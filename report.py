#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投资日报（包含 MA200 + VIX 22/34 双擎策略置顶看盘）
- 去除上下防抖轨，直接基于 MA200 进行判断
- 置顶模块极简处理：仅保留 QQQ 和 TQQQ 的“今日操作建议”
- 阅后即焚：渲染完成后自动删除 HTML，只保留 PDF
- 数据源：yfinance（日频）
"""

import os
import json
import csv
from datetime import datetime
from dateutil import tz
import numpy as np
import pandas as pd
import yfinance as yf
from jinja2 import Environment, BaseLoader

LOCAL_TZ = tz.gettz("Asia/Shanghai")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(PROJECT_DIR, "reports")

DEFAULT_CONFIG = {
    "title": "每日投资日报",
    "history_days": 400,
    "ma_windows": [20, 50, 200],
    "split_by_market": True,
    "export_pdf": True,
    "pdf_use_system_chrome_fallback": True,
    "print": {
        "page_size": "A3",
        "landscape": False,
        "margin_mm": 6,
        "base_font_px": 11,
        "table_font_px": 9,
        "dense": True,
        "print_two_columns": False,
        "pdf_scale": 0.92,
        "max_rows_per_block": 50
    }
}

HK_INDEX_SET = {"^HSI", "^HSCE", "^HSCC"}
US_INDEX_SET = {"^GSPC", "^NDX", "^IXIC", "^DJI", "^VIX", "^RUT"}

CN_COL_MAP = {
    "Name": "名称",
    "Ticker": "代码",
    "Close": "收盘价",
    "ChangePct": "涨跌幅(%)",
    "VolRatio": "量比",
    "Ret5D": "近5日(%)",
    "Ret20D": "近20日(%)",
    "MA20": "MA20",
    "MA50": "MA50",
    "MA200": "MA200",
    "StrategyHint": "策略状态",
    "DailyAction": "今日操作",
    "Trend": "趋势",
}

def infer_market_fallback(ticker: str) -> str:
    t = (ticker or "").upper().strip()
    if not t: return "其他"
    if t in HK_INDEX_SET or t.endswith(".HK"): return "港股"
    if t.endswith(".SS") or t.endswith(".SZ"): return "A股"
    if t in US_INDEX_SET: return "美股"
    if t.isdigit() and len(t) == 6 and (t.startswith("60") or t.startswith("00") or t.startswith("30")): return "A股"
    if t.startswith("^"): return "美股"
    if "." not in t and not t.startswith("^"): return "美股"
    if "=X" in t or "-" in t: return "全球"
    return "其他"

def ensure_default_watchlist_csv(path: str):
    if os.path.exists(path): return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "market", "ticker", "name"])
        w.writerow(["indices", "US", "^GSPC", "标普500指数"])
        w.writerow(["indices", "US", "^NDX", "纳斯达克100指数"])
        w.writerow(["indices", "US", "^IXIC", "纳斯达克综合指数"])
        w.writerow(["indices", "HK", "^HSI", "恒生指数"])
        w.writerow(["indices", "CN", "000300.SS", "沪深300指数"])
        w.writerow(["sectors", "US", "QQQ", "纳斯达克100ETF（QQQ）"])
        w.writerow(["sectors", "US", "SOXX", "半导体ETF（SOXX）"])
        w.writerow(["risk", "US", "^VIX", "恐慌指数VIX"])
        w.writerow(["risk", "GL", "USDCNY=X", "美元兑人民币"])
        w.writerow(["stocks", "US", "AAPL", "苹果"])
        w.writerow(["stocks", "HK", "0700.HK", "腾讯控股"])
        w.writerow(["stocks", "CN", "600519.SS", "贵州茅台"])

def load_watchlist_from_csv(path: str) -> dict:
    wl = {}
    if not os.path.exists(path): return wl
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        headers = [h.strip().lower() for h in (reader.fieldnames or [])]
        has_market = "market" in headers
        for row in reader:
            cat = (row.get("category") or "").strip()
            ticker = (row.get("ticker") or "").strip()
            name = (row.get("name") or "").strip()
            if not (cat and ticker and name): continue
            market = (row.get("market") or "").strip() if has_market else ""
            market = market.upper()
            if market in ("美股", "US", "USA"): market = "US"
            elif market in ("港股", "HK", "HKG"): market = "HK"
            elif market in ("A股", "CN", "CHN", "CHINA"): market = "CN"
            elif market in ("GL", "GLOBAL", "世界", "全球"): market = "GL"
            elif market in ("OTHER", "OTHERS", "其他"): market = "OT"
            wl.setdefault(cat, [])
            wl[cat].append({"ticker": ticker, "name": name, "market": market})
    
    if "strategy" not in wl:
        wl["strategy"] = [
            {"ticker": "TQQQ", "name": "核心策略大仓 (TQQQ)", "market": "US"},
            {"ticker": "^VIX", "name": "恐慌风控开关 (VIX)", "market": "US"},
            {"ticker": "SGOV", "name": "空仓短债理财 (SGOV)", "market": "US"}
        ]
    return wl

def deep_merge_dict(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge_dict(out[k], v)
        else:
            out[k] = v
    return out

def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
        if hasattr(x, "iloc"): x = x.iloc[0]
        elif isinstance(x, (list, tuple, np.ndarray)) and len(x) == 1: x = x[0]
        return float(x)
    except Exception:
        return np.nan

def _normalize_ma_windows(ma_windows):
    if ma_windows is None: ma_windows = []
    if isinstance(ma_windows, (int, float)): ma_windows = [int(ma_windows)]
    out = []
    for x in ma_windows:
        try: out.append(int(x))
        except Exception: continue
    if 200 not in out: out.append(200)
    return sorted(list(set([w for w in out if w > 0])))

def fetch_and_calculate(ticker_map: dict, history_days: int = 400, ma_windows=None) -> pd.DataFrame:
    if not ticker_map: return pd.DataFrame()
    ma_windows = _normalize_ma_windows(ma_windows)
    
    original_tickers = list(ticker_map.keys())
    tickers_to_fetch = list(set(original_tickers + ["^VIX"]))
    
    print(f"   -> 下载 {len(tickers_to_fetch)} 个标的历史收盘数据...")

    try:
        raw_data = yf.download(
            tickers_to_fetch, period=f"{history_days}d", interval="1d",
            auto_adjust=False, group_by="ticker", threads=True, progress=False
        )
    except Exception as e:
        print(f"   [Error] 下载失败: {e}")
        return pd.DataFrame()

    if raw_data is None or raw_data.empty: return pd.DataFrame()

    if "^VIX" in raw_data.columns.levels[0]:
        vix_df = raw_data["^VIX"]["Close"].dropna().sort_index()
    else:
        vix_df = pd.Series(dtype=float)

    rows = []
    for t in original_tickers:
        try:
            if len(tickers_to_fetch) == 1: df_t = raw_data
            else:
                if not hasattr(raw_data.columns, "levels") or t not in raw_data.columns.levels[0]: continue
                df_t = raw_data[t]

            df_t = df_t.dropna(subset=["Close"]).sort_index()
            if df_t.empty: continue

            close = df_t["Close"]
            
            if not vix_df.empty:
                vix_aligned = vix_df.reindex(close.index).ffill()
            else:
                vix_aligned = pd.Series(np.nan, index=close.index)

            curr = _safe_float(close.iloc[-1])
            prev = _safe_float(close.iloc[-2]) if len(close) >= 2 else np.nan
            pct = (curr / prev - 1) * 100 if (not np.isnan(prev) and prev != 0) else np.nan

            curr_vix = _safe_float(vix_aligned.iloc[-1]) if len(vix_aligned) >= 1 else np.nan
            prev_vix = _safe_float(vix_aligned.iloc[-2]) if len(vix_aligned) >= 2 else np.nan

            ma_vals = {}
            ma200_series = None
            for w in ma_windows:
                s = close.rolling(w).mean()
                ma_vals[f"MA{w}"] = _safe_float(s.iloc[-1]) if len(close) >= w else np.nan
                if w == 200: ma200_series = s

            if ma200_series is None: ma200_series = close.rolling(200).mean()
            curr_ma200 = _safe_float(ma200_series.iloc[-1]) if len(close) >= 200 else np.nan
            prev_ma200 = _safe_float(ma200_series.iloc[-2]) if len(close) >= 201 else np.nan

            strategy_hint = ""
            daily_action = "观望"
            
            if (not np.isnan(prev) and not np.isnan(prev_ma200) and not np.isnan(curr) and not np.isnan(curr_ma200)):
                if t == "^VIX":
                    if curr > 34:
                        strategy_hint = "🔴 极度恐慌 (触发满仓抄底特权)"
                        daily_action = "买点出现"
                    elif curr > 22:
                        strategy_hint = "🟠 高压警戒 (破均线坚决卖出)"
                        daily_action = "警戒区域"
                    else:
                        strategy_hint = "🟢 情绪平稳 (大盘安全)"
                        daily_action = "安全期"
                        
                elif t in ["SGOV", "SHV", "BIL"]:
                    strategy_hint = "💰 空仓期专属避险理财 (年化约4.5%)"
                    daily_action = "稳定收息"
                    
                else:
                    want_sell = (curr < curr_ma200) and (curr_vix > 22.0)
                    want_buy_panic = (prev_vix > 34.0)
                    
                    is_sell_state = want_sell and not want_buy_panic
                    is_buy_state = (curr > curr_ma200) or want_buy_panic

                    prev_want_sell = (prev < prev_ma200) and (prev_vix > 22.0)
                    prev_vix_2 = _safe_float(vix_aligned.iloc[-3]) if len(vix_aligned) >= 3 else np.nan
                    prev_want_buy_panic = (prev_vix_2 > 34.0)
                    
                    prev_is_sell_state = prev_want_sell and not prev_want_buy_panic
                    prev_is_buy_state = (prev > prev_ma200) or prev_want_buy_panic

                    if is_buy_state and not prev_is_buy_state and not is_sell_state:
                        strategy_hint = "！VIX恐慌抄底" if want_buy_panic else "！向上突破MA200"
                        daily_action = "买入"
                    elif is_sell_state and not prev_is_sell_state:
                        strategy_hint = f"！跌破MA200且VIX恐慌({curr_vix:.1f})"
                        daily_action = "卖出"
                    elif is_buy_state:
                        strategy_hint = "运行于均线之上 (多头持仓)"
                    elif is_sell_state:
                        strategy_hint = "防线跌穿且处于高压恐慌中 (空仓)"
                    elif curr < curr_ma200:
                        strategy_hint = f"阴跌假摔中 (VIX={curr_vix:.1f} < 22)"
                    else:
                        strategy_hint = "处于MA200附近震荡"

            ret_vals = {}
            for w in [5, 20]:
                if len(close) > w: ret_vals[f"Ret{w}D"] = _safe_float((close.iloc[-1] / close.iloc[-1 - w] - 1) * 100)
                else: ret_vals[f"Ret{w}D"] = np.nan

            vol_ratio = np.nan
            if "Volume" in df_t.columns:
                v_curr = _safe_float(df_t["Volume"].iloc[-1])
                if len(df_t) >= 22:
                    v_avg = df_t["Volume"].iloc[-21:-1].mean()
                    if v_avg and v_avg > 0: vol_ratio = v_curr / v_avg

            rows.append({
                "Name": ticker_map.get(t, t),
                "Ticker": t,
                "Close": curr,
                "ChangePct": pct,
                "VolRatio": vol_ratio,
                "StrategyHint": strategy_hint,
                "DailyAction": daily_action,
                **ma_vals,
                **ret_vals,
            })
        except Exception:
            continue
    return pd.DataFrame(rows)

def add_trend_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    def judge(row):
        score, valid = 0, 0
        for ma in ["MA20", "MA50", "MA200"]:
            if ma in row and not pd.isna(row.get(ma)) and not pd.isna(row.get("Close")):
                valid += 1
                score += 1 if row["Close"] > row[ma] else -1
        if valid == 0: return ""
        return "强" if score == valid else ("弱" if score == -valid else "中")
    out["Trend"] = out.apply(judge, axis=1)
    return out

def format_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    if "ChangePct" in out.columns: out = out.sort_values("ChangePct", ascending=False)
    numeric_cols = ["Close", "ChangePct", "VolRatio", "Ret5D", "Ret20D", "MA20", "MA50", "MA200"]
    for c in numeric_cols:
        if c in out.columns: out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    return out

def cn_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df is None: return df
    return df.rename(columns={c: CN_COL_MAP.get(c, c) for c in df.columns})

# —— HTML 模板 ——
HTML_TEMPLATE = r"""
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }} - {{ report_date }}</title>

<style>
  :root{
    --fg:#111827; --muted:#667085; --border:#D0D5DD; --head:#F2F4F7; --stripe:#FAFAFB;
    --blue:#0B5ED7;
  }
  html,body{ margin:0; padding:0; color:var(--fg); }
  body{
    font-family: -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Hiragino Sans GB","Microsoft YaHei",Arial,sans-serif;
    font-size: {{ base_font_px }}px;
    line-height: 1.25;
  }
  .wrap{ padding: 10px 12px; max-width: 1280px; margin:0 auto; }
  h1{ margin:0; font-size: 18px; }
  .meta{ margin-top:4px; color:var(--muted); font-size: 11px; }

  .grid{
    display:grid;
    grid-template-columns: 1fr;
    gap: 8px;
    margin-top: 10px;
  }

  .card{
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 10px;
    box-shadow: 0 1px 1px rgba(0,0,0,.03);
    break-inside: avoid;
    page-break-inside: avoid;
  }
  .card h2{
    margin: 0 0 6px;
    font-size: 14px;
    color: var(--blue);
  }

  table{
    width:100%;
    border-collapse: collapse;
    table-layout: fixed;
    font-size: {{ table_font_px }}px;
  }
  th, td{
    border: 1px solid var(--border);
    padding: {{ "2px 6px" if dense else "6px 8px" }};
    vertical-align: middle;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  th{
    background: var(--head);
    text-align: center;
    font-weight: 600;
  }
  td{ text-align: right; }
  td:nth-child(1), td:nth-child(2){ text-align: left; }
  tbody tr:nth-child(even) td{ background: var(--stripe); }

  .pos{ color:#067647; font-weight: 600; }
  .neg{ color:#B42318; font-weight: 600; }

  @media print {
    @page {
      size: {{ page_size }} {{ "landscape" if landscape else "portrait" }};
      margin: {{ margin_mm }}mm;
    }
    .wrap{ padding: 0; max-width: none; }

    {% if print_two_columns %}
    .grid{ grid-template-columns: 1fr 1fr; gap: 6px; }
    {% endif %}

    .card{ box-shadow:none; border-radius: 0; padding: 6px 8px; }
    table{ font-size: {{ table_font_px }}px; }
    th,td{ padding: 2px 4px; }
    tr, td, th { page-break-inside: avoid; break-inside: avoid; }
  }
</style>
</head>

<body>
<div class="wrap">
  <h1>{{ title }}（{{ report_date }}）</h1>
  <div class="meta">策略参数：收盘价突破 MA200 | 卖出触发 VIX > 22 | 抄底触发 VIX > 34</div>
  <div class="meta">数据源：yfinance　|　生成时间：{{ generated_at }}</div>

  <div class="grid">
    {% for b in blocks %}
      <div class="card">
        <h2>{{ b.title }}</h2>
        {{ b.html_table | safe }}
      </div>
    {% endfor %}
  </div>
</div>
</body>
</html>
"""

def df_to_html_table(df: pd.DataFrame) -> str:
    if df is None or df.empty: return ""
    df2 = df.copy()

    def fmt_pct(x):
        if x is None or (isinstance(x, float) and np.isnan(x)): return ""
        try: v = float(x)
        except Exception: return str(x)
        cls = "pos" if v > 0 else ("neg" if v < 0 else "")
        s = f"{v:.2f}"
        return f'<span class="{cls}">{s}</span>' if cls else s

    for col in ["涨跌幅(%)", "近5日(%)", "近20日(%)"]:
        if col in df2.columns: df2[col] = df2[col].apply(fmt_pct)

    if "策略状态" in df2.columns:
        def _fmt_hint(x):
            s = str(x).strip() if x is not None else ""
            if "恐慌抄底" in s or "突破" in s or "买点出现" in s: return f'<span class="pos">🟢 {s}</span>'
            if "砸穿" in s or "跌破" in s or "极度恐慌" in s: return f'<span class="neg">🔴 {s}</span>'
            if "多头" in s or "平稳" in s: return f'<span class="pos" style="opacity:0.8">{s}</span>'
            if "空头" in s or "警戒" in s or "阴跌" in s: return f'<span class="neg" style="opacity:0.8">{s}</span>'
            if "理财" in s: return f'<span style="color:#B8860B; font-weight:bold;">{s}</span>' 
            return f'<span style="color:var(--muted)">{s}</span>'
        df2["策略状态"] = df2["策略状态"].apply(_fmt_hint)

    if "今日操作" in df2.columns:
        def _fmt_action(x):
            s = str(x).strip() if x is not None else ""
            if "买入" in s or "抄底" in s: return f'<span class="pos">🟢 {s} (次日开盘)</span>'
            if "卖出" in s: return f'<span class="neg">🔴 {s} (次日开盘)</span>'
            if "理财" in s or "收息" in s: return f'<span style="color:#B8860B;">💵 {s}</span>'
            if "警戒" in s: return f'<span class="neg" style="opacity:0.8">⚠️ {s}</span>'
            return f'<span style="color:var(--muted)">➖ {s}</span>'
        df2["今日操作"] = df2["今日操作"].apply(_fmt_action)

    return df2.to_html(index=False, escape=False, border=0)

def _mac_default_chrome_path() -> str:
    return "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

def export_pdf_with_playwright(html_path: str, pdf_path: str, page_size: str, landscape: bool, margin_mm: int, scale: float, use_system_chrome_fallback: bool) -> bool:
    try: from playwright.sync_api import sync_playwright
    except Exception:
        print("⚠️ 未安装 playwright：跳过自动导出 PDF。")
        return False

    html_abs = os.path.abspath(html_path)
    pdf_abs = os.path.abspath(pdf_path)

    def _do_export(p, executable_path=None):
        browser = p.chromium.launch(executable_path=executable_path) if executable_path else p.chromium.launch()
        page = browser.new_page()
        page.goto(f"file://{html_abs}", wait_until="networkidle")
        page.pdf(
            path=pdf_abs, format=page_size, landscape=landscape, scale=float(scale),
            margin={"top": f"{margin_mm}mm", "bottom": f"{margin_mm}mm", "left": f"{margin_mm}mm", "right": f"{margin_mm}mm"},
            print_background=True,
        )
        browser.close()

    try:
        with sync_playwright() as p:
            try:
                _do_export(p)
                return True
            except Exception as e:
                msg = str(e)
                if "Executable doesn't exist" in msg or "browser has been closed" in msg:
                    if use_system_chrome_fallback and os.path.exists(_mac_default_chrome_path()):
                        print("⚠️ Playwright Chromium 缺失，尝试使用系统 Chrome 导出 PDF...")
                        _do_export(p, executable_path=_mac_default_chrome_path())
                        return True
                    print("⚠️ Playwright Chromium 缺失。请运行：python3 -m playwright install chromium")
                    return False
                print(f"⚠️ playwright 导出 PDF 失败：{e}")
                return False
    except Exception as e:
        print(f"⚠️ playwright 运行异常：{e}")
        return False

# ====== 今日极简操作指令 (仅处理 QQQ 和 TQQQ) ======
def get_today_action_block(tickers=["QQQ", "TQQQ"], ma_window=200, vix_sell_th=22.0, vix_buy_th=34.0):
    print(f"   -> [今日操作建议] 正在拉取 {', '.join(tickers)} 状态...")
    try:
        fetch_list = tickers + ["^VIX"]
        df_raw = yf.download(fetch_list, period="1y", auto_adjust=True, progress=False)
        
        if df_raw.empty:
            return {"title": "🎯 今日操作建议", "html_table": "<p>数据获取失败，请检查网络。</p>"}
            
        if isinstance(df_raw.columns, pd.MultiIndex):
            vix_close = df_raw['Close']['^VIX']
        else:
            vix_close = df_raw['Close'] if len(tickers) == 0 else df_raw['Close']['^VIX'] # 兼容写法

        html_rows = ""
        for tk in tickers:
            if isinstance(df_raw.columns, pd.MultiIndex):
                if tk not in df_raw['Close']: continue
                tk_close = df_raw['Close'][tk]
            else:
                tk_close = df_raw['Close']
            
            df = pd.DataFrame({"close": tk_close, "vix": vix_close}).dropna()
            df["ma200"] = df["close"].rolling(ma_window).mean()
            df = df.dropna()
            
            if len(df) < 2:
                html_rows += f"<tr><td style='padding: 8px; border: 1px solid #D0D5DD;'><b>{tk}</b></td><td style='padding: 8px; border: 1px solid #D0D5DD;'>有效数据不足</td></tr>"
                continue
                
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            is_buy_today = (latest["close"] > latest["ma200"]) or (latest["vix"] > vix_buy_th)
            is_sell_today = (latest["close"] < latest["ma200"]) and (latest["vix"] > vix_sell_th)
            
            is_buy_yest = (prev["close"] > prev["ma200"]) or (prev["vix"] > vix_buy_th)
            is_sell_yest = (prev["close"] < prev["ma200"]) and (prev["vix"] > vix_sell_th)

            if is_buy_today and not is_buy_yest and not is_sell_today:
                action_text = "🚨 【买入】触发买入信号！(突破 MA200 或 VIX > 34)"
                color = "#067647" 
            elif is_sell_today and not is_sell_yest:
                action_text = "🚨 【卖出】触发卖出信号！(跌破 MA200 且 VIX > 22)"
                color = "#B42318" 
            else:
                if is_buy_today:
                    action_text = "✅ 【持有】建议持有做多 (MAIN)。今日无新操作信号。"
                    color = "#067647"
                elif is_sell_today:
                    action_text = "🛑 【空仓】建议空仓吃息 (CASH)。今日无新操作信号。"
                    color = "#B42318"
                elif latest["close"] < latest["ma200"]:
                    action_text = f"⚠️ 【观望】均线以下阴跌中 (VIX {latest['vix']:.1f} 未达恐慌卖出阈值)。维持原仓。"
                    color = "#B8860B" 
                else:
                    action_text = "⚠️ 【观望】处于均线胶着状态。"
                    color = "#667085"
            
            html_rows += f"""
            <tr>
                <td style="padding: 10px; border: 1px solid #D0D5DD; background: #FAFAFB; font-weight: bold; width: 15%; text-align: center;">{tk}</td>
                <td style="padding: 10px; border: 1px solid #D0D5DD; color: {color}; font-weight: bold; font-size: 14px;">{action_text}</td>
            </tr>
            """
            
        html = f"""
        <table style="width:100%; border-collapse: collapse; text-align: left; font-size: 13px;">
            {html_rows}
        </table>
        """
        return {"title": "🎯 今日操作建议 (QQQ & TQQQ)", "html_table": html}
    except Exception as e:
        return {"title": "🎯 今日操作建议", "html_table": f"<p>执行异常: {e}</p>"}

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    config_path = os.path.join(PROJECT_DIR, "config.json")
    watchlist_csv_path = os.path.join(PROJECT_DIR, "watchlist.csv")

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = deep_merge_dict(DEFAULT_CONFIG, json.load(f))
    else:
        cfg = DEFAULT_CONFIG
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)

    ensure_default_watchlist_csv(watchlist_csv_path)
    wl = load_watchlist_from_csv(watchlist_csv_path)
    if not wl:
        print("watchlist.csv 为空，无法生成。")
        return

    dt = datetime.now(tz=LOCAL_TZ)
    report_date = dt.strftime("%Y-%m-%d")

    title = cfg.get("title", "每日投资日报")
    history_days = int(cfg.get("history_days", 400))
    ma_windows = _normalize_ma_windows(cfg.get("ma_windows", [20, 50, 200]))

    split_by_market = bool(cfg.get("split_by_market", True))
    print_cfg = cfg.get("print", {})

    page_size = str(print_cfg.get("page_size", "A3"))
    landscape = bool(print_cfg.get("landscape", False))
    margin_mm = int(print_cfg.get("margin_mm", 6))
    base_font_px = int(print_cfg.get("base_font_px", 11))
    table_font_px = int(print_cfg.get("table_font_px", 9))
    dense = bool(print_cfg.get("dense", True))
    print_two_columns = bool(print_cfg.get("print_two_columns", False))
    pdf_scale = float(print_cfg.get("pdf_scale", 0.92))

    max_rows = print_cfg.get("max_rows_per_block", None)
    if max_rows is not None:
        try: max_rows = int(max_rows)
        except Exception: max_rows = None

    blocks = []
    
    # ====== 将极简今日操作指令植入 PDF 最顶层 ======
    action_block = get_today_action_block(["QQQ", "TQQQ"])
    if action_block:
        blocks.append(action_block)

    sections = [
        ("strategy", "🔥 核心策略收盘级复盘 (MA200 + VIX 22/34 双擎)",
         ["Name", "Ticker", "Close", "ChangePct", "MA200", "StrategyHint", "DailyAction"]),
        ("indices", "大盘指数（趋势）",
         ["Name", "Ticker", "Close", "ChangePct", "Ret5D", "Ret20D", "MA20", "MA50", "MA200", "StrategyHint", "DailyAction", "Trend"]),
        ("sectors", "行业/主题指数 (ETF)",
         ["Name", "Ticker", "Close", "ChangePct", "Ret5D", "Ret20D", "MA20", "MA50", "MA200", "StrategyHint", "DailyAction", "Trend", "VolRatio"]),
        ("stocks", "重点个股",
         ["Name", "Ticker", "Close", "ChangePct", "Ret5D", "Ret20D", "MA20", "MA50", "MA200", "StrategyHint", "DailyAction", "Trend", "VolRatio"]),
        ("risk", "宏观 & 风险监控",
         ["Name", "Ticker", "Close", "ChangePct", "Ret5D", "Ret20D", "MA20"]),
    ]

    market_order = [("US", "美股"), ("HK", "港股"), ("CN", "A股"), ("GL", "全球"), ("OT", "其他"), ("", "其他")]

    for key, section_title, cols_to_show in sections:
        items = wl.get(key, [])
        if not items: continue

        ticker_map = {it["ticker"]: it["name"] for it in items}
        ticker_market_map = {it["ticker"]: (it.get("market") or "").upper().strip() for it in items}

        print(f"正在处理: {section_title} ...")
        df = fetch_and_calculate(ticker_map, history_days, ma_windows)
        df = add_trend_flags(df)
        df_show = format_table(df)

        final_cols = [c for c in cols_to_show if c in df_show.columns]
        df_final = df_show[final_cols].copy()

        def market_cn_from_watchlist(tk: str) -> str:
            m = (ticker_market_map.get(tk, "") or "").upper().strip()
            if m == "US": return "美股"
            if m == "HK": return "港股"
            if m == "CN": return "A股"
            if m == "GL": return "全球"
            if m == "OT": return "其他"
            return infer_market_fallback(tk)

        if key == "strategy":
            df_final = cn_cols(df_final)
            html_table = df_to_html_table(df_final)
            if html_table: blocks.append({"title": section_title, "html_table": html_table})
            continue

        if split_by_market and "Ticker" in df_final.columns and not df_final.empty:
            df_final["_MarketCN"] = df_final["Ticker"].apply(market_cn_from_watchlist)

            for _, mk_cn in market_order:
                sub = df_final[df_final["_MarketCN"] == mk_cn].drop(columns=["_MarketCN"])
                if sub.empty: continue
                if max_rows: sub = sub.head(max_rows)
                sub = cn_cols(sub)
                html_table = df_to_html_table(sub)
                if not html_table: continue
                blocks.append({"title": f"{section_title} - {mk_cn}", "html_table": html_table})
        else:
            if df_final is None or df_final.empty: continue
            if max_rows: df_final = df_final.head(max_rows)
            df_final = cn_cols(df_final)
            html_table = df_to_html_table(df_final)
            if not html_table: continue
            blocks.append({"title": section_title, "html_table": html_table})

    if not blocks:
        print("⚠️ 没有可输出的数据块。")
        return

    env = Environment(loader=BaseLoader())
    template = env.from_string(HTML_TEMPLATE)
    html_content = template.render(
        title=title,
        report_date=report_date,
        generated_at=dt.strftime("%Y-%m-%d %H:%M:%S"),
        blocks=blocks,
        page_size=page_size,
        landscape=landscape,
        margin_mm=margin_mm,
        base_font_px=base_font_px,
        table_font_px=table_font_px,
        dense=dense,
        print_two_columns=print_two_columns
    )

    html_path = os.path.join(REPORT_DIR, f"{report_date}_temp.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    if cfg.get("export_pdf", True):
        pdf_path = os.path.join(REPORT_DIR, f"{report_date}.pdf")
        use_fallback = bool(cfg.get("pdf_use_system_chrome_fallback", True))
        print("   -> 正在使用 playwright 导出 PDF...")
        ok = export_pdf_with_playwright(html_path, pdf_path, page_size, landscape, margin_mm, pdf_scale, use_fallback)
        if ok: 
            print(f"✅ 成功生成纯净版 PDF: {pdf_path}")
        else:
            print("❌ PDF 生成失败。")

    if os.path.exists(html_path):
        os.remove(html_path)
        print("🗑️ 已清理中间生成的 HTML 文件。")

if __name__ == "__main__":
    main()