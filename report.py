#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投资日报（包含 MA200 + VIX 22/34 双擎策略置顶看盘）- 网页轻量版
- 移除 Playwright/PDF 依赖，仅生成轻量 index.html
- 信号降噪：除了 QQQ 和 TQQQ，其他标的均隐藏操作和趋势信号
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
# 生成的文件将放在 public 文件夹中，方便直接部署为静态网站
REPORT_DIR = os.path.join(PROJECT_DIR, "public")

DEFAULT_CONFIG = {
    "title": "每日投资日报",
    "history_days": 400,
    "ma_windows": [20, 50, 200],
    "split_by_market": True,
    "base_font_px": 14,
    "table_font_px": 12,
    "dense": False
}

HK_INDEX_SET = {"^HSI", "^HSCE", "^HSCC"}
US_INDEX_SET = {"^GSPC", "^NDX", "^IXIC", "^DJI", "^VIX", "^RUT"}

CN_COL_MAP = {
    "Name": "名称", "Ticker": "代码", "Close": "收盘价", "ChangePct": "涨跌幅(%)",
    "VolRatio": "量比", "Ret5D": "近5日(%)", "Ret20D": "近20日(%)",
    "MA20": "MA20", "MA50": "MA50", "MA200": "MA200",
    "StrategyHint": "策略状态", "DailyAction": "今日操作", "Trend": "趋势",
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
        w.writerow(["sectors", "US", "QQQ", "纳斯达克100ETF（QQQ）"])
        w.writerow(["strategy", "US", "TQQQ", "纳斯达克三倍做多（TQQQ）"])
        w.writerow(["risk", "US", "^VIX", "恐慌指数VIX"])
        w.writerow(["stocks", "US", "AAPL", "苹果"])

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
    return wl

def deep_merge_dict(base: dict, updates: dict) -> dict:
    out = dict(base)
    for k, v in (updates or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict): out[k] = deep_merge_dict(out[k], v)
        else: out[k] = v
    return out

def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
        if hasattr(x, "iloc"): x = x.iloc[0]
        elif isinstance(x, (list, tuple, np.ndarray)) and len(x) == 1: x = x[0]
        return float(x)
    except Exception:
        return np.nan

def fetch_and_calculate(ticker_map: dict, history_days: int = 400, ma_windows=[20, 50, 200]) -> pd.DataFrame:
    if not ticker_map: return pd.DataFrame()
    original_tickers = list(ticker_map.keys())
    tickers_to_fetch = list(set(original_tickers + ["^VIX"]))
    
    try:
        raw_data = yf.download(
            tickers_to_fetch, period=f"{history_days}d", interval="1d",
            auto_adjust=False, group_by="ticker", threads=True, progress=False
        )
    except Exception:
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
            vix_aligned = vix_df.reindex(close.index).ffill() if not vix_df.empty else pd.Series(np.nan, index=close.index)

            curr = _safe_float(close.iloc[-1])
            prev = _safe_float(close.iloc[-2]) if len(close) >= 2 else np.nan
            pct = (curr / prev - 1) * 100 if (not np.isnan(prev) and prev != 0) else np.nan
            curr_vix = _safe_float(vix_aligned.iloc[-1]) if len(vix_aligned) >= 1 else np.nan
            prev_vix = _safe_float(vix_aligned.iloc[-2]) if len(vix_aligned) >= 2 else np.nan

            ma_vals = {}
            for w in ma_windows:
                s = close.rolling(w).mean()
                ma_vals[f"MA{w}"] = _safe_float(s.iloc[-1]) if len(close) >= w else np.nan
                if w == 200: ma200_series = s

            curr_ma200 = _safe_float(ma200_series.iloc[-1]) if len(close) >= 200 else np.nan
            prev_ma200 = _safe_float(ma200_series.iloc[-2]) if len(close) >= 201 else np.nan

            strategy_hint = ""
            daily_action = "观望"
            
            if not np.isnan(prev) and not np.isnan(curr_ma200):
                if t == "^VIX":
                    if curr > 34: strategy_hint, daily_action = "🔴 极度恐慌 (抄底)", "买点出现"
                    elif curr > 22: strategy_hint, daily_action = "🟠 高压警戒 (破均线卖出)", "警戒区域"
                    else: strategy_hint, daily_action = "🟢 情绪平稳", "安全期"
                elif t in ["SGOV", "SHV", "BIL"]:
                    strategy_hint, daily_action = "💰 空仓收息", "稳定收息"
                else:
                    want_sell = (curr < curr_ma200) and (curr_vix > 22.0)
                    want_buy_panic = (prev_vix > 34.0)
                    is_sell_state = want_sell and not want_buy_panic
                    is_buy_state = (curr > curr_ma200) or want_buy_panic
                    
                    prev_want_sell = (prev < prev_ma200) and (prev_vix > 22.0)
                    prev_want_buy_panic = (_safe_float(vix_aligned.iloc[-3]) > 34.0) if len(vix_aligned)>=3 else False
                    prev_is_sell_state = prev_want_sell and not prev_want_buy_panic
                    prev_is_buy_state = (prev > prev_ma200) or prev_want_buy_panic

                    if is_buy_state and not prev_is_buy_state and not is_sell_state:
                        strategy_hint, daily_action = ("！VIX恐慌抄底" if want_buy_panic else "！突破MA200"), "买入"
                    elif is_sell_state and not prev_is_sell_state:
                        strategy_hint, daily_action = f"！跌穿且恐慌(VIX={curr_vix:.1f})", "卖出"
                    elif is_buy_state: strategy_hint = "均线之上 (持仓)"
                    elif is_sell_state: strategy_hint = "防线跌穿且高压 (空仓)"
                    elif curr < curr_ma200: strategy_hint = f"阴跌假摔 (VIX={curr_vix:.1f})"
                    else: strategy_hint = "MA200附近震荡"

            ret_vals = {}
            for w in [5, 20]: ret_vals[f"Ret{w}D"] = _safe_float((close.iloc[-1] / close.iloc[-1-w] - 1)*100) if len(close)>w else np.nan

            vol_ratio = np.nan
            if "Volume" in df_t.columns and len(df_t) >= 22:
                v_avg = df_t["Volume"].iloc[-21:-1].mean()
                if v_avg and v_avg > 0: vol_ratio = _safe_float(df_t["Volume"].iloc[-1]) / v_avg

            rows.append({
                "Name": ticker_map.get(t, t), "Ticker": t, "Close": curr, "ChangePct": pct,
                "VolRatio": vol_ratio, "StrategyHint": strategy_hint, "DailyAction": daily_action,
                **ma_vals, **ret_vals,
            })
        except Exception: continue
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

# —— HTML 模板 (适配手机浏览) ——
HTML_TEMPLATE = r"""
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>{{ title }} - {{ report_date }}</title>
<style>
  :root{ --fg:#111827; --muted:#667085; --border:#E4E7EC; --head:#F9FAFB; --stripe:#F9FAFB; --blue:#0B5ED7; }
  body{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; font-size: {{ base_font_px }}px; line-height: 1.4; color:var(--fg); margin:0; padding:12px; background:#F2F4F7;}
  .wrap{ max-width: 800px; margin:0 auto; background:#fff; border-radius:12px; padding:16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
  h1{ margin:0 0 4px; font-size: 20px; color:#101828; }
  .meta{ color:var(--muted); font-size: 12px; margin-bottom: 12px;}
  .card{ border: 1px solid var(--border); border-radius: 8px; margin-bottom: 16px; overflow:hidden;}
  .card h2{ margin: 0; padding: 10px 12px; font-size: 15px; background: var(--head); border-bottom: 1px solid var(--border); color: #344054; }
  .table-wrap { overflow-x: auto; }
  table{ width:100%; border-collapse: collapse; font-size: {{ table_font_px }}px; white-space: nowrap; }
  th, td{ border-bottom: 1px solid var(--border); padding: 8px 12px; text-align: right; }
  th{ background: #fff; text-align: right; font-weight: 600; color:#475467; }
  td:nth-child(1), th:nth-child(1), td:nth-child(2), th:nth-child(2){ text-align: left; }
  tbody tr:hover{ background: var(--stripe); }
  .pos{ color:#027A48; font-weight: 600; }
  .neg{ color:#B42318; font-weight: 600; }
</style>
</head>
<body>
<div class="wrap">
  <h1>{{ title }}</h1>
  <div class="meta">{{ report_date }} | 美股收盘生成 | yfinance</div>
  {% for b in blocks %}
    <div class="card">
      <h2>{{ b.title }}</h2>
      <div class="table-wrap">{{ b.html_table | safe }}</div>
    </div>
  {% endfor %}
</div>
</body>
</html>
"""

def get_today_action_block(tickers=["QQQ", "TQQQ"]):
    try:
        fetch_list = tickers + ["^VIX"]
        df_raw = yf.download(fetch_list, period="1y", auto_adjust=True, progress=False)
        if df_raw.empty: return None
        vix_close = df_raw['Close']['^VIX'] if isinstance(df_raw.columns, pd.MultiIndex) else (df_raw['Close'] if len(tickers)==0 else df_raw['Close']['^VIX'])

        html_rows = ""
        for tk in tickers:
            tk_close = df_raw['Close'][tk] if isinstance(df_raw.columns, pd.MultiIndex) else df_raw['Close']
            df = pd.DataFrame({"close": tk_close, "vix": vix_close}).dropna()
            df["ma200"] = df["close"].rolling(200).mean()
            df = df.dropna()
            if len(df) < 2: continue
            
            latest, prev = df.iloc[-1], df.iloc[-2]
            is_buy = (latest["close"] > latest["ma200"]) or (latest["vix"] > 34.0)
            is_sell = (latest["close"] < latest["ma200"]) and (latest["vix"] > 22.0)
            prev_buy = (prev["close"] > prev["ma200"]) or (prev["vix"] > 34.0)
            prev_sell = (prev["close"] < prev["ma200"]) and (prev["vix"] > 22.0)

            if is_buy and not prev_buy and not is_sell:
                txt, color = "🚨 【买入】触发买入信号！(突破MA200或VIX>34)", "#027A48"
            elif is_sell and not prev_sell:
                txt, color = "🚨 【卖出】触发卖出信号！(跌破MA200且VIX>22)", "#B42318"
            elif is_buy: txt, color = "✅ 【持有】建议持有做多。无新信号。", "#027A48"
            elif is_sell: txt, color = "🛑 【空仓】建议空仓吃息。无新信号。", "#B42318"
            elif latest["close"] < latest["ma200"]: txt, color = f"⚠️ 【观望】均线下阴跌(VIX {latest['vix']:.1f})。维持原仓", "#B8860B"
            else: txt, color = "⚠️ 【观望】处于均线胶着状态。", "#667085"
            
            html_rows += f"<tr><td style='padding:12px; border-bottom:1px solid #E4E7EC; font-weight:bold; width:20%;'>{tk}</td><td style='padding:12px; border-bottom:1px solid #E4E7EC; color:{color}; font-weight:bold;'>{txt}</td></tr>"
        
        return {"title": "🎯 核心标的今日操作", "html_table": f"<table style='width:100%; border-collapse:collapse; text-align:left;'>{html_rows}</table>"}
    except Exception: return None

def df_to_html_table(df: pd.DataFrame) -> str:
    if df is None or df.empty: return ""
    df2 = df.copy()
    def fmt_pct(x):
        if x is None or (isinstance(x, float) and np.isnan(x)): return ""
        try: v = float(x)
        except: return str(x)
        cls = "pos" if v > 0 else ("neg" if v < 0 else "")
        return f'<span class="{cls}">{v:.2f}</span>' if cls else f"{v:.2f}"

    for col in ["涨跌幅(%)", "近5日(%)", "近20日(%)"]:
        if col in df2.columns: df2[col] = df2[col].apply(fmt_pct)
        
    numeric_cols = ["收盘价", "量比", "MA20", "MA50", "MA200"]
    for c in numeric_cols:
        if c in df2.columns: df2[c] = pd.to_numeric(df2[c], errors="coerce").round(2)

    if "策略状态" in df2.columns:
        def _fmt_hint(x):
            s = str(x).strip() if x is not None else ""
            if s == "-": return '<span style="color:#D0D5DD">-</span>'
            if "恐慌" in s or "突破" in s or "买点" in s: return f'<span class="pos">🟢 {s}</span>'
            if "砸穿" in s or "跌穿" in s or "高压" in s: return f'<span class="neg">🔴 {s}</span>'
            if "持仓" in s or "平稳" in s: return f'<span class="pos" style="opacity:0.8">{s}</span>'
            if "空仓" in s or "阴跌" in s: return f'<span class="neg" style="opacity:0.8">{s}</span>'
            if "收息" in s: return f'<span style="color:#B8860B; font-weight:bold;">{s}</span>' 
            return f'<span style="color:var(--muted)">{s}</span>'
        df2["策略状态"] = df2["策略状态"].apply(_fmt_hint)

    if "今日操作" in df2.columns:
        def _fmt_action(x):
            s = str(x).strip() if x is not None else ""
            if s == "-": return '<span style="color:#D0D5DD">-</span>'
            if "买入" in s: return f'<span class="pos">🟢 {s}</span>'
            if "卖出" in s: return f'<span class="neg">🔴 {s}</span>'
            if "收息" in s: return f'<span style="color:#B8860B;">💵 {s}</span>'
            if "警戒" in s: return f'<span class="neg" style="opacity:0.8">⚠️ {s}</span>'
            return f'<span style="color:var(--muted)">➖ {s}</span>'
        df2["今日操作"] = df2["今日操作"].apply(_fmt_action)

    return df2.to_html(index=False, escape=False, border=0)

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    config_path = os.path.join(PROJECT_DIR, "config.json")
    cfg = deep_merge_dict(DEFAULT_CONFIG, json.load(open(config_path, "r", encoding="utf-8")) if os.path.exists(config_path) else {})
    
    watchlist_path = os.path.join(PROJECT_DIR, "watchlist.csv")
    ensure_default_watchlist_csv(watchlist_path)
    wl = load_watchlist_from_csv(watchlist_path)
    
    dt = datetime.now(tz=LOCAL_TZ)
    report_date = dt.strftime("%Y-%m-%d")
    
    blocks = []
    action_block = get_today_action_block(["QQQ", "TQQQ"])
    if action_block: blocks.append(action_block)

    sections = [
        ("indices", "大盘指数", ["Name", "Ticker", "Close", "ChangePct", "Ret5D", "Ret20D", "MA20", "MA50", "MA200", "StrategyHint", "DailyAction", "Trend"]),
        ("sectors", "行业/主题指数 (ETF)", ["Name", "Ticker", "Close", "ChangePct", "Ret5D", "Ret20D", "MA200", "StrategyHint", "DailyAction", "Trend", "VolRatio"]),
        ("strategy", "核心策略标的", ["Name", "Ticker", "Close", "ChangePct", "Ret5D", "MA200", "StrategyHint", "DailyAction", "Trend"]),
        ("stocks", "重点个股", ["Name", "Ticker", "Close", "ChangePct", "Ret5D", "MA200", "StrategyHint", "DailyAction", "Trend", "VolRatio"]),
        ("risk", "风险与宏观", ["Name", "Ticker", "Close", "ChangePct", "Ret5D", "Ret20D", "MA20"])
    ]

    for key, title, cols in sections:
        items = wl.get(key, [])
        if not items: continue
        df = fetch_and_calculate({it["ticker"]: it["name"] for it in items}, cfg.get("history_days", 400), cfg.get("ma_windows", [20, 50, 200]))
        df = add_trend_flags(df)
        if df.empty: continue
        
        # 只保留所需列，如果是涨跌幅则按涨跌幅倒序
        final_cols = [c for c in cols if c in df.columns]
        df_show = df[final_cols].copy()
        if "ChangePct" in df_show.columns: df_show = df_show.sort_values("ChangePct", ascending=False)
        
        # 🟢 【核心优化】：对非 QQQ / TQQQ 的标的，隐藏策略和操作列
        for col in ["StrategyHint", "DailyAction", "Trend"]:
            if col in df_show.columns:
                df_show.loc[~df_show["Ticker"].isin(["QQQ", "TQQQ"]), col] = "-"
        
        df_show = df_show.rename(columns={c: CN_COL_MAP.get(c, c) for c in df_show.columns})
        html_table = df_to_html_table(df_show)
        if html_table: blocks.append({"title": title, "html_table": html_table})

    env = Environment(loader=BaseLoader())
    html_content = env.from_string(HTML_TEMPLATE).render(
        title=cfg.get("title", "每日投资日报"), report_date=report_date, blocks=blocks,
        base_font_px=cfg.get("base_font_px", 14), table_font_px=cfg.get("table_font_px", 12)
    )

    with open(os.path.join(REPORT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ 网页版日报生成完毕 (public/index.html)！")

if __name__ == "__main__": main()