#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投资监控雷达 - 极致聚焦版 (修复版)
- 修复了雅虎财经批量下载数据格式错位导致的模块丢失问题
- 结构：核心指令 -> 宏观大盘 -> QQQ十大权重股(含PE/PS) -> 全球市场 -> 行业ETF -> 其他个股
"""

import os
import json
import csv
import time
from datetime import datetime
from dateutil import tz
import numpy as np
import pandas as pd
import yfinance as yf
from jinja2 import Environment, BaseLoader

LOCAL_TZ = tz.gettz("Asia/Shanghai")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(PROJECT_DIR, "public")

DEFAULT_CONFIG = {
    "title": "每日投资监控雷达",
    "history_days": 400,
    "ma_windows": [20, 50, 200],
    "base_font_px": 14,
    "table_font_px": 12
}

# 🟢 列名中文映射
CN_COL_MAP = {
    "Name": "名称", "Ticker": "代码", "Close": "收盘价", 
    "ChangePct": "日涨幅(%)", "Ret5D": "周涨幅(%)", 
    "Ret20D": "月涨幅(%)", "Ret250D": "年涨幅(%)",
    "VolRatio": "量比", "MA200": "MA200", "Trend": "趋势",
    "StrategyHint": "策略状态", "DailyAction": "今日操作",
    "PE_Trailing": "历史P/E", "PE_Forward": "远期P/E", "PS": "市销率(P/S)",
}

# 🚀 QQQ 前十大权重股字典（内置中文名称）
QQQ_TOP_10_MAP = {
    "NVDA": "英伟达",
    "AAPL": "苹果",
    "MSFT": "微软",
    "AMZN": "亚马逊",
    "META": "Meta(脸书)",
    "AVGO": "博通",
    "GOOGL": "谷歌(A类)",
    "GOOG": "谷歌(C类)",
    "TSLA": "特斯拉",
    "COST": "好市多"
}

def ensure_default_watchlist_csv(path: str):
    if os.path.exists(path): return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "market", "ticker", "name"])
        w.writerow(["strategy", "US", "QQQ", "纳斯达克100ETF"])
        w.writerow(["strategy", "US", "TQQQ", "纳斯达克三倍做多"])
        w.writerow(["risk", "US", "^VIX", "恐慌指数VIX"])

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
            
            market = (row.get("market") or "").strip() if has_market else "US"
            market = market.upper()
            if market in ("美股", "USA"): market = "US"
            elif market in ("港股", "HKG"): market = "HK"
            elif market in ("A股", "CHN", "CHINA"): market = "CN"
            elif market in ("日本", "JAPAN"): market = "JP"
            elif market in ("欧洲", "EUROPE", "EURO"): market = "EU"
            
            wl.setdefault(cat, [])
            wl[cat].append({"category": cat, "ticker": ticker, "name": name, "market": market})
    return wl

def fetch_valuation_data(tickers):
    """专门为核心个股抓取 P/E 和 P/S 数据"""
    results = {}
    for t in tickers:
        try:
            time.sleep(1) # 避开 Yahoo API 频率限制防封锁
            info = yf.Ticker(t).info
            results[t] = {
                "PE_Trailing": info.get('trailingPE', np.nan),
                "PE_Forward": info.get('forwardPE', np.nan),
                "PS": info.get('priceToSalesTrailing12Months', np.nan)
            }
        except:
            results[t] = {"PE_Trailing": np.nan, "PE_Forward": np.nan, "PS": np.nan}
    return results

def _safe_float(x):
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)): return np.nan
        if hasattr(x, "iloc"): x = x.iloc[0]
        return float(x)
    except: return np.nan

def fetch_and_calculate(ticker_map: dict, history_days: int = 400) -> pd.DataFrame:
    if not ticker_map: return pd.DataFrame()
    tickers = list(ticker_map.keys())
    fetch_list = list(set(tickers + ["^VIX"]))
    
    raw_data = None
    for attempt in range(3):
        try:
            # 🐞 核心修复：补回了 group_by="ticker" 参数，否则下方解析会全部跳过！
            raw_data = yf.download(fetch_list, period=f"{history_days}d", interval="1d", auto_adjust=False, group_by="ticker", threads=False, progress=False)
            if raw_data is not None and not raw_data.empty: break
            time.sleep(2)
        except: time.sleep(2)
            
    if raw_data is None or raw_data.empty: return pd.DataFrame()
    
    if "^VIX" in raw_data.columns.levels[0]: vix_df = raw_data["^VIX"]["Close"].dropna().sort_index()
    else: vix_df = pd.Series(dtype=float)

    rows = []
    for t in tickers:
        try:
            if len(fetch_list) == 1: df_t = raw_data
            else:
                if not hasattr(raw_data.columns, "levels") or t not in raw_data.columns.levels[0]: continue
                df_t = raw_data[t]

            df_t = df_t.dropna(subset=["Close"]).sort_index()
            if df_t.empty: continue
            close = df_t["Close"]
            vix_aligned = vix_df.reindex(close.index).ffill() if not vix_df.empty else pd.Series(np.nan, index=close.index)

            curr = _safe_float(close.iloc[-1])
            prev = _safe_float(close.iloc[-2]) if len(close) >= 2 else np.nan
            pct = (curr / prev - 1) * 100 if prev else np.nan
            curr_vix = _safe_float(vix_aligned.iloc[-1]) if len(vix_aligned) >= 1 else np.nan
            prev_vix = _safe_float(vix_aligned.iloc[-2]) if len(vix_aligned) >= 2 else np.nan

            ma200_series = close.rolling(200).mean()
            curr_ma200 = _safe_float(ma200_series.iloc[-1]) if len(close) >= 200 else np.nan
            prev_ma200 = _safe_float(ma200_series.iloc[-2]) if len(close) >= 201 else np.nan

            strategy_hint, daily_action = "", "观望"
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
                    prev_is_sell_state = prev_want_sell and not (_safe_float(vix_aligned.iloc[-3]) > 34.0 if len(vix_aligned)>=3 else False)
                    prev_is_buy_state = (prev > prev_ma200) or (_safe_float(vix_aligned.iloc[-3]) > 34.0 if len(vix_aligned)>=3 else False)

                    if is_buy_state and not prev_is_buy_state and not is_sell_state:
                        strategy_hint, daily_action = ("！VIX恐慌抄底" if want_buy_panic else "！突破MA200"), "买入"
                    elif is_sell_state and not prev_is_sell_state:
                        strategy_hint, daily_action = f"！跌穿且恐慌(VIX={curr_vix:.1f})", "卖出"
                    elif is_buy_state: strategy_hint = "均线之上 (持仓)"
                    elif is_sell_state: strategy_hint = "防线跌穿且高压 (空仓)"
                    elif curr < curr_ma200: strategy_hint = f"阴跌假摔 (VIX={curr_vix:.1f})"
                    else: strategy_hint = "MA200附近震荡"

            ret20 = (curr / close.iloc[-21] - 1) * 100 if len(close) > 21 else np.nan
            ret250 = (curr / close.iloc[-251] - 1) * 100 if len(close) > 251 else np.nan

            vol_ratio = np.nan
            if "Volume" in df_t.columns and len(df_t) >= 22:
                v_avg = df_t["Volume"].iloc[-21:-1].mean()
                if v_avg and v_avg > 0: vol_ratio = _safe_float(df_t["Volume"].iloc[-1]) / v_avg

            rows.append({
                "Name": ticker_map.get(t, t), "Ticker": t, "Close": curr, "ChangePct": pct,
                "VolRatio": vol_ratio, "StrategyHint": strategy_hint, "DailyAction": daily_action,
                "MA200": curr_ma200, "Ret20D": ret20, "Ret250D": ret250
            })
        except: continue
    return pd.DataFrame(rows)

def add_trend_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    out["Trend"] = out.apply(lambda r: "强" if not pd.isna(r.get("MA200")) and r["Close"] > r["MA200"] else "弱", axis=1)
    return out

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

            if is_buy and not prev_buy and not is_sell: txt, color = "🚨 【买入】触发买入信号！(突破MA200或VIX>34)", "#027A48"
            elif is_sell and not prev_sell: txt, color = "🚨 【卖出】触发卖出信号！(跌破MA200且VIX>22)", "#B42318"
            elif is_buy: txt, color = "✅ 【持有】建议持有做多。无新信号。", "#027A48"
            elif is_sell: txt, color = "🛑 【空仓】建议空仓吃息。无新信号。", "#B42318"
            elif latest["close"] < latest["ma200"]: txt, color = f"⚠️ 【观望】均线下阴跌(VIX {latest['vix']:.1f})。维持原仓", "#B8860B"
            else: txt, color = "⚠️ 【观望】处于均线胶着状态。", "#667085"
            html_rows += f"<tr><td style='padding:12px; border-bottom:1px solid #E4E7EC; font-weight:bold; width:20%;'>{tk}</td><td style='padding:12px; border-bottom:1px solid #E4E7EC; color:{color}; font-weight:bold;'>{txt}</td></tr>"
        return {"title": "🎯 今日核心动作 (QQQ & TQQQ)", "html_table": f"<table style='width:100%; border-collapse:collapse; text-align:left;'>{html_rows}</table>"}
    except: return None

def df_to_html_table(df: pd.DataFrame) -> str:
    if df is None or df.empty: return ""
    df2 = df.copy()
    def fmt_pct(x):
        if pd.isna(x): return "-"
        try: v = float(x)
        except: return str(x)
        cls = "pos" if v > 0 else ("neg" if v < 0 else "")
        return f'<span class="{cls}">{v:.2f}</span>' if cls else f"{v:.2f}"

    for col in ["日涨幅(%)", "周涨幅(%)", "月涨幅(%)", "年涨幅(%)"]:
        if col in df2.columns: df2[col] = df2[col].apply(fmt_pct)
        
    # 市销率 > 20 倍标红报警
    if "市销率(P/S)" in df2.columns:
        df2["市销率(P/S)"] = df2["市销率(P/S)"].apply(
            lambda x: f'<span class="neg" style="font-weight:bold;">{x:.1f}x</span>' if (not pd.isna(x) and x > 20) else (f"{x:.1f}x" if not pd.isna(x) else "-")
        )

    for c in ["收盘价", "量比", "MA200", "历史P/E", "远期P/E"]:
        if c in df2.columns: 
            df2[c] = df2[c].apply(lambda x: f"{x:.2f}x" if "P/E" in c and not pd.isna(x) else (f"{x:.2f}" if not pd.isna(x) else "-"))

    if "策略状态" in df2.columns:
        def _fmt_hint(x):
            s = str(x).strip() if x is not None else ""
            if "恐慌" in s or "突破" in s or "买点" in s: return f'<span class="pos">🟢 {s}</span>'
            if "砸穿" in s or "跌穿" in s or "高压" in s: return f'<span class="neg">🔴 {s}</span>'
            if "持仓" in s or "平稳" in s: return f'<span class="pos" style="opacity:0.8">{s}</span>'
            if "空仓" in s or "阴跌" in s: return f'<span class="neg" style="opacity:0.8">{s}</span>'
            if "收息" in s: return f'<span style="color:#B8860B; font-weight:bold;">{s}</span>' 
            return f'<span style="color:var(--muted)">{s}</span>'
        df2["策略状态"] = df2["策略状态"].apply(_fmt_hint)

    return df2.to_html(index=False, escape=False, border=0)

# ==========================================
# HTML 网页模板
# ==========================================
HTML_TEMPLATE = r"""
<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>{{ title }} - {{ report_date }}</title>
<style>
  :root{ --fg:#111827; --muted:#667085; --border:#E4E7EC; --head:#F9FAFB; --stripe:#F9FAFB; }
  body{ font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; font-size: {{ base_font_px }}px; line-height: 1.4; color:var(--fg); margin:0; padding:12px; background:#F2F4F7;}
  .wrap{ max-width: 850px; margin:0 auto; background:#fff; border-radius:12px; padding:16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}
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
  <div class="meta">{{ report_date }} | yfinance 驱动</div>
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

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    cfg = DEFAULT_CONFIG
    
    watchlist_path = os.path.join(PROJECT_DIR, "watchlist.csv")
    ensure_default_watchlist_csv(watchlist_path)
    wl = load_watchlist_from_csv(watchlist_path)
    
    all_items = []
    for items in wl.values(): all_items.extend(items)
    
    dt = datetime.now(tz=LOCAL_TZ)
    report_date = dt.strftime("%Y-%m-%d")
    blocks = []
    
    # 模块 1: 置顶核心操作
    action_block = get_today_action_block(["QQQ", "TQQQ"])
    if action_block: blocks.append(action_block)

    # 模块架构定义
    sections = [
        ("🎯 核心资产状态", 
         lambda it: it["ticker"] in ["QQQ", "TQQQ", "SGOV"], 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D", "MA200", "StrategyHint", "Trend"], "core"),
         
        ("🇺🇸 美国宏观大盘 & 风险锚", 
         lambda it: it["category"] in ["indices", "risk"] and it["market"] == "US" and it["ticker"] not in ["QQQ", "TQQQ", "SGOV"], 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D", "MA200"], "us_macro"),

        # 🚀 QQQ 前十大权重股估值透视模块
        ("🚀 QQQ 权重前十大股票 (估值透视)", 
         None, 
         ["Name", "Ticker", "Close", "ChangePct", "PS", "PE_Trailing", "PE_Forward", "Ret20D", "Ret250D", "MA200"], "qqq_top"),
         
        ("🌏 亚洲市场横向对比", 
         lambda it: it["category"] == "indices" and it["market"] in ["HK", "CN", "JP", "IN", "ASIA"], 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D"], "asia"),
         
        ("🇪🇺 欧洲及其他市场", 
         lambda it: it["category"] == "indices" and it["market"] in ["EU", "UK", "EUROPE", "GL", "OT"], 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D"], "eu"),
         
        ("📊 行业与主题 ETF", 
         lambda it: it["category"] == "sectors" and it["ticker"] not in ["QQQ", "TQQQ", "SGOV"], 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D", "MA200", "VolRatio"], "sectors"),
         
        # 🏢 自动排重：剔除已经在前十大里出现过的股票
        ("🏢 其他重点个股", 
         lambda it: it["category"] == "stocks" and it["ticker"] not in QQQ_TOP_10_MAP, 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D", "MA200", "VolRatio"], "stocks")
    ]

    for title, condition, cols, key in sections:
        if key == "qqq_top":
            # 专门拉取前十大的数据
            df = fetch_and_calculate(QQQ_TOP_10_MAP, cfg.get("history_days", 400))
            if df.empty: continue
            
            # 抓取 P/E 和 P/S
            val_data = fetch_valuation_data(list(QQQ_TOP_10_MAP.keys()))
            df["PS"] = np.nan
            df["PE_Trailing"] = np.nan
            df["PE_Forward"] = np.nan
            
            for t, vals in val_data.items():
                df.loc[df["Ticker"] == t, "PS"] = vals["PS"]
                df.loc[df["Ticker"] == t, "PE_Trailing"] = vals["PE_Trailing"]
                df.loc[df["Ticker"] == t, "PE_Forward"] = vals["PE_Forward"]
                
        else:
            items = [it for it in all_items if condition(it)]
            if not items: continue
            df = fetch_and_calculate({it["ticker"]: it["name"] for it in items}, cfg.get("history_days", 400))
        
        df = add_trend_flags(df)
        if df.empty: continue
        
        final_cols = [c for c in cols if c in df.columns]
        df_show = df[final_cols].copy()
        
        # 将日涨幅排在前面
        if "ChangePct" in df_show.columns: df_show = df_show.sort_values("ChangePct", ascending=False)
        
        # 如果是前十大，按固定顺序排列
        if key == "qqq_top":
            sorter = list(QQQ_TOP_10_MAP.keys())
            df_show['Ticker'] = pd.Categorical(df_show['Ticker'], categories=sorter, ordered=True)
            df_show = df_show.sort_values('Ticker')
            
        df_show = df_show.rename(columns={c: CN_COL_MAP.get(c, c) for c in df_show.columns})
        html_table = df_to_html_table(df_show)
        if html_table: blocks.append({"title": title, "html_table": html_table})

    env = Environment(loader=BaseLoader())
    html_content = env.from_string(HTML_TEMPLATE).render(
        title=cfg.get("title", "每日投资监控雷达"), report_date=report_date, blocks=blocks,
        base_font_px=cfg.get("base_font_px", 14), table_font_px=cfg.get("table_font_px", 12)
    )

    with open(os.path.join(REPORT_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ 终极网页版日报生成完毕 (public/index.html)！")

if __name__ == "__main__": main()
