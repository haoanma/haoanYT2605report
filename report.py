#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投资监控雷达 - 极致聚焦版 (Model 10: Prometheus 驱动)
- 信号引擎：
    买入：MA200突破 / VIX恐慌抄底(>34) / MA20防踏空接回 / VIX平息再入场(5日消退) / MA50夺回
    卖出：MA200+VIX22 清仓避险 / 动态Bias止盈(rolling 252日85th分位)
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
    "history_days": 500,  # 确保有足够数据计算 252日滚动分位数
    "ma_windows": [20, 50, 200],
    "base_font_px": 14,
    "table_font_px": 12
}

# 🟢 列名中文映射
CN_COL_MAP = {
    "Name": "名称", "Ticker": "代码", "Close": "收盘价", 
    "ChangePct": "日涨幅(%)", "Ret5D": "周涨幅(%)", 
    "Ret20D": "月涨幅(%)", "Ret250D": "年涨幅(%)",
    "VolRatio": "量比", "MA20": "MA20", "MA50": "MA50", "MA200": "MA200", "Trend": "趋势",
    "StrategyHint": "策略状态", "DailyAction": "今日操作",
    "PE_Trailing": "历史P/E", "PE_Forward": "远期P/E", "PS": "市销率(P/S)",
}

# 🚀 QQQ 前十大权重股字典（内置中文名称）
QQQ_TOP_10_MAP = {
    "NVDA": "英伟达", "AAPL": "苹果", "MSFT": "微软", "AMZN": "亚马逊",
    "META": "Meta(脸书)", "AVGO": "博通", "GOOGL": "谷歌(A类)", "GOOG": "谷歌(C类)",
    "TSLA": "特斯拉", "COST": "好市多"
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
    results = {}
    for t in tickers:
        try:
            time.sleep(1)
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

# ==========================================
# 📊 通用大表策略判定 (Model 10)
# ==========================================
def fetch_and_calculate(ticker_map: dict, history_days: int = 500) -> pd.DataFrame:
    if not ticker_map: return pd.DataFrame()
    tickers = list(ticker_map.keys())
    fetch_list = list(set(tickers + ["^VIX"]))
    
    raw_data = None
    for attempt in range(3):
        try:
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
            if len(df_t) < 201: continue
            close = df_t["Close"]
            vix_aligned = vix_df.reindex(close.index).ffill() if not vix_df.empty else pd.Series(np.nan, index=close.index)

            curr = _safe_float(close.iloc[-1])
            prev = _safe_float(close.iloc[-2])
            pct = (curr / prev - 1) * 100 if prev else np.nan
            
            curr_vix = _safe_float(vix_aligned.iloc[-1]) if len(vix_aligned) >= 1 else np.nan
            prev_vix = _safe_float(vix_aligned.iloc[-2]) if len(vix_aligned) >= 2 else np.nan

            ma20 = close.rolling(20).mean()
            ma50 = close.rolling(50).mean()
            ma200 = close.rolling(200).mean()
            
            curr_ma20 = _safe_float(ma20.iloc[-1])
            curr_ma50 = _safe_float(ma50.iloc[-1])
            curr_ma200 = _safe_float(ma200.iloc[-1])
            prev_ma200 = _safe_float(ma200.iloc[-2])

            # Model 10 动态偏离度
            bias = (close / ma200) - 1.0
            dynamic_threshold = (bias.rolling(252, min_periods=126).quantile(0.85) * 0.90).clip(lower=0.40)
            
            curr_bias = _safe_float(bias.iloc[-1])
            curr_dyn_thresh = _safe_float(dynamic_threshold.iloc[-1])
            is_overextended = curr_bias > curr_dyn_thresh

            strategy_hint, daily_action = "", "观望"
            if not np.isnan(prev) and not np.isnan(curr_ma200):
                if t == "^VIX":
                    if curr > 34: strategy_hint, daily_action = "🔴 极度恐慌 (抄底)", "买点出现"
                    elif curr > 22: strategy_hint, daily_action = "🟠 高压警戒 (破均线卖出)", "警戒区域"
                    else: strategy_hint, daily_action = "🟢 情绪平稳", "安全期"
                elif t in ["SGOV", "SHV", "BIL"]:
                    strategy_hint, daily_action = "💰 空仓收息", "稳定收息"
                else:
                    # --- Model 10 Prometheus 信号判定 ---
                    # 辅助数据：5日前快照
                    vix_5d_ago   = _safe_float(vix_aligned.iloc[-6]) if len(vix_aligned) >= 6 else np.nan
                    close_5d_ago = _safe_float(close.iloc[-6])        if len(close) >= 6    else np.nan
                    prev_ma200_val = _safe_float(ma200.iloc[-2])
                    prev_ma20_val  = _safe_float(ma20.iloc[-2])
                    prev_ma50_val  = _safe_float(ma50.iloc[-2])
                    prev_vix       = _safe_float(vix_aligned.iloc[-2]) if len(vix_aligned) >= 2 else np.nan
                    prev_close     = _safe_float(close.iloc[-2])

                    ma200_rising      = (curr_ma200 > _safe_float(ma200.iloc[-6])) if len(ma200) >= 6 else False
                    recent_min        = float(close.iloc[-6:-1].min()) if len(close) >= 6 else curr
                    recently_dipped   = recent_min < curr_ma50

                    # 卖出
                    sell_normal  = (curr < curr_ma200) and (curr_vix > 22.0) and not ((prev_close < prev_ma200_val) and (prev_vix > 22.0))
                    p_overext    = (_safe_float(bias.iloc[-2]) > _safe_float(dynamic_threshold.iloc[-2]))
                    sell_profit  = is_overextended and (curr < curr_ma20) and not (p_overext and (prev_close < prev_ma20_val))

                    # 买入
                    buy_ma200    = (prev_close <= prev_ma200_val) and (curr > curr_ma200)
                    buy_vix_panic = (curr <= curr_ma200) and (curr_vix > 34.0) and (prev_vix <= 34.0)
                    buy_reentry  = (prev_close <= prev_ma20_val) and (curr > curr_ma20) and (curr > curr_ma200)
                    vix_calming  = (not np.isnan(vix_5d_ago)) and (vix_5d_ago > 30.0) and (curr_vix < 25.0)
                    price_stable = (not np.isnan(close_5d_ago)) and (curr > close_5d_ago)
                    p_vix_calm   = (not np.isnan(vix_5d_ago)) and (vix_5d_ago > 30.0) and (prev_vix < 25.0) and (prev_close > close_5d_ago)
                    buy_vix_calm = vix_calming and price_stable and (curr > curr_ma50) and not p_vix_calm
                    buy_ma50     = (prev_close <= prev_ma50_val) and (curr > curr_ma50) and ma200_rising and recently_dipped and (curr > curr_ma200)

                    is_sell_state = sell_normal or sell_profit
                    is_buy_state  = (curr > curr_ma200) or (curr_vix > 34.0)

                    if is_sell_state:
                        strategy_hint, daily_action = ("🚨 极值止盈" if sell_profit else "🚨 跌破防线(VIX高企)"), "卖出"
                    elif buy_vix_panic:
                        strategy_hint, daily_action = "🔥 VIX极度恐慌抄底", "买入"
                    elif buy_vix_calm:
                        strategy_hint, daily_action = "🕊️ VIX平息重入(5日消退)", "买入"
                    elif buy_ma200:
                        strategy_hint, daily_action = "🚀 突破MA200长牛起航", "买入"
                    elif buy_ma50:
                        strategy_hint, daily_action = "📈 MA50夺回(回调结束)", "买入"
                    elif buy_reentry:
                        strategy_hint, daily_action = "↩️ MA20接回(防踏空)", "买入"
                    elif is_buy_state:
                        strategy_hint = "均线之上 (持仓)"
                    else:
                        strategy_hint = f"阴跌空仓 (VIX={curr_vix:.1f})"

            ret20 = (curr / close.iloc[-21] - 1) * 100 if len(close) > 21 else np.nan
            ret250 = (curr / close.iloc[-251] - 1) * 100 if len(close) > 251 else np.nan

            vol_ratio = np.nan
            if "Volume" in df_t.columns and len(df_t) >= 22:
                v_avg = df_t["Volume"].iloc[-21:-1].mean()
                if v_avg and v_avg > 0: vol_ratio = _safe_float(df_t["Volume"].iloc[-1]) / v_avg

            rows.append({
                "Name": ticker_map.get(t, t), "Ticker": t, "Close": curr, "ChangePct": pct,
                "VolRatio": vol_ratio, "StrategyHint": strategy_hint, "DailyAction": daily_action,
                "MA20": curr_ma20, "MA50": curr_ma50, "MA200": curr_ma200, "Ret20D": ret20, "Ret250D": ret250
            })
        except: continue
    return pd.DataFrame(rows)

def add_trend_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    out = df.copy()
    out["Trend"] = out.apply(lambda r: "强" if not pd.isna(r.get("MA200")) and r["Close"] > r["MA200"] else "弱", axis=1)
    return out

# ==========================================
# 🧠 核心舱：Model 10 Prometheus 状态机
# ==========================================
def get_today_action_block(tickers=["QQQ", "TQQQ"]):
    try:
        fetch_list = tickers + ["^VIX"]
        df_raw = yf.download(fetch_list, period="2y", auto_adjust=False, group_by="ticker", threads=False, progress=False)
        if df_raw.empty: return None

        if "^VIX" in df_raw.columns.levels[0]:
            vix_series = df_raw["^VIX"]["Close"].dropna()
        else: return None

        html_rows = ""
        for tk in tickers:
            if tk not in df_raw.columns.levels[0]: continue
            close = df_raw[tk]["Close"].dropna()
            if len(close) < 252: continue

            df = pd.DataFrame({"close": close})
            df["ma20"] = df["close"].rolling(20).mean()
            df["ma50"] = df["close"].rolling(50).mean()
            df["ma200"] = df["close"].rolling(200).mean()
            df["vix"] = vix_series.reindex(df.index).ffill()
            df = df.dropna()
            if len(df) < 2: continue

            # Model 10 动态偏离度计算 (252日分位数)
            bias = (df["close"] / df["ma200"]) - 1.0
            dynamic_threshold = (bias.rolling(252, min_periods=126).quantile(0.85) * 0.90).clip(lower=0.40)
            
            curr = df.iloc[-1]
            prev = df.iloc[-2]
            
            curr_bias = bias.iloc[-1]
            curr_dyn_thresh = dynamic_threshold.iloc[-1]
            is_overextended = curr_bias > curr_dyn_thresh

            # --- 辅助数据：5日前快照（用于 VIX 平息 & MA50 夺回信号）---
            row_5d = df.iloc[-6] if len(df) >= 6 else None
            vix_5d_ago   = row_5d["vix"]   if row_5d is not None else np.nan
            close_5d_ago = row_5d["close"] if row_5d is not None else np.nan
            ma200_5d_ago = row_5d["ma200"] if row_5d is not None else np.nan

            # MA200 斜率向上（5日对比）
            ma200_rising = curr["ma200"] > ma200_5d_ago if not np.isnan(ma200_5d_ago) else False

            # 5日内最低收盘是否曾低于 MA50（确认"确实发生过回调"）
            recent_min = df["close"].iloc[-6:-1].min() if len(df) >= 6 else curr["close"]
            recently_dipped_ma50 = recent_min < curr["ma50"]

            # 边缘触发器 (Edge Triggers)
            cross_up_ma200 = (prev["close"] <= prev["ma200"]) and (curr["close"] > curr["ma200"])
            cross_up_ma20  = (prev["close"] <= prev["ma20"])  and (curr["close"] > curr["ma20"])
            cross_up_ma50  = (prev["close"] <= prev["ma50"])  and (curr["close"] > curr["ma50"])
            cross_down_ma20 = (prev["close"] >= prev["ma20"]) and (curr["close"] < curr["ma20"])

            # --- 卖出逻辑 (Sell Absolute Priority) ---
            # 盾1: 正常避险 (跌破年线 且 VIX > 22)
            c_sell_normal = (curr["close"] < curr["ma200"]) and (curr["vix"] > 22.0)
            p_sell_normal = (prev["close"] < prev["ma200"]) and (prev["vix"] > 22.0)
            sell_normal = c_sell_normal and not p_sell_normal

            # 盾2: 动态止盈 (处于超买期，且跌破 20日线)
            c_sell_profit = is_overextended and (curr["close"] < curr["ma20"])
            p_sell_profit = (bias.iloc[-2] > dynamic_threshold.iloc[-2]) and (prev["close"] < prev["ma20"])
            sell_profit = c_sell_profit and not p_sell_profit

            # --- 买入逻辑 ---
            # 刀1: MA200 收复（趋势右侧买）
            buy_ma200 = cross_up_ma200

            # 刀2: 左侧恐慌飞刀 (VIX 每次重新突破 34)
            buy_vix_panic = (curr["close"] <= curr["ma200"]) and (curr["vix"] > 34.0) and (prev["vix"] <= 34.0)

            # 刀3: MA20 防踏空接回（止盈后价格重新站上 MA20，仍在 MA200 之上）
            buy_reentry = cross_up_ma20 and (curr["close"] > curr["ma200"])

            # 刀4: VIX 平息再入场（5日前 VIX>30，现在<25，且价格企稳回升）
            #   比原版"VIX<20"更敏感，能在极端恐慌消退时提前介入
            vix_calming    = (not np.isnan(vix_5d_ago)) and (vix_5d_ago > 30.0) and (curr["vix"] < 25.0)
            price_stable   = (not np.isnan(close_5d_ago)) and (curr["close"] > close_5d_ago)
            p_vix_calming  = (not np.isnan(vix_5d_ago)) and (vix_5d_ago > 30.0) and (prev["vix"] < 25.0) and (prev["close"] > close_5d_ago)
            buy_vix_calm   = vix_calming and price_stable and (curr["close"] > curr["ma50"]) and not p_vix_calming

            # 刀5: MA50 夺回买入（牛市回调后重新站上 MA50，MA200 仍向上）
            buy_ma50 = cross_up_ma50 and ma200_rising and recently_dipped_ma50 and (curr["close"] > curr["ma200"])

            # 状态判定（卖出优先；买入按信号强度排列）
            if sell_normal or sell_profit:
                if sell_normal:
                    signal, color = "🚨 盾1：清仓避险", "#B42318"
                    desc = f"<b>衰退确认！</b>跌破年线且 VIX 高压 ({curr['vix']:.1f})"
                else:
                    signal, color = "🚨 盾2：动态止盈", "#B42318"
                    desc = f"<b>高位超买！</b>偏离度({curr_bias*100:.1f}%) 突破动态阈值({curr_dyn_thresh*100:.1f}%) 且跌穿 MA20"

            elif buy_vix_panic or buy_vix_calm or buy_ma200 or buy_ma50 or buy_reentry:
                if buy_vix_panic:
                    signal, color = "🔥 刀2：左侧抄底", "#027A48"
                    desc = f"<b>极度恐慌！</b>迎着暴跌 (VIX {curr['vix']:.1f}) 抢夺带血筹码"
                elif buy_vix_calm:
                    signal, color = "🕊️ 刀4：VIX平息重入", "#027A48"
                    desc = f"<b>恐慌消退！</b>VIX 从高位({vix_5d_ago:.1f})回落至 {curr['vix']:.1f}，价格企稳 MA50，提前介入"
                elif buy_ma200:
                    signal, color = "🚀 刀1：右侧顺势", "#027A48"
                    desc = "<b>长牛起航！</b>向上有效突破 MA200 牛熊分界线"
                elif buy_ma50:
                    signal, color = "📈 刀5：MA50夺回", "#027A48"
                    desc = f"<b>回调结束！</b>牛市健康回调后重新站上 MA50，MA200 趋势向上"
                else:
                    signal, color = "↩️ 刀3：MA20接回", "#027A48"
                    desc = "<b>防踏空！</b>止盈后价格重新站上 MA20，续仓跟牛"
                    
            else:
                if curr["close"] > curr["ma200"]:
                    color_bias = "#B42318" if is_overextended else "#B8860B"
                    signal, color = "🛡️ 多头死拿", color_bias
                    alert = "⚠️ 处于超买区" if is_overextended else "安全持仓"
                    desc = f"均线之上耐心持仓 | 当前偏离度: {curr_bias*100:.1f}% (动态警戒线: {curr_dyn_thresh*100:.1f}%) - {alert}"
                else:
                    signal, color = "💤 空仓吃息", "#667085"
                    desc = f"身处熊市左侧，耐心等待系统拔刀 (当前 VIX: {curr['vix']:.1f})"

            html_rows += f"""
            <tr>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; font-size:14px; font-weight:900; width:15%;'>{tk}</td>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; color:{color}; font-weight:bold; width:22%;'>{signal}</td>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; color:#475467; font-size:12px;'>{desc}</td>
            </tr>
            """
            
        return {"title": "🧠 核心舱交易大脑 (Model 10: Prometheus)", "html_table": f"<table style='width:100%; border-collapse:collapse; text-align:left;'>{html_rows}</table>"}
    except Exception as e: 
        print(f"核心操作块解析失败: {e}")
        return None

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
        
    if "市销率(P/S)" in df2.columns:
        df2["市销率(P/S)"] = df2["市销率(P/S)"].apply(
            lambda x: f'<span class="neg" style="font-weight:bold;">{x:.1f}x</span>' if (not pd.isna(x) and x > 20) else (f"{x:.1f}x" if not pd.isna(x) else "-")
        )

    for c in ["收盘价", "量比", "MA20", "MA50", "MA200", "历史P/E", "远期P/E"]:
        if c in df2.columns: 
            df2[c] = df2[c].apply(lambda x: f"{x:.2f}x" if "P/E" in c and not pd.isna(x) else (f"{x:.2f}" if not pd.isna(x) else "-"))

    if "策略状态" in df2.columns:
        def _fmt_hint(x):
            s = str(x).strip() if x is not None else ""
            if "恐慌" in s or "突破" in s or "买点" in s or "重入" in s: return f'<span class="pos">🟢 {s}</span>'
            if "止盈" in s or "跌破" in s or "高压" in s: return f'<span class="neg">🔴 {s}</span>'
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
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D", "MA20", "MA50", "MA200", "StrategyHint", "Trend"], "core"),
         
        ("🇺🇸 美国宏观大盘 & 风险锚", 
         lambda it: it["category"] in ["indices", "risk"] and it["market"] == "US" and it["ticker"] not in ["QQQ", "TQQQ", "SGOV"], 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D", "MA20", "MA200"], "us_macro"),

        ("🚀 QQQ 权重前十大股票 (估值透视)", 
         None, 
         ["Name", "Ticker", "Close", "ChangePct", "PS", "PE_Trailing", "PE_Forward", "MA20", "MA200", "Ret250D"], "qqq_top"),
         
        ("🌏 亚洲市场横向对比", 
         lambda it: it["category"] == "indices" and it["market"] in ["HK", "CN", "JP", "IN", "ASIA"], 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D"], "asia"),
         
        ("🇪🇺 欧洲及其他市场", 
         lambda it: it["category"] == "indices" and it["market"] in ["EU", "UK", "EUROPE", "GL", "OT"], 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D"], "eu"),
         
        ("📊 行业与主题 ETF", 
         lambda it: it["category"] == "sectors" and it["ticker"] not in ["QQQ", "TQQQ", "SGOV"], 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D", "MA20", "MA200", "VolRatio"], "sectors"),
         
        ("🏢 其他重点个股", 
         lambda it: it["category"] == "stocks" and it["ticker"] not in QQQ_TOP_10_MAP, 
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D", "MA20", "MA200", "VolRatio"], "stocks")
    ]

    for title, condition, cols, key in sections:
        if key == "qqq_top":
            df = fetch_and_calculate(QQQ_TOP_10_MAP, cfg.get("history_days", 500))
            if df.empty: continue
            
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
            df = fetch_and_calculate({it["ticker"]: it["name"] for it in items}, cfg.get("history_days", 500))
        
        df = add_trend_flags(df)
        if df.empty: continue
        
        final_cols = [c for c in cols if c in df.columns]
        df_show = df[final_cols].copy()
        
        if "ChangePct" in df_show.columns: df_show = df_show.sort_values("ChangePct", ascending=False)
        
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
    print("✅ Model 10 Prometheus 网页版日报生成完毕 (public/index.html)！")

if __name__ == "__main__": main()
