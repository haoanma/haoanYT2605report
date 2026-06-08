#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投资监控雷达 - 极致聚焦版
- QQQ  信号引擎：Model 08 Diamond Hands
    买入：MA200突破 / VIX>34恐慌抄底 / MA20防踏空接回
    卖出：MA200+VIX22 清仓 / 固定Bias>64%止盈
- TQQQ 信号引擎：Model 08 基础 + Apollo VIX百分位体制叠加
    额外卖出：高恐慌体制(VIX>过去1年75th百分位) 且 跌破MA50 → 提前离场
    额外买入：低恐慌体制(VIX<25th百分位)接回用MA20；高恐慌体制接回要MA50确认
    止盈：动态Bias阈值(252日85th百分位) 替代固定64%
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
# 🔧 Apollo 信号计算辅助函数 (仅用于 TQQQ)
# ==========================================
def _compute_apollo_signals(close: pd.Series, vix: pd.Series, ma20: pd.Series,
                             ma50: pd.Series, ma200: pd.Series):
    """返回 Apollo 模型在序列末端的当前信号字典。"""
    bias        = (close / ma200) - 1.0
    bias_th     = bias.rolling(252, min_periods=63).quantile(0.85)
    is_oe_s     = (bias > bias_th).rolling(10, min_periods=1).max().fillna(False).astype(bool)
    peak_bias10 = bias.rolling(10, min_periods=1).max()   # 近10日Bias峰值，用于止盈描述

    vix_p75  = vix.rolling(252, min_periods=63).quantile(0.75)
    vix_p25  = vix.rolling(252, min_periods=63).quantile(0.25)
    high_fear = vix > vix_p75
    low_fear  = vix < vix_p25

    def edge(cond):
        c = cond.fillna(False).astype(bool)
        return c & (~c.shift(1, fill_value=False))

    # 买入
    bt  = edge(close > ma200)
    bv  = edge((vix > 34.0) & (close <= ma200))
    brc = edge((close > ma20)  & (close > ma200) & low_fear)   # 低恐慌: MA20
    brf = edge((close > ma50)  & (close > ma200) & (~low_fear)) # 非低恐慌: MA50

    # 卖出
    sff = edge(high_fear & (close < ma50))                     # Apollo新增: 高恐提前出
    sn  = edge((close < ma200) & (vix > 20.0))
    sp  = edge(is_oe_s & (close < ma20))

    raw_buy  = bt | bv | brc | brf
    raw_sell = sff | sn | sp
    _st = pd.Series(np.nan, index=close.index)
    _st.loc[raw_buy & (~raw_sell)] = 1.0
    _st.loc[raw_sell]              = 0.0
    is_buy_state = bool(_st.ffill().fillna(0.0).astype(bool).iloc[-1])

    curr_vix      = float(vix.iloc[-1])
    curr_p75      = float(vix_p75.iloc[-1]) if not np.isnan(vix_p75.iloc[-1]) else 30.0
    curr_p25      = float(vix_p25.iloc[-1]) if not np.isnan(vix_p25.iloc[-1]) else 15.0
    curr_bias     = float(bias.iloc[-1])
    curr_bth      = float(bias_th.iloc[-1]) if not np.isnan(bias_th.iloc[-1]) else 0.85
    curr_peak_b10 = float(peak_bias10.iloc[-1])

    regime = ("高恐慌" if bool(high_fear.iloc[-1]) else
              ("低恐慌" if bool(low_fear.iloc[-1]) else "正常"))

    return {
        "is_buy_state":       is_buy_state,
        "sell_fear_fast":     bool(sff.iloc[-1]),
        "sell_normal":        bool(sn.iloc[-1]),
        "sell_profit":        bool(sp.iloc[-1]),
        "buy_trend":          bool(bt.iloc[-1]),
        "buy_vix_panic":      bool(bv.iloc[-1]),
        "buy_reentry_calm":   bool(brc.iloc[-1]),
        "buy_reentry_fear":   bool(brf.iloc[-1]),
        "is_overextended":    bool(is_oe_s.iloc[-1]),
        "curr_bias":          curr_bias,
        "curr_bias_th":       curr_bth,
        "curr_peak_bias10":   curr_peak_b10,
        "curr_vix":           curr_vix,
        "vix_p75":            curr_p75,
        "vix_p25":            curr_p25,
        "regime":             regime,
    }


# ==========================================
# 📊 通用大表策略判定
#    QQQ 及其他: Model 08 Diamond Hands
#    TQQQ:       Model 08 基础 + Apollo 叠加
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

            ma20  = close.rolling(20).mean()
            ma50  = close.rolling(50).mean()
            ma200 = close.rolling(200).mean()

            curr_ma20  = _safe_float(ma20.iloc[-1])
            curr_ma50  = _safe_float(ma50.iloc[-1])
            curr_ma200 = _safe_float(ma200.iloc[-1])

            strategy_hint, daily_action = "", "观望"

            if not np.isnan(prev) and not np.isnan(curr_ma200):
                if t == "^VIX":
                    if curr > 34: strategy_hint, daily_action = "🔴 极度恐慌 (抄底)", "买点出现"
                    elif curr > 22: strategy_hint, daily_action = "🟠 高压警戒 (破均线卖出)", "警戒区域"
                    else: strategy_hint, daily_action = "🟢 情绪平稳", "安全期"

                elif t in ["SGOV", "SHV", "BIL"]:
                    strategy_hint, daily_action = "💰 空仓收息", "稳定收息"

                elif t == "TQQQ":
                    # ── Apollo 信号 ──
                    ap = _compute_apollo_signals(close, vix_aligned, ma20, ma50, ma200)
                    regime_tag = {"高恐慌": "🔴", "低恐慌": "🟢", "正常": "🟡"}.get(ap["regime"], "🟡")
                    regime_str = f"{regime_tag}{ap['regime']}体制 VIX={ap['curr_vix']:.1f}(p75={ap['vix_p75']:.1f})"

                    if ap["sell_fear_fast"]:
                        strategy_hint = f"⚡ Apollo高恐预警卖出 | {regime_str}"
                        daily_action  = "卖出"
                    elif ap["sell_profit"]:
                        strategy_hint = f"🚨 动态止盈(Bias>{ap['curr_bias_th']*100:.0f}%) | {regime_str}"
                        daily_action  = "卖出"
                    elif ap["sell_normal"]:
                        strategy_hint = f"🚨 跌破MA200+VIX高企 | {regime_str}"
                        daily_action  = "卖出"
                    elif ap["buy_vix_panic"]:
                        strategy_hint = f"🔥 VIX极度恐慌抄底 | {regime_str}"
                        daily_action  = "买入"
                    elif ap["buy_trend"]:
                        strategy_hint = f"🚀 突破MA200右侧 | {regime_str}"
                        daily_action  = "买入"
                    elif ap["buy_reentry_calm"]:
                        strategy_hint = f"↩️ 低恐接回(MA20) | {regime_str}"
                        daily_action  = "买入"
                    elif ap["buy_reentry_fear"]:
                        strategy_hint = f"↩️ 压力期接回(MA50) | {regime_str}"
                        daily_action  = "买入"
                    elif ap["is_buy_state"]:
                        bias_warn = f" ⚠️Bias={ap['curr_bias']*100:.0f}%>阈值{ap['curr_bias_th']*100:.0f}%" if ap["is_overextended"] else f" Bias={ap['curr_bias']*100:.0f}%"
                        strategy_hint = f"持仓中{bias_warn} | {regime_str}"
                    else:
                        strategy_hint = f"空仓等信号 | {regime_str}"

                else:
                    # ── Model 08 Diamond Hands (其他品种) ──
                    bias    = (close / ma200) - 1.0
                    is_oe_s = (bias > 0.64).rolling(10, min_periods=1).max().astype(bool)
                    curr_bias      = _safe_float(bias.iloc[-1])
                    is_overextended = bool(is_oe_s.iloc[-1])

                    _c_bt = close > ma200
                    _bt   = _c_bt & (~_c_bt.shift(1).fillna(False))
                    _c_bv = (vix_aligned > 34.0) & (close <= ma200)
                    _bv   = _c_bv & (~_c_bv.shift(1).fillna(False))
                    _c_br = (close > ma20) & (close > ma200)
                    _br   = _c_br & (~_c_br.shift(1).fillna(False))
                    _c_sn = (close < ma200) & (vix_aligned > 22.0)
                    _sn   = _c_sn & (~_c_sn.shift(1).fillna(False))
                    _c_sp = is_oe_s & (close < ma20)
                    _sp   = _c_sp & (~_c_sp.shift(1).fillna(False))
                    _raw_buy  = _bt | _bv | _br
                    _raw_sell = _sn | _sp
                    _st = pd.Series(np.nan, index=close.index)
                    _st.loc[_raw_buy & (~_raw_sell)] = 1.0
                    _st.loc[_raw_sell]               = 0.0
                    is_buy_state = bool(_st.ffill().fillna(0.0).astype(bool).iloc[-1])

                    sell_normal   = bool(_sn.iloc[-1])
                    sell_profit   = bool(_sp.iloc[-1])
                    buy_ma200     = bool(_bt.iloc[-1])
                    buy_vix_panic = bool(_bv.iloc[-1])
                    buy_reentry   = bool(_br.iloc[-1])

                    if sell_profit or sell_normal:
                        strategy_hint, daily_action = ("🚨 极值止盈" if sell_profit else "🚨 跌破防线(VIX高企)"), "卖出"
                    elif buy_vix_panic:
                        strategy_hint, daily_action = "🔥 VIX极度恐慌抄底", "买入"
                    elif buy_ma200:
                        strategy_hint, daily_action = "🚀 突破MA200长牛起航", "买入"
                    elif buy_reentry:
                        strategy_hint, daily_action = "↩️ MA20接回(防踏空)", "买入"
                    elif is_buy_state:
                        strategy_hint = "均线之上 (持仓)"
                    else:
                        strategy_hint = f"阴跌空仓 (VIX={curr_vix:.1f})"

            ret20  = (curr / close.iloc[-21]  - 1) * 100 if len(close) > 21  else np.nan
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
# 🧠 核心舱：QQQ → Model 08 | TQQQ → Apollo
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

            vix_aligned = vix_series.reindex(close.index).ffill().dropna()
            common_idx  = close.index.intersection(vix_aligned.index)
            close       = close.loc[common_idx]
            vix_aligned = vix_aligned.loc[common_idx]
            if len(close) < 2: continue

            ma20  = close.rolling(20).mean()
            ma50  = close.rolling(50).mean()
            ma200 = close.rolling(200).mean()

            if tk == "TQQQ":
                # ══ Apollo 信号 ══════════════════════════════════════
                ap = _compute_apollo_signals(close, vix_aligned, ma20, ma50, ma200)

                regime_color = {"高恐慌": "#B42318", "低恐慌": "#027A48", "正常": "#B8860B"}
                rc = regime_color.get(ap["regime"], "#B8860B")
                regime_badge = (
                    f"<span style='background:{rc};color:#fff;padding:2px 6px;"
                    f"border-radius:4px;font-size:11px;'>{ap['regime']}体制</span>"
                )
                vix_info = (
                    f"VIX <b>{ap['curr_vix']:.1f}</b> "
                    f"(1年区间 p25={ap['vix_p25']:.1f} / p75={ap['vix_p75']:.1f})"
                )
                bias_info = (
                    f"Bias <b>{ap['curr_bias']*100:.1f}%</b> "
                    f"/ 动态止盈线 {ap['curr_bias_th']*100:.1f}%"
                )

                if ap["sell_fear_fast"]:
                    signal = "⚡ Apollo盾0：高恐预警清仓"
                    color  = "#B42318"
                    desc   = (f"<b>高恐慌体制下跌破MA50！</b> {regime_badge} 提前离场，不等MA200破位。<br>"
                              f"{vix_info} | {bias_info}")
                elif ap["sell_profit"]:
                    signal = "🚨 Apollo盾2：动态止盈"
                    color  = "#B42318"
                    desc   = (f"<b>近10日曾超买(峰值Bias {ap['curr_peak_bias10']*100:.1f}% > 阈值{ap['curr_bias_th']*100:.1f}%)，今跌穿MA20！</b> {regime_badge}<br>"
                              f"当前Bias {ap['curr_bias']*100:.1f}% | {vix_info}")
                elif ap["sell_normal"]:
                    signal = "🚨 Apollo盾1：清仓避险"
                    color  = "#B42318"
                    desc   = (f"<b>跌破MA200且VIX高企！</b> {regime_badge}<br>"
                              f"{vix_info} | {bias_info}")
                elif ap["buy_vix_panic"]:
                    signal = "🔥 Apollo刀2：左侧抄底"
                    color  = "#027A48"
                    desc   = (f"<b>极度恐慌（VIX>{ap['curr_vix']:.0f}）</b>且跌破MA200，迎着暴跌建仓。 {regime_badge}<br>"
                              f"{vix_info}")
                elif ap["buy_trend"]:
                    signal = "🚀 Apollo刀1：右侧顺势"
                    color  = "#027A48"
                    desc   = (f"<b>向上突破MA200！</b>长牛起航。 {regime_badge}<br>"
                              f"{vix_info} | {bias_info}")
                elif ap["buy_reentry_calm"]:
                    signal = "↩️ Apollo刀3：低恐接回(MA20)"
                    color  = "#027A48"
                    desc   = (f"<b>低恐慌体制，MA20站上接回。</b> {regime_badge}<br>"
                              f"VIX处于历史低位，宽松条件再入场。 {vix_info}")
                elif ap["buy_reentry_fear"]:
                    signal = "↩️ Apollo刀3：压力期接回(MA50)"
                    color  = "#027A48"
                    desc   = (f"<b>非低恐慌体制，需MA50确认再入场。</b> {regime_badge}<br>"
                              f"比低恐慌期更严格，防止死猫反弹接刀。 {vix_info}")
                else:
                    if ap["is_buy_state"]:
                        color = "#B42318" if ap["is_overextended"] else "#B8860B"
                        signal = "🛡️ Apollo多头持仓"
                        oe_warn = " ⚠️超买区" if ap["is_overextended"] else ""
                        desc = (f"均线之上持仓{oe_warn}。 {regime_badge}<br>"
                                f"{bias_info} | {vix_info}")
                    else:
                        signal, color = "💤 Apollo空仓吃息", "#667085"
                        desc = (f"熊市等待信号。 {regime_badge}<br>"
                                f"{vix_info} | {bias_info}")

                model_tag = "<span style='font-size:10px;color:#667085;'>Apollo</span>"

            else:
                # ══ Model 08 Diamond Hands (QQQ 及其他) ══════════════
                bias_s  = (close / ma200) - 1.0
                is_oe_s = (bias_s > 0.64).rolling(10, min_periods=1).max().astype(bool)
                curr_bias = float(bias_s.iloc[-1])
                is_overextended = bool(is_oe_s.iloc[-1])

                _c_bt = close > ma200
                _bt   = _c_bt & (~_c_bt.shift(1).fillna(False))
                _c_bv = (vix_aligned > 34.0) & (close <= ma200)
                _bv   = _c_bv & (~_c_bv.shift(1).fillna(False))
                _c_br = (close > ma20) & (close > ma200)
                _br   = _c_br & (~_c_br.shift(1).fillna(False))
                _c_sn = (close < ma200) & (vix_aligned > 22.0)
                _sn   = _c_sn & (~_c_sn.shift(1).fillna(False))
                _c_sp = is_oe_s & (close < ma20)
                _sp   = _c_sp & (~_c_sp.shift(1).fillna(False))
                _raw_buy  = _bt | _bv | _br
                _raw_sell = _sn | _sp
                _st = pd.Series(np.nan, index=close.index)
                _st.loc[_raw_buy & (~_raw_sell)] = 1.0
                _st.loc[_raw_sell]               = 0.0
                is_buy_state = bool(_st.ffill().fillna(0.0).astype(bool).iloc[-1])

                curr_vix = float(vix_aligned.iloc[-1])

                sell_normal   = bool(_sn.iloc[-1])
                sell_profit   = bool(_sp.iloc[-1])
                buy_ma200     = bool(_bt.iloc[-1])
                buy_vix_panic = bool(_bv.iloc[-1])
                buy_reentry   = bool(_br.iloc[-1])

                if sell_normal or sell_profit:
                    if sell_normal:
                        signal, color = "🚨 盾1：清仓避险", "#B42318"
                        desc = f"<b>衰退确认！</b>跌破年线且 VIX 高压 ({curr_vix:.1f})"
                    else:
                        signal, color = "🚨 盾2：极值止盈", "#B42318"
                        desc = f"<b>高位超买！</b>偏离度({curr_bias*100:.1f}%) 超过64%固定阈值且跌穿 MA20"
                elif buy_vix_panic:
                    signal, color = "🔥 刀2：左侧抄底", "#027A48"
                    desc = f"<b>极度恐慌！</b>迎着暴跌 (VIX {curr_vix:.1f}) 抢夺带血筹码"
                elif buy_ma200:
                    signal, color = "🚀 刀1：右侧顺势", "#027A48"
                    desc = "<b>长牛起航！</b>向上有效突破 MA200 牛熊分界线"
                elif buy_reentry:
                    signal, color = "↩️ 刀3：MA20接回", "#027A48"
                    desc = "<b>防踏空！</b>止盈后价格重新站上 MA20，续仓跟牛"
                else:
                    if is_buy_state:
                        color_bias = "#B42318" if is_overextended else "#B8860B"
                        signal, color = "🛡️ 多头死拿", color_bias
                        alert = "⚠️ 处于超买区(Bias>64%)" if is_overextended else "安全持仓"
                        desc = f"均线之上耐心持仓 | 当前偏离度: {curr_bias*100:.1f}% (固定警戒线: 64%) - {alert}"
                    else:
                        signal, color = "💤 空仓吃息", "#667085"
                        desc = f"身处熊市左侧，耐心等待系统拔刀 (当前 VIX: {curr_vix:.1f})"

                model_tag = "<span style='font-size:10px;color:#667085;'>Model 08</span>"

            html_rows += f"""
            <tr>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; font-size:14px; font-weight:900; width:12%;'>{tk}<br>{model_tag}</td>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; color:{color}; font-weight:bold; width:24%;'>{signal}</td>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; color:#475467; font-size:12px;'>{desc}</td>
            </tr>
            """

        return {
            "title": "🧠 核心舱交易大脑  QQQ→Model 08 Diamond | TQQQ→Apollo VIX体制",
            "html_table": f"<table style='width:100%; border-collapse:collapse; text-align:left;'>{html_rows}</table>"
        }
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
    print("✅ Model 08 Diamond Hands 网页版日报生成完毕 (public/index.html)！")

if __name__ == "__main__": main()
