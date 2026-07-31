#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投资监控雷达 - 极致聚焦版
- QQQ  信号引擎：纯MA200（上穿买入，下穿卖出，无其他过滤）
- TQQQ 信号引擎：ADX20方向切换(下行才卖)。Wilder 14周期 ADX(T-1) 判趋势、+DI/-DI(T-1) 判方向。
    只有「跌破MA200 且 ADX(T-1)>20(趋势确认) 且 -DI(T-1)>+DI(T-1)(下行确认)」
    才卖出空仓；其余情况（MA200上方 / 震荡 / 跌破但方向偏多）一律无条件满仓持有，
    避免仅凭跌破MA200+趋势确认就卖在震荡或假下跌里（回测见 adx20_directional.py）。
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
    "history_days": 500,
    "ma_windows": [20, 50, 200],
    "base_font_px": 14,
    "table_font_px": 12
}

# TQQQ 核心舱: ADX20切换逻辑用到的阈值
TQQQ_ADX_PERIOD    = 14
TQQQ_ADX_THRESHOLD = 20

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
# 🔧 Wilder 14周期 ADX（手写实现，无未来函数）
# ==========================================
def _wilder_smooth(values: np.ndarray, period: int, mode: str) -> np.ndarray:
    """mode='sum': 首值=前period个观测之和，递推 S_t = S_{t-1} - S_{t-1}/period + x_t
    （用于 TR / +DM / -DM）。
    mode='avg': 首值=前period个观测均值，递推 A_t = ((period-1)*A_{t-1}+x_t)/period
    （用于 DX -> ADX）。"""
    n = len(values)
    out = np.full(n, np.nan)
    valid = ~np.isnan(values)
    if not valid.any():
        return out
    start = int(np.argmax(valid))
    if start + period > n:
        return out
    window = values[start:start + period]
    if mode == "sum":
        out[start + period - 1] = window.sum()
        for i in range(start + period, n):
            out[i] = out[i - 1] - out[i - 1] / period + values[i]
    else:
        out[start + period - 1] = window.mean()
        for i in range(start + period, n):
            out[i] = ((period - 1) * out[i - 1] + values[i]) / period
    return out


def _wilder_dmi(high: pd.Series, low: pd.Series, close: pd.Series, period: int = TQQQ_ADX_PERIOD):
    """返回 (adx, plus_di, minus_di)，与 adx20_directional.py 的 wilder_dmi 完全一致。"""
    prev_close = close.shift(1)
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move   = high - prev_high
    down_move = prev_low - low
    plus_dm  = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=close.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=close.index)

    tr.iloc[0] = np.nan
    plus_dm.iloc[0] = np.nan
    minus_dm.iloc[0] = np.nan

    tr_s    = pd.Series(_wilder_smooth(tr.to_numpy(), period, "sum"), index=close.index)
    plus_s  = pd.Series(_wilder_smooth(plus_dm.to_numpy(), period, "sum"), index=close.index)
    minus_s = pd.Series(_wilder_smooth(minus_dm.to_numpy(), period, "sum"), index=close.index)

    plus_di  = 100 * plus_s / tr_s
    minus_di = 100 * minus_s / tr_s
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)

    adx = pd.Series(_wilder_smooth(dx.to_numpy(), period, "avg"), index=close.index)
    return adx, plus_di, minus_di


# ==========================================
# 🔧 TQQQ 核心舱：ADX20 方向切换逻辑
# ==========================================
def _compute_adx20_switch_signal(close: pd.Series, high: pd.Series, low: pd.Series,
                                  adx_threshold: float = TQQQ_ADX_THRESHOLD) -> dict:
    """方向精细化版本（原 adx20_directional.py 回测验证）：
    跌破MA200 且 ADX(T-1)>threshold（趋势确认） 且 -DI(T-1)>+DI(T-1)（下行方向确认） -> 卖出/空仓
    其余情况（MA200上方 / 震荡 / 跌破但方向偏多）一律无条件满仓持有，
    避免仅凭"跌破MA200+趋势确认"就卖在震荡或假下跌里。"""
    ma200 = close.rolling(200, min_periods=200).mean()
    adx, plus_di, minus_di = _wilder_dmi(high, low, close, period=TQQQ_ADX_PERIOD)

    valid = ma200.dropna().index
    if len(valid) < 2:
        return {}
    close_v = close.loc[valid]
    ma200_v = ma200.loc[valid]
    adx_v   = adx.reindex(valid)
    pdi_v   = plus_di.reindex(valid)
    mdi_v   = minus_di.reindex(valid)

    above      = (close_v > ma200_v).fillna(False)
    above_prev = above.shift(1, fill_value=above.iloc[0])
    adx_prev   = adx_v.shift(1)
    pdi_prev   = pdi_v.shift(1)
    mdi_prev   = mdi_v.shift(1)
    trending   = (adx_prev > adx_threshold).fillna(False)
    downtrend  = (mdi_prev > pdi_prev).fillna(False)

    sell = (~above_prev) & trending & downtrend
    holding = ~sell
    prev_holding = holding.shift(1, fill_value=holding.iloc[0])
    buy_e  = holding & (~prev_holding)
    sell_e = (~holding) & prev_holding

    curr_close = float(close_v.iloc[-1])
    curr_ma200 = float(ma200_v.iloc[-1])
    curr_adx   = float(adx_v.iloc[-1]) if pd.notna(adx_v.iloc[-1]) else float("nan")
    curr_pdi   = float(pdi_v.iloc[-1]) if pd.notna(pdi_v.iloc[-1]) else float("nan")
    curr_mdi   = float(mdi_v.iloc[-1]) if pd.notna(mdi_v.iloc[-1]) else float("nan")
    bias_pct   = (curr_close / curr_ma200 - 1) * 100 if curr_ma200 else 0.0

    # 当天(T)口径的方向/趋势，用于显示，避免与 T-1 判定值混用
    curr_is_downtrend = bool(curr_mdi > curr_pdi) if not (np.isnan(curr_mdi) or np.isnan(curr_pdi)) else False
    curr_is_trending  = bool(curr_adx > adx_threshold) if not np.isnan(curr_adx) else False
    curr_above = bool(above.iloc[-1])
    holding_now = bool(holding.iloc[-1])  # 今日开盘按T-1信号执行完之后，当前实际仓位

    # 用"今天"自己的收盘数据当作明天的T-1，反推明天开盘会执行什么操作
    would_sell_next = (not curr_above) and curr_is_trending and curr_is_downtrend

    # 卖出信号「今日收盘刚形成、尚未执行」：当前持仓，但今日数据已满足卖出三条件 -> 明日开盘卖出
    pending_sell = bool(holding_now and would_sell_next)
    # 买入信号「今日收盘刚形成、尚未执行」：当前空仓，但今日数据已不再满足卖出三条件(收回MA200/ADX转弱/方向转多)
    # -> 明日开盘买入。这正是此前缺失的一半：例如今日按昨日信号卖出，但今日收盘已收复MA200，
    # 此时不该只显示"已卖出"，还要提示"明日将买回"。
    pending_buy = bool((not holding_now) and (not would_sell_next))

    if holding_now:
        next_action = "SELL" if would_sell_next else "HOLD"
    else:
        next_action = "HOLD_CASH" if would_sell_next else "BUY"

    executed_today = "SELL" if bool(sell_e.iloc[-1]) else ("BUY" if bool(buy_e.iloc[-1]) else "NONE")

    return {
        "holding":     holding_now,
        "today_buy":   bool(buy_e.iloc[-1]),
        "today_sell":  bool(sell_e.iloc[-1]),
        "executed_today": executed_today,
        "is_trending": bool(trending.iloc[-1]),
        "is_downtrend": bool(downtrend.iloc[-1]),
        "above_ma200": curr_above,
        "curr_is_downtrend": curr_is_downtrend,
        "curr_is_trending":  curr_is_trending,
        "pending_sell":      pending_sell,
        "pending_buy":       pending_buy,
        "next_action":       next_action,
        "curr_adx":    curr_adx,
        "curr_pdi":    curr_pdi,
        "curr_mdi":    curr_mdi,
        "curr_ma200":  curr_ma200,
        "curr_close":  curr_close,
        "bias_pct":    bias_pct,
    }


# ==========================================
# 📊 通用大表策略判定
#    QQQ 及其他: Model 08 Diamond Hands
#    TQQQ:       Model 08 基础 + Apollo 叠加
# ==========================================
def fetch_and_calculate(ticker_map: dict, history_days: int = 500) -> pd.DataFrame:
    if not ticker_map: return pd.DataFrame()
    tickers = list(ticker_map.keys())
    fetch_list = list(set(tickers))

    raw_data = None
    for attempt in range(3):
        try:
            raw_data = yf.download(fetch_list, period=f"{history_days}d", interval="1d", auto_adjust=False, group_by="ticker", threads=False, progress=False)
            if raw_data is not None and not raw_data.empty: break
            time.sleep(2)
        except: time.sleep(2)

    if raw_data is None or raw_data.empty: return pd.DataFrame()

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

            curr = _safe_float(close.iloc[-1])
            prev = _safe_float(close.iloc[-2])
            pct = (curr / prev - 1) * 100 if prev else np.nan

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
                    high, low = df_t["High"], df_t["Low"]
                    adx_sig = _compute_adx20_switch_signal(close, high, low)

                    if not adx_sig:
                        adx_hint, adx_action = "数据不足", "观望"
                    else:
                        adx_txt = (f"ADX={adx_sig['curr_adx']:.1f} +DI={adx_sig['curr_pdi']:.1f}/-DI={adx_sig['curr_mdi']:.1f}"
                                   if not np.isnan(adx_sig["curr_adx"]) else "ADX=NA")
                        # 优先展示"下一交易日该怎么做"(用今日收盘数据推演)，而不是只报"今日已执行什么"
                        if adx_sig["pending_buy"]:
                            prefix = "今日已按信号卖出，但" if adx_sig["today_sell"] else ""
                            adx_hint, adx_action = f"🔄 {prefix}收盘已转多/收复条件，买入信号已形成，下一开盘执行 | {adx_txt}", "明日买入"
                        elif adx_sig["pending_sell"]:
                            adx_hint, adx_action = f"🚨 卖出信号已形成，下一开盘执行 | {adx_txt}", "明日卖出"
                        elif adx_sig["today_sell"]:
                            adx_hint, adx_action = f"🚨 下行趋势确认卖出(-DI>+DI) | {adx_txt}", "卖出"
                        elif adx_sig["today_buy"]:
                            adx_hint, adx_action = f"🔄 转入持有(方向/体制转变) | {adx_txt}", "买入"
                        elif not adx_sig["holding"]:
                            adx_hint, adx_action = f"⛔ 下行趋势空仓中 | {adx_txt}", "观望"
                        elif adx_sig["above_ma200"]:
                            adx_hint, adx_action = f"🛡️ MA200上方持有 | {adx_txt}", "观望"
                        elif adx_sig["curr_is_trending"] and not adx_sig["curr_is_downtrend"]:
                            adx_hint, adx_action = f"📈 下方趋势偏多持有(+DI>-DI) | {adx_txt}", "观望"
                        else:
                            adx_hint, adx_action = f"💤 下方震荡持有 | {adx_txt}", "观望"

                    ret20  = (curr / close.iloc[-21]  - 1) * 100 if len(close) > 21  else np.nan
                    ret250 = (curr / close.iloc[-251] - 1) * 100 if len(close) > 251 else np.nan
                    vol_ratio = np.nan
                    if "Volume" in df_t.columns and len(df_t) >= 22:
                        v_avg = df_t["Volume"].iloc[-21:-1].mean()
                        if v_avg and v_avg > 0: vol_ratio = _safe_float(df_t["Volume"].iloc[-1]) / v_avg

                    rows.append({
                        "Ticker": t, "Close": curr, "ChangePct": pct, "VolRatio": vol_ratio,
                        "MA20": curr_ma20, "MA50": curr_ma50, "MA200": curr_ma200,
                        "Ret20D": ret20, "Ret250D": ret250,
                        "Name": f"{ticker_map.get(t, t)}(ADX20方向切换/下行才卖)",
                        "StrategyHint": adx_hint, "DailyAction": adx_action,
                    })
                    continue

                else:
                    # ── 纯 MA200（QQQ 及其他品种）──
                    _above = (close > ma200).fillna(False)
                    _buy_e = _above & (~_above.shift(1, fill_value=False))
                    _sel_e = (~_above) & _above.shift(1, fill_value=False)
                    bias_pct = (curr / curr_ma200 - 1) * 100 if curr_ma200 else 0.0

                    if bool(_sel_e.iloc[-1]):
                        strategy_hint, daily_action = "🚨 MA200破位 卖出", "卖出"
                    elif bool(_buy_e.iloc[-1]):
                        strategy_hint, daily_action = "🚀 MA200上穿 买入", "买入"
                    elif bool(_above.iloc[-1]):
                        strategy_hint = f"🛡️ MA200上方持仓 {bias_pct:+.1f}%"
                    else:
                        strategy_hint = f"💤 MA200下方空仓 {bias_pct:+.1f}%"

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
def _render_action_row(tk_label: str, model_tag_text: str, signal: str, color: str, desc: str) -> str:
    model_tag = f"<span style='font-size:10px;color:#667085;'>{model_tag_text}</span>"
    return f"""
    <tr>
        <td style='padding:12px; border-bottom:1px solid #E4E7EC; font-size:14px; font-weight:900; width:12%;'>{tk_label}<br>{model_tag}</td>
        <td style='padding:12px; border-bottom:1px solid #E4E7EC; color:{color}; font-weight:bold; width:24%;'>{signal}</td>
        <td style='padding:12px; border-bottom:1px solid #E4E7EC; color:#475467; font-size:12px;'>{desc}</td>
    </tr>
    """


def get_today_action_block(tickers=["QQQ", "TQQQ"]):
    try:
        df_raw = yf.download(tickers, period="2y", auto_adjust=False, group_by="ticker", threads=False, progress=False)
        if df_raw.empty: return None

        html_rows = ""
        for tk in tickers:
            if not hasattr(df_raw.columns, "levels") or tk not in df_raw.columns.levels[0]: continue
            close = df_raw[tk]["Close"].dropna()
            if len(close) < 252: continue

            ma200 = close.rolling(200).mean()

            if tk == "TQQQ":
                high = df_raw[tk]["High"].reindex(close.index)
                low  = df_raw[tk]["Low"].reindex(close.index)
                adx_sig = _compute_adx20_switch_signal(close, high, low)

                # ══ ADX20方向切换(下行才卖) ═════════════════════
                if not adx_sig:
                    signal2, color2, desc2 = "数据不足", "#667085", "历史数据不够，无法计算ADX/MA200"
                else:
                    adx_txt  = f"{adx_sig['curr_adx']:.1f}" if not np.isnan(adx_sig["curr_adx"]) else "NA"
                    dir_txt  = (f"-DI{adx_sig['curr_mdi']:.1f}>+DI{adx_sig['curr_pdi']:.1f}(下行)"
                                if adx_sig["curr_is_downtrend"] else
                                f"+DI{adx_sig['curr_pdi']:.1f}>-DI{adx_sig['curr_mdi']:.1f}(偏多)")
                    adx_info = (f"ADX(14) <b>{adx_txt}</b>（阈值20，{dir_txt}）| MA200 <b>{adx_sig['curr_ma200']:.2f}</b> | "
                                f"现价 <b>{adx_sig['curr_close']:.2f}</b> "
                                f"({'上方' if adx_sig['above_ma200'] else '下方'} {abs(adx_sig['bias_pct']):.1f}%)")
                    # 优先展示"下一交易日该怎么做"(用今日收盘数据推演)，而不是只报"今日已执行什么"——
                    # 例如今日按昨日信号卖出，但今日收盘已收复MA200/方向转多，不能只说"已卖出"，
                    # 还要提示"买入信号已形成，明日开盘买回"。
                    if adx_sig["pending_buy"]:
                        signal2 = "🔄 买入信号已形成：下一开盘执行"
                        color2  = "#027A48"
                        sold_note = ("<b>今日已按昨日信号卖出离场，但今日收盘MA200/ADX/方向条件已重新满足持有——</b>"
                                     if adx_sig["today_sell"] else "<b>今日收盘MA200/ADX/方向条件已重新满足持有条件——</b>")
                        desc2   = (f"{sold_note}买入信号已形成，将在下一交易日开盘买回。<br>{adx_info}")
                    elif adx_sig["pending_sell"]:
                        signal2 = "🚨 卖出信号已形成：下一开盘执行"
                        color2  = "#B42318"
                        desc2   = (f"<b>今日收盘价跌破MA200，且ADX&gt;20、-DI&gt;+DI下行确认，卖出信号已形成。</b>"
                                   f"按 T-1 信号→T 开盘执行规则，将在下一交易日开盘卖出离场（今日仍记为持仓）。<br>{adx_info}")
                    elif adx_sig["today_sell"]:
                        signal2 = "🚨 下行趋势确认：卖出"
                        color2  = "#B42318"
                        desc2   = (f"<b>价格跌破MA200，且ADX&gt;20确认趋势、-DI&gt;+DI确认方向向下，卖出离场。</b>"
                                   f"（震荡或方向偏多时不会卖）<br>{adx_info}")
                    elif adx_sig["today_buy"]:
                        signal2 = "🔄 转入持有：买入"
                        color2  = "#027A48"
                        desc2   = f"MA200回到上方，或震荡/方向转多导致「下行确认」条件解除，恢复满仓买入。<br>{adx_info}"
                    elif not adx_sig["holding"]:
                        signal2 = "⛔ 下行趋势：空仓中"
                        color2  = "#667085"
                        desc2   = f"仍处于「跌破MA200+趋势确认+下行确认」的卖出状态，继续空仓等待。<br>{adx_info}"
                    elif adx_sig["above_ma200"]:
                        signal2 = "🛡️ MA200上方持有"
                        color2  = "#B8860B"
                        desc2   = f"价格在MA200上方，维持满仓。<br>{adx_info}"
                    elif adx_sig["curr_is_trending"] and not adx_sig["curr_is_downtrend"]:
                        signal2 = "📈 下方趋势偏多：持有"
                        color2  = "#B8860B"
                        desc2   = f"ADX&gt;20确认趋势，但+DI&gt;-DI方向偏多（非下行确认），继续满仓持有。<br>{adx_info}"
                    else:
                        signal2 = "💤 下方震荡：持有"
                        color2  = "#B8860B"
                        desc2   = f"方向未确认或ADX偏低（震荡/假跌破/预热期），维持满仓。<br>{adx_info}"

                html_rows += _render_action_row(tk, "ADX20方向切换(下行才卖)", signal2, color2, desc2)
                continue

            else:
                # ══ 纯 MA200（QQQ）══════════════════════════════════
                _above = (close > ma200).fillna(False)
                _buy_e = _above  & (~_above.shift(1, fill_value=False))
                _sel_e = (~_above) & _above.shift(1, fill_value=False)
                curr_close  = float(close.iloc[-1])
                curr_ma200v = float(ma200.iloc[-1]) if not ma200.empty else 0.0
                bias_pct    = (curr_close / curr_ma200v - 1) * 100 if curr_ma200v else 0.0
                ma200_info  = (f"MA200 <b>{curr_ma200v:.2f}</b> | 现价 <b>{curr_close:.2f}</b> "
                               f"({'上方' if bool(_above.iloc[-1]) else '下方'} {abs(bias_pct):.1f}%)")

                if bool(_sel_e.iloc[-1]):
                    signal = "🚨 MA200破位：卖出"
                    color  = "#B42318"
                    desc   = f"<b>价格下穿MA200，趋势转坏，清仓离场。</b><br>{ma200_info}"
                elif bool(_buy_e.iloc[-1]):
                    signal = "🚀 MA200上穿：买入"
                    color  = "#027A48"
                    desc   = f"<b>价格上穿MA200，趋势回升，建仓入场。</b><br>{ma200_info}"
                elif bool(_above.iloc[-1]):
                    signal = "🛡️ MA200上方持仓"
                    color  = "#B8860B"
                    desc   = f"价格在MA200上方，持仓跟牛。<br>{ma200_info}"
                else:
                    signal = "💤 MA200下方空仓"
                    color  = "#667085"
                    desc   = f"价格在MA200下方，等待上穿信号。<br>{ma200_info}"

                html_rows += _render_action_row(tk, "MA200", signal, color, desc)

        return {
            "title": "🧠 核心舱交易大脑  QQQ→纯MA200 | TQQQ→ADX20方向切换(下行才卖)",
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
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D", "MA200"], "asia"),

        ("🇪🇺 欧洲及其他市场",
         lambda it: it["category"] == "indices" and it["market"] in ["EU", "UK", "EUROPE", "GL", "OT"],
         ["Name", "Ticker", "Close", "ChangePct", "Ret20D", "Ret250D", "MA200"], "eu"),
         
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
