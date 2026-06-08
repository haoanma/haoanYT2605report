#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投资监控雷达 - 极致聚焦版
- QQQ  信号引擎：纯MA200（上穿买入，下穿卖出，无其他过滤）
- TQQQ 信号引擎：纯MA200 + VIX恐慌抄底
    基础：仅 MA200（上穿买入，下穿卖出）
    抄底：VIX≥35 且价格在MA200以下 → 可抄底一次（每次MA200下穿周期限用一次）
    止损：抄底后跌超10%次日开盘卖出
    重置：抄底用后须等MA200上穿信号才重新开启抄底资格
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

# TQQQ 专用参数
TQQQ_VIX_TRIGGER = 35    # VIX 触达此值可抄底
TQQQ_STOP_PCT    = 0.10  # 抄底后跌超此幅度止损

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
# 🔧 TQQQ 状态机：纯MA200 + VIX恐慌抄底
# ==========================================
def _compute_ma200_vix_state(close: pd.Series, vix: pd.Series,
                              vix_trigger: float = TQQQ_VIX_TRIGGER,
                              stop_pct: float = TQQQ_STOP_PCT) -> dict:
    """
    在全部可用历史数据上运行状态机，返回当前最新持仓状态。
    规则：
      基础    — MA200上穿买入 / 下穿卖出
      抄底    — VIX≥vix_trigger 且价格<MA200 → 可抄底一次（每轮熊市限一次）
      止损    — 抄底后跌超 stop_pct → 次日卖出
      重置    — 抄底后须等MA200上穿才重新允许抄底
    """
    ma200 = close.rolling(200, min_periods=200).mean()
    valid = ma200.dropna().index
    if len(valid) < 2:
        return {}
    close = close.loc[valid]
    vix   = vix.reindex(valid).ffill().fillna(20.0)
    ma200 = ma200.loc[valid]

    above       = (close > ma200).fillna(False)
    buy_ma200_e = above  & (~above.shift(1, fill_value=above.iloc[0]))
    sell_ma200_e= (~above) & above.shift(1, fill_value=above.iloc[0])
    c_vix       = (vix >= vix_trigger) & (~above)
    buy_vix_e   = c_vix & (~c_vix.shift(1, fill_value=False))

    # 初始状态：价格在MA200上方则持仓，否则空仓
    holding       = bool(above.iloc[0])
    panic_allowed = not holding   # 若起始在MA200下方，允许抄底
    in_panic      = False
    panic_entry   = 0.0
    force_sell    = False

    for i in range(len(close)):
        cp = float(close.iloc[i])
        if i > 0:
            is_sell    = bool(sell_ma200_e.iloc[i - 1])
            is_ma_buy  = bool(buy_ma200_e.iloc[i - 1])
            is_vix_buy = bool(buy_vix_e.iloc[i - 1])

            if force_sell and holding:
                holding = False; in_panic = False
                panic_allowed = False; panic_entry = 0.0; force_sell = False
            elif is_sell and holding:
                was = in_panic
                holding = False; in_panic = False; force_sell = False
                if was: panic_allowed = False
            elif is_ma_buy and not holding:
                holding = True; in_panic = False; panic_allowed = True
            elif is_vix_buy and panic_allowed and not holding:
                holding = True; in_panic = True
                panic_entry = cp; panic_allowed = False

        if in_panic and holding and cp < panic_entry * (1 - stop_pct):
            force_sell = True

    curr_close  = float(close.iloc[-1])
    curr_ma200  = float(ma200.iloc[-1])
    curr_vix    = float(vix.iloc[-1])
    curr_above  = bool(above.iloc[-1])
    bias_pct    = (curr_close / curr_ma200 - 1) * 100 if curr_ma200 else 0.0
    panic_pnl   = (curr_close / panic_entry - 1) * 100 if in_panic and panic_entry > 0 else None
    stop_level  = panic_entry * (1 - stop_pct) if in_panic and panic_entry > 0 else None

    return {
        "holding":         holding,
        "in_panic":        in_panic,
        "panic_allowed":   panic_allowed,
        "panic_entry":     panic_entry,
        "panic_pnl":       panic_pnl,       # 抄底仓位当前浮盈%
        "stop_level":      stop_level,       # 止损价
        "force_sell":      force_sell,       # 止损已触发，明日执行
        "above_ma200":     curr_above,
        "curr_ma200":      curr_ma200,
        "curr_close":      curr_close,
        "curr_vix":        curr_vix,
        "bias_pct":        bias_pct,
        # 今日边沿信号
        "today_sell":      bool(sell_ma200_e.iloc[-1]),
        "today_buy_ma200": bool(buy_ma200_e.iloc[-1]),
        "today_buy_vix":   bool(buy_vix_e.iloc[-1]) and panic_allowed and not holding,
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
                    # ── 纯MA200 + VIX抄底 状态机 ──
                    st = _compute_ma200_vix_state(close, vix_aligned)
                    if not st:
                        strategy_hint = "数据不足"
                    elif st["force_sell"]:
                        strategy_hint = f"🛑 止损预警 明日卖出 | VIX={st['curr_vix']:.1f}"
                        daily_action  = "卖出"
                    elif st["today_sell"]:
                        strategy_hint = f"🚨 MA200破位卖出 | VIX={st['curr_vix']:.1f}"
                        daily_action  = "卖出"
                    elif st["today_buy_vix"]:
                        strategy_hint = f"🔥 VIX≥{TQQQ_VIX_TRIGGER}抄底买入 | VIX={st['curr_vix']:.1f}"
                        daily_action  = "买入"
                    elif st["today_buy_ma200"]:
                        strategy_hint = f"🚀 MA200上穿买入 | VIX={st['curr_vix']:.1f}"
                        daily_action  = "买入"
                    elif st["holding"] and st["in_panic"]:
                        pnl = f"{st['panic_pnl']:+.1f}%" if st["panic_pnl"] is not None else "?"
                        strategy_hint = f"⚡ 抄底持仓({pnl}) 止损线{st['stop_level']:.3f} | VIX={st['curr_vix']:.1f}"
                    elif st["holding"]:
                        strategy_hint = f"🛡️ MA200上方持仓 Bias={st['bias_pct']:+.1f}% | VIX={st['curr_vix']:.1f}"
                    elif not st["panic_allowed"]:
                        strategy_hint = f"💤 空仓等MA200上穿 | VIX={st['curr_vix']:.1f}"
                    else:
                        vix_tag = f"🔥VIX={st['curr_vix']:.1f}≥{TQQQ_VIX_TRIGGER}可抄底" if st["curr_vix"] >= TQQQ_VIX_TRIGGER and not st["above_ma200"] else f"VIX={st['curr_vix']:.1f}"
                        strategy_hint = f"💤 空仓等信号 | {vix_tag}"

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

            ma200 = close.rolling(200).mean()

            if tk == "TQQQ":
                # ══ 纯MA200 + VIX≥35 恐慌抄底 状态机 ══════════════
                st = _compute_ma200_vix_state(close, vix_aligned)
                if not st:
                    signal, color, desc = "数据不足", "#667085", "历史数据不够，无法计算MA200"
                else:
                    ma200_info = (f"MA200 <b>{st['curr_ma200']:.2f}</b> | "
                                  f"现价 <b>{st['curr_close']:.2f}</b> "
                                  f"({'上方' if st['above_ma200'] else '下方'} {abs(st['bias_pct']):.1f}%)")
                    vix_info   = f"VIX <b>{st['curr_vix']:.1f}</b> (抄底线≥{TQQQ_VIX_TRIGGER})"

                    if st["force_sell"]:
                        signal = "🛑 止损预警：明日开盘卖出"
                        color  = "#B42318"
                        desc   = (f"<b>抄底仓位跌破止损线(买入价×{1-TQQQ_STOP_PCT:.0%})！</b>"
                                  f"将于明日开盘执行卖出。<br>{vix_info} | {ma200_info}")

                    elif st["today_sell"]:
                        signal = "🚨 MA200破位：清仓"
                        color  = "#B42318"
                        desc   = f"<b>价格下穿MA200，趋势转坏，清仓离场。</b><br>{vix_info} | {ma200_info}"

                    elif st["today_buy_vix"]:
                        signal = f"🔥 VIX≥{TQQQ_VIX_TRIGGER}：恐慌抄底"
                        color  = "#027A48"
                        desc   = (f"<b>VIX触达恐慌区间，价格在MA200以下，一次性抄底机会。</b>"
                                  f"止损线：买入价×{1-TQQQ_STOP_PCT:.0%}<br>{vix_info} | {ma200_info}")

                    elif st["today_buy_ma200"]:
                        signal = "🚀 MA200上穿：买入"
                        color  = "#027A48"
                        desc   = (f"<b>价格上穿MA200，趋势回升，建仓入场。</b>"
                                  f"抄底资格同步重置。<br>{vix_info} | {ma200_info}")

                    elif st["holding"] and st["in_panic"]:
                        pnl    = f"{st['panic_pnl']:+.1f}%" if st["panic_pnl"] is not None else "?"
                        sl_px  = f"{st['stop_level']:.3f}" if st["stop_level"] else "?"
                        pnl_v  = st["panic_pnl"] or 0
                        color  = "#B42318" if pnl_v < -7 else ("#027A48" if pnl_v > 0 else "#B8860B")
                        signal = f"⚡ 抄底持仓中 ({pnl})"
                        desc   = (f"持有VIX恐慌抄底仓位 | 浮盈 <b>{pnl}</b> | "
                                  f"止损线 {sl_px} ({TQQQ_STOP_PCT:.0%})<br>{vix_info} | {ma200_info}")

                    elif st["holding"]:
                        color  = "#B8860B"
                        signal = "🛡️ MA200上方持仓"
                        desc   = (f"价格在MA200上方，持仓跟牛。"
                                  f"{'⚠️偏离较高，注意回调' if st['bias_pct'] > 60 else ''}<br>"
                                  f"{vix_info} | {ma200_info}")

                    elif not st["panic_allowed"]:
                        signal = "💤 空仓：等MA200上穿"
                        color  = "#667085"
                        desc   = (f"本轮熊市抄底资格已用，须等价格上穿MA200才可再次入场。<br>"
                                  f"{vix_info} | {ma200_info}")

                    else:
                        # 空仓 + 抄底资格可用
                        if st["curr_vix"] >= TQQQ_VIX_TRIGGER and not st["above_ma200"]:
                            signal = f"🔥 VIX={st['curr_vix']:.1f}≥{TQQQ_VIX_TRIGGER}（可抄底，等边沿）"
                            color  = "#027A48"
                            desc   = (f"VIX处于恐慌区间且价格在MA200以下，满足抄底条件，等待明日信号确认。<br>"
                                      f"{vix_info} | {ma200_info}")
                        else:
                            signal = "💤 空仓等信号"
                            color  = "#667085"
                            desc   = (f"等待 VIX≥{TQQQ_VIX_TRIGGER} 抄底 或 MA200上穿买入。<br>"
                                      f"{vix_info} | {ma200_info}")

                model_tag = "<span style='font-size:10px;color:#667085;'>MA200+VIX35</span>"

            else:
                # ══ 纯 MA200（QQQ）══════════════════════════════════
                _above = (close > ma200).fillna(False)
                _buy_e = _above  & (~_above.shift(1, fill_value=False))
                _sel_e = (~_above) & _above.shift(1, fill_value=False)
                curr_close  = float(close.iloc[-1])
                curr_ma200v = float(ma200.iloc[-1]) if not ma200.empty else 0.0
                bias_pct    = (curr_close / curr_ma200v - 1) * 100 if curr_ma200v else 0.0
                curr_vix    = float(vix_aligned.iloc[-1])
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

                model_tag = "<span style='font-size:10px;color:#667085;'>MA200</span>"

            html_rows += f"""
            <tr>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; font-size:14px; font-weight:900; width:12%;'>{tk}<br>{model_tag}</td>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; color:{color}; font-weight:bold; width:24%;'>{signal}</td>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; color:#475467; font-size:12px;'>{desc}</td>
            </tr>
            """

        return {
            "title": f"🧠 核心舱交易大脑  QQQ→纯MA200 | TQQQ→MA200+VIX≥{TQQQ_VIX_TRIGGER}恐慌抄底",
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
