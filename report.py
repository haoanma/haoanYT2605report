#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
投资日报（修复版）- 网页轻量化 + 宏观泡沫预警雷达
- 已修复：加入 fetch_macro_and_bubble_indicators 函数
- 已修复：整合 P/S, CAPE, 利差等机构级指标
"""

import os
import json
import csv
from datetime import datetime
from dateutil import tz
import numpy as np
import pandas as pd
import yfinance as yf
import nasdaqdatalink
from jinja2 import Environment, BaseLoader

# 你的专属 API Key
NASDAQ_API_KEY = "xM_zshfJy_hqkhJwmxKF"

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

CN_COL_MAP = {
    "Name": "名称", "Ticker": "代码", "Close": "收盘价", 
    "ChangePct": "日涨幅(%)", "Ret5D": "周涨幅(%)", 
    "Ret20D": "月涨幅(%)", "Ret250D": "年涨幅(%)",
    "VolRatio": "量比", "MA20": "MA20", "MA50": "MA50", "MA200": "MA200",
    "StrategyHint": "策略状态", "DailyAction": "今日操作", "Trend": "趋势",
}

# ==========================================
# 🚨 核心风控：宏观泡沫与流动性预警
# ==========================================
def fetch_macro_and_bubble_indicators(api_key: str) -> dict:
    if not api_key: return None
    nasdaqdatalink.ApiConfig.api_key = api_key
    try:
        # 1. 抓取纳斯达克底层数据
        fed_rate_data = nasdaqdatalink.get("FRED/FEDFUNDS", rows=2)
        fed_curr = fed_rate_data.iloc[-1].values[0]
        
        cape_data = nasdaqdatalink.get("MULTPL/SHILLER_PE_RATIO_MONTH", rows=1)
        cape_curr = cape_data.iloc[-1].values[0]
        cape_color = "#B42318" if cape_curr > 35 else "#027A48"
        
        t10y2y = nasdaqdatalink.get("FRED/T10Y2Y", rows=2)
        yield_curr = t10y2y.iloc[-1].values[0]
        yield_prev = t10y2y.iloc[-2].values[0]
        yield_trend = "倒挂转正(高危)" if yield_curr > 0 and yield_prev < 0 else ("倒挂中" if yield_curr < 0 else "正常")
        yield_color = "#B42318" if "高危" in yield_trend else "#027A48"
        
        hy_spread = nasdaqdatalink.get("FRED/BAMLH0A0HYM2", rows=1)
        cred_curr = hy_spread.iloc[-1].values[0]
        cred_color = "#B42318" if cred_curr > 5.0 else "#027A48"

        # 2. 抓取巨头 P/S 估值
        tech_titans = ["NVDA", "MSFT", "AAPL"]
        ps_html_parts = []
        for tk in tech_titans:
            try:
                info = yf.Ticker(tk).info
                ps = info.get('priceToSalesTrailing12Months', 0)
                color = "#B42318" if ps > 20 else ("#B8860B" if ps > 10 else "#027A48")
                ps_html_parts.append(f"{tk}: <span style='color:{color}; font-weight:bold;'>{ps:.1f}x</span>")
            except: ps_html_parts.append(f"{tk}: N/A")
        ps_display = " | ".join(ps_html_parts)

        html = f"""
        <table style="width:100%; border-collapse: collapse; text-align: left; font-size: 13px;">
            <tr>
                <td style="padding: 10px; border: 1px solid #E4E7EC; background: #FAFAFB; font-weight: bold; width: 35%;">
                    📉 估值扭曲警报<br><span style="font-size:11px; color:#667085; font-weight:normal;">巨头市销率(P/S) & 席勒市盈率(CAPE)</span>
                </td>
                <td style="padding: 10px; border: 1px solid #E4E7EC;">
                    标普500 CAPE: <b style="color:{cape_color}">{cape_curr:.1f}倍</b><br>
                    核心巨头 P/S: {ps_display} <br>
                    <span style="font-size:11px; color:#667085;">* 注：P/S > 20倍为纯数学级泡沫信号</span>
                </td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #E4E7EC; background: #FAFAFB; font-weight: bold;">
                    🚰 宏观流动性与衰退<br><span style="font-size:11px; color:#667085; font-weight:normal;">基准利率 & 期限利差 (10Y-2Y)</span>
                </td>
                <td style="padding: 10px; border: 1px solid #E4E7EC;">
                    联邦基金利率: <b>{fed_curr:.2f}%</b> <br>
                    10Y-2Y 美债利差: <b style="color:{yield_color}">{yield_curr:.2f}%</b> ({yield_trend})
                </td>
            </tr>
            <tr>
                <td style="padding: 10px; border: 1px solid #E4E7EC; background: #FAFAFB; font-weight: bold;">
                    🧨 资金链断裂预警<br><span style="font-size:11px; color:#667085; font-weight:normal;">美国垃圾债信用利差</span>
                </td>
                <td style="padding: 10px; border: 1px solid #E4E7EC; color: {cred_color};">
                    当前利差: <b>{cred_curr:.2f}%</b> <br>
                    <span style="font-size:11px; color:#667085;">* 注：利差突破5%代表系统性债务风险飙升</span>
                </td>
            </tr>
        </table>
        """
        return {"title": "🚨 宏观泡沫与流动性预警 (深度数据源)", "html_table": html}
    except Exception as e:
        print(f"宏观数据获取失败: {e}")
        return None

# (此处省略中间 fetch_and_calculate, add_trend_flags 等函数，请确保保留你的原版内容)

def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    # ... (省略 watchlist 加载逻辑)
    
    blocks = []
    # 1. 核心操作
    action_block = get_today_action_block(["QQQ", "TQQQ"])
    if action_block: blocks.append(action_block)

    # 2. 插入宏观泡沫预警块 (新增)
    macro_block = fetch_macro_and_bubble_indicators(NASDAQ_API_KEY)
    if macro_block: blocks.append(macro_block)

    # ... (后续生成 sections 的循环)
