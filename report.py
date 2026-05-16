def get_today_action_block(tickers=["QQQ", "TQQQ"]):
    try:
        fetch_list = tickers + ["^VIX"]
        # 必须带上 group_by="ticker" 确保格式对齐
        df_raw = yf.download(fetch_list, period="1y", auto_adjust=False, group_by="ticker", threads=False, progress=False)
        if df_raw.empty: return None

        if "^VIX" in df_raw.columns.levels[0]:
            vix_series = df_raw["^VIX"]["Close"].dropna()
        else: return None

        html_rows = ""
        for tk in tickers:
            if tk not in df_raw.columns.levels[0]: continue
            close = df_raw[tk]["Close"].dropna()
            if len(close) < 201: continue

            df = pd.DataFrame({"close": close})
            df["ma20"] = df["close"].rolling(20).mean()
            df["ma200"] = df["close"].rolling(200).mean()
            df["vix"] = vix_series.reindex(df.index).ffill()
            df = df.dropna()
            if len(df) < 2: continue

            curr = df.iloc[-1]
            prev = df.iloc[-2]

            # 1. 基础指标与偏离度 (Bias) 计算
            bias_pct = ((curr["close"] - curr["ma200"]) / curr["ma200"]) * 100
            bubble_threshold = 64.0 # 极值防线阈值，可根据需求改为 64.0 钻石版

            # 2. 边缘触发器 (Edge Triggers) - 只认当天的交叉瞬间
            cross_up_ma200 = (prev["close"] <= prev["ma200"]) and (curr["close"] > curr["ma200"])
            cross_down_ma200 = (prev["close"] >= prev["ma200"]) and (curr["close"] < curr["ma200"])
            cross_up_ma20 = (prev["close"] <= prev["ma20"]) and (curr["close"] > curr["ma20"])
            cross_down_ma20 = (prev["close"] >= prev["ma20"]) and (curr["close"] < curr["ma20"])
            
            vix_spike_panic = (curr["vix"] > 34) and not (prev["vix"] > 34)
            
            # 3. 判定卖出防守盾牌 (Sell Absolute Priority)
            # 盾1: 跌破年线且 VIX > 22 (衰退确认)
            is_sell_shield1 = (curr["close"] < curr["ma200"] and curr["vix"] > 22) and not (prev["close"] < prev["ma200"] and prev["vix"] > 22)
            # 盾2: 泡沫极度偏离且跌破 20 日线 (狂热刺破)
            is_sell_shield2 = (bias_pct > bubble_threshold) and cross_down_ma20

            # 4. 判定买入进攻尖刀
            is_buy_knife1 = cross_up_ma200
            is_buy_knife2 = (curr["close"] <= curr["ma200"]) and vix_spike_panic
            is_buy_knife3 = cross_up_ma20 and (curr["close"] > curr["ma200"])

            # 5. 状态机风控大脑介入 (强行拦截优先级)
            if is_sell_shield1 or is_sell_shield2:
                if is_sell_shield1:
                    signal, color = "🚨 清仓避险", "#B42318"
                    desc = f"<b>🛡️ 盾1生效</b>：衰退确认！跌破年线且VIX飙升至 {curr['vix']:.1f}"
                else:
                    signal, color = "🚨 极值止盈", "#B42318"
                    desc = f"<b>🛡️ 盾2生效</b>：泡沫刺破！年线偏离达 {bias_pct:.1f}% 且跌穿20日均线"
                    
            elif is_buy_knife1 or is_buy_knife2 or is_buy_knife3:
                if is_buy_knife2:
                    signal, color = "🔥 左侧抄底", "#027A48"
                    desc = f"<b>⚔️ 刀2出鞘</b>：非理性暴跌！迎着极度恐慌 (VIX {curr['vix']:.1f}) 抢夺带血筹码"
                elif is_buy_knife1:
                    signal, color = "🚀 右侧顺势", "#027A48"
                    desc = "<b>⚔️ 刀1出鞘</b>：长牛起航！有效突破 MA200 年线牛熊分界"
                else:
                    signal, color = "⚡ 防踏空买回", "#027A48"
                    desc = "<b>⚔️ 刀3出鞘</b>：假摔纠错！重返 MA20 日线，顺大势吃大肉"
                    
            else:
                # 若无边缘触发信号，维持原有状态
                if curr["close"] > curr["ma200"]:
                    signal, color = "🛡️ 多头死拿", "#B8860B"
                    desc = f"均线之上耐心持仓 (当前Bias偏离度: {bias_pct:.1f}%)"
                else:
                    signal, color = "💤 空仓吃息", "#667085"
                    desc = f"身处熊市左侧，耐心等待系统拔刀 (当前VIX: {curr['vix']:.1f})"

            html_rows += f"""
            <tr>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; font-size:14px; font-weight:900; width:15%;'>{tk}</td>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; color:{color}; font-weight:bold; width:20%;'>{signal}</td>
                <td style='padding:12px; border-bottom:1px solid #E4E7EC; color:#475467; font-size:12px;'>{desc}</td>
            </tr>
            """
            
        return {"title": "🧠 核心舱交易逻辑 (绝对防守法则)", "html_table": f"<table style='width:100%; border-collapse:collapse; text-align:left;'>{html_rows}</table>"}
    except Exception as e: 
        print(f"核心操作块解析失败: {e}")
        return None
