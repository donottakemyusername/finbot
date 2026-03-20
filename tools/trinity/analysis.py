"""tools/trinity/analysis.py
=============================
Layer 4: 主入口。在原有基础上新增 price_chart_data，
让前端能画出真实的价格+均线图来验证分析结论。
"""
from __future__ import annotations
import json, os
import anthropic
import pandas as pd

from tools.trinity.indicators import (
    fetch_multi_timeframe,
    compute_all_hard_signals,
    compute_bollinger_trinity,
    compute_macd_signals,
    compute_divergence_summary,
    compute_ma_analysis_summary,
)
from tools.trinity.state import compute_time_space_state
from tools.trinity.prompt import call_claude_for_soft_signals
from tools.trinity.verify import verify_trinity_output


def _merge_claude_with_python(claude_output: dict, hard_signals: dict,
                              time_space: dict) -> dict:
    """v2: 将Claude精简输出与Python预算字段合并为完整 pattern_analysis。

    Claude只输出 structure + composite（~12字段）。
    divergence / ma_analysis / key_support / key_resistance / structure_overridden
    全部由Python确定性填充，不再依赖Claude。
    """
    divergence  = compute_divergence_summary(hard_signals)
    ma_analysis = compute_ma_analysis_summary(hard_signals)

    structure = claude_output.get("structure", {})
    # 注入Python预算的支撑压力位
    structure["key_support"]    = hard_signals.get("key_support")
    structure["key_resistance"] = hard_signals.get("key_resistance")
    # structure_overridden 由时空状态决定
    state = time_space.get("daily_state", {})
    structure["structure_overridden"] = state.get("is_extreme", False)

    return {
        "divergence":  divergence,
        "ma_analysis": ma_analysis,
        "structure":   structure,
        "composite":   claude_output.get("composite", {}),
    }


def _enforce_hard_overrides(pattern_analysis: dict, hard_signals: dict) -> dict:
    """后置校验：修正suggested_action里的止损价 + 剥离公式文本。

    v2简化版：key_support/key_resistance已由Python在merge阶段注入，
    这里只处理Claude自由文本中的止损数字和公式泄露。
    """
    composite = pattern_analysis.get("composite", {})

    hs_lsl        = hard_signals.get("long_stop_loss")
    hs_ssl        = hard_signals.get("short_stop_loss")

    # ── 止损价 —— 替换 suggested_action 里的错误止损数字 ─────────────────────
    action = composite.get("suggested_action", "")
    if action and hs_lsl is not None and hs_ssl is not None:
        # 仅在止损价偏差超过 0.5% 时才替换，避免不必要修改
        import re
        # 匹配"止损设在 XXX"或"止损 XXX"或"stop XXX"后面跟随的数字
        pattern = r'(止损[^\d]{0,4})([\d]+\.?\d*)'
        matches = list(re.finditer(pattern, action))
        if matches:
            # 根据信号方向决定应该用哪个止损
            signal = composite.get("signal", "hold")
            if signal in ("buy", "strong_buy"):
                correct_stop = hs_lsl
            elif signal in ("sell", "strong_sell"):
                correct_stop = hs_ssl
            else:
                # hold 信号下根据止损上下文中的持仓方向关键词来决定
                correct_stop = None  # 不做全局替换，按上下文逐个处理

            for m in matches:
                old_val = float(m.group(2))
                if correct_stop is not None:
                    # buy/sell 信号：全局替换
                    stop_to_use = correct_stop
                else:
                    # hold 信号：根据止损前后文判断方向
                    ctx_start = max(0, m.start() - 15)
                    ctx = action[ctx_start:m.end()]
                    if any(kw in ctx for kw in ["多头", "做多", "多仓", "持股", "持仓"]):
                        stop_to_use = hs_lsl
                    elif any(kw in ctx for kw in ["空头", "做空", "空仓", "卖出"]):
                        stop_to_use = hs_ssl
                    else:
                        # 无上下文关键词时，根据止损数值与价格的相对位置判断
                        price = hard_signals.get("current_price", 0)
                        if old_val < price:
                            stop_to_use = hs_lsl  # 止损在价格下方 → 做多止损
                        else:
                            # 止损在价格上方，通常是做空止损。
                            # 但若整体建议是多头观望/不建议追空，应强制换为 long_stop_loss。
                            # 典型bug：Claude在hold信号里写了short_stop_loss数值，
                            # _enforce_hard_overrides误判为"正确"（差值=0），实际方向错误。
                            _long_bias_kws = [
                                "不建议追空", "多头持仓者", "持股者", "轻仓观望",
                                "持多仓", "多头仓位", "已持多仓", "做多止损",
                            ]
                            if any(kw in action for kw in _long_bias_kws):
                                stop_to_use = hs_lsl  # 整体是多头观望，强制用做多止损
                            else:
                                stop_to_use = hs_ssl  # 真正的做空场景
                # 替换条件：偏差 > 0.5%，OR 数值正确但方向错误（wrong type already == correct value 的罕见情况已由上方逻辑修正，这里兜底）
                if abs(old_val - stop_to_use) / max(stop_to_use, 1) > 0.005 or old_val != stop_to_use:
                    action = action.replace(m.group(0), f"{m.group(1)}{stop_to_use}")

        # ── 剥离禁止出现的公式说明（如"（= 支撑3.79 × 0.97）"等） ────────────
        import re as _re
        # 匹配括号型公式：（= ...× 0.97）等
        action = _re.sub(r'[（(]\s*=?\s*[^）)]*×\s*[\d.]+\s*[）)]', '', action)
        # 匹配"（做多止损，支撑 $X × 0.97）"类型
        action = _re.sub(r'[（(][^）)]{0,30}[\d.]+\s*×\s*[\d.]+[^）)]{0,10}[）)]', '', action)
        # 匹配"（根据key_support XXX × 0.97计算）"
        action = _re.sub(r'[（(][^）)]{0,50}计算[^）)]{0,10}[）)]', '', action)
        composite["suggested_action"] = action.strip()

    # ── 超扩延 + 高布林带 + 风险收益比 硬约束 ──────────────────────────────────
    dist_ma55  = hard_signals.get("dist_from_ma55", 0)
    bb_pos     = hard_signals.get("price_position", 0.5)
    cur_price  = hard_signals.get("current_price", 0) or 0
    key_res    = hard_signals.get("key_resistance")
    lsl        = hard_signals.get("long_stop_loss")
    ma55_val   = hard_signals.get("ma55")
    signal     = composite.get("signal", "hold")

    # 风险收益比计算
    rr_ratio = None
    if key_res and lsl and cur_price and key_res > cur_price and cur_price > lsl:
        upside   = (key_res - cur_price) / cur_price
        downside = (cur_price - lsl)     / cur_price
        rr_ratio = round(upside / downside, 2) if downside > 0 else 0

    overextended = dist_ma55 > 0.15 and bb_pos > 0.80
    poor_rr      = rr_ratio is not None and rr_ratio < 1.0

    if signal in ("buy", "strong_buy") and (overextended or poor_rr):
        composite["position_size"] = "light"
        risk_parts = []
        if overextended:
            risk_parts.append(
                f"价格偏离MA55达{dist_ma55*100:.1f}%且布林带位置{int(bb_pos*100)}%，超扩延不宜追高"
            )
        if poor_rr:
            risk_parts.append(
                f"风险收益比{rr_ratio:.2f}<1（至压力位空间不足止损范围），不建议新建仓"
            )
        composite["key_risk"] = "；".join(risk_parts)

        if overextended and poor_rr:
            ma55_str = f"${ma55_val:.2f}" if ma55_val else "MA55"
            composite["suggested_action"] = (
                f"方向看多但当前不宜新建仓，等待回踩{ma55_str}附近黄金棒确认再入场；"
                f"已持仓者观望，止损设在 {lsl}"
            )
            composite["entry_side"] = "wait"

    pattern_analysis["composite"] = composite
    return pattern_analysis


def trinity_analysis(
    ticker: str,
    period: str = "2y",
    holding_days_min: int = 1,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """
    三位一体完整分析，额外返回 price_chart_data 供前端画图。

    price_chart_data 包含：
      - 最近120个交易日的 OHLC + 成交量
      - MA55 / MA233 数值序列
      - 布林带上中下轨
      - MACD柱状图 / DIF / DEA
    前端用这些数据画出真实均线图，可视化验证分析结论。
    """
    ticker = ticker.upper()
    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # ── Step 1: 数据 ──────────────────────────────────────────────────────────
    dfs        = fetch_multi_timeframe(ticker)
    df_daily   = dfs.get("daily",   pd.DataFrame())
    df_weekly  = dfs.get("weekly",  pd.DataFrame())
    df_monthly = dfs.get("monthly", pd.DataFrame())
    df_hourly  = dfs.get("hourly",  pd.DataFrame())

    if df_daily.empty or len(df_daily) < 60:
        return {"error": f"数据不足，无法分析 {ticker}"}

    # ── Step 2: 硬指标 ────────────────────────────────────────────────────────
    hard_signals = compute_all_hard_signals(df_daily)

    bb_hourly = (compute_bollinger_trinity(df_hourly)
                 if not df_hourly.empty and len(df_hourly) >= 25
                 else {"error": "60分钟数据不足"})

    monthly_macd = (compute_macd_signals(df_monthly)
                    if not df_monthly.empty and len(df_monthly) >= 30
                    else {})

    weekly_macd = (compute_macd_signals(df_weekly)
                   if not df_weekly.empty and len(df_weekly) >= 30
                   else {})

    # ── Step 3: 时空状态机 ────────────────────────────────────────────────────
    time_space = compute_time_space_state(
        df_daily=df_daily,
        df_monthly=df_monthly,
        df_weekly=df_weekly,
        bb_hourly=bb_hourly,
        holding_days_min=holding_days_min,
    )

    # ── Step 4: Claude 软判断（v2: 只输出 structure + composite）────────────
    claude_input = {
        **hard_signals,
        "bb_hourly_j_minus1":  bb_hourly,
        "weekly_macd_j1":      weekly_macd,
        "monthly_macd_j2":     monthly_macd,
    }
    claude_output = call_claude_for_soft_signals(
        ticker=ticker,
        hard_signals=claude_input,
        time_space=time_space,
        client=client,
    )

    # ── Step 4b: Python合并 — divergence/ma_analysis/支撑压力由Python填充 ──
    pattern_analysis = _merge_claude_with_python(claude_output, hard_signals, time_space)

    # ── Step 5: 构建前端图表数据（最近120日）────────────────────────────────
    price_chart_data = _build_chart_data(df_daily)

    # ── Step 5b: 后置校验 — 修正suggested_action中止损价 + 剥离公式 ────────
    pattern_analysis = _enforce_hard_overrides(pattern_analysis, hard_signals)

    # ── Step 6: 精简摘要 ──────────────────────────────────────────────────────
    composite = pattern_analysis.get("composite", {})
    structure = pattern_analysis.get("structure", {})
    div       = pattern_analysis.get("divergence", {})
    ma_ana    = pattern_analysis.get("ma_analysis", {})
    state     = time_space.get("daily_state", {})
    main_wave = time_space.get("main_wave", {})
    exit_g    = time_space.get("exit_guidance", {})

    is_locked = (main_wave.get("bollinger_locked", False)
                 and main_wave.get("monthly_extreme_strong", False))

    summary = {
        "state_label":          state.get("state_label", "未知"),
        "state_code":           state.get("current_state", "unknown"),
        "is_extreme":           state.get("is_extreme", False),
        "is_bullish":           state.get("is_bullish", False),
        "bars_in_state":        state.get("bars_in_state", 0),
        "state_anomaly":        state.get("state_anomaly", False),
        "weekly_state_label":   time_space.get("weekly_state", {}).get("state_label", "未知"),
        "weekly_is_bullish":    time_space.get("weekly_state", {}).get("is_bullish", False),
        "j1_weekly_confirmed":  main_wave.get("j1_weekly_confirmed", False),
        "first_assumption":     time_space.get("first_assumption", ""),
        "main_wave_locked":     is_locked,
        "main_wave_note":       main_wave.get("note", ""),
        "pattern_type":         structure.get("pattern_type", "unknown"),
        "current_stage":        structure.get("current_stage", "unknown"),
        "likely_next":          structure.get("likely_next_move", "unknown"),
        "key_support":          structure.get("key_support") or hard_signals.get("key_support"),
        "key_resistance":       structure.get("key_resistance") or hard_signals.get("key_resistance"),
        "long_stop_loss":       hard_signals.get("long_stop_loss"),    # Python预算：key_support×0.97
        "short_stop_loss":      hard_signals.get("short_stop_loss"),   # Python预算：key_resistance×1.03
        "trend_alignment":      hard_signals.get("trend_alignment", "mixed"),
        "ma55":                 hard_signals.get("ma55"),
        "ma233":                hard_signals.get("ma233"),
        "current_price":        hard_signals.get("current_price"),
        "ma_breakout_type":     ma_ana.get("ma55_breakout_type", "none"),
        "ma_breakout_type_py":  hard_signals.get("ma_breakout_type_py", "unknown"),
        "ma_breakout_direction": ma_ana.get("ma55_breakout_direction", ""),
        "pullback_opportunity": ma_ana.get("pullback_opportunity", False),
        "divergence_type":      div.get("divergence_type", "none"),
        "divergence_strength":  div.get("divergence_strength", "none"),
        "divergence_note":      div.get("divergence_note", ""),
        "top_div_maturity":     hard_signals.get("top_div_maturity", "none"),
        "bot_div_maturity":     hard_signals.get("bot_div_maturity", "none"),
        "structure_type_py":    hard_signals.get("structure_type_py", "unknown"),
        "structure_stage_py":   hard_signals.get("structure_current_stage_py", "unknown"),
        "structure_d_to_a":     hard_signals.get("structure_d_to_a_py", False),
        "latest_golden_candle": hard_signals.get("latest_golden_candle"),
        "signal":               composite.get("signal", "hold"),
        "confidence":           composite.get("confidence", "low"),
        "entry_side":           composite.get("entry_side", "wait"),
        "override_active":      composite.get("override_active", False),
        "override_reason":      composite.get("override_reason", ""),
        "suggested_action":     composite.get("suggested_action", ""),
        "key_risk":             composite.get("key_risk", ""),
        "position_size":        composite.get("position_size", "light"),
        "exit_mode":            exit_g.get("mode", "normal"),
        "exit_trigger":         exit_g.get("exit_trigger", ""),
        "reduce_1st_long":      exit_g.get("reduce_1st_long", exit_g.get("reduce_1st", "")),
        "reduce_2nd_long":      exit_g.get("reduce_2nd_long", exit_g.get("reduce_2nd", "")),
        "reduce_1st_short":     exit_g.get("reduce_1st_short", ""),
        "reduce_2nd_short":     exit_g.get("reduce_2nd_short", ""),
        # keep legacy fields for frontend compatibility
        "reduce_1st":           exit_g.get("reduce_1st_long", exit_g.get("reduce_1st", "")),
        "reduce_2nd":           exit_g.get("reduce_2nd_long", exit_g.get("reduce_2nd", "")),
        "holding_constraint":   exit_g.get("holding_constraint_note", ""),
        "multi_timeframe_conflict": time_space.get("multi_timeframe_conflict", False),
        "mtf_conflict_type":       time_space.get("mtf_conflict_type", ""),
        "ma_inverted":             hard_signals.get("ma_inverted", False),
        "ma_alignment_bracket":    hard_signals.get("trend_alignment_bracket", ""),
    }

    # ── Step 7: 验证层 — 确定性规则校验，修正所有已知违规 ────────────────────
    summary = verify_trinity_output(summary, hard_signals, time_space)

    return {
        "ticker":            ticker,
        "time_space_state":  time_space,
        "hard_signals":      hard_signals,
        "pattern_analysis":  pattern_analysis,
        "price_chart_data":  price_chart_data,
        "summary":           summary,
    }


def _build_chart_data(df: pd.DataFrame, n: int = 120) -> list[dict]:
    """
    构建前端图表所需的时序数据。
    每个数据点包含：日期、OHLC、成交量、MA55、MA233、布林带、MACD。
    只返回最近 n 个交易日，控制数据量。
    """
    df = df.copy().tail(n + 233)  # 多取一些用于计算均线

    # 均线
    df["ma55"]  = df["Close"].rolling(55).mean()
    df["ma233"] = df["Close"].rolling(233).mean()

    # 布林带（20日）
    df["bb_mid"]   = df["Close"].rolling(20).mean()
    df["bb_std"]   = df["Close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    # MACD
    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["dif"]      = exp12 - exp26
    df["dea"]      = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd_bar"] = 2 * (df["dif"] - df["dea"])

    # 只取最近 n 条
    df = df.tail(n).dropna(subset=["ma55"])

    result = []
    for idx, row in df.iterrows():
        date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
        result.append({
            "date":     date_str,
            "open":     _r(row.get("Open")),
            "high":     _r(row.get("High")),
            "low":      _r(row.get("Low")),
            "close":    _r(row.get("Close")),
            "volume":   int(row.get("Volume", 0)) if pd.notna(row.get("Volume", 0)) else 0,
            "ma55":     _r(row.get("ma55")),
            "ma233":    _r(row.get("ma233")),
            "bb_upper": _r(row.get("bb_upper")),
            "bb_mid":   _r(row.get("bb_mid")),
            "bb_lower": _r(row.get("bb_lower")),
            "dif":      _r(row.get("dif")),
            "dea":      _r(row.get("dea")),
            "macd_bar": _r(row.get("macd_bar")),
        })
    return result


def _r(v, decimals: int = 4):
    """安全四舍五入，处理 NaN。"""
    try:
        f = float(v)
        return round(f, decimals) if pd.notna(f) else None
    except Exception:
        return None