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
)
from tools.trinity.state import compute_time_space_state
from tools.trinity.prompt import call_claude_for_soft_signals


def _enforce_hard_overrides(pattern_analysis: dict, hard_signals: dict) -> dict:
    """后置校验：将Claude可能篡改的字段强制覆盖为Python预算值。

    已知复发问题：Claude经常忽略硬指标里的 key_support/key_resistance，
    自行挑选历史高低点，导致支撑压力位和止损价全部偏移。
    """
    structure = pattern_analysis.get("structure", {})
    composite = pattern_analysis.get("composite", {})

    # ── 支撑 / 压力 ─────────────────────────────────────────────────────────
    hs_support    = hard_signals.get("key_support")
    hs_resistance = hard_signals.get("key_resistance")
    hs_lsl        = hard_signals.get("long_stop_loss")
    hs_ssl        = hard_signals.get("short_stop_loss")

    if hs_support is not None:
        structure["key_support"] = hs_support
    if hs_resistance is not None:
        structure["key_resistance"] = hs_resistance

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
                # hold 信号下取离当前价较近的止损
                price = hard_signals.get("current_price", 0)
                correct_stop = hs_lsl if abs(price - hs_lsl) < abs(price - hs_ssl) else hs_ssl

            for m in matches:
                old_val = float(m.group(2))
                if abs(old_val - correct_stop) / max(correct_stop, 1) > 0.005:
                    action = action.replace(m.group(0), f"{m.group(1)}{correct_stop}")
            composite["suggested_action"] = action

    pattern_analysis["structure"] = structure
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

    # ── Step 4: Claude 软判断 ─────────────────────────────────────────────────
    claude_input = {
        **hard_signals,
        "bb_hourly_j_minus1":  bb_hourly,
        "weekly_macd_j1":      weekly_macd,
        "monthly_macd_j2":     monthly_macd,
    }
    pattern_analysis = call_claude_for_soft_signals(
        ticker=ticker,
        hard_signals=claude_input,
        time_space=time_space,
        client=client,
    )

    # ── Step 5: 构建前端图表数据（最近120日）────────────────────────────────
    price_chart_data = _build_chart_data(df_daily)

    # ── Step 5b: 后置校验 — 强制覆盖Claude可能篡改的硬指标字段 ───────────────
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
    }

    return {
        "ticker":            ticker,
        "time_space_state":  time_space,
        "hard_signals":      hard_signals,
        "pattern_analysis":  pattern_analysis,
        "price_chart_data":  price_chart_data,   # ← 新增，供前端画图
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