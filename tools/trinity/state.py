"""tools/trinity/state.py
=========================
Layer 2: 时空状态机，100% Python实现，无需Claude API。

六种状态（按课程严格定义）：
  中性偏强 → 底部金叉后，DIF第一次上穿零轴
  极强     → DIF上穿零轴 → DEA上穿零轴（这段区间）
  强       → DEA上穿零轴 → 高位死叉
  中性偏弱 → 高位死叉 → DIF第一次下穿零轴
  极弱     → DIF下穿零轴 → DEA下穿零轴
  弱       → DEA下穿零轴 → 下一次底部金叉
"""
from __future__ import annotations
import pandas as pd

STATE_LABELS = {
    "mid_strong": "中性偏强", "extreme_strong": "极强", "strong": "强",
    "mid_weak": "中性偏弱", "extreme_weak": "极弱", "weak": "弱", "unknown": "未知",
}
IS_BULLISH = {"mid_strong", "extreme_strong", "strong"}
IS_BEARISH = {"mid_weak", "extreme_weak", "weak"}
IS_EXTREME = {"extreme_strong", "extreme_weak"}


def _macd_series(df: pd.DataFrame):
    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    dif   = exp12 - exp26
    dea   = dif.ewm(span=9, adjust=False).mean()
    return dif.values, dea.values


def detect_state_events(df: pd.DataFrame) -> list[dict]:
    """按时间顺序提取所有状态切换事件。"""
    if df.empty or len(df) < 30:
        return []
    dif, dea = _macd_series(df)
    n = len(dif)
    events = []
    for i in range(1, n):
        d0, d1 = float(dif[i-1]), float(dif[i])
        e0, e1 = float(dea[i-1]), float(dea[i])
        if d0 <= 0 < d1:   events.append({"bar": i, "event": "dif_cross_zero_up"})
        elif d0 >= 0 > d1: events.append({"bar": i, "event": "dif_cross_zero_dn"})
        if e0 <= 0 < e1:   events.append({"bar": i, "event": "dea_cross_zero_up"})
        elif e0 >= 0 > e1: events.append({"bar": i, "event": "dea_cross_zero_dn"})
        if d0 <= e0 and d1 > e1:
            events.append({"bar": i, "event": "bottom_cross" if d1 < 0 else "high_golden_cross"})
        elif d0 >= e0 and d1 < e1:
            events.append({"bar": i, "event": "top_death_cross" if d1 > 0 else "low_death_cross"})
    priority = {"dif_cross_zero_up": 0, "dif_cross_zero_dn": 0,
                "dea_cross_zero_up": 1, "dea_cross_zero_dn": 1,
                "bottom_cross": 2, "top_death_cross": 2,
                "high_golden_cross": 3, "low_death_cross": 3}
    events.sort(key=lambda e: (e["bar"], priority.get(e["event"], 9)))
    return events


def compute_current_state(events: list[dict], total_bars: int) -> dict:
    """根据事件序列推算当前所处的六种状态之一。"""
    if not events:
        return _unknown()
    last_bottom = next((e for e in reversed(events) if e["event"] == "bottom_cross"), None)
    if last_bottom is None:
        return _unknown()

    current_state = "mid_strong"
    last_event_bar = last_bottom["bar"]
    last_event_name = "bottom_cross"

    transition_map = {
        "mid_strong":     ["dif_cross_zero_up"],
        "extreme_strong": ["dea_cross_zero_up"],
        "strong":         ["top_death_cross"],
        "mid_weak":       ["dif_cross_zero_dn"],
        "extreme_weak":   ["dea_cross_zero_dn"],
        "weak":           ["bottom_cross"],
    }
    next_state_map = {
        "dif_cross_zero_up": "extreme_strong", "dea_cross_zero_up": "strong",
        "top_death_cross": "mid_weak", "dif_cross_zero_dn": "extreme_weak",
        "dea_cross_zero_dn": "weak", "bottom_cross": "mid_strong",
    }
    for ev in events:
        if ev["bar"] <= last_bottom["bar"]:
            continue
        if ev["event"] in transition_map.get(current_state, []):
            current_state  = next_state_map[ev["event"]]
            last_event_bar = ev["bar"]
            last_event_name = ev["event"]

    bars_in_state = max(0, total_bars - 1 - last_event_bar)
    return {
        "current_state": current_state,
        "state_label":   STATE_LABELS.get(current_state, "未知"),
        "is_bullish":    current_state in IS_BULLISH,
        "is_bearish":    current_state in IS_BEARISH,
        "is_extreme":    current_state in IS_EXTREME,
        "bars_in_state": bars_in_state,
        "last_event":    last_event_name,
    }


def _unknown() -> dict:
    return {"current_state": "unknown", "state_label": "未知",
            "is_bullish": False, "is_bearish": False, "is_extreme": False,
            "bars_in_state": 0, "last_event": "none"}


def detect_boundary_window(events: list[dict], total_bars: int, window: int = 3) -> dict:
    """检测是否处于临界3根K窗口（金叉/死叉前后各3根K）。"""
    current_bar = total_bars - 1
    boundary_map = {
        "bottom_cross":      "extreme_strong_boundary",
        "top_death_cross":   "extreme_weak_boundary",
        "high_golden_cross": "strong_boundary",
        "low_death_cross":   "weak_boundary",
    }
    for ev in reversed(events):
        if ev["event"] not in boundary_map:
            continue
        distance = current_bar - ev["bar"]
        if 0 <= distance <= window:
            return {"in_boundary": True, "boundary_type": boundary_map[ev["event"]],
                    "trigger_event": ev["event"], "bars_since_event": distance}
    return {"in_boundary": False, "boundary_type": None, "trigger_event": None, "bars_since_event": None}


def compute_main_wave_lock(state_daily: dict, state_monthly: dict, bb_hourly: dict) -> dict:
    """
    判断主涨段锁定状态（7 Key Points核心规则）。
    月线极强(J+2) + 60分钟布林带未跌破(J-1) = 锁定中
    """
    monthly_extreme = state_monthly.get("current_state") == "extreme_strong"
    daily_bullish   = state_daily.get("is_bullish", False)
    bb_ok   = not bb_hourly.get("error")
    locked  = bb_ok and not bb_hourly.get("below_mid_2bars", False)

    if locked and monthly_extreme:
        note = "主涨段锁定中：忽略所有背离和结构信号，只看J-1布林带"
    elif not locked:
        note = "锁定解除：回归三要素判断（均线+结构+背离）"
    else:
        note = "月线未达极强：非主涨段锁定状态"

    return {
        "monthly_extreme_strong":   monthly_extreme,
        "daily_bullish":            daily_bullish,
        "time_space_condition_met": monthly_extreme and daily_bullish,
        "bollinger_locked":         locked,
        "bollinger_lock_broken":    not locked,
        "j1_below_mid_2bars":       bb_hourly.get("below_mid_2bars", False),
        "note":                     note,
    }


def compute_exit_guidance(state: dict, main_wave: dict, holding_days_min: int = 1) -> dict:
    """
    止盈观察级别建议。
    holding_days_min: 最少持有天数限制（0=无限制，1=至少持有1天，30=30天锁定期）
    """
    is_locked     = main_wave.get("bollinger_locked") and main_wave.get("monthly_extreme_strong")
    is_extreme    = state.get("is_extreme", False)
    holding_note  = ""
    if holding_days_min == 30:
        holding_note = "⚠️ 注意：持仓有30天限制，减仓信号出现后需等满持有期再操作"
    elif holding_days_min == 1:
        holding_note = "注意：至少持有1天，日内信号仅供参考，次日才可操作"

    if is_locked:
        return {
            "mode": "main_wave_locked",
            "description": "主涨段锁定中",
            "exit_trigger": "60分钟连续2根K线跌破布林带中轨",
            "reduce_1st": None, "reduce_2nd": None,
            "ignore_signals": ["顶背离", "结构前高", "小级别MACD"],
            "holding_constraint_note": holding_note,
        }
    elif is_extreme:
        return {
            "mode": "extreme_state",
            "description": f"极端状态（{state.get('state_label')}），结构信号无效",
            "exit_trigger": "均线跌破信号（MA55或MA233）",
            "reduce_1st": None, "reduce_2nd": None,
            "ignore_signals": ["顶背离", "结构"],
            "holding_constraint_note": holding_note,
        }
    else:
        return {
            "mode": "normal",
            "description": "正常状态，三要素均有效",
            "exit_trigger": "背离+破位双确认",
            "reduce_1st": "15分钟顶背离 + 5分钟破MA55 → 减仓20-30%",
            "reduce_2nd": "60分钟顶背离 + 15分钟破MA55 → 再减仓50%",
            "ignore_signals": [],
            "holding_constraint_note": holding_note,
        }


def compute_time_space_state(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
    bb_hourly: dict,
    holding_days_min: int = 1,
) -> dict:
    """完整时空状态计算主入口。"""
    events_daily   = detect_state_events(df_daily)
    state_daily    = compute_current_state(events_daily, len(df_daily))
    boundary_daily = detect_boundary_window(events_daily, len(df_daily))
    events_monthly = detect_state_events(df_monthly)
    state_monthly  = compute_current_state(events_monthly, len(df_monthly))
    main_wave      = compute_main_wave_lock(state_daily, state_monthly, bb_hourly)
    exit_guidance  = compute_exit_guidance(state_daily, main_wave, holding_days_min)

    cs = state_daily["current_state"]
    first_assumption = {
        "extreme_strong": "A类上涨（五段式），极端状态忽略背离持有",
        "strong":         "B类上涨 或 D类下跌（强弱对称）",
        "mid_strong":     "C类（震荡整理为主）",
        "mid_weak":       "C类（震荡整理为主）",
        "weak":           "B类下跌 或 D类上涨（强弱对称）",
        "extreme_weak":   "A类下跌，极端状态忽略背离持空",
    }.get(cs, "未知")

    return {
        "daily_state":      state_daily,
        "monthly_state":    state_monthly,
        "boundary_window":  boundary_daily,
        "main_wave":        main_wave,
        "exit_guidance":    exit_guidance,
        "first_assumption": first_assumption,
    }