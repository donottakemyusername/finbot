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

更新：
- compute_time_space_state return 新增 extreme_bars_warning（极端状态<3根K线时=True）
  prompt.py 读取此字段，强制将confidence降为low
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
    """根据事件序列推算当前所处的六种状态之一。

    特殊情况处理：
    对于长期牛股（如CIEN、AVGO），DIF可能全程在零轴上方，永远不会出现 bottom_cross。
    此时用 dea_cross_zero_up 或最早的 high_golden_cross 作为起点，
    起点状态定为"强"（因为DEA已过零轴，已经过了极强阶段）。
    """
    if not events:
        return _unknown()

    # ── 优先找 bottom_cross（标准起点）─────────────────────────────────────
    last_bottom = next((e for e in reversed(events) if e["event"] == "bottom_cross"), None)

    # ── fallback：找不到 bottom_cross，说明DIF全程在零轴上方（长期牛股）──
    if last_bottom is None:
        first_dea_up = next((e for e in events if e["event"] == "dea_cross_zero_up"), None)
        if first_dea_up is not None:
            last_bottom = first_dea_up
            start_state = "strong"
        else:
            first_hgc = next((e for e in events if e["event"] == "high_golden_cross"), None)
            if first_hgc is not None:
                last_bottom = first_hgc
                start_state = "strong"
            else:
                return _unknown()
    else:
        start_state = "mid_strong"

    current_state = start_state
    last_event_bar = last_bottom["bar"]
    last_event_name = last_bottom["event"]

    transition_map = {
        "mid_strong":     ["dif_cross_zero_up"],
        "extreme_strong": ["dea_cross_zero_up", "dif_cross_zero_dn", "high_golden_cross"],
        "strong":         ["top_death_cross"],
        "mid_weak":       ["dif_cross_zero_dn", "high_golden_cross"],
        "extreme_weak":   ["dea_cross_zero_dn", "dif_cross_zero_up", "dea_cross_zero_up", "high_golden_cross"],
        "weak":           ["bottom_cross", "dif_cross_zero_up", "dea_cross_zero_up", "high_golden_cross"],
    }
    next_state_map = {
        "dif_cross_zero_up":  "extreme_strong",
        "dea_cross_zero_up":  "strong",
        "top_death_cross":    "mid_weak",
        "dif_cross_zero_dn":  "extreme_weak",
        "dea_cross_zero_dn":  "weak",
        "bottom_cross":       "mid_strong",
        "high_golden_cross":  "strong",
    }

    # 正常状态流转顺序（用于检测异常跳变）——注意这是一个循环
    _NORMAL_ORDER = [
        "mid_strong", "extreme_strong", "strong",
        "mid_weak", "extreme_weak", "weak",
    ]
    _N = len(_NORMAL_ORDER)
    _ORDER_IDX = {s: i for i, s in enumerate(_NORMAL_ORDER)}
    state_anomaly = False
    prev_state    = current_state

    for ev in events:
        if ev["bar"] <= last_bottom["bar"]:
            continue
        if ev["event"] in transition_map.get(current_state, []):
            new_state = next_state_map[ev["event"]]
            # 用循环距离检测异常跳变（正向=顺着周期方向）
            idx_old = _ORDER_IDX.get(current_state, 0)
            idx_new = _ORDER_IDX.get(new_state, 0)
            fwd_gap = (idx_new - idx_old) % _N   # 正向距离
            bwd_gap = (idx_old - idx_new) % _N   # 反向距离
            # 只记录最后一次跳变是否异常（而非任意一次）
            state_anomaly = fwd_gap > 2 or (bwd_gap >= 1 and bwd_gap < _N - 1)
            prev_state     = current_state
            current_state  = new_state
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
        "prev_state":    prev_state,
        "state_anomaly": state_anomaly,
    }


def _unknown() -> dict:
    return {"current_state": "unknown", "state_label": "未知",
            "is_bullish": False, "is_bearish": False, "is_extreme": False,
            "bars_in_state": 0, "last_event": "none",
            "prev_state": "unknown", "state_anomaly": False}


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


def compute_main_wave_lock(
    state_daily: dict,
    state_weekly: dict,
    state_monthly: dict,
    bb_hourly: dict,
) -> dict:
    """
    判断主涨段锁定状态（7 Key Points核心规则）。
    月线极强(J+2) + 60分钟布林带未跌破(J-1) = 锁定中
    """
    monthly_extreme  = state_monthly.get("current_state") == "extreme_strong"
    weekly_bullish   = state_weekly.get("is_bullish", False)
    weekly_state_lbl = state_weekly.get("state_label", "未知")
    daily_bullish    = state_daily.get("is_bullish", False)
    bb_ok   = not bb_hourly.get("error")
    locked  = bb_ok and not bb_hourly.get("below_mid_2bars", False)

    if locked and monthly_extreme:
        if weekly_bullish:
            note = "主涨段锁定中：月线极强+周线多头双确认，忽略所有背离和结构信号，只看J-1布林带"
        else:
            note = f"主涨段锁定中（周线{weekly_state_lbl}，中间层偏弱，注意持仓强度）：只看J-1布林带"
    elif not locked:
        note = "锁定解除：回归三要素判断（均线+结构+背离）"
    else:
        note = "月线未达极强：非主涨段锁定状态"

    return {
        "monthly_extreme_strong":   monthly_extreme,
        "weekly_bullish":           weekly_bullish,
        "weekly_state_label":       weekly_state_lbl,
        "daily_bullish":            daily_bullish,
        "time_space_condition_met": monthly_extreme and daily_bullish,
        "j1_weekly_confirmed":      monthly_extreme and weekly_bullish,
        "bollinger_locked":         locked,
        "bollinger_lock_broken":    not locked,
        "j1_below_mid_2bars":       bb_hourly.get("below_mid_2bars", False),
        "note":                     note,
    }


def compute_exit_guidance(state: dict, main_wave: dict, holding_days_min: int = 1) -> dict:
    """止盈观察级别建议。"""
    is_locked    = main_wave.get("bollinger_locked") and main_wave.get("monthly_extreme_strong")
    is_extreme   = state.get("is_extreme", False)
    holding_note = ""
    if holding_days_min == 30:
        holding_note = "⚠️ 注意：持仓有30天限制，减仓信号出现后需等满持有期再操作"
    elif holding_days_min == 1:
        holding_note = "注意：至少持有1天，日内信号仅供参考，次日才可操作"

    if is_locked:
        return {
            "mode":                 "main_wave_locked",
            "description":          "主涨段锁定中",
            "exit_trigger":         "60分钟连续2根K线跌破布林带中轨",
            "reduce_1st_long":      None,
            "reduce_2nd_long":      None,
            "reduce_1st_short":     None,
            "reduce_2nd_short":     None,
            "reduce_1st":           None,
            "reduce_2nd":           None,
            "ignore_signals":       ["顶背离", "结构前高", "小级别MACD"],
            "holding_constraint_note": holding_note,
        }
    elif is_extreme:
        return {
            "mode":                 "extreme_state",
            "description":          f"极端状态（{state.get('state_label')}），结构信号无效",
            "exit_trigger":         "均线跌破信号（MA55或MA233）",
            "reduce_1st_long":      None,
            "reduce_2nd_long":      None,
            "reduce_1st_short":     None,
            "reduce_2nd_short":     None,
            "reduce_1st":           None,
            "reduce_2nd":           None,
            "ignore_signals":       ["顶背离", "结构"],
            "holding_constraint_note": holding_note,
        }
    else:
        return {
            "mode":                 "normal",
            "description":          "正常状态，三要素均有效",
            "exit_trigger":         "背离+破位双确认",
            "reduce_1st_long":      "15分钟顶背离 + 5分钟破MA55（向下）→ 减仓20-30%",
            "reduce_2nd_long":      "60分钟顶背离 + 15分钟破MA55（向下）→ 再减仓50%",
            "reduce_1st_short":     "15分钟底背离 + 5分钟站上MA55（向上）→ 平空20-30%",
            "reduce_2nd_short":     "60分钟底背离 + 15分钟站上MA55（向上）→ 再平空50%",
            "reduce_1st":           "15分钟顶背离 + 5分钟破MA55（向下）→ 减仓20-30%",
            "reduce_2nd":           "60分钟顶背离 + 15分钟破MA55（向下）→ 再减仓50%",
            "ignore_signals":       [],
            "holding_constraint_note": holding_note,
        }


def compute_time_space_state(
    df_daily: pd.DataFrame,
    df_monthly: pd.DataFrame,
    bb_hourly: dict,
    df_weekly: pd.DataFrame | None = None,
    holding_days_min: int = 1,
) -> dict:
    """完整时空状态计算主入口。"""
    events_daily   = detect_state_events(df_daily)
    state_daily    = compute_current_state(events_daily, len(df_daily))
    boundary_daily = detect_boundary_window(events_daily, len(df_daily))

    events_monthly = detect_state_events(df_monthly)
    state_monthly  = compute_current_state(events_monthly, len(df_monthly))

    if df_weekly is not None and not df_weekly.empty and len(df_weekly) >= 30:
        events_weekly = detect_state_events(df_weekly)
        state_weekly  = compute_current_state(events_weekly, len(df_weekly))
    else:
        state_weekly = _unknown()

    main_wave     = compute_main_wave_lock(state_daily, state_weekly, state_monthly, bb_hourly)
    exit_guidance = compute_exit_guidance(state_daily, main_wave, holding_days_min)

    cs = state_daily["current_state"]
    first_assumption = {
        "extreme_strong": "A类上涨（五段式），极端状态忽略背离持有",
        "strong":         "B类上涨 或 D类下跌（强弱对称）",
        "mid_strong":     "C类（震荡整理为主）",
        "mid_weak":       "C类（震荡整理为主）",
        "weak":           "B类下跌 或 D类上涨（强弱对称）",
        "extreme_weak":   "A类下跌，极端状态忽略背离持空",
    }.get(cs, "未知")

    # ── extreme_bars_warning：极端状态刚触发（<3根K线），信号极不可靠 ──────
    # prompt.py 读取此字段，强制将 confidence 降为 low，避免追单
    is_extreme    = state_daily.get("is_extreme", False)
    bars_in_state = state_daily.get("bars_in_state", 0)
    extreme_bars_warning = bool(is_extreme and bars_in_state < 3)

    # ── state_anomaly：状态机检测到异常跳变（跳过了中间状态） ──────────
    state_anomaly = state_daily.get("state_anomaly", False)

    # ── multi_timeframe_conflict：日线与周/月线方向矛盾 ──────────────────
    # 课程定义：周线分两级处理
    #   硬冲突（弱/极弱）→ confidence≤medium，严格限制信号
    #   软冲突（中性偏弱）→ 仅要求轻仓+提示风险，不封锁信号（日线强时仍可买）
    daily_bull  = state_daily.get("is_bullish", False)
    daily_bear  = state_daily.get("is_bearish", False)
    weekly_bull = state_weekly.get("is_bullish", False)
    weekly_bear = state_weekly.get("is_bearish", False)
    weekly_state_code = state_weekly.get("current_state", "unknown")
    month_bull  = state_monthly.get("is_bullish", False)
    month_bear  = state_monthly.get("is_bearish", False)

    # 周线是否为"强空头"（弱/极弱 → 硬冲突）vs "弱调整期"（中性偏弱 → 软冲突）
    weekly_hard_bear = weekly_state_code in ("weak", "extreme_weak")
    weekly_soft_bear = weekly_state_code == "mid_weak"

    multi_timeframe_conflict = False
    mtf_conflict_severity    = "none"   # "hard" | "soft" | "none"
    mtf_conflict_type = ""

    if daily_bull and (weekly_hard_bear or month_bear):
        # 硬冲突：周线真空头（弱/极弱）或月线偏空
        multi_timeframe_conflict = True
        mtf_conflict_severity    = "hard"
        parts = []
        if weekly_hard_bear:
            parts.append(f"周线{state_weekly.get('state_label', '空')}（真空头）")
        if month_bear:
            parts.append(f"月线{state_monthly.get('state_label', '空')}")
        mtf_conflict_type = f"日线多头（{state_daily.get('state_label')}）但{'、'.join(parts)}，大级别偏空"
    elif daily_bull and weekly_soft_bear:
        # 软冲突：周线中性偏弱（正常调整期），不封锁信号，仅要求轻仓
        multi_timeframe_conflict = True
        mtf_conflict_severity    = "soft"
        mtf_conflict_type = (
            f"日线多头（{state_daily.get('state_label')}）但周线中性偏弱（调整期，非趋势反转），"
            f"可操作但需轻仓"
        )
    elif daily_bear and (weekly_bull or month_bull):
        multi_timeframe_conflict = True
        mtf_conflict_severity    = "hard"
        parts = []
        if weekly_bull:
            parts.append(f"周线{state_weekly.get('state_label', '多')}")
        if month_bull:
            parts.append(f"月线{state_monthly.get('state_label', '多')}")
        mtf_conflict_type = f"日线空头（{state_daily.get('state_label')}）但{'、'.join(parts)}，大级别偏多"

    return {
        "daily_state":           state_daily,
        "weekly_state":          state_weekly,
        "monthly_state":         state_monthly,
        "boundary_window":       boundary_daily,
        "main_wave":             main_wave,
        "exit_guidance":         exit_guidance,
        "first_assumption":      first_assumption,
        "extreme_bars_warning":  extreme_bars_warning,  # ← True时prompt强制confidence=low
        "state_anomaly":         state_anomaly,         # ← True时prompt提示状态跳变异常
        "multi_timeframe_conflict": multi_timeframe_conflict,
        "mtf_conflict_severity":    mtf_conflict_severity,
        "mtf_conflict_type":        mtf_conflict_type,
    }