"""tools/trinity/verify.py
==========================
三位一体输出验证层（Python 确定性规则校验）。

在 Claude 输出之后、前端展示之前运行，强制执行所有已知规则。
发现违规时就地修正并记录到 summary['_corrections']。

规则来源：
  - 三位一体课程硬规则
  - 历史 bug 总结（每次发现新问题后在此新增规则）
"""
from __future__ import annotations


# ─── 优先级顺序（数字越小越先执行，越高优先级规则可覆盖低优先级）──────────
_CONFIDENCE_ORDER = ["low", "medium", "high"]
_POSITION_ORDER   = ["none", "light", "moderate", "heavy"]


def verify_trinity_output(
    summary:      dict,
    hard_signals: dict,
    time_space:   dict,
) -> dict:
    """
    对 trinity_analysis 的 summary 做最终规则校验。
    就地修改 summary，返回修改后的 summary。
    所有修改记录在 summary['_corrections'] 列表中。
    """
    summary     = summary.copy()
    corrections: list[str] = []

    # ── 读取常用字段 ──────────────────────────────────────────────────────────
    signal         = summary.get("signal", "hold")
    confidence     = summary.get("confidence", "low")
    position_size  = summary.get("position_size", "light")
    entry_side     = summary.get("entry_side", "wait")
    state_code     = summary.get("state_code", "unknown")
    is_locked      = summary.get("main_wave_locked", False)
    likely_next    = summary.get("likely_next", "unknown")
    struct_type    = (summary.get("pattern_type")
                      or summary.get("structure_type_py") or "unknown")

    # hard_signals
    ma_inverted    = hard_signals.get("ma_inverted", False)
    dist_ma55      = hard_signals.get("dist_from_ma55", 0)
    bb_pos         = hard_signals.get("price_position", 0.5)
    cur_price      = float(hard_signals.get("current_price") or 0)
    key_res        = hard_signals.get("key_resistance")
    lsl            = hard_signals.get("long_stop_loss")
    ma55_val       = hard_signals.get("ma55")
    top_div_valid  = hard_signals.get("top_divergence_hard_valid", False)
    bot_div_valid  = hard_signals.get("bot_divergence_hard_valid", False)
    live_warning   = hard_signals.get("live_top_div_warning", False)
    adj_suff       = hard_signals.get("adjustment_sufficient", False)
    trend_align    = hard_signals.get("trend_alignment", "mixed")
    overext_hard   = hard_signals.get("overextension_hard", False)

    # time_space
    extreme_warn   = time_space.get("extreme_bars_warning", False)
    mtf_conflict   = time_space.get("multi_timeframe_conflict", False)

    # ── 辅助函数 ─────────────────────────────────────────────────────────────
    def _cap_confidence(target: str, reason: str) -> None:
        nonlocal confidence
        if _CONFIDENCE_ORDER.index(confidence) > _CONFIDENCE_ORDER.index(target):
            corrections.append(f"[confidence] {confidence} → {target}：{reason}")
            confidence = target

    def _cap_position(target: str, reason: str) -> None:
        nonlocal position_size
        if _POSITION_ORDER.index(position_size) > _POSITION_ORDER.index(target):
            corrections.append(f"[position_size] {position_size} → {target}：{reason}")
            position_size = target

    def _set_signal(target: str, reason: str) -> None:
        nonlocal signal
        if signal != target:
            corrections.append(f"[signal] {signal} → {target}：{reason}")
            signal = target

    def _set_entry(target: str, reason: str) -> None:
        nonlocal entry_side
        if entry_side != target:
            corrections.append(f"[entry_side] {entry_side} → {target}：{reason}")
            entry_side = target

    def _append_risk(note: str) -> None:
        existing = summary.get("key_risk", "")
        if note not in existing:
            summary["key_risk"] = (existing + "；" + note).lstrip("；")

    # ═══════════════════════════════════════════════════════════════════════════
    # 规则集（按优先级从高到低）
    # ═══════════════════════════════════════════════════════════════════════════

    # ── R01：主涨段锁定 → 强制 hold ──────────────────────────────────────────
    if is_locked and signal != "hold":
        _set_signal("hold", "主涨段锁定期间忽略方向信号")
        summary["override_active"] = True
        summary["override_reason"] = "主涨段锁定"

    # ── R02：极端状态刚触发（<3 根 K 线）→ 降级，等待确认 ─────────────────────
    if extreme_warn:
        _cap_confidence("low",    "极端状态刚触发 <3 根 K 线，信号不可靠")
        _cap_position("light",    "极端状态刚触发，轻仓观察")

    # ── R03：极强+空头排列 / 极弱+多头排列 → 降为 hold ────────────────────────
    if state_code == "extreme_strong" and trend_align == "bearish":
        if signal in ("buy", "strong_buy"):
            _set_signal("hold", "极强但均线空头排列（疑似熊市反弹）")
        _cap_confidence("medium", "极强+空头排列矛盾")

    if state_code == "extreme_weak" and trend_align == "bullish":
        if signal in ("sell", "strong_sell"):
            _set_signal("hold", "极弱但均线多头排列（疑似洗盘）")
        _cap_confidence("medium", "极弱+多头排列矛盾")

    # ── R04：极端状态信号边界 ──────────────────────────────────────────────────
    if state_code == "extreme_weak" and signal == "strong_buy":
        _set_signal("hold", "极弱状态禁止 strong_buy")
    if state_code == "extreme_strong" and signal == "strong_sell":
        _set_signal("hold", "极强状态禁止 strong_sell")

    # ── R05：信号与时空状态严重矛盾 → confidence 降低 ──────────────────────────
    _WEAK_STATES   = ("weak", "extreme_weak", "mid_weak")
    _STRONG_STATES = ("strong", "extreme_strong", "mid_strong")

    if signal in ("buy", "strong_buy") and state_code in _WEAK_STATES:
        _cap_confidence("low", f"买入信号与弱状态（{state_code}）矛盾")
    if signal in ("sell", "strong_sell") and state_code in _STRONG_STATES:
        _cap_confidence("low", f"卖出信号与强状态（{state_code}）矛盾")

    # ── R06：多时间框架级别冲突 → confidence ≤ medium ─────────────────────────
    if mtf_conflict:
        _cap_confidence("medium", "多时间框架级别冲突")
        _cap_position("moderate", "级别冲突时仓位不宜过重")

    # ── R07：均线倒置（MA55 < MA233）→ confidence ≤ medium ────────────────────
    if ma_inverted:
        _cap_confidence("medium", "均线倒置（MA55 < MA233），长期趋势未修复")
        if signal in ("buy", "strong_buy"):
            _append_risk("均线倒置：MA55 需上穿 MA233 才算趋势真正修复")

    # ── R08：顶背离有效 → 降低仓位，不允许 strong_buy ──────────────────────────
    if top_div_valid:
        _cap_position("moderate", "顶背离有效，降低仓位上限")
        if signal == "strong_buy":
            _set_signal("buy", "顶背离有效时不允许 strong_buy")

    # ── R09：实时顶背离预警 → confidence ≤ medium，仓位轻 ─────────────────────
    if live_warning and signal in ("buy", "strong_buy"):
        _cap_confidence("medium", "实时顶背离预警（当前价超前高但 MACD 动能弱）")
        _cap_position("light",    "实时顶背离预警，轻仓观察")
        _append_risk("实时顶背离预警：新高动能不足，注意见顶风险")

    # ── R10：底背离调整不充分 → 底背离无效（兜底校验）─────────────────────────
    if bot_div_valid and not adj_suff:
        hard_signals["bot_divergence_hard_valid"] = False
        corrections.append("[bot_div] 底背离无效：60 日内 DIF/DEA 未穿越零轴，调整不充分")

    # ── R11：价格超扩延 → 不宜追高 ──────────────────────────────────────────────
    # 超扩延条件：MA55偏离>15%且布林>80%，或者 overextension_hard=True（含MA233偏离>40%）
    overextended = (dist_ma55 > 0.15 and bb_pos > 0.80) or overext_hard
    if overextended and signal in ("buy", "strong_buy"):
        _cap_position("light", f"超扩延（MA55偏离{dist_ma55*100:.1f}%，布林{int(bb_pos*100)}%）")
        _set_entry("wait",    "超扩延不宜追高，等待回踩 MA55")
        ma55_str = f"${ma55_val:.2f}" if ma55_val else "MA55"
        _append_risk(f"价格超扩延，等待回踩 {ma55_str} 附近黄金棒确认再入场")

    # ── R12：风险收益比计算 ────────────────────────────────────────────────────
    rr_ratio = None
    if key_res and lsl and cur_price and key_res > cur_price and cur_price > lsl:
        upside   = (key_res - cur_price) / cur_price
        downside = (cur_price - lsl)     / cur_price
        rr_ratio = round(upside / downside, 2) if downside > 0 else 0.0
    summary["rr_ratio"] = rr_ratio

    # buy/strong_buy：RR < 1 → 不建议新建仓
    if rr_ratio is not None and rr_ratio < 1.0 and signal in ("buy", "strong_buy"):
        _cap_position("light", f"风险收益比 {rr_ratio:.2f} < 1")
        _set_entry("wait",     f"风险收益比 {rr_ratio:.2f} 不足 1")
        _append_risk(
            f"风险收益比 {rr_ratio:.2f}（至压力 ${key_res:.2f} 空间 < 止损 ${lsl:.2f} 距离）"
        )

    # hold：RR 较差（<0.5）→ 附加警告，禁止文字建议加仓
    if rr_ratio is not None and rr_ratio < 0.5 and signal == "hold":
        severity = "极差" if rr_ratio < 0.2 else "较差"
        _append_risk(
            f"风险收益比{severity} {rr_ratio:.2f}（至近端压力 ${key_res:.2f} 仅"
            f" {(key_res - cur_price):.2f} 点，止损距离 {(cur_price - lsl):.2f} 点）"
            f"，不建议在当前位置加仓，等待突破 ${key_res:.2f} 后确认再操作"
        )
        corrections.append(
            f"[rr_warning] HOLD信号下RR={rr_ratio:.2f}{severity}，已附加key_risk警告"
        )

    # ── R13：黄金棒陈旧检查 ───────────────────────────────────────────────────
    # 若最新黄金棒的收盘价比当前价高 >3%，说明价格已跌穿黄金棒信号区，信号失效
    latest_gc = hard_signals.get("latest_golden_candle")
    if latest_gc and latest_gc.get("confirmed") and cur_price > 0:
        gc_price = float(latest_gc.get("price", 0))
        if gc_price > 0 and gc_price > cur_price * 1.03:
            stale_pct = (gc_price - cur_price) / cur_price * 100
            _append_risk(
                f"黄金棒（${gc_price:.2f}）在当前价${cur_price:.2f}上方{stale_pct:.1f}%，"
                f"价格已跌穿黄金棒信号区，多头支撑信号已失效"
            )
            corrections.append(
                f"[golden_candle] 黄金棒${gc_price:.2f}高于当前价{stale_pct:.1f}%，信号陈旧"
            )

    # ── R14：价格贴近或已跌破关键支撑 → 破位预警 ────────────────────────────
    key_sup = hard_signals.get("key_support") or summary.get("key_support")
    if key_sup and cur_price > 0:
        dist_to_sup = (cur_price - float(key_sup)) / cur_price
        if 0 < dist_to_sup < 0.03:
            _append_risk(
                f"当前价${cur_price:.2f}距关键支撑${float(key_sup):.2f}仅"
                f"{dist_to_sup*100:.1f}%，破位风险高，请设好止损"
            )
            corrections.append(
                f"[support_proximity] 价格距支撑仅{dist_to_sup*100:.1f}%，已附加破位预警"
            )
        elif dist_to_sup < 0:
            # 价格已跌破支撑位
            broken_pct = abs(dist_to_sup) * 100
            _append_risk(
                f"关键支撑${float(key_sup):.2f}已被跌破（当前价${cur_price:.2f}，"
                f"跌破{broken_pct:.1f}%），结构支撑失效，多头止损参考价需重新评估"
            )
            corrections.append(
                f"[support_broken] 价格已跌破支撑${float(key_sup):.2f}（跌破{broken_pct:.1f}%），已附加破位警告"
            )

    # ── R16：B 类结构未突破 → likely_next 不能为 "up" ─────────────────────────
    if struct_type == "B" and likely_next == "up":
        broken_out = (key_res is not None and cur_price > float(key_res) * 1.03)
        if not broken_out:
            summary["likely_next"] = "sideways"
            corrections.append("[structure] B 类未突破压力位 → likely_next: up → sideways")

    # ── R17：high confidence 需两维度共振，单维度降为 medium ──────────────────
    if confidence == "high":
        top_div_risk   = top_div_valid or live_warning
        state_misalign = (
            (signal in ("buy",  "strong_buy")  and state_code in _WEAK_STATES) or
            (signal in ("sell", "strong_sell") and state_code in _STRONG_STATES)
        )
        if top_div_risk or state_misalign or ma_inverted or mtf_conflict:
            _cap_confidence("medium", "存在矛盾信号，high confidence 需两维度完全共振")

    # ═══════════════════════════════════════════════════════════════════════════
    # 写回修正后的字段
    # ═══════════════════════════════════════════════════════════════════════════
    summary["signal"]        = signal
    summary["confidence"]    = confidence
    summary["position_size"] = position_size
    summary["entry_side"]    = entry_side
    summary["_corrections"]  = corrections

    return summary
