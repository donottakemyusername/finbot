"""tools/trinity/prompt.py
==========================
Layer 3: 构建Claude软判断prompt。

v2架构：Claude只输出 structure + composite（~12字段），
        divergence / ma_analysis / 支撑压力 / 止损价 全部由Python确定性计算。
"""
from __future__ import annotations
import json, os
import anthropic

from tools.trinity.state import STATE_LABELS

SYSTEM_PROMPT = """
你是三位一体（Trinity Trading System）交易系统的分析引擎。

职责（v2精简版）：
1. 接收Python预算好的量化数据
2. 你只做两件事：① 结构分类（ABCD型）② 综合信号判断
3. 严格按JSON格式输出，不输出任何其他内容

⚠️ 你不需要输出 divergence 和 ma_analysis —— 这两个模块已完全由Python计算。
   你只需在 composite 的文本字段中引用预算好的数字（如止损价）。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心规则
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【规则1】主涨段锁定时
→ signal=hold, override_active=true

【规则2】极端状态（极强/极弱）
→ structure_overridden=true
→ 极强：signal只能hold或buy；极弱：signal只能hold或sell
→ 极强+空头排列 → signal=hold, confidence≤medium
→ 极弱+多头排列 → signal=hold, confidence≤medium
→ extreme_bars_warning=true → confidence=low, position_size=light,
  suggested_action含"等待3根K线确认", key_risk含"信号不可靠"

【规则3】时空状态与信号
→ 中性偏强/中性偏弱：C类震荡优先hold，三要素共振才给方向性信号
→ 强：偏向hold/buy    → 弱：偏向hold/sell

【规则4】confidence约束
→ high需两维度共振，单维度只给medium/low
→ 时空与信号矛盾时降为low

【规则5】级别冲突（multi_timeframe_conflict=true）
→ 大级别优先，confidence≤medium，position_size≤moderate
→ key_risk必须提及周/月线方向

【规则6】止损写法
→ 做多止损 = long_stop_loss数字，做空止损 = short_stop_loss数字
→ 格式：「止损设在 <数字>」，不写百分比/公式/括号
→ hold+多头观望 → 只写long_stop_loss

【规则7】状态标签精确
→ 弱≠极弱，极强≠强，必须用state_label原文

【规则8】B类结构方向判断
→ B类（双平台）本质是震荡，likely_next_move默认填"sideways"
→ 仅当价格已明确突破双平台上边界（偏离高点>3%）且时空状态为强时，才可填"up"
→ structure_note须写明"双平台上沿突破"或"双平台震荡整理"，不可简单写"上涨"

【规则9】均线倒置（ma_inverted=true）
→ 均线倒置指 MA55 < MA233，长期趋势未修复，属于"恢复性反弹"而非标准趋势延伸
→ 均线倒置时 confidence 不能给 high，最高 medium
→ key_risk 须包含"均线倒置，MA55需上穿MA233才算趋势修复"
""".strip()


FEW_SHOT_EXAMPLES = """
---
## Few-Shot 案例（只输出 structure + composite）

### 案例1：中性偏强+顶背离+均线破位 ⚠️ 最容易出错

输入摘要：
- daily_state: "mid_strong"（中性偏强）← 关键
- Python预判：top_divergence=valid, ma_breakout=A类向下
- turning_points: 4个拐点 [66, 96, 75, 85]

错误：signal="sell", confidence="high" ← 中性偏强不能high sell

正确输出：
```json
{
  "structure": {
    "pattern_type": "D",
    "trend_direction": "down",
    "current_stage": "3",
    "d_to_a_probability": "low",
    "likely_next_move": "sideways",
    "structure_note": "高位回落D类第三段，96是前高压力"
  },
  "composite": {
    "signal": "hold",
    "confidence": "medium",
    "entry_side": "wait",
    "signals_aligned": false,
    "override_active": false,
    "override_reason": "",
    "primary_basis": "时空中性偏强与顶背离矛盾，降低置信度",
    "suggested_action": "多头持仓者观望，不建议追空；止损设在 64.02",
    "key_risk": "中性偏强做空风险大，可能震荡整理而非单边下跌",
    "position_size": "light"
  }
}
```

---
### 案例2：主涨段锁定（极强状态）

输入摘要：
- daily_state: "extreme_strong", main_wave locked
- Python预判：top_divergence=valid（但锁定中无效）

正确输出：
```json
{
  "structure": {
    "pattern_type": "A",
    "trend_direction": "up",
    "current_stage": "3",
    "d_to_a_probability": "na",
    "likely_next_move": "up",
    "structure_note": "极强状态，结构仅供参考"
  },
  "composite": {
    "signal": "hold",
    "confidence": "high",
    "entry_side": "wait",
    "signals_aligned": true,
    "override_active": true,
    "override_reason": "主涨段锁定中，忽略顶背离",
    "primary_basis": "J-1布林带锁定",
    "suggested_action": "持有多头，止损设在60分钟布林带中轨下方",
    "key_risk": "末期加速可能突然结束",
    "position_size": "heavy"
  }
}
```

---
### 案例3：极弱+空头+调整不充分

输入摘要：
- daily_state: "extreme_weak", trend_alignment: "bearish"
- Python预判：bot_divergence=invalid（调整不充分）

正确输出：
```json
{
  "structure": {
    "pattern_type": "A",
    "trend_direction": "down",
    "current_stage": "3",
    "d_to_a_probability": "na",
    "likely_next_move": "down",
    "structure_note": "极弱A类主跌段，不要抄底"
  },
  "composite": {
    "signal": "sell",
    "confidence": "high",
    "entry_side": "wait",
    "signals_aligned": true,
    "override_active": false,
    "override_reason": "",
    "primary_basis": "极弱+空头排列+调整不充分",
    "suggested_action": "持空或观望，多头仓位建议减仓；止损设在 21.26",
    "key_risk": "极弱状态下底背离可能背了又背",
    "position_size": "light"
  }
}
```
---
"""


# ─────────────────────────────────────────────────────────────────────────────
# Token优化：精简数据
# ─────────────────────────────────────────────────────────────────────────────

# Claude不需要的大序列（Python已消化）
_HEAVY_KEYS = {
    "macd_bar_history_60",
    "recent_30_closes",
    "price_vs_ma55_last10",
    # v2新增：背离原始数据也不再需要Claude处理
    "top_divergence_raw",
    "bot_divergence_raw",
    # v2+新增：关键K线列表由Python处理，只传 latest_golden_candle 单条
    "key_candles_last20",
}


def _slim_hard_signals(signals: dict) -> dict:
    """去掉Claude不需要的字段，只保留决策摘要。"""
    return {k: v for k, v in signals.items() if k not in _HEAVY_KEYS}


def _slim_time_space(ts: dict) -> dict:
    """精简time_space：只保留状态摘要。"""
    slim = {}
    for k, v in ts.items():
        if isinstance(v, dict):
            slim[k] = {sk: sv for sk, sv in v.items()
                       if not isinstance(sv, list) or len(sv) <= 5}
        else:
            slim[k] = v
    return slim


def build_prompt(ticker: str, hard_signals: dict, time_space: dict) -> str:
    state      = time_space.get("daily_state", {})
    main_wave  = time_space.get("main_wave", {})
    is_locked  = main_wave.get("bollinger_locked") and main_wave.get("monthly_extreme_strong")
    is_extreme = state.get("is_extreme", False)
    state_code = state.get("current_state", "unknown")

    extreme_bars_warning = time_space.get("extreme_bars_warning", False)
    bars_in_state        = state.get("bars_in_state", 0)

    # ── emphasis 块（告诉Claude当前最重要的约束）────────────────────────────
    if is_locked:
        emphasis = "⚠️ 主涨段锁定 → signal=hold, override_active=true。"
    elif is_extreme and extreme_bars_warning:
        emphasis = (
            f"🚨 极端状态刚触发（{bars_in_state}根K线）→ confidence=low, position_size=light, "
            f"signal=hold（不追单）, suggested_action含'等待3根K线确认'。"
        )
    elif is_extreme:
        ma_alignment = hard_signals.get("trend_alignment", "mixed")
        if (state_code == "extreme_strong" and ma_alignment == "bearish"):
            emphasis = f"🚨 极强但均线空头排列 → signal=hold, confidence≤medium（熊市反弹？）"
        elif (state_code == "extreme_weak" and ma_alignment == "bullish"):
            emphasis = f"🚨 极弱但均线多头排列 → signal=hold, confidence≤medium（洗盘？）"
        else:
            emphasis = (
                f"⚠️ 极端状态：{state.get('state_label')}，structure_overridden=true。"
                + (" signal只能hold/buy。" if state_code == "extreme_strong" else " signal只能hold/sell。")
            )
    elif state_code in ("mid_strong", "mid_weak"):
        emphasis = f"当前{state.get('state_label')}，C类震荡优先 → signal优先hold。"
    elif state_code == "strong":
        emphasis = "当前：强，偏多 → hold/buy为主。"
    elif state_code == "weak":
        emphasis = "当前：弱，偏空 → hold/sell为主。"
    else:
        emphasis = "正常状态，综合判断。"

    # ── 异常/冲突附加 ────────────────────────────────────────────────────────
    extras = []
    if time_space.get("state_anomaly", False):
        prev = state.get("prev_state", "unknown")
        extras.append(f"🚨 状态跳变：{STATE_LABELS.get(prev, prev)}→{state.get('state_label')}，confidence应降低。")
    if time_space.get("multi_timeframe_conflict", False):
        extras.append(f"🚨 级别冲突：{time_space.get('mtf_conflict_type', '')} → confidence≤medium。")
    if hard_signals.get("ma_inverted", False):
        extras.append(
            "⚠️ 均线倒置（MA55 < MA233）：价格虽超越双均线，但长期趋势未修复。"
            "confidence最高medium，key_risk须提及'MA55需上穿MA233才算趋势修复'。"
        )
    extra_text = "\n".join(extras)

    # ── Python预算摘要（Claude只需引用数字，不需要重新计算）───────────────
    _ks  = hard_signals.get('key_support')
    _kr  = hard_signals.get('key_resistance')
    _lsl = hard_signals.get('long_stop_loss')
    _ssl = hard_signals.get('short_stop_loss')

    # 背离摘要（Python已判断，Claude参考用于composite决策）
    top_div_valid = hard_signals.get('top_divergence_hard_valid', False)
    bot_div_valid = hard_signals.get('bot_divergence_hard_valid', False)
    top_note = hard_signals.get('top_divergence_note_py', '')
    bot_note = hard_signals.get('bot_divergence_note_py', '')
    adj_suff = hard_signals.get('adjustment_sufficient', False)
    # 注意：adjustment_sufficient=True 仅代表"底背离如出现则有效"的前提条件，
    # 不代表反弹信号已成型。在文字中只能写"调整充分，若出现底背离则有效"，
    # 禁止写"反弹信号已成型/初步成型"。

    # 背离成熟度（Python计算，课程：矛盾激化才有分析价值）
    top_mat      = hard_signals.get('top_div_maturity', 'none')
    top_mat_bars = hard_signals.get('top_div_bars_since')
    bot_mat      = hard_signals.get('bot_div_maturity', 'none')
    bot_mat_bars = hard_signals.get('bot_div_bars_since')
    _top_mat_str = (f"，成熟度={top_mat}（距形成{top_mat_bars}根K线）"
                    if top_div_valid and top_mat not in ("none", "unknown") else "")
    _bot_mat_str = (f"，成熟度={bot_mat}（距形成{bot_mat_bars}根K线）"
                    if bot_div_valid and bot_mat not in ("none", "unknown") else "")

    live_warning = hard_signals.get('live_top_div_warning', False)
    live_note    = hard_signals.get('live_top_div_note', '')
    _live_str    = f"\n实时预警：{live_note}" if live_warning else ""

    _adj_note = (
        "是（DIF/DEA已穿越零轴，底背离若出现则条件有效；但调整充分≠反弹信号，不可写'反弹信号成型'）"
        if adj_suff else "否（DIF/DEA未充分穿越零轴，底背离即使出现也无效）"
    )
    div_summary = (
        f"顶背离：{'✅有效' if top_div_valid else '❌无效'}（{top_note}）{_top_mat_str}\n"
        f"底背离：{'✅有效' if bot_div_valid else '❌无效'}（{bot_note}）{_bot_mat_str}\n"
        f"调整充分：{_adj_note}"
        f"{_live_str}"
    )

    # 均线摘要
    ma_type     = hard_signals.get('ma_breakout_type_py', 'unknown')
    ma_dir      = hard_signals.get('ma_breakout_direction_py', 'none')
    align_zh    = hard_signals.get('trend_alignment_zh', '混沌排列')
    align_br    = hard_signals.get('trend_alignment_bracket', '')
    ma_inverted = hard_signals.get('ma_inverted', False)
    ma_summary  = f"均线排列：{align_zh}{align_br}，突破类型：{ma_type}类（{ma_dir}）"
    if ma_inverted:
        ma_summary += (
            "\n⚠️ 均线倒置警告：MA55 < MA233，属于恢复性反弹而非趋势延伸。"
            "A类突破的可信度低于标准多头排列，需MA55上穿MA233才算长期趋势修复。"
        )

    # 结构预判（Python硬计算，Claude验证后输出）
    struct_type  = hard_signals.get('structure_type_py', 'unknown')
    struct_stage = hard_signals.get('structure_current_stage_py', 'unknown')
    struct_conf  = hard_signals.get('structure_confidence_py', 'none')
    struct_note  = hard_signals.get('structure_note_py', '')
    struct_d2a   = hard_signals.get('structure_d_to_a_py', False)
    structure_summary = (
        f"Python预判结构：{struct_type}类（置信度={struct_conf}，当前第{struct_stage}拐点）\n"
        f"说明：{struct_note}"
        + ("\n⚠️ D→A转化信号：第三段已超越第一段×1.5" if struct_d2a else "")
    )

    # 关键K线（Python检测）
    golden = hard_signals.get('latest_golden_candle')
    if golden:
        _pat   = "阳包阴" if golden.get("pattern") == "bullish_engulfing" else "下影线"
        _conf  = "已确认（下一根未破低）" if golden.get("confirmed") else "待确认"
        _vshrk = "缩量✅" if golden.get("vol_shrink") else "非缩量"
        _nsup  = "（临近支撑位）" if golden.get("near_support") else ""
        golden_text = f"关键K线（黄金棒）：{_pat}{_nsup}，{_vshrk}，{_conf}"
    else:
        golden_text = "关键K线：近期未检测到"

    pinned_values = (
        f"支撑={_ks} | 压力={_kr}\n"
        f"做多止损={_lsl} | 做空止损={_ssl}\n"
        f"suggested_action里止损格式：「止损设在 <数字>」"
    )

    # ── 精简硬指标 ────────────────────────────────────────────────────────────
    slim_signals = _slim_hard_signals(hard_signals)
    slim_ts      = _slim_time_space(time_space)

    return f"""
## 分析任务
股票代码：{ticker}
时空状态：{state.get('state_label', '未知')}（{state_code}，{bars_in_state}根K线）
主涨段：{'🔒锁定' if is_locked else '未锁定'}
极端警告：{'🚨是（<3根K线）' if extreme_bars_warning else '否'}
{extra_text}

{emphasis}

---

## Python预算摘要（直接引用，不要重新计算）

{div_summary}

{ma_summary}

{structure_summary}

{golden_text}

{pinned_values}

---

## 硬指标数据

```json
{json.dumps(slim_signals, ensure_ascii=False, indent=2)}
```

时空状态：
```json
{json.dumps(slim_ts, ensure_ascii=False, indent=2)}
```

{FEW_SHOT_EXAMPLES}

---

## 结构分类（Python预判已提供，请验证后输出）

Python已根据拐点数量和段幅度预判了结构类型（见"Python预算摘要"）。
你的任务：① 验证预判是否合理 ② 若有明显不符可调整，须在 structure_note 中说明原因。

规则回顾：
A类（五段式）：6个拐点，高低交替，第三段 > 第一段 × 1.5（第三段为主升/主跌浪）
B类（双平台）：多个拐点形成两个横盘区域，相邻同向拐点差 < 3%
C类（单平台）：一个横盘区域，方向未定（常见于中性偏强/中性偏弱阶段）
D类（三段式）：4个拐点，第三段决定后续演化方向（D4是重要决策点）
→ D→A转化：若 d_to_a_py=True，第五段方向与第一段相同概率高

⚠️ 默认使用 Python预判的 pattern_type，除非拐点序列明显支持其他分类。

---

## 止盈规则（suggested_action中可引用）

做多止盈：① 15分钟顶背离+5分钟破MA55 → 减仓20-30%  ② 60分钟顶背离+15分钟破MA55 → 再减50%
做空止盈：① 15分钟底背离+5分钟站上MA55 → 平空20-30%  ② 60分钟底背离+15分钟站上MA55 → 再平50%
⚠️ 不设固定止盈价格，绝不写"目标价XXX"或"止盈区XXX-YYY"

---

## 输出格式（严格JSON，只有两个section）

{{
  "structure": {{
    "pattern_type": "<A/B/C/D/unknown>",
    "trend_direction": "<up/down/sideways/unknown>",
    "current_stage": "<1/2/3/4/5/6/unknown>",
    "d_to_a_probability": "<high/medium/low/na>",
    "likely_next_move": "<up/down/sideways/unknown>",
    "structure_note": "<40字以内中文说明>"
  }},
  "composite": {{
    "signal": "<strong_buy/buy/hold/sell/strong_sell>",
    "confidence": "<high/medium/low>",
    "entry_side": "<left_side/right_side/wait>",
    "signals_aligned": <true/false>,
    "override_active": <true/false>,
    "override_reason": "<若override=true说明原因>",
    "primary_basis": "<主要依据，30字以内>",
    "suggested_action": "<方向+止损价，50字以内>",
    "key_risk": "<最大风险，40字以内>",
    "position_size": "<light/moderate/heavy>"
  }}
}}
""".strip()


def call_claude_for_soft_signals(
    ticker: str, hard_signals: dict, time_space: dict,
    client: anthropic.Anthropic | None = None,
    model: str = "claude-haiku-4-5-20251001",
) -> dict:
    """一次Claude API调用，返回结构分类+综合信号（v2精简版）。"""
    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    try:
        resp = client.messages.create(
            model=model, max_tokens=800,  # v2: 输出更短，从1500降到800
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(ticker, hard_signals, time_space)}],
        )
        raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        # 只截取第一个{到最后一个}，防止Claude在JSON后追加说明文字
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start: end + 1]
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return _fallback(f"JSON解析失败: {e}")
    except Exception as e:
        return _fallback(str(e))


def _fallback(err: str) -> dict:
    """v2 fallback：只包含 structure + composite。"""
    return {
        "error": err,
        "structure": {"pattern_type": "unknown", "trend_direction": "unknown",
                      "current_stage": "unknown", "d_to_a_probability": "na",
                      "likely_next_move": "unknown",
                      "structure_note": "分析失败"},
        "composite": {"signal": "hold", "confidence": "low", "entry_side": "wait",
                      "signals_aligned": False, "override_active": False, "override_reason": "",
                      "primary_basis": "分析失败", "suggested_action": "等待系统恢复",
                      "key_risk": err, "position_size": "light"},
    }
