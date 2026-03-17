"""tools/trinity/prompt.py
==========================
Layer 3: 构建Claude软判断prompt，含few-shot learning案例。

更新：
- build_prompt 读取 time_space.extreme_bars_warning，
  极端状态 < 3根K线时在 emphasis 里强制加警告，要求 confidence=low
"""
from __future__ import annotations
import json, os
import anthropic

SYSTEM_PROMPT = """
你是三位一体（Trinity Trading System）交易系统的分析引擎。
三个维度：均线（MA）、结构（Structure）、时空（Time-Space）。

职责：
1. 接收已计算好的量化硬指标数据
2. 对需要模式识别的软指标做出判断
3. 严格按照要求的JSON格式输出，不输出任何其他内容

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心优先级规则（铁律，不可违反）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

【规则1】主涨段锁定时
→ composite.signal 必须为 hold
→ 所有背离无效（divergence_valid 全部 false）
→ override_active = true

【规则2】极端状态（极强/极弱）
→ 结构信号无效（structure_overridden = true）
→ 只看均线方向
→ 极强：signal 只能是 hold 或 buy，不能是 sell
→ 极弱：signal 只能是 hold 或 sell，不能是 buy

【规则2补充】极端状态持续不足3根K线时（extreme_bars_warning=true）
→ confidence 必须为 low，无论其他信号多强
→ key_risk 必须包含"极端状态刚触发（N根K线），信号不可靠，不建议追单"
→ suggested_action 必须包含"等待极端状态持续确认（至少3根K线）再操作"
→ position_size 必须为 light
→ 原因：极端状态刚触发的第1-2根K线是历史上最容易反转的阶段，
   AVGO/TTMI等案例证明此时做空往往是最危险的操作

【规则3】时空状态与信号的对应约束（最重要，必须遵守）
→ 中性偏强（mid_strong）：第一假设是震荡C类，signal 优先 hold，
   只有均线+背离+结构三要素同时看多时才给 buy，
   绝不能仅凭顶背离或均线破位就给 sell 或 high confidence sell
→ 强（strong）：趋势多头，signal 偏向 hold/buy，给 sell 需要极强理由（顶背离+破MA55+结构顶部共振）
→ 中性偏弱（mid_weak）：第一假设是震荡C类，signal 优先 hold，
   只有均线+背离+结构三要素同时看空时才给 sell
→ 弱（weak）：趋势空头，signal 偏向 hold/sell
→ 极强/极弱：只看均线，忽略结构和背离

【规则4】背离有效性（⚠️ 必须先看Python预判字段，不要自行重新计算）
→ 硬指标里已有 top_divergence_hard_valid 和 bot_divergence_hard_valid 两个布尔值
  这是Python按"没有新高新低不看背离"规则严格计算的结果，直接作为最终判断
→ top_divergence_hard_valid=false → top_divergence_valid 必须输出 false，不得覆盖
→ bot_divergence_hard_valid=false → bot_divergence_valid 必须输出 false，不得覆盖
→ 背离是拐点的必要不充分条件，单独背离不足以给 high confidence
→ 调整充分性：adjustment_sufficient=false 时，底背离无效，不建议抄底

【规则5】止损方向必须明确区分
→ 看多持仓的止损：设在支撑位下方（key_support 下方 2-3%）
→ 看空做空的止损：设在压力位上方（key_resistance 上方 2-3%）
→ 分析中不可把做空止损和做多持仓的操作混在一起
→ suggested_action 必须明确当前操作方向（做多/做空/观望）

【规则6】confidence 约束
→ high confidence 需要至少两个维度信号共振（均线+背离，或均线+结构，或背离+结构）
→ 单一维度信号（只有背离，或只有均线破位）只能给 medium 或 low
→ 时空状态和信号方向矛盾时（如中性偏强却想给sell），confidence 降为 low
""".strip()


FEW_SHOT_EXAMPLES = """
---
## Few-Shot 案例（学习判断逻辑）

### 案例1：标准底背离 + D类结构买点（中性偏强状态）

输入数据摘要：
- trend_alignment: "bullish"（多头排列）
- dist_from_ma55: -0.02（价格轻微跌破MA55）
- price_vs_ma55_last10: [T,T,T,T,T,F,F,T,T,F]（C类回抽特征）
- bot_divergence_raw: {price_new_low: true, macd_bar_smaller: true, price_change_pct: -0.04, macd_change_pct: -0.35}
- turning_points: 5个拐点，价格序列 [100, 115, 108, 118, 111]
- daily_state: "mid_strong"（中性偏强）
- main_wave locked: false

正确输出：
```json
{
  "divergence": {
    "top_divergence_valid": false,
    "bot_divergence_valid": true,
    "divergence_strength": "medium",
    "divergence_type": "bottom",
    "divergence_note": "价格创新低但MACD面积缩小35%，底背离有效"
  },
  "ma_analysis": {
    "ma55_breakout_type": "C",
    "ma55_breakout_direction": "up",
    "ma55_breakout_valid": true,
    "pullback_opportunity": true,
    "pullback_side": "buy",
    "overextension_warning": false,
    "ma_note": "快速突破MA55后回踩不破，C类强势形态"
  },
  "structure": {
    "pattern_type": "D",
    "trend_direction": "up",
    "current_stage": "3",
    "d_to_a_probability": "high",
    "key_support": 108.0,
    "key_resistance": 120.0,
    "likely_next_move": "up",
    "structure_overridden": false,
    "structure_note": "D类第三段涨幅超第一段1.5倍，大概率转A五段式"
  },
  "composite": {
    "signal": "buy",
    "confidence": "high",
    "entry_side": "right_side",
    "signals_aligned": true,
    "override_active": false,
    "override_reason": "",
    "primary_basis": "底背离+C类突破+D转A结构，三要素共振",
    "suggested_action": "做多方向：等回踩MA55支撑后买入，止损设在 105.8（= long_stop_loss）",
    "key_risk": "背离可能背了又背，需确认5分钟MA55不破才入场",
    "position_size": "moderate"
  }
}
```

---
### 案例2：主涨段锁定中，忽略顶背离（极强状态）

输入数据摘要：
- trend_alignment: "bullish"
- daily_state: "extreme_strong"（极强）
- main_wave: {bollinger_locked: true, monthly_extreme_strong: true}
- top_divergence_raw: {price_new_high: true, macd_bar_lower: true}
- bb_hourly: {below_mid_2bars: false}

正确输出：
```json
{
  "divergence": {
    "top_divergence_valid": false,
    "bot_divergence_valid": false,
    "divergence_strength": "none",
    "divergence_type": "none",
    "divergence_note": "主涨段锁定中，所有背离信号无效"
  },
  "ma_analysis": {
    "ma55_breakout_type": "A",
    "ma55_breakout_direction": "up",
    "ma55_breakout_valid": true,
    "pullback_opportunity": false,
    "pullback_side": "none",
    "overextension_warning": false,
    "ma_note": "典型多头突破，价格远离MA55，均线支撑有效"
  },
  "structure": {
    "pattern_type": "A",
    "trend_direction": "up",
    "current_stage": "3",
    "d_to_a_probability": "na",
    "key_support": null,
    "key_resistance": null,
    "likely_next_move": "up",
    "structure_overridden": true,
    "structure_note": "极强状态，结构判断仅供参考，不作为操作依据"
  },
  "composite": {
    "signal": "hold",
    "confidence": "high",
    "entry_side": "wait",
    "signals_aligned": true,
    "override_active": true,
    "override_reason": "主涨段锁定中（月线极强+60分钟布林带未跌破），忽略顶背离",
    "primary_basis": "J-1布林带锁定",
    "suggested_action": "持有多头仓位，止损设在60分钟布林带中轨下方，等跌破才考虑减仓",
    "key_risk": "末期加速可能突然结束，需盯紧60分钟布林带中轨",
    "position_size": "heavy"
  }
}
```

---
### 案例3：极弱状态，空头排列，调整不充分不抄底

输入数据摘要：
- trend_alignment: "bearish"（空头排列）
- daily_state: "extreme_weak"（极弱）
- dist_from_ma55: -0.06
- bot_divergence_raw: {price_new_low: true, macd_bar_smaller: true}
- adjustment_sufficient: false

正确输出：
```json
{
  "divergence": {
    "top_divergence_valid": false,
    "bot_divergence_valid": false,
    "divergence_strength": "none",
    "divergence_type": "none",
    "divergence_note": "调整不充分（DIF/DEA未穿零轴），底背离无效"
  },
  "ma_analysis": {
    "ma55_breakout_type": "A",
    "ma55_breakout_direction": "down",
    "ma55_breakout_valid": true,
    "pullback_opportunity": true,
    "pullback_side": "sell",
    "overextension_warning": false,
    "ma_note": "典型跌破MA55，空头排列，反弹到MA55是做空/减仓机会"
  },
  "structure": {
    "pattern_type": "A",
    "trend_direction": "down",
    "current_stage": "3",
    "d_to_a_probability": "na",
    "key_support": null,
    "key_resistance": null,
    "likely_next_move": "down",
    "structure_overridden": false,
    "structure_note": "极弱A类下跌主跌段，不要抄底"
  },
  "composite": {
    "signal": "sell",
    "confidence": "high",
    "entry_side": "wait",
    "signals_aligned": true,
    "override_active": false,
    "override_reason": "",
    "primary_basis": "极弱状态+空头排列+调整不充分，三要素看空共振",
    "suggested_action": "持空或观望：若有多头仓位建议减仓；止损设在 short_stop_loss（若key_resistance有值则用预算值，如null则不写具体价格）",
    "key_risk": "极弱状态下底背离可能背了又背，不要抄底",
    "position_size": "light"
  }
}
```

---
### 案例4：中性偏强但有顶背离+均线破位（RKLB类型）⚠️ 最容易出错的案例

输入数据摘要：
- trend_alignment: "mixed"（混沌排列）
- daily_state: "mid_strong"（中性偏强）← 关键！
- bars_below_ma55_last10: 10（全部在MA55下方，A类跌破）
- top_divergence_raw: {price_new_high: true, macd_bar_lower: true}（有顶背离）
- key_resistance: 96.3（真实前高）
- dist_from_ma55: -0.02（轻微跌破）

错误输出（不要这样做）：
→ signal: "sell", confidence: "high" ← 错！中性偏强不能给high confidence sell

正确输出：
```json
{
  "divergence": {
    "top_divergence_valid": true,
    "bot_divergence_valid": false,
    "divergence_strength": "medium",
    "divergence_type": "top",
    "divergence_note": "价格创新高但MACD动能缩小，顶背离有效，注意拐点风险"
  },
  "ma_analysis": {
    "ma55_breakout_type": "B",
    "ma55_breakout_direction": "down",
    "ma55_breakout_valid": true,
    "pullback_opportunity": true,
    "pullback_side": "sell",
    "overextension_warning": false,
    "ma_note": "慢速盘整跌破MA55，10根K线徘徊下方，距离均线较近，短期偏空"
  },
  "structure": {
    "pattern_type": "D",
    "trend_direction": "down",
    "current_stage": "3",
    "d_to_a_probability": "low",
    "key_support": 66.0,
    "key_resistance": 96.3,
    "likely_next_move": "sideways",
    "structure_overridden": false,
    "structure_note": "高位反弹结构，96.3是真实前高压力，当前在反弹第三段"
  },
  "composite": {
    "signal": "hold",
    "confidence": "medium",
    "entry_side": "wait",
    "signals_aligned": false,
    "override_active": false,
    "override_reason": "",
    "primary_basis": "时空中性偏强与顶背离+均线破位信号矛盾，降低置信度",
    "suggested_action": "持多头仓位者观望，等确认跌破66支撑再减仓；不建议现在新开空仓",
    "key_risk": "中性偏强状态下做空风险大，可能形成震荡整理而非单边下跌",
    "position_size": "light"
  }
}
```

---
### 案例5：极弱状态刚触发（<3根K线），强基本面票 ⚠️ 最容易追空的陷阱

输入数据摘要：
- daily_state: "extreme_weak"（极弱）
- bars_in_state: 1（刚触发1根K线）
- extreme_bars_warning: true ← 关键！
- trend_alignment: "mixed"（混沌）
- monthly_state: "strong"（月线强）
- top_divergence_hard_valid: false

错误输出（不要这样做）：
→ signal: "sell", confidence: "high" ← 错！极弱1根K线不可靠，月线还是强

正确输出：
```json
{
  "divergence": {
    "top_divergence_valid": false,
    "bot_divergence_valid": false,
    "divergence_strength": "none",
    "divergence_type": "none",
    "divergence_note": "极弱刚触发，信号不稳定，背离参考价值低"
  },
  "ma_analysis": {
    "ma55_breakout_type": "A",
    "ma55_breakout_direction": "down",
    "ma55_breakout_valid": true,
    "pullback_opportunity": false,
    "pullback_side": "none",
    "overextension_warning": false,
    "ma_note": "已跌破MA55，但极弱仅1根K线，随时可能反转"
  },
  "structure": {
    "pattern_type": "unknown",
    "trend_direction": "unknown",
    "current_stage": "unknown",
    "d_to_a_probability": "na",
    "key_support": null,
    "key_resistance": null,
    "likely_next_move": "unknown",
    "structure_overridden": true,
    "structure_note": "极端状态结构信号无效"
  },
  "composite": {
    "signal": "hold",
    "confidence": "low",
    "entry_side": "wait",
    "signals_aligned": false,
    "override_active": true,
    "override_reason": "极端状态刚触发（1根K线），信号极不可靠",
    "primary_basis": "极弱刚触发，等待3根K线确认",
    "suggested_action": "观望，等待极端状态持续确认（至少3根K线）再操作；月线仍强，不建议追空",
    "key_risk": "极端状态刚触发（1根K线），信号不可靠，做空风险极大",
    "position_size": "light"
  }
}
```
---
"""


def build_prompt(ticker: str, hard_signals: dict, time_space: dict) -> str:
    state      = time_space.get("daily_state", {})
    main_wave  = time_space.get("main_wave", {})
    is_locked  = main_wave.get("bollinger_locked") and main_wave.get("monthly_extreme_strong")
    is_extreme = state.get("is_extreme", False)
    state_code = state.get("current_state", "unknown")

    # ── 读取极端状态警告 ──────────────────────────────────────────────────────
    extreme_bars_warning = time_space.get("extreme_bars_warning", False)
    bars_in_state        = state.get("bars_in_state", 0)

    if is_locked:
        emphasis = (
            "⚠️ 主涨段锁定（月线极强 + 60分钟布林带未跌破）。\n"
            "→ composite.signal必须为hold，所有背离无效，override_active=true。"
        )
    elif is_extreme and extreme_bars_warning:
        # 极端状态刚触发（< 3根K线）：最不可靠的信号，强制降级
        direction = "极强" if state_code == "extreme_strong" else "极弱"
        emphasis = (
            f"🚨🚨 极端时空状态：{state.get('state_label')}，但仅持续 {bars_in_state} 根K线！\n"
            f"→ 极端状态刚触发（< 3根K线）是历史上最容易反转的阶段。\n"
            f"→ confidence 必须为 low，position_size 必须为 light。\n"
            f"→ signal 为 hold（不追单），suggested_action 必须包含'等待3根K线确认'。\n"
            f"→ key_risk 必须注明'极端状态刚触发（{bars_in_state}根K线），信号不可靠，不建议追单'。\n"
            + ("→ 极强状态：signal只能是hold或buy。" if state_code == "extreme_strong"
               else "→ 极弱状态：signal只能是hold或sell。")
        )
    elif is_extreme:
        emphasis = (
            f"⚠️ 极端时空状态：{state.get('state_label')}，已持续 {bars_in_state} 根K线。\n"
            "→ 结构信号无效（structure_overridden=true），只看均线方向。\n"
            + ("→ 极强状态：signal只能是hold或buy。" if state_code == "extreme_strong"
               else "→ 极弱状态：signal只能是hold或sell。")
        )
    elif state_code in ("mid_strong", "mid_weak"):
        emphasis = (
            f"当前时空状态：{state.get('state_label')}，第一假设是C类震荡整理。\n"
            "→ signal优先输出hold，除非均线+背离+结构三要素完全共振才给buy/sell。\n"
            "→ 单一维度信号（只有背离或只有均线破位）不足以给high confidence的方向性信号。\n"
            "→ 时空状态和信号方向矛盾时，confidence必须降为low或medium。"
        )
    elif state_code == "strong":
        emphasis = (
            "当前时空状态：强，趋势偏多。\n"
            "→ signal偏向hold/buy，给sell需要顶背离+破MA55+结构顶部三要素同时出现。"
        )
    elif state_code == "weak":
        emphasis = (
            "当前时空状态：弱，趋势偏空。\n"
            "→ signal偏向hold/sell，给buy需要底背离+站上MA55+结构底部三要素同时出现。"
        )
    else:
        emphasis = "当前处于正常状态，三要素均有效，请综合判断。"

    return f"""
## 分析任务
股票代码：{ticker}
时空状态：{state.get('state_label', '未知')}（{state_code}，已持续 {bars_in_state} 根K线）
主涨段状态：{'🔒 锁定中' if is_locked else '未锁定'}
极端状态警告：{'🚨 是（<3根K线，confidence强制为low）' if extreme_bars_warning else '否'}

{emphasis}

---

## 已计算的硬指标数据

```json
{json.dumps(hard_signals, ensure_ascii=False, indent=2)}
```

时空状态详情：
```json
{json.dumps(time_space, ensure_ascii=False, indent=2)}
```

{FEW_SHOT_EXAMPLES}

---

## 三位一体知识库

### 均线突破四种类型

**A类（典型突破）**：强势穿越均线，价格快速远离MA55超过2%后续不回头，方向性最强。
  识别：bars_above >= 8 且 dist > 0.02，或 bars_below >= 8 且 dist < -0.02。
  ⚠️ 关键约束：dist 符号必须与 bars 方向一致。若 bars_below >= 8 但 dist 变成正值（价格已穿回上方），
  说明旧的A类跌破已经结束，不能再标注"A类跌破"——此时按新方向的bars_above重新判断。

**B类（慢速/盘整突破）**：在MA55上下盘整徘徊，多根K线缓慢穿越；可靠性中等。
  识别：bars_above或below在3-7之间，dist_from_ma55接近0（±2%内），长时间徘徊。
  ⚠️ 补充：bars_above=1或2，dist在0%~2%之间（刚刚突破还未走远），也归B类（新突破待确认）。

**C类（突破后回抽）**：快速突破后回踩不破，再继续原方向。强势形态。
  识别：price_vs_ma55_last10序列先连续同向，后出现1-2根反向，再恢复同向。

**D类（反向测试）**：碰到MA55后直接反弹，从未穿越。
  识别：全部0或全部1，dist_from_ma55接近0但未穿越。

### 均线排列（trend_alignment）
⚠️ 排列描述必须直接使用Python hard_signals中的 trend_alignment_zh 字段（中文已预算好），禁止自行判断：
- trend_alignment_zh = "多头排列" → 报告/ma_note里写"多头排列"（price > MA55 > MA233）
- trend_alignment_zh = "空头排列" → 报告/ma_note里写"空头排列"（price < MA55 < MA233）
- trend_alignment_zh = "混沌排列" → 报告/ma_note里写"混沌排列"，括号里写 trend_alignment_bracket 字段内容
⚠️ 绝对不要写与 trend_alignment_zh 相反的排列描述！
⚠️ 特别注意：价格在MA55下方不等于"空头排列"，只有 price < MA55 < MA233 才是空头排列，
   若 MA233 < price < MA55（价格夹在两条均线之间），必须写"混沌排列（MA233 < 价格 < MA55）"。

### 背离有效性

⚠️ 先看 top_divergence_hard_valid / bot_divergence_hard_valid，Python已严格预判。
顶背离：top_divergence_hard_valid=true（price_new_high 且 macd_bar_lower 同时成立）
底背离：bot_divergence_hard_valid=true（price_new_low 且 macd_bar_smaller 同时成立）
调整充分：adjustment_sufficient=true（DIF和DEA都穿越过零轴）

⚠️ divergence_note 写作规则：
⚠️ 最高优先级：hard_signals 里有预算好的 top_divergence_note_py 和 bot_divergence_note_py 字段。
   直接将这两个字段的文字原样复制到 divergence_note 里，禁止根据原始数字自行构造背离描述！
   turning_points 列表仅用于结构分类（A/B/C/D型），绝对禁止用于背离描述！

强度：
- strong: |price_change_pct| > 0.05 且 |macd_change_pct| > 0.30
- medium: 0.02-0.05 且 0.15-0.30
- weak: 其他

### 结构分类（看turning_points序列）

A类（五段式）：6个拐点，高低交替，第三段 > 第一段 * 1.5
B类（双平台）：两个横盘区域，相邻高点差距 < 3%
C类（单平台）：一个横盘区域，方向未定
D类（三段式）：4个拐点，第三段决定演化方向

### 支撑压力位规则（重要，必须遵守）

硬指标数据里已预计算了 key_support 和 key_resistance，来源是结构拐点（turning_points历史高低点），
这是固定不变的历史价格，优先级最高。

输出 structure.key_support 和 structure.key_resistance 时的规则：
1. 直接使用硬指标里的 key_support / key_resistance 值，不要自己重新计算
2. 绝对不要用布林带下轨（bb_lower）作为支撑——布林带是动态的会随时间移动
3. 绝对不要用布林带上轨（bb_upper）作为压力
4. 只有当硬指标里 key_support=null 时，才可以用MA233（ma233_fallback）作为备选支撑
5. support_source="structural_trough" = 结构低点，权重最高
6. resistance_source="structural_peak" = 结构前高，权重最高
7. resistance_source="ma55_plus5pct_fallback" = 价格已超越所有结构前高，阻力使用MA55+5%做备用参考。
   此时在报告中应注明"当前价已突破所有结构前高，阻力为参考位（MA55上方5%）"，而非写成"结构高点"。
8. 绝对不要新增任何非Python计算的支撑/阻力价格，包括"MA55 + X%"、"前高 + Y点"等自创价位。

### 止损设置规则（必须区分方向，百分比固定不可更改）

硬指标里已预计算了 long_stop_loss（= key_support × 0.97）和 short_stop_loss（= key_resistance × 1.03）。
⚠️ 直接读取这两个字段的预算值，绝对不要自己用 key_support / key_resistance / MA55 重新做乘法！
做多持仓的止损 → 直接使用 long_stop_loss 字段的数字（已是support下方3%）
做空持仓的止损 → 直接使用 short_stop_loss 字段的数字（已是resistance上方3%）
⚠️ 绝对不要用 MA55 × 1.03 作为做空止损！做空止损必须用 short_stop_loss（= key_resistance × 1.03）。
   只有 key_resistance=null 时，short_stop_loss 才为 null，此时 suggested_action 中省略具体止损价。
⚠️ suggested_action 和所有止损描述里，只写具体美元止损价数字，绝对不要写"上方X%"、"下方X%"、"约XXX"，
   百分比说明已内置在预算值中，重复%计算只会产生错误！
⚠️ 在 suggested_action 里写止损时，格式固定为：「止损设在 <止损价数字>」，不附加任何百分比解释。

### 止盈规则（课程核心：条件触发，方向敏感）

⚠️ 三位一体课程**不设固定止盈价格**，止盈是信号条件触发，不是到达某个价位就卖出。
绝对不要在 suggested_action 或 structure_note 中写"目标价 xxx"、"下跌目标 xxx"、"目标：xxx支撑位"、"看向xxx"、"目标看向支撑xxx"、"完全平仓：跌破xxx"、"跌破xxx全部离场"。
key_support / key_resistance 不是止盈目标价，不要用这两个数字描述止盈目标。

⚠️ 止盈方向必须与持仓方向一致，两套规则绝对不能混用：

做多持仓的止盈（减仓）条件：
- 第1次减仓：15分钟顶背离 + 5分钟K线跌破MA55（向下）→ 减仓20-30%
- 第2次减仓：60分钟顶背离 + 15分钟K线跌破MA55（向下）→ 再减仓50%

做空持仓的止盈（平空）条件：
- 第1次平空：15分钟底背离 + 5分钟K线站上MA55（向上）→ 平空20-30%
- 第2次平空：60分钟底背离 + 15分钟K线站上MA55（向上）→ 再平空50%

---

## 输出格式（严格JSON，不要有任何其他文字）

{{
  "divergence": {{
    "top_divergence_valid": <true/false>,
    "bot_divergence_valid": <true/false>,
    "divergence_strength": "<strong/medium/weak/none>",
    "divergence_type": "<top/bottom/both/none>",
    "divergence_note": "<30字以内中文说明，直接复制top_divergence_note_py或bot_divergence_note_py>"
  }},
  "ma_analysis": {{
    "ma55_breakout_type": "<A/B/C/D/none>",
    "ma55_breakout_direction": "<up/down/none>",
    "ma55_breakout_valid": <true/false>,
    "pullback_opportunity": <true/false>,
    "pullback_side": "<buy/sell/none>",
    "overextension_warning": <true/false>,
    "ma_note": "<30字以内中文说明，排列描述直接使用trend_alignment_zh + trend_alignment_bracket>"
  }},
  "structure": {{
    "pattern_type": "<A/B/C/D/unknown>",
    "trend_direction": "<up/down/sideways/unknown>",
    "current_stage": "<1/2/3/4/5/6/unknown>",
    "d_to_a_probability": "<high/medium/low/na>",
    "key_support": <直接使用硬指标里的key_support值，不要自行计算>,
    "key_resistance": <直接使用硬指标里的key_resistance值，不要自行计算>,
    "likely_next_move": "<up/down/sideways/unknown>",
    "structure_overridden": <true/false>,
    "structure_note": "<40字以内中文说明>"
  }},
  "composite": {{
    "signal": "<strong_buy/buy/hold/sell/strong_sell>",
    "confidence": "<high/medium/low>",
    "entry_side": "<left_side/right_side/wait>",
    "signals_aligned": <true/false>,
    "override_active": <true/false>,
    "override_reason": "<若override=true说明原因，否则空字符串>",
    "primary_basis": "<主要判断依据，30字以内>",
    "suggested_action": "<明确做多/做空/观望方向，含具体止损价数字（直接读预算值），50字以内>",
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
    """一次Claude API调用，返回所有软判断JSON。"""
    if client is None:
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    try:
        resp = client.messages.create(
            model=model, max_tokens=1500, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(ticker, hard_signals, time_space)}],
        )
        raw = resp.content[0].text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(raw)
    except json.JSONDecodeError as e:
        return _fallback(f"JSON解析失败: {e}")
    except Exception as e:
        return _fallback(str(e))


def _fallback(err: str) -> dict:
    return {
        "error": err,
        "divergence": {"top_divergence_valid": False, "bot_divergence_valid": False,
                       "divergence_strength": "none", "divergence_type": "none",
                       "divergence_note": "分析失败"},
        "ma_analysis": {"ma55_breakout_type": "unknown", "ma55_breakout_valid": False,
                        "pullback_opportunity": False, "pullback_side": "none",
                        "overextension_warning": False, "ma_note": "分析失败"},
        "structure": {"pattern_type": "unknown", "trend_direction": "unknown",
                      "current_stage": "unknown", "d_to_a_probability": "na",
                      "key_support": None, "key_resistance": None,
                      "likely_next_move": "unknown", "structure_overridden": False,
                      "structure_note": "分析失败"},
        "composite": {"signal": "hold", "confidence": "low", "entry_side": "wait",
                      "signals_aligned": False, "override_active": False, "override_reason": "",
                      "primary_basis": "分析失败", "suggested_action": "等待系统恢复",
                      "key_risk": err, "position_size": "light"},
    }