"""tools/trinity/prompt.py
==========================
Layer 3: 构建Claude软判断prompt，含few-shot learning案例。
一次API调用解决所有模式识别问题。
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

核心优先级规则（必须遵守）：
1. 主涨段锁定时 → composite.signal=hold，忽略所有背离和结构
2. 时空极端状态时 → 结构信号降权，以均线为主
3. 正常状态时 → 三要素综合判断，共振才给high confidence
4. 背离必须在创新高/新低时才有效
""".strip()


FEW_SHOT_EXAMPLES = """
---
## Few-Shot 案例（学习判断逻辑）

### 案例1：标准底背离 + D类结构买点

输入数据摘要：
- trend_alignment: "bullish"（多头排列）
- dist_from_ma55: -0.02（价格轻微跌破MA55）
- price_vs_ma55_last10: [T,T,T,T,T,F,F,T,T,F]（先上后下再上，C类特征）
- bot_divergence_raw: {price_new_low: true, macd_bar_smaller: true, price_change_pct: -0.04, macd_change_pct: -0.35}
- turning_points: 5个拐点，价格序列 [100, 115, 108, 118, 111]（第三段上涨 > 第一段）
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
    "structure_note": "D类三段式第三段，第三段涨幅超第一段1.5倍，大概率转A"
  },
  "composite": {
    "signal": "buy",
    "confidence": "high",
    "entry_side": "right_side",
    "signals_aligned": true,
    "override_active": false,
    "override_reason": "",
    "primary_basis": "底背离有效+C类突破+D转A结构",
    "suggested_action": "等回踩MA55支撑后买入，止损设在前低111下方",
    "key_risk": "背离可能背了又背，需确认5分钟破位才入场",
    "position_size": "moderate"
  }
}
```

---
### 案例2：主涨段锁定中，忽略顶背离

输入数据摘要：
- trend_alignment: "bullish"
- daily_state: "extreme_strong"（极强）
- main_wave: {bollinger_locked: true, monthly_extreme_strong: true}
- top_divergence_raw: {price_new_high: true, macd_bar_lower: true}（看起来有顶背离）
- bb_hourly: {below_mid_2bars: false}（60分钟布林带未跌破）

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
    "ma55_breakout_type": "B",
    "ma55_breakout_direction": "up",
    "ma55_breakout_valid": true,
    "pullback_opportunity": false,
    "pullback_side": "none",
    "overextension_warning": false,
    "ma_note": "多头排列，均线支撑有效"
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
    "suggested_action": "持有不动，等60分钟连续2根K线跌破布林带中轨才考虑减仓",
    "key_risk": "末期加速可能突然结束，需盯紧60分钟布林带中轨",
    "position_size": "heavy"
  }
}
```

---
### 案例3：极弱状态，空头排列，不要抄底

输入数据摘要：
- trend_alignment: "bearish"（空头排列）
- daily_state: "extreme_weak"（极弱）
- dist_from_ma55: -0.06（价格在MA55下方6%）
- bot_divergence_raw: {price_new_low: true, macd_bar_smaller: true}
- adjustment_sufficient: false（DIF/DEA未双双穿越零轴，调整不充分）

正确输出：
```json
{
  "divergence": {
    "top_divergence_valid": false,
    "bot_divergence_valid": false,
    "divergence_strength": "none",
    "divergence_type": "none",
    "divergence_note": "调整不充分（DIF/DEA未穿零轴），底背离无效，不考虑抄底"
  },
  "ma_analysis": {
    "ma55_breakout_type": "A",
    "ma55_breakout_direction": "down",
    "ma55_breakout_valid": true,
    "pullback_opportunity": true,
    "pullback_side": "sell",
    "overextension_warning": false,
    "ma_note": "慢速跌破MA55，空头排列确认，反弹到MA55是减仓机会"
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
    "structure_note": "极弱A类下跌第三段（主跌段），跌幅最大段"
  },
  "composite": {
    "signal": "sell",
    "confidence": "high",
    "entry_side": "wait",
    "signals_aligned": true,
    "override_active": false,
    "override_reason": "",
    "primary_basis": "极弱状态+空头排列+调整不充分",
    "suggested_action": "不抄底，等DIF和DEA都穿越零轴后再考虑介入",
    "key_risk": "极弱状态下底背离可能背了又背",
    "position_size": "light"
  }
}
```
---
"""


def build_prompt(ticker: str, hard_signals: dict, time_space: dict) -> str:
    state     = time_space.get("daily_state", {})
    main_wave = time_space.get("main_wave", {})
    is_locked = main_wave.get("bollinger_locked") and main_wave.get("monthly_extreme_strong")
    is_extreme = state.get("is_extreme", False)

    if is_locked:
        emphasis = (
            "⚠️ 当前处于主涨段锁定状态（月线极强 + 60分钟布林带未跌破）。\n"
            "规则：composite.signal必须为hold，所有背离无效（divergence_valid全部false），"
            "override_active=true。"
        )
    elif is_extreme:
        emphasis = (
            f"⚠️ 当前处于极端时空状态：{state.get('state_label')}。\n"
            "规则：结构信号仅供参考（structure_overridden=true），以均线为主要判断依据。"
        )
    else:
        emphasis = "当前处于正常状态，三要素均有效，请综合判断。"

    return f"""
## 分析任务
股票代码：{ticker}
时空状态：{state.get('state_label', '未知')}（{state.get('current_state', 'unknown')}）
主涨段状态：{'🔒 锁定中' if is_locked else '未锁定'}

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

### 均线突破四种类型（关键：看price_vs_ma55_last10序列）

**A类（慢速突破/跌破）**：连续3+根K线收盘在MA55的±1%范围，缓慢穿越。可靠性最低。
  识别：bars_above_ma55_last10在3-7之间，dist_from_ma55接近0，价格长时间在均线附近徘徊。

**B类（有效突破）**：单日大幅穿越，收盘偏离MA55超过2%，后续不回头。可靠性最高。
  识别：bars_above_ma55_last10 >= 8 且 dist_from_ma55 > 0.02（或 < -0.02）。

**C类（突破后回抽）**：快速突破后回踩MA55但不跌破，再继续原方向。常见强势形态。
  识别：price_vs_ma55_last10序列先有连续True，后出现1-2个False（回踩），再转True。

**D类（反向测试）**：价格碰到MA55后直接反弹，从未穿越。
  识别：bars_above_ma55_last10全是0或1，dist_from_ma55接近0但方向相反。

### 背离有效性判断

顶背离必须同时满足：price_new_high=true 且 macd_bar_lower=true
底背离必须同时满足：price_new_low=true 且 macd_bar_smaller=true
调整充分性前提：adjustment_sufficient=true（DIF和DEA都穿越过零轴）

强度：
- strong: |price_change_pct| > 0.05 且 |macd_change_pct| > 0.30
- medium: |price_change_pct| 0.02-0.05 且 |macd_change_pct| 0.15-0.30
- weak: 其他情况

### 结构分类（看turning_points价格序列）

A类（五段式）：6个拐点，高低点交替，第三段幅度 > 第一段 * 1.5
B类（双平台）：相邻高点差距<3%，形成两个横盘平台区域
C类（单平台）：只有一个横盘平台，震荡中
D类（三段式）：4个拐点，第三段长度决定后续（>1.5倍→转A，≈1倍→转C，<1倍→转B）

当前阶段：数turning_points里高低点的个数，除以2得段数，当前在第几段。

---

## 输出格式（严格JSON，不要有任何其他文字）

{{
  "divergence": {{
    "top_divergence_valid": <true/false>,
    "bot_divergence_valid": <true/false>,
    "divergence_strength": "<strong/medium/weak/none>",
    "divergence_type": "<top/bottom/both/none>",
    "divergence_note": "<30字以内中文说明>"
  }},
  "ma_analysis": {{
    "ma55_breakout_type": "<A/B/C/D/none>",
    "ma55_breakout_direction": "<up/down/none>",
    "ma55_breakout_valid": <true/false>,
    "pullback_opportunity": <true/false>,
    "pullback_side": "<buy/sell/none>",
    "overextension_warning": <true/false>,
    "ma_note": "<30字以内中文说明>"
  }},
  "structure": {{
    "pattern_type": "<A/B/C/D/unknown>",
    "trend_direction": "<up/down/sideways/unknown>",
    "current_stage": "<1/2/3/4/5/6/unknown>",
    "d_to_a_probability": "<high/medium/low/na>",
    "key_support": <float或null>,
    "key_resistance": <float或null>,
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
    "suggested_action": "<具体建议，50字以内>",
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
    """一次Claude API调用，返回所有软判断JSON。约$0.004/次。"""
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
                       "divergence_strength": "none", "divergence_type": "none", "divergence_note": "分析失败"},
        "ma_analysis": {"ma55_breakout_type": "unknown", "ma55_breakout_valid": False,
                        "pullback_opportunity": False, "pullback_side": "none",
                        "overextension_warning": False, "ma_note": "分析失败"},
        "structure": {"pattern_type": "unknown", "trend_direction": "unknown",
                      "current_stage": "unknown", "d_to_a_probability": "na",
                      "key_support": None, "key_resistance": None,
                      "likely_next_move": "unknown", "structure_overridden": False, "structure_note": "分析失败"},
        "composite": {"signal": "hold", "confidence": "low", "entry_side": "wait",
                      "signals_aligned": False, "override_active": False, "override_reason": "",
                      "primary_basis": "分析失败", "suggested_action": "等待系统恢复",
                      "key_risk": err, "position_size": "light"},
    }