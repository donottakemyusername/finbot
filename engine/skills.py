"""engine/skills.py
===================
PM Skill Router — detects intent from user query and returns:
  - skill label
  - layout hint for frontend
  - tool priority hints (which tools Claude should bias toward)
  - reflection protocol instructions

Skill Labels
------------
morning_brief     "Morning brief on AAPL, NVDA, AMD"
deep_dive         "Deep dive on MSFT" / "Initiate coverage on TSLA"
macro_regime      "What's the macro regime?" / "Risk-on or risk-off?"
portfolio         "Analyze my portfolio" / "My positions: AAPL 20%, NVDA 15%"
earnings_preview  "NVDA earnings preview" / "Before earnings on MSFT"
trinity           "三位一体" / "时空状态" / "主涨段"
default           Everything else → standard stock analysis

Reflection Protocol
-------------------
After tool calls, the system prompt appends a confidence check:
Claude evaluates whether it has sufficient data to make a high-confidence
recommendation, and optionally calls more tools to fill gaps.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────────────────────
# Skill patterns (order matters — first match wins)
# ─────────────────────────────────────────────────────────────────────────────

_PATTERNS: list[tuple[str, list[str]]] = [
    ("morning_brief", [
        r"morning\s*brief",
        r"开盘前|早报|早盘",
        r"watchlist",
        r"scan\s+(?:my\s+)?(?:stocks|names|watchlist|list)",
        r"overnight\s+(?:changes|moves|scan)",
        r"quick\s+scan",
        r"(?:check|update\s+me\s+on)\s+(?:my\s+)?(?:watchlist|names|stocks)",
    ]),
    ("macro_regime", [
        r"macro\s*regime",
        r"market\s*regime",
        r"risk[\s-]?(?:on|off)",
        r"yield\s+curve",
        r"credit\s+spread",
        r"vix|volatility\s+regime",
        r"macro\s+(?:environment|backdrop|outlook|picture)",
        r"what(?:'s|\s+is)\s+the\s+macro",
        r"fed\s+(?:policy|rate|funds)",
        r"宏观|利率|信用利差|曲线|恐慌指数",
        r"inflation\s+regime",
        r"stagflation|goldilocks|late[\s-]cycle",
        r"market\s+environment",
    ]),
    ("portfolio", [
        r"(?:my\s+)?portfolio",
        r"my\s+(?:book|positions|holdings|names)",
        r"portfolio\s*(?:risk|exposure|analysis|check)",
        r"sector\s+exposure",
        r"concentration\s+risk",
        r"portfolio\s+beta",
        r"仓位|持仓|组合|配置",
    ]),
    ("earnings_preview", [
        r"earnings\s+(?:preview|call|report|season)",
        r"before\s+(?:earnings|the\s+report|results)",
        r"q[1-4]\s+(?:results|earnings|preview)",
        r"(?:upcoming|next)\s+earnings",
        r"财报|业绩|季报|年报\s*(?:预测|预览|前瞻)",
    ]),
    ("deep_dive", [
        r"initiat(?:e|ion)(?:\s+(?:of\s+)?coverage)?",
        r"deep[\s-]?dive",
        r"investment\s+(?:memo|thesis|case)",
        r"research\s+report",
        r"full\s+(?:analysis|research|breakdown)",
        r"tell\s+me\s+everything\s+about",
        r"comprehensive\s+(?:analysis|view)",
        r"深度研究|全面分析|开始覆盖",
    ]),
    ("trinity", [
        r"三位一体|时空状态|主涨段|均线突破|做T|加仓",
        r"trinity\s*(?:analysis|system)?",
        r"时空|结构|背离",
    ]),
]


@dataclass
class SkillResult:
    skill:       str
    layout:      str
    description: str
    tool_hints:  list[str] = field(default_factory=list)
    reflection:  str = ""


_SKILL_META: dict[str, dict] = {
    "morning_brief": {
        "layout":      "morning_brief",
        "description": "Watchlist morning brief — compact overview + trinity state for each name",
        "tool_hints":  ["get_stock_overview", "trinity_analysis"],
        "reflection":  (
            "You are running a MORNING BRIEF for the PM's watchlist. "
            "For each ticker: call get_stock_overview AND trinity_analysis. "
            "Summarize compactly: price change, trinity state, signal, stop-loss. "
            "Output a structured table or grid — not long paragraphs. "
            "End with: which names are most actionable today and why."
        ),
    },
    "macro_regime": {
        "layout":      "macro",
        "description": "Macro regime classification — yield curve, credit spreads, VIX, Fed policy",
        "tool_hints":  ["macro_regime"],
        "reflection":  (
            "You are assessing the MACRO REGIME for portfolio construction. "
            "Call macro_regime first. Then synthesize: "
            "1) Current regime label + risk score "
            "2) What this means for equity conviction (the equity_adj field) "
            "3) Which sectors/factors to overweight vs underweight given this regime "
            "4) One key macro risk to watch. "
            "Be direct and actionable — this is for a PM making allocation decisions."
        ),
    },
    "portfolio": {
        "layout":      "portfolio",
        "description": "Portfolio exposure analysis — sector, beta, concentration, factor tilt",
        "tool_hints":  ["portfolio_exposure", "macro_regime"],
        "reflection":  (
            "You are analyzing the PM's PORTFOLIO EXPOSURE. "
            "Call portfolio_exposure with the positions from the user's message. "
            "Then call macro_regime to get current regime context. "
            "Synthesize: "
            "1) Key concentration risks "
            "2) Portfolio beta and what it means in current regime "
            "3) Factor tilt vs current macro regime (e.g. growth-heavy in late-cycle = risk) "
            "4) Top 2-3 specific actions to reduce risk or improve positioning. "
            "Parse position data from natural language if needed (e.g. 'AAPL 20%' → {ticker:AAPL, weight:0.20})."
        ),
    },
    "earnings_preview": {
        "layout":      "deep_dive",
        "description": "Earnings preview — fundamentals + valuation + EDGAR research",
        "tool_hints":  ["analyze_fundamentals", "analyze_valuation", "deep_research_edgar"],
        "reflection":  (
            "You are running an EARNINGS PREVIEW for the PM. "
            "Call: analyze_fundamentals, analyze_valuation, AND deep_research_edgar (10-Q, mda + risk_factors). "
            "Structure your output as: "
            "1) What consensus expects (growth, margins) "
            "2) What the fundamentals/valuation say about setup "
            "3) Key risks from the 10-Q "
            "4) Bull/bear case with specific catalysts "
            "5) Positioning recommendation (hold into earnings / trim / avoid). "
            "CONFIDENCE CHECK: If fundamental data is incomplete, call the tool again with end_date."
        ),
    },
    "deep_dive": {
        "layout":      "deep_dive",
        "description": "Investment initiation — full analysis + EDGAR + valuation + trinity",
        "tool_hints":  ["get_full_analysis", "deep_research_edgar", "trinity_analysis"],
        "reflection":  (
            "You are writing an INVESTMENT INITIATION / DEEP DIVE for the PM. "
            "Call: get_full_analysis (include_deep_research=true), AND trinity_analysis. "
            "Structure as an investment memo: "
            "## Verdict: [BUY/HOLD/SELL] — [confidence] "
            "## Investment Thesis (2-3 bullets) "
            "## Key Catalysts "
            "## Key Risks "
            "## Valuation: intrinsic value vs market, upside/downside % "
            "## Technical Setup: trinity state, entry, stop-loss "
            "## Macro Context: how current regime affects conviction "
            "Be specific with numbers. No vague statements."
        ),
    },
    "trinity": {
        "layout":      "default",
        "description": "Trinity technical analysis — 三位一体",
        "tool_hints":  ["trinity_analysis"],
        "reflection":  "",
    },
    "default": {
        "layout":      "default",
        "description": "Standard stock analysis",
        "tool_hints":  [],
        "reflection":  "",
    },
}

# Reflection protocol appended for PM skills
_REFLECTION_SUFFIX = """

━━━ REFLECTION PROTOCOL ━━━
Before finalizing your response, check:
1. CONFIDENCE: Do you have enough data to give a HIGH-confidence recommendation?
   - If missing key data (e.g. no valuation, no macro context), call the relevant tool.
   - If tool returned an error, acknowledge it and state what's missing.
2. MACRO CONTEXT: Has macro_regime been called?
   - If not called AND the PM's question involves timing/sizing → call macro_regime.
   - If called, always mention the equity_adj and what it means for conviction.
3. ACTIONABILITY: End with a clear 1-2 sentence action statement:
   "Given [regime] + [signal], the recommended action is [X] with [stop/size]."
State your confidence level (HIGH/MEDIUM/LOW) at the top of your response.
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def detect_skill(user_message: str) -> SkillResult:
    """
    Detect PM skill intent from user message using keyword/regex matching.

    Returns SkillResult with skill label, layout hint, tool hints, and
    reflection protocol to inject into the system prompt.
    """
    msg_lower = user_message.lower()

    for skill_name, patterns in _PATTERNS:
        for pattern in patterns:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                meta = _SKILL_META[skill_name]
                reflection = meta.get("reflection", "")
                if reflection and skill_name not in ("trinity", "default"):
                    reflection += _REFLECTION_SUFFIX
                return SkillResult(
                    skill=skill_name,
                    layout=meta["layout"],
                    description=meta["description"],
                    tool_hints=list(meta["tool_hints"]),
                    reflection=reflection,
                )

    # Default
    meta = _SKILL_META["default"]
    return SkillResult(
        skill="default",
        layout="default",
        description=meta["description"],
        tool_hints=[],
        reflection="",
    )


def skill_system_prompt_addendum(skill_result: SkillResult) -> str:
    """
    Returns the text to append to SYSTEM_PROMPT for the detected skill.
    Empty string for 'default' skill.
    """
    if skill_result.skill in ("default", "trinity"):
        return ""

    lines = [
        "\n\n━━━ PM SKILL ACTIVATED ━━━",
        f"Skill: {skill_result.skill.upper().replace('_', ' ')}",
        f"Description: {skill_result.description}",
    ]
    if skill_result.tool_hints:
        lines.append(f"Priority tools: {', '.join(skill_result.tool_hints)}")
    if skill_result.reflection:
        lines.append("")
        lines.append(skill_result.reflection)
    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)
