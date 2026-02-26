"""engine/aggregator.py
=======================
Combines outputs from technical, fundamental, and valuation agents
into a final buy / hold / sell verdict with reasoning.

Two modes
---------
1. Rule-based (fast)  — weighted voting across all agents
2. AI-enhanced        — sends all signals to Claude for final reasoning

Both modes are available; the MCP tool calls AI-enhanced by default.
"""

from __future__ import annotations

import json
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Signal weights for rule-based aggregation
WEIGHTS = {
    "technical":   0.35,
    "fundamental": 0.35,
    "valuation":   0.30,
}

# Map string signals to numeric scores
SIGNAL_SCORE = {
    "buy": 1, "bullish": 1,
    "hold": 0, "neutral": 0,
    "sell": -1, "bearish": -1,
}


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_signals_rule_based(
    technical_result: dict,
    fundamental_result: dict,
    valuation_result: dict,
) -> dict:
    """
    Weighted average of the three agent signals.

    Returns
    -------
    {
        "signal": "buy" | "hold" | "sell",
        "confidence": 72,
        "weighted_score": 0.44,
        "breakdown": { ... }
    }
    """
    tech_sig = technical_result.get("overall_technical_signal", "hold")
    fund_sig = fundamental_result.get("overall_signal",          "neutral")
    valu_sig = valuation_result.get("overall_signal",            "neutral")

    scores = {
        "technical":   SIGNAL_SCORE.get(tech_sig, 0),
        "fundamental": SIGNAL_SCORE.get(fund_sig, 0),
        "valuation":   SIGNAL_SCORE.get(valu_sig, 0),
    }

    weighted = sum(scores[k] * WEIGHTS[k] for k in scores)

    if weighted > 0.15:
        signal = "buy"
    elif weighted < -0.15:
        signal = "sell"
    else:
        signal = "hold"

    confidence = round(min(abs(weighted) / 0.50 * 100, 100))

    # Individual indicator breakdown for dashboard
    indicator_signals = {}
    for key, val in technical_result.get("indicators", {}).items():
        indicator_signals[val["name"]] = {
            "signal": val["signal"],
            "backtest_win_rate_%": val.get("backtest", {}).get("win_rate_%"),
            "backtest_trades":     val.get("backtest", {}).get("n_trades"),
        }

    for section, data in fundamental_result.get("sections", {}).items():
        indicator_signals[f"Fundamental: {section.title()}"] = {
            "signal": data.get("signal", "neutral"),
        }

    for method, data in valuation_result.get("methods", {}).items():
        indicator_signals[f"Valuation: {method.replace('_', ' ').title()}"] = {
            "signal": data.get("signal", "neutral"),
            "gap_%": data.get("gap_%"),
        }

    return {
        "signal":         signal,
        "confidence":     confidence,
        "weighted_score": round(weighted, 3),
        "agent_signals": {
            "technical":   {"signal": tech_sig, "votes": technical_result.get("vote_summary")},
            "fundamental": {"signal": fund_sig, "votes": fundamental_result.get("vote_summary")},
            "valuation":   {"signal": valu_sig, "gap_%": valuation_result.get("weighted_gap_%")},
        },
        "indicator_breakdown": indicator_signals,
    }


# ─────────────────────────────────────────────────────────────────────────────
# AI-enhanced final verdict
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an expert quantitative equity analyst. You will be given structured
analysis from three agents (technical, fundamental, valuation) for a stock.

Your job:
1. Synthesize all signals into a final verdict: BUY, HOLD, or SELL.
2. Provide a confidence score 0–100.
3. Write a 3–5 sentence reasoning paragraph in clear English covering:
   - What the technical signals say about price momentum
   - What fundamentals reveal about business quality
   - What valuation says about whether the price is fair
   - Any key conflicts or caveats the investor should know
4. List 3 key supporting arguments and 2 key risks.

Respond ONLY with valid JSON matching this schema exactly:
{
  "verdict": "BUY" | "HOLD" | "SELL",
  "confidence": <int 0-100>,
  "reasoning": "<paragraph>",
  "supporting_arguments": ["...", "...", "..."],
  "key_risks": ["...", "..."]
}
"""


def get_ai_verdict(
    ticker: str,
    technical_result: dict,
    fundamental_result: dict,
    valuation_result: dict,
    rule_based: dict,
    deep_research_result: dict | None = None,
) -> dict:
    """
    Call Claude to produce a final narrative verdict.
    Falls back to rule-based result if the API key is missing.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return {**rule_based, "reasoning": "AI verdict unavailable (no API key). Rule-based result shown."}

    client = anthropic.Anthropic(api_key=api_key)

    payload = {
        "ticker": ticker,
        "rule_based_verdict": rule_based,
        "technical_summary": {
            "overall": technical_result.get("overall_technical_signal"),
            "votes":   technical_result.get("vote_summary"),
            "indicators": {
                k: {"signal": v["signal"], "reason": v.get("reason"), "backtest_win_%": v.get("backtest", {}).get("win_rate_%")}
                for k, v in technical_result.get("indicators", {}).items()
            },
        },
        "fundamental_summary": {
            "overall":    fundamental_result.get("overall_signal"),
            "confidence": fundamental_result.get("confidence"),
            "sections": {
                s: {"signal": d["signal"], "details": d.get("details")}
                for s, d in fundamental_result.get("sections", {}).items()
            },
        },
        "valuation_summary": {
            "overall":      valuation_result.get("overall_signal"),
            "weighted_gap": valuation_result.get("weighted_gap_%"),
            "methods": {
                m: {"signal": d["signal"], "gap_%": d.get("gap_%")}
                for m, d in valuation_result.get("methods", {}).items()
            },
        },
    }

    if deep_research_result:
        # Include only brief MDA excerpt to avoid context overflow
        filings = deep_research_result.get("filings", [])
        if filings:
            mda = filings[0].get("sections", {}).get("mda", "")
            payload["edgar_mda_excerpt"] = mda[:2000]

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": json.dumps(payload, indent=2)}
            ],
        )
        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        ai_output = json.loads(raw)
    except Exception as exc:
        ai_output = {
            "verdict": rule_based["signal"].upper(),
            "confidence": rule_based["confidence"],
            "reasoning": f"AI verdict generation failed ({exc}). Rule-based result used.",
            "supporting_arguments": [],
            "key_risks": [],
        }

    return {
        **rule_based,
        "ai_verdict":          ai_output.get("verdict", rule_based["signal"].upper()),
        "ai_confidence":       ai_output.get("confidence", rule_based["confidence"]),
        "reasoning":           ai_output.get("reasoning", ""),
        "supporting_arguments": ai_output.get("supporting_arguments", []),
        "key_risks":           ai_output.get("key_risks", []),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full analysis pipeline (called by MCP tool)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_analysis(
    ticker: str,
    technical_result: dict,
    fundamental_result: dict,
    valuation_result: dict,
    deep_research_result: dict | None = None,
    use_ai: bool = True,
) -> dict:
    """Aggregate all agents and return a final verdict dict."""
    rule_based = aggregate_signals_rule_based(
        technical_result, fundamental_result, valuation_result
    )

    if use_ai:
        verdict = get_ai_verdict(
            ticker, technical_result, fundamental_result,
            valuation_result, rule_based, deep_research_result,
        )
    else:
        verdict = {**rule_based, "reasoning": "Rule-based aggregation (AI disabled)."}

    return {
        "ticker":    ticker,
        "price":     technical_result.get("price"),
        "as_of":     technical_result.get("as_of"),
        **verdict,
    }