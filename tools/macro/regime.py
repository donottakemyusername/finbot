"""tools/macro/regime.py
========================
Macro regime classifier.

Combines yield curve, credit spreads, VIX, and Fed policy into a
single market regime label with PM-relevant conviction adjustments.

Regime Labels
-------------
GOLDILOCKS      Ideal equity environment: curve normal/steep, spreads tight, VIX low
RISK_ON         Healthy expansion: most indicators positive
LATE_CYCLE      Curve flattening, spreads drifting wider — reduce growth/duration risk
RISK_OFF        Vol spike or spread stress — defensive posture
STAGFLATION     High inflation + inverted curve + elevated vol
TIGHTENING      Fed actively hiking, real yields rising — growth multiple compression
RECESSION_RISK  Deeply inverted curve + stressed spreads + high VIX

Equity Conviction Adjustment (how regime modifies stock-level signals)
-----------------------------------------------------------------------
GOLDILOCKS      → +1 boost (upgrade hold→buy, buy→strong_buy)
RISK_ON         → no change
LATE_CYCLE      → −0.5 (penalize high-beta, growth; favor defensive/value)
RISK_OFF        → −1 (downgrade all — cash/defensive preference)
STAGFLATION     → −0.5 (commodities > equities; avoid long-duration growth)
TIGHTENING      → −0.5 (compress PE multiples, avoid unprofitable tech)
RECESSION_RISK  → −1.5 (strong sell bias; preserve capital)
"""
from __future__ import annotations

from tools.macro.indicators import (
    fetch_yield_curve,
    fetch_credit_spreads,
    fetch_vix_data,
    fetch_fed_policy,
)

# ─── Regime definitions ───────────────────────────────────────────────────────

REGIME_META = {
    "GOLDILOCKS": {
        "label":       "Goldilocks",
        "description": "Ideal equity environment — low vol, tight spreads, normal curve.",
        "equity_adj":  +1.0,
        "bias":        "Overweight equities. Favor growth and cyclicals.",
        "avoid":       "Cash drag, excessive defensiveness.",
        "color":       "emerald",
        "icon":        "🌟",
    },
    "RISK_ON": {
        "label":       "Risk-On / Expansion",
        "description": "Healthy expansion — constructive backdrop for equities.",
        "equity_adj":  0.0,
        "bias":        "Normal allocation. Earnings quality matters.",
        "avoid":       "Over-concentration in single factor.",
        "color":       "teal",
        "icon":        "📈",
    },
    "LATE_CYCLE": {
        "label":       "Late Cycle",
        "description": "Flattening curve, rising credit costs — transition phase.",
        "equity_adj":  -0.5,
        "bias":        "Rotate toward value, quality, and defensive sectors.",
        "avoid":       "High-beta growth, long-duration assets, leveraged names.",
        "color":       "amber",
        "icon":        "⏳",
    },
    "TIGHTENING": {
        "label":       "Fed Tightening",
        "description": "Rates elevated, real yields rising — multiple compression.",
        "equity_adj":  -0.5,
        "bias":        "Profitable companies with pricing power. Short duration.",
        "avoid":       "Unprofitable growth, high PE names, REITs.",
        "color":       "orange",
        "icon":        "🔧",
    },
    "RISK_OFF": {
        "label":       "Risk-Off",
        "description": "Vol spike or spread widening — defensive posture warranted.",
        "equity_adj":  -1.0,
        "bias":        "Reduce equities. Favor defensive sectors, T-bills.",
        "avoid":       "Cyclicals, financials, high-beta tech.",
        "color":       "red",
        "icon":        "🛡️",
    },
    "STAGFLATION": {
        "label":       "Stagflation Risk",
        "description": "High inflation + weak growth — commodities beat equities.",
        "equity_adj":  -0.5,
        "bias":        "Commodities, energy, TIPS, real assets.",
        "avoid":       "Consumer discretionary, long-duration tech.",
        "color":       "orange",
        "icon":        "⚠️",
    },
    "RECESSION_RISK": {
        "label":       "Recession Risk",
        "description": "Deeply inverted curve + stressed credit — capital preservation.",
        "equity_adj":  -1.5,
        "bias":        "Cash, T-bills, investment-grade bonds. Min equities.",
        "avoid":       "Cyclicals, small cap, leveraged balance sheets.",
        "color":       "red",
        "icon":        "🔴",
    },
}


def _classify_regime(
    curve: dict,
    spreads: dict,
    vix: dict,
    fed: dict,
) -> str:
    """Rule-based regime classification."""
    vix_v    = vix.get("vix") or 20
    vix_reg  = vix.get("vix_regime", "normal")
    ig_reg   = spreads.get("ig_regime", "normal")
    hy_reg   = spreads.get("hy_regime", "normal")
    cr_reg   = curve.get("curve_regime", "normal")
    spread   = curve.get("spread_10_2_bps")
    stance   = fed.get("policy_stance", "neutral")
    breakeven = fed.get("breakeven_10y") or 2.0
    real_yield = fed.get("real_yield_10y")

    # Recession Risk: worst signals across all dimensions
    if (cr_reg == "deeply_inverted"
            and hy_reg in ("elevated", "stressed")
            and vix_reg in ("stressed", "crisis")):
        return "RECESSION_RISK"

    # Risk-Off: vol or spread spike
    if vix_reg in ("stressed", "crisis") or hy_reg == "stressed":
        return "RISK_OFF"

    # Stagflation: inverted curve + high inflation
    if cr_reg in ("inverted", "deeply_inverted") and breakeven > 3.0:
        return "STAGFLATION"

    # Late cycle: inverted or flat curve
    if cr_reg in ("inverted", "flat") and ig_reg in ("normal", "elevated"):
        return "LATE_CYCLE"

    # Tightening: restrictive policy + rising real yields
    if stance in ("restrictive", "very_restrictive"):
        if real_yield is not None and real_yield > 1.5:
            return "TIGHTENING"

    # Goldilocks: best of all worlds
    if (cr_reg in ("normal", "steep")
            and ig_reg == "tight"
            and hy_reg in ("tight", "normal")
            and vix_reg in ("low", "normal")):
        return "GOLDILOCKS"

    # Default: Risk-On
    return "RISK_ON"


def _compute_regime_score(curve: dict, spreads: dict, vix: dict, fed: dict) -> dict:
    """
    Compute a numeric risk score (0=safest, 100=most stressed) and
    a breakdown of each pillar's contribution.
    """
    score = 0
    breakdown = {}

    # Yield curve (0-25 pts)
    cr = curve.get("curve_regime", "normal")
    c_score = {"steep": 0, "normal": 5, "flat": 12, "inverted": 20, "deeply_inverted": 25, "unknown": 10}.get(cr, 10)
    score += c_score
    breakdown["yield_curve"] = {"regime": cr, "score": c_score, "max": 25}

    # Credit spreads (0-30 pts)
    ig = spreads.get("ig_regime", "normal")
    hy = spreads.get("hy_regime", "normal")
    ig_s = {"tight": 0, "normal": 5, "elevated": 12, "stressed": 20, "unknown": 7}.get(ig, 7)
    hy_s = {"tight": 0, "normal": 5, "elevated": 15, "stressed": 25, "unknown": 10}.get(hy, 10)
    cs = min(ig_s + hy_s, 30)
    score += cs
    breakdown["credit_spreads"] = {"ig_regime": ig, "hy_regime": hy, "score": cs, "max": 30}

    # VIX (0-25 pts)
    vr = vix.get("vix_regime", "normal")
    v_score = {"low": 0, "normal": 5, "elevated": 12, "stressed": 20, "crisis": 25, "unknown": 8}.get(vr, 8)
    score += v_score
    breakdown["volatility"] = {"vix_regime": vr, "score": v_score, "max": 25}

    # Fed policy (0-20 pts)
    st = fed.get("policy_stance", "neutral")
    f_score = {"accommodative": 0, "neutral": 5, "restrictive": 12, "very_restrictive": 20, "unknown": 7}.get(st, 7)
    score += f_score
    breakdown["fed_policy"] = {"stance": st, "score": f_score, "max": 20}

    return {"total": min(score, 100), "breakdown": breakdown}


def run_macro_regime() -> dict:
    """
    Master function: fetch all macro data and classify the current regime.

    Returns
    -------
    {
        "regime":         str,   # GOLDILOCKS | RISK_ON | LATE_CYCLE | ...
        "regime_label":   str,   # human-readable
        "description":    str,
        "equity_adj":     float, # conviction modifier for stock signals
        "bias":           str,   # PM-facing guidance
        "avoid":          str,   # what to avoid
        "risk_score":     int,   # 0-100 composite stress score
        "risk_breakdown": dict,
        "pillars": {
            "yield_curve":     dict,
            "credit_spreads":  dict,
            "volatility":      dict,
            "fed_policy":      dict,
        },
        "chart_data": {
            "yield_curve_history": list,
            "credit_spread_history": list,
            "vix_history": list,
        },
        "icon":  str,
        "color": str,
        "as_of": str,
    }
    """
    from datetime import date

    # Fetch all pillars
    curve   = fetch_yield_curve()
    spreads = fetch_credit_spreads()
    vix     = fetch_vix_data()
    fed     = fetch_fed_policy()

    regime      = _classify_regime(curve, spreads, vix, fed)
    meta        = REGIME_META[regime]
    risk        = _compute_regime_score(curve, spreads, vix, fed)

    return {
        "regime":         regime,
        "regime_label":   meta["label"],
        "description":    meta["description"],
        "equity_adj":     meta["equity_adj"],
        "bias":           meta["bias"],
        "avoid":          meta["avoid"],
        "risk_score":     risk["total"],
        "risk_breakdown": risk["breakdown"],
        "icon":           meta["icon"],
        "color":          meta["color"],
        "pillars": {
            "yield_curve":    curve,
            "credit_spreads": spreads,
            "volatility":     vix,
            "fed_policy":     fed,
        },
        "chart_data": {
            "yield_curve_history":   curve.get("history_10_2", []),
            "credit_spread_history": spreads.get("history", []),
            "vix_history":           vix.get("vix_history", []),
        },
        "as_of": date.today().isoformat(),
    }
