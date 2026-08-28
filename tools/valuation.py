"""tools/valuation.py
=====================
Four complementary valuation methodologies, adapted from the provided sample.

Methods
-------
1. Discounted Cash Flow (DCF)          — weight 35 %
2. Owner Earnings (Buffett)            — weight 35 %
3. EV/EBITDA implied equity value      — weight 20 %
4. Residual Income Model (RIM/EBO)     — weight 10 %

The weighted average gap vs. market cap drives the final signal.
Gap > +15 % → bullish (undervalued)
Gap < -15 % → bearish (overvalued)
Otherwise   → neutral
"""

from __future__ import annotations

from statistics import median

from tools.data import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
)

# ─────────────────────────────────────────────────────────────────────────────
# Individual models
# ─────────────────────────────────────────────────────────────────────────────

def _dcf(
    fcf: float,
    growth_rate: float = 0.05,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.03,
    years: int = 5,
) -> float:
    if fcf is None or fcf <= 0:
        return 0.0
    pv = sum(
        fcf * (1 + growth_rate) ** yr / (1 + discount_rate) ** yr
        for yr in range(1, years + 1)
    )
    term = (fcf * (1 + growth_rate) ** years * (1 + terminal_growth)) / (discount_rate - terminal_growth)
    pv_term = term / (1 + discount_rate) ** years
    return pv + pv_term


def _owner_earnings(
    net_income: float,
    depreciation: float,
    capex: float,
    wc_change: float,
    growth_rate: float = 0.05,
    required_return: float = 0.15,
    margin_of_safety: float = 0.25,
    years: int = 5,
) -> float:
    if not all(isinstance(x, (int, float)) for x in [net_income, depreciation, capex, wc_change]):
        return 0.0
    oe = net_income + depreciation - capex - wc_change
    if oe <= 0:
        return 0.0

    pv = sum(
        oe * (1 + growth_rate) ** yr / (1 + required_return) ** yr
        for yr in range(1, years + 1)
    )
    tg = min(growth_rate, 0.03)
    term = (oe * (1 + growth_rate) ** years * (1 + tg)) / (required_return - tg)
    pv_term = term / (1 + required_return) ** years
    return (pv + pv_term) * (1 - margin_of_safety)


def _ev_ebitda(metrics: list[dict]) -> float:
    if not metrics:
        return 0.0
    m0 = metrics[0]
    ev       = m0.get("enterprise_value")
    ev_ratio = m0.get("enterprise_value_to_ebitda_ratio")
    mktcap   = m0.get("market_cap")

    if not ev or not ev_ratio or ev_ratio == 0:
        return 0.0

    ebitda   = ev / ev_ratio
    ratios   = [m.get("enterprise_value_to_ebitda_ratio") for m in metrics
                if m.get("enterprise_value_to_ebitda_ratio")]
    if not ratios:
        return 0.0
    med      = median(ratios)
    ev_impl  = med * ebitda
    net_debt = (ev or 0) - (mktcap or 0)
    return max(ev_impl - net_debt, 0.0)


def _residual_income(
    market_cap: float,
    net_income: float,
    pb_ratio: float,
    bv_growth: float = 0.03,
    cost_of_equity: float = 0.10,
    terminal_growth: float = 0.03,
    years: int = 5,
) -> float:
    if not (market_cap and net_income and pb_ratio and pb_ratio > 0):
        return 0.0
    book_val = market_cap / pb_ratio
    ri0      = net_income - cost_of_equity * book_val
    if ri0 <= 0:
        return 0.0

    pv_ri = sum(
        ri0 * (1 + bv_growth) ** yr / (1 + cost_of_equity) ** yr
        for yr in range(1, years + 1)
    )
    term    = ri0 * (1 + bv_growth) ** (years + 1) / (cost_of_equity - terminal_growth)
    pv_term = term / (1 + cost_of_equity) ** years
    return (book_val + pv_ri + pv_term) * 0.80   # 20% margin of safety


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

WEIGHTS = {"dcf": 0.35, "owner_earnings": 0.35, "ev_ebitda": 0.20, "residual_income": 0.10}

# Plausibility bounds; violations produce a data_quality_flag instead of a signal.
_IV_RATIO_MIN  = 0.1    # intrinsic value must be >= 0.1× market cap
_IV_RATIO_MAX  = 10.0   # intrinsic value must be <= 10× market cap
_GAP_PCT_LIMIT = 500.0  # |weighted_gap_%| must be <= 500%


def _sanity_check(result: dict, market_cap: float) -> dict:
    """Mutate result in-place: replace signal with data_quality_flag when bounds are breached."""
    flags: list[str] = []

    for name, data in result.get("methods", {}).items():
        iv = data.get("intrinsic_value")
        if iv and iv > 0 and market_cap > 0:
            ratio = iv / market_cap
            if ratio > _IV_RATIO_MAX:
                flags.append(
                    f"{name}: intrinsic value is {ratio:.0f}\u00d7 market cap "
                    f"(ceiling {_IV_RATIO_MAX:.0f}\u00d7) — likely a unit mismatch"
                )
            elif ratio < _IV_RATIO_MIN:
                flags.append(
                    f"{name}: intrinsic value is {ratio:.3f}\u00d7 market cap "
                    f"(floor {_IV_RATIO_MIN:.1f}\u00d7)"
                )

    gap = abs(result.get("weighted_gap_%") or 0)
    if gap > _GAP_PCT_LIMIT:
        flags.append(
            f"Weighted gap {result['weighted_gap_%']:.1f}% exceeds \u00b1{_GAP_PCT_LIMIT:.0f}% "
            f"plausibility bound — review growth-rate inputs"
        )

    if flags:
        result["data_quality_flag"]    = True
        result["data_quality_reasons"] = flags
        result["data_quality_note"] = (
            "Valuation output exceeded sanity bounds. "
            "Do not act on this signal without manually reviewing input data."
        )
        result["overall_signal"]  = "data_quality_flag"
        result["confidence"]      = 0
        result["interpretation"]  = (
            "\u26a0\ufe0f Data quality flag — valuation requires manual review before use."
        )
    else:
        result["data_quality_flag"] = False

    return result


def _normalized_growth_rate(value: object, default: float = 0.05) -> float:
    """Normalize provider percentages and bound a one-stage forecast assumption."""
    try:
        growth = float(value)
    except (TypeError, ValueError):
        return default
    if growth != growth:
        return default
    if abs(growth) > 1:
        growth /= 100
    # A five-year single-stage DCF is not credible with triple-digit growth.
    return min(max(growth, -0.50), 0.30)


def run_valuation_analysis(
    ticker: str,
    end_date: str | None = None,
) -> dict:
    """
    Run all four valuation models for a ticker.

    Returns
    -------
    {
        "ticker": "AAPL",
        "market_cap": 2_800_000_000_000,
        "overall_signal": "bullish" | "neutral" | "bearish",
        "confidence": 82,
        "weighted_gap_%": 23.4,
        "methods": {
            "dcf":             { "signal": ..., "value": ..., "gap_%": ..., "weight_%": 35 },
            "owner_earnings":  { ... },
            "ev_ebitda":       { ... },
            "residual_income": { ... },
        }
    }
    """
    metrics_list = get_financial_metrics(ticker, period="ttm", limit=8, end_date=end_date)
    if not metrics_list:
        return {"ticker": ticker, "error": "No financial metrics found"}

    m0 = metrics_list[0]

    # Line items (need 2 periods for WC change)
    lines = search_line_items(
        ticker,
        line_items=["free_cash_flow", "net_income", "depreciation_and_amortization",
                    "capital_expenditure", "working_capital"],
        period="ttm",
        limit=2,
        end_date=end_date,
    )

    li_curr = lines[0] if len(lines) > 0 else {}
    li_prev = lines[1] if len(lines) > 1 else {}

    wc_change = 0.0
    if li_curr.get("working_capital") and li_prev.get("working_capital"):
        wc_change = li_curr["working_capital"] - li_prev["working_capital"]

    g = _normalized_growth_rate(m0.get("earnings_growth"))

    dcf_val = _dcf(li_curr.get("free_cash_flow"), growth_rate=g)
    oe_val  = _owner_earnings(
        net_income=li_curr.get("net_income"),
        depreciation=li_curr.get("depreciation_and_amortization"),
        capex=li_curr.get("capital_expenditure"),
        wc_change=wc_change,
        growth_rate=g,
    )
    ev_val  = _ev_ebitda(metrics_list)
    rim_val = _residual_income(
        market_cap=m0.get("market_cap"),
        net_income=li_curr.get("net_income"),
        pb_ratio=m0.get("price_to_book_ratio"),
        bv_growth=float(m0.get("book_value_growth") or 0.03),
    )

    market_cap = get_market_cap(ticker, end_date)
    if not market_cap:
        return {"ticker": ticker, "error": "Market cap unavailable"}

    method_vals = {
        "dcf":             (dcf_val, WEIGHTS["dcf"]),
        "owner_earnings":  (oe_val,  WEIGHTS["owner_earnings"]),
        "ev_ebitda":       (ev_val,  WEIGHTS["ev_ebitda"]),
        "residual_income": (rim_val, WEIGHTS["residual_income"]),
    }

    # Gaps & weighted average
    gaps: dict[str, float] = {}
    for name, (val, _) in method_vals.items():
        if val and val > 0:
            gaps[name] = (val - market_cap) / market_cap

    total_weight = sum(WEIGHTS[k] for k in gaps)
    if total_weight == 0:
        return {"ticker": ticker, "error": "All valuation methods returned zero"}

    weighted_gap = sum(gaps[k] * WEIGHTS[k] for k in gaps) / total_weight

    signal     = "bullish" if weighted_gap > 0.15 else "bearish" if weighted_gap < -0.15 else "neutral"
    model_coverage = total_weight / sum(WEIGHTS.values())
    confidence = round(min(abs(weighted_gap) / 0.30 * 100, 100) * model_coverage)

    methods = {}
    for name, (val, weight) in method_vals.items():
        gap = gaps.get(name)
        methods[name] = {
            "signal": ("bullish" if gap and gap > 0.15 else
                       "bearish" if gap and gap < -0.15 else "neutral"),
            "intrinsic_value": round(val, 2) if val else None,
            "market_cap": round(market_cap, 2),
            "gap_%": round(gap * 100, 1) if gap is not None else None,
            "weight_%": round(weight * 100),
        }

    result = {
        "ticker": ticker,
        "market_cap": round(market_cap, 2),
        "overall_signal": signal,
        "confidence": confidence,
        "weighted_gap_%": round(weighted_gap * 100, 1),
        "assumptions": {
            "forecast_growth_rate": g,
            "growth_source": "earnings_growth, normalized and bounded to [-50%, 30%]",
            "available_model_weight_%": round(model_coverage * 100),
        },
        "interpretation": (
            "Stock appears undervalued" if signal == "bullish" else
            "Stock appears overvalued"  if signal == "bearish" else
            "Stock appears fairly valued"
        ),
        "methods": methods,
    }
    return _sanity_check(result, market_cap)