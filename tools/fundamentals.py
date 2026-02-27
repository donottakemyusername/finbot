"""tools/fundamentals.py
========================
Fundamental analysis: profitability, growth, financial health,
valuation ratios.  Adapted and extended from the provided sample.

Each section returns a sub-signal ("bullish" / "neutral" / "bearish")
plus a human-readable detail string.  All sub-signals are aggregated
into an overall signal with a confidence score.
"""

from __future__ import annotations

from tools.data import get_financial_metrics

# ─────────────────────────────────────────────────────────────────────────────
# Thresholds (easily tunable)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLDS = {
    # Profitability
    "roe_min":              0.15,   # Return on Equity > 15 %
    "net_margin_min":       0.20,   # Net margin > 20 %
    "op_margin_min":        0.15,   # Operating margin > 15 %
    # Growth
    "revenue_growth_min":   0.10,   # Revenue growth > 10 %
    "earnings_growth_min":  0.10,
    "bv_growth_min":        0.10,
    # Financial health
    "current_ratio_min":    1.5,
    "debt_equity_max":      0.5,
    "fcf_eps_ratio_min":    0.8,    # FCF per share / EPS > 80 %
    # Valuation (above these = expensive → bearish)
    "pe_max":               25,
    "pb_max":               3,
    "ps_max":               5,
    # Efficiency
    "roa_min":              0.05,
    "asset_turnover_min":   0.5,
    # Dividends (optional — only scored if data present)
    "dividend_yield_min":   0.02,
    "payout_ratio_max":     0.60,
}


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _score(items: list[tuple]) -> int:
    """Count how many (value, threshold, direction) tuples pass.
    direction: 'above' → value > threshold,  'below' → value < threshold
    """
    score = 0
    for value, threshold, direction in items:
        if value is None:
            continue
        if direction == "above" and value > threshold:
            score += 1
        elif direction == "below" and value < threshold:
            score += 1
    return score


def _sub_signal(score: int, max_score: int) -> str:
    ratio = score / max_score if max_score > 0 else 0
    if ratio >= 0.67:
        return "bullish"
    elif ratio <= 0.33:
        return "bearish"
    return "neutral"


def _fmt(label: str, value, fmt: str = ".2%") -> str:
    if value is None:
        return f"{label}: N/A"
    return f"{label}: {value:{fmt}}"


# ─────────────────────────────────────────────────────────────────────────────
# Analysis sections
# ─────────────────────────────────────────────────────────────────────────────

def _profitability(m: dict) -> tuple[str, dict]:
    roe  = m.get("return_on_equity")
    nm   = m.get("net_margin")
    om   = m.get("operating_margin")
    roa  = m.get("return_on_assets")
    at   = m.get("asset_turnover")

    items = [
        (roe,  THRESHOLDS["roe_min"],          "above"),
        (nm,   THRESHOLDS["net_margin_min"],   "above"),
        (om,   THRESHOLDS["op_margin_min"],    "above"),
        (roa,  THRESHOLDS["roa_min"],          "above"),
        (at,   THRESHOLDS["asset_turnover_min"],"above"),
    ]
    valid = [(v, t, d) for v, t, d in items if v is not None]
    sc = _score(valid)
    sig = _sub_signal(sc, len(valid))

    return sig, {
        "signal": sig,
        "score": f"{sc}/{len(valid)}",
        "details": " | ".join([
            _fmt("ROE",  roe),  _fmt("Net Margin", nm),
            _fmt("Op Margin", om), _fmt("ROA", roa),
            _fmt("Asset Turnover", at, ".2f"),
        ]),
    }


def _growth(m: dict) -> tuple[str, dict]:
    rg  = m.get("revenue_growth")
    eg  = m.get("earnings_growth")
    bvg = m.get("book_value_growth")

    items = [
        (rg,  THRESHOLDS["revenue_growth_min"],  "above"),
        (eg,  THRESHOLDS["earnings_growth_min"], "above"),
        (bvg, THRESHOLDS["bv_growth_min"],       "above"),
    ]
    valid = [(v, t, d) for v, t, d in items if v is not None]
    sc = _score(valid)
    sig = _sub_signal(sc, max(len(valid), 1))

    return sig, {
        "signal": sig,
        "score": f"{sc}/{len(valid)}",
        "details": " | ".join([
            _fmt("Revenue Growth", rg), _fmt("Earnings Growth", eg),
            _fmt("Book Value Growth", bvg),
        ]),
    }


def _health(m: dict) -> tuple[str, dict]:
    cr  = m.get("current_ratio")
    de  = m.get("debt_to_equity")
    fcf = m.get("free_cash_flow_per_share")
    eps = m.get("earnings_per_share")

    fcf_eps_ok = None
    if fcf is not None and eps and eps > 0:
        fcf_eps_ok = fcf / eps

    items = [
        (cr,       THRESHOLDS["current_ratio_min"], "above"),
        (de,       THRESHOLDS["debt_equity_max"],   "below"),
        (fcf_eps_ok, THRESHOLDS["fcf_eps_ratio_min"], "above"),
    ]
    valid = [(v, t, d) for v, t, d in items if v is not None]
    sc = _score(valid)
    sig = _sub_signal(sc, max(len(valid), 1))

    return sig, {
        "signal": sig,
        "score": f"{sc}/{len(valid)}",
        "details": " | ".join([
            _fmt("Current Ratio", cr, ".2f"),
            _fmt("D/E", de, ".2f"),
            _fmt("FCF/EPS", fcf_eps_ok, ".2f") if fcf_eps_ok else "FCF/EPS: N/A",
        ]),
    }


def _valuation_ratios(m: dict) -> tuple[str, dict]:
    """High ratios → bearish signal (overvalued)."""
    pe = m.get("price_to_earnings_ratio")
    pb = m.get("price_to_book_ratio")
    ps = m.get("price_to_sales_ratio")

    items = [
        (pe, THRESHOLDS["pe_max"], "below"),
        (pb, THRESHOLDS["pb_max"], "below"),
        (ps, THRESHOLDS["ps_max"], "below"),
    ]
    valid = [(v, t, d) for v, t, d in items if v is not None]
    sc = _score(valid)
    # For valuation, LOW ratios → bullish
    sig = _sub_signal(sc, max(len(valid), 1))

    return sig, {
        "signal": sig,
        "score": f"{sc}/{len(valid)}",
        "details": " | ".join([
            _fmt("P/E", pe, ".2f"), _fmt("P/B", pb, ".2f"), _fmt("P/S", ps, ".2f"),
        ]),
    }


def _dividends(m: dict) -> tuple[str, dict] | None:
    dy  = m.get("dividend_yield")
    pr  = m.get("payout_ratio")
    if dy is None and pr is None:
        return None

    items = [
        (dy, THRESHOLDS["dividend_yield_min"], "above"),
        (pr, THRESHOLDS["payout_ratio_max"],   "below"),
    ]
    valid = [(v, t, d) for v, t, d in items if v is not None]
    sc = _score(valid)
    sig = _sub_signal(sc, max(len(valid), 1))

    return sig, {
        "signal": sig,
        "score": f"{sc}/{len(valid)}",
        "details": " | ".join([_fmt("Dividend Yield", dy/100), _fmt("Payout Ratio", pr)]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_fundamental_analysis(
    ticker: str,
    end_date: str | None = None,
    limit: int = 10,
) -> dict:
    """
    Run all fundamental analysis sections for a ticker.

    Returns
    -------
    {
        "ticker": "AAPL",
        "overall_signal": "bullish" | "neutral" | "bearish",
        "confidence": 75,
        "sections": {
            "profitability": { "signal": ..., "details": ... },
            "growth":        { ... },
            "health":        { ... },
            "valuation":     { ... },
            "dividends":     { ... },   # optional
        }
    }
    """
    metrics_list = get_financial_metrics(ticker, period="ttm", limit=limit, end_date=end_date)
    if not metrics_list:
        return {"ticker": ticker, "error": "No financial metrics found"}

    m = metrics_list[0]  # most recent TTM snapshot

    sections: dict = {}

    prof_sig, sections["profitability"] = _profitability(m)
    grow_sig, sections["growth"]        = _growth(m)
    hlth_sig, sections["health"]        = _health(m)
    valu_sig, sections["valuation"]     = _valuation_ratios(m)

    sigs = [prof_sig, grow_sig, hlth_sig, valu_sig]

    div = _dividends(m)
    if div:
        div_sig, sections["dividends"] = div
        sigs.append(div_sig)

    bullish = sigs.count("bullish")
    bearish = sigs.count("bearish")
    total   = len(sigs)

    if bullish > bearish:
        overall = "bullish"
    elif bearish > bullish:
        overall = "bearish"
    else:
        overall = "neutral"

    confidence = round(max(bullish, bearish) / total * 100)

    return {
        "ticker": ticker,
        "overall_signal": overall,
        "confidence": confidence,
        "vote_summary": {"bullish": bullish, "bearish": bearish, "neutral": sigs.count("neutral")},
        "sections": sections,
        "as_of": m.get("report_period", "N/A"),
    }