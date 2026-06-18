"""tools/portfolio.py
=====================
Portfolio exposure analysis for PM-facing workflows.

Input: list of {ticker, weight} positions (weights should sum to ~1).
Output: sector breakdown, beta-weighted market exposure, concentration flags,
        factor tilt summary, and regime-adjusted conviction notes.

Beta is computed as 1-year daily returns regression against SPY.
Sector/industry data sourced from yfinance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Any


def _compute_beta(ticker: str, market_ticker: str = "SPY", years: int = 1) -> float | None:
    """Compute trailing 1-year beta vs SPY using daily log returns."""
    try:
        tickers = [ticker, market_ticker]
        df = yf.download(tickers, period=f"{years}y", progress=False, auto_adjust=True)["Close"]
        if df.empty or df.shape[1] < 2:
            return None
        returns = df.pct_change().dropna()
        stock_col = ticker
        mkt_col   = market_ticker
        if stock_col not in returns.columns or mkt_col not in returns.columns:
            return None
        cov    = np.cov(returns[stock_col], returns[mkt_col])
        beta   = cov[0, 1] / cov[1, 1]
        return round(float(beta), 2)
    except Exception:
        return None


def _get_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info or {}
        return info
    except Exception:
        return {}


def run_portfolio_exposure(positions: list[dict]) -> dict:
    """
    Analyze a portfolio of positions.

    Parameters
    ----------
    positions : list of {ticker: str, weight: float, cost_basis?: float}
        weights are fractions (0.15 = 15%). They don't need to sum to 1.

    Returns
    -------
    {
        "total_weight":         float,  # sum of input weights
        "positions":            list of enriched position dicts,
        "sector_exposure":      dict {sector: weight},
        "top_sectors":          list,
        "beta_weighted_exposure": float,  # portfolio beta
        "concentration_flags":  list of warning strings,
        "factor_tilt": {
            "growth_weight":    float,
            "value_weight":     float,
            "cyclical_weight":  float,
            "defensive_weight": float,
        },
        "risk_summary":         str,
        "regime_note":          str,
    }
    """
    if not positions:
        return {"error": "No positions provided"}

    # Normalize weights
    total_w = sum(p.get("weight", 0) for p in positions)
    if total_w <= 0:
        return {"error": "All weights are zero"}

    enriched      = []
    sector_agg    = {}
    total_beta_w  = 0.0
    total_w_valid = 0.0

    GROWTH_SECTORS     = {"Technology", "Communication Services", "Consumer Discretionary"}
    VALUE_SECTORS      = {"Financials", "Energy", "Materials", "Utilities", "Real Estate"}
    CYCLICAL_SECTORS   = {"Industrials", "Materials", "Consumer Discretionary", "Energy", "Financials"}
    DEFENSIVE_SECTORS  = {"Health Care", "Consumer Staples", "Utilities"}

    factor_growth    = 0.0
    factor_value     = 0.0
    factor_cyclical  = 0.0
    factor_defensive = 0.0

    for pos in positions:
        ticker = pos.get("ticker", "").upper().strip()
        weight = pos.get("weight", 0)
        if not ticker or weight <= 0:
            continue

        w_norm = weight / total_w  # normalized fraction

        info   = _get_info(ticker)
        beta   = _compute_beta(ticker)
        sector = info.get("sector", "Unknown")
        industry = info.get("industry", "Unknown")
        name   = info.get("shortName", ticker)

        # Current price
        try:
            price = round(float(info.get("regularMarketPrice") or info.get("previousClose") or 0), 2)
        except Exception:
            price = None

        # Sector aggregation
        sector_agg[sector] = round(sector_agg.get(sector, 0) + w_norm, 4)

        # Beta contribution
        if beta is not None:
            total_beta_w  += beta * w_norm
            total_w_valid += w_norm

        # Factor tilts
        if sector in GROWTH_SECTORS:     factor_growth    += w_norm
        if sector in VALUE_SECTORS:      factor_value     += w_norm
        if sector in CYCLICAL_SECTORS:   factor_cyclical  += w_norm
        if sector in DEFENSIVE_SECTORS:  factor_defensive += w_norm

        enriched.append({
            "ticker":     ticker,
            "name":       name,
            "weight":     round(weight, 4),
            "weight_pct": round(w_norm * 100, 1),
            "sector":     sector,
            "industry":   industry,
            "beta":       beta,
            "price":      price,
            "cost_basis": pos.get("cost_basis"),
        })

    # Sort by weight descending
    enriched.sort(key=lambda x: x["weight"], reverse=True)

    # Portfolio beta
    portfolio_beta = round(total_beta_w, 2) if total_w_valid > 0 else None

    # Sector breakdown sorted
    sector_sorted = dict(sorted(sector_agg.items(), key=lambda x: x[1], reverse=True))

    # Top sectors list
    top_sectors = [
        {"sector": k, "weight_pct": round(v * 100, 1)}
        for k, v in sector_sorted.items()
    ]

    # Concentration flags
    flags = []
    max_single = max((p["weight_pct"] for p in enriched), default=0)
    if max_single > 25:
        top_pos = enriched[0]
        flags.append(f"Single-name concentration: {top_pos['ticker']} is {max_single:.1f}% of portfolio — exceeds 25% threshold")

    top_sector_weight = top_sectors[0]["weight_pct"] if top_sectors else 0
    if top_sector_weight > 40:
        flags.append(f"Sector concentration: {top_sectors[0]['sector']} at {top_sector_weight:.1f}% — consider diversifying")

    if portfolio_beta and portfolio_beta > 1.4:
        flags.append(f"High beta portfolio ({portfolio_beta:.2f}x SPY) — amplified drawdown risk in risk-off regimes")

    if portfolio_beta and portfolio_beta < 0.5:
        flags.append(f"Low beta ({portfolio_beta:.2f}x) — may underperform in strong bull markets")

    # Risk summary
    n = len(enriched)
    if n <= 5:
        risk_summary = f"Concentrated portfolio ({n} names). High idiosyncratic risk."
    elif n <= 15:
        risk_summary = f"Moderately concentrated ({n} names). Monitor single-name concentration."
    else:
        risk_summary = f"Diversified portfolio ({n} names). Watch for factor concentration."

    if flags:
        risk_summary += f" {len(flags)} concentration flag(s) detected."

    # Regime note (generic — macro_regime tool provides full context)
    if portfolio_beta and portfolio_beta > 1.2:
        regime_note = "High-beta portfolio will amplify returns in GOLDILOCKS/RISK_ON but suffer disproportionately in RISK_OFF or RECESSION_RISK regimes. Run macro_regime to assess current environment."
    elif factor_defensive > 0.4:
        regime_note = "Defensive tilt — well-positioned for RISK_OFF/LATE_CYCLE but will lag in GOLDILOCKS. Consider adding cyclical exposure if macro improves."
    else:
        regime_note = "Balanced portfolio. Run macro_regime tool alongside this analysis for regime-adjusted conviction."

    return {
        "total_weight":           round(total_w, 4),
        "n_positions":            n,
        "positions":              enriched,
        "sector_exposure":        {k: round(v * 100, 1) for k, v in sector_sorted.items()},
        "top_sectors":            top_sectors,
        "portfolio_beta":         portfolio_beta,
        "concentration_flags":    flags,
        "factor_tilt": {
            "growth_pct":    round(factor_growth * 100, 1),
            "value_pct":     round(factor_value * 100, 1),
            "cyclical_pct":  round(factor_cyclical * 100, 1),
            "defensive_pct": round(factor_defensive * 100, 1),
        },
        "risk_summary":  risk_summary,
        "regime_note":   regime_note,
    }
