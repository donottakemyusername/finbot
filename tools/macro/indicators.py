"""tools/macro/indicators.py
============================
Raw macro data fetchers.

Sources
-------
- FRED API (https://fred.stlouisfed.org) — free, requires FRED_API_KEY in .env
  Set FRED_API_KEY= in your .env to activate FRED data.
  Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html
- yfinance — VIX, VIX term structure, SPY beta reference

Key FRED Series
---------------
DGS2       2-Year Treasury Yield
DGS10      10-Year Treasury Yield
DGS30      30-Year Treasury Yield
BAMLC0A0CM ICE BofA IG Corporate OAS (bps)
BAMLH0A0HYM2 ICE BofA HY Index OAS (bps)
DFF        Fed Funds Effective Rate
T10YIE     10-Year Breakeven Inflation Rate
"""
from __future__ import annotations

import os
import time
from datetime import date, timedelta
from typing import Any

import requests
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE    = "https://api.stlouisfed.org/fred/series/observations"
_SLEEP       = 0.2


def _fred_get(series_id: str, n_obs: int = 5) -> list[dict]:
    """Fetch last n observations from FRED. Returns [] if key missing."""
    if not FRED_API_KEY:
        return []
    params = {
        "series_id":           series_id,
        "api_key":             FRED_API_KEY,
        "file_type":           "json",
        "sort_order":          "desc",
        "observation_start":   (date.today() - timedelta(days=30)).isoformat(),
        "limit":               n_obs,
    }
    try:
        time.sleep(_SLEEP)
        r = requests.get(FRED_BASE, params=params, timeout=10)
        r.raise_for_status()
        obs = r.json().get("observations", [])
        return [o for o in obs if o.get("value") not in (".", None, "")]
    except Exception:
        return []


def _latest_fred(series_id: str) -> float | None:
    """Return the most recent non-null FRED value."""
    obs = _fred_get(series_id, n_obs=5)
    for o in obs:
        try:
            v = float(o["value"])
            return v
        except (ValueError, TypeError):
            continue
    return None


def _hist_fred(series_id: str, days: int = 90) -> list[tuple[str, float]]:
    """Return (date, value) pairs for the last `days` calendar days."""
    if not FRED_API_KEY:
        return []
    params = {
        "series_id":         series_id,
        "api_key":           FRED_API_KEY,
        "file_type":         "json",
        "sort_order":        "asc",
        "observation_start": (date.today() - timedelta(days=days)).isoformat(),
    }
    try:
        time.sleep(_SLEEP)
        r = requests.get(FRED_BASE, params=params, timeout=10)
        r.raise_for_status()
        result = []
        for o in r.json().get("observations", []):
            try:
                result.append((o["date"], float(o["value"])))
            except (ValueError, TypeError):
                continue
        return result
    except Exception:
        return []


def fetch_yield_curve() -> dict:
    """
    Fetch 2Y, 10Y, 30Y treasury yields and compute curve slope.

    Returns
    -------
    {
        "y2":       float | None,  # 2-Year yield %
        "y10":      float | None,  # 10-Year yield %
        "y30":      float | None,  # 30-Year yield %
        "spread_10_2": float | None,  # 10Y - 2Y spread bps
        "inverted": bool,
        "curve_regime": "steep" | "normal" | "flat" | "inverted" | "deeply_inverted",
        "source":   "fred" | "unavailable",
        "history_10_2": list of {date, spread} for charting (last 90 days),
    }
    """
    y2  = _latest_fred("DGS2")
    y10 = _latest_fred("DGS10")
    y30 = _latest_fred("DGS30")

    spread = None
    if y2 is not None and y10 is not None:
        spread = round((y10 - y2) * 100, 1)  # in bps

    inverted = spread is not None and spread < 0

    if spread is None:
        regime = "unknown"
    elif spread > 100:
        regime = "steep"
    elif spread > 10:
        regime = "normal"
    elif spread > -50:
        regime = "flat" if spread >= 0 else "inverted"
    else:
        regime = "deeply_inverted"

    # Historical for chart
    hist_raw = _hist_fred("DGS10", days=90)
    hist2_raw = _hist_fred("DGS2", days=90)
    hist2_map = {d: v for d, v in hist2_raw}
    history = [
        {"date": d, "spread": round((v - hist2_map[d]) * 100, 1)}
        for d, v in hist_raw
        if d in hist2_map
    ]

    return {
        "y2":           y2,
        "y10":          y10,
        "y30":          y30,
        "spread_10_2_bps": spread,
        "inverted":     inverted,
        "curve_regime": regime,
        "source":       "fred" if y10 is not None else "unavailable",
        "history_10_2": history[-60:],  # last 60 data points
    }


def fetch_credit_spreads() -> dict:
    """
    Fetch IG and HY credit spreads from FRED (BAMLC0A0CM, BAMLH0A0HYM2).

    Returns
    -------
    {
        "ig_spread_bps":  float | None,
        "hy_spread_bps":  float | None,
        "ig_regime":  "tight" | "normal" | "elevated" | "stressed",
        "hy_regime":  "tight" | "normal" | "elevated" | "stressed",
        "risk_appetite": "high" | "moderate" | "low" | "very_low",
        "source": "fred" | "unavailable",
        "history": list of {date, ig, hy},
    }
    """
    ig = _latest_fred("BAMLC0A0CM")
    hy = _latest_fred("BAMLH0A0HYM2")

    def _ig_regime(v):
        if v is None:   return "unknown"
        if v < 90:      return "tight"
        if v < 140:     return "normal"
        if v < 200:     return "elevated"
        return "stressed"

    def _hy_regime(v):
        if v is None:   return "unknown"
        if v < 300:     return "tight"
        if v < 420:     return "normal"
        if v < 600:     return "elevated"
        return "stressed"

    ig_r = _ig_regime(ig)
    hy_r = _hy_regime(hy)

    risk_score = sum([
        ig_r == "tight", hy_r == "tight",
        ig_r == "normal", hy_r == "normal",
    ])
    if risk_score >= 3:     risk_appetite = "high"
    elif risk_score >= 1:   risk_appetite = "moderate"
    elif ig_r == "elevated" or hy_r == "elevated": risk_appetite = "low"
    else:                   risk_appetite = "very_low"

    # Historical
    ig_hist = _hist_fred("BAMLC0A0CM", days=90)
    hy_hist = _hist_fred("BAMLH0A0HYM2", days=90)
    hy_map  = {d: v for d, v in hy_hist}
    history = [
        {"date": d, "ig": round(v, 1), "hy": round(hy_map.get(d, 0), 1)}
        for d, v in ig_hist
        if d in hy_map
    ]

    return {
        "ig_spread_bps": ig,
        "hy_spread_bps": hy,
        "ig_regime":     ig_r,
        "hy_regime":     hy_r,
        "risk_appetite": risk_appetite,
        "source":        "fred" if ig is not None else "unavailable",
        "history":       history[-60:],
    }


def fetch_vix_data() -> dict:
    """
    Fetch VIX spot + term structure from yfinance.

    Returns
    -------
    {
        "vix":       float,
        "vix9d":     float | None,
        "vix3m":     float | None,
        "term_slope": float | None,  # vix3m - vix (contango > 0, backwardation < 0)
        "vix_regime": "low" | "normal" | "elevated" | "stressed" | "crisis",
        "structure":  "contango" | "backwardation" | "flat",
        "vix_history": list of {date, vix},
    }
    """
    def _last(ticker: str) -> float | None:
        try:
            df = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if df.empty:
                return None
            val = float(df["Close"].iloc[-1].iloc[0] if hasattr(df["Close"].iloc[-1], "iloc") else df["Close"].iloc[-1])
            return round(val, 2)
        except Exception:
            return None

    vix   = _last("^VIX")
    vix9d = _last("^VIX9D")
    vix3m = _last("^VIX3M")

    # VIX history for chart
    vix_hist_df = yf.download("^VIX", period="3mo", progress=False, auto_adjust=True)
    vix_history = []
    if not vix_hist_df.empty:
        for idx, row in vix_hist_df.iterrows():
            try:
                close_val = row["Close"]
                if hasattr(close_val, "iloc"):
                    close_val = close_val.iloc[0]
                vix_history.append({
                    "date": idx.strftime("%Y-%m-%d"),
                    "vix":  round(float(close_val), 2),
                })
            except Exception:
                continue

    if vix is None:
        return {
            "vix": None, "vix9d": None, "vix3m": None,
            "term_slope": None, "vix_regime": "unknown",
            "structure": "unknown", "vix_history": [],
        }

    if vix < 14:        vix_regime = "low"
    elif vix < 20:      vix_regime = "normal"
    elif vix < 26:      vix_regime = "elevated"
    elif vix < 35:      vix_regime = "stressed"
    else:               vix_regime = "crisis"

    term_slope = None
    structure  = "unknown"
    if vix3m is not None:
        term_slope = round(vix3m - vix, 2)
        if term_slope > 1:    structure = "contango"
        elif term_slope < -1: structure = "backwardation"
        else:                 structure = "flat"

    return {
        "vix":         vix,
        "vix9d":       vix9d,
        "vix3m":       vix3m,
        "term_slope":  term_slope,
        "vix_regime":  vix_regime,
        "structure":   structure,
        "vix_history": vix_history[-60:],
    }


def fetch_fed_policy() -> dict:
    """
    Fetch Fed Funds Rate and 10Y breakeven inflation from FRED.

    Returns
    -------
    {
        "fed_funds_rate": float | None,
        "breakeven_10y":  float | None,  # market-implied inflation
        "real_yield_10y": float | None,  # DGS10 - T10YIE
        "policy_stance":  "accommodative" | "neutral" | "restrictive" | "very_restrictive",
        "source": "fred" | "unavailable",
    }
    """
    dff   = _latest_fred("DFF")
    t10yi = _latest_fred("T10YIE")
    y10   = _latest_fred("DGS10")

    real_yield = None
    if y10 is not None and t10yi is not None:
        real_yield = round(y10 - t10yi, 2)

    if dff is None:
        stance = "unknown"
    elif dff < 1.5:   stance = "accommodative"
    elif dff < 3.5:   stance = "neutral"
    elif dff < 5.5:   stance = "restrictive"
    else:             stance = "very_restrictive"

    return {
        "fed_funds_rate":  dff,
        "breakeven_10y":   t10yi,
        "real_yield_10y":  real_yield,
        "policy_stance":   stance,
        "source":          "fred" if dff is not None else "unavailable",
    }
