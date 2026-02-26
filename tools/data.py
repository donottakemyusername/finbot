"""tools/data.py
=================
Unified data layer.  All other tools import from here — never call yfinance
or the external APIs directly.

Sources
-------
* yfinance          – OHLCV price history, basic info
* financialdatasets – Income statement, balance sheet, cash flow, metrics
* EDGAR             – 10-K / 10-Q filings (see deep_research.py)
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()


FD_API_KEY = os.getenv("FINANCIAL_DATASETS_API_KEY", "")
FD_BASE = "https://api.financialdatasets.ai"

# ─────────────────────────────────────────────────────────────────────────────
# Price data (yfinance)
# ─────────────────────────────────────────────────────────────────────────────

def get_price_history(
    ticker: str,
    years: int = 5,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Return OHLCV DataFrame indexed by date.

    Columns: Open, High, Low, Close, Volume
    """
    end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today()
    start = end - timedelta(days=years * 365)

    df = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise ValueError(f"No price data found for {ticker}")
    return df


def get_ticker_info(ticker: str) -> dict[str, Any]:
    """Return yfinance .info dict (sector, industry, description, etc.)."""
    t = yf.Ticker(ticker)
    return t.info or {}


# ─────────────────────────────────────────────────────────────────────────────
# Financial Datasets helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fd_get(path: str, params: dict | None = None) -> Any:
    """Authenticated GET to financialdatasets.ai — returns parsed JSON body."""
    if not FD_API_KEY:
        raise RuntimeError(
            "FINANCIAL_DATASETS_API_KEY is not set. "
            "Add it to your .env file."
        )
    headers = {"X-API-KEY": FD_API_KEY}
    url = f"{FD_BASE}{path}"
    resp = requests.get(url, headers=headers, params=params or {}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_financial_metrics(
    ticker: str,
    period: str = "ttm",
    limit: int = 8,
    end_date: str | None = None,
) -> list[dict]:
    """Return a list of financial metric snapshots (most recent first)."""
    params: dict = {"ticker": ticker, "period": period, "limit": limit}
    if end_date:
        params["report_period_lte"] = end_date
    data = _fd_get("/financial-metrics/search", params)
    return data.get("financial_metrics", [])


def get_income_statements(
    ticker: str,
    period: str = "ttm",
    limit: int = 5,
) -> list[dict]:
    data = _fd_get("/income-statements/search", {"ticker": ticker, "period": period, "limit": limit})
    return data.get("income_statements", [])


def get_balance_sheets(
    ticker: str,
    period: str = "ttm",
    limit: int = 5,
) -> list[dict]:
    data = _fd_get("/balance-sheets/search", {"ticker": ticker, "period": period, "limit": limit})
    return data.get("balance_sheets", [])


def get_cash_flow_statements(
    ticker: str,
    period: str = "ttm",
    limit: int = 5,
) -> list[dict]:
    data = _fd_get("/cash-flow-statements/search", {"ticker": ticker, "period": period, "limit": limit})
    return data.get("cash_flow_statements", [])


def get_market_cap(ticker: str, end_date: str | None = None) -> float | None:
    """Best-effort market cap: try financialdatasets first, fall back to yfinance."""
    try:
        metrics = get_financial_metrics(ticker, period="ttm", limit=1, end_date=end_date)
        if metrics and metrics[0].get("market_cap"):
            return float(metrics[0]["market_cap"])
    except Exception:
        pass
    info = get_ticker_info(ticker)
    return info.get("marketCap")


def search_line_items(
    ticker: str,
    line_items: list[str],
    period: str = "ttm",
    limit: int = 2,
    end_date: str | None = None,
) -> list[dict]:
    """Pull specific line items from cash-flow / income / balance-sheet data
    and return them merged into a flat dict list, most recent first.
    """
    cf = get_cash_flow_statements(ticker, period=period, limit=limit)
    inc = get_income_statements(ticker, period=period, limit=limit)
    bs = get_balance_sheets(ticker, period=period, limit=limit)

    merged: list[dict] = []
    for i in range(min(limit, len(cf))):
        row: dict = {}
        if i < len(cf):
            row.update(cf[i])
        if i < len(inc):
            row.update(inc[i])
        if i < len(bs):
            row.update(bs[i])
        merged.append(row)

    # Filter to only requested line items + date fields
    keep = set(line_items) | {"report_period", "ticker"}
    return [{k: v for k, v in r.items() if k in keep} for r in merged]