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


def get_financial_metrics_from_yfinance(ticker: str) -> list[dict]:
    """Build a financial metrics dict from yfinance .info — used as fallback."""
    info = get_ticker_info(ticker)
    if not info:
        return []
    # Map yfinance keys to financialdatasets field names
    metrics = {
        'ticker':                   ticker.upper(),
        'report_period':            'ttm',
        # Profitability
        'return_on_equity':         info.get('returnOnEquity'),
        'net_margin':               info.get('profitMargins'),
        'operating_margin':         info.get('operatingMargins'),
        'return_on_assets':         info.get('returnOnAssets'),
        'asset_turnover':           None,
        # Growth
        'revenue_growth':           info.get('revenueGrowth'),
        'earnings_growth':          info.get('earningsGrowth'),
        'book_value_growth':        None,
        # Health
        'current_ratio':            info.get('currentRatio'),
        'debt_to_equity':           (info.get('debtToEquity') or 0) / 100 if info.get('debtToEquity') else None,
        'free_cash_flow_per_share': info.get('freeCashflow') / max(info.get('sharesOutstanding', 1), 1) if info.get('freeCashflow') else None,
        'earnings_per_share':       info.get('trailingEps'),
        # Valuation
        'price_to_earnings_ratio':  info.get('trailingPE'),
        'price_to_book_ratio':      info.get('priceToBook'),
        'price_to_sales_ratio':     info.get('priceToSalesTrailing12Months'),
        # Dividends
        'dividend_yield':           info.get('dividendYield'),
        'payout_ratio':             info.get('payoutRatio'),
        # Market
        'market_cap':               info.get('marketCap'),
        'enterprise_value':         info.get('enterpriseValue'),
        'ebitda':                   info.get('ebitda'),
    }
    return [metrics]


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
    """Return financial metrics — tries financialdatasets.ai first, falls back to yfinance."""
    if FD_API_KEY:
        try:
            params: dict = {"ticker": ticker, "period": period, "limit": limit}
            if end_date:
                params["report_period_lte"] = end_date
            data = _fd_get("/financial-metrics/search", params)
            metrics = data.get("financial_metrics", [])
            if metrics:
                return metrics
        except Exception:
            pass
    # Fallback: build metrics from yfinance
    return get_financial_metrics_from_yfinance(ticker)


def _yf_financials(ticker: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Pull income, balance sheet, cash flow from yfinance as fallback."""
    t = yf.Ticker(ticker)
    info = t.info or {}

    def _row(d: dict) -> dict:
        return {k: (float(v) if hasattr(v, '__float__') else v) for k, v in d.items()}

    # Income statement fields
    inc = [{
        'ticker':           ticker.upper(),
        'report_period':    'ttm',
        'revenue':          info.get('totalRevenue'),
        'net_income':       info.get('netIncomeToCommon'),
        'operating_income': info.get('operatingCashflow'),
        'ebitda':           info.get('ebitda'),
        'eps':              info.get('trailingEps'),
        'earnings_growth':  info.get('earningsGrowth'),
        'revenue_growth':   info.get('revenueGrowth'),
    }]

    # Balance sheet fields
    bs = [{
        'ticker':                    ticker.upper(),
        'report_period':             'ttm',
        'total_assets':              info.get('totalAssets'),
        'total_liabilities':         info.get('totalDebt'),
        'total_equity':              info.get('bookValue'),
        'cash_and_equivalents':      info.get('totalCash'),
        'book_value_per_share':      info.get('bookValue'),
        'shares_outstanding':        info.get('sharesOutstanding'),
        'current_assets':            info.get('currentAssets'),
        'current_liabilities':       info.get('currentLiabilities'),
        'long_term_debt':            info.get('longTermDebt'),
    }]

    # Cash flow fields
    cf = [{
        'ticker':              ticker.upper(),
        'report_period':       'ttm',
        'free_cash_flow':      info.get('freeCashflow'),
        'operating_cash_flow': info.get('operatingCashflow'),
        'capital_expenditure': info.get('capitalExpenditures'),
        'net_income':          info.get('netIncomeToCommon'),
    }]

    return inc, bs, cf


def get_income_statements(
    ticker: str,
    period: str = "ttm",
    limit: int = 5,
) -> list[dict]:
    if FD_API_KEY:
        try:
            data = _fd_get("/income-statements/search", {"ticker": ticker, "period": period, "limit": limit})
            result = data.get("income_statements", [])
            if result: return result
        except Exception:
            pass
    inc, _, _ = _yf_financials(ticker)
    return inc


def get_balance_sheets(
    ticker: str,
    period: str = "ttm",
    limit: int = 5,
) -> list[dict]:
    if FD_API_KEY:
        try:
            data = _fd_get("/balance-sheets/search", {"ticker": ticker, "period": period, "limit": limit})
            result = data.get("balance_sheets", [])
            if result: return result
        except Exception:
            pass
    _, bs, _ = _yf_financials(ticker)
    return bs


def get_cash_flow_statements(
    ticker: str,
    period: str = "ttm",
    limit: int = 5,
) -> list[dict]:
    if FD_API_KEY:
        try:
            data = _fd_get("/cash-flow-statements/search", {"ticker": ticker, "period": period, "limit": limit})
            result = data.get("cash_flow_statements", [])
            if result: return result
        except Exception:
            pass
    _, _, cf = _yf_financials(ticker)
    return cf


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