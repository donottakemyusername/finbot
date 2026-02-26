"""server.py
===========
MCP Server for the Stock Analyst chatbot.

Registered tools
----------------
1.  get_stock_overview         – ticker info, price, sector
2.  run_technical_analysis     – all TA indicators + backtest per indicator
3.  run_single_indicator       – one indicator + backtest (e.g. "bollinger")
4.  run_fundamental_analysis   – profitability, growth, health, valuation ratios
5.  run_valuation_analysis     – DCF, owner earnings, EV/EBITDA, RIM
6.  run_deep_research          – EDGAR 10-K / 10-Q extraction
7.  get_full_analysis          – runs everything + final AI verdict

Transport
---------
Set MCP_TRANSPORT=stdio  (default, works with Claude Desktop / mcp CLI)
Set MCP_TRANSPORT=http   (SSE server on PORT, works with custom frontends)

Usage
-----
stdio:  python server.py
http:   MCP_TRANSPORT=http PORT=8000 python server.py
"""

from __future__ import annotations

import json
import os
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from tools.data import get_ticker_info, get_price_history
from tools.technicals import run_technical_analysis, INDICATORS
from tools.fundamentals import run_fundamental_analysis
from tools.valuation import run_valuation_analysis
from tools.deep_research import run_deep_research, get_filing_summary
from engine.aggregator import run_full_analysis

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
mcp = FastMCP(
    name="stock-analyst",
    version="1.0.0",
    description=(
        "AI-powered stock analysis: technical indicators with backtesting, "
        "fundamental scoring, intrinsic value models, EDGAR deep research, "
        "and a final buy/hold/sell verdict."
    ),
)
# ─────────────────────────────────────────────────────────────────────────────


# ── Tool 1: Stock Overview ────────────────────────────────────────────────────

@mcp.tool()
def get_stock_overview(ticker: str) -> str:
    """
    Get a quick overview of a stock: company name, sector, industry,
    current price, 52-week range, and market cap.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL")
    """
    info = get_ticker_info(ticker)
    try:
        df = get_price_history(ticker, years=1)
        price     = round(float(df["Close"].iloc[-1]), 2)
        high_52w  = round(float(df["High"].max()), 2)
        low_52w   = round(float(df["Low"].min()), 2)
        price_chg = round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2)
    except Exception:
        price = high_52w = low_52w = price_chg = None

    result = {
        "ticker":      ticker.upper(),
        "name":        info.get("longName", info.get("shortName", "N/A")),
        "sector":      info.get("sector", "N/A"),
        "industry":    info.get("industry", "N/A"),
        "description": (info.get("longBusinessSummary", "") or "")[:400],
        "price":       price,
        "price_change_1y_%": price_chg,
        "52w_high":    high_52w,
        "52w_low":     low_52w,
        "market_cap":  info.get("marketCap"),
        "employees":   info.get("fullTimeEmployees"),
        "website":     info.get("website"),
    }
    return json.dumps(result, indent=2)


# ── Tool 2: Full Technical Analysis ──────────────────────────────────────────

@mcp.tool()
def analyze_technicals(
    ticker: str,
    years: int = 5,
    end_date: str | None = None,
) -> str:
    """
    Run all technical indicators (Bollinger Bands, SMA 50/200, EMA 12/26,
    RSI 14, MACD) for a stock. Each indicator returns:
      - Current signal: buy / hold / sell
      - Reason (plain English explanation)
      - 5-year backtest: win rate, total return, n_trades, Sharpe ratio

    Args:
        ticker  : Stock ticker (e.g. "AAPL")
        years   : Years of price history to use (default 5)
        end_date: Optional end date "YYYY-MM-DD" (defaults to today)
    """
    result = run_technical_analysis(ticker, years=years, end_date=end_date)
    return json.dumps(result, indent=2)


# ── Tool 3: Single Indicator ─────────────────────────────────────────────────

@mcp.tool()
def analyze_single_indicator(
    ticker: str,
    indicator: str,
    years: int = 5,
) -> str:
    """
    Run a SINGLE technical indicator for a stock with backtest.

    Args:
        ticker    : Stock ticker (e.g. "TSLA")
        indicator : One of: bollinger, sma, ema, rsi, macd
        years     : Years of backtest history (default 5)
    """
    valid = list(INDICATORS.keys())
    if indicator not in valid:
        return json.dumps({"error": f"Unknown indicator '{indicator}'. Valid: {valid}"})

    result = run_technical_analysis(ticker, years=years, indicators=[indicator])
    ind_data = result["indicators"].get(indicator, {})
    return json.dumps({
        "ticker":    ticker,
        "price":     result["price"],
        "as_of":     result["as_of"],
        "indicator": ind_data,
    }, indent=2)


# ── Tool 4: Fundamental Analysis ─────────────────────────────────────────────

@mcp.tool()
def analyze_fundamentals(
    ticker: str,
    end_date: str | None = None,
) -> str:
    """
    Analyze company fundamentals: profitability (ROE, margins), growth
    (revenue, earnings), financial health (current ratio, D/E, FCF),
    and valuation ratios (P/E, P/B, P/S).

    Returns an overall signal (bullish / neutral / bearish) with
    scored sub-sections and detailed metric values.

    Args:
        ticker  : Stock ticker (e.g. "MSFT")
        end_date: Optional cutoff date "YYYY-MM-DD"
    """
    result = run_fundamental_analysis(ticker, end_date=end_date)
    return json.dumps(result, indent=2)


# ── Tool 5: Valuation Analysis ───────────────────────────────────────────────

@mcp.tool()
def analyze_valuation(
    ticker: str,
    end_date: str | None = None,
) -> str:
    """
    Run four intrinsic value models (DCF, Owner Earnings, EV/EBITDA,
    Residual Income) and compare to current market cap.

    Returns overall signal, % gap to intrinsic value, and per-model breakdown.

    Args:
        ticker  : Stock ticker (e.g. "NVDA")
        end_date: Optional cutoff date "YYYY-MM-DD"
    """
    result = run_valuation_analysis(ticker, end_date=end_date)
    return json.dumps(result, indent=2)


# ── Tool 6: Deep Research (EDGAR) ────────────────────────────────────────────

@mcp.tool()
def deep_research_edgar(
    ticker: str,
    form_type: str = "10-K",
    sections: list[str] | None = None,
) -> str:
    """
    Pull the latest SEC filing (10-K or 10-Q) and extract key sections:
      - business: Company description and strategy
      - risk_factors: Key risks
      - mda: Management's Discussion & Analysis
      - financial_statements: Financial highlights

    Args:
        ticker    : Stock ticker (e.g. "AAPL")
        form_type : "10-K" (annual) or "10-Q" (quarterly)
        sections  : List of sections to extract (default: all four)
    """
    result = run_deep_research(ticker, form_type=form_type, sections=sections)
    return json.dumps(result, indent=2)


@mcp.tool()
def list_edgar_filings(ticker: str, form_type: str = "10-K") -> str:
    """
    List available SEC filings for a ticker without fetching content.
    Useful for checking what filings exist before deep_research_edgar.

    Args:
        ticker    : Stock ticker
        form_type : "10-K" or "10-Q"
    """
    result = get_filing_summary(ticker, form_type=form_type)
    return json.dumps(result, indent=2)


# ── Tool 7: Full Analysis (master tool) ──────────────────────────────────────

@mcp.tool()
def get_full_analysis(
    ticker: str,
    include_deep_research: bool = False,
    years: int = 5,
) -> str:
    """
    Run the complete analysis pipeline for a stock and return a final
    AI-generated buy / hold / sell verdict with reasoning.

    This tool:
      1. Runs all technical indicators + backtests
      2. Runs fundamental analysis
      3. Runs all valuation models
      4. Optionally fetches EDGAR 10-K (deep research mode)
      5. Aggregates everything with Claude into a final verdict

    The response includes:
      - Final verdict: BUY / HOLD / SELL
      - Confidence score (0-100)
      - AI reasoning paragraph
      - Supporting arguments & key risks
      - Per-indicator signal dashboard
      - Per-indicator backtest win rates

    Args:
        ticker               : Stock ticker (e.g. "AAPL")
        include_deep_research: If True, fetch EDGAR 10-K (slower but more thorough)
        years                : Years of price history for technical backtest (default 5)
    """
    # Run all agents
    tech  = run_technical_analysis(ticker, years=years)
    fund  = run_fundamental_analysis(ticker)
    valu  = run_valuation_analysis(ticker)
    edgar = run_deep_research(ticker, form_type="10-K", sections=["mda"]) if include_deep_research else None

    # Aggregate
    result = run_full_analysis(
        ticker=ticker,
        technical_result=tech,
        fundamental_result=fund,
        valuation_result=valu,
        deep_research_result=edgar,
        use_ai=True,
    )

    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()

    if transport == "http":
        port = int(os.getenv("PORT", 8000))
        print(f"Starting MCP server on HTTP/SSE port {port} ...")
        mcp.run(transport="sse", port=port)
    else:
        print("Starting MCP server on stdio transport ...")
        mcp.run(transport="stdio")