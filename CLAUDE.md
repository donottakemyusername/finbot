# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP Stock Analyst FinBot — an AI-powered stock analysis system exposing financial analysis tools to Claude via the Model Context Protocol (MCP). It combines technical, fundamental, valuation, and trinity (三位一体) analysis into a multi-agent consensus verdict.

## Environment Setup

```bash
pip install -r requirements.txt
# Copy and fill in .env:
# ANTHROPIC_API_KEY, FINANCIAL_DATASETS_API_KEY (optional), EDGAR_USER_AGENT
```

## Running the System

```bash
# MCP server (stdio, for Claude Desktop)
python server.py

# MCP server (HTTP/SSE mode)
MCP_TRANSPORT=http PORT=8000 python server.py

# Interactive CLI chatbot
python chatbot.py

# FastAPI web server (Railway deployment mode)
uvicorn chatbot:create_api --factory --host 0.0.0.0 --port 8000

# MCP dev mode
mcp dev server.py
```

## Frontend

```bash
cd stock-ui
npm install
npm run dev       # Dev server
npm run build     # Production build
npm run lint      # ESLint
```

## Testing

The trinity module has a test suite at `tools/trinity/test.py`. There is no pytest setup; run:
```bash
python tools/trinity/test.py
```

## Architecture

The system is a **multi-agent pipeline** that aggregates three independent signals into an AI-synthesized verdict:

```
Ticker → [Technical Agent (35%)] + [Fundamental Agent (35%)] + [Valuation Agent (30%)]
       → Rule-based weighted aggregation
       → Claude AI synthesis → Final verdict (BUY/HOLD/SELL) + reasoning
```

### Key Modules

| Module | Responsibility |
|--------|---------------|
| `server.py` | MCP server (FastMCP), exposes 9 tools via stdio or HTTP/SSE |
| `chatbot.py` | Interactive CLI + FastAPI backend with tool-calling loop |
| `tools/data.py` | Unified data layer: yfinance, financialdatasets API, EDGAR |
| `tools/technicals.py` | 5 TA indicators (BB, SMA, EMA, RSI, MACD) + 5-year backtest per indicator |
| `tools/fundamentals.py` | ROE, margins, growth, financial health → bullish/neutral/bearish signal |
| `tools/valuation.py` | DCF (35%), Owner Earnings (35%), EV/EBITDA (20%), Residual Income (10%) → gap% signal |
| `tools/deep_research.py` | SEC EDGAR 10-K/10-Q extraction via free SEC API |
| `engine/backtest.py` | Event-driven long-only backtester (0.1% commission) |
| `engine/aggregator.py` | Signal aggregation + rule-based weighting + Claude AI verdict |
| `tools/trinity/` | Trinity Trading System (三位一体) — advanced Chinese technical analysis |

### MCP Tools Exposed (server.py)

- `get_stock_overview` — Basic info + 52-week range
- `analyze_technicals` — All TA indicators + backtests
- `analyze_single_indicator` — One indicator at a time
- `analyze_fundamentals` — Profitability, growth, health metrics
- `analyze_valuation` — DCF, Owner Earnings, EV/EBITDA, RIM
- `deep_research_edgar` — 10-K/10-Q SEC filing extraction
- `list_edgar_filings` — Available EDGAR filings for a ticker
- `get_full_analysis` — Master tool aggregating all agents
- `trinity_analysis` — Trinity system with time-space state + structure analysis

### Trinity System (`tools/trinity/`)

The Trinity Trading System (三位一体) is a Chinese technical analysis framework:
- `indicators.py` — Hard indicators: MA alignment, MACD, Bollinger, support/resistance
- `state.py` — Time-space state classification (极强/强/中性偏强/etc.)
- `prompt.py` — Claude prompts for soft signal synthesis
- `analysis.py` — Main entry point merging all trinity dimensions
- Supports multi-timeframe: daily, weekly, monthly, hourly

### Signal Aggregation Logic

- Technical: >3 of 5 indicators = BUY; <2 = SELL; else HOLD
- Fundamental: >66% bullish sections = BULLISH; <33% = BEARISH
- Valuation: intrinsic_value gap >+15% = BULLISH; <-15% = BEARISH
- Combined weights: Technical 35% + Fundamental 35% + Valuation 30%
- Final verdict: Claude (haiku-4.5) synthesizes into narrative + confidence score

### Transport Modes

- `stdio`: Default for Claude Desktop integration
- `http`: SSE server for custom frontends/Railway deployment

### Deployment

Railway.app config in `railway.toml` runs `uvicorn chatbot:create_api --factory` with `/health` healthcheck.
