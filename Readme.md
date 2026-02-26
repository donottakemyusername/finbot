# Stock Analyst MCP — Setup Guide

## Quick Start

### 1. Clone & install

```bash
cd stock-analyst-mcp
pip install -r requirements.txt
cp .env.example .env
# Fill in your keys in .env
```

### 2. Set up environment variables

Edit `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
FINANCIAL_DATASETS_API_KEY=fd-...
EDGAR_USER_AGENT="Your Name your@email.com"
```

> **EDGAR is free** — no key needed, just set a user agent so SEC can contact you if needed.

---

## Running the MCP Server

### Option A: Claude Desktop (stdio) — Recommended to start

Add to your `claude_desktop_config.json`:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`  
**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "stock-analyst": {
      "command": "python",
      "args": ["/absolute/path/to/stock-analyst-mcp/server.py"],
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "FINANCIAL_DATASETS_API_KEY": "fd-...",
        "EDGAR_USER_AGENT": "Your Name your@email.com"
      }
    }
  }
}
```

Restart Claude Desktop. You should see the tools appear.

### Option B: HTTP/SSE Server (for custom frontends)

```bash
MCP_TRANSPORT=http PORT=8000 python server.py
```

Then configure your MCP client to connect to `http://localhost:8000/sse`.

### Option C: MCP CLI (for testing)

```bash
pip install mcp[cli]
mcp dev server.py
```

---

## Available MCP Tools

| Tool | Description |
|------|-------------|
| `get_stock_overview` | Company info, price, 52w range |
| `analyze_technicals` | All TA indicators + backtests |
| `analyze_single_indicator` | One indicator (e.g. bollinger) + backtest |
| `analyze_fundamentals` | ROE, margins, growth, health, ratios |
| `analyze_valuation` | DCF, owner earnings, EV/EBITDA, RIM |
| `deep_research_edgar` | EDGAR 10-K/10-Q section extraction |
| `list_edgar_filings` | List available filings (no content fetch) |
| `get_full_analysis` | **Master tool** — runs everything + AI verdict |

---

## Example Prompts (in Claude Desktop)

```
Analyze AAPL technically — what do Bollinger Bands and RSI say, and what's the 5-year backtest win rate?

Run a full analysis on NVDA including deep research from their latest 10-K.

Compare the fundamentals of MSFT and GOOGL.

What's the intrinsic value gap for TSLA using DCF?

Run get_full_analysis on META and tell me if I should buy, hold, or sell.
```

---

## Dashboard Frontend

The React dashboard (`frontend/dashboard.jsx`) is a standalone component.

**To run it:**

```bash
# Option 1: Drop into any React project
cp frontend/dashboard.jsx src/components/StockDashboard.jsx

# Option 2: Quick standalone with Vite
npm create vite@latest stock-dashboard -- --template react
cd stock-dashboard
npm install
cp ../frontend/dashboard.jsx src/App.jsx
npm run dev
```

**To wire it to live data:** Replace the `runAnalysis` mock in dashboard.jsx with a real fetch to your MCP HTTP server or a thin FastAPI wrapper around the Python tools.

---

## Project Structure

```
stock-analyst-mcp/
├── server.py                  ← MCP server (entry point)
├── tools/
│   ├── data.py                ← yfinance + financialdatasets fetchers
│   ├── technicals.py          ← BB, SMA, EMA, RSI, MACD + backtest
│   ├── fundamentals.py        ← Profitability, growth, health, ratios
│   ├── valuation.py           ← DCF, owner earnings, EV/EBITDA, RIM
│   └── deep_research.py       ← EDGAR 10-K/10-Q parser
├── engine/
│   ├── backtest.py            ← Generic long-only backtester
│   └── aggregator.py          ← Rule-based + AI verdict aggregation
├── frontend/
│   └── dashboard.jsx          ← React signal dashboard
├── .env.example
├── requirements.txt
└── README.md
```

---

## Architecture Notes

### How signals aggregate

```
Technical (35%)    Fundamental (35%)    Valuation (30%)
     ↓                    ↓                   ↓
  buy/hold/sell      bullish/neutral/bearish  bullish/neutral/bearish
     └────────────────────┴───────────────────┘
                          ↓
                  Weighted average score
                  Score > 0.15  → BUY
                  Score < -0.15 → SELL
                  Otherwise     → HOLD
                          ↓
                  Claude AI (claude-sonnet-4-6)
                  Synthesizes all data → final verdict + reasoning
```

### Backtest methodology

- **Long-only**, enter on next open after signal bar
- **0.1% commission** per side
- **Exit**: indicator-specific (e.g. BB exits at upper band or SMA; RSI exits at overbought)
- Reports: win rate, compounded return, Sharpe ratio, max drawdown, avg hold days
- Compared against buy & hold over same period