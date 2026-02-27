"""chatbot.py
=============
Option B: Custom chatbot with explicit Anthropic tool-calling loop.

Flow
----
User message
    → Claude sees message + all tool schemas
    → Claude decides which tool(s) to call
    → We execute the tool(s) locally
    → Results sent back to Claude
    → Claude synthesizes final response
    → (repeat for multi-turn)

Run
---
    python chatbot.py                  # interactive CLI
    python chatbot.py --stream         # with streaming output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Generator

import anthropic
from dotenv import load_dotenv

# ── Local tool implementations ───────────────────────────────────────────────
from tools.data import get_ticker_info, get_price_history
from tools.technicals import run_technical_analysis, INDICATORS
from tools.fundamentals import run_fundamental_analysis
from tools.valuation import run_valuation_analysis
from tools.deep_research import run_deep_research, get_filing_summary
from engine.aggregator import run_full_analysis

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# 1.  TOOL SCHEMAS  (explicit JSON Schema — Claude uses these to decide what to call)
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "name": "get_stock_overview",
        "description": (
            "Get a quick overview of a stock: company name, sector, industry, "
            "current price, 52-week high/low, 1-year price change, and market cap. "
            "Use this when the user asks about what a company does, its sector, or basic price info."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'AAPL', 'TSLA', 'MSFT'",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "analyze_technicals",
        "description": (
            "Run ALL technical indicators for a stock: Bollinger Bands, SMA 50/200 "
            "(Golden/Death Cross), EMA 12/26, RSI 14, and MACD 12/26/9. "
            "Each indicator returns its current buy/hold/sell signal, plain-English reason, "
            "and a 5-year backtest result (win rate, total return, Sharpe ratio, n_trades). "
            "Use this when the user asks for a full technical analysis or 'all indicators'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "years": {
                    "type": "integer",
                    "description": "Years of price history for backtest (default 5)",
                    "default": 5,
                },
                "end_date": {
                    "type": "string",
                    "description": "Optional end date YYYY-MM-DD (defaults to today)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "analyze_single_indicator",
        "description": (
            "Run a SINGLE technical indicator for a stock with a 5-year backtest. "
            "Use this when the user asks about a specific indicator by name, e.g. "
            "'What does Bollinger Bands say about AAPL?' or 'Show me the RSI for TSLA'. "
            "Available indicators: bollinger, sma, ema, rsi, macd."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "indicator": {
                    "type": "string",
                    "enum": ["bollinger", "sma", "ema", "rsi", "macd"],
                    "description": "Which indicator to run",
                },
                "years": {
                    "type": "integer",
                    "description": "Years of backtest history (default 5)",
                    "default": 5,
                },
            },
            "required": ["ticker", "indicator"],
        },
    },
    {
        "name": "analyze_multiple_indicators",
        "description": (
            "Run a SPECIFIC SET of technical indicators for a stock. "
            "Use this when the user asks about 2-4 specific indicators, e.g. "
            "'Compare Bollinger and RSI for AAPL' or 'Show me MACD and SMA for NVDA'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "indicators": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["bollinger", "sma", "ema", "rsi", "macd"],
                    },
                    "description": "List of indicators to run",
                },
                "years": {
                    "type": "integer",
                    "description": "Years of backtest history (default 5)",
                    "default": 5,
                },
            },
            "required": ["ticker", "indicators"],
        },
    },
    {
        "name": "analyze_fundamentals",
        "description": (
            "Analyze company fundamentals across 5 areas: "
            "profitability (ROE, net margin, operating margin, ROA), "
            "growth (revenue, earnings, book value growth), "
            "financial health (current ratio, debt/equity, FCF conversion), "
            "valuation ratios (P/E, P/B, P/S), and dividends if applicable. "
            "Returns bullish/neutral/bearish signal per section with scores and metric values. "
            "Use when user asks about fundamentals, financials, earnings quality, or business health."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "end_date": {
                    "type": "string",
                    "description": "Optional cutoff date YYYY-MM-DD",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "analyze_valuation",
        "description": (
            "Run four intrinsic value models and compare to current market cap: "
            "1) DCF (Discounted Cash Flow, 35% weight), "
            "2) Owner Earnings / Buffett model (35% weight), "
            "3) EV/EBITDA implied equity value (20% weight), "
            "4) Residual Income Model (10% weight). "
            "Returns overall signal, weighted % gap to intrinsic value, and per-model breakdown. "
            "Use when user asks about intrinsic value, whether a stock is over/undervalued, or DCF."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "end_date": {
                    "type": "string",
                    "description": "Optional cutoff date YYYY-MM-DD",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "deep_research_edgar",
        "description": (
            "Pull the latest 10-K (annual) or 10-Q (quarterly) SEC filing for a company "
            "and extract key sections: business description, risk factors, "
            "management discussion & analysis (MD&A), and financial statements. "
            "Use when the user asks for deep research, wants to know about risks, "
            "strategy, or management commentary from official filings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "form_type": {
                    "type": "string",
                    "enum": ["10-K", "10-Q"],
                    "description": "10-K for annual report, 10-Q for quarterly (default 10-K)",
                    "default": "10-K",
                },
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["business", "risk_factors", "mda", "financial_statements"],
                    },
                    "description": "Which sections to extract (default: all four)",
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_full_analysis",
        "description": (
            "Run the COMPLETE analysis pipeline and return a final AI-generated "
            "buy/hold/sell verdict with reasoning. This runs: all technical indicators "
            "with backtests, full fundamental analysis, all valuation models, and "
            "optionally EDGAR deep research. Returns final verdict, confidence score, "
            "reasoning paragraph, supporting arguments, key risks, and a full indicator dashboard. "
            "Use when user asks for a full analysis, final recommendation, or 'should I buy X'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol",
                },
                "include_deep_research": {
                    "type": "boolean",
                    "description": "If true, also fetches EDGAR 10-K (slower but more thorough)",
                    "default": False,
                },
                "years": {
                    "type": "integer",
                    "description": "Years of price history for technical backtest (default 5)",
                    "default": 5,
                },
            },
            "required": ["ticker"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TOOL DISPATCHER  (maps tool name → actual Python function call)
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name and return its JSON string result."""

    try:
        if tool_name == "get_stock_overview":
            ticker = tool_input["ticker"]
            info = get_ticker_info(ticker)
            try:
                df = get_price_history(ticker, years=1)
                price    = round(float(df["Close"].iloc[-1]), 2)
                high_52w = round(float(df["High"].max()), 2)
                low_52w  = round(float(df["Low"].min()), 2)
                chg_1y   = round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2)
            except Exception:
                price = high_52w = low_52w = chg_1y = None
            result = {
                "ticker":            ticker.upper(),
                "name":              info.get("longName", info.get("shortName", "N/A")),
                "sector":            info.get("sector", "N/A"),
                "industry":          info.get("industry", "N/A"),
                "description":       (info.get("longBusinessSummary", "") or "")[:400],
                "price":             price,
                "price_change_1y_%": chg_1y,
                "52w_high":          high_52w,
                "52w_low":           low_52w,
                "market_cap":        info.get("marketCap"),
            }
            return json.dumps(result, indent=2)

        elif tool_name == "analyze_technicals":
            result = run_technical_analysis(
                ticker=tool_input["ticker"],
                years=tool_input.get("years", 5),
                end_date=tool_input.get("end_date"),
            )
            return json.dumps(result, indent=2)

        elif tool_name == "analyze_single_indicator":
            result = run_technical_analysis(
                ticker=tool_input["ticker"],
                years=tool_input.get("years", 5),
                indicators=[tool_input["indicator"]],
            )
            ind = result["indicators"].get(tool_input["indicator"], {})
            return json.dumps({
                "ticker":    tool_input["ticker"],
                "price":     result["price"],
                "as_of":     result["as_of"],
                "indicator": ind,
            }, indent=2)

        elif tool_name == "analyze_multiple_indicators":
            result = run_technical_analysis(
                ticker=tool_input["ticker"],
                years=tool_input.get("years", 5),
                indicators=tool_input["indicators"],
            )
            return json.dumps(result, indent=2)

        elif tool_name == "analyze_fundamentals":
            result = run_fundamental_analysis(
                ticker=tool_input["ticker"],
                end_date=tool_input.get("end_date"),
            )
            return json.dumps(result, indent=2)

        elif tool_name == "analyze_valuation":
            result = run_valuation_analysis(
                ticker=tool_input["ticker"],
                end_date=tool_input.get("end_date"),
            )
            return json.dumps(result, indent=2)

        elif tool_name == "deep_research_edgar":
            result = run_deep_research(
                ticker=tool_input["ticker"],
                form_type=tool_input.get("form_type", "10-K"),
                sections=tool_input.get("sections"),
            )
            return json.dumps(result, indent=2)

        elif tool_name == "get_full_analysis":
            ticker  = tool_input["ticker"]
            years   = tool_input.get("years", 5)
            deep    = tool_input.get("include_deep_research", False)

            tech  = run_technical_analysis(ticker, years=years)
            fund  = run_fundamental_analysis(ticker)
            valu  = run_valuation_analysis(ticker)
            edgar = run_deep_research(ticker, form_type="10-K", sections=["mda"]) if deep else None

            result = run_full_analysis(
                ticker=ticker,
                technical_result=tech,
                fundamental_result=fund,
                valuation_result=valu,
                deep_research_result=edgar,
                use_ai=True,
            )
            return json.dumps(result, indent=2)

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": tool_name})


# ─────────────────────────────────────────────────────────────────────────────
# 3.  SYSTEM PROMPT
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert stock analysis assistant with access to real-time financial tools.

When a user asks about a stock, decide which tool(s) to call based on their query:
- Specific indicator question (e.g. "what does Bollinger say") → analyze_single_indicator
- Multiple specific indicators → analyze_multiple_indicators  
- "Full technical analysis" or "all indicators" → analyze_technicals
- Fundamentals / financials / earnings → analyze_fundamentals
- Intrinsic value / DCF / overvalued? → analyze_valuation
- 10-K / 10-Q / risk factors / deep research → deep_research_edgar
- "Full analysis" / "should I buy?" / final verdict → get_full_analysis
- Basic company info / price → get_stock_overview

You can call multiple tools in parallel when needed (e.g. user asks for both technicals and fundamentals).

After receiving tool results, synthesize them into a clear, concise response. Always:
1. Lead with the signal (Buy / Hold / Sell) and confidence
2. Explain the key reasons in plain English
3. Mention backtest win rates for technical indicators
4. Flag any conflicting signals between indicators
5. End with a brief risk caveat

Format your response in a readable way using headers and bullet points where appropriate.
Do NOT dump raw JSON at the user — always interpret the data.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CHATBOT CLASS  (multi-turn conversation with tool loop)
# ─────────────────────────────────────────────────────────────────────────────

class StockAnalystChatbot:
    def __init__(self, stream: bool = False):
        self.client   = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.history: list[dict] = []
        self.stream   = stream
        self.model    = "claude-sonnet-4-6"

    def chat(self, user_message: str) -> tuple[str, dict]:
        """
        Send a message and get a response.
        Handles multi-step tool calling automatically.
        Returns the final text response.
        """
        self.history.append({"role": "user", "content": user_message})
        self._tool_data: dict = {}

        while True:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=self.history,
            )

            # ── Case 1: Claude wants to call tools ───────────────────────────
            if response.stop_reason == "tool_use":
                # Add Claude's response (which contains tool_use blocks) to history
                self.history.append({
                    "role": "assistant",
                    "content": response.content,
                })

                # Execute all requested tools
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  🔧 Calling tool: {block.name}({json.dumps(block.input)})")
                        result = dispatch_tool(block.name, block.input)
                        tool_results.append({
                            "type":        "tool_result",
                            "tool_use_id": block.id,
                            "content":     result,
                        })
                        # Store raw parsed result for frontend visualizations
                        try:
                            parsed = json.loads(result)
                            self._tool_data[block.name] = parsed
                        except Exception:
                            pass

                # Send tool results back to Claude
                self.history.append({
                    "role": "user",
                    "content": tool_results,
                })
                # Loop again — Claude may call more tools or produce final response

            # ── Case 2: Claude has a final text response ──────────────────────
            elif response.stop_reason == "end_turn":
                text = "".join(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )
                self.history.append({"role": "assistant", "content": text})
                return text, self._tool_data

            else:
                return f"Unexpected stop reason: {response.stop_reason}", {}

    def reset(self):
        """Clear conversation history."""
        self.history = []


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FastAPI backend  (used by React dashboard)
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Request model (module-level so Pydantic can resolve it)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from pydantic import BaseModel as _BaseModel
    class ChatRequest(_BaseModel):
        message: str
        session_id: str = "default"
except ImportError:
    pass

def create_api():
    """
    Returns a FastAPI app.  Import and run with uvicorn:
        uvicorn chatbot:create_api --factory --reload
    """
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel
    except ImportError:
        raise ImportError("Run: pip install fastapi uvicorn")

    app = FastAPI(title="Stock Analyst API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten in production
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # One chatbot instance per session (for demo; use Redis for production)
    sessions: dict[str, StockAnalystChatbot] = {}

    @app.post("/chat")
    def chat_endpoint(req: ChatRequest):
        if req.session_id not in sessions:
            sessions[req.session_id] = StockAnalystChatbot()
        bot = sessions[req.session_id]

        tool_calls_made: list[str] = []
        original_dispatch = dispatch_tool

        def tracking_dispatch(name, inp):
            tool_calls_made.append(name)
            return original_dispatch(name, inp)

        import chatbot as _self
        _self.dispatch_tool = tracking_dispatch
        try:
            response, tool_data = bot.chat(req.message)
        except Exception as e:
            _self.dispatch_tool = original_dispatch
            return {"response": f"Error: {str(e)}", "tool_calls": tool_calls_made, "tool_data": {}, "session_id": req.session_id}
        _self.dispatch_tool = original_dispatch

        return {"response": response, "tool_calls": tool_calls_made, "tool_data": tool_data, "session_id": req.session_id}

    @app.delete("/chat/{session_id}")
    def reset_session(session_id: str):
        if session_id in sessions:
            sessions[session_id].reset()
        return {"status": "reset"}

    @app.get("/tools")
    def list_tools():
        """Return all available tool schemas — useful for the frontend."""
        return {"tools": [{"name": t["name"], "description": t["description"]} for t in TOOL_SCHEMAS]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


# ─────────────────────────────────────────────────────────────────────────────
# 6.  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_cli():
    parser = argparse.ArgumentParser(description="Stock Analyst CLI Chatbot")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    args = parser.parse_args()

    bot = StockAnalystChatbot(stream=args.stream)

    print("\n🤖 Stock Analyst Chatbot")
    print("=" * 50)
    print("Ask me anything about stocks. Examples:")
    print("  • Analyze AAPL using Bollinger Bands and RSI")
    print("  • What's the intrinsic value of NVDA?")
    print("  • Run a full analysis on TSLA")
    print("  • Compare fundamentals of MSFT and GOOGL")
    print("  • Pull the 10-K risk factors for AAPL")
    print("\nType 'reset' to clear history, 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break
        if user_input.lower() == "reset":
            bot.reset()
            print("🔄 Conversation reset.\n")
            continue

        print("\nAssistant: ", end="", flush=True)
        response = bot.chat(user_input)
        print(response)
        print()


if __name__ == "__main__":
    run_cli()