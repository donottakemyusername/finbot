"""chatbot.py
=============
Option B: Custom chatbot with explicit Anthropic tool-calling loop.
三位一体 (Trinity Trading System) 已集成为独立工具。

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
    uvicorn chatbot:create_api --factory --reload --port 8000  # FastAPI
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import anthropic
from dotenv import load_dotenv

from tools.data import get_ticker_info, get_price_history
from tools.technicals import run_technical_analysis, INDICATORS
from tools.fundamentals import run_fundamental_analysis
from tools.valuation import run_valuation_analysis
from tools.deep_research import run_deep_research, get_filing_summary
from engine.aggregator import run_full_analysis

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TOOL SCHEMAS
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
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. 'AAPL'"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "analyze_technicals",
        "description": (
            "Run ALL technical indicators for a stock: Bollinger Bands, SMA 50/200, "
            "EMA 12/26, RSI 14, and MACD 12/26/9. Each indicator returns signal + 5-year backtest. "
            "Use when the user asks for full technical analysis or 'all indicators'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "years":  {"type": "integer", "default": 5},
                "end_date": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "analyze_single_indicator",
        "description": (
            "Run a SINGLE technical indicator for a stock with backtest. "
            "Use when user asks about a specific indicator, e.g. 'What does Bollinger say about AAPL?'. "
            "Available: bollinger, sma, ema, rsi, macd."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker":    {"type": "string"},
                "indicator": {"type": "string", "enum": ["bollinger", "sma", "ema", "rsi", "macd"]},
                "years":     {"type": "integer", "default": 5},
            },
            "required": ["ticker", "indicator"],
        },
    },
    {
        "name": "analyze_multiple_indicators",
        "description": (
            "Run a SPECIFIC SET of technical indicators. "
            "Use when user asks about 2-4 specific indicators, e.g. 'Compare Bollinger and RSI'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string"},
                "indicators": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["bollinger", "sma", "ema", "rsi", "macd"]},
                },
                "years": {"type": "integer", "default": 5},
            },
            "required": ["ticker", "indicators"],
        },
    },
    {
        "name": "analyze_fundamentals",
        "description": (
            "Analyze company fundamentals: profitability, growth, financial health, "
            "valuation ratios, and dividends. "
            "Use when user asks about fundamentals, financials, or business health."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker":   {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "analyze_valuation",
        "description": (
            "Run four intrinsic value models: DCF, Owner Earnings, EV/EBITDA, Residual Income. "
            "Use when user asks about intrinsic value, over/undervalued, or DCF."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker":   {"type": "string"},
                "end_date": {"type": "string"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "deep_research_edgar",
        "description": (
            "Pull the latest 10-K or 10-Q SEC filing and extract key sections: "
            "business, risk factors, MD&A, financial statements. "
            "Use when user asks for deep research, risks, or management commentary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker":    {"type": "string"},
                "form_type": {"type": "string", "enum": ["10-K", "10-Q"], "default": "10-K"},
                "sections": {
                    "type": "array",
                    "items": {"type": "string",
                              "enum": ["business", "risk_factors", "mda", "financial_statements"]},
                },
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "get_full_analysis",
        "description": (
            "Run the COMPLETE analysis pipeline: all technical indicators, fundamentals, "
            "valuation models, and optionally EDGAR deep research. "
            "Returns final AI-generated buy/hold/sell verdict with reasoning. "
            "Use when user asks for a full analysis or 'should I buy X'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker":                {"type": "string"},
                "include_deep_research": {"type": "boolean", "default": False},
                "years":                 {"type": "integer", "default": 5},
            },
            "required": ["ticker"],
        },
    },
    # ── 三位一体工具 ──────────────────────────────────────────────────────────
    {
        "name": "trinity_analysis",
        "description": (
            "三位一体技术分析（Trinity Trading System）：基于均线、结构、时空三个维度深度分析股票。\n"
            "功能：\n"
            "  • 时空状态识别（极强/强/中性偏强/中性偏弱/极弱/弱）\n"
            "  • 结构分类（A五段式/B双平台/C单平台/D三段式）\n"
            "  • 主涨段确认与锁定状态判断\n"
            "  • 均线突破类型（A慢速/B有效/C回抽/D反向测试）\n"
            "  • 顶底背离检测\n"
            "  • 分层止盈建议\n"
            "使用场景：当用户问到技术形态、趋势状态、主涨段、"
            "应该加仓还是做T、止盈点位、均线突破类型时使用。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "股票代码，如 'AAPL', 'TSLA'",
                },
                "holding_days_min": {
                    "type": "integer",
                    "description": (
                        "交易限制：最少持有天数。"
                        "0=无限制，1=至少持有1天（默认），30=30天锁定期。"
                        "设置后会在止盈建议中加入相应提示。"
                    ),
                    "default": 1,
                },
            },
            "required": ["ticker"],
        },
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  TOOL DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool by name and return its JSON string result."""
    try:
        if tool_name == "get_stock_overview":
            ticker = tool_input["ticker"]
            info = get_ticker_info(ticker)
            try:
                df    = get_price_history(ticker, years=1)
                price    = round(float(df["Close"].iloc[-1]), 2)
                high_52w = round(float(df["High"].max()), 2)
                low_52w  = round(float(df["Low"].min()), 2)
                chg_1y   = round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2)
            except Exception:
                price = high_52w = low_52w = chg_1y = None
            return json.dumps({
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
            }, indent=2)

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
                "ticker": tool_input["ticker"], "price": result["price"],
                "as_of": result["as_of"], "indicator": ind,
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
            ticker = tool_input["ticker"]
            years  = tool_input.get("years", 5)
            deep   = tool_input.get("include_deep_research", False)
            tech   = run_technical_analysis(ticker, years=years)
            fund   = run_fundamental_analysis(ticker)
            valu   = run_valuation_analysis(ticker)
            edgar  = run_deep_research(ticker, form_type="10-K", sections=["mda"]) if deep else None
            result = run_full_analysis(
                ticker=ticker,
                technical_result=tech,
                fundamental_result=fund,
                valuation_result=valu,
                deep_research_result=edgar,
                use_ai=True,
            )
            return json.dumps(result, indent=2)

        # ── 三位一体 ──────────────────────────────────────────────────────────
        elif tool_name == "trinity_analysis":
            from tools.trinity.analysis import trinity_analysis
            result = trinity_analysis(
                ticker=tool_input["ticker"],
                holding_days_min=tool_input.get("holding_days_min", 1),
                client=None,  # 会从环境变量读取ANTHROPIC_API_KEY
            )
            return json.dumps(result, ensure_ascii=False, indent=2)

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
- Specific indicator question → analyze_single_indicator
- Multiple specific indicators → analyze_multiple_indicators
- "Full technical analysis" / "all indicators" → analyze_technicals
- Fundamentals / financials / earnings → analyze_fundamentals
- Intrinsic value / DCF / overvalued? → analyze_valuation
- 10-K / 10-Q / risk factors / deep research → deep_research_edgar
- "Full analysis" / "should I buy?" / final verdict → get_full_analysis
- Basic company info / price → get_stock_overview
- 技术形态 / 趋势状态 / 时空 / 主涨段 / 加仓还是做T / 止盈点位 / 均线突破类型 → trinity_analysis

You can call multiple tools in parallel when needed.

After receiving tool results, synthesize them into a clear, concise response. Always:
1. Lead with the signal (Buy / Hold / Sell) and confidence
2. Explain the key reasons in plain English
3. Mention backtest win rates for technical indicators
4. Flag any conflicting signals between indicators
5. End with a brief risk caveat

For trinity_analysis results, always mention:
- The time-space state (时空状态)
- Whether main wave is locked (主涨段是否锁定)
- The specific exit trigger (具体止盈触发条件)

Format your response using headers and bullet points where appropriate.
Do NOT dump raw JSON — always interpret the data.

CRITICAL: If a tool returns an "error" key, report the exact error. Do NOT make up data.

You support Chinese — if the user writes in Chinese, respond in Chinese.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 4.  CHATBOT CLASS
# ─────────────────────────────────────────────────────────────────────────────

class StockAnalystChatbot:
    def __init__(self, stream: bool = False):
        self.client  = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.history: list[dict] = []
        self.stream  = stream
        self.model   = "claude-haiku-4-5-20251001"

    def chat(self, user_message: str) -> tuple[str, dict]:
        print(f"\n👤 User: {user_message}")
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

            if response.stop_reason == "tool_use":
                self.history.append({"role": "assistant", "content": response.content})
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"  🔧 Calling tool: {block.name}({json.dumps(block.input)[:80]}...)")
                        result = dispatch_tool(block.name, block.input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                        try:
                            self._tool_data[block.name] = json.loads(result)
                        except Exception:
                            pass
                self.history.append({"role": "user", "content": tool_results})

            elif response.stop_reason == "end_turn":
                text = "".join(b.text for b in response.content if hasattr(b, "text"))
                preview = text[:300] + ("..." if len(text) > 300 else "")
                print(f"\n🤖 Claude: {preview}\n")
                self.history.append({"role": "assistant", "content": text})
                return text, self._tool_data

            else:
                return f"Unexpected stop reason: {response.stop_reason}", {}

    def reset(self):
        self.history = []


# ─────────────────────────────────────────────────────────────────────────────
# 5.  PYDANTIC MODELS (module-level for FastAPI)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from pydantic import BaseModel as _BaseModel

    class ChatRequest(_BaseModel):
        message: str
        session_id: str = "default"

except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FASTAPI BACKEND
# ─────────────────────────────────────────────────────────────────────────────

def create_api():
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
    except ImportError:
        raise ImportError("Run: pip install fastapi uvicorn")

    app = FastAPI(title="AlphaLens Stock Analyst API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
            return {"response": f"Error: {str(e)}", "tool_calls": tool_calls_made,
                    "tool_data": {}, "session_id": req.session_id}
        _self.dispatch_tool = original_dispatch

        return {
            "response":   response,
            "tool_calls": tool_calls_made,
            "tool_data":  tool_data,
            "session_id": req.session_id,
        }

    @app.delete("/chat/{session_id}")
    def reset_session(session_id: str):
        if session_id in sessions:
            sessions[session_id].reset()
        return {"status": "reset"}

    @app.get("/tools")
    def list_tools():
        return {"tools": [{"name": t["name"], "description": t["description"]}
                          for t in TOOL_SCHEMAS]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


# ─────────────────────────────────────────────────────────────────────────────
# 7.  CLI
# ─────────────────────────────────────────────────────────────────────────────

def run_cli():
    parser = argparse.ArgumentParser(description="AlphaLens Stock Analyst CLI")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    bot = StockAnalystChatbot(stream=args.stream)
    print("\n🤖 AlphaLens Stock Analyst")
    print("=" * 50)
    print("Examples:")
    print("  • Analyze AAPL using Bollinger and RSI")
    print("  • TSLA的三位一体时空状态是什么？")
    print("  • NVDA现在是主涨段吗？该加仓还是做T？")
    print("  • Run a full analysis on MSFT")
    print("  • 我有30天持仓限制，分析AAPL的止盈策略")
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
        response, _ = bot.chat(user_input)
        print(f"\nAssistant: {response}\n")


if __name__ == "__main__":
    run_cli()