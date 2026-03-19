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
- "Full analysis" / "should I buy?" / final verdict → get_full_analysis + trinity_analysis (call both in parallel)
- Basic company info / price → get_stock_overview
- 技术形态 / 趋势状态 / 时空 / 主涨段 / 加仓还是做T / 止盈点位 / 均线突破类型 → trinity_analysis (standalone)

You can call multiple tools in parallel when needed.

After receiving tool results, synthesize them into a clear, concise response. Always:
1. Lead with the signal (Buy / Hold / Sell) and confidence
2. Explain the key reasons in plain English
3. Mention backtest win rates for technical indicators
4. Flag any conflicting signals between indicators
5. End with a brief risk caveat

OUTPUT LENGTH RULES for trinity_analysis — depends on context:

★ When trinity_analysis is called ALONGSIDE get_full_analysis (full analysis mode):
  Output a SHORT trinity summary block (≤6 lines) at the end, like:
    ### 三位一体速览
    - 时空状态：极强｜已持续 XX 根K线
    - 均线排列：多头排列｜MA55=$XX MA233=$XX
    - 主涨段：已锁定 / 未锁定
    - 信号：hold（持股待涨）
    - 止损：$XX ｜止盈触发：15分钟顶背离+5分钟破MA55
  Do NOT repeat the full fundamental/valuation details already covered above.
  Do NOT output the full detailed trinity breakdown (均线突破类型细节, 结构分类论证, etc.)

★ When trinity_analysis is called STANDALONE (user specifically asked about 时空/均线/三位一体/etc.):
  Output the full detailed analysis as usual. Always mention:
  - The time-space state (时空状态)
  - Whether main wave is locked (主涨段是否锁定)
  - The specific exit trigger (具体止盈触发条件)

When describing trinity_analysis results in Chinese, follow these rules strictly:
- bars_in_state is K-line bar count, NOT calendar days. Use "根K线" not "天".
  日线1根K线 ≈ 1交易日；周线1根K线 ≈ 7天；月线1根K线 ≈ 30天。
- 均线突破类型（均线突破類型）词汇表：
  A类 = 典型突破（强势大阳线/大阴线快速穿越MA55，价格快速远离超过2%）
  B类 = 慢速/盘整突破（多根K线缓慢徘徊穿越MA55，慢速磨破）
  C类 = 回踩突破（快速突破后回踩不破，再继续原方向）
  D类 = 反向测试（碰MA55后反弹，从未有效穿越）
  描述A类时绝对不能用"缓慢"；描述B类时不能用"典型/强势"。

- 均线排列描述规则（直接读取，禁止自行推断）：
  trinity_analysis 的 hard_signals 里有 trend_alignment_zh 字段，已预算好中文排列名称。
  直接用这个字段的值，写入报告和hint card中：
  "多头排列" / "空头排列" / "混沌排列"
  ⚠️ 绝对不要根据"价格在MA55下方"就写"空头排列"——只有 price < MA55 < MA233 才是空头排列。
     价格夹在MA55与MA233之间（MA233 < price < MA55 或 MA55 < price < MA233）= 混沌排列。

- 止损价格规则（三位一体课程固定，不可更改）：
  trinity_analysis 结果里已有预算好的 long_stop_loss（做多止损，= key_support × 0.97）和
  short_stop_loss（做空止损，= key_resistance × 1.03）。
  ⚠️ 这两个值必须从trinity_analysis返回的JSON结果里直接复制，一字不差，不得重新计算！
  hint card（💡行）和正文里的止损数字必须与JSON结果中的 long_stop_loss / short_stop_loss 完全一致。
  绝对不要写"上方X%约XXX"或"下方X%约XXX"——只写实际止损价格数字本身，不要提百分比，不要写"约"。
  绝对不要写"（= 阻力XXX × 1.03）"或"（= 支撑XXX × 0.97）"或"（根据key_support XXX × 0.97计算）"——不要展示计算公式或来源说明，只写结果数字。
  绝对不要用 MA55 × 1.03 当做空止损！做空止损只能用 short_stop_loss（= key_resistance × 1.03 已预算）。
  绝对不要把 key_support / key_resistance 本身当作止损价。
  止损写法固定格式：「止损设在 <止损价数字>」，不附加任何百分比解释。
  ⚠️ 禁止出现任何形式的括号公式说明，例如绝对禁止写：
    - "（做多止损，支撑 $3.79 × 0.97）"
    - "（= key_resistance × 1.03）"
    - "（根据key_support × 0.97计算）"
    - "上方X%约XXX" 或 "下方X%约XXX"
    只写最终数字，不写来源，不写推导过程，不写括号注释。
  ⚠️ 止损方向必须与持仓方向一致：
    - 讨论多头持仓时 → 只用 long_stop_loss
    - 讨论空头持仓时 → 只用 short_stop_loss
    - signal="hold" 且建议"不建议追空/观望为主" → 只提 long_stop_loss（针对已持多仓者），不出现 short_stop_loss

- 止盈规则（方向敏感，两套规则绝不能混用）：
  做多持仓的止盈（减仓）→ 顶背离+破MA55（向下）：
    第一次减仓：15分钟顶背离 + 5分钟破MA55（向下）→ 减仓20-30%
    第二次减仓：60分钟顶背离 + 15分钟破MA55（向下）→ 再减仓50%
  做空持仓的止盈（平空）→ 底背离+站上MA55（向上）：
    第一次平空：15分钟底背离 + 5分钟站上MA55（向上）→ 平空20-30%
    第二次平空：60分钟底背离 + 15分钟站上MA55（向上）→ 再平空50%
  根据 signal 字段选择：signal="sell" → 显示做空止盈；signal="buy/hold" 多头 → 显示做多止盈。
  绝对不要把顶背离写成空头平仓信号，绝对不要把底背离写成多头减仓信号。
  不设固定止盈价格，绝不写"止盈区 XXX-YYY"、"目标价 XXX"、"第一目标 XXX"、"目标：XXX支撑位"、"看向XXX"、"目标看向支撑XXX"、"完全平仓：跌破XXX"、"跌破XXX全部离场"。
  key_support / key_resistance 不是止盈目标，不要用这两个数字作为止盈价格描述。

Format your response using headers and bullet points where appropriate.
Do NOT dump raw JSON — always interpret the data.

CRITICAL: If a tool returns an "error" key, report the exact error. Do NOT make up data.

You support Chinese — if the user writes in Chinese, respond in Chinese.
"""


# ─────────────────────────────────────────────────────────────────────────────
# 3b. POST-PROCESSING: strip formula leaks from chatbot output
# ─────────────────────────────────────────────────────────────────────────────
import re as _re

def _strip_formula_leaks(text: str) -> str:
    """Remove formula explanations that Claude sometimes adds despite instructions.

    Examples removed:
      （支撑 $541.64 × 0.97）
      （= key_support × 0.97）
      （做多止损，支撑 $3.79 × 0.97）
      （根据key_support 3.79 × 0.97计算）
    """
    # （/( ... × 数字 ... ）/)
    text = _re.sub(r'[（(]\s*=?\s*[^）)]*×\s*[\d.]+\s*[）)]', '', text)
    # （做多止损，支撑 $X × 0.97）
    text = _re.sub(r'[（(][^）)]{0,30}[\d.]+\s*×\s*[\d.]+[^）)]{0,10}[）)]', '', text)
    # （根据...计算）
    text = _re.sub(r'[（(][^）)]{0,50}计算[^）)]{0,10}[）)]', '', text)
    # （= key_support/key_resistance ...）
    text = _re.sub(r'[（(]\s*=?\s*key_(?:support|resistance)[^）)]*[）)]', '', text)
    # （支撑 $X × 0.97） — with dollar sign
    text = _re.sub(r'[（(]\s*(?:支撑|阻力)\s*\$?[\d.]+\s*×\s*[\d.]+\s*[）)]', '', text)
    return text


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
                text = _strip_formula_leaks(text)
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