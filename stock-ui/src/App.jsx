import { useState, useRef, useEffect } from "react";

const API_BASE = "http://localhost:8000";

// ─── Colour helpers ──────────────────────────────────────────────────────────
const SIGNAL_COLORS = {
  buy:     { bg: "bg-emerald-500/15", border: "border-emerald-500/40", text: "text-emerald-400", dot: "bg-emerald-400" },
  bullish: { bg: "bg-emerald-500/15", border: "border-emerald-500/40", text: "text-emerald-400", dot: "bg-emerald-400" },
  hold:    { bg: "bg-amber-500/15",   border: "border-amber-500/40",   text: "text-amber-400",   dot: "bg-amber-400"   },
  neutral: { bg: "bg-amber-500/15",   border: "border-amber-500/40",   text: "text-amber-400",   dot: "bg-amber-400"   },
  sell:    { bg: "bg-red-500/15",     border: "border-red-500/40",     text: "text-red-400",     dot: "bg-red-400"     },
  bearish: { bg: "bg-red-500/15",     border: "border-red-500/40",     text: "text-red-400",     dot: "bg-red-400"     },
};
const sigColors = (s) => SIGNAL_COLORS[(s || "").toLowerCase()] || SIGNAL_COLORS.neutral;
const sigLabel  = (s) => (s || "neutral").toUpperCase();

const VERDICT_CONFIG = {
  BUY:  { gradient: "from-emerald-600 to-teal-700",  icon: "↑" },
  HOLD: { gradient: "from-amber-600  to-orange-700", icon: "→" },
  SELL: { gradient: "from-red-600    to-rose-700",   icon: "↓" },
};

const TOOL_LABELS = {
  get_stock_overview:          "📊 Company Overview",
  analyze_technicals:          "📈 Full Technical Analysis",
  analyze_single_indicator:    "📉 Single Indicator",
  analyze_multiple_indicators: "📉 Multiple Indicators",
  analyze_fundamentals:        "📋 Fundamentals",
  analyze_valuation:           "💰 Valuation Models",
  deep_research_edgar:         "📄 EDGAR Deep Research",
  get_full_analysis:           "🔬 Full Analysis",
};

// ─── Sub-components ──────────────────────────────────────────────────────────

function SignalBadge({ signal, size = "sm" }) {
  const c = sigColors(signal);
  const sz = size === "lg"
    ? "px-3 py-1 text-sm font-bold"
    : "px-2 py-0.5 text-xs font-semibold";
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border ${c.bg} ${c.border} ${c.text} ${sz}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${c.dot}`} />
      {sigLabel(signal)}
    </span>
  );
}

function ToolPill({ name }) {
  return (
    <span className="inline-flex items-center gap-1 bg-blue-500/15 border border-blue-500/30 text-blue-300 text-xs px-2 py-0.5 rounded-full">
      {TOOL_LABELS[name] || name}
    </span>
  );
}

function IndicatorCard({ name, data }) {
  const c = sigColors(data.signal);
  const isTA = data.backtest_win_rate_pct !== undefined;
  const gapColor = data.gap_pct > 0 ? "text-emerald-400"
                 : data.gap_pct < 0 ? "text-red-400"
                 : "text-gray-400";
  return (
    <div className={`rounded-xl border ${c.border} ${c.bg} p-3 flex flex-col gap-2`}>
      <div className="flex items-start justify-between gap-2">
        <span className="text-white/80 text-xs font-medium leading-tight">{name}</span>
        <SignalBadge signal={data.signal} />
      </div>
      {isTA && (
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-white/50">
            <span>5Y Win Rate</span>
            <span className={c.text}>{data.backtest_win_rate_pct}%</span>
          </div>
          <div className="w-full h-1 bg-white/10 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full ${
                data.backtest_win_rate_pct >= 60 ? "bg-emerald-500"
              : data.backtest_win_rate_pct >= 50 ? "bg-amber-500"
              : "bg-red-500"}`}
              style={{ width: `${data.backtest_win_rate_pct}%` }}
            />
          </div>
          <p className="text-xs text-white/25">{data.backtest_trades} trades</p>
        </div>
      )}
      {data.gap_pct !== undefined && (
        <div className="flex justify-between text-xs">
          <span className="text-white/40">Gap</span>
          <span className={`font-semibold ${gapColor}`}>
            {data.gap_pct > 0 ? "+" : ""}{data.gap_pct}%
          </span>
        </div>
      )}
    </div>
  );
}

function VerdictCard({ data }) {
  if (!data) return null;
  const cfg = VERDICT_CONFIG[data.ai_verdict] || VERDICT_CONFIG.HOLD;
  return (
    <div className={`rounded-2xl bg-gradient-to-r ${cfg.gradient} p-5 space-y-4`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span className="text-4xl font-black">{cfg.icon} {data.ai_verdict}</span>
          <div>
            <p className="text-white/60 text-xs">AI Confidence</p>
            <p className="text-xl font-bold">{data.ai_confidence}%</p>
          </div>
        </div>
        <div className="text-right text-xs text-white/50">
          <p className="font-mono font-bold">{data.ticker}</p>
          <p>${data.price} · {data.as_of}</p>
        </div>
      </div>

      <div className="w-full h-1.5 bg-white/20 rounded-full">
        <div className="h-full bg-white/50 rounded-full transition-all"
          style={{ width: `${data.ai_confidence}%` }} />
      </div>

      {data.reasoning && (
        <p className="text-white/85 text-sm leading-relaxed border-t border-white/20 pt-3">
          {data.reasoning}
        </p>
      )}

      {(data.supporting_arguments?.length > 0 || data.key_risks?.length > 0) && (
        <div className="grid grid-cols-2 gap-4 pt-1">
          <div>
            <p className="text-white/50 text-xs uppercase tracking-wider mb-2">Supporting</p>
            <ul className="space-y-1.5">
              {data.supporting_arguments?.map((a, i) => (
                <li key={i} className="flex gap-1.5 text-xs text-white/75 leading-relaxed">
                  <span className="text-emerald-300 flex-shrink-0 mt-0.5">✓</span>{a}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <p className="text-white/50 text-xs uppercase tracking-wider mb-2">Key Risks</p>
            <ul className="space-y-1.5">
              {data.key_risks?.map((r, i) => (
                <li key={i} className="flex gap-1.5 text-xs text-white/75 leading-relaxed">
                  <span className="text-red-300 flex-shrink-0 mt-0.5">⚠</span>{r}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}

      {data.indicator_breakdown && Object.keys(data.indicator_breakdown).length > 0 && (
        <div className="border-t border-white/20 pt-3">
          <p className="text-white/50 text-xs uppercase tracking-wider mb-2">Signal Dashboard</p>
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(data.indicator_breakdown).map(([name, d]) => (
              <IndicatorCard
                key={name}
                name={name.replace(/^(Fundamental|Valuation): /, "")}
                data={d}
              />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Markdown-lite renderer ───────────────────────────────────────────────────
// Renders **bold**, `code`, ## headers, and bullet points from Claude's text

function MarkdownText({ text }) {
  const lines = text.split("\n");
  return (
    <div className="space-y-1">
      {lines.map((line, i) => {
        if (!line.trim()) return <div key={i} className="h-1" />;

        // ## header
        if (line.startsWith("## ")) {
          return <p key={i} className="font-semibold text-white/95 mt-2">{line.slice(3)}</p>;
        }
        // ### header
        if (line.startsWith("### ")) {
          return <p key={i} className="font-medium text-white/80 mt-1 text-xs uppercase tracking-wider">{line.slice(4)}</p>;
        }
        // Bullet
        if (line.startsWith("- ") || line.startsWith("• ")) {
          const content = line.slice(2);
          return (
            <div key={i} className="flex gap-2">
              <span className="text-white/30 flex-shrink-0">•</span>
              <span>{renderInline(content)}</span>
            </div>
          );
        }
        // Numbered
        const numMatch = line.match(/^(\d+)\.\s(.+)/);
        if (numMatch) {
          return (
            <div key={i} className="flex gap-2">
              <span className="text-white/40 flex-shrink-0 font-mono text-xs mt-0.5">{numMatch[1]}.</span>
              <span>{renderInline(numMatch[2])}</span>
            </div>
          );
        }
        return <p key={i}>{renderInline(line)}</p>;
      })}
    </div>
  );
}

function renderInline(text) {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={i} className="font-semibold text-white">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={i} className="bg-white/10 text-blue-300 px-1 rounded text-xs font-mono">{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

// ─── Message bubble ───────────────────────────────────────────────────────────

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} gap-3`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-xs font-bold flex-shrink-0 mt-1">
          S
        </div>
      )}
      <div className={`max-w-[82%] space-y-2 flex flex-col ${isUser ? "items-end" : "items-start"}`}>
        {msg.tool_calls?.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {msg.tool_calls.map((t, i) => <ToolPill key={i} name={t} />)}
          </div>
        )}
        <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
          isUser
            ? "bg-blue-600 text-white rounded-tr-sm"
            : "bg-white/8 border border-white/10 text-white/85 rounded-tl-sm w-full"
        }`}>
          {isUser ? msg.content : <MarkdownText text={msg.content} />}
        </div>
        {msg.verdict && <VerdictCard data={msg.verdict} />}
      </div>
    </div>
  );
}

function TypingIndicator({ toolsInProgress }) {
  return (
    <div className="flex gap-3 items-start">
      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-xs font-bold flex-shrink-0">
        S
      </div>
      <div className="bg-white/8 border border-white/10 rounded-2xl rounded-tl-sm px-4 py-3 min-w-32">
        {toolsInProgress.length > 0 ? (
          <div className="space-y-1.5">
            {toolsInProgress.map((t, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-white/50">
                <div className="w-3 h-3 border border-blue-400 border-t-transparent rounded-full animate-spin" />
                {TOOL_LABELS[t] || t}
              </div>
            ))}
          </div>
        ) : (
          <div className="flex gap-1 items-center h-4">
            {[0, 1, 2].map(i => (
              <div key={i} className="w-1.5 h-1.5 bg-white/40 rounded-full animate-bounce"
                style={{ animationDelay: `${i * 0.15}s` }} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

const QUICK_ACTIONS = [
  "Full analysis on AAPL 📊",
  "Bollinger + RSI for TSLA 📉",
  "Is NVDA overvalued? (DCF) 💰",
  "MSFT fundamentals 📋",
  "AAPL 10-K risk factors 📄",
];

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function StockAnalystChat() {
  const [messages, setMessages]               = useState([{
    role: "assistant",
    content: "Hi! I'm your AI stock analyst powered by Claude.\n\nI can run **technical indicators** (Bollinger, RSI, MACD, SMA, EMA) with 5-year backtests, **fundamental analysis**, **valuation models** (DCF, Owner Earnings), and pull **SEC filings** from EDGAR.\n\nWhat stock would you like to analyze?",
  }]);
  const [input, setInput]                     = useState("");
  const [loading, setLoading]                 = useState(false);
  const [toolsInProgress, setToolsInProgress] = useState([]);
  const [sessionId]                           = useState(() => crypto.randomUUID());
  const [apiStatus, setApiStatus]             = useState("checking");
  const messagesEndRef                        = useRef(null);
  const textareaRef                           = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.ok ? setApiStatus("ok") : setApiStatus("error"))
      .catch(() => setApiStatus("error"));
  }, []);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "48px";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  const sendMessage = async (text) => {
    const userText = (text || input).trim();
    if (!userText || loading) return;
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userText }]);
    setLoading(true);
    setToolsInProgress([]);

    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userText, session_id: sessionId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      setMessages(prev => [...prev, {
        role: "assistant",
        content: data.response,
        tool_calls: data.tool_calls || [],
        verdict: data.verdict || null,
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: "assistant",
        content: `⚠️ **Backend error:** ${err.message}\n\nMake sure the server is running:\n\`uvicorn chatbot:create_api --factory --reload\``,
      }]);
    } finally {
      setLoading(false);
      setToolsInProgress([]);
    }
  };

  const showQuickActions = messages.length === 1;

  return (
    <div className="min-h-screen bg-[#0d1117] text-white flex flex-col" style={{ fontFamily: "system-ui,sans-serif" }}>

      {/* Header */}
      <div className="border-b border-white/8 px-6 py-3 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-sm font-bold">S</div>
          <div className="leading-tight">
            <p className="font-semibold text-white/90 text-sm">Stock Analyst</p>
            <p className="text-white/30 text-xs">Claude claude-sonnet-4-6 · MCP tool-calling</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${
            apiStatus === "ok"       ? "bg-emerald-400"
          : apiStatus === "error"   ? "bg-red-400"
          : "bg-amber-400 animate-pulse"}`} />
          <span className="text-white/30 text-xs">
            {apiStatus === "ok" ? "Connected" : apiStatus === "error" ? "Backend offline" : "Connecting…"}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-6 space-y-6">
          {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
          {loading && <TypingIndicator toolsInProgress={toolsInProgress} />}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Quick actions */}
      {showQuickActions && (
        <div className="max-w-3xl mx-auto px-6 pb-3 w-full">
          <div className="flex flex-wrap gap-2">
            {QUICK_ACTIONS.map((a, i) => (
              <button key={i} onClick={() => sendMessage(a.replace(/ [^\s]+$/, ""))}
                className="text-xs bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 rounded-full px-3 py-1.5 text-white/55 hover:text-white/90 transition-all">
                {a}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="border-t border-white/8 px-6 py-4 flex-shrink-0">
        <div className="max-w-3xl mx-auto flex gap-3 items-end">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); } }}
            placeholder="Ask about any stock… e.g. 'Analyze AAPL using Bollinger and RSI'"
            className="flex-1 bg-[#161b22] border border-white/15 focus:border-blue-400/60 rounded-xl px-4 py-3 text-sm text-white placeholder-white/40 resize-none focus:outline-none transition-colors"
            style={{ minHeight: "48px", maxHeight: "120px" }}
          />
          <button
            onClick={() => sendMessage()}
            disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-35 disabled:cursor-not-allowed rounded-xl px-5 py-3 text-sm font-semibold transition-colors flex-shrink-0"
          >
            {loading ? "…" : "↑"}
          </button>
        </div>
        <p className="text-white/15 text-xs mt-2 text-center">
          For informational purposes only. Not financial advice.
        </p>
      </div>
    </div>
  );
}