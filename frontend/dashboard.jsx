import { useState, useRef, useEffect } from "react";
import {
  AreaChart, Area, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  Cell
} from "recharts";

const API_BASE = "http://localhost:8000";

// ─── Colours ──────────────────────────────────────────────────────────────────
const SIG = {
  buy:     { bg: "bg-emerald-500/15", border: "border-emerald-500/40", text: "text-emerald-400", dot: "bg-emerald-400", hex: "#10b981" },
  bullish: { bg: "bg-emerald-500/15", border: "border-emerald-500/40", text: "text-emerald-400", dot: "bg-emerald-400", hex: "#10b981" },
  hold:    { bg: "bg-amber-500/15",   border: "border-amber-500/40",   text: "text-amber-400",   dot: "bg-amber-400",   hex: "#f59e0b" },
  neutral: { bg: "bg-amber-500/15",   border: "border-amber-500/40",   text: "text-amber-400",   dot: "bg-amber-400",   hex: "#f59e0b" },
  sell:    { bg: "bg-red-500/15",     border: "border-red-500/40",     text: "text-red-400",     dot: "bg-red-400",     hex: "#ef4444" },
  bearish: { bg: "bg-red-500/15",     border: "border-red-500/40",     text: "text-red-400",     dot: "bg-red-400",     hex: "#ef4444" },
};
const sc = (s) => SIG[(s || "").toLowerCase()] || SIG.neutral;
const sl = (s) => (s || "neutral").toUpperCase();

const VERDICT_CONFIG = {
  BUY:  { gradient: "from-emerald-600 to-teal-700",  icon: "↑" },
  HOLD: { gradient: "from-amber-600  to-orange-700", icon: "→" },
  SELL: { gradient: "from-red-600    to-rose-700",   icon: "↓" },
};

const TOOL_LABELS = {
  get_stock_overview:          "📊 Overview",
  analyze_technicals:          "📈 Full Technical",
  analyze_single_indicator:    "📉 Indicator",
  analyze_multiple_indicators: "📉 Indicators",
  analyze_fundamentals:        "📋 Fundamentals",
  analyze_valuation:           "💰 Valuation",
  deep_research_edgar:         "📄 EDGAR",
  get_full_analysis:           "🔬 Full Analysis",
};

const GRID  = { stroke: "rgba(255,255,255,0.06)" };
const ATICK = { fill: "rgba(255,255,255,0.3)", fontSize: 10 };

// ─── Shared tooltip ───────────────────────────────────────────────────────────
const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1a1f2e] border border-white/15 rounded-lg px-3 py-2 text-xs shadow-xl">
      {label && <p className="text-white/40 mb-1">{label}</p>}
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || "#fff" }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
        </p>
      ))}
    </div>
  );
};

// ─── Section wrapper ──────────────────────────────────────────────────────────
const Section = ({ title, children }) => (
  <div className="bg-white/4 border border-white/8 rounded-xl p-4 space-y-3">
    <p className="text-white/50 text-xs font-semibold uppercase tracking-wider">{title}</p>
    {children}
  </div>
);

// ─── Signal grid ──────────────────────────────────────────────────────────────
function SignalGrid({ indicators }) {
  const entries = Object.entries(indicators);
  if (!entries.length) return null;
  return (
    <Section title="Signal Summary">
      <div className="grid grid-cols-2 gap-2">
        {entries.map(([key, val]) => {
          const c = sc(val.signal);
          return (
            <div key={key} className={`flex items-center justify-between rounded-lg px-3 py-2 border ${c.border} ${c.bg}`}>
              <span className="text-white/70 text-xs truncate mr-2">{val.name || key}</span>
              <span className={`text-xs font-bold flex-shrink-0 ${c.text}`}>{sl(val.signal)}</span>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ─── Win-rate bar chart ───────────────────────────────────────────────────────
function WinRateChart({ indicators }) {
  const data = Object.entries(indicators)
    .filter(([, v]) => v.backtest?.["win_rate_%"] !== undefined)
    .map(([key, val]) => ({
      name: (val.name || key).replace(/ \d+.*/, ""),
      winRate: val.backtest["win_rate_%"] || 0,
      signal: val.signal,
      trades: val.backtest.n_trades || 0,
    }));
  if (!data.length) return null;

  return (
    <Section title="5-Year Backtest Win Rates">
      <ResponsiveContainer width="100%" height={Math.max(90, data.length * 38)}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 40 }}>
          <CartesianGrid {...GRID} horizontal={false} />
          <XAxis type="number" domain={[0, 100]} tick={ATICK} tickFormatter={v => `${v}%`} />
          <YAxis type="category" dataKey="name" tick={ATICK} width={85} />
          <Tooltip content={<Tip />} formatter={v => [`${v}%`, "Win Rate"]} />
          <ReferenceLine x={50} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4" />
          <Bar dataKey="winRate" radius={[0, 4, 4, 0]} name="Win Rate">
            {data.map((d, i) => <Cell key={i} fill={sc(d.signal).hex} fillOpacity={0.8} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-3 pt-1 border-t border-white/8">
        {data.map((d, i) => (
          <span key={i} className="text-xs text-white/30">{d.name}: <span className="text-white/50">{d.trades} trades</span></span>
        ))}
      </div>
    </Section>
  );
}

// ─── Equity curve ─────────────────────────────────────────────────────────────
function EquityCurve({ trades, name, winRate, totalReturn, buyHold }) {
  if (!trades?.length) return null;

  let equity = 10000;
  const data = [{ i: 0, v: 10000 }];
  trades.forEach((t, idx) => {
    equity *= 1 + t.pct_return / 100;
    data.push({ i: idx + 1, v: Math.round(equity) });
  });

  const col = totalReturn >= 0 ? "#10b981" : "#ef4444";
  const id  = `eq_${name.replace(/\s/g, "_")}`;

  return (
    <Section title={`${name} — Equity Curve`}>
      <div className="flex gap-4 text-xs mb-1">
        <span style={{ color: col }}>Strategy: {totalReturn > 0 ? "+" : ""}{totalReturn?.toFixed(1)}%</span>
        <span className="text-indigo-400">B&H: {buyHold > 0 ? "+" : ""}{buyHold?.toFixed(1)}%</span>
        <span className="text-white/40">Win Rate: <span className="text-white/70">{winRate?.toFixed(1)}%</span></span>
        <span className="text-white/40">Trades: <span className="text-white/70">{trades.length}</span></span>
      </div>
      <ResponsiveContainer width="100%" height={130}>
        <AreaChart data={data}>
          <defs>
            <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%"  stopColor={col} stopOpacity={0.3} />
              <stop offset="95%" stopColor={col} stopOpacity={0.02} />
            </linearGradient>
          </defs>
          <CartesianGrid {...GRID} />
          <XAxis dataKey="i" tick={ATICK} label={{ value: "Trade #", fill: "rgba(255,255,255,0.2)", fontSize: 9, position: "insideBottomRight", offset: -4 }} />
          <YAxis tick={ATICK} tickFormatter={v => `$${(v / 1000).toFixed(1)}k`} />
          <Tooltip content={<Tip />} formatter={v => [`$${v.toLocaleString()}`, "Portfolio"]} labelFormatter={l => `Trade #${l}`} />
          <ReferenceLine y={10000} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
          <Area type="monotone" dataKey="v" stroke={col} fill={`url(#${id})`} strokeWidth={1.5} dot={false} name="Portfolio" />
        </AreaChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ─── Fundamentals radar ───────────────────────────────────────────────────────
function FundamentalsRadar({ sections }) {
  const scoreMap = { bullish: 90, neutral: 50, bearish: 15 };
  const data = Object.entries(sections).map(([key, val]) => ({
    subject: key.charAt(0).toUpperCase() + key.slice(1),
    score: scoreMap[val.signal] ?? 50,
    signal: val.signal,
  }));
  if (!data.length) return null;

  return (
    <Section title="Fundamentals Radar">
      <ResponsiveContainer width="100%" height={200}>
        <RadarChart data={data}>
          <PolarGrid stroke="rgba(255,255,255,0.1)" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 10 }} />
          <Radar dataKey="score" stroke="#818cf8" fill="#818cf8" fillOpacity={0.2} strokeWidth={1.5} name="Score" />
          <Tooltip content={<Tip />} formatter={v => [v >= 80 ? "Bullish" : v >= 40 ? "Neutral" : "Bearish", "Signal"]} />
        </RadarChart>
      </ResponsiveContainer>
      <div className="space-y-1.5 border-t border-white/8 pt-2">
        {Object.entries(sections).map(([key, val]) => {
          const c = sc(val.signal);
          return (
            <div key={key} className="flex gap-2 text-xs">
              <span className={`font-semibold flex-shrink-0 w-20 ${c.text}`}>{sl(val.signal)}</span>
              <span className="text-white/40">{val.details}</span>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

// ─── Valuation gap chart ──────────────────────────────────────────────────────
function ValuationGapChart({ methods, weightedGap }) {
  const data = Object.entries(methods)
    .filter(([, v]) => v.gap_pct !== null && v.gap_pct !== undefined)
    .map(([key, val]) => ({
      name: key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase()),
      gap: val.gap_pct,
      signal: val.signal,
    }));
  if (!data.length) return null;

  return (
    <Section title="Intrinsic Value Gap vs Market Cap">
      <div className="flex justify-between items-center">
        <p className="text-white/30 text-xs">+gap = undervalued, −gap = overvalued (±15% threshold)</p>
        <span className={`text-sm font-bold ${weightedGap > 0 ? "text-emerald-400" : "text-red-400"}`}>
          Weighted: {weightedGap > 0 ? "+" : ""}{weightedGap?.toFixed(1)}%
        </span>
      </div>
      <ResponsiveContainer width="100%" height={Math.max(110, data.length * 44)}>
        <BarChart data={data} layout="vertical" margin={{ left: 8, right: 50 }}>
          <CartesianGrid {...GRID} horizontal={false} />
          <XAxis type="number" tick={ATICK} tickFormatter={v => `${v > 0 ? "+" : ""}${v}%`} />
          <YAxis type="category" dataKey="name" tick={ATICK} width={115} />
          <Tooltip content={<Tip />} formatter={v => [`${v > 0 ? "+" : ""}${Number(v).toFixed(1)}%`, "Gap"]} />
          <ReferenceLine x={0}    stroke="rgba(255,255,255,0.25)" />
          <ReferenceLine x={15}   stroke="#10b981" strokeDasharray="3 3" strokeOpacity={0.5} />
          <ReferenceLine x={-15}  stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.5} />
          <Bar dataKey="gap" radius={[0, 4, 4, 0]} name="Gap %">
            {data.map((d, i) => (
              <Cell key={i} fill={d.gap > 15 ? "#10b981" : d.gap < -15 ? "#ef4444" : "#f59e0b"} fillOpacity={0.8} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ─── Stock overview card ──────────────────────────────────────────────────────
function OverviewCard({ data }) {
  const chg = data["price_change_1y_%"] ?? data.price_change_1y_pct;
  const isUp = chg >= 0;
  return (
    <Section title="Company Overview">
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-lg font-bold text-white">{data.name}</p>
          <p className="text-white/40 text-xs mt-0.5">{data.sector} · {data.industry}</p>
        </div>
        <div className="text-right flex-shrink-0">
          <p className="text-xl font-bold text-white">${data.price}</p>
          <p className={`text-xs font-semibold ${isUp ? "text-emerald-400" : "text-red-400"}`}>
            {isUp ? "▲" : "▼"} {Math.abs(chg)?.toFixed(1)}% (1Y)
          </p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3 border-t border-white/8 pt-3">
        {[
          ["52W High",   `$${data["52w_high"]}`],
          ["52W Low",    `$${data["52w_low"]}`],
          ["Market Cap", data.market_cap ? `$${(data.market_cap / 1e9).toFixed(1)}B` : "N/A"],
        ].map(([label, val]) => (
          <div key={label}>
            <p className="text-white/30 text-xs">{label}</p>
            <p className="text-white/80 text-sm font-semibold">{val}</p>
          </div>
        ))}
      </div>
      {data.description && (
        <p className="text-white/35 text-xs leading-relaxed border-t border-white/8 pt-2">{data.description}</p>
      )}
    </Section>
  );
}

// ─── Full analysis verdict ────────────────────────────────────────────────────
function VerdictCard({ data }) {
  if (!data?.ai_verdict) return null;
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
          <p className="font-mono font-bold text-white/80">{data.ticker}</p>
          <p>${data.price} · {data.as_of}</p>
        </div>
      </div>
      <div className="w-full h-1.5 bg-white/20 rounded-full">
        <div className="h-full bg-white/50 rounded-full" style={{ width: `${data.ai_confidence}%` }} />
      </div>
      {data.reasoning && (
        <p className="text-white/85 text-sm leading-relaxed border-t border-white/20 pt-3">{data.reasoning}</p>
      )}
      {(data.supporting_arguments?.length > 0 || data.key_risks?.length > 0) && (
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-white/50 text-xs uppercase tracking-wider mb-2">Supporting</p>
            <ul className="space-y-1.5">
              {data.supporting_arguments?.map((a, i) => (
                <li key={i} className="flex gap-1.5 text-xs text-white/75 leading-relaxed">
                  <span className="text-emerald-300 flex-shrink-0">✓</span>{a}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <p className="text-white/50 text-xs uppercase tracking-wider mb-2">Key Risks</p>
            <ul className="space-y-1.5">
              {data.key_risks?.map((r, i) => (
                <li key={i} className="flex gap-1.5 text-xs text-white/75 leading-relaxed">
                  <span className="text-red-300 flex-shrink-0">⚠</span>{r}
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Tool visualisation dispatcher ───────────────────────────────────────────
function ToolViz({ toolName, data }) {
  if (!data) return null;

  if (["analyze_technicals", "analyze_single_indicator", "analyze_multiple_indicators"].includes(toolName)) {
    const indicators = data.indicators || {};
    if (!Object.keys(indicators).length) return null;
    return (
      <div className="space-y-3">
        <SignalGrid indicators={indicators} />
        <WinRateChart indicators={indicators} />
        {Object.entries(indicators).map(([key, val]) =>
          val.backtest?.trades?.length > 0 ? (
            <EquityCurve
              key={key}
              trades={val.backtest.trades}
              name={val.name || key}
              winRate={val.backtest["win_rate_%"]}
              totalReturn={val.backtest["total_return_%"]}
              buyHold={val.backtest["buy_hold_%"]}
            />
          ) : null
        )}
      </div>
    );
  }

  if (toolName === "analyze_fundamentals") {
    const sections = data.sections || {};
    if (!Object.keys(sections).length) return null;
    return <FundamentalsRadar sections={sections} />;
  }

  if (toolName === "analyze_valuation") {
    const methods = data.methods || {};
    if (!Object.keys(methods).length) return null;
    return <ValuationGapChart methods={methods} weightedGap={data["weighted_gap_%"]} />;
  }

  if (toolName === "get_stock_overview" && data.name) {
    return <OverviewCard data={data} />;
  }

  if (toolName === "get_full_analysis") {
    return (
      <div className="space-y-3">
        <VerdictCard data={data} />
        {data.indicator_breakdown && (
          <WinRateChart indicators={
            Object.fromEntries(
              Object.entries(data.indicator_breakdown)
                .filter(([, v]) => v.backtest_win_rate_pct !== undefined)
                .map(([k, v]) => [k, {
                  name: k,
                  signal: v.signal,
                  backtest: { "win_rate_%": v.backtest_win_rate_pct, n_trades: v.backtest_trades },
                }])
            )
          } />
        )}
      </div>
    );
  }

  return null;
}

// ─── Markdown renderer ────────────────────────────────────────────────────────
function renderInline(text) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/).map((p, i) => {
    if (p.startsWith("**") && p.endsWith("**")) return <strong key={i} className="font-semibold text-white">{p.slice(2, -2)}</strong>;
    if (p.startsWith("`") && p.endsWith("`")) return <code key={i} className="bg-white/10 text-blue-300 px-1 rounded text-xs font-mono">{p.slice(1, -1)}</code>;
    return p;
  });
}

function MDText({ text }) {
  return (
    <div className="space-y-1">
      {text.split("\n").map((line, i) => {
        if (!line.trim()) return <div key={i} className="h-1" />;
        if (line.startsWith("## "))  return <p key={i} className="font-semibold text-white/95 mt-2">{line.slice(3)}</p>;
        if (line.startsWith("### ")) return <p key={i} className="text-white/50 text-xs uppercase tracking-wider mt-1">{line.slice(4)}</p>;
        if (line.match(/^[-•] /))    return <div key={i} className="flex gap-2"><span className="text-white/30 flex-shrink-0">•</span><span>{renderInline(line.slice(2))}</span></div>;
        const nm = line.match(/^(\d+)\.\s(.+)/);
        if (nm) return <div key={i} className="flex gap-2"><span className="text-white/40 font-mono text-xs mt-0.5 flex-shrink-0">{nm[1]}.</span><span>{renderInline(nm[2])}</span></div>;
        return <p key={i}>{renderInline(line)}</p>;
      })}
    </div>
  );
}

// ─── Message bubble ───────────────────────────────────────────────────────────
function Bubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} gap-3`}>
      {!isUser && (
        <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-xs font-bold flex-shrink-0 mt-1">A</div>
      )}
      <div className={`max-w-[85%] flex flex-col gap-2 ${isUser ? "items-end" : "items-start w-full"}`}>

        {/* Tool pills */}
        {msg.tool_calls?.length > 0 && (
          <div className="flex flex-wrap gap-1">
            {msg.tool_calls.map((t, i) => (
              <span key={i} className="inline-flex items-center gap-1 bg-blue-500/15 border border-blue-500/30 text-blue-300 text-xs px-2 py-0.5 rounded-full">
                {TOOL_LABELS[t] || t}
              </span>
            ))}
          </div>
        )}

        {/* Text */}
        <div className={`rounded-2xl px-4 py-3 text-sm leading-relaxed w-full ${
          isUser
            ? "bg-blue-600 text-white rounded-tr-sm max-w-fit"
            : "bg-white/8 border border-white/10 text-white/85 rounded-tl-sm"
        }`}>
          {isUser ? msg.content : <MDText text={msg.content} />}
        </div>

        {/* Charts — one block per tool */}
        {msg.tool_calls?.map((toolName, i) => {
          const d = msg.tool_data?.[toolName];
          return d ? <div key={i} className="w-full"><ToolViz toolName={toolName} data={d} /></div> : null;
        })}

      </div>
    </div>
  );
}

function TypingDots() {
  return (
    <div className="flex gap-3 items-start">
      <div className="w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-xs font-bold flex-shrink-0">A</div>
      <div className="bg-white/8 border border-white/10 rounded-2xl rounded-tl-sm px-4 py-3">
        <div className="flex gap-1 items-center h-4">
          {[0, 1, 2].map(i => (
            <div key={i} className="w-1.5 h-1.5 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: `${i * 0.15}s` }} />
          ))}
        </div>
      </div>
    </div>
  );
}

const QUICK = ["Full analysis on AAPL", "Bollinger + RSI for TSLA", "Is NVDA overvalued?", "MSFT fundamentals"];

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([{
    role: "assistant",
    content: "Hi! I'm **AlphaLens** — your AI stock analyst.\n\nAsk me about any stock and I'll show you interactive charts: equity curves, win-rate comparisons, fundamentals radar, valuation gaps, and a final buy/hold/sell verdict.\n\nWhat would you like to analyze?",
  }]);
  const [input, setInput]       = useState("");
  const [loading, setLoading]   = useState(false);
  const [sessionId]             = useState(() => crypto.randomUUID());
  const [status, setStatus]     = useState("checking");
  const endRef                  = useRef(null);
  const textareaRef             = useRef(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, loading]);
  useEffect(() => {
    fetch(`${API_BASE}/health`).then(r => setStatus(r.ok ? "ok" : "error")).catch(() => setStatus("error"));
  }, []);
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "48px";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  const send = async (text) => {
    const msg = (text || input).trim();
    if (!msg || loading) return;
    setInput("");
    setMessages(p => [...p, { role: "user", content: msg }]);
    setLoading(true);
    try {
      const res  = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg, session_id: sessionId }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setMessages(p => [...p, {
        role: "assistant",
        content:     data.response,
        tool_calls:  data.tool_calls || [],
        tool_data:   data.tool_data  || {},
      }]);
    } catch (e) {
      setMessages(p => [...p, { role: "assistant", content: `⚠️ **Error:** ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0d1117] text-white flex flex-col" style={{ fontFamily: "system-ui,sans-serif" }}>

      {/* Header */}
      <div className="border-b border-white/8 px-6 py-3 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center font-bold text-sm">A</div>
          <div>
            <p className="font-semibold text-sm">AlphaLens</p>
            <p className="text-white/30 text-xs">Claude Sonnet · MCP tool-calling</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${status === "ok" ? "bg-emerald-400" : status === "error" ? "bg-red-400" : "bg-amber-400 animate-pulse"}`} />
          <span className="text-white/30 text-xs">{status === "ok" ? "Connected" : status === "error" ? "Offline" : "Connecting…"}</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-6 py-6 space-y-6">
          {messages.map((m, i) => <Bubble key={i} msg={m} />)}
          {loading && <TypingDots />}
          <div ref={endRef} />
        </div>
      </div>

      {/* Quick actions — only on first load */}
      {messages.length === 1 && (
        <div className="max-w-3xl mx-auto px-6 pb-3 w-full">
          <div className="flex flex-wrap gap-2">
            {QUICK.map((q, i) => (
              <button key={i} onClick={() => send(q)}
                className="text-xs bg-white/5 hover:bg-white/10 border border-white/10 hover:border-white/20 rounded-full px-3 py-1.5 text-white/55 hover:text-white/90 transition-all">
                {q}
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
            onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
            placeholder="Ask about any stock…"
            className="flex-1 bg-[#161b22] border border-white/15 focus:border-blue-400/60 rounded-xl px-4 py-3 text-sm text-white placeholder-white/30 resize-none focus:outline-none transition-colors"
            style={{ minHeight: "48px", maxHeight: "120px" }}
          />
          <button onClick={() => send()} disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-500 disabled:opacity-35 disabled:cursor-not-allowed rounded-xl px-5 py-3 text-sm font-semibold transition-colors flex-shrink-0">
            {loading ? "…" : "↑"}
          </button>
        </div>
        <p className="text-white/15 text-xs mt-2 text-center">For informational purposes only. Not financial advice.</p>
      </div>
    </div>
  );
}