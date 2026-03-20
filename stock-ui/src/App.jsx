import { useState, useRef, useEffect } from "react";
import {
  AreaChart, Area, BarChart, Bar, ComposedChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  Cell, Legend,
} from "recharts";

const API_BASE = "https://finbot-production-5739.up.railway.app";

// ─── Colours ──────────────────────────────────────────────────────────────────
const SIG = {
  buy:          { bg: "bg-emerald-500/15", border: "border-emerald-500/40", text: "text-emerald-400", dot: "bg-emerald-400", hex: "#10b981" },
  bullish:      { bg: "bg-emerald-500/15", border: "border-emerald-500/40", text: "text-emerald-400", dot: "bg-emerald-400", hex: "#10b981" },
  strong_buy:   { bg: "bg-emerald-500/25", border: "border-emerald-400/60", text: "text-emerald-300", dot: "bg-emerald-300", hex: "#6ee7b7" },
  hold:         { bg: "bg-amber-500/15",   border: "border-amber-500/40",   text: "text-amber-400",   dot: "bg-amber-400",   hex: "#f59e0b" },
  neutral:      { bg: "bg-amber-500/15",   border: "border-amber-500/40",   text: "text-amber-400",   dot: "bg-amber-400",   hex: "#f59e0b" },
  sell:         { bg: "bg-red-500/15",     border: "border-red-500/40",     text: "text-red-400",     dot: "bg-red-400",     hex: "#ef4444" },
  bearish:      { bg: "bg-red-500/15",     border: "border-red-500/40",     text: "text-red-400",     dot: "bg-red-400",     hex: "#ef4444" },
  strong_sell:  { bg: "bg-red-500/25",     border: "border-red-400/60",     text: "text-red-300",     dot: "bg-red-300",     hex: "#fca5a5" },
};
const sc = (s) => SIG[(s || "").toLowerCase()] || SIG.neutral;
const sl = (s) => (s || "neutral").toUpperCase();

// Trinity 时空状态颜色
const STATE_COLORS = {
  extreme_strong: { bg: "bg-emerald-500/25", border: "border-emerald-400", text: "text-emerald-300", label: "极强 🔥" },
  strong:         { bg: "bg-emerald-500/15", border: "border-emerald-500", text: "text-emerald-400", label: "强 ↑" },
  mid_strong:     { bg: "bg-teal-500/15",    border: "border-teal-500",    text: "text-teal-400",    label: "中性偏强" },
  mid_weak:       { bg: "bg-orange-500/15",  border: "border-orange-500",  text: "text-orange-400",  label: "中性偏弱" },
  extreme_weak:   { bg: "bg-red-500/25",     border: "border-red-400",     text: "text-red-300",     label: "极弱 ❄️" },
  weak:           { bg: "bg-red-500/15",     border: "border-red-500",     text: "text-red-400",     label: "弱 ↓" },
  unknown:        { bg: "bg-white/10",        border: "border-white/20",    text: "text-white/50",    label: "未知" },
};
const stateColor = (code) => STATE_COLORS[code] || STATE_COLORS.unknown;

const VERDICT_CONFIG = {
  BUY:        { gradient: "from-emerald-600 to-teal-700",  icon: "↑" },
  STRONG_BUY: { gradient: "from-emerald-500 to-green-600", icon: "↑↑" },
  HOLD:       { gradient: "from-amber-600  to-orange-700", icon: "→" },
  SELL:       { gradient: "from-red-600    to-rose-700",   icon: "↓" },
  STRONG_SELL:{ gradient: "from-red-700    to-rose-800",   icon: "↓↓" },
};

const TOOL_LABELS = {
  get_stock_overview:          "📊 公司概览",
  analyze_technicals:          "📈 完整技术分析",
  analyze_single_indicator:    "📉 单一指标",
  analyze_multiple_indicators: "📉 多指标分析",
  analyze_fundamentals:        "📋 基本面分析",
  analyze_valuation:           "💰 估值分析",
  deep_research_edgar:         "📄 EDGAR研报",
  get_full_analysis:           "🔬 综合分析",
  trinity_analysis:            "☯️ 三位一体",
};

const GRID  = { stroke: "rgba(255,255,255,0.06)" };
const ATICK = { fill: "rgba(255,255,255,0.3)", fontSize: 9 };

// ─── Shared tooltip ───────────────────────────────────────────────────────────
const Tip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#1a1f2e] border border-white/15 rounded-lg px-2 py-1.5 text-xs shadow-xl max-w-[180px]">
      {label && <p className="text-white/40 mb-1 truncate">{label}</p>}
      {payload.map((p, i) => (
        <p key={i} style={{ color: p.color || "#fff" }} className="truncate">
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(2) : p.value}
        </p>
      ))}
    </div>
  );
};

const Section = ({ title, children }) => (
  <div className="bg-white/4 border border-white/8 rounded-xl p-3 sm:p-4 space-y-3">
    <p className="text-white/50 text-xs font-semibold uppercase tracking-wider">{title}</p>
    {children}
  </div>
);

// ─── Trinity: 均线 + 布林带价格图 ─────────────────────────────────────────────
function TrinityPriceChart({ chartData, summary }) {
  if (!chartData?.length) return null;

  // 只取最近60条避免太密集
  const data = chartData.slice(-60);
  // 显示每隔N条的日期label
  const labelStep = Math.max(1, Math.floor(data.length / 6));

  const priceMin = Math.min(...data.map(d => d.low).filter(Boolean)) * 0.995;
  const priceMax = Math.max(...data.map(d => d.high).filter(Boolean)) * 1.005;

  const breakoutType = summary?.ma_breakout_type;
  const breakoutDir  = summary?.ma_breakout_direction;  // "up" | "down" | ""
  const breakoutColors = { A: "#f59e0b", B: "#10b981", C: "#3b82f6", D: "#a78bfa" };
  const breakoutColor  = breakoutColors[breakoutType] || "#fff";
  const dirLabel = breakoutDir === "up" ? "向上" : breakoutDir === "down" ? "向下" : "";

  return (
    <Section title={`价格 · MA55 · MA233 · 布林带 — 近60日`}>
      {/* 突破类型标注 */}
      {breakoutType && breakoutType !== "none" && (
        <div className="flex items-center gap-2 text-xs mb-1">
          <span style={{ color: breakoutColor }} className="font-bold">
            均线突破类型：{breakoutType}类
          </span>
          <span className="text-white/30">
            {{ A: "典型突破", B: "慢速/盘整突破", C: "突破后回抽", D: "反向测试" }[breakoutType]}
            {dirLabel && `（${dirLabel}）`}
          </span>
        </div>
      )}

      <div className="flex gap-3 flex-wrap text-xs mb-1">
        <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-yellow-400 inline-block" />MA55</span>
        <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-blue-400 inline-block" />MA233</span>
        <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-purple-400/60 inline-block" />布林带</span>
        {summary?.key_support && (
          <span className="text-emerald-400">支撑: ${(+summary.key_support).toFixed(2)}</span>
        )}
        {summary?.key_resistance && (
          <span className="text-red-400">压力: ${(+summary.key_resistance).toFixed(2)}</span>
        )}
      </div>

      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
          <CartesianGrid {...GRID} />
          <XAxis
            dataKey="date"
            tick={ATICK}
            tickFormatter={(v, i) => i % labelStep === 0 ? v?.slice(5) : ""}
            interval={0}
          />
          <YAxis domain={[priceMin, priceMax]} tick={ATICK} width={48} tickFormatter={v => `$${v.toFixed(0)}`} />
          <Tooltip content={<Tip />} />

          {/* 布林带区域 */}
          <Area dataKey="bb_upper" stroke="rgba(167,139,250,0.4)" strokeWidth={1} fill="rgba(167,139,250,0.05)" name="布林上轨" dot={false} />
          <Area dataKey="bb_lower" stroke="rgba(167,139,250,0.4)" strokeWidth={1} fill="rgba(167,139,250,0.05)" name="布林下轨" dot={false} />

          {/* 收盘价 */}
          <Line dataKey="close" stroke="#e2e8f0" strokeWidth={1.5} dot={false} name="收盘价" />

          {/* MA55 */}
          <Line dataKey="ma55"  stroke="#fbbf24" strokeWidth={2} dot={false} name="MA55" />
          {/* MA233 */}
          <Line dataKey="ma233" stroke="#60a5fa" strokeWidth={2} dot={false} name="MA233" />

          {/* 支撑/压力线 */}
          {summary?.key_support    && <ReferenceLine y={summary.key_support}    stroke="#10b981" strokeDasharray="4 4" label={{ value: `支撑 $${(+summary.key_support).toFixed(2)}`,    fill: "#10b981", fontSize: 9 }} />}
          {summary?.key_resistance && <ReferenceLine y={summary.key_resistance} stroke="#ef4444" strokeDasharray="4 4" label={{ value: `压力 $${(+summary.key_resistance).toFixed(2)}`, fill: "#ef4444", fontSize: 9 }} />}
        </ComposedChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ─── Trinity: MACD图 ──────────────────────────────────────────────────────────
function TrinityMacdChart({ chartData }) {
  if (!chartData?.length) return null;
  const data = chartData.slice(-60);
  const labelStep = Math.max(1, Math.floor(data.length / 6));

  return (
    <Section title="MACD — DIF · DEA · 柱状图">
      <div className="flex gap-3 flex-wrap text-xs mb-1">
        <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-white inline-block" />DIF(快线)</span>
        <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-yellow-400 inline-block" />DEA(慢线)</span>
        <span className="text-white/40 text-xs">柱状图：红=多头动能 绿=空头动能</span>
      </div>
      <ResponsiveContainer width="100%" height={130}>
        <ComposedChart data={data} margin={{ left: 0, right: 8, top: 4, bottom: 0 }}>
          <CartesianGrid {...GRID} />
          <XAxis dataKey="date" tick={ATICK} tickFormatter={(v, i) => i % labelStep === 0 ? v?.slice(5) : ""} interval={0} />
          <YAxis tick={ATICK} width={48} />
          <Tooltip content={<Tip />} />
          <ReferenceLine y={0} stroke="rgba(255,255,255,0.25)" />
          <Bar dataKey="macd_bar" name="MACD柱" radius={[1,1,0,0]}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.macd_bar >= 0 ? "#10b981" : "#ef4444"} fillOpacity={0.7} />
            ))}
          </Bar>
          <Line dataKey="dif" stroke="#e2e8f0" strokeWidth={1.5} dot={false} name="DIF" />
          <Line dataKey="dea" stroke="#fbbf24" strokeWidth={1.5} dot={false} name="DEA" />
        </ComposedChart>
      </ResponsiveContainer>
    </Section>
  );
}

// ─── helpers ──────────────────────────────────────────────────────────────────
const labelToStateColor = (label) => {
  if (!label || label === "未知") return STATE_COLORS.unknown;
  if (label.includes("极强")) return STATE_COLORS.extreme_strong;
  if (label.includes("中性偏强")) return STATE_COLORS.mid_strong;
  if (label.includes("强"))   return STATE_COLORS.strong;
  if (label.includes("极弱")) return STATE_COLORS.extreme_weak;
  if (label.includes("中性偏弱")) return STATE_COLORS.mid_weak;
  if (label.includes("弱"))   return STATE_COLORS.weak;
  return STATE_COLORS.unknown;
};

const POS_SIZE_CONFIG = {
  heavy:  { label: "重仓", pct: 90, color: "bg-emerald-400" },
  medium: { label: "中仓", pct: 60, color: "bg-amber-400"   },
  light:  { label: "轻仓", pct: 30, color: "bg-orange-400"  },
  none:   { label: "观望", pct:  5, color: "bg-white/20"    },
};

// ─── Trinity: 主卡片 ──────────────────────────────────────────────────────────
const MATURITY_LABEL = {
  forming:      { text: "初期（<3根K线，谨慎）",     color: "text-orange-400", dot: "bg-orange-400" },
  intensifying: { text: "激化中（矛盾加剧）",          color: "text-yellow-400", dot: "bg-yellow-400" },
  mature:       { text: "成熟（信号有效）",            color: "text-emerald-400", dot: "bg-emerald-400" },
};
const BREAKOUT_DESC = { A: "典型突破", B: "慢速/盘整突破", C: "突破后回抽", D: "反向测试" };

function TrinityCard({ data }) {
  const [activeTab, setActiveTab] = useState("时空");
  if (!data?.summary) return null;
  const s   = data.summary;
  const sc_ = sc(s.signal);
  const st  = stateColor(s.state_code);
  const verdictKey = (s.signal || "hold").toUpperCase().replace("_", "_");
  const cfg = VERDICT_CONFIG[verdictKey] || VERDICT_CONFIG.HOLD;

  const hasDivergence = s.divergence_type && s.divergence_type !== "none";
  const tabs = ["时空", "结构", "均线", "技术", ...(hasDivergence ? ["背离"] : [])];

  // Last bar for technical tab
  const lastBar   = data.price_chart_data?.slice(-1)[0];
  const bbPct     = (lastBar?.bb_upper && lastBar?.bb_lower && lastBar?.bb_upper !== lastBar?.bb_lower)
    ? Math.round(((lastBar.close - lastBar.bb_lower) / (lastBar.bb_upper - lastBar.bb_lower)) * 100)
    : null;
  const posCfg    = POS_SIZE_CONFIG[s.position_size] || POS_SIZE_CONFIG.light;

  // Multi-timeframe states for 时空 tab
  const monthlyLabel = data.time_space_state?.monthly_state?.state_label || "数据不足";
  const weeklyC  = labelToStateColor(s.weekly_state_label);
  const monthlyC = labelToStateColor(monthlyLabel);

  const isTop = s.divergence_type?.includes("top");
  const maturity = isTop ? s.top_div_maturity : s.bot_div_maturity;
  const matLabel = MATURITY_LABEL[maturity];

  // Resolve structure type: prefer Claude output, fallback to Python classifier
  const rawType = s.pattern_type && s.pattern_type !== "unknown" ? s.pattern_type : s.structure_type_py;
  const structureType = rawType?.toUpperCase() || null;
  const STRUCT_DESC = { A: "五段式", B: "双平台", C: "单平台", D: "三段式" };
  // Resolve current stage: prefer Claude, fallback to Python
  const currentStage = (s.current_stage && s.current_stage !== "unknown") ? s.current_stage : s.structure_stage_py;

  const isShort = s.signal === "sell";
  const r1l = s.reduce_1st_long || s.reduce_1st;
  const r2l = s.reduce_2nd_long || s.reduce_2nd;
  const r1s = s.reduce_1st_short;
  const r2s = s.reduce_2nd_short;

  // Distance from current price to MA55
  const ma55Dist = s.current_price && s.ma55
    ? (((+s.current_price - +s.ma55) / +s.ma55) * 100).toFixed(1)
    : null;

  return (
    <div className="space-y-3">

      {/* ── 顶部：信号 + 时空状态 ── */}
      <div className={`rounded-2xl bg-gradient-to-r ${cfg.gradient} p-4 sm:p-5`}>
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-2xl sm:text-3xl font-black">{cfg.icon} {(s.signal || "HOLD").replace("_", " ").toUpperCase()}</span>
              <span className={`text-xs px-2 py-0.5 rounded-full border ${st.border} ${st.bg} ${st.text} font-semibold`}>
                {st.label}
              </span>
            </div>
            <p className="text-white/60 text-xs">
              {s.bars_in_state > 0 ? `已持续 ${s.bars_in_state} 根K线` : ""}
              {s.main_wave_locked ? "  🔒 主涨段锁定中" : ""}
            </p>
          </div>
          <div className="text-right">
            <p className="text-white font-mono text-lg font-bold">{data.ticker}</p>
            {s.current_price && <p className="text-white/70 text-sm">${s.current_price}</p>}
            <p className="text-white/40 text-xs">{s.confidence} confidence</p>
          </div>
        </div>

        {/* 置信度进度条 */}
        <div className="w-full h-1.5 bg-white/20 rounded-full mt-3">
          <div className="h-full bg-white/60 rounded-full" style={{ width: s.confidence === "high" ? "85%" : s.confidence === "medium" ? "55%" : "30%" }} />
        </div>

        {/* 建议行动 */}
        {s.suggested_action && (
          <p className="text-white/85 text-xs sm:text-sm mt-3 pt-3 border-t border-white/20 leading-relaxed">
            💡 {s.suggested_action}
          </p>
        )}
        {s.key_risk && (
          <p className="text-white/60 text-xs mt-1.5">⚠️ {s.key_risk}</p>
        )}
      </div>

      {/* ── 均线 + MACD 真实图表 ── */}
      {data.price_chart_data?.length > 0 && (
        <>
          <TrinityPriceChart chartData={data.price_chart_data} summary={s} />
          <TrinityMacdChart  chartData={data.price_chart_data} />
        </>
      )}

      {/* ── 三维度 Tabs ── */}
      <div className="bg-white/4 border border-white/8 rounded-xl overflow-hidden">
        {/* Tab bar */}
        <div className="flex border-b border-white/8">
          {tabs.map(tab => {
            const isActive = activeTab === tab;
            // badge for divergence tab
            const badge = tab === "背离" && hasDivergence;
            return (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-2.5 text-xs font-semibold transition-all relative
                  ${isActive
                    ? "text-white bg-white/8"
                    : "text-white/35 hover:text-white/60 hover:bg-white/4"
                  }`}
              >
                {tab}
                {badge && (
                  <span className={`ml-1 inline-block w-1.5 h-1.5 rounded-full align-middle -mt-0.5 ${isTop ? "bg-red-400" : "bg-emerald-400"}`} />
                )}
                {isActive && (
                  <span className="absolute bottom-0 left-0 right-0 h-0.5 bg-white/50 rounded-t" />
                )}
              </button>
            );
          })}
        </div>

        {/* Tab content */}
        <div className="p-3 sm:p-4">

          {/* ─ 时空 ─ */}
          {activeTab === "时空" && (
            <div className="space-y-3">
              {/* Multi-timeframe grid */}
              <div className="grid grid-cols-3 gap-2">
                {[
                  { label: "日线", c: st,       stateLabel: st.label,            bars: s.bars_in_state },
                  { label: "周线", c: weeklyC,  stateLabel: s.weekly_state_label || "未知", bars: null },
                  { label: "月线", c: monthlyC, stateLabel: monthlyLabel,        bars: null },
                ].map(({ label, c, stateLabel, bars }) => (
                  <div key={label} className={`rounded-xl border ${c.border} ${c.bg} px-2.5 py-2.5 text-center`}>
                    <p className="text-white/35 text-xs mb-1">{label}</p>
                    <p className={`text-xs font-bold leading-tight ${c.text}`}>{stateLabel}</p>
                    {bars > 0 && <p className="text-white/25 text-xs mt-0.5">{bars}根</p>}
                  </div>
                ))}
              </div>

              {/* State anomaly note */}
              {s.state_anomaly && (
                <div className="flex items-start gap-2 bg-amber-500/8 border border-amber-500/20 rounded-lg px-3 py-2 text-xs">
                  <span className="text-amber-400 flex-shrink-0">⚡</span>
                  <span className="text-amber-300/80">状态发生回落，动能钝化，留意上升持续性</span>
                </div>
              )}

              {/* MTF conflict */}
              {s.multi_timeframe_conflict && (
                <div className="flex items-start gap-2 bg-red-500/8 border border-red-500/20 rounded-lg px-3 py-2 text-xs">
                  <span className="text-red-400 flex-shrink-0">⚠️</span>
                  <span className="text-red-300/80">多级别冲突：{s.mtf_conflict_type || "日强周弱"}</span>
                </div>
              )}

              {/* Main wave + first assumption */}
              <div className="flex items-center justify-between">
                {s.main_wave_locked
                  ? <span className="text-emerald-300 text-xs font-semibold bg-emerald-500/20 border border-emerald-500/30 rounded-lg px-2.5 py-1">🔒 主涨段锁定</span>
                  : <span className="text-white/30 text-xs bg-white/5 border border-white/10 rounded-lg px-2.5 py-1">主涨段未启动</span>
                }
                {s.first_assumption && (
                  <span className="text-white/40 text-xs text-right max-w-[55%] leading-tight">{s.first_assumption}</span>
                )}
              </div>

              {s.main_wave_note && (
                <p className="text-white/40 text-xs leading-relaxed">{s.main_wave_note}</p>
              )}
            </div>
          )}

          {/* ─ 结构 ─ */}
          {activeTab === "结构" && (
            <div className="space-y-3">
              {/* Type + stage hero */}
              <div className="flex gap-3">
                <div className="flex-1 bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-center">
                  <p className="text-white/40 text-xs mb-1">结构类型</p>
                  {structureType
                    ? <>
                        <p className="text-white text-xl font-black">{structureType} 类</p>
                        <p className="text-white/30 text-xs mt-0.5">{STRUCT_DESC[structureType] || "整理中"}</p>
                      </>
                    : <p className="text-white/30 text-sm mt-1">分析中…</p>
                  }
                </div>
                <div className="flex-1 bg-white/5 border border-white/10 rounded-xl px-3 py-2.5 text-center">
                  <p className="text-white/40 text-xs mb-1">当前阶段</p>
                  {currentStage
                    ? <p className="text-white text-xl font-black">第 {currentStage} 笔</p>
                    : <p className="text-white/30 text-sm mt-1">待确认</p>
                  }
                  <p className={`text-xs mt-0.5 font-semibold
                    ${s.likely_next === "up" ? "text-emerald-400" : s.likely_next === "down" ? "text-red-400" : "text-amber-400"}`}>
                    {s.likely_next === "up" ? "↑ 下一步：上涨" : s.likely_next === "down" ? "↓ 下一步：下跌" : "→ 震荡整理"}
                  </p>
                </div>
              </div>
              {/* Key levels */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                {s.key_support && (
                  <div className="bg-emerald-500/8 border border-emerald-500/20 rounded-lg px-3 py-2">
                    <p className="text-emerald-400/60 mb-0.5">支撑位</p>
                    <p className="text-emerald-300 font-mono font-bold">${(+s.key_support).toFixed(2)}</p>
                  </div>
                )}
                {s.key_resistance && (
                  <div className="bg-red-500/8 border border-red-500/20 rounded-lg px-3 py-2">
                    <p className="text-red-400/60 mb-0.5">压力位</p>
                    <p className="text-red-300 font-mono font-bold">${(+s.key_resistance).toFixed(2)}</p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ─ 均线 ─ */}
          {activeTab === "均线" && (
            <div className="space-y-3">
              {/* Alignment hero */}
              <div className={`rounded-xl px-4 py-3 border flex items-center justify-between
                ${s.trend_alignment === "bullish" ? "bg-emerald-500/10 border-emerald-500/30"
                  : s.trend_alignment === "bearish" ? "bg-red-500/10 border-red-500/30"
                  : "bg-amber-500/10 border-amber-500/30"}`}>
                <div>
                  <p className={`text-base font-bold
                    ${s.trend_alignment === "bullish" ? "text-emerald-400"
                      : s.trend_alignment === "bearish" ? "text-red-400"
                      : "text-amber-400"}`}>
                    {s.trend_alignment === "bullish" ? "多头排列 ↑"
                      : s.trend_alignment === "bearish" ? "空头排列 ↓"
                      : "混沌排列 ↔"}
                  </p>
                  <p className="text-white/40 text-xs mt-0.5">
                    {s.trend_alignment === "bullish" ? "价格 > MA55 > MA233"
                      : s.trend_alignment === "bearish" ? "价格 < MA55 < MA233"
                      : "均线尚未形成方向"}
                  </p>
                </div>
                {s.ma_breakout_type && s.ma_breakout_type !== "none" && (
                  <div className="text-right">
                    <p className="text-purple-300 text-xs font-bold">{s.ma_breakout_type} 类突破</p>
                    <p className="text-white/30 text-xs">{BREAKOUT_DESC[s.ma_breakout_type] || ""}</p>
                  </div>
                )}
              </div>
              {/* MA values */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="bg-yellow-400/8 border border-yellow-400/20 rounded-lg px-3 py-2">
                  <p className="text-yellow-400/60 mb-0.5">MA55</p>
                  <p className="text-yellow-300 font-mono font-bold">${s.ma55?.toFixed(2) || "—"}</p>
                  {ma55Dist !== null && (
                    <p className={`text-xs mt-0.5 ${+ma55Dist >= 0 ? "text-emerald-400/70" : "text-red-400/70"}`}>
                      {+ma55Dist >= 0 ? "+" : ""}{ma55Dist}% 距离
                    </p>
                  )}
                </div>
                <div className="bg-blue-400/8 border border-blue-400/20 rounded-lg px-3 py-2">
                  <p className="text-blue-400/60 mb-0.5">MA233</p>
                  <p className="text-blue-300 font-mono font-bold">${s.ma233?.toFixed(2) || "—"}</p>
                  <p className="text-white/20 text-xs mt-0.5">长期均线</p>
                </div>
              </div>
            </div>
          )}

          {/* ─ 技术 ─ */}
          {activeTab === "技术" && (
            <div className="space-y-3">
              {/* Bollinger Band position gauge */}
              {bbPct !== null && (
                <div>
                  <div className="flex justify-between text-xs mb-1.5">
                    <span className="text-white/40">布林带位置</span>
                    <span className={`font-semibold ${bbPct > 80 ? "text-red-400" : bbPct < 20 ? "text-emerald-400" : "text-white/70"}`}>
                      {bbPct}%
                      {bbPct > 80 ? " · 接近上轨" : bbPct < 20 ? " · 接近下轨" : " · 中间区域"}
                    </span>
                  </div>
                  <div className="relative h-3 bg-white/8 rounded-full overflow-hidden">
                    {/* gradient band: lower=green, mid=neutral, upper=red */}
                    <div className="absolute inset-0 bg-gradient-to-r from-emerald-500/40 via-white/10 to-red-500/40 rounded-full" />
                    {/* position marker */}
                    <div
                      className="absolute top-0.5 bottom-0.5 w-2 bg-white rounded-full shadow-lg transition-all"
                      style={{ left: `calc(${Math.min(Math.max(bbPct, 2), 98)}% - 4px)` }}
                    />
                  </div>
                  <div className="flex justify-between text-xs text-white/20 mt-1">
                    <span>下轨 ${lastBar?.bb_lower?.toFixed(2)}</span>
                    <span>中轨 ${lastBar?.bb_mid?.toFixed(2)}</span>
                    <span>上轨 ${lastBar?.bb_upper?.toFixed(2)}</span>
                  </div>
                </div>
              )}

              {/* MACD current values */}
              {lastBar?.dif != null && (
                <div className="grid grid-cols-3 gap-2 text-xs">
                  {[
                    { label: "DIF", val: lastBar.dif,      color: "text-white/80" },
                    { label: "DEA", val: lastBar.dea,      color: "text-yellow-400" },
                    { label: "柱", val: lastBar.macd_bar,  color: lastBar.macd_bar >= 0 ? "text-emerald-400" : "text-red-400" },
                  ].map(({ label, val, color }) => (
                    <div key={label} className="bg-white/5 border border-white/8 rounded-lg px-2.5 py-2 text-center">
                      <p className="text-white/30 text-xs mb-0.5">{label}</p>
                      <p className={`font-mono font-semibold ${color}`}>{val?.toFixed(2)}</p>
                    </div>
                  ))}
                </div>
              )}

              {/* Position size recommendation */}
              <div>
                <div className="flex justify-between text-xs mb-1.5">
                  <span className="text-white/40">建议仓位</span>
                  <span className={`font-semibold ${posCfg.color.replace("bg-", "text-")}`}>{posCfg.label}</span>
                </div>
                <div className="h-2 bg-white/8 rounded-full overflow-hidden">
                  <div className={`h-full ${posCfg.color} rounded-full transition-all`} style={{ width: `${posCfg.pct}%` }} />
                </div>
              </div>

              {/* Pullback opportunity */}
              {s.pullback_opportunity && (
                <div className="flex items-center gap-2 bg-emerald-500/8 border border-emerald-500/20 rounded-lg px-3 py-2 text-xs">
                  <span className="text-emerald-400">✓</span>
                  <span className="text-emerald-300/80">回踩MA55机会 — 右侧入场信号</span>
                </div>
              )}
            </div>
          )}

          {/* ─ 背离 ─ */}
          {activeTab === "背离" && hasDivergence && (
            <div className="space-y-3">
              <div className={`rounded-xl px-4 py-3 border flex items-center gap-3
                ${isTop ? "bg-red-500/10 border-red-500/30" : "bg-emerald-500/10 border-emerald-500/30"}`}>
                <span className="text-2xl">{isTop ? "📉" : "📈"}</span>
                <div className="flex-1">
                  <p className={`text-base font-bold ${isTop ? "text-red-400" : "text-emerald-400"}`}>
                    {isTop ? "顶背离" : "底背离"}
                    <span className="text-white/40 font-normal ml-2 text-xs">({s.divergence_strength})</span>
                  </p>
                  <p className="text-white/50 text-xs mt-0.5 leading-relaxed">
                    {s.divergence_note || (isTop ? "MACD动能衰竭，注意拐点风险" : "MACD动能收缩，可能出现反弹")}
                  </p>
                </div>
              </div>
              {matLabel && (
                <div className="flex items-center justify-between bg-white/5 border border-white/10 rounded-lg px-3 py-2.5">
                  <span className="text-white/40 text-xs">成熟度</span>
                  <div className="flex items-center gap-1.5">
                    <span className={`w-2 h-2 rounded-full ${matLabel.dot}`} />
                    <span className={`text-xs font-semibold ${matLabel.color}`}>{matLabel.text}</span>
                  </div>
                </div>
              )}
            </div>
          )}

        </div>
      </div>

      {/* ── 止盈止损策略 ── */}
      <div className="bg-white/4 border border-white/8 rounded-xl overflow-hidden">
        <div className="px-3 sm:px-4 py-2 border-b border-white/8">
          <p className="text-white/50 text-xs font-semibold uppercase tracking-wider">止盈止损策略</p>
        </div>
        <div className="px-3 sm:px-4 py-3 space-y-2">
          {s.main_wave_locked ? (
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-3 py-2">
              <p className="text-emerald-400 text-xs font-semibold mb-1">🔒 主涨段锁定模式</p>
              <p className="text-white/60 text-xs">止盈触发：{s.exit_trigger}</p>
              <p className="text-white/40 text-xs mt-1">锁定期间忽略：顶背离、结构前高、小级别MACD信号</p>
            </div>
          ) : (<>
            {!isShort && r1l && (
              <div className="flex items-start gap-3 text-xs">
                <span className="text-amber-400 font-bold flex-shrink-0 w-16">减仓①</span>
                <span className="text-white/60 leading-relaxed">{r1l}</span>
              </div>
            )}
            {!isShort && r2l && (
              <div className="flex items-start gap-3 text-xs">
                <span className="text-red-400 font-bold flex-shrink-0 w-16">减仓②</span>
                <span className="text-white/60 leading-relaxed">{r2l}</span>
              </div>
            )}
            {isShort && r1s && (
              <div className="flex items-start gap-3 text-xs">
                <span className="text-sky-400 font-bold flex-shrink-0 w-16">平空①</span>
                <span className="text-white/60 leading-relaxed">{r1s}</span>
              </div>
            )}
            {isShort && r2s && (
              <div className="flex items-start gap-3 text-xs">
                <span className="text-blue-400 font-bold flex-shrink-0 w-16">平空②</span>
                <span className="text-white/60 leading-relaxed">{r2s}</span>
              </div>
            )}
            {!r1l && !r1s && s.exit_trigger && (
              <p className="text-white/60 text-xs">止盈触发：{s.exit_trigger}</p>
            )}
            {s.long_stop_loss && (
              <div className="flex items-center justify-between pt-1 border-t border-white/8 text-xs">
                <span className="text-white/30">止损价</span>
                <span className="text-red-400 font-mono font-bold">${(+s.long_stop_loss).toFixed(2)}</span>
              </div>
            )}
          </>)}
          {s.holding_constraint && (
            <p className="text-orange-400 text-xs border-t border-white/8 pt-2">{s.holding_constraint}</p>
          )}
        </div>
      </div>

      {/* ── 入场方向 ── */}
      <div className={`flex items-center gap-3 rounded-xl px-3 py-2.5 border ${sc_.border} ${sc_.bg}`}>
        <span className={`text-lg font-black ${sc_.text}`}>{sc_.hex === "#10b981" ? "↑" : sc_.hex === "#ef4444" ? "↓" : "→"}</span>
        <div>
          <p className={`text-xs font-bold ${sc_.text}`}>
            {s.entry_side === "left_side" ? "左侧入场（提前布局）" : s.entry_side === "right_side" ? "右侧入场（确认后跟进）" : "等待信号"}
          </p>
          {s.override_active && (
            <p className="text-white/40 text-xs">{s.override_reason}</p>
          )}
        </div>
      </div>

    </div>
  );
}

// ─── 现有组件（保持不变）────────────────────────────────────────────────────

function SignalGrid({ indicators }) {
  const entries = Object.entries(indicators);
  if (!entries.length) return null;
  return (
    <Section title="信号摘要">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
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

function WinRateChart({ indicators }) {
  const data = Object.entries(indicators)
    .filter(([, v]) => v.backtest?.["win_rate_%"] !== undefined)
    .map(([key, val]) => ({
      name: (val.name || key).replace(/ \d+.*/, "").replace("Bollinger Bands", "BB").replace("Moving Average", "MA"),
      winRate: val.backtest["win_rate_%"] || 0,
      signal: val.signal,
      trades: val.backtest.n_trades || 0,
    }));
  if (!data.length) return null;
  return (
    <Section title="5年回测胜率">
      <div className="overflow-x-auto">
        <div style={{ minWidth: 260 }}>
          <ResponsiveContainer width="100%" height={Math.max(90, data.length * 38)}>
            <BarChart data={data} layout="vertical" margin={{ left: 4, right: 36, top: 0, bottom: 0 }}>
              <CartesianGrid {...GRID} horizontal={false} />
              <XAxis type="number" domain={[0, 100]} tick={ATICK} tickFormatter={v => `${v}%`} />
              <YAxis type="category" dataKey="name" tick={ATICK} width={70} />
              <Tooltip content={<Tip />} formatter={v => [`${v}%`, "胜率"]} />
              <ReferenceLine x={50} stroke="rgba(255,255,255,0.2)" strokeDasharray="4 4" />
              <Bar dataKey="winRate" radius={[0, 4, 4, 0]} name="Win Rate">
                {data.map((d, i) => <Cell key={i} fill={sc(d.signal).hex} fillOpacity={0.8} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
      <div className="flex flex-wrap gap-2 pt-1 border-t border-white/8">
        {data.map((d, i) => (
          <span key={i} className="text-xs text-white/30">{d.name}: <span className="text-white/50">{d.trades}t</span></span>
        ))}
      </div>
    </Section>
  );
}

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
    <Section title={`${name} — 资金曲线`}>
      <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs mb-1">
        <span style={{ color: col }}>策略: {totalReturn > 0 ? "+" : ""}{totalReturn?.toFixed(1)}%</span>
        <span className="text-indigo-400">买持: {buyHold > 0 ? "+" : ""}{buyHold?.toFixed(1)}%</span>
        <span className="text-white/40">胜率: <span className="text-white/70">{winRate?.toFixed(1)}%</span></span>
        <span className="text-white/40">交易次数: <span className="text-white/70">{trades.length}</span></span>
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
          <XAxis dataKey="i" tick={ATICK} />
          <YAxis tick={ATICK} tickFormatter={v => `$${(v / 1000).toFixed(1)}k`} width={42} />
          <Tooltip content={<Tip />} formatter={v => [`$${v.toLocaleString()}`, "组合"]} labelFormatter={l => `第${l}笔`} />
          <ReferenceLine y={10000} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
          <Area type="monotone" dataKey="v" stroke={col} fill={`url(#${id})`} strokeWidth={1.5} dot={false} name="Portfolio" />
        </AreaChart>
      </ResponsiveContainer>
    </Section>
  );
}

function FundamentalsRadar({ sections }) {
  const scoreMap = { bullish: 90, neutral: 50, bearish: 15 };
  const data = Object.entries(sections).map(([key, val]) => ({
    subject: key.charAt(0).toUpperCase() + key.slice(1),
    score: scoreMap[val.signal] ?? 50,
    signal: val.signal,
  }));
  if (!data.length) return null;
  return (
    <Section title="基本面雷达图">
      <ResponsiveContainer width="100%" height={180}>
        <RadarChart data={data}>
          <PolarGrid stroke="rgba(255,255,255,0.1)" />
          <PolarAngleAxis dataKey="subject" tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 9 }} />
          <Radar dataKey="score" stroke="#818cf8" fill="#818cf8" fillOpacity={0.2} strokeWidth={1.5} name="评分" />
          <Tooltip content={<Tip />} formatter={v => [v >= 80 ? "看涨" : v >= 40 ? "中性" : "看跌", "信号"]} />
        </RadarChart>
      </ResponsiveContainer>
      <div className="space-y-1.5 border-t border-white/8 pt-2">
        {Object.entries(sections).map(([key, val]) => {
          const c = sc(val.signal);
          return (
            <div key={key} className="flex gap-2 text-xs">
              <span className={`font-semibold flex-shrink-0 w-16 ${c.text}`}>{sl(val.signal)}</span>
              <span className="text-white/40 leading-relaxed">{val.details}</span>
            </div>
          );
        })}
      </div>
    </Section>
  );
}

function ValuationGapChart({ methods, weightedGap }) {
  const data = Object.entries(methods)
    .filter(([, v]) => v.gap_pct !== null && v.gap_pct !== undefined)
    .map(([key, val]) => ({
      name: key.replace(/_/g, " ").replace(/\b\w/g, l => l.toUpperCase())
               .replace("Owner Earnings", "OE").replace("Residual Income Model", "RIM").replace("Ev Ebitda", "EV/EBITDA"),
      gap: val.gap_pct,
      signal: val.signal,
    }));
  if (!data.length) return null;
  return (
    <Section title="内在价值差距">
      <div className="flex flex-wrap justify-between items-center gap-1">
        <p className="text-white/30 text-xs">+ = 低估 · − = 高估</p>
        <span className={`text-sm font-bold ${weightedGap > 0 ? "text-emerald-400" : "text-red-400"}`}>
          加权: {weightedGap > 0 ? "+" : ""}{weightedGap?.toFixed(1)}%
        </span>
      </div>
      <div className="overflow-x-auto">
        <div style={{ minWidth: 260 }}>
          <ResponsiveContainer width="100%" height={Math.max(110, data.length * 44)}>
            <BarChart data={data} layout="vertical" margin={{ left: 4, right: 44, top: 0, bottom: 0 }}>
              <CartesianGrid {...GRID} horizontal={false} />
              <XAxis type="number" tick={ATICK} tickFormatter={v => `${v > 0 ? "+" : ""}${v}%`} />
              <YAxis type="category" dataKey="name" tick={ATICK} width={72} />
              <Tooltip content={<Tip />} formatter={v => [`${v > 0 ? "+" : ""}${Number(v).toFixed(1)}%`, "差距"]} />
              <ReferenceLine x={0}   stroke="rgba(255,255,255,0.25)" />
              <ReferenceLine x={15}  stroke="#10b981" strokeDasharray="3 3" strokeOpacity={0.5} />
              <ReferenceLine x={-15} stroke="#ef4444" strokeDasharray="3 3" strokeOpacity={0.5} />
              <Bar dataKey="gap" radius={[0, 4, 4, 0]} name="Gap %">
                {data.map((d, i) => (
                  <Cell key={i} fill={d.gap > 15 ? "#10b981" : d.gap < -15 ? "#ef4444" : "#f59e0b"} fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </Section>
  );
}

function OverviewCard({ data }) {
  const chg = data["price_change_1y_%"] ?? data.price_change_1y_pct;
  const isUp = chg >= 0;
  return (
    <Section title="公司概览">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-base sm:text-lg font-bold text-white truncate">{data.name}</p>
          <p className="text-white/40 text-xs mt-0.5 truncate">{data.sector} · {data.industry}</p>
        </div>
        <div className="text-right flex-shrink-0">
          <p className="text-lg sm:text-xl font-bold text-white">${data.price}</p>
          <p className={`text-xs font-semibold ${isUp ? "text-emerald-400" : "text-red-400"}`}>
            {isUp ? "▲" : "▼"} {Math.abs(chg)?.toFixed(1)}% (1年)
          </p>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 border-t border-white/8 pt-3">
        {[["52周高", `$${data["52w_high"]}`], ["52周低", `$${data["52w_low"]}`],
          ["市值", data.market_cap ? `$${(data.market_cap / 1e9).toFixed(1)}B` : "N/A"]].map(([label, val]) => (
          <div key={label}>
            <p className="text-white/30 text-xs">{label}</p>
            <p className="text-white/80 text-xs sm:text-sm font-semibold">{val}</p>
          </div>
        ))}
      </div>
      {data.description && (
        <p className="text-white/35 text-xs leading-relaxed border-t border-white/8 pt-2 line-clamp-4 sm:line-clamp-none">{data.description}</p>
      )}
    </Section>
  );
}

function VerdictCard({ data }) {
  if (!data?.ai_verdict) return null;
  const cfg = VERDICT_CONFIG[data.ai_verdict] || VERDICT_CONFIG.HOLD;
  return (
    <div className={`rounded-2xl bg-gradient-to-r ${cfg.gradient} p-4 sm:p-5 space-y-3 sm:space-y-4`}>
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-3">
          <span className="text-3xl sm:text-4xl font-black">{cfg.icon} {data.ai_verdict}</span>
          <div>
            <p className="text-white/60 text-xs">AI 置信度</p>
            <p className="text-lg sm:text-xl font-bold">{data.ai_confidence}%</p>
          </div>
        </div>
        <div className="text-right text-xs text-white/50 flex-shrink-0">
          <p className="font-mono font-bold text-white/80">{data.ticker}</p>
          <p>${data.price}</p>
          <p className="hidden sm:block">{data.as_of}</p>
        </div>
      </div>
      <div className="w-full h-1.5 bg-white/20 rounded-full">
        <div className="h-full bg-white/50 rounded-full" style={{ width: `${data.ai_confidence}%` }} />
      </div>
      {data.reasoning && (
        <p className="text-white/85 text-xs sm:text-sm leading-relaxed border-t border-white/20 pt-3">{data.reasoning}</p>
      )}
      {(data.supporting_arguments?.length > 0 || data.key_risks?.length > 0) && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
          <div>
            <p className="text-white/50 text-xs uppercase tracking-wider mb-2">支撑因素</p>
            <ul className="space-y-1.5">
              {data.supporting_arguments?.map((a, i) => (
                <li key={i} className="flex gap-1.5 text-xs text-white/75 leading-relaxed">
                  <span className="text-emerald-300 flex-shrink-0">✓</span>{a}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <p className="text-white/50 text-xs uppercase tracking-wider mb-2">主要风险</p>
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

  // ── 三位一体 ──
  if (toolName === "trinity_analysis") {
    return <TrinityCard data={data} />;
  }

  if (["analyze_technicals", "analyze_single_indicator", "analyze_multiple_indicators"].includes(toolName)) {
    const indicators = data.indicators || {};
    if (!Object.keys(indicators).length) return null;
    return (
      <div className="space-y-3">
        <SignalGrid indicators={indicators} />
        <WinRateChart indicators={indicators} />
        {Object.entries(indicators).map(([key, val]) =>
          val.backtest?.trades?.length > 0 ? (
            <EquityCurve key={key} trades={val.backtest.trades} name={val.name || key}
              winRate={val.backtest["win_rate_%"]} totalReturn={val.backtest["total_return_%"]}
              buyHold={val.backtest["buy_hold_%"]} />
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
                  name: k, signal: v.signal,
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
  const isUser     = msg.role === "user";
  const hasTrinity = !!msg.tool_data?.trinity_analysis;
  const [expanded, setExpanded] = useState(!hasTrinity);

  const nonEmptyLines = (msg.content || "").split("\n").filter(l => l.trim());
  const PREVIEW_LINES = 3;
  const needsToggle   = hasTrinity && nonEmptyLines.length > PREVIEW_LINES;
  const displayText   = expanded ? msg.content : nonEmptyLines.slice(0, PREVIEW_LINES).join("\n");

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} gap-2 sm:gap-3`}>
      {!isUser && (
        <div className="w-6 h-6 sm:w-7 sm:h-7 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-xs font-bold flex-shrink-0 mt-1">A</div>
      )}
      <div className={`max-w-[88%] sm:max-w-[85%] flex flex-col gap-2 ${isUser ? "items-end" : "items-start w-full"}`}>
        {msg.tool_calls?.length > 0 && (
          <div className="flex gap-1 overflow-x-auto pb-0.5 max-w-full">
            {msg.tool_calls.map((t, i) => (
              <span key={i} className="inline-flex items-center gap-1 bg-blue-500/15 border border-blue-500/30 text-blue-300 text-xs px-2 py-0.5 rounded-full whitespace-nowrap flex-shrink-0">
                {TOOL_LABELS[t] || t}
              </span>
            ))}
          </div>
        )}
        <div className={`rounded-2xl px-3 sm:px-4 py-2.5 sm:py-3 text-sm leading-relaxed w-full ${
          isUser
            ? "bg-blue-600 text-white rounded-tr-sm max-w-fit"
            : "bg-white/8 border border-white/10 text-white/85 rounded-tl-sm"
        }`}>
          {isUser ? msg.content : (
            <>
              <MDText text={displayText} />
              {needsToggle && (
                <button
                  onClick={() => setExpanded(e => !e)}
                  className="mt-2 flex items-center gap-1 text-xs text-white/35 hover:text-white/65 transition-colors"
                >
                  {expanded
                    ? <><span>收起</span><span className="text-[10px]">▲</span></>
                    : <><span>展开完整分析</span><span className="text-white/25 ml-1">（{nonEmptyLines.length} 行）</span><span className="text-[10px]">▾</span></>
                  }
                </button>
              )}
            </>
          )}
        </div>
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
    <div className="flex gap-2 sm:gap-3 items-start">
      <div className="w-6 h-6 sm:w-7 sm:h-7 rounded-full bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center text-xs font-bold flex-shrink-0">A</div>
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

const QUICK = [
  "AAPL三位一体分析",
  "TSLA的时空状态和主涨段",
  "NVDA综合分析",
  "MSFT布林带+RSI分析",
];

// ─── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  const [messages, setMessages] = useState([{
    role: "assistant",
    content: "你好！我是 **AlphaLens** — 搭载三位一体交易系统的AI股票分析师。\n\n直接告诉我你想分析的股票，例如：\n- AAPL三位一体分析（时空状态+均线+结构）\n- TSLA现在是主涨段吗？该加仓还是做T？\n- NVDA综合分析\n- MSFT布林带+RSI分析",
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
      textareaRef.current.style.height = "44px";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  const send = async (text) => {
    const msg = (text || input).trim();
    if (!msg || loading) return;
    setInput("");
    setMessages(p => [...p, { role: "user", content: msg }]);
    setLoading(true);

    const doFetch = async () => {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 180000); // 3min for trinity
      try {
        const res = await fetch(`${API_BASE}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: msg, session_id: sessionId }),
          signal: controller.signal,
        });
        clearTimeout(timeout);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return await res.json();
      } catch (e) {
        clearTimeout(timeout);
        throw e;
      }
    };

    try {
      let data;
      try {
        data = await doFetch();
      } catch (e) {
        // Cold start: Railway server waking up — retry once after 15s
        if (e.message === "Failed to fetch" || e.name === "TypeError") {
          setMessages(p => [...p, {
            role: "assistant",
            content: "⏳ 服务器冷启动中，15秒后自动重试...",
            _retrying: true,
          }]);
          await new Promise(r => setTimeout(r, 15000));
          setMessages(p => p.filter(m => !m._retrying));
          data = await doFetch();
        } else {
          throw e;
        }
      }
      setMessages(p => [...p, {
        role: "assistant",
        content:    data.response,
        tool_calls: [...new Set(data.tool_calls || [])],
        tool_data:  data.tool_data || {},
      }]);
    } catch (e) {
      const errMsg = e.name === "AbortError"
        ? "⚠️ **请求超时。** 三位一体分析通常需要 2-3 分钟，请稍后重试。"
        : `⚠️ **出错了：** ${e.message}`;
      setMessages(p => p.filter(m => !m._retrying));
      setMessages(p => [...p, { role: "assistant", content: errMsg }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-[#0d1117] text-white flex flex-col"
      style={{ fontFamily: "system-ui,sans-serif", height: "100dvh",
               paddingTop: "env(safe-area-inset-top)", paddingBottom: "env(safe-area-inset-bottom)" }}>

      {/* Header */}
      <div className="border-b border-white/8 px-4 sm:px-6 py-2.5 sm:py-3 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-2 sm:gap-3">
          <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-gradient-to-br from-blue-500 to-violet-600 flex items-center justify-center font-bold text-sm">A</div>
          <div>
            <p className="font-semibold text-sm">AlphaLens <span className="text-white/30 text-xs">☯️ Trinity</span></p>
            <p className="text-white/30 text-xs hidden sm:block">Claude Sonnet · 三位一体 · MCP</p>
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${status === "ok" ? "bg-emerald-400" : status === "error" ? "bg-red-400" : "bg-amber-400 animate-pulse"}`} />
          <span className="text-white/30 text-xs">{status === "ok" ? "在线" : status === "error" ? "离线" : "…"}</span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto overscroll-contain">
        <div className="max-w-3xl mx-auto px-3 sm:px-6 py-4 sm:py-6 space-y-4 sm:space-y-6">
          {messages.map((m, i) => <Bubble key={i} msg={m} />)}
          {loading && <TypingDots />}
          <div ref={endRef} />
        </div>
      </div>

      {/* Quick actions */}
      {messages.length === 1 && (
        <div className="max-w-3xl mx-auto px-3 sm:px-6 pb-2 w-full">
          <div className="flex gap-2 overflow-x-auto pb-1">
            {QUICK.map((q, i) => (
              <button key={i} onClick={() => send(q)}
                className="text-xs bg-white/5 hover:bg-white/10 active:bg-white/15 border border-white/10 hover:border-white/20 rounded-full px-3 py-1.5 text-white/55 hover:text-white/90 transition-all whitespace-nowrap flex-shrink-0 touch-manipulation">
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="border-t border-white/8 px-3 sm:px-6 py-3 sm:py-4 flex-shrink-0">
        <div className="max-w-3xl mx-auto flex gap-2 sm:gap-3 items-end">
          <textarea ref={textareaRef} value={input} onChange={e => setInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === "Enter" && !e.shiftKey && !("ontouchstart" in window)) {
                e.preventDefault(); send();
              }
            }}
            placeholder="输入股票代码或问题，例如：TSLA三位一体分析"
            style={{ minHeight: "44px", maxHeight: "120px", fontSize: "16px" }}
            className="flex-1 bg-[#161b22] border border-white/15 focus:border-blue-400/60 rounded-xl px-3.5 sm:px-4 py-2.5 sm:py-3 text-white placeholder-white/30 resize-none focus:outline-none transition-colors leading-relaxed"
          />
          <button onClick={() => send()} disabled={loading || !input.trim()}
            className="bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:opacity-35 disabled:cursor-not-allowed rounded-xl w-11 h-11 flex items-center justify-center text-base font-semibold transition-colors flex-shrink-0 touch-manipulation">
            {loading
              ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              : "↑"}
          </button>
        </div>
        <p className="text-white/15 text-xs mt-2 text-center">仅供参考，不构成任何投资建议。</p>
      </div>
    </div>
  );
}