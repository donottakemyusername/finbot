"""tools/trinity/indicators.py
================================
Layer 1: 纯Python硬指标计算，无需Claude API。
数据源：yfinance（主）

更新：
- compute_ma_signals 新增 trend_alignment_zh / trend_alignment_bracket（防止Claude排列描述出错）
- compute_structural_levels fallback 从2年低价改为MA233（更有操作意义）
- compute_all_hard_signals 新增 extreme_bars_warning 占位（由state.py注入）
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import yfinance as yf

try:
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def fetch_multi_timeframe(ticker: str) -> dict[str, pd.DataFrame]:
    t = ticker.upper()
    dfs: dict[str, pd.DataFrame] = {}

    def _clean(df):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna()

    try:
        dfs["monthly"] = _clean(yf.download(t, period="5y", interval="1mo", progress=False, auto_adjust=True))
    except Exception:
        dfs["monthly"] = pd.DataFrame()

    try:
        dfs["weekly"] = _clean(yf.download(t, period="5y", interval="1wk", progress=False, auto_adjust=True))
    except Exception:
        dfs["weekly"] = pd.DataFrame()

    try:
        dfs["daily"] = _clean(yf.download(t, period="2y", interval="1d", progress=False, auto_adjust=True))
    except Exception:
        dfs["daily"] = pd.DataFrame()

    try:
        dfs["hourly"] = _clean(yf.download(t, period="730d", interval="60m", progress=False, auto_adjust=True))
    except Exception:
        dfs["hourly"] = pd.DataFrame()

    return dfs


def compute_ma_signals(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 240:
        return {"error": "数据不足，至少需要240个交易日"}
    df = df.copy()
    df["ma55"]  = df["Close"].rolling(55).mean()
    df["ma233"] = df["Close"].rolling(233).mean()
    price  = float(df["Close"].iloc[-1])
    ma55   = float(df["ma55"].iloc[-1])
    ma233  = float(df["ma233"].iloc[-1])

    if price > ma55 > ma233:
        alignment = "bullish"
    elif price < ma55 < ma233:
        alignment = "bearish"
    else:
        alignment = "mixed"

    ma55_slope  = float((df["ma55"].iloc[-1]  - df["ma55"].iloc[-10])  / df["ma55"].iloc[-10])
    ma233_slope = float((df["ma233"].iloc[-1] - df["ma233"].iloc[-10]) / df["ma233"].iloc[-10])
    dist_ma55   = float((price - ma55)  / ma55)
    dist_ma233  = float((price - ma233) / ma233)

    recent = df.tail(10).reset_index(drop=True)
    price_vs_ma55 = [
        {"bar": int(i), "close": round(float(recent["Close"].iloc[i]), 4),
         "ma55": round(float(recent["ma55"].iloc[i]), 4),
         "above_ma55": bool(recent["Close"].iloc[i] > recent["ma55"].iloc[i])}
        for i in range(len(recent))
    ]
    bars_above = sum(1 for r in price_vs_ma55 if r["above_ma55"])

    # ── 中文排列描述（供prompt直接读取，防止Claude自行推断出错）──────────────
    alignment_zh = {"bullish": "多头排列", "bearish": "空头排列", "mixed": "混沌排列"}[alignment]

    # 混沌排列的括号注释，精确描述价格夹在哪两条均线之间
    if alignment == "mixed":
        if price > ma233 and price < ma55:
            # 价格在MA233上方、MA55下方：MA233 < 价格 < MA55
            alignment_bracket = "（MA233 < 价格 < MA55）"
        elif price > ma55 and price < ma233:
            # 价格在MA55上方、MA233下方（均线倒置，极少见）
            alignment_bracket = "（MA55 < 价格 < MA233）"
        elif price < ma233 and price < ma55:
            # 价格在两条均线下方，但MA233 > MA55（罕见倒置状态）
            alignment_bracket = "（价格 < MA233，均线倒置）"
        else:
            # 价格在两线上方，但MA55 < MA233（均线本身倒置，长期趋势未修复）
            alignment_bracket = "（均线倒置：MA55 < MA233，长期趋势未修复）"
    else:
        alignment_bracket = ""

    return {
        "current_price":           round(price, 4),
        "ma55":                    round(ma55, 4),
        "ma233":                   round(ma233, 4),
        "trend_alignment":         alignment,
        "trend_alignment_zh":      alignment_zh,        # ← Claude直接读取，禁止自行推断排列
        "trend_alignment_bracket": alignment_bracket,   # ← 混沌排列的精确括号注释
        "ma_inverted":             bool(ma55 < ma233),  # ← 均线倒置标志（MA55 < MA233）
        "ma55_slope":              round(ma55_slope, 6),
        "ma233_slope":             round(ma233_slope, 6),
        "dist_from_ma55":          round(dist_ma55, 4),
        "dist_from_ma233":         round(dist_ma233, 4),
        "overextension_hard":      bool(abs(dist_ma55) > 0.15 or abs(dist_ma233) > 0.40),
        "price_vs_ma55_last10":    price_vs_ma55,
        "bars_above_ma55_last10":  bars_above,
        "bars_below_ma55_last10":  10 - bars_above,
    }


def compute_macd_signals(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 60:
        return {"error": "数据不足"}
    df = df.copy()
    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["dif"]      = exp12 - exp26
    df["dea"]      = df["dif"].ewm(span=9, adjust=False).mean()
    df["macd_bar"] = 2 * (df["dif"] - df["dea"])
    dif_now  = float(df["dif"].iloc[-1])
    dea_now  = float(df["dea"].iloc[-1])
    dif_prev = float(df["dif"].iloc[-2])
    dea_prev = float(df["dea"].iloc[-2])
    if dif_prev < dea_prev and dif_now > dea_now:
        cross_signal = "golden_cross"
    elif dif_prev > dea_prev and dif_now < dea_now:
        cross_signal = "death_cross"
    else:
        cross_signal = "none"
    price = float(df["Close"].iloc[-1])
    # 临轴金叉/死叉：DIF与DEA发生交叉，且DIF和DEA都在零轴附近（各自绝对值 < 价格的0.5%）
    zero_axis_cross = (
        cross_signal != "none"
        and abs(dif_now) < price * 0.005
        and abs(dea_now) < price * 0.005
    )
    recent60 = df.tail(60)
    dif_crossed = bool(recent60["dif"].max() > 0 and recent60["dif"].min() < 0)
    dea_crossed = bool(recent60["dea"].max() > 0 and recent60["dea"].min() < 0)
    return {
        "dif_current":          round(dif_now, 6),
        "dea_current":          round(dea_now, 6),
        "macd_bar_current":     round(float(df["macd_bar"].iloc[-1]), 6),
        "zero_axis_position":   "above" if dif_now > 0 else "below",
        "cross_signal":         cross_signal,
        "zero_axis_cross":      zero_axis_cross,
        "dif_crossed_zero_60d": dif_crossed,
        "dea_crossed_zero_60d": dea_crossed,
        "adjustment_sufficient": dif_crossed and dea_crossed,
        "macd_bar_history_60":  [round(float(x), 6) for x in df["macd_bar"].tail(60).tolist()],
    }


def compute_bollinger_trinity(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> dict:
    if df.empty or len(df) < period + 5:
        return {"error": "数据不足"}
    df = df.copy()
    df["bb_mid"]   = df["Close"].rolling(period).mean()
    df["bb_std"]   = df["Close"].rolling(period).std()
    df["bb_upper"] = df["bb_mid"] + std * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - std * df["bb_std"]
    price = float(df["Close"].iloc[-1])
    mid   = float(df["bb_mid"].iloc[-1])
    upper = float(df["bb_upper"].iloc[-1])
    lower = float(df["bb_lower"].iloc[-1])
    band_width = upper - lower
    position   = (price - lower) / band_width if band_width > 0 else 0.5
    below_mid_2bars = all(
        float(df["Close"].iloc[i]) < float(df["bb_mid"].iloc[i]) for i in [-2, -1]
    )
    return {
        "bb_upper":        round(upper, 4),
        "bb_mid":          round(mid, 4),
        "bb_lower":        round(lower, 4),
        "current_price":   round(price, 4),
        "price_position":  round(position, 4),
        "above_mid":       bool(price > mid),
        "below_mid_2bars": bool(below_mid_2bars),
    }


def _find_peaks_simple(values: np.ndarray, distance: int = 5) -> list[int]:
    peaks = []
    for i in range(distance, len(values) - distance):
        window = values[i - distance: i + distance + 1]
        if values[i] == max(window):
            peaks.append(i)
    return peaks


def compute_turning_points_and_divergence(df: pd.DataFrame) -> dict:
    if df.empty or len(df) < 40:
        return {"error": "数据不足"}
    df = df.copy()
    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    dif   = exp12 - exp26
    dea   = dif.ewm(span=9, adjust=False).mean()
    bars  = 2 * (dif - dea)
    closes   = df["Close"].values.astype(float)
    bar_vals = bars.values.astype(float)

    if HAS_SCIPY:
        std_c = float(np.std(closes))
        peaks, _ = find_peaks(closes, distance=10, prominence=std_c * 0.5)
        troughs, _ = find_peaks(-closes, distance=10, prominence=std_c * 0.5)
        peaks = list(peaks); troughs = list(troughs)
    else:
        peaks   = _find_peaks_simple(closes)
        troughs = _find_peaks_simple(-closes)

    # 过滤掉距末尾太近的峰/谷（< 5根K线）
    min_tail_dist = 5
    n_bars = len(closes)
    peaks   = [p for p in peaks   if p < n_bars - min_tail_dist]
    troughs = [t for t in troughs if t < n_bars - min_tail_dist]

    def get_last2(idx): return list(idx[-2:]) if len(idx) >= 2 else []

    last_peaks   = get_last2(peaks)
    last_troughs = get_last2(troughs)

    top_div = None
    if len(last_peaks) == 2:
        p1, p2 = last_peaks
        b1, b2 = float(bar_vals[p1]), float(bar_vals[p2])
        top_div = {
            "peak1_price":      round(float(closes[p1]), 4),
            "peak2_price":      round(float(closes[p2]), 4),
            "peak1_macd_bar":   round(b1, 6),
            "peak2_macd_bar":   round(b2, 6),
            "price_new_high":   bool(closes[p2] > closes[p1]),
            "macd_bar_lower":   bool(b2 < b1),
            "price_change_pct": round(float((closes[p2] - closes[p1]) / closes[p1]), 4),
            "macd_change_pct":  round(float((b2 - b1) / abs(b1)) if b1 != 0 else 0, 4),
        }

    bot_div = None
    if len(last_troughs) == 2:
        t1, t2 = last_troughs
        b1, b2 = float(bar_vals[t1]), float(bar_vals[t2])
        bot_div = {
            "trough1_price":    round(float(closes[t1]), 4),
            "trough2_price":    round(float(closes[t2]), 4),
            "trough1_macd_bar": round(b1, 6),
            "trough2_macd_bar": round(b2, 6),
            "price_new_low":    bool(closes[t2] < closes[t1]),
            "macd_bar_smaller": bool(abs(b2) < abs(b1)),
            "price_change_pct": round(float((closes[t2] - closes[t1]) / closes[t1]), 4),
            "macd_change_pct":  round(float((abs(b2) - abs(b1)) / abs(b1)) if b1 != 0 else 0, 4),
        }

    all_tp = sorted(
        [(int(i), "peak") for i in peaks[-6:]] + [(int(i), "trough") for i in troughs[-6:]],
        key=lambda x: x[0]
    )
    turning_points = [
        {"bar_index": idx, "type": tp, "price": round(float(closes[idx]), 4),
         "macd_bar": round(float(bar_vals[idx]), 6)}
        for idx, tp in all_tp
    ]

    # ── 硬性有效性预判 ────────────────────────────────────────────────────────
    current_price   = float(closes[-1])
    current_bar     = float(bar_vals[-1])

    # ── 实时顶背离预警：当前价已超越最近历史高点，但MACD动能更弱 ──────────────
    # 注：这不是"确认"背离（当前点尚未形成回落确认），而是早期警示信号
    live_top_div_warning = False
    live_top_div_note    = ""
    if len(last_peaks) >= 1:
        ref_idx   = last_peaks[-1]
        ref_price = float(closes[ref_idx])
        ref_bar   = float(bar_vals[ref_idx])
        if current_price > ref_price and current_bar < ref_bar:
            bar_drop_pct = abs((current_bar - ref_bar) / abs(ref_bar)) * 100 if ref_bar != 0 else 0
            live_top_div_warning = True
            live_top_div_note = (
                f"⚠️ 实时顶背离预警：当前价${current_price:.2f}已超越前高${ref_price:.2f}，"
                f"但MACD柱动能缩减{bar_drop_pct:.1f}%（前高柱={ref_bar:+.4f}，当前={current_bar:+.4f}）。"
                f"尚未确认（需等待回落），但需警惕见顶风险。"
            )

    top_hard_valid = bool(
        top_div is not None
        and top_div["price_new_high"]
        and top_div["macd_bar_lower"]
    )
    bot_hard_valid = bool(
        bot_div is not None
        and bot_div["price_new_low"]
        and bot_div["macd_bar_smaller"]
    )

    # ── 过期背离检测 ──────────────────────────────────────────────────────────
    top_div_stale = False
    if top_hard_valid and top_div is not None:
        peak2_px = top_div["peak2_price"]
        if current_price < peak2_px * 0.85 or current_price > peak2_px * 1.20:
            top_hard_valid = False
            top_div_stale  = True

    bot_div_stale = False
    if bot_hard_valid and bot_div is not None:
        trough2_px = bot_div["trough2_price"]
        if current_price > trough2_px * 1.15:
            bot_hard_valid = False
            bot_div_stale  = True

    # ── 预计算背离说明文字（防止Claude逻辑倒置）──────────────────────────────
    def _top_note() -> str:
        if top_div_stale:
            return "顶背离形成于历史高位，已引发调整完成，不是当前风险点"
        if top_div is None:
            return "近期高点结构不完整，顶背离暂时不成立"
        if top_hard_valid:
            pct = abs(top_div.get("macd_change_pct", 0)) * 100
            return f"价格创新高但MACD动能缩减{pct:.1f}%，顶背离有效"
        if top_div.get("price_new_high"):
            return "价格虽创新高但MACD动能同步创高，顶背离不成立"
        p1 = top_div.get("peak1_price", "?")
        p2 = top_div.get("peak2_price", "?")
        return f"次高点(${p2})未超越前高(${p1})，无顶背离条件（注：当前价若已突破前高，则为新一轮上行，非背离结构）"

    def _bot_note() -> str:
        if bot_div_stale:
            return "底背离形成于历史低位，已引发反弹完成，不是当前买点依据"
        if bot_div is None:
            return "近期低点结构不完整，底背离暂时不成立"
        if bot_hard_valid:
            pct = abs(bot_div.get("macd_change_pct", 0)) * 100
            return f"价格创新低但MACD面积缩减{pct:.1f}%，底背离有效"
        if bot_div.get("price_new_low"):
            return "价格虽创新低但MACD动能同步扩大，底背离不成立"
        return "当前价未创新低，底背离不成立"

    return {
        "top_divergence_raw":        top_div,
        "bot_divergence_raw":        bot_div,
        "top_divergence_hard_valid": top_hard_valid,
        "bot_divergence_hard_valid": bot_hard_valid,
        "top_div_stale":             top_div_stale,
        "bot_div_stale":             bot_div_stale,
        "live_top_div_warning":      live_top_div_warning,
        "live_top_div_note":         live_top_div_note,
        "top_divergence_note_py":    _top_note(),
        "bot_divergence_note_py":    _bot_note(),
        "turning_points":            turning_points,
        "recent_30_closes":          [round(float(x), 4) for x in closes[-30:]],
        # 背离形成时第二拐点的K线索引，供 compute_divergence_maturity 计算成熟度
        "top_div_peak2_bar":   int(last_peaks[1])   if len(last_peaks)   >= 2 else None,
        "bot_div_trough2_bar": int(last_troughs[1]) if len(last_troughs) >= 2 else None,
    }


def compute_structural_levels(df: pd.DataFrame, div_result: dict) -> dict:
    """
    从结构拐点中提取关键支撑和压力位。
    优先级：结构高低点 > MA233/MA55 fallback > 2年极值（最后备用）。

    修复：fallback改为MA233（比2年历史低点更有操作意义）。
    AVGO案例：当价格在历史高位回调时，2年低点（$145）毫无参考价值，
    MA233（$306）才是真正的中期支撑。
    """
    if df.empty:
        return {"key_support": None, "key_resistance": None,
                "support_source": "none", "resistance_source": "none",
                "long_stop_loss": None, "short_stop_loss": None}

    closes = df["Close"].values.astype(float)
    price  = float(closes[-1])

    # 预计算MA233和MA55供fallback使用
    ma233_val = None
    ma55_val  = None
    if len(df) >= 233:
        v = df["Close"].rolling(233).mean().iloc[-1]
        if pd.notna(v):
            ma233_val = round(float(v), 4)
    if len(df) >= 55:
        v = df["Close"].rolling(55).mean().iloc[-1]
        if pd.notna(v):
            ma55_val = round(float(v), 4)

    turning_points = div_result.get("turning_points", [])

    # key_resistance = 当前价上方价格最近的结构高点（前高）
    peaks_above   = [tp for tp in turning_points if tp["type"] == "peak"   and tp["price"] > price]
    # key_support   = 当前价下方价格最近的结构低点（前低）
    troughs_below = [tp for tp in turning_points if tp["type"] == "trough" and tp["price"] < price]

    # ── 压力位 ────────────────────────────────────────────────────────────────
    key_resistance    = None
    resistance_source = "none"
    if peaks_above:
        nearest_peak   = min(peaks_above, key=lambda x: x["price"])
        key_resistance = nearest_peak["price"]
        resistance_source = "structural_peak"
    else:
        # fallback1：MA55上方5%（只有在仍高于当前价时才有意义）
        if ma55_val is not None and ma55_val * 1.05 > price:
            key_resistance = round(ma55_val * 1.05, 4)
            resistance_source = "ma55_plus5pct_fallback"
        # fallback2：当前价上方5%（价格已远超MA55，用当前价做参考）
        elif ma55_val is not None:
            key_resistance = round(price * 1.05, 4)
            resistance_source = "price_plus5pct_fallback"
        else:
            key_resistance = round(float(df["High"].max()), 4)
            resistance_source = "period_high"

    # ── 支撑位 ────────────────────────────────────────────────────────────────
    key_support    = None
    support_source = "none"

    # 也收集"刚跌破"的近距离谷底（高于价格但 <3%）
    all_troughs = [tp for tp in turning_points if tp["type"] == "trough"]
    near_troughs_above = [tp for tp in all_troughs
                          if tp["price"] >= price and (tp["price"] - price) / price < 0.03]

    # 候选集合：严格低于价格的谷底 + 刚被跌破的近距离谷底
    candidates = list(troughs_below) + near_troughs_above
    if candidates:
        # 选离当前价最近的谷底
        best = min(candidates, key=lambda x: abs(x["price"] - price))
        key_support    = best["price"]
        support_source = ("structural_trough_just_broken"
                          if best["price"] >= price
                          else "structural_trough")
    else:
        # fallback1：MA233（中期均线支撑，比2年历史低点有操作意义）
        if ma233_val is not None and ma233_val < price:
            key_support    = ma233_val
            support_source = "ma233_fallback"
        # fallback2：MA55（若MA233也在价格上方，如极端上涨票）
        elif ma55_val is not None and ma55_val < price:
            key_support    = ma55_val
            support_source = "ma55_fallback"
        else:
            # 最后备用：2年最低价
            key_support    = round(float(df["Low"].min()), 4)
            support_source = "period_low"

    # 预计算止损价（Claude直接读取，禁止自行重算）
    long_stop_loss  = round(key_support  * 0.97, 2) if key_support  is not None else None
    short_stop_loss = round(key_resistance * 1.03, 2) if key_resistance is not None else None

    return {
        "key_support":        key_support,
        "key_resistance":     key_resistance,
        "support_source":     support_source,
        "resistance_source":  resistance_source,
        "long_stop_loss":     long_stop_loss,
        "short_stop_loss":    short_stop_loss,
    }


def compute_all_hard_signals(df: pd.DataFrame) -> dict:
    ma       = compute_ma_signals(df)
    macd     = compute_macd_signals(df)
    bb       = compute_bollinger_trinity(df)
    div      = compute_turning_points_and_divergence(df)
    levels   = compute_structural_levels(df, div)
    breakout = compute_ma_breakout_type(ma)

    # Python硬判断：结构分类 + 背离成熟度 + 关键K线
    struct_cls   = compute_structure_classification(div.get("turning_points", []), len(df))
    div_maturity = compute_divergence_maturity(len(df), div)
    key_candles  = detect_key_candles(df, levels.get("key_support"))

    combined = {**ma, **macd, **bb, **div, **levels, **breakout,
                **struct_cls, **div_maturity, **key_candles}

    # 课程硬规则：调整不充分时底背离无效
    if not combined.get("adjustment_sufficient", False):
        combined["bot_divergence_hard_valid"] = False

    return combined


def compute_ma_breakout_type(ma_signals: dict) -> dict:
    """
    Python硬判断均线突破类型（ABCD），减少Claude不确定性。

    A类（典型突破）：bars_above>=8 且 dist>2% （或反向）
    B类（慢速/盘整突破）：bars 3-7 或 刚突破(1-2)，dist在±2%内
    C类（回抽突破）：price_vs_ma55序列先连续同向后1-2根反向再恢复
    D类（反向测试）：全部在同侧，dist接近0但未穿越
    """
    if "error" in ma_signals:
        return {"ma_breakout_type_py": "unknown", "ma_breakout_direction_py": "none"}

    bars_above = ma_signals.get("bars_above_ma55_last10", 0)
    bars_below = ma_signals.get("bars_below_ma55_last10", 0)
    dist       = ma_signals.get("dist_from_ma55", 0)
    pv_list    = ma_signals.get("price_vs_ma55_last10", [])

    above_flags = [r["above_ma55"] for r in pv_list] if pv_list else []

    # ── C类检测：序列先连续同向≥ 5根，后1-2根反向，再恢复同向 ────────
    def _detect_c_type(flags: list[bool]) -> tuple[bool, str]:
        if len(flags) < 8:
            return False, "none"
        # 向上回抽：多数True，中间出现少量False，最后回到True
        for start in range(len(flags) - 7):
            seg = flags[start:start + 8]
            # 前5根全True，第6-7根False，第8根True
            if all(seg[:5]) and not all(seg[5:7]) and seg[-1]:
                return True, "up"
            # 前5根全False，第6-7根True，第8根False
            if not any(seg[:5]) and any(seg[5:7]) and not seg[-1]:
                return True, "down"
        return False, "none"

    is_c, c_dir = _detect_c_type(above_flags)
    if is_c:
        return {"ma_breakout_type_py": "C", "ma_breakout_direction_py": c_dir}

    # ── D类：全部在同侧，距离接近0（±2%内），未穿越 ───────────────
    if (bars_above == 10 or bars_below == 10) and abs(dist) < 0.02:
        direction = "up" if bars_above == 10 else "down"
        return {"ma_breakout_type_py": "D", "ma_breakout_direction_py": direction}

    # ── A类：强势突破，bars>=8 且 dist > 2% 且方向一致 ──────────────
    if bars_above >= 8 and dist > 0.02:
        return {"ma_breakout_type_py": "A", "ma_breakout_direction_py": "up"}
    if bars_below >= 8 and dist < -0.02:
        return {"ma_breakout_type_py": "A", "ma_breakout_direction_py": "down"}

    # ── B类：慢速盘整突破（中间地带） ────────────────────────────
    if abs(dist) < 0.02:
        direction = "up" if dist >= 0 else "down"
        return {"ma_breakout_type_py": "B", "ma_breakout_direction_py": direction}

    # ── 其他默认为A类（距离远且方向明确） ──────────────────────
    direction = "up" if dist > 0 else "down"
    return {"ma_breakout_type_py": "A", "ma_breakout_direction_py": direction}


# ─────────────────────────────────────────────────────────────────────────────
# 纯Python计算的摘要（v2架构：从Claude输出中移除，改由Python确定性生成）
# ─────────────────────────────────────────────────────────────────────────────

def compute_divergence_summary(hard_signals: dict) -> dict:
    """100% Python确定性计算背离摘要，不再依赖Claude判断。"""
    top_valid = hard_signals.get("top_divergence_hard_valid", False)
    bot_valid = hard_signals.get("bot_divergence_hard_valid", False)

    # divergence_type
    if top_valid and bot_valid:
        div_type = "both"
    elif top_valid:
        div_type = "top"
    elif bot_valid:
        div_type = "bottom"
    else:
        div_type = "none"

    # divergence_strength（从原始百分比计算）
    strength = "none"
    raw = None
    if top_valid:
        raw = hard_signals.get("top_divergence_raw") or {}
    elif bot_valid:
        raw = hard_signals.get("bot_divergence_raw") or {}
    if raw:
        p_pct = abs(raw.get("price_change_pct", 0))
        m_pct = abs(raw.get("macd_change_pct", 0))
        if p_pct > 0.05 and m_pct > 0.30:
            strength = "strong"
        elif p_pct > 0.02 or m_pct > 0.15:
            strength = "medium"
        else:
            strength = "weak"

    # divergence_note（直接用Python预判文字）
    if top_valid:
        note = hard_signals.get("top_divergence_note_py", "")
    elif bot_valid:
        note = hard_signals.get("bot_divergence_note_py", "")
    else:
        # 无有效背离时，取不成立的说明
        note = hard_signals.get("top_divergence_note_py", "") or hard_signals.get("bot_divergence_note_py", "")

    return {
        "top_divergence_valid": top_valid,
        "bot_divergence_valid": bot_valid,
        "divergence_strength":  strength,
        "divergence_type":      div_type,
        "divergence_note":      note,
    }


def compute_ma_analysis_summary(hard_signals: dict) -> dict:
    """100% Python确定性计算均线分析摘要，不再依赖Claude判断。"""
    breakout_type = hard_signals.get("ma_breakout_type_py", "unknown")
    breakout_dir  = hard_signals.get("ma_breakout_direction_py", "none")
    breakout_valid = breakout_type not in ("none", "unknown")
    overextension = hard_signals.get("overextension_hard", False)
    dist = hard_signals.get("dist_from_ma55", 0)
    alignment_zh = hard_signals.get("trend_alignment_zh", "混沌排列")
    alignment_bracket = hard_signals.get("trend_alignment_bracket", "")

    # pullback判断
    pullback = False
    pullback_side = "none"
    if breakout_type in ("A", "C") and abs(dist) < 0.05:
        pullback = True
        pullback_side = "buy" if breakout_dir == "up" else "sell"
    elif breakout_type == "B":
        pullback = True
        pullback_side = "buy" if dist >= 0 else "sell"

    # ma_note模板生成
    type_desc = {
        "A": "典型突破（向上）" if breakout_dir == "up" else "典型跌破（向下）",
        "B": "慢速盘整突破（向上）" if breakout_dir == "up" else "慢速盘整跌破（向下）",
        "C": "突破后回抽确认（向上）" if breakout_dir == "up" else "跌破后反抽确认（向下）",
        "D": "反向测试MA55未穿越",
    }.get(breakout_type, "")

    ma_note = f"{alignment_zh}{alignment_bracket}"
    if breakout_valid and type_desc:
        ma_note += f"，{breakout_type}类{type_desc}"
    if overextension:
        ma_note += "，⚠️价格过度偏离均线"

    return {
        "ma55_breakout_type":      breakout_type,
        "ma55_breakout_direction": breakout_dir,
        "ma55_breakout_valid":     breakout_valid,
        "pullback_opportunity":    pullback,
        "pullback_side":           pullback_side,
        "overextension_warning":   overextension,
        "ma_note":                 ma_note,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 新增：结构分类 / 背离成熟度 / 关键K线（v2+）
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_alternation(turning_points: list[dict]) -> list[dict]:
    """确保拐点序列严格高低交替；同类型相邻时保留最极端值。"""
    if not turning_points:
        return []
    clean = [turning_points[0].copy()]
    for tp in turning_points[1:]:
        if tp["type"] == clean[-1]["type"]:
            if tp["type"] == "peak" and tp["price"] > clean[-1]["price"]:
                clean[-1] = tp.copy()
            elif tp["type"] == "trough" and tp["price"] < clean[-1]["price"]:
                clean[-1] = tp.copy()
        else:
            clean.append(tp.copy())
    return clean


def compute_structure_classification(turning_points: list[dict], n_bars: int) -> dict:
    """
    Python硬判断结构分类（ABCD）。

    A类（五段式）：≥6个拐点，第三段幅度 > 第一段 × 1.5
    B类（双平台）：相邻同向拐点价格差 < 3%（双高或双低）
    C类（单平台）：拐点不足或不满足A/B条件的过渡结构
    D类（三段式）：恰好4个拐点；若第三段>第一段×1.5则 d_to_a_py=True

    文档原则：D4之后最有可能转化为A5，第三段超越第一段×1.5是关键判据。
    """
    clean = _enforce_alternation(turning_points)
    n = len(clean)

    if n < 2:
        return {
            "structure_type_py":          "unknown",
            "structure_current_stage_py": "unknown",
            "structure_confidence_py":    "none",
            "structure_d_to_a_py":        False,
            "structure_note_py":          "拐点不足，无法判断结构",
        }

    # 各段幅度（绝对价格距离，方向无关）
    segments = [abs(clean[i]["price"] - clean[i - 1]["price"]) for i in range(1, n)]
    seg1 = segments[0] if len(segments) >= 1 else 0.0
    seg3 = segments[2] if len(segments) >= 3 else 0.0

    # 当前处于第几拐点（即已形成n个拐点，正在运行第n+1段）
    current_stage = str(n)

    # ── D类：恰好4个拐点 ──────────────────────────────────────────────────
    if n == 4:
        d_to_a = bool(seg1 > 0 and seg3 > seg1 * 1.5)
        return {
            "structure_type_py":          "D",
            "structure_current_stage_py": current_stage,
            "structure_confidence_py":    "high",
            "structure_d_to_a_py":        d_to_a,
            "structure_note_py": (
                f"D4已现，第三段({seg3:.2f})>第一段({seg1:.2f})×1.5，D→A转化形成中"
                if d_to_a else
                f"D类三段式，第四拐点位，等待第五段方向确认"
            ),
        }

    # ── A类（确认）：≥6个拐点且第三段>第一段×1.5 ────────────────────────
    if n >= 6 and seg1 > 0 and seg3 > seg1 * 1.5:
        return {
            "structure_type_py":          "A",
            "structure_current_stage_py": current_stage,
            "structure_confidence_py":    "high",
            "structure_d_to_a_py":        False,
            "structure_note_py":          f"A类五段式已确认，第三段({seg3:.2f})>第一段({seg1:.2f})×1.5",
        }

    # ── B类：相邻同向拐点差<3% ────────────────────────────────────────────
    peaks   = [tp for tp in clean if tp["type"] == "peak"]
    troughs = [tp for tp in clean if tp["type"] == "trough"]

    has_double_peak = (
        len(peaks) >= 2 and
        abs(peaks[-1]["price"] - peaks[-2]["price"]) / max(peaks[-2]["price"], 0.01) < 0.03
    )
    has_double_trough = (
        len(troughs) >= 2 and
        abs(troughs[-1]["price"] - troughs[-2]["price"]) / max(troughs[-2]["price"], 0.01) < 0.03
    )

    if has_double_peak or has_double_trough:
        platform_desc = "高点双平台" if has_double_peak else "低点双平台"
        return {
            "structure_type_py":          "B",
            "structure_current_stage_py": current_stage,
            "structure_confidence_py":    "medium",
            "structure_d_to_a_py":        False,
            "structure_note_py":          f"B类双平台（{platform_desc}，相邻差<3%）",
        }

    # ── A类（进行中）：≥5个拐点，第三段>第一段但<1.5× ────────────────────
    if n >= 5 and seg1 > 0 and seg3 > seg1:
        return {
            "structure_type_py":          "A",
            "structure_current_stage_py": current_stage,
            "structure_confidence_py":    "medium",
            "structure_d_to_a_py":        False,
            "structure_note_py":          f"A类进行中（第三段{seg3:.2f}>第一段{seg1:.2f}，待确认×1.5）",
        }

    # ── C类：默认 ─────────────────────────────────────────────────────────
    return {
        "structure_type_py":          "C",
        "structure_current_stage_py": current_stage,
        "structure_confidence_py":    "low",
        "structure_d_to_a_py":        False,
        "structure_note_py":          f"C类单平台或过渡结构（{n}个拐点，方向待定）",
    }


def compute_divergence_maturity(n_bars: int, div_result: dict) -> dict:
    """
    背离成熟度评估（课程："背离是过程，矛盾激化时才有分析价值"）。

    forming      (<3根K线)：背离刚形成，不宜操作
    intensifying (3-10根)：背离激化中，开始具有分析价值
    mature       (>10根)： 背离成熟，分析价值高
    """
    def _classify(bars: int | None) -> str:
        if bars is None:
            return "unknown"
        if bars < 3:
            return "forming"
        elif bars <= 10:
            return "intensifying"
        return "mature"

    top_valid    = div_result.get("top_divergence_hard_valid", False)
    top_peak2    = div_result.get("top_div_peak2_bar")
    top_bars     = max(0, n_bars - 1 - top_peak2) if (top_valid and top_peak2 is not None) else None
    top_maturity = _classify(top_bars) if top_valid else "none"

    bot_valid    = div_result.get("bot_divergence_hard_valid", False)
    bot_trough2  = div_result.get("bot_div_trough2_bar")
    bot_bars     = max(0, n_bars - 1 - bot_trough2) if (bot_valid and bot_trough2 is not None) else None
    bot_maturity = _classify(bot_bars) if bot_valid else "none"

    return {
        "top_div_maturity":   top_maturity,
        "top_div_bars_since": top_bars,
        "bot_div_maturity":   bot_maturity,
        "bot_div_bars_since": bot_bars,
    }


def detect_key_candles(df: pd.DataFrame, key_support: float | None = None) -> dict:
    """
    关键K线识别（Golden Candle / 黄金棒）。

    课程条件（三位一体高级课程·关键K）：
    1. 阳包阴（包住前阴线≥50%）或显著下影线（>30%实体区间）
    2. 量能缩量（相比前一根K线）
    3. 下一根K线不破低点（确认信号）
    4. 临近结构支撑位（±5%，加分项）

    文档："缩量阳包阴，下一根不破低点 → 可补仓"
    """
    if df.empty or len(df) < 5:
        return {"key_candles_last20": [], "latest_golden_candle": None,
                "golden_candle_near_support": False}

    df = df.copy()
    has_vol  = "Volume" in df.columns and float(df["Volume"].sum()) > 0
    lookback = min(20, len(df) - 1)
    start_i  = len(df) - lookback

    golden_candles = []
    for i in range(start_i, len(df)):
        if i == 0:
            continue
        o, c = float(df["Open"].iloc[i]),  float(df["Close"].iloc[i])
        h, l = float(df["High"].iloc[i]),  float(df["Low"].iloc[i])
        o_prev = float(df["Open"].iloc[i - 1])
        c_prev = float(df["Close"].iloc[i - 1])

        is_bullish  = c > o
        full_range  = h - l if h > l else 0.001
        lower_shad  = (min(o, c) - l) / full_range  # 下影线占比

        # 条件1a: 阳包阴（包住前阴线≥50%）
        prev_bearish   = c_prev < o_prev
        engulf_half    = (
            is_bullish and prev_bearish
            and c > o_prev                            # 收盘超越前一根开盘
            and o < c_prev                            # 开盘低于前一根收盘
            and (c - o) > (o_prev - c_prev) * 0.5    # 包住≥50%
        )

        # 条件1b: 显著下影线（>30%）
        long_shadow = is_bullish and lower_shad > 0.30

        if not (engulf_half or long_shadow):
            continue

        # 条件2: 缩量
        vol_shrink = False
        if has_vol and i > 0:
            v_c = float(df["Volume"].iloc[i])
            v_p = float(df["Volume"].iloc[i - 1])
            vol_shrink = (v_c < v_p) if v_p > 0 else False

        # 条件3: 下一根不破低点
        confirmed = False
        if i + 1 < len(df):
            confirmed = float(df["Low"].iloc[i + 1]) >= l

        # 加分：临近支撑位（±5%）
        near_support = (
            key_support is not None and key_support > 0
            and abs(l - key_support) / key_support < 0.05
        )

        golden_candles.append({
            "bar_index":        i,
            "price":            round(c, 4),
            "pattern":          "bullish_engulfing" if engulf_half else "lower_shadow",
            "lower_shadow_pct": round(lower_shad, 3),
            "vol_shrink":       vol_shrink,
            "near_support":     near_support,
            "confirmed":        confirmed,
        })

    confirmed_list = [k for k in golden_candles if k["confirmed"]]
    latest   = confirmed_list[-1] if confirmed_list else (golden_candles[-1] if golden_candles else None)
    near_any = any(k["near_support"] for k in golden_candles[-3:]) if golden_candles else False

    return {
        "key_candles_last20":         golden_candles,
        "latest_golden_candle":       latest,
        "golden_candle_near_support": near_any,
    }