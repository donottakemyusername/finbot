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
            alignment_bracket = ""
    else:
        alignment_bracket = ""

    return {
        "current_price":           round(price, 4),
        "ma55":                    round(ma55, 4),
        "ma233":                   round(ma233, 4),
        "trend_alignment":         alignment,
        "trend_alignment_zh":      alignment_zh,        # ← Claude直接读取，禁止自行推断排列
        "trend_alignment_bracket": alignment_bracket,   # ← 混沌排列的精确括号注释
        "ma55_slope":              round(ma55_slope, 6),
        "ma233_slope":             round(ma233_slope, 6),
        "dist_from_ma55":          round(dist_ma55, 4),
        "dist_from_ma233":         round(dist_ma233, 4),
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

    # 过滤掉距末尾太近的峰/谷（< 10根K线）
    min_tail_dist = 10
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
    current_price = float(closes[-1])

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
        return "当前价未超越前高，顶背离不成立"

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
        "top_divergence_note_py":    _top_note(),
        "bot_divergence_note_py":    _bot_note(),
        "turning_points":            turning_points,
        "recent_30_closes":          [round(float(x), 4) for x in closes[-30:]],
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

    # key_resistance = 当前价上方最近的结构高点（前高）
    peaks_above   = [tp for tp in turning_points if tp["type"] == "peak"   and tp["price"] > price]
    # key_support   = 当前价下方最近的结构低点（前低）
    troughs_below = [tp for tp in turning_points if tp["type"] == "trough" and tp["price"] < price]

    # ── 压力位 ────────────────────────────────────────────────────────────────
    key_resistance    = None
    resistance_source = "none"
    if peaks_above:
        nearest_peak   = max(peaks_above, key=lambda x: x["bar_index"])
        key_resistance = nearest_peak["price"]
        resistance_source = "structural_peak"
    else:
        # fallback1：MA55上方5%（比历史最高价更有近期操作意义）
        if ma55_val is not None:
            key_resistance = round(ma55_val * 1.05, 4)
            resistance_source = "ma55_plus5pct_fallback"
        else:
            key_resistance = round(float(df["High"].max()), 4)
            resistance_source = "period_high"

    # ── 支撑位 ────────────────────────────────────────────────────────────────
    key_support    = None
    support_source = "none"
    if troughs_below:
        nearest_trough = max(troughs_below, key=lambda x: x["bar_index"])
        key_support    = nearest_trough["price"]
        support_source = "structural_trough"
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
    ma     = compute_ma_signals(df)
    macd   = compute_macd_signals(df)
    bb     = compute_bollinger_trinity(df)
    div    = compute_turning_points_and_divergence(df)
    levels = compute_structural_levels(df, div)
    combined = {**ma, **macd, **bb, **div, **levels}

    # 课程硬规则：调整不充分时底背离无效
    if not combined.get("adjustment_sufficient", False):
        combined["bot_divergence_hard_valid"] = False

    return combined