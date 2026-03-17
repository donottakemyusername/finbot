"""tools/trinity/indicators.py
================================
Layer 1: 纯Python硬指标计算，无需Claude API。
数据源：yfinance（主）
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
    return {
        "current_price": round(price, 4), "ma55": round(ma55, 4), "ma233": round(ma233, 4),
        "trend_alignment": alignment, "ma55_slope": round(ma55_slope, 6),
        "ma233_slope": round(ma233_slope, 6), "dist_from_ma55": round(dist_ma55, 4),
        "dist_from_ma233": round(dist_ma233, 4),
        "price_vs_ma55_last10": price_vs_ma55,
        "bars_above_ma55_last10": bars_above, "bars_below_ma55_last10": 10 - bars_above,
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
    zero_axis_cross = (cross_signal != "none" and abs(dif_now) < price * 0.005)
    recent30 = df.tail(30)
    dif_crossed = bool(recent30["dif"].max() > 0 and recent30["dif"].min() < 0)
    dea_crossed = bool(recent30["dea"].max() > 0 and recent30["dea"].min() < 0)
    return {
        "dif_current": round(dif_now, 6), "dea_current": round(dea_now, 6),
        "macd_bar_current": round(float(df["macd_bar"].iloc[-1]), 6),
        "zero_axis_position": "above" if dif_now > 0 else "below",
        "cross_signal": cross_signal, "zero_axis_cross": zero_axis_cross,
        "dif_crossed_zero_30d": dif_crossed, "dea_crossed_zero_30d": dea_crossed,
        "adjustment_sufficient": dif_crossed and dea_crossed,
        "macd_bar_history_60": [round(float(x), 6) for x in df["macd_bar"].tail(60).tolist()],
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
        "bb_upper": round(upper, 4), "bb_mid": round(mid, 4), "bb_lower": round(lower, 4),
        "current_price": round(price, 4), "price_position": round(position, 4),
        "above_mid": bool(price > mid), "below_mid_2bars": bool(below_mid_2bars),
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
        peaks,   _ = find_peaks(closes,  distance=5, prominence=std_c * 0.3)
        troughs, _ = find_peaks(-closes, distance=5, prominence=std_c * 0.3)
        peaks = list(peaks); troughs = list(troughs)
    else:
        peaks   = _find_peaks_simple(closes)
        troughs = _find_peaks_simple(-closes)

    def get_last2(idx): return list(idx[-2:]) if len(idx) >= 2 else []

    last_peaks   = get_last2(peaks)
    last_troughs = get_last2(troughs)

    top_div = None
    if len(last_peaks) == 2:
        p1, p2 = last_peaks
        b1, b2 = float(bar_vals[p1]), float(bar_vals[p2])
        top_div = {
            "peak1_price": round(float(closes[p1]), 4), "peak2_price": round(float(closes[p2]), 4),
            "peak1_macd_bar": round(b1, 6), "peak2_macd_bar": round(b2, 6),
            "price_new_high": bool(closes[p2] > closes[p1]), "macd_bar_lower": bool(b2 < b1),
            "price_change_pct": round(float((closes[p2] - closes[p1]) / closes[p1]), 4),
            "macd_change_pct": round(float((b2 - b1) / abs(b1)) if b1 != 0 else 0, 4),
        }
    bot_div = None
    if len(last_troughs) == 2:
        t1, t2 = last_troughs
        b1, b2 = float(bar_vals[t1]), float(bar_vals[t2])
        bot_div = {
            "trough1_price": round(float(closes[t1]), 4), "trough2_price": round(float(closes[t2]), 4),
            "trough1_macd_bar": round(b1, 6), "trough2_macd_bar": round(b2, 6),
            "price_new_low": bool(closes[t2] < closes[t1]), "macd_bar_smaller": bool(abs(b2) < abs(b1)),
            "price_change_pct": round(float((closes[t2] - closes[t1]) / closes[t1]), 4),
            "macd_change_pct": round(float((abs(b2) - abs(b1)) / abs(b1)) if b1 != 0 else 0, 4),
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
    return {
        "top_divergence_raw": top_div, "bot_divergence_raw": bot_div,
        "turning_points": turning_points,
        "recent_30_closes": [round(float(x), 4) for x in closes[-30:]],
    }


def compute_all_hard_signals(df: pd.DataFrame) -> dict:
    ma   = compute_ma_signals(df)
    macd = compute_macd_signals(df)
    bb   = compute_bollinger_trinity(df)
    div  = compute_turning_points_and_divergence(df)
    return {**ma, **macd, **bb, **div}