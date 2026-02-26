"""tools/technicals.py
======================
Technical analysis indicators, buy/sell signal generation, and
per-indicator backtesting.

Indicators implemented
----------------------
* Bollinger Bands (BB)
* Simple Moving Average crossover (SMA 50/200)
* Exponential Moving Average crossover (EMA 12/26)
* Relative Strength Index (RSI 14)
* MACD (12/26/9)
* Average True Range (ATR) — used for stop-loss sizing, not a signal itself

Each indicator exposes:
    compute_{indicator}(df)  → df with extra columns
    signal_{indicator}(df)   → pd.Series of +1 / -1 / 0
    analyse_{indicator}(ticker, years) → dict with signal + backtest result
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tools.data import get_price_history
from engine.backtest import BacktestResult, run_backtest

# ─────────────────────────────────────────────────────────────────────────────
# Bollinger Bands
# ─────────────────────────────────────────────────────────────────────────────

def compute_bollinger(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    df = df.copy()
    df["BB_SMA"]   = df["Close"].rolling(period).mean()
    df["BB_STD"]   = df["Close"].rolling(period).std()
    df["BB_Upper"] = df["BB_SMA"] + num_std * df["BB_STD"]
    df["BB_Lower"] = df["BB_SMA"] - num_std * df["BB_STD"]
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_SMA"]
    return df


def signal_bollinger(df: pd.DataFrame) -> pd.Series:
    """
    +1 (buy)  when price crosses below lower band (oversold)
    -1 (sell) when price crosses above upper band (overbought) or reverts to SMA
     0        otherwise
    """
    df = compute_bollinger(df)
    sig = pd.Series(0, index=df.index)
    in_trade = False

    for i in range(1, len(df)):
        price = df["Close"].iloc[i]
        lower = df["BB_Lower"].iloc[i]
        upper = df["BB_Upper"].iloc[i]
        sma   = df["BB_SMA"].iloc[i]

        if pd.isna(lower):
            continue
        if not in_trade and price < lower:
            sig.iloc[i] = 1
            in_trade = True
        elif in_trade and (price > upper or price >= sma):
            sig.iloc[i] = -1
            in_trade = False

    return sig


def current_bollinger_signal(df: pd.DataFrame) -> dict:
    df = compute_bollinger(df)
    last = df.iloc[-1]
    price = last["Close"]
    if pd.isna(last["BB_Lower"]):
        return {"signal": "neutral", "reason": "Insufficient data"}

    if price < last["BB_Lower"]:
        signal = "buy"
        reason = f"Price ${price:.2f} below lower band ${last['BB_Lower']:.2f}"
    elif price > last["BB_Upper"]:
        signal = "sell"
        reason = f"Price ${price:.2f} above upper band ${last['BB_Upper']:.2f}"
    elif price > last["BB_SMA"]:
        signal = "hold"
        reason = f"Price above SMA, within bands (width: {last['BB_Width']:.2%})"
    else:
        signal = "hold"
        reason = f"Price below SMA, within bands (width: {last['BB_Width']:.2%})"

    return {
        "signal": signal,
        "reason": reason,
        "price": round(price, 2),
        "upper_band": round(last["BB_Upper"], 2),
        "lower_band": round(last["BB_Lower"], 2),
        "sma": round(last["BB_SMA"], 2),
        "band_width_%": round(last["BB_Width"] * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SMA Crossover (50 / 200 — Golden Cross / Death Cross)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sma(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.DataFrame:
    df = df.copy()
    df[f"SMA_{fast}"] = df["Close"].rolling(fast).mean()
    df[f"SMA_{slow}"] = df["Close"].rolling(slow).mean()
    return df


def signal_sma(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> pd.Series:
    """Golden cross (+1) / Death cross (-1)."""
    df = compute_sma(df, fast, slow)
    f_col, s_col = f"SMA_{fast}", f"SMA_{slow}"
    above = df[f_col] > df[s_col]
    sig = pd.Series(0, index=df.index)
    sig[above & ~above.shift(1).fillna(False)]  = 1   # just crossed above
    sig[~above & above.shift(1).fillna(True)]   = -1  # just crossed below
    return sig


def current_sma_signal(df: pd.DataFrame, fast: int = 50, slow: int = 200) -> dict:
    df = compute_sma(df, fast, slow)
    last = df.iloc[-1]
    f_col, s_col = f"SMA_{fast}", f"SMA_{slow}"

    if pd.isna(last[s_col]):
        return {"signal": "neutral", "reason": "Insufficient data for SMA200"}

    gap_pct = (last[f_col] - last[s_col]) / last[s_col] * 100

    # Check for recent crossover (last 5 bars)
    recent = df.tail(5)
    crossed_above = any(recent[f_col].iloc[i] > recent[s_col].iloc[i] and
                        recent[f_col].iloc[i-1] <= recent[s_col].iloc[i-1]
                        for i in range(1, len(recent)))
    crossed_below = any(recent[f_col].iloc[i] < recent[s_col].iloc[i] and
                        recent[f_col].iloc[i-1] >= recent[s_col].iloc[i-1]
                        for i in range(1, len(recent)))

    if crossed_above:
        signal = "buy"
        reason = f"Golden Cross: SMA{fast} just crossed above SMA{slow}"
    elif crossed_below:
        signal = "sell"
        reason = f"Death Cross: SMA{fast} just crossed below SMA{slow}"
    elif last[f_col] > last[s_col]:
        signal = "hold"
        reason = f"Bullish alignment: SMA{fast} ({last[f_col]:.2f}) > SMA{slow} ({last[s_col]:.2f}), gap {gap_pct:+.1f}%"
    else:
        signal = "sell"
        reason = f"Bearish alignment: SMA{fast} ({last[f_col]:.2f}) < SMA{slow} ({last[s_col]:.2f}), gap {gap_pct:+.1f}%"

    return {
        "signal": signal,
        "reason": reason,
        f"sma_{fast}": round(last[f_col], 2),
        f"sma_{slow}": round(last[s_col], 2),
        "gap_%": round(gap_pct, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# EMA Crossover (12 / 26)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ema(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.DataFrame:
    df = df.copy()
    df[f"EMA_{fast}"] = df["Close"].ewm(span=fast, adjust=False).mean()
    df[f"EMA_{slow}"] = df["Close"].ewm(span=slow, adjust=False).mean()
    return df


def signal_ema(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> pd.Series:
    df = compute_ema(df, fast, slow)
    f_col, s_col = f"EMA_{fast}", f"EMA_{slow}"
    above = df[f_col] > df[s_col]
    sig = pd.Series(0, index=df.index)
    sig[above & ~above.shift(1).fillna(False)]  = 1
    sig[~above & above.shift(1).fillna(True)]   = -1
    return sig


def current_ema_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26) -> dict:
    df = compute_ema(df, fast, slow)
    last = df.iloc[-1]
    f_col, s_col = f"EMA_{fast}", f"EMA_{slow}"
    gap_pct = (last[f_col] - last[s_col]) / last[s_col] * 100

    recent = df.tail(3)
    crossed_above = any(recent[f_col].iloc[i] > recent[s_col].iloc[i] and
                        recent[f_col].iloc[i-1] <= recent[s_col].iloc[i-1]
                        for i in range(1, len(recent)))
    crossed_below = any(recent[f_col].iloc[i] < recent[s_col].iloc[i] and
                        recent[f_col].iloc[i-1] >= recent[s_col].iloc[i-1]
                        for i in range(1, len(recent)))

    signal = ("buy" if crossed_above else
              "sell" if (crossed_below or last[f_col] < last[s_col]) else "hold")
    reason = (
        f"EMA{fast} ({last[f_col]:.2f}) {'>' if last[f_col] > last[s_col] else '<'} "
        f"EMA{slow} ({last[s_col]:.2f}), gap {gap_pct:+.1f}%"
    )
    return {"signal": signal, "reason": reason,
            f"ema_{fast}": round(last[f_col], 2),
            f"ema_{slow}": round(last[s_col], 2),
            "gap_%": round(gap_pct, 2)}


# ─────────────────────────────────────────────────────────────────────────────
# RSI
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0)
    loss   = (-delta).clip(lower=0)
    avg_g  = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_l  = loss.ewm(com=period - 1, min_periods=period).mean()
    rs     = avg_g / avg_l.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def signal_rsi(df: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> pd.Series:
    """
    +1 when RSI crosses above oversold threshold
    -1 when RSI crosses below overbought threshold
    """
    df = compute_rsi(df)
    sig = pd.Series(0, index=df.index)
    rsi = df["RSI"]
    sig[(rsi > oversold) & (rsi.shift(1) <= oversold)] = 1
    sig[(rsi < overbought) & (rsi.shift(1) >= overbought)] = -1
    return sig


def current_rsi_signal(df: pd.DataFrame, oversold: int = 30, overbought: int = 70) -> dict:
    df = compute_rsi(df)
    rsi_val = df["RSI"].iloc[-1]
    if pd.isna(rsi_val):
        return {"signal": "neutral", "reason": "Insufficient data"}

    if rsi_val < oversold:
        signal, reason = "buy", f"RSI {rsi_val:.1f} is oversold (< {oversold})"
    elif rsi_val > overbought:
        signal, reason = "sell", f"RSI {rsi_val:.1f} is overbought (> {overbought})"
    elif rsi_val < 50:
        signal, reason = "hold", f"RSI {rsi_val:.1f} is neutral-bearish (30–50)"
    else:
        signal, reason = "hold", f"RSI {rsi_val:.1f} is neutral-bullish (50–70)"

    return {"signal": signal, "reason": reason, "rsi": round(rsi_val, 2),
            "oversold_threshold": oversold, "overbought_threshold": overbought}


# ─────────────────────────────────────────────────────────────────────────────
# MACD
# ─────────────────────────────────────────────────────────────────────────────

def compute_macd(
    df: pd.DataFrame,
    fast: int = 12, slow: int = 26, signal_period: int = 9
) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast,   adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow,   adjust=False).mean()
    df["MACD"]        = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal_period, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    return df


def signal_macd(df: pd.DataFrame) -> pd.Series:
    """
    +1 when MACD crosses above signal line
    -1 when MACD crosses below signal line
    """
    df = compute_macd(df)
    above = df["MACD"] > df["MACD_Signal"]
    sig = pd.Series(0, index=df.index)
    sig[above & ~above.shift(1).fillna(False)]  = 1
    sig[~above & above.shift(1).fillna(True)]   = -1
    return sig


def current_macd_signal(df: pd.DataFrame) -> dict:
    df = compute_macd(df)
    last    = df.iloc[-1]
    prev    = df.iloc[-2]
    macd_v  = last["MACD"]
    sig_v   = last["MACD_Signal"]
    hist_v  = last["MACD_Hist"]

    if pd.isna(macd_v):
        return {"signal": "neutral", "reason": "Insufficient data"}

    crossed_above = macd_v > sig_v and prev["MACD"] <= prev["MACD_Signal"]
    crossed_below = macd_v < sig_v and prev["MACD"] >= prev["MACD_Signal"]

    if crossed_above:
        signal = "buy"
        reason = f"MACD ({macd_v:.3f}) just crossed above signal ({sig_v:.3f})"
    elif crossed_below:
        signal = "sell"
        reason = f"MACD ({macd_v:.3f}) just crossed below signal ({sig_v:.3f})"
    elif macd_v > sig_v:
        signal = "hold"
        reason = f"MACD ({macd_v:.3f}) above signal ({sig_v:.3f}), histogram {hist_v:.3f}"
    else:
        signal = "sell"
        reason = f"MACD ({macd_v:.3f}) below signal ({sig_v:.3f}), histogram {hist_v:.3f}"

    return {"signal": signal, "reason": reason,
            "macd": round(macd_v, 4), "signal_line": round(sig_v, 4),
            "histogram": round(hist_v, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# ATR (volatility / stop-loss sizing — not a directional signal)
# ─────────────────────────────────────────────────────────────────────────────

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift(1)).abs(),
        (df["Low"]  - df["Close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = df["TR"].ewm(com=period - 1, min_periods=period).mean()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Master: run ALL technicals for a ticker
# ─────────────────────────────────────────────────────────────────────────────

INDICATORS = {
    "bollinger":  (signal_bollinger, current_bollinger_signal, "Bollinger Bands"),
    "sma":        (signal_sma,       current_sma_signal,       "SMA 50/200"),
    "ema":        (signal_ema,       current_ema_signal,       "EMA 12/26"),
    "rsi":        (signal_rsi,       current_rsi_signal,       "RSI 14"),
    "macd":       (signal_macd,      current_macd_signal,      "MACD 12/26/9"),
}


def run_technical_analysis(
    ticker: str,
    years: int = 5,
    end_date: str | None = None,
    indicators: list[str] | None = None,
) -> dict:
    """
    Run all (or a subset of) technical indicators for a ticker.
    Returns current signal + 5-year backtest result for each indicator.

    Returns
    -------
    {
        "ticker": "AAPL",
        "price": 185.50,
        "indicators": {
            "bollinger": {
                "name": "Bollinger Bands",
                "signal": "buy",
                "reason": "...",
                "backtest": { ... BacktestResult.summary() ... }
            },
            ...
        },
        "overall_signal": "buy" | "sell" | "hold"
    }
    """
    df = get_price_history(ticker, years=years, end_date=end_date)

    to_run = indicators or list(INDICATORS.keys())
    results: dict = {}

    for key in to_run:
        if key not in INDICATORS:
            continue
        sig_fn, cur_fn, label = INDICATORS[key]

        # Current signal
        current = cur_fn(df)

        # Backtest
        bt: BacktestResult = run_backtest(ticker, df, sig_fn, strategy_name=label)

        results[key] = {
            "name": label,
            **current,
            "backtest": bt.summary(),
        }

    # Overall signal: simple majority vote
    signals = [v["signal"] for v in results.values()]
    buy_count  = signals.count("buy")
    sell_count = signals.count("sell")
    hold_count = signals.count("hold")

    if buy_count > sell_count and buy_count > hold_count:
        overall = "buy"
    elif sell_count > buy_count and sell_count > hold_count:
        overall = "sell"
    else:
        overall = "hold"

    return {
        "ticker": ticker,
        "price": round(float(df["Close"].iloc[-1]), 2),
        "as_of": str(df.index[-1].date()),
        "indicators": results,
        "overall_technical_signal": overall,
        "vote_summary": {"buy": buy_count, "sell": sell_count, "hold": hold_count},
    }