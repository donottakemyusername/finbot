"""engine/backtest.py
====================
Generic event-driven backtester.

Each strategy is a pure function:
    signal_fn(df: pd.DataFrame) -> pd.Series[int]
        returns  +1 (buy), -1 (sell),  0 (hold) at each bar

The engine simulates:
  * Enter long on the next open after a +1 signal
  * Exit on the next open after a -1 signal OR a neutral cross
  * No short selling (long-only)
  * 0.1% round-trip commission per trade

Returns a BacktestResult dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pct_return: float
    holding_days: int
    win: bool
    exit_reason: str


@dataclass
class BacktestResult:
    ticker: str
    strategy: str
    n_trades: int
    win_rate: float          # 0-100
    avg_return_pct: float
    total_return_pct: float  # compounded
    buy_hold_pct: float
    avg_hold_days: float
    max_drawdown_pct: float
    sharpe: float
    trades: list[Trade] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "ticker": self.ticker,
            "strategy": self.strategy,
            "n_trades": self.n_trades,
            "win_rate_%": round(self.win_rate, 1),
            "avg_trade_%": round(self.avg_return_pct, 2),
            "total_return_%": round(self.total_return_pct, 1),
            "buy_hold_%": round(self.buy_hold_pct, 1),
            "avg_hold_days": round(self.avg_hold_days, 1),
            "max_drawdown_%": round(self.max_drawdown_pct, 1),
            "sharpe": round(self.sharpe, 2),
        }


COMMISSION = 0.001  # 0.1% per side


def run_backtest(
    ticker: str,
    df: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    strategy_name: str = "custom",
) -> BacktestResult:
    """Run a long-only backtest given a signal function.

    Parameters
    ----------
    ticker       : ticker symbol (for labelling)
    df           : OHLCV DataFrame (must have Close, Open columns)
    signal_fn    : function(df) → Series of int (+1 buy, -1 sell, 0 hold)
    strategy_name: label for this strategy
    """
    df = df.copy().dropna()
    signals = signal_fn(df)

    if signals is None or signals.empty:
        return _empty_result(ticker, strategy_name, df)

    trades: list[Trade] = []
    in_trade = False
    entry_price = 0.0
    entry_date = ""
    equity = 10_000.0
    equity_curve: list[float] = [equity]

    for i in range(1, len(df)):
        sig  = signals.iloc[i - 1]   # signal generated at bar i-1, act on bar i open
        price = float(df["Open"].iloc[i])
        date  = str(df.index[i].date())

        if not in_trade and sig == 1:
            entry_price = price * (1 + COMMISSION)
            entry_date  = date
            in_trade = True

        elif in_trade and sig == -1:
            exit_price = price * (1 - COMMISSION)
            pct = (exit_price - entry_price) / entry_price * 100
            hold = (pd.Timestamp(date) - pd.Timestamp(entry_date)).days
            equity *= (1 + pct / 100)
            equity_curve.append(equity)
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=date,
                entry_price=entry_price,
                exit_price=exit_price,
                pct_return=pct,
                holding_days=max(hold, 1),
                win=pct > 0,
                exit_reason="signal",
            ))
            in_trade = False

    if not trades:
        return _empty_result(ticker, strategy_name, df)

    rets   = np.array([t.pct_return for t in trades])
    eq_arr = np.array(equity_curve)
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / peak * 100

    hold_days = np.array([t.holding_days for t in trades])
    daily_r   = rets / hold_days
    sharpe    = float(daily_r.mean() / daily_r.std() * np.sqrt(252)) if daily_r.std() > 0 else 0.0

    bh = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1) * 100

    return BacktestResult(
        ticker=ticker,
        strategy=strategy_name,
        n_trades=len(trades),
        win_rate=float(np.mean([t.win for t in trades]) * 100),
        avg_return_pct=float(rets.mean()),
        total_return_pct=float((equity / 10_000 - 1) * 100),
        buy_hold_pct=bh,
        avg_hold_days=float(hold_days.mean()),
        max_drawdown_pct=float(dd.min()),
        sharpe=sharpe,
        trades=trades,
    )


def _empty_result(ticker: str, strategy: str, df: pd.DataFrame) -> BacktestResult:
    bh = (float(df["Close"].iloc[-1]) / float(df["Close"].iloc[0]) - 1) * 100 if len(df) > 1 else 0.0
    return BacktestResult(
        ticker=ticker, strategy=strategy,
        n_trades=0, win_rate=0, avg_return_pct=0,
        total_return_pct=0, buy_hold_pct=bh,
        avg_hold_days=0, max_drawdown_pct=0, sharpe=0,
    )