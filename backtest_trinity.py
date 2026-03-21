"""backtest_trinity.py
=====================
历史回测脚本：给定一个日期和一组股票，
用三位一体分析判断当时是否应该买入，并计算事后收益率验证准确性。

用法:
    python backtest_trinity.py --date 2024-06-01 --tickers AAPL TSLA NVDA
    python backtest_trinity.py                      # 使用脚本底部的默认配置
    python backtest_trinity.py --no-claude          # 跳过Claude软分析（只用硬指标+时空状态）
"""
from __future__ import annotations

import argparse
import os
import sys

# Fix Windows GBK console encoding for Chinese characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
from datetime import date, timedelta, datetime

import anthropic
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── 把项目根目录加入 Python 路径（兼容直接运行）─────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from tools.trinity.indicators import (
    compute_all_hard_signals,
    compute_bollinger_trinity,
    compute_macd_signals,
    compute_divergence_summary,
    compute_ma_analysis_summary,
)
from tools.trinity.state import compute_time_space_state
from tools.trinity.prompt import call_claude_for_soft_signals
from tools.trinity.verify import verify_trinity_output
from tools.trinity.analysis import _merge_claude_with_python, _enforce_hard_overrides


# ─────────────────────────────────────────────────────────────────────────────
# 数据获取：截止到指定日期
# ─────────────────────────────────────────────────────────────────────────────

def fetch_as_of(ticker: str, as_of: date) -> dict[str, pd.DataFrame]:
    """
    下载截止到 as_of（含）的历史数据，模拟在当时能看到的数据。
    对于较久远的历史日期，60m 小时数据可能不可用（yfinance 限制）。
    """
    t = ticker.upper()
    end_str = (as_of + timedelta(days=1)).strftime("%Y-%m-%d")  # yfinance end 是开区间

    # 各时间框架需要多少历史深度
    start_daily   = (as_of - timedelta(days=800)).strftime("%Y-%m-%d")   # 约 2 年
    start_weekly  = (as_of - timedelta(days=1900)).strftime("%Y-%m-%d")  # 约 5 年
    start_monthly = (as_of - timedelta(days=1900)).strftime("%Y-%m-%d")  # 约 5 年
    # 60 天前的小时数据 yfinance 无法获取
    start_hourly  = (as_of - timedelta(days=58)).strftime("%Y-%m-%d")

    def _dl(interval: str, start: str) -> pd.DataFrame:
        try:
            df = yf.download(t, start=start, end=end_str,
                             interval=interval, progress=False, auto_adjust=True)
            df = df.copy()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df.dropna()
        except Exception:
            return pd.DataFrame()

    dfs = {
        "daily":   _dl("1d",  start_daily),
        "weekly":  _dl("1wk", start_weekly),
        "monthly": _dl("1mo", start_monthly),
        "hourly":  _dl("60m", start_hourly),
    }
    return dfs


def fetch_forward_returns(ticker: str, as_of: date,
                          horizons=(5, 10, 21, 63)) -> dict[str, float | None]:
    """
    计算 as_of 之后各 horizon 个交易日的实际收益率，用于验证信号准确性。
    horizon 单位：交易日（约：5≈1周, 21≈1月, 63≈3月）
    """
    end_dt = as_of + timedelta(days=max(horizons) * 2)  # 多取一些确保有足够交易日
    try:
        df = yf.download(ticker.upper(),
                         start=as_of.strftime("%Y-%m-%d"),
                         end=end_dt.strftime("%Y-%m-%d"),
                         interval="1d", progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna()
    except Exception:
        return {f"{h}d": None for h in horizons}

    if df.empty:
        return {f"{h}d": None for h in horizons}

    buy_price = float(df["Close"].iloc[0])
    result = {}
    for h in horizons:
        if len(df) > h:
            sell_price = float(df["Close"].iloc[h])
            result[f"{h}d"] = round((sell_price - buy_price) / buy_price * 100, 2)
        else:
            result[f"{h}d"] = None
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 核心：历史日期三位一体分析
# ─────────────────────────────────────────────────────────────────────────────

def trinity_analysis_as_of(
    ticker: str,
    as_of: date,
    use_claude: bool = True,
    client: anthropic.Anthropic | None = None,
) -> dict:
    """
    在指定历史日期运行三位一体分析。
    use_claude=False 时跳过软分析，只返回硬指标 + 时空状态（快速 / 省钱）。
    """
    ticker = ticker.upper()

    # ── 1. 获取历史数据 ───────────────────────────────────────────────────────
    dfs        = fetch_as_of(ticker, as_of)
    df_daily   = dfs.get("daily",   pd.DataFrame())
    df_weekly  = dfs.get("weekly",  pd.DataFrame())
    df_monthly = dfs.get("monthly", pd.DataFrame())
    df_hourly  = dfs.get("hourly",  pd.DataFrame())

    if df_daily.empty or len(df_daily) < 60:
        return {"ticker": ticker, "as_of": str(as_of), "error": "历史数据不足（<60日）"}

    # 确认实际最后一个交易日
    actual_last_date = str(df_daily.index[-1].date()
                           if hasattr(df_daily.index[-1], "date")
                           else df_daily.index[-1])[:10]

    # ── 2. 硬指标 ─────────────────────────────────────────────────────────────
    hard_signals = compute_all_hard_signals(df_daily)

    bb_hourly = (compute_bollinger_trinity(df_hourly)
                 if not df_hourly.empty and len(df_hourly) >= 25
                 else {"error": "60m 数据不可用（历史过远）"})

    monthly_macd = (compute_macd_signals(df_monthly)
                    if not df_monthly.empty and len(df_monthly) >= 30
                    else {})

    weekly_macd = (compute_macd_signals(df_weekly)
                   if not df_weekly.empty and len(df_weekly) >= 30
                   else {})

    # ── 3. 时空状态 ───────────────────────────────────────────────────────────
    time_space = compute_time_space_state(
        df_daily=df_daily,
        df_monthly=df_monthly,
        df_weekly=df_weekly,
        bb_hourly=bb_hourly,
        holding_days_min=1,
    )

    # ── 4. Claude 软判断（可选）───────────────────────────────────────────────
    if use_claude:
        if client is None:
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        claude_input = {
            **hard_signals,
            "bb_hourly_j_minus1": bb_hourly,
            "weekly_macd_j1":     weekly_macd,
            "monthly_macd_j2":    monthly_macd,
        }
        claude_output = call_claude_for_soft_signals(
            ticker=ticker,
            hard_signals=claude_input,
            time_space=time_space,
            client=client,
        )
        pattern_analysis = _merge_claude_with_python(claude_output, hard_signals, time_space)
        pattern_analysis = _enforce_hard_overrides(pattern_analysis, hard_signals)
        composite = pattern_analysis.get("composite", {})
        signal     = composite.get("signal", "hold")
        confidence = composite.get("confidence", "low")
        entry_side = composite.get("entry_side", "wait")
        suggested_action = composite.get("suggested_action", "")
        key_risk   = composite.get("key_risk", "")
    else:
        # 无 Claude：根据硬指标 + 时空状态给出简单信号
        state = time_space.get("daily_state", {})
        is_bullish = state.get("is_bullish", False)
        is_extreme = state.get("is_extreme", False)
        trend = hard_signals.get("trend_alignment", "mixed")
        div_sum = compute_divergence_summary(hard_signals)
        div_type = div_sum.get("divergence_type", "none")

        if is_bullish and trend == "bullish" and div_type != "bearish_divergence":
            signal = "buy"
        elif not is_bullish and trend == "bearish":
            signal = "sell"
        else:
            signal = "hold"

        confidence = "medium" if not is_extreme else "low"
        entry_side = "long" if signal == "buy" else ("short" if signal == "sell" else "wait")
        suggested_action = f"时空状态={state.get('state_label','?')} 趋势排列={trend}"
        key_risk = "无Claude分析，仅供参考"
        pattern_analysis = {}

    # ── 5. 构建摘要 ───────────────────────────────────────────────────────────
    state = time_space.get("daily_state", {})
    summary = {
        "ticker":            ticker,
        "as_of":             str(as_of),
        "actual_last_bar":   actual_last_date,
        "current_price":     hard_signals.get("current_price"),
        "ma55":              hard_signals.get("ma55"),
        "ma233":             hard_signals.get("ma233"),
        "trend_alignment":   hard_signals.get("trend_alignment", "mixed"),
        "state_label":       state.get("state_label", "未知"),
        "state_code":        state.get("current_state", "unknown"),
        "is_bullish":        state.get("is_bullish", False),
        "weekly_state":      time_space.get("weekly_state", {}).get("state_label", "未知"),
        "divergence_type":   compute_divergence_summary(hard_signals).get("divergence_type", "none"),
        "signal":            signal,
        "confidence":        confidence,
        "entry_side":        entry_side,
        "suggested_action":  suggested_action,
        "key_risk":          key_risk,
        "long_stop_loss":    hard_signals.get("long_stop_loss"),
        "key_support":       hard_signals.get("key_support"),
        "key_resistance":    hard_signals.get("key_resistance"),
    }

    # 验证层
    summary = verify_trinity_output(summary, hard_signals, time_space)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 批量回测
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    tickers: list[str],
    as_of: date,
    use_claude: bool = True,
    forward_horizons: tuple[int, ...] = (5, 21, 63),
) -> pd.DataFrame:
    """
    对多个股票在同一历史日期做三位一体分析，返回结果 DataFrame。
    结果包含：信号、置信度、建议操作、以及事后实际收益率。
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if use_claude else None

    import time as _time

    rows = []
    for i, ticker in enumerate(tickers):
        # 每只股票之间间隔 3 秒，避免连续请求触发 rate limit
        if use_claude and i > 0:
            _time.sleep(3)
        print(f"  Analyzing {ticker} ...")
        try:
            summary = trinity_analysis_as_of(ticker, as_of, use_claude=use_claude, client=client)
        except Exception as e:
            summary = {"ticker": ticker, "as_of": str(as_of), "error": str(e)}

        if "error" in summary:
            rows.append({"ticker": ticker, "error": summary["error"]})
            continue

        # 事后收益率
        fwd = fetch_forward_returns(ticker, as_of, horizons=forward_horizons)

        row = {
            "ticker":          summary["ticker"],
            "as_of":           summary["as_of"],
            "close":           summary.get("current_price"),
            "state":           summary.get("state_label"),
            "weekly_state":    summary.get("weekly_state"),
            "trend":           summary.get("trend_alignment"),
            "divergence":      summary.get("divergence_type"),
            "signal":          summary.get("signal"),
            "confidence":      summary.get("confidence"),
            "entry":           summary.get("entry_side"),
            "stop_loss":       summary.get("long_stop_loss"),
            "key_support":     summary.get("key_support"),
            "key_resistance":  summary.get("key_resistance"),
            "action_summary":  summary.get("suggested_action", "")[:80],
            "key_risk":        summary.get("key_risk", "")[:60],
        }
        for k, v in fwd.items():
            row[f"ret_{k}"] = f"{v:+.1f}%" if v is not None else "N/A"

        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 打印结果
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_EMOJI = {
    "strong_buy": "**BUY+**",
    "buy":        "**BUY** ",
    "hold":       "--HOLD-",
    "sell":       "!!SELL!",
    "strong_sell":"!!SEL!!",
}

def print_results(df: pd.DataFrame, as_of: date) -> None:
    print(f"\n{'='*80}")
    print(f"  Trinity Backtest  |  As of: {as_of}")
    print(f"{'='*80}\n")

    for _, row in df.iterrows():
        ticker = row.get("ticker", "?")
        if "error" in row and pd.notna(row.get("error")):
            print(f"  {ticker:6s}  [ERR] {row['error']}\n")
            continue

        signal = row.get("signal", "hold")
        tag    = SIGNAL_EMOJI.get(signal, "  ??  ")
        conf   = row.get("confidence", "")
        entry  = row.get("entry", "")

        print(f"  {ticker:6s}  {tag} {signal.upper():12s}  conf={conf:6s}  side={entry}")
        print(f"         price={row.get('close')}  "
              f"state={row.get('state')}  weekly={row.get('weekly_state')}  "
              f"trend={row.get('trend')}  div={row.get('divergence')}")
        print(f"         stop={row.get('stop_loss')}  support={row.get('key_support')}  "
              f"resist={row.get('key_resistance')}")

        # forward returns
        ret_cols = [c for c in row.index if c.startswith("ret_")]
        if ret_cols:
            ret_str = "  ".join(f"{c.replace('ret_','')}={row[c]}" for c in ret_cols)
            print(f"         fwd returns: {ret_str}")

        if row.get("action_summary"):
            print(f"         action: {row['action_summary']}")
        if row.get("key_risk"):
            print(f"         risk:   {row['key_risk']}")
        print()

    # summary table
    signal_col = "signal"
    if signal_col in df.columns:
        print(f"  {'-'*60}")
        print(f"  Signal distribution: "
              + "  ".join(f"{SIGNAL_EMOJI.get(s,'?')} {s}={n}"
                          for s, n in df[signal_col].value_counts().items()))
        buy_mask = df[signal_col].isin(["buy", "strong_buy"])
        ret_cols = [c for c in df.columns if c.startswith("ret_")]
        if buy_mask.any() and ret_cols:
            for rc in ret_cols:
                vals = pd.to_numeric(
                    df.loc[buy_mask, rc].str.replace("%", "", regex=False),
                    errors="coerce"
                ).dropna()
                if not vals.empty:
                    label = rc.replace("ret_", "")
                    print(f"  BUY signal avg return ({label}): {vals.mean():+.2f}%  "
                          f"win rate: {(vals > 0).mean()*100:.0f}%")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

# ============================================================
# 默认配置（直接运行脚本时使用）
# ============================================================
DEFAULT_DATE    = "2024-10-01"
DEFAULT_TICKERS = ["AAPL", "TSLA", "NVDA", "MSFT", "AMZN"]
DEFAULT_CLAUDE  = True           # False = 跳过Claude软分析（快速模式）
# ============================================================


def _parse_tickers(raw: list[str]) -> list[str]:
    """Accept both space-separated and comma-separated ticker lists, deduplicated."""
    result = []
    seen = set()
    for item in raw:
        for t in item.split(","):
            t = t.strip().upper()
            if t and t not in seen:
                result.append(t)
                seen.add(t)
    return result


def main():
    parser = argparse.ArgumentParser(description="Trinity historical backtest")
    parser.add_argument("--date",      default=DEFAULT_DATE,
                        help="Analysis date YYYY-MM-DD")
    # Accept both --tickers and --ticker (with or without s)
    parser.add_argument("--tickers", "--ticker", nargs="+", default=DEFAULT_TICKERS,
                        dest="tickers",
                        help="Ticker symbols, space- or comma-separated")
    parser.add_argument("--no-claude", action="store_true",
                        help="Skip Claude soft analysis (fast mode, hard signals only)")
    parser.add_argument("--horizons",  nargs="+", type=int, default=[5, 21, 63],
                        help="Forward-return windows in trading days (default: 5 21 63)")
    parser.add_argument("--output",    default=None,
                        help="Save results to CSV file path (optional)")
    args = parser.parse_args()

    try:
        as_of = datetime.strptime(args.date, "%Y-%m-%d").date()
    except ValueError:
        print(f"Invalid date format: {args.date}, use YYYY-MM-DD")
        sys.exit(1)

    if as_of >= date.today():
        print("[WARN] as_of date must be in the past (historical backtest)")
        sys.exit(1)

    use_claude = DEFAULT_CLAUDE and not args.no_claude
    tickers    = _parse_tickers(args.tickers)

    print(f"\nTrinity Historical Backtest")
    print(f"  Date:    {as_of}")
    print(f"  Tickers: {', '.join(tickers)}")
    print(f"  Claude:  {'enabled' if use_claude else 'disabled (fast mode)'}")
    print(f"  Forward windows: {args.horizons} trading days\n")

    df = run_backtest(
        tickers=tickers,
        as_of=as_of,
        use_claude=use_claude,
        forward_horizons=tuple(args.horizons),
    )

    print_results(df, as_of)

    if args.output:
        df.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Results saved: {args.output}")


if __name__ == "__main__":
    main()
