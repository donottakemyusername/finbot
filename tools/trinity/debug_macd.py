# debug_macd.py
# 打印任意股票在各个时间周期的MACD数据，用于手动验证时空状态机是否正确。
# 用法：
#   cd mcp-finbot/tools/trinity
#   python debug_macd.py MSFT
#   python debug_macd.py MSFT --tf monthly weekly daily
import sys, argparse
import pandas as pd
import yfinance as yf

# ── 参数 ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("tickers", nargs="+", help="股票代码，支持多个，例如 MSFT AAPL NVDA")
parser.add_argument(
    "--tf", nargs="+",
    default=["monthly", "weekly", "daily"],
    help="时间周期列表，可选: monthly weekly daily hourly"
)
parser.add_argument("--tail", type=int, default=20, help="打印最近N根K线（默认20）")
args = parser.parse_args()

TF_CONFIG = {
    "monthly": ("5y", "1mo"),
    "weekly":  ("5y", "1wk"),
    "daily":   ("2y", "1d"),
    "hourly":  ("730d", "60m"),
}

def compute_macd(df):
    exp12 = df["Close"].ewm(span=12, adjust=False).mean()
    exp26 = df["Close"].ewm(span=26, adjust=False).mean()
    dif   = exp12 - exp26
    dea   = dif.ewm(span=9, adjust=False).mean()
    bar   = 2 * (dif - dea)
    return dif, dea, bar

def detect_events(dif_vals, dea_vals):
    events = []
    for i in range(1, len(dif_vals)):
        d0, d1 = float(dif_vals[i-1]), float(dif_vals[i])
        e0, e1 = float(dea_vals[i-1]), float(dea_vals[i])
        if   d0 <= 0 < d1:             events.append((i, "DIF↑零轴"))
        elif d0 >= 0 > d1:             events.append((i, "DIF↓零轴"))
        if   e0 <= 0 < e1:             events.append((i, "DEA↑零轴"))
        elif e0 >= 0 > e1:             events.append((i, "DEA↓零轴"))
        if d0 <= e0 and d1 > e1:
            events.append((i, "底部金叉" if d1 < 0 else "高位金叉"))
        elif d0 >= e0 and d1 < e1:
            events.append((i, "顶部死叉" if d1 > 0 else "低位死叉"))
    return events

def state_from_events(events):
    """简化状态机，返回当前状态名"""
    state = "unknown"
    for _, ev in events:
        if ev == "底部金叉":       state = "中性偏强"
        elif ev == "DIF↑零轴":    state = "极强"
        elif ev == "DEA↑零轴":    state = "强"
        elif ev == "顶部死叉":     state = "中性偏弱"
        elif ev == "DIF↓零轴":    state = "极弱"
        elif ev == "DEA↓零轴":    state = "弱"
        elif ev == "高位金叉" and state == "中性偏弱": state = "强"
    return state

# ── 主循环 ────────────────────────────────────────────────────────────────────
for ticker in [t.upper() for t in args.tickers]:
    print(f"\n{'='*70}")
    print(f"  {ticker}  MACD时空状态调试报告")
    print(f"{'='*70}")

    for tf in args.tf:
        if tf not in TF_CONFIG:
            print(f"\n[SKIP] 未知周期: {tf}")
            continue

        period, interval = TF_CONFIG[tf]
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna()

        if len(df) < 30:
            print(f"\n[{tf.upper()}] 数据不足（{len(df)}根），跳过")
            continue

        dif, dea, bar = compute_macd(df)
        dif_vals = dif.values
        dea_vals = dea.values

        events = detect_events(dif_vals, dea_vals)
        current_state = state_from_events(events)

        tail = min(args.tail, len(df))
        df_tail = df.tail(tail).copy()
        dif_tail = dif.tail(tail)
        dea_tail = dea.tail(tail)
        bar_tail = bar.tail(tail)

        tf_label = {"monthly":"月线","weekly":"周线","daily":"日线","hourly":"60分线"}.get(tf, tf)
        print(f"\n{'─'*70}")
        print(f"  [{tf_label}]  当前状态: 【{current_state}】  共{len(df)}根K线")
        print(f"{'─'*70}")
        print(f"  {'日期':<12} {'收盘':>8} {'DIF':>10} {'DEA':>10} {'BAR':>10}  {'零轴':>6}  事件")
        print(f"  {'─'*12} {'─'*8} {'─'*10} {'─'*10} {'─'*10}  {'─'*6}")

        event_dict: dict[int, list[str]] = {}
        for bar_i, ev_name in events:
            event_dict.setdefault(bar_i, []).append(ev_name)

        for i, (idx, row) in enumerate(df_tail.iterrows()):
            abs_i = len(df) - tail + i
            date_str = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
            close = float(row["Close"])
            d = float(dif_tail.iloc[i])
            e = float(dea_tail.iloc[i])
            b = float(bar_tail.iloc[i])
            zero_side = "上" if d > 0 else "下"
            evs = " ← " + " / ".join(event_dict[abs_i]) if abs_i in event_dict else ""
            print(f"  {date_str:<12} {close:>8.2f} {d:>10.4f} {e:>10.4f} {b:>10.4f}  零轴{zero_side}{evs}")

        recent_events = [(i, ev) for i, ev in events if i >= len(df) - 60]
        if recent_events:
            print(f"\n  近60根K线内状态切换事件：")
            for bar_i, ev_name in recent_events:
                date = str(df.index[bar_i].date()) if hasattr(df.index[bar_i], "date") else str(df.index[bar_i])[:10]
                print(f"    bar{bar_i:4d}  {date}  {ev_name}")
        else:
            print(f"\n  近60根K线内：无状态切换（状态稳定）")

        dif_crossed = any(dif_vals[i-1]*dif_vals[i] < 0 for i in range(max(1,len(dif_vals)-60), len(dif_vals)))
        dea_crossed = any(dea_vals[i-1]*dea_vals[i] < 0 for i in range(max(1,len(dea_vals)-60), len(dea_vals)))
        print(f"\n  快照: DIF={dif_vals[-1]:.4f} ({'零轴上' if dif_vals[-1]>0 else '零轴下'})  "
              f"DEA={dea_vals[-1]:.4f} ({'零轴上' if dea_vals[-1]>0 else '零轴下'})  "
              f"调整充分={'✅' if dif_crossed and dea_crossed else '❌'}")

print(f"\n{'='*70}\n")
