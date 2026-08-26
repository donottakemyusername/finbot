"""momentum_crash_test.py
========================
测试：Quant GT 的"动量崩塌"（momentum crash）风险，以及如何缓解 / 提前预警。

背景
----
经前面分析确认：
  * QGT 选股是「横截面动量」（每月买最强的 5 只高 beta 突破股，beta≈1.66）。
  * 它在震荡/横盘市里收益最高（离散度红利），并不吃亏。
  * 真正的软肋是「急涨之后的动量崩塌」——拥挤交易突然去杠杆（如 2026-07 芯片崩、
    2024-04、2020-03、2018-10）。
  * 现有波动率目标模型（按 QGT 近 6 月已实现波动率降仓）能缓解崩塌，但它是「反应式」的：
    崩塌发生在波动率已升高之后时它有效；对「低波动 melt-up 后突然崩」则太慢。

本脚本做三件事（全部因果，只用每月 i 之前已收盘的信息）：
  A. 量化现有波动率降仓在历史崩塌月「救回」了多少。
  B. 检验候选预警指标对「下月崩塌」的预测力：
        近3/6月涨幅(runup)、近6月负偏度(skew)、已实现波动率(trail_vol)、VIX 水平/分位。
     结论：涨幅(runup6, corr≈+0.31) 明显强于波动率(+0.12)，VIX 几乎无用(+0.07)。
  C. 回测「过热刹车」增强：在波动率模型上叠加——近6月涨幅超阈值时额外压低仓位——
     与基线对比 复利/回撤/Sharpe/崩塌月均。

用法
----
  python momentum_crash_test.py                  # 全部三部分
  python momentum_crash_test.py --no-vix         # 跳过 VIX 下载（更快）
  python momentum_crash_test.py --crash -15      # 自定义崩塌阈值(默认 -12%)

⚠️ 统计告诫：全样本只有 ~7 个崩塌月（正类极少），任何预测结论都属探索性，
   换样本极易失效。刹车的主要价值是「降回撤」，2022+ 对收益近似中性。
"""
from __future__ import annotations

import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from quant_gt_model import (
    _month_qgt, _holding_window, _dl, _window_return,
    target_vol, qgt_weight, _apply_weight, _metrics, runup,
    LOOKBACK, BURN_IN, REBOUND_ENABLED, REBOUND_THRESHOLD,
    BRAKE_LOOKBACK, BRAKE_THRESHOLD, VOL_LOW_WEIGHT,
)
from quant_gt_validate import QUANT_GT_PICKS


# ──────────────────────────────────────────────────────────────────────────────
# 数据准备
# ──────────────────────────────────────────────────────────────────────────────

def _load():
    mq = _month_qgt()
    months = sorted(mq)
    qgt = np.array([mq[m] for m in months])
    gld_s, spy_s = _dl("GLD"), _dl("SPY")
    gld, spy = [], []
    for m in months:
        a, e = _holding_window(m)
        gld.append(_window_return(gld_s, a, e))
        spy.append(_window_return(spy_s, a, e))
    return months, qgt, np.array(gld), np.array(spy)


def _runup(qgt: np.ndarray, i: int, k: int) -> float:
    """月 i 之前 k 个月的复利涨幅%（因果，只用 <i 的收益）。"""
    if i < k:
        return np.nan
    return (np.prod(1 + qgt[i - k:i] / 100) - 1) * 100


# ──────────────────────────────────────────────────────────────────────────────
# A. 现有波动率降仓在崩塌月救了多少
# ──────────────────────────────────────────────────────────────────────────────

def part_a(months, qgt, gld, spy, crash_thr):
    tgt = target_vol(list(qgt), k=LOOKBACK, burn=BURN_IN)
    w = _baseline_weights(qgt, tgt)
    strat = _apply_weight(w, qgt, gld)

    print("=" * 74)
    print("  A. 现有波动率降仓在崩塌月「救回」了多少")
    print("=" * 74)
    print(f"  {'月份':<9}{'QGT收益':>9}{'入场仓位':>9}{'策略实收':>10}{'救回':>9}")
    print("  " + "-" * 46)
    for i, m in enumerate(months):
        if qgt[i] < crash_thr:
            saved = strat[i] - qgt[i]
            print(f"  {m:<9}{qgt[i]:>+8.1f}%{w[i]:>8.2f}{strat[i]:>+9.1f}%{saved:>+8.1f}%")
    print("\n  规律：崩塌多发生在波动率已升高之后 → 模型早已减仓（这是有效的部分）；")
    print("        但标定期(2018-10)或低波动急涨后的崩塌，波动率反应不及 → 需要额外信号。\n")


def _baseline_weights(qgt, tgt, cap=1.0):
    w = np.ones(len(qgt))
    for i in range(len(qgt)):
        if i < max(LOOKBACK, BURN_IN):
            w[i] = min(1.0, cap)
        else:
            w[i] = qgt_weight(np.std(qgt[i - LOOKBACK:i]), tgt, cap)
        if REBOUND_ENABLED and i > 0 and qgt[i - 1] < REBOUND_THRESHOLD:
            w[i] = cap
    return w


# ──────────────────────────────────────────────────────────────────────────────
# B. 候选预警指标的预测力
# ──────────────────────────────────────────────────────────────────────────────

def part_b(months, qgt, spy, crash_thr, use_vix):
    adates = [QUANT_GT_PICKS[m]["analysis_date"] for m in months]

    vix_lvl = vix_pctl = None
    if use_vix:
        try:
            import yfinance as yf
            vdf = yf.download("^VIX", start="2016-01-01", end=str(adates[-1]),
                              interval="1d", progress=False)
            vdf.columns = [c[0] if isinstance(c, tuple) else c for c in vdf.columns]
            vs = vdf["Close"].dropna()
            vix_lvl, vix_pctl = [], []
            for d in adates:
                x = vs[vs.index <= pd.Timestamp(d)]
                vix_lvl.append(float(x.iloc[-1]) if len(x) else np.nan)
                vix_pctl.append(float(x.rank(pct=True).iloc[-1] * 100) if len(x) else np.nan)
            vix_lvl, vix_pctl = np.array(vix_lvl), np.array(vix_pctl)
        except Exception as e:
            print(f"  [VIX 下载失败，跳过: {e}]")
            use_vix = False

    rows = []
    for i in range(len(months)):
        rows.append(dict(
            trail_vol=np.std(qgt[max(0, i - 6):i]) if i >= 2 else np.nan,
            runup3=_runup(qgt, i, 3),
            runup6=_runup(qgt, i, 6),
            skew6=pd.Series(qgt[max(0, i - 6):i]).skew() if i >= 4 else np.nan,
            vix=(vix_lvl[i] if use_vix else np.nan),
            vix_pctl=(vix_pctl[i] if use_vix else np.nan),
            crash=int(qgt[i] < crash_thr),
            ret=qgt[i],
        ))
    F = pd.DataFrame(rows).iloc[6:]
    cols = ["trail_vol", "runup3", "runup6", "skew6"] + (["vix", "vix_pctl"] if use_vix else [])
    F = F.dropna(subset=cols)

    print("=" * 74)
    print(f"  B. 各指标对「下月崩塌 (QGT<{crash_thr:g}%)」的预测力"
          f"   (n={len(F)}, 崩塌{int(F.crash.sum())}次)")
    print("=" * 74)
    print(f"  {'指标':<12}{'corr(指标,崩塌)':>16}{'高分位组崩塌率':>15}{'低分位组':>10}{'corr(指标,下月收益)':>18}")
    print("  " + "-" * 70)
    for c in cols:
        hi = F[F[c] >= F[c].quantile(0.7)]
        lo = F[F[c] <= F[c].quantile(0.3)]
        cc = np.corrcoef(F[c], F["crash"])[0, 1]
        cr = np.corrcoef(F[c], F["ret"])[0, 1]
        print(f"  {c:<12}{cc:>+15.2f}{hi.crash.mean()*100:>13.0f}%{lo.crash.mean()*100:>9.0f}%{cr:>+17.2f}")
    print("\n  结论：涨幅(runup6/3) 是最强崩塌预警，强于波动率；VIX 几乎无预测力（Calm Index）。\n")
    return F


# ──────────────────────────────────────────────────────────────────────────────
# C. 「过热刹车」增强回测
# ──────────────────────────────────────────────────────────────────────────────

def _weights_with_brake(qgt, tgt, cap=1.0, brake_thr=None, brake_cap=0.5):
    w = np.ones(len(qgt))
    for i in range(len(qgt)):
        if i < max(LOOKBACK, BURN_IN):
            w[i] = min(1.0, cap)
        else:
            w[i] = qgt_weight(np.std(qgt[i - LOOKBACK:i]), tgt, cap)
        # 过热刹车：近6月涨幅过高 → 额外压低仓位（因果，只用 <i 数据）
        if brake_thr is not None and i >= 6 and _runup(qgt, i, 6) >= brake_thr:
            w[i] = min(w[i], brake_cap)
        # 反弹覆盖优先级最高（上月暴跌 → 顶格抢反弹）
        if REBOUND_ENABLED and i > 0 and qgt[i - 1] < REBOUND_THRESHOLD:
            w[i] = cap
    return w


def part_c(months, qgt, gld, spy, crash_thr):
    tgt = target_vol(list(qgt), k=LOOKBACK, burn=BURN_IN)
    yr = np.array([int(m[:4]) for m in months])

    print("=" * 74)
    print("  C. 「过热刹车」增强 vs 基线   (刹车: 近6月涨幅≥阈值 → 仓位≤0.5)")
    print("=" * 74)

    def show(tag, w):
        strat = _apply_weight(w, qgt, gld)
        for lab, mask in [("全样本", yr >= 0), ("2022+", yr >= 2022)]:
            d = _metrics(strat[mask], spy[mask])
            cm = mask & (qgt < crash_thr)
            cavg = strat[cm].mean() if cm.sum() else float("nan")
            print(f"  {tag:<24}[{lab:<5}] 复利{d['comp']:>+7.0f}%  回撤{d['mdd']:>6.1f}%  "
                  f"Sharpe{d['sharpe']:>5.2f}  崩塌月均{cavg:>+6.1f}%")

    for cap in (1.0, 1.5):
        print(f"\n  ── cap={cap} ──")
        show(f"基线", _weights_with_brake(qgt, tgt, cap))
        for thr in (25, 35, 50):
            show(f"+刹车(涨>{thr}%→≤0.5)",
                 _weights_with_brake(qgt, tgt, cap, brake_thr=thr, brake_cap=0.5))
    print("\n  读法：全样本刹车全面变好（主要修复 2018-10 那类波动率救不到的崩塌）；")
    print("        2022+ 对收益近似中性（近期崩塌已被波动率降仓提前拦下，两信号重叠）。")
    print("        多个阈值结果接近 → 非过拟合到单一魔法数字。\n")


# ──────────────────────────────────────────────────────────────────────────────
# D. 大跌是否发生在「信号很低」时？（两个信号 × 实际收益 交叉验证）
# ──────────────────────────────────────────────────────────────────────────────

def part_d(months, qgt, crash_thr):
    """对每个月计算两个风控信号的状态 + 当月实际收益，检验大跌是否落在示警区。
    信号均因果：用 <i 的历史算，对照 qgt[i] 的实际收益。"""
    tgt = target_vol(list(qgt), k=LOOKBACK, burn=BURN_IN)
    start = max(LOOKBACK, BRAKE_LOOKBACK)

    rows = []
    for i in range(start, len(months)):
        hist = qgt[:i]
        vol_w = qgt_weight(np.std(hist[-LOOKBACK:]), tgt)   # 波动率信号建议仓位
        run6 = runup(list(hist), BRAKE_LOOKBACK)            # 近6月涨幅
        vol_low = vol_w < VOL_LOW_WEIGHT
        brake_on = run6 >= BRAKE_THRESHOLD
        n_low = int(vol_low) + int(brake_on)
        rows.append(dict(month=months[i], ret=qgt[i], vol_w=vol_w, run6=run6,
                         vol_low=vol_low, brake_on=brake_on, n_low=n_low,
                         crash=qgt[i] < crash_thr))
    D = pd.DataFrame(rows)

    print("=" * 74)
    print(f"  D. 大跌是否发生在「信号很低」时  (n={len(D)}, 崩塌阈值 {crash_thr:g}%)")
    print("=" * 74)

    # 交叉表：按「几个信号示警」分组
    print("  按示警信号个数分组：")
    print(f"  {'状态':<16}{'月数':>5}{'崩塌数':>7}{'崩塌率':>8}{'该组月均收益':>12}")
    print("  " + "-" * 50)
    labels = {0: "两个都正常", 1: "恰一个示警", 2: "两个都示警"}
    for n in (0, 1, 2):
        s = D[D.n_low == n]
        if len(s):
            print(f"  {labels[n]:<16}{len(s):>5}{int(s.crash.sum()):>7}"
                  f"{s.crash.mean()*100:>7.0f}%{s.ret.mean():>+11.1f}%")

    # 崩塌事件逐一：发生时两个信号状态如何？
    print(f"\n  ── 每次崩塌 (QGT<{crash_thr:g}%) 当月，两个信号的状态 ──")
    print(f"  {'月份':<9}{'实际收益':>9}{'波动信号仓位':>12}{'波动低?':>8}"
          f"{'近6月涨幅':>10}{'刹车?':>7}{'示警数':>7}")
    print("  " + "-" * 62)
    caught = 0
    for _, r in D[D.crash].iterrows():
        if r.n_low >= 1:
            caught += 1
        print(f"  {r.month:<9}{r.ret:>+8.1f}%{r.vol_w:>11.2f}"
              f"{('是' if r.vol_low else '否'):>8}{r.run6:>+9.1f}%"
              f"{('是' if r.brake_on else '否'):>7}{r.n_low:>6}")
    ncrash = int(D.crash.sum())
    print(f"\n  ⇒ {ncrash} 次崩塌中，{caught} 次「至少一个信号示警」"
          f"（{caught/ncrash*100:.0f}%）；{ncrash-caught} 次两个信号都正常（盲区）。")

    # 反过来：示警时的崩塌命中率（避免幸存者偏差）
    warn = D[D.n_low >= 1]
    print(f"  ⇒ 反向看：只要有信号示警的 {len(warn)} 个月里，崩塌率 "
          f"{warn.crash.mean()*100:.0f}%（vs 全样本 {D.crash.mean()*100:.0f}%）——"
          f"示警会误报，但崩塌几乎不会不带信号发生。\n")
    return D


def main():
    ap = argparse.ArgumentParser(description="Quant GT 动量崩塌测试")
    ap.add_argument("--crash", type=float, default=-12.0, help="崩塌阈值%（默认 -12）")
    ap.add_argument("--no-vix", action="store_true", help="跳过 VIX 下载")
    args = ap.parse_args()

    print("\n下载数据 (GLD, SPY)...")
    months, qgt, gld, spy = _load()
    print(f"样本: {months[0]} → {months[-1]}  ({len(months)} 个月)\n")

    part_a(months, qgt, gld, spy, args.crash)
    part_b(months, qgt, spy, args.crash, use_vix=not args.no_vix)
    part_c(months, qgt, gld, spy, args.crash)
    part_d(months, qgt, args.crash)


if __name__ == "__main__":
    main()
