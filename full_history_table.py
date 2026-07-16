"""full_history_table.py - 全历史月度策略明细表"""
import pandas as pd, numpy as np, os, sys
os.chdir(r"c:\Users\bowen\PycharmProjects\mcp-finbot")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import warnings; warnings.filterwarnings("ignore")

# 复用两套预测：
#   2016-2021 → 全量模型预测（in-sample，标注 IS）
#   2022-2026 → walk-forward 预测（out-of-sample，标注 OOS）
from quant_gt_predictor import (mf, ALL_FEAT, predict_proba,
                                 test_wf, pred2, sc2)
from quant_gt_validate import QUANT_GT_PICKS

# ── 策略：≥55% 满仓 / 40-55% 半仓 / <40% 空仓 ─────────────────────────────
def position(p):
    if p >= 0.55:   return 1.00, "满仓100%"
    elif p >= 0.40: return 0.50, "半仓 50%"
    else:           return 0.00, "空仓  0%"

# ── 全量模型概率（in-sample for 2016-2021）
full_probs = dict(zip(mf["month"], predict_proba(mf[ALL_FEAT].values)))

# ── Walk-forward 概率（out-of-sample for 2022-2026）
wf_sorted = test_wf.sort_values("analysis_date")
wf_probs  = dict(zip(wf_sorted["month"], pred2(wf_sorted[ALL_FEAT].values)))

def get_prob(month):
    yr = int(month[:4])
    if yr >= 2022:
        return wf_probs.get(month, full_probs.get(month))
    return full_probs.get(month)

# ── 构建行 ──────────────────────────────────────────────────────────────────
rows = []
eq   = 1.0
eq_qgt = 1.0

for m in sorted(QUANT_GT_PICKS.keys()):
    info  = QUANT_GT_PICKS[m]
    picks = info["picks"]   # list of (ticker, return)
    tickers = [t for t, _ in picks]
    rets    = [r for _, r in picks]
    qgt_avg = np.mean(rets)

    prob = get_prob(m)
    if prob is None:
        continue

    wt, decision = position(prob)
    strat_ret = qgt_avg * wt

    eq     *= (1 + strat_ret/100)
    eq_qgt *= (1 + qgt_avg/100)

    yr = int(m[:4])
    sample_type = "OOS" if yr >= 2022 else "IS"

    rows.append({
        "月份":       m,
        "样本":       sample_type,
        "选股":       " / ".join(tickers),
        "各股收益":   " / ".join(f"{t}:{r:+.1f}%" for t, r in picks),
        "QGT均收益":  round(qgt_avg, 2),
        "模型概率":   round(prob * 100, 1),
        "仓位决策":   decision,
        "策略收益":   round(strat_ret, 2),
        "策略累计":   round((eq - 1) * 100, 1),
        "QGT累计":    round((eq_qgt - 1) * 100, 1),
    })

df = pd.DataFrame(rows)

# ── 保存 CSV ────────────────────────────────────────────────────────────────
df.to_csv("full_history_strategy.csv", index=False, encoding="utf-8-sig")

# ── 控制台打印（紧凑版）────────────────────────────────────────────────────
print("=" * 110)
print("  Quant GT × 量化模型  全历史月度明细")
print("  策略：≥55% 满仓 | 40-55% 半仓 | <40% 空仓    (OOS=样本外真实预测 IS=训练内参考)")
print("=" * 110)

DECISION_SYMBOL = {"满仓100%": "●●●", "半仓 50%": "●●○", "空仓  0%": "○○○"}

header = (f"{'月份':<10} {'S':>3} {'选股':<30} {'QGT均':>8} "
          f"{'概率':>7} {'仓位':>5} {'策略收益':>9} {'策略累计':>10} {'QGT累计':>10}")
print(f"\n{header}")
print("-" * 110)

prev_yr = None
for _, r in df.iterrows():
    yr = r["月份"][:4]
    if yr != prev_yr:
        yrdata = df[df["月份"].str.startswith(yr)]
        yr_qgt   = (np.prod(1 + yrdata["QGT均收益"].values/100) - 1)*100
        yr_strat = (np.prod(1 + yrdata["策略收益"].values/100) - 1)*100
        print(f"\n  ── {yr}年 ──  QGT年复利={yr_qgt:+.1f}%  策略年复利={yr_strat:+.1f}%  "
              f"({'OOS 真实测试' if int(yr)>=2022 else 'IS 参考'})")
        prev_yr = yr

    sym   = DECISION_SYMBOL.get(r["仓位决策"], "   ")
    tickers_short = r["选股"][:28]
    color_qgt   = "+" if r["QGT均收益"] > 0 else "-" if r["QGT均收益"] < 0 else " "
    color_strat = "+" if r["策略收益"] > 0 else "-" if r["策略收益"] < 0 else " "

    line = (f"  {r['月份']:<8} {r['样本']:>3} {tickers_short:<30} "
            f"{r['QGT均收益']:>+7.1f}%  {r['模型概率']:>5.1f}%  {sym} "
            f"{r['策略收益']:>+7.1f}%  {r['策略累计']:>+8.1f}%  {r['QGT累计']:>+8.1f}%")
    print(line)

# ── 汇总统计 ────────────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print("  汇总统计")
print("=" * 110)

# 全部
total_strat = (np.prod(1 + df["策略收益"].values/100) - 1)*100
total_qgt   = (np.prod(1 + df["QGT均收益"].values/100) - 1)*100
oos = df[df["样本"] == "OOS"]
is_ = df[df["样本"] == "IS"]

oos_strat = (np.prod(1 + oos["策略收益"].values/100) - 1)*100
oos_qgt   = (np.prod(1 + oos["QGT均收益"].values/100) - 1)*100
is_strat  = (np.prod(1 + is_["策略收益"].values/100) - 1)*100
is_qgt    = (np.prod(1 + is_["QGT均收益"].values/100) - 1)*100

def mdd(rets):
    eq = np.cumprod(1 + np.array(rets)/100)
    pk = np.maximum.accumulate(eq)
    return ((eq - pk) / pk * 100).min()

def sharpe(rets):
    r = np.array(rets)
    return r.mean()/r.std()*np.sqrt(12) if r.std()>0 else 0

print(f"\n  {'集合':<12} {'策略复利':>10} {'QGT复利':>10} {'策略MaxDD':>10} {'QGT MaxDD':>10} {'策略Sharpe':>11} {'QGT Sharpe':>11}")
print(f"  {'-'*75}")
for label, sub in [("全部(参考)", df), ("OOS 2022+(真实)", oos), ("IS 2016-21(参考)", is_)]:
    s_c = (np.prod(1 + sub["策略收益"].values/100)-1)*100
    q_c = (np.prod(1 + sub["QGT均收益"].values/100)-1)*100
    s_d = mdd(sub["策略收益"].tolist())
    q_d = mdd(sub["QGT均收益"].tolist())
    s_sh = sharpe(sub["策略收益"].tolist())
    q_sh = sharpe(sub["QGT均收益"].tolist())
    print(f"  {label:<18} {s_c:>+9.1f}%  {q_c:>+9.1f}%  {s_d:>+9.1f}%  {q_d:>+9.1f}%  {s_sh:>10.2f}  {q_sh:>10.2f}")

# 仓位分布
print(f"\n  仓位决策分布（OOS 2022-2026）:")
for d, grp in oos.groupby("仓位决策"):
    avg = grp["QGT均收益"].mean()
    win = (grp["QGT均收益"] > 0).mean()*100
    print(f"    {d}  n={len(grp):>3}月  QGT均收益={avg:>+.1f}%  胜率={win:.0f}%")

print(f"\n  详细数据已保存至 full_history_strategy.csv")
