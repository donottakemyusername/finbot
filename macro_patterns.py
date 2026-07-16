"""macro_patterns.py - 深挖触发大回撤的具体条件"""
import pandas as pd
import numpy as np
import yfinance as yf
import os, sys
from datetime import timedelta

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.chdir(r"c:\Users\bowen\PycharmProjects\mcp-finbot")

from quant_gt_validate import QUANT_GT_PICKS

# ─── 构建月度基础表 ──────────────────────────────────────────────────────────
files = {"2019":"results_2019.csv","2022":"results_2022.csv",
         "2023":"results_2023.csv","2024":"results_2024.csv",
         "2025+26":"quant_gt_trinity_results.csv"}
all_dfs = []
for yr, f in files.items():
    d = pd.read_csv(f, encoding="utf-8-sig"); d["year_label"]=yr; all_dfs.append(d)
stock_df = pd.concat(all_dfs, ignore_index=True)
stock_df = stock_df[~stock_df["trinity_signal"].isin(["ERROR","SKIP"])].copy()
stock_df["quant_gt_return"] = pd.to_numeric(stock_df["quant_gt_return"], errors="coerce")

month_qgt = {m: stock_df[stock_df["month"]==m]["quant_gt_return"].mean()
             for m in sorted(stock_df["month"].unique())}

# ─── 下载宏观数据 ────────────────────────────────────────────────────────────
print("下载数据...")
def dl(ticker, start="2018-01-01", end="2026-07-15"):
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df["Close"].dropna()

spy = dl("SPY"); qqq = dl("QQQ"); vix = dl("^VIX")
tlt = dl("TLT"); hyg = dl("HYG")

def px(s, d):
    mask = s.index <= pd.Timestamp(d)
    return float(s[mask].iloc[-1]) if mask.any() else None

def ret(s, d, days):
    p1 = px(s, d); p0 = px(s, pd.Timestamp(d)-timedelta(days=days))
    return (p1-p0)/p0*100 if p1 and p0 and p0!=0 else None

# ─── 行业分类 ─────────────────────────────────────────────────────────────────
SECTOR = {
    "NVDA":"Semi","AMD":"Semi","MSTR":"Crypto","COIN":"Crypto",
    "SMCI":"Semi","CVNA":"ConsDisc","TSLA":"ConsDisc","APP":"Tech",
    "PLTR":"Tech","NET":"Tech","CRWD":"Tech","RKLB":"Aero",
    "HOOD":"Fintech","RBLX":"Tech","AFRM":"Fintech","SOFI":"Fintech",
    "MU":"Semi","WDC":"Semi","SNDK":"Semi","CIEN":"Semi",
    "ALAB":"Semi","MRVL":"Semi","MXL":"Semi","INTC":"Semi",
    "AAOI":"Semi","VIAV":"Semi","LITE":"Semi","FORM":"Semi",
    "COHR":"Semi","IONQ":"Tech","SYM":"Tech",
    "OXY":"Energy","EQT":"Energy","HAL":"Energy","VLO":"Energy",
    "CTRA":"Energy","DVN":"Energy","AA":"Metal","BKR":"Energy",
    "EOG":"Energy","MPC":"Energy","SLB":"Energy",
    "AU":"Gold","NEM":"Gold","RGLD":"Gold",
    "ALNY":"Biotech","BIIB":"Biotech","LLY":"Pharma","RVMD":"Biotech",
    "VRTX":"Biotech","GILD":"Biotech","INSM":"Biotech",
    "META":"Tech","HUBS":"Tech","CRM":"Tech","MDB":"Tech",
    "DELL":"Tech","ETHA":"Crypto","RDDT":"Tech","WBD":"Media",
    "IRM":"REIT","TPR":"ConsDisc","BJ":"Retail","PM":"Tobacco",
    "OKTA":"Tech","ZS":"Tech","ANET":"Semi","NRG":"Utility",
    "VST":"Utility","VRT":"Indus","FSLR":"Solar","TER":"Semi",
    "RKT":"Fintech","GDDY":"Tech","WSM":"Retail","AXON":"Tech",
    "SCCO":"Metal","FCX":"Metal","LVS":"Gaming","GE":"Indus",
    "HUBS":"Tech","CRM":"Tech","SMCI":"Semi","PLTR":"Tech",
    "MDB":"Tech","MRVL":"Semi","NVDA":"Semi","RCL":"Travel",
    "OVV":"Energy","DELL":"Tech","MPC":"Energy","HAL":"Energy",
    "LLY":"Pharma","CRWD":"Tech","PGR":"Insure","WSM":"Retail",
    "URA":"Nuclear","TWLO":"Tech","WDAY":"Tech","OKTA":"Tech",
    "SBUX":"Retail","CHD":"ConsStaple","MKC":"ConsStaple",
    "HRL":"ConsStaple","KEYS":"Tech","NTNX":"Tech","W":"Retail",
    "TTD":"AdTech","CDNS":"EDA","ANET":"Semi","QCOM":"Semi",
    "ALGN":"Medical","IR":"Indus","VEEV":"Health","BX":"PE",
    "BR":"Fintech","CPRT":"ConsDisc","DXCM":"Medical","TTWO":"Gaming",
    "HSY":"ConsStaple","WDC":"Semi","KLAC":"Semi","TGT":"Retail",
    "LRCX":"Semi","DG":"Retail","TSLA":"ConsDisc",
    "ON":"Semi","F":"Auto","DLTR":"Retail","CIEN":"Semi",
    "CVX":"Energy","MNST":"Bev","LNG":"Energy","VICI":"REIT",
    "CDNS":"EDA","DECK":"ConsDisc","DKS":"Retail","ALNY":"Biotech",
}

def sector_conc(picks):
    from collections import Counter
    secs = [SECTOR.get(t, "Other") for t in picks]
    c = Counter(secs)
    return max(c.values())/len(picks), c.most_common(1)[0][0]

# ─── 构建月度特征矩阵 ─────────────────────────────────────────────────────────
months_sorted = sorted(QUANT_GT_PICKS.keys())
rows = []
prev_dominant = None
prev_qgt = None
semi_streak = 0

for i, m in enumerate(months_sorted):
    if m not in month_qgt:
        continue
    info = QUANT_GT_PICKS[m]
    ad   = info["analysis_date"]
    picks = [p[0] for p in info["picks"]]
    qgt_r = month_qgt[m]

    conc, dom = sector_conc(picks)

    # 连续同一行业月数
    if dom == prev_dominant:
        streak = rows[-1]["sector_streak"] + 1 if rows else 1
    else:
        streak = 1
    prev_dominant = dom

    # 前月 QGT 收益
    prev_r = prev_qgt if prev_qgt is not None else 0.0
    prev_qgt = qgt_r

    # 过去3月 QGT 累计
    prev3 = [month_qgt.get(months_sorted[j]) for j in range(i-3, i) if j>=0]
    prev3_sum = sum(x for x in prev3 if x is not None)

    # 宏观指标
    vix_level   = px(vix, ad)
    spy_1m      = ret(spy, ad, 30)
    qqq_1m      = ret(qqq, ad, 30)
    spy_3m      = ret(spy, ad, 90)
    qqq_3m      = ret(qqq, ad, 90)
    tlt_1m      = ret(tlt, ad, 30)
    hyg_1m      = ret(hyg, ad, 30)

    # 前1月个股最大涨幅（Quant GT 可能选了已经涨很多的股）
    pick_prior_rets = []
    for t in picks:
        try:
            r1m = ret(dl(t), ad, 30)
            if r1m is not None:
                pick_prior_rets.append(r1m)
        except:
            pass
    avg_pick_prior = np.mean(pick_prior_rets) if pick_prior_rets else None
    max_pick_prior = max(pick_prior_rets) if pick_prior_rets else None

    rows.append({
        "month": m, "analysis_date": ad, "qgt_avg": qgt_r,
        "dominant_sector": dom, "sector_conc": conc, "sector_streak": streak,
        "prev_month_qgt": prev_r, "prev3_qgt_sum": prev3_sum,
        "vix_level": vix_level, "spy_1m": spy_1m, "qqq_1m": qqq_1m,
        "spy_3m": spy_3m, "qqq_3m": qqq_3m,
        "tlt_1m": tlt_1m, "hyg_1m": hyg_1m,
        "avg_pick_prior_1m": avg_pick_prior,
        "max_pick_prior_1m": max_pick_prior,
        "bad_month": qgt_r < -5,
    })

mf = pd.DataFrame(rows)
print(f"特征矩阵: {mf.shape}")

# ─── 相关系数（含新特征）────────────────────────────────────────────────────
print("\n" + "="*65)
print("  所有特征与 QGT 月收益的相关系数（从强到弱）")
print("="*65 + "\n")

feat_cols = ["vix_level","spy_1m","qqq_1m","spy_3m","qqq_3m",
             "tlt_1m","hyg_1m","sector_conc","sector_streak",
             "prev_month_qgt","prev3_qgt_sum",
             "avg_pick_prior_1m","max_pick_prior_1m"]
feat_labels = {
    "vix_level":"VIX水平", "spy_1m":"SPY近1月涨",
    "qqq_1m":"QQQ近1月涨", "spy_3m":"SPY近3月涨", "qqq_3m":"QQQ近3月涨",
    "tlt_1m":"TLT近1月（债）", "hyg_1m":"HYG近1月（高收益债）",
    "sector_conc":"行业集中度", "sector_streak":"连续同行业月数",
    "prev_month_qgt":"上月QGT收益", "prev3_qgt_sum":"前3月QGT累计",
    "avg_pick_prior_1m":"选股前1月平均涨幅", "max_pick_prior_1m":"选股前1月最大涨幅",
}

corr_res = []
for col in feat_cols:
    valid = mf[[col,"qgt_avg"]].dropna()
    if len(valid) < 8: continue
    r = valid[col].corr(valid["qgt_avg"])
    corr_res.append((feat_labels.get(col,col), r, len(valid)))
corr_res.sort(key=lambda x: abs(x[1]), reverse=True)
for label, r, n in corr_res:
    bar = "█" * max(1, int(abs(r)*25))
    s = "+" if r>0 else "-"
    print(f"  {label:<25} r={r:>+.3f}  n={n:>2}  {s}{bar}")

# ─── 关键模式分析 ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  回撤触发条件：多因子组合")
print("="*65)

# Pattern 1: 前3月累计 + 高选股前涨幅
print("\n  [条件1] 前3月QGT累计 > +25%（过热）时的表现:")
hot  = mf[mf["prev3_qgt_sum"] > 25]
cold = mf[mf["prev3_qgt_sum"] <= 25]
print(f"    过热 ({len(hot)}个月): 均收益={hot['qgt_avg'].mean():+.1f}%  胜率={(hot['qgt_avg']>0).mean()*100:.0f}%  大回撤率={(hot['bad_month']).mean()*100:.0f}%")
print(f"    正常 ({len(cold)}个月): 均收益={cold['qgt_avg'].mean():+.1f}%  胜率={(cold['qgt_avg']>0).mean()*100:.0f}%  大回撤率={(cold['bad_month']).mean()*100:.0f}%")

# Pattern 2: 连续同一行业 >= 3个月
print("\n  [条件2] 连续同一行业 ≥3 个月时的表现:")
long_str  = mf[mf["sector_streak"] >= 3]
short_str = mf[mf["sector_streak"] < 3]
print(f"    连续≥3月 ({len(long_str)}个月): 均收益={long_str['qgt_avg'].mean():+.1f}%  胜率={(long_str['qgt_avg']>0).mean()*100:.0f}%  大回撤率={(long_str['bad_month']).mean()*100:.0f}%")
print(f"    连续<3月 ({len(short_str)}个月): 均收益={short_str['qgt_avg'].mean():+.1f}%  胜率={(short_str['qgt_avg']>0).mean()*100:.0f}%  大回撤率={(short_str['bad_month']).mean()*100:.0f}%")

# Pattern 3: 选股前1月平均已涨 > 20%
print("\n  [条件3] 选股时该批股票前1月已平均涨 > +20% 时:")
high_pre = mf[mf["avg_pick_prior_1m"] > 20]
low_pre  = mf[mf["avg_pick_prior_1m"] <= 20]
print(f"    前涨>20% ({len(high_pre)}个月): 均收益={high_pre['qgt_avg'].mean():+.1f}%  胜率={(high_pre['qgt_avg']>0).mean()*100:.0f}%  大回撤率={(high_pre['bad_month']).mean()*100:.0f}%")
print(f"    前涨≤20% ({len(low_pre)}个月): 均收益={low_pre['qgt_avg'].mean():+.1f}%  胜率={(low_pre['qgt_avg']>0).mean()*100:.0f}%  大回撤率={(low_pre['bad_month']).mean()*100:.0f}%")

# Pattern 4: VIX + SPY 组合
print("\n  [条件4] VIX>20 且 SPY近1月 <0%（恐慌 + 下跌）:")
fear = mf[(mf["vix_level"]>20) & (mf["spy_1m"]<0)]
calm = mf[~((mf["vix_level"]>20) & (mf["spy_1m"]<0))]
print(f"    恐慌期 ({len(fear)}个月): 均收益={fear['qgt_avg'].mean():+.1f}%  胜率={(fear['qgt_avg']>0).mean()*100:.0f}%  大回撤率={(fear['bad_month']).mean()*100:.0f}%")
print(f"    正常期 ({len(calm)}个月): 均收益={calm['qgt_avg'].mean():+.1f}%  胜率={(calm['qgt_avg']>0).mean()*100:.0f}%  大回撤率={(calm['bad_month']).mean()*100:.0f}%")

# Pattern 5: 上月 QGT 超高
print("\n  [条件5] 上月QGT收益 > +30%（强势后惯性衰减）:")
after_big = mf[mf["prev_month_qgt"] > 30]
other     = mf[mf["prev_month_qgt"] <= 30]
print(f"    超高月后 ({len(after_big)}个月): 均收益={after_big['qgt_avg'].mean():+.1f}%  胜率={(after_big['qgt_avg']>0).mean()*100:.0f}%  大回撤率={(after_big['bad_month']).mean()*100:.0f}%")
print(f"    普通月后 ({len(other)}个月):  均收益={other['qgt_avg'].mean():+.1f}%  胜率={(other['qgt_avg']>0).mean()*100:.0f}%  大回撤率={(other['bad_month']).mean()*100:.0f}%")

# ─── 综合评分（简单预警系统）───────────────────────────────────────────────
print("\n" + "="*65)
print("  简单预警评分（每条件触发 +1 分，分越高风险越大）")
print("="*65)

def risk_score(r):
    score = 0
    reasons = []
    if pd.notna(r.get("prev3_qgt_sum")) and r["prev3_qgt_sum"] > 25:
        score += 2; reasons.append(f"前3月累计过热(+{r['prev3_qgt_sum']:.0f}%)")
    if pd.notna(r.get("sector_streak")) and r["sector_streak"] >= 3:
        score += 2; reasons.append(f"连续{r['sector_streak']:.0f}月同行业({r['dominant_sector']})")
    if pd.notna(r.get("avg_pick_prior_1m")) and r["avg_pick_prior_1m"] > 20:
        score += 1; reasons.append(f"选股前1月均涨{r['avg_pick_prior_1m']:.0f}%")
    if pd.notna(r.get("vix_level")) and r["vix_level"] > 20:
        score += 1; reasons.append(f"VIX={r['vix_level']:.0f}")
    if pd.notna(r.get("spy_1m")) and r["spy_1m"] < -3:
        score += 1; reasons.append(f"SPY近1月{r['spy_1m']:.1f}%")
    if pd.notna(r.get("prev_month_qgt")) and r["prev_month_qgt"] > 30:
        score += 1; reasons.append(f"上月QGT+{r['prev_month_qgt']:.0f}%")
    return score, reasons

mf["risk_score"] = mf.apply(lambda r: risk_score(r)[0], axis=1)

print(f"\n{'月份':<12} {'QGT均收':>9} {'风险分':>6} {'大回撤?':>8}  触发条件")
print("-"*90)
for _, r in mf.sort_values("analysis_date").iterrows():
    sc, reasons = risk_score(r)
    flag = "⚠️ YES" if r["bad_month"] else "  no"
    if sc >= 2 or r["bad_month"]:
        print(f"  {r['month']:<10} {r['qgt_avg']:>+8.1f}%  {sc:>5}   {flag}  {' | '.join(reasons[:3])}")

# 评分系统有效性
print("\n  预警系统准确性:")
for threshold in [2, 3, 4]:
    warned = mf[mf["risk_score"] >= threshold]
    not_warned = mf[mf["risk_score"] < threshold]
    if len(warned) == 0: continue
    tp = warned["bad_month"].sum()
    fp = len(warned) - tp
    fn = not_warned["bad_month"].sum()
    tn = len(not_warned) - fn
    precision = tp/(tp+fp)*100 if (tp+fp) else 0
    recall    = tp/(tp+fn)*100 if (tp+fn) else 0
    print(f"  阈值≥{threshold}: 预警{len(warned)}个月  命中大回撤={tp}/{(mf['bad_month']).sum()}  精确率={precision:.0f}%  召回率={recall:.0f}%")
