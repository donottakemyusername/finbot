"""quant_gt_predictor.py
=======================
基于宏观 + 行业动量特征，预测 Quant GT 下一个月的涨跌概率。

训练：2016-2023（排除 2017 作为测试集，walk-forward 也验证）
测试：2017（out-of-sample）+ 2024-2026（真实前向验证）

输出：
  - 特征重要性
  - 每月风险评分（0-100）+ 涨跌概率
  - Walk-forward 回测曲线
  - 最终模型应用于历史所有月份
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import date, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (brier_score_loss, roc_auc_score,
                              classification_report, confusion_matrix)
from sklearn.pipeline import Pipeline

os.chdir(r"c:\Users\bowen\PycharmProjects\mcp-finbot")
sys.path.insert(0, ".")
from quant_gt_validate import QUANT_GT_PICKS

# ──────────────────────────────────────────────────────────────────────────────
# 1. 月度 QGT 收益表
# ──────────────────────────────────────────────────────────────────────────────
month_qgt: dict[str, float] = {}
for m, info in QUANT_GT_PICKS.items():
    rets = [r for _, r in info["picks"]]
    month_qgt[m] = float(np.mean(rets))

months_sorted = sorted(month_qgt.keys())
print(f"总月份数: {len(months_sorted)}  ({months_sorted[0]} → {months_sorted[-1]})")

# ──────────────────────────────────────────────────────────────────────────────
# 2. 下载宏观数据
# ──────────────────────────────────────────────────────────────────────────────
print("\n下载宏观数据...")
def dl(ticker):
    df = yf.download(ticker, start="2015-01-01", end="2026-07-15",
                     interval="1d", progress=False, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df["Close"].dropna()

SPY  = dl("SPY");  QQQ  = dl("QQQ");  VIX  = dl("^VIX")
TLT  = dl("TLT");  HYG  = dl("HYG");  IWM  = dl("IWM")
print("  宏观数据下载完成")

def px(s, d):
    mask = s.index <= pd.Timestamp(d)
    return float(s[mask].iloc[-1]) if mask.any() else np.nan

def pret(s, d, days):
    p1 = px(s, d)
    p0 = px(s, pd.Timestamp(d) - timedelta(days=days))
    return (p1-p0)/p0*100 if (not np.isnan(p1)) and (not np.isnan(p0)) and p0!=0 else np.nan

def above_ma(s, d, window=200):
    mask = s.index <= pd.Timestamp(d)
    sub  = s[mask]
    if len(sub) < window: return np.nan
    return float(sub.iloc[-1] > sub.iloc[-window:].mean())

# ──────────────────────────────────────────────────────────────────────────────
# 3. 行业分类
# ──────────────────────────────────────────────────────────────────────────────
SECTOR = {
    "NVDA":"Semi","AMD":"Semi","MU":"Semi","WDC":"Semi","SNDK":"Semi",
    "CIEN":"Semi","ALAB":"Semi","MRVL":"Semi","MXL":"Semi","INTC":"Semi",
    "AAOI":"Semi","VIAV":"Semi","LITE":"Semi","FORM":"Semi","COHR":"Semi",
    "ANET":"Semi","TER":"Semi","KLAC":"Semi","LRCX":"Semi","SMCI":"Semi",
    "MSTR":"Crypto","COIN":"Crypto","ETHA":"Crypto",
    "CVNA":"ConsDisc","TSLA":"ConsDisc","W":"Retail","AMZN":"Tech",
    "APP":"Tech","PLTR":"Tech","NET":"Tech","CRWD":"Tech","RKLB":"Aero",
    "HOOD":"Fintech","RBLX":"Tech","AFRM":"Fintech","SOFI":"Fintech",
    "RKT":"Fintech","IBKR":"Fintech","ROST":"Retail","KR":"Retail",
    "OXY":"Energy","EQT":"Energy","HAL":"Energy","VLO":"Energy",
    "CTRA":"Energy","DVN":"Energy","BKR":"Energy","EOG":"Energy",
    "MPC":"Energy","SLB":"Energy","LNG":"Energy","AA":"Metal",
    "SCCO":"Metal","FCX":"Metal",
    "AU":"Gold","NEM":"Gold","RGLD":"Gold",
    "ALNY":"Biotech","BIIB":"Biotech","LLY":"Pharma","RVMD":"Biotech",
    "VRTX":"Biotech","GILD":"Biotech","INSM":"Biotech","MRNA":"Biotech",
    "ARWR":"Biotech","RMD":"Medical","DXCM":"Medical","ALGN":"Medical",
    "PODD":"Medical","GH":"Medical",
    "META":"Tech","HUBS":"Tech","CRM":"Tech","MDB":"Tech","DELL":"Tech",
    "RDDT":"Tech","OKTA":"Tech","ZS":"Tech","FTNT":"Tech","PANW":"Tech",
    "NOW":"Tech","DDOG":"Tech","U":"Tech","IONQ":"Tech","SYM":"Tech",
    "NFLX":"Media","ROKU":"Media","TTD":"AdTech","WBD":"Media",
    "IRM":"REIT","VICI":"REIT","TPR":"ConsDisc","BJ":"Retail",
    "PM":"Tobacco","NEM":"Gold","NUE":"Metal","DECK":"ConsDisc",
    "TWLO":"Tech","WDAY":"Tech","PYPL":"Fintech","KEYS":"Tech",
    "NTNX":"Tech","CDNS":"EDA","QCOM":"Semi","VST":"Utility",
    "VRT":"Indus","FSLR":"Solar","NRG":"Utility","BX":"PE",
    "ZM":"Tech","PINS":"Social","SNAP":"Social","Z":"PropTech",
    "ZG":"PropTech","CVNA":"ConsDisc","BA":"Aero","TSLA":"ConsDisc",
    "GE":"Indus","CARR":"Indus","WAT":"Indus","J":"Indus",
    "SWKS":"Semi","CGNX":"Indus","NUE":"Metal",
    "WSM":"Retail","PGR":"Insure","URA":"Nuclear","AXON":"Tech",
    "GDDY":"Tech","REMX":"ETF","LVS":"Gaming","ADSK":"Tech",
    "SATS":"Tech","CFG":"Bank","KEY":"Bank","GS":"Bank","BAC":"Bank",
    "ABBV":"Pharma","INCY":"Pharma","ISRG":"Medical","EA":"Gaming",
    "CSGP":"PropTech","COO":"Medical","PYPL":"Fintech",
    "WMG":"Media","HAL":"Energy","RCL":"Travel","OVV":"Energy",
    "MPC":"Energy","LLY":"Pharma","HSY":"ConsStaple","TTWO":"Gaming",
    "MKC":"ConsStaple","HRL":"ConsStaple","SBUX":"Retail","CHD":"ConsStaple",
    "DG":"Retail","TGT":"Retail","NTAP":"Tech","ROST":"Retail",
    "AMZN":"Tech","DECK":"ConsDisc","CMG":"Retail","PANW":"Tech",
    "W":"Retail","ALGN":"Medical","NFLX":"Media",
    "STT":"Bank","CPRT":"ConsDisc","BR":"Fintech","VEEV":"Health",
    "IR":"Indus","QCOM":"Semi","WDC":"Semi","DLTR":"Retail",
    "SPY":"ETF","QQQ":"ETF","TLT":"ETF","NEM":"Gold","BX":"PE",
}

SECTOR_GROUPS = {
    "Semi": "Tech/Semi", "Tech": "Tech/Semi", "EDA": "Tech/Semi",
    "AdTech": "Tech/Semi", "Fintech": "Tech/Semi", "Social": "Tech/Semi",
    "Crypto": "Risk/Speculative", "Aero": "Risk/Speculative",
    "Biotech": "Health", "Medical": "Health", "Pharma": "Health", "Health": "Health",
    "Energy": "Commodity", "Metal": "Commodity", "Gold": "Commodity",
    "Nuclear": "Commodity", "Solar": "Commodity",
    "ConsDisc": "Consumer", "Retail": "Consumer", "Tobacco": "Consumer",
    "ConsStaple": "Consumer", "Gaming": "Consumer", "Media": "Consumer",
    "Bank": "Finance", "PE": "Finance", "Insure": "Finance",
    "Indus": "Cyclical", "Utility": "Cyclical", "REIT": "Cyclical",
    "PropTech": "Cyclical",
    "ETF": "ETF",
}

def sector_features(picks):
    from collections import Counter
    secs = [SECTOR.get(t, "Other") for t in picks]
    groups = [SECTOR_GROUPS.get(s, "Other") for s in secs]
    c = Counter(groups)
    conc = max(c.values()) / len(picks)
    dom  = c.most_common(1)[0][0]
    return conc, dom

# ──────────────────────────────────────────────────────────────────────────────
# 4. 构建特征矩阵
# ──────────────────────────────────────────────────────────────────────────────
print("\n构建特征矩阵...")

all_months = months_sorted
prev_dom   = None
rows = []

for i, m in enumerate(all_months):
    info  = QUANT_GT_PICKS[m]
    ad    = info["analysis_date"]
    picks = [p[0] for p in info["picks"]]
    qgt_r = month_qgt[m]

    # 行业特征
    conc, dom = sector_features(picks)
    streak = (rows[-1]["sector_streak"]+1
              if rows and dom == rows[-1]["dominant_sector_group"] else 1)

    # 宏观
    vix_lv   = px(VIX, ad)
    spy_1m   = pret(SPY, ad, 30)
    spy_3m   = pret(SPY, ad, 90)
    qqq_1m   = pret(QQQ, ad, 30)
    qqq_3m   = pret(QQQ, ad, 90)
    tlt_1m   = pret(TLT, ad, 30)
    hyg_1m   = pret(HYG, ad, 30)
    iwm_1m   = pret(IWM, ad, 30)
    spy_ma   = above_ma(SPY, ad, 200)
    qqq_ma   = above_ma(QQQ, ad, 200)

    # 前期 QGT 动量
    prev_qgt = month_qgt.get(all_months[i-1]) if i > 0 else 0.0
    prev3    = [month_qgt.get(all_months[j]) for j in range(max(0,i-3),i)]
    prev3_sum = sum(x for x in prev3 if x is not None)

    # 前3月最大单月
    prev3_max = max((month_qgt.get(all_months[j], 0) for j in range(max(0,i-3),i)), default=0.0)

    # 月份（seasonality）
    month_num = ad.month

    rows.append({
        "month": m,
        "analysis_date": ad,
        "year": int(m[:4]),
        "qgt_avg": qgt_r,
        "up": int(qgt_r > 0),             # 二元目标
        "big_down": int(qgt_r < -5),       # 大回撤目标
        # features
        "vix_level":    vix_lv,
        "spy_1m":       spy_1m,
        "spy_3m":       spy_3m,
        "qqq_1m":       qqq_1m,
        "qqq_3m":       qqq_3m,
        "tlt_1m":       tlt_1m,
        "hyg_1m":       hyg_1m,
        "iwm_1m":       iwm_1m,
        "spy_above200": spy_ma,
        "qqq_above200": qqq_ma,
        "sector_conc":  conc,
        "sector_streak":streak,
        "dominant_sector_group": dom,
        "prev_month_qgt": prev_qgt if prev_qgt is not None else 0.0,
        "prev3_qgt_sum":  prev3_sum,
        "prev3_qgt_max":  prev3_max,
        "month_num":    month_num,
    })

mf = pd.DataFrame(rows)
print(f"  特征矩阵: {mf.shape[0]} 行 × {mf.shape[1]} 列")
print(f"  up month: {mf['up'].sum()}/{len(mf)}  big_down: {mf['big_down'].sum()}/{len(mf)}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. 特征列 & 哑变量
# ──────────────────────────────────────────────────────────────────────────────
FEAT_COLS = [
    "vix_level","spy_1m","spy_3m","qqq_1m","qqq_3m",
    "tlt_1m","hyg_1m","iwm_1m",
    "spy_above200","qqq_above200",
    "sector_conc","sector_streak",
    "prev_month_qgt","prev3_qgt_sum","prev3_qgt_max",
    "month_num",
]

# 行业 one-hot (top 5 groups + other)
mf = pd.get_dummies(mf, columns=["dominant_sector_group"], prefix="sec")
sec_cols = [c for c in mf.columns if c.startswith("sec_")]
ALL_FEAT = FEAT_COLS + sec_cols

# 填充缺失值（用均值）
mf[ALL_FEAT] = mf[ALL_FEAT].fillna(mf[ALL_FEAT].mean())

# ──────────────────────────────────────────────────────────────────────────────
# 6. 训练 / 测试分割
# ──────────────────────────────────────────────────────────────────────────────
# Test set: 2017 (held out completely)  +  2024-2026 (future validation)
# Train: everything else (2016,2018,2019,2020,2021,2022,2023)

TEST_YEAR_OOS  = 2017                          # 完全样本外
TEST_YEAR_FWD  = [2024, 2025, 2026]            # 前向验证

train_mask = mf["year"].isin([2016,2018,2019,2020,2021,2022,2023])
oos_mask   = mf["year"] == TEST_YEAR_OOS
fwd_mask   = mf["year"].isin(TEST_YEAR_FWD)

df_train = mf[train_mask].copy()
df_oos   = mf[oos_mask].copy()
df_fwd   = mf[fwd_mask].copy()

print(f"\n训练集: {len(df_train)} 月  OOS(2017): {len(df_oos)} 月  前向(2024+): {len(df_fwd)} 月")

X_train = df_train[ALL_FEAT].values
y_train = df_train["up"].values
y_down  = df_train["big_down"].values

# ──────────────────────────────────────────────────────────────────────────────
# 7. 模型训练（Logistic + GBM，Calibrated）
# ──────────────────────────────────────────────────────────────────────────────
print("\n训练模型...")

# Model A: Logistic Regression (interpretable)
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    LogisticRegression(C=0.5, class_weight="balanced",
                                   solver="lbfgs", max_iter=1000, random_state=42))
])
pipe_lr.fit(X_train, y_train)

# Model B: Gradient Boosting (performance)
gbm = GradientBoostingClassifier(
    n_estimators=80, max_depth=2, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=4, random_state=42
)
cal_gbm = CalibratedClassifierCV(gbm, cv=5, method="isotonic")
scaler_gbm = StandardScaler()
X_tr_scaled = scaler_gbm.fit_transform(X_train)
cal_gbm.fit(X_tr_scaled, y_train)

# Ensemble: average of two
def predict_proba(X_raw):
    p_lr  = pipe_lr.predict_proba(X_raw)[:, 1]
    p_gbm = cal_gbm.predict_proba(scaler_gbm.transform(X_raw))[:, 1]
    return (p_lr + p_gbm) / 2

mf["prob_up"] = predict_proba(mf[ALL_FEAT].values)
mf["risk_score"] = (100 * (1 - mf["prob_up"])).round(1)  # 0=极安全 100=极危险

# ──────────────────────────────────────────────────────────────────────────────
# 8. 模型评估
# ──────────────────────────────────────────────────────────────────────────────
def eval_set(df_set, label):
    if df_set.empty: return
    X = df_set[ALL_FEAT].values
    y = df_set["up"].values
    p = predict_proba(X)
    threshold = 0.55
    y_pred = (p >= threshold).astype(int)
    print(f"\n  ─── {label} ({len(df_set)} 月) ───")
    print(f"  ROC-AUC = {roc_auc_score(y, p):.3f}   Brier = {brier_score_loss(y, p):.3f}")
    print(f"  阈值{threshold} 时: 准确率={np.mean(y_pred==y)*100:.0f}%  "
          f"预测上涨={y_pred.sum()}月  实际上涨={y.sum()}月")
    cm = confusion_matrix(y, y_pred)
    print(f"  混淆矩阵:\n{cm}")

    # 按概率分档
    df_set = df_set.copy()
    df_set["prob_up"] = p
    df_set["bucket"] = pd.cut(p, bins=[0,.35,.45,.55,.65,.75,1.01],
                               labels=["<35%","35-45%","45-55%","55-65%","65-75%",">75%"])
    print(f"\n  概率分档实际表现:")
    print(f"  {'概率区间':<10} {'月数':>5} {'实际胜率':>9} {'均收益':>9}")
    for bucket, grp in df_set.groupby("bucket", observed=True):
        win = (grp["qgt_avg"]>0).mean()*100
        avg = grp["qgt_avg"].mean()
        print(f"    {str(bucket):<10} {len(grp):>5} {win:>8.0f}%  {avg:>+8.1f}%")

print("\n" + "="*65)
print("  模型评估")
print("="*65)
eval_set(df_train, "训练集 (2016/18-23)")
eval_set(df_oos,   "样本外 OOS (2017)")
eval_set(df_fwd,   "前向验证 (2024-2026)")

# ──────────────────────────────────────────────────────────────────────────────
# 9. 特征重要性
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  特征重要性 (Logistic 系数 + GBM 重要性)")
print("="*65)

scaler_coef = pipe_lr.named_steps["scaler"]
clf_coef    = pipe_lr.named_steps["clf"]
coef_scaled = clf_coef.coef_[0] * scaler_coef.scale_
feat_importance = sorted(zip(ALL_FEAT, coef_scaled), key=lambda x: abs(x[1]), reverse=True)

FEAT_LABELS = {
    "vix_level":"VIX水平","spy_1m":"SPY近1月","spy_3m":"SPY近3月",
    "qqq_1m":"QQQ近1月","qqq_3m":"QQQ近3月","tlt_1m":"TLT近1月(债)",
    "hyg_1m":"HYG近1月(信用)","iwm_1m":"IWM近1月(小盘)",
    "spy_above200":"SPY>200MA","qqq_above200":"QQQ>200MA",
    "sector_conc":"行业集中度","sector_streak":"连续同行业月数",
    "prev_month_qgt":"上月QGT收益","prev3_qgt_sum":"前3月QGT累计",
    "prev3_qgt_max":"前3月最大单月","month_num":"月份序号",
}

print(f"\n{'特征':<22} {'逻辑系数':>10}  方向")
print("-"*45)
for feat, coef in feat_importance[:14]:
    label = FEAT_LABELS.get(feat, feat[:22])
    direction = "↑看涨" if coef > 0 else "↓看跌"
    bar = "█" * max(1, int(abs(coef)*8))
    print(f"  {label:<20} {coef:>+9.3f}  {direction}  {bar}")

# ──────────────────────────────────────────────────────────────────────────────
# 10. Walk-forward 回测（策略：prob_up>0.55 满仓，否则空仓）
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  Walk-forward 回测（训练2016-2021，测试2022+）")
print("="*65)

TRAIN_END_YEAR = 2021

train_wf = mf[mf["year"] <= TRAIN_END_YEAR].copy()
test_wf  = mf[mf["year"] >  TRAIN_END_YEAR].copy()

# 重新在训练集上fit
X_wf = train_wf[ALL_FEAT].values; y_wf = train_wf["up"].values
p2 = Pipeline([("s", StandardScaler()),
               ("c", LogisticRegression(C=0.5, class_weight="balanced",
                                        max_iter=1000, random_state=42))])
p2.fit(X_wf, y_wf)
gbm2 = GradientBoostingClassifier(n_estimators=80, max_depth=2, learning_rate=0.05,
                                   subsample=0.8, min_samples_leaf=4, random_state=42)
cal2 = CalibratedClassifierCV(gbm2, cv=5, method="isotonic")
sc2  = StandardScaler(); sc2.fit(X_wf)
cal2.fit(sc2.transform(X_wf), y_wf)

def pred2(X_raw):
    p_lr  = p2.predict_proba(X_raw)[:, 1]
    p_gbm = cal2.predict_proba(sc2.transform(X_raw))[:, 1]
    return (p_lr + p_gbm) / 2

test_wf["prob_up_wf"] = pred2(test_wf[ALL_FEAT].values)

THRESHOLD = 0.55
print(f"\n  阈值: prob_up > {THRESHOLD} → 入场  否则 → 空仓(0%)\n")
print(f"{'月份':<12} {'概率':>8} {'信号':>8} {'QGT实际':>10} {'策略收益':>10}  状态")
print("-"*65)

qgt_rets = []; strat_rets = []
for _, r in test_wf.sort_values("analysis_date").iterrows():
    signal = "BUY" if r["prob_up_wf"] >= THRESHOLD else "CASH"
    strat  = r["qgt_avg"] if signal == "BUY" else 0.0
    qgt_rets.append(r["qgt_avg"])
    strat_rets.append(strat)
    flag = ""
    if signal=="BUY" and r["qgt_avg"]<-5: flag="⚠ FP"
    if signal=="CASH" and r["qgt_avg"]>10: flag="✗ FN"
    if signal=="BUY" and r["qgt_avg"]>5:  flag="✓"
    if signal=="CASH" and r["qgt_avg"]<0: flag="✓ TN"
    print(f"  {r['month']:<10} {r['prob_up_wf']:>7.1%} {signal:>8} {r['qgt_avg']:>+9.1f}% {strat:>+9.1f}%  {flag}")

qgt_arr = np.array(qgt_rets); strat_arr = np.array(strat_rets)
qgt_c   = (np.prod(1 + qgt_arr/100) - 1)*100
strat_c = (np.prod(1 + strat_arr/100) - 1)*100
print(f"\n  纯QGT复利:  {qgt_c:>+.1f}%  ({(qgt_arr>0).mean()*100:.0f}% 月胜率)")
print(f"  模型策略:   {strat_c:>+.1f}%  ({(strat_arr>0).mean()*100:.0f}% 月胜率)")
print(f"  空仓月数:   {(strat_arr==0).sum()}/{len(strat_arr)}")

# ──────────────────────────────────────────────────────────────────────────────
# 11. 全历史 + 最新月份风险评分
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("  全历史月度风险评分（概率 + 实际收益）")
print("="*65)

print(f"\n{'月份':<12} {'涨跌概率':>9} {'风险评分':>8} {'QGT实际':>9}  判断")
print("-"*55)
for _, r in mf.sort_values("analysis_date").iterrows():
    prob  = r["prob_up"]
    score = r["risk_score"]
    actual = r["qgt_avg"]
    if prob >= 0.65:   verdict = "强看涨"
    elif prob >= 0.55: verdict = "看涨"
    elif prob >= 0.45: verdict = "中性"
    elif prob >= 0.35: verdict = "看跌"
    else:              verdict = "强看跌"
    flag = " ✓" if (prob>=0.55 and actual>0) or (prob<0.45 and actual<0) else ""
    flag += " ✗" if (prob>=0.55 and actual<-5) or (prob<0.45 and actual>10) else ""
    print(f"  {r['month']:<10} {prob:>8.1%} {score:>8.1f}  {actual:>+8.1f}%  {verdict}{flag}")

# ──────────────────────────────────────────────────────────────────────────────
# 12. 保存（统一口径：OOS 月份用 walk-forward 概率，IS 月份用全量模型概率）
# ──────────────────────────────────────────────────────────────────────────────
# Merge walk-forward predictions back for OOS months (2022+)
# test_wf already has prob_up_wf computed in step 10
wf_prob_map = dict(zip(test_wf["month"], test_wf["prob_up_wf"]))

def unified_prob(row):
    """OOS (year >= 2022): walk-forward only.  IS: full model."""
    if row["year"] >= 2022:
        return wf_prob_map.get(row["month"], row["prob_up"])
    return row["prob_up"]

mf["prob_up"] = mf.apply(unified_prob, axis=1)
mf["risk_score"] = (100 * (1 - mf["prob_up"])).round(1)

out_cols = ["month","analysis_date","year","qgt_avg","up","big_down",
            "prob_up","risk_score"] + FEAT_COLS[:8]
mf[out_cols].to_csv("quant_gt_predictions.csv", index=False, encoding="utf-8-sig")
print(f"\n  详细预测已保存至 quant_gt_predictions.csv")
print(f"  注：OOS月份(2022+)使用walk-forward概率，IS月份使用全量模型概率")

# ──────────────────────────────────────────────────────────────────────────────
# 13. 保存 walk-forward 模型供 API 服务 (quant_gt_serve.py)
# ──────────────────────────────────────────────────────────────────────────────
try:
    import joblib
    joblib.dump({
        "p2": p2,
        "cal2": cal2,
        "sc2": sc2,
        "ALL_FEAT": ALL_FEAT,
        "FEAT_COLS": FEAT_COLS,
        "SECTOR": SECTOR,
        "SECTOR_GROUPS": SECTOR_GROUPS,
        "month_qgt": month_qgt,
    }, "quant_gt_wf_models.pkl")
    print("  Walk-forward 模型已保存至 quant_gt_wf_models.pkl")
except Exception as _e:
    print(f"  [警告] 模型保存失败: {_e}")
