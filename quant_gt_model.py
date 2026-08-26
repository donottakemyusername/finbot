"""quant_gt_model.py
===================
Quant GT 仓位管理模型 —— 波动率目标法 (Volatility Targeting)。

核心思想
--------
经过严格的样本外检验，我们发现：
  1. 宏观因子**无法预测** Quant GT 下个月的涨跌方向（滚动重训 ROC-AUC ≈ 0.46，比抛硬币还差）。
  2. 但 Quant GT 收益的**波动率是可预测的**（波动聚集效应）——高波动月倾向于扎堆出现。

因此本模型不做方向预测，而是做**风险敞口管理**：
  - 平时满仓（甚至可小幅加杠杆）跟随 Quant GT 动量组合；
  - 当 Quant GT 自身近 K 个月的已实现波动率升高时，自动降低仓位，
    把腾出的资金放进防御资产（默认 GLD 黄金）；
  - 波动回落时再加回来。

这个信号只用「已收盘的历史月度收益」计算，无未来函数。

另外叠加一个「超卖反弹覆盖」：上月 QGT 暴跌 < -15% 时，下月不降仓、反而顶格做多。
依据：历史上 QGT 单月跌破 -15% 后，下月 5/5 反弹、均值 +17%（超卖强制平仓反转）。

样本外回测结论（2022+，防御资产=GLD，含 5%/年融资成本，含反弹覆盖）
-------------------------------------------------------
  纯 QGT       : 复利 +463%   最大回撤 -30.7%   beta 1.89   Sharpe 1.02
  本模型(cap=1.5): 复利 +925%   最大回撤 -12.4%   beta 1.29   Sharpe 1.61   (收益、回撤、Sharpe 全面胜)
  全样本(cap=1.5): 复利 +6362%（纯QGT +4838%）  最大回撤 -22.0%   Sharpe 1.41

用法
----
  python quant_gt_model.py                 # 跑完整回测 + 重新生成 quant_gt_predictions.csv
  python quant_gt_model.py --cap 1.0       # 不加杠杆版本
  from quant_gt_model import predict_month  # 给下个月出仓位建议（供 chatbot API 调用）
"""
from __future__ import annotations

import os
import sys
import argparse
import warnings

warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from datetime import date, timedelta

_BASE = os.path.dirname(os.path.abspath(__file__))

# ── 模型超参数 ────────────────────────────────────────────────────────────────
LOOKBACK      = 6       # 计算已实现波动率的回看月数
CAP           = 1.0     # 最大 QGT 仓位（1.0 = 不加杠杆，默认；1.5 = 激进可加杠杆）
BURN_IN       = 30      # 标定波动率目标所用的前置月数（回测中保证因果）
BORROW_ANNUAL = 0.05    # 杠杆部分的年化融资成本（回测中扣除，力求真实）
DEFENSIVE     = "GLD"   # 降仓时买入的防御资产

# 超卖反弹覆盖：当上月 QGT 暴跌超过阈值时，下月不降仓，反而顶格做多。
# 依据：历史上 QGT 单月跌破 -15% 后，下月 5/5 反弹，均值 +17%（超卖强制平仓反转效应）。
# 注意：样本仅 5 次，且本数据集不含 2008 式持续熊市级联，顶格/加杠杆抄底有真实尾部风险。
REBOUND_THRESHOLD = -15.0   # 上月收益低于此值 → 触发反弹覆盖
REBOUND_ENABLED   = True    # 是否启用反弹覆盖

# 过热刹车（momentum crash 预警）：近 K 月涨幅过高时额外压低仓位。
# 依据：近6月涨幅对「下月崩塌」的预测力(corr≈+0.31)强于已实现波动率(+0.12)；VIX 几乎无用。
# 它补的是波动率模型的漏洞——「低波动急涨后突然崩」（如 2018-10）波动率反应不及。
# 默认 ADVISORY（只提示不自动改仓），由你看到信号后自行减仓。
BRAKE_LOOKBACK    = 6       # 计算累计涨幅的回看月数
BRAKE_THRESHOLD   = 25.0    # 近 K 月复利涨幅 ≥ 此值 → 判定过热（跨 25/35/50 阈值结果稳健）
BRAKE_CAP         = 0.5     # 过热时建议的仓位上限
BRAKE_ENABLED     = False   # True=自动把刹车应用到最终仓位；False=仅提示(默认)
VOL_LOW_WEIGHT    = 0.6     # 波动率信号「偏低」的判定线（用于「两个信号都低」预警）

# z 风控综合分（经稳健性验证的最终推荐模型，取代二值刹车）：
#   把已实现波动率 vol 与近6月涨幅 runup6 各自做「因果扩展窗口 z-score」，取平均得连续风险分 z；
#   z>0（风险高于历史均值）时平滑降仓：w = clip(1 - Z_SCALE·max(0,z), Z_FLOOR, 1)，再与波动率目标仓位取小。
# 稳健性验证：跨 k∈[0.2,0.8] 结果平滑、两个子区间(16-21/22+)与样本外均改善；
#   overext/crowd 经隔离检验为冗余(与 runup6 高相关)已剔除，只留 vol+runup6 两个已有信号。
# 效果：全样本 Sharpe 1.40→1.52、回撤 -23%→-18.7%；样本外(2022+) Sharpe 1.60→1.61、收益也反超。
RISK_ZSCORE_ENABLED = True    # 启用 z 综合风控（最终推荐模型）
Z_SCALE             = 0.45    # 风险分缩放 k（跨 0.2~0.8 稳健，取中值，非峰值）
Z_FLOOR             = 0.2     # 降仓下限（最低保留 20% QGT，留反弹参与）
Z_MIN_HISTORY       = 12      # 计算扩展 z 所需的最少历史样本

# ──────────────────────────────────────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────────────────────────────────────

def _month_qgt() -> dict[str, float]:
    """{ 'YYYY-MM': 该月 5 只 picks 的等权平均收益% }"""
    from quant_gt_validate import QUANT_GT_PICKS
    return {m: float(np.mean([r for _, r in info["picks"]]))
            for m, info in QUANT_GT_PICKS.items()}


def _holding_window(month: str):
    """返回该月的 (analysis_date, end_date)，用于计算防御资产同期收益。"""
    from quant_gt_validate import QUANT_GT_PICKS
    info = QUANT_GT_PICKS[month]
    return info["analysis_date"], (info["end_date"] or date.today())


def _dl(ticker: str) -> pd.Series:
    import yfinance as yf
    df = yf.download(ticker, start="2014-01-01", end=date.today().strftime("%Y-%m-%d"),
                     interval="1d", progress=False, auto_adjust=True)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df["Close"].dropna()


def _window_return(series: pd.Series, start, end) -> float:
    a, e = pd.Timestamp(start), pd.Timestamp(end)
    sa, se = series[series.index <= a], series[series.index <= e]
    if sa.empty or se.empty:
        return float("nan")
    return (float(se.iloc[-1]) / float(sa.iloc[-1]) - 1) * 100

# ──────────────────────────────────────────────────────────────────────────────
# 核心信号：波动率目标仓位
# ──────────────────────────────────────────────────────────────────────────────

def target_vol(returns_pct: list[float], k: int = LOOKBACK, burn: int = BURN_IN) -> float:
    """用前 `burn` 个月的数据标定一个固定的波动率目标（因果、只算一次）。"""
    trailing = [np.std(returns_pct[max(0, i - k):i]) for i in range(k, burn)]
    return float(np.median(trailing)) if trailing else float(np.std(returns_pct[:k]))


def qgt_weight(trailing_vol: float, tgt: float, cap: float = CAP) -> float:
    """波动率目标法：仓位 = 目标波动 / 当前波动，封顶 cap，封底 0。"""
    if trailing_vol is None or trailing_vol <= 0 or np.isnan(trailing_vol):
        return min(1.0, cap)
    return float(np.clip(tgt / trailing_vol, 0.0, cap))


def runup(returns_pct: list[float], k: int = BRAKE_LOOKBACK) -> float:
    """近 k 个月的复利涨幅%（因果，只用已收盘的历史）。"""
    if len(returns_pct) < k:
        return float("nan")
    return float((np.prod(1 + np.asarray(returns_pct[-k:]) / 100) - 1) * 100)


# ── z 综合风控（最终模型） ────────────────────────────────────────────────────

def _trailing_vol_series(qgt: np.ndarray, k: int = LOOKBACK) -> np.ndarray:
    """每月 i 的已实现波动率 = std(qgt[i-k:i])（因果，只用之前的月份）。"""
    return np.array([np.std(qgt[max(0, i - k):i]) if i >= 2 else np.nan
                     for i in range(len(qgt))])


def _runup_series(qgt: np.ndarray, k: int = 6) -> np.ndarray:
    """每月 i 的近 k 月复利涨幅（因果）。"""
    return np.array([(np.prod(1 + qgt[i - k:i] / 100) - 1) * 100 if i >= k else np.nan
                     for i in range(len(qgt))])


def _expanding_z(x: np.ndarray) -> np.ndarray:
    """因果扩展窗口 z-score：z[i] 只用 x[:i]（之前的样本）标定均值/标准差。"""
    z = np.full(len(x), np.nan)
    for i in range(len(x)):
        h = x[:i][~np.isnan(x[:i])]
        if len(h) >= Z_MIN_HISTORY and not np.isnan(x[i]) and h.std() > 0:
            z[i] = (x[i] - h.mean()) / h.std()
    return z


def risk_zscore_series(qgt: np.ndarray, k: int = LOOKBACK) -> np.ndarray:
    """综合风险分 = mean(z(波动率), z(近6月涨幅))，逐月因果。"""
    return np.nanmean(np.vstack([_expanding_z(_trailing_vol_series(qgt, k)),
                                 _expanding_z(_runup_series(qgt))]), axis=0)


def _model_weights(qgt: np.ndarray, tgt: float, cap: float = CAP, k: int = LOOKBACK):
    """最终模型逐月仓位：波动率目标 + z综合风控降仓 + 反弹覆盖。返回 (w, z_series)。"""
    z = risk_zscore_series(qgt, k)
    w = np.ones(len(qgt))
    for i in range(len(qgt)):
        # 1) 波动率目标基础仓位（标定期满仓）
        w[i] = min(1.0, cap) if i < max(k, BURN_IN) else qgt_weight(np.std(qgt[i - k:i]), tgt, cap)
        # 2) z综合风控：风险高于历史均值时平滑降仓（与基础仓位取更保守者）
        if RISK_ZSCORE_ENABLED and not np.isnan(z[i]):
            w[i] = min(w[i], float(np.clip(1.0 - Z_SCALE * max(0.0, z[i]), Z_FLOOR, 1.0)))
        # 3) 超卖反弹覆盖：上月暴跌 → 顶格（优先级最高，因果安全）
        if REBOUND_ENABLED and i > 0 and qgt[i - 1] < REBOUND_THRESHOLD:
            w[i] = cap
    return w, z


def _next_month_risk(hist: np.ndarray, k: int = LOOKBACK):
    """为「下一个月」计算 (波动率, 近6月涨幅, z_vol, z_runup, z综合)，只用已收盘历史。"""
    hist = np.asarray(hist, float)
    vol_pred = float(np.std(hist[-k:])) if len(hist) >= 2 else float("nan")
    run_pred = float((np.prod(1 + hist[-6:] / 100) - 1) * 100) if len(hist) >= 6 else float("nan")

    def _z(val, series):
        h = series[~np.isnan(series)]
        if len(h) >= Z_MIN_HISTORY and h.std() > 0 and not np.isnan(val):
            return float((val - h.mean()) / h.std())
        return float("nan")

    zv = _z(vol_pred, _trailing_vol_series(hist, k))
    zr = _z(run_pred, _runup_series(hist))
    z = float(np.nanmean([zv, zr])) if not (np.isnan(zv) and np.isnan(zr)) else float("nan")
    return vol_pred, run_pred, zv, zr, z

# ──────────────────────────────────────────────────────────────────────────────
# 回测
# ──────────────────────────────────────────────────────────────────────────────

def _metrics(ret_pct: np.ndarray, spy_pct: np.ndarray) -> dict:
    r = np.asarray(ret_pct, float) / 100.0
    sp = np.asarray(spy_pct, float) / 100.0
    comp = (np.prod(1 + r) - 1) * 100
    eq = np.cumprod(1 + r)
    mdd = ((eq / np.maximum.accumulate(eq)) - 1).min() * 100
    beta = np.cov(r, sp)[0, 1] / np.var(sp)
    sharpe = (r.mean() * 12) / (r.std() * np.sqrt(12)) if r.std() > 0 else 0.0
    return dict(comp=comp, mdd=mdd, beta=beta, sharpe=sharpe,
                vol=r.std() * np.sqrt(12) * 100, win=(r > 0).mean() * 100)


def _apply_weight(w: np.ndarray, qgt: np.ndarray, defensive: np.ndarray) -> np.ndarray:
    """w<=1: (1-w) 放防御资产；w>1: 杠杆做多 QGT，扣融资成本。"""
    w = np.asarray(w, float)
    base = np.where(w <= 1.0,
                    w * qgt + (1 - w) * defensive,          # 降仓 → 防御资产
                    w * qgt)                                 # 加杠杆
    lev = np.clip(w - 1.0, 0, None)                          # 借入比例
    borrow_cost_pct = lev * (BORROW_ANNUAL / 12) * 100       # 每月融资成本
    return base - borrow_cost_pct


def run_backtest(cap: float = CAP, k: int = LOOKBACK, save_csv: bool = True) -> pd.DataFrame:
    mq = _month_qgt()
    months = sorted(mq)
    qgt = np.array([mq[m] for m in months])

    print("下载防御资产 / 基准数据 (GLD, SPY)...")
    gld_s, spy_s = _dl(DEFENSIVE), _dl("SPY")
    gld, spy = [], []
    for m in months:
        a, e = _holding_window(m)
        gld.append(_window_return(gld_s, a, e))
        spy.append(_window_return(spy_s, a, e))
    gld, spy = np.array(gld), np.array(spy)

    # 因果标定波动目标（仅用前 BURN_IN 个月）
    tgt = target_vol(list(qgt), k=k, burn=BURN_IN)

    # 逐月仓位：波动率目标 + z综合风控 + 反弹覆盖（全部因果，见 _model_weights）
    w, zrisk = _model_weights(qgt, tgt, cap=cap, k=k)

    strat = _apply_weight(w, qgt, gld)

    # 每月 5 只 picks（含各自收益），方便事后按标的/板块分析
    from quant_gt_validate import QUANT_GT_PICKS
    tickers   = [" ".join(t for t, _ in QUANT_GT_PICKS[m]["picks"]) for m in months]
    picks_ret = ["|".join(f"{t}:{r:+.1f}" for t, r in QUANT_GT_PICKS[m]["picks"]) for m in months]

    df = pd.DataFrame(dict(
        month=months, year=[int(m[:4]) for m in months],
        tickers=tickers, picks_detail=picks_ret,
        qgt_return=qgt.round(2), qgt_weight=w.round(3),
        defensive_weight=(1 - np.clip(w, 0, 1)).round(3),
        risk_z=np.round(zrisk, 2),
        gld_return=gld.round(2), spy_return=spy.round(2),
        strat_return=strat.round(2),
    ))

    # ── 打印报告 ──────────────────────────────────────────────────────────────
    def report(label, mask):
        sub = df[mask]
        base = _metrics(sub.qgt_return.values, sub.spy_return.values)
        m = _metrics(sub.strat_return.values, sub.spy_return.values)
        print(f"\n=== {label}  (n={len(sub)})  vol目标={tgt:.1f}  回看={k}月  cap={cap} ===")
        print(f"  {'策略':<20}{'复利':>10}{'最大回撤':>10}{'Beta':>8}{'年化波动':>10}{'Sharpe':>9}{'月胜率':>8}")
        print(f"  {'纯 QGT (基准)':<18}{base['comp']:>+9.0f}%{base['mdd']:>9.1f}%{base['beta']:>+8.2f}"
              f"{base['vol']:>9.0f}%{base['sharpe']:>9.2f}{base['win']:>7.0f}%")
        beat_r = "✓" if m['comp'] > base['comp'] else " "
        beat_s = "✓" if m['sharpe'] > base['sharpe'] else " "
        print(f"  {'波动率目标模型':<18}{m['comp']:>+9.0f}%{beat_r}{m['mdd']:>8.1f}%{m['beta']:>+8.2f}"
              f"{m['vol']:>9.0f}%{m['sharpe']:>8.2f}{beat_s}{m['win']:>7.0f}%")

    print("\n" + "=" * 78)
    print("  Quant GT 波动率目标模型 —— 样本外回测")
    print("=" * 78)
    report("全样本 2016+ (含标定期)", df.index >= 0)
    report("样本外 2022+ (最诚实口径)", (df.year >= 2022).values)
    report("近期 2024+", (df.year >= 2024).values)

    if save_csv:
        path = os.path.join(_BASE, "quant_gt_predictions.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n  逐月明细已保存至 {os.path.basename(path)}")

    return df


# ──────────────────────────────────────────────────────────────────────────────
# 服务接口（供 chatbot.py 调用）—— 为下一个月出仓位建议
# ──────────────────────────────────────────────────────────────────────────────

def predict_month(month: str, tickers: list[str] | None = None, cap: float = CAP) -> dict:
    """
    为指定月份给出仓位建议。

    注意：本模型的仓位只取决于 Quant GT 组合近 K 个月的已实现波动率，
    与当月具体选了哪 5 只票无关（方向不可预测，已验证）。tickers 仅作记录。
    """
    mq = _month_qgt()
    months = sorted(mq)
    hist = [mq[m] for m in months if m < month]          # 严格取该月之前的历史
    if len(hist) < LOOKBACK:
        return {"error": f"历史月份不足 {LOOKBACK} 个，无法计算波动率目标"}

    hist = np.asarray(hist, float)
    tgt = target_vol([mq[m] for m in months], k=LOOKBACK, burn=BURN_IN)

    # ── 信号 1：波动率目标基础仓位 ──（波动越高 → 仓位越低）
    trailing_vol = float(np.std(hist[-LOOKBACK:]))
    vol_w = qgt_weight(trailing_vol, tgt, cap)
    all_tv = [np.std(hist[max(0, i - LOOKBACK):i]) for i in range(LOOKBACK, len(hist))]
    vol_pct = float((np.array(all_tv) < trailing_vol).mean() * 100) if all_tv else float("nan")

    # ── 信号 2：z 综合风控 ──（vol 与 近6月涨幅 的因果 z 分数平均；z>0 平滑降仓）
    vol_pred, run6, zv, zr, z = _next_month_risk(hist, LOOKBACK)
    if RISK_ZSCORE_ENABLED and not np.isnan(z):
        z_cap = float(np.clip(1.0 - Z_SCALE * max(0.0, z), Z_FLOOR, 1.0))
        risk_w = min(vol_w, z_cap)
    else:
        z_cap = 1.0
        risk_w = vol_w

    # ── 信号 3：超卖反弹覆盖 ──（上月暴跌 → 顶格，优先级最高）
    rebound = bool(REBOUND_ENABLED and hist[-1] < REBOUND_THRESHOLD)

    # ── 最终仓位 ──
    w = cap if rebound else risk_w
    w_if_no_rebound = risk_w          # 反弹覆盖前，风控模型本来建议的仓位（供你手动对照）

    def _fmt(x):
        if x >= 1.0:
            return f"加杠杆 {x*100:.0f}% QGT（借入 {(x-1)*100:.0f}%）" if x > 1.0 else "满仓 100% QGT"
        return f"{x*100:.0f}% QGT + {(1-x)*100:.0f}% {DEFENSIVE}（防御，赚 {DEFENSIVE} 收益、非0）"

    defensive_w = 0.0 if w >= 1.0 else round(1 - w, 3)
    risk_high = (not np.isnan(z)) and z > 0.5            # 风险分明显偏高
    conflict = rebound and risk_high                    # 反弹覆盖 vs 风控红灯 冲突

    if conflict:
        note = (f"⚠️ 冲突：上月暴跌触发反弹覆盖→顶格，但 z 风控分 {z:.2f} 偏高（风控本建议仅 {risk_w*100:.0f}%）。"
                f"历史反弹 5/5 但样本小；若本次是板块基本面崩塌，建议手动采纳风控档位 ~{risk_w*100:.0f}%。")
    elif rebound:
        note = "上月暴跌触发超卖反弹覆盖：顶格抢反弹（历史 5/5 但样本小，有尾部风险）。"
    elif risk_high:
        note = f"z 风控分 {z:.2f} 偏高（风险高于历史均值）→ 已降仓至 {w*100:.0f}%，多出的钱进 {DEFENSIVE} 防御。"
    else:
        note = f"z 风控分 {'%.2f' % z if not np.isnan(z) else 'NA'} 正常；仓位由波动率决定。方向不可预测，故不做涨跌判断。"

    return {
        "month": month,
        "tickers": tickers or [],
        "model": "volatility_targeting + z_risk (final)",
        # —— 最终建议 ——
        "qgt_weight": round(w, 3),
        "defensive_asset": DEFENSIVE,
        "defensive_weight": defensive_w,
        "position": _fmt(w),
        "risk_model_weight": round(risk_w, 3),          # 风控模型建议（反弹覆盖前）
        # —— 各信号透明展示 ——
        "signals": {
            "volatility": {
                "suggested_weight": round(vol_w, 3),
                "trailing_vol": round(trailing_vol, 2),
                "vol_target": round(tgt, 2),
                "vol_percentile": (round(vol_pct, 0) if not np.isnan(vol_pct) else None),
            },
            "risk_zscore": {
                "z_composite": (round(z, 2) if not np.isnan(z) else None),
                "z_vol": (round(zv, 2) if not np.isnan(zv) else None),
                "z_runup6": (round(zr, 2) if not np.isnan(zr) else None),
                "runup_6m_pct": (round(run6, 1) if not np.isnan(run6) else None),
                "suggested_cap": round(z_cap, 3),        # z 建议的仓位上限
                "level": "偏高(降仓)" if risk_high else "正常",
            },
            "rebound_override": {
                "prev_month_return": round(float(hist[-1]), 2),
                "threshold_pct": REBOUND_THRESHOLD,
                "triggered": rebound,
            },
        },
        "rebound_vs_risk_conflict": conflict,   # 反弹覆盖与风控红灯冲突 → 建议手动采纳风控档位
        "note": note,
    }


def get_history(n: int = 12) -> list[dict]:
    """读取最近 n 个月的回测明细（供 API /quant-gt/history）。"""
    path = os.path.join(_BASE, "quant_gt_predictions.csv")
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path).sort_values("month").tail(n)
    rows = []
    for _, r in df.iterrows():
        w = float(r["qgt_weight"])
        rows.append({
            "month": r["month"],
            "qgt_weight": round(w, 3),
            "defensive_weight": round(float(r["defensive_weight"]), 3),
            "qgt_return": round(float(r["qgt_return"]), 2),
            "strat_return": round(float(r["strat_return"]), 2),
            "position": ("满仓/杠杆" if w >= 1.0 else f"{w*100:.0f}% QGT + 防御"),
        })
    return rows


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Quant GT 波动率目标模型")
    ap.add_argument("--cap", type=float, default=CAP, help="最大 QGT 仓位 (1.0=不加杠杆, 1.5=激进)")
    ap.add_argument("--lookback", type=int, default=LOOKBACK, help="波动率回看月数")
    args = ap.parse_args()
    run_backtest(cap=args.cap, k=args.lookback)
