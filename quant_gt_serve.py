"""quant_gt_serve.py
Standalone prediction service for Quant GT monthly model.
Loaded by chatbot.py FastAPI endpoints — no dependency on quant_gt_predictor.py.
"""
from __future__ import annotations
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from collections import Counter
from datetime import date, timedelta

_BASE = os.path.dirname(os.path.abspath(__file__))

# ── Lazy singletons ───────────────────────────────────────────────────────────
_models = None
_macro  = None


def _get_models():
    global _models
    if _models is None:
        path = os.path.join(_BASE, "quant_gt_wf_models.pkl")
        _models = joblib.load(path)
    return _models


def _get_macro(force_refresh=False):
    global _macro
    if _macro is None or force_refresh:
        today = date.today().strftime("%Y-%m-%d")
        def dl(ticker):
            df = yf.download(ticker, start="2015-01-01", end=today,
                             interval="1d", progress=False, auto_adjust=True)
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            return df["Close"].dropna()
        _macro = {k: dl(v) for k, v in
                  [("SPY","SPY"),("QQQ","QQQ"),("VIX","^VIX"),
                   ("TLT","TLT"),("HYG","HYG"),("IWM","IWM")]}
    return _macro


# ── Feature helpers ───────────────────────────────────────────────────────────

def _px(s: pd.Series, d) -> float:
    mask = s.index <= pd.Timestamp(d)
    return float(s[mask].iloc[-1]) if mask.any() else float("nan")


def _pret(s: pd.Series, d, days: int) -> float:
    p1 = _px(s, d)
    p0 = _px(s, pd.Timestamp(d) - timedelta(days=days))
    if any(np.isnan(x) for x in [p1, p0]) or p0 == 0:
        return float("nan")
    return (p1 - p0) / p0 * 100


def _above_ma(s: pd.Series, d, window: int = 200) -> float:
    mask = s.index <= pd.Timestamp(d)
    sub  = s[mask]
    if len(sub) < window:
        return float("nan")
    return float(sub.iloc[-1] > sub.iloc[-window:].mean())


def _sector_features(tickers: list[str], SECTOR: dict, SECTOR_GROUPS: dict):
    secs   = [SECTOR.get(t, "Other") for t in tickers]
    groups = [SECTOR_GROUPS.get(s, "Other") for s in secs]
    c      = Counter(groups)
    conc   = max(c.values()) / len(tickers) if tickers else 0.5
    dom    = c.most_common(1)[0][0] if tickers else "Other"
    return conc, dom


def _first_trading_day(month_str: str) -> date:
    yr, mo = map(int, month_str.split("-"))
    d = date(yr, mo, 1)
    today = date.today()
    if d > today:
        return today          # future month → use today's macro data
    for offset in range(5):
        candidate = d + timedelta(days=offset)
        if candidate.weekday() < 5:
            return candidate
    return d


# ── Public API ────────────────────────────────────────────────────────────────

def predict_month(month_str: str, tickers: list[str]) -> dict:
    """
    Predict up-probability for (month, tickers) using saved walk-forward model.

    Returns dict with: prob_up, prob_up_pct, risk_score, position,
                       position_weight, dominant_sector, features.
    """
    mdl = _get_models()
    p2_m, cal2_m, sc2_m = mdl["p2"], mdl["cal2"], mdl["sc2"]
    ALL_FEAT     = mdl["ALL_FEAT"]
    SECTOR       = mdl["SECTOR"]
    SECTOR_GROUPS = mdl["SECTOR_GROUPS"]
    month_qgt    = mdl["month_qgt"]          # {month: avg_return}

    macro = _get_macro()
    SPY, QQQ, VIX = macro["SPY"], macro["QQQ"], macro["VIX"]
    TLT, HYG, IWM = macro["TLT"], macro["HYG"], macro["IWM"]

    ad = _first_trading_day(month_str)

    # Macro
    vix_lv  = _px(VIX, ad)
    spy_1m  = _pret(SPY, ad, 30)
    spy_3m  = _pret(SPY, ad, 90)
    qqq_1m  = _pret(QQQ, ad, 30)
    qqq_3m  = _pret(QQQ, ad, 90)
    tlt_1m  = _pret(TLT, ad, 30)
    hyg_1m  = _pret(HYG, ad, 30)
    iwm_1m  = _pret(IWM, ad, 30)
    spy_ma  = _above_ma(SPY, ad, 200)
    qqq_ma  = _above_ma(QQQ, ad, 200)

    # Sector
    conc, dom = _sector_features(tickers, SECTOR, SECTOR_GROUPS)

    # QGT momentum (from historical picks only — avoids look-ahead)
    all_months_hist = sorted(month_qgt.keys())
    prev_months = [m for m in all_months_hist if m < month_str]

    prev_qgt  = float(month_qgt.get(prev_months[-1], 0.0)) if prev_months else 0.0
    prev3     = [float(month_qgt.get(m, 0.0)) for m in prev_months[-3:]]
    prev3_sum = float(sum(prev3))
    prev3_max = float(max(prev3)) if prev3 else 0.0

    # Streak: consecutive months with the same dominant sector group
    streak = 1
    for pm in reversed(prev_months):
        pm_picks_tickers = list(month_qgt.keys())   # sector from current tickers
        # We approximate streak using the same sector for consecutive months
        # (exact re-computation would need tickers per historical month)
        break
    # Use sector_streak = 1 for new input (safe conservative default)

    month_num = ad.month

    row: dict = {
        "vix_level":      vix_lv,
        "spy_1m":         spy_1m,
        "spy_3m":         spy_3m,
        "qqq_1m":         qqq_1m,
        "qqq_3m":         qqq_3m,
        "tlt_1m":         tlt_1m,
        "hyg_1m":         hyg_1m,
        "iwm_1m":         iwm_1m,
        "spy_above200":   spy_ma,
        "qqq_above200":   qqq_ma,
        "sector_conc":    conc,
        "sector_streak":  streak,
        "prev_month_qgt": prev_qgt,
        "prev3_qgt_sum":  prev3_sum,
        "prev3_qgt_max":  prev3_max,
        "month_num":      month_num,
    }

    # One-hot sector columns that the model was trained with
    sec_cols = [c for c in ALL_FEAT if c.startswith("sec_")]
    for sc_col in sec_cols:
        group_name = sc_col[len("sec_"):]
        row[sc_col] = 1.0 if dom == group_name else 0.0

    row_df = pd.DataFrame([row])
    for col in ALL_FEAT:
        if col not in row_df.columns:
            row_df[col] = 0.0

    X = row_df[ALL_FEAT].fillna(0.0).values

    p_lr  = float(p2_m.predict_proba(X)[0, 1])
    p_gbm = float(cal2_m.predict_proba(sc2_m.transform(X))[0, 1])
    prob  = (p_lr + p_gbm) / 2.0
    risk  = round((1.0 - prob) * 100, 1)

    if prob >= 0.55:
        position, weight = "满仓 100%", 1.0
    elif prob >= 0.40:
        position, weight = "半仓 50%", 0.5
    else:
        position, weight = "空仓 0%", 0.0

    def safe(v):
        return None if (v is None or np.isnan(v)) else round(float(v), 2)

    return {
        "month":            month_str,
        "analysis_date":    ad.strftime("%Y-%m-%d"),
        "tickers":          tickers,
        "dominant_sector":  dom,
        "prob_up":          round(prob, 4),
        "prob_up_pct":      round(prob * 100, 1),
        "risk_score":       risk,
        "position":         position,
        "position_weight":  weight,
        "features": {
            "vix":            safe(vix_lv),
            "spy_1m":         safe(spy_1m),
            "spy_3m":         safe(spy_3m),
            "qqq_1m":         safe(qqq_1m),
            "tlt_1m":         safe(tlt_1m),
            "hyg_1m":         safe(hyg_1m),
            "iwm_1m":         safe(iwm_1m),
            "spy_above_200ma": bool(spy_ma > 0.5) if not np.isnan(spy_ma) else None,
            "prev_month_qgt": round(prev_qgt, 2),
            "prev3_sum":      round(prev3_sum, 2),
        },
    }


def get_history(n: int = 12) -> list[dict]:
    """Return the last n months from quant_gt_predictions.csv."""
    path = os.path.join(_BASE, "quant_gt_predictions.csv")
    df   = pd.read_csv(path)
    df   = df.sort_values("month").tail(n)
    rows = []
    for _, r in df.iterrows():
        prob = float(r["prob_up"])
        if prob >= 0.55:
            pos, pw = "满仓 100%", 1.0
        elif prob >= 0.40:
            pos, pw = "半仓 50%", 0.5
        else:
            pos, pw = "空仓 0%", 0.0
        rows.append({
            "month":           r["month"],
            "prob_up_pct":     round(prob * 100, 1),
            "risk_score":      float(r["risk_score"]),
            "position":        pos,
            "position_weight": pw,
            "qgt_avg":         round(float(r["qgt_avg"]), 2),
            "up":              int(r["up"]),
        })
    return rows
