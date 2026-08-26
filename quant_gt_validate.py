"""quant_gt_validate.py
======================
用三位一体（硬指标 + 时空状态）验证 Quant GT 每月初的选股信号。

对每个月第一天可获得的历史信息运行三位一体分析，判断哪些股票当时应该买入，
并与 Quant GT 的实际持仓收益对比。

用法:
    python quant_gt_validate.py                    # 全部月份，无Claude（快速）
    python quant_gt_validate.py --claude           # 启用Claude软分析（更慢更准）
    python quant_gt_validate.py --year 2025        # 只跑2025年
    python quant_gt_validate.py --month 2025-04    # 只跑某个月
    python quant_gt_validate.py --output results.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time
from datetime import date, datetime

# Fix Windows GBK console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from backtest_trinity import trinity_analysis_as_of, fetch_forward_returns

# ─────────────────────────────────────────────────────────────────────────────
# Quant GT 月度选股数据（来源：截图）
#
# 格式：
#   key   = "YYYY-MM"
#   value = dict(
#       analysis_date = 当月第一个交易日（用于三位一体分析）
#       end_date      = 持仓结束日（用于精确收益计算）
#       picks = list of (ticker, actual_return_pct)
#                actual_return_pct 来自截图中的 RETURN 列
#   )
# ─────────────────────────────────────────────────────────────────────────────

QUANT_GT_PICKS: dict[str, dict] = {
    # ═══════════════════ 2016 (partial) ═════════
    "2016-11": dict(
        analysis_date=date(2016, 11, 1),
        end_date=date(2016, 12, 1),
        picks=[
            ("MU",   +14.53),
            ("COHR", +8.05),
            ("NVDA", +28.49),
            ("LVS",  +6.98),
            ("ADSK", +0.51),
        ],
    ),
    "2016-12": dict(
        analysis_date=date(2016, 12, 1),
        end_date=date(2017, 1, 3),
        picks=[
            ("COHR", -0.83),
            ("NVDA", +13.48),
            ("SATS", -0.10),
            ("CFG",  +7.59),
            ("NFLX", +6.38),
        ],
    ),
    # ═══════════════════ 2017 ═══════════════════
    "2017-01": dict(
        analysis_date=date(2017, 1, 3),
        end_date=date(2017, 2, 1),
        picks=[
            ("NVDA", +5.75),
            ("KEY",  -1.73),
            ("CFG",  +0.27),
            ("COHR", +1.64),
            ("GS",   -5.02),
        ],
    ),
    "2017-02": dict(
        analysis_date=date(2017, 2, 1),
        end_date=date(2017, 3, 1),
        picks=[
            ("NVDA", -6.16),
            ("AMD",  +38.35),
            ("CFG",  +5.15),
            ("KEY",  +5.00),
            ("BAC",  +10.45),
        ],
    ),
    "2017-03": dict(
        analysis_date=date(2017, 3, 1),
        end_date=date(2017, 4, 3),
        picks=[
            ("AMD",  -3.18),
            ("NVDA", +5.02),
            ("BAC",  -6.78),
            ("COHR", -0.14),
            ("CFG",  -9.84),
        ],
    ),
    "2017-04": dict(
        analysis_date=date(2017, 4, 3),
        end_date=date(2017, 5, 1),
        picks=[
            ("AMD",  -8.01),
            ("INCY", -7.57),
            ("ANET", +5.95),
            ("MU",   -3.80),
            ("TSLA", +9.72),
        ],
    ),
    "2017-05": dict(
        analysis_date=date(2017, 5, 1),
        end_date=date(2017, 6, 1),
        picks=[
            ("ANET", +5.73),
            ("AMD",  -16.23),
            ("VRTX", +4.22),
            ("TSLA", +9.24),
            ("MU",   +10.41),
        ],
    ),
    "2017-06": dict(
        analysis_date=date(2017, 6, 1),
        end_date=date(2017, 7, 3),
        picks=[
            ("VRTX", +4.43),
            ("ANET", +2.39),
            ("TSLA", +7.63),
            ("LRCX", -8.11),
            ("ISRG", +2.82),
        ],
    ),
    "2017-07": dict(
        analysis_date=date(2017, 7, 3),
        end_date=date(2017, 8, 1),
        picks=[
            ("NVDA", +11.57),
            ("VRTX", +18.18),
            ("TSLA", -12.76),
            ("ANET", -0.42),
            ("EA",   +10.38),
        ],
    ),
    "2017-08": dict(
        analysis_date=date(2017, 8, 1),
        end_date=date(2017, 9, 1),
        picks=[
            ("NVDA", +4.94),
            ("VRTX", +5.05),
            ("PYPL", +5.05),
            ("CSGP", +3.97),
            ("COO",  -0.05),
        ],
    ),
    "2017-09": dict(
        analysis_date=date(2017, 9, 1),
        end_date=date(2017, 10, 2),
        picks=[
            ("NVDA", +6.35),
            ("VRTX", -5.93),
            ("PYPL", +4.07),
            ("BA",   +6.25),
            ("CSGP", -6.36),
        ],
    ),
    "2017-10": dict(
        analysis_date=date(2017, 10, 2),
        end_date=date(2017, 11, 1),
        picks=[
            ("REMX", +5.93),
            ("NVDA", +15.71),
            ("BA",   +1.43),
            ("VRTX", -2.57),
            ("FCX",  +3.00),
        ],
    ),
    "2017-11": dict(
        analysis_date=date(2017, 11, 1),
        end_date=date(2017, 12, 1),
        picks=[
            ("REMX", +3.03),
            ("MU",   -6.83),
            ("ABBV", +7.06),
            ("BA",   +7.44),
            ("NVDA", -4.78),
        ],
    ),
    "2017-12": dict(
        analysis_date=date(2017, 12, 1),
        end_date=date(2018, 1, 2),
        picks=[
            ("MU",   -0.46),
            ("REMX", +7.22),
            ("IBKR", +3.58),
            ("LRCX", -2.53),
            ("ANET", +1.31),
        ],
    ),
    # ═══════════════════ 2018 ═══════════════════
    "2018-01": dict(
        analysis_date=date(2018, 1, 2),
        end_date=date(2018, 2, 1),
        picks=[
            ("IBKR", +8.20),
            ("MU",   +3.61),
            ("ANET", +16.03),
            ("DLTR", +5.19),
            ("CPRT", +1.10),
        ],
    ),
    "2018-02": dict(
        analysis_date=date(2018, 2, 1),
        end_date=date(2018, 3, 1),
        picks=[
            ("NTAP", -1.21),
            ("ANET", -0.94),
            ("IBKR", +9.08),
            ("ROST", -4.63),
            ("KR",   -9.67),
        ],
    ),
    "2018-03": dict(
        analysis_date=date(2018, 3, 1),
        end_date=date(2018, 4, 2),
        picks=[
            ("NFLX", -0.31),
            ("KR",   -12.35),
            ("AMZN", -6.34),
            ("DECK", -4.89),
            ("NTAP", -0.49),
        ],
    ),
    "2018-04": dict(
        analysis_date=date(2018, 4, 2),
        end_date=date(2018, 5, 1),
        picks=[
            ("NFLX", +6.34),
            ("AMZN", +10.27),
            ("DECK", +3.34),
            ("NOW",  +0.55),
            ("BA",   +2.24),
        ],
    ),
    "2018-05": dict(
        analysis_date=date(2018, 5, 1),
        end_date=date(2018, 6, 1),
        picks=[
            ("NFLX", +14.01),
            ("DXCM", +21.31),
            ("FTNT", +11.91),
            ("PANW", +9.51),
            ("NOW",  +8.29),
        ],
    ),
    "2018-06": dict(
        analysis_date=date(2018, 6, 1),
        end_date=date(2018, 7, 2),
        picks=[
            ("DXCM", +9.71),
            ("NFLX", +8.93),
            ("TTD",  +7.56),
            ("PANW", -2.43),
            ("CMG",  -0.81),
        ],
    ),
    "2018-07": dict(
        analysis_date=date(2018, 7, 2),
        end_date=date(2018, 8, 1),
        picks=[
            ("TTD",  -9.08),
            ("DXCM", -2.65),
            ("CMG",  +1.64),
            ("NFLX", -12.87),
            ("ALGN", +6.49),
        ],
    ),
    "2018-08": dict(
        analysis_date=date(2018, 8, 1),
        end_date=date(2018, 9, 4),
        picks=[
            ("TTD",  +68.73),
            ("W",    +23.14),
            ("AMD",  +39.69),
            ("DXCM", +53.12),
            ("ALGN", +8.41),
        ],
    ),
    "2018-09": dict(
        analysis_date=date(2018, 9, 4),
        end_date=date(2018, 10, 1),
        picks=[
            ("TTD",  +6.77),
            ("AMD",  +19.79),
            ("DXCM", -2.22),
            ("W",    +10.57),
            ("ALGN", +1.74),
        ],
    ),
    "2018-10": dict(
        analysis_date=date(2018, 10, 1),
        end_date=date(2018, 11, 1),
        picks=[
            ("AMD",  -40.01),
            ("TTD",  -17.62),
            ("DXCM", -5.22),
            ("W",    -35.41),
            ("FTNT", -11.61),
        ],
    ),
    "2018-11": dict(
        analysis_date=date(2018, 11, 1),
        end_date=date(2018, 12, 3),
        picks=[
            ("TTD",  +18.59),
            ("DXCM", +0.06),
            ("FTNT", -7.24),
            ("W",    +14.99),
            ("LLY",  +9.10),
        ],
    ),
    "2018-12": dict(
        analysis_date=date(2018, 12, 3),
        end_date=date(2019, 1, 2),
        picks=[
            ("SBUX", -5.48),
            ("MKC",  -8.55),
            ("LLY",  -3.25),
            ("DXCM", -12.45),
            ("HRL",  -6.34),
        ],
    ),
    # ═══════════════════ 2020 ═══════════════════
    "2020-01": dict(
        analysis_date=date(2020, 1, 2),
        end_date=date(2020, 2, 3),
        picks=[
            ("TSLA", +58.69),
            ("ALGN", -8.43),
            ("STT",  -4.22),
            ("DXCM", +10.29),
            ("AMD",  -0.98),
        ],
    ),
    "2020-02": dict(
        analysis_date=date(2020, 2, 3),
        end_date=date(2020, 3, 2),
        picks=[
            ("TSLA", +5.59),
            ("AMD",  +2.20),
            ("ALGN", -14.99),
            ("Z",    +19.93),
            ("ZG",   +20.12),
        ],
    ),
    "2020-03": dict(
        analysis_date=date(2020, 3, 2),
        end_date=date(2020, 4, 1),
        picks=[
            ("TSLA", -29.14),
            ("ZG",   -42.67),
            ("Z",    -39.58),
            ("AMD",  -6.83),
            ("DXCM", -8.13),
        ],
    ),
    "2020-04": dict(
        analysis_date=date(2020, 4, 1),
        end_date=date(2020, 5, 1),
        picks=[
            ("TSLA", +49.79),
            ("DXCM", +29.54),
            ("AMD",  +15.60),
            ("NFLX", +10.37),
            ("QQQ",  +16.09),
        ],
    ),
    "2020-05": dict(
        analysis_date=date(2020, 5, 1),
        end_date=date(2020, 6, 1),
        picks=[
            ("ZM",   +34.69),
            ("TSLA", +13.65),
            ("DXCM", +15.56),
            ("NEM",  +0.51),
            ("TLT",  -3.22),
        ],
    ),
    "2020-06": dict(
        analysis_date=date(2020, 6, 1),
        end_date=date(2020, 7, 1),
        picks=[
            ("ZM",   +37.21),
            ("W",    +14.08),
            ("DXCM", +6.91),
            ("NEM",  +4.78),
            ("TWLO", +11.41),
        ],
    ),
    "2020-07": dict(
        analysis_date=date(2020, 7, 1),
        end_date=date(2020, 8, 3),
        picks=[
            ("W",    +38.67),
            ("ZM",   +2.18),
            ("TWLO", +29.36),
            ("DXCM", +7.75),
            ("CVNA", +31.08),
        ],
    ),
    "2020-08": dict(
        analysis_date=date(2020, 8, 3),
        end_date=date(2020, 9, 1),
        picks=[
            ("W",    +9.27),
            ("TWLO", -2.77),
            ("CVNA", +37.23),
            ("TSLA", +73.25),
            ("ZM",   +71.13),
        ],
    ),
    "2020-09": dict(
        analysis_date=date(2020, 9, 1),
        end_date=date(2020, 10, 1),
        picks=[
            ("W",    -0.50),
            ("TSLA", -12.22),
            ("CVNA", +5.66),
            ("TWLO", -8.90),
            ("TTD",  +8.90),
        ],
    ),
    "2020-10": dict(
        analysis_date=date(2020, 10, 1),
        end_date=date(2020, 11, 2),
        picks=[
            ("TSLA", -10.61),
            ("CVNA", -17.64),
            ("ZM",   -3.08),
            ("W",    -15.92),
            ("ZG",   -13.68),
        ],
    ),
    "2020-11": dict(
        analysis_date=date(2020, 11, 2),
        end_date=date(2020, 12, 1),
        picks=[
            ("ZM",   -5.96),
            ("TSLA", +51.68),
            ("PINS", +20.19),
            ("ZG",   +23.91),
            ("Z",    +22.51),
        ],
    ),
    "2020-12": dict(
        analysis_date=date(2020, 12, 1),
        end_date=date(2021, 1, 4),
        picks=[
            ("PINS", -6.92),
            ("SNAP", +12.34),
            ("ZM",   -21.70),
            ("TTD",  -10.85),
            ("ROKU", +17.03),
        ],
    ),
    # ═══════════════════ 2021 ═══════════════════
    "2021-01": dict(
        analysis_date=date(2021, 1, 4),
        end_date=date(2021, 2, 1),
        picks=[
            ("SNAP", +7.06),
            ("PINS", +5.47),
            ("MSTR", +49.00),
            ("TTD",  -2.02),
            ("ROKU", +16.28),
        ],
    ),
    "2021-02": dict(
        analysis_date=date(2021, 2, 1),
        end_date=date(2021, 3, 1),
        picks=[
            ("ARWR", +3.99),
            ("QQQ",  +0.36),
            ("SPY",  +3.18),
            ("GH",   -3.75),
            ("PODD", -3.13),
        ],
    ),
    "2021-03": dict(
        analysis_date=date(2021, 3, 1),
        end_date=date(2021, 4, 1),
        picks=[
            ("ARWR", -17.14),
            ("GH",   +3.02),
            ("SWKS", +2.71),
            ("CGNX", -0.71),
            ("WAT",  +3.16),
        ],
    ),
    "2021-04": dict(
        analysis_date=date(2021, 4, 1),
        end_date=date(2021, 5, 3),
        picks=[
            ("MSTR", -2.11),
            ("SWKS", -2.15),
            ("J",    +3.84),
            ("WAT",  +6.22),
            ("GH",   +3.32),
        ],
    ),
    "2021-05": dict(
        analysis_date=date(2021, 5, 3),
        end_date=date(2021, 6, 1),
        picks=[
            ("J",    +6.30),
            ("SWKS", -6.03),
            ("CARR", +5.64),
            ("WAT",  +6.87),
            ("QQQ",  -1.16),
        ],
    ),
    "2021-06": dict(
        analysis_date=date(2021, 6, 1),
        end_date=date(2021, 7, 1),
        picks=[
            ("J",    -6.28),
            ("CARR", +4.31),
            ("WAT",  +7.11),
            ("NUE",  -7.62),
            ("SPY",  +1.49),
        ],
    ),
    "2021-07": dict(
        analysis_date=date(2021, 7, 1),
        end_date=date(2021, 8, 2),
        picks=[
            ("NUE",  +7.94),
            ("J",    +1.06),
            ("CARR", +13.74),
            ("WAT",  +12.97),
            ("QQQ",  +3.45),
        ],
    ),
    "2021-08": dict(
        analysis_date=date(2021, 8, 2),
        end_date=date(2021, 9, 1),
        picks=[
            ("MRNA", +6.34),
            ("NET",  +0.61),
            ("NVDA", +14.16),
            ("FTNT", +15.79),
            ("BX",   +8.37),
        ],
    ),
    "2021-09": dict(
        analysis_date=date(2021, 9, 1),
        end_date=date(2021, 10, 1),
        picks=[
            ("MRNA", -6.40),
            ("NET",  -5.82),
            ("DDOG", +2.85),
            ("RMD",  -8.62),
            ("FTNT", -8.31),
        ],
    ),
    "2021-10": dict(
        analysis_date=date(2021, 10, 1),
        end_date=date(2021, 11, 1),
        picks=[
            ("MRNA", -6.51),
            ("DDOG", +17.91),
            ("NET",  +72.33),
            ("MDB",  +10.12),
            ("ZS",   +21.39),
        ],
    ),
    "2021-11": dict(
        analysis_date=date(2021, 11, 1),
        end_date=date(2021, 12, 1),
        picks=[
            ("NET",  -2.38),
            ("MDB",  -3.83),
            ("DDOG", +9.19),
            ("MRNA", -1.49),
            ("U",    +14.12),
        ],
    ),
    "2021-12": dict(
        analysis_date=date(2021, 12, 1),
        end_date=date(2022, 1, 3),
        picks=[
            ("NET",  -31.06),
            ("TSLA", -1.12),
            ("DDOG", -2.24),
            ("U",    -17.17),
            ("MDB",  +2.59),
        ],
    ),
    # ═══════════════════ 2019 ═══════════════════
    "2019-01": dict(
        analysis_date=date(2019, 1, 2),
        end_date=date(2019, 2, 1),
        picks=[
            ("SBUX", +7.71),
            ("MKC",  -10.20),
            ("CHD",  -0.57),
            ("HRL",  +0.12),
            ("TSLA", -0.24),
        ],
    ),
    "2019-02": dict(
        analysis_date=date(2019, 2, 1),
        end_date=date(2019, 3, 1),
        picks=[
            ("TWLO", +10.73),
            ("WDAY", +10.62),
            ("OKTA", +5.40),
            ("SBUX", +2.97),
            ("CHD",  +1.67),
        ],
    ),
    "2019-03": dict(
        analysis_date=date(2019, 3, 1),
        end_date=date(2019, 4, 1),
        picks=[
            ("TWLO", +6.76),
            ("OKTA", -3.50),
            ("WDAY", -3.17),
            ("KEYS", +3.53),
            ("NTNX", +4.05),
        ],
    ),
    "2019-04": dict(
        analysis_date=date(2019, 4, 1),
        end_date=date(2019, 5, 1),
        picks=[
            ("W",    +8.59),
            ("TTD",  +10.57),
            ("TWLO", +8.07),
            ("KEYS", +0.00),
            ("CMG",  -3.23),
        ],
    ),
    "2019-05": dict(
        analysis_date=date(2019, 5, 1),
        end_date=date(2019, 6, 3),
        picks=[
            ("W",    -11.79),
            ("TTD",  -10.32),
            ("CMG",  -4.57),
            ("CDNS", -9.04),
            ("ANET", -22.61),
        ],
    ),
    "2019-06": dict(
        analysis_date=date(2019, 6, 3),
        end_date=date(2019, 7, 1),
        picks=[
            ("QCOM", +20.24),
            ("ALGN", -0.78),
            ("OKTA", +10.10),
            ("TTD",  +18.06),
            ("CDNS", +14.61),
        ],
    ),
    "2019-07": dict(
        analysis_date=date(2019, 7, 1),
        end_date=date(2019, 8, 1),
        picks=[
            ("OKTA", +4.76),
            ("QCOM", -14.62),
            ("IR",   -10.84),
            ("VEEV", -0.06),
            ("TTD",  +12.03),
        ],
    ),
    "2019-08": dict(
        analysis_date=date(2019, 8, 1),
        end_date=date(2019, 9, 1),
        picks=[
            ("OKTA", -4.55),
            ("VEEV", -3.53),
            ("BX",   +2.67),
            ("BR",   +1.73),
            ("CPRT", -2.64),
        ],
    ),
    "2019-09": dict(
        analysis_date=date(2019, 9, 1),
        end_date=date(2019, 10, 1),
        picks=[
            ("OKTA", -21.86),
            ("DXCM", -12.00),
            ("TTWO", -4.40),
            ("HSY",  -2.33),
            ("BX",   +0.45),
        ],
    ),
    "2019-10": dict(
        analysis_date=date(2019, 10, 1),
        end_date=date(2019, 11, 1),
        picks=[
            ("WDC",  -13.73),
            ("KLAC", +5.81),
            ("DXCM", +4.01),
            ("TGT",  +0.68),
            ("TTWO", -3.13),
        ],
    ),
    "2019-11": dict(
        analysis_date=date(2019, 11, 1),
        end_date=date(2019, 12, 2),
        picks=[
            ("KLAC", -3.63),
            ("TGT",  +16.32),
            ("WDC",  -2.84),
            ("LRCX", -2.67),
            ("DG",   -1.98),
        ],
    ),
    "2019-12": dict(
        analysis_date=date(2019, 12, 2),
        end_date=date(2020, 1, 2),
        picks=[
            ("TSLA", +28.87),
            ("LRCX", +10.99),
            ("KLAC", +10.27),
            ("MPC",  +0.00),
            ("TGT",  +2.42),
        ],
    ),
    # ═══════════════════ 2022 ═══════════════════
    "2022-01": dict(
        analysis_date=date(2022, 1, 3),
        end_date=date(2022, 2, 1),
        picks=[
            ("ON",   -13.42),
            ("F",    -3.10),
            ("TSLA", -18.52),
            ("DLTR", -6.84),
            ("ANET", -13.77),
        ],
    ),
    "2022-02": dict(
        analysis_date=date(2022, 2, 1),
        end_date=date(2022, 3, 1),
        picks=[
            ("F",    -15.53),
            ("DLTR", +8.26),
            ("ANET", -1.74),
            ("ON",   +5.36),
            ("CIEN", +2.99),
        ],
    ),
    "2022-03": dict(
        analysis_date=date(2022, 3, 1),
        end_date=date(2022, 4, 1),
        picks=[
            ("HAL",  +11.77),
            ("AA",   +17.72),
            ("DVN",  -1.05),
            ("EOG",  +2.83),
            ("VRTX", +14.12),
        ],
    ),
    "2022-04": dict(
        analysis_date=date(2022, 4, 1),
        end_date=date(2022, 5, 2),
        picks=[
            ("OXY",  -4.03),
            ("AA",   -26.93),
            ("HAL",  -7.14),
            ("DVN",  -3.23),
            ("CVX",  -3.97),
        ],
    ),
    "2022-05": dict(
        analysis_date=date(2022, 5, 2),
        end_date=date(2022, 6, 1),
        picks=[
            ("OXY",  +29.53),
            ("EQT",  +22.83),
            ("HAL",  +16.10),
            ("AA",   -8.75),
            ("BKR",  +18.84),
        ],
    ),
    "2022-06": dict(
        analysis_date=date(2022, 6, 1),
        end_date=date(2022, 7, 1),
        picks=[
            ("EQT",  -27.92),
            ("OXY",  -15.88),
            ("VLO",  -17.29),
            ("CTRA", -25.24),
            ("HAL",  -22.98),
        ],
    ),
    "2022-07": dict(
        analysis_date=date(2022, 7, 1),
        end_date=date(2022, 8, 1),
        picks=[
            ("EQT",  +23.32),
            ("VLO",  +1.15),
            ("OXY",  +9.70),
            ("CTRA", +15.23),
            ("MPC",  +8.46),
        ],
    ),
    "2022-08": dict(
        analysis_date=date(2022, 8, 1),
        end_date=date(2022, 9, 1),
        picks=[
            ("EQT",  +9.55),
            ("RBA",  -3.59),
            ("MNST", -10.63),
            ("LLY",  -8.09),
            ("VICI", -3.61),
        ],
    ),
    "2022-09": dict(
        analysis_date=date(2022, 9, 1),
        end_date=date(2022, 10, 3),
        picks=[
            ("SMCI", -11.42),
            ("ALNY", -0.79),
            ("RBA",  -9.09),
            ("DECK", -0.70),
            ("CDNS", -3.87),
        ],
    ),
    "2022-10": dict(
        analysis_date=date(2022, 10, 3),
        end_date=date(2022, 11, 1),
        picks=[
            ("ALNY", +2.13),
            ("SMCI", +24.03),
            ("DKS",  +10.14),
            ("DECK", +12.62),
            ("ON",   -0.44),
        ],
    ),
    "2022-11": dict(
        analysis_date=date(2022, 11, 1),
        end_date=date(2022, 12, 1),
        picks=[
            ("ALNY", +5.20),
            ("DKS",  +3.00),
            ("AXON", +20.91),
            ("DECK", +11.93),
            ("LNG",  -0.56),
        ],
    ),
    "2022-12": dict(
        analysis_date=date(2022, 12, 1),
        end_date=date(2023, 1, 3),
        picks=[
            ("AXON", -5.66),
            ("DXCM", -1.80),
            ("SLB",  +1.03),
            ("SMCI", -9.48),
            ("BIIB", -9.42),
        ],
    ),
    # ═══════════════════ 2023 ═══════════════════
    "2023-01": dict(
        analysis_date=date(2023, 1, 3),
        end_date=date(2023, 2, 1),
        picks=[
            ("AXON", +17.44),
            ("SMCI", -13.36),
            ("SLB",  +7.66),
            ("GILD", -1.26),
            ("AU",   +4.97),
        ],
    ),
    "2023-02": dict(
        analysis_date=date(2023, 2, 1),
        end_date=date(2023, 3, 1),
        picks=[
            ("SCCO", +1.65),
            ("FCX",  -3.77),
            ("AXON", +5.87),
            ("WMG",  -13.62),
            ("HAL",  -11.14),
        ],
    ),
    "2023-03": dict(
        analysis_date=date(2023, 3, 1),
        end_date=date(2023, 4, 3),
        picks=[
            ("SCCO", +1.38),
            ("NVDA", +18.63),
            ("LVS",  +2.05),
            ("GE",   +13.09),
            ("FCX",  -3.45),
        ],
    ),
    "2023-04": dict(
        analysis_date=date(2023, 4, 3),
        end_date=date(2023, 5, 1),
        picks=[
            ("META", +14.26),
            ("NVDA", +1.20),
            ("HUBS", -0.11),
            ("GE",   +3.92),
            ("AMD",  -5.86),
        ],
    ),
    "2023-05": dict(
        analysis_date=date(2023, 5, 1),
        end_date=date(2023, 6, 1),
        picks=[
            ("META", +11.43),
            ("NVDA", +38.25),
            ("MSTR", -6.72),
            ("HUBS", +20.19),
            ("AMD",  +28.85),
        ],
    ),
    "2023-06": dict(
        analysis_date=date(2023, 6, 1),
        end_date=date(2023, 7, 3),
        picks=[
            ("META", +7.82),
            ("SMCI", +12.49),
            ("NVDA", +10.47),
            ("HUBS", +4.20),
            ("CRM",  +1.13),
        ],
    ),
    "2023-07": dict(
        analysis_date=date(2023, 7, 3),
        end_date=date(2023, 8, 1),
        picks=[
            ("SMCI", +26.52),
            ("PLTR", +25.63),
            ("NVDA", +9.27),
            ("MDB",  +3.25),
            ("META", +10.76),
        ],
    ),
    "2023-08": dict(
        analysis_date=date(2023, 8, 1),
        end_date=date(2023, 9, 1),
        picks=[
            ("SMCI", -14.11),
            ("PLTR", -22.10),
            ("MDB",  -5.95),
            ("NVDA", +7.10),
            ("MRVL", -9.14),
        ],
    ),
    "2023-09": dict(
        analysis_date=date(2023, 9, 1),
        end_date=date(2023, 10, 2),
        picks=[
            ("SMCI", -0.72),
            ("PLTR", +5.74),
            ("MDB",  -13.32),
            ("NVDA", -11.52),
            ("RCL",  -7.52),
        ],
    ),
    "2023-10": dict(
        analysis_date=date(2023, 10, 2),
        end_date=date(2023, 11, 1),
        picks=[
            ("SMCI", -12.25),
            ("OVV",  +0.90),
            ("DELL", -1.92),
            ("MPC",  +0.91),
            ("HAL",  -2.05),
        ],
    ),
    "2023-11": dict(
        analysis_date=date(2023, 11, 1),
        end_date=date(2023, 12, 1),
        picks=[
            ("DELL", +6.79),
            ("MPC",  -2.39),
            ("OVV",  -8.00),
            ("HAL",  -6.37),
            ("LLY",  +6.61),
        ],
    ),
    "2023-12": dict(
        analysis_date=date(2023, 12, 1),
        end_date=date(2024, 1, 2),
        picks=[
            ("DELL", +5.00),
            ("CRWD", +6.68),
            ("PGR",  -2.66),
            ("WSM",  +6.97),
            ("URA",  -5.38),
        ],
    ),
    # ═══════════════════ 2024 ═══════════════════
    "2024-01": dict(
        analysis_date=date(2024, 1, 2),
        end_date=date(2024, 2, 1),
        picks=[
            ("MSTR", -27.54),
            ("CRWD", +17.76),
            ("GDDY", +1.88),
            ("WSM",  -2.68),
            ("RBLX", -12.39),
        ],
    ),
    "2024-02": dict(
        analysis_date=date(2024, 2, 1),
        end_date=date(2024, 3, 1),
        picks=[
            ("CRWD", +7.80),
            ("MSTR", +98.49),
            ("RBLX", +2.04),
            ("RKT",  +0.40),
            ("AFRM", -8.20),
        ],
    ),
    "2024-03": dict(
        analysis_date=date(2024, 3, 1),
        end_date=date(2024, 4, 1),
        picks=[
            ("SMCI", +14.53),
            ("AFRM", -0.90),
            ("COIN", +29.19),
            ("CRWD", +0.55),
            ("AMD",  -9.00),
        ],
    ),
    "2024-04": dict(
        analysis_date=date(2024, 4, 1),
        end_date=date(2024, 5, 1),
        picks=[
            ("SMCI", -23.03),
            ("MSTR", -38.27),
            ("CVNA", -5.71),
            ("COIN", -24.01),
            ("NVDA", -5.78),
        ],
    ),
    "2024-05": dict(
        analysis_date=date(2024, 5, 1),
        end_date=date(2024, 6, 3),
        picks=[
            ("MSTR", +57.66),
            ("SMCI", +3.16),
            ("CVNA", +25.07),
            ("VST",  +31.98),
            ("COIN", +16.83),
        ],
    ),
    "2024-06": dict(
        analysis_date=date(2024, 6, 3),
        end_date=date(2024, 7, 1),
        picks=[
            ("VST",  -11.14),
            ("CVNA", +24.89),
            ("MSTR", -11.93),
            ("APP",  +2.49),
            ("VRT",  -10.99),
        ],
    ),
    "2024-07": dict(
        analysis_date=date(2024, 7, 1),
        end_date=date(2024, 8, 1),
        picks=[
            ("VST",  -10.43),
            ("CVNA", +14.23),
            ("FSLR", -2.51),
            ("MSTR", +14.54),
            ("NVDA", -4.81),
        ],
    ),
    "2024-08": dict(
        analysis_date=date(2024, 8, 1),
        end_date=date(2024, 9, 3),
        picks=[
            ("CVNA", +0.99),
            ("FSLR", +3.45),
            ("ALNY", +4.18),
            ("NVDA", -1.29),
            ("TER",  +3.67),
        ],
    ),
    "2024-09": dict(
        analysis_date=date(2024, 9, 3),
        end_date=date(2024, 10, 1),
        picks=[
            ("ALNY", +5.54),
            ("CVNA", +17.14),
            ("TSLA", +22.02),
            ("IRM",  +5.44),
            ("RKT",  -1.18),
        ],
    ),
    "2024-10": dict(
        analysis_date=date(2024, 10, 1),
        end_date=date(2024, 11, 1),
        picks=[
            ("ALNY", -2.85),
            ("PLTR", +12.68),
            ("RKT",  -15.46),
            ("IRM",  +4.94),
            ("CVNA", +41.88),
        ],
    ),
    "2024-11": dict(
        analysis_date=date(2024, 11, 1),
        end_date=date(2024, 12, 2),
        picks=[
            ("APP",  +96.21),
            ("PLTR", +60.84),
            ("AFRM", +62.94),
            ("CVNA", +5.79),
            ("ALNY", -5.50),
        ],
    ),
    "2024-12": dict(
        analysis_date=date(2024, 12, 2),
        end_date=date(2025, 1, 2),
        picks=[
            ("APP",  -1.41),
            ("MSTR", -23.76),
            ("PLTR", +12.99),
            ("SOFI", -9.04),
            ("AFRM", -13.44),
        ],
    ),
    # ═══════════════════ 2025 ═══════════════════
    "2025-01": dict(
        analysis_date=date(2025, 1, 2),
        end_date=date(2025, 2, 3),
        picks=[
            ("APP",  +6.78),
            ("MSTR", +3.30),
            ("PLTR", +5.16),
            ("SOFI", -0.13),
            ("AFRM", -8.24),
        ],
    ),
    "2025-02": dict(
        analysis_date=date(2025, 2, 3),
        end_date=date(2025, 3, 3),
        picks=[
            ("APP",  -0.93),
            ("RKLB", -21.88),
            ("IONQ", -32.30),
            ("PLTR", +10.51),
            ("MSTR", -5.44),
        ],
    ),
    "2025-03": dict(
        analysis_date=date(2025, 3, 3),
        end_date=date(2025, 4, 1),
        picks=[
            ("HOOD", -23.61),
            ("APP",  -24.56),
            ("RKLB", -16.71),
            ("PLTR", -5.26),
            ("NET",  -21.77),
        ],
    ),
    "2025-04": dict(
        analysis_date=date(2025, 4, 1),
        end_date=date(2025, 5, 1),
        picks=[
            ("PLTR", +43.09),
            ("HOOD", +22.32),
            ("OKTA", +8.15),
            ("NET",  +10.10),
            ("TPR",  +0.38),
        ],
    ),
    "2025-05": dict(
        analysis_date=date(2025, 5, 1),
        end_date=date(2025, 6, 2),
        picks=[
            ("AU",   +13.37),
            ("OKTA", -9.04),
            ("PM",   +6.06),
            ("NEM",  +4.70),
            ("BJ",   -3.15),
        ],
    ),
    "2025-06": dict(
        analysis_date=date(2025, 6, 2),
        end_date=date(2025, 7, 1),
        picks=[
            ("AU",   +3.60),
            ("NEM",  +9.31),
            ("PLTR", +2.91),
            ("PM",   +1.60),
            ("RGLD", -0.10),
        ],
    ),
    "2025-07": dict(
        analysis_date=date(2025, 7, 1),
        end_date=date(2025, 8, 1),
        picks=[
            ("NRG",  +1.19),
            ("AU",   +0.68),
            ("CVNA", +13.42),
            ("RBLX", +26.57),
            ("ZS",   -10.51),
        ],
    ),
    "2025-08": dict(
        analysis_date=date(2025, 8, 1),
        end_date=date(2025, 9, 2),
        picks=[
            ("HOOD", +2.60),
            ("SYM",  -9.34),
            ("RKLB", +8.22),
            ("RBLX", -8.46),
            ("COIN", -10.49),
        ],
    ),
    "2025-09": dict(
        analysis_date=date(2025, 9, 2),
        end_date=date(2025, 10, 1),
        picks=[
            ("RKLB", +1.35),
            ("HOOD", +40.78),
            ("SYM",  +17.04),
            ("ALAB", +8.28),
            ("ETHA", -0.06),
        ],
    ),
    "2025-10": dict(
        analysis_date=date(2025, 10, 1),
        end_date=date(2025, 11, 3),
        picks=[
            ("ALAB", -0.21),
            ("RDDT", -0.91),
            ("ETHA", -13.49),
            ("SOFI", +10.96),
            ("RKLB", +33.03),
        ],
    ),
    "2025-11": dict(
        analysis_date=date(2025, 11, 3),
        end_date=date(2025, 12, 1),
        picks=[
            ("WDC",  +5.35),
            ("CIEN", +0.59),
            ("ALAB", -18.69),
            ("WBD",  +5.87),
            ("APP",  -9.10),
        ],
    ),
    "2025-12": dict(
        analysis_date=date(2025, 12, 1),
        end_date=date(2026, 1, 2),
        picks=[
            ("CIEN", +23.81),
            ("WDC",  +10.61),
            ("MU",   +27.05),
            ("WBD",  +20.86),
            ("INSM", -15.00),
        ],
    ),
    # ═══════════════════ 2026 ═══════════════════
    "2026-01": dict(
        analysis_date=date(2026, 1, 2),
        end_date=date(2026, 2, 2),
        picks=[
            ("CIEN", +2.88),
            ("WDC",  +37.56),
            ("WBD",  -4.41),
            ("MU",   +39.66),
            ("COHR", +11.06),
        ],
    ),
    "2026-02": dict(
        analysis_date=date(2026, 2, 2),
        end_date=date(2026, 3, 2),
        picks=[
            ("PL",   -4.70),
            ("HL",   +9.05),
            ("RVMD", +5.16),
            ("MU",   -2.60),
            ("LITE", +83.21),
        ],
    ),
    "2026-03": dict(
        analysis_date=date(2026, 3, 2),
        end_date=date(2026, 4, 1),
        picks=[
            ("LITE", -1.08),
            ("SNDK", +5.44),
            ("MU",   -13.09),
            ("FORM", +2.23),
            ("WDC",  +3.70),
        ],
    ),
    "2026-04": dict(
        analysis_date=date(2026, 4, 1),
        end_date=date(2026, 5, 1),
        picks=[
            ("SNDK", +62.35),
            ("LITE", +24.24),
            ("AAOI", +80.45),
            ("FORM", +37.54),
            ("VIAV", +56.58),
        ],
    ),
    "2026-05": dict(
        analysis_date=date(2026, 5, 1),
        end_date=date(2026, 5, 29),
        picks=[
            ("AAOI", -2.62),
            ("SNDK", +60.05),
            ("LITE", -6.36),
            ("VIAV", -9.27),
            ("CIEN", +10.05),
        ],
    ),
    "2026-06": dict(
        analysis_date=date(2026, 6, 1),
        end_date=date(2026, 6, 30),
        picks=[
            ("SNDK", +31.34),
            ("AAOI", -0.73),
            ("VIAV", +1.12),
            ("DOCN", -3.07),
            ("INTC", +27.60),
        ],
    ),
    "2026-07": dict(
        analysis_date=date(2026, 7, 1),
        end_date=date(2026, 7, 31),
        picks=[
            ("SNDK", -39.34),
            ("MXL",  -42.06),
            ("MRVL", -31.49),
            ("INTC", -31.13),
            ("DOCN", -20.88),
        ],
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# 精确收益率（用实际持仓日期）
# ─────────────────────────────────────────────────────────────────────────────

def fetch_exact_return(ticker: str, start: date, end: date | None) -> float | None:
    """用实际起止日计算持仓期收益率（与截图 RETURN 列比较）。"""
    import yfinance as yf
    from datetime import timedelta
    end_dt = end if end else date.today()
    try:
        df = yf.download(
            ticker.upper(),
            start=start.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=3)).strftime("%Y-%m-%d"),
            interval="1d", progress=False, auto_adjust=True,
        )
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df.dropna()
        if len(df) < 2:
            return None
        buy  = float(df["Close"].iloc[0])
        sell = float(df["Close"].iloc[-1])
        return round((sell - buy) / buy * 100, 2)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 主批量流程
# ─────────────────────────────────────────────────────────────────────────────

def run_validation(
    months: list[str],
    use_claude: bool = False,
    sleep_between: float = 1.5,
) -> pd.DataFrame:
    """
    对所有指定月份运行三位一体验证。
    返回包含三位一体信号 + Quant GT 实际收益的汇总 DataFrame。
    """
    import anthropic

    client = None
    if use_claude:
        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            base_url="https://api.anthropic.com",
        )

    all_rows: list[dict] = []
    total = sum(len(QUANT_GT_PICKS[m]["picks"]) for m in months)
    done  = 0

    for month_key in months:
        info          = QUANT_GT_PICKS[month_key]
        analysis_date = info["analysis_date"]
        end_date      = info["end_date"]
        picks         = info["picks"]

        print(f"\n{'─'*70}")
        print(f"  Month: {month_key}  |  Analysis date: {analysis_date}  "
              f"|  End: {end_date or '持仓中'}")
        print(f"{'─'*70}")

        for ticker, actual_ret in picks:
            done += 1
            print(f"  [{done}/{total}] {ticker} ...", end=" ", flush=True)

            try:
                summary = trinity_analysis_as_of(
                    ticker=ticker,
                    as_of=analysis_date,
                    use_claude=use_claude,
                    client=client,
                )
            except Exception as e:
                print(f"ERROR: {e}")
                all_rows.append({
                    "month": month_key,
                    "ticker": ticker,
                    "analysis_date": str(analysis_date),
                    "trinity_signal": "ERROR",
                    "quant_gt_return": actual_ret,
                    "error": str(e),
                })
                continue

            if "error" in summary:
                print(f"SKIP: {summary['error']}")
                all_rows.append({
                    "month": month_key,
                    "ticker": ticker,
                    "analysis_date": str(analysis_date),
                    "trinity_signal": "SKIP",
                    "quant_gt_return": actual_ret,
                    "error": summary["error"],
                })
                continue

            sig   = summary.get("signal", "hold")
            conf  = summary.get("confidence", "")
            state = summary.get("state_label", "")
            wkly  = summary.get("weekly_state", "")
            trend = summary.get("trend_alignment", "")
            div   = summary.get("divergence_type", "none")

            # 三位一体是否推荐买入
            trinity_buy = sig in ("buy", "strong_buy")
            # Quant GT 实际盈利？
            qgt_profit  = actual_ret > 0

            # 评估：Trinity 买 + QGT 赚钱 → TP (True Positive)
            #       Trinity 不买 + QGT 亏 → TN
            #       Trinity 买 + QGT 亏   → FP
            #       Trinity 不买 + QGT 赚 → FN
            if trinity_buy and qgt_profit:
                verdict = "TP"
            elif not trinity_buy and not qgt_profit:
                verdict = "TN"
            elif trinity_buy and not qgt_profit:
                verdict = "FP"
            else:
                verdict = "FN"

            row = {
                "month":           month_key,
                "ticker":          ticker,
                "analysis_date":   str(analysis_date),
                "close":           summary.get("current_price"),
                "state":           state,
                "weekly_state":    wkly,
                "trend":           trend,
                "divergence":      div,
                "trinity_signal":  sig,
                "confidence":      conf,
                "trinity_buy":     "YES" if trinity_buy else "no",
                "quant_gt_return": actual_ret,
                "quant_gt_profit": "YES" if qgt_profit else "no",
                "verdict":         verdict,
                "action_summary":  summary.get("suggested_action", "")[:80],
                "key_risk":        summary.get("key_risk", "")[:60],
            }
            all_rows.append(row)

            # 打印摘要
            verdict_icon = {"TP": "✓ ", "TN": "✓~", "FP": "✗ ", "FN": "✗~"}.get(verdict, "??")
            print(
                f"sig={sig:10s} conf={conf:6s} state={state}  "
                f"QGT={actual_ret:+.1f}%  [{verdict_icon}{verdict}]"
            )

            if use_claude and done < total:
                _time.sleep(sleep_between)

    return pd.DataFrame(all_rows)


# ─────────────────────────────────────────────────────────────────────────────
# 打印汇总统计
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*80}")
    print("  QUANT GT × 三位一体 验证报告")
    print(f"{'='*80}\n")

    # 过滤掉错误行
    ok = df[df["trinity_signal"].notna() & ~df["trinity_signal"].isin(["ERROR", "SKIP"])]

    if ok.empty:
        print("  无有效数据。")
        return

    # ── 每月汇总 ─────────────────────────────────────────────────────────────
    print("  ── 每月汇总 ──────────────────────────────────────────────────────\n")
    print(f"  {'月份':<10} {'三位一体BUY':<12} {'QGT赚钱':<10} "
          f"{'TP':>4} {'TN':>4} {'FP':>4} {'FN':>4}  {'吻合率':>8}  {'QGT月均'}")
    print(f"  {'-'*80}")

    for month_key in sorted(ok["month"].unique()):
        sub = ok[ok["month"] == month_key]
        n_buy   = (sub["trinity_buy"] == "YES").sum()
        n_prof  = (sub["quant_gt_profit"] == "YES").sum()
        tp = (sub["verdict"] == "TP").sum()
        tn = (sub["verdict"] == "TN").sum()
        fp = (sub["verdict"] == "FP").sum()
        fn = (sub["verdict"] == "FN").sum()
        match_rate = (tp + tn) / len(sub) * 100 if len(sub) else 0
        avg_ret = sub["quant_gt_return"].mean()
        print(
            f"  {month_key:<10} {n_buy:>5}/{len(sub):<5}  {n_prof:>4}/{len(sub):<4}  "
            f"{tp:>4} {tn:>4} {fp:>4} {fn:>4}   {match_rate:>6.0f}%   {avg_ret:>+.1f}%"
        )

    print()

    # ── 整体混淆矩阵 ─────────────────────────────────────────────────────────
    tp_all = (ok["verdict"] == "TP").sum()
    tn_all = (ok["verdict"] == "TN").sum()
    fp_all = (ok["verdict"] == "FP").sum()
    fn_all = (ok["verdict"] == "FN").sum()
    total  = len(ok)

    precision = tp_all / (tp_all + fp_all) * 100 if (tp_all + fp_all) else 0
    recall    = tp_all / (tp_all + fn_all) * 100 if (tp_all + fn_all) else 0
    accuracy  = (tp_all + tn_all) / total * 100 if total else 0

    print(f"  ── 整体指标 ──────────────────────────────────────────────────────\n")
    print(f"  总样本: {total}  |  TP={tp_all}  TN={tn_all}  FP={fp_all}  FN={fn_all}")
    print(f"  Accuracy  (正确吻合率):  {accuracy:.1f}%")
    print(f"  Precision (买了真赚):    {precision:.1f}%")
    print(f"  Recall    (赚钱被捕捉):  {recall:.1f}%")

    # ── 三位一体 BUY 推荐的实际收益 ─────────────────────────────────────────
    buy_rows  = ok[ok["trinity_buy"] == "YES"]
    nobuy_rows = ok[ok["trinity_buy"] == "no"]
    if not buy_rows.empty:
        avg_buy   = buy_rows["quant_gt_return"].mean()
        win_buy   = (buy_rows["quant_gt_return"] > 0).mean() * 100
        avg_nobuy = nobuy_rows["quant_gt_return"].mean() if not nobuy_rows.empty else float("nan")
        print(f"\n  三位一体 BUY  推荐 ({len(buy_rows)}支):  "
              f"QGT平均收益 {avg_buy:+.1f}%  胜率 {win_buy:.0f}%")
        print(f"  三位一体 跳过 ({len(nobuy_rows)}支):  "
              f"QGT平均收益 {avg_nobuy:+.1f}%")

    # ── 信号与实际收益关系 ────────────────────────────────────────────────────
    print(f"\n  ── 按三位一体信号分组的 QGT 实际收益 ─────────────────────────────\n")
    for sig, grp in ok.groupby("trinity_signal"):
        avg = grp["quant_gt_return"].mean()
        win = (grp["quant_gt_return"] > 0).mean() * 100
        print(f"  {sig:12s}  n={len(grp):3d}  QGT均收益={avg:+.1f}%  胜率={win:.0f}%")

    # ── 本月（最新）详情 ─────────────────────────────────────────────────────
    latest = sorted(ok["month"].unique())[-1]
    latest_sub = ok[ok["month"] == latest]
    print(f"\n  ── 最新月 {latest} 详情 ────────────────────────────────────────\n")
    print(f"  {'Ticker':<6} {'Trinity':>10} {'Conf':>6} {'State':<12} {'BUY?':>5}  "
          f"{'QGT Ret':>8}  Verdict")
    print(f"  {'-'*72}")
    for _, r in latest_sub.iterrows():
        print(
            f"  {r['ticker']:<6} {r['trinity_signal']:>10} {r['confidence']:>6} "
            f"{str(r['state']):<12} {r['trinity_buy']:>5}  "
            f"{r['quant_gt_return']:>+7.1f}%  {r['verdict']}"
        )

    print()


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quant GT × 三位一体 历史验证"
    )
    parser.add_argument("--claude", action="store_true",
                        help="启用 Claude 软分析（默认仅硬指标，快速）")
    parser.add_argument("--year", type=int, default=None,
                        help="只分析某年，例如 --year 2025")
    parser.add_argument("--month", type=str, default=None,
                        help="只分析某月，例如 --month 2025-04")
    parser.add_argument("--output", type=str, default="quant_gt_trinity_results.csv",
                        help="输出 CSV 文件路径（默认 quant_gt_trinity_results.csv）")
    args = parser.parse_args()

    # 过滤月份
    all_months = sorted(QUANT_GT_PICKS.keys())
    if args.month:
        if args.month not in QUANT_GT_PICKS:
            print(f"[ERROR] 月份 {args.month} 不在数据中。可选: {all_months}")
            sys.exit(1)
        months = [args.month]
    elif args.year:
        months = [m for m in all_months if m.startswith(str(args.year))]
        if not months:
            print(f"[ERROR] 没有 {args.year} 年的数据。")
            sys.exit(1)
    else:
        months = all_months

    use_claude = args.claude
    n_stocks   = sum(len(QUANT_GT_PICKS[m]["picks"]) for m in months)

    print(f"\nQuant GT × 三位一体 验证")
    print(f"  月份范围: {months[0]} → {months[-1]}  ({len(months)} 个月, {n_stocks} 支股票)")
    print(f"  Claude:   {'启用（较慢）' if use_claude else '关闭（快速硬指标模式）'}")
    print(f"  输出:     {args.output}\n")

    df = run_validation(months=months, use_claude=use_claude)

    print_summary(df)

    df.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"  详细结果已保存至: {args.output}\n")


if __name__ == "__main__":
    main()
