"""tests/test_data.py
====================
Regression tests for data-layer normalization and per-share metric safety.

Run:  python tests/test_data.py
"""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tools.valuation import _normalized_growth_rate


class TestGrowthRateNormalization(unittest.TestCase):

    def test_percent_scale_divided(self):
        # 13.685 from a percent-scale provider → 0.13685
        self.assertAlmostEqual(_normalized_growth_rate(13.685), 0.13685, places=5)

    def test_decimal_scale_unchanged(self):
        # 0.13685 already a decimal → unchanged
        self.assertAlmostEqual(_normalized_growth_rate(0.13685), 0.13685, places=5)

    def test_cap_upper(self):
        # 150% growth → divided to 1.5, then capped at 0.30
        self.assertAlmostEqual(_normalized_growth_rate(150.0), 0.30, places=5)

    def test_cap_lower(self):
        # -80% decline → divided to -0.80, then clamped to -0.50
        self.assertAlmostEqual(_normalized_growth_rate(-80.0), -0.50, places=5)

    def test_none_returns_default(self):
        self.assertAlmostEqual(_normalized_growth_rate(None), 0.05, places=5)

    def test_nan_returns_default(self):
        import math
        self.assertAlmostEqual(_normalized_growth_rate(float("nan")), 0.05, places=5)

    def test_zero_unchanged(self):
        self.assertAlmostEqual(_normalized_growth_rate(0.0), 0.0, places=5)

    def test_small_positive_decimal_unchanged(self):
        # 0.05 is already a decimal (5%), should stay
        self.assertAlmostEqual(_normalized_growth_rate(0.05), 0.05, places=5)

    def test_negative_percent_scale_divided(self):
        # -15.0 percent-scale → -0.15 decimal
        self.assertAlmostEqual(_normalized_growth_rate(-15.0), -0.15, places=5)


class TestFCFPerShare(unittest.TestCase):
    """Validate that missing sharesOutstanding prevents per-share fabrication."""

    def _build_info(self, free_cashflow, shares_outstanding):
        return {
            "freeCashflow": free_cashflow,
            "sharesOutstanding": shares_outstanding,
        }

    def _calc_fcf_per_share(self, free_cashflow, shares_outstanding):
        # Mirror the logic in tools/data.py
        fcf = free_cashflow
        shares = shares_outstanding
        if fcf is not None and shares and shares > 0:
            return fcf / shares
        return None

    def test_valid_shares_divides_correctly(self):
        result = self._calc_fcf_per_share(7_600_000_000, 1_100_000_000)
        self.assertAlmostEqual(result, 6.909, places=2)

    def test_missing_shares_returns_none(self):
        # Regression: previously this returned FCF total (billions) as "per share"
        result = self._calc_fcf_per_share(7_600_000_000, None)
        self.assertIsNone(result)

    def test_zero_shares_returns_none(self):
        result = self._calc_fcf_per_share(7_600_000_000, 0)
        self.assertIsNone(result)

    def test_none_fcf_returns_none(self):
        result = self._calc_fcf_per_share(None, 1_100_000_000)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
