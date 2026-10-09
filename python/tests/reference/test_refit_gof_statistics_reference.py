"""Reference values for the KS, AD and CvM statistics of the multi-family refit cell.

The expected KS and CvM values below were computed with ``scipy.stats.kstest``
and ``scipy.stats.cramervonmises`` against the scipy CDF of the stated model;
the expected AD values were computed with an independent 40-digit ``mpmath``
evaluation of the textbook sum over the same CDF values.  The samples and
parameters are fixed here, so the test does not fit anything.  scipy is not a
dependency of the test suite: only the recorded numbers are used.
"""

from __future__ import annotations

import unittest
from math import isfinite

import mpmath

from veridist.inference import GofStatistic, _edf_statistics, _model_statistics

REAL_SAMPLE = (-1.3, -0.4, 0.2, 0.5, 0.9, 1.4, 2.1, 3.3)
POSITIVE_SAMPLE = (0.35, 0.8, 1.1, 1.6, 2.2, 2.9, 4.1, 6.5)

# family, sample, parameters, KS, AD, CvM
REFERENCE = (
    (
        "normal",
        REAL_SAMPLE,
        {"mu": 0.8, "sigma": 1.4},
        0.09652833548352147,
        0.11868838753202428,
        0.013860050246114954,
    ),
    (
        "gumbel_right",
        REAL_SAMPLE,
        {"location": 0.3, "scale": 1.2},
        0.1024871246280182,
        0.15396370875359501,
        0.014149463181787801,
    ),
    (
        "gamma",
        POSITIVE_SAMPLE,
        {"shape": 2.1, "scale": 1.5},
        0.23886135155368005,
        0.80974849140768079,
        0.13421531147526744,
    ),
    (
        "weibull_min",
        POSITIVE_SAMPLE,
        {"shape": 1.4, "scale": 2.8},
        0.1381113409974096,
        0.17976144488419952,
        0.026793867598873165,
    ),
    (
        "lognormal",
        POSITIVE_SAMPLE,
        {"mu_log": 0.7, "sigma_log": 0.9},
        0.12416918212798778,
        0.18936664185198229,
        0.021603046734949984,
    ),
)


class ModelStatisticsReferenceTests(unittest.TestCase):
    def test_statistics_match_recorded_reference_values(self) -> None:
        for family, sample, parameters, ks, ad, cvm in REFERENCE:
            with self.subTest(family=family):
                got = _model_statistics(family, sorted(sample), parameters)
                self.assertAlmostEqual(got[GofStatistic.KS], ks, delta=1e-12)
                self.assertAlmostEqual(got[GofStatistic.AD], ad, delta=1e-12)
                self.assertAlmostEqual(got[GofStatistic.CVM], cvm, delta=1e-12)


class EdfStatisticsTests(unittest.TestCase):
    def test_small_case_by_hand(self) -> None:
        # n = 3 and fitted-CDF values 0.05, 0.4, 0.9.
        # KS: i=0 -> max(1/3 - 0.05, 0.05), i=1 -> max(2/3 - 0.4, 0.4 - 1/3),
        #     i=2 -> max(1 - 0.9, 0.9 - 2/3); the largest is 1/3 - 0.05.
        got = _edf_statistics((0.05, 0.4, 0.9))
        self.assertAlmostEqual(got[GofStatistic.KS], 1.0 / 3.0 - 0.05, places=14)
        # CvM = 1/36 + (0.05 - 1/6)^2 + (0.4 - 1/2)^2 + (0.9 - 5/6)^2
        self.assertAlmostEqual(
            got[GofStatistic.CVM],
            1.0 / 36.0 + (0.05 - 1.0 / 6.0) ** 2 + (0.4 - 0.5) ** 2 + (0.9 - 5.0 / 6.0) ** 2,
            places=14,
        )
        # AD = -3 - (1/3) * (1*ln(0.05 * 0.1) + 3*ln(0.4 * 0.6) + 5*ln(0.9 * 0.95))
        expected = (
            -3
            - (
                1 * mpmath.log(mpmath.mpf("0.05") * mpmath.mpf("0.1"))
                + 3 * mpmath.log(mpmath.mpf("0.4") * mpmath.mpf("0.6"))
                + 5 * mpmath.log(mpmath.mpf("0.9") * mpmath.mpf("0.95"))
            )
            / 3
        )
        self.assertAlmostEqual(got[GofStatistic.AD], float(expected), places=12)

    def test_probabilities_at_the_ends_are_clipped_for_ad_and_cvm_only(self) -> None:
        got = _edf_statistics((0.0, 0.5, 1.0))
        # KS sees the raw probabilities: i=0 -> 1/3, i=1 -> 1/6, i=2 -> 1/3.
        self.assertAlmostEqual(got[GofStatistic.KS], 1.0 / 3.0, places=14)
        low, high = 1e-15, 1.0 - 1e-15
        expected_ad = (
            -3.0
            - (
                1.0 * (mpmath.log(low) + mpmath.log(1 - mpmath.mpf(high)))
                + 3.0 * (mpmath.log(0.5) + mpmath.log(0.5))
                + 5.0 * (mpmath.log(mpmath.mpf(high)) + mpmath.log(1 - mpmath.mpf(low)))
            )
            / 3.0
        )
        self.assertTrue(isfinite(got[GofStatistic.AD]))
        self.assertAlmostEqual(got[GofStatistic.AD], float(expected_ad), places=9)
        expected_cvm = 1.0 / 36.0 + (low - 1.0 / 6.0) ** 2 + 0.0 + (high - 5.0 / 6.0) ** 2
        self.assertAlmostEqual(got[GofStatistic.CVM], expected_cvm, places=14)

    def test_statistics_agree_with_the_exponential_cell(self) -> None:
        from math import expm1

        from veridist.inference import _empirical_statistics

        values = (0.31, 0.62, 0.9, 1.4, 2.2, 3.5, 0.15, 0.77)
        rate = len(values) / sum(values)
        probabilities = tuple(-expm1(-rate * value) for value in sorted(values))
        generic = _edf_statistics(probabilities)
        exponential = _empirical_statistics(values)
        for statistic in GofStatistic:
            with self.subTest(statistic=statistic):
                self.assertAlmostEqual(generic[statistic], exponential[statistic], places=12)


if __name__ == "__main__":
    unittest.main()
