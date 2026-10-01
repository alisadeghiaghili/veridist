"""Independent mpmath reference contracts for the gamma incomplete functions.

These pin the fix for the regularized lower/upper incomplete gamma ratios
used by ``cdf``/``sf``/``ppf`` for the gamma family: a modified Lentz
continued fraction that previously destroyed the sign of its intermediate
terms, which made ``cdf`` silently wrong (often ``0.0``) whenever
``x / scale >= shape + 1``.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import mpmath

from tests.reference import log_density_oracle as oracle
from veridist.families.registry import FamilyId
from veridist.statistics import distributions
from veridist.statistics.distributions import _regularized_gamma_pq, cdf, ppf, sf
from veridist.statistics.log_density import LogDensitySuccess, evaluate_log_density

_SHAPES = (0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0)
_MULTIPLIERS = (1e-3, 0.1, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0)
_PROBABILITIES = (1e-12, 1e-6, 0.05, 0.5, 0.95, 1.0 - 1e-6, 1.0 - 1e-12)
_ABSOLUTE_FLOOR = 1e-300

# For this one (shape, probability) pair the true quantile underflows below
# the smallest positive binary64 value (confirmed independently against
# ``scipy.special.gammaincinv`` in a scratch check, which also returns 0.0
# for both). Zero is then the correctly rounded quantile, so its CDF cannot
# reproduce the requested probability and the round-trip check does not apply.
_UNREPRESENTABLE_QUANTILES = frozenset({(0.01, 1e-12), (0.01, 1e-6)})


def _grid_points(shape: float) -> list[float]:
    points = [shape * multiplier for multiplier in _MULTIPLIERS]
    points.append(shape + 1.0 - 1e-9)
    points.append(shape + 1.0 + 1e-9)
    return [point for point in points if point > 0.0]


def _relative_error(got: float, true_value: mpmath.mpf) -> float:
    if true_value == 0:
        return 0.0 if got == 0.0 else float("inf")
    return float(abs(mpmath.mpf(got) - true_value) / true_value)


class GammaIncompleteGridTests(unittest.TestCase):
    """``cdf``/``sf`` match mpmath's regularized incomplete gamma everywhere."""

    def test_cdf_and_sf_match_mpmath_grid(self) -> None:
        checked = 0
        with mpmath.workdps(50):
            for shape in _SHAPES:
                for x in _grid_points(shape):
                    with self.subTest(shape=shape, x=x):
                        a = mpmath.mpf(shape)
                        xv = mpmath.mpf(x)
                        true_p = mpmath.gammainc(a, 0, xv, regularized=True)
                        true_q = mpmath.gammainc(a, xv, mpmath.inf, regularized=True)
                        got_p = cdf("gamma", x, {"shape": shape, "scale": 1.0})
                        got_q = sf("gamma", x, {"shape": shape, "scale": 1.0})
                        if true_p >= _ABSOLUTE_FLOOR:
                            self.assertLessEqual(_relative_error(got_p, true_p), 1e-12)
                        else:
                            self.assertLessEqual(abs(got_p - float(true_p)), _ABSOLUTE_FLOOR)
                        if true_q >= _ABSOLUTE_FLOOR:
                            self.assertLessEqual(_relative_error(got_q, true_q), 1e-12)
                        else:
                            self.assertLessEqual(abs(got_q - float(true_q)), _ABSOLUTE_FLOOR)
                        checked += 1
        self.assertGreater(checked, 100)


class GammaIncompletePinnedRegressionTests(unittest.TestCase):
    """The four previously wrong table rows, pinned against mpmath references.

    Every constant below was computed with ``mpmath`` at 50 decimal digits of
    precision: ``mpmath.gammainc(a, 0, x, regularized=True)`` for the CDF rows
    and ``mpmath.gammainc(a, x, mpmath.inf, regularized=True)`` for the SF
    rows. Before the fix these four calls returned ``0.0``, ``9.08e-4``,
    ``0.0`` and ``5.3696...`` respectively.
    """

    def test_cdf_shape_five_at_six_point_five(self) -> None:
        got = cdf("gamma", 6.5, {"shape": 5.0, "scale": 1.0})
        expected = 0.77632818318850068659755759254321381340078306042997
        self.assertLessEqual(abs(got - expected) / expected, 1e-9)

    def test_sf_shape_two_at_ten(self) -> None:
        got = sf("gamma", 10.0, {"shape": 2.0, "scale": 1.0})
        expected = 0.00049939922738733336689150667116605671261709897753221
        self.assertLessEqual(abs(got - expected) / expected, 1e-9)

    def test_sf_shape_two_at_fifty(self) -> None:
        got = sf("gamma", 50.0, {"shape": 2.0, "scale": 1.0})
        expected = 9.8366242246159806933884483642877641312394465212743e-21
        self.assertLessEqual(abs(got - expected) / expected, 1e-9)

    def test_ppf_shape_two_at_point_nine_five(self) -> None:
        got = ppf("gamma", 0.95, {"shape": 2.0, "scale": 1.0})
        expected = 4.743864518390577
        self.assertLessEqual(abs(got - expected) / expected, 1e-10)


class GammaPpfRoundTripTests(unittest.TestCase):
    """``cdf(ppf(p)) == p`` across the shape/probability grid."""

    def test_round_trip_across_shape_and_probability_grid(self) -> None:
        checked = 0
        for shape in _SHAPES:
            for probability in _PROBABILITIES:
                with self.subTest(shape=shape, probability=probability):
                    quantile = ppf("gamma", probability, {"shape": shape, "scale": 1.0})
                    if (shape, probability) in _UNREPRESENTABLE_QUANTILES:
                        self.assertEqual(quantile, 0.0)
                        continue
                    recovered = cdf("gamma", quantile, {"shape": shape, "scale": 1.0})
                    self.assertLessEqual(abs(recovered - probability) / probability, 1e-10)
                    checked += 1
        self.assertGreater(checked, 50)


class GammaLogDensityUnaffectedTests(unittest.TestCase):
    """The density path shares no code with the incomplete-gamma fix."""

    def test_log_density_matches_mpmath_oracle_on_the_same_grid(self) -> None:
        checked = 0
        for shape in _SHAPES:
            for x in _grid_points(shape):
                with self.subTest(shape=shape, x=x):
                    expected = oracle.gamma(x, shape=shape, scale=1.0)
                    result = evaluate_log_density(FamilyId.GAMMA, x, shape=shape, scale=1.0)
                    self.assertIsInstance(result, LogDensitySuccess)
                    assert isinstance(result, LogDensitySuccess)
                    if expected == 0.0:
                        self.assertEqual(result.log_density, 0.0)
                    else:
                        relative = abs(result.log_density - expected) / abs(expected)
                        self.assertLessEqual(relative, 1e-12)
                    checked += 1
        self.assertGreater(checked, 100)


class GammaIncompleteInternalRobustnessTests(unittest.TestCase):
    """Non-convergence and sign-preserving-clamp paths behave as specified."""

    def test_series_branch_raises_on_exhausted_iteration_budget(self) -> None:
        with patch.object(distributions, "_GAMMA_MAXIT", 0):
            with self.assertRaises(ArithmeticError):
                _regularized_gamma_pq(2.0, 0.5)

    def test_continued_fraction_branch_raises_on_exhausted_iteration_budget(self) -> None:
        with patch.object(distributions, "_GAMMA_MAXIT", 0):
            with self.assertRaises(ArithmeticError):
                _regularized_gamma_pq(2.0, 10.0)

    def test_continued_fraction_clamp_keeps_signs_and_stays_a_valid_probability(self) -> None:
        # Raising FPMIN far above the natural magnitude of the continued
        # fraction's intermediate terms forces both sign-preserving clamps
        # (``d`` and ``c``) to engage on nearly every iteration. The result
        # must still be a valid, finite probability pair that sums to one,
        # confirming the clamp keeps the sign of the clamped term instead of
        # discarding it the way the earlier ``max(abs(...), tiny)`` form did.
        with patch.object(distributions, "_GAMMA_FPMIN", 1e10):
            lower, upper = _regularized_gamma_pq(2.0, 10.0)
        self.assertTrue(0.0 <= lower <= 1.0)
        self.assertTrue(0.0 <= upper <= 1.0)
        self.assertAlmostEqual(lower + upper, 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
