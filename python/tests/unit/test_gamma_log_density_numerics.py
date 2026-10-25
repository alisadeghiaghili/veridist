"""Unit tests of the exact-ratio helpers and the large-shape incomplete-gamma prefactor."""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

import mpmath

from veridist.families.registry import FamilyId
from veridist.statistics.distributions import (
    _STABLE_PREFACTOR_SHAPE,
    _log_gamma_prefactor,
    _log_gamma_prefactor_terms,
    _log_regularized_gamma_q,
    logpdf,
)
from veridist.statistics.log_density import (
    LogDensitySuccess,
    _log_exact_ratio,
    _log_ratio,
    _ratio_and_log,
    _ratio_terms,
    evaluate_log_density,
)

#: ``(shape, scale, observation, log-density)`` recorded from the previous implementation at
#: points where it was accurate (ratio of at least one half for shapes of 8 or more, any ratio
#: below shape 8); the fix must leave them unchanged.
UNCHANGED_VALUES = (
    (8.0, 1.0, 4.0, -2.8211008332261804),
    (8.0, 1.0, 6.0, -1.9828450764690297),
    (10.0, 0.37, 3.7, -1.0843093697911916),
    (30.0, 13.0, 507.0, -6.5787005868698),
    (100.0, 1.0, 200.0, -34.60078608131977),
    (1000.0, 1e-06, 0.0006, -100.87218709028673),
    (10000.0, 1.0, 10000.001, -5.52411715257609),
    (10000.0, 1000000.0, 500000000000.0, -450903.02159633447),
    (3.0, 1.0, 3e-10, -44.54762446340464),
    (7.5, 0.37, 2.7750000000000004e-05, -68.27733285219662),
    (0.05, 2.0, 100.0, -57.37844823676841),
)


def _mp_log_ratio(top: int, bottom: int) -> mpmath.mpf:
    with mpmath.workdps(60):
        return mpmath.log(mpmath.mpf(top) / mpmath.mpf(bottom))


class ExactRatioLogarithmTests(unittest.TestCase):
    def test_log_exact_ratio_matches_mpmath_on_both_sides_of_one(self) -> None:
        cases = (
            (1, 1 << 1074),
            (3, 7),
            (7, 3),
            (1 << 1100, 3),
            ((1 << 70) + 12345, (1 << 75) + 999),
            (5, 1 << 2000),
            (1, 3),
        )
        for top, bottom in cases:
            with self.subTest(top=top.bit_length(), bottom=bottom.bit_length()):
                expected = float(_mp_log_ratio(top, bottom))
                self.assertLessEqual(
                    abs(_log_exact_ratio(top, bottom) - expected), 4e-16 * max(1.0, abs(expected))
                )

    def test_ratio_terms_reproduce_the_exact_binary_quotient(self) -> None:
        top, bottom = _ratio_terms(5e-324, 1e300, 7.5)
        self.assertEqual(
            Fraction(top, bottom), Fraction(5e-324) / (Fraction(1e300) * Fraction(7.5))
        )

    def test_ratio_and_log_keep_the_log_when_the_ratio_underflows(self) -> None:
        ratio, log_ratio = _ratio_and_log(5e-324, 1e300, 7.5)
        self.assertEqual(ratio, 0.0)
        top, bottom = _ratio_terms(5e-324, 1e300, 7.5)
        self.assertAlmostEqual(log_ratio, float(_mp_log_ratio(top, bottom)), delta=1e-12)
        ratio, log_ratio = _ratio_and_log(3.0, 4.0)
        self.assertEqual(ratio, 0.75)
        self.assertAlmostEqual(log_ratio, math.log(0.75), delta=4e-16)

    def test_weibull_log_ratio_far_from_one_has_no_cancellation(self) -> None:
        for numerator, denominator in (
            (1.5e-300, 1e-300),
            (2.2250738585072014e-308, 2.5e-310),
            (5e-324, 1e-300),
            (1.5e-8, 1e-8),
        ):
            with self.subTest(numerator=numerator, denominator=denominator):
                top, bottom = _ratio_terms(numerator, denominator)
                expected = float(_mp_log_ratio(top, bottom))
                self.assertLessEqual(
                    abs(_log_ratio(numerator, denominator) - expected),
                    4e-16 * max(1.0, abs(expected)),
                )

    def test_weibull_array_and_scalar_kernels_agree_far_from_one(self) -> None:
        import numpy as np

        points = (5e-324, 1e-310, 2.2250738585072014e-308, 1.5e-8, 2.0, 9e5)
        compared = 0
        for shape, scale in ((3.0, 1e-300), (30.0, 2.5e-310), (1000.0, 1e-8), (1000.0, 1.0)):
            for point in points:
                result = evaluate_log_density(FamilyId.WEIBULL_MIN, point, shape=shape, scale=scale)
                if not isinstance(result, LogDensitySuccess):
                    continue
                vector = logpdf("weibull_min", np.array([point]), shape=shape, scale=scale)
                compared += 1
                with self.subTest(shape=shape, scale=scale, point=point):
                    self.assertTrue(
                        math.isclose(float(vector[0]), result.log_density, rel_tol=1e-13)
                    )
        self.assertGreater(compared, 0)


class GammaPrefactorTests(unittest.TestCase):
    def test_below_the_threshold_the_prefactor_is_the_direct_expression(self) -> None:
        for shape, value in ((3.0, 2.0), (300.0, 310.0), (511.0, 40.0), (0.2, 1e-3)):
            log_value = math.log(value)
            with self.subTest(shape=shape, value=value):
                self.assertEqual(
                    _log_gamma_prefactor(shape, value, log_value),
                    -value + shape * log_value - math.lgamma(shape),
                )
                self.assertEqual(
                    math.fsum(_log_gamma_prefactor_terms(shape, value, log_value)),
                    math.fsum((-value, shape * log_value, -math.lgamma(shape))),
                )

    def test_at_and_above_the_threshold_it_matches_mpmath(self) -> None:
        for shape in (_STABLE_PREFACTOR_SHAPE, 1000.0, 20000.0):
            for ratio in (1e-3, 0.3, 0.6, 0.999, 1.0, 1.001, 1.6, 4.0):
                value = shape * ratio
                with self.subTest(shape=shape, ratio=ratio), mpmath.workdps(60):
                    x, k = mpmath.mpf(value), mpmath.mpf(shape)
                    expected = float(-x + k * mpmath.log(x) - mpmath.loggamma(k))
                    computed = _log_gamma_prefactor(shape, value, math.log(value))
                    self.assertLessEqual(abs(computed - expected), 1e-14 * max(1.0, abs(expected)))

    def test_the_threshold_is_continuous(self) -> None:
        below = math.nextafter(_STABLE_PREFACTOR_SHAPE, 0.0)
        for ratio in (0.2, 0.9, 1.0, 1.4, 3.0):
            value = _STABLE_PREFACTOR_SHAPE * ratio
            log_value = math.log(value)
            with self.subTest(ratio=ratio):
                direct = _log_gamma_prefactor(below, value, log_value)
                stable = _log_gamma_prefactor(_STABLE_PREFACTOR_SHAPE, value, log_value)
                self.assertAlmostEqual(direct, stable, delta=2e-12 + 1e-13 * abs(direct))

    def test_the_log_survival_uses_the_same_prefactor(self) -> None:
        # log Q(1000, 1300) = log(1.8736155715785551242e-18), from mpmath at 60 digits.
        self.assertAlmostEqual(
            _log_regularized_gamma_q(1000.0, 1300.0, math.log(1300.0)),
            math.log(1.8736155715785551242e-18),
            delta=5e-13,
        )


class UnchangedValueTests(unittest.TestCase):
    def test_values_that_were_already_accurate_are_unchanged(self) -> None:
        for shape, scale, observation, expected in UNCHANGED_VALUES:
            with self.subTest(shape=shape, scale=scale, observation=observation):
                result = evaluate_log_density(FamilyId.GAMMA, observation, shape=shape, scale=scale)
                assert isinstance(result, LogDensitySuccess)
                self.assertTrue(math.isclose(result.log_density, expected, rel_tol=2e-15))


if __name__ == "__main__":
    unittest.main()
