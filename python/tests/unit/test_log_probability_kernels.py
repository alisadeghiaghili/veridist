"""Exactness pins for the log-probability kernels: helpers, constants, routing and the switch.

Expected values are mpmath results at 50 digits (rounded to binary64) or closed forms written
here independently of the implementation.  Accuracy sweeps live in ``tests/reference`` and
``tests/statistical``; this module pins the pieces they cannot see: the constants, the branch
points, the form each window is computed from, and the iteration counts of the gamma loops.
"""

from __future__ import annotations

import math
import unittest
from typing import Any
from unittest import mock

import numpy as np

from veridist.statistics import _loggamma as gamma_kernel
from veridist.statistics import _logmath as logmath
from veridist.statistics import _logprob as logprob
from veridist.statistics._logerrors import LogProbabilityError, LogProbabilityErrorCode


def _array(*values: float) -> Any:
    return np.array(values, dtype=np.float64)


def _relative(computed: float, expected: float) -> float:
    return abs(computed - expected) / abs(expected)


#: ``(d, log(1 - exp(d)))`` from mpmath at 50 digits.
LOG1MEXP = (
    (-1e-300, -690.7755278982137),
    (-1e-17, -39.14394658089878),
    (-1e-05, -11.512930464966061),
    (-0.3, -1.3502256128148467),
    (-0.5, -0.9327521295671886),
    (-0.6931471805599453, -0.6931471805599453),
    (-0.7, -0.6863410028083852),
    (-1.0, -0.4586751453870819),
    (-5.0, -0.006760749449488557),
    (-36.7, -1.1518409493076097e-16),
    (-50.0, -1.9287498479639178e-22),
    (-700.0, -9.85967654375977e-305),
)

#: ``(delta, log(1 + delta), log1p(delta) - delta)`` from mpmath at 50 digits.
LOG1PMX = (
    (1e-09, 1.000000082240371e-09, -4.999999996666667e-19),
    (-0.0001, -0.00010000500033334732, -5.000333358335334e-09),
    (0.02, 0.01980262729617973, -0.00019737270382028698),
    (0.03, 0.02955880224154443, -0.00044119775845559726),
    (-0.03, -0.030459207484708574, -0.00045920748470854587),
    (0.04, 0.03922071315328133, -0.0007792868467187037),
    (-0.04, -0.040821994520255166, -0.0008219945202551296),
    (0.5, 0.4054651081081644, -0.09453489189183562),
    (3.0, 1.3862943611198906, -1.6137056388801094),
    (-0.4, -0.5108256237659907, -0.1108256237659907),
    (-0.5, -0.6931471805599453, -0.19314718055994531),
    (-0.75, -1.3862943611198906, -0.6362943611198906),
    (1e10, 23.025850930040455, -9999999976.97415),
)

#: ``(shape, y, sum, iterations)``: ``sum_n y**n / (shape (shape + 1) ... (shape + n))``.
SERIES = (
    (0.003, 0.5, 548.8443070543457, 15),
    (1.0, 0.9, 1.621781234618833, 17),
    (8.0, 8.5, 0.5585158244764084, 35),
    (100.0, 99.0, 0.11933789727975501, 91),
    (1000.0, 1000.5, 0.04047485042306586, 266),
    (3900.0, 3900.9999999999995, 0.020413971755334406, 509),
    (50.0, 0.0, 0.02, 1),
)

#: ``(shape, y, h, iterations)`` with ``Q(shape, y) = exp(-y + shape log y - lgamma(shape)) h``.
FRACTION = (
    (0.003, 1.5, 0.44872872635386624, 66),
    (1.0, 3.0, 0.3333333333333333, 1),
    (8.0, 9.0, 0.30728946811070695, 8),
    (100.0, 101.0, 0.11272421482644741, 39),
    (1000.0, 1100.0, 0.009139399538168738, 33),
    (3900.0, 3901.0, 0.01973020717659542, 145),
    (2.5, 100000.0, 1.0000150000749996e-05, 2),
)

#: ``(shape, y, log Q(shape, y))`` for shapes below 0.01 and ``y`` up to ``shape + 1``.
SMALL_SHAPE_UPPER = (
    (0.003, 1e-300, -0.13480053976259845),
    (0.003, 1e-05, -3.4332456518532677),
    (0.003, 0.1, -5.2102410379286255),
    (0.003, 0.5, -6.387799889057365),
    (0.003, 0.999, -7.321337253822239),
    (0.003, 1.0, -7.323012155871869),
    (0.003, 1.0029, -7.327866322333725),
    (0.0099, 1e-300, -0.0010781545104826301),
    (0.0099, 1e-05, -2.276128196516442),
    (0.0099, 0.1, -4.01989229547973),
    (0.0099, 0.5, -5.190323521638653),
    (0.0099, 0.999, -6.120427050711869),
    (0.0099, 1.0, -6.122096795639239),
    (0.0099, 1.0089, -6.136934051401488),
)


class Log1mexpTests(unittest.TestCase):
    def test_matches_mpmath_to_relative_precision_on_both_branches(self) -> None:
        for value, expected in LOG1MEXP:
            with self.subTest(value=value):
                computed = float(logmath.log1mexp(_array(value))[0])
                self.assertLessEqual(_relative(computed, expected), 4e-16)

    def test_the_points_at_the_ends_of_the_range(self) -> None:
        computed = logmath.log1mexp(_array(0.0, -math.inf, 1e-3))
        self.assertEqual(computed[0], -math.inf)
        self.assertEqual(computed[1], 0.0)
        self.assertTrue(math.isnan(computed[2]))

    def test_shape_and_dtype_are_preserved(self) -> None:
        computed = logmath.log1mexp(np.full((2, 3), -1.0))
        self.assertEqual(computed.shape, (2, 3))
        self.assertEqual(computed.dtype, np.float64)


class LogRatioTests(unittest.TestCase):
    def test_log_of_the_ratio_near_the_scale_and_far_from_it(self) -> None:
        scale = 3.0
        values = (1.5, 1.4999999999999998, 4.4999999999999996, 4.500000000000001, 9.0, 0.01, 3.0)
        computed = logmath.log_ratio(_array(*values), scale)
        for value, result in zip(values, computed, strict=True):
            with self.subTest(value=value):
                if value == scale:
                    self.assertEqual(result, 0.0)
                else:
                    self.assertLessEqual(_relative(float(result), math.log(value / scale)), 2e-15)

    def test_a_value_near_the_scale_keeps_every_digit_of_the_log1p(self) -> None:
        scale = 3.0
        value = scale * (1.0 + 2.0**-40)
        expected = math.log1p((value - scale) / scale)
        self.assertEqual(float(logmath.log_ratio(_array(value), scale)[0]), expected)

    def test_subnormal_and_extreme_arguments(self) -> None:
        computed = logmath.log_ratio(_array(5e-324, 1e-320, 1e308, 0.0, math.inf), 1.0)
        self.assertLessEqual(_relative(float(computed[0]), -744.4400719213812), 2e-16)
        self.assertLessEqual(_relative(float(computed[1]), math.log(1e-320)), 5e-14)
        self.assertLessEqual(_relative(float(computed[2]), math.log(1e308)), 2e-16)
        self.assertEqual(computed[3], -math.inf)
        self.assertEqual(computed[4], math.inf)

    def test_the_near_scale_window_is_pinned(self) -> None:
        self.assertEqual(logmath._NEAR_SCALE, 0.5)

    def test_a_subnormal_scale_and_argument_give_the_exact_binary_exponent_difference(self) -> None:
        computed = logmath.log_ratio(_array(math.ldexp(1.0, -1070)), math.ldexp(1.0, -1060))
        self.assertEqual(float(computed[0]), -10.0 * math.log(2.0))


class ConstantsTests(unittest.TestCase):
    def test_the_published_constants(self) -> None:
        self.assertEqual(logprob.NARROW_SWITCH, 0.02)
        self.assertEqual(len(logprob.QUADRATURE_NODES), 4)
        self.assertEqual(len(logprob.QUADRATURE_WEIGHTS), 4)
        self.assertEqual(gamma_kernel.GAMMA_MAX_ITERATIONS, 512)
        self.assertEqual(gamma_kernel.GAMMA_MAX_SHAPE, 3900.0)
        self.assertEqual(gamma_kernel.SADDLE_POINT_SHAPE, 8.0)
        self.assertEqual(gamma_kernel.SMALL_SHAPE, 0.01)
        self.assertEqual(gamma_kernel._NEAR_ONE, 0.03)
        self.assertEqual(gamma_kernel._NEAR_ONE_LAST, 15)
        self.assertEqual(gamma_kernel._EPSILON, 2e-16)
        self.assertEqual(gamma_kernel._FPMIN, 1e-300)
        self.assertEqual(logprob._TINY_LOG_HAZARD, -40.0)
        self.assertEqual(logprob._GUMBEL_LINEAR_FROM, 40.0)
        self.assertEqual(logprob._NORMAL_PRODUCT, 1e-290)
        self.assertEqual(logprob._ERFC_FLOOR, 1e-300)

    def test_the_quadrature_rule_is_the_8_point_gauss_legendre_rule(self) -> None:
        nodes, weights = np.polynomial.legendre.leggauss(8)
        positive = nodes > 0
        np.testing.assert_allclose(logprob.QUADRATURE_NODES, nodes[positive][::1], rtol=1e-15)
        np.testing.assert_allclose(logprob.QUADRATURE_WEIGHTS, weights[positive][::1], rtol=1e-14)
        total = 2.0 * math.fsum(logprob.QUADRATURE_WEIGHTS)
        self.assertAlmostEqual(total, 2.0, places=15)
        # it integrates every polynomial of degree 15 exactly: x**14 on [-1, 1] is 2 / 15
        moment = math.fsum(
            2.0 * weight * node**14
            for node, weight in zip(
                logprob.QUADRATURE_NODES, logprob.QUADRATURE_WEIGHTS, strict=True
            )
        )
        self.assertAlmostEqual(moment, 2.0 / 15.0, places=15)

    def test_the_tiny_hazard_matches_its_logarithm(self) -> None:
        self.assertEqual(logprob._TINY_HAZARD, math.exp(logprob._TINY_LOG_HAZARD))


class ShapeRegionTests(unittest.TestCase):
    def test_every_region_is_named_with_its_bounds(self) -> None:
        cases = (
            (0.003, "shape below 0.01"),
            (0.009999999999999998, "shape below 0.01"),
            (0.01, "shape 0.01 to 8"),
            (7.999999999999999, "shape 0.01 to 8"),
            (8.0, "shape 8 to 1000"),
            (999.9999999999999, "shape 8 to 1000"),
            (1000.0, "shape 1000 to 3900"),
            (3900.0, "shape 1000 to 3900"),
            (3900.000000000001, "shape above 3900"),
            (1e6, "shape above 3900"),
        )
        for shape, expected in cases:
            with self.subTest(shape=shape):
                self.assertEqual(gamma_kernel.shape_region(shape), expected)


class LogOnePlusMinusTests(unittest.TestCase):
    def test_matches_mpmath_in_the_series_the_log1p_and_the_log_branches(self) -> None:
        for delta, log_ratio, expected in LOG1PMX:
            with self.subTest(delta=delta):
                computed = float(gamma_kernel._log1pmx(_array(delta), _array(log_ratio))[0])
                self.assertLessEqual(_relative(computed, expected), 5e-15)

    def test_a_ratio_far_below_one_uses_the_logarithm_not_the_rounded_difference(self) -> None:
        # ratio 1e-300: delta rounds to -1 exactly, the logarithm carries the information
        computed = float(gamma_kernel._log1pmx(_array(-1.0), _array(math.log(1e-300)))[0])
        self.assertLessEqual(_relative(computed, -689.7755278982137), 1e-15)
        computed = float(gamma_kernel._log1pmx(_array(-0.75), _array(math.log(0.25)))[0])
        self.assertLessEqual(_relative(computed, -0.6362943611198906), 1e-15)

    def test_the_boundary_between_series_and_log1p_is_continuous(self) -> None:
        for delta in (0.03, -0.03):
            below = math.nextafter(delta, 0.0)
            above = math.nextafter(delta, 10.0 * delta)
            values = [
                float(gamma_kernel._log1pmx(_array(point), _array(math.log1p(point)))[0])
                for point in (below, delta, above)
            ]
            self.assertLessEqual(abs(values[0] - values[1]), 1e-17)
            self.assertLessEqual(abs(values[2] - values[1]), 1e-17)

    def test_the_series_is_exact_to_the_last_digit_near_zero(self) -> None:
        delta = 1e-9
        expected = -(delta**2) / 2 + delta**3 / 3
        computed = float(gamma_kernel._log1pmx(_array(delta), _array(math.log1p(delta)))[0])
        self.assertLessEqual(_relative(computed, expected), 1e-15)


class SumThreeTests(unittest.TestCase):
    def test_a_cancelling_sum_keeps_its_value(self) -> None:
        computed = gamma_kernel._sum_three(_array(1e16), _array(1.0), _array(-1e16))
        self.assertEqual(float(computed[0]), 1.0)
        computed = gamma_kernel._sum_three(_array(0.1), _array(0.2), _array(0.3))
        self.assertEqual(float(computed[0]), math.fsum((0.1, 0.2, 0.3)))
        computed = gamma_kernel._sum_three(_array(-1.0), _array(1e-30), _array(1.0))
        self.assertEqual(float(computed[0]), 1e-30)
        computed = gamma_kernel._sum_three(_array(1e-30), _array(-1.0), _array(1.0))
        self.assertEqual(float(computed[0]), 1e-30)


class SeriesAndFractionTests(unittest.TestCase):
    def test_series_sums_and_iteration_counts(self) -> None:
        for shape, y, expected, iterations in SERIES:
            with self.subTest(shape=shape, y=y):
                total, count = gamma_kernel._series_sum(shape, _array(y), 512)
                self.assertLessEqual(_relative(float(total[0]), expected), 5e-14)
                self.assertEqual(count, iterations)

    def test_fraction_values_and_iteration_counts(self) -> None:
        for shape, y, expected, iterations in FRACTION:
            with self.subTest(shape=shape, y=y):
                value, count = gamma_kernel._fraction_value(shape, _array(y), 512)
                self.assertLessEqual(_relative(float(value[0]), expected), 5e-14)
                self.assertEqual(count, iterations)

    def test_the_count_is_that_of_the_slowest_lane_and_lanes_do_not_interfere(self) -> None:
        shape = 100.0
        grid = _array(1e-300, 0.5, 30.0, 99.0, 100.5, 3.0, 99.9999)
        together, count = gamma_kernel._series_sum(shape, grid, 512)
        for index, point in enumerate(grid):
            alone, alone_count = gamma_kernel._series_sum(shape, _array(point), 512)
            self.assertEqual(float(together[index]), float(alone[0]))
            self.assertLessEqual(alone_count, count)
        self.assertEqual(
            count, max(gamma_kernel._series_sum(shape, grid[i : i + 1], 512)[1] for i in range(7))
        )
        grid = _array(100.5, 101.0, 130.0, 400.0, 1e4, 150.0)
        together, count = gamma_kernel._fraction_value(shape, grid, 512)
        for index, point in enumerate(grid):
            alone, alone_count = gamma_kernel._fraction_value(shape, _array(point), 512)
            self.assertEqual(float(together[index]), float(alone[0]))
            self.assertLessEqual(alone_count, count)
        self.assertEqual(
            count,
            max(gamma_kernel._fraction_value(shape, grid[i : i + 1], 512)[1] for i in range(6)),
        )

    def test_the_series_cap_is_exactly_512_iterations(self) -> None:
        converges = math.nextafter(3951.0, 0.0)
        total, count = gamma_kernel._series_sum(3950.0, _array(converges), 512)
        self.assertEqual(count, 512)
        self.assertTrue(math.isfinite(float(total[0])))
        with self.assertRaises(LogProbabilityError) as raised:
            gamma_kernel._series_sum(3960.0, _array(math.nextafter(3961.0, 0.0)), 512)
        self.assertIs(raised.exception.code, LogProbabilityErrorCode.GAMMA_NONCONVERGENCE)
        self.assertEqual(raised.exception.region, "shape above 3900")
        # 513 iterations is what the same lane needs
        _, count = gamma_kernel._series_sum(3960.0, _array(math.nextafter(3961.0, 0.0)), 513)
        self.assertEqual(count, 513)

    def test_the_fraction_cap_is_exactly_512_iterations(self) -> None:
        # shapes whose fraction at y = shape + 1 needs exactly 512 and exactly 513 iterations
        converges, fails = 131773.0, 145479.0
        value, count = gamma_kernel._fraction_value(converges, _array(converges + 1.0), 512)
        self.assertEqual(count, 512)
        self.assertTrue(math.isfinite(float(value[0])))
        with self.assertRaises(LogProbabilityError) as raised:
            gamma_kernel._fraction_value(fails, _array(fails + 1.0), 512)
        self.assertIs(raised.exception.code, LogProbabilityErrorCode.GAMMA_NONCONVERGENCE)
        _, count = gamma_kernel._fraction_value(fails, _array(fails + 1.0), 513)
        self.assertEqual(count, 513)

    def test_a_zero_lane_count_is_an_empty_result(self) -> None:
        empty = np.empty(0)
        self.assertEqual(gamma_kernel._series_sum(10.0, empty, 512)[0].shape, (0,))
        self.assertEqual(gamma_kernel._series_sum(10.0, empty, 512)[1], 0)
        self.assertEqual(gamma_kernel._fraction_value(10.0, empty, 512)[0].shape, (0,))
        self.assertEqual(gamma_kernel._fraction_value(10.0, empty, 512)[1], 0)

    def test_the_fraction_survives_a_zero_denominator(self) -> None:
        # y + 1 - shape = 0 would be a zero first denominator; the series/fraction split avoids it
        # (the fraction starts at y >= shape + 1), and the clamps keep a vanishing d or c finite
        value, count = gamma_kernel._fraction_value(2.0, _array(3.0), 512)
        self.assertTrue(math.isfinite(float(value[0])))
        self.assertGreater(count, 0)


class SmallShapeTests(unittest.TestCase):
    def test_log_upper_ratio_matches_mpmath(self) -> None:
        for shape, y, expected in SMALL_SHAPE_UPPER:
            with self.subTest(shape=shape, y=y):
                computed = float(
                    gamma_kernel._log_upper_small_shape(shape, _array(y), _array(math.log(y)))[0]
                )
                self.assertLessEqual(abs(computed - expected) / max(1.0, abs(expected)), 1e-14)

    def test_the_small_shape_form_is_used_below_the_threshold_only(self) -> None:
        below, above = 0.0099999999999999, 0.01
        y = _array(0.5)
        log_y = np.log(y)
        lower_below, upper_below = gamma_kernel.log_regularized_gamma(below, y, log_y)
        lower_above, upper_above = gamma_kernel.log_regularized_gamma(above, y, log_y)
        direct = gamma_kernel._log_upper_small_shape(below, y, log_y)
        self.assertEqual(float(upper_below[0]), float(direct[0]))
        self.assertEqual(float(upper_above[0]), float(logmath.log1mexp(lower_above)[0]))
        self.assertLessEqual(abs(float(upper_below[0]) - float(upper_above[0])), 1e-12)
        self.assertLessEqual(abs(float(lower_below[0]) - float(lower_above[0])), 1e-12)


class PrefactorTests(unittest.TestCase):
    def test_both_forms_agree_across_the_saddle_point_shape(self) -> None:
        below, above = math.nextafter(8.0, 0.0), 8.0
        for y in (0.5, 3.0, 8.0, 12.0, 30.0):
            y_array = _array(y)
            first = float(gamma_kernel.log_prefactor(below, y_array, np.log(y_array))[0])
            second = float(gamma_kernel.log_prefactor(above, y_array, np.log(y_array))[0])
            direct = above * math.log(y) - y - math.lgamma(above)
            self.assertLessEqual(abs(second - direct), 3e-14)
            self.assertLessEqual(abs(first - second), 1e-12)

    def test_the_direct_form_below_the_saddle_point_shape(self) -> None:
        for shape, y in ((0.5, 0.25), (3.0, 7.0), (7.9, 1e-5)):
            y_array = _array(y)
            computed = float(gamma_kernel.log_prefactor(shape, y_array, np.log(y_array))[0])
            self.assertEqual(computed, shape * math.log(y) - y - math.lgamma(shape))

    def test_an_underflowed_argument_keeps_its_logarithm(self) -> None:
        for shape in (3.0, 100.0):
            log_y = -800.0
            computed = float(gamma_kernel.log_prefactor(shape, _array(0.0), _array(log_y))[0])
            self.assertLessEqual(_relative(computed, shape * log_y - math.lgamma(shape)), 1e-15)


class GammaLaneTests(unittest.TestCase):
    def test_zero_and_infinite_points(self) -> None:
        y = _array(0.0, math.inf, math.inf, 0.0)
        log_y = _array(-math.inf, math.inf, 710.0, -800.0)
        lower, upper = gamma_kernel.log_regularized_gamma(3.0, y, log_y)
        self.assertEqual(float(lower[0]), -math.inf)
        self.assertEqual(float(upper[0]), 0.0)
        self.assertEqual(float(lower[1]), 0.0)
        self.assertEqual(float(upper[1]), -math.inf)
        self.assertEqual(float(lower[2]), 0.0)
        self.assertEqual(float(upper[2]), -math.inf)
        self.assertLessEqual(_relative(float(lower[3]), 3.0 * -800.0 - math.lgamma(4.0)), 1e-15)
        self.assertLessEqual(abs(float(upper[3])), 1e-300)

    def test_the_series_and_fraction_split_at_shape_plus_one(self) -> None:
        shape = 7.0
        below, above = math.nextafter(8.0, 0.0), 8.0
        y = _array(below, above)
        lower, upper = gamma_kernel.log_regularized_gamma(shape, y, np.log(y))
        # below the split the series gives the lower ratio and the upper one is derived from it
        self.assertEqual(float(upper[0]), float(logmath.log1mexp(lower[:1])[0]))
        # at the split the fraction gives the upper ratio and the lower one is derived from it
        self.assertEqual(float(lower[1]), float(logmath.log1mexp(upper[1:])[0]))
        self.assertLessEqual(abs(float(lower[0]) - float(lower[1])), 1e-14)

    def test_the_shape_limit_is_refused_before_any_work(self) -> None:
        y = _array(1.0)
        gamma_kernel.log_regularized_gamma(3900.0, y, np.log(y))
        with self.assertRaises(LogProbabilityError) as raised:
            gamma_kernel.log_regularized_gamma(math.nextafter(3900.0, 4000.0), y, np.log(y))
        self.assertIs(raised.exception.code, LogProbabilityErrorCode.GAMMA_SHAPE_REFUSED)
        self.assertEqual(raised.exception.region, "shape above 3900")
        self.assertEqual(str(raised.exception), "gamma_shape_refused (shape above 3900)")

    def test_empty_input_gives_empty_output(self) -> None:
        empty = np.empty(0)
        lower, upper = gamma_kernel.log_regularized_gamma(3.0, empty, empty)
        self.assertEqual((lower.shape, upper.shape), ((0,), (0,)))


class HazardKernelTests(unittest.TestCase):
    def test_the_survival_is_the_negative_cumulative_hazard_exactly(self) -> None:
        x = _array(0.5, 1.0, 2.5, 40.0)
        np.testing.assert_array_equal(logprob.log_sf("exponential", x, rate=0.25), -0.25 * x)
        shape, scale = 1.7, 3.0
        expected = -np.exp(shape * np.log(x / scale))
        np.testing.assert_allclose(
            logprob.log_sf("weibull_min", x, shape=shape, scale=scale), expected, rtol=1e-14
        )

    def test_the_cdf_in_the_upper_tail_has_full_relative_precision(self) -> None:
        # log F = log(1 - exp(-x)) = -u - u**2 / 2 with u = exp(-x)
        x = _array(30.0, 37.0, 40.0, 100.0)
        computed = logprob.log_cdf("exponential", x, rate=1.0)
        expected = -np.exp(-x) - np.exp(-2.0 * x) / 2.0
        np.testing.assert_allclose(computed, expected, rtol=3e-16)

    def test_the_cdf_in_the_lower_tail_is_the_log_of_the_hazard(self) -> None:
        x = _array(1e-20, 1e-300, 1e-310, 5e-324)
        computed = logprob.log_cdf("exponential", x, rate=2.0)
        for value, point in zip(computed, x, strict=True):
            self.assertLessEqual(
                _relative(float(value), math.log(2.0) + math.log(float(point))), 1e-15
            )
        weibull = logprob.log_cdf("weibull_min", x, shape=2.0, scale=3.0)
        for value, point in zip(weibull, x, strict=True):
            expected = 2.0 * (math.log(float(point)) - math.log(3.0))
            self.assertLessEqual(_relative(float(value), expected), 1e-14)

    def test_the_tiny_hazard_switch_is_continuous(self) -> None:
        threshold = math.exp(-40.0)
        x = _array(math.nextafter(threshold, 0.0), threshold, math.nextafter(threshold, 1.0))
        for computed in (
            logprob.log_cdf("exponential", x, rate=1.0),
            logprob.log_cdf("weibull_min", x, shape=1.0, scale=1.0),
        ):
            self.assertLessEqual(abs(float(computed[0]) - float(computed[2])), 1e-14)
            self.assertLessEqual(abs(float(computed[0]) + 40.0), 1e-14)

    def test_outside_the_support_and_at_infinity(self) -> None:
        x = _array(-3.0, 0.0, math.inf)
        for family, parameters in (
            ("exponential", {"rate": 0.5}),
            ("weibull_min", {"shape": 2.0, "scale": 3.0}),
            ("gamma", {"shape": 2.0, "scale": 3.0}),
            ("lognormal", {"mu_log": 0.0, "sigma_log": 1.0}),
        ):
            with self.subTest(family=family):
                cdf = logprob.log_cdf(family, x, **parameters)
                sf = logprob.log_sf(family, x, **parameters)
                self.assertEqual(list(cdf), [-math.inf, -math.inf, 0.0])
                self.assertEqual(list(sf), [0.0, 0.0, -math.inf])
                self.assertEqual(math.copysign(1.0, float(sf[0])), 1.0)
                self.assertEqual(math.copysign(1.0, float(sf[1])), 1.0)

    def test_a_hazard_beyond_the_range_is_minus_infinity(self) -> None:
        sf = logprob.log_sf("weibull_min", _array(1e300), shape=50.0, scale=1.0)
        self.assertEqual(float(sf[0]), -math.inf)
        cdf = logprob.log_cdf("weibull_min", _array(1e300), shape=50.0, scale=1.0)
        self.assertEqual(float(cdf[0]), 0.0)

    def test_the_exponential_product_threshold_selects_a_subnormal_safe_log(self) -> None:
        # rate * x = 1e-324 * ... is not representable: the logarithms of the factors are added
        computed = logprob.log_cdf("exponential", _array(1e-310), rate=1e-300)
        self.assertLessEqual(
            _relative(float(computed[0]), math.log(1e-300) + math.log(1e-310)), 1e-15
        )
        near = _array(1e-290 / 1e3, 1e-290 / 1e3 * (1.0 + 1e-15))
        values = logprob.log_cdf("exponential", near, rate=1e3)
        self.assertLessEqual(abs(float(values[0]) - math.log(1e-290)), 1e-12)


class GumbelKernelTests(unittest.TestCase):
    def test_the_cdf_is_minus_exp_of_minus_z_exactly(self) -> None:
        x = _array(-5.0, 0.0, 3.0, 30.0)
        computed = logprob.log_cdf("gumbel_right", x, location=1.0, scale=2.0)
        np.testing.assert_array_equal(computed, -np.exp(-(x - 1.0) / 2.0))

    def test_the_survival_is_the_linear_form_from_z_40(self) -> None:
        z = _array(39.99999999999999, 40.0, 40.00000000000001, 700.0, 1e5)
        computed = logprob.log_sf("gumbel_right", z, location=0.0, scale=1.0)
        self.assertEqual(float(computed[1]), -40.0)
        self.assertEqual(float(computed[3]), -700.0)
        self.assertEqual(float(computed[4]), -1e5)
        self.assertLessEqual(abs(float(computed[0]) + 40.0), 1e-14)
        self.assertLessEqual(abs(float(computed[2]) + 40.0), 1e-14)
        below = logprob.log_sf("gumbel_right", _array(5.0), location=0.0, scale=1.0)
        self.assertLessEqual(
            _relative(float(below[0]), math.log(-math.expm1(-math.exp(-5.0)))), 1e-15
        )

    def test_the_far_lower_tail_and_the_infinities(self) -> None:
        z = _array(-800.0, -math.inf, math.inf)
        cdf = logprob.log_cdf("gumbel_right", z, location=0.0, scale=1.0)
        sf = logprob.log_sf("gumbel_right", z, location=0.0, scale=1.0)
        self.assertEqual(list(cdf), [-math.inf, -math.inf, 0.0])
        self.assertEqual(list(sf), [0.0, 0.0, -math.inf])


class NormalKernelTests(unittest.TestCase):
    def test_the_two_tails_are_reflections_bit_for_bit(self) -> None:
        z = np.linspace(-9.0, 9.0, 37)
        cdf = logprob.log_cdf("normal", z, mu=0.0, sigma=1.0)
        sf = logprob.log_sf("normal", -z, mu=0.0, sigma=1.0)
        np.testing.assert_array_equal(cdf, sf)

    def test_the_median_is_one_half(self) -> None:
        expected = math.log(0.5)
        for family, parameters in (
            ("normal", {"mu": 3.0, "sigma": 2.0}),
            ("lognormal", {"mu_log": 0.5, "sigma_log": 2.0}),
        ):
            point = _array(3.0 if family == "normal" else math.exp(0.5))
            self.assertLessEqual(
                abs(float(logprob.log_cdf(family, point, **parameters)[0]) - expected), 1e-15
            )
            self.assertLessEqual(
                abs(float(logprob.log_sf(family, point, **parameters)[0]) - expected), 1e-15
            )

    def test_the_lognormal_position_is_the_standardised_log(self) -> None:
        x = _array(0.5, 2.0, 40.0)
        direct = logprob.log_cdf("lognormal", x, mu_log=1.0, sigma_log=0.7)
        z = (np.log(x) - 1.0) / 0.7
        np.testing.assert_array_equal(direct, logprob.log_phi(z))


class MedianAndFlagTests(unittest.TestCase):
    """The side a window is computed from follows the median; the integration coordinate follows
    the support."""

    def _kernel(self, family: str, **parameters: float) -> Any:
        return logprob._kernel(family, parameters)

    def test_closed_form_medians(self) -> None:
        self.assertEqual(self._kernel("exponential", rate=4.0).median(), math.log(2.0) / 4.0)
        self.assertEqual(self._kernel("normal", mu=3.0, sigma=2.0).median(), 3.0)
        weibull = self._kernel("weibull_min", shape=2.0, scale=5.0).median()
        self.assertAlmostEqual(weibull, 5.0 * math.sqrt(math.log(2.0)), places=14)
        self.assertNotEqual(weibull, 5.0 * math.log(2.0) * 2.0)
        gumbel = self._kernel("gumbel_right", location=1.0, scale=2.0).median()
        self.assertAlmostEqual(gumbel, 1.0 - 2.0 * math.log(math.log(2.0)), places=14)
        self.assertEqual(
            self._kernel("lognormal", mu_log=0.5, sigma_log=1.0).median(), math.exp(0.5)
        )

    def test_the_lognormal_median_saturates_at_infinity(self) -> None:
        self.assertEqual(self._kernel("lognormal", mu_log=709.9, sigma_log=1.0).median(), math.inf)
        self.assertEqual(
            self._kernel("lognormal", mu_log=700.0, sigma_log=1.0).median(), math.exp(700.0)
        )

    def test_gamma_median_is_close_to_the_true_median(self) -> None:
        # mpmath: 2.1757300955477636, 9.668714614714132, 99.66686491931549 (scale 1)
        for shape, true in (
            (2.5, 2.1757300955477636),
            (10.0, 9.668714614714132),
            (100.0, 99.66686491931549),
        ):
            median = self._kernel("gamma", shape=shape, scale=2.0).median()
            self.assertLessEqual(abs(median / (2.0 * true) - 1.0), 0.01)
            self.assertEqual(median, 2.0 * shape * (1.0 - 1.0 / (9.0 * shape)) ** 3)
        small = self._kernel("gamma", shape=0.5, scale=2.0).median()
        self.assertEqual(small, 2.0 * math.exp((math.lgamma(1.5) - math.log(2.0)) / 0.5))
        # the true median 0.2274682115597864 of shape 0.5 lies within a factor of two
        self.assertLessEqual(abs(math.log(small / (2.0 * 0.2274682115597864))), math.log(2.0))
        at_one = self._kernel("gamma", shape=1.0, scale=1.0).median()
        self.assertEqual(at_one, (1.0 - 1.0 / 9.0) ** 3)

    def test_support_flags(self) -> None:
        lifetimes = {
            "exponential": {"rate": 1.0},
            "weibull_min": {"shape": 1.0, "scale": 1.0},
            "lognormal": {"mu_log": 0.0, "sigma_log": 1.0},
            "gamma": {"shape": 1.0, "scale": 1.0},
        }
        for family, parameters in lifetimes.items():
            self.assertIs(self._kernel(family, **parameters).lifetime, True)
        self.assertIs(self._kernel("normal", mu=0.0, sigma=1.0).lifetime, False)
        self.assertIs(self._kernel("gumbel_right", location=0.0, scale=1.0).lifetime, False)

    def test_the_gamma_density_beyond_the_range_is_minus_infinity(self) -> None:
        kernel = self._kernel("gamma", shape=9.0, scale=1.0)
        value = kernel.log_density(_array(800.0, 5.0))
        self.assertEqual(float(value[0]), -math.inf)
        self.assertTrue(math.isfinite(float(value[1])))

    def test_the_series_serves_below_shape_plus_one_and_the_fraction_from_it(self) -> None:
        y = _array(7.999999999999999, 8.0, 12.0)
        series = mock.patch.object(gamma_kernel, "_series_sum", wraps=gamma_kernel._series_sum)
        fraction = mock.patch.object(
            gamma_kernel, "_fraction_value", wraps=gamma_kernel._fraction_value
        )
        with series as series_spy, fraction as fraction_spy:
            gamma_kernel.log_regularized_gamma(7.0, y, np.log(y))
        self.assertEqual(series_spy.call_args.args[1].tolist(), [7.999999999999999])
        self.assertEqual(fraction_spy.call_args.args[1].tolist(), [8.0, 12.0])
        # shape + 1 is 8.5: the argument 8.25 is still a series lane, not a fraction one
        y = _array(8.25, 8.5)
        with series as series_spy, fraction as fraction_spy:
            gamma_kernel.log_regularized_gamma(7.5, y, np.log(y))
        self.assertEqual(series_spy.call_args.args[1].tolist(), [8.25])
        self.assertEqual(fraction_spy.call_args.args[1].tolist(), [8.5])

    def test_the_huge_argument_threshold(self) -> None:
        self.assertEqual(gamma_kernel._HUGE_ARGUMENT, 1e100)
        y = _array(9.999999999999998e99, 1e100, 1e101)
        with mock.patch.object(
            gamma_kernel, "_fraction_value", wraps=gamma_kernel._fraction_value
        ) as fraction_spy:
            lower, upper = gamma_kernel.log_regularized_gamma(2.5, y, np.log(y))
        self.assertEqual(fraction_spy.call_args.args[1].tolist(), [9.999999999999998e99])
        for value, expected in zip(upper, (-9.999999999999998e99, -1e100, -1e101), strict=True):
            self.assertLessEqual(_relative(float(value), expected), 1e-15)


class WindowRoutingTests(unittest.TestCase):
    PARAMETERS = {
        "exponential": {"rate": 0.5},
        "weibull_min": {"shape": 1.7, "scale": 3.0},
        "gumbel_right": {"location": 1.0, "scale": 2.0},
        "normal": {"mu": 1.0, "sigma": 2.0},
        "lognormal": {"mu_log": 0.3, "sigma_log": 0.9},
        "gamma": {"shape": 2.5, "scale": 1.5},
    }

    def test_open_ends_are_the_single_tail_kernels_bit_for_bit(self) -> None:
        points = _array(0.4, 2.0, 7.0)
        for family, parameters in self.PARAMETERS.items():
            with self.subTest(family=family):
                np.testing.assert_array_equal(
                    logprob.log_window(family, -math.inf, points, **parameters),
                    logprob.log_cdf(family, points, **parameters),
                )
                np.testing.assert_array_equal(
                    logprob.log_window(family, points, math.inf, **parameters),
                    logprob.log_sf(family, points, **parameters),
                )
                both = logprob.log_window(family, _array(-math.inf), _array(math.inf), **parameters)
                self.assertEqual(float(both[0]), 0.0)

    def test_the_support_bound_of_the_positive_families_is_an_open_end(self) -> None:
        points = _array(0.4, 2.0)
        for family in ("exponential", "weibull_min", "lognormal", "gamma"):
            parameters = self.PARAMETERS[family]
            reference = logprob.log_cdf(family, points, **parameters)
            for bound in (0.0, -1.0, -math.inf):
                with self.subTest(family=family, bound=bound):
                    np.testing.assert_array_equal(
                        logprob.log_window(family, bound, points, **parameters), reference
                    )
            both = logprob.log_window(
                family, _array(0.0, -2.0), _array(math.inf, math.inf), **parameters
            )
            self.assertEqual(list(both), [0.0, 0.0])
            gone = logprob.log_window(family, _array(-3.0, -1.0), _array(0.0, -0.5), **parameters)
            self.assertEqual(list(gone), [-math.inf, -math.inf])

    def test_location_families_keep_negative_ends(self) -> None:
        for family in ("normal", "gumbel_right"):
            parameters = self.PARAMETERS[family]
            computed = logprob.log_window(family, _array(-3.0), _array(-1.0), **parameters)
            self.assertTrue(math.isfinite(float(computed[0])))
            self.assertLess(float(computed[0]), 0.0)

    def test_a_mixed_batch_equals_its_single_elements(self) -> None:
        lower = _array(-math.inf, 0.2, 0.2, 3.0, 5.0, 0.01, -2.0, 1.0)
        upper = _array(2.0, math.inf, 0.2000001, 9.0, 5.5, 40.0, -1.0, math.inf)
        for family, parameters in self.PARAMETERS.items():
            with self.subTest(family=family):
                together = logprob.log_window(family, lower, upper, **parameters)
                for index in range(8):
                    single = logprob.log_window(
                        family, lower[index : index + 1], upper[index : index + 1], **parameters
                    )
                    self.assertEqual(float(together[index]), float(single[0]))

    def test_the_form_follows_the_side_of_the_window(self) -> None:
        # lower tail: F(upper) <= 1/2, upper tail: S(lower) <= 1/2, across the median otherwise
        for family, parameters in self.PARAMETERS.items():
            with self.subTest(family=family):
                spec = logprob._kernel(family, parameters)
                median = spec.median()
                lo = _array(median * 0.2 if family != "normal" else median - 3.0)
                hi = _array(median * 0.5 if family != "normal" else median - 1.5)
                lower_tail = float(logprob.log_window(family, lo, hi, **parameters)[0])
                at_hi = logprob.log_cdf(family, hi, **parameters)
                at_lo = logprob.log_cdf(family, lo, **parameters)
                self.assertEqual(lower_tail, float((at_hi + logmath.log1mexp(at_lo - at_hi))[0]))
                lo2 = _array(median * 2.0 if family != "normal" else median + 1.5)
                hi2 = _array(median * 4.0 if family != "normal" else median + 3.0)
                upper_tail = float(logprob.log_window(family, lo2, hi2, **parameters)[0])
                s_lo = logprob.log_sf(family, lo2, **parameters)
                s_hi = logprob.log_sf(family, hi2, **parameters)
                self.assertEqual(upper_tail, float((s_lo + logmath.log1mexp(s_hi - s_lo))[0]))
                across = float(logprob.log_window(family, lo, hi2, **parameters)[0])
                tails = np.exp(at_lo) + np.exp(s_hi)
                self.assertEqual(across, float(np.log1p(-tails)[0]))

    def test_a_probability_below_the_smallest_logarithm_is_minus_infinity(self) -> None:
        # both ends so deep in the Gumbel lower tail that log F is -inf
        computed = logprob.log_window(
            "gumbel_right", _array(-800.0), _array(-780.0), location=0.0, scale=1.0
        )
        self.assertEqual(float(computed[0]), -math.inf)
        computed = logprob.log_window(
            "gumbel_right", _array(780.0), _array(800.0), location=0.0, scale=1.0
        )
        self.assertLessEqual(
            abs(float(computed[0]) - (-780.0 + math.log1p(-math.exp(-20.0)))), 1e-12
        )


class SwitchTests(unittest.TestCase):
    """The quadrature and the direct form agree across the switch, and the switch is at 0.02."""

    PARAMETERS = WindowRoutingTests.PARAMETERS

    def _both(self, family: str, lower: float, upper: float) -> tuple[float, float, float]:
        kernel = logprob._kernel(family, self.PARAMETERS[family])
        low, high = _array(lower), _array(upper)
        direct = float(logprob._finite_window(kernel, low, high, switch=0.0)[0])
        quadrature = float(logprob._finite_window(kernel, low, high, switch=2.0)[0])
        default = float(logprob._finite_window(kernel, low, high)[0])
        return direct, quadrature, default

    def test_the_two_forms_agree_on_windows_either_side_of_the_switch(self) -> None:
        cases = {
            "exponential": ((0.05, 0.0504), (0.05, 0.051), (0.05, 0.07), (6.0, 6.002), (6.0, 6.5)),
            "weibull_min": ((0.5, 0.505), (0.5, 0.52), (9.0, 9.01), (9.0, 9.5), (2.0, 2.2)),
            "gumbel_right": ((-4.0, -3.999), (-4.0, -3.95), (9.0, 9.05), (9.0, 10.0), (1.0, 1.01)),
            "normal": ((-7.0, -6.999), (-7.0, -6.99), (8.0, 8.002), (8.0, 8.05), (1.0, 1.01)),
            "lognormal": ((0.05, 0.0501), (0.05, 0.0508), (60.0, 60.05), (60.0, 61.0), (1.3, 1.31)),
            "gamma": ((0.2, 0.2003), (0.2, 0.2041), (9.0, 9.01), (9.0, 9.3), (3.0, 3.04)),
        }
        for family, windows in cases.items():
            for lower, upper in windows:
                with self.subTest(family=family, lower=lower, upper=upper):
                    direct, quadrature, default = self._both(family, lower, upper)
                    scale = max(1.0, abs(direct))
                    self.assertLessEqual(abs(direct - quadrature) / scale, 2e-12)
                    self.assertIn(default, (direct, quadrature))

    def _chooses_quadrature(self, family: str, lower: float, upper: float) -> bool:
        kernel = logprob._kernel(family, self.PARAMETERS[family])
        with mock.patch.object(logprob, "_quadrature", return_value=_array(123.0)) as quadrature:
            value = logprob._finite_window(kernel, _array(lower), _array(upper))
        self.assertEqual(quadrature.called, float(value[0]) == 123.0)
        return bool(quadrature.called)

    def test_the_switch_is_at_a_complement_of_0_02_in_the_lower_tail(self) -> None:
        # exponential, rate 0.5: the complement is 1 - F(lower) / F(upper)
        for complement, expected in ((0.0199, True), (0.0201, False), (0.0, True), (0.9, False)):
            lower = self._lower_for_complement(0.4, complement) if complement else 0.4
            window = (lower, 0.4) if complement else (0.4, 0.4000001)
            self.assertIs(self._chooses_quadrature("exponential", *window), expected)

    def test_the_switch_is_at_a_complement_of_0_02_in_the_upper_tail(self) -> None:
        # exponential, rate 0.5, S(1.4) = 0.5 is the median: complement 1 - S(upper) / S(lower)
        lower = 4.0
        for complement, expected in ((0.0199, True), (0.0201, False)):
            upper = lower - 2.0 * math.log1p(-complement)
            self.assertIs(self._chooses_quadrature("exponential", lower, upper), expected)

    def test_the_switch_is_at_a_probability_of_0_02_across_the_median(self) -> None:
        # normal(1, 2): P = F(1 + h) - F(1 - h) = erf(h / (2 sqrt 2))
        for probability, expected in ((0.0199, True), (0.0201, False)):
            half = 2.0 * math.sqrt(2.0) * self._inverse_erf(probability)
            self.assertIs(self._chooses_quadrature("normal", 1.0 - half, 1.0 + half), expected)

    @staticmethod
    def _inverse_erf(value: float) -> float:
        """``erf`` inverted by bisection to full precision."""

        low, high = 0.0, 1.0
        for _ in range(100):
            middle = 0.5 * (low + high)
            low, high = (middle, high) if math.erf(middle) < value else (low, middle)
        return low

    @staticmethod
    def _lower_for_complement(upper: float, complement: float) -> float:
        """The ``lower`` with ``F(lower) / F(upper) = 1 - complement`` (exponential, rate 0.5)."""

        target = (1.0 - complement) * (-math.expm1(-0.5 * upper))
        return -math.log1p(-target) / 0.5

    def test_the_forms_agree_wherever_the_complement_is_moderate(self) -> None:
        # two positions per family, 28 widths each; the forms must agree where both are valid
        cases = {
            "exponential": ((0.05, "lower"), (6.0, "upper")),
            "weibull_min": ((0.5, "lower"), (9.0, "upper")),
            "gumbel_right": ((-4.0, "lower"), (9.0, "upper")),
            "normal": ((-7.0, "lower"), (8.0, "upper")),
            "lognormal": ((0.05, "lower"), (60.0, "upper")),
            "gamma": ((0.2, "lower"), (9.0, "upper")),
        }
        compared = 0
        for family, positions in cases.items():
            parameters = self.PARAMETERS[family]
            kernel = logprob._kernel(family, parameters)
            for start, side in positions:
                lower = _array(*([start] * 28))
                upper = lower + abs(start) * np.geomspace(1e-4, 1.0, 28)
                if side == "lower":
                    low_cdf = logprob.log_cdf(family, lower, **parameters)
                    complement = -np.expm1(low_cdf - logprob.log_cdf(family, upper, **parameters))
                else:
                    high_sf = logprob.log_sf(family, upper, **parameters)
                    complement = -np.expm1(high_sf - logprob.log_sf(family, lower, **parameters))
                direct = logprob._finite_window(kernel, lower, upper, switch=0.0)
                quadrature = logprob._finite_window(kernel, lower, upper, switch=2.0)
                usable = (complement > 0.005) & (complement < 0.1)
                self.assertTrue(usable.any(), (family, side))
                scale = np.maximum(1.0, np.abs(direct))
                self.assertLessEqual(
                    float((np.abs(direct - quadrature) / scale)[usable].max()), 2e-12
                )
                compared += int(usable.sum())
        self.assertGreater(compared, 100)

    def test_the_quadrature_integrates_exactly_what_the_density_is(self) -> None:
        # exponential in t = log x: integral of rate exp(-rate x) over [x1, x2] by the 8 nodes
        kernel = logprob._kernel("exponential", {"rate": 1.0})
        value = float(logprob._quadrature(kernel, _array(1.0), _array(1.0001))[0])
        self.assertLessEqual(abs(value - math.log(math.exp(-1.0) - math.exp(-1.0001))), 1e-12)

    def test_an_unrepresentable_density_gives_minus_infinity_in_the_quadrature(self) -> None:
        kernel = logprob._kernel("gumbel_right", {"location": 0.0, "scale": 1.0})
        value = logprob._quadrature(kernel, _array(-800.0), _array(-799.999))
        self.assertEqual(float(value[0]), -math.inf)


class InputValidationTests(unittest.TestCase):
    def test_integer_and_python_inputs_are_accepted(self) -> None:
        computed = logprob.log_cdf("normal", [0, 1, 2], mu=0, sigma=1)
        self.assertEqual(computed.dtype, np.float64)
        self.assertEqual(computed.shape, (3,))

    def test_the_result_has_the_broadcast_shape(self) -> None:
        computed = logprob.log_window("normal", np.zeros((2, 3)), np.ones((3,)), mu=0.0, sigma=1.0)
        self.assertEqual(computed.shape, (2, 3))
        scalar = logprob.log_cdf("normal", 0.5, mu=0.0, sigma=1.0)
        self.assertEqual(scalar.shape, ())
        self.assertEqual(
            logprob.log_cdf("normal", np.empty((0, 2)), mu=0.0, sigma=1.0).shape, (0, 2)
        )

    def test_the_input_is_never_modified(self) -> None:
        x = np.array([0.5, 1.5, 2.5])
        copy = x.copy()
        logprob.log_cdf("gamma", x, shape=2.0, scale=1.0)
        logprob.log_window("gamma", x, x + 1.0, shape=2.0, scale=1.0)
        np.testing.assert_array_equal(x, copy)
        low, high = np.array([0.5, 1.5]), np.array([2.0, 3.0])
        logprob.log_window("normal", low, high, mu=0.0, sigma=1.0)
        np.testing.assert_array_equal(low, [0.5, 1.5])
        np.testing.assert_array_equal(high, [2.0, 3.0])

    def test_non_real_inputs_are_a_type_error(self) -> None:
        for bad in (
            np.array(["a"]),
            np.array([1 + 2j]),
            np.array([True]),
            np.array([None], dtype=object),
        ):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(TypeError):
                    logprob.log_cdf("normal", bad, mu=0.0, sigma=1.0)

    def test_the_family_and_parameters_are_validated_like_the_registry(self) -> None:
        with self.assertRaises(TypeError):
            logprob.log_cdf(3, [1.0], mu=0.0, sigma=1.0)
        with self.assertRaises(ValueError):
            logprob.log_cdf("no_such_family", [1.0], mu=0.0, sigma=1.0)
        with self.assertRaises(TypeError):
            logprob.log_cdf("normal", [1.0], mu=0.0)
        with self.assertRaises(ValueError):
            logprob.log_cdf("normal", [1.0], mu=0.0, sigma=-1.0)
        with self.assertRaises(ValueError):
            logprob.log_cdf("gamma", [1.0], shape=math.nan, scale=1.0)


if __name__ == "__main__":
    unittest.main()
