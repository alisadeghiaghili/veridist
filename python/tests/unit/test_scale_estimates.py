"""The estimates of the one-pass families: numerics, degenerate rules and failure codes."""

from __future__ import annotations

import math
import unittest
from fractions import Fraction
from typing import Any
from unittest import mock

import mpmath
import numpy as np

from tests.unit.scale_fit_oracle import (
    exponential_rate,
    gamma_estimates,
    gaussian_estimates,
    lognormal_estimates,
)
from veridist.families.exponential import (
    ExponentialFitFailure,
    ExponentialFitFailureCode,
    ExponentialFitSuccess,
)
from veridist.families.gamma import GammaFitFailure, GammaFitFailureCode, GammaFitSuccess
from veridist.families.lognormal import (
    LognormalFitFailure,
    LognormalFitFailureCode,
    LognormalFitSuccess,
)
from veridist.families.normal import NormalFitFailure, NormalFitFailureCode, NormalFitSuccess
from veridist.scale import _estimates
from veridist.scale._estimates import (
    MAX_SHAPE_BISECTIONS,
    fit_exponential_state,
    fit_gamma_state,
    fit_lognormal_state,
    fit_normal_state,
    gap_unresolved,
    gaussian_log_likelihood,
    log_minus_digamma,
    profile_gap,
    rounded_sqrt,
    solve_shape,
)
from veridist.scale._evidence import StateEvidence
from veridist.scale._one_pass import (
    ExactStatistic,
    ExponentialState,
    GammaState,
    LognormalState,
    NormalState,
)


def folded(state_type: Any, values: list[float] | np.ndarray) -> Any:
    return state_type.empty().update(np.asarray(values, dtype=np.float64))


def relative(value: float, reference: Any) -> float:
    return float(abs((mpmath.mpf(value) - reference) / reference))


class GapUnresolvedTests(unittest.TestCase):
    def test_the_bound_is_two_to_the_minus_49_of_the_magnitude(self) -> None:
        self.assertTrue(gap_unresolved(2.0**-49, 1.0))
        self.assertFalse(gap_unresolved(float(np.nextafter(2.0**-49, 1.0)), 1.0))
        self.assertTrue(gap_unresolved(2.0**-48, 2.0))
        self.assertFalse(gap_unresolved(float(np.nextafter(2.0**-48, 1.0)), 2.0))

    def test_zero_and_negative_gaps_are_unresolved(self) -> None:
        self.assertTrue(gap_unresolved(0.0, 0.0))
        self.assertTrue(gap_unresolved(-1e-3, 5.0))
        self.assertFalse(gap_unresolved(1e-300, 0.0))

    def test_a_clear_gap_is_resolved(self) -> None:
        self.assertFalse(gap_unresolved(0.5, 3.0))


class LogMinusDigammaTests(unittest.TestCase):
    def test_below_ten_matches_the_reference(self) -> None:
        for shape in (1e-3, 0.1, 0.5, 1.0, 2.0, 5.0, 9.0, 9.99, 9.999999):
            reference = mpmath.log(shape) - mpmath.digamma(shape)
            with self.subTest(shape=shape):
                self.assertLess(relative(log_minus_digamma(shape), reference), 1e-14)

    def test_from_ten_on_the_series_is_correct_to_rounding(self) -> None:
        for shape in (10.0, 10.5, 12.0, 15.0, 19.99, 20.0, 100.0, 1e3, 1e5, 1e7, 1e9):
            reference = mpmath.log(shape) - mpmath.digamma(shape)
            with self.subTest(shape=shape):
                self.assertLess(relative(log_minus_digamma(shape), reference), 4e-16)

    def test_the_value_is_positive_and_decreasing(self) -> None:
        previous = math.inf
        for shape in (0.01, 0.1, 1.0, 9.0, 10.0, 11.0, 100.0, 1e4, 1e8):
            value = log_minus_digamma(shape)
            self.assertGreater(value, 0.0)
            self.assertLess(value, previous)
            previous = value


class SolveShapeTests(unittest.TestCase):
    def test_the_root_is_found_across_the_whole_range(self) -> None:
        for shape in (1e-3, 0.05, 0.5, 1.0, 3.0, 9.999, 10.0, 25.0, 300.0, 1e4, 1e6, 4e8):
            gap = float(mpmath.log(shape) - mpmath.digamma(shape))
            with self.subTest(shape=shape):
                solved = solve_shape(gap)
                assert solved is not None
                self.assertLess(abs(solved / shape - 1.0), 2e-14)

    def test_a_root_beyond_the_largest_shape_is_none(self) -> None:
        beyond = float(mpmath.log(mpmath.e**20) - mpmath.digamma(mpmath.e**20))
        self.assertIsNone(solve_shape(beyond))
        self.assertIsNone(solve_shape(beyond * 0.5))
        self.assertIsNone(solve_shape(1e-12))

    def test_a_root_just_inside_the_largest_shape_is_found(self) -> None:
        shape = float(mpmath.e**20) * 0.999
        gap = float(mpmath.log(shape) - mpmath.digamma(shape))
        solved = solve_shape(gap)
        assert solved is not None
        self.assertLess(abs(solved / shape - 1.0), 1e-6)

    def test_the_iteration_cap_bounds_the_search(self) -> None:
        upper = math.exp(20.0)
        with mock.patch.object(_estimates, "MAX_SHAPE_BISECTIONS", 1):
            self.assertEqual(solve_shape(1e-3), math.sqrt(1.0 * upper))
            self.assertEqual(solve_shape(100.0), math.sqrt((1.0 / upper) * 1.0))
        with mock.patch.object(_estimates, "MAX_SHAPE_BISECTIONS", 0):
            self.assertEqual(solve_shape(1.0), math.sqrt((1.0 / upper) * upper))

    def test_the_default_cap_is_ample_for_binary64(self) -> None:
        self.assertEqual(MAX_SHAPE_BISECTIONS, 100)
        calls = 0
        original = _estimates.log_minus_digamma

        def counting(shape: float) -> float:
            nonlocal calls
            calls += 1
            return original(shape)

        with mock.patch.object(_estimates, "log_minus_digamma", counting):
            solve_shape(0.1)
        self.assertLess(calls, 70)
        self.assertGreater(calls, 50)

    def test_the_root_is_a_binary64_neighbour_of_the_sign_change(self) -> None:
        gap = 0.0123456789
        solved = solve_shape(gap)
        assert solved is not None
        below = float(np.nextafter(solved, 0.0))
        above = float(np.nextafter(solved, math.inf))
        self.assertGreater(log_minus_digamma(below) + 1e-15, gap)
        self.assertLess(log_minus_digamma(above) - 1e-15, gap)


class ProfileGapTests(unittest.TestCase):
    def test_a_consistent_pair_gives_the_log_gap(self) -> None:
        gap, magnitude = profile_gap(
            ExactStatistic.of(Fraction(3)), ExactStatistic.of(Fraction(1, 2))
        )
        self.assertAlmostEqual(gap, math.log(3.0) - 0.5, delta=2e-16)
        self.assertAlmostEqual(magnitude, math.log(3.0) + 0.5, delta=2e-16)

    def test_the_rounding_of_the_mean_is_corrected_exactly(self) -> None:
        # The stored rounded values are deliberately wrong; the exact values decide.
        gap, magnitude = profile_gap(
            ExactStatistic(Fraction(5, 2), 2.0), ExactStatistic(Fraction(1, 3), 0.25)
        )
        expected = float(mpmath.log(mpmath.mpf(5) / 2) - mpmath.mpf(1) / 3)
        self.assertAlmostEqual(gap, expected, delta=4e-16)
        self.assertEqual(magnitude, math.log(2.0) + 0.25)

    def test_a_tiny_excess_of_the_mean_survives(self) -> None:
        mean = Fraction(1) + Fraction(1, 1 << 60)
        gap, magnitude = profile_gap(ExactStatistic(mean, 1.0), ExactStatistic(Fraction(0), 0.0))
        self.assertEqual(gap, 2.0**-60)
        self.assertEqual(magnitude, 0.0)

    def test_a_tiny_excess_of_the_mean_log_survives(self) -> None:
        mean_log = Fraction(1, 1 << 60)
        gap, _ = profile_gap(ExactStatistic(Fraction(1), 1.0), ExactStatistic(mean_log, 0.0))
        self.assertEqual(gap, -(2.0**-60))

    def test_the_magnitude_adds_the_absolute_values(self) -> None:
        _, magnitude = profile_gap(
            ExactStatistic.of(Fraction(1, 4)), ExactStatistic.of(Fraction(-3, 2))
        )
        self.assertAlmostEqual(magnitude, math.log(4.0) + 1.5, delta=2e-16)


class RoundedSqrtTests(unittest.TestCase):
    def test_exact_roots_are_exact(self) -> None:
        for value, root in ((Fraction(4), 2.0), (Fraction(1, 4), 0.5), (Fraction(9, 16), 0.75),
                            (Fraction(1 << 2000), 2.0**1000), (Fraction(1, 1 << 2000), 2.0**-1000)):
            with self.subTest(value=value):
                self.assertEqual(rounded_sqrt(value), root)

    def test_the_result_is_the_correctly_rounded_root(self) -> None:
        rng = np.random.default_rng(3)
        numerators = rng.integers(1, 2**62, 40).tolist()
        denominators = rng.integers(1, 2**40, 40).tolist()
        exponents = rng.integers(-1000, 1000, 40).tolist()
        for numerator, denominator, exponent in zip(numerators, denominators, exponents):
            value = Fraction(numerator, denominator) * Fraction(2) ** exponent
            root = rounded_sqrt(value)
            below = Fraction(float(np.nextafter(root, 0.0)))
            above = Fraction(float(np.nextafter(root, math.inf)))
            low, high = (Fraction(root) + below) / 2, (Fraction(root) + above) / 2
            with self.subTest(value=float(value)):
                self.assertLessEqual(low * low, value)
                self.assertLessEqual(value, high * high)

    def test_a_value_just_above_a_rounding_boundary_rounds_up(self) -> None:
        # The midpoint of 1.0 and its successor, squared, plus one part in 2**200.
        midpoint = (Fraction(1) + Fraction(1) + Fraction(2) ** -52) / 2
        self.assertEqual(rounded_sqrt(midpoint * midpoint), 1.0)
        self.assertEqual(rounded_sqrt(midpoint * midpoint + Fraction(1, 1 << 200)), 1.0 + 2.0**-52)
        self.assertEqual(rounded_sqrt(midpoint * midpoint - Fraction(1, 1 << 200)), 1.0)

    def test_ties_go_to_even_and_the_subnormal_range_is_reached(self) -> None:
        self.assertEqual(rounded_sqrt(Fraction(1, 1 << 2150)), 0.0)
        self.assertEqual(rounded_sqrt(Fraction(1, 1 << 2148)), 5e-324)
        self.assertEqual(rounded_sqrt(Fraction(9, 1 << 2150)), 2 * 5e-324)
        self.assertEqual(rounded_sqrt(Fraction(25, 1 << 2150)), 2 * 5e-324)

    def test_a_root_beyond_the_range_overflows(self) -> None:
        with self.assertRaises(OverflowError):
            rounded_sqrt(Fraction(1 << 2048))
        self.assertEqual(rounded_sqrt(Fraction(1 << 2046)), 2.0**1023)


class GaussianLogLikelihoodTests(unittest.TestCase):
    def test_the_terms_are_the_normal_log_likelihood(self) -> None:
        value = gaussian_log_likelihood(4, 2.0, Fraction(16), 0.0)
        self.assertAlmostEqual(
            value, -4 * math.log(2.0) - 4 * 0.5 * math.log(2 * math.pi) - 2.0, delta=1e-15
        )

    def test_the_constant_is_added(self) -> None:
        base = gaussian_log_likelihood(4, 2.0, Fraction(16), 0.0)
        shifted = gaussian_log_likelihood(4, 2.0, Fraction(16), 5.5)
        self.assertAlmostEqual(shifted, base + 5.5, delta=1e-15)

    def test_extreme_scales_do_not_overflow(self) -> None:
        large = gaussian_log_likelihood(10, 1e200, Fraction(10) * Fraction(1e200) ** 2, 0.0)
        small = gaussian_log_likelihood(10, 1e-200, Fraction(10) * Fraction(1e-200) ** 2, 0.0)
        half_log = 5 * math.log(2 * math.pi)
        self.assertAlmostEqual(large, -10 * math.log(1e200) - half_log - 5.0, delta=1e-12)
        self.assertAlmostEqual(small, -10 * math.log(1e-200) - half_log - 5.0, delta=1e-12)


class ExponentialEstimateTests(unittest.TestCase):
    def test_the_rate_is_count_over_total_time(self) -> None:
        result = fit_exponential_state(folded(ExponentialState, [1.0, 2.0, 3.0, 4.0]))
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.rate, 0.4)
        self.assertEqual(result.total_time, 10.0)
        self.assertEqual(result.mean, 2.5)
        self.assertEqual((result.observation_count, result.event_count), (4, 4))
        self.assertEqual(result.censored_count, 0)
        self.assertEqual(result.log_likelihood, 4 * math.log(0.4) - 0.4 * 10.0)

    def test_the_total_is_the_exact_sum_rounded_once(self) -> None:
        values = [1e16, 1.0, 1.0, 1.0, 1.0]
        result = fit_exponential_state(folded(ExponentialState, values))
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.total_time, 1e16 + 4.0)
        self.assertEqual(result.rate, 5 / (1e16 + 4.0))

    def test_matches_the_independent_rate(self) -> None:
        values = np.random.default_rng(5).exponential(3.0, 777)
        result = fit_exponential_state(folded(ExponentialState, values))
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.rate, exponential_rate(values))

    def test_an_empty_state_is_an_empty_sample(self) -> None:
        result = fit_exponential_state(ExponentialState.empty())
        self.assertEqual(
            result, ExponentialFitFailure(ExponentialFitFailureCode.EMPTY_SAMPLE, 0, 0, 0.0)
        )

    def test_all_zero_times_are_an_unbounded_likelihood(self) -> None:
        result = fit_exponential_state(folded(ExponentialState, [0.0, 0.0, 0.0]))
        self.assertEqual(
            result,
            ExponentialFitFailure(ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD, 3, 3, 0.0),
        )

    def test_a_total_beyond_the_float_range_is_a_numerical_overflow(self) -> None:
        result = fit_exponential_state(folded(ExponentialState, [1.7e308, 1.7e308]))
        self.assertEqual(
            result,
            ExponentialFitFailure(ExponentialFitFailureCode.NUMERICAL_OVERFLOW, 2, 2, None),
        )

    def test_a_rate_beyond_the_float_range_is_a_numerical_overflow(self) -> None:
        result = fit_exponential_state(folded(ExponentialState, [5e-324]))
        self.assertEqual(
            result,
            ExponentialFitFailure(ExponentialFitFailureCode.NUMERICAL_OVERFLOW, 1, 1, None),
        )

    def test_a_single_observation_is_fitted(self) -> None:
        result = fit_exponential_state(folded(ExponentialState, [2.0]))
        assert isinstance(result, ExponentialFitSuccess)
        self.assertEqual(result.rate, 0.5)


class NormalEstimateTests(unittest.TestCase):
    def test_a_small_sample_has_known_estimates(self) -> None:
        result = fit_normal_state(folded(NormalState, [1.0, 2.0, 3.0, 4.0]))
        assert isinstance(result, NormalFitSuccess)
        self.assertEqual(result.mu, 2.5)
        self.assertEqual(result.sigma, math.sqrt(1.25))
        expected = -4 * (math.log(math.sqrt(1.25)) + 0.5 * math.log(2.0 * math.pi)) - 2.0
        self.assertAlmostEqual(result.log_likelihood, expected, delta=1e-14)
        self.assertEqual((result.observation_count, result.event_count), (4, 4))
        self.assertEqual(result.censored_count, 0)
        self.assertTrue(result.converged)
        self.assertTrue(result.complete)
        self.assertEqual(result.restart_failures, 0)

    def test_sigma_divides_by_n_not_n_minus_one(self) -> None:
        result = fit_normal_state(folded(NormalState, [0.0, 2.0]))
        assert isinstance(result, NormalFitSuccess)
        self.assertEqual(result.sigma, 1.0)

    def test_matches_the_independent_estimates(self) -> None:
        for seed in (1, 2, 3):
            values = np.random.default_rng(seed).normal(-3.0, 4.0, 900)
            result = fit_normal_state(folded(NormalState, values))
            assert isinstance(result, NormalFitSuccess)
            self.assertEqual((result.mu, result.sigma), gaussian_estimates(values))

    def test_the_evidence_holds_the_statistics_of_the_state(self) -> None:
        state = folded(NormalState, [1.0, 2.0, 3.0])
        result = fit_normal_state(state)
        assert isinstance(result, NormalFitSuccess)
        evidence = result._evidence
        assert isinstance(evidence, StateEvidence)
        self.assertEqual(evidence.statistics, state.finalize())
        self.assertEqual((evidence.exact, evidence.censored, evidence.fixed), ((), (), ()))

    def test_an_empty_state_is_an_empty_sample(self) -> None:
        self.assertEqual(
            fit_normal_state(NormalState.empty()),
            NormalFitFailure(NormalFitFailureCode.EMPTY_SAMPLE, 0, 0, 0),
        )

    def test_identical_values_are_degenerate(self) -> None:
        for values in ([2.0], [2.0, 2.0], [0.1] * 7, [1e150] * 3, [5e-300] * 4):
            with self.subTest(values=values[:2], count=len(values)):
                self.assertEqual(
                    fit_normal_state(folded(NormalState, values)),
                    NormalFitFailure(
                        NormalFitFailureCode.DEGENERATE_SAMPLE, len(values), len(values), 0
                    ),
                )

    def test_a_tiny_spread_relative_to_the_values_is_estimated_exactly(self) -> None:
        for offset, spread in ((1e9, 1.0), (1e12, 2.0**-9), (1e15, 1.0), (3.0, 2.0**-30)):
            values = [offset - spread, offset, offset + spread, offset]
            result = fit_normal_state(folded(NormalState, values))
            assert isinstance(result, NormalFitSuccess)
            with self.subTest(offset=offset, spread=spread):
                self.assertEqual((result.mu, result.sigma), gaussian_estimates(np.array(values)))
                self.assertAlmostEqual(result.sigma / spread, math.sqrt(0.5), delta=1e-12)

    def test_the_smallest_possible_spread_is_estimated(self) -> None:
        values = [1e9, float(np.nextafter(1e9, 2e9))]
        result = fit_normal_state(folded(NormalState, values))
        assert isinstance(result, NormalFitSuccess)
        self.assertEqual(result.sigma, float(np.spacing(1e9)) / 2.0)

    def test_values_beyond_the_square_range_are_estimated(self) -> None:
        values = [1e200, 2e200, 3e200, 4e200]
        result = fit_normal_state(folded(NormalState, values))
        assert isinstance(result, NormalFitSuccess)
        self.assertEqual((result.mu, result.sigma), gaussian_estimates(np.array(values)))
        self.assertAlmostEqual(result.sigma / 1e200, math.sqrt(1.25), delta=2e-16)
        self.assertTrue(math.isfinite(result.log_likelihood))

    def test_values_below_the_square_range_are_estimated(self) -> None:
        values = [1e-200, 2e-200, 3e-200, 4e-200]
        result = fit_normal_state(folded(NormalState, values))
        assert isinstance(result, NormalFitSuccess)
        self.assertEqual((result.mu, result.sigma), gaussian_estimates(np.array(values)))
        self.assertAlmostEqual(result.sigma / 1e-200, math.sqrt(1.25), delta=2e-16)
        self.assertTrue(math.isfinite(result.log_likelihood))

    def test_a_spread_that_rounds_to_zero_is_optimizer_exhausted(self) -> None:
        result = fit_normal_state(folded(NormalState, [0.0, 5e-324]))
        self.assertEqual(
            result, NormalFitFailure(NormalFitFailureCode.OPTIMIZER_EXHAUSTED, 2, 2, 0)
        )

    def test_a_total_beyond_the_float_range_is_optimizer_exhausted(self) -> None:
        self.assertEqual(
            fit_normal_state(folded(NormalState, [1.7e308, 1.7e308, 1.7e308])),
            NormalFitFailure(NormalFitFailureCode.OPTIMIZER_EXHAUSTED, 3, 3, 0),
        )

    def test_an_arithmetic_failure_in_the_estimate_is_optimizer_exhausted(self) -> None:
        state = folded(NormalState, [1.0, 2.0])
        with mock.patch.object(_estimates, "rounded_sqrt", side_effect=OverflowError):
            self.assertEqual(
                fit_normal_state(state),
                NormalFitFailure(NormalFitFailureCode.OPTIMIZER_EXHAUSTED, 2, 2, 0),
            )


class LognormalEstimateTests(unittest.TestCase):
    def test_a_small_sample_has_known_estimates(self) -> None:
        values = [1.0, math.e, math.e**2]
        result = fit_lognormal_state(folded(LognormalState, values))
        assert isinstance(result, LognormalFitSuccess)
        logs = np.log(np.asarray(values))
        self.assertAlmostEqual(result.mu_log, 1.0, delta=1e-15)
        self.assertAlmostEqual(result.sigma_log, math.sqrt(2.0 / 3.0), delta=1e-15)
        self.assertEqual((result.mu_log, result.sigma_log), gaussian_estimates(logs))
        self.assertEqual((result.observation_count, result.event_count), (3, 3))
        self.assertEqual(result.censored_count, 0)
        self.assertTrue(result.converged)

    def test_matches_the_independent_estimates(self) -> None:
        for seed in (1, 2, 3):
            values = np.random.default_rng(seed).lognormal(0.5, 0.8, 900)
            result = fit_lognormal_state(folded(LognormalState, values))
            assert isinstance(result, LognormalFitSuccess)
            self.assertEqual((result.mu_log, result.sigma_log), lognormal_estimates(values))

    def test_the_likelihood_includes_the_jacobian(self) -> None:
        values = np.array([0.5, 1.5, 4.0, 9.0])
        result = fit_lognormal_state(folded(LognormalState, values))
        assert isinstance(result, LognormalFitSuccess)
        mu, sigma = result.mu_log, result.sigma_log
        expected = math.fsum(
            -math.log(x) - math.log(sigma) - 0.5 * math.log(2 * math.pi)
            - (math.log(x) - mu) ** 2 / (2 * sigma**2)
            for x in values.tolist()
        )
        self.assertAlmostEqual(result.log_likelihood, expected, delta=1e-13)

    def test_the_evidence_holds_the_statistics_of_the_state(self) -> None:
        state = folded(LognormalState, [1.0, 2.0, 3.0])
        result = fit_lognormal_state(state)
        assert isinstance(result, LognormalFitSuccess)
        assert isinstance(result._evidence, StateEvidence)
        self.assertEqual(result._evidence.statistics, state.finalize())

    def test_an_empty_state_is_an_empty_sample(self) -> None:
        self.assertEqual(
            fit_lognormal_state(LognormalState.empty()),
            LognormalFitFailure(LognormalFitFailureCode.EMPTY_SAMPLE, 0, 0, 0),
        )

    def test_identical_values_are_optimizer_exhausted_like_the_in_memory_fit(self) -> None:
        for values in ([2.0], [2.0] * 10, [0.1] * 3, [7.5, 7.5]):
            with self.subTest(count=len(values)):
                self.assertEqual(
                    fit_lognormal_state(folded(LognormalState, values)),
                    LognormalFitFailure(
                        LognormalFitFailureCode.OPTIMIZER_EXHAUSTED, len(values), len(values), 0
                    ),
                )

    def test_an_arithmetic_failure_in_the_estimate_is_optimizer_exhausted(self) -> None:
        state = folded(LognormalState, [1.0, 2.0])
        with mock.patch.object(_estimates, "rounded_sqrt", side_effect=ValueError):
            self.assertEqual(
                fit_lognormal_state(state),
                LognormalFitFailure(LognormalFitFailureCode.OPTIMIZER_EXHAUSTED, 2, 2, 0),
            )


class GammaEstimateTests(unittest.TestCase):
    def test_matches_the_independent_root_for_moderate_shapes(self) -> None:
        for seed, (shape, scale) in enumerate([(0.5, 1.0), (3.0, 2.0), (25.0, 0.1)], start=1):
            values = np.random.default_rng(seed).gamma(shape, scale, 1500)
            result = fit_gamma_state(folded(GammaState, values))
            assert isinstance(result, GammaFitSuccess)
            expected_shape, expected_scale = gamma_estimates(values)
            with self.subTest(shape=shape):
                self.assertLess(abs(result.shape / expected_shape - 1.0), 1e-13)
                self.assertLess(abs(result.scale / expected_scale - 1.0), 1e-13)

    def test_scale_is_the_mean_over_the_shape(self) -> None:
        values = np.random.default_rng(9).gamma(2.0, 3.0, 400)
        result = fit_gamma_state(folded(GammaState, values))
        assert isinstance(result, GammaFitSuccess)
        mean = math.fsum(values.tolist()) / len(values)
        self.assertAlmostEqual(result.shape * result.scale, mean, delta=2e-15 * mean)

    def test_the_likelihood_is_the_gamma_log_likelihood(self) -> None:
        values = np.random.default_rng(10).gamma(4.0, 0.5, 300)
        result = fit_gamma_state(folded(GammaState, values))
        assert isinstance(result, GammaFitSuccess)
        shape, scale = result.shape, result.scale
        expected = math.fsum(
            (shape - 1.0) * math.log(x) - x / scale - math.lgamma(shape) - shape * math.log(scale)
            for x in values.tolist()
        )
        self.assertAlmostEqual(result.log_likelihood, expected, delta=1e-12)

    def test_a_small_sample_is_fitted_with_the_score_equation_satisfied(self) -> None:
        values = [1.0, 2.0, 4.0, 8.0]
        result = fit_gamma_state(folded(GammaState, values))
        assert isinstance(result, GammaFitSuccess)
        logs = np.log(np.asarray(values))
        gap = math.log(sum(values) / 4) - float(np.mean(logs))
        self.assertAlmostEqual(
            math.log(result.shape) - float(mpmath.digamma(result.shape)), gap, delta=1e-14
        )
        self.assertEqual((result.observation_count, result.event_count), (4, 4))
        self.assertEqual(result.censored_count, 0)
        self.assertEqual(result.location, 0.0)
        self.assertTrue(result.converged)

    def test_the_evidence_holds_the_statistics_of_the_state(self) -> None:
        state = folded(GammaState, [1.0, 2.0, 4.0])
        result = fit_gamma_state(state)
        assert isinstance(result, GammaFitSuccess)
        assert isinstance(result._evidence, StateEvidence)
        self.assertEqual(result._evidence.statistics, state.finalize())

    def test_an_empty_state_is_an_empty_sample(self) -> None:
        self.assertEqual(
            fit_gamma_state(GammaState.empty()),
            GammaFitFailure(GammaFitFailureCode.EMPTY_SAMPLE, 0, 0, 0),
        )

    def test_identical_values_are_degenerate(self) -> None:
        for values in ([2.0], [2.0, 2.0], [0.1] * 7, [1e150] * 3, [5e-300] * 4, [1.0] * 5):
            with self.subTest(values=values[:2], count=len(values)):
                self.assertEqual(
                    fit_gamma_state(folded(GammaState, values)),
                    GammaFitFailure(
                        GammaFitFailureCode.DEGENERATE_SAMPLE, len(values), len(values), 0
                    ),
                )

    def test_a_shape_beyond_the_searched_range_is_a_boundary_solution(self) -> None:
        values = np.random.default_rng(3).normal(1000.0, 0.001, 200)
        result = fit_gamma_state(folded(GammaState, values))
        self.assertEqual(
            result, GammaFitFailure(GammaFitFailureCode.BOUNDARY_SOLUTION, 200, 200, 0)
        )

    def test_a_total_beyond_the_float_range_is_optimizer_exhausted(self) -> None:
        self.assertEqual(
            fit_gamma_state(folded(GammaState, [1.7e308, 1.7e308])),
            GammaFitFailure(GammaFitFailureCode.OPTIMIZER_EXHAUSTED, 2, 2, 0),
        )

    def test_a_scale_beyond_the_float_range_is_optimizer_exhausted(self) -> None:
        result = fit_gamma_state(folded(GammaState, [1e308, 1e-300]))
        self.assertEqual(result, GammaFitFailure(GammaFitFailureCode.OPTIMIZER_EXHAUSTED, 2, 2, 0))

    def test_a_scale_that_underflows_is_optimizer_exhausted(self) -> None:
        result = fit_gamma_state(folded(GammaState, [5e-324, 1e-323]))
        self.assertEqual(result, GammaFitFailure(GammaFitFailureCode.OPTIMIZER_EXHAUSTED, 2, 2, 0))


if __name__ == "__main__":
    unittest.main()
