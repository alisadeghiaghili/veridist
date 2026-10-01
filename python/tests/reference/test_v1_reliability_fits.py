"""Independent invariants for the 0.7 reliability fitting cells."""

from __future__ import annotations

import math
import unittest

import numpy as np

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime


class V1ReliabilityFitTests(unittest.TestCase):
    def test_weibull_shape_one_reduces_to_exponential_likelihood(self) -> None:
        from veridist.families.weibull import fit_weibull

        observations = (
            ExactLifetime(1.0),
            RightCensoredLifetime(2.0),
            ExactLifetime(3.0),
        )
        result = fit_weibull(observations, fixed_shape=1.0)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.shape, 1.0, places=15)
        self.assertAlmostEqual(result.scale, 3.0, places=15)
        expected = 2 * math.log(1.0 / 3.0) - 6.0 / 3.0
        self.assertAlmostEqual(result.log_likelihood, expected, places=14)
        self.assertEqual((result.event_count, result.censored_count), (2, 1))

    def test_lognormal_exact_fit_matches_log_space_closed_form(self) -> None:
        from veridist.families.lognormal import fit_lognormal

        observations = tuple(ExactLifetime(value) for value in (1.0, math.e, math.e**2))
        result = fit_lognormal(observations)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.mu_log, 1.0, places=14)
        self.assertAlmostEqual(result.sigma_log, math.sqrt(2.0 / 3.0), places=14)
        self.assertEqual(result.restart_failures, 0)

    def test_censored_cells_return_finite_declared_estimates(self) -> None:
        from veridist.families.lognormal import fit_lognormal
        from veridist.families.weibull import fit_weibull

        observations = (
            ExactLifetime(0.5),
            ExactLifetime(1.0),
            RightCensoredLifetime(2.0),
        )
        weibull = fit_weibull(observations)
        lognormal = fit_lognormal(observations)
        self.assertTrue(weibull.complete)
        self.assertTrue(lognormal.complete)
        self.assertTrue(math.isfinite(weibull.log_likelihood))
        self.assertTrue(math.isfinite(lognormal.log_likelihood))

    def test_typed_non_estimates_preserve_the_reason(self) -> None:
        from veridist.families.lognormal import LognormalFitFailureCode, fit_lognormal
        from veridist.families.weibull import WeibullFitFailureCode, fit_weibull

        all_censored = (RightCensoredLifetime(1.0), RightCensoredLifetime(2.0))
        zero_event = (ExactLifetime(0.0),)
        self.assertEqual(fit_weibull(()).code, WeibullFitFailureCode.EMPTY_SAMPLE)
        self.assertEqual(fit_lognormal(()).code, LognormalFitFailureCode.EMPTY_SAMPLE)
        self.assertEqual(
            fit_weibull(all_censored).code, WeibullFitFailureCode.NO_OBSERVED_EVENTS
        )
        self.assertEqual(
            fit_lognormal(all_censored).code, LognormalFitFailureCode.NO_OBSERVED_EVENTS
        )
        self.assertEqual(fit_weibull(zero_event).code, WeibullFitFailureCode.INVALID_SUPPORT)
        self.assertEqual(fit_lognormal(zero_event).code, LognormalFitFailureCode.INVALID_SUPPORT)

    def test_invalid_fixed_shape_and_degenerate_log_sample_are_non_estimates(self) -> None:
        from veridist.families.lognormal import LognormalFitFailureCode, fit_lognormal
        from veridist.families.weibull import WeibullFitFailureCode, fit_weibull

        self.assertEqual(
            fit_weibull((ExactLifetime(1.0),), fixed_shape=-1.0).code,
            WeibullFitFailureCode.OPTIMIZER_EXHAUSTED,
        )
        with self.assertRaises(TypeError):
            fit_weibull((ExactLifetime(1.0),), fixed_shape="one")
        self.assertEqual(
            fit_lognormal((ExactLifetime(1.0), ExactLifetime(1.0))).code,
            LognormalFitFailureCode.OPTIMIZER_EXHAUSTED,
        )

    def test_estimate_value_objects_reject_non_finite_parameters(self) -> None:
        from veridist.families.lognormal import LognormalFitSuccess
        from veridist.families.weibull import WeibullFitSuccess

        with self.assertRaises(ValueError):
            WeibullFitSuccess(0.0, 1.0, 0.0, 1, 1, 0)
        with self.assertRaises(ValueError):
            WeibullFitSuccess(1.0, float("inf"), 0.0, 1, 1, 0)
        with self.assertRaises(ValueError):
            LognormalFitSuccess(0.0, 0.0, 0.0, 1, 1, 0)
        with self.assertRaises(ValueError):
            LognormalFitSuccess(0.0, 1.0, float("nan"), 1, 1, 0)

    def test_weibull_reports_degenerate_sample_instead_of_a_boundary_success(self) -> None:
        # Previously `fit_weibull` returned `WeibullFitSuccess(shape=403.43,
        # converged=True)` for these two samples: 403.43 == e**6, the hard-
        # coded search bound. No finite-shape MLE exists when every exact
        # time is tied (and no censored time exceeds it), so this is now
        # declared non-estimable up front, before any search is attempted.
        from veridist.families.weibull import WeibullFitFailureCode, fit_weibull

        single = fit_weibull((ExactLifetime(5.0),))
        self.assertEqual(single.code, WeibullFitFailureCode.DEGENERATE_SAMPLE)
        self.assertFalse(single.converged)
        self.assertEqual(single.restart_failures, 0)

        repeated = fit_weibull((ExactLifetime(5.0),) * 5)
        self.assertEqual(repeated.code, WeibullFitFailureCode.DEGENERATE_SAMPLE)

        # A tied censored time at the common exact time does not add
        # information either; still degenerate.
        with_tied_censoring = fit_weibull(
            (ExactLifetime(5.0), ExactLifetime(5.0), RightCensoredLifetime(5.0))
        )
        self.assertEqual(with_tied_censoring.code, WeibullFitFailureCode.DEGENERATE_SAMPLE)

        # A censored time strictly beyond the tied exact times means the MLE
        # is no longer degenerate by this rule, and a shape is found.
        not_degenerate = fit_weibull(
            (ExactLifetime(5.0), ExactLifetime(5.0), RightCensoredLifetime(5.1))
        )
        self.assertNotEqual(
            getattr(not_degenerate, "code", None), WeibullFitFailureCode.DEGENERATE_SAMPLE
        )

    def test_weibull_reports_boundary_solution_when_the_expanded_bracket_still_fails(
        self,
    ) -> None:
        # A sample with near-zero (but not exactly zero) spread has a finite
        # MLE shape that can lie far beyond the original [-6, 6] log-shape
        # search bracket. Bracket expansion up to the hard limit [-20, 20]
        # finds genuine interior optima for moderate spreads (see the
        # scale-invariance reference test); only once the hard limit itself
        # is reached without an interior optimum is this reported as a
        # failure rather than a false "converged" success.
        from veridist.families.weibull import WeibullFitFailureCode, fit_weibull

        tiny_spread = tuple(
            ExactLifetime(5.0 + offset)
            for offset in (
                1e-12, -1e-12, 2e-12, -2e-12, 0.5e-12, -0.5e-12,
                1.5e-12, -1.5e-12, 3e-13, -3e-13, 7e-13, -7e-13,
            )
        )
        result = fit_weibull(tiny_spread)
        self.assertEqual(result.code, WeibullFitFailureCode.BOUNDARY_SOLUTION)
        self.assertFalse(result.converged)

    def test_lognormal_reports_boundary_solution_when_the_expanded_bracket_still_fails(
        self,
    ) -> None:
        # A single exact observation tied down by a handful of right-censored
        # observations far out on the tail pulls the profile mu estimate
        # beyond even the 64x-widened hard limit around the single exact
        # time: no interior optimum exists within the declared search range.
        from veridist.families.lognormal import LognormalFitFailureCode, fit_lognormal

        observations = (ExactLifetime(1.0),) + (RightCensoredLifetime(math.exp(50.0)),) * 10
        result = fit_lognormal(observations)
        self.assertEqual(result.code, LognormalFitFailureCode.BOUNDARY_SOLUTION)
        self.assertFalse(result.converged)
        self.assertEqual(result.restart_failures, 0)

    def test_lognormal_boundary_solution_can_come_from_sigma_alone(self) -> None:
        # Near-tied exact times (spread far below 1, so mu's search half-width
        # clamps to its 8x/64x floor and stays wide relative to the data) let
        # mu settle near the common center, while a single right-censored
        # observation placed far below that center still cannot stop sigma
        # from shrinking toward the degenerate point-mass limit: mu converges
        # but log-sigma is pinned at the -20 hard limit.
        from veridist.families.lognormal import LognormalFitFailureCode, fit_lognormal

        rng = np.random.default_rng(0)
        near_tied = 5.0 + rng.normal(0.0, 1e-12, 20)
        observations = tuple(ExactLifetime(float(value)) for value in near_tied) + (
            RightCensoredLifetime(0.001),
        )
        result = fit_lognormal(observations)
        self.assertEqual(result.code, LognormalFitFailureCode.BOUNDARY_SOLUTION)
        self.assertFalse(result.converged)

    def test_right_tail_underflow_no_longer_aborts_the_optimizer(self) -> None:
        # Previously `_log_sf` raised ValueError whenever `erfc` underflowed to
        # exactly 0.0, which aborted the whole fit with OPTIMIZER_EXHAUSTED even
        # when the true optimum was fine elsewhere. It now falls back to
        # `_log_normal_sf`'s stable asymptotic tail and never raises for a
        # finite z (see test_log_normal_sf_reference.py for the dedicated
        # accuracy check). Reference value for z = (log(1e300) - 0.0) / 1e-6
        # computed with mpmath.log(mpmath.ncdf(-z)) at mp.dps = 60:
        # -238585414971527931.747538998559834503340290434572838408427696
        from veridist.families.lognormal import _log_sf

        result = _log_sf(1e300, 0.0, 1e-6)
        self.assertTrue(math.isfinite(result))
        expected = -238585414971527931.74753899855983450334029043457
        self.assertAlmostEqual(result, expected, delta=abs(expected) * 1e-9)


if __name__ == "__main__":
    unittest.main()
