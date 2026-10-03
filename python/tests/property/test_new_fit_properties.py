"""Invariance, degeneracy, boundary and reducer-consistency properties of the fits."""

from __future__ import annotations

import random
import unittest
from collections.abc import Callable
from math import log

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue, RightCensoredValue
from veridist.engine.errors import CapabilityError
from veridist.families import (
    fit,
    fit_exponential,
    fit_gamma,
    fit_gumbel_right,
    fit_lognormal,
    fit_normal,
    fit_weibull,
)
from veridist.families.exponential import ExponentialFitSuccess
from veridist.families.gamma import GammaFitFailure, GammaFitFailureCode, GammaFitSuccess
from veridist.families.gumbel import GumbelFitFailure, GumbelFitFailureCode, GumbelFitSuccess
from veridist.families.normal import NormalFitFailure, NormalFitFailureCode, NormalFitSuccess
from veridist.families.registry import FamilyId
from veridist.families.results import FitFailure
from veridist.statistics.lifetime_log_likelihood import (
    reduce_lifetime_log_likelihood_chunks,
    reduce_value_log_likelihood_chunks,
)
from veridist.statistics.log_likelihood import LogLikelihoodSuccess

LIFETIME_FAMILIES = (
    FamilyId.EXPONENTIAL,
    FamilyId.WEIBULL_MIN,
    FamilyId.LOGNORMAL,
    FamilyId.GAMMA,
)


def _data(family: FamilyId, seed: int, count: int = 40) -> list[float]:
    rng = random.Random(seed)
    if family is FamilyId.NORMAL:
        return [rng.gauss(20.0, 4.0) for _ in range(count)]
    if family is FamilyId.GUMBEL_RIGHT:
        return [20.0 - 4.0 * log(-log(rng.random())) for _ in range(count)]
    if family is FamilyId.GAMMA:
        return [rng.gammavariate(2.0, 3.0) for _ in range(count)]
    if family is FamilyId.WEIBULL_MIN:
        return [rng.weibullvariate(5.0, 1.8) for _ in range(count)]
    if family is FamilyId.LOGNORMAL:
        return [rng.lognormvariate(1.0, 0.6) for _ in range(count)]
    return [rng.expovariate(0.3) for _ in range(count)]


def _observations(family: FamilyId, data: list[float], cutoff: float | None) -> list[object]:
    exact, censored = (
        (ExactLifetime, RightCensoredLifetime)
        if family in LIFETIME_FAMILIES
        else (ExactValue, RightCensoredValue)
    )
    return [exact(x) if cutoff is None or x < cutoff else censored(cutoff) for x in data]


def _value_data() -> list[float]:
    return _data(FamilyId.NORMAL, 5, 30)


class ReducerConsistencyTests(unittest.TestCase):
    """The reducer at the fitted parameters equals the fit's log-likelihood (rel 1e-12)."""

    def test_every_family_with_and_without_censoring(self) -> None:
        compared = 0
        for family in FamilyId:
            for seed in (1, 2, 3):
                data = _data(family, seed)
                for cutoff in (None, sorted(data)[28]):
                    observations = _observations(family, data, cutoff)
                    result = fit(family, observations)
                    with self.subTest(family=family, seed=seed, censored=cutoff is not None):
                        self.assertEqual(result.family, family)  # type: ignore[union-attr]
                        reducer = (
                            reduce_lifetime_log_likelihood_chunks
                            if family in LIFETIME_FAMILIES
                            else reduce_value_log_likelihood_chunks
                        )
                        total = reducer(family, [observations], **result.parameters)  # type: ignore[union-attr]
                        assert isinstance(total, LogLikelihoodSuccess)
                        want = result.log_likelihood  # type: ignore[union-attr]
                        self.assertLessEqual(
                            abs(total.total_log_likelihood - want), 1e-12 * abs(want)
                        )
                        compared += 1
        self.assertEqual(compared, 36)


class InvarianceTests(unittest.TestCase):
    def _transform_values(self, observations: list[object], a: float, b: float) -> list[object]:
        return [
            type(o)(a * o.value + b)
            for o in observations  # type: ignore[attr-defined]
        ]

    def _transform_lifetimes(self, observations: list[object], c: float) -> list[object]:
        return [type(o)(c * o.time) for o in observations]  # type: ignore[attr-defined]

    def test_normal_is_location_scale_equivariant(self) -> None:
        data = _data(FamilyId.NORMAL, 8)
        for cutoff, factors, tolerance in (
            (None, (2.5, -0.4), 1e-12),
            (sorted(data)[30], (2.5, 0.013), 1e-6),
        ):
            base = fit_normal(_observations(FamilyId.NORMAL, data, cutoff))  # type: ignore[arg-type]
            assert isinstance(base, NormalFitSuccess)
            for a in factors:
                for b in (37.5, -1.0e4):
                    moved = fit_normal(
                        self._transform_values(_observations(FamilyId.NORMAL, data, cutoff), a, b)  # type: ignore[arg-type]
                    )
                    with self.subTest(censored=cutoff is not None, a=a, b=b):
                        assert isinstance(moved, NormalFitSuccess)
                        scale = abs(a) * base.sigma
                        self.assertLessEqual(abs(moved.mu - (a * base.mu + b)), tolerance * scale)
                        self.assertLessEqual(abs(moved.sigma - scale), tolerance * scale)
                        want = base.log_likelihood - base.event_count * log(abs(a))
                        self.assertLessEqual(
                            abs(moved.log_likelihood - want), tolerance * abs(want)
                        )

    def test_gumbel_is_location_scale_equivariant_for_positive_scale_factors(self) -> None:
        data = _data(FamilyId.GUMBEL_RIGHT, 9)
        for cutoff, tolerance in ((None, 1e-6), (sorted(data)[30], 1e-6)):
            observations = _observations(FamilyId.GUMBEL_RIGHT, data, cutoff)
            base = fit_gumbel_right(observations)  # type: ignore[arg-type]
            assert isinstance(base, GumbelFitSuccess)
            for a, b in ((2.5, 37.5), (0.013, -1.0e4)):
                moved = fit_gumbel_right(self._transform_values(observations, a, b))  # type: ignore[arg-type]
                with self.subTest(censored=cutoff is not None, a=a, b=b):
                    assert isinstance(moved, GumbelFitSuccess)
                    scale = a * base.scale
                    self.assertLessEqual(
                        abs(moved.location - (a * base.location + b)), tolerance * scale
                    )
                    self.assertLessEqual(abs(moved.scale - scale), tolerance * scale)
                    want = base.log_likelihood - base.event_count * log(a)
                    self.assertLessEqual(abs(moved.log_likelihood - want), tolerance * abs(want))

    def test_gamma_is_scale_equivariant_and_keeps_its_shape(self) -> None:
        data = _data(FamilyId.GAMMA, 10)
        for cutoff in (None, sorted(data)[30]):
            observations = _observations(FamilyId.GAMMA, data, cutoff)
            base = fit_gamma(observations)  # type: ignore[arg-type]
            assert isinstance(base, GammaFitSuccess)
            for c in (1.0e-3, 7.25, 3.0e4):
                moved = fit_gamma(self._transform_lifetimes(observations, c))  # type: ignore[arg-type]
                with self.subTest(censored=cutoff is not None, c=c):
                    assert isinstance(moved, GammaFitSuccess)
                    self.assertLessEqual(abs(moved.shape - base.shape), 1e-6 * base.shape)
                    self.assertLessEqual(abs(moved.scale - c * base.scale), 1e-6 * c * base.scale)
                    want = base.log_likelihood - base.event_count * log(c)
                    self.assertLessEqual(abs(moved.log_likelihood - want), 1e-6 * abs(want))


class DegenerateSampleTests(unittest.TestCase):
    def test_identical_exact_values_and_a_single_value_are_degenerate(self) -> None:
        fits: list[tuple[Callable[..., object], type, type, object]] = [
            (fit_normal, ExactValue, RightCensoredValue, NormalFitFailureCode.DEGENERATE_SAMPLE),
            (
                fit_gumbel_right,
                ExactValue,
                RightCensoredValue,
                GumbelFitFailureCode.DEGENERATE_SAMPLE,
            ),
            (
                fit_gamma,
                ExactLifetime,
                RightCensoredLifetime,
                GammaFitFailureCode.DEGENERATE_SAMPLE,
            ),
        ]
        for function, exact, censored, code in fits:
            for observations in (
                [exact(2.0)],
                [exact(2.0)] * 5,
                [exact(2.0)] * 3 + [censored(1.0), censored(2.0)],
            ):
                with self.subTest(fit=function.__name__, n=len(observations)):
                    result = function(observations)
                    self.assertIsInstance(
                        result, (NormalFitFailure, GumbelFitFailure, GammaFitFailure)
                    )
                    self.assertIs(result.code, code)  # type: ignore[attr-defined]
                    self.assertFalse(result.converged)  # type: ignore[attr-defined]
                    self.assertEqual(
                        result.event_count, sum(type(o) is exact for o in observations)
                    )  # type: ignore[attr-defined]

    def test_a_censoring_point_above_the_common_value_is_not_degenerate(self) -> None:
        for function, exact, censored in (
            (fit_normal, ExactValue, RightCensoredValue),
            (fit_gumbel_right, ExactValue, RightCensoredValue),
            (fit_gamma, ExactLifetime, RightCensoredLifetime),
        ):
            result = function([exact(2.0), exact(2.0), censored(3.0)])
            with self.subTest(fit=function.__name__):
                if isinstance(result, FitFailure):
                    self.assertNotEqual(result.code.value, "DEGENERATE_SAMPLE")

    def test_empty_and_event_free_samples_have_their_own_codes(self) -> None:
        for function, censored in (
            (fit_normal, RightCensoredValue),
            (fit_gumbel_right, RightCensoredValue),
            (fit_gamma, RightCensoredLifetime),
        ):
            with self.subTest(fit=function.__name__):
                self.assertEqual(function([]).code.value, "EMPTY_SAMPLE")  # type: ignore[attr-defined]
                self.assertEqual(function([censored(1.0)]).code.value, "NO_OBSERVED_EVENTS")  # type: ignore[attr-defined]


class BoundaryAndFailureTests(unittest.TestCase):
    def test_gamma_under_very_heavy_censoring_never_reports_a_boundary_as_converged(self) -> None:
        # Two events and fifty far-right censoring times: the likelihood has no
        # maximum inside the search limits (its supremum needs a scale beyond e**20 times
        # the mean event time), so the only honest answer is a boundary failure.
        observations = [ExactLifetime(1.0), ExactLifetime(2.0)] + [
            RightCensoredLifetime(1000.0)
        ] * 50
        result = fit_gamma(observations)
        assert isinstance(result, GammaFitFailure)
        self.assertIs(result.code, GammaFitFailureCode.BOUNDARY_SOLUTION)
        self.assertFalse(result.converged)
        self.assertEqual((result.event_count, result.censored_count), (2, 50))

    def test_any_gamma_success_under_heavy_censoring_is_a_local_optimum(self) -> None:
        rng = random.Random(77)
        successes = failures = 0
        for _ in range(6):
            events = rng.randint(2, 4)
            observations: list[object] = [
                ExactLifetime(rng.uniform(0.5, 6.0)) for _ in range(events)
            ]
            observations += [RightCensoredLifetime(rng.uniform(4.0, 40.0))] * rng.randint(20, 40)
            result = fit_gamma(observations)  # type: ignore[arg-type]
            if isinstance(result, GammaFitFailure):
                failures += 1
                self.assertIn(
                    result.code,
                    (
                        GammaFitFailureCode.BOUNDARY_SOLUTION,
                        GammaFitFailureCode.OPTIMIZER_EXHAUSTED,
                    ),
                )
                continue
            assert isinstance(result, GammaFitSuccess)
            successes += 1
            for shape_factor, scale_factor in ((1.01, 1.0), (0.99, 1.0), (1.0, 1.01), (1.0, 0.99)):
                nearby = reduce_lifetime_log_likelihood_chunks(
                    FamilyId.GAMMA,
                    [observations],
                    shape=result.shape * shape_factor,
                    scale=result.scale * scale_factor,
                )
                assert isinstance(nearby, LogLikelihoodSuccess)
                self.assertLessEqual(nearby.total_log_likelihood, result.log_likelihood + 1e-9)
        self.assertEqual(successes + failures, 6)

    def test_gamma_without_censoring_reports_a_boundary_when_the_sample_is_almost_constant(
        self,
    ) -> None:
        result = fit_gamma([ExactLifetime(100.0 + d) for d in (0.0, 0.001, -0.001, 0.002, -0.002)])
        assert isinstance(result, GammaFitFailure)
        self.assertIs(result.code, GammaFitFailureCode.BOUNDARY_SOLUTION)

    def test_unrepresentable_deviations_are_optimizer_failures_not_exceptions(self) -> None:
        huge = 1.7e308
        normal = fit_normal([ExactValue(huge), ExactValue(huge / 2), RightCensoredValue(-huge)])
        assert isinstance(normal, NormalFitFailure)
        self.assertIs(normal.code, NormalFitFailureCode.OPTIMIZER_EXHAUSTED)
        gumbel = fit_gumbel_right(
            [ExactValue(huge), ExactValue(huge / 2), RightCensoredValue(-huge)]
        )
        assert isinstance(gumbel, GumbelFitFailure)
        self.assertIs(gumbel.code, GumbelFitFailureCode.OPTIMIZER_EXHAUSTED)
        gumbel_pair = fit_gumbel_right([ExactValue(huge), ExactValue(-huge)])
        assert isinstance(gumbel_pair, GumbelFitFailure)
        self.assertIs(gumbel_pair.code, GumbelFitFailureCode.OPTIMIZER_EXHAUSTED)
        gamma = fit_gamma(
            [ExactLifetime(1e-300), ExactLifetime(2e-300), RightCensoredLifetime(1e300)]
        )
        assert isinstance(gamma, GammaFitFailure)
        self.assertIs(gamma.code, GammaFitFailureCode.OPTIMIZER_EXHAUSTED)

    def test_gamma_requires_strictly_positive_times(self) -> None:
        result = fit_gamma([ExactLifetime(0.0), ExactLifetime(1.0)])
        assert isinstance(result, GammaFitFailure)
        self.assertIs(result.code, GammaFitFailureCode.INVALID_SUPPORT)

    def test_two_distinct_values_have_an_interior_optimum(self) -> None:
        for function, exact in (
            (fit_normal, ExactValue),
            (fit_gumbel_right, ExactValue),
            (fit_gamma, ExactLifetime),
        ):
            result = function([exact(1.0), exact(2.0)])
            with self.subTest(fit=function.__name__):
                self.assertTrue(result.converged)  # type: ignore[attr-defined]
                self.assertEqual(result.event_count, 2)  # type: ignore[attr-defined]


class AdmissionTests(unittest.TestCase):
    def test_frequency_weights_expand_exactly(self) -> None:
        values = [ExactValue(1.0), ExactValue(4.0), RightCensoredValue(6.0)]
        weights = [2, 1, 3]
        expanded = [values[0]] * 2 + [values[1]] + [values[2]] * 3
        for function in (fit_normal, fit_gumbel_right):
            self.assertEqual(function(values, frequency_weights=weights), function(expanded))
        lifetimes = [ExactLifetime(1.0), ExactLifetime(4.0), RightCensoredLifetime(6.0)]
        expanded_lifetimes = [lifetimes[0]] * 2 + [lifetimes[1]] + [lifetimes[2]] * 3
        self.assertEqual(
            fit_gamma(lifetimes, frequency_weights=weights), fit_gamma(expanded_lifetimes)
        )

    def test_unsupported_semantics_are_capability_errors_and_bad_options_are_rejected(self) -> None:
        values = [ExactValue(1.0), ExactValue(2.0)]
        lifetimes = [ExactLifetime(1.0), ExactLifetime(2.0)]
        for function, data in (
            (fit_normal, values),
            (fit_gumbel_right, values),
            (fit_gamma, lifetimes),
        ):
            with self.subTest(fit=function.__name__):
                with self.assertRaises(CapabilityError):
                    function(data, analytic_weights=[1.0, 1.0])
                with self.assertRaises(CapabilityError):
                    function(data, truncation=(0.0, 5.0))
                with self.assertRaises(CapabilityError):
                    function(data, censoring="left")
                with self.assertRaises(CapabilityError):
                    function(data, censoring="interval")
                with self.assertRaises(ValueError):
                    function(data, censoring="sideways")
                with self.assertRaises(ValueError):
                    function(data, frequency_weights=[1])
                with self.assertRaises(TypeError):
                    function(data, frequency_weights=[1, True])
                with self.assertRaises(TypeError):
                    function(data, frequency_weights=[1, -1])

    def test_each_fit_rejects_the_other_observation_pair_with_type_error(self) -> None:
        lifetimes = [ExactLifetime(1.0), ExactLifetime(2.0)]
        values = [ExactValue(1.0), ExactValue(2.0)]
        for function in (fit_normal, fit_gumbel_right):
            with self.assertRaises(TypeError) as caught:
                function(lifetimes)
            self.assertIn("real values", str(caught.exception))
        for function in (fit_gamma, fit_weibull, fit_lognormal, fit_exponential):
            with self.assertRaises(TypeError):
                function(values)  # type: ignore[arg-type]


class ParameterRecoveryTests(unittest.TestCase):
    def test_large_samples_recover_the_generating_parameters(self) -> None:
        normal = fit_normal(_observations(FamilyId.NORMAL, _data(FamilyId.NORMAL, 21, 4000), None))  # type: ignore[arg-type]
        assert isinstance(normal, NormalFitSuccess)
        self.assertAlmostEqual(normal.mu, 20.0, delta=0.4)
        self.assertAlmostEqual(normal.sigma, 4.0, delta=0.3)
        gumbel = fit_gumbel_right(
            _observations(FamilyId.GUMBEL_RIGHT, _data(FamilyId.GUMBEL_RIGHT, 22, 4000), None)  # type: ignore[arg-type]
        )
        assert isinstance(gumbel, GumbelFitSuccess)
        self.assertAlmostEqual(gumbel.location, 20.0, delta=0.4)
        self.assertAlmostEqual(gumbel.scale, 4.0, delta=0.3)
        gamma = fit_gamma(_observations(FamilyId.GAMMA, _data(FamilyId.GAMMA, 23, 4000), None))  # type: ignore[arg-type]
        assert isinstance(gamma, GammaFitSuccess)
        self.assertAlmostEqual(gamma.shape, 2.0, delta=0.2)
        self.assertAlmostEqual(gamma.scale, 3.0, delta=0.3)


if __name__ == "__main__":
    unittest.main()


class ExponentialZeroTimeTests(unittest.TestCase):
    def test_fit_and_reducer_agree_on_samples_containing_zero_times(self) -> None:
        samples = (
            [ExactLifetime(0.0), ExactLifetime(1.5), ExactLifetime(3.0)],
            [ExactLifetime(0.0), RightCensoredLifetime(0.0), ExactLifetime(2.0)],
            [ExactLifetime(0.0)] * 2 + [ExactLifetime(4.0), RightCensoredLifetime(0.0)] * 2,
            [RightCensoredLifetime(0.0), ExactLifetime(0.5), RightCensoredLifetime(7.0)],
        )
        for observations in samples:
            result = fit_exponential(observations)
            with self.subTest(n=len(observations)):
                self.assertIsInstance(result, ExponentialFitSuccess)
                assert isinstance(result, ExponentialFitSuccess)
                total = reduce_lifetime_log_likelihood_chunks(
                    FamilyId.EXPONENTIAL, [observations], rate=result.rate
                )
                assert isinstance(total, LogLikelihoodSuccess)
                want = result.log_likelihood
                self.assertLessEqual(abs(total.total_log_likelihood - want), 1e-12 * abs(want))
                self.assertEqual(total.observation_count, result.observation_count)

    def test_a_sample_of_only_zero_times_has_no_finite_rate(self) -> None:
        result = fit_exponential([ExactLifetime(0.0), RightCensoredLifetime(0.0)])
        self.assertEqual(result.code.value, "UNBOUNDED_LIKELIHOOD")  # type: ignore[union-attr]
