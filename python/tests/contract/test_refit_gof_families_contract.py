"""Contracts of the refit Monte Carlo cell for the non-exponential families."""

from __future__ import annotations

import dataclasses
import unittest
from collections.abc import Callable, Sequence
from typing import Any
from unittest.mock import patch

import numpy as np

from veridist import cdf, fit, inference, sample
from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue, RightCensoredValue
from veridist.engine.errors import VeridistError
from veridist.families.registry import FamilyId
from veridist.families.results import FitFailure, FitSuccess
from veridist.inference import GofFitError, GofStatistic, refit_monte_carlo_gof

REAL_LINE = ("normal", "gumbel_right")
POSITIVE = ("gamma", "weibull_min", "lognormal")
NON_EXPONENTIAL = ("normal", "gamma", "weibull_min", "lognormal", "gumbel_right")
PARAMETERS: dict[str, dict[str, float]] = {
    "normal": {"mu": 3.0, "sigma": 1.5},
    "gamma": {"shape": 2.0, "scale": 1.5},
    "weibull_min": {"shape": 1.7, "scale": 2.0},
    "lognormal": {"mu_log": 0.5, "sigma_log": 0.6},
    "gumbel_right": {"location": 2.0, "scale": 1.3},
}
KS = frozenset({GofStatistic.KS})
ALL = frozenset({GofStatistic.KS, GofStatistic.AD, GofStatistic.CVM})


def typed(family: str, values: Sequence[float]) -> list[Any]:
    kind = ExactValue if family in REAL_LINE else ExactLifetime
    return [kind(value) for value in values]


def data(family: str, size: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(value) for value in sample(family, size, rng=rng, **PARAMETERS[family])]


def run(family: str, observations: Sequence[float], **overrides: Any) -> Any:
    arguments: dict[str, Any] = {
        "observations": observations,
        "family": family,
        "statistics": KS,
        "replicates": 15,
        "rng": np.random.default_rng(5),
    }
    arguments.update(overrides)
    return refit_monte_carlo_gof(**arguments)


def reference_ks_p_value(
    family: str,
    observations: Sequence[float],
    replicates: int,
    seed: int,
    counted: Callable[[int], bool] = lambda index: True,
) -> tuple[float, int]:
    """An independent evaluation of the refit Monte Carlo KS p-value.

    Uses only the public fit, sample and cdf operations.  Replicates for which
    ``counted(index)`` is false are drawn (they consume the generator) but are
    left out of the tally, as a failed refit is.
    """

    def kolmogorov_smirnov(ordered: Sequence[float], parameters: Any) -> float:
        probabilities = cdf(family, np.asarray(ordered), **parameters)
        size = len(ordered)
        ranks = np.arange(1, size + 1)
        return float(
            max((ranks / size - probabilities).max(), (probabilities - (ranks - 1) / size).max())
        )

    fitted = fit(family, typed(family, observations))
    observed = kolmogorov_smirnov(sorted(observations), fitted.parameters)
    rng = np.random.default_rng(seed)
    exceedances = successful = 0
    for index in range(replicates):
        drawn = sorted(
            float(v) for v in sample(family, len(observations), rng=rng, **fitted.parameters)
        )
        if not counted(index):
            continue
        refit = fit(family, typed(family, drawn))
        successful += 1
        if kolmogorov_smirnov(drawn, refit.parameters) >= observed:
            exceedances += 1
    return (exceedances + 1.0) / (successful + 1.0), successful


class ReferenceAgreementTests(unittest.TestCase):
    def test_p_values_match_an_independent_implementation(self) -> None:
        for number, family in enumerate(NON_EXPONENTIAL):
            with self.subTest(family=family):
                observations = data(family, 12, 100 + number)
                expected, successful = reference_ks_p_value(family, observations, 15, 5)
                result = run(family, observations)
                self.assertEqual(result.p_values, {GofStatistic.KS: expected})
                self.assertEqual(result.successful_replicates, successful)
                self.assertEqual(result.failed_replicates, 0)
                self.assertEqual(result.requested_replicates, 15)

    def test_generator_is_consumed_one_sample_per_replicate(self) -> None:
        for family in NON_EXPONENTIAL:
            with self.subTest(family=family):
                observations = data(family, 9, 7)
                used = np.random.default_rng(11)
                run(family, observations, rng=used, replicates=6)
                reference = np.random.default_rng(11)
                fitted = fit(family, typed(family, observations))
                for _ in range(6):
                    sample(family, 9, rng=reference, **fitted.parameters)
                self.assertEqual(used.bit_generator.state, reference.bit_generator.state)

    def test_same_seed_same_result_and_all_statistics_reported(self) -> None:
        for family in NON_EXPONENTIAL:
            with self.subTest(family=family):
                observations = data(family, 15, 21)
                first = run(family, observations, statistics=ALL, rng=np.random.default_rng(9))
                second = run(family, observations, statistics=ALL, rng=np.random.default_rng(9))
                self.assertEqual(first, second)
                self.assertEqual(set(first.p_values), set(ALL))
                self.assertEqual(first.primary_statistic, GofStatistic.AD)
                for value in first.p_values.values():
                    self.assertGreaterEqual(value, 1.0 / 16.0)
                    self.assertLessEqual(value, 1.0)
                self.assertEqual(first.method, "refit_monte_carlo")
                self.assertEqual(first.rng_policy, "caller_owned_generator")

    def test_standard_error_and_interval_describe_the_primary_statistic(self) -> None:
        observations = data("normal", 15, 3)
        result = run("normal", observations, statistics=ALL, replicates=20)
        primary = result.p_values[GofStatistic.AD]
        standard_error = (primary * (1.0 - primary) / 20.0) ** 0.5
        self.assertEqual(result.monte_carlo_standard_error, standard_error)
        self.assertEqual(
            result.interval,
            (max(0.0, primary - 1.96 * standard_error), min(1.0, primary + 1.96 * standard_error)),
        )

    def test_family_id_string_and_alias_give_identical_results(self) -> None:
        observations = data("weibull_min", 12, 4)
        by_string = run("weibull_min", observations)
        self.assertEqual(run(FamilyId.WEIBULL_MIN, observations), by_string)
        self.assertEqual(run("weibull", observations), by_string)

    def test_recorded_p_values_for_a_fixed_sample_and_seed(self) -> None:
        # Regression values recorded from this implementation (the statistical
        # evidence is in tests/statistical); they pin the draw order and the
        # statistics across all three EDF statistics.
        expected = {
            "normal": (0.125, 0.125, 0.1875),
            "gamma": (0.8125, 0.8125, 0.6875),
            "weibull_min": (0.8125, 0.8125, 0.8125),
            "lognormal": (0.125, 0.125, 0.125),
            "gumbel_right": (0.4375, 0.4375, 0.375),
        }
        for number, family in enumerate(NON_EXPONENTIAL):
            with self.subTest(family=family):
                result = run(
                    family,
                    data(family, 12, 100 + number),
                    statistics=ALL,
                    rng=np.random.default_rng(5),
                )
                got = (
                    result.p_values[GofStatistic.AD],
                    result.p_values[GofStatistic.CVM],
                    result.p_values[GofStatistic.KS],
                )
                self.assertEqual(got, expected[family])


class ArgumentValidationTests(unittest.TestCase):
    def test_family_must_be_registered(self) -> None:
        with self.assertRaises(ValueError):
            run("poisson", [1.0, 2.0, 3.0])
        for bad in (3, None, 1.5):
            with self.subTest(family=bad), self.assertRaises(TypeError):
                run(bad, [1.0, 2.0, 3.0])  # type: ignore[arg-type]

    def test_shared_arguments_are_validated_for_every_family(self) -> None:
        for family in NON_EXPONENTIAL:
            observations = data(family, 6, 1)
            with self.subTest(family=family):
                for statistics in (frozenset(), frozenset({"KS"}), {GofStatistic.KS}):
                    with self.assertRaises(TypeError):
                        run(family, observations, statistics=statistics)
                for replicates in (0, -1, True, 2.0):
                    with self.assertRaises(ValueError):
                        run(family, observations, replicates=replicates)
                with self.assertRaises(TypeError):
                    run(family, observations, rng=object())

    def test_observations_must_be_plain_real_numbers(self) -> None:
        rejected: tuple[Any, ...] = (
            [ExactLifetime(1.0), ExactLifetime(2.0), ExactLifetime(3.0)],
            [RightCensoredLifetime(1.0), 2.0, 3.0],
            [ExactValue(1.0), ExactValue(2.0), ExactValue(3.0)],
            [RightCensoredValue(1.0), 2.0, 3.0],
            [1.0, 2.0, "3.0"],
            [1.0, 2.0, None],
            [1.0, 2.0, True],
        )
        for family in (*NON_EXPONENTIAL, "exponential"):
            for observations in rejected:
                with self.subTest(family=family, observations=observations):
                    with self.assertRaisesRegex(TypeError, "censored"):
                        run(family, observations)

    def test_observations_must_not_be_empty_or_non_finite(self) -> None:
        for family in NON_EXPONENTIAL:
            with self.subTest(family=family):
                with self.assertRaisesRegex(ValueError, "must not be empty"):
                    run(family, [])
                for bad in (float("nan"), float("inf"), float("-inf")):
                    with self.assertRaisesRegex(ValueError, "must be finite"):
                        run(family, [1.0, 2.0, bad, 4.0])
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    run(family, [1.0, 2.0, 10**400])

    def test_lifetime_families_need_strictly_positive_values(self) -> None:
        for family in POSITIVE:
            for bad in (0.0, -1.0):
                with self.subTest(family=family, value=bad):
                    with self.assertRaisesRegex(ValueError, "strictly positive"):
                        run(family, [1.0, 2.0, 3.0, bad])

    def test_real_line_families_accept_zero_and_negative_values(self) -> None:
        for family in REAL_LINE:
            with self.subTest(family=family):
                result = run(family, [-2.5, 0.0, 0.4, 1.1, 2.0])
                self.assertEqual(result.successful_replicates, 15)

    def test_minimum_sample_size_is_three_for_two_parameter_families(self) -> None:
        for family in NON_EXPONENTIAL:
            with self.subTest(family=family):
                with self.assertRaisesRegex(ValueError, "at least 3 observations"):
                    run(family, [1.0, 2.0])
                result = run(family, [1.0, 2.0, 4.0])
                self.assertEqual(result.successful_replicates, 15)

    def test_integers_and_numpy_values_are_accepted(self) -> None:
        reference = run("gamma", [1.0, 2.0, 4.0, 7.0])
        self.assertEqual(run("gamma", [1, 2, 4, 7]), reference)
        self.assertEqual(run("gamma", np.array([1.0, 2.0, 4.0, 7.0])), reference)
        self.assertEqual(run("gamma", iter([1.0, 2.0, 4.0, 7.0])), reference)
        self.assertEqual(run("gamma", np.array([1, 2, 4, 7], dtype=np.int64)), reference)


class ObservedFitFailureTests(unittest.TestCase):
    def test_a_failed_observed_fit_raises_a_typed_error_naming_the_code(self) -> None:
        for family in NON_EXPONENTIAL:
            constant = [2.0, 2.0, 2.0, 2.0]
            failure = fit(family, typed(family, constant))
            self.assertIsInstance(failure, FitFailure)
            with self.subTest(family=family):
                generator = np.random.default_rng(1)
                before = generator.bit_generator.state
                with self.assertRaises(GofFitError) as caught:
                    run(family, constant, rng=generator)
                error = caught.exception
                self.assertEqual(error.code, failure.code.value)
                self.assertEqual(error.family, FamilyId(family))
                self.assertIn(failure.code.value, str(error))
                self.assertIn(family, str(error))
                self.assertIsInstance(error, VeridistError)
                self.assertIsInstance(error, RuntimeError)
                # No replicate was drawn: the generator is untouched.
                self.assertEqual(generator.bit_generator.state, before)

    def test_a_fit_that_did_not_converge_is_not_used(self) -> None:
        observations = data("gamma", 10, 2)
        real = fit("gamma", typed("gamma", observations))

        def fitter(family: Any, values: Any, /, **options: Any) -> Any:
            return dataclasses.replace(real, converged=False)

        with patch.object(inference, "_fit_family", side_effect=fitter):
            with self.assertRaises(GofFitError) as caught:
                run("gamma", observations)
        self.assertEqual(caught.exception.code, "NOT_CONVERGED")

    def test_an_unrepresentable_observed_cdf_is_a_fit_error(self) -> None:
        observations = data("normal", 10, 2)

        def broken(family: Any, x: Any, /, **parameters: Any) -> Any:
            return np.full(len(x), np.nan)

        with patch.object(inference, "cdf", side_effect=broken):
            with self.assertRaises(GofFitError) as caught:
                run("normal", observations)
        self.assertEqual(caught.exception.code, "NOT_REPRESENTABLE")


class ReplicateFailureAccountingTests(unittest.TestCase):
    """Replicate refits that fail are counted, never retried and never tallied."""

    OUTCOMES = ("ok", "failure", "not_converged", "arithmetic", "value")

    def scripted_fitter(self, family: str, observations: Sequence[float]) -> Callable[..., Any]:
        real = fit(family, typed(family, observations))
        failure = fit(family, typed(family, [2.0, 2.0, 2.0, 2.0]))
        calls = {"count": 0}

        def fitter(name: Any, values: Any, /, **options: Any) -> Any:
            index = calls["count"]
            calls["count"] += 1
            if index == 0:  # the observed sample
                return fit(name, values)
            outcome = self.OUTCOMES[(index - 1) % len(self.OUTCOMES)]
            if outcome == "ok":
                return fit(name, values)
            if outcome == "failure":
                return failure
            if outcome == "not_converged":
                return dataclasses.replace(real, converged=False)
            if outcome == "arithmetic":
                raise ArithmeticError("synthetic refit failure")
            raise ValueError("synthetic refit failure")

        return fitter

    def test_failures_are_counted_and_left_out_of_the_p_value(self) -> None:
        for number, family in enumerate(NON_EXPONENTIAL):
            observations = data(family, 12, 300 + number)
            with self.subTest(family=family):
                fitter = self.scripted_fitter(family, observations)
                with patch.object(inference, "_fit_family", side_effect=fitter):
                    result = run(family, observations, replicates=20)
                expected, successful = reference_ks_p_value(
                    family, observations, 20, 5, counted=lambda index: index % 5 == 0
                )
                self.assertEqual(result.requested_replicates, 20)
                self.assertEqual(result.successful_replicates, 4)
                self.assertEqual(result.failed_replicates, 16)
                self.assertEqual(successful, 4)
                self.assertEqual(result.p_values, {GofStatistic.KS: expected})

    def test_every_replicate_failing_is_an_error(self) -> None:
        observations = data("normal", 10, 8)
        real = fit("normal", typed("normal", observations))

        calls = {"count": 0}

        def fitter(name: Any, values: Any, /, **options: Any) -> Any:
            calls["count"] += 1
            if calls["count"] == 1:  # the observed sample
                return real
            raise ArithmeticError("synthetic refit failure")

        with patch.object(inference, "_fit_family", side_effect=fitter):
            with self.assertRaisesRegex(RuntimeError, "all Monte Carlo refits failed"):
                run("normal", observations)

    def test_non_finite_draws_count_as_failed_replicates(self) -> None:
        observations = data("weibull_min", 10, 9)
        real_sample = inference.sample
        calls = {"count": 0}

        def sometimes_infinite(*args: Any, **kwargs: Any) -> Any:
            calls["count"] += 1
            values = real_sample(*args, **kwargs)
            if calls["count"] % 2 == 0:
                values[0] = np.inf
            return values

        with patch.object(inference, "sample", side_effect=sometimes_infinite):
            result = run("weibull_min", observations, replicates=10)
        self.assertEqual((result.successful_replicates, result.failed_replicates), (5, 5))

    def test_unrepresentable_replicate_cdfs_count_as_failed_replicates(self) -> None:
        observations = data("gamma", 10, 9)
        real_cdf = inference.cdf
        calls = {"count": 0}

        def sometimes_nan(*args: Any, **kwargs: Any) -> Any:
            calls["count"] += 1
            values = real_cdf(*args, **kwargs)
            if calls["count"] % 3 == 0:
                values[2] = np.nan
            return values

        with patch.object(inference, "cdf", side_effect=sometimes_nan):
            result = run("gamma", observations, replicates=9)
        # The first call is the observed sample's; calls 3, 6, 9 are failed replicates.
        self.assertEqual((result.successful_replicates, result.failed_replicates), (6, 3))

    def test_programming_errors_are_not_swallowed(self) -> None:
        observations = data("normal", 10, 8)

        calls = {"count": 0}

        def broken(name: Any, values: Any, /, **options: Any) -> Any:
            calls["count"] += 1
            if calls["count"] == 1:  # the observed sample
                return fit(name, values)
            raise TypeError("a bug, not a failed refit")

        with patch.object(inference, "_fit_family", side_effect=broken):
            with self.assertRaises(TypeError):
                run("normal", observations)


class FitProtocolTests(unittest.TestCase):
    def test_fit_results_satisfy_the_protocols_the_cell_relies_on(self) -> None:
        for family in NON_EXPONENTIAL:
            with self.subTest(family=family):
                good = fit(family, typed(family, data(family, 8, 1)))
                bad = fit(family, typed(family, [3.0, 3.0, 3.0]))
                self.assertIsInstance(good, FitSuccess)
                self.assertIsInstance(bad, FitFailure)


if __name__ == "__main__":
    unittest.main()
