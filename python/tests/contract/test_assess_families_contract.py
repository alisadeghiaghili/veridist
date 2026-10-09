"""Contracts of ``assess_families``: per-family evidence and adequacy-gated selection."""

from __future__ import annotations

import contextlib
import dataclasses
import unittest
from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import numpy as np

from veridist import fit, inference, sample
from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue
from veridist.families.registry import FamilyId
from veridist.inference import (
    FamilyAssessment,
    FamilyCandidate,
    GofStatistic,
    RefitMonteCarloGof,
    SelectionCode,
    assess_families,
    compare_models,
    information_criteria,
    refit_monte_carlo_gof,
)

REGISTRY_ORDER = ("normal", "gamma", "weibull_min", "lognormal", "gumbel_right", "exponential")
REAL_LINE = ("normal", "gumbel_right")
AD_ONLY = frozenset({GofStatistic.AD})


def typed(family: str, values: Any) -> list[Any]:
    kind = ExactValue if family in REAL_LINE else ExactLifetime
    return [kind(value) for value in values]


def positive_sample(size: int = 20, seed: int = 1) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in sample("gamma", size, rng=rng, shape=3.0, scale=1.0)]


def assess(observations: Any = None, **overrides: Any) -> FamilyAssessment:
    arguments: dict[str, Any] = {
        "observations": positive_sample() if observations is None else observations,
        "replicates": 9,
        "rng": np.random.default_rng(3),
    }
    arguments.update(overrides)
    return assess_families(**arguments)


@contextlib.contextmanager
def prescribed_p_values(p_values: dict[str, float]) -> Iterator[None]:
    """Replace the Monte Carlo step so that each family gets a chosen AD p-value."""

    def result(family: Any, *args: Any) -> RefitMonteCarloGof:
        p_value = p_values[family.value]
        return RefitMonteCarloGof(
            9, 9, 0, 0.1, (0.0, 1.0), {GofStatistic.AD: p_value}, GofStatistic.AD
        )

    def model(family: Any, *args: Any) -> RefitMonteCarloGof:
        return result(family)

    def exponential(*args: Any) -> RefitMonteCarloGof:
        return result(FamilyId.EXPONENTIAL)

    with (
        patch.object(inference, "_model_refit_gof", side_effect=model),
        patch.object(inference, "_exponential_refit_gof", side_effect=exponential),
    ):
        yield


def lowest_aic(rows: Any) -> FamilyCandidate:
    return min(rows, key=lambda row: float("inf") if row.aic is None else row.aic)


class RowContentTests(unittest.TestCase):
    def test_default_families_are_the_registry_order_for_a_positive_sample(self) -> None:
        result = assess()
        self.assertEqual(tuple(row.family.value for row in result.candidates), REGISTRY_ORDER)
        self.assertEqual(
            tuple(type(row.family) for row in result.candidates), (FamilyId,) * len(REGISTRY_ORDER)
        )

    def test_default_families_drop_those_whose_support_excludes_the_sample(self) -> None:
        for values in ([-1.0, 0.5, 1.5, 2.0, 3.0], [0.0, 0.5, 1.5, 2.0, 3.0]):
            with self.subTest(sample=values):
                result = assess(values)
                self.assertEqual(
                    tuple(row.family.value for row in result.candidates),
                    ("normal", "gumbel_right"),
                )

    def test_every_row_carries_consistent_evidence(self) -> None:
        observations = positive_sample()
        result = assess(observations, statistic=GofStatistic.CVM, adequacy_threshold=0.3)
        self.assertEqual(result.sample_size, len(observations))
        self.assertEqual(result.statistic, GofStatistic.CVM)
        self.assertEqual(result.adequacy_threshold, 0.3)
        self.assertEqual(result.requested_replicates, 9)
        self.assertEqual(result.method, "refit_monte_carlo")
        self.assertEqual(result.rng_policy, "caller_owned_generator")
        for row in result.candidates:
            with self.subTest(family=row.family.value):
                family = row.family.value
                fitted = fit(family, typed(family, observations))
                self.assertIsNone(row.failure_code)
                self.assertEqual(row.log_likelihood, fitted.log_likelihood)
                self.assertEqual(row.free_parameters, len(fitted.parameters))
                criteria = information_criteria(
                    log_likelihood=fitted.log_likelihood,
                    sample_size=len(observations),
                    free_parameters=len(fitted.parameters),
                )
                self.assertEqual(row.aic, criteria.aic)
                self.assertEqual(row.bic, criteria.bic)
                assert row.gof is not None
                self.assertEqual(set(row.gof.p_values), {GofStatistic.CVM})
                self.assertEqual(row.gof.primary_statistic, GofStatistic.CVM)
                self.assertEqual(row.gof.requested_replicates, 9)
                self.assertEqual(row.p_value, row.gof.p_values[GofStatistic.CVM])
                assert row.p_value is not None
                self.assertEqual(row.adequate, row.p_value >= 0.3)

    def test_free_parameter_counts_follow_the_fitted_parameter_set(self) -> None:
        counts = {row.family.value: row.free_parameters for row in assess().candidates}
        self.assertEqual(
            counts,
            {
                "normal": 2,
                "gamma": 2,
                "weibull_min": 2,
                "lognormal": 2,
                "gumbel_right": 2,
                "exponential": 1,
            },
        )

    def test_the_default_statistic_is_anderson_darling(self) -> None:
        result = assess()
        self.assertEqual(result.statistic, GofStatistic.AD)
        for row in result.candidates:
            assert row.gof is not None
            self.assertEqual(set(row.gof.p_values), {GofStatistic.AD})

    def test_the_default_adequacy_threshold_is_five_percent(self) -> None:
        self.assertEqual(assess().adequacy_threshold, 0.05)

    def test_explicit_families_keep_the_callers_order_and_accept_strings_and_ids(self) -> None:
        result = assess(families=["lognormal", FamilyId.NORMAL, "weibull"])
        self.assertEqual(
            tuple(row.family for row in result.candidates),
            (FamilyId.LOGNORMAL, FamilyId.NORMAL, FamilyId.WEIBULL_MIN),
        )
        result = assess(families=iter(["gamma"]))
        self.assertEqual(tuple(row.family for row in result.candidates), (FamilyId.GAMMA,))

    def test_dataclasses_are_frozen(self) -> None:
        result = assess(families=["normal"])
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.sample_size = 1  # type: ignore[misc]
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.candidates[0].aic = 1.0  # type: ignore[misc]
        self.assertIsInstance(result.candidates, tuple)
        self.assertIsInstance(result.candidates[0], FamilyCandidate)

    def test_numpy_replicates_are_accepted(self) -> None:
        result = assess(families=["normal"], replicates=np.int64(7))
        self.assertEqual(result.requested_replicates, 7)
        self.assertIs(type(result.requested_replicates), int)


class RandomnessTests(unittest.TestCase):
    def test_the_generator_is_shared_sequentially_in_family_order(self) -> None:
        observations = positive_sample()
        families = ["gamma", "normal", "exponential", "weibull_min"]
        result = assess(observations, families=families, rng=np.random.default_rng(17))
        shared = np.random.default_rng(17)
        for row in result.candidates:
            direct = refit_monte_carlo_gof(
                observations=observations,
                family=row.family,
                statistics=AD_ONLY,
                replicates=9,
                rng=shared,
            )
            self.assertEqual(row.gof, direct)

    def test_the_family_order_changes_the_random_stream_not_the_fits(self) -> None:
        observations = positive_sample()
        forward = assess(observations, families=["normal", "gamma"], rng=np.random.default_rng(2))
        backward = assess(observations, families=["gamma", "normal"], rng=np.random.default_rng(2))
        by_family = {row.family: row for row in backward.candidates}
        for row in forward.candidates:
            other = by_family[row.family]
            self.assertEqual(
                (row.log_likelihood, row.aic, row.bic),
                (other.log_likelihood, other.aic, other.bic),
            )
        # The first family of either order draws from the start of the stream.
        self.assertEqual(
            forward.candidates[0].gof,
            refit_monte_carlo_gof(
                observations=observations,
                family="normal",
                statistics=AD_ONLY,
                replicates=9,
                rng=np.random.default_rng(2),
            ),
        )
        self.assertEqual(
            backward.candidates[0].gof,
            refit_monte_carlo_gof(
                observations=observations,
                family="gamma",
                statistics=AD_ONLY,
                replicates=9,
                rng=np.random.default_rng(2),
            ),
        )

    def test_the_same_seed_gives_the_identical_assessment(self) -> None:
        observations = positive_sample()
        first = assess(observations, rng=np.random.default_rng(99))
        second = assess(observations, rng=np.random.default_rng(99))
        self.assertEqual(first, second)

    def test_the_generator_advances_only_for_assessed_families(self) -> None:
        observations = [-1.0, 0.5, 1.5, 2.0, 3.0]
        used = np.random.default_rng(4)
        assess(observations, families=["normal", "lognormal", "exponential"], rng=used)
        reference = np.random.default_rng(4)
        refit_monte_carlo_gof(
            observations=observations,
            family="normal",
            statistics=AD_ONLY,
            replicates=9,
            rng=reference,
        )
        self.assertEqual(used.bit_generator.state, reference.bit_generator.state)


class SelectionTests(unittest.TestCase):
    def test_selection_is_compare_models_over_the_candidate_rows(self) -> None:
        observations = positive_sample(30, 6)
        for threshold in (0.0, 0.2, 0.5):
            with self.subTest(threshold=threshold):
                result = assess(observations, adequacy_threshold=threshold)
                expected = compare_models(
                    candidates=[
                        {"family": row.family.value, "aic": row.aic, "p_value": row.p_value}
                        for row in result.candidates
                    ],
                    adequacy_threshold=threshold,
                )
                self.assertEqual(result.selection, expected)

    def test_a_zero_threshold_selects_the_lowest_aic(self) -> None:
        result = assess(adequacy_threshold=0.0)
        self.assertEqual(result.selection.code, SelectionCode.SELECTED)
        self.assertEqual(
            result.selection.selected_family, lowest_aic(result.candidates).family.value
        )
        self.assertTrue(all(row.adequate for row in result.candidates))

    def test_no_adequate_family_gives_the_typed_empty_outcome(self) -> None:
        with prescribed_p_values({name: 0.01 for name in REGISTRY_ORDER}):
            result = assess()
        self.assertEqual(result.selection.code, SelectionCode.NONE_ADEQUATE)
        self.assertIsNone(result.selection.selected_family)
        self.assertFalse(any(row.adequate for row in result.candidates))
        self.assertTrue(all(row.aic is not None for row in result.candidates))

    def test_the_lowest_aic_family_is_skipped_when_it_is_not_adequate(self) -> None:
        observations = positive_sample(30, 6)
        with prescribed_p_values({name: 0.5 for name in REGISTRY_ORDER}):
            baseline = assess(observations)
        ranked = sorted(baseline.candidates, key=lambda row: row.aic or 0.0)
        self.assertEqual(baseline.selection.selected_family, ranked[0].family.value)
        prescribed = {row.family.value: 0.5 for row in ranked}
        prescribed[ranked[0].family.value] = 0.01
        with prescribed_p_values(prescribed):
            result = assess(observations)
        self.assertEqual(result.selection.code, SelectionCode.SELECTED)
        self.assertEqual(result.selection.selected_family, ranked[1].family.value)
        self.assertEqual(
            [row.adequate for row in result.candidates],
            [row.family != ranked[0].family for row in result.candidates],
        )

    def test_a_less_adequate_lower_aic_family_loses_to_a_better_fitting_one(self) -> None:
        observations = positive_sample(30, 6)
        with prescribed_p_values({name: 0.5 for name in REGISTRY_ORDER}):
            ranked = sorted(assess(observations).candidates, key=lambda row: row.aic or 0.0)
        # Only the worst-AIC family is adequate, so it is the one selected.
        prescribed = {row.family.value: 0.01 for row in ranked}
        prescribed[ranked[-1].family.value] = 0.5
        with prescribed_p_values(prescribed):
            result = assess(observations)
        self.assertEqual(result.selection.selected_family, ranked[-1].family.value)

    def test_the_adequacy_gate_is_inclusive(self) -> None:
        baseline = assess(families=["normal"])
        p_value = baseline.candidates[0].p_value
        assert p_value is not None
        at = assess(families=["normal"], adequacy_threshold=p_value)
        self.assertTrue(at.candidates[0].adequate)
        self.assertEqual(at.selection.selected_family, "normal")


class IneligibleFamilyTests(unittest.TestCase):
    def test_a_failed_observed_fit_is_reported_and_not_eligible(self) -> None:
        constant = [2.0, 2.0, 2.0, 2.0]
        used = np.random.default_rng(1)
        result = assess(
            constant,
            families=["normal", "gamma", "weibull_min", "lognormal", "gumbel_right"],
            rng=used,
        )
        self.assertEqual(result.selection.code, SelectionCode.NONE_ADEQUATE)
        self.assertEqual(used.bit_generator.state, np.random.default_rng(1).bit_generator.state)
        for row in result.candidates:
            with self.subTest(family=row.family.value):
                family = row.family.value
                failure: Any = fit(family, typed(family, constant))
                self.assertEqual(row.failure_code, failure.code.value)
                self.assertEqual(
                    (row.log_likelihood, row.free_parameters, row.aic, row.bic, row.gof),
                    (None, None, None, None, None),
                )
                self.assertIsNone(row.p_value)
                self.assertFalse(row.adequate)

    def test_a_constant_sample_still_fits_the_exponential_family(self) -> None:
        result = assess([2.0, 2.0, 2.0, 2.0])
        rows = {row.family.value: row for row in result.candidates}
        self.assertIsNone(rows["exponential"].failure_code)
        self.assertIsNotNone(rows["exponential"].gof)
        self.assertTrue(
            all(row.failure_code for name, row in rows.items() if name != "exponential")
        )

    def test_one_failed_family_does_not_stop_the_others(self) -> None:
        observations = positive_sample()
        real_fit = inference._fit_family

        def fitter(family: Any, values: Any, /, **options: Any) -> Any:
            if family is FamilyId.GAMMA:
                return real_fit(family, typed("gamma", [3.0, 3.0, 3.0]))
            return real_fit(family, values, **options)

        with patch.object(inference, "_fit_family", side_effect=fitter):
            result = assess(observations, adequacy_threshold=0.0)
        rows = {row.family.value: row for row in result.candidates}
        self.assertEqual(rows["gamma"].failure_code, "DEGENERATE_SAMPLE")
        self.assertIsNone(rows["gamma"].aic)
        self.assertFalse(rows["gamma"].adequate)
        for name in ("normal", "weibull_min", "lognormal", "gumbel_right", "exponential"):
            self.assertIsNone(rows[name].failure_code)
            self.assertIsNotNone(rows[name].gof)
        self.assertNotEqual(result.selection.selected_family, "gamma")

    def test_a_family_that_did_not_converge_is_reported(self) -> None:
        observations = positive_sample()
        real_fit = inference._fit_family

        def fitter(family: Any, values: Any, /, **options: Any) -> Any:
            result = real_fit(family, values, **options)
            if family is FamilyId.LOGNORMAL:
                return dataclasses.replace(result, converged=False)
            return result

        with patch.object(inference, "_fit_family", side_effect=fitter):
            result = assess(observations, families=["lognormal", "normal"])
        self.assertEqual(result.candidates[0].failure_code, "NOT_CONVERGED")
        self.assertIsNone(result.candidates[1].failure_code)

    def test_an_unrepresentable_cdf_makes_that_family_ineligible(self) -> None:
        observations = positive_sample()
        real_cdf = inference.cdf

        def cdf_with_a_broken_normal(family: Any, x: Any, /, **parameters: Any) -> Any:
            if family is FamilyId.NORMAL:
                return np.full(len(x), np.nan)
            return real_cdf(family, x, **parameters)

        with patch.object(inference, "cdf", side_effect=cdf_with_a_broken_normal):
            result = assess(observations, families=["normal", "gamma"])
        self.assertEqual(result.candidates[0].failure_code, "NOT_REPRESENTABLE")
        self.assertIsNone(result.candidates[1].failure_code)

    def test_a_sample_outside_the_support_is_reported_for_explicit_families(self) -> None:
        observations = [-1.0, 0.5, 1.5, 2.0, 3.0]
        result = assess(observations, families=["lognormal", "normal", "exponential", "gamma"])
        codes = {row.family.value: row.failure_code for row in result.candidates}
        self.assertEqual(
            codes,
            {
                "lognormal": "INVALID_SUPPORT",
                "normal": None,
                "exponential": "INVALID_SUPPORT",
                "gamma": "INVALID_SUPPORT",
            },
        )

    def test_a_sample_below_the_family_minimum_is_reported(self) -> None:
        result = assess([1.0, 2.5])
        codes = {row.family.value: row.failure_code for row in result.candidates}
        self.assertEqual(
            codes,
            {
                "normal": "SAMPLE_TOO_SMALL",
                "gamma": "SAMPLE_TOO_SMALL",
                "weibull_min": "SAMPLE_TOO_SMALL",
                "lognormal": "SAMPLE_TOO_SMALL",
                "gumbel_right": "SAMPLE_TOO_SMALL",
                "exponential": None,
            },
        )
        self.assertEqual(result.sample_size, 2)

    def test_a_single_observation_is_enough_for_exponential_only(self) -> None:
        result = assess([2.0], families=["normal", "exponential"])
        self.assertEqual(result.candidates[0].failure_code, "SAMPLE_TOO_SMALL")
        self.assertIsNone(result.candidates[1].failure_code)


class ArgumentValidationTests(unittest.TestCase):
    def test_statistic_replicates_rng_and_threshold(self) -> None:
        with self.assertRaises(TypeError):
            assess(statistic="AD")
        for bad in (0, -3, True, 2.0, None):
            with self.subTest(replicates=bad), self.assertRaises(ValueError):
                assess(replicates=bad)
        with self.assertRaises(TypeError):
            assess(rng=object())
        for bad in (float("nan"), -0.1, 1.5, True, 1, "0.05", None):
            with self.subTest(threshold=bad), self.assertRaises(ValueError):
                assess(adequacy_threshold=bad)

    def test_invalid_arguments_are_rejected_before_any_random_draw(self) -> None:
        used = np.random.default_rng(8)
        before = used.bit_generator.state
        with self.assertRaises(ValueError):
            assess(rng=used, adequacy_threshold=2.0)
        with self.assertRaises(ValueError):
            assess(rng=used, families=["normal", "normal"])
        self.assertEqual(used.bit_generator.state, before)

    def test_families_must_be_a_non_empty_duplicate_free_iterable(self) -> None:
        with self.assertRaises(TypeError):
            assess(families="normal")
        with self.assertRaises(ValueError):
            assess(families=[])
        with self.assertRaises(ValueError):
            assess(families=["normal", FamilyId.NORMAL])
        with self.assertRaises(ValueError):
            assess(families=["weibull", "weibull_min"])
        with self.assertRaises(ValueError):
            assess(families=["poisson"])
        with self.assertRaises(TypeError):
            assess(families=[3])

    def test_observations_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            assess([])
        for bad in (float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "must be finite"):
                assess([1.0, 2.0, bad])
        with self.assertRaisesRegex(ValueError, "must be finite"):
            assess([1.0, 2.0, 10**400])
        for censored in (
            [ExactLifetime(1.0), ExactLifetime(2.0), ExactLifetime(3.0)],
            [RightCensoredLifetime(1.0), 2.0, 3.0],
            [1.0, "2.0", 3.0],
        ):
            with self.subTest(observations=censored):
                with self.assertRaisesRegex(TypeError, "censored"):
                    assess(censored)

    def test_the_exponential_row_uses_the_exponential_cell(self) -> None:
        observations = positive_sample()
        result = assess(observations, families=["exponential"], rng=np.random.default_rng(12))
        direct = refit_monte_carlo_gof(
            observations=observations,
            family="exponential",
            statistics=AD_ONLY,
            replicates=9,
            rng=np.random.default_rng(12),
        )
        self.assertEqual(result.candidates[0].gof, direct)


if __name__ == "__main__":
    unittest.main()
