"""Contracts for fit uncertainty: the shared object, typed absence, validation, derived values."""

from __future__ import annotations

import dataclasses
import math
import random
import unittest
from collections.abc import Mapping
from unittest import mock

import numpy as np

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue, RightCensoredValue
from veridist.families import (
    DerivedEstimate,
    FitSuccess,
    FitUncertainty,
    ParameterInterval,
    UncertaintyUnavailable,
    UncertaintyUnavailableReason,
    fit,
    fit_exponential,
    fit_weibull,
)
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.families.weibull import WeibullFitSuccess
from veridist.statistics.distributions import ppf, sf

LIFETIME_FAMILIES = (
    FamilyId.EXPONENTIAL,
    FamilyId.WEIBULL_MIN,
    FamilyId.LOGNORMAL,
    FamilyId.GAMMA,
)
MEANS = {
    FamilyId.EXPONENTIAL: lambda p: 1.0 / p["rate"],
    FamilyId.WEIBULL_MIN: lambda p: p["scale"] * math.gamma(1.0 + 1.0 / p["shape"]),
    FamilyId.LOGNORMAL: lambda p: math.exp(p["mu_log"] + 0.5 * p["sigma_log"] ** 2),
    FamilyId.GAMMA: lambda p: p["shape"] * p["scale"],
    FamilyId.NORMAL: lambda p: p["mu"],
    FamilyId.GUMBEL_RIGHT: lambda p: p["location"] + 0.5772156649015329 * p["scale"],
}


def _observations(family: FamilyId, *, censored: bool) -> list[object]:
    rng = random.Random(41)
    exact_kind, censored_kind = (
        (ExactLifetime, RightCensoredLifetime)
        if family in LIFETIME_FAMILIES
        else (ExactValue, RightCensoredValue)
    )
    values = [rng.uniform(1.0, 9.0) for _ in range(30)]
    cutoff = 7.0 if censored else 1e9
    return [exact_kind(v) if v < cutoff else censored_kind(cutoff) for v in values]


def _uncertainty(family: FamilyId, *, censored: bool = False) -> tuple[FitSuccess, FitUncertainty]:
    result = fit(family, _observations(family, censored=censored))
    assert isinstance(result, FitSuccess)
    uncertainty = result.uncertainty
    assert isinstance(uncertainty, FitUncertainty)
    return result, uncertainty


class UncertaintyObjectContract(unittest.TestCase):
    def test_every_success_exposes_uncertainty_in_canonical_order(self) -> None:
        for family in FamilyId:
            for censored in (False, True):
                with self.subTest(family=family, censored=censored):
                    result, uncertainty = _uncertainty(family, censored=censored)
                    names = tuple(
                        parameter.name for parameter in FAMILY_REGISTRY.lookup(family).parameters
                    )
                    self.assertEqual(uncertainty.family, family)
                    self.assertEqual(uncertainty.parameter_names, names)
                    self.assertEqual(tuple(result.parameters), names)
                    self.assertEqual(uncertainty.estimates, tuple(result.parameters.values()))
                    self.assertEqual(tuple(uncertainty.standard_errors), names)

    def test_covariance_is_symmetric_positive_definite_tuples(self) -> None:
        for family in FamilyId:
            with self.subTest(family=family):
                _, uncertainty = _uncertainty(family, censored=True)
                covariance = uncertainty.covariance
                size = len(uncertainty.parameter_names)
                self.assertIsInstance(covariance, tuple)
                self.assertEqual(len(covariance), size)
                for row in covariance:
                    self.assertIsInstance(row, tuple)
                    self.assertEqual(len(row), size)
                for i in range(size):
                    self.assertGreater(covariance[i][i], 0.0)
                    for j in range(size):
                        self.assertEqual(covariance[i][j], covariance[j][i])
                if size == 2:
                    self.assertGreater(
                        covariance[0][0] * covariance[1][1] - covariance[0][1] ** 2, 0.0
                    )

    def test_standard_errors_are_the_covariance_diagonal_roots(self) -> None:
        for family in FamilyId:
            with self.subTest(family=family):
                _, uncertainty = _uncertainty(family)
                for index, name in enumerate(uncertainty.parameter_names):
                    self.assertEqual(
                        uncertainty.standard_errors[name],
                        math.sqrt(uncertainty.covariance[index][index]),
                    )

    def test_standard_errors_and_intervals_are_read_only_mappings(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.WEIBULL_MIN)
        for mapping in (uncertainty.standard_errors, uncertainty.confidence_intervals()):
            self.assertIsInstance(mapping, Mapping)
            with self.assertRaises(TypeError):
                mapping["shape"] = 1.0  # type: ignore[index]

    def test_the_uncertainty_object_is_immutable(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.NORMAL)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            uncertainty.estimates = (0.0, 1.0)  # type: ignore[misc]

    def test_uncertainty_is_computed_once_per_result(self) -> None:
        result = fit_weibull(_observations(FamilyId.WEIBULL_MIN, censored=True))
        assert isinstance(result, FitSuccess)
        import veridist.families.uncertainty as module

        with mock.patch.object(
            module, "compute_uncertainty", wraps=module.compute_uncertainty
        ) as spy:
            first = result.uncertainty
            second = result.uncertainty
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(first, second)

    def test_a_failure_has_no_uncertainty_and_is_not_a_success(self) -> None:
        failure = fit(FamilyId.WEIBULL_MIN, [])
        self.assertNotIsInstance(failure, FitSuccess)
        self.assertFalse(hasattr(failure, "uncertainty"))

    def test_frequency_weights_equal_repeated_observations(self) -> None:
        values = [ExactLifetime(t) for t in (2.0, 3.5, 5.0, 8.0, 9.5)] + [
            RightCensoredLifetime(7.0)
        ]
        weights = [2, 1, 3, 1, 2, 2]
        weighted = fit(FamilyId.WEIBULL_MIN, values, frequency_weights=weights)
        repeated = fit(
            FamilyId.WEIBULL_MIN,
            [v for v, w in zip(values, weights, strict=True) for _ in range(w)],
        )
        assert isinstance(weighted, FitSuccess) and isinstance(repeated, FitSuccess)
        left, right = weighted.uncertainty, repeated.uncertainty
        assert isinstance(left, FitUncertainty) and isinstance(right, FitUncertainty)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(
                    left.covariance[i][j],
                    right.covariance[i][j],
                    delta=1e-7 * abs(right.covariance[i][j]) + 1e-12,
                )


class IntervalContract(unittest.TestCase):
    def test_intervals_bracket_the_estimate_and_widen_with_the_level(self) -> None:
        for family in FamilyId:
            for method in ("wald", "profile"):
                with self.subTest(family=family, method=method):
                    result, uncertainty = _uncertainty(family, censored=True)
                    narrow = uncertainty.confidence_intervals(0.8, method)
                    wide = uncertainty.confidence_intervals(0.99, method)
                    for name, estimate in result.parameters.items():
                        self.assertLess(wide[name][0], narrow[name][0])
                        self.assertLess(narrow[name][0], estimate)
                        self.assertLess(estimate, narrow[name][1])
                        self.assertLess(narrow[name][1], wide[name][1])

    def test_positive_parameters_never_get_a_non_positive_bound(self) -> None:
        for family in FamilyId:
            spec = FAMILY_REGISTRY.lookup(family)
            for method in ("wald", "profile"):
                _, uncertainty = _uncertainty(family, censored=True)
                intervals = uncertainty.confidence_intervals(0.999, method)
                for parameter in spec.parameters:
                    if parameter.role.value == "positive":
                        self.assertGreater(intervals[parameter.name][0], 0.0, (family, method))

    def test_interval_details_carry_method_level_and_flags(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.GAMMA, censored=True)
        for method in ("wald", "profile"):
            details = uncertainty.interval_details(0.9, method)
            intervals = uncertainty.confidence_intervals(0.9, method)
            for name, detail in details.items():
                self.assertIsInstance(detail, ParameterInterval)
                self.assertEqual((detail.method, detail.level), (method, 0.9))
                self.assertEqual((detail.lower, detail.upper), intervals[name])
                self.assertFalse(detail.lower_unbounded)
                self.assertFalse(detail.upper_unbounded)

    def test_default_arguments_are_the_95_percent_wald_interval(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.LOGNORMAL)
        self.assertEqual(
            dict(uncertainty.confidence_intervals()),
            dict(uncertainty.confidence_intervals(0.95, "wald")),
        )

    def test_invalid_level_and_method_are_rejected(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.NORMAL)
        for level in (0.0, 1.0, -0.1, 1.5, math.nan, math.inf):
            with self.subTest(level=level), self.assertRaises(ValueError):
                uncertainty.confidence_intervals(level)
        for level in (True, "0.95", None):
            with self.subTest(level=level), self.assertRaises(TypeError):
                uncertainty.confidence_intervals(level)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            uncertainty.confidence_intervals(0.95, "bootstrap")
        with self.assertRaises(TypeError):
            uncertainty.confidence_intervals(0.95, 1)  # type: ignore[arg-type]

    def test_numpy_scalar_levels_are_accepted(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.NORMAL)
        self.assertEqual(
            dict(uncertainty.confidence_intervals(np.float64(0.9))),
            dict(uncertainty.confidence_intervals(0.9)),
        )

    def test_a_profile_side_that_does_not_cross_is_reported_unbounded(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.WEIBULL_MIN)
        import veridist.families.uncertainty as module

        original = module._solve_side
        sides: list[float] = []

        def never_crossing_upper(
            curve: object, direction: float, target: float, maximum: float
        ) -> float | None:
            sides.append(direction)
            return None if direction > 0 else original(curve, direction, target, maximum)  # type: ignore[arg-type]

        with mock.patch.object(module, "_solve_side", never_crossing_upper):
            details = uncertainty.interval_details(0.95, "profile")
            intervals = uncertainty.confidence_intervals(0.95, "profile")
        # Two parameters, lower side first, for each of the two calls above.
        self.assertEqual(sides, [-1.0, 1.0] * 4)
        for name, detail in details.items():
            self.assertFalse(detail.lower_unbounded)
            self.assertTrue(detail.upper_unbounded)
            self.assertEqual(detail.upper, math.inf)
            self.assertEqual(intervals[name][1], math.inf)

    def test_a_lower_side_that_does_not_cross_is_zero_or_minus_infinity(self) -> None:
        for family, positive, location in (
            (FamilyId.WEIBULL_MIN, "scale", None),
            (FamilyId.NORMAL, "sigma", "mu"),
        ):
            _, uncertainty = _uncertainty(family)
            import veridist.families.uncertainty as module

            original = module._solve_side

            def never_crossing_lower(
                curve: object,
                direction: float,
                target: float,
                maximum: float,
                original: object = original,
            ) -> float | None:
                return None if direction < 0 else original(curve, direction, target, maximum)  # type: ignore[operator]

            with mock.patch.object(module, "_solve_side", never_crossing_lower):
                details = uncertainty.interval_details(0.95, "profile")
            self.assertEqual(details[positive].lower, 0.0)
            self.assertTrue(details[positive].lower_unbounded)
            if location is not None:
                self.assertEqual(details[location].lower, -math.inf)


class ExactIntervalContract(unittest.TestCase):
    def test_exact_interval_for_uncensored_exponential_data(self) -> None:
        result = fit_exponential([ExactLifetime(t) for t in (1.0, 2.0, 4.0, 8.0, 3.0)])
        assert isinstance(result, FitSuccess)
        uncertainty = result.uncertainty
        assert isinstance(uncertainty, FitUncertainty)
        interval = uncertainty.confidence_intervals(0.95, "exact")["rate"]
        # 2 * rate * T ~ chi-square(2 r) with r = 5 events and T = 18:
        # scipy.stats.chi2.ppf(0.025, 10) = 3.2469727802368413,
        # scipy.stats.chi2.ppf(0.975, 10) = 20.483177350807388.
        self.assertAlmostEqual(interval[0], 3.2469727802368413 / 36.0, delta=1e-12)
        self.assertAlmostEqual(interval[1], 20.483177350807388 / 36.0, delta=1e-12)
        details = uncertainty.interval_details(0.95, "exact")
        self.assertEqual(tuple(details), ("rate",))
        self.assertEqual(details["rate"].method, "exact")

    def test_exact_interval_with_censoring_explains_why_it_is_unavailable(self) -> None:
        result = fit_exponential(
            [ExactLifetime(1.0), ExactLifetime(4.0), RightCensoredLifetime(6.0)]
        )
        assert isinstance(result, FitSuccess)
        uncertainty = result.uncertainty
        assert isinstance(uncertainty, FitUncertainty)
        with self.assertRaises(ValueError) as caught:
            uncertainty.confidence_intervals(0.95, "exact")
        self.assertIn("censor", str(caught.exception))
        with self.assertRaises(ValueError):
            uncertainty.mean(method="exact")

    def test_exact_interval_exists_only_for_the_exponential_family(self) -> None:
        for family in FamilyId:
            if family is FamilyId.EXPONENTIAL:
                continue
            _, uncertainty = _uncertainty(family)
            with self.subTest(family=family), self.assertRaises(ValueError):
                uncertainty.confidence_intervals(0.95, "exact")

    def test_exact_derived_intervals_are_the_decreasing_images_of_the_rate_interval(self) -> None:
        result = fit_exponential([ExactLifetime(t) for t in (1.0, 2.0, 4.0, 8.0, 3.0)])
        assert isinstance(result, FitSuccess)
        uncertainty = result.uncertainty
        assert isinstance(uncertainty, FitUncertainty)
        low, high = uncertainty.confidence_intervals(0.9, "exact")["rate"]
        mean = uncertainty.mean(0.9, "exact")
        self.assertEqual((mean.lower, mean.upper), (1.0 / high, 1.0 / low))
        quantile = uncertainty.quantile(0.1, 0.9, "exact")
        scale = -math.log1p(-0.1)
        self.assertAlmostEqual(quantile.lower, scale / high, delta=1e-12)
        self.assertAlmostEqual(quantile.upper, scale / low, delta=1e-12)
        survival = uncertainty.survival(2.0, 0.9, "exact")
        self.assertAlmostEqual(survival.lower, math.exp(-high * 2.0), delta=1e-12)
        self.assertAlmostEqual(survival.upper, math.exp(-low * 2.0), delta=1e-12)
        self.assertEqual(survival.method, "exact")


class DerivedQuantityContract(unittest.TestCase):
    def test_point_estimates_are_the_family_functions_at_the_estimate(self) -> None:
        for family in FamilyId:
            with self.subTest(family=family):
                result, uncertainty = _uncertainty(family, censored=True)
                parameters = dict(result.parameters)
                self.assertAlmostEqual(
                    uncertainty.mean().estimate,
                    MEANS[family](parameters),
                    delta=1e-12 * abs(MEANS[family](parameters)),
                )
                self.assertEqual(uncertainty.quantile(0.1).estimate, ppf(family, 0.1, **parameters))
                self.assertEqual(uncertainty.survival(5.0).estimate, sf(family, 5.0, **parameters))

    def test_results_are_frozen_with_method_and_level(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.WEIBULL_MIN)
        for estimate in (
            uncertainty.mean(0.9),
            uncertainty.quantile(0.1, 0.9),
            uncertainty.survival(5.0, 0.9),
        ):
            self.assertIsInstance(estimate, DerivedEstimate)
            self.assertEqual((estimate.method, estimate.level), ("wald", 0.9))
            self.assertLess(estimate.lower, estimate.estimate)
            self.assertLess(estimate.estimate, estimate.upper)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                estimate.estimate = 0.0  # type: ignore[misc]

    def test_b10_is_the_zero_point_one_quantile(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.WEIBULL_MIN)
        self.assertEqual(uncertainty.quantile(0.1), uncertainty.quantile(probability=0.1))

    def test_positive_quantities_get_positive_bounds_and_probabilities_stay_inside_zero_one(
        self,
    ) -> None:
        for family in (
            FamilyId.EXPONENTIAL,
            FamilyId.WEIBULL_MIN,
            FamilyId.LOGNORMAL,
            FamilyId.GAMMA,
        ):
            _, uncertainty = _uncertainty(family, censored=True)
            self.assertGreater(uncertainty.mean(0.999).lower, 0.0)
            self.assertGreater(uncertainty.quantile(0.01, 0.999).lower, 0.0)
        for family in FamilyId:
            _, uncertainty = _uncertainty(family, censored=True)
            survival = uncertainty.survival(5.0, 0.999)
            self.assertGreater(survival.lower, 0.0)
            self.assertLess(survival.upper, 1.0)

    def test_real_line_quantities_use_the_identity_scale(self) -> None:
        result, uncertainty = _uncertainty(FamilyId.NORMAL)
        z = 1.959963984540054
        mean = uncertainty.mean()
        self.assertAlmostEqual(
            mean.estimate - mean.lower, z * uncertainty.standard_errors["mu"], delta=1e-9
        )
        self.assertAlmostEqual(
            mean.upper - mean.estimate, z * uncertainty.standard_errors["mu"], delta=1e-9
        )
        self.assertEqual(result.parameters["mu"], mean.estimate)

    def test_argument_validation(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.WEIBULL_MIN)
        for bad in (0.0, 1.0, -0.5, 2.0, math.nan):
            with self.subTest(probability=bad), self.assertRaises(ValueError):
                uncertainty.quantile(bad)
        for bad in (True, "0.1", None):
            with self.subTest(probability=bad), self.assertRaises(TypeError):
                uncertainty.quantile(bad)  # type: ignore[arg-type]
        for bad in (math.inf, -math.inf, math.nan):
            with self.subTest(time=bad), self.assertRaises(ValueError):
                uncertainty.survival(bad)
        with self.assertRaises(TypeError):
            uncertainty.survival(True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            uncertainty.mean(0.95, "bootstrap")
        with self.assertRaises(ValueError):
            uncertainty.mean(1.2)

    def test_survival_outside_the_support_or_underflowing_is_a_degenerate_interval(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.WEIBULL_MIN)
        for method in ("wald", "profile"):
            before = uncertainty.survival(-1.0, method=method)
            self.assertEqual((before.estimate, before.lower, before.upper), (1.0, 1.0, 1.0))
            far = uncertainty.survival(1e300, method=method)
            self.assertEqual((far.estimate, far.lower, far.upper), (0.0, 0.0, 0.0))

    def test_profile_intervals_are_available_for_exponential_and_weibull(self) -> None:
        for family in (FamilyId.EXPONENTIAL, FamilyId.WEIBULL_MIN):
            _, uncertainty = _uncertainty(family, censored=True)
            for estimate in (
                uncertainty.mean(method="profile"),
                uncertainty.quantile(0.1, method="profile"),
                uncertainty.survival(5.0, method="profile"),
            ):
                self.assertEqual(estimate.method, "profile")
                self.assertLess(estimate.lower, estimate.estimate)
                self.assertLess(estimate.estimate, estimate.upper)
                self.assertFalse(estimate.lower_unbounded or estimate.upper_unbounded)

    def test_profile_elsewhere_raises_not_implemented_and_names_the_alternative(self) -> None:
        for family in (FamilyId.LOGNORMAL, FamilyId.GAMMA, FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT):
            _, uncertainty = _uncertainty(family)
            for call in (
                lambda: uncertainty.mean(method="profile"),
                lambda: uncertainty.quantile(0.1, method="profile"),
                lambda: uncertainty.survival(5.0, method="profile"),
            ):
                with self.subTest(family=family), self.assertRaises(NotImplementedError) as caught:
                    call()
                self.assertIn("wald", str(caught.exception))
                self.assertIn(family.value, str(caught.exception))

    def test_weibull_profile_of_the_mean_is_wider_than_zero_and_contains_the_estimate(self) -> None:
        _, uncertainty = _uncertainty(FamilyId.WEIBULL_MIN)
        mean = uncertainty.mean(method="profile")
        self.assertLess(mean.lower, mean.estimate)
        self.assertGreater(mean.upper, mean.estimate)


class UncertaintyUnavailableContract(unittest.TestCase):
    def test_a_fixed_weibull_shape_has_no_uncertainty(self) -> None:
        result = fit_weibull(_observations(FamilyId.WEIBULL_MIN, censored=True), fixed_shape=1.7)
        assert isinstance(result, FitSuccess)
        unavailable = result.uncertainty
        self.assertIsInstance(unavailable, UncertaintyUnavailable)
        assert isinstance(unavailable, UncertaintyUnavailable)
        self.assertIs(unavailable.reason, UncertaintyUnavailableReason.FIXED_PARAMETER)
        self.assertIn("shape", unavailable.detail)

    def test_a_result_built_without_a_fit_has_no_data(self) -> None:
        result = WeibullFitSuccess(1.5, 4.0, -10.0, 5, 4, 1)
        unavailable = result.uncertainty
        assert isinstance(unavailable, UncertaintyUnavailable)
        self.assertIs(unavailable.reason, UncertaintyUnavailableReason.NO_DATA)

    def test_parameters_that_are_not_a_maximum_give_a_typed_reason_not_an_exception(self) -> None:
        template = fit_weibull(_observations(FamilyId.WEIBULL_MIN, censored=True))
        assert isinstance(template, WeibullFitSuccess)
        far_scale = dataclasses.replace(template, scale=1e6)
        unavailable = far_scale.uncertainty
        assert isinstance(unavailable, UncertaintyUnavailable)
        self.assertIs(unavailable.reason, UncertaintyUnavailableReason.NOT_POSITIVE_DEFINITE)
        overflowing = dataclasses.replace(template, shape=1e6)
        unavailable = overflowing.uncertainty
        assert isinstance(unavailable, UncertaintyUnavailable)
        self.assertIs(unavailable.reason, UncertaintyUnavailableReason.NOT_COMPUTABLE)

    def test_the_cache_follows_the_parameters_of_the_result(self) -> None:
        template = fit_weibull(_observations(FamilyId.WEIBULL_MIN, censored=True))
        assert isinstance(template, WeibullFitSuccess)
        changed = dataclasses.replace(template, scale=template.scale * 1.01)
        original = template.uncertainty
        other = changed.uncertainty
        assert isinstance(original, FitUncertainty) and isinstance(other, FitUncertainty)
        self.assertNotEqual(original.estimates, other.estimates)
        self.assertEqual(template.uncertainty, original)

    def test_reasons_are_stable_strings(self) -> None:
        self.assertEqual(
            [reason.value for reason in UncertaintyUnavailableReason],
            [
                "NO_DATA",
                "FIXED_PARAMETER",
                "SINGULAR_INFORMATION",
                "NOT_POSITIVE_DEFINITE",
                "NOT_COMPUTABLE",
            ],
        )


if __name__ == "__main__":
    unittest.main()
