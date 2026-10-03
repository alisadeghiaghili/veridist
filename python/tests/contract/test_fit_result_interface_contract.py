"""Contracts for the common fit-result interface and the ``fit`` dispatcher."""

from __future__ import annotations

import random
import unittest
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

import veridist.families as families_package
from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue, RightCensoredValue
from veridist.families import (
    FitFailure,
    FitSuccess,
    fit,
    fit_exponential,
    fit_gamma,
    fit_gumbel_right,
    fit_lognormal,
    fit_normal,
    fit_weibull,
)
from veridist.families import dispatch as dispatch_module
from veridist.families.registry import FAMILY_REGISTRY, FamilyId, Operation

LIFETIME_FAMILIES = (
    FamilyId.EXPONENTIAL,
    FamilyId.WEIBULL_MIN,
    FamilyId.LOGNORMAL,
    FamilyId.GAMMA,
)
REAL_LINE_FAMILIES = (FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT)
FITS = {
    FamilyId.EXPONENTIAL: fit_exponential,
    FamilyId.WEIBULL_MIN: fit_weibull,
    FamilyId.LOGNORMAL: fit_lognormal,
    FamilyId.GAMMA: fit_gamma,
    FamilyId.NORMAL: fit_normal,
    FamilyId.GUMBEL_RIGHT: fit_gumbel_right,
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


class CommonResultInterfaceTests(unittest.TestCase):
    def test_every_family_has_a_fit_and_the_dispatch_table_matches_the_registry(self) -> None:
        self.assertEqual(set(FITS), set(FamilyId))
        for spec in FAMILY_REGISTRY.list():
            self.assertTrue(spec.supports(Operation.FIT))
        self.assertEqual(set(dispatch_module._FITTERS), set(FamilyId))
        self.assertEqual(set(dispatch_module._OPTIONS), set(FamilyId))

    def test_verify_fitters_rejects_any_mismatch(self) -> None:
        registry = FAMILY_REGISTRY.families
        fitters = dict(dispatch_module._FITTERS)
        options = dict(dispatch_module._OPTIONS)
        dispatch_module._verify_fitters(registry, fitters, options)
        missing_fitter = {k: v for k, v in fitters.items() if k is not FamilyId.GAMMA}
        missing_options = {k: v for k, v in options.items() if k is not FamilyId.GAMMA}
        with self.assertRaises(RuntimeError):
            dispatch_module._verify_fitters(registry, missing_fitter, options)
        with self.assertRaises(RuntimeError):
            dispatch_module._verify_fitters(registry, fitters, missing_options)

    def test_every_success_satisfies_fit_success_and_not_fit_failure(self) -> None:
        for family in FamilyId:
            for censored in (False, True):
                with self.subTest(family=family, censored=censored):
                    result = fit(family, _observations(family, censored=censored))
                    self.assertIsInstance(result, FitSuccess)
                    self.assertNotIsInstance(result, FitFailure)
                    assert isinstance(result, FitSuccess)
                    self.assertIs(type(result.family), FamilyId)
                    self.assertIs(result.family, family)
                    self.assertEqual(result.observation_count, 30)
                    self.assertEqual(
                        result.event_count + result.censored_count, result.observation_count
                    )
                    self.assertEqual(result.censored_count > 0, censored)
                    self.assertIs(result.converged, True)
                    self.assertIs(type(result.log_likelihood), float)

    def test_every_failure_satisfies_fit_failure_and_not_fit_success(self) -> None:
        for family in FamilyId:
            censored_kind = (
                RightCensoredLifetime if family in LIFETIME_FAMILIES else RightCensoredValue
            )
            failures = {
                "EMPTY_SAMPLE": [],
                "NO_OBSERVED_EVENTS": [censored_kind(2.0), censored_kind(3.0)],
            }
            for expected_code, observations in failures.items():
                with self.subTest(family=family, code=expected_code):
                    result = fit(family, observations)
                    self.assertIsInstance(result, FitFailure)
                    self.assertNotIsInstance(result, FitSuccess)
                    assert isinstance(result, FitFailure)
                    self.assertIs(result.family, family)
                    self.assertIsInstance(result.code, StrEnum)
                    self.assertEqual(result.code.value, expected_code)
                    self.assertEqual(result.observation_count, len(observations))
                    self.assertEqual(result.event_count, 0)
                    self.assertEqual(result.censored_count, len(observations))

    def test_parameters_are_read_only_with_canonical_registry_names_in_order(self) -> None:
        for family in FamilyId:
            result = fit(family, _observations(family, censored=False))
            assert isinstance(result, FitSuccess)
            with self.subTest(family=family):
                canonical = tuple(p.name for p in FAMILY_REGISTRY.families[family].parameters)
                self.assertIsInstance(result.parameters, MappingProxyType)
                self.assertEqual(tuple(result.parameters), canonical)
                for name, value in result.parameters.items():
                    self.assertIs(type(value), float)
                    self.assertEqual(getattr(result, name), value)
                with self.assertRaises(TypeError):
                    result.parameters[canonical[0]] = 1.0  # type: ignore[index]
                # The mapping is a valid input for the registry (names and roles).
                FAMILY_REGISTRY.families[family].validate_parameters(**result.parameters)

    def test_family_specific_attributes_remain_for_compatibility(self) -> None:
        expected = {
            FamilyId.EXPONENTIAL: ("rate",),
            FamilyId.WEIBULL_MIN: ("shape", "scale"),
            FamilyId.LOGNORMAL: ("mu_log", "sigma_log"),
            FamilyId.GAMMA: ("shape", "scale"),
            FamilyId.NORMAL: ("mu", "sigma"),
            FamilyId.GUMBEL_RIGHT: ("location", "scale"),
        }
        for family, names in expected.items():
            result = fit(family, _observations(family, censored=False))
            with self.subTest(family=family):
                self.assertEqual(tuple(result.parameters), names)  # type: ignore[union-attr]

    def test_family_is_a_family_id_and_still_equals_its_string_value(self) -> None:
        for family in LIFETIME_FAMILIES:
            result = fit(family, _observations(family, censored=False))
            self.assertIs(type(result.family), FamilyId)  # type: ignore[union-attr]
            self.assertEqual(result.family, family.value)  # type: ignore[union-attr]

    def test_unsupported_result_construction_is_rejected(self) -> None:
        from veridist.families.lognormal import LognormalFitSuccess
        from veridist.families.weibull import WeibullFitSuccess

        with self.assertRaises(ValueError):
            WeibullFitSuccess(1.0, 1.0, -1.0, 2, 2, 0, family="lognormal")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LognormalFitSuccess(0.0, 1.0, -1.0, 2, 2, 0, family="weibull_min")  # type: ignore[arg-type]
        self.assertIs(
            WeibullFitSuccess(1.0, 1.0, -1.0, 2, 2, 0, family="weibull_min").family,
            FamilyId.WEIBULL_MIN,
        )  # type: ignore[arg-type]
        self.assertIs(
            LognormalFitSuccess(0.0, 1.0, -1.0, 2, 2, 0, family="lognormal").family,
            FamilyId.LOGNORMAL,
        )  # type: ignore[arg-type]


class FitDispatcherTests(unittest.TestCase):
    def test_dispatch_returns_exactly_what_the_family_fit_returns(self) -> None:
        for family, function in FITS.items():
            for censored in (False, True):
                observations = _observations(family, censored=censored)
                with self.subTest(family=family, censored=censored):
                    self.assertEqual(fit(family, observations), function(observations))

    def test_family_may_be_given_as_its_string_value_or_alias(self) -> None:
        observations = _observations(FamilyId.WEIBULL_MIN, censored=True)
        expected = fit_weibull(observations)
        self.assertEqual(fit("weibull_min", observations), expected)
        self.assertEqual(fit("weibull", observations), expected)
        self.assertEqual(fit(FamilyId.WEIBULL_MIN, observations), expected)
        values = _observations(FamilyId.NORMAL, censored=False)
        self.assertEqual(fit("gaussian", values), fit_normal(values))
        self.assertEqual(fit("gumbel", values), fit_gumbel_right(values))

    def test_family_must_be_a_family_id_or_a_known_string(self) -> None:
        for bad in (None, 3, 1.5, b"normal", ["normal"], object()):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                fit(bad, [])  # type: ignore[arg-type]
        for name in ("", "Normal", "student_t", "weibull-min"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                fit(name, [])

    def test_options_are_forwarded_and_unknown_options_are_named(self) -> None:
        observations = _observations(FamilyId.WEIBULL_MIN, censored=False)
        self.assertEqual(
            fit("weibull", observations, fixed_shape=1.7),
            fit_weibull(observations, fixed_shape=1.7),
        )
        weights = [2] * len(observations)
        self.assertEqual(
            fit(
                FamilyId.GAMMA,
                _observations(FamilyId.GAMMA, censored=False),
                frequency_weights=weights,
            ),
            fit_gamma(_observations(FamilyId.GAMMA, censored=False), frequency_weights=weights),
        )
        for family in FamilyId:
            data = _observations(family, censored=False)
            with self.subTest(family=family):
                with self.assertRaises(TypeError) as caught:
                    fit(family, data, bandwidth=3)
                self.assertIn("bandwidth", str(caught.exception))
                self.assertIn(family.value, str(caught.exception))
        with self.assertRaises(TypeError) as caught:
            fit(FamilyId.NORMAL, [], fixed_shape=1.0, zeta=2)
        self.assertIn("fixed_shape", str(caught.exception))
        self.assertIn("zeta", str(caught.exception))
        with self.assertRaises(TypeError) as caught:
            fit(
                FamilyId.EXPONENTIAL,
                _observations(FamilyId.EXPONENTIAL, censored=False),
                censoring="right",
            )
        self.assertIn("censoring", str(caught.exception))

    def test_observations_must_be_the_type_pair_valid_for_the_family(self) -> None:
        lifetimes = [ExactLifetime(1.0), ExactLifetime(2.0), RightCensoredLifetime(3.0)]
        values = [ExactValue(1.0), ExactValue(2.0), RightCensoredValue(3.0)]
        for family in LIFETIME_FAMILIES:
            with self.subTest(family=family), self.assertRaises(TypeError):
                fit(family, values)
        for family in REAL_LINE_FAMILIES:
            with self.subTest(family=family), self.assertRaises(TypeError):
                fit(family, lifetimes)
        for family in FamilyId:
            with self.subTest(family=family, kind="junk"), self.assertRaises(TypeError):
                fit(family, [1.0, 2.0])

    def test_dispatcher_is_exported_from_the_families_package(self) -> None:
        self.assertIs(families_package.fit, fit)
        for name in (
            "fit",
            "fit_gamma",
            "fit_normal",
            "fit_gumbel_right",
            "FitSuccess",
            "FitFailure",
        ):
            self.assertIn(name, families_package.__all__)
        self.assertIsInstance(FITS, Mapping)


if __name__ == "__main__":
    unittest.main()
