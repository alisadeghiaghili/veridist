"""Boundary probes for the registry lookup and the shared fit machinery."""

from __future__ import annotations

import unittest
import warnings

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue, RightCensoredValue
from veridist.families._reliability import (
    admitted_observations,
    admitted_real_observations,
    is_degenerate_sample,
    maximize_nested,
)
from veridist.families.registry import FAMILY_REGISTRY, FamilyId, Operation, Support
from veridist.statistics.distributions import cdf


class RegistryLookupProbeTests(unittest.TestCase):
    def test_lookup_accepts_enum_value_and_alias_and_returns_the_same_spec(self) -> None:
        for family in FamilyId:
            spec = FAMILY_REGISTRY.families[family]
            self.assertIs(FAMILY_REGISTRY.lookup(family), spec)
            self.assertIs(FAMILY_REGISTRY.lookup(family.value), spec)
        self.assertIs(FAMILY_REGISTRY.lookup("gaussian"), FAMILY_REGISTRY.families[FamilyId.NORMAL])

    def test_lookup_rejects_other_types_and_unknown_names(self) -> None:
        for bad in (None, 1, b"normal", ["normal"]):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                FAMILY_REGISTRY.lookup(bad)
        with self.assertRaises(ValueError):
            FAMILY_REGISTRY.lookup("nope")

    def test_operations_support_and_exponential_membership(self) -> None:
        self.assertEqual(
            {operation.value for operation in Operation},
            {"logpdf", "cdf", "sf", "ppf", "sample", "fit"},
        )
        self.assertEqual(
            {support.value for support in Support},
            {"real_line", "positive", "non_negative"},
        )
        self.assertEqual(len(FAMILY_REGISTRY.list()), 6)
        self.assertEqual(FamilyId("exponential"), FamilyId.EXPONENTIAL)

    def test_legacy_aliases_that_once_raised_assertion_error_now_resolve(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            self.assertEqual(cdf("gaussian", 0.0, {"mu": 0.0, "sigma": 1.0}), 0.5)
        self.assertEqual(
            cdf("gumbel", 0.0, location=0.0, scale=1.0),
            cdf(FamilyId.GUMBEL_RIGHT, 0.0, location=0.0, scale=1.0),
        )
        self.assertEqual(
            cdf("weibull", 2.0, shape=2.0, scale=2.0),
            cdf(FamilyId.WEIBULL_MIN, 2.0, shape=2.0, scale=2.0),
        )


class SharedMachineryProbeTests(unittest.TestCase):
    def test_degenerate_rule_boundaries(self) -> None:
        self.assertTrue(is_degenerate_sample([2.0], []))
        self.assertTrue(is_degenerate_sample([2.0, 2.0], [2.0, 1.0]))
        self.assertFalse(is_degenerate_sample([2.0, 2.0], [2.0000001]))
        self.assertFalse(is_degenerate_sample([2.0, 3.0], [0.0]))

    def test_nested_maximization_finds_an_interior_optimum_and_reports_boundaries(self) -> None:
        def at_outer(outer: float):  # type: ignore[no-untyped-def]
            return lambda inner: -((outer - 1.5) ** 2) - (inner + 0.25 * outer - 2.0) ** 2

        outer, inner, value, boundary = maximize_nested(
            at_outer,
            outer_bounds=(-6.0, 6.0),
            outer_hard_limits=(-20.0, 20.0),
            inner_bounds=(-6.0, 6.0),
            inner_hard_limits=(-20.0, 20.0),
        )
        self.assertFalse(boundary)
        self.assertAlmostEqual(outer, 1.5, places=6)
        self.assertAlmostEqual(inner, 2.0 - 0.375, places=6)
        self.assertAlmostEqual(value, 0.0, places=10)

        def rising(outer: float):  # type: ignore[no-untyped-def]
            return lambda inner: outer + 0.0 * inner

        *_, boundary = maximize_nested(
            rising,
            outer_bounds=(-6.0, 6.0),
            outer_hard_limits=(-20.0, 20.0),
            inner_bounds=(-6.0, 6.0),
            inner_hard_limits=(-20.0, 20.0),
        )
        self.assertTrue(boundary)

        def drifting_inner(outer: float):  # type: ignore[no-untyped-def]
            return lambda inner: inner - 0.0 * outer

        *_, boundary = maximize_nested(
            drifting_inner,
            outer_bounds=(-6.0, 6.0),
            outer_hard_limits=(-20.0, 20.0),
            inner_bounds=(-6.0, 6.0),
            inner_hard_limits=(-20.0, 20.0),
        )
        self.assertTrue(boundary)

    def test_each_admission_function_accepts_only_its_own_pair(self) -> None:
        options = {
            "frequency_weights": None,
            "analytic_weights": None,
            "censoring": "right",
            "truncation": None,
        }
        lifetimes = (ExactLifetime(1.0), RightCensoredLifetime(2.0))
        values = (ExactValue(1.0), RightCensoredValue(2.0))
        self.assertEqual(admitted_observations(lifetimes, **options), lifetimes)  # type: ignore[arg-type]
        self.assertEqual(admitted_real_observations(values, **options), values)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            admitted_observations(values, **options)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            admitted_real_observations(lifetimes, **options)  # type: ignore[arg-type]
        self.assertEqual(
            admitted_real_observations(values, **{**options, "frequency_weights": [2, 1]}),  # type: ignore[arg-type]
            (values[0], values[0], values[1]),
        )


if __name__ == "__main__":
    unittest.main()


class NewFitResultValidationTests(unittest.TestCase):
    def test_success_classes_reject_non_finite_or_non_positive_estimates(self) -> None:
        from math import inf, nan

        from veridist.families.gamma import GammaFitSuccess
        from veridist.families.gumbel import GumbelFitSuccess
        from veridist.families.normal import NormalFitSuccess

        counts = (3, 3, 0)
        for bad in (0.0, -1.0, nan, inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    NormalFitSuccess(0.0, bad, -1.0, *counts)
                with self.assertRaises(ValueError):
                    GumbelFitSuccess(0.0, bad, -1.0, *counts)
                with self.assertRaises(ValueError):
                    GammaFitSuccess(bad, 1.0, -1.0, *counts)
                with self.assertRaises(ValueError):
                    GammaFitSuccess(1.0, bad, -1.0, *counts)
        for bad in (nan, inf, -inf):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    NormalFitSuccess(bad, 1.0, -1.0, *counts)
                with self.assertRaises(ValueError):
                    NormalFitSuccess(0.0, 1.0, bad, *counts)
                with self.assertRaises(ValueError):
                    GumbelFitSuccess(bad, 1.0, -1.0, *counts)
                with self.assertRaises(ValueError):
                    GumbelFitSuccess(0.0, 1.0, bad, *counts)
                with self.assertRaises(ValueError):
                    GammaFitSuccess(1.0, 1.0, bad, *counts)
        good = NormalFitSuccess(-2.0, 0.5, -1.0, *counts)
        self.assertEqual(dict(good.parameters), {"mu": -2.0, "sigma": 0.5})
        self.assertIs(good.family, FamilyId.NORMAL)
        self.assertEqual(
            dict(GammaFitSuccess(1.5, 2.0, -1.0, *counts).parameters), {"shape": 1.5, "scale": 2.0}
        )


class NewFitInternalsTests(unittest.TestCase):
    def test_exp_or_inf_never_raises(self) -> None:
        from math import exp, inf

        from veridist.families._reliability import exp_or_inf

        self.assertEqual(exp_or_inf(0.0), 1.0)
        self.assertEqual(exp_or_inf(-5.0), exp(-5.0))
        self.assertEqual(exp_or_inf(709.0), exp(709.0))
        self.assertEqual(exp_or_inf(709.5), inf)
        self.assertEqual(exp_or_inf(1e300), inf)

    def test_gumbel_log_survival_branches_match_the_definition(self) -> None:
        from math import exp, log

        from veridist.families.gumbel import _log_survival

        self.assertEqual(_log_survival(-701.0), 0.0)
        self.assertEqual(_log_survival(-1e300), 0.0)
        self.assertEqual(_log_survival(800.0), -800.0)
        self.assertEqual(_log_survival(1e300), -1e300)
        for z in (-50.0, -0.5, 0.0, 0.3, 2.0, 30.0, 600.0):
            with self.subTest(z=z):
                tail = exp(-z)
                direct = log(-(exp(-tail) - 1.0)) if 1e-8 < tail < 30.0 else None
                value = _log_survival(z)
                self.assertLess(value, 1e-300)
                if direct is not None:
                    self.assertAlmostEqual(value, direct, places=9)

    def test_a_boundary_flag_from_the_search_becomes_a_boundary_failure(self) -> None:
        from unittest.mock import patch

        from veridist.families import gumbel as gumbel_module
        from veridist.families import normal as normal_module
        from veridist.families.gumbel import GumbelFitFailureCode, fit_gumbel_right
        from veridist.families.normal import NormalFitFailureCode, fit_normal

        censored_values = [ExactValue(1.0), ExactValue(2.0), RightCensoredValue(3.0)]
        plain_values = [ExactValue(1.0), ExactValue(2.0)]
        with patch.object(normal_module, "maximize_nested", return_value=(0.0, 0.0, 0.0, True)):
            self.assertIs(fit_normal(censored_values).code, NormalFitFailureCode.BOUNDARY_SOLUTION)  # type: ignore[union-attr]
        with patch.object(gumbel_module, "maximize_nested", return_value=(0.0, 0.0, 0.0, True)):
            self.assertIs(
                fit_gumbel_right(censored_values).code,  # type: ignore[union-attr]
                GumbelFitFailureCode.BOUNDARY_SOLUTION,
            )
        with patch.object(gumbel_module, "expand_bracket", return_value=(0.0, 0.0, True)):
            self.assertIs(
                fit_gumbel_right(plain_values).code,  # type: ignore[union-attr]
                GumbelFitFailureCode.BOUNDARY_SOLUTION,
            )

    def test_unrepresentable_deviations_from_the_exact_mean_fail_softly(self) -> None:
        from veridist.families.gumbel import GumbelFitFailureCode, fit_gumbel_right
        from veridist.families.normal import NormalFitFailureCode, fit_normal

        data = [ExactValue(-0.9e308), ExactValue(-0.1e308), RightCensoredValue(1.7e308)]
        self.assertIs(fit_normal(data).code, NormalFitFailureCode.OPTIMIZER_EXHAUSTED)  # type: ignore[union-attr]
        self.assertIs(fit_gumbel_right(data).code, GumbelFitFailureCode.OPTIMIZER_EXHAUSTED)  # type: ignore[union-attr]

    def test_a_gamma_shape_beyond_the_incomplete_gamma_budget_is_an_optimizer_failure(
        self,
    ) -> None:
        from veridist.families.gamma import GammaFitFailureCode, fit_gamma

        cluster = (1.0, 1.000001, 0.999999, 1.0000005)
        result = fit_gamma(
            [ExactLifetime(t) for t in cluster] + [RightCensoredLifetime(1.0000002)] * 3
        )
        self.assertIs(result.code, GammaFitFailureCode.OPTIMIZER_EXHAUSTED)  # type: ignore[union-attr]
