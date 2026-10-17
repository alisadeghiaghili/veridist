"""Profile-likelihood intervals of the multi-pass scale fits: agreement, passes, refusals."""

from __future__ import annotations

import dataclasses
import unittest
from dataclasses import replace
from functools import cache
from math import log
from typing import Any
from unittest import mock

import numpy as np

from tests.unit.scale_checkpoint_fixtures import ReadLog
from tests.unit.scale_inference_oracle import gumbel_data, weibull_data
from veridist.domain.lifetimes import ExactLifetime
from veridist.domain.values import ExactValue
from veridist.engine.errors import FailureCode
from veridist.engine.pass_budget import PassBudgetError
from veridist.families._evidence import FitEvidence
from veridist.families.dispatch import fit as in_memory_fit
from veridist.families.registry import FamilyId
from veridist.families.uncertainty import DerivedEstimate, ParameterInterval
from veridist.scale import _profile_intervals as pi
from veridist.scale._errors import (
    ScaleInferenceError,
    ScaleInferenceErrorCode,
    ScaleSourceError,
    ScaleSourceErrorCode,
)
from veridist.scale._evidence import ProfileEvidence
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._power_sums import MAX_POWER_SUM_CANDIDATES, PowerSumAccumulator
from veridist.scale._sources import NumpySource

DATA = {
    "weibull_min": lambda size, seed: weibull_data(1.7, 40.0, size, seed),
    "gumbel_right": lambda size, seed: gumbel_data(5.0, 2.0, size, seed),
}


@cache
def scale_fit(family: str, size: int, seed: int) -> tuple[np.ndarray, ScaleFitResult]:
    values = DATA[family](size, seed)
    return values, fit(NumpySource(values), family)


def in_memory_uncertainty(family: str, values: np.ndarray) -> Any:
    if family == "weibull_min":
        observations: Any = tuple(ExactLifetime(float(v)) for v in values)
    else:
        observations = tuple(ExactValue(float(v)) for v in values)
    return in_memory_fit(family, observations).uncertainty()


class QuantityTests(unittest.TestCase):
    def test_the_constructors_validate_and_name_their_quantity(self) -> None:
        self.assertEqual(pi.mean().name, "mean")
        self.assertEqual(pi.quantile(0.1).name, "quantile:0.1")
        self.assertEqual(pi.survival(30.0).name, "survival:30.0")
        self.assertEqual(pi.survival(-2.5).kind, pi.QuantityKind.SURVIVAL)
        for bad in (0.0, 1.0, -0.1, 1.5):
            with self.assertRaisesRegex(ValueError, "strictly between 0 and 1"):
                pi.quantile(bad)
        with self.assertRaisesRegex(ValueError, "finite"):
            pi.survival(float("inf"))
        with self.assertRaisesRegex(ValueError, "finite"):
            pi.survival(float("nan"))
        with self.assertRaises(TypeError):
            pi.quantile(1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            pi.Quantity("mean")  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "no argument"):
            pi.Quantity(pi.QuantityKind.MEAN, 1.0)
        with self.assertRaises(TypeError):
            pi.Quantity(pi.QuantityKind.QUANTILE)

    def test_the_plan_constants_are_pinned(self) -> None:
        self.assertEqual(pi.PROFILE_PASS_BUDGET_DEFAULT, 6)
        self.assertEqual(pi.WINDOW_FACTOR, 4.0)
        self.assertEqual(pi.WINDOW_GROWTH, 2.0)
        self.assertEqual(pi.MAX_WINDOW_EXPANSIONS, 3)
        self.assertEqual(pi.MAX_QUANTITIES, 64)
        self.assertEqual(pi.PROFILE_MIN_PASSES, 2)
        self.assertEqual(pi.PROFILE_MAX_PASSES, 9)
        # the confirmation holds the two ends and the two extremes of every quantity and of
        # the location-like parameter: it must fit in one frame
        self.assertGreaterEqual(MAX_POWER_SUM_CANDIDATES, 2 * (1 + pi.MAX_QUANTITIES) + 2)
        self.assertEqual(
            pi.profile_plan_identity(),
            (
                6,
                (4.0).hex(),
                (2.0).hex(),
                3,
                (20.0).hex(),
                (8, 16, 32, 64, 128),
                (1e-6).hex(),
                (1e-5).hex(),
            ),
        )


class ReferenceTests(unittest.TestCase):
    """The intervals equal the in-memory profile intervals and the ``mpmath`` definition."""

    def test_the_parameters_equal_the_in_memory_profile_intervals(self) -> None:
        for family in DATA:
            for size, seed in ((400, 1), (3000, 2)):
                for level in (0.9, 0.95, 0.99):
                    with self.subTest(family=family, size=size, level=level):
                        values, result = scale_fit(family, size, seed)
                        out = pi.profile_intervals(NumpySource(values), result, level=level)
                        reference = in_memory_uncertainty(family, values).interval_details(
                            level, "profile"
                        )
                        self.assertEqual(set(out.parameters), set(reference))
                        for name, interval in out.parameters.items():
                            expected = reference[name]
                            self.assertEqual((interval.method, interval.level), ("profile", level))
                            self.assertFalse(interval.lower_unbounded or interval.upper_unbounded)
                            self.assertAlmostEqual(
                                interval.lower / expected.lower, 1.0, delta=1e-7, msg=name
                            )
                            self.assertAlmostEqual(
                                interval.upper / expected.upper, 1.0, delta=1e-7, msg=name
                            )

    def test_the_weibull_quantities_equal_the_in_memory_profile_intervals(self) -> None:
        values, result = scale_fit("weibull_min", 800, 3)
        requested = [pi.mean(), pi.quantile(0.1), pi.quantile(0.5), pi.survival(30.0)]
        out = pi.profile_intervals(NumpySource(values), result, quantities=requested)
        uncertainty = in_memory_uncertainty("weibull_min", values)
        reference = {
            "mean": uncertainty.mean(0.95, "profile"),
            "quantile:0.1": uncertainty.quantile(0.1, 0.95, "profile"),
            "quantile:0.5": uncertainty.quantile(0.5, 0.95, "profile"),
            "survival:30.0": uncertainty.survival(30.0, 0.95, "profile"),
        }
        self.assertEqual(list(out.derived), list(reference))
        for name, expected in reference.items():
            got = out.derived[name]
            assert isinstance(got, DerivedEstimate)
            with self.subTest(name=name):
                self.assertEqual((got.method, got.level), ("profile", 0.95))
                self.assertAlmostEqual(got.estimate / expected.estimate, 1.0, delta=1e-6)
                self.assertAlmostEqual(got.lower / expected.lower, 1.0, delta=1e-6)
                self.assertAlmostEqual(got.upper / expected.upper, 1.0, delta=1e-6)
                self.assertLess(got.lower, got.estimate)
                self.assertLess(got.estimate, got.upper)

    def test_a_larger_level_gives_a_wider_interval_around_the_estimate(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        previous: dict[str, ParameterInterval] | None = None
        for level in (0.5, 0.8, 0.95, 0.999):
            out = pi.profile_intervals(NumpySource(values), result, level=level)
            for name, interval in out.parameters.items():
                estimate = result.fit.parameters[name]  # type: ignore[union-attr]
                self.assertLess(interval.lower, estimate)
                self.assertGreater(interval.upper, estimate)
                if previous is not None:
                    self.assertLess(interval.lower, previous[name].lower)
                    self.assertGreater(interval.upper, previous[name].upper)
            previous = dict(out.parameters)


class ResultTests(unittest.TestCase):
    def test_the_result_describes_the_computation(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        out = pi.profile_intervals(NumpySource(values), result, level=0.9)
        self.assertIs(out.family, FamilyId.WEIBULL_MIN)
        self.assertEqual(out.level, 0.9)
        self.assertAlmostEqual(out.deviance, 2.705543454095404, places=12)
        self.assertEqual(list(out.parameters), ["shape", "scale"])
        self.assertEqual(out.derived, {})
        self.assertEqual(out.scale.row_count, 600)
        self.assertEqual(out.scale.state, result.scale.state)
        self.assertEqual(out.scale.partition_count, 1)
        with self.assertRaises(TypeError):
            out.parameters["shape"] = None  # type: ignore[index]

    def test_the_resolution_reports_the_surrogate_and_the_confirmation(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        out = pi.profile_intervals(NumpySource(values), result)
        resolution = out.resolution
        self.assertIn(resolution.surrogate_degree, (8, 16, 32, 64, 128))
        self.assertEqual(resolution.tolerance, 600 * 2.0**-40)
        self.assertLessEqual(2 * 600 * resolution.surrogate_gap, 1e-6 + resolution.tolerance)
        self.assertLessEqual(resolution.confirmation_residual, 1e-5 + resolution.tolerance)

    def test_the_gumbel_order_of_the_parameters_is_the_registry_order(self) -> None:
        values, result = scale_fit("gumbel_right", 600, 5)
        out = pi.profile_intervals(NumpySource(values), result)
        self.assertEqual(list(out.parameters), ["location", "scale"])
        self.assertIs(out.family, FamilyId.GUMBEL_RIGHT)

    def test_the_gumbel_quantities_are_typed_unavailable(self) -> None:
        values, result = scale_fit("gumbel_right", 600, 5)
        out = pi.profile_intervals(
            NumpySource(values),
            result,
            quantities=[pi.mean(), pi.quantile(0.5), pi.survival(4.0)],
        )
        self.assertEqual(list(out.derived), ["mean", "quantile:0.5", "survival:4.0"])
        for answer in out.derived.values():
            assert isinstance(answer, pi.ProfileUnavailable)
            self.assertIs(answer.reason, pi.ProfileUnavailableReason.NOT_IMPLEMENTED)
            self.assertIn("Gumbel", answer.detail)

    def test_a_survival_of_zero_or_one_has_a_degenerate_interval(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        out = pi.profile_intervals(
            NumpySource(values),
            result,
            level=0.9,
            quantities=[pi.survival(1e-300), pi.survival(1e9), pi.survival(-3.0)],
        )
        self.assertEqual(
            out.derived["survival:1e-300"], DerivedEstimate(1.0, 1.0, 1.0, "profile", 0.9)
        )
        self.assertEqual(
            out.derived["survival:1000000000.0"], DerivedEstimate(0.0, 0.0, 0.0, "profile", 0.9)
        )
        self.assertEqual(
            out.derived["survival:-3.0"], DerivedEstimate(1.0, 1.0, 1.0, "profile", 0.9)
        )

    def test_an_unrepresentable_quantity_is_typed_unavailable(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        with mock.patch.object(pi, "gamma", side_effect=OverflowError):
            out = pi.profile_intervals(
                NumpySource(values), result, quantities=[pi.mean(), pi.quantile(0.1)]
            )
        answer = out.derived["mean"]
        assert isinstance(answer, pi.ProfileUnavailable)
        self.assertIs(answer.reason, pi.ProfileUnavailableReason.NOT_REPRESENTABLE)
        self.assertIsInstance(out.derived["quantile:0.1"], DerivedEstimate)

    def test_a_non_finite_estimate_is_typed_unavailable(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        with mock.patch.object(pi, "gamma", return_value=float("inf")):
            out = pi.profile_intervals(NumpySource(values), result, quantities=[pi.mean()])
        self.assertIsInstance(out.derived["mean"], pi.ProfileUnavailable)


class FallbackErrorTests(unittest.TestCase):
    def test_an_unavailable_covariance_gives_the_inverse_root_of_the_count(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        basis = result.fit._evidence.basis  # type: ignore[union-attr]
        from veridist.families.uncertainty import (
            UncertaintyUnavailable,
            UncertaintyUnavailableReason,
        )

        unavailable = UncertaintyUnavailable(UncertaintyUnavailableReason.NOT_COMPUTABLE)
        stand_in = 1.0 / 600**0.5
        for answer in (unavailable,):
            fake = mock.Mock()
            fake.fit.uncertainty.return_value = answer
            self.assertEqual(pi._relative_error(fake, basis), stand_in)
        real = pi._relative_error(result, basis)
        self.assertGreater(real, 0.0)
        shape = result.fit.shape  # type: ignore[union-attr]
        self.assertEqual(real, result.fit.uncertainty().standard_errors["shape"] / shape)  # type: ignore[union-attr]
        values, gumbel = scale_fit("gumbel_right", 600, 5)
        gbasis = gumbel.fit._evidence.basis  # type: ignore[union-attr]
        scale = gumbel.fit.scale  # type: ignore[union-attr]
        self.assertEqual(
            pi._relative_error(gumbel, gbasis),
            gumbel.fit.uncertainty().standard_errors["scale"] / scale,  # type: ignore[union-attr]
        )

    def test_a_non_positive_or_non_finite_error_gives_the_stand_in(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        basis = result.fit._evidence.basis  # type: ignore[union-attr]
        for bad in (0.0, float("nan"), float("inf")):
            fake = mock.Mock()
            uncertainty = mock.Mock()
            uncertainty.estimates = (1.0, 1.0)
            uncertainty.parameter_names = ("shape", "scale")
            uncertainty.standard_errors = {"shape": bad, "scale": bad}
            fake.fit.uncertainty.return_value = uncertainty
            self.assertEqual(pi._relative_error(fake, basis), 1.0 / 600**0.5)


class PassTests(unittest.TestCase):
    def roles(self, out: pi.ProfileIntervals) -> list[tuple[str, int]]:
        assert out.scale.passes is not None
        return [(item.role, item.points) for item in out.scale.passes.records]

    def test_a_run_reads_a_window_refinements_and_a_confirmation(self) -> None:
        values, result = scale_fit("weibull_min", 3000, 2)
        source = ReadLog(NumpySource(values, partition_rows=1000))
        out = pi.profile_intervals(source, result)
        passes = out.scale.passes
        assert passes is not None
        self.assertEqual(
            (passes.pass_budget, passes.minimum_passes, passes.maximum_passes), (6, 2, 9)
        )
        roles = self.roles(out)
        self.assertEqual(roles[0], ("profile_window", 10))
        self.assertEqual(roles[-1][0], "profile_confirmation")
        self.assertTrue(all(role == "profile_refinement" for role, _ in roles[1:-1]))
        self.assertEqual(passes.pass_count, len(roles))
        self.assertEqual(len(source.reads), 3 * passes.pass_count)
        degree = out.resolution.surrogate_degree
        self.assertEqual(
            [points for role, points in roles if role == "profile_refinement"],
            [d // 2 for d in (16, 32, 64, 128) if d <= degree],
        )
        window, last = passes.records[0], passes.records[-1]
        assert window.log_low is not None and window.log_high is not None
        self.assertLess(window.log_low, log(result.fit.shape))  # type: ignore[union-attr]
        self.assertGreater(window.log_high, log(result.fit.shape))  # type: ignore[union-attr]
        assert last.log_low is not None and last.log_high is not None
        self.assertLessEqual(last.log_low, last.log_high)

    def test_the_candidates_of_the_confirmation_are_the_ends_and_the_extremes(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        shape_only = pi.profile_intervals(NumpySource(values), result)
        more = pi.profile_intervals(
            NumpySource(values), result, quantities=[pi.mean(), pi.quantile(0.1)]
        )
        self.assertLessEqual(self.roles(shape_only)[-1][1], 6)
        self.assertGreater(self.roles(more)[-1][1], self.roles(shape_only)[-1][1])
        self.assertLessEqual(self.roles(more)[-1][1], 2 + 2 * 3)

    def test_a_budget_below_the_least_is_refused_before_any_read(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        source = ReadLog(NumpySource(values))
        with self.assertRaises(PassBudgetError) as caught:
            pi.profile_intervals(source, result, inference_pass_budget=1)
        self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_INSUFFICIENT)
        self.assertEqual(
            dict(caught.exception.context),
            {"pass_budget": 1, "minimum_passes": 2, "maximum_passes": 9},
        )
        self.assertEqual(source.reads, [])

    def test_a_run_that_needs_more_passes_than_the_budget_is_refused_before_the_excess_read(
        self,
    ) -> None:
        values, result = scale_fit("weibull_min", 3000, 2)
        needed = pi.profile_intervals(NumpySource(values), result).scale.passes
        assert needed is not None and needed.pass_count > 2
        source = ReadLog(NumpySource(values))
        with self.assertRaises(PassBudgetError) as caught:
            pi.profile_intervals(source, result, inference_pass_budget=needed.pass_count - 1)
        self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_EXCEEDED)
        self.assertEqual(
            dict(caught.exception.context),
            {
                "max_passes": needed.pass_count - 1,
                "actual_pass_count": needed.pass_count - 1,
                "attempted_pass": needed.pass_count,
            },
        )
        self.assertEqual(len(source.reads), needed.pass_count - 1)

    def test_exactly_the_passes_needed_are_enough(self) -> None:
        values, result = scale_fit("weibull_min", 3000, 2)
        needed = pi.profile_intervals(NumpySource(values), result).scale.passes
        assert needed is not None
        out = pi.profile_intervals(
            NumpySource(values), result, inference_pass_budget=needed.pass_count
        )
        assert out.scale.passes is not None
        self.assertEqual(out.scale.passes.pass_count, needed.pass_count)
        self.assertEqual(out.scale.passes.pass_budget, needed.pass_count)

    def test_a_window_that_is_too_narrow_is_widened_and_gives_the_same_intervals(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        base = pi.profile_intervals(NumpySource(values), result)
        with mock.patch.object(pi, "WINDOW_FACTOR", 0.3):
            widened = pi.profile_intervals(NumpySource(values), result)
        roles = [role for role, _ in self.roles(widened)]
        self.assertGreater(roles.count("profile_window"), 1)
        self.assertEqual(
            roles[: roles.count("profile_window")],
            ["profile_window"] * roles.count("profile_window"),
        )
        for name, interval in base.parameters.items():
            self.assertAlmostEqual(widened.parameters[name].lower / interval.lower, 1.0, delta=1e-9)
            self.assertAlmostEqual(widened.parameters[name].upper / interval.upper, 1.0, delta=1e-9)

    def test_the_window_is_cut_to_the_search_domain(self) -> None:
        self.assertEqual(pi._window(0.0, 1.0), (-1.0, 1.0))
        self.assertEqual(pi._window(19.5, 3.0), (16.5, 20.0))
        self.assertEqual(pi._window(-19.5, 3.0), (-20.0, -16.5))
        self.assertEqual(pi._window(0.0, 1e9), (-20.0, 20.0))


class RefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values, self.result = scale_fit("weibull_min", 600, 5)

    def test_a_result_that_cannot_be_profiled_is_refused(self) -> None:
        def refused(result: Any) -> None:
            with self.assertRaises(ScaleInferenceError) as caught:
                pi.profile_intervals(NumpySource(self.values), result)
            self.assertIs(caught.exception.code, ScaleInferenceErrorCode.FIT_NOT_PROFILABLE)

        refused(object())
        refused(fit(NumpySource(self.values), "lognormal"))
        refused(fit(NumpySource(self.values), "normal"))
        refused(fit(NumpySource(np.full(10, 3.0)), "weibull_min"))
        refused(fit(NumpySource(np.array([-1.0, 2.0, 3.0])), "weibull_min"))
        success: Any = self.result.fit
        evidence = ProfileEvidence(success.family, (1.0, 1.0), lambda: None)  # type: ignore[arg-type, return-value]
        stripped = replace(success, _evidence=evidence)
        refused(replace(self.result, fit=stripped))
        refused(replace(self.result, fit=replace(success, _evidence=FitEvidence())))

    def test_the_arguments_are_validated_before_anything_is_read(self) -> None:
        source = ReadLog(NumpySource(self.values))
        cases: list[tuple[type[Exception], dict[str, Any]]] = [
            (ValueError, {"level": 0.0}),
            (ValueError, {"level": 1.0}),
            (TypeError, {"level": "0.95"}),
            (TypeError, {"quantities": [object()]}),
            (ValueError, {"quantities": [pi.mean(), pi.mean()]}),
            (ValueError, {"quantities": [pi.quantile(0.5)] * 2}),
            (ValueError, {"quantities": [pi.survival(float(i)) for i in range(65)]}),
            (TypeError, {"inference_pass_budget": 3.0}),
            (ValueError, {"inference_pass_budget": 0}),
            (ValueError, {"max_rows": 0}),
            (TypeError, {"spool": "dir"}),
            (ValueError, {"executor": "threads"}),
            (TypeError, {"executor": 3}),
            (ValueError, {"resume": True}),
            (TypeError, {"resume": 1}),
            (ValueError, {"recompute_changed": True}),
            (TypeError, {"checkpoint": 5}),
        ]
        for error, arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(error):
                pi.profile_intervals(source, self.result, **arguments)
        with self.assertRaises(TypeError):
            pi.profile_intervals(object(), self.result)  # type: ignore[arg-type]
        for bad in ("mean", 5, None):
            with self.assertRaisesRegex(TypeError, "sequence of Quantity"):
                pi.profile_intervals(source, self.result, quantities=bad)  # type: ignore[arg-type]
        self.assertEqual(source.reads, [])

    def test_the_most_quantities_are_accepted(self) -> None:
        out = pi.profile_intervals(
            NumpySource(self.values),
            self.result,
            quantities=[pi.survival(float(i + 1)) for i in range(64)],
        )
        self.assertEqual(len(out.derived), 64)
        assert out.scale.passes is not None
        self.assertLessEqual(out.scale.passes.records[-1].points, 2 + 2 * 65)

    def test_another_source_is_partition_changed(self) -> None:
        other = weibull_data(1.7, 40.0, 600, 99)
        with self.assertRaises(ScaleSourceError) as caught:
            pi.profile_intervals(NumpySource(other), self.result)
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)
        with self.assertRaises(ScaleSourceError) as caught:
            pi.profile_intervals(NumpySource(self.values[:-1]), self.result)
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)

    def test_a_sum_that_is_not_finite_or_not_positive_is_not_computable(self) -> None:
        for bad in (0.0, float("inf"), float("nan"), -1.0):
            with self.subTest(bad=bad):
                with mock.patch.object(
                    PowerSumAccumulator, "totals", lambda self, bad=bad: [bad] * len(self.sums)
                ):
                    with self.assertRaises(ScaleInferenceError) as caught:
                        pi.profile_intervals(NumpySource(self.values), self.result)
                self.assertIs(caught.exception.code, ScaleInferenceErrorCode.PROFILE_NOT_COMPUTABLE)

    def test_a_single_use_source_needs_a_spool(self) -> None:
        class Once(NumpySource):
            single_use = True

        with self.assertRaises(ScaleSourceError) as caught:
            pi.profile_intervals(Once(self.values), self.result)
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.SPOOL_REQUIRED)

    def test_a_window_that_never_brackets_is_not_converged(self) -> None:
        with (
            mock.patch.object(pi, "WINDOW_FACTOR", 1e-9),
            mock.patch.object(pi, "MAX_WINDOW_EXPANSIONS", 1),
            mock.patch.object(pi, "WINDOW_GROWTH", 1.5),
        ):
            source = ReadLog(NumpySource(self.values))
            with self.assertRaises(ScaleInferenceError) as caught:
                pi.profile_intervals(source, self.result)
        self.assertIs(caught.exception.code, ScaleInferenceErrorCode.PROFILE_NOT_CONVERGED)
        self.assertEqual(len(source.reads), 2)

    def test_a_region_that_reaches_the_edge_of_the_domain_is_unbounded(self) -> None:
        edge = log(self.result.fit.shape) + 1e-6  # type: ignore[union-attr]
        with mock.patch.object(pi, "LOG_DOMAIN_LIMIT", edge):
            with self.assertRaises(ScaleInferenceError) as caught:
                pi.profile_intervals(NumpySource(self.values), self.result)
        self.assertIs(caught.exception.code, ScaleInferenceErrorCode.PROFILE_UNBOUNDED)

    def test_a_region_open_on_either_side_is_unbounded_at_the_domain_edge(self) -> None:
        centre = log(self.result.fit.shape)  # type: ignore[union-attr]
        real = pi.LikelihoodForm.deviance

        for side in ("low", "high"):

            def short(form: Any, point: float, psi: float, side: str = side) -> float:
                if side == "low" and point < centre - 0.5:
                    return 0.0
                if side == "high" and point > centre + 0.5:
                    return 0.0
                return real(form, point, psi)

            with self.subTest(side=side):
                with (
                    mock.patch.object(pi, "WINDOW_FACTOR", 1e6),
                    mock.patch.object(pi.LikelihoodForm, "deviance", short),
                ):
                    with self.assertRaises(ScaleInferenceError) as caught:
                        pi.profile_intervals(NumpySource(self.values), self.result)
                self.assertIs(caught.exception.code, ScaleInferenceErrorCode.PROFILE_UNBOUNDED)

    def test_the_confirmation_allows_for_rounding_of_the_sums(self) -> None:
        with mock.patch.object(pi, "CONFIRMATION_TOLERANCE", 0.0):
            out = pi.profile_intervals(NumpySource(self.values), self.result)
        self.assertGreater(out.resolution.tolerance, out.resolution.confirmation_residual)

    def test_a_surrogate_that_does_not_converge_is_refused_at_the_highest_degree(self) -> None:
        source = ReadLog(NumpySource(self.values))
        with mock.patch.object(pi, "SURROGATE_TOLERANCE", -1e9):
            with self.assertRaises(ScaleInferenceError) as caught:
                pi.profile_intervals(source, self.result)
        self.assertIs(caught.exception.code, ScaleInferenceErrorCode.PROFILE_NOT_CONVERGED)
        self.assertEqual(len(source.reads), 5)

    def test_a_confirmation_that_disagrees_with_the_surrogate_is_refused(self) -> None:
        with mock.patch.object(pi, "CONFIRMATION_TOLERANCE", -1.0):
            with self.assertRaises(ScaleInferenceError) as caught:
                pi.profile_intervals(NumpySource(self.values), self.result)
        self.assertIs(caught.exception.code, ScaleInferenceErrorCode.PROFILE_NOT_CONVERGED)

    def test_an_estimate_that_is_not_inside_its_own_region_is_refused(self) -> None:
        with mock.patch.object(pi, "surrogate_deviance", return_value=1e6):
            with self.assertRaises(ScaleInferenceError) as caught:
                pi.profile_intervals(NumpySource(self.values), self.result)
        self.assertIs(caught.exception.code, ScaleInferenceErrorCode.PROFILE_NOT_CONVERGED)


class ImmutabilityTests(unittest.TestCase):
    def test_the_values_are_immutable_and_slotted(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        out = pi.profile_intervals(NumpySource(values), result, quantities=[pi.mean()])
        for value in (
            out,
            out.resolution,
            pi.mean(),
            pi.ProfileUnavailable(pi.ProfileUnavailableReason.NOT_IMPLEMENTED),
        ):
            with self.subTest(type=type(value).__name__):
                first = next(iter(type(value).__dataclass_fields__))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(value, first, None)
                self.assertFalse(hasattr(value, "__dict__"))


class ErrorTests(unittest.TestCase):
    def test_the_inference_error_carries_only_its_code(self) -> None:
        import pickle

        error = ScaleInferenceError(ScaleInferenceErrorCode.PROFILE_UNBOUNDED)
        self.assertEqual(str(error), "PROFILE_UNBOUNDED")
        self.assertIs(error.code, ScaleInferenceErrorCode.PROFILE_UNBOUNDED)
        copy = pickle.loads(pickle.dumps(error))
        self.assertIs(copy.code, ScaleInferenceErrorCode.PROFILE_UNBOUNDED)
        self.assertEqual(ScaleInferenceError("FIT_NOT_PROFILABLE").code.value, "FIT_NOT_PROFILABLE")
        with self.assertRaises(ValueError):
            ScaleInferenceError("nothing")
        self.assertEqual(
            [item.value for item in ScaleInferenceErrorCode],
            [
                "FIT_NOT_PROFILABLE",
                "PROFILE_NOT_CONVERGED",
                "PROFILE_UNBOUNDED",
                "PROFILE_NOT_COMPUTABLE",
                "GOF_FIT_NOT_AVAILABLE",
                "GOF_NOT_REPRESENTABLE",
                "GOF_RESOLUTION_UNSUPPORTED",
            ],
        )


class UncertaintyPointerTests(unittest.TestCase):
    def test_the_in_memory_style_profile_request_points_to_the_function(self) -> None:
        values, result = scale_fit("weibull_min", 600, 5)
        with self.assertRaisesRegex(NotImplementedError, "profile_intervals"):
            result.fit.uncertainty().confidence_intervals(0.95, "profile")  # type: ignore[union-attr]
        with self.assertRaisesRegex(NotImplementedError, "further passes"):
            result.fit.uncertainty().quantile(0.1, 0.95, "profile")  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
