"""The information a multi-pass fit keeps instead of its data, and the uncertainty it reports."""

from __future__ import annotations

import math
import unittest
from typing import Any

from tests.unit.scale_multi_pass_oracle import gumbel_sample, weibull_sample
from veridist.domain.lifetimes import ExactLifetime
from veridist.domain.values import ExactValue
from veridist.families._models import Evaluation, GumbelModel, WeibullModel
from veridist.families.gumbel import fit_gumbel_right
from veridist.families.registry import FamilyId
from veridist.families.uncertainty import (
    FitUncertainty,
    UncertaintyUnavailable,
    UncertaintyUnavailableReason,
)
from veridist.families.weibull import fit_weibull
from veridist.scale._evidence import (
    GumbelSumsModel,
    ProfileEvidence,
    WaldUncertainty,
    WeibullSumsModel,
    gumbel_evaluation,
    weibull_evaluation,
)
from veridist.scale._fit import fit
from veridist.scale._planner import ProfileSums
from veridist.scale._sources import NumpySource


def close(test: unittest.TestCase, value: float, reference: float, tolerance: float) -> None:
    test.assertLessEqual(abs(value - reference), tolerance * max(abs(reference), 1e-300))


class WeibullInformationTests(unittest.TestCase):
    def test_the_hessian_is_the_in_memory_hessian_at_the_fitted_parameters(self) -> None:
        for shape, size, seed in ((0.6, 200, 1), (1.7, 500, 2), (9.0, 300, 3)):
            values = weibull_sample(shape, 1000.0, size, seed)
            result: Any = fit(NumpySource(values), "weibull_min").fit
            theta = (result.shape, result.scale)
            model = WeibullModel(tuple(float(v) for v in values), ())
            reference = model.evaluate(theta)
            evidence = result._evidence
            uncertainty = evidence.uncertainty(FamilyId.WEIBULL_MIN, result.parameters)
            assert isinstance(uncertainty, WaldUncertainty)
            sums_model = uncertainty._model
            assert isinstance(sums_model, WeibullSumsModel)
            ours = sums_model.evaluate(theta)
            with self.subTest(shape=shape):
                close(self, ours.value, reference.value, 1e-12)
                for row in range(2):
                    for column in range(2):
                        close(self, ours.hessian[row][column], reference.hessian[row][column], 1e-6)
                self.assertEqual(ours.gradient, (0.0, 0.0))

    def test_the_covariance_agrees_with_the_in_memory_fit(self) -> None:
        values = weibull_sample(1.7, 1000.0, 400, 7)
        ours: Any = fit(NumpySource(values), "weibull_min").fit
        theirs = fit_weibull([ExactLifetime(float(v)) for v in values])
        a: Any = ours.uncertainty()
        b: Any = theirs.uncertainty()  # type: ignore[union-attr]
        for row in range(2):
            for column in range(2):
                close(self, a.covariance[row][column], b.covariance[row][column], 1e-6)
        for name in ("shape", "scale"):
            close(self, a.standard_errors[name], b.standard_errors[name], 1e-6)
            for ours_bound, their_bound in zip(
                a.confidence_intervals()[name], b.confidence_intervals()[name], strict=True
            ):
                close(self, ours_bound, their_bound, 1e-6)
        close(self, a.mean().upper, b.mean().upper, 1e-6)
        close(self, a.quantile(0.1).lower, b.quantile(0.1).lower, 1e-6)
        close(self, a.survival(800.0).upper, b.survival(800.0).upper, 1e-6)


class GumbelInformationTests(unittest.TestCase):
    def test_the_hessian_is_the_in_memory_hessian_at_the_fitted_parameters(self) -> None:
        for location, scale, size, seed in (
            (0.0, 1.0, 300, 1),
            (1e6, 3.0, 500, 2),
            (-50.0, 0.01, 200, 3),
        ):
            values = gumbel_sample(location, scale, size, seed)
            result: Any = fit(NumpySource(values), "gumbel_right").fit
            theta = (result.location, result.scale)
            model = GumbelModel(tuple(float(v) for v in values), ())
            reference = model.evaluate(theta)
            uncertainty = result._evidence.uncertainty(FamilyId.GUMBEL_RIGHT, result.parameters)
            assert isinstance(uncertainty, WaldUncertainty)
            sums_model = uncertainty._model
            assert isinstance(sums_model, GumbelSumsModel)
            ours = sums_model.evaluate(theta)
            with self.subTest(location=location, scale=scale):
                close(self, ours.value, reference.value, 1e-9)
                for row in range(2):
                    for column in range(2):
                        close(self, ours.hessian[row][column], reference.hessian[row][column], 1e-5)

    def test_the_covariance_agrees_with_the_in_memory_fit(self) -> None:
        values = gumbel_sample(5.0, 2.0, 400, 7)
        ours: Any = fit(NumpySource(values), "gumbel_right").fit
        theirs = fit_gumbel_right([ExactValue(float(v)) for v in values])
        a: Any = ours.uncertainty()
        b: Any = theirs.uncertainty()  # type: ignore[union-attr]
        for row in range(2):
            for column in range(2):
                close(self, a.covariance[row][column], b.covariance[row][column], 1e-6)
        for name in ("location", "scale"):
            close(self, a.standard_errors[name], b.standard_errors[name], 1e-6)
        close(self, a.mean().lower, b.mean().lower, 1e-6)
        close(self, a.quantile(0.9).upper, b.quantile(0.9).upper, 1e-6)


class EvaluationFormulaTests(unittest.TestCase):
    def test_weibull_evaluation_from_known_sums(self) -> None:
        sums = ProfileSums(4.0, -2.0, 1.5)
        evaluation = weibull_evaluation(10, 2.0, 3.0, -0.5, sums, -7.0)
        # offset u = -0.25, S1/S0 = -0.5, variance = 0.375 - 0.25 = 0.125
        total_first = 10 * (-0.5 + 0.25)
        total_second = 10 * (0.125 + 0.0625)
        mixed = 2.0 * total_first / 3.0
        self.assertEqual(evaluation.value, -7.0)
        self.assertEqual(evaluation.gradient, (0.0, 0.0))
        self.assertEqual(evaluation.hessian[0][0], -10 / 4.0 - total_second)
        self.assertEqual(evaluation.hessian[0][1], mixed)
        self.assertEqual(evaluation.hessian[1][0], mixed)
        self.assertEqual(evaluation.hessian[1][1], -10 * 4.0 / 9.0)

    def test_gumbel_evaluation_from_known_sums(self) -> None:
        sums = ProfileSums(4.0, -2.0, 1.5)
        evaluation = gumbel_evaluation(10, 2.0, 3.0, 0.5, -0.75, sums, -9.0)
        # t1 = log_ratio - rate*first = 0.5 + 1.5 = 2.0; variance = 0.125
        t1 = 2.0
        t2 = 9.0 * 0.125 + (3.0 * -0.5 - 0.5) ** 2
        cross = 3.0 * (-0.75 + 0.5)
        self.assertEqual(evaluation.value, -9.0)
        self.assertEqual(evaluation.hessian[0][0], -10 / 4.0)
        self.assertEqual(evaluation.hessian[0][1], -10 * t1 / 4.0)
        self.assertEqual(evaluation.hessian[1][0], -10 * t1 / 4.0)
        self.assertAlmostEqual(
            evaluation.hessian[1][1], 10 * (1.0 + 2.0 * cross - t2) / 4.0, places=12
        )

    def test_a_negative_rounded_variance_is_clipped(self) -> None:
        sums = ProfileSums(1.0, -1.0, 0.9999999999)
        evaluation = weibull_evaluation(4, 1.0, 1.0, 0.0, sums, 0.0)
        self.assertEqual(evaluation.hessian[0][0], -4.0 - 4 * (1.0 + 0.0))


class ModelTests(unittest.TestCase):
    def evaluation(self) -> Evaluation:
        return Evaluation(-1.0, (0.0, 0.0), ((-2.0, 0.5), (0.5, -3.0)))

    def test_the_models_keep_the_registry_names_and_flags(self) -> None:
        weibull_model = WeibullSumsModel((2.0, 3.0), self.evaluation())
        gumbel_model = GumbelSumsModel((0.5, 1.5), self.evaluation())
        self.assertEqual(weibull_model.names, ("shape", "scale"))
        self.assertEqual(weibull_model.positive, (True, True))
        self.assertTrue(weibull_model.positive_support)
        self.assertEqual(gumbel_model.names, ("location", "scale"))
        self.assertEqual(gumbel_model.positive, (False, True))
        self.assertFalse(gumbel_model.positive_support)

    def test_the_information_exists_at_the_fitted_point_only(self) -> None:
        model = WeibullSumsModel((2.0, 3.0), self.evaluation())
        self.assertEqual(model.evaluate((2.0, 3.0)), self.evaluation())
        self.assertEqual(model.evaluate((2.0, 3.0), precise=False), self.evaluation())
        self.assertEqual(model.log_likelihood((2.0, 3.0)), -1.0)
        for other in ((2.0, 3.0000001), (2.1, 3.0), (3.0, 2.0)):
            with self.subTest(other=other), self.assertRaises(ValueError):
                model.evaluate(other)

    def test_the_closed_forms_of_the_in_memory_models_are_inherited(self) -> None:
        weibull_model = WeibullSumsModel((2.0, 3.0), self.evaluation())
        self.assertEqual(weibull_model.mean((2.0, 3.0)), 3.0 * math.gamma(1.5))
        self.assertAlmostEqual(
            weibull_model.quantile((2.0, 3.0), 0.5), 3.0 * math.log(2.0) ** 0.5, places=12
        )
        self.assertAlmostEqual(weibull_model.survival((2.0, 3.0), 3.0), math.exp(-1.0), places=12)
        gumbel_model = GumbelSumsModel((0.5, 1.5), self.evaluation())
        self.assertAlmostEqual(
            gumbel_model.mean((0.5, 1.5)), 0.5 + 0.5772156649015329 * 1.5, places=12
        )


class EvidenceTests(unittest.TestCase):
    def build(self, evaluation: Evaluation | Exception) -> ProfileEvidence:
        def make() -> Evaluation:
            if isinstance(evaluation, Exception):
                raise evaluation
            return evaluation

        return ProfileEvidence(FamilyId.WEIBULL_MIN, (2.0, 3.0), make)

    def test_the_evidence_holds_no_observation(self) -> None:
        evidence = self.build(Evaluation(-1.0, (0.0, 0.0), ((-2.0, 0.5), (0.5, -3.0))))
        self.assertEqual((evidence.exact, evidence.censored, evidence.fixed), ((), (), ()))

    def test_the_uncertainty_is_computed_once_per_parameter_vector(self) -> None:
        evidence = self.build(Evaluation(-1.0, (0.0, 0.0), ((-2.0, 0.5), (0.5, -3.0))))
        parameters = {"shape": 2.0, "scale": 3.0}
        first = evidence.uncertainty(FamilyId.WEIBULL_MIN, parameters)
        self.assertIsInstance(first, WaldUncertainty)
        self.assertIsInstance(first, FitUncertainty)
        self.assertIs(evidence.uncertainty(FamilyId.WEIBULL_MIN, parameters), first)
        assert isinstance(first, WaldUncertainty)
        self.assertEqual(first.parameter_names, ("shape", "scale"))
        self.assertEqual(first.estimates, (2.0, 3.0))
        determinant = 2.0 * 3.0 - 0.25
        self.assertAlmostEqual(first.covariance[0][0], 3.0 / determinant, places=12)
        self.assertAlmostEqual(first.covariance[0][1], 0.5 / determinant, places=12)

    def test_information_that_cannot_be_prepared_is_unavailable(self) -> None:
        for error in (ZeroDivisionError("x"), OverflowError("x"), ValueError("x")):
            evidence = self.build(error)
            result = evidence.uncertainty(FamilyId.WEIBULL_MIN, {"shape": 2.0, "scale": 3.0})
            with self.subTest(error=type(error).__name__):
                assert isinstance(result, UncertaintyUnavailable)
                self.assertIs(result.reason, UncertaintyUnavailableReason.NOT_COMPUTABLE)

    def test_a_matrix_that_cannot_be_inverted_is_unavailable(self) -> None:
        evidence = self.build(Evaluation(-1.0, (0.0, 0.0), ((1.0, 0.0), (0.0, 1.0))))
        result = evidence.uncertainty(FamilyId.WEIBULL_MIN, {"shape": 2.0, "scale": 3.0})
        assert isinstance(result, UncertaintyUnavailable)
        self.assertIs(result.reason, UncertaintyUnavailableReason.NOT_POSITIVE_DEFINITE)

    def test_the_family_selects_the_model(self) -> None:
        evaluation = Evaluation(-1.0, (0.0, 0.0), ((-2.0, 0.5), (0.5, -3.0)))
        gumbel = ProfileEvidence(FamilyId.GUMBEL_RIGHT, (0.5, 1.5), lambda: evaluation)
        result = gumbel.uncertainty(FamilyId.GUMBEL_RIGHT, {"location": 0.5, "scale": 1.5})
        assert isinstance(result, WaldUncertainty)
        self.assertIsInstance(result._model, GumbelSumsModel)
        self.assertEqual(result.parameter_names, ("location", "scale"))

    def test_profile_requests_are_refused_and_other_methods_are_not(self) -> None:
        evidence = self.build(Evaluation(-1.0, (0.0, 0.0), ((-2.0, 0.5), (0.5, -3.0))))
        result = evidence.uncertainty(FamilyId.WEIBULL_MIN, {"shape": 2.0, "scale": 3.0})
        assert isinstance(result, WaldUncertainty)
        with self.assertRaisesRegex(NotImplementedError, "further passes"):
            result.interval_details(0.95, "profile")
        wald = result.interval_details(0.9, "wald")
        self.assertEqual(wald["shape"].method, "wald")
        self.assertEqual(wald["shape"].level, 0.9)
        self.assertEqual(
            result.confidence_intervals(0.9), {k: (v.lower, v.upper) for k, v in wald.items()}
        )
        with self.assertRaisesRegex(NotImplementedError, "further passes"):
            result.quantile(0.5, 0.95, "profile")


if __name__ == "__main__":
    unittest.main()
