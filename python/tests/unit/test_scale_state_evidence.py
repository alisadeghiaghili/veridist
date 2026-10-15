"""State-backed fit evidence: the models it builds and the uncertainty it reports."""

from __future__ import annotations

import math
import unittest
from fractions import Fraction
from typing import Any

import numpy as np

from veridist.families._evidence import FitEvidence
from veridist.families._models import GammaModel, GaussianModel, build_model
from veridist.families.registry import FamilyId
from veridist.families.uncertainty import (
    FitUncertainty,
    UncertaintyUnavailable,
    UncertaintyUnavailableReason,
)
from veridist.scale._evidence import StateEvidence, model_of
from veridist.scale._one_pass import (
    ExactStatistic,
    GammaState,
    LognormalState,
    NormalState,
)


def folded(state_type: Any, values: list[float] | np.ndarray) -> Any:
    return state_type.empty().update(np.asarray(values, dtype=np.float64))


def exact(value: Fraction) -> ExactStatistic:
    return ExactStatistic.of(value)


class ModelFromStatisticsTests(unittest.TestCase):
    def test_normal_model_attributes_for_a_small_sample(self) -> None:
        model = model_of(folded(NormalState, [1.0, 2.0, 3.0, 4.0]).finalize())
        assert isinstance(model, GaussianModel)
        self.assertEqual(model.family, FamilyId.NORMAL)
        self.assertEqual(model.names, ("mu", "sigma"))
        self.assertEqual(model.events, 4)
        self.assertEqual(model.center, 2.5)
        self.assertEqual(model.sum_deviations, 0.0)
        self.assertEqual(model.sum_squares, 5.0)
        self.assertEqual(model.constant, 0.0)
        self.assertEqual(model.censored, ())
        self.assertFalse(model.positive_support)

    def test_normal_model_sums_are_exact_about_the_rounded_center(self) -> None:
        values = [0.1, 0.2, 0.4]
        model = model_of(folded(NormalState, values).finalize())
        assert isinstance(model, GaussianModel)
        total = sum((Fraction(value) for value in values), Fraction(0))
        center = float(total / 3)
        self.assertEqual(model.center, center)
        self.assertEqual(model.sum_deviations, float(total - 3 * Fraction(center)))
        squares = sum((Fraction(value - center) ** 2 for value in values), Fraction(0))
        self.assertAlmostEqual(model.sum_squares, float(squares), delta=1e-16)
        self.assertNotEqual(model.sum_deviations, 0.0)

    def test_lognormal_model_is_the_normal_model_of_the_logarithms(self) -> None:
        values = [0.5, 1.5, 4.0, 9.0]
        model = model_of(folded(LognormalState, values).finalize())
        assert isinstance(model, GaussianModel)
        logs = np.log(np.asarray(values))
        self.assertEqual(model.family, FamilyId.LOGNORMAL)
        self.assertEqual(model.names, ("mu_log", "sigma_log"))
        self.assertTrue(model.positive_support)
        self.assertEqual(model.events, 4)
        self.assertAlmostEqual(model.center, float(np.mean(logs)), delta=2e-16)
        self.assertEqual(model.constant, -math.fsum(logs.tolist()))
        self.assertEqual(model.censored, ())

    def test_gamma_model_attributes(self) -> None:
        values = [1.0, 2.0, 4.0, 8.0]
        model = model_of(folded(GammaState, values).finalize())
        assert isinstance(model, GammaModel)
        self.assertEqual(model.events, 4)
        self.assertEqual(model.sum_exact, 15.0)
        self.assertEqual(model.sum_logs, math.fsum(np.log(np.asarray(values)).tolist()))
        self.assertEqual(model.censored, ())

    def test_each_model_evaluates_like_the_model_of_the_values(self) -> None:
        cases = (
            (FamilyId.NORMAL, NormalState, np.random.default_rng(1).normal(2.0, 1.5, 400)),
            (FamilyId.LOGNORMAL, LognormalState, np.random.default_rng(2).lognormal(0.3, 0.6, 400)),
            (FamilyId.GAMMA, GammaState, np.random.default_rng(3).gamma(2.5, 1.5, 400)),
        )
        thetas = {
            FamilyId.NORMAL: (1.7, 1.4),
            FamilyId.LOGNORMAL: (0.25, 0.65),
            FamilyId.GAMMA: (2.7, 1.3),
        }
        for family, state_type, values in cases:
            statistics = folded(state_type, values).finalize()
            expected = build_model(family, tuple(values.tolist()), ())
            actual = model_of(statistics)
            theta = thetas[family]
            reference, candidate = expected.evaluate(theta), actual.evaluate(theta)
            with self.subTest(family=family):
                self.assertAlmostEqual(
                    candidate.value / reference.value, 1.0, delta=1e-13
                )
                for left, right in zip(candidate.gradient, reference.gradient, strict=True):
                    self.assertAlmostEqual(left, right, delta=1e-10 * (1.0 + abs(right)))
                for row_left, row_right in zip(candidate.hessian, reference.hessian, strict=True):
                    for left, right in zip(row_left, row_right, strict=True):
                        self.assertAlmostEqual(left / right, 1.0, delta=1e-12)


class StateEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        values = np.random.default_rng(4).normal(5.0, 2.0, 300)
        self.statistics = folded(NormalState, values).finalize()
        self.evidence = StateEvidence(self.statistics)
        self.parameters = {"mu": 5.0, "sigma": 2.0}

    def test_it_is_a_fit_evidence_without_observations(self) -> None:
        self.assertIsInstance(self.evidence, FitEvidence)
        self.assertEqual(self.evidence.exact, ())
        self.assertEqual(self.evidence.censored, ())
        self.assertEqual(self.evidence.fixed, ())
        self.assertIs(self.evidence.statistics, self.statistics)

    def test_the_covariance_inverts_the_observed_information_of_the_statistics(self) -> None:
        result = self.evidence.uncertainty(FamilyId.NORMAL, self.parameters)
        assert isinstance(result, FitUncertainty)
        self.assertEqual(result.family, FamilyId.NORMAL)
        self.assertEqual(result.parameter_names, ("mu", "sigma"))
        self.assertEqual(result.estimates, (5.0, 2.0))
        information = model_of(self.statistics).evaluate((5.0, 2.0)).hessian
        for row in range(2):
            for column in range(2):
                product = math.fsum(
                    result.covariance[row][k] * -information[k][column] for k in range(2)
                )
                self.assertAlmostEqual(product, 1.0 if row == column else 0.0, delta=1e-12)

    def test_the_standard_errors_of_a_normal_fit_follow_the_closed_form(self) -> None:
        mu = self.statistics.mean.rounded
        centered = self.statistics.centered_sum_of_squares
        sigma = math.sqrt(float(centered.exact / 300))
        result = self.evidence.uncertainty(FamilyId.NORMAL, {"mu": mu, "sigma": sigma})
        assert isinstance(result, FitUncertainty)
        self.assertAlmostEqual(result.standard_errors["mu"], sigma / math.sqrt(300), delta=1e-14)
        self.assertAlmostEqual(
            result.standard_errors["sigma"], sigma / math.sqrt(600), delta=1e-14
        )

    def test_the_result_is_cached_per_parameter_vector(self) -> None:
        first = self.evidence.uncertainty(FamilyId.NORMAL, self.parameters)
        self.assertIs(self.evidence.uncertainty(FamilyId.NORMAL, self.parameters), first)
        other = self.evidence.uncertainty(FamilyId.NORMAL, {"mu": 5.0, "sigma": 2.5})
        self.assertIsNot(other, first)
        assert isinstance(other, FitUncertainty)
        self.assertEqual(other.estimates, (5.0, 2.5))

    def test_statistics_that_cannot_be_prepared_are_unavailable_not_an_error(self) -> None:
        huge = exact(Fraction(10**300))
        broken = type(self.statistics)(
            3, huge, ExactStatistic(Fraction(10**400), 1.0), huge, huge
        )
        result = StateEvidence(broken).uncertainty(FamilyId.NORMAL, self.parameters)
        self.assertIsInstance(result, UncertaintyUnavailable)
        assert isinstance(result, UncertaintyUnavailable)
        self.assertEqual(result.reason, UncertaintyUnavailableReason.NOT_COMPUTABLE)

    def test_a_singular_information_is_reported_as_unavailable(self) -> None:
        result = self.evidence.uncertainty(FamilyId.NORMAL, {"mu": 5.0, "sigma": 1e200})
        self.assertIsInstance(result, UncertaintyUnavailable)


if __name__ == "__main__":
    unittest.main()
