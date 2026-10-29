"""The contribution evaluator: classification, closed ratios, failures and batch independence."""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from veridist.domain import (
    ExactLifetime,
    ExactValue,
    IntervalCensoredLifetime,
    IntervalCensoredValue,
    LeftCensoredLifetime,
    LeftCensoredValue,
    RightCensoredLifetime,
    RightCensoredValue,
)
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.statistics import _log_contributions as contributions
from veridist.statistics._log_contributions import ContributionEvaluator, is_new_data
from veridist.statistics.lifetime_log_likelihood import _evaluate_log_survival
from veridist.statistics.log_density import LogDensityErrorCode


def evaluator(family: FamilyId, **parameters: float) -> ContributionEvaluator:
    validated = FAMILY_REGISTRY.families[family].validate_parameters(**parameters)
    return ContributionEvaluator(family, validated, _evaluate_log_survival)


class ClassificationTests(unittest.TestCase):
    def test_only_the_earlier_untruncated_kinds_are_not_new_data(self) -> None:
        old: list[Any] = [
            ExactLifetime(1.0),
            RightCensoredLifetime(1.0),
            ExactValue(1.0),
            RightCensoredValue(1.0),
        ]
        for row in old:
            self.assertFalse(is_new_data(row))
        new: list[Any] = [
            LeftCensoredLifetime(1.0),
            IntervalCensoredLifetime(1.0, 2.0),
            LeftCensoredValue(1.0),
            IntervalCensoredValue(1.0, 2.0),
            ExactLifetime(1.0, truncation_lower=0.5),
            ExactLifetime(1.0, truncation_upper=2.0),
            RightCensoredValue(1.0, truncation_lower=0.5),
            ExactValue(1.0, truncation_upper=2.0),
        ]
        for row in new:
            self.assertTrue(is_new_data(row), row)

    def test_the_batch_size_is_the_documented_one(self) -> None:
        self.assertEqual(contributions.BATCH_SIZE, 512)


class QuotientTests(unittest.TestCase):
    def test_the_log_quotient_is_the_log_of_the_ratio_near_and_far_from_one(self) -> None:
        x = np.array([1.0000001, 3.0, 0.4, 1e-310, 1e300, 2.0])
        a = np.array([1.0, 1.0, 1.0, 1e-300, 1e-10, 2.0])
        got = contributions._log_quotient(x, a)
        for value, xi, ai in zip(got.tolist(), x.tolist(), a.tolist(), strict=True):
            expected = math.log(xi) - math.log(ai)
            self.assertAlmostEqual(value, expected, delta=1e-13 * max(1.0, abs(expected)))
        self.assertEqual(got[-1], 0.0)

    def test_near_one_it_keeps_the_exact_relative_difference(self) -> None:
        a = np.array([1e6])
        x = a * (1.0 + 2.0**-40)
        self.assertEqual(contributions._log_quotient(x, a)[0], math.log1p((x[0] - a[0]) / a[0]))


class ClosedRatioTests(unittest.TestCase):
    def test_the_exponential_terms_are_the_closed_forms(self) -> None:
        ev = evaluator(FamilyId.EXPONENTIAL, rate=0.1)
        rows = [
            RightCensoredLifetime(1e6 + 0.3, truncation_lower=1e6),
            ExactLifetime(1e6 + 0.3, truncation_lower=1e6),
        ]
        right, exact = ev.evaluate(rows)
        self.assertEqual(right, -0.1 * ((1e6 + 0.3) - 1e6))
        self.assertEqual(exact, math.log(0.1) - 0.1 * ((1e6 + 0.3) - 1e6))

    def test_the_ratio_is_used_only_below_truncation_with_an_open_upper_end(self) -> None:
        ev = evaluator(FamilyId.EXPONENTIAL, rate=0.1)
        two_sided = ExactLifetime(5.0, truncation_lower=2.0, truncation_upper=1e12)
        one_sided = ExactLifetime(5.0, truncation_lower=2.0)
        a, b = ev.evaluate([two_sided, one_sided])
        self.assertAlmostEqual(a, b, delta=1e-12)
        kind = contributions.CensoringKind.EXACT
        two = contributions._Pending(0, kind, 5.0, None, (2.0, 1e12), -1.0)
        one = contributions._Pending(0, kind, 5.0, None, (2.0, math.inf), -1.0)
        self.assertEqual(ev._ratio_rows([two]), {})
        self.assertEqual(list(ev._ratio_rows([one])), [0])
        left_bounded = contributions._Pending(
            0, contributions.CensoringKind.LEFT, 5.0, (-math.inf, 5.0), (2.0, math.inf)
        )
        self.assertEqual(ev._ratio_rows([left_bounded]), {})
        right = RightCensoredLifetime(5.0, truncation_upper=9.0)
        left = LeftCensoredLifetime(5.0, truncation_lower=2.0)
        for row in (right, left):
            self.assertTrue(math.isfinite(ev.evaluate([row])[0]))

    def test_other_families_do_not_use_closed_ratios(self) -> None:
        ev = evaluator(FamilyId.LOGNORMAL, mu_log=0.0, sigma_log=1.0)
        pending = [
            contributions._Pending(
                0, contributions.CensoringKind.EXACT, 2.0, None, (1.0, math.inf), -1.0
            )
        ]
        self.assertEqual(ev._ratio_rows(pending), {})

    def test_the_weibull_ratio_matches_the_difference_of_the_two_terms(self) -> None:
        ev = evaluator(FamilyId.WEIBULL_MIN, shape=2.0, scale=3.0)
        rows = [
            RightCensoredLifetime(4.0, truncation_lower=2.5),
            ExactLifetime(4.0, truncation_lower=2.5),
        ]
        right, exact = ev.evaluate(rows)
        survival = lambda x: -((x / 3.0) ** 2.0)  # noqa: E731
        self.assertAlmostEqual(right, survival(4.0) - survival(2.5), delta=1e-14)
        density = math.log(2.0 / 3.0) + math.log(4.0 / 3.0) - (4.0 / 3.0) ** 2.0
        self.assertAlmostEqual(exact, density - survival(2.5), delta=1e-14)


class FailureTests(unittest.TestCase):
    def test_a_support_violation_is_a_code_not_an_exception(self) -> None:
        ev = evaluator(FamilyId.WEIBULL_MIN, shape=2.0, scale=3.0)
        outcomes = ev.evaluate([ExactLifetime(0.0), RightCensoredLifetime(0.0), ExactLifetime(1.0)])
        self.assertEqual(outcomes[0], LogDensityErrorCode.SUPPORT_VIOLATION)
        self.assertEqual(outcomes[1], LogDensityErrorCode.SUPPORT_VIOLATION)
        self.assertIsInstance(outcomes[2], float)

    def test_a_survival_term_the_earlier_evaluator_refuses_is_its_code(self) -> None:
        ev = evaluator(FamilyId.WEIBULL_MIN, shape=50.0, scale=1.0)
        outcome = ev.evaluate([RightCensoredLifetime(1e10)])
        self.assertEqual(outcome, [LogDensityErrorCode.NUMERICAL_OVERFLOW])

    def test_a_term_that_is_not_finite_is_its_code(self) -> None:
        self.assertEqual(contributions._settle(1.5), 1.5)
        self.assertEqual(contributions._settle(-math.inf), LogDensityErrorCode.NUMERICAL_OVERFLOW)
        self.assertEqual(contributions._settle(math.inf), LogDensityErrorCode.NONFINITE_LOG_DENSITY)
        self.assertEqual(contributions._settle(math.nan), LogDensityErrorCode.NONFINITE_LOG_DENSITY)

    def test_the_kernel_is_built_only_when_a_row_needs_it(self) -> None:
        ev = evaluator(FamilyId.NORMAL, mu=0.0, sigma=1.0)
        ev.evaluate([ExactValue(0.0), RightCensoredValue(1.0)])
        self.assertIsNone(ev._kernel)
        ev.evaluate([LeftCensoredValue(1.0)])
        self.assertIsNotNone(ev._kernel)
        self.assertIs(ev.kernel(), ev._kernel)

    def test_a_refused_batch_keeps_the_rows_the_kernel_can_serve(self) -> None:
        ev = evaluator(FamilyId.GAMMA, shape=5000.0, scale=1.0)
        rows = [ExactLifetime(5000.0), LeftCensoredLifetime(5000.0), ExactLifetime(5001.0)]
        outcomes = ev.evaluate(rows)
        self.assertIsInstance(outcomes[0], float)
        self.assertEqual(outcomes[1], LogDensityErrorCode.NONFINITE_LOG_DENSITY)
        self.assertIsInstance(outcomes[2], float)
        two = ev.evaluate([LeftCensoredLifetime(5000.0), IntervalCensoredLifetime(1.0, 2.0)])
        self.assertEqual(two, [LogDensityErrorCode.NONFINITE_LOG_DENSITY] * 2)

    def test_an_empty_batch_is_empty(self) -> None:
        self.assertEqual(evaluator(FamilyId.NORMAL, mu=0.0, sigma=1.0).evaluate([]), [])


class IndependenceTests(unittest.TestCase):
    def test_a_row_does_not_depend_on_its_batch(self) -> None:
        ev = evaluator(FamilyId.LOGNORMAL, mu_log=1.0, sigma_log=0.7)
        rows: list[Any] = [
            ExactLifetime(2.0, truncation_upper=30.0),
            RightCensoredLifetime(3.0, truncation_lower=1.0),
            LeftCensoredLifetime(4.0),
            IntervalCensoredLifetime(2.0, 2.0000001),
            IntervalCensoredLifetime(1.0, 9.0, truncation_lower=0.5, truncation_upper=20.0),
            ExactLifetime(2.5),
        ]
        singles = [ev.evaluate([row])[0] for row in rows]
        self.assertEqual(ev.evaluate(rows), singles)
        self.assertEqual(ev.evaluate(rows[::-1]), singles[::-1])
        self.assertEqual(ev.evaluate(rows * 3), singles * 3)


if __name__ == "__main__":
    unittest.main()
