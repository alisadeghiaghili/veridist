"""The facts a fit records about its rows and its maximum: summary, claim and diagnostics."""

from __future__ import annotations

import dataclasses
import unittest

from veridist import (
    ExactLifetime,
    ExactValue,
    IntervalCensoredLifetime,
    LeftCensoredLifetime,
    LeftCensoredValue,
    RightCensoredLifetime,
)
from veridist.families._fit_facts import (
    FitDiagnostics,
    ObservationSummary,
    OptimumKind,
    summarize_observations,
)


class SummaryTests(unittest.TestCase):
    def test_every_kind_and_every_truncation_is_counted_once(self) -> None:
        rows = [
            ExactLifetime(2.0),
            ExactLifetime(3.0, truncation_lower=1.0),
            RightCensoredLifetime(3.0, truncation_upper=9.0),
            LeftCensoredLifetime(4.0, truncation_lower=1.0, truncation_upper=9.0),
            IntervalCensoredLifetime(1.0, 3.0),
            IntervalCensoredLifetime(1.0, 3.0, truncation_lower=0.5),
        ]
        summary = summarize_observations(rows)
        self.assertEqual(
            dataclasses.astuple(summary),
            (2, 1, 1, 2, 2, 2, 1, 1),
        )
        self.assertEqual(summary.total, 6)
        self.assertEqual(summary.censored, 4)
        self.assertEqual(summary.truncated, 4)
        self.assertTrue(summary.is_new_data)

    def test_plain_rows_are_not_new_data(self) -> None:
        summary = summarize_observations([ExactValue(1.0), ExactValue(2.0)])
        self.assertFalse(summary.is_new_data)
        self.assertEqual(summary.truncated, 0)
        self.assertEqual(summary.censored, 0)
        plain = summarize_observations([ExactLifetime(1.0), RightCensoredLifetime(2.0)])
        self.assertFalse(plain.is_new_data)
        self.assertEqual((plain.exact, plain.right_censored), (1, 1))

    def test_each_new_feature_alone_is_new_data(self) -> None:
        for row in (
            LeftCensoredValue(1.0),
            ExactValue(1.0, truncation_lower=0.0 - 1.0),
            ExactValue(1.0, truncation_upper=2.0),
        ):
            self.assertTrue(summarize_observations([row]).is_new_data)
        self.assertTrue(summarize_observations([IntervalCensoredLifetime(1.0, 2.0)]).is_new_data)

    def test_an_empty_sample_and_a_foreign_object(self) -> None:
        self.assertEqual(summarize_observations([]), ObservationSummary())
        self.assertEqual(ObservationSummary().total, 0)
        with self.assertRaises(TypeError):
            summarize_observations([1.0])

    def test_the_counts_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            ObservationSummary(exact=-1, untruncated=-1)
        with self.assertRaises(ValueError):
            ObservationSummary(exact=True, untruncated=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ObservationSummary(exact=1.0, untruncated=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ObservationSummary(exact=2, untruncated=1)
        with self.assertRaises(ValueError):
            ObservationSummary(exact=1, untruncated=0, left_truncated=2)

    def test_the_summary_is_frozen(self) -> None:
        summary = ObservationSummary(exact=1, untruncated=1)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            summary.exact = 2  # type: ignore[misc]
        self.assertEqual(summary, ObservationSummary(exact=1, untruncated=1))
        self.assertEqual(hash(summary), hash(ObservationSummary(exact=1, untruncated=1)))


class OptimumTests(unittest.TestCase):
    def test_the_closed_set_of_claims(self) -> None:
        self.assertEqual(
            {kind.value for kind in OptimumKind},
            {"unassessed", "closed_form", "global_maximum", "stationary_point"},
        )


class DiagnosticsTests(unittest.TestCase):
    def test_the_defaults_are_a_single_newton_start(self) -> None:
        facts = FitDiagnostics(passes=6, iterations=4)
        self.assertEqual(
            (facts.shifted_directions, facts.stencil_failures, facts.starts), (0, 0, 1)
        )
        self.assertEqual(facts.distinct_stationary_points, 1)
        self.assertIsNone(facts.stationary_points_seen)
        self.assertEqual(facts.method, "newton")

    def test_the_counters_are_non_negative_integers(self) -> None:
        for name in (
            "passes",
            "iterations",
            "shifted_directions",
            "stencil_failures",
            "starts",
            "distinct_stationary_points",
        ):
            for bad in (-1, 1.5, True):
                values = {"passes": 1, "iterations": 1, name: bad}
                with self.assertRaises(ValueError, msg=(name, bad)):
                    FitDiagnostics(**values)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
