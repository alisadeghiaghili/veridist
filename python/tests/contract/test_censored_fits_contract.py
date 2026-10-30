"""In-memory fits of left- and interval-censored and truncated data (CENS-FIT, CENS-PROV).

``fit`` and ``fit_<family>`` take every cell of the combination matrix without analytic
weights: each result is the family's existing result type (or its failure type with the family's
codes), carrying the counts of its rows by kind and truncation (``observation_summary``), the
coarsening assumption behind the estimate, and what is claimed about the point (``optimum``).
Data the earlier releases accepted keeps its results bit for bit.
"""

from __future__ import annotations

import math
import unittest
from itertools import product
from typing import Any
from unittest import mock

from tests.reference import censored_support as support
from tests.reference.censored_fit_goldens import GOLDEN
from veridist import (
    CoarseningAssumption,
    ExactLifetime,
    ExactValue,
    FamilyId,
    IntervalCensoredLifetime,
    LeftCensoredLifetime,
    RightCensoredLifetime,
    fit,
    fit_exponential,
    fit_gamma,
    fit_gumbel_right,
    fit_lognormal,
    fit_normal,
    fit_weibull,
)
from veridist.families import FitSuccess, fit_exponential_chunks
from veridist.families._censored_rows import build_rows
from veridist.families._evidence import FitEvidence
from veridist.families._fit_facts import OptimumKind
from veridist.families.exponential import ExponentialFitFailure, ExponentialFitSuccess
from veridist.families.uncertainty import UncertaintyUnavailable, UncertaintyUnavailableReason
from veridist.reporting.exponential import ReportLocale, render_exponential_report

LIFETIME_FAMILIES = ("exponential", "weibull_min", "lognormal", "gamma")
VALUE_FAMILIES = ("normal", "gumbel_right")
#: The gamma fits cost many kernel evaluations: its cells are few here (the whole grid is in
#: ``tests/statistical/test_censored_gamma_reference.py``).
FAST_FAMILIES = ("exponential", "weibull_min", "lognormal", "normal", "gumbel_right")
FITS = {
    "exponential": fit_exponential,
    "weibull_min": fit_weibull,
    "lognormal": fit_lognormal,
    "gamma": fit_gamma,
    "normal": fit_normal,
    "gumbel_right": fit_gumbel_right,
}
KINDS = ("exact", "right", "left", "interval")
TRUNCATIONS = ("none", "left", "right", "two_sided")
WINDOWS = {
    "none": None,
    "left": (0.5, math.inf),
    "right": (-math.inf, 11.0),
    "two_sided": (0.5, 11.0),
}
TIMES = (2.2, 3.1, 4.7, 1.9, 5.9, 3.5, 2.8, 4.1, 6.3, 3.3)


def cell_sample(family: str, kind: str, truncation: str) -> list[Any]:
    """Ten exact rows and four rows of ``kind``, every row with the window of ``truncation``."""

    window = WINDOWS[truncation]
    rows = [support.observation(family, "exact", time, None, window) for time in TIMES]
    for offset in (0.0, 0.4, 0.8, 1.2):
        point = 3.0 + offset
        upper = point + 1.5 if kind == "interval" else None
        rows.append(support.observation(family, kind, point, upper, window))
    return rows


def is_new(kind: str, truncation: str) -> bool:
    return kind in ("left", "interval") or truncation != "none"


#: The stored results of the earlier release come from one platform's libm and BLAS.  Where the
#: fit is iterative (a bounded golden-section or Newton search that stops at a relative
#: tolerance of about 1e-9) the last digits that a stopping rule leaves depend on the platform:
#: the stored values agree with another platform's to about 2e-8 (relative), and the check against
#: them allows ``RELEASE_TOLERANCE`` (the accuracy of those searches, 1e-7).  A closed form made
#: of correctly rounded operations only (sums by ``fsum``, divisions, ``sqrt``) is the same on
#: every platform and is compared exactly.
RELEASE_TOLERANCE = 1e-7
CLOSED_FORM_PARAMETERS = frozenset({"exponential/exact", "exponential/right", "normal/exact"})


def earlier_release_observations(key: str, rows: Any) -> list[Any]:
    family = key.split("/")[0]
    lifetime = family in LIFETIME_FAMILIES
    exact_type = ExactLifetime if lifetime else ExactValue
    right_type = support.TYPES[lifetime]["right"]
    return [exact_type(v) if k else right_type(v) for k, v in rows]


class GoldenTests(unittest.TestCase):
    def test_data_the_earlier_releases_accepted_keeps_its_results_bit_for_bit(self) -> None:
        """Data of exact and right-censored rows without a window never reaches the new
        estimator: ``fit`` returns what the family's own earlier code path returns, bit for bit.

        The expected result is computed in this process, by the family's fit with the new
        estimator made unreachable (so the earlier path is the only one that can produce it), and
        compared with ``fit`` and with a second call; equal floats are compared with ``==``.
        """

        def unreachable(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("the new estimator was reached")

        for key, (rows, _, _) in GOLDEN.items():
            family = key.split("/")[0]
            observations = earlier_release_observations(key, rows)
            with self.subTest(case=key):
                with mock.patch("veridist.families._censored_fit.estimate", unreachable):
                    expected = FITS[family](observations)
                self.assertTrue(hasattr(expected, "parameters"), expected)
                for result in (fit(family, observations), FITS[family](observations)):
                    self.assertEqual(
                        tuple(result.parameters.values()), tuple(expected.parameters.values())
                    )
                    self.assertEqual(result.log_likelihood, expected.log_likelihood)
                    self.assertEqual(result, expected)
                    self.assertIs(result.optimum, OptimumKind.UNASSESSED)
                    self.assertIsNone(result.coarsening_assumption)
                    self.assertIsNone(result.diagnostics)

    def test_data_the_earlier_releases_accepted_keeps_the_results_they_recorded(self) -> None:
        for key, (rows, parameters, likelihood) in GOLDEN.items():
            family = key.split("/")[0]
            observations = earlier_release_observations(key, rows)
            with self.subTest(case=key):
                result = fit(family, observations)
                found = tuple(result.parameters.values())
                if key in CLOSED_FORM_PARAMETERS:
                    self.assertEqual(found, parameters)
                else:
                    for value, recorded in zip(found, parameters, strict=True):
                        self.assertAlmostEqual(value / recorded, 1.0, delta=RELEASE_TOLERANCE)
                self.assertAlmostEqual(
                    result.log_likelihood, likelihood, delta=RELEASE_TOLERANCE * abs(likelihood)
                )
                self.assertIs(result.optimum, OptimumKind.UNASSESSED)
                self.assertIsNone(result.coarsening_assumption)
                self.assertIsNone(result.diagnostics)
                summary = result.observation_summary
                self.assertEqual(summary.exact, sum(k for k, _ in rows))
                self.assertEqual(summary.right_censored, sum(1 - k for k, _ in rows))
                self.assertEqual(summary.untruncated, len(rows))
                self.assertFalse(summary.is_new_data)
                self.assertEqual(result.event_count, summary.exact)
                self.assertEqual(result.censored_count, summary.censored)

    def test_the_exponential_keeps_its_censoring_assumption_for_that_data(self) -> None:
        result = fit_exponential([ExactLifetime(2.0), RightCensoredLifetime(3.0)])
        self.assertEqual(result.censoring_assumption, "independent_right_censoring")
        self.assertEqual(result.provenance.state_complexity, "O(1)")
        self.assertIs(result.optimum, OptimumKind.UNASSESSED)

    def test_failures_of_that_data_keep_their_codes(self) -> None:
        self.assertEqual(fit_normal([]).code.value, "EMPTY_SAMPLE")
        self.assertEqual(fit_normal([ExactValue(1.0)]).code.value, "DEGENERATE_SAMPLE")
        self.assertEqual(
            fit_lognormal([RightCensoredLifetime(2.0)]).code.value, "NO_OBSERVED_EVENTS"
        )
        self.assertEqual(
            fit_lognormal([ExactLifetime(2.0), ExactLifetime(2.0)]).code.value,
            "OPTIMIZER_EXHAUSTED",
        )
        failure = fit_gamma([])
        self.assertIsNone(failure.coarsening_assumption)
        self.assertEqual(failure.observation_summary.total, 0)


class CellTests(unittest.TestCase):
    def check_family_cells(self, family: str) -> None:
        for kind, truncation in product(KINDS, TRUNCATIONS):
            with self.subTest(family=family, kind=kind, truncation=truncation):
                sample = cell_sample(family, kind, truncation)
                result = FITS[family](sample)
                self.assertEqual(result.observation_count, 14)
                self.assertEqual(result.event_count, 14 if kind == "exact" else 10)
                self.assertEqual(result.censored_count, 0 if kind == "exact" else 4)
                summary = result.observation_summary
                self.assertEqual(summary.total, 14)
                self.assertEqual(summary.is_new_data, is_new(kind, truncation))
                self.assertEqual(summary.truncated, 0 if truncation == "none" else 14)
                new = is_new(kind, truncation)
                self.assertEqual(
                    result.coarsening_assumption,
                    CoarseningAssumption.INDEPENDENT_CENSORING_AND_TRUNCATION if new else None,
                )
                self.assertTrue(isinstance(result, FitSuccess), result)
                self.assertTrue(result.converged)
                self.assertTrue(math.isfinite(result.log_likelihood))
                self.assertEqual(result.diagnostics is not None, new)

    def test_the_dispatcher_and_the_family_function_agree(self) -> None:
        for family in FAST_FAMILIES:
            sample = cell_sample(family, "interval", "two_sided")
            self.assertEqual(FITS[family](sample), fit(family, sample))

    def check_claim(self, family: str) -> None:
        concave = OptimumKind.GLOBAL_MAXIMUM
        stationary = OptimumKind.STATIONARY_POINT
        for kind, truncation in product(KINDS, TRUNCATIONS):
            if not is_new(kind, truncation):
                continue
            with self.subTest(family=family, kind=kind, truncation=truncation):
                result = FITS[family](cell_sample(family, kind, truncation))
                if family == "exponential":
                    if kind in ("exact", "right") and truncation == "left":
                        expected = OptimumKind.CLOSED_FORM
                    else:
                        expected = concave if truncation in ("none", "left") else stationary
                else:
                    expected = concave if truncation == "none" else stationary
                self.assertIs(result.optimum, expected)
                if expected is concave:
                    self.assertEqual(result.diagnostics.starts, 1)
                elif expected is stationary and result.diagnostics.method == "newton":
                    self.assertEqual(result.diagnostics.starts, 2)

    def test_the_gamma_has_no_concavity_guarantee_and_tries_two_starts(self) -> None:
        sample = cell_sample("gamma", "left", "none")
        result = fit_gamma(sample)
        self.assertIs(result.optimum, OptimumKind.STATIONARY_POINT)
        self.assertEqual(result.diagnostics.starts, 2)
        self.assertEqual(result.observation_summary.left_censored, 4)
        self.assertEqual(result, fit("gamma", sample))
        self.assertIs(
            result.coarsening_assumption, CoarseningAssumption.INDEPENDENT_CENSORING_AND_TRUNCATION
        )

    def test_the_left_truncated_weibull_profile_records_its_stationary_points(self) -> None:
        for kind in ("exact", "right"):
            result = fit_weibull(cell_sample("weibull_min", kind, "left"))
            self.assertEqual(result.diagnostics.method, "profile")
            self.assertEqual(result.diagnostics.stationary_points_seen, 1)
            self.assertIs(result.optimum, OptimumKind.STATIONARY_POINT)
        newton = fit_weibull(cell_sample("weibull_min", "left", "left"))
        self.assertEqual(newton.diagnostics.method, "newton")
        self.assertIsNone(newton.diagnostics.stationary_points_seen)

    def test_the_results_have_the_existing_types_and_fields(self) -> None:
        from veridist.families import (
            GumbelFitSuccess,
            LognormalFitSuccess,
            NormalFitSuccess,
            WeibullFitSuccess,
        )

        expected = {
            "exponential": ExponentialFitSuccess,
            "weibull_min": WeibullFitSuccess,
            "lognormal": LognormalFitSuccess,
            "normal": NormalFitSuccess,
            "gumbel_right": GumbelFitSuccess,
        }
        for family, kind in expected.items():
            result = FITS[family](cell_sample(family, "interval", "two_sided"))
            self.assertIs(type(result), kind)
            self.assertEqual(result.family, FamilyId(family))
            self.assertEqual(list(result.parameters), list(support_names(family)))

    def test_the_summary_counts_every_kind_and_every_truncation(self) -> None:
        sample = [
            ExactLifetime(2.0),
            RightCensoredLifetime(3.0, truncation_lower=1.0),
            LeftCensoredLifetime(4.0, truncation_upper=9.0),
            IntervalCensoredLifetime(1.0, 3.0, truncation_lower=0.5, truncation_upper=9.0),
            ExactLifetime(2.5, truncation_lower=1.0),
        ]
        summary = fit_weibull(sample).observation_summary
        self.assertEqual(
            (
                summary.exact,
                summary.right_censored,
                summary.left_censored,
                summary.interval_censored,
            ),
            (2, 1, 1, 1),
        )
        self.assertEqual(
            (
                summary.untruncated,
                summary.left_truncated,
                summary.right_truncated,
                summary.two_sided_truncated,
            ),
            (1, 2, 1, 1),
        )

    def test_a_fit_is_the_same_for_every_run(self) -> None:
        sample = cell_sample("lognormal", "interval", "two_sided")
        first, second = fit_lognormal(sample), fit_lognormal(list(reversed(sample)))
        self.assertEqual(first, second)
        self.assertEqual(first.diagnostics, second.diagnostics)


def _generate() -> None:
    for family in FAST_FAMILIES:

        def cells(self: CellTests, family: str = family) -> None:
            self.check_family_cells(family)

        def claim(self: CellTests, family: str = family) -> None:
            self.check_claim(family)

        setattr(CellTests, f"test_every_{family}_cell_is_fitted", cells)
        setattr(CellTests, f"test_the_claim_for_every_{family}_cell", claim)


_generate()


def support_names(family: str) -> tuple[str, ...]:
    return {
        "exponential": ("rate",),
        "weibull_min": ("shape", "scale"),
        "lognormal": ("mu_log", "sigma_log"),
        "gamma": ("shape", "scale"),
        "normal": ("mu", "sigma"),
        "gumbel_right": ("location", "scale"),
    }[family]


class FailureTests(unittest.TestCase):
    def test_a_failure_carries_the_summary_and_the_assumption(self) -> None:
        sample = [RightCensoredLifetime(3.0, truncation_lower=1.0), RightCensoredLifetime(4.0)]
        for family in ("weibull_min", "lognormal", "gamma"):
            failure = FITS[family](sample)
            self.assertEqual(failure.code.value, "NO_OBSERVED_EVENTS")
            self.assertEqual(failure.observation_count, 2)
            self.assertEqual(failure.event_count, 0)
            self.assertEqual(failure.censored_count, 2)
            self.assertEqual(failure.observation_summary.right_censored, 2)
            self.assertEqual(failure.observation_summary.left_truncated, 1)
            self.assertIs(
                failure.coarsening_assumption,
                CoarseningAssumption.INDEPENDENT_CENSORING_AND_TRUNCATION,
            )
        failure = fit_exponential(sample)
        self.assertEqual((failure.code.value, failure.total_time), ("NO_OBSERVED_EVENTS", 7.0))
        self.assertIs(
            failure.coarsening_assumption, CoarseningAssumption.INDEPENDENT_CENSORING_AND_TRUNCATION
        )

    def test_a_left_or_interval_row_is_informative_so_it_is_not_no_observed_events(self) -> None:
        sample = [
            RightCensoredLifetime(3.0),
            RightCensoredLifetime(4.0),
            IntervalCensoredLifetime(1.0, 6.0),
        ]
        result = fit_exponential(sample)
        self.assertIsInstance(result, ExponentialFitSuccess)
        self.assertEqual(result.event_count, 0)
        self.assertEqual(result.censored_count, 3)

    def test_the_codes_of_the_families_are_the_existing_ones(self) -> None:
        for family in ("normal", "gumbel_right"):
            degenerate = FITS[family](
                [ExactValue(5.0), support.observation(family, "left", 5.0, None, None)]
            )
            self.assertEqual(degenerate.code.value, "DEGENERATE_SAMPLE")
        lognormal = fit_lognormal([ExactLifetime(5.0), LeftCensoredLifetime(5.0)])
        self.assertEqual(lognormal.code.value, "DEGENERATE_SAMPLE")
        self.assertIn("DEGENERATE_SAMPLE", {code.value for code in type(lognormal.code)})
        for family in ("weibull_min", "lognormal", "gamma"):
            invalid = FITS[family](
                [ExactLifetime(0.0), LeftCensoredLifetime(2.0), ExactLifetime(1.0)]
            )
            self.assertEqual(invalid.code.value, "INVALID_SUPPORT")

    def test_a_flat_likelihood_is_never_an_estimate(self) -> None:
        sample = [LeftCensoredLifetime(5.0), LeftCensoredLifetime(6.0), LeftCensoredLifetime(7.0)]
        for family in ("exponential", "weibull_min", "lognormal"):
            with self.subTest(family=family):
                failure = FITS[family](sample)
                self.assertEqual(failure.code.value, "OPTIMIZER_EXHAUSTED")
                self.assertFalse(hasattr(failure, "parameters"))
        with self.assertRaises(ValueError):
            ExponentialFitFailure(
                type(fit_exponential(sample).code).OPTIMIZER_EXHAUSTED, 0, 0, None
            )

    def test_the_exponential_failures_of_the_iterative_cells_have_facts_that_match(self) -> None:
        codes = type(fit_exponential(sample_flat()).code)
        self.assertEqual(
            {code.value for code in codes},
            {
                "EMPTY_SAMPLE",
                "NO_OBSERVED_EVENTS",
                "UNBOUNDED_LIKELIHOOD",
                "NUMERICAL_OVERFLOW",
                "OPTIMIZER_EXHAUSTED",
                "BOUNDARY_SOLUTION",
            },
        )
        for code in (codes.OPTIMIZER_EXHAUSTED, codes.BOUNDARY_SOLUTION):
            self.assertEqual(ExponentialFitFailure(code, 3, 0, None).code, code)
            for total in (0.0, 2.5):
                with self.assertRaises(ValueError):
                    ExponentialFitFailure(code, 3, 0, total)
            with self.assertRaises(ValueError):
                ExponentialFitFailure(code, 0, 0, None)

    def test_exact_times_at_their_entry_times_are_an_unbounded_likelihood(self) -> None:
        failure = fit_exponential(
            [ExactLifetime(2.0, truncation_lower=2.0), ExactLifetime(3.0, truncation_lower=3.0)]
        )
        self.assertEqual(failure.code.value, "UNBOUNDED_LIKELIHOOD")
        self.assertEqual(failure.total_time, 0.0)


def sample_flat() -> list[Any]:
    return [LeftCensoredLifetime(5.0), LeftCensoredLifetime(6.0)]


class WeightTests(unittest.TestCase):
    def test_integer_frequency_weights_equal_replicated_rows_bit_for_bit(self) -> None:
        for family in FAST_FAMILIES[1:]:
            sample = cell_sample(family, "interval", "left")
            weights = [1, 2, 3, 1, 2, 1, 1, 3, 1, 2, 1, 2, 1, 1]
            replicated = [
                row for row, weight in zip(sample, weights, strict=True) for _ in range(weight)
            ]
            with self.subTest(family=family):
                weighted = FITS[family](sample, frequency_weights=weights)
                direct = FITS[family](replicated)
                self.assertEqual(weighted, direct)
                self.assertEqual(weighted.log_likelihood, direct.log_likelihood)
                self.assertEqual(weighted.observation_count, sum(weights))
                self.assertEqual(weighted.observation_summary, direct.observation_summary)

    def test_a_zero_weight_removes_the_row_and_a_bad_weight_is_refused(self) -> None:
        sample = cell_sample("normal", "left", "none")
        zero = [0, *([1] * 13)]
        self.assertEqual(fit_normal(sample, frequency_weights=zero), fit_normal(sample[1:]))
        for bad, error in ((-1, TypeError), (1.5, TypeError), (True, TypeError)):
            with self.assertRaises(error):
                fit_normal(sample, frequency_weights=[bad] + [1] * 13)
        with self.assertRaises(ValueError):
            fit_normal(sample, frequency_weights=[1, 2])


class ExponentialStreamTests(unittest.TestCase):
    def test_a_generator_of_new_data_is_fitted_and_only_the_new_rows_are_kept(self) -> None:
        sample = cell_sample("exponential", "interval", "none")
        listed = fit_exponential(sample)
        streamed = fit_exponential(row for row in sample)
        self.assertAlmostEqual(streamed.rate, listed.rate, delta=1e-12 * listed.rate)
        self.assertEqual(streamed.observation_summary, listed.observation_summary)

    def test_chunks_are_fitted(self) -> None:
        sample = cell_sample("exponential", "left", "left")
        whole = fit_exponential_chunks([sample])
        parts = fit_exponential_chunks([sample[:5], sample[5:9], [], sample[9:]])
        self.assertAlmostEqual(parts.rate, whole.rate, delta=1e-12 * whole.rate)
        self.assertEqual(parts.observation_count, 14)

    def test_the_plain_rows_are_summarised_and_the_result_equals_the_fit_of_all_rows(self) -> None:
        extra = [LeftCensoredLifetime(3.0), IntervalCensoredLifetime(1.0, 6.0)]
        plain = [ExactLifetime(v) for v in TIMES] + [RightCensoredLifetime(7.0)] * 3
        result = fit_exponential([*plain, *extra])
        self.assertEqual(result.observation_summary.exact, 10)
        self.assertEqual(result.observation_summary.right_censored, 3)
        self.assertEqual(result.observation_summary.untruncated, 15)
        # the same likelihood: all rows kept explicitly
        explicit = fit_exponential(
            [
                *[ExactLifetime(v, truncation_upper=99.0) for v in TIMES],
                *[RightCensoredLifetime(7.0, truncation_upper=99.0)] * 3,
                *extra,
            ]
        )
        self.assertEqual(result.event_count, explicit.event_count)

    def test_new_rows_after_an_overflow_of_the_plain_rows_are_a_numerical_overflow(self) -> None:
        sample = [ExactLifetime(1.7e308)] * 3 + [LeftCensoredLifetime(2.0)]
        failure = fit_exponential(sample)
        self.assertEqual(failure.code.value, "NUMERICAL_OVERFLOW")
        self.assertEqual(failure.observation_count, 4)
        self.assertEqual(failure.event_count, 3)
        self.assertIsNone(failure.total_time)

    def test_an_object_that_is_not_a_lifetime_is_still_a_type_error(self) -> None:
        with self.assertRaises(TypeError):
            fit_exponential([LeftCensoredLifetime(3.0), ExactValue(1.0)])
        with self.assertRaises(TypeError):
            fit_exponential([LeftCensoredLifetime(3.0), 2.0])
        with self.assertRaises(TypeError):
            fit_exponential_chunks([[LeftCensoredLifetime(3.0)], 7])

    def test_the_result_validates_what_it_claims(self) -> None:
        result = fit_exponential(cell_sample("exponential", "left", "right"))
        self.assertEqual(result.censoring_assumption, "independent_censoring_and_truncation")
        self.assertEqual(result.provenance.state_complexity, "O(n)")
        self.assertTrue(result.provenance.raw_data_retained)
        self.assertEqual(result.mean, 1.0 / result.rate)
        closed = fit_exponential(cell_sample("exponential", "exact", "left"))
        self.assertEqual(closed.optimum, OptimumKind.CLOSED_FORM)
        self.assertEqual(closed.provenance.state_complexity, "O(1)")
        self.assertEqual(closed.rate, closed.event_count / closed.total_time)
        self.assertEqual(
            closed.log_likelihood,
            closed.event_count * math.log(closed.rate) - closed.rate * closed.total_time,
        )
        with self.assertRaises(ValueError):
            ExponentialFitSuccess(
                result.rate,
                3,
                3,
                2.0,
                result.mean,
                -1.0,
                0,
                censoring_assumption="independent_right_censoring",
                observation_summary=result.observation_summary,
                optimum=OptimumKind.GLOBAL_MAXIMUM,
            )
        with self.assertRaises(ValueError):
            ExponentialFitSuccess(0.5, 3, 3, 2.0, 2.0, -1.0, 0, optimum=OptimumKind.CLOSED_FORM)


class ReportTests(unittest.TestCase):
    def test_the_report_of_a_new_data_result_names_its_assumption(self) -> None:
        result = fit_exponential(cell_sample("exponential", "interval", "none"))
        for locale in ReportLocale:
            html = render_exponential_report(result, locale)
            self.assertIn("independent_censoring_and_truncation", html)
            self.assertNotIn("independent_right_censoring", html)
        failure = fit_exponential(sample_flat())
        for locale in ReportLocale:
            html = render_exponential_report(failure, locale)
            self.assertIn("OPTIMIZER_EXHAUSTED", html)
            self.assertIn("independent_censoring_and_truncation", html)

    def test_the_report_of_earlier_data_is_unchanged(self) -> None:
        result = fit_exponential([ExactLifetime(2.0), RightCensoredLifetime(3.0)])
        html = render_exponential_report(result, ReportLocale.EN)
        self.assertIn("independent_right_censoring", html)
        failure = fit_exponential([RightCensoredLifetime(3.0)])
        self.assertIn(
            "independent_right_censoring", render_exponential_report(failure, ReportLocale.EN)
        )

    def test_the_boundary_failure_has_a_message_in_every_locale(self) -> None:
        failure = ExponentialFitFailure(
            type(fit_exponential(sample_flat()).code).BOUNDARY_SOLUTION, 3, 0, None
        )
        messages = {render_exponential_report(failure, locale) for locale in ReportLocale}
        self.assertEqual(len(messages), 3)


class UncertaintyTests(unittest.TestCase):
    def test_the_gamma_has_a_covariance_and_wald_intervals(self) -> None:
        result = fit_gamma(cell_sample("gamma", "interval", "left"))
        uncertainty = result.uncertainty()
        self.assertNotIsInstance(uncertainty, UncertaintyUnavailable)
        self.assertGreater(
            uncertainty.covariance[0][0] * uncertainty.covariance[1][1],
            uncertainty.covariance[0][1] ** 2,
        )
        intervals = uncertainty.interval_details(0.9, "wald")
        for name, value in result.parameters.items():
            self.assertLess(intervals[name].lower, value)
            self.assertGreater(intervals[name].upper, value)

    def test_every_family_has_a_covariance_and_wald_and_profile_intervals(self) -> None:
        for family in FAST_FAMILIES:
            sample = cell_sample(family, "interval", "left")
            result = FITS[family](sample)
            with self.subTest(family=family):
                uncertainty = result.uncertainty()
                self.assertNotIsInstance(uncertainty, UncertaintyUnavailable)
                matrix = uncertainty.covariance
                self.assertEqual(len(matrix), len(result.parameters))
                for i, row in enumerate(matrix):
                    self.assertGreater(row[i], 0.0)
                    for j in range(len(matrix)):
                        self.assertEqual(matrix[i][j], matrix[j][i])
                if len(matrix) == 2:
                    self.assertGreater(matrix[0][0] * matrix[1][1], matrix[0][1] ** 2)
                for method in ("wald", "profile"):
                    intervals = uncertainty.interval_details(0.9, method)
                    for name, value in result.parameters.items():
                        interval = intervals[name]
                        self.assertLess(interval.lower, value)
                        self.assertGreater(interval.upper, value)
                        self.assertEqual(interval.method, method)

    def test_the_wald_interval_is_the_estimate_plus_or_minus_its_standard_error(self) -> None:
        result = fit_normal(cell_sample("normal", "left", "two_sided"))
        uncertainty = result.uncertainty()
        errors = uncertainty.standard_errors
        low, high = uncertainty.confidence_intervals(0.95, "wald")["mu"]
        self.assertAlmostEqual(high - result.mu, 1.959963984540054 * errors["mu"], places=9)
        self.assertAlmostEqual(result.mu - low, 1.959963984540054 * errors["mu"], places=9)

    def test_derived_quantities_have_intervals(self) -> None:
        for family in ("weibull_min", "normal"):
            result = FITS[family](cell_sample(family, "interval", "two_sided"))
            uncertainty = result.uncertainty()
            with self.subTest(family=family):
                for estimate in (
                    uncertainty.mean(),
                    uncertainty.quantile(0.5),
                    uncertainty.survival(3.0),
                ):
                    self.assertLess(estimate.lower, estimate.estimate)
                    self.assertGreater(estimate.upper, estimate.estimate)

    def test_an_exponential_profile_interval_of_the_mean(self) -> None:
        exponential = fit_exponential(cell_sample("exponential", "interval", "right"))
        estimate = exponential.uncertainty().mean(0.9, "profile")
        self.assertLess(estimate.lower, estimate.estimate)
        self.assertGreater(estimate.upper, estimate.estimate)

    def test_a_profile_interval_of_a_derived_quantity_is_not_implemented_for_the_normal(
        self,
    ) -> None:
        with self.assertRaises(NotImplementedError):
            fit_normal(cell_sample("normal", "interval", "left")).uncertainty().mean(0.9, "profile")

    def test_the_exact_interval_stays_available_for_uncensored_data_with_a_left_window(
        self,
    ) -> None:
        result = fit_exponential(cell_sample("exponential", "exact", "left"))
        low, high = result.uncertainty().confidence_intervals(0.95, "exact")["rate"]
        self.assertLess(low, result.rate)
        self.assertGreater(high, result.rate)
        for sample in (
            cell_sample("exponential", "right", "left"),
            cell_sample("exponential", "interval", "none"),
            cell_sample("exponential", "exact", "right"),
        ):
            with self.assertRaises(ValueError):
                fit_exponential(sample).uncertainty().confidence_intervals(0.95, "exact")
        with self.assertRaisesRegex(ValueError, "only for the exponential"):
            fit_normal(cell_sample("normal", "left", "none")).uncertainty().confidence_intervals(
                0.95, "exact"
            )

    def test_an_information_that_cannot_be_inverted_is_a_typed_unavailability(self) -> None:
        # Far from the maximum of a truncated sample the observed information is not positive
        # definite: typed, never an exception and never a number.
        normal = FitEvidence(
            rows=build_rows(FamilyId.NORMAL, cell_sample("normal", "interval", "two_sided"))
        )
        far = normal.uncertainty(FamilyId.NORMAL, {"mu": -5.0, "sigma": 50.0})
        assert isinstance(far, UncertaintyUnavailable)
        self.assertIs(far.reason, UncertaintyUnavailableReason.NOT_POSITIVE_DEFINITE)
        gamma = FitEvidence(
            rows=build_rows(FamilyId.GAMMA, cell_sample("gamma", "interval", "none"))
        )
        refused = gamma.uncertainty(FamilyId.GAMMA, {"shape": 1e5, "scale": 1.0})
        assert isinstance(refused, UncertaintyUnavailable)
        self.assertIs(refused.reason, UncertaintyUnavailableReason.NOT_COMPUTABLE)

    def test_the_uncertainty_is_cached_and_the_parameters_are_the_estimates(self) -> None:
        result = fit_gamma(cell_sample("gamma", "left", "none"))
        first = result.uncertainty()
        self.assertIs(result.uncertainty(), first)
        self.assertEqual(first.estimates, tuple(result.parameters.values()))

    def test_a_result_built_by_hand_has_no_data(self) -> None:
        from veridist.families import NormalFitSuccess

        built = NormalFitSuccess(1.0, 2.0, -3.0, 2, 2, 0)
        self.assertIsInstance(built.uncertainty(), UncertaintyUnavailable)
        self.assertIs(built.uncertainty().reason, UncertaintyUnavailableReason.NO_DATA)


class PlainSummaryOverflowTests(unittest.TestCase):
    def test_a_compensated_total_that_overflows_at_the_end_is_a_numerical_overflow(self) -> None:
        from veridist.families.exponential import _fit_new_data
        from veridist.statistics.exponential import ExponentialReductionState

        state = ExponentialReductionState(2, 2, 1.7e308, 1.7e308)
        failure = _fit_new_data(state, [LeftCensoredLifetime(2.0)])
        self.assertEqual(failure.code.value, "NUMERICAL_OVERFLOW")
        self.assertEqual((failure.observation_count, failure.event_count), (3, 2))
        self.assertIsNone(failure.total_time)


class NuisanceEvaluationTests(unittest.TestCase):
    def test_a_vector_that_cannot_be_built_is_not_evaluable(self) -> None:
        from veridist.families._models import ExponentialModel
        from veridist.families.uncertainty import _evaluate_at

        model = ExponentialModel(3, 6.0, 0)

        def overflowing(phi: float) -> tuple[float, ...]:
            return (math.exp(phi),)

        self.assertIsNone(_evaluate_at(model, overflowing, 800.0))
        evaluation = _evaluate_at(model, overflowing, 0.0)
        assert evaluation is not None
        self.assertAlmostEqual(evaluation.value, 3 * math.log(1.0) - 6.0)
        self.assertIsNone(_evaluate_at(model, lambda phi: (-1.0,), 0.0))


if __name__ == "__main__":
    unittest.main()
