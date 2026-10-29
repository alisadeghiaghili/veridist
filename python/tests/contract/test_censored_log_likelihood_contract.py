"""The streaming log-likelihood reducers with censored, truncated and weighted rows.

CENS-LL-04 (terms equal the exact sum of the scalar terms, invariant to order and chunking),
CENS-PROV-01 (the coarsening assumption is recorded exactly when the data needs it) and the
weights semantics (a frequency weight ``w`` is ``w`` replicated rows, bit for bit).
"""

from __future__ import annotations

import json
import math
import random
import unittest
from typing import Any

from tests.reference import censored_support as support
from veridist import (
    CapabilityError,
    CoarseningAssumption,
    ExactLifetime,
    ExactValue,
    FamilyId,
    IntervalCensoredLifetime,
    IntervalCensoredValue,
    LeftCensoredLifetime,
    LeftCensoredValue,
    RightCensoredLifetime,
    RightCensoredValue,
    reduce_lifetime_log_likelihood_chunks,
    reduce_value_log_likelihood_chunks,
)
from veridist.engine.errors import CapabilityCode
from veridist.statistics import LogLikelihoodErrorCode, LogLikelihoodFailure, LogLikelihoodSuccess
from veridist.statistics.log_density import LogDensityErrorCode, evaluate_log_density

WEIBULL = {"shape": 1.5, "scale": 6.0}
NORMAL = {"mu": 4.0, "sigma": 2.0}
ASSUMPTION = CoarseningAssumption.INDEPENDENT_CENSORING_AND_TRUNCATION


def lifetimes() -> list[Any]:
    return [
        ExactLifetime(3.0),
        RightCensoredLifetime(7.0),
        LeftCensoredLifetime(2.0),
        IntervalCensoredLifetime(4.0, 5.5),
        ExactLifetime(5.0, truncation_lower=2.0),
        RightCensoredLifetime(6.0, truncation_upper=20.0),
        IntervalCensoredLifetime(1.0, 2.0, truncation_lower=0.5, truncation_upper=9.0),
    ]


def values() -> list[Any]:
    return [
        ExactValue(3.0),
        RightCensoredValue(7.0),
        LeftCensoredValue(2.0),
        IntervalCensoredValue(4.0, 5.5),
        ExactValue(5.0, truncation_lower=2.0),
        RightCensoredValue(6.0, truncation_upper=20.0),
        IntervalCensoredValue(-1.0, 2.0, truncation_lower=-3.0, truncation_upper=9.0),
    ]


def lifetime_total(chunks: Any, **options: Any) -> Any:
    return reduce_lifetime_log_likelihood_chunks(FamilyId.WEIBULL_MIN, chunks, **options, **WEIBULL)


def value_total(chunks: Any, **options: Any) -> Any:
    return reduce_value_log_likelihood_chunks(FamilyId.NORMAL, chunks, **options, **NORMAL)


class SumTests(unittest.TestCase):
    def test_the_total_is_the_exact_sum_of_the_one_row_terms(self) -> None:
        for family, parameters, rows, reducer in (
            (FamilyId.WEIBULL_MIN, WEIBULL, lifetimes(), reduce_lifetime_log_likelihood_chunks),
            (FamilyId.NORMAL, NORMAL, values(), reduce_value_log_likelihood_chunks),
        ):
            with self.subTest(family=family):
                terms = [support.term(family.value, parameters, row) for row in rows]
                result = reducer(family, [rows], **parameters)
                self.assertIsInstance(result, LogLikelihoodSuccess)
                self.assertEqual(result.observation_count, len(rows))
                self.assertEqual(result.total_log_likelihood, math.fsum(terms))

    def test_the_total_is_invariant_to_order_and_chunking(self) -> None:
        rows = lifetimes() * 40
        reference = lifetime_total([rows]).total_log_likelihood
        generator = random.Random(7)
        for _ in range(6):
            shuffled = rows[:]
            generator.shuffle(shuffled)
            cuts = sorted(generator.sample(range(1, len(shuffled)), 5))
            chunks = [
                shuffled[a:b] for a, b in zip([0, *cuts], [*cuts, len(shuffled)], strict=True)
            ]
            self.assertEqual(lifetime_total(chunks).total_log_likelihood, reference)

    def test_the_batch_boundary_does_not_change_the_total(self) -> None:
        rows = (lifetimes() * 200)[:1300]  # more than two batches of 512
        whole = lifetime_total([rows]).total_log_likelihood
        singles = lifetime_total([[row] for row in rows]).total_log_likelihood
        self.assertEqual(whole, singles)

    def test_exact_and_right_censored_rows_keep_their_scalar_terms_bit_for_bit(self) -> None:
        from veridist.statistics.lifetime_log_likelihood import _evaluate_log_survival

        parameters = {"shape": 1.5, "scale": 6.0}
        rows = [
            ExactLifetime(0.3),
            ExactLifetime(11.0),
            RightCensoredLifetime(0.3),
            RightCensoredLifetime(40.0),
        ]
        terms = [
            evaluate_log_density(FamilyId.WEIBULL_MIN, 0.3, **parameters).log_density,
            evaluate_log_density(FamilyId.WEIBULL_MIN, 11.0, **parameters).log_density,
            _evaluate_log_survival(FamilyId.WEIBULL_MIN, parameters, 0.3).log_density,
            _evaluate_log_survival(FamilyId.WEIBULL_MIN, parameters, 40.0).log_density,
        ]
        for row, term in zip(rows, terms, strict=True):
            self.assertEqual(support.term("weibull_min", parameters, row), term)

    def test_an_empty_input_is_the_empty_sum(self) -> None:
        result = lifetime_total([])
        self.assertEqual((result.observation_count, result.total_log_likelihood), (0, 0.0))
        self.assertIsNone(result.coarsening_assumption)


class FrequencyWeightTests(unittest.TestCase):
    def test_a_weight_k_equals_k_replicated_rows_bit_for_bit(self) -> None:
        for reducer, family, parameters, rows in (
            (reduce_lifetime_log_likelihood_chunks, FamilyId.WEIBULL_MIN, WEIBULL, lifetimes()),
            (reduce_value_log_likelihood_chunks, FamilyId.NORMAL, NORMAL, values()),
        ):
            weights = [3, 1, 0, 7, 2, 5, 4]
            replicated = [row for row, w in zip(rows, weights, strict=True) for _ in range(w)]
            weighted = reducer(family, [rows], frequency_weights=[weights], **parameters)
            expected = reducer(family, [replicated], **parameters)
            self.assertEqual(weighted.total_log_likelihood, expected.total_log_likelihood)
            self.assertEqual(weighted.observation_count, expected.observation_count)
            self.assertEqual(weighted.observation_count, sum(weights))
            self.assertEqual(weighted.coarsening_assumption, expected.coarsening_assumption)

    def test_weights_follow_the_chunks(self) -> None:
        rows = lifetimes()
        weights = [2, 1, 3, 1, 2, 2, 1]
        whole = lifetime_total([rows], frequency_weights=[weights])
        split = lifetime_total([rows[:3], rows[3:]], frequency_weights=[weights[:3], weights[3:]])
        self.assertEqual(whole, split)

    def test_a_zero_weight_removes_the_row_even_one_that_would_fail(self) -> None:
        bad = ExactLifetime(0.0)  # outside the Weibull support
        kept = lifetime_total([[ExactLifetime(3.0)]])
        dropped = lifetime_total([[bad, ExactLifetime(3.0)]], frequency_weights=[[0, 1]])
        self.assertEqual(kept, dropped)
        failed = lifetime_total([[bad, ExactLifetime(3.0)]], frequency_weights=[[1, 1]])
        self.assertIsInstance(failed, LogLikelihoodFailure)

    def test_a_row_of_another_group_fails_even_with_weight_zero(self) -> None:
        result = lifetime_total([[ExactLifetime(3.0), ExactValue(3.0)]], frequency_weights=[[1, 0]])
        self.assertIsInstance(result, LogLikelihoodFailure)
        self.assertEqual(result.scalar_error_code, LogDensityErrorCode.NONFINITE_OBSERVATION)
        self.assertEqual(result.processed_count, 1)

    def test_the_observation_limit_counts_replicates(self) -> None:
        result = lifetime_total([[ExactLifetime(3.0)]], frequency_weights=[[(1 << 64)]])
        self.assertIsInstance(result, LogLikelihoodFailure)
        self.assertIs(result.code, LogLikelihoodErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        self.assertEqual(result.processed_count, 0)

    def test_weight_chunks_must_pair_with_observation_chunks(self) -> None:
        rows = [ExactLifetime(3.0), ExactLifetime(4.0)]
        for weights in ([[1]], [[1, 1, 1]], [[1, 1], [1]], []):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                lifetime_total([rows], frequency_weights=weights)
        with self.assertRaises(ValueError):
            lifetime_total([rows, rows], frequency_weights=[[1, 1]])

    def test_analytic_weights_are_refused_with_the_typed_reason(self) -> None:
        for reducer, family, parameters in (
            (reduce_lifetime_log_likelihood_chunks, FamilyId.WEIBULL_MIN, WEIBULL),
            (reduce_value_log_likelihood_chunks, FamilyId.NORMAL, NORMAL),
        ):
            with self.assertRaises(CapabilityError) as captured:
                reducer(family, [[]], analytic_weights=[[1.0]], **parameters)
            self.assertIs(captured.exception.code, CapabilityCode.ANALYTIC_WEIGHTS_UNSUPPORTED)
            self.assertIn("later release", str(captured.exception))


class CoarseningAssumptionTests(unittest.TestCase):
    def test_the_assumption_is_recorded_exactly_for_new_data(self) -> None:
        old = [ExactLifetime(3.0), RightCensoredLifetime(7.0)]
        self.assertIsNone(lifetime_total([old]).coarsening_assumption)
        for new in (
            LeftCensoredLifetime(2.0),
            IntervalCensoredLifetime(1.0, 2.0),
            ExactLifetime(3.0, truncation_lower=1.0),
            RightCensoredLifetime(3.0, truncation_upper=9.0),
        ):
            with self.subTest(new=new):
                self.assertIs(lifetime_total([[*old, new]]).coarsening_assumption, ASSUMPTION)
                self.assertIs(lifetime_total([[new]]).coarsening_assumption, ASSUMPTION)
        self.assertIsNone(
            value_total([[ExactValue(1.0), RightCensoredValue(2.0)]]).coarsening_assumption
        )
        self.assertIs(value_total([[LeftCensoredValue(1.0)]]).coarsening_assumption, ASSUMPTION)

    def test_a_failure_carries_the_assumption_seen_so_far(self) -> None:
        failure = lifetime_total([[LeftCensoredLifetime(2.0), ExactLifetime(0.0)]])
        self.assertIsInstance(failure, LogLikelihoodFailure)
        self.assertIs(failure.coarsening_assumption, ASSUMPTION)
        plain = lifetime_total([[ExactLifetime(0.0)]])
        self.assertIsNone(plain.coarsening_assumption)

    def test_the_assumption_is_a_closed_value_with_a_stable_name(self) -> None:
        self.assertEqual(list(CoarseningAssumption), [ASSUMPTION])
        self.assertEqual(ASSUMPTION.value, "independent_censoring_and_truncation")

    def test_the_json_carries_the_key_only_when_the_result_has_one(self) -> None:
        old = json.loads(lifetime_total([[ExactLifetime(3.0)]]).to_json())
        self.assertEqual(
            sorted(old), ["family", "observation_count", "outcome", "total_log_likelihood"]
        )
        new = json.loads(lifetime_total([[LeftCensoredLifetime(3.0)]]).to_json())
        self.assertEqual(new["coarsening_assumption"], ASSUMPTION.value)
        failure = json.loads(
            lifetime_total([[LeftCensoredLifetime(2.0), ExactLifetime(0.0)]]).to_json()
        )
        self.assertEqual(failure["coarsening_assumption"], ASSUMPTION.value)
        self.assertNotIn(
            "coarsening_assumption", json.loads(lifetime_total([[ExactLifetime(0.0)]]).to_json())
        )

    def test_result_equality_and_construction_still_work_without_the_field(self) -> None:
        result = lifetime_total([[ExactLifetime(3.0)]])
        rebuilt = LogLikelihoodSuccess(
            result.family, result.parameter_fingerprint, 1, result.total_log_likelihood
        )
        self.assertEqual(result, rebuilt)
        self.assertNotEqual(
            result,
            LogLikelihoodSuccess(
                result.family,
                result.parameter_fingerprint,
                1,
                result.total_log_likelihood,
                ASSUMPTION,
            ),
        )
        with self.assertRaises(TypeError):
            LogLikelihoodSuccess(
                result.family,
                result.parameter_fingerprint,
                1,
                0.0,
                "independent",  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            LogLikelihoodFailure(
                FamilyId.NORMAL,
                LogLikelihoodErrorCode.OBSERVATION_LIMIT_EXCEEDED,
                0,
                None,
                "independent",  # type: ignore[arg-type]
            )


class FailureTests(unittest.TestCase):
    def test_a_term_that_is_not_representable_is_a_typed_failure_with_a_count(self) -> None:
        # Both probabilities are below the smallest binary64 logarithm; the left-truncated row has
        # a numerator and a denominator that are both minus infinity, which has no difference.
        overflow = LogDensityErrorCode.NUMERICAL_OVERFLOW
        huge = (
            (RightCensoredLifetime(1e10, truncation_upper=2e10), overflow),
            (
                LeftCensoredLifetime(2e10, truncation_lower=1e10),
                LogDensityErrorCode.NONFINITE_LOG_DENSITY,
            ),
            (IntervalCensoredLifetime(1e10, 1.5e10), overflow),
        )
        for bad, code in huge:
            with self.subTest(bad=bad):
                rows = [ExactLifetime(3.0), bad, ExactLifetime(4.0)]
                result = reduce_lifetime_log_likelihood_chunks(
                    FamilyId.WEIBULL_MIN, [rows], shape=50.0, scale=1.0
                )
                self.assertIsInstance(result, LogLikelihoodFailure)
                self.assertIs(result.code, LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE)
                self.assertIs(result.scalar_error_code, code)
                self.assertEqual(result.processed_count, 1)

    def test_a_right_censored_row_at_zero_is_a_support_violation_as_before(self) -> None:
        for row in (RightCensoredLifetime(0.0), RightCensoredLifetime(0.0, truncation_upper=9.0)):
            result = lifetime_total([[row]])
            self.assertIs(result.scalar_error_code, LogDensityErrorCode.SUPPORT_VIOLATION)

    def test_a_gamma_shape_the_kernels_refuse_is_a_typed_failure_for_window_rows_only(self) -> None:
        parameters = {"shape": 5000.0, "scale": 1.0}
        exact = reduce_lifetime_log_likelihood_chunks(
            FamilyId.GAMMA, [[ExactLifetime(5000.0)]], **parameters
        )
        self.assertIsInstance(exact, LogLikelihoodSuccess)
        for row in (LeftCensoredLifetime(5000.0), ExactLifetime(5000.0, truncation_upper=6000.0)):
            result = reduce_lifetime_log_likelihood_chunks(FamilyId.GAMMA, [[row]], **parameters)
            self.assertIsInstance(result, LogLikelihoodFailure)
            self.assertIs(result.scalar_error_code, LogDensityErrorCode.NONFINITE_LOG_DENSITY)

    def test_a_refused_batch_is_settled_row_by_row(self) -> None:
        parameters = {"shape": 5000.0, "scale": 1.0}
        rows = [ExactLifetime(5000.0), LeftCensoredLifetime(5000.0), ExactLifetime(5001.0)]
        result = reduce_lifetime_log_likelihood_chunks(FamilyId.GAMMA, [rows], **parameters)
        self.assertIsInstance(result, LogLikelihoodFailure)
        self.assertEqual(result.processed_count, 1)

    def test_other_objects_and_the_wrong_group_are_typed_failures(self) -> None:
        for row in (3.0, "x", None, ExactValue(3.0), LeftCensoredValue(3.0)):
            with self.subTest(row=row):
                result = lifetime_total([[ExactLifetime(3.0), row]])
                self.assertIsInstance(result, LogLikelihoodFailure)
                self.assertEqual(
                    result.scalar_error_code, LogDensityErrorCode.NONFINITE_OBSERVATION
                )
                self.assertEqual(result.processed_count, 1)
        result = value_total([[ExactLifetime(3.0)]])
        self.assertIsInstance(result, LogLikelihoodFailure)

    def test_a_chunk_that_is_not_iterable_is_a_type_error(self) -> None:
        with self.assertRaises(TypeError):
            lifetime_total([3.0])
        for weights in ([3], [None]):
            with self.assertRaises(TypeError):
                lifetime_total([[ExactLifetime(3.0)]], frequency_weights=weights)

    def test_the_family_group_is_checked_before_reading(self) -> None:
        with self.assertRaises(ValueError):
            reduce_lifetime_log_likelihood_chunks(FamilyId.NORMAL, [[]], **NORMAL)
        with self.assertRaises(ValueError):
            reduce_value_log_likelihood_chunks(FamilyId.WEIBULL_MIN, [[]], **WEIBULL)


if __name__ == "__main__":
    unittest.main()
