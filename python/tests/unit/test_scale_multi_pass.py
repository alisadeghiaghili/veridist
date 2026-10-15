"""The multi-pass scale fit of ``weibull_min`` and ``gumbel_right``: results, passes, refusals."""

from __future__ import annotations

import hashlib
import math
import unittest
from collections.abc import Iterator
from typing import Any
from unittest import mock

import numpy as np

from tests.unit.scale_fit_oracle import ListSource
from tests.unit.scale_multi_pass_oracle import gumbel_sample, weibull_sample
from veridist.engine.errors import FailureCode
from veridist.engine.pass_budget import PassBudgetError
from veridist.engine.provenance import (
    ScalePassProvenance,
    scale_provenance_to_canonical_json_bytes,
)
from veridist.families.gumbel import GumbelFitFailure, GumbelFitFailureCode, GumbelFitSuccess
from veridist.families.registry import FamilyId
from veridist.families.weibull import WeibullFitFailure, WeibullFitFailureCode, WeibullFitSuccess
from veridist.scale import _multi_pass
from veridist.scale._accumulator import ExactSum
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode, ScaleStateError
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._multi_pass import ProfileAccumulator
from veridist.scale._pilot import GumbelPilotState, WeibullPilotState
from veridist.scale._planner import ProfileRefusal, ProfileSums
from veridist.scale._sources import DEFAULT_MAX_ROWS, FingerprintLevel, NumpySource, Partition


def roles(result: ScaleFitResult) -> str:
    assert result.scale.passes is not None
    return "".join(item.role[0] for item in result.scale.passes.records)


class AccumulatorTests(unittest.TestCase):
    def test_the_sums_are_the_exact_sums_of_the_one_formula_terms(self) -> None:
        centred = -np.array([0.0, 0.5, 1.25, 3.0, 0.125])
        accumulator = ProfileAccumulator(2)
        accumulator.update(centred, (0.5, 3.0))
        self.assertEqual(accumulator.count, 5)
        for index, parameter in enumerate((0.5, 3.0)):
            weights = np.exp(parameter * centred)
            moment = weights * centred
            expected = ProfileSums(
                self.summed(weights).total(),
                self.summed(moment).total(),
                self.summed(moment * centred).total(),
            )
            self.assertEqual(accumulator.totals()[index], expected)
            self.assertAlmostEqual(
                expected.s0, math.fsum(math.exp(parameter * float(v)) for v in centred), delta=1e-15
            )

    @staticmethod
    def summed(values: np.ndarray) -> ExactSum:
        total = ExactSum()
        total.update(values)
        return total

    def test_the_totals_do_not_depend_on_batching_or_merge_order(self) -> None:
        centred = -np.abs(np.random.default_rng(3).normal(0.0, 4.0, 1000))
        parameters = (0.3, 1.7, 20.0)
        whole = ProfileAccumulator(3)
        whole.update(centred, parameters)
        pieces = []
        for part in (centred[:1], centred[1:400], centred[400:401], centred[401:]):
            piece = ProfileAccumulator(3)
            piece.update(part, parameters)
            pieces.append(piece)
        forward = pieces[0].merge(pieces[1]).merge(pieces[2]).merge(pieces[3])
        backward = pieces[3].merge(pieces[2]).merge(pieces[1]).merge(pieces[0])
        self.assertEqual(forward.count, 1000)
        self.assertEqual(forward.totals(), whole.totals())
        self.assertEqual(backward.totals(), whole.totals())

    def test_an_empty_accumulator_has_zero_sums_and_is_the_identity(self) -> None:
        empty = ProfileAccumulator(1)
        self.assertEqual(empty.count, 0)
        self.assertEqual(empty.totals(), [ProfileSums(0.0, 0.0, 0.0)])
        full = ProfileAccumulator(1)
        full.update(np.array([0.0, -2.0]), (1.5,))
        self.assertEqual(empty.merge(full).totals(), full.totals())
        self.assertEqual(full.merge(empty).count, 2)

    def test_the_weights_cannot_overflow_for_any_candidate(self) -> None:
        centred = np.array([0.0, -1500.0, -3.0])
        accumulator = ProfileAccumulator(2)
        accumulator.update(centred, (math.exp(20.0), math.exp(-20.0)))
        large, small = accumulator.totals()
        self.assertEqual(large.s0, 1.0)
        self.assertAlmostEqual(small.s0, 3.0, places=5)

    def test_an_empty_batch_adds_nothing(self) -> None:
        accumulator = ProfileAccumulator(1)
        accumulator.update(np.empty(0), (1.0,))
        self.assertEqual(accumulator.count, 0)


class ResultTests(unittest.TestCase):
    def test_weibull_returns_the_in_memory_success_type(self) -> None:
        values = weibull_sample(1.7, 1000.0, 100, 11)
        result = fit(NumpySource(values), FamilyId.WEIBULL_MIN)
        self.assertIsInstance(result.fit, WeibullFitSuccess)
        success: Any = result.fit
        self.assertEqual(success.family, FamilyId.WEIBULL_MIN)
        self.assertEqual(
            (success.observation_count, success.event_count, success.censored_count), (100, 100, 0)
        )
        self.assertTrue(success.converged)
        self.assertEqual(success.restart_failures, 0)
        self.assertEqual(success.location, 0.0)
        self.assertEqual(set(success.parameters), {"shape", "scale"})

    def test_gumbel_returns_the_in_memory_success_type(self) -> None:
        values = gumbel_sample(5.0, 1.0, 100, 11)
        result = fit(NumpySource(values), "gumbel_right")
        self.assertIsInstance(result.fit, GumbelFitSuccess)
        success: Any = result.fit
        self.assertEqual(
            (success.observation_count, success.event_count, success.censored_count), (100, 100, 0)
        )
        self.assertEqual(set(success.parameters), {"location", "scale"})

    def test_the_pass_sequences_of_typical_samples_are_pinned(self) -> None:
        cases = (
            (weibull_sample(1.7, 1000.0, 100, 11), "weibull_min", "sgiiii"),
            (weibull_sample(0.3, 1000.0, 100, 12), "weibull_min", "sgii"),
            (weibull_sample(0.3, 1000.0, 100, 11), "weibull_min", "sgiii"),
            (weibull_sample(20.0, 1000.0, 100, 11), "weibull_min", "sgiiii"),
            (gumbel_sample(5.0, 1.0, 100, 11), "gumbel_right", "sgiii"),
            (gumbel_sample(5.0, 1.0, 1000, 11), "gumbel_right", "sgiiii"),
        )
        for values, family, expected in cases:
            result = fit(NumpySource(values), family)
            with self.subTest(family=family, expected=expected):
                self.assertEqual(roles(result), expected)
                assert result.scale.passes is not None
                self.assertEqual(result.scale.passes.pass_count, len(expected))

    def test_a_tight_sample_triggers_an_expansion_and_a_tighter_one_three(self) -> None:
        rng = np.random.default_rng(1)
        one = fit(NumpySource(1000.0 * (1.0 + 1e-5 * rng.standard_normal(50))), "weibull_min")
        self.assertEqual(roles(one), "sgeiii")
        assert isinstance(one.fit, WeibullFitSuccess)
        self.assertGreater(one.fit.shape, math.exp(8.0))
        three = fit(NumpySource(1000.0 * (1.0 + 2e-8 * rng.standard_normal(50))), "weibull_min")
        self.assertEqual(roles(three), "sgeeeii")
        assert isinstance(three.fit, WeibullFitSuccess)
        self.assertGreater(three.fit.shape, math.exp(16.0))
        assert three.scale.passes is not None
        bands = [
            (r.log_low, r.log_high) for r in three.scale.passes.records if r.role == "expansion"
        ]
        self.assertEqual(bands, [(8.5, 12.0), (12.5, 16.0), (16.5, 20.0)])

    def test_the_grid_record_is_the_plan_window(self) -> None:
        result = fit(NumpySource(weibull_sample(1.7, 1000.0, 50, 2)), "weibull_min")
        assert result.scale.passes is not None
        records = result.scale.passes.records
        self.assertEqual((records[0].role, records[0].points), ("statistics", 0))
        self.assertEqual((records[1].role, records[1].points), ("grid", 8))
        self.assertEqual((records[1].log_low, records[1].log_high), (-8.0, 8.0))
        for record in records[2:]:
            self.assertEqual((record.role, record.points, record.log_low), ("iteration", 1, None))

    def test_the_default_budget_and_the_published_bounds_are_in_the_record(self) -> None:
        result = fit(NumpySource(weibull_sample(1.7, 1000.0, 50, 2)), "weibull_min")
        assert result.scale.passes is not None
        self.assertEqual(
            (
                result.scale.passes.pass_budget,
                result.scale.passes.minimum_passes,
                result.scale.passes.maximum_passes,
            ),
            (8, 3, 17),
        )

    def test_extreme_magnitudes_are_fitted_without_overflow(self) -> None:
        wide = fit(NumpySource(np.array([1e-300, 1e300, 1.0])), "weibull_min")
        self.assertIsInstance(wide.fit, WeibullFitSuccess)
        huge = fit(NumpySource(np.array([-1e308, 1e308, 0.0])), "gumbel_right")
        self.assertIsInstance(huge.fit, GumbelFitSuccess)
        subnormal = fit(NumpySource(np.array([5e-324, 1e-323, 2e-323])), "weibull_min")
        self.assertIsInstance(subnormal.fit, WeibullFitSuccess)

    def test_integer_and_float32_sources_are_fitted_like_their_float64_widening(self) -> None:
        ints = np.array([3, 5, 8, 13, 21, 34], dtype=np.int64)
        for family in ("weibull_min", "gumbel_right"):
            self.assertEqual(
                fit(NumpySource(ints), family).fit, fit(NumpySource(ints.astype(float)), family).fit
            )


class FailureTests(unittest.TestCase):
    def test_an_empty_source_is_an_empty_sample(self) -> None:
        for source in (ListSource([]), NumpySource(np.empty(0))):
            weibull_result = fit(source, "weibull_min")
            gumbel_result = fit(source, "gumbel_right")
            self.assertEqual(
                weibull_result.fit, WeibullFitFailure(WeibullFitFailureCode.EMPTY_SAMPLE, 0, 0, 0)
            )
            self.assertEqual(
                gumbel_result.fit, GumbelFitFailure(GumbelFitFailureCode.EMPTY_SAMPLE, 0, 0, 0)
            )
            self.assertEqual(roles(weibull_result), "s")
            self.assertEqual(weibull_result.scale.row_count, 0)

    def test_a_value_that_is_not_positive_is_an_invalid_support_with_the_row_count(self) -> None:
        for bad in (0.0, -2.0):
            source = ListSource([np.array([1.0, 2.0, bad]), np.array([5.0, 6.0])])
            result = fit(source, "weibull_min")
            with self.subTest(bad=bad):
                self.assertEqual(
                    result.fit, WeibullFitFailure(WeibullFitFailureCode.INVALID_SUPPORT, 5, 5, 0)
                )
                self.assertEqual(result.scale.row_count, 5)
                self.assertIsNone(result.scale.state)
                self.assertEqual(roles(result), "s")
                self.assertEqual(source.batch_calls, [0, 1])

    def test_gumbel_accepts_every_sign(self) -> None:
        result = fit(NumpySource(np.array([-3.0, 0.0, 4.0, -0.0])), "gumbel_right")
        self.assertIsInstance(result.fit, GumbelFitSuccess)

    def test_equal_values_and_a_single_row_are_a_degenerate_sample(self) -> None:
        for values in ([4.0], [2.5, 2.5, 2.5]):
            weibull_result = fit(NumpySource(np.array(values)), "weibull_min")
            gumbel_result = fit(NumpySource(np.array(values)), "gumbel_right")
            count = len(values)
            with self.subTest(values=values):
                self.assertEqual(
                    weibull_result.fit,
                    WeibullFitFailure(WeibullFitFailureCode.DEGENERATE_SAMPLE, count, count, 0),
                )
                self.assertEqual(
                    gumbel_result.fit,
                    GumbelFitFailure(GumbelFitFailureCode.DEGENERATE_SAMPLE, count, count, 0),
                )
                self.assertEqual(roles(weibull_result), "s")
                assert weibull_result.scale.state is not None

    def test_a_shape_beyond_the_domain_is_a_boundary_solution_after_the_last_expansion(
        self,
    ) -> None:
        values = np.array([1000.0, math.nextafter(1000.0, 2000.0)])
        result = fit(NumpySource(values), "weibull_min")
        self.assertEqual(
            result.fit, WeibullFitFailure(WeibullFitFailureCode.BOUNDARY_SOLUTION, 2, 2, 0)
        )
        self.assertEqual(roles(result), "sgeee")
        assert result.scale.state is not None

    def test_a_shape_just_inside_the_domain_is_fitted(self) -> None:
        rng = np.random.default_rng(1)
        result = fit(NumpySource(1000.0 * (1.0 + 2e-9 * rng.standard_normal(50))), "weibull_min")
        self.assertIsInstance(result.fit, WeibullFitFailure)  # shape above exp(20) here
        result = fit(NumpySource(1000.0 * (1.0 + 2e-8 * rng.standard_normal(50))), "weibull_min")
        self.assertIsInstance(result.fit, WeibullFitSuccess)

    def test_a_refusal_of_the_planner_is_the_matching_failure_code(self) -> None:
        values = np.array([1.0, 2.0, 4.0])
        for refusal, weibull_code, gumbel_code in (
            (
                ProfileRefusal.BOUNDARY,
                WeibullFitFailureCode.BOUNDARY_SOLUTION,
                GumbelFitFailureCode.BOUNDARY_SOLUTION,
            ),
            (
                ProfileRefusal.EXHAUSTED,
                WeibullFitFailureCode.OPTIMIZER_EXHAUSTED,
                GumbelFitFailureCode.OPTIMIZER_EXHAUSTED,
            ),
        ):
            with mock.patch.object(_multi_pass, "solve_profile", return_value=refusal):
                weibull_result = fit(NumpySource(values), "weibull_min")
                gumbel_result = fit(NumpySource(values), "gumbel_right")
            self.assertEqual(weibull_result.fit, WeibullFitFailure(weibull_code, 3, 3, 0))
            self.assertEqual(gumbel_result.fit, GumbelFitFailure(gumbel_code, 3, 3, 0))
            self.assertEqual(roles(weibull_result), "s")

    def test_a_deviation_beyond_the_float_range_is_an_exhausted_optimizer(self) -> None:
        values = np.array([-1.7e308, -1.7e308, -1.7e308, -1.7e308, 1.7e308])
        result = fit(NumpySource(values), "gumbel_right")
        self.assertEqual(
            result.fit, GumbelFitFailure(GumbelFitFailureCode.OPTIMIZER_EXHAUSTED, 5, 5, 0)
        )
        self.assertEqual(roles(result), "s")

    def test_a_scale_that_cannot_be_represented_is_an_exhausted_optimizer(self) -> None:
        values = weibull_sample(1.7, 1000.0, 20, 4)
        with mock.patch.object(_multi_pass, "exp", side_effect=OverflowError):
            result = fit(NumpySource(values), "weibull_min")
        self.assertEqual(
            result.fit, WeibullFitFailure(WeibullFitFailureCode.OPTIMIZER_EXHAUSTED, 20, 20, 0)
        )

    def test_a_scale_that_underflows_to_zero_is_an_exhausted_optimizer(self) -> None:
        values = weibull_sample(1.7, 1000.0, 20, 4)
        with mock.patch.object(_multi_pass, "exp", return_value=0.0):
            result = fit(NumpySource(values), "weibull_min")
        self.assertEqual(
            result.fit, WeibullFitFailure(WeibullFitFailureCode.OPTIMIZER_EXHAUSTED, 20, 20, 0)
        )

    def test_a_location_that_is_not_finite_is_an_exhausted_optimizer(self) -> None:
        values = gumbel_sample(5.0, 2.0, 30, 4)
        with mock.patch.object(_multi_pass, "fsum", return_value=math.inf):
            result = fit(NumpySource(values), "gumbel_right")
        self.assertEqual(
            result.fit, GumbelFitFailure(GumbelFitFailureCode.OPTIMIZER_EXHAUSTED, 30, 30, 0)
        )

    def test_a_sum_of_values_beyond_the_float_range_is_still_fitted(self) -> None:
        values = np.array([1.5e308, 1.7e308, 1.6e308, 1.65e308, 1.55e308])
        result = fit(NumpySource(values), "gumbel_right")
        self.assertIsInstance(result.fit, GumbelFitSuccess)

    def test_the_value_policy_refusals_are_raised_not_failures(self) -> None:
        for family in ("weibull_min", "gumbel_right"):
            with self.subTest(family=family), self.assertRaises(ScaleSourceError):
                fit(NumpySource(np.array([1.0, np.nan, 3.0])), family)
        source = ListSource([np.array([1.0, 2.0, np.inf])])
        with self.assertRaises(ScaleStateError):
            fit(source, "weibull_min")


class BudgetTests(unittest.TestCase):
    def test_a_budget_below_the_minimum_is_refused_before_any_read(self) -> None:
        for family in ("weibull_min", "gumbel_right"):
            for budget in (1, 2):
                source = ListSource([np.array([1.0, 2.0, 4.0])])
                with (
                    self.subTest(family=family, budget=budget),
                    self.assertRaises(PassBudgetError) as caught,
                ):
                    fit(source, family, pass_budget=budget)
                self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_INSUFFICIENT)
                self.assertEqual(source.partition_calls, 0)
                self.assertEqual(source.batch_calls, [])

    def test_a_run_needing_more_passes_than_the_budget_is_refused_before_the_excess_pass(
        self,
    ) -> None:
        values = weibull_sample(1.7, 1000.0, 100, 11)
        needed = fit(NumpySource(values), "weibull_min").scale.passes
        assert needed is not None
        self.assertEqual(needed.pass_count, 6)
        for budget in (3, 4, 5):
            source = ListSource([values[:50], values[50:]])
            with self.subTest(budget=budget), self.assertRaises(PassBudgetError) as caught:
                fit(source, "weibull_min", pass_budget=budget)
            self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_EXCEEDED)
            self.assertEqual(caught.exception.context["max_passes"], budget)
            self.assertEqual(caught.exception.context["actual_pass_count"], budget)
            self.assertEqual(caught.exception.context["attempted_pass"], budget + 1)
            self.assertEqual(source.batch_calls, [0, 1] * budget)
        # with budget 4 the refusal happens while a bracket is known
        source = ListSource([values])
        with self.assertRaises(PassBudgetError) as caught:
            fit(source, "weibull_min", pass_budget=3)
        width = caught.exception.context["bracket_log_width"]
        assert isinstance(width, float)
        self.assertGreater(width, 0.0)
        self.assertLess(width, 16.0 / 7.0 + 1e-9)

    def test_the_exact_budget_is_enough_and_a_larger_one_changes_nothing(self) -> None:
        values = weibull_sample(1.7, 1000.0, 100, 11)
        base = fit(NumpySource(values), "weibull_min")
        exact = fit(NumpySource(values), "weibull_min", pass_budget=6)
        large = fit(NumpySource(values), "weibull_min", pass_budget=100)
        self.assertEqual(exact.fit, base.fit)
        self.assertEqual(large.fit, base.fit)
        assert exact.scale.passes is not None and large.scale.passes is not None
        self.assertEqual(exact.scale.passes.pass_budget, 6)
        self.assertEqual(large.scale.passes.pass_budget, 100)
        self.assertEqual(exact.scale.passes.records, base.scale.passes.records)  # type: ignore[union-attr]

    def test_the_budget_of_the_boundary_case_is_five_passes(self) -> None:
        values = np.array([1000.0, math.nextafter(1000.0, 2000.0)])
        self.assertIsInstance(
            fit(NumpySource(values), "weibull_min", pass_budget=5).fit, WeibullFitFailure
        )
        with self.assertRaises(PassBudgetError):
            fit(NumpySource(values), "weibull_min", pass_budget=4)

    def test_a_one_pass_family_ignores_the_budget_above_one(self) -> None:
        values = np.array([1.0, 2.0, 4.0, 8.0])
        self.assertEqual(
            fit(NumpySource(values), "normal", pass_budget=1).fit,
            fit(NumpySource(values), "normal").fit,
        )
        self.assertIsNone(fit(NumpySource(values), "normal").scale.passes)


class ChangingSource:
    """A source whose rows change after the first pass."""

    def __init__(self, first: np.ndarray, later: np.ndarray) -> None:
        self.first, self.later = first, later
        self.reads = 0

    def partitions(self) -> tuple[Partition, ...]:
        return (Partition("only", 0, None, None, FingerprintLevel.NONE),)

    def batches(self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS) -> Iterator[Any]:
        self.reads += 1
        yield self.first if self.reads == 1 else self.later


class SourceChangeTests(unittest.TestCase):
    def test_a_later_pass_that_reads_other_rows_is_refused(self) -> None:
        values = weibull_sample(1.7, 1000.0, 30, 6)
        for family in ("weibull_min", "gumbel_right"):
            source = ChangingSource(values, values[:-1])
            with self.subTest(family=family), self.assertRaises(ScaleSourceError) as caught:
                fit(source, family)
            self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)
            self.assertEqual(source.reads, 2)

    def test_a_later_pass_with_a_wrong_dtype_is_a_type_error(self) -> None:
        values = weibull_sample(1.7, 1000.0, 30, 6)
        source = ChangingSource(values, values.astype(np.float32))
        with self.assertRaises(TypeError):
            fit(source, "weibull_min")


class ProvenanceTests(unittest.TestCase):
    def test_the_state_is_the_first_pass_state(self) -> None:
        values = weibull_sample(1.7, 1000.0, 77, 5)
        result = fit(NumpySource(values), "weibull_min")
        state = WeibullPilotState.empty().update(values)
        assert result.scale.state is not None
        self.assertEqual(result.scale.state.state_tag, "scale.weibull_pilot")
        self.assertEqual(result.scale.state.state_schema_version, 1)
        self.assertEqual(
            result.scale.state.digest_sha256, hashlib.sha256(state.to_bytes()).hexdigest()
        )
        gumbel_values = gumbel_sample(5.0, 2.0, 77, 5)
        other = fit(NumpySource(gumbel_values), "gumbel_right")
        gumbel_state = GumbelPilotState.empty().update(gumbel_values)
        assert other.scale.state is not None
        self.assertEqual(
            other.scale.state.digest_sha256, hashlib.sha256(gumbel_state.to_bytes()).hexdigest()
        )

    def test_the_record_describes_the_source_and_the_passes(self) -> None:
        values = weibull_sample(1.7, 1000.0, 123, 5)
        source = NumpySource(values)
        record = fit(source, "weibull_min", max_rows=50).scale
        self.assertEqual(record.source_kind, "numpy")
        self.assertEqual(record.row_count, 123)
        self.assertEqual(record.max_rows, 50)
        self.assertEqual(record.partition_count, 1)
        assert isinstance(record.passes, ScalePassProvenance)
        self.assertEqual(record.passes.pass_count, len(record.passes.records))
        data = scale_provenance_to_canonical_json_bytes(record)
        self.assertIn(b'"passes":{"maximum_passes":17,"minimum_passes":3,"pass_budget":8,', data)
        self.assertIn(b'"role":"grid"', data)

    def test_every_partition_is_read_once_per_pass(self) -> None:
        values = weibull_sample(1.7, 1000.0, 90, 5)
        source = ListSource([values[:30], values[30:31], values[31:]])
        result = fit(source, "weibull_min", max_rows=25)
        assert result.scale.passes is not None
        self.assertEqual(source.partition_calls, 1)
        self.assertEqual(source.batch_calls, [0, 1, 2] * result.scale.passes.pass_count)
        self.assertEqual(
            [item.partition_id for item in result.scale.partitions], ["p0", "p1", "p2"]
        )

    def test_a_custom_source_is_recorded_as_custom(self) -> None:
        result = fit(ListSource([np.array([1.0, 2.0, 5.0])]), "gumbel_right")
        self.assertEqual(result.scale.source_kind, "custom")


class UncertaintyTests(unittest.TestCase):
    def test_a_successful_fit_reports_wald_uncertainty_and_refuses_profile(self) -> None:
        for family, values in (
            ("weibull_min", weibull_sample(1.7, 1000.0, 200, 5)),
            ("gumbel_right", gumbel_sample(5.0, 2.0, 200, 5)),
        ):
            result = fit(NumpySource(values), family)
            success: Any = result.fit
            uncertainty = success.uncertainty()
            with self.subTest(family=family):
                self.assertEqual(set(uncertainty.standard_errors), set(success.parameters))
                for lower, upper in uncertainty.confidence_intervals(0.95, "wald").values():
                    self.assertLess(lower, upper)
                with self.assertRaises(NotImplementedError):
                    uncertainty.confidence_intervals(0.95, "profile")
                with self.assertRaises(NotImplementedError):
                    uncertainty.interval_details(0.9, "profile")
                for derived in (
                    lambda: uncertainty.mean(0.95, "profile"),
                    lambda: uncertainty.quantile(0.1, 0.95, "profile"),
                    lambda: uncertainty.survival(1.0, 0.95, "profile"),
                ):
                    with self.assertRaises(NotImplementedError):
                        derived()
                self.assertIs(success.uncertainty(), uncertainty)

    def test_derived_quantities_have_wald_intervals(self) -> None:
        result = fit(NumpySource(weibull_sample(1.7, 1000.0, 200, 5)), "weibull_min")
        uncertainty: Any = result.fit.uncertainty()  # type: ignore[union-attr]
        mean = uncertainty.mean()
        self.assertTrue(mean.lower < mean.estimate < mean.upper)
        quantile = uncertainty.quantile(0.1)
        self.assertTrue(quantile.lower < quantile.estimate < quantile.upper)
        self.assertEqual(uncertainty.mean(method="wald"), mean)

    def test_an_unsupported_method_is_a_value_error(self) -> None:
        result = fit(NumpySource(weibull_sample(1.7, 1000.0, 50, 5)), "weibull_min")
        uncertainty: Any = result.fit.uncertainty()  # type: ignore[union-attr]
        with self.assertRaises(ValueError):
            uncertainty.confidence_intervals(0.95, "nope")
        with self.assertRaises(ValueError):
            uncertainty.confidence_intervals(0.95, "exact")


if __name__ == "__main__":
    unittest.main()
