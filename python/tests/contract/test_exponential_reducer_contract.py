"""EXP09-EXP14 contracts for the fixed-state exponential reduction seam."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import tracemalloc
import unittest
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from math import isclose, isfinite
from pathlib import Path
from unittest.mock import PropertyMock, patch

import veridist.families.exponential as exponential_module
from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.engine.checkpoint import CheckpointRecord, SQLiteCheckpointStore
from veridist.engine.retry import apply_pure_update
from veridist.families.exponential import (
    ExponentialFitFailure,
    ExponentialFitFailureCode,
    ExponentialFitProvenance,
    ExponentialFitSuccess,
    fit_exponential,
    fit_exponential_chunks,
    fit_exponential_reduction_state,
)
from veridist.statistics.exponential import (
    ExponentialCheckpointReducer,
    ExponentialReductionState,
    _ReductionOverflow,
    reduce_exponential_chunks,
)


class ExponentialReducerContracts(unittest.TestCase):
    """Streaming, reproducibility, state, and provenance constraints."""

    def test_checkpoint_reducer_round_trips_state_and_reduces_canonical_json(self) -> None:
        reducer = ExponentialCheckpointReducer()
        updated = reducer.reduce(ExponentialReductionState.empty(), b'[[1.5,true],[2.25,false]]')
        restored = reducer.decode_state(reducer.encode_state(updated))
        self.assertEqual(restored, updated)
        self.assertEqual((restored.observation_count, restored.event_count), (2, 1))
        self.assertEqual(restored.summed_time, 3.75)

    def test_checkpoint_reducer_rejects_malformed_state_and_payload(self) -> None:
        reducer = ExponentialCheckpointReducer()
        with self.assertRaises((KeyError, ValueError, TypeError)):
            reducer.decode_state(b"[]")
        with self.assertRaises((ValueError, TypeError)):
            reducer.reduce(ExponentialReductionState.empty(), b"{}")

    def test_checkpoint_reducer_commits_through_sqlite_store(self) -> None:
        reducer = ExponentialCheckpointReducer()
        initial_state = reducer.encode_state(ExponentialReductionState.empty())
        initial = CheckpointRecord.create(
            format_version=1,
            source_id="source",
            source_schema="exponential-v1",
            source_revision="revision",
            reducer_id=reducer.reducer_id,
            accumulator_schema=reducer.accumulator_schema,
            plan_digest="plan",
            cursor=0,
            committed_ranges=(),
            generation=0,
            operation_token=None,
            operation_digest=None,
            state=initial_state,
        )
        payload = b"[[1.5,true],[2.25,false]]"
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteCheckpointStore.create(Path(directory) / "state.sqlite3", initial)
            committed = apply_pure_update(
                store=store,
                source_revision="revision",
                payload=payload,
                payload_sha256=hashlib.sha256(payload).hexdigest(),
                row_start=0,
                row_stop=2,
                operation_token="chunk-1",
                reducer=reducer,
            )
            self.assertEqual(reducer.decode_state(committed.state).summed_time, 3.75)

    def test_exp09_ragged_and_empty_chunks_match_canonical_observation_order(self) -> None:
        chunks = (
            (ExactLifetime(Decimal("1")),),
            (),
            (RightCensoredLifetime(Decimal("2")), ExactLifetime(Decimal("3"))),
        )

        streaming = fit_exponential_chunks(chunks)
        canonical = fit_exponential(tuple(item for chunk in chunks for item in chunk))

        self.assertEqual(streaming, canonical)

    def test_exp10_reduction_state_is_frozen_slotted_and_fixed_shape(self) -> None:
        state = ExponentialReductionState.empty().add(ExactLifetime(Decimal("1")))

        self.assertEqual(
            tuple(ExponentialReductionState.__slots__),
            ("observation_count", "event_count", "total_time", "compensation"),
        )
        with self.assertRaises((FrozenInstanceError, TypeError)):
            state.event_count = 99  # type: ignore[misc]
        self.assertFalse(hasattr(state, "raw_rows"))

    def test_exp11_compensated_sum_preserves_small_term_in_adversarial_order(self) -> None:
        state = ExponentialReductionState.empty()
        for time in (Decimal("10000000000000000"), Decimal("1"), Decimal("1")):
            state = state.add(ExactLifetime(time))

        self.assertEqual(state.summed_time, 10_000_000_000_000_002.0)
        self.assertEqual(state.event_count, 3)

    def test_exp11_merge_preserves_compensation_and_declares_only_tolerance_across_partitions(
        self,
    ) -> None:
        left = reduce_exponential_chunks(
            ((ExactLifetime(Decimal("1e16")), ExactLifetime(Decimal("1"))),)
        )
        right = reduce_exponential_chunks(
            ((ExactLifetime(Decimal("1")), ExactLifetime(Decimal("1"))),)
        )
        merged = left.merge(right)
        direct = reduce_exponential_chunks(
            (
                (
                    ExactLifetime(Decimal("1e16")),
                    ExactLifetime(Decimal("1")),
                    ExactLifetime(Decimal("1")),
                    ExactLifetime(Decimal("1")),
                ),
            )
        )

        self.assertEqual((merged.observation_count, merged.event_count), (4, 4))
        self.assertEqual(merged.summed_time, direct.summed_time)
        self.assertTrue(isclose(merged.summed_time, direct.summed_time, rel_tol=1e-15, abs_tol=0.0))

    def test_exp12_partition_order_is_tolerance_claim_not_bit_identical_contract(self) -> None:
        observations = tuple(ExactLifetime(Decimal("0.1")) for _ in range(10_001))
        canonical = fit_exponential(observations)
        partitioned = fit_exponential_chunks(
            (observations[:3333], observations[3333:7777], observations[7777:])
        )

        self.assertIsInstance(canonical, ExponentialFitSuccess)
        self.assertIsInstance(partitioned, ExponentialFitSuccess)
        assert isinstance(canonical, ExponentialFitSuccess)
        assert isinstance(partitioned, ExponentialFitSuccess)
        self.assertTrue(isclose(canonical.rate, partitioned.rate, rel_tol=1e-15, abs_tol=0.0))
        self.assertEqual(canonical.provenance.reduction_order, "canonical_input_order")
        self.assertEqual(canonical.provenance.partition_order_contract, "tolerance_only")

    def test_exp13_provenance_is_closed_and_never_retains_source_locator_or_rows(self) -> None:
        result = fit_exponential((ExactLifetime(Decimal("2")),))

        self.assertIsInstance(result, ExponentialFitSuccess)
        assert isinstance(result, ExponentialFitSuccess)
        provenance = result.provenance
        self.assertEqual(provenance.accumulator_schema_version, "1")
        self.assertEqual(provenance.state_complexity, "O(1)")
        self.assertFalse(provenance.raw_data_retained)
        self.assertFalse(hasattr(provenance, "__dict__"))
        self.assertEqual(
            tuple(type(provenance).__slots__),
            (
                "accumulator_schema_version",
                "state_complexity",
                "reduction_order",
                "partition_order_contract",
                "raw_data_retained",
            ),
        )

    def test_exp14_large_generator_uses_fixed_state_and_never_materializes_input(self) -> None:
        generated = (ExactLifetime(Decimal("2")) for _ in range(20_000))
        state = reduce_exponential_chunks((generated,))

        self.assertEqual(
            (state.observation_count, state.event_count, state.total_time),
            (20_000, 20_000, 40_000.0),
        )
        self.assertEqual(
            tuple(ExponentialReductionState.__slots__),
            ("observation_count", "event_count", "total_time", "compensation"),
        )
        self.assertFalse(hasattr(state, "__dict__"))

    def test_exp14_memory_growth_is_bounded_for_unique_generated_observations(self) -> None:
        def peak_for(size: int) -> int:
            tracemalloc.start()
            try:
                result = fit_exponential(ExactLifetime(Decimal(index + 1)) for index in range(size))
                self.assertIsInstance(result, ExponentialFitSuccess)
                return tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

        small_peak = peak_for(256)
        large_peak = peak_for(30_000)
        self.assertLess(
            large_peak - small_peak,
            1_500_000,
            "the fixed-cardinality reducer must not retain generated observations",
        )

    def test_exp14_aggregate_overflow_is_a_typed_non_estimate(self) -> None:
        result = fit_exponential((ExactLifetime(1e308), ExactLifetime(1e308)))

        self.assertEqual(
            result,
            ExponentialFitFailure(
                code=ExponentialFitFailureCode.NUMERICAL_OVERFLOW,
                observation_count=2,
                event_count=2,
                total_time=None,
            ),
        )
        chunked = fit_exponential_chunks(((ExactLifetime(1e308),), (ExactLifetime(1e308),)))
        self.assertEqual(chunked, result)

    def test_exp14_overflow_scans_to_completion_and_reports_full_derived_counts(self) -> None:
        result = fit_exponential(
            (
                ExactLifetime(1e308),
                ExactLifetime(1e308),
                RightCensoredLifetime(Decimal("2")),
                ExactLifetime(Decimal("3")),
            )
        )
        self.assertEqual(
            result,
            ExponentialFitFailure(
                code=ExponentialFitFailureCode.NUMERICAL_OVERFLOW,
                observation_count=4,
                event_count=3,
                total_time=None,
            ),
        )

    def test_exp14_all_censored_overflow_is_typed_but_invalid_input_after_overflow_is_not_masked(
        self,
    ) -> None:
        overflow = fit_exponential((RightCensoredLifetime(1e308), RightCensoredLifetime(1e308)))
        self.assertEqual(overflow.code, ExponentialFitFailureCode.NUMERICAL_OVERFLOW)
        with self.assertRaises(TypeError):
            fit_exponential((ExactLifetime(1e308), ExactLifetime(1e308), object()))  # type: ignore[arg-type]

    def test_exp14_rejects_forged_impossible_reduction_state(self) -> None:
        for arguments in (
            (0, 0, 1.0, 0.0),
            (1, 0, -1.0, 0.0),
            (1, 1, 1.0, -0.1),
            (1, 1, 1e-15, -2e-15),
        ):
            with self.assertRaises((TypeError, ValueError)):
                ExponentialReductionState(*arguments)

    def test_exp14_reduction_state_rejects_invalid_counts_and_nonfinite_totals(self) -> None:
        for arguments in (
            (True, 0, 0.0, 0.0),
            (1, 2, 1.0, 0.0),
            (1, 1, float("inf"), 0.0),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    ExponentialReductionState(*arguments)

    def test_exp14_final_sum_and_compensation_overflow_are_explicit(self) -> None:
        maximum = float.fromhex("0x1.fffffffffffffp+1023")
        total = float.fromhex("0x1.0000000000000p+1023")
        state = ExponentialReductionState(1, 1, total, maximum)
        with self.assertRaises(_ReductionOverflow):
            _ = state.summed_time
        with self.assertRaises(_ReductionOverflow):
            state._add_value(0.5 * float.fromhex("0x1.0000000000000p+971"), 1, 1)

    def test_exp14_reduction_boundaries_reject_wrong_containers_and_merge_types(self) -> None:
        state = ExponentialReductionState.empty()
        with self.assertRaises(TypeError):
            state.merge(object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            reduce_exponential_chunks(1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            reduce_exponential_chunks((1,))  # type: ignore[arg-type]

    def test_exp14_public_constructors_reject_inconsistent_facts_and_have_no_dict(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            ExponentialFitSuccess(
                rate=0.0,
                observation_count=1,
                event_count=1,
                total_time=1.0,
                mean=1.0,
                log_likelihood=0.0,
                censored_count=0,
            )
        with self.assertRaises((TypeError, ValueError)):
            ExponentialFitSuccess(
                rate=2.0,
                observation_count=1,
                event_count=1,
                total_time=1.0,
                mean=0.5,
                log_likelihood=0.0,
                censored_count=0,
                family="spoofed",
            )
        with self.assertRaises((TypeError, ValueError)):
            ExponentialFitFailure(
                code=ExponentialFitFailureCode.NO_OBSERVED_EVENTS,
                observation_count=1,
                event_count=1,
                total_time=1.0,
            )
        result = fit_exponential((ExactLifetime(Decimal("1")),))
        self.assertFalse(hasattr(result, "__dict__"))

    def test_exp14_success_constructor_rejects_each_inconsistent_derived_fact(self) -> None:
        valid = fit_exponential((ExactLifetime(Decimal("1")),))
        assert isinstance(valid, ExponentialFitSuccess)
        invalid_changes = (
            {"observation_count": True},
            {"censored_count": 1},
            {"total_time": float("nan")},
            {"mean": 2.0},
            {"log_likelihood": 1.0},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    replace(valid, **changes)

    def test_exp14_provenance_and_failure_constructors_reject_spoofed_facts(self) -> None:
        with self.assertRaises(ValueError):
            ExponentialFitProvenance(raw_data_retained=True)
        invalid_failures = (
            ("EMPTY_SAMPLE", 0, 0, 0.0),
            (ExponentialFitFailureCode.EMPTY_SAMPLE, True, 0, 0.0),
            (ExponentialFitFailureCode.NO_OBSERVED_EVENTS, 1, 2, 1.0),
        )
        for arguments in invalid_failures:
            with self.subTest(arguments=arguments):
                with self.assertRaises((TypeError, ValueError)):
                    ExponentialFitFailure(*arguments)  # type: ignore[arg-type]

    def test_exp14_zero_time_failure_facts_require_builtin_finite_floats(self) -> None:
        class FloatSubclass(float):
            pass

        legal = (
            (ExponentialFitFailureCode.EMPTY_SAMPLE, 0, 0),
            (ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD, 1, 1),
        )
        for code, observations, events in legal:
            with self.subTest(code=code):
                failure = ExponentialFitFailure(code, observations, events, 0.0)
                self.assertIsInstance(failure.total_time, float)
                for value in (0, False, Decimal("0"), FloatSubclass(0.0)):
                    with self.assertRaises(TypeError):
                        ExponentialFitFailure(code, observations, events, value)  # type: ignore[arg-type]

    def test_exp14_failure_fact_validator_is_total_over_the_closed_code_set(self) -> None:
        self.assertEqual(
            set(exponential_module._FAILURE_FACT_VALIDATORS),
            set(ExponentialFitFailureCode),
        )

    def test_exp14_reduction_finalizer_type_and_terminal_overflow(self) -> None:
        with self.assertRaises(TypeError):
            fit_exponential_reduction_state(object())  # type: ignore[arg-type]
        state = ExponentialReductionState(1, 1, 1.0, 0.0)
        overflow = _ReductionOverflow(1, 1)
        with patch.object(
            ExponentialReductionState,
            "summed_time",
            new_callable=PropertyMock,
            side_effect=overflow,
        ):
            result = fit_exponential_reduction_state(state)
        self.assertIsInstance(result, ExponentialFitFailure)
        assert isinstance(result, ExponentialFitFailure)
        self.assertIs(result.code, ExponentialFitFailureCode.NUMERICAL_OVERFLOW)

    def test_exp15_tiny_total_time_overflows_the_rate_as_a_typed_non_estimate(self) -> None:
        # events / total_time overflows to inf for a single, vanishingly small
        # exact lifetime; this used to escape as a raw ValueError from the
        # ExponentialFitSuccess constructor instead of a typed failure.
        result = fit_exponential((ExactLifetime(1e-320),))
        expected = ExponentialFitFailure(
            code=ExponentialFitFailureCode.NUMERICAL_OVERFLOW,
            observation_count=1,
            event_count=1,
            total_time=None,
        )
        self.assertEqual(result, expected)
        self.assertEqual(
            fit_exponential_reduction_state(ExponentialReductionState(1, 1, 1e-320, 0.0)),
            expected,
        )

    def test_exp15_finite_rate_with_overflowing_inverse_is_a_typed_non_estimate(self) -> None:
        # A single event accumulated against the largest representable total
        # time yields a finite but subnormal rate whose reciprocal (the
        # reported mean) overflows back out to inf.
        rate = 1.0 / sys.float_info.max
        self.assertTrue(isfinite(rate))
        self.assertFalse(isfinite(1.0 / rate))
        state = ExponentialReductionState(1, 1, sys.float_info.max, 0.0)
        self.assertEqual(
            fit_exponential_reduction_state(state),
            ExponentialFitFailure(
                code=ExponentialFitFailureCode.NUMERICAL_OVERFLOW,
                observation_count=1,
                event_count=1,
                total_time=None,
            ),
        )
