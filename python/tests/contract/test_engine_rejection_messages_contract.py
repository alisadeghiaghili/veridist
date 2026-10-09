"""Rejected engine inputs name the offending argument and the violated rule.

These messages are what a caller sees when an argument is rejected before any
work starts, so each case checks the exception type together with the whole
message, not merely that something was raised.
"""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from veridist.engine.checkpoint import (
    CheckpointRecord,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from veridist.engine.data_source import DataSourceMetadata, Replayability, plan_passes
from veridist.engine.delivery import (
    AdapterKind,
    BoundedChunkBuffer,
    BufferedChunk,
    ChunkEnvelope,
    DeliveryValidator,
)
from veridist.engine.errors import (
    CapabilityCode,
    CapabilityError,
    EngineContractError,
    FailureCode,
)
from veridist.engine.outcome import (
    FailureRecord,
    FailureStage,
    KnownCoverage,
    KnownExtent,
    RowRange,
    UnknownMissingRanges,
    classify_execution_outcome,
)
from veridist.engine.pass_budget import PassEnforcer
from veridist.engine.provenance import (
    AdapterProvenance,
    EstimatorProvenance,
    RngPolicy,
    RngProvenance,
    SourceHash,
    SourceHashAlgorithm,
    SpoolCleanupStatus,
    SpoolObservation,
    SpoolRetention,
    checkpoint_observation_from_resume,
    failure_record_from_error,
    snapshot_execution_observation,
    to_canonical_json_bytes,
)
from veridist.engine.resume import ResumeExpectation
from veridist.engine.retry import (
    IdempotentSink,
    PureReducer,
    SinkResult,
    apply_pure_update,
    apply_sink_update,
)
from veridist.engine.streaming import IterableDataSource, StreamSourceError, iter_stream

SHA = "a" * 64


class MessageTestCase(unittest.TestCase):
    def check(
        self,
        error: type[BaseException],
        message: str,
        call: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        with self.assertRaises(error) as captured:
            call(*args, **kwargs)
        self.assertIs(type(captured.exception), error)
        self.assertEqual(str(captured.exception), message)


def metadata(replayability: Replayability = Replayability.SINGLE_PASS) -> DataSourceMetadata:
    return DataSourceMetadata(
        source_id="source",
        schema_version="1",
        provenance_schema_version="1",
        replayability=replayability,
        source_hash=SHA,
    )


def record() -> CheckpointRecord:
    return CheckpointRecord.create(
        format_version=1,
        source_id="source",
        source_schema="schema",
        source_revision="revision",
        reducer_id="reducer",
        accumulator_schema="accumulator",
        plan_digest="plan",
        cursor=0,
        committed_ranges=(),
        generation=0,
        operation_token=None,
        operation_digest=None,
        state=b"0",
    )


class PlanningAndDeliveryMessageTests(MessageTestCase):
    def test_a_plan_needs_at_least_one_pass(self) -> None:
        source = SimpleNamespace(metadata=metadata())
        self.check(
            ValueError,
            "required_passes must be at least one",
            plan_passes,
            source,
            required_passes=0,
        )

    def test_a_delivery_validator_rejects_a_bad_source_id_and_start_positions(self) -> None:
        self.check(TypeError, "source_id must be a string", DeliveryValidator, 5)
        self.check(ValueError, "source_id must be non-empty", DeliveryValidator, "  ")
        self.check(
            ValueError,
            "initial_offset must be non-negative",
            DeliveryValidator,
            "s",
            initial_offset=-1,
        )
        self.check(
            ValueError,
            "initial_sequence must be non-negative",
            DeliveryValidator,
            "s",
            initial_sequence=-1,
        )

    def test_finishing_a_delivery_rejects_negative_expectations(self) -> None:
        validator = DeliveryValidator("s")
        self.check(
            ValueError,
            "expected_row_stop must be non-negative",
            validator.finish,
            expected_row_stop=-1,
        )
        self.check(
            ValueError,
            "expected_chunk_count must be non-negative",
            validator.finish,
            expected_row_stop=0,
            expected_chunk_count=-1,
        )

    def test_a_chunk_buffer_rejects_bad_byte_budgets(self) -> None:
        self.check(
            TypeError,
            "buffer byte budgets must be integers",
            BoundedChunkBuffer,
            chunk_bytes="2",
            max_inflight_bytes=4,
        )
        self.check(
            TypeError,
            "buffer byte budgets must be integers",
            BoundedChunkBuffer,
            chunk_bytes=2,
            max_inflight_bytes=4.0,
        )
        self.check(
            ValueError,
            "chunk_bytes must be positive",
            BoundedChunkBuffer,
            chunk_bytes=0,
            max_inflight_bytes=4,
        )
        self.check(
            ValueError,
            "max_inflight_bytes must be at least chunk_bytes",
            BoundedChunkBuffer,
            chunk_bytes=5,
            max_inflight_bytes=4,
        )

    def test_the_smallest_legal_budget_is_one_byte(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=1, max_inflight_bytes=1)
        self.assertEqual((buffer.chunk_bytes, buffer.max_inflight_bytes), (1, 1))

    def test_a_released_chunk_cannot_be_buffered(self) -> None:
        envelope = ChunkEnvelope("s", "c", 0, 0, 1, 1)
        chunk = BufferedChunk(envelope=envelope, payload=object())
        self.assertIs(chunk.released, False)
        chunk.release()
        self.assertIs(chunk.released, True)
        self.check(
            RuntimeError,
            "cannot buffer an already released chunk",
            chunk._compose_release_callback,
            lambda: None,
        )
        buffer = BoundedChunkBuffer(chunk_bytes=1, max_inflight_bytes=4)
        self.check(
            RuntimeError,
            "cannot buffer an already released chunk",
            lambda item: buffer.put(item, timeout=2.0),
            chunk,
        )

    def test_releasing_more_bytes_than_are_in_flight_is_an_accounting_error(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=2, max_inflight_bytes=4)
        self.check(RuntimeError, "buffer inflight byte accounting underflow", buffer._release, 1)

    def test_a_new_buffer_is_not_cancelled_until_it_is_cancelled(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=2, max_inflight_bytes=4)
        self.assertIs(buffer.cancelled, False)
        buffer.cancel()
        self.assertIs(buffer.cancelled, True)


class ErrorTypeMessageTests(MessageTestCase):
    def test_an_engine_error_needs_a_failure_code_and_a_mapping_context(self) -> None:
        self.check(TypeError, "code must be a FailureCode", EngineContractError, "CANCELLED")
        self.check(
            TypeError,
            "failure context must be a mapping",
            EngineContractError,
            FailureCode.CANCELLED,
            [("a", 1)],
        )

    def test_an_engine_error_carries_its_code_as_its_only_argument(self) -> None:
        error = EngineContractError(FailureCode.CANCELLED, {"stage": "x"})
        self.assertEqual(error.args, ("CANCELLED",))
        self.assertIs(error.code, FailureCode.CANCELLED)
        self.assertEqual(dict(error.context), {"stage": "x"})

    def test_context_values_and_keys_are_screened(self) -> None:
        for context, message in (
            ({"ratio": float("nan")}, "failure context floats must be finite"),
            ({1: "x"}, "failure context mapping keys must be strings"),
            ({"file_path": "x"}, "failure context contains a sensitive key"),
            ({"value": object()}, "failure context contains an unsafe value type"),
            ({"nested": {"token": 1}}, "failure context contains a sensitive key"),
        ):
            with self.subTest(context=context):
                self.check(TypeError, message, EngineContractError, FailureCode.CANCELLED, context)

    def test_a_capability_error_needs_a_capability_code(self) -> None:
        self.check(TypeError, "code must be a CapabilityCode", CapabilityError, "x")
        code = next(iter(CapabilityCode))
        error = CapabilityError(code)
        self.assertIs(error.code, code)
        self.assertEqual(error.args, (code.value,))


class OutcomeMessageTests(MessageTestCase):
    def test_a_row_range_needs_non_negative_integers(self) -> None:
        self.check(TypeError, "start must be an integer", RowRange, "0", 1)
        self.check(TypeError, "stop must be an integer", RowRange, 0, 1.0)
        self.check(ValueError, "start must be non-negative", RowRange, -1, 1)
        self.check(ValueError, "stop must be non-negative", RowRange, 0, -1)

    def test_processed_ranges_must_be_an_ordered_non_overlapping_tuple(self) -> None:
        extent = KnownExtent(0, 10)

        def build(ranges: Any) -> KnownCoverage:
            return KnownCoverage(extent, ranges, 2, 0)

        self.check(TypeError, "processed_ranges must be a tuple", build, [RowRange(0, 1)])
        self.check(
            TypeError,
            "processed_ranges must contain RowRange values",
            build,
            (RowRange(0, 1), "x"),
        )
        self.check(
            ValueError,
            "processed_ranges must be ordered",
            build,
            (RowRange(4, 6), RowRange(0, 2)),
        )
        self.check(
            ValueError,
            "processed_ranges must not overlap or repeat",
            build,
            (RowRange(0, 4), RowRange(2, 6)),
        )
        self.check(
            ValueError,
            "processed_ranges must not overlap or repeat",
            build,
            (RowRange(0, 4), RowRange(0, 6)),
        )

    def test_chunk_counts_are_named_integers_and_empty_cannot_exceed_accepted(self) -> None:
        extent = KnownExtent(0, 4)
        self.check(
            TypeError,
            "accepted_chunk_count must be an integer",
            KnownCoverage,
            extent,
            (),
            "1",
            0,
        )
        self.check(
            ValueError,
            "accepted_chunk_count must be non-negative",
            KnownCoverage,
            extent,
            (),
            -1,
            0,
        )
        self.check(
            TypeError,
            "empty_chunk_count must be an integer",
            KnownCoverage,
            extent,
            (),
            1,
            0.0,
        )
        self.check(
            ValueError,
            "empty_chunk_count must be non-negative",
            KnownCoverage,
            extent,
            (),
            1,
            -1,
        )
        self.check(
            ValueError,
            "empty_chunk_count cannot exceed accepted_chunk_count",
            KnownCoverage,
            extent,
            (),
            1,
            2,
        )
        self.check(
            ValueError,
            "empty_chunk_count cannot exceed accepted_chunk_count",
            UnknownMissingRanges,
            (),
            1,
            2,
        )

    def test_classification_rejects_unsupported_inputs_with_a_reason(self) -> None:
        coverage = KnownCoverage(KnownExtent(0, 4), (RowRange(0, 2),), 1, 0)
        self.check(
            TypeError,
            "coverage must be a supported coverage value",
            classify_execution_outcome,
            "coverage",
            None,
        )
        self.check(
            TypeError,
            "failure must be a FailureRecord or None",
            classify_execution_outcome,
            coverage,
            "failure",
        )
        self.check(
            ValueError,
            "incomplete or unknown coverage requires a typed failure",
            classify_execution_outcome,
            coverage,
            None,
        )
        self.check(
            ValueError,
            "incomplete or unknown coverage requires a typed failure",
            classify_execution_outcome,
            UnknownMissingRanges((), 0, 0),
            None,
        )
        failure = FailureRecord(FailureCode.CANCELLED, FailureStage.CANCELLATION)
        self.assertEqual(classify_execution_outcome(coverage, failure).status.value, "partial")


class PassBudgetAndProvenanceMessageTests(MessageTestCase):
    def test_a_pass_enforcer_needs_a_positive_integer_budget(self) -> None:
        for bad in (0, True, 1.0):
            with self.subTest(bad=bad):
                self.check(
                    ValueError,
                    "max_passes must be a positive integer",
                    PassEnforcer,
                    max_passes=bad,
                )

    def test_provenance_tokens_hashes_and_counts_are_named_in_their_errors(self) -> None:
        self.check(
            TypeError, "adapter version must be a string", AdapterProvenance, AdapterKind.CSV, 1
        )
        self.check(
            ValueError,
            "adapter version must be an allowlisted token",
            AdapterProvenance,
            AdapterKind.CSV,
            "not a token",
        )
        self.check(
            TypeError, "source hash must be a string", SourceHash, SourceHashAlgorithm.SHA256, 5
        )
        self.check(
            ValueError,
            "source hash must be a lowercase SHA-256 value",
            SourceHash,
            SourceHashAlgorithm.SHA256,
            "ABC",
        )
        self.check(
            TypeError,
            "estimator settings must be a string",
            EstimatorProvenance,
            "family",
            "estimator",
            "1",
            5,
        )

        def seeded(value: Any) -> RngProvenance:
            return RngProvenance(RngPolicy.EXPLICIT_SEED, "pcg64", value)

        self.check(TypeError, "seed must be an integer", seeded, "1")
        self.check(TypeError, "seed must be an integer", seeded, True)
        self.check(ValueError, "seed must be non-negative", seeded, -1)

        def spooled(budget: Any) -> SpoolObservation:
            return SpoolObservation(
                budget, SpoolRetention.DELETE_ON_CLOSE, SpoolCleanupStatus.COMPLETED
            )

        self.check(TypeError, "disk_budget_bytes must be an integer", spooled, "1")
        self.check(ValueError, "disk_budget_bytes must be non-negative", spooled, -1)
        self.check(ValueError, "disk_budget_bytes must be positive", spooled, 0)

    def test_the_projection_helpers_name_the_argument_of_the_wrong_type(self) -> None:
        self.check(
            TypeError,
            "plan must be an ExecutionPlan",
            snapshot_execution_observation,
            plan="plan",
            pass_enforcer=None,
            buffer=None,
            adapter=None,
            spool=None,
        )
        plan = plan_passes(SimpleNamespace(metadata=metadata()), required_passes=1)
        self.check(
            TypeError,
            "pass_enforcer must be a PassEnforcer",
            snapshot_execution_observation,
            plan=plan,
            pass_enforcer="enforcer",
            buffer=None,
            adapter=None,
            spool=None,
        )
        self.check(
            TypeError,
            "buffer must be a BoundedChunkBuffer",
            snapshot_execution_observation,
            plan=plan,
            pass_enforcer=PassEnforcer(max_passes=1),
            buffer="buffer",
            adapter=None,
            spool=None,
        )
        self.check(
            TypeError,
            "resume must be PublicResumeMetadata",
            checkpoint_observation_from_resume,
            "resume",
            accumulator_schema_version="1",
            final_generation=1,
            retry_count=0,
            commit_count=1,
            store_version="1",
        )
        self.check(
            TypeError,
            "error must be an EngineContractError",
            failure_record_from_error,
            ValueError("x"),
            FailureStage.DELIVERY,
        )
        self.check(
            TypeError,
            "stage must be a FailureStage",
            failure_record_from_error,
            EngineContractError(FailureCode.CANCELLED),
            "delivery",
        )
        self.check(
            TypeError, "report must be an ExecutionReport", to_canonical_json_bytes, object()
        )


class _Reducer(PureReducer[int]):
    reducer_id = "reducer"
    accumulator_schema = "accumulator"

    def decode_state(self, state: bytes) -> int:
        return int(state)

    def reduce(self, accumulator: int, payload: bytes) -> int:
        return accumulator + len(payload)

    def encode_state(self, accumulator: int) -> bytes:
        return str(accumulator).encode()


class _Sink(IdempotentSink):
    def apply_once(
        self, operation_token: str, operation_digest: str, payload: bytes
    ) -> SinkResult:
        return SinkResult.APPLIED


class CheckpointAndRetryMessageTests(MessageTestCase):
    def test_a_checkpoint_record_names_the_empty_field(self) -> None:
        self.check(
            ValueError,
            "source_id must be non-empty",
            CheckpointRecord.create,
            format_version=1,
            source_id=" ",
            source_schema="s",
            source_revision="r",
            reducer_id="x",
            accumulator_schema="a",
            plan_digest="p",
            cursor=0,
            committed_ranges=(),
            generation=0,
            operation_token=None,
            operation_digest=None,
            state=b"",
        )

    def test_resume_expectations_name_the_empty_field(self) -> None:
        self.check(
            ValueError,
            "plan_digest must be non-empty",
            ResumeExpectation,
            format_version=1,
            source_id="s",
            source_schema="s",
            source_revision="r",
            reducer_id="x",
            accumulator_schema="a",
            plan_digest=" ",
            cursor=0,
        )

    def test_a_sqlite_store_rejects_a_non_positive_timeout_and_a_negative_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "store.sqlite3"
            self.check(
                ValueError, "timeout must be positive", SQLiteCheckpointStore, path, timeout=0
            )
            self.check(
                ValueError, "timeout must be positive", SQLiteCheckpointStore, path, timeout=-1.0
            )
            store = SQLiteCheckpointStore(path)
            self.check(
                ValueError,
                "expected_generation must be non-negative",
                store.compare_and_swap,
                -1,
                record(),
            )

    def test_a_retried_update_names_the_invalid_request_argument(self) -> None:
        payload = b"ab"
        base: dict[str, Any] = {
            "store": InMemoryCheckpointStore(record()),
            "source_revision": "revision",
            "payload": payload,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "row_start": 0,
            "row_stop": 2,
            "operation_token": "token",
        }
        for update, extra in (
            (apply_pure_update, {"reducer": _Reducer()}),
            (apply_sink_update, {"sink": _Sink()}),
        ):
            for override, message in (
                ({"row_start": -1}, "row range must be an ordered non-negative interval"),
                ({"row_start": 3}, "row range must be an ordered non-negative interval"),
                ({"operation_token": " "}, "operation_token must be non-empty"),
                ({"max_attempts": 0}, "max_attempts must be positive"),
            ):
                with self.subTest(update=update.__name__, override=override):
                    self.check(ValueError, message, update, **{**base, **extra, **override})


class StreamSourceMessageTests(MessageTestCase):
    def test_an_iterable_source_names_what_it_was_given_wrongly(self) -> None:
        self.check(
            TypeError, "metadata must be DataSourceMetadata", IterableDataSource, [1], "metadata"
        )
        self.check(
            TypeError,
            "single-pass sources require an iterable, not an iterator factory",
            IterableDataSource,
            lambda: iter([1]),
            metadata(),
        )
        self.check(TypeError, "chunks must be an iterable", IterableDataSource, 5, metadata())
        self.check(
            ValueError,
            "replayable sources require an explicit iterator factory",
            IterableDataSource,
            [1, 2],
            metadata(Replayability.REPLAYABLE),
        )

    def test_a_replayable_factory_must_return_an_iterator(self) -> None:
        source = IterableDataSource(lambda: [1, 2], metadata(Replayability.REPLAYABLE))
        self.check(TypeError, "stream factory must return an iterator", source.iter_chunks)

    def test_a_single_pass_source_reports_the_attempted_pass(self) -> None:
        source = IterableDataSource([1, 2], metadata())
        self.assertEqual(list(source.iter_chunks()), [1, 2])
        for _attempt in range(2):
            with self.assertRaises(StreamSourceError) as captured:
                source.iter_chunks()
            self.assertIs(captured.exception.code, FailureCode.PASS_BUDGET_EXCEEDED)
            self.assertEqual(
                dict(captured.exception.context), {"max_passes": 1, "attempted_pass": 2}
            )

    def test_a_replayable_source_hands_out_a_fresh_iterator_each_time(self) -> None:
        source = IterableDataSource(lambda: iter([1, 2]), metadata(Replayability.REPLAYABLE))
        self.assertEqual(list(source.iter_chunks()), [1, 2])
        self.assertEqual(list(source.iter_chunks()), [1, 2])

    def test_iter_stream_names_the_unsupported_source(self) -> None:
        self.check(TypeError, "source must be a StreamSource or iterable", iter_stream, 5)
        self.assertEqual(list(iter_stream([1, 2])), [1, 2])


if __name__ == "__main__":
    unittest.main()
