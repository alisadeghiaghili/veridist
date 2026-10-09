"""Typed engine failures carry exactly the machine-readable facts they document.

``EngineContractError.context`` is part of the public failure surface: a caller
reads ``context["expected_generation"]`` rather than parsing text. These tests
pin the keys and values of each documented failure, and the plan provenance a
planner returns.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from types import SimpleNamespace
from typing import Any

from veridist.engine.checkpoint import (
    CheckpointCommitUncertain,
    CheckpointRecord,
    InMemoryCheckpointStore,
)
from veridist.engine.data_source import (
    CheckpointMetadata,
    DataSourceCapabilityError,
    DataSourceMetadata,
    Replayability,
    SpoolPolicy,
    plan_passes,
)
from veridist.engine.delivery import (
    BoundedChunkBuffer,
    BufferedChunk,
    ChunkEnvelope,
    DeliveryContractError,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.resume import ResumeExpectation, resume_checkpoint
from veridist.engine.retry import (
    IdempotentSink,
    PureReducer,
    SinkResult,
    apply_pure_update,
    apply_sink_update,
)

SHA = "b" * 64


def record(*, generation: int = 0, cursor: int = 0, state: bytes = b"0") -> CheckpointRecord:
    return CheckpointRecord.create(
        format_version=1,
        source_id="source",
        source_schema="schema",
        source_revision="revision",
        reducer_id="reducer",
        accumulator_schema="accumulator",
        plan_digest="plan",
        cursor=cursor,
        committed_ranges=() if cursor == 0 else ((0, cursor),),
        generation=generation,
        operation_token=None if generation == 0 else "token-0",
        operation_digest=None if generation == 0 else "digest-0",
        state=state,
    )


def variant(base: CheckpointRecord, **changes: Any) -> CheckpointRecord:
    """Rebuild ``base`` with some fields changed and a valid checksum."""

    fields: dict[str, Any] = {
        "format_version": base.format_version,
        "source_id": base.source_id,
        "source_schema": base.source_schema,
        "source_revision": base.source_revision,
        "reducer_id": base.reducer_id,
        "accumulator_schema": base.accumulator_schema,
        "plan_digest": base.plan_digest,
        "cursor": base.cursor,
        "committed_ranges": base.committed_ranges,
        "generation": base.generation,
        "operation_token": base.operation_token,
        "operation_digest": base.operation_digest,
        "state": base.state,
    }
    fields.update(changes)
    return CheckpointRecord.create(**fields)


def source(
    replayability: Replayability,
    *,
    source_hash: str | None = SHA,
    redaction_reason: str | None = None,
    checkpoint_schema_version: str | None = None,
) -> Any:
    return SimpleNamespace(
        metadata=DataSourceMetadata(
            source_id="source-1",
            schema_version="schema-1",
            provenance_schema_version="1",
            replayability=replayability,
            source_hash=source_hash,
            redaction_reason=redaction_reason,
            checkpoint_schema_version=checkpoint_schema_version,
        )
    )


class StorePlanningContextTests(unittest.TestCase):
    def test_a_stale_expected_generation_reports_both_generations(self) -> None:
        store = InMemoryCheckpointStore(record())
        with self.assertRaises(EngineContractError) as captured:
            store.compare_and_swap(3, record(generation=4, cursor=1))
        self.assertIs(captured.exception.code, FailureCode.CHECKPOINT_CONFLICT)
        self.assertEqual(
            dict(captured.exception.context), {"actual_generation": 0, "expected_generation": 3}
        )

    def test_a_candidate_that_skips_a_generation_reports_what_it_should_have_been(self) -> None:
        store = InMemoryCheckpointStore(record())
        with self.assertRaises(EngineContractError) as captured:
            store.compare_and_swap(0, record(generation=5, cursor=1))
        self.assertIs(captured.exception.code, FailureCode.CHECKPOINT_CONFLICT)
        self.assertEqual(
            dict(captured.exception.context),
            {"candidate_generation": 5, "expected_generation": 1},
        )

    def test_a_plan_records_the_source_identity_and_the_pass_requirement(self) -> None:
        plan = plan_passes(source(Replayability.REPLAYABLE), required_passes=2)
        self.assertEqual(
            dict(plan.provenance),
            {
                "source_id": "source-1",
                "schema_version": "schema-1",
                "provenance_schema_version": "1",
                "replayability": "replayable",
                "required_passes": 2,
                "source_hash": SHA,
            },
        )
        self.assertEqual(plan.required_passes, 2)
        self.assertIs(plan.spool_enabled, False)

    def test_a_redacted_source_records_the_reason_instead_of_a_hash(self) -> None:
        plan = plan_passes(
            source(Replayability.REPLAYABLE, source_hash=None, redaction_reason="policy"),
            required_passes=1,
        )
        self.assertEqual(
            dict(plan.provenance),
            {
                "source_id": "source-1",
                "schema_version": "schema-1",
                "provenance_schema_version": "1",
                "replayability": "replayable",
                "required_passes": 1,
                "redaction_reason": "policy",
            },
        )

    def test_a_spooled_plan_records_the_spool_requirements(self) -> None:
        spool = SpoolPolicy(
            enabled=True, disk_budget_bytes=4096, retention="delete_on_close", cleanup_required=True
        )
        plan = plan_passes(source(Replayability.SINGLE_PASS), required_passes=2, spool=spool)
        self.assertIs(plan.spool_enabled, True)
        self.assertEqual(
            dict(plan.provenance["spool"]),  # type: ignore[arg-type]
            {"disk_budget_bytes": 4096, "retention": "delete_on_close", "cleanup_required": True},
        )

    def test_a_checkpoint_replayable_source_needs_a_checkpoint_only_for_a_second_pass(self) -> None:
        checkpointed = source(
            Replayability.CHECKPOINT_REPLAYABLE, checkpoint_schema_version="schema-1"
        )
        plan = plan_passes(checkpointed, required_passes=1)
        self.assertEqual(plan.required_passes, 1)
        with self.assertRaises(DataSourceCapabilityError) as captured:
            plan_passes(checkpointed, required_passes=2)
        self.assertIs(captured.exception.code, FailureCode.CHECKPOINT_REQUIRED)
        self.assertEqual(
            dict(captured.exception.context),
            {"required_passes": 2, "replayability": "checkpoint_replayable"},
        )

    def test_a_checkpoint_for_another_source_or_schema_fails_the_preflight(self) -> None:
        checkpointed = source(
            Replayability.CHECKPOINT_REPLAYABLE, checkpoint_schema_version="schema-1"
        )
        for checkpoint, code in (
            (
                CheckpointMetadata("other-source", "schema-1"),
                FailureCode.CHECKPOINT_SOURCE_ID_MISMATCH,
            ),
            (
                CheckpointMetadata("source-1", "other-schema"),
                FailureCode.CHECKPOINT_SCHEMA_MISMATCH,
            ),
        ):
            with self.subTest(code=code), self.assertRaises(DataSourceCapabilityError) as captured:
                plan_passes(checkpointed, required_passes=2, checkpoint=checkpoint)
            self.assertIs(captured.exception.code, code)
            self.assertEqual(dict(captured.exception.context), {"stage": "checkpoint_preflight"})
        accepted = plan_passes(
            checkpointed,
            required_passes=2,
            checkpoint=CheckpointMetadata("source-1", "schema-1"),
        )
        self.assertEqual(accepted.required_passes, 2)

    def test_a_single_pass_source_without_a_spool_reports_why_it_cannot_run_twice(self) -> None:
        with self.assertRaises(DataSourceCapabilityError) as captured:
            plan_passes(source(Replayability.SINGLE_PASS), required_passes=2)
        self.assertIs(captured.exception.code, FailureCode.SPOOL_REQUIRED)
        self.assertEqual(
            dict(captured.exception.context),
            {"required_passes": 2, "replayability": "single_pass", "spool_enabled": False},
        )


class BufferContextTests(unittest.TestCase):
    def test_a_zero_byte_lease_is_rejected_with_its_size(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=8)
        chunk = BufferedChunk(envelope=ChunkEnvelope("s", "c", 0, 0, 0, 0), payload=object())
        with self.assertRaises(DeliveryContractError) as captured:
            buffer.put(chunk, timeout=2.0)
        self.assertIs(captured.exception.code, FailureCode.INVALID_RETAINED_BYTES)
        self.assertEqual(dict(captured.exception.context), {"byte_size": 0})

    def test_an_oversized_lease_is_rejected_with_its_size_and_the_limit(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=8)
        chunk = BufferedChunk(envelope=ChunkEnvelope("s", "c", 0, 0, 1, 5), payload=object())
        with self.assertRaises(DeliveryContractError) as captured:
            buffer.put(chunk, timeout=2.0)
        self.assertIs(captured.exception.code, FailureCode.CHUNK_TOO_LARGE)
        self.assertEqual(dict(captured.exception.context), {"byte_size": 5, "chunk_bytes": 4})

    def test_a_cancelled_buffer_reports_a_cancellation_without_context(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=8)
        buffer.cancel()
        chunk = BufferedChunk(envelope=ChunkEnvelope("s", "c", 0, 0, 1, 1), payload=object())
        with self.assertRaises(DeliveryContractError) as captured:
            buffer.put(chunk, timeout=2.0)
        self.assertIs(captured.exception.code, FailureCode.CANCELLED)
        self.assertEqual(dict(captured.exception.context), {})


class _Reducer(PureReducer[int]):
    reducer_id = "reducer"
    accumulator_schema = "accumulator"

    def decode_state(self, state: bytes) -> int:
        return int(state)

    def reduce(self, accumulator: int, payload: bytes) -> int:
        return accumulator + len(payload)

    def encode_state(self, accumulator: int) -> bytes:
        return str(accumulator).encode()


class _RecordingSink(IdempotentSink):
    def __init__(self, outcome: Any = SinkResult.APPLIED) -> None:
        self.calls: list[tuple[str, str, bytes]] = []
        self.outcome = outcome

    def apply_once(
        self, operation_token: str, operation_digest: str, payload: bytes
    ) -> SinkResult:
        self.calls.append((operation_token, operation_digest, payload))
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome  # type: ignore[no-any-return]


class _UncertainStore:
    """Every commit is uncertain; a later read shows what ``observe`` returns."""

    def __init__(self, base: CheckpointRecord, observe: Any = None) -> None:
        self.base = base
        self.observe = observe
        self.candidates: list[CheckpointRecord] = []

    def read(self) -> CheckpointRecord:
        if not self.candidates or self.observe is None:
            return self.base
        return self.observe(self.base, self.candidates[-1])  # type: ignore[no-any-return]

    def compare_and_swap(
        self, expected_generation: int, candidate: CheckpointRecord
    ) -> CheckpointRecord:
        self.candidates.append(candidate)
        raise CheckpointCommitUncertain("uncertain")


PAYLOAD = b"abc"
PAYLOAD_SHA = hashlib.sha256(PAYLOAD).hexdigest()


def update_arguments(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "source_revision": "revision",
        "payload": PAYLOAD,
        "payload_sha256": PAYLOAD_SHA,
        "row_start": 0,
        "row_stop": 3,
        "operation_token": "token-1",
    }
    arguments.update(overrides)
    return arguments


def expected_operation_digest(base: CheckpointRecord, **overrides: Any) -> str:
    arguments = update_arguments(**overrides)
    value = {
        "operation_token": arguments["operation_token"],
        "payload_sha256": arguments["payload_sha256"],
        "plan_digest": base.plan_digest,
        "reducer_id": base.reducer_id,
        "row_start": arguments["row_start"],
        "row_stop": arguments["row_stop"],
        "source_id": base.source_id,
        "source_revision": base.source_revision,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class RetryContextTests(unittest.TestCase):
    def test_an_update_that_does_not_start_at_the_cursor_reports_the_positions(self) -> None:
        store = InMemoryCheckpointStore(record())
        for update, extra in (
            (apply_pure_update, {"reducer": _Reducer()}),
            (apply_sink_update, {"sink": _RecordingSink()}),
        ):
            with self.subTest(update=update.__name__), self.assertRaises(
                EngineContractError
            ) as captured:
                update(store=store, **extra, **update_arguments(row_start=1, row_stop=3))
            self.assertIs(captured.exception.code, FailureCode.RANGE_MISMATCH)
            self.assertEqual(
                dict(captured.exception.context),
                {"checkpoint_cursor": 0, "row_start": 1, "row_stop": 3},
            )

    def test_a_committed_update_carries_the_documented_operation_digest(self) -> None:
        base = record()
        for update, extra in (
            (apply_pure_update, {"reducer": _Reducer()}),
            (apply_sink_update, {"sink": _RecordingSink()}),
        ):
            store = InMemoryCheckpointStore(base)
            with self.subTest(update=update.__name__):
                committed = update(store=store, **extra, **update_arguments())
                self.assertEqual(committed.operation_token, "token-1")
                self.assertEqual(committed.operation_digest, expected_operation_digest(base))
                self.assertEqual((committed.cursor, committed.generation), (3, 1))

    def test_a_sink_receives_the_token_the_digest_and_the_payload(self) -> None:
        base = record()
        sink = _RecordingSink()
        apply_sink_update(store=InMemoryCheckpointStore(base), sink=sink, **update_arguments())
        self.assertEqual(sink.calls, [("token-1", expected_operation_digest(base), PAYLOAD)])

    def test_a_sink_failure_reports_only_the_exception_type(self) -> None:
        for raised, reported in ((RuntimeError("secret detail"), "RuntimeError"),):
            with self.subTest(raised=raised), self.assertRaises(EngineContractError) as captured:
                apply_sink_update(
                    store=InMemoryCheckpointStore(record()),
                    sink=_RecordingSink(raised),
                    **update_arguments(),
                )
            self.assertIs(captured.exception.code, FailureCode.SINK_FAILURE)
            self.assertEqual(dict(captured.exception.context), {"failure_type": reported})

    def test_an_unrecognised_sink_result_is_a_sink_failure_without_context(self) -> None:
        with self.assertRaises(EngineContractError) as captured:
            apply_sink_update(
                store=InMemoryCheckpointStore(record()),
                sink=_RecordingSink("maybe"),
                **update_arguments(),
            )
        self.assertIs(captured.exception.code, FailureCode.SINK_FAILURE)
        self.assertEqual(dict(captured.exception.context), {})

    def test_exhausted_retries_report_the_attempts_that_were_made(self) -> None:
        for update, extra, expected in (
            (apply_pure_update, {"reducer": _Reducer()}, {}),
            (
                apply_sink_update,
                {"sink": _RecordingSink()},
                {"effect_status": "applied_or_unknown"},
            ),
        ):
            for attempts in (None, 2):
                store = _UncertainStore(record())
                options = {} if attempts is None else {"max_attempts": attempts}
                with self.subTest(update=update.__name__, attempts=attempts), self.assertRaises(
                    EngineContractError
                ) as captured:
                    update(store=store, **extra, **update_arguments(), **options)
                self.assertIs(captured.exception.code, FailureCode.RETRY_EXHAUSTED)
                made = 3 if attempts is None else attempts
                self.assertEqual(
                    dict(captured.exception.context),
                    {"attempts": made, "operation_id_present": True, **expected},
                )
                self.assertEqual(len(store.candidates), made)

    def test_an_uncertain_commit_that_did_land_is_recognised(self) -> None:
        def landed(base: CheckpointRecord, candidate: CheckpointRecord) -> CheckpointRecord:
            return candidate

        for update, extra in (
            (apply_pure_update, {"reducer": _Reducer()}),
            (apply_sink_update, {"sink": _RecordingSink()}),
        ):
            store = _UncertainStore(record(), landed)
            with self.subTest(update=update.__name__):
                result = update(store=store, **extra, **update_arguments())
                self.assertIs(result, store.candidates[0])

    def test_a_different_record_at_the_next_generation_is_a_conflict_not_a_success(self) -> None:
        def other_token(base: CheckpointRecord, candidate: CheckpointRecord) -> CheckpointRecord:
            return variant(candidate, operation_token="someone-else")

        def other_digest(base: CheckpointRecord, candidate: CheckpointRecord) -> CheckpointRecord:
            return variant(candidate, operation_digest="another-digest")

        def later_generation(
            base: CheckpointRecord, candidate: CheckpointRecord
        ) -> CheckpointRecord:
            return variant(candidate, generation=candidate.generation + 4)

        for name, observe in (
            ("token", other_token),
            ("digest", other_digest),
            ("generation", later_generation),
        ):
            for update, extra in (
                (apply_pure_update, {"reducer": _Reducer()}),
                (apply_sink_update, {"sink": _RecordingSink()}),
            ):
                store = _UncertainStore(record(), observe)
                with self.subTest(differs=name, update=update.__name__), self.assertRaises(
                    EngineContractError
                ) as captured:
                    update(store=store, **extra, **update_arguments())
                self.assertIs(captured.exception.code, FailureCode.CHECKPOINT_CONFLICT)


class ResumeMetadataTests(unittest.TestCase):
    def test_a_resumed_checkpoint_publishes_its_public_facts_and_hides_the_revision(self) -> None:
        stored = record(generation=2, cursor=7, state=b"41")
        resumed = resume_checkpoint(
            store=InMemoryCheckpointStore(stored),
            expected=ResumeExpectation(
                format_version=1,
                source_id="source",
                source_schema="schema",
                source_revision="revision",
                reducer_id="reducer",
                accumulator_schema="accumulator",
                plan_digest="plan",
                cursor=7,
            ),
            reducer=_Reducer(),
        )
        self.assertEqual(resumed.accumulator, 41)
        self.assertEqual((resumed.cursor, resumed.generation), (7, 2))
        self.assertEqual(resumed.committed_ranges, ((0, 7),))
        public = resumed.public_metadata
        self.assertEqual(
            (
                public.format_version,
                public.source_id,
                public.source_schema,
                public.reducer_id,
                public.accumulator_schema,
                public.plan_digest,
                public.cursor,
                public.committed_ranges,
                public.generation,
            ),
            (1, "source", "schema", "reducer", "accumulator", "plan", 7, ((0, 7),), 2),
        )
        self.assertNotIn("revision", repr(public))


if __name__ == "__main__":
    unittest.main()
