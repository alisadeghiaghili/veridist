"""DS-09 contracts for deterministic and strictly validated checkpoint resume."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from veridist.engine.checkpoint import (
    CheckpointRecord,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.resume import ResumeExpectation, resume_checkpoint
from veridist.engine.retry import PureReducer, apply_pure_update
from veridist.execution import fit_exponential_checkpointed_chunks

SOURCE_ID = "dataset:resume-001"
SOURCE_SCHEMA = "source-v1"
SOURCE_REVISION = "private-etag-91"
PLAN_DIGEST = hashlib.sha256(b"resume-plan-v1").hexdigest()


def sha256(value: bytes | bytearray) -> str:
    return hashlib.sha256(value).hexdigest()


class IntegerSumReducer(PureReducer[int]):
    reducer_id = "integer-sum-v1"
    accumulator_schema = "integer-v1"

    def __init__(self, *, fail_decode: bool = False) -> None:
        self.fail_decode = fail_decode
        self.decode_calls = 0

    def decode_state(self, state: bytes) -> int:
        self.decode_calls += 1
        if self.fail_decode:
            raise RuntimeError("private state must not escape")
        return int(state.decode("ascii"))

    def reduce(self, accumulator: int, payload: bytes) -> int:
        return accumulator + int(payload.decode("ascii"))

    def encode_state(self, accumulator: int) -> bytes:
        return str(accumulator).encode("ascii")


def checkpoint(
    *,
    format_version: int = 1,
    source_id: str = SOURCE_ID,
    source_schema: str = SOURCE_SCHEMA,
    source_revision: str = SOURCE_REVISION,
    reducer_id: str = "integer-sum-v1",
    accumulator_schema: str = "integer-v1",
    plan_digest: str = PLAN_DIGEST,
    cursor: int = 1,
    state: bytes = b"1",
) -> CheckpointRecord:
    return CheckpointRecord.create(
        format_version=format_version,
        source_id=source_id,
        source_schema=source_schema,
        source_revision=source_revision,
        reducer_id=reducer_id,
        accumulator_schema=accumulator_schema,
        plan_digest=plan_digest,
        cursor=cursor,
        committed_ranges=() if cursor == 0 else ((0, cursor),),
        generation=cursor,
        operation_token=None if cursor == 0 else f"chunk-{cursor}",
        operation_digest=None if cursor == 0 else sha256(f"operation-{cursor}".encode()),
        state=state,
    )


def exponential_checkpoint_record() -> CheckpointRecord:
    """A fresh, empty-state checkpoint compatible with the exponential reducer."""

    state = (
        b'{"compensation":"0x0.0p+0","event_count":0,'
        b'"observation_count":0,"total_time":"0x0.0p+0"}'
    )
    return CheckpointRecord.create(
        format_version=1,
        source_id="source",
        source_schema="exponential-v1",
        source_revision=SOURCE_REVISION,
        reducer_id="exponential-reduction-v1",
        accumulator_schema="exponential-reduction-v1",
        plan_digest=PLAN_DIGEST,
        cursor=0,
        committed_ranges=(),
        generation=0,
        operation_token=None,
        operation_digest=None,
        state=state,
    )


def expectation(**overrides: object) -> ResumeExpectation:
    values: dict[str, object] = {
        "format_version": 1,
        "source_id": SOURCE_ID,
        "source_schema": SOURCE_SCHEMA,
        "source_revision": SOURCE_REVISION,
        "reducer_id": "integer-sum-v1",
        "accumulator_schema": "integer-v1",
        "plan_digest": PLAN_DIGEST,
        "cursor": 1,
    }
    values.update(overrides)
    return ResumeExpectation(**values)  # type: ignore[arg-type]


class CheckpointResumeContractTests(unittest.TestCase):
    def test_ds09_checkpointed_chunks_reject_invalid_chunk_shapes(self) -> None:
        with self.assertRaises(TypeError):
            fit_exponential_checkpointed_chunks(
                store=object(), source_revision=SOURCE_REVISION, chunks=1
            )
        with self.assertRaises(TypeError):
            fit_exponential_checkpointed_chunks(
                store=object(), source_revision=SOURCE_REVISION, chunks=("rows",)
            )
        with self.assertRaises(ValueError):
            fit_exponential_checkpointed_chunks(
                store=object(), source_revision=SOURCE_REVISION, chunks=(b"{}",)
            )

    def test_ds09_checkpointed_exponential_chunks_return_a_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteCheckpointStore.create(
                Path(directory) / "fit.sqlite3", exponential_checkpoint_record()
            )
            fit = fit_exponential_checkpointed_chunks(
                store=store,
                source_revision=SOURCE_REVISION,
                chunks=(b"[[1.5,true],[2.25,false]]",),
            )
        self.assertEqual(fit.observation_count, 2)
        self.assertEqual(fit.event_count, 1)
        self.assertEqual(fit.total_time, 3.75)

    def test_ds09_legacy_bytes_chunk_replay_does_not_double_count(self) -> None:
        """DS2b repro: resending a committed legacy chunk must not double-count.

        Before the fix, `row_start` was always read from the live cursor, so
        a replayed chunk always looked like a brand-new operation at a new
        offset, and its rows were counted twice.
        """

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteCheckpointStore.create(
                Path(directory) / "fit.sqlite3", exponential_checkpoint_record()
            )
            first = fit_exponential_checkpointed_chunks(
                store=store,
                source_revision=SOURCE_REVISION,
                chunks=(b"[[1.0,true]]",),
            )
            self.assertEqual(first.observation_count, 1)
            self.assertEqual(first.total_time, 1.0)

            with self.assertWarns(DeprecationWarning):
                second = fit_exponential_checkpointed_chunks(
                    store=store,
                    source_revision=SOURCE_REVISION,
                    chunks=(b"[[1.0,true]]", b"[[2.0,false]]"),
                )
        self.assertEqual(second.observation_count, 2)
        self.assertEqual(second.event_count, 1)
        self.assertEqual(second.total_time, 3.0)

    def test_ds09_offset_form_full_replay_is_a_no_op(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteCheckpointStore.create(
                Path(directory) / "fit.sqlite3", exponential_checkpoint_record()
            )
            chunks = ((0, b"[[1.0,true]]"), (1, b"[[2.0,false]]"))
            fit_exponential_checkpointed_chunks(
                store=store, source_revision=SOURCE_REVISION, chunks=chunks
            )
            before = store.read()

            replayed = fit_exponential_checkpointed_chunks(
                store=store, source_revision=SOURCE_REVISION, chunks=chunks
            )
            after = store.read()

        self.assertEqual(replayed.observation_count, 2)
        self.assertEqual(replayed.event_count, 1)
        self.assertEqual(replayed.total_time, 3.0)
        self.assertEqual(after.generation, before.generation)
        self.assertEqual(after.state, before.state)

    def test_ds09_offset_form_gap_and_partial_overlap_are_range_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteCheckpointStore.create(
                Path(directory) / "gap.sqlite3", exponential_checkpoint_record()
            )
            fit_exponential_checkpointed_chunks(
                store=store,
                source_revision=SOURCE_REVISION,
                chunks=((0, b"[[1.0,true],[2.0,true]]"),),
            )
            self.assertEqual(store.read().cursor, 2)

            with self.assertRaises(EngineContractError) as gap:
                fit_exponential_checkpointed_chunks(
                    store=store,
                    source_revision=SOURCE_REVISION,
                    chunks=((3, b"[[4.0,true]]"),),
                )
            self.assertIs(gap.exception.code, FailureCode.RANGE_MISMATCH)

            with self.assertRaises(EngineContractError) as overlap:
                fit_exponential_checkpointed_chunks(
                    store=store,
                    source_revision=SOURCE_REVISION,
                    chunks=((1, b"[[4.0,true],[5.0,true]]"),),
                )
            self.assertIs(overlap.exception.code, FailureCode.RANGE_MISMATCH)
            self.assertEqual(store.read().cursor, 2)

    def test_ds09_offset_form_rejects_malformed_tuples(self) -> None:
        malformed = (
            (0,),
            (0, 1, 2),
            (1.5, b"[[1.0,true]]"),
            (True, b"[[1.0,true]]"),
            (0, "not-bytes"),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteCheckpointStore.create(
                Path(directory) / "fit.sqlite3", exponential_checkpoint_record()
            )
            for bad in malformed:
                with self.subTest(bad=bad), self.assertRaises(TypeError):
                    fit_exponential_checkpointed_chunks(
                        store=store, source_revision=SOURCE_REVISION, chunks=(bad,)
                    )

    def test_ds09_legacy_chunk_matching_length_but_different_payload_is_new_data(self) -> None:
        """A same-length but different legacy chunk must not be skipped.

        The legacy form can only recognize replay through the checkpoint's
        recorded operation digest, not merely by matching row count, so this
        chunk is applied as new data at the current cursor.
        """

        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteCheckpointStore.create(
                Path(directory) / "fit.sqlite3", exponential_checkpoint_record()
            )
            fit_exponential_checkpointed_chunks(
                store=store,
                source_revision=SOURCE_REVISION,
                chunks=((0, b"[[1.0,true],[2.0,true]]"),),
            )
            self.assertEqual(store.read().cursor, 2)

            with self.assertWarns(DeprecationWarning):
                result = fit_exponential_checkpointed_chunks(
                    store=store,
                    source_revision=SOURCE_REVISION,
                    chunks=(b"[[9.0,true],[9.0,true]]",),
                )
        self.assertEqual(result.observation_count, 4)
        self.assertEqual(result.total_time, 1.0 + 2.0 + 9.0 + 9.0)

    def test_ds09_sqlite_resume_continues_transactional_reduction(self) -> None:
        reducer = IntegerSumReducer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.sqlite3"
            SQLiteCheckpointStore.create(path, checkpoint(cursor=0, state=b"0"))
            store = SQLiteCheckpointStore(path)
            current = apply_pure_update(
                store=store,
                source_revision=SOURCE_REVISION,
                payload=b"1",
                payload_sha256=sha256(b"1"),
                row_start=0,
                row_stop=1,
                operation_token="chunk-1",
                reducer=reducer,
            )
            self.assertEqual(current.generation, 1)
            resumed = resume_checkpoint(
                store=SQLiteCheckpointStore(path),
                expected=expectation(cursor=1),
                reducer=reducer,
            )
            self.assertEqual(resumed.accumulator, 1)
            self.assertEqual(resumed.public_metadata.generation, 1)
            current = apply_pure_update(
                store=SQLiteCheckpointStore(path),
                source_revision=SOURCE_REVISION,
                payload=b"2",
                payload_sha256=sha256(b"2"),
                row_start=1,
                row_stop=2,
                operation_token="chunk-2",
                reducer=reducer,
            )
            self.assertEqual(int(current.state), 3)
            self.assertEqual(SQLiteCheckpointStore(path).read().generation, 2)

    def test_ds09_compatible_resume_matches_canonical_reduction(self) -> None:
        reducer = IntegerSumReducer()
        store = InMemoryCheckpointStore(checkpoint())
        resumed = resume_checkpoint(store=store, expected=expectation(), reducer=reducer)

        canonical_prefix = sum((1,))
        self.assertEqual(resumed.accumulator, canonical_prefix)
        self.assertEqual(resumed.cursor, 1)
        self.assertEqual(resumed.committed_ranges, ((0, 1),))
        self.assertEqual(reducer.decode_calls, 1)

        current = store.read()
        for index, value in enumerate((2, 3), start=1):
            payload = str(value).encode("ascii")
            current = apply_pure_update(
                store=store,
                source_revision=SOURCE_REVISION,
                payload=payload,
                payload_sha256=sha256(payload),
                row_start=index,
                row_stop=index + 1,
                operation_token=f"chunk-{index + 1}",
                reducer=reducer,
            )

        canonical = sum((1, 2, 3))
        self.assertEqual(int(current.state.decode("ascii")), canonical)
        self.assertEqual(current.committed_ranges, ((0, 3),))

    def test_ds09_validation_order_is_fixed_and_precedes_decode_or_write(self) -> None:
        cases = (
            (
                replace(checkpoint(), checksum="corrupt"),
                expectation(format_version=2, source_id="other"),
                FailureCode.CHECKPOINT_CHECKSUM_MISMATCH,
            ),
            (
                checkpoint(format_version=2, source_id="other"),
                expectation(format_version=2, source_id="other"),
                FailureCode.CHECKPOINT_FORMAT_UNSUPPORTED,
            ),
            (
                checkpoint(source_id="other", source_revision="other-revision"),
                expectation(),
                FailureCode.SOURCE_ID_MISMATCH,
            ),
            (
                checkpoint(source_schema="source-v2", source_revision="other-revision"),
                expectation(),
                FailureCode.SOURCE_SCHEMA_MISMATCH,
            ),
            (
                checkpoint(source_revision="other-revision", reducer_id="other"),
                expectation(),
                FailureCode.SOURCE_REVISION_MISMATCH,
            ),
            (
                checkpoint(reducer_id="other", plan_digest=sha256(b"other-plan")),
                expectation(),
                FailureCode.REDUCER_MISMATCH,
            ),
            (
                checkpoint(accumulator_schema="other", plan_digest=sha256(b"other-plan")),
                expectation(),
                FailureCode.ACCUMULATOR_SCHEMA_MISMATCH,
            ),
            (
                checkpoint(plan_digest=sha256(b"other-plan"), cursor=2),
                expectation(),
                FailureCode.PLAN_MISMATCH,
            ),
            (checkpoint(cursor=2), expectation(), FailureCode.RANGE_MISMATCH),
        )

        for record, expected, code in cases:
            reducer = IntegerSumReducer()
            store = InMemoryCheckpointStore(record)
            with self.subTest(code=code), self.assertRaises(EngineContractError) as caught:
                resume_checkpoint(store=store, expected=expected, reducer=reducer)
            self.assertIs(caught.exception.code, code)
            self.assertEqual(store.read_count, 1)
            self.assertEqual(store.write_count, 0)
            self.assertEqual(reducer.decode_calls, 0)

    def test_ds09_decode_is_last_and_failure_has_no_raw_exception_chain(self) -> None:
        reducer = IntegerSumReducer(fail_decode=True)
        store = InMemoryCheckpointStore(checkpoint())

        with self.assertRaises(EngineContractError) as caught:
            resume_checkpoint(store=store, expected=expectation(), reducer=reducer)

        self.assertIs(caught.exception.code, FailureCode.REDUCER_FAILURE)
        self.assertEqual(caught.exception.context["failure_type"], "RuntimeError")
        self.assertIsNone(caught.exception.__cause__)
        self.assertEqual(store.write_count, 0)

    def test_ds09_missing_private_revision_fails_before_checkpoint_read(self) -> None:
        for revision in (None, " "):
            store = InMemoryCheckpointStore(checkpoint())
            reducer = IntegerSumReducer()
            with self.subTest(revision=revision), self.assertRaises(
                EngineContractError
            ) as caught:
                resume_checkpoint(
                    store=store,
                    expected=expectation(source_revision=revision),
                    reducer=reducer,
                )
            self.assertIs(caught.exception.code, FailureCode.SOURCE_REVISION_UNAVAILABLE)
            self.assertEqual(store.read_count, 0)
            self.assertEqual(store.write_count, 0)
            self.assertEqual(reducer.decode_calls, 0)

    def test_ds09_unknown_version_is_rejected_without_automigration(self) -> None:
        record = checkpoint(format_version=7)
        store = InMemoryCheckpointStore(record)
        reducer = IntegerSumReducer()

        with self.assertRaises(EngineContractError) as caught:
            resume_checkpoint(
                store=store,
                expected=expectation(format_version=7),
                reducer=reducer,
            )

        self.assertIs(caught.exception.code, FailureCode.CHECKPOINT_FORMAT_UNSUPPORTED)
        self.assertIs(store.read(), record)
        self.assertEqual(store.write_count, 0)
        self.assertEqual(reducer.decode_calls, 0)

    def test_ds09_public_resume_metadata_never_contains_private_revision(self) -> None:
        resumed = resume_checkpoint(
            store=InMemoryCheckpointStore(checkpoint()),
            expected=expectation(),
            reducer=IntegerSumReducer(),
        )

        public = resumed.public_metadata
        self.assertFalse(hasattr(public, "source_revision"))
        self.assertNotIn(SOURCE_REVISION, repr(public))
        self.assertEqual(public.source_id, SOURCE_ID)
        self.assertEqual(public.source_schema, SOURCE_SCHEMA)

    def test_ds09_sequential_updates_keep_one_canonical_range_and_fixed_shape(self) -> None:
        reducer = IntegerSumReducer()
        store = InMemoryCheckpointStore(checkpoint(cursor=0, state=b"0"))
        initial = store.read()
        initial_orchestration_size = (
            sys.getsizeof(initial)
            + sys.getsizeof(initial.committed_ranges)
            + sys.getsizeof(initial.operation_token)
            + sys.getsizeof(initial.operation_digest)
        )

        for index in range(1, 2_049):
            payload = bytearray(b"1")
            current = apply_pure_update(
                store=store,
                source_revision=SOURCE_REVISION,
                payload=payload,  # type: ignore[arg-type]
                payload_sha256=sha256(payload),
                row_start=index - 1,
                row_stop=index,
                operation_token=f"chunk-{index}",
                reducer=reducer,
            )
            payload[0] = ord("9")
            self.assertEqual(int(current.state), index)
            self.assertLessEqual(len(current.committed_ranges), 1)

        final = store.read()
        final_orchestration_size = (
            sys.getsizeof(final)
            + sys.getsizeof(final.committed_ranges)
            + sys.getsizeof(final.operation_token)
            + sys.getsizeof(final.operation_digest)
        )
        self.assertEqual(final.committed_ranges, ((0, 2_048),))
        self.assertEqual(int(final.state), 2_048)
        self.assertLessEqual(final_orchestration_size - initial_orchestration_size, 256)
        self.assertEqual(len(final.__slots__), len(initial.__slots__))


if __name__ == "__main__":
    unittest.main()
