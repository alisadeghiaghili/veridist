"""Checkpointed reductions do not re-read a record their own commit just returned."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId
from veridist.engine.checkpoint import (
    CheckpointRecord,
    CheckpointStore,
    InMemoryCheckpointStore,
    SQLiteCheckpointStore,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.execution import (
    create_checkpointed_csv_store,
    fit_exponential_checkpointed_chunks,
    fit_exponential_checkpointed_csv,
)

_SOURCE_ID = "src_0123456789abcdef0123456789abcdef"
_REVISION = "revision-a"

AfterCommit = Callable[[CheckpointStore, int], None]


def _empty_record(revision: str = _REVISION) -> CheckpointRecord:
    return CheckpointRecord.create(
        format_version=1,
        source_id="source",
        source_schema="exponential-v1",
        source_revision=revision,
        reducer_id="exponential-reduction-v1",
        accumulator_schema="exponential-reduction-v1",
        plan_digest="plan",
        cursor=0,
        committed_ranges=(),
        generation=0,
        operation_token=None,
        operation_digest=None,
        state=(
            b'{"compensation":"0x0.0p+0","event_count":0,'
            b'"observation_count":0,"total_time":"0x0.0p+0"}'
        ),
    )


def _payload(start: int, count: int) -> bytes:
    rows = [[float(start + index + 1), (start + index) % 2 == 0] for index in range(count)]
    return json.dumps(rows, separators=(",", ":")).encode("utf-8")


class RecordingStore:
    """Counts reads and compare-and-swaps, with an optional hook after each commit."""

    def __init__(self, inner: CheckpointStore, after_commit: AfterCommit | None = None) -> None:
        self.inner = inner
        self.reads = 0
        self.swaps = 0
        self._after_commit = after_commit

    def read(self) -> CheckpointRecord:
        self.reads += 1
        return self.inner.read()

    def compare_and_swap(
        self, expected_generation: int, candidate: CheckpointRecord
    ) -> CheckpointRecord:
        self.swaps += 1
        committed = self.inner.compare_and_swap(expected_generation, candidate)
        if self._after_commit is not None:
            self._after_commit(self.inner, self.swaps)
        return committed


def _foreign_advance_after_first_commit(inner: CheckpointStore, swaps: int) -> None:
    """Commit one transition from 'another process' directly on the real store."""

    if swaps != 1:
        return
    base = inner.read()
    inner.compare_and_swap(
        base.generation,
        base.next_generation(
            cursor=base.cursor + 1,
            committed_ranges=((0, base.cursor + 1),),
            operation_token="foreign-writer",
            operation_digest=hashlib.sha256(b"foreign").hexdigest(),
            state=base.state,
        ),
    )


def _revision_change_after_first_commit(inner: CheckpointStore, swaps: int) -> None:
    if swaps != 1:
        return
    base = inner.read()
    inner.compare_and_swap(
        base.generation,
        CheckpointRecord.create(
            format_version=base.format_version,
            source_id=base.source_id,
            source_schema=base.source_schema,
            source_revision="changed-elsewhere",
            reducer_id=base.reducer_id,
            accumulator_schema=base.accumulator_schema,
            plan_digest=base.plan_digest,
            cursor=base.cursor,
            committed_ranges=base.committed_ranges,
            generation=base.generation + 1,
            operation_token="other",
            operation_digest=hashlib.sha256(b"other").hexdigest(),
            state=base.state,
        ),
    )


def _write_csv(path: Path, rows: int) -> str:
    lines = ["time,event_observed"]
    lines.extend(f"{(index % 9) + 1}.5,{index % 2}" for index in range(rows))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ChunkedReadOverheadTests(unittest.TestCase):
    def test_one_store_read_per_commit_plus_first_and_final(self) -> None:
        store = InMemoryCheckpointStore(_empty_record())
        chunk_count = 6
        chunks = tuple((index * 3, _payload(index * 3, 3)) for index in range(chunk_count))
        fit = fit_exponential_checkpointed_chunks(
            store=store, source_revision=_REVISION, chunks=chunks
        )
        self.assertEqual(fit.observation_count, 18)  # type: ignore[union-attr]
        self.assertEqual(store.write_count, chunk_count)
        # Per chunk: only the read inside the CAS boundary; plus the first
        # chunk's cursor read and the final state read.
        self.assertEqual(store.read_count, chunk_count + 2)

    def test_replayed_chunks_are_skipped_without_rereading_the_store(self) -> None:
        store = InMemoryCheckpointStore(_empty_record())
        chunks = tuple((index * 3, _payload(index * 3, 3)) for index in range(4))
        fit_exponential_checkpointed_chunks(store=store, source_revision=_REVISION, chunks=chunks)
        reads_before = store.read_count
        writes_before = store.write_count
        fit = fit_exponential_checkpointed_chunks(
            store=store, source_revision=_REVISION, chunks=chunks
        )
        self.assertEqual(fit.observation_count, 12)  # type: ignore[union-attr]
        self.assertEqual(store.write_count, writes_before)
        self.assertEqual(store.read_count - reads_before, 2)

    def test_a_foreign_commit_between_chunks_is_rejected_not_double_counted(self) -> None:
        inner = InMemoryCheckpointStore(_empty_record())
        store = RecordingStore(inner, _foreign_advance_after_first_commit)
        chunks = ((0, _payload(0, 3)), (3, _payload(3, 3)))
        with self.assertRaises(EngineContractError) as caught:
            fit_exponential_checkpointed_chunks(
                store=store, source_revision=_REVISION, chunks=chunks
            )
        self.assertIs(caught.exception.code, FailureCode.RANGE_MISMATCH)
        committed = inner.read()
        self.assertEqual(committed.generation, 2)
        self.assertEqual(committed.operation_token, "foreign-writer")
        self.assertEqual(committed.cursor, 4)

    def test_a_revision_change_by_another_writer_is_rejected_before_the_next_commit(
        self,
    ) -> None:
        inner = InMemoryCheckpointStore(_empty_record())
        store = RecordingStore(inner, _revision_change_after_first_commit)
        chunks = ((0, _payload(0, 3)), (3, _payload(3, 3)))
        with self.assertRaises(EngineContractError) as caught:
            fit_exponential_checkpointed_chunks(
                store=store, source_revision=_REVISION, chunks=chunks
            )
        self.assertIs(caught.exception.code, FailureCode.SOURCE_REVISION_MISMATCH)
        self.assertEqual(store.swaps, 1)


class CsvReadOverheadTests(unittest.TestCase):
    def _run(
        self, directory: str, rows: int, hook: AfterCommit | None = None
    ) -> tuple[str, RecordingStore, SQLiteCheckpointStore]:
        source = Path(directory) / "lifetimes.csv"
        revision = _write_csv(source, rows)
        sqlite_store = create_checkpointed_csv_store(
            Path(directory) / "store.sqlite3",
            csv_path=source,
            source_id=PublicSourceId(_SOURCE_ID),
        )
        store = RecordingStore(sqlite_store, hook)
        result = fit_exponential_checkpointed_csv(
            path=source,
            schema=CsvLifetimeSchema("time", "event_observed"),
            source_id=PublicSourceId(_SOURCE_ID),
            limits=CsvLifetimeLimits(2048, 4096),
            store=store,
            source_revision=revision,
            cancel=None,
        )
        return result.code, store, sqlite_store

    def test_reads_per_chunk_drop_to_the_one_inside_the_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, store, sqlite_store = self._run(directory, 300)
            self.assertEqual(code, "COMPLETE")
            self.assertGreater(store.swaps, 3)
            self.assertEqual(sqlite_store.read().cursor, 300)
            # preflight + first-chunk refresh + one read per commit + final.
            self.assertEqual(store.reads, store.swaps + 3)

    def test_a_fully_committed_source_costs_three_reads_and_no_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            revision = _write_csv(source, 120)
            sqlite_store = create_checkpointed_csv_store(
                Path(directory) / "store.sqlite3",
                csv_path=source,
                source_id=PublicSourceId(_SOURCE_ID),
            )

            def run(store: CheckpointStore) -> str:
                return fit_exponential_checkpointed_csv(
                    path=source,
                    schema=CsvLifetimeSchema("time", "event_observed"),
                    source_id=PublicSourceId(_SOURCE_ID),
                    limits=CsvLifetimeLimits(2048, 4096),
                    store=store,
                    source_revision=revision,
                    cancel=None,
                ).code

            self.assertEqual(run(sqlite_store), "COMPLETE")
            resumed = RecordingStore(sqlite_store)
            self.assertEqual(run(resumed), "COMPLETE")
            self.assertEqual(resumed.swaps, 0)
            self.assertEqual(resumed.reads, 3)

    def test_a_foreign_commit_between_chunks_is_a_typed_failure_not_a_silent_merge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            code, store, sqlite_store = self._run(
                directory, 300, _foreign_advance_after_first_commit
            )
            self.assertEqual(code, "RANGE_MISMATCH")
            self.assertEqual(store.swaps, 1)
            committed = sqlite_store.read()
            self.assertEqual(committed.operation_token, "foreign-writer")
            self.assertEqual(committed.generation, 2)


class _StatementSpy:
    """A connection wrapper recording the leading keyword(s) of each statement."""

    recorded: list[str]

    def __init__(self, real: sqlite3.Connection, recorded: list[str]) -> None:
        self._real = real
        self._recorded = recorded

    def execute(self, statement: str, *parameters: object) -> sqlite3.Cursor:
        words = statement.split()
        self._recorded.append(" ".join(words[:2]).upper())
        return self._real.execute(statement, *parameters)

    def close(self) -> None:
        self._real.close()


class SqliteDurabilityPragmaTests(unittest.TestCase):
    """``synchronous = FULL`` is requested on write paths only."""

    def _statements(self, action: str) -> list[str]:
        recorded: list[str] = []
        original_connect = SQLiteCheckpointStore._connect

        def spying_connect(self: SQLiteCheckpointStore) -> _StatementSpy:
            return _StatementSpy(original_connect(self), recorded)

        initial = _empty_record()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            if action == "create":
                with patch.object(SQLiteCheckpointStore, "_connect", spying_connect):
                    SQLiteCheckpointStore.create(path, initial)
                return recorded
            store = SQLiteCheckpointStore.create(path, initial)
            with patch.object(SQLiteCheckpointStore, "_connect", spying_connect):
                if action == "read":
                    store.read()
                else:
                    store.compare_and_swap(
                        0,
                        initial.next_generation(
                            cursor=1,
                            committed_ranges=((0, 1),),
                            operation_token="chunk-0-1",
                            operation_digest=hashlib.sha256(b"x").hexdigest(),
                            state=initial.state,
                        ),
                    )
        return recorded

    def test_read_does_not_request_synchronous(self) -> None:
        self.assertEqual(self._statements("read"), ["SELECT FORMAT_VERSION,"])

    def test_compare_and_swap_requests_synchronous_before_the_transaction(self) -> None:
        statements = self._statements("swap")
        self.assertEqual(statements[:2], ["PRAGMA SYNCHRONOUS", "BEGIN IMMEDIATE"])
        self.assertEqual(statements.count("PRAGMA SYNCHRONOUS"), 1)

    def test_create_requests_synchronous_before_the_transaction(self) -> None:
        statements = self._statements("create")
        self.assertEqual(statements[:2], ["PRAGMA SYNCHRONOUS", "BEGIN IMMEDIATE"])
