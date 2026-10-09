"""Storage-level contracts of the SQLite checkpoint store and the chunk buffer.

The on-disk record format, the way a damaged row is rejected, the machine-readable
cause of a storage failure, and the resolution of a commit whose acknowledgement
was lost are all observable by a caller who resumes a run later, possibly with a
different version of the library.
"""

from __future__ import annotations

import hashlib
import inspect
import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from tests.contract.buffer_watchdog import bounded_buffer_call
from veridist.engine.checkpoint import (
    CheckpointCommitUncertain,
    CheckpointRecord,
    SQLiteCheckpointStore,
    _sqlite_error_context,
)
from veridist.engine.delivery import (
    BoundedChunkBuffer,
    BufferedChunk,
    ChunkEnvelope,
    DeliveryContractError,
)
from veridist.engine.errors import EngineContractError, FailureCode


def initial_record() -> CheckpointRecord:
    return CheckpointRecord.create(
        format_version=1,
        source_id="source",
        source_schema="schema",
        source_revision="private-revision",
        reducer_id="reducer",
        accumulator_schema="accumulator",
        plan_digest="plan",
        cursor=0,
        committed_ranges=(),
        generation=0,
        operation_token=None,
        operation_digest=None,
        state=b"\x00binary-state",
    )


def next_record(record: CheckpointRecord, **changes: Any) -> CheckpointRecord:
    fields: dict[str, Any] = {
        "cursor": 1,
        "committed_ranges": ((0, 1),),
        "operation_token": "chunk-1",
        "operation_digest": "digest-1",
        "state": b"\xffnext-state",
    }
    fields.update(changes)
    return record.next_generation(**fields)


def rewrite_row(path: Path, statement: str, parameters: tuple[object, ...] = ()) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(statement, parameters)
        connection.commit()
    finally:
        connection.close()


def stored_payload(path: Path) -> object:
    connection = sqlite3.connect(path)
    try:
        return connection.execute("SELECT payload FROM checkpoint").fetchone()[0]
    finally:
        connection.close()


class _FakeSqliteError(sqlite3.Error):
    def __init__(self, name: object) -> None:
        super().__init__("detail that must never be exposed")
        self.sqlite_errorname = name


class OnDiskFormatTests(unittest.TestCase):
    def test_the_stored_payload_is_the_compact_sorted_json_the_checksum_covers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            record = initial_record()
            SQLiteCheckpointStore.create(path, record)
            payload = stored_payload(path)
        self.assertEqual(
            payload,
            '{"accumulator_schema":"accumulator","committed_ranges":[],"cursor":0,'
            '"format_version":1,"generation":0,"operation_digest":null,'
            '"operation_token":null,"plan_digest":"plan","reducer_id":"reducer",'
            '"source_id":"source","source_revision":"private-revision",'
            '"source_schema":"schema","state":"AGJpbmFyeS1zdGF0ZQ=="}',
        )
        assert isinstance(payload, str)
        self.assertEqual(record.checksum, hashlib.sha256(payload.encode("utf-8")).hexdigest())

    def test_the_documented_defaults_do_not_drift(self) -> None:
        for function in (SQLiteCheckpointStore.__init__, SQLiteCheckpointStore.create):
            with self.subTest(function=function.__qualname__):
                self.assertEqual(inspect.signature(function).parameters["timeout"].default, 5.0)

    def test_creating_a_store_creates_its_missing_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a" / "b" / "checkpoint.sqlite3"
            store = SQLiteCheckpointStore.create(path, initial_record())
            self.assertTrue(path.is_file())
            self.assertEqual(store.read(), initial_record())

    def test_a_committed_swap_returns_the_candidate_without_a_second_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteCheckpointStore.create(Path(directory) / "c.sqlite3", initial_record())
            candidate = next_record(initial_record())
            with patch.object(SQLiteCheckpointStore, "read", side_effect=AssertionError("re-read")):
                self.assertIs(store.compare_and_swap(0, candidate), candidate)


class DamagedRowTests(unittest.TestCase):
    def read_error(self, path: Path) -> EngineContractError:
        with self.assertRaises(EngineContractError) as captured:
            SQLiteCheckpointStore(path).read()
        return captured.exception

    def create_store(self, directory: str) -> Path:
        path = Path(directory) / "checkpoint.sqlite3"
        SQLiteCheckpointStore.create(path, initial_record())
        return path

    def test_a_payload_stored_as_a_blob_is_not_decoded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_store(directory)
            payload = stored_payload(path)
            assert isinstance(payload, str)
            rewrite_row(path, "UPDATE checkpoint SET payload = ?", (payload.encode("utf-8"),))
            self.assertIs(self.read_error(path).code, FailureCode.CHECKPOINT_DECODE_FAILED)

    def test_a_checksum_stored_as_a_blob_is_not_decoded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_store(directory)
            rewrite_row(
                path,
                "UPDATE checkpoint SET checksum = ?",
                (initial_record().checksum.encode("ascii"),),
            )
            self.assertIs(self.read_error(path).code, FailureCode.CHECKPOINT_DECODE_FAILED)

    def test_a_state_that_is_not_strict_base64_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self.create_store(directory)
            payload = stored_payload(path)
            assert isinstance(payload, str)
            damaged = payload.replace('"state":"', '"state":"!')
            self.assertNotEqual(damaged, payload)
            rewrite_row(path, "UPDATE checkpoint SET payload = ?", (damaged,))
            self.assertIs(self.read_error(path).code, FailureCode.CHECKPOINT_DECODE_FAILED)

    def test_a_file_that_is_not_a_database_reports_the_sqlite_error_name_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            path.write_bytes(b"this is not a sqlite database file" * 8)
            error = self.read_error(path)
        self.assertIs(error.code, FailureCode.CHECKPOINT_STORAGE_FAILED)
        self.assertEqual(dict(error.context), {"sqlite_errorname": "SQLITE_NOTADB"})

    def test_an_unopenable_location_reports_the_sqlite_error_name_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteCheckpointStore(Path(directory) / "missing-directory" / "c.sqlite3")
            with self.assertRaises(EngineContractError) as captured:
                store.compare_and_swap(0, next_record(initial_record()))
        self.assertIs(captured.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)
        self.assertEqual(dict(captured.exception.context), {"sqlite_errorname": "SQLITE_CANTOPEN"})

    def test_a_failed_create_reports_the_sqlite_error_name_and_leaves_no_file(self) -> None:
        def broken_schema(connection: sqlite3.Connection) -> None:
            connection.execute("SELECT * FROM a_table_that_does_not_exist")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            with patch.object(
                SQLiteCheckpointStore, "_create_schema", staticmethod(broken_schema)
            ), self.assertRaises(EngineContractError) as captured:
                SQLiteCheckpointStore.create(path, initial_record())
            self.assertFalse(path.exists())
        self.assertIs(captured.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)
        self.assertEqual(dict(captured.exception.context), {"sqlite_errorname": "SQLITE_ERROR"})

    def test_only_a_textual_non_empty_error_name_is_exposed(self) -> None:
        self.assertEqual(
            _sqlite_error_context(_FakeSqliteError("SQLITE_BUSY")),
            {"sqlite_errorname": "SQLITE_BUSY"},
        )
        for name in ("", None, 7):
            with self.subTest(name=name):
                self.assertEqual(_sqlite_error_context(_FakeSqliteError(name)), {})
        self.assertEqual(_sqlite_error_context(sqlite3.Error("plain")), {})


class LockTimeoutTests(unittest.TestCase):
    def locked_swap_seconds(self, store: SQLiteCheckpointStore, path: Path) -> float:
        blocker = sqlite3.connect(path, isolation_level=None)
        try:
            blocker.execute("BEGIN IMMEDIATE")
            started = time.perf_counter()
            with self.assertRaises(EngineContractError) as captured:
                store.compare_and_swap(0, next_record(initial_record()))
            elapsed = time.perf_counter() - started
        finally:
            blocker.close()
        self.assertIs(captured.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)
        self.assertEqual(dict(captured.exception.context), {"sqlite_errorname": "SQLITE_BUSY"})
        return elapsed

    def test_a_locked_database_is_given_up_after_the_configured_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            SQLiteCheckpointStore.create(path, initial_record())
            elapsed = self.locked_swap_seconds(SQLiteCheckpointStore(path, timeout=0.05), path)
        self.assertLess(elapsed, 2.5)

    def test_the_timeout_given_to_create_is_kept_by_the_returned_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            store = SQLiteCheckpointStore.create(path, initial_record(), timeout=0.05)
            elapsed = self.locked_swap_seconds(store, path)
        self.assertLess(elapsed, 2.5)


class LostAcknowledgementTests(unittest.TestCase):
    def store_with_lost_acknowledgement(self, directory: str) -> SQLiteCheckpointStore:
        return SQLiteCheckpointStore.create(Path(directory) / "c.sqlite3", initial_record())

    def test_a_commit_that_certainly_did_not_land_is_reported_as_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_with_lost_acknowledgement(directory)
            with patch.object(
                SQLiteCheckpointStore, "_commit", side_effect=sqlite3.OperationalError("lost")
            ), self.assertRaises(CheckpointCommitUncertain) as captured:
                store.compare_and_swap(0, next_record(initial_record()))
        self.assertEqual(str(captured.exception), "checkpoint commit acknowledgement is uncertain")

    def test_a_store_that_cannot_be_read_back_is_reported_as_uncertain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_with_lost_acknowledgement(directory)
            unreadable = EngineContractError(FailureCode.CHECKPOINT_STORAGE_FAILED)
            with patch.object(
                SQLiteCheckpointStore, "_commit", side_effect=sqlite3.OperationalError("lost")
            ), patch.object(
                SQLiteCheckpointStore, "read", side_effect=unreadable
            ), self.assertRaises(CheckpointCommitUncertain) as captured:
                store.compare_and_swap(0, next_record(initial_record()))
        self.assertEqual(str(captured.exception), "checkpoint commit acknowledgement is uncertain")

    def test_a_commit_that_did_land_is_recognised_from_the_stored_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_with_lost_acknowledgement(directory)
            candidate = next_record(initial_record())
            real_commit = SQLiteCheckpointStore._commit

            def commit_then_lose_the_acknowledgement(connection: sqlite3.Connection) -> None:
                real_commit(connection)
                raise sqlite3.OperationalError("acknowledgement lost")

            with patch.object(
                SQLiteCheckpointStore,
                "_commit",
                staticmethod(commit_then_lose_the_acknowledgement),
            ):
                self.assertEqual(store.compare_and_swap(0, candidate), candidate)

    def test_a_different_record_at_the_candidate_generation_is_a_conflict(self) -> None:
        candidate = next_record(initial_record())
        look_alikes = {
            "state": next_record(initial_record(), state=b"another state"),
            "digest": next_record(initial_record(), operation_digest="another-digest"),
            "token": next_record(initial_record(), operation_token="another-token"),
        }
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_with_lost_acknowledgement(directory)
            for differs, observed in look_alikes.items():
                self.assertEqual(observed.generation, candidate.generation)
                with self.subTest(differs=differs), patch.object(
                    SQLiteCheckpointStore, "read", return_value=observed
                ), self.assertRaises(EngineContractError) as captured:
                    store._reconcile_uncertain_commit(0, candidate)
                self.assertIs(captured.exception.code, FailureCode.CHECKPOINT_CONFLICT)

    def test_an_identical_record_is_the_committed_result(self) -> None:
        candidate = next_record(initial_record())
        with tempfile.TemporaryDirectory() as directory:
            store = self.store_with_lost_acknowledgement(directory)
            with patch.object(SQLiteCheckpointStore, "read", return_value=candidate):
                self.assertEqual(store._reconcile_uncertain_commit(0, candidate), candidate)


def chunk(byte_size: int = 1) -> BufferedChunk:
    return BufferedChunk(envelope=ChunkEnvelope("s", "c", 0, 0, 1, byte_size), payload=object())


class BufferBlockingTests(unittest.TestCase):
    def test_a_timed_out_put_leaves_no_waiting_producer_behind(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=2, max_inflight_bytes=2)
        buffer.put(chunk(2), timeout=2.0)
        for _attempt in range(2):
            with self.assertRaises(DeliveryContractError) as captured:
                bounded_buffer_call(buffer, lambda: buffer.put(chunk(2), timeout=0.02))
            self.assertIs(captured.exception.code, FailureCode.BUFFER_TIMEOUT)
            self.assertEqual(dict(captured.exception.context), {"operation": "put"})
            self.assertEqual(buffer.waiting_producers, 0)
        self.assertEqual(buffer.observation.backpressure_event_count, 2)

    def test_read_and_put_passes_its_timeout_on_to_the_put(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=2, max_inflight_bytes=2)
        buffer.put(chunk(2), timeout=2.0)
        with self.assertRaises(DeliveryContractError) as captured:
            bounded_buffer_call(
                buffer, lambda: buffer.read_and_put(lambda: chunk(2), timeout=0.02), timeout=2.0
            )
        self.assertIs(captured.exception.code, FailureCode.BUFFER_TIMEOUT)

    def deliver_after(self, buffer: BoundedChunkBuffer, delay: float) -> threading.Thread:
        producer = threading.Thread(
            target=lambda: (time.sleep(delay), buffer.put(chunk(1), timeout=5.0)), daemon=True
        )
        producer.start()
        return producer

    def test_a_get_without_a_timeout_waits_for_the_next_chunk(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=2, max_inflight_bytes=2)
        producer = self.deliver_after(buffer, 0.1)
        received = bounded_buffer_call(buffer, buffer.get, timeout=5.0)
        producer.join(5.0)
        self.assertEqual(received.envelope.chunk_id, "c")

    def test_a_get_waits_for_the_whole_timeout_not_less(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=2, max_inflight_bytes=2)
        producer = self.deliver_after(buffer, 0.1)
        received = bounded_buffer_call(buffer, lambda: buffer.get(timeout=3.0), timeout=5.0)
        producer.join(5.0)
        self.assertEqual(received.envelope.chunk_id, "c")

    def test_a_get_shorter_than_a_second_still_waits_for_a_chunk_that_arrives_in_time(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=2, max_inflight_bytes=2)
        producer = self.deliver_after(buffer, 0.1)
        received = bounded_buffer_call(buffer, lambda: buffer.get(timeout=0.9), timeout=5.0)
        producer.join(5.0)
        self.assertEqual(received.envelope.chunk_id, "c")

    def test_a_get_on_an_empty_buffer_times_out(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=2, max_inflight_bytes=2)
        with self.assertRaises(DeliveryContractError) as captured:
            bounded_buffer_call(buffer, lambda: buffer.get(timeout=0.02))
        self.assertIs(captured.exception.code, FailureCode.BUFFER_TIMEOUT)
        self.assertEqual(dict(captured.exception.context), {"operation": "get"})


if __name__ == "__main__":
    unittest.main()
