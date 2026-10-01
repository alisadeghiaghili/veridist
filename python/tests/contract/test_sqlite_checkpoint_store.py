"""CKPT-SQL contracts for the local transactional checkpoint backend."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from veridist.engine.checkpoint import (
    CheckpointCommitUncertain,
    CheckpointRecord,
    SQLiteCheckpointStore,
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


def next_record(record: CheckpointRecord) -> CheckpointRecord:
    return record.next_generation(
        cursor=1,
        committed_ranges=((0, 1),),
        operation_token="chunk-1",
        operation_digest="digest-1",
        state=b"\xffnext-state",
    )


def execute_database_update(path: Path, statement: str, parameters: tuple[object, ...]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(statement, parameters)
        connection.commit()
    finally:
        connection.close()


class SQLiteCheckpointStoreContractTests(unittest.TestCase):
    def test_ckpt_sql01_create_reopen_and_binary_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            initial = initial_record()
            store = SQLiteCheckpointStore.create(path, initial)
            self.assertEqual(store.read(), initial)
            candidate = next_record(initial)
            self.assertEqual(store.compare_and_swap(0, candidate), candidate)
            self.assertEqual(SQLiteCheckpointStore(path).read(), candidate)

    def test_ckpt_sql02_create_and_stale_writer_conflicts_are_typed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            initial = initial_record()
            first = SQLiteCheckpointStore.create(path, initial)
            with self.assertRaises(EngineContractError) as existing:
                SQLiteCheckpointStore.create(path, initial)
            self.assertIs(existing.exception.code, FailureCode.CHECKPOINT_ALREADY_EXISTS)

            candidate = next_record(initial)
            self.assertEqual(first.compare_and_swap(0, candidate), candidate)
            with self.assertRaises(EngineContractError) as stale:
                SQLiteCheckpointStore(path).compare_and_swap(0, candidate)
            self.assertIs(stale.exception.code, FailureCode.CHECKPOINT_CONFLICT)

    def test_ckpt_sql03_corruption_and_invalid_candidate_fail_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            initial = initial_record()
            store = SQLiteCheckpointStore.create(path, initial)
            invalid = replace(next_record(initial), checksum="not-a-checksum")
            with self.assertRaises(EngineContractError) as candidate_error:
                store.compare_and_swap(0, invalid)
            self.assertIs(candidate_error.exception.code, FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)
            self.assertEqual(store.read(), initial)

            execute_database_update(
                path,
                "UPDATE checkpoint SET payload = ? WHERE singleton = 1",
                ("{}",),
            )
            with self.assertRaises(EngineContractError) as malformed:
                SQLiteCheckpointStore(path).read()
            self.assertIs(malformed.exception.code, FailureCode.CHECKPOINT_DECODE_FAILED)

    def test_ckpt_sql04_missing_store_and_unsupported_format_have_distinct_codes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            with self.assertRaises(EngineContractError) as missing:
                SQLiteCheckpointStore(path).read()
            self.assertIs(missing.exception.code, FailureCode.CHECKPOINT_NOT_FOUND)

            initial = initial_record()
            store = SQLiteCheckpointStore.create(path, initial)
            execute_database_update(
                path,
                "UPDATE checkpoint SET format_version = 99 WHERE singleton = 1",
                (),
            )
            with self.assertRaises(EngineContractError) as unsupported:
                store.read()
            self.assertIs(unsupported.exception.code, FailureCode.CHECKPOINT_FORMAT_UNSUPPORTED)

    def test_ckpt_sql04_rejects_invalid_storage_shape_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            SQLiteCheckpointStore.create(path, initial_record())
            execute_database_update(
                path,
                "UPDATE checkpoint SET generation = ? WHERE singleton = 1",
                ("invalid",),
            )
            with self.assertRaises(EngineContractError) as malformed:
                SQLiteCheckpointStore(path).read()
            self.assertIs(malformed.exception.code, FailureCode.CHECKPOINT_DECODE_FAILED)

            path.unlink()
            SQLiteCheckpointStore.create(path, initial_record())
            execute_database_update(
                path,
                "UPDATE checkpoint SET checksum = ? WHERE singleton = 1",
                ("invalid",),
            )
            with self.assertRaises(EngineContractError) as corrupted:
                SQLiteCheckpointStore(path).read()
            self.assertIs(corrupted.exception.code, FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)

    def test_ckpt_sql04_rejects_non_object_and_inconsistent_payload_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            SQLiteCheckpointStore.create(path, initial_record())
            execute_database_update(
                path,
                "UPDATE checkpoint SET payload = ? WHERE singleton = 1",
                ("[]",),
            )
            with self.assertRaises(EngineContractError) as non_object:
                SQLiteCheckpointStore(path).read()
            self.assertIs(non_object.exception.code, FailureCode.CHECKPOINT_DECODE_FAILED)

            path.unlink()
            SQLiteCheckpointStore.create(path, initial_record())
            execute_database_update(
                path,
                "UPDATE checkpoint SET generation = ? WHERE singleton = 1",
                (2,),
            )
            with self.assertRaises(EngineContractError) as inconsistent:
                SQLiteCheckpointStore(path).read()
            self.assertIs(inconsistent.exception.code, FailureCode.CHECKPOINT_DECODE_FAILED)

    def test_ckpt_sql04_empty_row_and_storage_errors_are_typed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "checkpoint.sqlite3"
            store = SQLiteCheckpointStore.create(path, initial_record())
            self.assertEqual(store.path, path)
            execute_database_update(path, "DELETE FROM checkpoint WHERE singleton = 1", ())
            with self.assertRaises(EngineContractError) as empty:
                store.read()
            self.assertIs(empty.exception.code, FailureCode.CHECKPOINT_NOT_FOUND)

            with self.assertRaises(EngineContractError) as invalid_database:
                SQLiteCheckpointStore(root).read()
            self.assertIs(invalid_database.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)

            broken_parent = root / "not-a-directory"
            broken_parent.write_text("file", encoding="utf-8")
            with self.assertRaises(EngineContractError) as create_error:
                SQLiteCheckpointStore.create(broken_parent / "checkpoint.sqlite3", initial_record())
            self.assertIs(create_error.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)

            path.unlink()
            with self.assertRaises(EngineContractError) as update_error:
                store.compare_and_swap(0, next_record(initial_record()))
            self.assertIs(update_error.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)

    def test_ckpt_sql04_create_race_and_invalid_database_errors_are_typed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "checkpoint.sqlite3"
            original_schema = SQLiteCheckpointStore._create_schema

            def occupied_schema(connection: sqlite3.Connection) -> None:
                original_schema(connection)
                connection.execute(
                    "INSERT INTO checkpoint VALUES (1, 1, 0, '{}', 'checksum')"
                )

            with patch.object(
                SQLiteCheckpointStore,
                "_create_schema",
                staticmethod(occupied_schema),
            ):
                with self.assertRaises(EngineContractError) as collision:
                    SQLiteCheckpointStore.create(path, initial_record())
            self.assertIs(collision.exception.code, FailureCode.CHECKPOINT_ALREADY_EXISTS)

            invalid_path = root / "not-a-database.sqlite3"
            invalid_path.write_text("not a SQLite database", encoding="utf-8")
            with self.assertRaises(EngineContractError) as read_error:
                SQLiteCheckpointStore(invalid_path).read()
            self.assertIs(read_error.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)

            path = root / "schema-error.sqlite3"
            with patch.object(
                SQLiteCheckpointStore,
                "_create_schema",
                side_effect=sqlite3.OperationalError("table checkpoint already exists"),
            ):
                with self.assertRaises(EngineContractError) as raced_schema:
                    SQLiteCheckpointStore.create(path, initial_record())
            self.assertIs(raced_schema.exception.code, FailureCode.CHECKPOINT_ALREADY_EXISTS)
            # create() rolls back and deletes its own partial file on failure, so
            # the path is free again without an explicit cleanup step here.
            self.assertFalse(path.exists())
            with patch.object(
                SQLiteCheckpointStore,
                "_create_schema",
                side_effect=sqlite3.OperationalError("disk I/O error"),
            ):
                with self.assertRaises(EngineContractError) as storage_error:
                    SQLiteCheckpointStore.create(path, initial_record())
            self.assertIs(storage_error.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)

            path = root / "missing-table.sqlite3"
            SQLiteCheckpointStore.create(path, initial_record())
            execute_database_update(path, "DROP TABLE checkpoint", ())
            with self.assertRaises(EngineContractError) as missing_table:
                SQLiteCheckpointStore(path).read()
            self.assertIs(missing_table.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)

    def test_ckpt_sql04_rejects_invalid_timeout_and_generation_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            with self.assertRaises(ValueError):
                SQLiteCheckpointStore(path, timeout=0)
            store = SQLiteCheckpointStore.create(path, initial_record())
            candidate = next_record(initial_record())
            for generation in (-1, True):
                with self.subTest(generation=generation), self.assertRaises(ValueError):
                    store.compare_and_swap(generation, candidate)  # type: ignore[arg-type]
            with self.assertRaises(EngineContractError) as mismatch:
                store.compare_and_swap(1, candidate)
            self.assertIs(mismatch.exception.code, FailureCode.CHECKPOINT_CONFLICT)
            self.assertEqual(store.read(), initial_record())

    def test_ckpt_sql04_create_is_atomic_across_schema_and_insert(self) -> None:
        # sqlite3.Connection is an immutable C type: its `execute` cannot be
        # patched directly. Wrap the real connection instead, so only the
        # schema-creation half of create()'s transaction succeeds before the
        # INSERT fails.
        original_connect = SQLiteCheckpointStore._connect

        class FailingConnection:
            def __init__(self, real: sqlite3.Connection) -> None:
                self._real = real

            def execute(self, statement: str, *parameters: object) -> sqlite3.Cursor:
                if statement.strip().upper().startswith("INSERT INTO CHECKPOINT"):
                    raise sqlite3.OperationalError("simulated insert failure")
                return self._real.execute(statement, *parameters)

            def close(self) -> None:
                self._real.close()

        def failing_connect(self: SQLiteCheckpointStore) -> FailingConnection:
            return FailingConnection(original_connect(self))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            with patch.object(SQLiteCheckpointStore, "_connect", failing_connect):
                with self.assertRaises(EngineContractError) as failed:
                    SQLiteCheckpointStore.create(path, initial_record())
            self.assertIs(failed.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)
            # The schema-creation half of the transaction must not survive on
            # its own: either nothing is left, or a fully valid record is.
            self.assertFalse(path.exists())

            store = SQLiteCheckpointStore.create(path, initial_record())
            self.assertEqual(store.read(), initial_record())

    def test_ckpt_sql05_reconciles_lost_commit_acknowledgement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            initial = initial_record()
            candidate = next_record(initial)
            store = SQLiteCheckpointStore.create(path, initial)
            original_commit = SQLiteCheckpointStore._commit

            def commit_then_lose_acknowledgement(connection: sqlite3.Connection) -> None:
                original_commit(connection)
                raise sqlite3.OperationalError("acknowledgement unavailable")

            with patch.object(
                SQLiteCheckpointStore,
                "_commit",
                side_effect=commit_then_lose_acknowledgement,
            ):
                self.assertEqual(store.compare_and_swap(0, candidate), candidate)
            self.assertEqual(SQLiteCheckpointStore(path).read(), candidate)

    def test_ckpt_sql05_reports_uncommitted_lost_acknowledgement_without_leaks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private-checkpoint.sqlite3"
            initial = initial_record()
            store = SQLiteCheckpointStore.create(path, initial)
            with patch.object(
                SQLiteCheckpointStore,
                "_commit",
                side_effect=sqlite3.OperationalError("private failure text"),
            ):
                with self.assertRaises(CheckpointCommitUncertain) as uncertain:
                    store.compare_and_swap(0, next_record(initial))
            self.assertNotIn("private", str(uncertain.exception))
            self.assertEqual(store.read(), initial)

    def test_ckpt_sql05_reconciliation_handles_unreadable_and_competing_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            initial = initial_record()
            candidate = next_record(initial)
            store = SQLiteCheckpointStore.create(path, initial)
            with patch.object(
                SQLiteCheckpointStore,
                "read",
                side_effect=EngineContractError(FailureCode.CHECKPOINT_STORAGE_FAILED),
            ):
                with self.assertRaises(CheckpointCommitUncertain):
                    store._reconcile_uncertain_commit(0, candidate)

            competing = next_record(candidate)
            with patch.object(SQLiteCheckpointStore, "read", return_value=competing):
                with self.assertRaises(EngineContractError) as conflict:
                    store._reconcile_uncertain_commit(0, candidate)
            self.assertIs(conflict.exception.code, FailureCode.CHECKPOINT_CONFLICT)

    def test_ckpt_sql06_terminated_writer_leaves_a_complete_old_record(self) -> None:
        program = """
import os
import sqlite3
import sys

connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute('BEGIN IMMEDIATE')
connection.execute('UPDATE checkpoint SET generation = 99 WHERE singleton = 1')
os._exit(0)
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            initial = initial_record()
            SQLiteCheckpointStore.create(path, initial)
            process = subprocess.run(
                [sys.executable, "-c", program, str(path)],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(process.returncode, 0)
            self.assertEqual(SQLiteCheckpointStore(path).read(), initial)

    def test_ckpt_sql06_lock_timeout_is_typed_and_does_not_mutate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            initial = initial_record()
            SQLiteCheckpointStore.create(path, initial)
            lock_holder = sqlite3.connect(path, isolation_level=None)
            try:
                lock_holder.execute("BEGIN IMMEDIATE")
                with self.assertRaises(EngineContractError) as locked:
                    SQLiteCheckpointStore(path, timeout=0.01).compare_and_swap(
                        0,
                        next_record(initial),
                    )
            finally:
                lock_holder.execute("ROLLBACK")
                lock_holder.close()
            self.assertIs(locked.exception.code, FailureCode.CHECKPOINT_STORAGE_FAILED)
            self.assertEqual(SQLiteCheckpointStore(path).read(), initial)

    def test_ckpt_sql05_independent_processes_allow_one_generation_commit(self) -> None:
        program = """
import sys
from veridist.engine.checkpoint import CheckpointRecord, SQLiteCheckpointStore
from veridist.engine.errors import EngineContractError

store = SQLiteCheckpointStore(sys.argv[1])
candidate = CheckpointRecord.create(
    format_version=1, source_id='source', source_schema='schema',
    source_revision='private-revision', reducer_id='reducer',
    accumulator_schema='accumulator', plan_digest='plan', cursor=1,
    committed_ranges=((0, 1),), generation=1, operation_token='chunk-1',
    operation_digest='digest-1', state=b'\\xffnext-state',
)
try:
    store.compare_and_swap(0, candidate)
except EngineContractError as error:
    print(error.code.value)
else:
    print('COMMITTED')
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.sqlite3"
            SQLiteCheckpointStore.create(path, initial_record())
            processes = [
                subprocess.Popen(
                    [sys.executable, "-c", program, str(path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                for _ in range(2)
            ]
            outputs = [process.communicate(timeout=10) for process in processes]
        self.assertTrue(all(process.returncode == 0 for process in processes))
        self.assertEqual(
            sorted(stdout.strip() for stdout, _ in outputs),
            ["CHECKPOINT_CONFLICT", "COMMITTED"],
        )
        self.assertTrue(all(not stderr for _, stderr in outputs))


if __name__ == "__main__":
    unittest.main()
