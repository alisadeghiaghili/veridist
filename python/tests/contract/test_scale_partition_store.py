"""The SQLite partition-state store: schema, commit protocol, integrity and failure model.

Evidence for the store of ADR-0015 and ADR-0023 carried over to partition states: create and
reopen, racing identical keys, malformed and corrupted rows, storage failures without leaked
detail, a commit that is lost or never happens, and connections that are always closed.
"""

from __future__ import annotations

import pickle
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any
from unittest import mock

from tests.unit.scale_checkpoint_fixtures import Tracker, execute, make_single_row_database, rows
from veridist.engine.checkpoint import SQLiteCheckpointStore
from veridist.scale._errors import ScaleCheckpointError, ScaleCheckpointErrorCode
from veridist.scale._partition_store import (
    DEFAULT_TIMEOUT_SECONDS,
    PARTITION_STORE_FORMAT_VERSION,
    STORE_KIND,
    LedgerEntry,
    ManifestEntry,
    SQLitePartitionStore,
    StoredJob,
    StoredState,
    _row_checksum,
)
from veridist.scale._sources import canonical_digest

FP_A = "sha256:" + "a" * 64
FP_B = "sha256:" + "b" * 64
FP_C = "sha256:" + "c" * 64
COMPONENTS = (("family", "weibull_min"), ("max_rows", "262144"))
MANIFEST = (
    ManifestEntry("p#1", FP_B, "metadata"),
    ManifestEntry("p#0", FP_A, "metadata"),
)
KEY = "k" * 64
STATE = b"\x00state\xff"


def code_of(
    case: unittest.TestCase, call: Any, *arguments: Any, **options: Any
) -> ScaleCheckpointError:
    with case.assertRaises(ScaleCheckpointError) as caught:
        call(*arguments, **options)
    assert isinstance(caught.exception, ScaleCheckpointError)
    return caught.exception


class StoreCase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "job.db"
        self.store = SQLitePartitionStore(self.path)

    def started(self) -> SQLitePartitionStore:
        self.store.create_job("d" * 64, COMPONENTS, MANIFEST)
        return self.store

    def expect(
        self, call: Any, code: ScaleCheckpointErrorCode, *arguments: Any, **options: Any
    ) -> ScaleCheckpointError:
        error = code_of(self, call, *arguments, **options)
        self.assertIs(error.code, code)
        return error


class ConstructionTests(StoreCase):
    def test_constants_name_the_store_and_its_format(self) -> None:
        self.assertEqual(STORE_KIND, "sqlite_partition")
        self.assertEqual(PARTITION_STORE_FORMAT_VERSION, 2)
        self.assertEqual(DEFAULT_TIMEOUT_SECONDS, 5.0)

    def test_the_path_is_kept_and_the_timeout_validated(self) -> None:
        self.assertEqual(self.store.path, self.path)
        self.assertEqual(SQLitePartitionStore(str(self.path)).path, self.path)
        for bad in (0, -1, 0.0, True, "5", None):
            with (
                self.subTest(timeout=bad),
                self.assertRaisesRegex(ValueError, "^timeout must be positive$"),
            ):
                SQLitePartitionStore(self.path, timeout=bad)  # type: ignore[arg-type]
        SQLitePartitionStore(self.path, timeout=0.5)
        SQLitePartitionStore(self.path, timeout=2)

    def test_nothing_is_created_by_construction_or_by_reads(self) -> None:
        self.assertIsNone(self.store.read_job())
        self.assertEqual(self.store.read_pass(0, KEY), {})
        self.assertEqual(self.store.read_ledger(), {})
        self.assertFalse(self.path.exists())

    def test_a_database_that_holds_nothing_has_no_job(self) -> None:
        sqlite3.connect(self.path).close()
        self.assertIsNone(self.store.read_job())
        self.assertEqual(self.store.read_pass(0, KEY), {})
        self.assertEqual(self.store.read_ledger(), {})


class JobTests(StoreCase):
    def test_a_job_round_trips_with_its_manifest_in_identifier_order(self) -> None:
        self.store.create_job("d" * 64, COMPONENTS, MANIFEST)
        job = SQLitePartitionStore(self.path).read_job()
        assert job is not None
        self.assertEqual(job.digest, "d" * 64)
        self.assertEqual(job.components, COMPONENTS)
        self.assertEqual(job.manifest, (MANIFEST[1], MANIFEST[0]))

    def test_a_job_without_partitions_or_components_round_trips(self) -> None:
        self.store.create_job("e" * 64, (), ())
        job = self.store.read_job()
        assert job is not None
        self.assertEqual((job.components, job.manifest), ((), ()))

    def test_the_store_has_exactly_the_four_tables_of_its_format(self) -> None:
        self.started()
        names = {
            row[0] for row in rows(self.path, "SELECT name FROM sqlite_master WHERE type='table'")
        }
        self.assertEqual(names, {"job", "manifest", "partition_state", "pass_ledger"})
        self.assertEqual(
            rows(self.path, "SELECT singleton, format_version FROM job"),
            [(1, PARTITION_STORE_FORMAT_VERSION)],
        )

    def test_a_second_job_is_refused_and_changes_nothing(self) -> None:
        self.started()
        before = self.path.read_bytes()
        self.expect(
            self.store.create_job, ScaleCheckpointErrorCode.JOB_EXISTS, "f" * 64, COMPONENTS, ()
        )
        self.assertEqual(self.path.read_bytes(), before)

    def test_the_parent_directories_are_made(self) -> None:
        store = SQLitePartitionStore(self.root / "a" / "b" / "job.db")
        store.create_job("d" * 64, COMPONENTS, MANIFEST)
        self.assertIsNotNone(store.read_job())

    def test_an_unmakeable_directory_is_a_storage_failure(self) -> None:
        blocker = self.root / "file"
        blocker.write_bytes(b"x")
        store = SQLitePartitionStore(blocker / "job.db")
        self.expect(
            store.create_job, ScaleCheckpointErrorCode.STORE_STORAGE_FAILED, "d" * 64, (), ()
        )

    def test_a_job_can_be_created_in_an_empty_database_file(self) -> None:
        sqlite3.connect(self.path).close()
        self.started()
        self.assertIsNotNone(self.store.read_job())

    def test_a_job_row_without_content_reads_as_no_job(self) -> None:
        self.started()
        execute(self.path, "DELETE FROM job")
        self.assertIsNone(self.store.read_job())

    def test_a_failed_creation_removes_the_file_it_made_and_only_that(self) -> None:
        with mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_lose_before_commit)):
            self.expect(
                self.store.create_job,
                ScaleCheckpointErrorCode.STORE_COMMIT_UNCERTAIN,
                "d" * 64,
                COMPONENTS,
                MANIFEST,
            )
        self.assertFalse(self.path.exists())
        sqlite3.connect(self.path).close()
        with mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_lose_before_commit)):
            self.expect(
                self.store.create_job,
                ScaleCheckpointErrorCode.STORE_COMMIT_UNCERTAIN,
                "d" * 64,
                COMPONENTS,
                MANIFEST,
            )
        self.assertTrue(self.path.exists())

    def test_a_creation_whose_acknowledgement_is_lost_stands(self) -> None:
        with mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_lose_after_commit)):
            self.store.create_job("d" * 64, COMPONENTS, MANIFEST)
        job = self.store.read_job()
        assert job is not None
        self.assertEqual(job.digest, "d" * 64)

    def test_a_creation_that_dies_before_its_commit_leaves_no_job(self) -> None:
        with (
            mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_die)),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.store.create_job("d" * 64, COMPONENTS, MANIFEST)
        self.assertFalse(self.path.exists())


def _lose_after_commit(connection: sqlite3.Connection) -> None:
    connection.execute("COMMIT")
    raise sqlite3.OperationalError("the acknowledgement was lost")


def _lose_before_commit(connection: sqlite3.Connection) -> None:
    raise sqlite3.OperationalError("the commit did not happen")


def _die(connection: sqlite3.Connection) -> None:
    raise KeyboardInterrupt


class FormatTests(StoreCase):
    def single_row_database(self) -> None:
        make_single_row_database(self.path)

    def test_a_single_row_checkpoint_database_is_an_unsupported_format_everywhere(self) -> None:
        self.single_row_database()
        before = self.path.read_bytes()
        unsupported = ScaleCheckpointErrorCode.STORE_FORMAT_UNSUPPORTED
        self.expect(self.store.read_job, unsupported)
        self.expect(self.store.read_pass, unsupported, 0, KEY)
        self.expect(self.store.read_ledger, unsupported)
        self.expect(self.store.create_job, unsupported, "d" * 64, COMPONENTS, MANIFEST)
        self.expect(self.store.commit_partition, unsupported, 0, KEY, "p", FP_A, "metadata", STATE)
        self.expect(self.store.commit_pass, unsupported, 0, KEY, "[]", STATE, prune=False)
        self.expect(self.store.amend_job, unsupported)
        self.assertEqual(self.path.read_bytes(), before)

    def test_the_single_row_store_does_not_read_this_format(self) -> None:
        self.started()
        with self.assertRaises(Exception) as caught:
            SQLiteCheckpointStore(self.path).read()
        self.assertNotIsInstance(caught.exception, ScaleCheckpointError)

    def test_a_database_of_something_else_is_an_unsupported_format(self) -> None:
        connection = sqlite3.connect(self.path)
        connection.execute("CREATE TABLE elsewhere (x INTEGER)")
        connection.commit()
        connection.close()
        self.expect(self.store.read_job, ScaleCheckpointErrorCode.STORE_FORMAT_UNSUPPORTED)

    def test_another_format_version_is_unsupported_and_not_migrated(self) -> None:
        self.started()
        execute(self.path, "UPDATE job SET format_version = 3")
        before = self.path.read_bytes()
        self.expect(self.store.read_job, ScaleCheckpointErrorCode.STORE_FORMAT_UNSUPPORTED)
        self.expect(
            self.store.commit_partition,
            ScaleCheckpointErrorCode.STORE_FORMAT_UNSUPPORTED,
            0,
            KEY,
            "p#0",
            FP_A,
            "metadata",
            STATE,
        )
        self.assertEqual(self.path.read_bytes(), before)

    def test_a_missing_database_cannot_be_written_and_is_not_created(self) -> None:
        self.expect(
            self.store.commit_partition,
            ScaleCheckpointErrorCode.STORE_STORAGE_FAILED,
            0,
            KEY,
            "p#0",
            FP_A,
            "metadata",
            STATE,
        )
        self.assertFalse(self.path.exists())

    def test_a_write_to_an_empty_database_is_an_unsupported_format(self) -> None:
        sqlite3.connect(self.path).close()
        self.expect(
            self.store.commit_pass,
            ScaleCheckpointErrorCode.STORE_FORMAT_UNSUPPORTED,
            0,
            KEY,
            "[]",
            STATE,
            prune=False,
        )


class PartitionStateTests(StoreCase):
    def test_a_state_round_trips_with_its_fingerprint_and_level(self) -> None:
        store = self.started()
        self.assertTrue(store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE))
        found = SQLitePartitionStore(self.path).read_pass(0, KEY)
        self.assertEqual(found, {"p#0": StoredState(FP_A, "metadata", STATE)})

    def test_every_binary_state_survives_byte_for_byte(self) -> None:
        store = self.started()
        for size, partition in ((0, "p#0"), (256, "p#1")):
            data = bytes(range(256))[:size] if size else b""
            store.commit_partition(0, KEY, partition, FP_A, "metadata", data)
            self.assertEqual(store.read_pass(0, KEY)[partition].data, data)

    def test_the_row_holds_the_digest_of_the_state(self) -> None:
        import hashlib

        store = self.started()
        store.commit_partition(3, KEY, "p#0", FP_A, "content", STATE)
        (row,) = rows(
            self.path,
            "SELECT pass_index, pass_key, partition_id, fingerprint, fingerprint_level, "
            "state, state_sha256 FROM partition_state",
        )
        self.assertEqual(
            row,
            (3, KEY, "p#0", FP_A, "content", STATE, hashlib.sha256(STATE).hexdigest()),
        )

    def test_committing_a_key_again_with_the_same_bytes_is_already_applied(self) -> None:
        store = self.started()
        self.assertTrue(store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE))
        self.assertFalse(store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE))
        self.assertEqual(rows(self.path, "SELECT COUNT(*) FROM partition_state"), [(1,)])

    def test_the_same_key_with_other_bytes_is_a_conflict_and_changes_nothing(self) -> None:
        store = self.started()
        store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE)
        before = self.path.read_bytes()
        self.expect(
            store.commit_partition,
            ScaleCheckpointErrorCode.PARTITION_STATE_CONFLICT,
            0,
            KEY,
            "p#0",
            FP_A,
            "metadata",
            STATE + b"!",
        )
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(store.read_pass(0, KEY)["p#0"].data, STATE)

    def test_the_pass_index_the_key_and_the_partition_make_the_key(self) -> None:
        store = self.started()
        for index, key, partition in (
            (0, KEY, "p#0"),
            (1, KEY, "p#0"),
            (0, "z" * 64, "p#0"),
            (0, KEY, "p#1"),
        ):
            self.assertTrue(
                store.commit_partition(index, key, partition, FP_A, "metadata", bytes([index]))
            )
        self.assertEqual(set(store.read_pass(0, KEY)), {"p#0", "p#1"})
        self.assertEqual(set(store.read_pass(1, KEY)), {"p#0"})
        self.assertEqual(set(store.read_pass(0, "z" * 64)), {"p#0"})
        self.assertEqual(store.read_pass(2, KEY), {})

    def test_two_store_objects_race_one_key(self) -> None:
        self.started()
        results: list[bool] = []
        failures: list[BaseException] = []

        def commit() -> None:
            try:
                store = SQLitePartitionStore(self.path, timeout=30.0)
                results.append(store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE))
            except BaseException as error:
                failures.append(error)

        threads = [threading.Thread(target=commit) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(60.0)
            self.assertFalse(thread.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(sorted(results), [False, False, False, True])
        self.assertEqual(self.store.read_pass(0, KEY)["p#0"].data, STATE)

    def test_a_commit_that_dies_before_it_is_committed_leaves_the_state_absent(self) -> None:
        store = self.started()
        with (
            mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_die)),
            self.assertRaises(KeyboardInterrupt),
        ):
            store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE)
        self.assertEqual(rows(self.path, "SELECT COUNT(*) FROM partition_state"), [(0,)])
        self.assertEqual(store.read_pass(0, KEY), {})
        self.assertTrue(store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE))

    def test_a_lost_acknowledgement_of_an_applied_commit_is_recognised(self) -> None:
        store = self.started()
        with mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_lose_after_commit)):
            self.assertTrue(store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE))
        self.assertEqual(store.read_pass(0, KEY)["p#0"].data, STATE)

    def test_a_commit_that_did_not_happen_is_uncertain_and_leaves_nothing(self) -> None:
        store = self.started()
        with mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_lose_before_commit)):
            self.expect(
                store.commit_partition,
                ScaleCheckpointErrorCode.STORE_COMMIT_UNCERTAIN,
                0,
                KEY,
                "p#0",
                FP_A,
                "metadata",
                STATE,
            )
        self.assertEqual(store.read_pass(0, KEY), {})

    def test_a_commit_that_cannot_be_read_back_is_uncertain(self) -> None:
        store = self.started()
        unreadable = ScaleCheckpointError(ScaleCheckpointErrorCode.STORE_STORAGE_FAILED)
        with (
            mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_lose_after_commit)),
            mock.patch.object(SQLitePartitionStore, "read_pass", side_effect=unreadable),
        ):
            self.expect(
                store.commit_partition,
                ScaleCheckpointErrorCode.STORE_COMMIT_UNCERTAIN,
                0,
                KEY,
                "p#0",
                FP_A,
                "metadata",
                STATE,
            )

    def test_a_row_that_was_changed_after_a_lost_acknowledgement_is_not_taken_for_ours(
        self,
    ) -> None:
        store = self.started()

        def lose_and_replace(connection: sqlite3.Connection) -> None:
            connection.execute("ROLLBACK")
            raise sqlite3.OperationalError("lost")

        other = SQLitePartitionStore(self.path)
        other.commit_partition(0, KEY, "p#0", FP_A, "metadata", b"other")
        with mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(lose_and_replace)):
            self.expect(
                store.commit_partition,
                ScaleCheckpointErrorCode.PARTITION_STATE_CONFLICT,
                0,
                KEY,
                "p#0",
                FP_A,
                "metadata",
                STATE,
            )


class IntegrityTests(StoreCase):
    def stored(self) -> SQLitePartitionStore:
        store = self.started()
        store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE)
        store.commit_pass(0, _key_of("[1]"), "[1]", STATE, prune=False)
        return store

    def test_an_altered_state_is_a_digest_mismatch_and_no_state_is_returned(self) -> None:
        store = self.stored()
        store.commit_partition(0, KEY, "p#1", FP_B, "metadata", b"fine")
        execute(
            self.path, "UPDATE partition_state SET state = ? WHERE partition_id = 'p#0'", b"altered"
        )
        self.expect(store.read_pass, ScaleCheckpointErrorCode.STORE_DIGEST_MISMATCH, 0, KEY)

    def test_every_field_of_a_state_row_is_covered_by_its_checksum(self) -> None:
        for column, value in (
            ("fingerprint", FP_C),
            ("fingerprint_level", "content"),
            ("partition_id", "p#9"),
            ("pass_key", "q" * 64),
            ("pass_index", 5),
            ("state_sha256", "0" * 64),
        ):
            with self.subTest(column=column):
                self.setUp()
                store = self.stored()
                execute(self.path, f"UPDATE partition_state SET {column} = ?", value)
                index = 5 if column == "pass_index" else 0
                key = "q" * 64 if column == "pass_key" else KEY
                self.expect(
                    store.read_pass, ScaleCheckpointErrorCode.STORE_DIGEST_MISMATCH, index, key
                )

    def test_a_row_of_the_wrong_type_is_a_decode_failure(self) -> None:
        store = self.stored()
        execute(self.path, "UPDATE partition_state SET state = 'text'")
        self.expect(store.read_pass, ScaleCheckpointErrorCode.STORE_DECODE_FAILED, 0, KEY)
        self.setUp()
        store = self.stored()
        execute(self.path, "UPDATE partition_state SET fingerprint = x'00'")
        self.expect(store.read_pass, ScaleCheckpointErrorCode.STORE_DECODE_FAILED, 0, KEY)

    def test_the_job_row_is_covered_by_its_checksum(self) -> None:
        for column, value in (("job_digest", "x" * 64), ("components", "family\x1fother")):
            with self.subTest(column=column):
                self.setUp()
                store = self.started()
                execute(self.path, f"UPDATE job SET {column} = ?", value)
                self.expect(store.read_job, ScaleCheckpointErrorCode.STORE_DIGEST_MISMATCH)

    def test_a_job_row_of_the_wrong_type_is_a_decode_failure(self) -> None:
        store = self.started()
        execute(self.path, "UPDATE job SET job_digest = x'00'")
        self.expect(store.read_job, ScaleCheckpointErrorCode.STORE_DECODE_FAILED)

    def test_a_component_without_a_value_is_a_decode_failure_even_with_a_valid_checksum(
        self,
    ) -> None:
        store = self.started()
        text = "no separator"
        checksum = _row_checksum("job", PARTITION_STORE_FORMAT_VERSION, "d" * 64, text)
        execute(self.path, "UPDATE job SET components = ?, checksum = ?", text, checksum)
        self.expect(store.read_job, ScaleCheckpointErrorCode.STORE_DECODE_FAILED)

    def test_a_manifest_row_is_covered_by_its_checksum(self) -> None:
        for column, value in (
            ("fingerprint", FP_C),
            ("fingerprint_level", "content"),
            ("partition_id", "p#7"),
        ):
            with self.subTest(column=column):
                self.setUp()
                store = self.started()
                execute(
                    self.path, f"UPDATE manifest SET {column} = ? WHERE partition_id = 'p#0'", value
                )
                self.expect(store.read_job, ScaleCheckpointErrorCode.STORE_DIGEST_MISMATCH)

    def test_a_manifest_row_of_the_wrong_type_is_a_decode_failure(self) -> None:
        store = self.started()
        execute(
            self.path, "UPDATE manifest SET fingerprint_level = x'00' WHERE partition_id = 'p#0'"
        )
        self.expect(store.read_job, ScaleCheckpointErrorCode.STORE_DECODE_FAILED)

    def test_the_ledger_merged_state_is_verified_by_its_digest(self) -> None:
        store = self.stored()
        execute(self.path, "UPDATE pass_ledger SET merged = ?", b"altered")
        self.expect(store.read_ledger, ScaleCheckpointErrorCode.STORE_DIGEST_MISMATCH)

    def test_the_ledger_key_must_be_the_digest_of_the_descriptor(self) -> None:
        store = self.stored()
        execute(self.path, "UPDATE pass_ledger SET descriptor = '[2]'")
        self.expect(store.read_ledger, ScaleCheckpointErrorCode.STORE_DIGEST_MISMATCH)

    def test_every_field_of_a_ledger_row_is_covered_by_its_checksum(self) -> None:
        store = self.stored()
        execute(self.path, "UPDATE pass_ledger SET pass_index = 4")
        self.expect(store.read_ledger, ScaleCheckpointErrorCode.STORE_DIGEST_MISMATCH)

    def test_a_ledger_row_of_the_wrong_type_is_a_decode_failure(self) -> None:
        store = self.stored()
        execute(self.path, "UPDATE pass_ledger SET merged = 'text'")
        self.expect(store.read_ledger, ScaleCheckpointErrorCode.STORE_DECODE_FAILED)

    def test_errors_carry_no_path_identifier_fingerprint_or_state(self) -> None:
        store = self.stored()
        execute(self.path, "UPDATE partition_state SET state = ?", b"altered")
        error = code_of(self, store.read_pass, 0, KEY)
        text = repr(error) + str(error) + repr(error.args) + repr(vars(error))
        for secret in (str(self.root), "job.db", "p#0", FP_A, "altered", STATE.decode("latin-1")):
            self.assertNotIn(secret, text)


def _key_of(descriptor: str) -> str:
    import hashlib

    return hashlib.sha256(descriptor.encode("utf-8")).hexdigest()


class AmendTests(StoreCase):
    def populated(self) -> SQLitePartitionStore:
        store = self.started()
        store.commit_partition(0, KEY, "p#0", FP_A, "metadata", b"zero")
        store.commit_partition(0, KEY, "p#1", FP_B, "metadata", b"one")
        store.commit_pass(0, _key_of("[1]"), "[1]", b"merged", prune=False)
        return store

    def test_added_partitions_join_the_manifest_and_the_ledger_is_emptied(self) -> None:
        store = self.populated()
        store.amend_job(add=[ManifestEntry("p#2", FP_C, "metadata")])
        job = store.read_job()
        assert job is not None
        self.assertEqual([entry.partition_id for entry in job.manifest], ["p#0", "p#1", "p#2"])
        self.assertEqual(store.read_ledger(), {})
        self.assertEqual(set(store.read_pass(0, KEY)), {"p#0", "p#1"})

    def test_removed_partitions_leave_the_manifest_and_lose_their_states(self) -> None:
        store = self.populated()
        store.amend_job(remove=["p#0"])
        job = store.read_job()
        assert job is not None
        self.assertEqual([entry.partition_id for entry in job.manifest], ["p#1"])
        self.assertEqual(set(store.read_pass(0, KEY)), {"p#1"})
        self.assertEqual(store.read_ledger(), {})

    def test_replaced_partitions_take_a_new_fingerprint_and_lose_their_states(self) -> None:
        store = self.populated()
        store.amend_job(replace=[ManifestEntry("p#0", FP_C, "content")])
        job = store.read_job()
        assert job is not None
        self.assertEqual(job.manifest[0], ManifestEntry("p#0", FP_C, "content"))
        self.assertEqual(job.manifest[1], ManifestEntry("p#1", FP_B, "metadata"))
        self.assertEqual(set(store.read_pass(0, KEY)), {"p#1"})

    def test_a_reset_drops_every_state_and_the_ledger_and_keeps_the_job(self) -> None:
        store = self.populated()
        store.amend_job(reset=True)
        self.assertEqual(store.read_pass(0, KEY), {})
        self.assertEqual(store.read_ledger(), {})
        job = store.read_job()
        assert job is not None
        self.assertEqual(len(job.manifest), 2)
        self.assertEqual(job.digest, "d" * 64)

    def test_an_amendment_with_nothing_to_do_still_drops_the_ledger_only(self) -> None:
        store = self.populated()
        store.amend_job()
        self.assertEqual(store.read_ledger(), {})
        self.assertEqual(set(store.read_pass(0, KEY)), {"p#0", "p#1"})

    def test_an_amendment_whose_acknowledgement_is_lost_stands(self) -> None:
        store = self.populated()
        with mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_lose_after_commit)):
            store.amend_job(add=[ManifestEntry("p#2", FP_C, "metadata")])
        job = store.read_job()
        assert job is not None
        self.assertEqual(len(job.manifest), 3)

    def test_an_amendment_that_did_not_happen_is_uncertain(self) -> None:
        store = self.populated()
        with mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_lose_before_commit)):
            self.expect(
                store.amend_job,
                ScaleCheckpointErrorCode.STORE_COMMIT_UNCERTAIN,
                add=[ManifestEntry("p#2", FP_C, "metadata")],
            )
        job = store.read_job()
        assert job is not None
        self.assertEqual(len(job.manifest), 2)
        self.assertEqual(len(store.read_ledger()), 1)

    def test_an_amendment_that_cannot_be_applied_changes_nothing(self) -> None:
        store = self.populated()
        before = self.path.read_bytes()
        self.expect(
            store.amend_job,
            ScaleCheckpointErrorCode.STORE_STORAGE_FAILED,
            add=[ManifestEntry("p#0", FP_C, "metadata")],
        )
        self.assertEqual(self.path.read_bytes(), before)


class LedgerTests(StoreCase):
    def test_a_pass_round_trips(self) -> None:
        store = self.started()
        key = _key_of("[0]")
        self.assertTrue(store.commit_pass(0, key, "[0]", b"merged", prune=False))
        self.assertEqual(
            SQLitePartitionStore(self.path).read_ledger(), {0: LedgerEntry(key, "[0]", b"merged")}
        )

    def test_the_ledger_is_in_pass_order(self) -> None:
        store = self.started()
        for index in (2, 0, 1):
            descriptor = f"[{index}]"
            store.commit_pass(index, _key_of(descriptor), descriptor, bytes([index]), prune=False)
        self.assertEqual(list(store.read_ledger()), [0, 1, 2])

    def test_committing_a_pass_again_is_already_applied_and_other_content_is_a_conflict(
        self,
    ) -> None:
        store = self.started()
        key = _key_of("[0]")
        store.commit_pass(0, key, "[0]", b"merged", prune=False)
        self.assertFalse(store.commit_pass(0, key, "[0]", b"merged", prune=False))
        before = self.path.read_bytes()
        conflict = ScaleCheckpointErrorCode.PARTITION_STATE_CONFLICT
        self.expect(store.commit_pass, conflict, 0, key, "[0]", b"other", prune=False)
        other = _key_of("[9]")
        self.expect(store.commit_pass, conflict, 0, other, "[9]", b"merged", prune=False)
        self.assertEqual(self.path.read_bytes(), before)

    def test_pruning_drops_the_states_of_earlier_passes_in_the_same_transaction(self) -> None:
        store = self.started()
        for index in (0, 1, 2):
            store.commit_partition(index, KEY, "p#0", FP_A, "metadata", bytes([index]))
        store.commit_pass(2, _key_of("[2]"), "[2]", b"m", prune=True)
        self.assertEqual(store.read_pass(0, KEY), {})
        self.assertEqual(store.read_pass(1, KEY), {})
        self.assertEqual(set(store.read_pass(2, KEY)), {"p#0"})

    def test_without_pruning_every_state_is_kept(self) -> None:
        store = self.started()
        for index in (0, 1):
            store.commit_partition(index, KEY, "p#0", FP_A, "metadata", bytes([index]))
        store.commit_pass(1, _key_of("[1]"), "[1]", b"m", prune=False)
        self.assertEqual(set(store.read_pass(0, KEY)), {"p#0"})

    def test_a_conflicting_pass_does_not_prune(self) -> None:
        store = self.started()
        store.commit_partition(0, KEY, "p#0", FP_A, "metadata", b"s")
        store.commit_pass(1, _key_of("[1]"), "[1]", b"m", prune=False)
        self.expect(
            store.commit_pass,
            ScaleCheckpointErrorCode.PARTITION_STATE_CONFLICT,
            1,
            _key_of("[1]"),
            "[1]",
            b"x",
            prune=True,
        )
        self.assertEqual(set(store.read_pass(0, KEY)), {"p#0"})

    def test_a_lost_acknowledgement_of_a_pass_is_recognised_and_an_absent_pass_is_uncertain(
        self,
    ) -> None:
        store = self.started()
        key = _key_of("[0]")
        with mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_lose_after_commit)):
            self.assertTrue(store.commit_pass(0, key, "[0]", b"m", prune=False))
        with mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_lose_before_commit)):
            self.expect(
                store.commit_pass,
                ScaleCheckpointErrorCode.STORE_COMMIT_UNCERTAIN,
                1,
                _key_of("[1]"),
                "[1]",
                b"m",
                prune=False,
            )
        self.assertEqual(list(store.read_ledger()), [0])


class StorageFailureTests(StoreCase):
    def test_a_directory_in_place_of_the_database_is_a_storage_failure(self) -> None:
        self.path.mkdir()
        for call, arguments in (
            (self.store.read_job, ()),
            (self.store.read_pass, (0, KEY)),
            (self.store.read_ledger, ()),
        ):
            with self.subTest(call=call.__name__):
                error = self.expect(call, ScaleCheckpointErrorCode.STORE_STORAGE_FAILED, *arguments)
                self.assertTrue((error.detail or "").startswith("SQLITE_CANTOPEN"))
                self.assertNotIn(str(self.root), str(error))

    def test_a_file_that_is_not_a_database_is_a_storage_failure(self) -> None:
        self.path.write_bytes(b"this is not a database" * 100)
        error = self.expect(self.store.read_job, ScaleCheckpointErrorCode.STORE_STORAGE_FAILED)
        self.assertEqual(error.detail, "SQLITE_NOTADB")
        self.assertEqual(str(error), "STORE_STORAGE_FAILED [SQLITE_NOTADB]")
        self.assertEqual(self.path.read_bytes(), b"this is not a database" * 100)

    def test_a_locked_database_is_a_bounded_storage_failure(self) -> None:
        self.started()
        store = SQLitePartitionStore(self.path, timeout=0.05)
        holder = sqlite3.connect(self.path, isolation_level=None)
        try:
            holder.execute("BEGIN IMMEDIATE")
            error = self.expect(
                store.commit_partition,
                ScaleCheckpointErrorCode.STORE_STORAGE_FAILED,
                0,
                KEY,
                "p#0",
                FP_A,
                "metadata",
                STATE,
            )
            self.assertEqual(error.detail, "SQLITE_BUSY")
        finally:
            holder.execute("ROLLBACK")
            holder.close()
        self.assertTrue(store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE))

    def test_a_storage_error_without_a_symbolic_name_has_no_detail(self) -> None:
        from veridist.scale._partition_store import _storage_failed

        self.assertIsNone(_storage_failed().detail)
        self.assertIsNone(_storage_failed(sqlite3.Error("free text with /a/path")).detail)


class ConnectionTests(StoreCase):
    def test_every_operation_closes_every_connection_it_opened(self) -> None:
        tracker = Tracker()
        with tracker.watching():
            store = self.started()
            store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE)
            store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE)
            store.read_pass(0, KEY)
            store.commit_pass(0, _key_of("[0]"), "[0]", b"m", prune=True)
            store.read_ledger()
            store.read_job()
            store.amend_job(reset=True)
            store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE)
            code_of(
                self,
                store.commit_partition,
                0,
                KEY,
                "p#0",
                FP_A,
                "metadata",
                b"different",
            )
            code_of(self, store.create_job, "e" * 64, (), ())
            with (
                mock.patch.object(SQLitePartitionStore, "_commit", staticmethod(_die)),
                self.assertRaises(KeyboardInterrupt),
            ):
                store.commit_pass(1, _key_of("[1]"), "[1]", b"m", prune=False)
        self.assertGreater(tracker.opened, 10)
        self.assertEqual(tracker.opened, tracker.closed)

    def test_failed_opens_and_missing_stores_leave_nothing_open(self) -> None:
        tracker = Tracker()
        with tracker.watching():
            self.path.mkdir()
            code_of(self, self.store.read_job)
            code_of(self, self.store.read_ledger)
        self.assertEqual(tracker.opened, tracker.closed)

    def test_a_write_requests_full_durability_before_it_begins_and_reads_do_not(self) -> None:
        store = self.started()
        tracker = Tracker()
        with tracker.watching():
            store.read_job()
            reads = list(tracker.statements)
            tracker.statements.clear()
            store.commit_partition(0, KEY, "p#0", FP_A, "metadata", STATE)
            writes = list(tracker.statements)
        self.assertNotIn("PRAGMA synchronous = FULL", reads)
        self.assertEqual(writes[0], "PRAGMA synchronous = FULL")
        self.assertEqual(writes[1], "BEGIN IMMEDIATE")
        self.assertEqual(writes[-1], "COMMIT")


class ErrorTests(unittest.TestCase):
    def test_the_message_is_the_code_and_what_the_error_names(self) -> None:
        self.assertEqual(str(ScaleCheckpointError("JOB_EXISTS")), "JOB_EXISTS")
        error = ScaleCheckpointError(ScaleCheckpointErrorCode.JOB_MISMATCH, ("family", "column"))
        self.assertEqual(str(error), "JOB_MISMATCH (family, column)")
        self.assertEqual(error.components, ("family", "column"))
        detailed = ScaleCheckpointError(
            ScaleCheckpointErrorCode.STORE_STORAGE_FAILED, detail="SQLITE_FULL"
        )
        self.assertEqual(str(detailed), "STORE_STORAGE_FAILED [SQLITE_FULL]")
        self.assertIsInstance(error, ValueError)

    def test_the_codes_are_closed(self) -> None:
        with self.assertRaises(ValueError):
            ScaleCheckpointError("NOT_A_CODE")
        self.assertEqual(
            {code.value for code in ScaleCheckpointErrorCode},
            {
                "SOURCE_REVISION_UNAVAILABLE",
                "INCREMENTAL_UNSUPPORTED",
                "JOB_EXISTS",
                "JOB_MISMATCH",
                "PARTITION_MISSING",
                "PARTITION_ADDED",
                "PARTITION_STATE_CONFLICT",
                "LEDGER_MISMATCH",
                "STORE_FORMAT_UNSUPPORTED",
                "STORE_DECODE_FAILED",
                "STORE_DIGEST_MISMATCH",
                "STORE_STORAGE_FAILED",
                "STORE_COMMIT_UNCERTAIN",
            },
        )

    def test_an_error_pickles_with_everything_it_names(self) -> None:
        error = ScaleCheckpointError(
            ScaleCheckpointErrorCode.JOB_MISMATCH, ("family",), "SQLITE_BUSY"
        )
        copy = pickle.loads(pickle.dumps(error))
        self.assertIsInstance(copy, ScaleCheckpointError)
        self.assertEqual(
            (copy.code, copy.components, copy.detail, str(copy)),
            (error.code, error.components, error.detail, str(error)),
        )


class ValueTests(unittest.TestCase):
    def test_the_records_are_immutable_and_have_no_dictionary(self) -> None:
        for item, field in (
            (ManifestEntry("p", FP_A, "metadata"), "level"),
            (StoredJob("d", (), ()), "digest"),
            (StoredState(FP_A, "metadata", b""), "data"),
            (LedgerEntry("k", "[]", b""), "data"),
        ):
            with self.subTest(type=type(item).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(item, field, None)
                self.assertFalse(hasattr(item, "__dict__"))


class RowChecksumTests(unittest.TestCase):
    def test_a_row_checksum_is_a_canonical_digest_of_the_kind_and_fields(self) -> None:
        self.assertEqual(
            _row_checksum("manifest", "a", 1),
            canonical_digest("veridist.scale.partition_store", "manifest", "a", 1),
        )
        self.assertNotEqual(_row_checksum("manifest", "a"), _row_checksum("state", "a"))
        self.assertNotEqual(_row_checksum("manifest", "a", "b"), _row_checksum("manifest", "ab"))
