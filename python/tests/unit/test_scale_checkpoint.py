"""The checkpoint logic between the store and the executors: arguments, identity, resume rules.

Everything here runs in this process on small in-memory partitions; the fits themselves are in
``test_scale_checkpoint_fit``.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from veridist.scale._checkpoint import (
    COMPONENT_NAMES,
    HINT_CHANGED,
    NO_COMPONENT,
    CheckpointRequest,
    CheckpointRun,
    JobIdentity,
    job_identity,
    refuse_incremental,
    require_resumable,
    resolve_checkpoint,
)
from veridist.scale._environment import numeric_environment_digest
from veridist.scale._errors import (
    ScaleCheckpointError,
    ScaleCheckpointErrorCode,
    ScaleSourceError,
    ScaleSourceErrorCode,
)
from veridist.scale._one_pass import ExponentialState
from veridist.scale._partition_store import (
    PARTITION_STORE_FORMAT_VERSION,
    STORE_KIND,
    SQLitePartitionStore,
)
from veridist.scale._sources import (
    VALUE_POLICY_VERSION,
    FingerprintLevel,
    NumpySource,
    Partition,
    canonical_digest,
)
from veridist.scale._tasks import FoldJob, TaskOutcome

JOB = FoldJob(ExponentialState)
ARGUMENTS: dict[str, Any] = {
    "family": "exponential",
    "column": None,
    "source_kind": "numpy",
    "max_rows": 100,
    "state_schema": "scale.exponential:1",
    "plan": None,
}


def parts(count: int = 4, size: int = 25, seed: int = 3) -> tuple[NumpySource, np.ndarray]:
    values = np.random.default_rng(seed).exponential(3.0, count * size)
    return NumpySource(values, partition_rows=size), values


def identity_of(partitions: tuple[Partition, ...], **changes: Any) -> JobIdentity:
    return job_identity(partitions=partitions, **{**ARGUMENTS, **changes})


class CheckpointCase(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "job.db"
        self.store = SQLitePartitionStore(self.path)
        self.source, self.values = parts()
        self.partitions = self.source.partitions()

    def request(self, **changes: bool) -> CheckpointRequest:
        flags = {
            "resume": True,
            "incremental": False,
            "recompute_changed": False,
            "allow_removed": False,
        }
        return CheckpointRequest(self.store, **{**flags, **changes})

    def open(
        self, partitions: Any = None, multi_pass: bool = False, **changes: bool
    ) -> CheckpointRun:
        used = self.partitions if partitions is None else partitions
        return CheckpointRun.open(
            self.request(**changes), identity_of(used), used, multi_pass=multi_pass
        )

    def expect(
        self, code: ScaleCheckpointErrorCode, call: Any, *arguments: Any, **options: Any
    ) -> Any:
        with self.assertRaises(ScaleCheckpointError) as caught:
            call(*arguments, **options)
        self.assertIs(caught.exception.code, code)
        return caught.exception

    def outcome(self, partition: Partition) -> TaskOutcome:
        data = JOB.run(self.source, partition, 100)
        return TaskOutcome(partition.ordinal, data, 0)


class ResolveTests(CheckpointCase):
    def test_no_checkpoint_and_no_flag_is_no_request(self) -> None:
        self.assertIsNone(resolve_checkpoint(None, False, False, False, False))

    def test_a_flag_needs_a_checkpoint(self) -> None:
        for flags in (
            (True, False, False, False),
            (False, True, False, False),
            (False, False, True, False),
            (False, False, False, True),
        ):
            with (
                self.subTest(flags=flags),
                self.assertRaisesRegex(ValueError, "need a checkpoint$"),
            ):
                resolve_checkpoint(None, *flags)

    def test_a_path_or_a_store_is_accepted(self) -> None:
        request = resolve_checkpoint(self.path, False, False, False, False)
        assert request is not None
        self.assertEqual(request.store.path, self.path)
        self.assertFalse(self.path.exists())
        from_text = resolve_checkpoint(str(self.path), True, False, False, False)
        assert from_text is not None
        self.assertEqual(from_text.store.path, self.path)
        again = resolve_checkpoint(self.store, True, True, True, False)
        assert again is not None
        self.assertIs(again.store, self.store)
        self.assertEqual(
            (again.resume, again.incremental, again.recompute_changed, again.allow_removed),
            (True, True, True, False),
        )

    def test_the_flags_are_kept_in_their_own_fields(self) -> None:
        request = resolve_checkpoint(self.store, True, True, False, True)
        assert request is not None
        self.assertEqual(
            (request.resume, request.incremental, request.recompute_changed, request.allow_removed),
            (True, True, False, True),
        )
        other = resolve_checkpoint(self.store, False, False, True, False)
        assert other is not None
        self.assertEqual(
            (other.resume, other.incremental, other.recompute_changed, other.allow_removed),
            (False, False, True, False),
        )

    def test_a_path_like_of_bytes_is_a_type_error(self) -> None:
        class BytesPath:
            def __fspath__(self) -> bytes:
                return b"job.db"

        with self.assertRaises(TypeError):
            resolve_checkpoint(BytesPath(), False, False, False, False)

    def test_another_type_is_a_type_error(self) -> None:
        for bad in (5, b"path", object(), ["path"]):
            with self.subTest(checkpoint=bad), self.assertRaises(TypeError):
                resolve_checkpoint(bad, False, False, False, False)

    def test_a_flag_must_be_a_bool(self) -> None:
        names = ("resume", "incremental", "recompute_changed", "allow_removed")
        for position, name in enumerate(names):
            for bad in (1, 0, "yes", None):
                flags: list[Any] = [False] * 4
                flags[position] = bad
                with (
                    self.subTest(position=position, value=bad),
                    self.assertRaisesRegex(TypeError, f"^{name} must be a bool$"),
                ):
                    resolve_checkpoint(self.store, *flags)

    def test_incremental_needs_resume_and_allow_removed_needs_incremental(self) -> None:
        with self.assertRaisesRegex(ValueError, "^incremental needs resume=True$"):
            resolve_checkpoint(self.store, False, True, False, False)
        with self.assertRaisesRegex(ValueError, "^allow_removed needs incremental=True$"):
            resolve_checkpoint(self.store, True, False, False, True)
        self.assertIsNotNone(resolve_checkpoint(self.store, True, True, False, True))

    def test_a_flag_is_checked_before_the_checkpoint_is(self) -> None:
        with self.assertRaises(TypeError):
            resolve_checkpoint(None, "yes", False, False, False)


class ValueTests(CheckpointCase):
    def test_the_request_and_the_identity_are_immutable_and_have_no_dictionary(self) -> None:
        request = self.request()
        identity = identity_of(self.partitions)
        for item, field in ((request, "resume"), (identity, "components")):
            with self.subTest(type=type(item).__name__):
                with self.assertRaises(FrozenInstanceError):
                    setattr(item, field, None)
                self.assertFalse(hasattr(item, "__dict__"))


class RefusalTests(CheckpointCase):
    def test_a_partition_without_a_fingerprint_cannot_be_checkpointed(self) -> None:
        unresumable = Partition("a", 0, 3, None, FingerprintLevel.NONE)
        error = self.expect(
            ScaleCheckpointErrorCode.SOURCE_REVISION_UNAVAILABLE,
            require_resumable,
            (*self.partitions, unresumable),
        )
        self.assertEqual(str(error), "SOURCE_REVISION_UNAVAILABLE")
        require_resumable(self.partitions)
        require_resumable(())

    def test_incremental_fitting_is_refused_for_a_multi_pass_family_only(self) -> None:
        self.expect(
            ScaleCheckpointErrorCode.INCREMENTAL_UNSUPPORTED,
            refuse_incremental,
            self.request(incremental=True),
            multi_pass=True,
        )
        refuse_incremental(self.request(incremental=True), multi_pass=False)
        refuse_incremental(self.request(), multi_pass=True)
        refuse_incremental(self.request(), multi_pass=False)


class IdentityTests(CheckpointCase):
    def test_the_components_are_named_in_order(self) -> None:
        self.assertEqual(
            COMPONENT_NAMES,
            (
                "family",
                "column",
                "source_kind",
                "max_rows",
                "value_policy",
                "fingerprint_level",
                "state_schema",
                "plan",
                "environment",
            ),
        )
        identity = identity_of(self.partitions)
        self.assertEqual(tuple(name for name, _ in identity.components), COMPONENT_NAMES)

    def test_a_component_that_does_not_apply_is_a_dash(self) -> None:
        self.assertEqual(NO_COMPONENT, "-")

    def test_the_values_are_what_the_fit_is(self) -> None:
        values = dict(identity_of(self.partitions).components)
        self.assertEqual(values["family"], "exponential")
        self.assertEqual(values["column"], NO_COMPONENT)
        self.assertEqual(values["source_kind"], "numpy")
        self.assertEqual(values["max_rows"], "100")
        self.assertEqual(values["value_policy"], str(VALUE_POLICY_VERSION))
        self.assertEqual(values["fingerprint_level"], "content")
        self.assertEqual(values["state_schema"], "scale.exponential:1")
        self.assertEqual(values["plan"], NO_COMPONENT)
        self.assertEqual(values["environment"], numeric_environment_digest())

    def test_a_column_and_a_plan_enter_as_digests_and_never_as_text(self) -> None:
        values = dict(
            identity_of(self.partitions, column="secret_column", plan=(8, "x")).components
        )
        self.assertEqual(values["column"], canonical_digest("column", "secret_column"))
        self.assertEqual(values["plan"], canonical_digest(8, "x"))
        self.assertNotIn("secret_column", repr(values))

    def test_the_levels_are_the_sorted_set_of_the_partitions(self) -> None:
        mixed = (
            Partition("a", 0, 1, "sha256:" + "a" * 64, FingerprintLevel.METADATA),
            Partition("b", 1, 1, "sha256:" + "b" * 64, FingerprintLevel.CONTENT),
            Partition("c", 2, 1, "sha256:" + "c" * 64, FingerprintLevel.METADATA),
        )
        self.assertEqual(
            dict(identity_of(mixed).components)["fingerprint_level"], "content,metadata"
        )
        self.assertEqual(dict(identity_of(()).components)["fingerprint_level"], NO_COMPONENT)

    def test_the_digest_is_the_hash_of_the_canonical_record_and_follows_every_component(
        self,
    ) -> None:
        identity = identity_of(self.partitions)
        flat = [item for pair in identity.components for item in pair]
        expected = hashlib.sha256(canonical_digest(*flat).encode("ascii")).hexdigest()
        self.assertEqual(identity.digest, expected)
        self.assertEqual(len(identity.digest), 64)
        for index, (name, value) in enumerate(identity.components):
            with self.subTest(component=name):
                changed = list(identity.components)
                changed[index] = (name, value + "x")
                self.assertNotEqual(JobIdentity(tuple(changed)).digest, identity.digest)

    def test_equal_jobs_have_equal_identities_and_a_different_job_does_not(self) -> None:
        base = identity_of(self.partitions)
        self.assertEqual(identity_of(self.partitions), base)
        for change in (
            {"family": "gamma"},
            {"column": "x"},
            {"source_kind": "parquet"},
            {"max_rows": 99},
            {"state_schema": "other:1"},
            {"plan": (1,)},
        ):
            with self.subTest(change=change):
                self.assertNotEqual(identity_of(self.partitions, **change), base)

    def test_the_value_policy_and_the_environment_are_part_of_the_identity(self) -> None:
        base = identity_of(self.partitions)
        with mock.patch("veridist.scale._checkpoint.VALUE_POLICY_VERSION", 99):
            self.assertNotEqual(identity_of(self.partitions), base)
        with mock.patch("veridist.scale._checkpoint.numeric_environment_digest", return_value="e"):
            self.assertEqual(dict(identity_of(self.partitions).components)["environment"], "e")


class OpenTests(CheckpointCase):
    def test_a_new_job_is_created_with_the_manifest_of_the_source(self) -> None:
        run = self.open()
        job = self.store.read_job()
        assert job is not None
        self.assertEqual(job.digest, identity_of(self.partitions).digest)
        self.assertEqual(job.components, identity_of(self.partitions).components)
        self.assertEqual(
            [(entry.partition_id, entry.fingerprint, entry.level) for entry in job.manifest],
            sorted((p.id, p.fingerprint, p.fingerprint_level.value) for p in self.partitions),
        )
        record = run.provenance()
        self.assertEqual(
            (record.store_kind, record.store_format_version),
            (STORE_KIND, PARTITION_STORE_FORMAT_VERSION),
        )
        self.assertEqual(record.job_digest_sha256, job.digest)
        self.assertFalse(record.resumed)
        self.assertEqual(record.passes, ())

    def test_a_job_that_exists_is_not_overwritten_unless_resumed(self) -> None:
        self.open()
        before = self.path.read_bytes()
        self.expect(ScaleCheckpointErrorCode.JOB_EXISTS, self.open, resume=False)
        self.assertEqual(self.path.read_bytes(), before)

    def test_resuming_a_store_without_a_job_starts_one(self) -> None:
        run = self.open(resume=True)
        self.assertFalse(run.provenance().resumed)
        self.assertIsNotNone(self.store.read_job())

    def test_resuming_a_matching_job_continues_it(self) -> None:
        self.open()
        run = self.open()
        record = run.provenance()
        self.assertTrue(record.resumed)
        self.assertEqual(
            (record.added_partitions, record.removed_partitions, record.recomputed_partitions),
            (0, 0, 0),
        )
        self.assertFalse(record.restarted)

    def test_each_component_that_differs_is_named_and_nothing_is_written(self) -> None:
        self.open()
        before = self.path.read_bytes()
        for name in COMPONENT_NAMES:
            with self.subTest(component=name):
                identity = identity_of(self.partitions)
                altered = tuple(
                    (key, value + "!") if key == name else (key, value)
                    for key, value in identity.components
                )
                error = self.expect(
                    ScaleCheckpointErrorCode.JOB_MISMATCH,
                    CheckpointRun.open,
                    self.request(),
                    JobIdentity(altered),
                    self.partitions,
                    multi_pass=False,
                )
                self.assertEqual(error.components, (name,))
                self.assertEqual(str(error), f"JOB_MISMATCH ({name})")
        self.assertEqual(self.path.read_bytes(), before)

    def test_several_components_are_all_named_in_their_order(self) -> None:
        self.open()
        other = identity_of(self.partitions, family="gamma", max_rows=7, plan=(1,))
        error = self.expect(
            ScaleCheckpointErrorCode.JOB_MISMATCH,
            CheckpointRun.open,
            self.request(),
            other,
            self.partitions,
            multi_pass=False,
        )
        self.assertEqual(error.components, ("family", "max_rows", "plan"))

    def changed_partitions(self) -> tuple[Partition, ...]:
        edited = self.values.copy()
        edited[0] += 1.0
        return NumpySource(edited, partition_rows=25).partitions()

    def test_a_changed_partition_is_refused_before_anything_changes(self) -> None:
        self.open()
        before = self.path.read_bytes()
        changed = self.changed_partitions()
        with self.assertRaises(ScaleSourceError) as caught:
            self.open(changed)
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)
        self.assertEqual(caught.exception.hint, HINT_CHANGED)
        self.assertIn("recompute_changed=True", HINT_CHANGED)
        self.assertEqual(self.path.read_bytes(), before)

    def test_a_changed_partition_is_replaced_when_asked_and_its_state_is_dropped(self) -> None:
        run = self.open()
        step = run.begin_pass(0, JOB, self.partitions)
        for partition in self.partitions:
            step.commit(self.outcome(partition))
        changed = self.changed_partitions()
        again = self.open(changed, recompute_changed=True)
        record = again.provenance()
        self.assertEqual((record.recomputed_partitions, record.restarted), (1, False))
        job = self.store.read_job()
        assert job is not None
        self.assertEqual(
            {entry.partition_id: entry.fingerprint for entry in job.manifest},
            {partition.id: partition.fingerprint for partition in changed},
        )
        held = self.store.read_pass(0, again.begin_pass(0, JOB, changed)._key)
        self.assertEqual(set(held), {p.id for p in changed[1:]})

    def test_a_changed_partition_of_a_multi_pass_job_restarts_every_pass(self) -> None:
        run = self.open(multi_pass=True)
        step = run.begin_pass(0, JOB, self.partitions)
        for partition in self.partitions:
            step.commit(self.outcome(partition))
        step.finish(b"merged")
        again = self.open(self.changed_partitions(), multi_pass=True, recompute_changed=True)
        record = again.provenance()
        self.assertEqual((record.recomputed_partitions, record.restarted), (1, True))
        self.assertEqual(self.store.read_ledger(), {})
        self.assertEqual(self.store.read_pass(0, step._key), {})
        self.assertIsNone(again.begin_pass(0, JOB, self.partitions).replayed)

    def test_a_missing_partition_is_refused_unless_removal_is_allowed(self) -> None:
        self.open()
        before = self.path.read_bytes()
        fewer = self.partitions[:-1]
        self.expect(ScaleCheckpointErrorCode.PARTITION_MISSING, self.open, fewer)
        self.expect(ScaleCheckpointErrorCode.PARTITION_MISSING, self.open, fewer, incremental=True)
        self.assertEqual(self.path.read_bytes(), before)
        record = self.open(fewer, incremental=True, allow_removed=True).provenance()
        self.assertEqual((record.removed_partitions, record.added_partitions), (1, 0))
        job = self.store.read_job()
        assert job is not None
        self.assertEqual(len(job.manifest), 3)

    def test_a_new_partition_is_refused_unless_the_fit_is_incremental(self) -> None:
        self.open(self.partitions[:-1])
        before = self.path.read_bytes()
        self.expect(ScaleCheckpointErrorCode.PARTITION_ADDED, self.open)
        self.assertEqual(self.path.read_bytes(), before)
        record = self.open(incremental=True).provenance()
        self.assertEqual((record.added_partitions, record.removed_partitions), (1, 0))
        job = self.store.read_job()
        assert job is not None
        self.assertEqual(len(job.manifest), 4)

    def test_added_and_removed_partitions_are_both_accepted_together(self) -> None:
        self.open(self.partitions[:2])
        later = self.partitions[1:]
        record = self.open(later, incremental=True, allow_removed=True).provenance()
        self.assertEqual((record.added_partitions, record.removed_partitions), (2, 1))

    def test_the_refusals_come_in_the_order_changed_missing_added(self) -> None:
        self.open(self.partitions[:3])
        # Against the stored partitions 0, 1, 2: partition 0 changed, 3 is new.
        edited = self.changed_partitions()
        with self.assertRaises(ScaleSourceError):
            self.open(edited, incremental=True, allow_removed=True)
        self.expect(
            ScaleCheckpointErrorCode.PARTITION_ADDED, self.open, edited, recompute_changed=True
        )
        # Partition 0 is gone and 3 is new.
        later = self.partitions[1:]
        self.expect(ScaleCheckpointErrorCode.PARTITION_MISSING, self.open, later, incremental=True)
        self.expect(ScaleCheckpointErrorCode.PARTITION_MISSING, self.open, later[:-1])
        self.expect(ScaleCheckpointErrorCode.PARTITION_ADDED, self.open, self.partitions)

    def test_a_removal_or_an_addition_drops_the_ledger_but_keeps_other_states(self) -> None:
        run = self.open()
        step = run.begin_pass(0, JOB, self.partitions)
        for partition in self.partitions:
            step.commit(self.outcome(partition))
        step.finish(b"merged")
        self.open(self.partitions[:-1], incremental=True, allow_removed=True)
        self.assertEqual(self.store.read_ledger(), {})
        held = self.store.read_pass(0, step._key)
        self.assertEqual(set(held), {p.id for p in self.partitions[:-1]})


class PassTests(CheckpointCase):
    def test_a_pass_is_keyed_by_its_exact_parameters(self) -> None:
        run = self.open()
        step = run.begin_pass(0, JOB, self.partitions)
        descriptor = json.dumps(
            ["fold", ExponentialState.STATE_TAG, ExponentialState.SCHEMA_VERSION],
            separators=(",", ":"),
        )
        self.assertEqual(step._descriptor, descriptor)
        self.assertEqual(step._key, hashlib.sha256(descriptor.encode()).hexdigest())

    def test_a_fresh_pass_has_every_partition_pending_and_nothing_reused(self) -> None:
        run = self.open()
        step = run.begin_pass(0, JOB, self.partitions)
        self.assertIsNone(step.replayed)
        self.assertEqual(step.pending, self.partitions)
        self.assertEqual(step.reused, {})

    def test_committed_partitions_are_reused_and_the_rest_stay_pending(self) -> None:
        run = self.open()
        step = run.begin_pass(0, JOB, self.partitions)
        for partition in (self.partitions[2], self.partitions[0]):
            step.commit(self.outcome(partition))
        again = self.open().begin_pass(0, JOB, self.partitions)
        self.assertEqual(set(again.reused), {0, 2})
        self.assertEqual(again.reused[2], JOB.run(self.source, self.partitions[2], 100))
        self.assertEqual(again.pending, (self.partitions[1], self.partitions[3]))
        self.assertIsNone(again.replayed)

    def test_reused_states_are_handed_out_once_in_canonical_order_and_still_counted(self) -> None:
        run = self.open()
        step = run.begin_pass(0, JOB, self.partitions)
        for partition in (self.partitions[3], self.partitions[1]):
            step.commit(self.outcome(partition))
        again = self.open().begin_pass(0, JOB, self.partitions)
        expected = [JOB.run(self.source, self.partitions[n], 100) for n in (1, 3)]
        self.assertEqual(list(again.take_reused()), expected)
        self.assertEqual(again.reused, {})
        self.assertEqual(list(again.take_reused()), [])
        record = again.record()
        self.assertEqual((record.partitions, record.reused, record.read), (4, 2, 0))

    def test_a_state_is_dropped_by_the_record_as_soon_as_it_is_handed_out(self) -> None:
        run = self.open()
        step = run.begin_pass(0, JOB, self.partitions)
        for partition in self.partitions:
            step.commit(self.outcome(partition))
        again = self.open().begin_pass(0, JOB, self.partitions)
        handed = again.take_reused()
        next(handed)
        self.assertEqual(sorted(again.reused), [1, 2, 3])
        next(handed)
        self.assertEqual(sorted(again.reused), [2, 3])

    def test_a_pass_that_is_not_the_stored_one_reuses_nothing(self) -> None:
        run = self.open()
        run.begin_pass(0, JOB, self.partitions).commit(self.outcome(self.partitions[0]))
        later = self.open().begin_pass(1, JOB, self.partitions)
        self.assertEqual(later.reused, {})
        self.assertEqual(later.pending, self.partitions)

    def test_a_fresh_job_does_not_ask_the_store_for_states(self) -> None:
        run = self.open()
        with mock.patch.object(SQLitePartitionStore, "read_pass") as read:
            run.begin_pass(0, JOB, self.partitions)
        read.assert_not_called()

    def test_a_completed_pass_is_replayed_from_the_ledger_without_pending_work(self) -> None:
        run = self.open()
        step = run.begin_pass(0, JOB, self.partitions)
        for partition in self.partitions:
            step.commit(self.outcome(partition))
        step.finish(b"merged state")
        again = self.open().begin_pass(0, JOB, self.partitions)
        self.assertEqual(again.replayed, b"merged state")
        self.assertEqual(again.pending, ())
        self.assertEqual(again.reused, {})

    def test_a_pass_that_the_ledger_holds_with_other_parameters_is_a_mismatch(self) -> None:
        run = self.open()
        run.begin_pass(0, JOB, self.partitions).finish(b"merged")
        other = mock.Mock(spec=FoldJob)
        other.checkpoint_key.return_value = ("fold", "elsewhere", 1)
        self.expect(
            ScaleCheckpointErrorCode.LEDGER_MISMATCH,
            self.open().begin_pass,
            0,
            other,
            self.partitions,
        )

    def test_a_stored_state_of_another_fingerprint_is_a_changed_partition(self) -> None:
        run = self.open()
        step = run.begin_pass(0, JOB, self.partitions)
        self.store.commit_partition(
            0, step._key, self.partitions[0].id, "sha256:" + "9" * 64, "content", b"x"
        )
        with self.assertRaises(ScaleSourceError) as caught:
            self.open().begin_pass(0, JOB, self.partitions)
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)
        self.store.commit_partition(
            0,
            step._key,
            self.partitions[1].id,
            self.partitions[1].fingerprint or "",
            "metadata",
            b"x",
        )
        with self.assertRaises(ScaleSourceError):
            self.open().begin_pass(0, JOB, self.partitions[1:])

    def test_commit_writes_the_state_under_the_key_of_the_pass_and_counts_it(self) -> None:
        run = self.open()
        step = run.begin_pass(2, JOB, self.partitions)
        partition = self.partitions[1]
        outcome = self.outcome(partition)
        step.commit(outcome)
        stored = self.store.read_pass(2, step._key)
        self.assertEqual(list(stored), [partition.id])
        self.assertEqual(stored[partition.id].data, outcome.data)
        self.assertEqual(stored[partition.id].fingerprint, partition.fingerprint)
        self.assertEqual(stored[partition.id].level, "content")
        self.assertEqual(step.record().read, 1)

    def test_finish_commits_the_pass_to_the_ledger_and_prunes_only_for_multi_pass(self) -> None:
        for multi_pass in (False, True):
            with self.subTest(multi_pass=multi_pass):
                self.setUp()
                run = self.open(multi_pass=multi_pass)
                first = run.begin_pass(0, JOB, self.partitions)
                first.commit(self.outcome(self.partitions[0]))
                second = run.begin_pass(1, JOB, self.partitions)
                second.finish(b"m")
                self.assertEqual(bool(self.store.read_pass(0, first._key)), not multi_pass)
                entry = self.store.read_ledger()[1]
                self.assertEqual((entry.pass_key, entry.data), (second._key, b"m"))

    def test_a_replayed_pass_commits_nothing_again(self) -> None:
        run = self.open()
        run.begin_pass(0, JOB, self.partitions).finish(b"merged")
        again = self.open().begin_pass(0, JOB, self.partitions)
        before = self.path.read_bytes()
        again.finish(b"merged")
        self.assertEqual(self.path.read_bytes(), before)

    def test_the_record_of_a_pass_says_what_it_did(self) -> None:
        run = self.open()
        step = run.begin_pass(0, JOB, self.partitions)
        step.commit(self.outcome(self.partitions[0]))
        step.commit(self.outcome(self.partitions[1]))
        step.finish(b"merged")
        record = step.record()
        self.assertEqual(
            (record.pass_index, record.partitions, record.replayed, record.reused, record.read),
            (0, 4, False, 0, 2),
        )
        again = self.open().begin_pass(0, JOB, self.partitions)
        record = again.record()
        self.assertEqual(
            (record.pass_index, record.partitions, record.replayed, record.reused, record.read),
            (0, 4, True, 4, 0),
        )

    def test_the_provenance_lists_the_passes_in_order(self) -> None:
        run = self.open()
        run.begin_pass(0, JOB, self.partitions)
        run.begin_pass(1, JOB, self.partitions)
        record = run.provenance()
        self.assertEqual([item.pass_index for item in record.passes], [0, 1])
        self.assertEqual(replace(record, passes=()).incremental, self.request().incremental)
