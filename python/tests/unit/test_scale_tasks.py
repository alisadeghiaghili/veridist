"""Partition tasks: the job, the retry rule, the canonical merge and the worker initializer."""

from __future__ import annotations

import pickle
import types
import unittest
from dataclasses import FrozenInstanceError
from typing import Any
from unittest import mock

import numpy as np

from tests.unit.scale_fit_oracle import ListSource
from veridist.scale import _tasks
from veridist.scale._errors import (
    ScaleSourceError,
    ScaleSourceErrorCode,
    ScaleStateError,
    ScaleStateErrorCode,
)
from veridist.scale._one_pass import (
    ExponentialState,
    GammaState,
    LognormalState,
    NormalState,
)
from veridist.scale._sources import FingerprintLevel, Partition, fold_partition
from veridist.scale._tasks import (
    MAX_PARTITION_RETRIES,
    FoldJob,
    PartitionTask,
    TaskOutcome,
    initialize_worker,
    is_transient,
    merge_tree,
    run_partition_task,
)

PARTITION = Partition("p3", 3, 4, None, FingerprintLevel.NONE)


def unreadable(cause: BaseException | None) -> ScaleSourceError:
    """A source's ``SOURCE_UNREADABLE``, with ``cause`` as ``raise ... from cause`` gives it."""

    error = ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE)
    error.__cause__ = cause
    return error


class TransientTests(unittest.TestCase):
    def test_an_os_error_is_transient_unless_it_is_missing_or_forbidden(self) -> None:
        for error in (
            OSError("x"),
            ConnectionResetError(),
            TimeoutError(),
            BrokenPipeError(),
            InterruptedError(),
            BlockingIOError(),
            IsADirectoryError(),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertTrue(is_transient(error))
        for error in (FileNotFoundError("x"), PermissionError("x")):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(is_transient(error))

    def test_nothing_else_is_transient(self) -> None:
        for error in (
            ValueError("x"),
            RuntimeError("x"),
            MemoryError(),
            KeyboardInterrupt(),
            ScaleStateError(ScaleStateErrorCode.NON_FINITE_VALUE),
            ScaleSourceError(ScaleSourceErrorCode.NAN_VALUE),
        ):
            with self.subTest(error=type(error).__name__):
                self.assertFalse(is_transient(error))

    def test_an_unreadable_source_is_transient_when_its_cause_is(self) -> None:
        self.assertTrue(is_transient(unreadable(OSError("x"))))
        self.assertTrue(is_transient(unreadable(ConnectionResetError())))
        self.assertFalse(is_transient(unreadable(FileNotFoundError("x"))))
        self.assertFalse(is_transient(unreadable(PermissionError("x"))))
        self.assertFalse(is_transient(unreadable(ValueError("x"))))
        self.assertFalse(is_transient(unreadable(None)))

    def test_only_the_unreadable_code_looks_at_its_cause(self) -> None:
        for code in ScaleSourceErrorCode:
            if code is ScaleSourceErrorCode.SOURCE_UNREADABLE:
                continue
            error = ScaleSourceError(code)
            error.__cause__ = OSError("x")
            with self.subTest(code=code):
                self.assertFalse(is_transient(error))

    def test_the_retry_cap_is_two(self) -> None:
        self.assertEqual(MAX_PARTITION_RETRIES, 2)


class ScriptedJob:
    """A job that raises the scripted errors, in turn, and then returns its payload."""

    def __init__(self, errors: list[BaseException]) -> None:
        self.errors = list(errors)
        self.calls: list[tuple[Any, Partition, int]] = []

    def run(self, source: Any, partition: Partition, max_rows: int) -> bytes:
        self.calls.append((source, partition, max_rows))
        if self.errors:
            raise self.errors.pop(0)
        return b"payload"

    def restore(self, data: bytes) -> Any:
        raise AssertionError("not used")

    def identity(self) -> Any:
        raise AssertionError("not used")


def task(job: ScriptedJob, retries: int, source: Any = "source") -> PartitionTask:
    return PartitionTask(source, PARTITION, 17, job, retries)


class RunTaskTests(unittest.TestCase):
    def test_a_task_reads_its_partition_once_and_reports_the_outcome(self) -> None:
        job = ScriptedJob([])
        outcome = run_partition_task(task(job, 2, "the source"))
        self.assertEqual(outcome, TaskOutcome(3, b"payload", 0))
        self.assertEqual(job.calls, [("the source", PARTITION, 17)])

    def test_a_transient_error_is_read_again_and_counted(self) -> None:
        for failures in (1, 2):
            job = ScriptedJob([OSError("x")] * failures)
            outcome = run_partition_task(task(job, 2))
            with self.subTest(failures=failures):
                self.assertEqual(outcome, TaskOutcome(3, b"payload", failures))
                self.assertEqual(len(job.calls), failures + 1)

    def test_a_task_that_keeps_failing_gives_up_after_its_retries(self) -> None:
        for retries in (0, 1, 2, 3):
            errors = [OSError(f"e{index}") for index in range(retries + 3)]
            job = ScriptedJob(errors)
            with self.subTest(retries=retries):
                with self.assertRaises(OSError) as caught:
                    run_partition_task(task(job, retries))
                self.assertIs(caught.exception, errors[retries])
                self.assertEqual(len(job.calls), retries + 1)

    def test_the_last_read_may_be_the_one_that_succeeds(self) -> None:
        job = ScriptedJob([OSError("x")] * 3)
        outcome = run_partition_task(task(job, 3))
        self.assertEqual(outcome, TaskOutcome(3, b"payload", 3))
        self.assertEqual(len(job.calls), 4)

    def test_a_final_error_is_raised_at_once_as_it_is(self) -> None:
        for error in (
            FileNotFoundError("x"),
            PermissionError("x"),
            ValueError("x"),
            ScaleStateError(ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT),
            ScaleSourceError(ScaleSourceErrorCode.PARTITION_CHANGED),
            unreadable(FileNotFoundError("x")),
        ):
            job = ScriptedJob([error])
            with self.subTest(error=repr(error)):
                with self.assertRaises(type(error)) as caught:
                    run_partition_task(task(job, 5))
                self.assertIs(caught.exception, error)
                self.assertEqual(len(job.calls), 1)

    def test_a_final_error_after_a_transient_one_is_raised(self) -> None:
        final = ScaleStateError(ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)
        job = ScriptedJob([OSError("x"), final])
        with self.assertRaises(ScaleStateError) as caught:
            run_partition_task(task(job, 5))
        self.assertIs(caught.exception, final)
        self.assertEqual(len(job.calls), 2)

    def test_an_unreadable_source_with_a_transient_cause_is_read_again(self) -> None:
        job = ScriptedJob([unreadable(ConnectionResetError())])
        self.assertEqual(run_partition_task(task(job, 1)), TaskOutcome(3, b"payload", 1))

    def test_an_exception_that_is_not_an_exception_is_not_retried(self) -> None:
        job = ScriptedJob([KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            run_partition_task(task(job, 5))
        self.assertEqual(len(job.calls), 1)

    def test_a_task_has_no_retries_unless_it_is_given_some(self) -> None:
        self.assertEqual(PartitionTask("s", PARTITION, 1, ScriptedJob([])).retries, 0)

    def test_a_task_and_its_outcome_are_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            task(ScriptedJob([]), 1).retries = 3  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            TaskOutcome(0, b"", 0).data = b"x"  # type: ignore[misc]


STATES = (ExponentialState, NormalState, GammaState, LognormalState)


class FoldJobTests(unittest.TestCase):
    def source(self) -> ListSource:
        rng = np.random.default_rng(5)
        return ListSource([rng.gamma(3.0, 2.0, 40), rng.gamma(3.0, 2.0, 25)])

    def test_the_bytes_are_those_of_the_partition_state(self) -> None:
        source = self.source()
        partition = source.partitions()[1]
        for state_type in STATES:
            job = FoldJob(state_type)
            with self.subTest(state=state_type.__name__):
                expected = fold_partition(source, partition, state_type, max_rows=7).to_bytes()
                self.assertEqual(job.run(source, partition, 7), expected)

    def test_the_job_reads_the_batches_it_is_told_to(self) -> None:
        source = self.source()
        FoldJob(ExponentialState).run(source, source.partitions()[0], 16)
        self.assertEqual(source.batch_sizes, [16, 16, 8])

    def test_restore_and_identity(self) -> None:
        source = self.source()
        partition = source.partitions()[0]
        for state_type in STATES:
            job = FoldJob(state_type)
            data = job.run(source, partition, 100)
            with self.subTest(state=state_type.__name__):
                restored = job.restore(data)
                self.assertIs(type(restored), state_type)
                self.assertEqual(restored.to_bytes(), data)
                self.assertEqual(restored.count, 40)
                self.assertEqual(job.identity(), state_type.empty())
                self.assertEqual(restored.merge(job.identity()), restored)

    def test_a_job_pickles(self) -> None:
        job = FoldJob(NormalState)
        self.assertEqual(pickle.loads(pickle.dumps(job)), job)

    def test_a_task_pickles_with_its_job_and_partition(self) -> None:
        original = PartitionTask(None, PARTITION, 9, FoldJob(GammaState), 2)  # type: ignore[arg-type]
        self.assertEqual(pickle.loads(pickle.dumps(original)), original)
        outcome = TaskOutcome(4, b"abc", 1)
        self.assertEqual(pickle.loads(pickle.dumps(outcome)), outcome)


class MergeTreeTests(unittest.TestCase):
    @staticmethod
    def join(left: Any, right: Any) -> Any:
        return (left, right)

    def shape(self, count: int) -> Any:
        return merge_tree(list(range(count)), self.join, "identity")

    def test_no_items_give_the_identity_and_one_item_is_not_merged(self) -> None:
        self.assertEqual(self.shape(0), "identity")
        self.assertEqual(self.shape(1), 0)

    def test_neighbours_merge_level_by_level_with_the_lower_ordinal_first(self) -> None:
        self.assertEqual(self.shape(2), (0, 1))
        self.assertEqual(self.shape(3), ((0, 1), 2))
        self.assertEqual(self.shape(4), ((0, 1), (2, 3)))
        self.assertEqual(self.shape(5), (((0, 1), (2, 3)), 4))
        self.assertEqual(self.shape(6), (((0, 1), (2, 3)), (4, 5)))
        self.assertEqual(self.shape(7), (((0, 1), (2, 3)), ((4, 5), 6)))
        self.assertEqual(self.shape(8), (((0, 1), (2, 3)), ((4, 5), (6, 7))))

    def test_the_items_are_not_modified(self) -> None:
        items = [1, 2, 3]
        merge_tree(items, lambda left, right: left + right, 0)
        self.assertEqual(items, [1, 2, 3])

    def test_every_item_is_merged_exactly_once(self) -> None:
        for count in range(1, 20):
            merged = merge_tree([1] * count, lambda left, right: left + right, 0)
            with self.subTest(count=count):
                self.assertEqual(merged, count)

    def test_the_merge_is_called_once_per_pair(self) -> None:
        calls = mock.Mock(side_effect=lambda left, right: left + right)
        merge_tree([1, 2, 3, 4, 5], calls, 0)
        self.assertEqual(calls.call_count, 4)

    def test_the_identity_is_not_merged_into_anything(self) -> None:
        calls = mock.Mock(side_effect=lambda left, right: left + right)
        self.assertEqual(merge_tree([5], calls, 100), 5)
        calls.assert_not_called()

    def test_merging_exact_states_gives_the_state_of_all_the_rows(self) -> None:
        rng = np.random.default_rng(2)
        parts = [rng.exponential(2.0, size) for size in (3, 9, 1, 5, 7)]
        states = [ExponentialState.empty().update(part) for part in parts]
        merged = merge_tree(states, lambda left, right: left.merge(right), ExponentialState.empty())
        whole = ExponentialState.empty().update(np.concatenate(parts))
        self.assertEqual(merged.to_bytes(), whole.to_bytes())


class InitializeWorkerTests(unittest.TestCase):
    def test_every_module_is_imported_in_order(self) -> None:
        loaded: list[str] = []

        def fake(name: str) -> types.ModuleType:
            loaded.append(name)
            return types.ModuleType(name)

        with mock.patch.object(_tasks, "import_module", fake):
            initialize_worker(("numpy", "other.module", "third"))
        self.assertEqual(loaded, ["numpy", "other.module", "third"])

    def test_arrow_is_pinned_to_one_thread_and_nothing_else_is_touched(self) -> None:
        arrow = types.ModuleType("pyarrow")
        calls: list[tuple[str, int]] = []
        arrow.set_cpu_count = lambda count: calls.append(("cpu", count))  # type: ignore[attr-defined]
        arrow.set_io_thread_count = lambda count: calls.append(("io", count))  # type: ignore[attr-defined]
        other = types.ModuleType("pyarrow.parquet")
        other.set_cpu_count = lambda count: calls.append(("wrong", count))  # type: ignore[attr-defined]
        modules = {"pyarrow": arrow, "pyarrow.parquet": other}
        with mock.patch.object(_tasks, "import_module", modules.__getitem__):
            initialize_worker(("pyarrow", "pyarrow.parquet"))
        self.assertEqual(calls, [("cpu", 1), ("io", 1)])

    def test_the_real_modules_import(self) -> None:
        initialize_worker(("numpy",))
        initialize_worker(())
