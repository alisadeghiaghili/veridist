"""Executors: the request, how a pass is read, merging, failures, retries and the pool's life.

Most of these run in this process: the pool is a fake that runs tasks when the test says so,
or a real thread pool. Real worker processes are in ``test_scale_executor_processes``.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import os
import threading
import unittest
from dataclasses import FrozenInstanceError
from typing import Any
from unittest import mock

import numpy as np

from tests.unit.scale_executor_fixtures import (
    FakePool,
    NoLeaksTestCase,
    PartSource,
    PlainSource,
    PoolRecorder,
    Scripted,
    ScriptedPool,
    cores,
)
from veridist.engine.provenance import ScaleExecutionPassProvenance
from veridist.scale import _executors
from veridist.scale._arrow import ArrowSource
from veridist.scale._errors import (
    ScaleSourceError,
    ScaleSourceErrorCode,
    ScaleStateError,
    ScaleStateErrorCode,
)
from veridist.scale._executors import (
    IN_FLIGHT_PER_WORKER,
    MAX_WORKERS,
    START_METHOD,
    WAIT_SLICE_SECONDS,
    PassEngine,
    ProcessExecutor,
    SerialExecutor,
    ThreadExecutor,
    available_cpus,
    make_process_pool,
    make_thread_pool,
    resolve_executor,
)
from veridist.scale._one_pass import ExponentialState
from veridist.scale._sources import DowngradeReason, ExecutorKind, FingerprintLevel, Partition
from veridist.scale._tasks import FoldJob, initialize_worker

JOB = FoldJob(ExponentialState)


def parts(count: int, size: int = 6) -> list[np.ndarray]:
    rng = np.random.default_rng(count * 100 + size)
    return [rng.exponential(2.0, size) + 0.5 for _ in range(count)]


class Node:
    """A mergeable that records which partitions were merged into it, each as often as it was."""

    def __init__(self, members: tuple[int, ...]) -> None:
        self.members = members

    def merge(self, other: Node) -> Node:
        return Node(tuple(sorted(self.members + other.members)))


class ShapeJob:
    """A job whose value is the sorted ordinals of the partitions merged into it."""

    def run(self, source: Any, partition: Partition, max_rows: int) -> bytes:
        return str(partition.ordinal).encode()

    def restore(self, data: bytes) -> Node:
        return Node((int(data),))

    def identity(self) -> Node:
        return Node(())


class FailingJob(ShapeJob):
    """A job that raises, for the listed ordinals, what it is told to."""

    def __init__(self, errors: dict[int, BaseException]) -> None:
        self.errors = errors
        self.ran: list[int] = []
        self._lock = threading.Lock()

    def run(self, source: Any, partition: Partition, max_rows: int) -> bytes:
        with self._lock:
            self.ran.append(partition.ordinal)
        error = self.errors.get(partition.ordinal)
        if error is not None:
            raise error
        return super().run(source, partition, max_rows)


def state_bytes(arrays: list[np.ndarray]) -> bytes:
    return ExponentialState.empty().update(np.concatenate(arrays)).to_bytes()


class ExecutorObjectTests(unittest.TestCase):
    def test_the_constants_are_pinned(self) -> None:
        self.assertEqual(MAX_WORKERS, 64)
        self.assertEqual(WAIT_SLICE_SECONDS, 0.5)
        self.assertEqual(START_METHOD, "spawn")
        self.assertEqual(_executors.THREAD_NAME_PREFIX, "veridist-scale")
        self.assertEqual(_executors.NUMPY_MODULE, "numpy")
        self.assertEqual(_executors.SERIAL_NAME, "serial")
        self.assertEqual(IN_FLIGHT_PER_WORKER, 2)

    def test_the_kinds_and_worker_counts(self) -> None:
        self.assertIs(SerialExecutor().kind, ExecutorKind.SERIAL)
        self.assertEqual(SerialExecutor().workers, 1)
        self.assertIs(ThreadExecutor(3).kind, ExecutorKind.THREADS)
        self.assertEqual(ThreadExecutor(workers=3).workers, 3)
        self.assertIs(ProcessExecutor(5).kind, ExecutorKind.PROCESSES)
        self.assertEqual(ProcessExecutor(workers=5).workers, 5)

    def test_workers_must_be_a_positive_built_in_integer(self) -> None:
        for make in (ThreadExecutor, ProcessExecutor):
            for bad in (0, -1):
                with self.subTest(make=make.__name__, workers=bad):
                    with self.assertRaisesRegex(ValueError, "^workers must be at least 1$"):
                        make(bad)
            for wrong in (True, 2.0, "2", None):
                with self.subTest(make=make.__name__, workers=wrong):
                    with self.assertRaisesRegex(TypeError, "^workers must be a built-in integer$"):
                        make(wrong)  # type: ignore[arg-type]
            self.assertEqual(make(1).workers, 1)

    def test_a_pool_factory_is_callable_or_absent(self) -> None:
        for make in (ThreadExecutor, ProcessExecutor):
            with self.subTest(make=make.__name__):
                with self.assertRaisesRegex(TypeError, "^pool_factory must be callable or None$"):
                    make(2, pool_factory=3)  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    make(2, PoolRecorder())  # type: ignore[misc]
                self.assertEqual(make(2), make(2, pool_factory=PoolRecorder()))
                self.assertEqual(
                    repr(make(2, pool_factory=PoolRecorder())), f"{make.__name__}(workers=2)"
                )
                self.assertIsNone(make(2).pool_factory)
                recorder = PoolRecorder()
                self.assertIs(make(2, pool_factory=recorder).pool_factory, recorder)

    def test_a_pool_factory_is_keyword_only_and_not_part_of_the_value(self) -> None:
        with self.assertRaises(TypeError):
            ThreadExecutor(2, PoolRecorder())  # type: ignore[misc]
        self.assertEqual(ThreadExecutor(2), ThreadExecutor(2, pool_factory=PoolRecorder()))
        self.assertNotEqual(ThreadExecutor(2), ThreadExecutor(3))
        self.assertNotEqual(ThreadExecutor(2), ProcessExecutor(2))
        self.assertEqual(
            repr(ThreadExecutor(2, pool_factory=PoolRecorder())), "ThreadExecutor(workers=2)"
        )
        self.assertEqual(repr(ProcessExecutor(4)), "ProcessExecutor(workers=4)")
        self.assertEqual(SerialExecutor(), SerialExecutor())

    def test_executors_are_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            ThreadExecutor(2).workers = 3  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            ProcessExecutor(2).workers = 3  # type: ignore[misc]

    def test_the_executor_argument_is_resolved(self) -> None:
        self.assertIsNone(resolve_executor(None))
        self.assertEqual(resolve_executor("serial"), SerialExecutor())
        for executor in (SerialExecutor(), ThreadExecutor(2), ProcessExecutor(3)):
            self.assertIs(resolve_executor(executor), executor)

    def test_a_bad_executor_argument_is_refused(self) -> None:
        for name in ("threads", "processes", "Serial", ""):
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError, "^executor must be 'serial' or an executor object$"
                ):
                    resolve_executor(name)
        for wrong in (3, 2.0, b"serial", object(), ThreadExecutor, True):
            with self.subTest(wrong=repr(wrong)):
                with self.assertRaisesRegex(
                    TypeError, "^executor must be None, 'serial' or an executor$"
                ):
                    resolve_executor(wrong)

    def test_a_string_subclass_is_not_the_name(self) -> None:
        class Named(str):
            pass

        with self.assertRaises(TypeError):
            resolve_executor(Named("serial"))

    def test_the_core_count_is_at_least_one(self) -> None:
        with mock.patch.object(os, "cpu_count", return_value=None):
            self.assertEqual(available_cpus(), 1)
        with mock.patch.object(os, "cpu_count", return_value=12):
            self.assertEqual(available_cpus(), 12)
        self.assertGreaterEqual(available_cpus(), 1)


class PoolMakerTests(NoLeaksTestCase):
    def test_a_thread_pool_has_the_workers_and_the_name(self) -> None:
        with mock.patch.object(_executors, "ThreadPoolExecutor") as maker:
            pool = make_thread_pool(3, ("ignored",))
        maker.assert_called_once_with(max_workers=3, thread_name_prefix="veridist-scale")
        self.assertIs(pool, maker.return_value)

    def test_a_process_pool_spawns_workers_that_import_their_modules(self) -> None:
        with (
            mock.patch.object(_executors, "ProcessPoolExecutor") as maker,
            mock.patch.object(_executors, "get_context") as context,
        ):
            pool = make_process_pool(4, ("numpy", "pyarrow"))
        context.assert_called_once_with("spawn")
        maker.assert_called_once_with(
            max_workers=4,
            mp_context=context.return_value,
            initializer=initialize_worker,
            initargs=(("numpy", "pyarrow"),),
        )
        self.assertIs(pool, maker.return_value)

    def test_a_real_thread_pool_runs_tasks_and_shuts_down(self) -> None:
        pool = make_thread_pool(2, ())
        try:
            self.assertEqual(pool.submit(sum, [1, 2, 3]).result(timeout=10), 6)
            names = {thread.name for thread in threading.enumerate()}
            self.assertTrue(any(name.startswith("veridist-scale") for name in names))
        finally:
            pool.shutdown(wait=True)

    def test_the_default_start_method_exists_on_every_platform(self) -> None:
        import multiprocessing

        self.assertIn(START_METHOD, multiprocessing.get_all_start_methods())


class ModeTests(NoLeaksTestCase):
    def run_pass(self, executor: Any, source: Any) -> tuple[PassEngine, Any]:
        engine = PassEngine(executor, 100)
        with engine:
            merged = engine.run(source, source.partitions(), JOB)
        return engine, merged

    def last(self, engine: PassEngine) -> ScaleExecutionPassProvenance:
        provenance = engine.provenance()
        assert provenance is not None
        return provenance.passes[-1]

    def test_the_serial_executor_reads_in_turn_and_asks_nothing(self) -> None:
        source = PartSource(parts(4))
        engine, merged = self.run_pass(SerialExecutor(), source)
        self.assertEqual(self.last(engine), ScaleExecutionPassProvenance("serial", 1, 4, 0, None))
        self.assertEqual(source.asked, [])
        self.assertEqual(source.reads, [0, 1, 2, 3])
        self.assertEqual(merged.to_bytes(), state_bytes(source.parts))

    def test_no_executor_reads_in_turn_and_records_nothing(self) -> None:
        source = PartSource(parts(3))
        engine, merged = self.run_pass(None, source)
        self.assertIsNone(engine.provenance())
        self.assertEqual(source.asked, [])
        self.assertEqual(merged.to_bytes(), state_bytes(source.parts))

    def test_threads_and_processes_ask_the_source_about_their_own_kind(self) -> None:
        for make, kind in (
            (ThreadExecutor, ExecutorKind.THREADS),
            (ProcessExecutor, ExecutorKind.PROCESSES),
        ):
            source = PartSource(parts(5))
            recorder = PoolRecorder()
            with self.subTest(kind=kind), cores(16):
                engine, merged = self.run_pass(make(4, pool_factory=recorder), source)
                self.assertEqual(source.asked, [(kind, 5)])
                self.assertEqual(
                    self.last(engine), ScaleExecutionPassProvenance(kind.value, 4, 5, 0, None)
                )
                self.assertEqual(merged.to_bytes(), state_bytes(source.parts))

    def test_the_workers_are_the_request_capped_by_cores_partitions_and_the_constant(self) -> None:
        cases = (
            (4, 16, 5, 4),
            (100, 16, 50, 16),
            (100, 2, 50, 2),
            (100, 1000, 7, 7),
            (1000, 1000, 100, 64),
            (64, 1000, 100, 64),
            (65, 1000, 100, 64),
            (1, 16, 5, 1),
            (3, 3, 3, 3),
        )
        for requested, cpus, count, expected in cases:
            source = PartSource(parts(count, 2))
            recorder = PoolRecorder()
            with self.subTest(requested=requested, cpus=cpus, partitions=count), cores(cpus):
                engine, _ = self.run_pass(ThreadExecutor(requested, pool_factory=recorder), source)
                self.assertEqual(self.last(engine).workers, expected)
                self.assertEqual(recorder.calls[0][0], expected)
                self.assertEqual(engine.provenance().requested_workers, requested)  # type: ignore[union-attr]

    def test_a_downgraded_pass_is_serial_with_its_reason_and_starts_no_pool(self) -> None:
        for reason in DowngradeReason:
            source = PartSource(parts(4), obstacle=reason)
            recorder = PoolRecorder()
            with self.subTest(reason=reason), cores(16):
                engine, merged = self.run_pass(ThreadExecutor(4, pool_factory=recorder), source)
                self.assertEqual(
                    self.last(engine),
                    ScaleExecutionPassProvenance("serial", 1, 4, 0, reason.value),
                )
                self.assertEqual(recorder.calls, [])
                self.assertEqual(source.reads, [0, 1, 2, 3])
                self.assertEqual(merged.to_bytes(), state_bytes(source.parts))

    def test_one_partition_is_a_downgrade_and_the_source_is_asked_first(self) -> None:
        source = PartSource(parts(1))
        recorder = PoolRecorder()
        with cores(16):
            engine, _ = self.run_pass(ProcessExecutor(4, pool_factory=recorder), source)
        self.assertEqual(
            self.last(engine), ScaleExecutionPassProvenance("serial", 1, 1, 0, "single_partition")
        )
        self.assertEqual(source.asked, [(ExecutorKind.PROCESSES, 1)])
        self.assertEqual(recorder.calls, [])

    def test_the_sources_own_reason_wins_over_a_single_partition(self) -> None:
        source = PartSource(parts(1), obstacle=DowngradeReason.ENGINE_MANAGED)
        with cores(16):
            engine, _ = self.run_pass(ThreadExecutor(2, pool_factory=PoolRecorder()), source)
        self.assertEqual(self.last(engine).downgrade, "engine_managed")

    def test_a_source_that_says_nothing_is_not_read_in_parallel(self) -> None:
        source = PlainSource(parts(4))
        recorder = PoolRecorder()
        with cores(16):
            engine, merged = self.run_pass(ThreadExecutor(4, pool_factory=recorder), source)
        self.assertEqual(
            self.last(engine), ScaleExecutionPassProvenance("serial", 1, 4, 0, "undeclared_source")
        )
        self.assertEqual(recorder.calls, [])
        self.assertEqual(merged.to_bytes(), state_bytes(source.parts))

    def test_no_partitions_is_one_serial_pass_over_nothing(self) -> None:
        source = PartSource([])
        recorder = PoolRecorder()
        with cores(16):
            engine, merged = self.run_pass(ThreadExecutor(4, pool_factory=recorder), source)
        self.assertEqual(
            self.last(engine), ScaleExecutionPassProvenance("serial", 1, 0, 0, "single_partition")
        )
        self.assertEqual(merged, ExponentialState.empty())

    def test_a_serial_request_is_never_a_downgrade(self) -> None:
        source = PartSource(parts(1), obstacle=DowngradeReason.ENGINE_MANAGED)
        engine, _ = self.run_pass(SerialExecutor(), source)
        self.assertEqual(self.last(engine).downgrade, None)

    def test_every_pass_is_recorded_in_order(self) -> None:
        source = PartSource(parts(3))
        recorder = PoolRecorder()
        engine = PassEngine(ThreadExecutor(2, pool_factory=recorder), 100)
        with engine, cores(16):
            engine.run(source, source.partitions(), JOB)
            source.obstacle = DowngradeReason.SPOOL_FIRST_PASS
            engine.run(source, source.partitions(), JOB)
            source.obstacle = None
            engine.run(source, source.partitions(), JOB)
        provenance = engine.provenance()
        assert provenance is not None
        self.assertEqual(
            provenance.passes,
            (
                ScaleExecutionPassProvenance("threads", 2, 3, 0, None),
                ScaleExecutionPassProvenance("serial", 1, 3, 0, "spool_first_pass"),
                ScaleExecutionPassProvenance("threads", 2, 3, 0, None),
            ),
        )
        self.assertEqual(provenance.effective_workers, 2)
        self.assertEqual(provenance.downgrade, "spool_first_pass")
        self.assertEqual(len(recorder.calls), 1)


class MergeOrderTests(NoLeaksTestCase):
    def run_members(
        self, count: int, script: list[list[int]]
    ) -> tuple[tuple[int, ...], ScriptedPool]:
        source = PartSource(parts(count, 1))
        scripted = Scripted(count, script)
        with scripted as engine:
            merged = engine.run(source, source.partitions(), ShapeJob())
        return merged.members, scripted.pool

    def test_every_partition_is_merged_exactly_once_whatever_the_arrival_order(self) -> None:
        expected = (0, 1, 2, 3, 4)
        orders = (
            [[0], [1], [2], [3], [4]],
            [[4], [3], [2], [1], [0]],
            [[2], [4], [0], [3], [1]],
            [[1, 3], [0, 4, 2]],
        )
        for order in orders:
            with self.subTest(order=order):
                members, pool = self.run_members(5, order)
                self.assertEqual(members, expected)
                self.assertEqual(sorted(pool.ran), [0, 1, 2, 3, 4])

    def test_results_that_arrive_late_in_a_real_pool_are_merged_the_same(self) -> None:
        gates = [threading.Event() for _ in range(4)]

        class Reversed(ShapeJob):
            def run(self, source: Any, partition: Partition, max_rows: int) -> bytes:
                ordinal = partition.ordinal
                if ordinal + 1 < len(gates) and not gates[ordinal + 1].wait(10.0):
                    raise AssertionError("the next partition did not finish")
                gates[ordinal].set()
                return super().run(source, partition, max_rows)

        source = PartSource(parts(4, 1))
        with cores(16), PassEngine(ThreadExecutor(4), 10) as engine:
            merged = engine.run(source, source.partitions(), Reversed())
        self.assertEqual(merged.members, (0, 1, 2, 3))


class FailureTests(NoLeaksTestCase):
    def run_failing(
        self, errors: dict[int, BaseException], script: list[list[int]], count: int = 4
    ) -> tuple[BaseException, ScriptedPool, FailingJob]:
        source = PartSource(parts(count, 1))
        job = FailingJob(errors)
        scripted = Scripted(count, script)
        with self.assertRaises(BaseException) as caught, scripted as engine:
            engine.run(source, source.partitions(), job)
        return caught.exception, scripted.pool, job

    def test_the_failure_first_in_canonical_order_is_raised(self) -> None:
        late = ScaleStateError(ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)
        early = ScaleSourceError(ScaleSourceErrorCode.NAN_VALUE)
        # Partition 3 fails first, then partition 1; 0 is read last and is fine.
        error, pool, job = self.run_failing({1: early, 3: late}, [[3], [1], [0]])
        self.assertIs(error, early)
        self.assertEqual(pool.ran, [3, 1, 0])

    def test_partitions_before_the_failure_are_waited_for_and_after_it_cancelled(self) -> None:
        first = ScaleSourceError(ScaleSourceErrorCode.PARTITION_CHANGED)
        second = ScaleStateError(ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)
        error, pool, job = self.run_failing({0: first, 1: second}, [[1], [0]])
        self.assertIs(error, first)
        self.assertEqual(pool.ran, [1, 0])
        self.assertEqual(
            [future.cancelled() for future in pool.futures], [False, False, True, True]
        )

    def test_a_failure_does_not_wait_for_partitions_after_it(self) -> None:
        only = ScaleStateError(ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)
        error, pool, job = self.run_failing({1: only}, [[0], [1]])
        self.assertIs(error, only)
        self.assertEqual(pool.ran, [0, 1])
        self.assertEqual(
            [future.cancelled() for future in pool.futures], [False, False, True, True]
        )
        self.assertEqual(job.ran, [0, 1])

    def test_a_failure_in_the_last_partition_still_waits_for_all_the_others(self) -> None:
        only = ScaleSourceError(ScaleSourceErrorCode.NAN_VALUE)
        error, pool, job = self.run_failing({3: only}, [[3], [0], [1], [2]])
        self.assertIs(error, only)
        self.assertEqual(pool.ran, [3, 0, 1, 2])

    def test_a_later_success_does_not_hide_an_earlier_failure(self) -> None:
        only = ScaleSourceError(ScaleSourceErrorCode.NAN_VALUE)
        error, pool, job = self.run_failing({0: only}, [[1, 2, 3, 0]])
        self.assertIs(error, only)

    def test_an_interrupt_in_a_task_ends_the_pass_at_once(self) -> None:
        error, pool, job = self.run_failing({1: KeyboardInterrupt()}, [[1]])
        self.assertIsInstance(error, KeyboardInterrupt)
        self.assertEqual(pool.ran, [1])
        self.assertEqual([future.cancelled() for future in pool.futures], [True, False, True, True])
        self.assertEqual(pool.shutdowns, [(True, True)])

    def test_an_interrupt_beats_an_exception_that_failed_with_it(self) -> None:
        typed = ScaleSourceError(ScaleSourceErrorCode.NAN_VALUE)
        error, pool, job = self.run_failing({0: typed, 1: SystemExit(3)}, [[0, 1]])
        self.assertIsInstance(error, SystemExit)

    def test_an_interrupt_of_the_parent_cancels_the_tasks_and_shuts_the_pool_down(self) -> None:
        source = PartSource(parts(4, 1))
        holder: list[ScriptedPool] = []

        def make(workers: int, modules: tuple[str, ...]) -> ScriptedPool:
            pool = ScriptedPool(workers, modules, [[0]])
            holder.append(pool)
            return pool

        calls: list[Any] = []

        def interrupted(pending: Any, timeout: Any = None, return_when: Any = None) -> Any:
            calls.append((timeout, return_when))
            if len(calls) == 1:
                holder[0].run(0)
                return concurrent.futures.wait(pending, timeout=0)
            raise KeyboardInterrupt

        engine = PassEngine(ThreadExecutor(4, pool_factory=PoolRecorder(make)), 10)
        with cores(64), mock.patch.object(_executors, "wait", interrupted):
            with self.assertRaises(KeyboardInterrupt), engine:
                engine.run(source, source.partitions(), ShapeJob())
        pool = holder[0]
        self.assertEqual(pool.shutdowns, [(True, True)])
        self.assertEqual([future.cancelled() for future in pool.futures], [False, True, True, True])
        provenance = engine.provenance()
        assert provenance is not None
        self.assertEqual(len(provenance.passes), 1)

    def test_the_wait_is_a_short_slice_for_the_first_to_finish(self) -> None:
        seen: list[tuple[Any, Any]] = []
        real = concurrent.futures.wait

        def recording(pending: Any, timeout: Any = None, return_when: Any = None) -> Any:
            seen.append((timeout, return_when))
            return real(pending, timeout=timeout, return_when=return_when)

        source = PartSource(parts(4, 3))
        with cores(16), mock.patch.object(_executors, "wait", recording):
            with PassEngine(ThreadExecutor(2), 10) as engine:
                engine.run(source, source.partitions(), JOB)
        self.assertTrue(seen)
        self.assertEqual(set(seen), {(WAIT_SLICE_SECONDS, concurrent.futures.FIRST_COMPLETED)})

    def test_a_pass_that_waits_longer_than_a_slice_goes_on_waiting(self) -> None:
        gate = threading.Event()

        class Slow(ShapeJob):
            def run(self, source: Any, partition: Partition, max_rows: int) -> bytes:
                if partition.ordinal == 0 and not gate.wait(10.0):
                    raise AssertionError("gate")
                return super().run(source, partition, max_rows)

        waits: list[int] = []
        real = concurrent.futures.wait

        def counting(pending: Any, timeout: Any = None, return_when: Any = None) -> Any:
            waits.append(1)
            if len(waits) == 3:
                gate.set()
            return real(pending, timeout=0.01, return_when=return_when)

        source = PartSource(parts(2, 1))
        with cores(16), mock.patch.object(_executors, "wait", counting):
            with PassEngine(ThreadExecutor(2), 10) as engine:
                merged = engine.run(source, source.partitions(), Slow())
        self.assertEqual(merged.members, (0, 1))
        self.assertGreaterEqual(len(waits), 3)

    def test_a_real_thread_pool_raises_the_error_the_task_raised_and_stops(self) -> None:
        typed = ScaleStateError(ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)
        source = PartSource(parts(6, 1))
        job = FailingJob({2: typed})
        with cores(16), self.assertRaises(ScaleStateError) as caught:
            with PassEngine(ThreadExecutor(3), 10) as engine:
                engine.run(source, source.partitions(), job)
        self.assertIs(caught.exception, typed)
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)

    def test_a_real_thread_pool_ends_on_an_interrupt_in_a_worker(self) -> None:
        source = PartSource(parts(6, 1))
        job = FailingJob({1: KeyboardInterrupt()})
        with cores(16), self.assertRaises(KeyboardInterrupt):
            with PassEngine(ThreadExecutor(3), 10) as engine:
                engine.run(source, source.partitions(), job)


class PoolLifeTests(NoLeaksTestCase):
    def test_one_pool_serves_every_pass_and_is_shut_down_once_at_the_end(self) -> None:
        source = PartSource(parts(4))
        recorder = PoolRecorder()
        engine = PassEngine(ThreadExecutor(3, pool_factory=recorder), 10)
        with cores(16):
            with engine:
                for _ in range(3):
                    engine.run(source, source.partitions(), JOB)
                self.assertEqual(recorder.pools[0].shutdowns, [])
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(recorder.pools[0].shutdowns, [(True, True)])
        self.assertEqual(len(recorder.pools[0].tasks), 12)

    def test_the_pool_is_shut_down_when_the_fit_fails(self) -> None:
        source = PartSource(parts(4))
        recorder = PoolRecorder()
        with cores(16), self.assertRaises(RuntimeError):
            with PassEngine(ThreadExecutor(3, pool_factory=recorder), 10) as engine:
                engine.run(source, source.partitions(), JOB)
                raise RuntimeError("after the pass")
        self.assertEqual(recorder.pools[0].shutdowns, [(True, True)])

    def test_leaving_the_engine_does_not_hide_the_error(self) -> None:
        engine = PassEngine(None, 10)
        self.assertIsNone(engine.__exit__(None, None, None))
        with self.assertRaises(KeyError):
            with engine:
                raise KeyError("x")

    def test_the_pool_is_made_by_the_first_pass_that_runs_in_parallel(self) -> None:
        source = PartSource(parts(3), obstacle=DowngradeReason.SPOOL_FIRST_PASS)
        recorder = PoolRecorder()
        with cores(16), PassEngine(ThreadExecutor(2, pool_factory=recorder), 10) as engine:
            engine.run(source, source.partitions(), JOB)
            self.assertEqual(recorder.calls, [])
            source.obstacle = None
            engine.run(source, source.partitions(), JOB)
            self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(len(recorder.pools), 1)

    def test_an_engine_that_never_ran_in_parallel_has_no_pool_to_shut_down(self) -> None:
        recorder = PoolRecorder()
        with PassEngine(ThreadExecutor(2, pool_factory=recorder), 10):
            pass
        self.assertEqual(recorder.calls, [])

    def test_the_default_pool_of_each_kind(self) -> None:
        source = PartSource(parts(4))
        for make, factory, other in (
            (ThreadExecutor, "make_thread_pool", "make_process_pool"),
            (ProcessExecutor, "make_process_pool", "make_thread_pool"),
        ):
            fake = FakePool()
            with (
                self.subTest(kind=make.__name__),
                cores(16),
                mock.patch.object(_executors, factory, return_value=fake) as patched,
                mock.patch.object(
                    _executors, other, side_effect=AssertionError("the other kind of pool")
                ) as unwanted,
            ):
                with PassEngine(make(2), 10) as engine:
                    engine.run(source, source.partitions(), JOB)
                patched.assert_called_once_with(2, ("numpy",))
                unwanted.assert_not_called()
                self.assertEqual(fake.shutdowns, [(True, True)])

    def test_the_other_default_pool_is_not_made(self) -> None:
        source = PartSource(parts(4))
        with (
            cores(16),
            mock.patch.object(
                _executors, "make_process_pool", side_effect=AssertionError("a process pool")
            ) as processes,
            mock.patch.object(_executors, "make_thread_pool", return_value=FakePool()),
        ):
            with PassEngine(ThreadExecutor(2), 10) as engine:
                engine.run(source, source.partitions(), JOB)
        processes.assert_not_called()


class Marker:
    """What a source hands to worker processes."""

    WORKER_MODULES = ("first.module", "second.module")

    def __init__(self, inner: PartSource) -> None:
        self.inner = inner

    def batches(self, partition: Partition, *, max_rows: int = 100) -> Any:
        return self.inner.batches(partition, max_rows=max_rows)


class MarkedSource(PartSource):
    def __init__(self, arrays: list[np.ndarray]) -> None:
        super().__init__(arrays)
        self.marker = Marker(self)

    def worker_source(self) -> Marker:
        return self.marker


class ShippingTests(NoLeaksTestCase):
    def tasks_of(self, executor: Any, source: Any) -> tuple[list[Any], PoolRecorder]:
        recorder = PoolRecorder()
        with cores(16), PassEngine(executor(2, pool_factory=recorder), 10) as engine:
            engine.run(source, source.partitions(), JOB)
        return recorder.pools[0].tasks, recorder

    def test_worker_processes_get_the_source_the_source_names(self) -> None:
        source = MarkedSource(parts(3))
        tasks, recorder = self.tasks_of(ProcessExecutor, source)
        self.assertEqual(len(tasks), 3)
        for task in tasks:
            self.assertIs(task.source, source.marker)
        self.assertEqual(recorder.calls, [(2, ("numpy", "first.module", "second.module"))])

    def test_a_process_source_without_a_worker_source_is_sent_as_it_is(self) -> None:
        source = PartSource(parts(3))
        tasks, recorder = self.tasks_of(ProcessExecutor, source)
        for task in tasks:
            self.assertIs(task.source, source)
        self.assertEqual(recorder.calls, [(2, ("numpy",))])

    def test_threads_share_the_source_itself(self) -> None:
        source = MarkedSource(parts(3))
        tasks, recorder = self.tasks_of(ThreadExecutor, source)
        for task in tasks:
            self.assertIs(task.source, source)
        self.assertEqual(recorder.calls, [(2, ("numpy",))])

    def test_a_downgraded_process_pass_reads_the_source_itself(self) -> None:
        source = MarkedSource(parts(3))
        source.obstacle = DowngradeReason.ENGINE_MANAGED
        with PassEngine(ProcessExecutor(2, pool_factory=PoolRecorder()), 10) as engine:
            with mock.patch.object(
                _executors, "run_partition_task", wraps=_executors.run_partition_task
            ) as spy:
                engine.run(source, source.partitions(), JOB)
        self.assertEqual(spy.call_count, 3)
        for call in spy.call_args_list:
            self.assertIs(call.args[0].source, source)

    def test_each_task_carries_its_partition_the_batch_cap_the_job_and_the_retries(self) -> None:
        source = PartSource(parts(3))
        recorder = PoolRecorder()
        with cores(16), PassEngine(ThreadExecutor(2, pool_factory=recorder), 33) as engine:
            engine.run(source, source.partitions(), JOB)
        for task, partition in zip(recorder.pools[0].tasks, source.partitions(), strict=True):
            self.assertEqual(task.partition, partition)
            self.assertEqual(task.max_rows, 33)
            self.assertIs(task.job, JOB)
            self.assertEqual(task.retries, 2)


class FlakySource(PartSource):
    """A source whose reads of chosen partitions fail with the scripted errors, in turn."""

    def __init__(self, arrays: list[np.ndarray], failures: dict[int, list[BaseException]]) -> None:
        super().__init__(arrays)
        self.failures = {ordinal: list(errors) for ordinal, errors in failures.items()}

    def batches(self, partition: Partition, *, max_rows: int = 100) -> Any:
        pending = self.failures.get(partition.ordinal)
        if pending:
            self.reads.append(partition.ordinal)
            raise pending.pop(0)
        return super().batches(partition, max_rows=max_rows)


class RetryTests(NoLeaksTestCase):
    def fit(self, executor: Any, source: Any) -> tuple[Any, ScaleExecutionPassProvenance | None]:
        with cores(16), PassEngine(executor, 10) as engine:
            merged = engine.run(source, source.partitions(), JOB)
        provenance = engine.provenance()
        return merged, None if provenance is None else provenance.passes[-1]

    def test_a_transient_error_is_read_again_and_recorded_by_every_executor(self) -> None:
        for executor in (SerialExecutor(), ThreadExecutor(3)):
            source = FlakySource(parts(4), {1: [OSError("x"), ConnectionResetError()]})
            with self.subTest(executor=repr(executor)):
                merged, record = self.fit(executor, source)
                assert record is not None
                self.assertEqual(record.retries, 2)
                self.assertEqual(source.reads.count(1), 3)
                self.assertEqual(merged.to_bytes(), state_bytes(source.parts))

    def test_retries_of_several_partitions_add_up(self) -> None:
        source = FlakySource(parts(4), {0: [OSError("a")], 2: [OSError("b"), OSError("c")]})
        merged, record = self.fit(SerialExecutor(), source)
        assert record is not None
        self.assertEqual(record.retries, 3)
        self.assertEqual(merged.to_bytes(), state_bytes(source.parts))

    def test_a_partition_that_keeps_failing_raises_after_three_reads(self) -> None:
        for executor in (SerialExecutor(), ThreadExecutor(2)):
            errors = [OSError(f"e{index}") for index in range(5)]
            source = FlakySource(parts(4), {1: list(errors)})
            with self.subTest(executor=repr(executor)):
                with self.assertRaises(OSError) as caught:
                    self.fit(executor, source)
                self.assertIs(caught.exception, errors[2])
                self.assertEqual(source.reads.count(1), 3)

    def test_an_error_that_is_final_is_raised_after_one_read(self) -> None:
        for error in (
            FileNotFoundError("x"),
            PermissionError("x"),
            ScaleStateError(ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT),
            ScaleSourceError(ScaleSourceErrorCode.NAN_VALUE),
        ):
            source = FlakySource(parts(3), {1: [error]})
            with self.subTest(error=repr(error)):
                with self.assertRaises(type(error)) as caught:
                    self.fit(SerialExecutor(), source)
                self.assertIs(caught.exception, error)
                self.assertEqual(source.reads.count(1), 1)

    def test_no_executor_means_no_retry(self) -> None:
        source = FlakySource(parts(3), {1: [OSError("x")]})
        with self.assertRaises(OSError):
            self.fit(None, source)
        self.assertEqual(source.reads.count(1), 1)

    def test_a_source_that_cannot_be_read_again_is_not_retried(self) -> None:
        for flag in ("retryable", "single_use"):
            source = FlakySource(parts(3), {1: [OSError("x")]})
            setattr(source, flag, flag == "single_use")
            with self.subTest(flag=flag):
                with self.assertRaises(OSError):
                    self.fit(SerialExecutor(), source)
                self.assertEqual(source.reads.count(1), 1)

    def test_a_source_that_says_it_is_retryable_is_retried(self) -> None:
        source = FlakySource(parts(3), {1: [OSError("x")]})
        source.retryable = True  # type: ignore[attr-defined]
        source.single_use = False  # type: ignore[attr-defined]
        _, record = self.fit(SerialExecutor(), source)
        assert record is not None
        self.assertEqual(record.retries, 1)

    def test_a_failed_pass_is_recorded_with_the_retries_of_the_tasks_that_returned(self) -> None:
        source = FlakySource(
            parts(4), {0: [OSError("a")], 1: [ScaleSourceError(ScaleSourceErrorCode.NAN_VALUE)]}
        )
        engine = PassEngine(SerialExecutor(), 10)
        with self.assertRaises(ScaleSourceError), engine:
            engine.run(source, source.partitions(), JOB)
        provenance = engine.provenance()
        assert provenance is not None
        self.assertEqual(
            provenance.passes, (ScaleExecutionPassProvenance("serial", 1, 4, 1, None),)
        )

    def test_an_arrow_stream_is_not_read_again(self) -> None:
        import pyarrow as pa

        schema = pa.schema([pa.field("x", pa.float64())])

        def batches() -> Any:
            yield pa.record_batch([pa.array([1.0, 2.0])], schema=schema)
            raise OSError("broken stream")

        source = ArrowSource(pa.RecordBatchReader.from_batches(schema, batches()), "x")
        with self.assertRaises(OSError):
            self.fit(SerialExecutor(), source)


class ValueObjectTests(unittest.TestCase):
    def test_the_new_records_are_frozen_and_slotted(self) -> None:
        from veridist.engine.provenance import ScaleExecutionProvenance
        from veridist.scale._multi_pass import GumbelKernel, ProfileJob, WeibullKernel
        from veridist.scale._tasks import PartitionTask, TaskOutcome

        partition = Partition("p0", 0, 1, None, FingerprintLevel.NONE)
        pass_record = ScaleExecutionPassProvenance("serial", 1, 1, 0, None)
        instances: list[Any] = [
            JOB,
            PartitionTask(object(), partition, 1, JOB),  # type: ignore[arg-type]
            TaskOutcome(0, b"", 0),
            ProfileJob(WeibullKernel(1.0), (1.0,)),
            WeibullKernel(1.0),
            GumbelKernel(0.0, 1.0, 2.0),
            SerialExecutor(),
            ThreadExecutor(2),
            ProcessExecutor(2),
            pass_record,
            ScaleExecutionProvenance("serial", 1, (pass_record,)),
        ]
        for instance in instances:
            with self.subTest(kind=type(instance).__name__):
                members = dataclasses.fields(instance)
                if members:
                    with self.assertRaises(FrozenInstanceError):
                        setattr(instance, members[0].name, None)
                self.assertFalse(hasattr(instance, "__dict__"))
