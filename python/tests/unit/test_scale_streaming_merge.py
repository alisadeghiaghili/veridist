"""The engine merges each partition's state as it arrives and holds a constant number of them.

A pass over many partitions must not keep a state per partition: the memory of a pass is the
running value, the state being merged and the tasks in flight. These tests count the states
alive at once (through jobs that report their own allocations, so nothing depends on the
resident set), pin the window of in-flight tasks, and compare the merged value, bit for bit,
with the balanced-tree merge the engine used before (``scale_merge_oracle``) over random
partition counts, completion orders and executors, for the states of every family, with and
without a checkpoint.
"""

from __future__ import annotations

import concurrent.futures
import threading
import tracemalloc
import unittest
import weakref
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import mock

import numpy as np

from tests.unit.scale_executor_fixtures import (
    NoLeaksTestCase,
    PartSource,
    PoolRecorder,
    ScriptedPool,
    cores,
)
from tests.unit.scale_merge_oracle import merge_tree
from veridist.scale import _executors
from veridist.scale._checkpoint import CheckpointRequest, CheckpointRun, job_identity
from veridist.scale._executors import (
    IN_FLIGHT_PER_WORKER,
    PassEngine,
    SerialExecutor,
    ThreadExecutor,
)
from veridist.scale._gof import PitJob
from veridist.scale._multi_pass import GumbelKernel, ProfileAccumulator, ProfileJob, WeibullKernel
from veridist.scale._one_pass import ExponentialState, GammaState, LognormalState, NormalState
from veridist.scale._partition_store import SQLitePartitionStore
from veridist.scale._pilot import GumbelPilotState, WeibullPilotState
from veridist.scale._sources import NumpySource, Partition
from veridist.scale._tasks import FoldJob, Job, TaskOutcome

MAX_ROWS = 7


class Tally:
    """Counts what a job allocates: results not yet merged and merge values alive at once."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.produced = 0
        self.restored = 0
        self.alive = 0
        self.peak_unmerged = 0
        self.peak_alive = 0

    def produce(self) -> None:
        with self._lock:
            self.produced += 1
            self.peak_unmerged = max(self.peak_unmerged, self.produced - self.restored)

    def restore(self) -> None:
        with self._lock:
            self.restored += 1

    def node(self, members: tuple[int, ...]) -> Node:
        node = Node(self, members)
        with self._lock:
            self.alive += 1
            self.peak_alive = max(self.peak_alive, self.alive)
        weakref.finalize(node, self._gone)
        return node

    def _gone(self) -> None:
        with self._lock:
            self.alive -= 1


class Node:
    """A mergeable value that knows which partitions it holds; the tally counts its life."""

    def __init__(self, tally: Tally, members: tuple[int, ...]) -> None:
        self.tally = tally
        self.members = members

    def merge(self, other: Node) -> Node:
        return self.tally.node(tuple(sorted(self.members + other.members)))

    def to_bytes(self) -> bytes:
        return ",".join(map(str, self.members)).encode()


class CountingJob:
    """A job whose results and merge values are counted by ``tally``."""

    def __init__(self, tally: Tally) -> None:
        self.tally = tally

    def run(self, source: Any, partition: Partition, max_rows: int) -> bytes:
        self.tally.produce()
        return str(partition.ordinal).encode()

    def restore(self, data: bytes) -> Node:
        self.tally.restore()
        return self.tally.node((int(data),))

    def identity(self) -> Node:
        return self.tally.node(())

    def checkpoint_key(self) -> tuple[object, ...]:
        return ("counting", 1)


class RecordingPool(ScriptedPool):
    """A scripted pool that remembers the most tasks submitted and not yet merged."""

    tally: Tally | None = None
    peak_in_flight = 0

    def submit(self, function: Any, *arguments: Any) -> Any:
        future = super().submit(function, *arguments)
        assert self.tally is not None
        self.peak_in_flight = max(self.peak_in_flight, len(self.tasks) - self.tally.restored)
        return future


class ShufflingPool(ScriptedPool):
    """A scripted pool that finishes a random selection of its queued tasks at each wait."""

    seed = 0
    rng: np.random.Generator | None = None

    def wait(self, pending: Any, timeout: Any = None, return_when: Any = None) -> Any:
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)
        queued = [position for position, item in enumerate(self.queue) if not item[0].done()]
        if queued:
            chosen = self.rng.permutation(queued)[: int(self.rng.integers(1, len(queued) + 1))]
            for position in chosen:
                self.run(int(position))
        return concurrent.futures.wait(pending, timeout=0, return_when=return_when)


@contextmanager
def engine_on(
    pool_type: type[ScriptedPool],
    workers: int,
    *,
    max_rows: int = 10,
    checkpoint: CheckpointRun | None = None,
    **fields: Any,
) -> Iterator[tuple[PassEngine, list[ScriptedPool]]]:
    """An engine on a thread executor whose pool is a ``pool_type`` that the test steers."""

    pools: list[ScriptedPool] = []

    def make(count: int, modules: tuple[str, ...]) -> ScriptedPool:
        pool = pool_type(count, modules)
        for name, value in fields.items():
            setattr(pool, name, value)
        pools.append(pool)
        return pool

    executor = ThreadExecutor(workers, pool_factory=PoolRecorder(make))
    engine = PassEngine(executor, max_rows, checkpoint)
    with ExitStack() as stack:
        stack.enter_context(cores(64))
        stack.enter_context(
            mock.patch.object(_executors, "wait", lambda *a, **k: pools[0].wait(*a, **k))
        )
        stack.enter_context(engine)
        yield engine, pools


def pieces(count: int, size: int = 1) -> list[np.ndarray]:
    rng = np.random.default_rng(count)
    return [rng.exponential(2.0, size) + 0.5 for _ in range(count)]


def checkpoint_run(store: SQLitePartitionStore, partitions: tuple[Partition, ...]) -> CheckpointRun:
    identity = job_identity(
        family="exponential",
        column=None,
        source_kind="numpy",
        max_rows=10,
        partitions=partitions,
        state_schema="scale.test:1",
        plan=None,
    )
    request = CheckpointRequest(store, True, False, False, False)
    return CheckpointRun.open(request, identity, partitions, multi_pass=False)


def size_of_one_accumulator(candidates: int) -> int:
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        held = ProfileAccumulator.from_bytes(ProfileAccumulator(candidates).to_bytes(), candidates)
        size = tracemalloc.get_traced_memory()[0] - before
    finally:
        tracemalloc.stop()
    assert held.count == 0
    return size


class ConstantStatesTests(NoLeaksTestCase):
    def test_a_serial_pass_holds_a_constant_number_of_states(self) -> None:
        peaks = []
        for count in (3, 25, 200):
            tally = Tally()
            source = PartSource(pieces(count))
            with PassEngine(SerialExecutor(), 10) as engine:
                merged = engine.run(source, source.partitions(), CountingJob(tally))
            with self.subTest(partitions=count):
                self.assertEqual(merged.members, tuple(range(count)))
                self.assertEqual(tally.produced, count)
                self.assertEqual(tally.restored, count)
                self.assertLessEqual(tally.peak_unmerged, 1)
                # The running value, the state being merged and the value they make.
                self.assertLessEqual(tally.peak_alive, 3)
            peaks.append((tally.peak_unmerged, tally.peak_alive))
        self.assertEqual(len(set(peaks[1:])), 1)

    def test_no_executor_holds_a_constant_number_of_states_too(self) -> None:
        tally = Tally()
        source = PartSource(pieces(120))
        with PassEngine(None, 10) as engine:
            merged = engine.run(source, source.partitions(), CountingJob(tally))
        self.assertEqual(merged.members, tuple(range(120)))
        self.assertEqual((tally.peak_unmerged, tally.peak_alive), (1, 3))

    def test_a_real_thread_pool_holds_no_more_results_than_its_window(self) -> None:
        for workers in (2, 3):
            window = workers * IN_FLIGHT_PER_WORKER
            for count in (30, 200):
                tally = Tally()
                source = PartSource(pieces(count))
                with self.subTest(workers=workers, partitions=count):
                    with cores(16), PassEngine(ThreadExecutor(workers), 10) as engine:
                        merged = engine.run(source, source.partitions(), CountingJob(tally))
                    self.assertEqual(merged.members, tuple(range(count)))
                    self.assertLessEqual(tally.peak_unmerged, window)
                    self.assertLessEqual(tally.peak_alive, 3)

    def test_the_window_is_the_workers_times_the_constant_and_is_used(self) -> None:
        for workers in (2, 3, 5):
            tally = Tally()
            source = PartSource(pieces(200))
            with engine_on(RecordingPool, workers, tally=tally) as (engine, pools):
                merged = engine.run(source, source.partitions(), CountingJob(tally))
            pool = pools[0]
            assert isinstance(pool, RecordingPool)
            window = workers * IN_FLIGHT_PER_WORKER
            with self.subTest(workers=workers):
                self.assertEqual(merged.members, tuple(range(200)))
                self.assertEqual(pool.peak_in_flight, window)
                self.assertEqual(tally.peak_unmerged, window)
                self.assertEqual(len(pool.tasks), 200)
                self.assertLessEqual(tally.peak_alive, 3)

    def test_tasks_are_submitted_in_canonical_order_a_window_at_a_time(self) -> None:
        source = PartSource(pieces(20))
        seen: list[int] = []

        class Watching(ScriptedPool):
            def wait(self, pending: Any, timeout: Any = None, return_when: Any = None) -> Any:
                seen.append(len(self.tasks))
                return super().wait(pending, timeout, return_when)

        with engine_on(Watching, 2) as (engine, pools):
            engine.run(source, source.partitions(), CountingJob(Tally()))
        self.assertEqual(seen[0], 2 * IN_FLIGHT_PER_WORKER)
        self.assertEqual(seen, sorted(seen))
        self.assertEqual([task.partition.ordinal for task in pools[0].tasks], list(range(20)))

    def test_a_pass_with_fewer_partitions_than_the_window_submits_them_all_at_once(self) -> None:
        source = PartSource(pieces(3))
        tally = Tally()
        with engine_on(RecordingPool, 3, tally=tally) as (engine, pools):
            engine.run(source, source.partitions(), CountingJob(tally))
        assert isinstance(pools[0], RecordingPool)
        self.assertEqual(pools[0].peak_in_flight, 3)

    def test_the_partitions_of_a_resumed_pass_are_merged_one_at_a_time(self) -> None:
        count = 50
        with TemporaryDirectory() as directory:
            source = NumpySource(np.arange(count * 3, dtype=np.float64), partition_rows=3)
            partitions = source.partitions()
            store = SQLitePartitionStore(Path(directory) / "job.db")
            step = checkpoint_run(store, partitions).begin_pass(0, CountingJob(Tally()), partitions)
            for partition in partitions:
                if partition.ordinal % 5:
                    step.commit(TaskOutcome(partition.ordinal, str(partition.ordinal).encode(), 0))
            tally = Tally()
            run = checkpoint_run(store, partitions)
            with PassEngine(SerialExecutor(), 10, run) as engine:
                merged = engine.run(source, partitions, CountingJob(tally))
            self.assertEqual(merged.members, tuple(range(count)))
            self.assertEqual(tally.produced, count // 5)
            self.assertEqual(tally.restored, count)
            self.assertLessEqual(tally.peak_unmerged, 1)
            self.assertLessEqual(tally.peak_alive, 3)
            self.assertEqual(run.provenance().passes[0].reused, count - count // 5)

    def test_real_profile_accumulators_do_not_pile_up(self) -> None:
        candidates = (0.5, 1.0, 1.7, 2.5, 3.0, 4.0, 5.0, 6.0)
        job = ProfileJob(WeibullKernel(1.0), candidates)
        one = size_of_one_accumulator(len(candidates))
        peaks = []
        for count in (4, 60):
            source = PartSource(pieces(count, 3))
            tracemalloc.start()
            try:
                with PassEngine(SerialExecutor(), 10) as engine:
                    merged = engine.run(source, source.partitions(), job)
                peaks.append(tracemalloc.get_traced_memory()[1])
            finally:
                tracemalloc.stop()
            self.assertEqual(merged.count, 3 * count)
        self.assertLess(peaks[1] - peaks[0], one // 2)


class FailureWithAWindowTests(NoLeaksTestCase):
    def failing(
        self,
        errors: dict[int, BaseException],
        *,
        count: int = 10,
        workers: int = 2,
        script: list[list[int]] | None = None,
    ) -> tuple[BaseException, ScriptedPool, list[int]]:
        ran: list[int] = []
        lock = threading.Lock()

        class Failing(CountingJob):
            def run(self, source: Any, partition: Partition, max_rows: int) -> bytes:
                with lock:
                    ran.append(partition.ordinal)
                error = errors.get(partition.ordinal)
                if error is not None:
                    raise error
                return super().run(source, partition, max_rows)

        source = PartSource(pieces(count))
        seen: list[ScriptedPool] = []
        with self.assertRaises(BaseException) as caught:
            with engine_on(ScriptedPool, workers, script=script or []) as (engine, pools):
                seen = pools
                engine.run(source, source.partitions(), Failing(Tally()))
        return caught.exception, seen[0], ran

    def test_nothing_is_submitted_after_a_failure(self) -> None:
        only = ValueError("partition 1")
        error, pool, ran = self.failing({1: only})
        self.assertIs(error, only)
        self.assertEqual(len(pool.tasks), 2 * IN_FLIGHT_PER_WORKER)
        self.assertEqual(sorted(ran), [0, 1, 2, 3])

    def test_the_lowest_failing_ordinal_is_raised_when_tasks_were_submitted_after_a_merge(
        self,
    ) -> None:
        low, high = ValueError("5"), KeyError("7")
        error, pool, ran = self.failing({5: low, 7: high})
        self.assertIs(error, low)
        self.assertEqual(len(pool.tasks), 8)
        self.assertEqual(sorted(ran), list(range(8)))

    def test_a_failure_in_a_later_window_does_not_hide_an_earlier_one_still_running(self) -> None:
        early, late = ValueError("early"), KeyError("late")
        # Window of 4: 0..3 finish and merge, 4..7 are submitted; 6 fails first, then 4,
        # which cancels 5 and, with 6, 7 already cancelled by the first failure.
        error, pool, ran = self.failing({4: early, 6: late}, script=[[0, 1, 2, 3], [6], [4]])
        self.assertIs(error, early)
        self.assertEqual(len(pool.tasks), 8)
        self.assertEqual(ran, [0, 1, 2, 3, 6, 4])
        self.assertEqual(
            [future.cancelled() for future in pool.futures], [False] * 5 + [True, False, True]
        )

    def test_an_error_of_the_merge_is_raised_at_once_and_cancels_the_rest(self) -> None:
        class Refusing(CountingJob):
            def restore(self, data: bytes) -> Node:
                if data == b"3":
                    raise ValueError("refused")
                return super().restore(data)

        source = PartSource(pieces(12))
        with self.assertRaisesRegex(ValueError, "^refused$"):
            with engine_on(ScriptedPool, 2) as (engine, pools):
                engine.run(source, source.partitions(), Refusing(Tally()))
        self.assertEqual(pools[0].shutdowns, [(True, True)])
        self.assertEqual(len(pools[0].tasks), 2 * IN_FLIGHT_PER_WORKER)


class Sources:
    """Seeded multi-partition data of every shape a job needs."""

    def __init__(self, seed: int, count: int) -> None:
        rng = np.random.default_rng(seed)
        self.sizes = [int(rng.integers(0, 40)) for _ in range(count)]
        self.sizes[int(rng.integers(0, count))] = 25
        self.parts = [rng.gamma(3.0, 2.0, size) + 0.25 for size in self.sizes]
        whole = np.concatenate(self.parts)
        self.peak_log = float(np.log(whole.max()))
        center = float(whole.mean())
        peak = float(max(whole.max() - center, center - whole.min()))
        self.gumbel = GumbelKernel(float((whole.min() - center) / peak), center, peak)
        self.candidates = tuple(float(value) for value in rng.uniform(0.3, 4.0, 4))

    def jobs(self) -> list[tuple[str, Job]]:
        return [
            ("exponential", FoldJob(ExponentialState)),
            ("normal", FoldJob(NormalState)),
            ("gamma", FoldJob(GammaState)),
            ("lognormal", FoldJob(LognormalState)),
            ("weibull pilot", FoldJob(WeibullPilotState)),
            ("gumbel pilot", FoldJob(GumbelPilotState)),
            ("weibull profile", ProfileJob(WeibullKernel(self.peak_log), self.candidates)),
            ("gumbel profile", ProfileJob(self.gumbel, self.candidates)),
            ("pit", PitJob("normal", ("mu", "sigma"), (6.0, 3.0), 3, 1e-12)),
        ]


def old_merge(source: PartSource, job: Job) -> bytes:
    """What the engine returned before: every state restored, then the balanced tree."""

    states = [job.restore(job.run(source, p, MAX_ROWS)) for p in source.partitions()]
    merged = merge_tree(states, lambda left, right: left.merge(right), job.identity())
    return bytes(merged.to_bytes())


class BitIdentityTests(NoLeaksTestCase):
    def test_the_streamed_merge_is_the_tree_merge_for_every_state_and_serial_reads(self) -> None:
        for seed, count in ((1, 1), (2, 2), (3, 3), (4, 7), (5, 8), (6, 13)):
            data = Sources(seed, count)
            source = PartSource(data.parts)
            for name, job in data.jobs():
                oracle = old_merge(source, job)
                for executor in (None, SerialExecutor()):
                    with self.subTest(job=name, partitions=count, executor=executor):
                        with PassEngine(executor, MAX_ROWS) as engine:
                            merged = engine.run(source, source.partitions(), job)
                        self.assertEqual(merged.to_bytes(), oracle)

    def test_any_completion_order_gives_the_tree_merge(self) -> None:
        for seed in range(5):
            count = 2 + (seed * 7) % 17
            data = Sources(100 + seed, count)
            source = PartSource(data.parts)
            for name, job in data.jobs():
                oracle = old_merge(source, job)
                for workers in (2, 4):
                    with self.subTest(job=name, partitions=count, seed=seed, workers=workers):
                        with engine_on(
                            ShufflingPool, workers, max_rows=MAX_ROWS, seed=seed * 10 + workers
                        ) as (engine, pools):
                            merged = engine.run(source, source.partitions(), job)
                        self.assertEqual(merged.to_bytes(), oracle)
                        self.assertEqual(len(pools[0].tasks), count)

    def test_real_threads_give_the_tree_merge(self) -> None:
        for seed, count in ((11, 9), (12, 20)):
            data = Sources(seed, count)
            source = PartSource(data.parts)
            for name, job in data.jobs():
                oracle = old_merge(source, job)
                for workers in (2, 4):
                    with self.subTest(job=name, partitions=count, workers=workers):
                        with cores(16), PassEngine(ThreadExecutor(workers), MAX_ROWS) as engine:
                            merged = engine.run(source, source.partitions(), job)
                        self.assertEqual(merged.to_bytes(), oracle)

    def test_the_result_is_the_state_of_all_the_rows(self) -> None:
        data = Sources(21, 19)
        source = PartSource(data.parts)
        whole = ExponentialState.empty().update(np.concatenate(data.parts))
        with engine_on(ShufflingPool, 3, max_rows=MAX_ROWS, seed=5) as (engine, _):
            merged = engine.run(source, source.partitions(), FoldJob(ExponentialState))
        self.assertEqual(merged.to_bytes(), whole.to_bytes())

    def test_no_partitions_give_the_identity(self) -> None:
        source = PartSource(pieces(1))
        with PassEngine(SerialExecutor(), 10) as engine:
            merged = engine.run(source, (), FoldJob(ExponentialState))
        self.assertEqual(merged.to_bytes(), ExponentialState.empty().to_bytes())


class CheckpointIdentityTests(NoLeaksTestCase):
    def test_a_resumed_pass_is_the_tree_merge_whatever_was_committed_and_however_it_is_read(
        self,
    ) -> None:
        data = Sources(31, 6)
        rows = np.concatenate(data.parts)
        source = NumpySource(rows, partition_rows=max(1, len(rows) // 6))
        partitions = source.partitions()
        rng = np.random.default_rng(7)
        for name, job in (
            ("exponential", FoldJob(ExponentialState)),
            ("weibull profile", ProfileJob(WeibullKernel(data.peak_log), data.candidates)),
            ("pit", PitJob("normal", ("mu", "sigma"), (6.0, 3.0), 3, 1e-12)),
        ):
            states = [job.restore(job.run(source, p, MAX_ROWS)) for p in partitions]
            oracle = merge_tree(states, lambda a, b: a.merge(b), job.identity()).to_bytes()
            for committed in (0, len(partitions) // 2, len(partitions)):
                for shuffled in (False, True):
                    with self.subTest(job=name, committed=committed, shuffled=shuffled):
                        merged, in_ledger = self.resume(
                            source, partitions, job, rng, committed, shuffled
                        )
                        self.assertEqual(merged, oracle)
                        self.assertEqual(in_ledger, oracle)

    def resume(
        self,
        source: NumpySource,
        partitions: tuple[Partition, ...],
        job: Job,
        rng: np.random.Generator,
        committed: int,
        shuffled: bool,
    ) -> tuple[bytes, bytes]:
        with TemporaryDirectory() as directory:
            store = SQLitePartitionStore(Path(directory) / "job.db")
            step = checkpoint_run(store, partitions).begin_pass(0, job, partitions)
            for position in rng.permutation(len(partitions))[:committed]:
                partition = partitions[int(position)]
                step.commit(TaskOutcome(partition.ordinal, job.run(source, partition, MAX_ROWS), 0))
            run = checkpoint_run(store, partitions)
            if shuffled:
                with engine_on(
                    ShufflingPool, 3, max_rows=MAX_ROWS, checkpoint=run, seed=committed
                ) as (engine, _):
                    merged = engine.run(source, partitions, job)
            else:
                with PassEngine(SerialExecutor(), MAX_ROWS, run) as engine:
                    merged = engine.run(source, partitions, job)
            record = run.provenance().passes[0]
            self.assertEqual((record.reused, record.read), (committed, len(partitions) - committed))
            return bytes(merged.to_bytes()), store.read_ledger()[0].data


if __name__ == "__main__":
    unittest.main()
