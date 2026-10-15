"""A spool with an executor: the first pass is serial, later passes read the files in parallel."""

from __future__ import annotations

import os
import pickle
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pyarrow as pa

from tests.unit.scale_executor_fixtures import (
    FakePool,
    NoLeaksTestCase,
    PartSource,
    PoolRecorder,
    cores,
    parquet_dataset,
)
from tests.unit.scale_multi_pass_oracle import weibull_sample
from veridist.engine.provenance import ScaleExecutionPassProvenance
from veridist.scale import _executors, _spool
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._executors import PassEngine, ProcessExecutor, ThreadExecutor
from veridist.scale._fit import fit
from veridist.scale._one_pass import ExponentialState
from veridist.scale._sources import (
    DowngradeReason,
    ExecutorKind,
    FingerprintLevel,
    Partition,
    is_retryable,
)
from veridist.scale._spool import Spool, SpoolPolicy, SpoolReplay
from veridist.scale._tasks import FoldJob


def arrays(count: int = 4, size: int = 40) -> list[np.ndarray]:
    return [weibull_sample(1.7, 50.0, size, seed) for seed in range(count)]


class TempCase(NoLeaksTestCase):
    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def policy(self) -> SpoolPolicy:
        return SpoolPolicy(max_bytes=10_000_000, directory=self.directory)


class Counting(PartSource):
    """A declared multi-partition source that counts partition reads by ordinal."""


class SpooledFitTests(TempCase):
    def test_the_first_pass_is_serial_and_later_passes_use_the_executor(self) -> None:
        data = arrays()
        base = fit(Counting(data), "weibull_min", spool=self.policy())
        source = Counting(data)
        with cores(16):
            result = fit(source, "weibull_min", executor=ThreadExecutor(3), spool=self.policy())
        self.assertEqual(result.fit, base.fit)
        assert result.scale.execution is not None and base.scale.state is not None
        self.assertEqual(result.scale.state, base.scale.state)
        records = result.scale.execution.passes
        self.assertEqual(
            records[0], ScaleExecutionPassProvenance("serial", 1, 4, 0, "spool_first_pass")
        )
        for record in records[1:]:
            self.assertEqual(record, ScaleExecutionPassProvenance("threads", 3, 4, 0, None))
        self.assertGreaterEqual(len(records), 3)
        self.assertEqual(result.scale.execution.downgrade, "spool_first_pass")
        self.assertEqual(result.scale.execution.effective_workers, 3)
        self.assertEqual(sorted(source.reads), [0, 1, 2, 3])
        self.assertEqual(os.listdir(self.directory), [])

    def test_the_underlying_source_is_read_once_in_canonical_order(self) -> None:
        source = Counting(arrays(5))
        with cores(16):
            fit(source, "gumbel_right", executor=ThreadExecutor(4), spool=self.policy())
        self.assertEqual(source.reads, [0, 1, 2, 3, 4])

    def test_the_result_does_not_depend_on_the_executor_or_the_spool(self) -> None:
        data = arrays(6, 30)
        plain = fit(PartSource(data), "weibull_min")
        for executor in (None, "serial", ThreadExecutor(1), ThreadExecutor(4)):
            with self.subTest(executor=repr(executor)), cores(16):
                spooled = fit(
                    PartSource(data), "weibull_min", executor=executor, spool=self.policy()
                )
                self.assertEqual(spooled.fit, plain.fit)
                self.assertEqual(spooled.scale.state, plain.scale.state)

    def test_a_spool_over_a_source_that_is_never_parallel_still_reads_files_in_parallel(
        self,
    ) -> None:
        # A source that says nothing about parallel reads: only the spool's files are shared.
        from tests.unit.scale_executor_fixtures import PlainSource

        data = arrays(4)
        with cores(16):
            result = fit(
                PlainSource(data), "weibull_min", executor=ThreadExecutor(2), spool=self.policy()
            )
        assert result.scale.execution is not None
        self.assertEqual(result.scale.execution.passes[0].downgrade, "spool_first_pass")
        self.assertEqual(result.scale.execution.passes[1].executor, "threads")

    def test_a_one_pass_family_ignores_the_spool_and_runs_in_parallel(self) -> None:
        data = arrays(4)
        with cores(16):
            result = fit(
                PartSource(data), "exponential", executor=ThreadExecutor(2), spool=self.policy()
            )
        assert result.scale.execution is not None
        self.assertEqual(
            result.scale.execution.passes, (ScaleExecutionPassProvenance("threads", 2, 4, 0, None),)
        )
        self.assertIsNone(result.scale.spool)

    def test_parquet_through_a_spool_is_the_same_whatever_runs_it(self) -> None:
        source = parquet_dataset(Path(self.directory) / "data", weibull_sample(1.7, 50.0, 600, 3))
        plain = fit(source, "weibull_min")
        with cores(16):
            result = fit(source, "weibull_min", executor=ThreadExecutor(3), spool=self.policy())
        self.assertEqual(result.fit, plain.fit)
        self.assertEqual(result.scale.state, plain.scale.state)
        assert result.scale.spool is not None
        self.assertEqual(result.scale.spool.row_count, 600)

    def test_a_failure_in_the_first_pass_is_not_read_again(self) -> None:
        class Failing(PartSource):
            def batches(self, partition: Partition, *, max_rows: int = 100) -> Iterator[Any]:
                if partition.ordinal == 2:
                    self.reads.append(2)
                    raise OSError("transient")
                return super().batches(partition, max_rows=max_rows)

        source = Failing(arrays())
        with cores(16), self.assertRaises(OSError):
            fit(source, "weibull_min", executor="serial", spool=self.policy())
        self.assertEqual(source.reads.count(2), 1)
        self.assertEqual(os.listdir(self.directory), [])

    def test_a_changed_spool_file_is_spool_changed_when_read_on_threads(self) -> None:
        data = arrays(4)
        spool_dir: list[str] = []
        real = _executors.PassEngine.run
        calls = {"n": 0}

        def damaging(engine: Any, source: Any, partitions: Any, job: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 2:
                (private,) = os.listdir(self.directory)
                spool_dir.append(private)
                target = sorted(Path(self.directory, private).iterdir())[1]
                target.write_bytes(target.read_bytes()[:-9])
            return real(engine, source, partitions, job)

        with (
            cores(16),
            mock.patch.object(_executors.PassEngine, "run", damaging),
            self.assertRaises(ScaleSourceError) as caught,
        ):
            fit(PartSource(data), "weibull_min", executor=ThreadExecutor(3), spool=self.policy())
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.SPOOL_CHANGED)
        self.assertEqual(os.listdir(self.directory), [])

    def test_the_pool_is_gone_before_the_spool_directory_is(self) -> None:
        events: list[str] = []
        directory = self.directory

        class Recording(FakePool):
            def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
                events.append(f"shutdown with {len(os.listdir(directory))} spool directories")
                super().shutdown(wait, cancel_futures=cancel_futures)

        recorder = PoolRecorder(lambda workers, modules: Recording(workers, modules))
        with cores(16):
            fit(
                PartSource(arrays()),
                "weibull_min",
                executor=ThreadExecutor(2, pool_factory=recorder),
                spool=self.policy(),
            )
        self.assertEqual(events, ["shutdown with 1 spool directories"])
        self.assertEqual(os.listdir(self.directory), [])


class SpooledSourceTests(TempCase):
    def sealed(self, data: list[np.ndarray]) -> tuple[Spool, Any, tuple[Partition, ...]]:
        source = PartSource(data)
        spool = Spool(self.policy())
        self.addCleanup(spool._delete)
        spooled = spool.source(source)
        partitions = source.partitions()
        for partition in partitions:
            list(spooled.batches(partition, max_rows=1000))
        return spool, spooled, partitions

    def test_it_is_parallel_only_once_every_partition_is_written(self) -> None:
        source = PartSource(arrays(3))
        spool = Spool(self.policy())
        self.addCleanup(spool._delete)
        spooled = spool.source(source)
        partitions = source.partitions()
        for kind in (ExecutorKind.THREADS, ExecutorKind.PROCESSES):
            self.assertIs(
                spooled.parallel_obstacle(kind, partitions), DowngradeReason.SPOOL_FIRST_PASS
            )
        self.assertFalse(spooled.retryable)  # nothing written yet: this pass writes
        for index, partition in enumerate(partitions):
            list(spooled.batches(partition, max_rows=1000))
            expected = None if index == 2 else DowngradeReason.SPOOL_FIRST_PASS
            self.assertIs(spooled.parallel_obstacle(ExecutorKind.THREADS, partitions), expected)
            self.assertTrue(spooled.retryable)
        self.assertIsNone(spooled.parallel_obstacle(ExecutorKind.PROCESSES, partitions))
        self.assertTrue(is_retryable(spooled))

    def test_a_spool_that_has_begun_a_partition_is_not_sealed(self) -> None:
        source = PartSource(arrays(2))
        spool = Spool(self.policy())
        self.addCleanup(spool._delete)
        self.assertFalse(spool.sealed)
        spooled = spool.source(source)
        first, second = source.partitions()
        iterator = spooled.batches(first, max_rows=1000)
        next(iterator)
        self.assertFalse(spool.sealed)
        list(iterator)
        self.assertTrue(spool.sealed)
        self.assertTrue(spool.holds([first]))
        self.assertFalse(spool.holds([first, second]))
        self.assertTrue(spool.holds([]))

    def test_the_replay_reads_every_sealed_partition_as_the_spool_does(self) -> None:
        spool, spooled, partitions = self.sealed(arrays(3))
        replay = spooled.worker_source()
        self.assertIsInstance(replay, SpoolReplay)
        for partition in partitions:
            for size in (1000, 7):
                with self.subTest(ordinal=partition.ordinal, size=size):
                    mine = [b.tolist() for b in replay.batches(partition, max_rows=size)]
                    theirs = [b.tolist() for b in spool.batches(PartSource([]), partition, size)]
                    self.assertEqual(mine, theirs)
                    self.assertTrue(all(len(batch) <= size for batch in mine))

    def test_the_replay_pickles_and_reads_from_the_copy(self) -> None:
        _, spooled, partitions = self.sealed(arrays(2))
        replay = pickle.loads(pickle.dumps(spooled.worker_source()))
        self.assertEqual(sum(len(b) for b in replay.batches(partitions[1], max_rows=1000)), 40)
        self.assertEqual(SpoolReplay.WORKER_MODULES, ("pyarrow",))

    def test_the_replay_holds_the_partitions_sealed_when_it_was_made(self) -> None:
        source = PartSource(arrays(2))
        spool = Spool(self.policy())
        self.addCleanup(spool._delete)
        spooled = spool.source(source)
        first, second = source.partitions()
        list(spooled.batches(first, max_rows=1000))
        replay = spool.replay()
        list(spooled.batches(second, max_rows=1000))
        self.assertEqual(sum(len(b) for b in replay.batches(first)), 40)
        with self.assertRaises(ScaleSourceError) as caught:
            replay.batches(second)
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.SPOOL_CHANGED)

    def test_the_replay_refuses_what_is_not_a_sealed_partition(self) -> None:
        _, spooled, _ = self.sealed(arrays(1))
        replay = spooled.worker_source()
        for bad in (Partition("other", 0, 1, None, FingerprintLevel.NONE), "p0", None):  # type: ignore[attr-defined]
            with self.subTest(bad=repr(bad)), self.assertRaises(ScaleSourceError) as caught:
                replay.batches(bad)
            self.assertIs(caught.exception.code, ScaleSourceErrorCode.SPOOL_CHANGED)

    def test_the_replay_verifies_what_it_reads(self) -> None:
        spool, spooled, partitions = self.sealed(arrays(2))
        replay = spooled.worker_source()
        target = sorted(Path(spool._directory).iterdir())[0]
        target.write_bytes(target.read_bytes()[:-9])
        with self.assertRaises(ScaleSourceError) as caught:
            list(replay.batches(partitions[0]))
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.SPOOL_CHANGED)
        # a file that is gone is the same refusal
        for path in Path(spool._directory).iterdir():
            path.unlink()
        with self.assertRaises(ScaleSourceError) as caught:
            list(replay.batches(partitions[1]))
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.SPOOL_CHANGED)

    def test_the_replay_validates_the_batch_bound(self) -> None:
        _, spooled, partitions = self.sealed(arrays(1))
        replay = spooled.worker_source()
        for bad in (0, -1):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replay.batches(partitions[0], max_rows=bad)
        with self.assertRaises(TypeError):
            replay.batches(partitions[0], max_rows=2.0)  # type: ignore[arg-type]

    def test_the_replay_closes_its_file_after_each_read(self) -> None:
        spool, spooled, partitions = self.sealed(arrays(2))
        replay = spooled.worker_source()
        opened: list[Any] = []
        real = _spool._open_file

        def spying(pa_module: Any, path: str) -> tuple[Any, Any]:
            handle, reader = real(pa_module, path)
            opened.append(handle)
            return handle, reader

        with mock.patch.object(_spool, "_open_file", spying):
            for partition in partitions:
                list(replay.batches(partition))
        self.assertEqual(len(opened), 2)
        self.assertTrue(all(handle.closed for handle in opened))
        self.assertEqual(spool._handles, [])

    def test_the_replay_keeps_no_registry_and_the_spool_keeps_its_own(self) -> None:
        spool, spooled, partitions = self.sealed(arrays(1))
        iterator = spool.batches(PartSource([]), partitions[0], 5)
        next(iterator)
        self.assertEqual(len(spool._handles), 1)
        iterator.close()
        self.assertEqual(spool._handles, [])
        replay = spooled.worker_source()
        iterator = replay.batches(partitions[0], max_rows=5)
        next(iterator)
        self.assertEqual(spool._handles, [])
        iterator.close()

    def test_the_schema_of_a_spool_file(self) -> None:
        schema = _spool.spool_schema(pa)
        self.assertEqual(schema.names, ["x"])
        self.assertEqual(schema.field(0).type, pa.float64())
        self.assertFalse(schema.field(0).nullable)

    def test_the_engine_hands_processes_the_replay_and_threads_the_spool(self) -> None:
        spool, spooled, partitions = self.sealed(arrays(3))
        job = FoldJob(ExponentialState)
        for make, kind in ((ProcessExecutor, SpoolReplay), (ThreadExecutor, type(spooled))):
            recorder = PoolRecorder()
            with self.subTest(kind=kind.__name__), cores(16):
                with PassEngine(make(2, pool_factory=recorder), 100) as engine:
                    merged = engine.run(spooled, partitions, job)
                tasks = recorder.pools[0].tasks
                self.assertEqual(len(tasks), 3)
                for task in tasks:
                    self.assertIs(type(task.source), kind)
                    self.assertEqual(task.retries, 2)
                self.assertEqual(merged.count, 120)

    def test_a_process_pool_made_for_the_replay_imports_pyarrow(self) -> None:
        _, spooled, partitions = self.sealed(arrays(3))
        recorder = PoolRecorder()
        with cores(16), PassEngine(ProcessExecutor(2, pool_factory=recorder), 100) as engine:
            engine.run(spooled, partitions, FoldJob(ExponentialState))
        self.assertEqual(recorder.calls, [(2, ("numpy", "pyarrow"))])
