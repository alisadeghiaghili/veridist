"""``fit(..., checkpoint=...)``: interrupt anywhere, resume, and get the uninterrupted result.

Every interruption is a failure injected at a commit of the parent (``Interrupting``), so it
lands at the same point whether the partitions are read serially or on threads. A resumed
fit must be bit for bit the fit that was never interrupted: the result, the uncertainty, the
state digest and the whole provenance but the checkpoint and execution records.
"""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_checkpoint_fixtures import (
    MULTI_PASS,
    ONE_PASS,
    Interrupting,
    Interruption,
    ReadLog,
    Tracker,
    execute,
    make_single_row_database,
    rows,
    same_as,
    values_for,
)
from tests.unit.scale_executor_fixtures import NoLeaksTestCase
from veridist.families.gamma import GammaFitFailure, GammaFitFailureCode
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._errors import (
    ScaleCheckpointError,
    ScaleCheckpointErrorCode,
    ScaleSourceError,
    ScaleSourceErrorCode,
)
from veridist.scale._executors import ThreadExecutor
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._partition_store import SQLitePartitionStore, _row_checksum
from veridist.scale._sources import NumpySource
from veridist.scale._spool import ScaleSpoolPolicy

PARTITIONS = 4
ROWS = 480
MAX_ROWS = 64


def numpy_source(family: str, size: int = ROWS) -> tuple[NumpySource, np.ndarray]:
    values = values_for(family, size)
    return NumpySource(values, partition_rows=size // PARTITIONS), values


class CheckpointCase(NoLeaksTestCase):
    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "job.db"

    def run_fit(self, source: Any, family: str, **options: Any) -> ScaleFitResult:
        """A fit of ``source``, read through a log unless it is one already.

        A log is a source of its own kind, so a job started through one is resumed through
        one; ``bare`` fits the source as it is.
        """

        read = source if isinstance(source, ReadLog) else ReadLog(source)
        return fit(read, family, **{"max_rows": MAX_ROWS, **options})

    def bare(self, source: Any, family: str, **options: Any) -> ScaleFitResult:
        return fit(source, family, **{"max_rows": MAX_ROWS, **options})

    def interrupt(
        self,
        source: Any,
        family: str,
        *,
        executor: Any = None,
        error: BaseException | None = None,
        resume: bool = False,
        **limits: Any,
    ) -> Interrupting:
        """Run a fit that fails at the commit named by ``limits``; the committed work stays."""

        store = Interrupting(self.path, error=error, **limits)
        with self.assertRaises(type(store.error)):
            self.run_fit(source, family, checkpoint=store, resume=resume, executor=executor)
        return store

    def checkpoint_of(self, result: ScaleFitResult) -> Any:
        assert result.scale.checkpoint is not None
        return result.scale.checkpoint


class OnePassResumeTests(CheckpointCase):
    def test_an_interrupted_fit_resumes_to_the_uninterrupted_result_at_every_point(self) -> None:
        for family in ONE_PASS:
            source, _ = numpy_source(family)
            base = self.run_fit(source, family)
            for executor in (None, ThreadExecutor(3)):
                for after in range(1, PARTITIONS):
                    with self.subTest(family=family, executor=executor, after=after):
                        self.path = self.root / f"{family}-{after}-{executor is None}.db"
                        stopped = self.interrupt(
                            ReadLog(source), family, executor=executor, after=after
                        )
                        self.assertEqual(stopped.partition_commits, after)
                        log = ReadLog(source)
                        result = self.run_fit(
                            log,
                            family,
                            checkpoint=SQLitePartitionStore(self.path),
                            resume=True,
                            executor=executor,
                        )
                        self.assertTrue(same_as(result, base))
                        self.assertEqual(len(log.reads), PARTITIONS - after)
                        self.assertEqual(len(set(log.reads)), PARTITIONS - after)
                        record = self.checkpoint_of(result)
                        self.assertTrue(record.resumed)
                        self.assertEqual(record.reused_partitions, after)
                        self.assertEqual(record.read_partitions, PARTITIONS - after)
                        self.assertEqual(record.replayed_passes, 0)

    def test_serial_reading_commits_in_canonical_order_so_the_first_partitions_are_kept(
        self,
    ) -> None:
        source, _ = numpy_source("normal")
        self.interrupt(ReadLog(source), "normal", after=2)
        log = ReadLog(source)
        self.run_fit(log, "normal", checkpoint=self.path, resume=True)
        self.assertEqual(log.reads, [2, 3])

    def test_a_keyboard_interrupt_is_resumable_too(self) -> None:
        for executor in (None, ThreadExecutor(2)):
            for family in ("exponential", "gamma"):
                with self.subTest(family=family, executor=executor):
                    self.path = self.root / f"{family}-{executor is None}.db"
                    source, _ = numpy_source(family)
                    base = self.run_fit(source, family)
                    self.interrupt(
                        ReadLog(source),
                        family,
                        executor=executor,
                        error=KeyboardInterrupt(),
                        after=2,
                    )
                    result = self.run_fit(
                        source, family, checkpoint=self.path, resume=True, executor=executor
                    )
                    self.assertTrue(same_as(result, base))
                    self.assertEqual(self.checkpoint_of(result).reused_partitions, 2)

    def test_a_fit_interrupted_several_times_still_reaches_the_same_result(self) -> None:
        source, _ = numpy_source("lognormal", 960)
        base = self.run_fit(source, "lognormal")
        self.interrupt(source, "lognormal", after=1)
        for expected in (1, 2):
            store = Interrupting(self.path, after=1)
            with self.assertRaises(Interruption):
                self.run_fit(source, "lognormal", checkpoint=store, resume=True)
            held = SQLitePartitionStore(self.path)
            self.assertEqual(len(rows(held.path, "SELECT * FROM partition_state")), expected + 1)
        result = self.run_fit(source, "lognormal", checkpoint=self.path, resume=True)
        self.assertTrue(same_as(result, base))
        self.assertEqual(self.checkpoint_of(result).reused_partitions, 3)

    def test_a_failure_after_every_state_but_before_the_ledger_reads_nothing_on_resume(
        self,
    ) -> None:
        for executor in (None, ThreadExecutor(2)):
            with self.subTest(executor=executor):
                self.path = self.root / f"ledger-{executor is None}.db"
                source, _ = numpy_source("gamma")
                base = self.run_fit(source, "gamma")
                stopped = self.interrupt(source, "gamma", executor=executor, after_passes=0)
                self.assertEqual(stopped.partition_commits, PARTITIONS)
                log = ReadLog(source)
                result = self.run_fit(
                    log, "gamma", checkpoint=self.path, resume=True, executor=executor
                )
                self.assertEqual(log.reads, [])
                self.assertTrue(same_as(result, base))
                record = self.checkpoint_of(result)
                self.assertEqual(
                    (record.reused_partitions, record.read_partitions, record.replayed_passes),
                    (PARTITIONS, 0, 0),
                )

    def test_a_finished_job_is_replayed_from_the_ledger_without_a_read(self) -> None:
        source, _ = numpy_source("normal")
        base = self.run_fit(source, "normal")
        first = self.run_fit(source, "normal", checkpoint=self.path)
        self.assertTrue(same_as(first, base))
        self.assertFalse(self.checkpoint_of(first).resumed)
        self.assertEqual(self.checkpoint_of(first).read_partitions, PARTITIONS)
        log = ReadLog(source)
        again = self.run_fit(log, "normal", checkpoint=self.path, resume=True)
        self.assertEqual(log.reads, [])
        self.assertTrue(same_as(again, base))
        record = self.checkpoint_of(again)
        self.assertEqual((record.replayed_passes, record.read_partitions), (1, 0))
        self.assertEqual(record.passes[0].reused, PARTITIONS)

    def test_a_job_can_be_resumed_by_another_executor(self) -> None:
        source, _ = numpy_source("exponential")
        base = self.run_fit(source, "exponential")
        self.interrupt(source, "exponential", after=2)
        result = self.run_fit(
            source, "exponential", checkpoint=self.path, resume=True, executor=ThreadExecutor(2)
        )
        self.assertTrue(same_as(result, base))
        assert result.scale.execution is not None
        self.assertEqual(result.scale.execution.passes[0].tasks, 2)

    def test_a_fit_that_returns_a_failure_is_checkpointed_and_resumes_to_the_same_failure(
        self,
    ) -> None:
        values = values_for("gamma", ROWS)
        values[ROWS - 3] = -1.0
        source = NumpySource(values, partition_rows=ROWS // PARTITIONS)
        base = self.run_fit(source, "gamma")
        self.assertIsInstance(base.fit, GammaFitFailure)
        assert isinstance(base.fit, GammaFitFailure)
        self.assertIs(base.fit.code, GammaFitFailureCode.INVALID_SUPPORT)
        first = self.run_fit(source, "gamma", checkpoint=self.path)
        self.assertEqual(first.fit, base.fit)
        record = self.checkpoint_of(first)
        self.assertEqual(len(record.passes), 1)
        self.assertEqual(record.read_partitions, PARTITIONS - 1)
        again = self.run_fit(source, "gamma", checkpoint=self.path, resume=True)
        self.assertEqual(again.fit, base.fit)
        self.assertEqual(self.checkpoint_of(again).reused_partitions, PARTITIONS - 1)

    def test_no_connection_outlives_a_fit_however_it_ends(self) -> None:
        source, _ = numpy_source("gamma")
        tracker = Tracker()
        with tracker.watching():
            self.interrupt(source, "gamma", executor=ThreadExecutor(2), after=2)
            self.interrupt(source, "gamma", error=KeyboardInterrupt(), resume=True, after=1)
            self.run_fit(source, "gamma", checkpoint=self.path, resume=True)
            self.run_fit(source, "gamma", checkpoint=self.path, resume=True)
            with self.assertRaises(ScaleCheckpointError):
                self.run_fit(source, "gamma", checkpoint=self.path)
        self.assertGreater(tracker.opened, 8)
        self.assertEqual(tracker.opened, tracker.closed)

    def test_a_run_without_a_checkpoint_has_no_checkpoint_record(self) -> None:
        source, _ = numpy_source("exponential")
        self.assertIsNone(self.run_fit(source, "exponential").scale.checkpoint)
        self.assertFalse(self.path.exists())

    def test_the_result_of_a_checkpointed_fit_is_the_result_without_one(self) -> None:
        for family in ONE_PASS:
            with self.subTest(family=family):
                source, _ = numpy_source(family)
                base = self.run_fit(source, family)
                result = self.run_fit(source, family, checkpoint=self.root / f"{family}.db")
                self.assertTrue(same_as(result, base))
                self.assertEqual(result.scale.state, base.scale.state)


class MultiPassResumeTests(CheckpointCase):
    def totals(self, family: str) -> tuple[ScaleFitResult, int]:
        source, _ = numpy_source(family)
        base = self.run_fit(source, family)
        assert base.scale.passes is not None
        return base, base.scale.passes.pass_count

    def test_an_interrupted_fit_resumes_within_the_pass_and_from_the_last_completed_pass(
        self,
    ) -> None:
        for family in MULTI_PASS:
            base, count = self.totals(family)
            source, _ = numpy_source(family)
            total = PARTITIONS * count
            points = sorted(
                {
                    point
                    for point in (
                        1,
                        PARTITIONS - 1,
                        PARTITIONS,
                        PARTITIONS + 1,
                        2 * PARTITIONS + 2,
                        total - PARTITIONS,
                        total - 1,
                    )
                    if 0 < point < total
                }
            )
            for after in points:
                with self.subTest(family=family, after=after):
                    self.path = self.root / f"{family}-{after}.db"
                    stopped = self.interrupt(ReadLog(source), family, after=after)
                    self.assertEqual(stopped.partition_commits, after)
                    log = ReadLog(source)
                    result = self.run_fit(
                        log, family, checkpoint=SQLitePartitionStore(self.path), resume=True
                    )
                    self.assertTrue(same_as(result, base))
                    self.assertEqual(len(log.reads), total - after)
                    record = self.checkpoint_of(result)
                    self.assertEqual(record.replayed_passes, after // PARTITIONS)
                    self.assertEqual(
                        [item.replayed for item in record.passes],
                        [index < after // PARTITIONS for index in range(count)],
                    )
                    current = record.passes[after // PARTITIONS]
                    self.assertEqual(current.reused, after % PARTITIONS)
                    self.assertEqual(current.read, PARTITIONS - after % PARTITIONS)

    def test_threads_resume_to_the_same_result_at_a_few_points(self) -> None:
        for family in MULTI_PASS:
            base, count = self.totals(family)
            source, _ = numpy_source(family)
            for after in (2, PARTITIONS * 2 + 1):
                with self.subTest(family=family, after=after):
                    self.path = self.root / f"{family}-t{after}.db"
                    self.interrupt(source, family, executor=ThreadExecutor(3), after=after)
                    result = self.run_fit(
                        source,
                        family,
                        checkpoint=self.path,
                        resume=True,
                        executor=ThreadExecutor(2),
                    )
                    self.assertTrue(same_as(result, base))
                    record = self.checkpoint_of(result)
                    self.assertEqual(record.reused_partitions, after)
                    self.assertEqual(record.replayed_passes, after // PARTITIONS)

    def test_a_failure_between_the_last_state_of_a_pass_and_its_ledger_row_loses_no_read(
        self,
    ) -> None:
        for family in MULTI_PASS:
            base, count = self.totals(family)
            source, _ = numpy_source(family)
            for failed in (0, 1, count - 1):
                with self.subTest(family=family, ledger_rows=failed):
                    self.path = self.root / f"{family}-ledger{failed}.db"
                    self.interrupt(source, family, after_passes=failed)
                    log = ReadLog(source)
                    result = self.run_fit(log, family, checkpoint=self.path, resume=True)
                    self.assertTrue(same_as(result, base))
                    self.assertEqual(len(log.reads), PARTITIONS * (count - failed - 1))
                    record = self.checkpoint_of(result)
                    self.assertEqual(record.replayed_passes, failed)
                    self.assertEqual(record.passes[failed].reused, PARTITIONS)
                    self.assertEqual(record.passes[failed].read, 0)

    def test_a_finished_fit_replays_every_pass_from_the_ledger(self) -> None:
        for family in MULTI_PASS:
            base, count = self.totals(family)
            source, _ = numpy_source(family)
            self.path = self.root / f"{family}.db"
            first = self.run_fit(source, family, checkpoint=self.path)
            self.assertTrue(same_as(first, base))
            log = ReadLog(source)
            again = self.run_fit(log, family, checkpoint=self.path, resume=True)
            self.assertEqual(log.reads, [])
            self.assertTrue(same_as(again, base))
            self.assertEqual(self.checkpoint_of(again).replayed_passes, count)
            self.assertEqual(self.checkpoint_of(again).read_partitions, 0)

    def test_the_pass_sequence_is_the_same_as_the_uninterrupted_one(self) -> None:
        base, count = self.totals("weibull_min")
        source, _ = numpy_source("weibull_min")
        self.interrupt(source, "weibull_min", after=PARTITIONS * 3 + 1)
        result = self.run_fit(source, "weibull_min", checkpoint=self.path, resume=True)
        self.assertEqual(result.scale.passes, base.scale.passes)
        ledger = SQLitePartitionStore(self.path).read_ledger()
        self.assertEqual(sorted(ledger), list(range(count)))

    def test_only_the_states_of_the_latest_pass_are_kept(self) -> None:
        base, count = self.totals("gumbel_right")
        source, _ = numpy_source("gumbel_right")
        self.run_fit(source, "gumbel_right", checkpoint=self.path)
        held = {
            row[0] for row in rows(self.path, "SELECT DISTINCT pass_index FROM partition_state")
        }
        self.assertEqual(held, {count - 1})

    def test_a_multi_pass_fit_in_a_spool_resumes_with_the_spool_it_makes_again(self) -> None:
        values = values_for("weibull_min", ROWS)
        root = self.root / "data"
        for index, piece in enumerate(np.array_split(values, PARTITIONS)):
            write_parquet(root / f"part-{index}.parquet", {"x": piece})
        source = ParquetSource(root, "x")
        base = self.run_fit(source, "weibull_min")
        assert base.scale.passes is not None
        count = base.scale.passes.pass_count
        policy = ScaleSpoolPolicy(max_bytes=1 << 20, directory=self.root)
        for index, limits in enumerate(
            (
                {"after": PARTITIONS + 2},
                {"after": PARTITIONS * (count - 1) + 1},
                {"after_passes": count - 1},
            )
        ):
            with self.subTest(limits=limits):
                self.path = self.root / f"spool-{index}.db"
                store = Interrupting(self.path, **limits)
                with self.assertRaises(Interruption):
                    self.run_fit(source, "weibull_min", checkpoint=store, spool=policy)
                result = self.run_fit(
                    source, "weibull_min", checkpoint=self.path, resume=True, spool=policy
                )
                self.assertTrue(same_as(result, base))
                assert result.scale.spool is not None
                self.assertLessEqual(len(result.scale.spool.partitions), PARTITIONS)
                self.assertEqual(
                    sorted(path.name for path in self.root.glob("veridist-spool-*")), []
                )

    def test_a_fully_stored_last_pass_needs_no_spool_at_all(self) -> None:
        values = values_for("weibull_min", ROWS)
        root = self.root / "data"
        for index, piece in enumerate(np.array_split(values, PARTITIONS)):
            write_parquet(root / f"part-{index}.parquet", {"x": piece})
        source = ParquetSource(root, "x")
        base = self.run_fit(source, "weibull_min")
        assert base.scale.passes is not None
        policy = ScaleSpoolPolicy(max_bytes=1 << 20, directory=self.root)
        store = Interrupting(self.path, after_passes=base.scale.passes.pass_count - 1)
        with self.assertRaises(Interruption):
            self.run_fit(source, "weibull_min", checkpoint=store, spool=policy)
        result = self.run_fit(
            source, "weibull_min", checkpoint=self.path, resume=True, spool=policy
        )
        self.assertTrue(same_as(result, base))
        assert result.scale.spool is not None
        self.assertEqual(result.scale.spool.partitions, ())


class ChangedPartitionTests(CheckpointCase):
    def edited(self, family: str) -> tuple[NumpySource, np.ndarray]:
        values = values_for(family, ROWS)
        values[5] *= 1.5
        return NumpySource(values, partition_rows=ROWS // PARTITIONS), values

    def test_a_changed_partition_is_refused_before_any_read_and_changes_nothing(self) -> None:
        for family in ("exponential", "weibull_min"):
            with self.subTest(family=family):
                self.path = self.root / f"{family}.db"
                source, _ = numpy_source(family)
                self.interrupt(source, family, after=2)
                before = self.path.read_bytes()
                changed, _ = self.edited(family)
                log = ReadLog(changed)
                with self.assertRaises(ScaleSourceError) as caught:
                    self.run_fit(log, family, checkpoint=self.path, resume=True)
                self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)
                self.assertEqual(log.reads, [])
                self.assertEqual(self.path.read_bytes(), before)

    def test_a_changed_partition_is_read_again_when_asked_and_the_fit_is_that_of_the_new_data(
        self,
    ) -> None:
        source, _ = numpy_source("exponential")
        self.run_fit(source, "exponential", checkpoint=self.path)
        changed, _ = self.edited("exponential")
        base = self.run_fit(changed, "exponential")
        log = ReadLog(changed)
        result = self.run_fit(
            log, "exponential", checkpoint=self.path, resume=True, recompute_changed=True
        )
        self.assertTrue(same_as(result, base))
        self.assertEqual(log.reads, [0])
        record = self.checkpoint_of(result)
        self.assertEqual(
            (record.recomputed_partitions, record.restarted, record.read_partitions),
            (1, False, 1),
        )
        self.assertEqual(record.reused_partitions, PARTITIONS - 1)
        again = self.run_fit(ReadLog(changed), "exponential", checkpoint=self.path, resume=True)
        self.assertTrue(same_as(again, base))
        self.assertEqual(self.checkpoint_of(again).replayed_passes, 1)

    def test_a_changed_partition_of_a_multi_pass_fit_restarts_every_pass(self) -> None:
        source, _ = numpy_source("weibull_min")
        self.run_fit(source, "weibull_min", checkpoint=self.path)
        changed, _ = self.edited("weibull_min")
        base = self.run_fit(changed, "weibull_min")
        assert base.scale.passes is not None
        log = ReadLog(changed)
        result = self.run_fit(
            log, "weibull_min", checkpoint=self.path, resume=True, recompute_changed=True
        )
        self.assertTrue(same_as(result, base))
        record = self.checkpoint_of(result)
        self.assertEqual((record.recomputed_partitions, record.restarted), (1, True))
        self.assertEqual(record.replayed_passes, 0)
        self.assertEqual(record.reused_partitions, 0)
        self.assertEqual(len(log.reads), PARTITIONS * base.scale.passes.pass_count)


class IncrementalTests(CheckpointCase):
    def dataset(self, files: range, family: str = "exponential") -> tuple[Path, np.ndarray]:
        """Parquet files ``part-<n>`` of two row groups each, in a directory of their own."""

        root = self.root / "data"
        values = values_for(family, ROWS * 2, seed=9)
        pieces = np.array_split(values, 10)
        for index in files:
            if (root / f"part-{index}.parquet").exists():
                continue
            write_parquet(
                root / f"part-{index}.parquet",
                {"x": pieces[index]},
                row_group_size=len(pieces[index]) // 2,
            )
        return root, np.concatenate([pieces[index] for index in files])

    def test_new_partitions_are_read_and_merged_with_the_stored_ones(self) -> None:
        for family in ONE_PASS:
            with self.subTest(family=family):
                self.path = self.root / f"{family}.db"
                root = self.root / "data"
                if root.exists():
                    for file in root.iterdir():
                        file.unlink()
                self.dataset(range(3), family)
                first = self.run_fit(ParquetSource(root, "x"), family, checkpoint=self.path)
                self.assertEqual(self.checkpoint_of(first).read_partitions, 6)
                self.dataset(range(5), family)
                full = ParquetSource(root, "x")
                base = self.run_fit(full, family)
                log = ReadLog(full)
                result = self.run_fit(
                    log, family, checkpoint=self.path, resume=True, incremental=True
                )
                self.assertTrue(same_as(result, base))
                self.assertEqual(log.reads, [6, 7, 8, 9])
                record = self.checkpoint_of(result)
                self.assertEqual(
                    (record.added_partitions, record.removed_partitions, record.incremental),
                    (4, 0, True),
                )
                self.assertEqual((record.reused_partitions, record.read_partitions), (6, 4))
                again = self.run_fit(ReadLog(full), family, checkpoint=self.path, resume=True)
                self.assertTrue(same_as(again, base))
                self.assertEqual(self.checkpoint_of(again).replayed_passes, 1)

    def test_a_new_partition_without_the_incremental_request_is_refused(self) -> None:
        root, _ = self.dataset(range(3))
        self.run_fit(ParquetSource(root, "x"), "exponential", checkpoint=self.path)
        self.dataset(range(5))
        log = ReadLog(ParquetSource(root, "x"))
        before = self.path.read_bytes()
        with self.assertRaises(ScaleCheckpointError) as caught:
            self.run_fit(log, "exponential", checkpoint=self.path, resume=True)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.PARTITION_ADDED)
        self.assertEqual(log.reads, [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_a_removed_partition_is_refused_unless_it_is_allowed(self) -> None:
        root, _ = self.dataset(range(5))
        self.run_fit(ParquetSource(root, "x"), "exponential", checkpoint=self.path)
        (root / "part-4.parquet").unlink()
        log = ReadLog(ParquetSource(root, "x"))
        before = self.path.read_bytes()
        for options in ({}, {"incremental": True}):
            with self.subTest(options=options), self.assertRaises(ScaleCheckpointError) as caught:
                self.run_fit(log, "exponential", checkpoint=self.path, resume=True, **options)
            self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.PARTITION_MISSING)
        self.assertEqual(log.reads, [])
        self.assertEqual(self.path.read_bytes(), before)
        base = self.run_fit(ParquetSource(root, "x"), "exponential")
        result = self.run_fit(
            log,
            "exponential",
            checkpoint=self.path,
            resume=True,
            incremental=True,
            allow_removed=True,
        )
        self.assertTrue(same_as(result, base))
        self.assertEqual(log.reads, [])
        record = self.checkpoint_of(result)
        self.assertEqual((record.removed_partitions, record.added_partitions), (2, 0))
        self.assertEqual(record.reused_partitions, 8)

    def test_partitions_can_be_added_and_removed_together(self) -> None:
        root, _ = self.dataset(range(3))
        self.run_fit(ParquetSource(root, "x"), "exponential", checkpoint=self.path)
        (root / "part-0.parquet").unlink()
        self.dataset(range(1, 5))
        source = ParquetSource(root, "x")
        base = self.run_fit(source, "exponential")
        result = self.run_fit(
            source,
            "exponential",
            checkpoint=self.path,
            resume=True,
            incremental=True,
            allow_removed=True,
        )
        self.assertTrue(same_as(result, base))
        record = self.checkpoint_of(result)
        self.assertEqual((record.added_partitions, record.removed_partitions), (4, 2))

    def test_an_appended_file_changes_its_row_groups_and_is_a_changed_partition(self) -> None:
        root, _ = self.dataset(range(2))
        self.run_fit(ParquetSource(root, "x"), "exponential", checkpoint=self.path)
        extra = values_for("exponential", 40, seed=1)
        write_parquet(root / "part-0.parquet", {"x": extra}, row_group_size=10)
        with self.assertRaises(ScaleSourceError) as caught:
            self.run_fit(
                ParquetSource(root, "x"),
                "exponential",
                checkpoint=self.path,
                resume=True,
                incremental=True,
            )
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)

    def test_a_multi_pass_family_refuses_an_incremental_fit_before_anything_happens(self) -> None:
        root, _ = self.dataset(range(3), "weibull_min")
        log = ReadLog(ParquetSource(root, "x"))
        for family in MULTI_PASS:
            with self.subTest(family=family), self.assertRaises(ScaleCheckpointError) as caught:
                self.run_fit(log, family, checkpoint=self.path, resume=True, incremental=True)
            self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.INCREMENTAL_UNSUPPORTED)
        self.assertEqual(log.reads, [])
        self.assertFalse(self.path.exists())

    def test_a_multi_pass_job_refuses_a_new_partition_on_resume(self) -> None:
        root, _ = self.dataset(range(3), "weibull_min")
        self.run_fit(ParquetSource(root, "x"), "weibull_min", checkpoint=self.path)
        self.dataset(range(5), "weibull_min")
        with self.assertRaises(ScaleCheckpointError) as caught:
            self.run_fit(ParquetSource(root, "x"), "weibull_min", checkpoint=self.path, resume=True)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.PARTITION_ADDED)
        (root / "part-4.parquet").unlink()
        (root / "part-3.parquet").unlink()
        (root / "part-2.parquet").unlink()
        with self.assertRaises(ScaleCheckpointError) as caught:
            self.run_fit(ParquetSource(root, "x"), "weibull_min", checkpoint=self.path, resume=True)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.PARTITION_MISSING)


class JobIdentityTests(CheckpointCase):
    def refused(self, source: Any, family: str, *names: str, **options: Any) -> None:
        before = self.path.read_bytes()
        with self.assertRaises(ScaleCheckpointError) as caught:
            self.bare(source, family, checkpoint=self.path, resume=True, **options)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.JOB_MISMATCH)
        self.assertEqual(caught.exception.components, names)
        if isinstance(source, ReadLog):
            self.assertEqual(source.reads, [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_a_different_family_batch_size_source_column_or_level_is_refused_by_name(self) -> None:
        values = values_for("exponential", ROWS)
        root = self.root / "data"
        write_parquet(root / "a.parquet", {"x": values, "y": values}, row_group_size=ROWS // 4)
        self.bare(ParquetSource(root, "x"), "exponential", checkpoint=self.path)
        source = ParquetSource(root, "x")
        self.refused(source, "normal", "family", "state_schema")
        self.refused(source, "exponential", "max_rows", max_rows=MAX_ROWS + 1)
        self.refused(ParquetSource(root, "y"), "exponential", "column")
        self.refused(ReadLog(source), "exponential", "source_kind")
        self.refused(
            ParquetSource(root, "x", content_hash=True), "exponential", "fingerprint_level"
        )

    def test_a_multi_pass_job_is_not_a_one_pass_job(self) -> None:
        source, _ = numpy_source("weibull_min")
        self.bare(source, "weibull_min", checkpoint=self.path)
        self.refused(source, "gumbel_right", "family", "state_schema")
        self.refused(source, "exponential", "family", "state_schema", "plan")


class FlagTests(CheckpointCase):
    def test_a_store_that_holds_a_job_is_not_overwritten_without_resume(self) -> None:
        source, _ = numpy_source("exponential")
        self.run_fit(source, "exponential", checkpoint=self.path)
        before = self.path.read_bytes()
        log = ReadLog(source)
        with self.assertRaises(ScaleCheckpointError) as caught:
            self.run_fit(log, "exponential", checkpoint=self.path)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.JOB_EXISTS)
        self.assertEqual(log.reads, [])
        self.assertEqual(self.path.read_bytes(), before)

    def test_resuming_without_a_job_starts_one(self) -> None:
        source, _ = numpy_source("exponential")
        result = self.run_fit(source, "exponential", checkpoint=self.path, resume=True)
        record = self.checkpoint_of(result)
        self.assertEqual((record.resumed, record.read_partitions), (False, PARTITIONS))
        self.assertTrue(self.path.exists())

    def test_argument_errors_come_before_any_read_or_any_file(self) -> None:
        source, _ = numpy_source("exponential")
        log = ReadLog(source)
        calls: list[tuple[type[Exception], dict[str, Any]]] = [
            (ValueError, {"resume": True}),
            (ValueError, {"incremental": True}),
            (ValueError, {"recompute_changed": True}),
            (ValueError, {"allow_removed": True}),
            (ValueError, {"checkpoint": self.path, "incremental": True}),
            (ValueError, {"checkpoint": self.path, "allow_removed": True}),
            (TypeError, {"checkpoint": 3}),
            (TypeError, {"checkpoint": self.path, "resume": "yes"}),
        ]
        for error, options in calls:
            with self.subTest(options=options), self.assertRaises(error):
                self.run_fit(log, "exponential", **options)
        self.assertEqual(log.reads, [])
        self.assertFalse(self.path.exists())


class IntegrityTests(CheckpointCase):
    def test_an_altered_state_is_never_used(self) -> None:
        source, _ = numpy_source("exponential")
        self.interrupt(source, "exponential", after=3)
        execute(self.path, "UPDATE partition_state SET state = ? WHERE rowid = 2", b"altered")
        log = ReadLog(source)
        with self.assertRaises(ScaleCheckpointError) as caught:
            self.run_fit(log, "exponential", checkpoint=self.path, resume=True)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.STORE_DIGEST_MISMATCH)
        self.assertEqual(log.reads, [])

    def test_an_altered_ledger_is_never_replayed(self) -> None:
        source, _ = numpy_source("weibull_min")
        self.run_fit(source, "weibull_min", checkpoint=self.path)
        execute(self.path, "UPDATE pass_ledger SET merged = ? WHERE pass_index = 1", b"altered")
        log = ReadLog(source)
        with self.assertRaises(ScaleCheckpointError) as caught:
            self.run_fit(log, "weibull_min", checkpoint=self.path, resume=True)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.STORE_DIGEST_MISMATCH)
        self.assertEqual(log.reads, [])

    def test_a_ledger_whose_parameters_are_not_the_ones_the_planner_derives_is_refused(
        self,
    ) -> None:
        source, _ = numpy_source("weibull_min")
        self.run_fit(source, "weibull_min", checkpoint=self.path)
        other = '["profile"]'
        key = hashlib.sha256(other.encode()).hexdigest()
        merged = rows(self.path, "SELECT merged_sha256 FROM pass_ledger WHERE pass_index = 2")[0][0]
        execute(
            self.path,
            "UPDATE pass_ledger SET pass_key = ?, descriptor = ?, checksum = ? "
            "WHERE pass_index = 2",
            key,
            other,
            _row_checksum("ledger", 2, key, other, merged),
        )
        with self.assertRaises(ScaleCheckpointError) as caught:
            self.run_fit(source, "weibull_min", checkpoint=self.path, resume=True)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.LEDGER_MISMATCH)

    def test_a_single_row_checkpoint_database_is_not_taken_for_a_job(self) -> None:
        make_single_row_database(self.path)
        source, _ = numpy_source("exponential")
        with self.assertRaises(ScaleCheckpointError) as caught:
            self.run_fit(source, "exponential", checkpoint=self.path, resume=True)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.STORE_FORMAT_UNSUPPORTED)


class UnresumableSourceTests(CheckpointCase):
    def test_an_arrow_table_cannot_be_checkpointed(self) -> None:
        values = values_for("exponential", ROWS)
        table = pa.table({"x": values})
        for family in ("exponential", "weibull_min"):
            for options in ({}, {"resume": True}, {"resume": True, "incremental": True}):
                with self.subTest(family=family, options=options):
                    source = ArrowSource(table, "x")
                    with self.assertRaises(ScaleCheckpointError) as caught:
                        self.run_fit(source, family, checkpoint=self.path, **options)
                    expected = (
                        ScaleCheckpointErrorCode.INCREMENTAL_UNSUPPORTED
                        if options.get("incremental") and family == "weibull_min"
                        else ScaleCheckpointErrorCode.SOURCE_REVISION_UNAVAILABLE
                    )
                    self.assertIs(caught.exception.code, expected)
                    self.assertFalse(self.path.exists())

    def test_an_arrow_stream_is_refused_for_a_checkpoint_before_a_spool_is_asked_about(
        self,
    ) -> None:
        values = values_for("weibull_min", ROWS)
        reader = pa.RecordBatchReader.from_batches(
            pa.schema([("x", pa.float64())]), pa.table({"x": values}).to_batches()
        )
        with self.assertRaises(ScaleCheckpointError) as caught:
            self.run_fit(ArrowSource(reader, "x"), "weibull_min", checkpoint=self.path)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.SOURCE_REVISION_UNAVAILABLE)
        self.assertFalse(self.path.exists())
        with self.assertRaises(ScaleCheckpointError):
            self.run_fit(
                ArrowSource(
                    pa.RecordBatchReader.from_batches(
                        pa.schema([("x", pa.float64())]), pa.table({"x": values}).to_batches()
                    ),
                    "x",
                ),
                "weibull_min",
                checkpoint=self.path,
                spool=ScaleSpoolPolicy(max_bytes=1 << 20, directory=self.root),
            )


class ExecutionRecordTests(CheckpointCase):
    def test_a_replayed_pass_is_recorded_as_a_pass_with_no_task(self) -> None:
        source, _ = numpy_source("exponential")
        self.run_fit(source, "exponential", checkpoint=self.path)
        result = self.run_fit(
            source, "exponential", checkpoint=self.path, resume=True, executor=ThreadExecutor(3)
        )
        assert result.scale.execution is not None
        (record,) = result.scale.execution.passes
        self.assertEqual(
            (record.tasks, record.executor, record.downgrade), (0, "serial", "single_partition")
        )

    def test_the_tasks_of_a_pass_are_the_partitions_that_are_read(self) -> None:
        source, _ = numpy_source("normal")
        self.interrupt(source, "normal", after=1, executor=ThreadExecutor(3))
        result = self.run_fit(
            source, "normal", checkpoint=self.path, resume=True, executor=ThreadExecutor(3)
        )
        assert result.scale.execution is not None
        (record,) = result.scale.execution.passes
        self.assertEqual((record.tasks, record.executor, record.workers), (3, "threads", 3))
        self.assertIsNone(record.downgrade)

    def test_the_checkpoint_record_replaces_nothing_else_in_the_provenance(self) -> None:
        source, _ = numpy_source("exponential")
        base = self.run_fit(source, "exponential")
        result = self.run_fit(source, "exponential", checkpoint=self.path)
        self.assertEqual(replace(result.scale, checkpoint=None), base.scale)
