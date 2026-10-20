"""Inference passes are checkpointable passes: interrupt anywhere, resume, get the same result.

Every interruption is a failure injected at a commit of the parent (``Interrupting``), so it lands
at the same point whether the partitions are read serially or on threads. A resumed run is bit for
bit the run that was never interrupted, reads only the partitions that have no committed state,
replays the passes of its ledger without reading them, and keeps a job of its own: one store holds
one job, so the store of the fit is not the store of its inference.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from tests.unit.scale_checkpoint_fixtures import Interrupting, Interruption, ReadLog
from tests.unit.scale_executor_fixtures import NoLeaksTestCase
from tests.unit.scale_fit_oracle import ListSource
from tests.unit.scale_inference_fixtures import gof_bits, profile_bits, without_execution
from tests.unit.scale_inference_oracle import weibull_data
from veridist.scale import _gof as gof
from veridist.scale import _profile_intervals as pi
from veridist.scale._errors import (
    ScaleCheckpointError,
    ScaleCheckpointErrorCode,
    ScaleSourceError,
    ScaleSourceErrorCode,
)
from veridist.scale._executors import ThreadExecutor
from veridist.scale._fit import fit
from veridist.scale._partition_store import SQLitePartitionStore
from veridist.scale._sources import NumpySource

PARTITIONS = 4


class InferenceCheckpointCase(NoLeaksTestCase):
    """The data are small: the tests of the same behaviour at a larger size and at every
    interruption point are in ``tests/statistical/test_scale_inference_full_size.py``."""

    SIZE = 160
    MAX_ROWS = 40
    QUANTITIES: tuple[pi.Quantity, ...] = ()

    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.values = weibull_data(1.7, 40.0, self.SIZE, 8)
        self.source = NumpySource(self.values, partition_rows=self.SIZE // PARTITIONS)
        self.result = fit(self.source, "weibull_min")

    def profile(self, source: Any, **options: Any) -> pi.ProfileIntervals:
        options = {"quantities": list(self.QUANTITIES), "max_rows": self.MAX_ROWS, **options}
        return pi.profile_intervals(source, self.result, **options)

    def record(self, out: Any) -> Any:
        assert out.scale.checkpoint is not None
        return out.scale.checkpoint


class ProfileResumeTests(InferenceCheckpointCase):
    def check_resumes(self, executor: Any, commits: Callable[[int], Iterable[int]]) -> None:
        base = self.profile(ReadLog(self.source))
        assert base.scale.passes is not None
        total = base.scale.passes.pass_count * PARTITIONS
        self.assertGreaterEqual(total, 8)
        for after in commits(total):
            with self.subTest(executor=executor, after=after):
                path = self.root / f"p-{after}-{type(executor).__name__}.db"
                store = Interrupting(path, after=after)
                with self.assertRaises(Interruption):
                    self.profile(ReadLog(self.source), checkpoint=store, executor=executor)
                self.assertEqual(store.partition_commits, after)
                log = ReadLog(self.source)
                out = self.profile(
                    log, checkpoint=SQLitePartitionStore(path), resume=True, executor=executor
                )
                self.assertEqual(profile_bits(out), profile_bits(base))
                self.assertEqual(without_execution(out), without_execution(base))
                self.assertEqual(len(log.reads), total - after)
                record = self.record(out)
                self.assertTrue(record.resumed)
                self.assertEqual(record.replayed_passes, after // PARTITIONS)
                self.assertEqual(record.reused_partitions, after)
                self.assertEqual(record.read_partitions, total - after)

    def test_a_run_interrupted_inside_the_first_pass_resumes_to_the_same_result(self) -> None:
        self.check_resumes(None, lambda total: [1])

    def test_a_run_interrupted_at_the_end_of_a_pass_resumes_to_the_same_result(self) -> None:
        self.check_resumes(None, lambda total: [PARTITIONS])

    def test_a_run_interrupted_before_its_last_commit_resumes_to_the_same_result(self) -> None:
        self.check_resumes(None, lambda total: [total - 1])

    def test_a_run_on_threads_interrupted_inside_a_pass_resumes_to_the_same_result(self) -> None:
        self.check_resumes(ThreadExecutor(3), lambda total: [PARTITIONS + 2])

    def test_a_completed_job_is_replayed_without_reading(self) -> None:
        base = self.profile(ReadLog(self.source), checkpoint=self.root / "done.db")
        log = ReadLog(self.source)
        out = self.profile(log, checkpoint=self.root / "done.db", resume=True)
        self.assertEqual(profile_bits(out), profile_bits(base))
        self.assertEqual(log.reads, [])
        assert out.scale.passes is not None
        self.assertEqual(self.record(out).replayed_passes, out.scale.passes.pass_count)
        self.assertEqual(len(self.record(out).passes), out.scale.passes.pass_count)

    def test_an_interrupted_ledger_commit_is_resumed_from_the_stored_partitions(self) -> None:
        base = self.profile(ReadLog(self.source))
        path = self.root / "ledger.db"
        store = Interrupting(path, after_passes=1)
        with self.assertRaises(Interruption):
            self.profile(ReadLog(self.source), checkpoint=store)
        log = ReadLog(self.source)
        out = self.profile(log, checkpoint=SQLitePartitionStore(path), resume=True)
        self.assertEqual(profile_bits(out), profile_bits(base))
        self.assertEqual(len(log.reads), (len(base.scale.passes.records) - 2) * PARTITIONS)  # type: ignore[union-attr]

    def test_the_passes_of_a_run_with_a_checkpoint_are_those_of_a_run_without_one(self) -> None:
        base = self.profile(self.source)
        out = self.profile(self.source, checkpoint=self.root / "same.db")
        self.assertEqual(profile_bits(out), profile_bits(base))
        self.assertEqual(without_execution(out), without_execution(base))
        self.assertFalse(self.record(out).resumed)
        self.assertEqual(self.record(out).replayed_passes, 0)


class RefusalTests(InferenceCheckpointCase):
    def refused(self, code: ScaleCheckpointErrorCode, call: Any) -> ScaleCheckpointError:
        with self.assertRaises(ScaleCheckpointError) as caught:
            call()
        self.assertIs(caught.exception.code, code)
        return caught.exception

    def test_an_existing_job_is_not_overwritten(self) -> None:
        path = self.root / "job.db"
        self.profile(self.source, checkpoint=path)
        self.refused(
            ScaleCheckpointErrorCode.JOB_EXISTS, lambda: self.profile(self.source, checkpoint=path)
        )

    def test_the_store_of_the_fit_is_another_job(self) -> None:
        path = self.root / "fit.db"
        fitted = fit(self.source, "weibull_min", checkpoint=path)
        error = self.refused(
            ScaleCheckpointErrorCode.JOB_MISMATCH,
            lambda: pi.profile_intervals(self.source, fitted, checkpoint=path, resume=True),
        )
        self.assertIn("family", error.components)
        self.assertIn("state_schema", error.components)

    def test_another_level_or_quantity_is_another_job(self) -> None:
        path = self.root / "job.db"
        quantities = [pi.quantile(0.1)]
        with self.assertRaises(Interruption):
            self.profile(
                self.source,
                quantities=quantities,
                checkpoint=Interrupting(path, after=1),
            )
        for options in ({"level": 0.9}, {"quantities": [pi.quantile(0.2)]}):
            with self.subTest(options=options):
                error = self.refused(
                    ScaleCheckpointErrorCode.JOB_MISMATCH,
                    lambda options=options: self.profile(
                        self.source,
                        checkpoint=path,
                        resume=True,
                        **{"quantities": quantities, **options},
                    ),
                )
                self.assertEqual(error.components, ("plan",))

    def test_a_goodness_of_fit_store_is_not_a_profile_store(self) -> None:
        path = self.root / "gof.db"
        gof.large_n_gof(self.source, self.result, max_rows=self.MAX_ROWS, checkpoint=path)
        self.refused(
            ScaleCheckpointErrorCode.JOB_MISMATCH,
            lambda: self.profile(self.source, checkpoint=path, resume=True),
        )

    def test_a_changed_partition_is_refused_unless_it_is_read_again(self) -> None:
        path = self.root / "changed.db"
        self.profile(self.source, checkpoint=path)
        changed = self.values.copy()
        changed[5] += 1.0
        with self.assertRaises(ScaleSourceError) as caught:
            self.profile(
                NumpySource(changed, partition_rows=self.SIZE // PARTITIONS),
                checkpoint=path,
                resume=True,
            )
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)
        # read again, the changed rows are not those of the fit: the first sum says so
        with self.assertRaises(ScaleSourceError) as caught:
            self.profile(
                NumpySource(changed, partition_rows=self.SIZE // PARTITIONS),
                checkpoint=path,
                resume=True,
                recompute_changed=True,
            )
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)

    def test_a_source_without_fingerprints_cannot_be_checkpointed(self) -> None:
        source = ListSource([self.values[:400], self.values[400:]])
        self.refused(
            ScaleCheckpointErrorCode.SOURCE_REVISION_UNAVAILABLE,
            lambda: self.profile(source, checkpoint=self.root / "none.db"),
        )
        self.refused(
            ScaleCheckpointErrorCode.SOURCE_REVISION_UNAVAILABLE,
            lambda: gof.large_n_gof(source, self.result, checkpoint=self.root / "none.db"),
        )
        self.assertFalse((self.root / "none.db").exists())

    def test_the_store_may_be_given_as_an_object(self) -> None:
        store = SQLitePartitionStore(self.root / "object.db")
        out = self.profile(self.source, checkpoint=store)
        self.assertEqual(self.record(out).passes[0].partitions, PARTITIONS)


class GofResumeTests(InferenceCheckpointCase):
    def check_resumes(self, executor: Any, afters: range | tuple[int, ...]) -> None:
        base = gof.large_n_gof(
            ReadLog(self.source), self.result, bin_exponent=8, max_rows=self.MAX_ROWS
        )
        for after in afters:
            with self.subTest(executor=executor, after=after):
                path = self.root / f"g-{after}-{type(executor).__name__}.db"
                with self.assertRaises(Interruption):
                    gof.large_n_gof(
                        ReadLog(self.source),
                        self.result,
                        bin_exponent=8,
                        max_rows=self.MAX_ROWS,
                        checkpoint=Interrupting(path, after=after),
                        executor=executor,
                    )
                log = ReadLog(self.source)
                out = gof.large_n_gof(
                    log,
                    self.result,
                    bin_exponent=8,
                    max_rows=self.MAX_ROWS,
                    checkpoint=SQLitePartitionStore(path),
                    resume=True,
                    executor=executor,
                )
                self.assertEqual(gof_bits(out), gof_bits(base))
                self.assertEqual(len(log.reads), PARTITIONS - after)
                self.assertEqual(self.record(out).reused_partitions, after)

    def test_a_resumed_report_equals_the_uninterrupted_one_serially(self) -> None:
        self.check_resumes(None, (1, PARTITIONS - 1))

    def test_a_resumed_report_equals_the_uninterrupted_one_on_threads(self) -> None:
        self.check_resumes(ThreadExecutor(2), (2,))

    def test_a_completed_report_is_replayed_without_reading(self) -> None:
        path = self.root / "gdone.db"
        base = gof.large_n_gof(ReadLog(self.source), self.result, bin_exponent=8, checkpoint=path)
        log = ReadLog(self.source)
        out = gof.large_n_gof(log, self.result, bin_exponent=8, checkpoint=path, resume=True)
        self.assertEqual(gof_bits(out), gof_bits(base))
        self.assertEqual(log.reads, [])
        self.assertEqual(self.record(out).replayed_passes, 1)

    def test_another_number_of_bins_is_another_job(self) -> None:
        path = self.root / "gbins.db"
        gof.large_n_gof(self.source, self.result, bin_exponent=8, checkpoint=path)
        with self.assertRaises(ScaleCheckpointError) as caught:
            gof.large_n_gof(self.source, self.result, bin_exponent=9, checkpoint=path, resume=True)
        self.assertIs(caught.exception.code, ScaleCheckpointErrorCode.JOB_MISMATCH)
        self.assertEqual(caught.exception.components, ("plan",))
