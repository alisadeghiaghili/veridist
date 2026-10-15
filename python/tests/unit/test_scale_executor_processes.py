"""Real worker processes: a few fits and one pool, as few as the contract can be shown with.

Everything else about executors is tested in this process with a fake or a thread pool
(``test_scale_executors``); starting processes is slow, and a process that was forked by
the mutation runner needs a fresh interpreter first (``tests/process_isolation.py``).
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from tests.unit.scale_executor_fixtures import (
    ChangedRows,
    NoLeaksTestCase,
    PoolRecorder,
    parquet_dataset,
    worker_facts,
)
from tests.unit.scale_multi_pass_oracle import weibull_sample
from veridist.engine.provenance import ScaleExecutionPassProvenance
from veridist.scale._errors import (
    ScaleSourceError,
    ScaleSourceErrorCode,
    ScaleStateError,
    ScaleStateErrorCode,
)
from veridist.scale._executors import (
    ProcessExecutor,
    make_process_pool,
)
from veridist.scale._fit import fit
from veridist.scale._spool import SpoolPolicy

WORKER_SECONDS = 120


def counting_factory() -> PoolRecorder:
    """A factory of real process pools that remembers how often it was asked."""

    return PoolRecorder(lambda workers, modules: make_process_pool(workers, modules))  # type: ignore[arg-type,return-value]


class ProcessTests(NoLeaksTestCase):
    real_processes = True

    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def dataset(self, family: str = "weibull_min", size: int = 480) -> Any:
        rng = np.random.default_rng(11)
        values = (
            rng.exponential(3.0, size)
            if family == "exponential"
            else weibull_sample(1.7, 40.0, size, 11)
        )
        return parquet_dataset(self.root / family, values), values

    def test_a_process_pool_gives_the_serial_result_bit_for_bit(self) -> None:
        source, _ = self.dataset("exponential")
        base = fit(source, "exponential")
        recorder = counting_factory()
        result = fit(source, "exponential", executor=ProcessExecutor(2, pool_factory=recorder))
        self.assertEqual(result.fit, base.fit)
        self.assertEqual(result.scale.state, base.scale.state)
        self.assertEqual(
            result.fit.uncertainty().covariance,  # type: ignore[union-attr]
            base.fit.uncertainty().covariance,  # type: ignore[union-attr]
        )
        assert result.scale.execution is not None
        self.assertEqual(
            result.scale.execution.passes,
            (ScaleExecutionPassProvenance("processes", 2, 6, 0, None),),
        )
        self.assertEqual(recorder.calls, [(2, ("numpy", "pyarrow", "pyarrow.parquet"))])
        self.assertEqual(multiprocessing.active_children(), [])

    def test_one_pool_serves_every_pass_of_a_multi_pass_fit(self) -> None:
        source, _ = self.dataset()
        base = fit(source, "weibull_min")
        recorder = counting_factory()
        result = fit(source, "weibull_min", executor=ProcessExecutor(2, pool_factory=recorder))
        self.assertEqual(result.fit, base.fit)
        self.assertEqual(result.scale.state, base.scale.state)
        assert result.scale.execution is not None and result.scale.passes is not None
        self.assertGreaterEqual(result.scale.passes.pass_count, 3)
        self.assertEqual(len(result.scale.execution.passes), result.scale.passes.pass_count)
        self.assertEqual(len(recorder.calls), 1)
        self.assertEqual(multiprocessing.active_children(), [])

    def test_failures_in_a_worker_are_the_typed_errors_of_the_serial_path(self) -> None:
        for name, edit, error_type, code in (
            ("negative", -1.0, ScaleStateError, ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT),
            ("nan", float("nan"), ScaleSourceError, ScaleSourceErrorCode.NAN_VALUE),
        ):
            values = np.random.default_rng(2).exponential(3.0, 240)
            values[100] = edit
            source = parquet_dataset(self.root / name, values)
            with self.subTest(case=name):
                with self.assertRaises(error_type) as serial:
                    fit(source, "exponential")
                with self.assertRaises(error_type) as parallel:
                    fit(source, "exponential", executor=ProcessExecutor(2))
                self.assertIs(parallel.exception.code, code)  # type: ignore[attr-defined]
                self.assertIs(parallel.exception.code, serial.exception.code)  # type: ignore[attr-defined]
                self.assertEqual(str(parallel.exception), str(serial.exception))
                self.assertEqual(multiprocessing.active_children(), [])

    def test_a_partition_that_changed_is_partition_changed_from_a_worker(self) -> None:
        source, _ = self.dataset("exponential", 240)
        with self.assertRaises(ScaleSourceError) as caught:
            fit(ChangedRows(source), "exponential", executor=ProcessExecutor(2))
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)
        self.assertEqual(multiprocessing.active_children(), [])

    def test_a_spooled_fit_reads_the_spool_in_parallel_after_the_first_pass(self) -> None:
        source, _ = self.dataset()
        base = fit(source, "weibull_min")
        spool_directory = self.root / "spool"
        spool_directory.mkdir()
        result = fit(
            source,
            "weibull_min",
            executor=ProcessExecutor(2),
            spool=SpoolPolicy(max_bytes=10_000_000, directory=spool_directory),
        )
        self.assertEqual(result.fit, base.fit)
        self.assertEqual(result.scale.state, base.scale.state)
        assert result.scale.execution is not None
        first, *later = result.scale.execution.passes
        self.assertEqual(first, ScaleExecutionPassProvenance("serial", 1, 6, 0, "spool_first_pass"))
        self.assertTrue(later)
        for record in later:
            self.assertEqual(record, ScaleExecutionPassProvenance("processes", 2, 6, 0, None))
        self.assertEqual(os.listdir(spool_directory), [])
        self.assertEqual(multiprocessing.active_children(), [])

    def test_a_worker_is_spawned_imports_its_modules_and_pins_arrow(self) -> None:
        pool = make_process_pool(1, ("numpy", "pyarrow", "pyarrow.parquet"))
        try:
            facts = pool.submit(worker_facts).result(timeout=WORKER_SECONDS)
            self.assertEqual(pool._mp_context.get_start_method(), "spawn")  # type: ignore[attr-defined]
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        self.assertNotEqual(facts["pid"], os.getpid())
        self.assertTrue(facts["numpy"] and facts["pyarrow"] and facts["parquet"])
        self.assertEqual((facts["arrow_cpus"], facts["arrow_io"]), (1, 1))
        self.assertFalse(facts["scipy"])
