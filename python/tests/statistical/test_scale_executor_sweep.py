"""Serial, thread and process executors agree bit for bit on multi-file Parquet.

For one-pass and multi-pass families, over files and row groups, with 1, 2 and 4 workers, the
result bits, the state digest and the whole provenance but the execution record are those of
the serial fit, and the pool leaves nothing running. Starting processes is slow, so the
process runs cover four families and the thread runs cover all six.
"""

from __future__ import annotations

import multiprocessing
import struct
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tests.unit.scale_executor_fixtures import NoLeaksTestCase, parquet_dataset
from tests.unit.scale_multi_pass_oracle import gumbel_sample, weibull_sample
from veridist.scale._executors import ProcessExecutor, ThreadExecutor
from veridist.scale._fit import ScaleFitResult, fit

THREAD_FAMILIES = ("exponential", "normal", "gamma", "lognormal", "weibull_min", "gumbel_right")
PROCESS_FAMILIES = ("exponential", "gamma", "weibull_min", "gumbel_right")
WORKERS = (1, 2, 4)
SIZE = 6000


def values_of(family: str) -> np.ndarray:
    rng = np.random.default_rng(21)
    if family == "exponential":
        return rng.exponential(3.0, SIZE)
    if family == "normal":
        return rng.normal(10.0, 3.0, SIZE)
    if family == "gamma":
        return rng.gamma(3.0, 2.0, SIZE)
    if family == "lognormal":
        return rng.lognormal(1.0, 0.5, SIZE)
    if family == "weibull_min":
        return weibull_sample(1.7, 40.0, SIZE, 21)
    return gumbel_sample(5.0, 2.0, SIZE, 21)


def bits(result: ScaleFitResult) -> tuple[bytes, ...]:
    success: Any = result.fit
    numbers = [*success.parameters.values(), success.log_likelihood]
    uncertainty = success.uncertainty()
    for row in uncertainty.covariance:
        numbers.extend(row)
    for lower, upper in uncertainty.confidence_intervals(0.95, "wald").values():
        numbers.extend((lower, upper))
    return tuple(struct.pack("<d", float(value)) for value in numbers)


class ExecutorSweepTests(NoLeaksTestCase):
    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def check(self, make: Any, families: tuple[str, ...]) -> None:
        for family in families:
            source = parquet_dataset(self.root / family, values_of(family), files=4, row_groups=3)
            base = fit(source, family)
            for workers in WORKERS:
                with self.subTest(family=family, executor=make.__name__, workers=workers):
                    result = fit(source, family, executor=make(workers))
                    self.assertEqual(result.fit, base.fit)
                    self.assertEqual(bits(result), bits(base))
                    self.assertEqual(result.scale.state, base.scale.state)
                    self.assertEqual(replace(result.scale, execution=None), base.scale)
                    assert result.scale.execution is not None
                    self.assertEqual(result.scale.execution.requested_workers, workers)
                    self.assertIsNone(result.scale.execution.downgrade)
                    self.assertEqual(result.scale.execution.retries, 0)
                    self.assertEqual(multiprocessing.active_children(), [])

    def test_threads(self) -> None:
        self.check(ThreadExecutor, THREAD_FAMILIES)

    def test_processes(self) -> None:
        self.check(ProcessExecutor, PROCESS_FAMILIES)
