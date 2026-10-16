"""A few million rows, interrupted and resumed several times, give the uninterrupted fit.

A one-pass fit over 3 million rows and a multi-pass fit over 300 thousand are each stopped at
three commits of the parent and continued, serially and on threads. The result bits and the
state digest equal those of the fit that was never stopped, no partition is read twice, and
the whole run stays within its time limit.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from tests.unit.scale_checkpoint_fixtures import Interrupting, Interruption, bits, same_as
from tests.unit.scale_executor_fixtures import NoLeaksTestCase, parquet_dataset
from tests.unit.scale_multi_pass_oracle import weibull_sample
from veridist.scale._arrow import ParquetSource
from veridist.scale._executors import ThreadExecutor
from veridist.scale._fit import ScaleFitResult, fit

LIMIT_SECONDS = 30.0
ONE_PASS_ROWS = 3_000_000
MULTI_PASS_ROWS = 300_000
FILES = 4
ROW_GROUPS = 3
PARTITIONS = FILES * ROW_GROUPS


class CheckpointSweepTests(NoLeaksTestCase):
    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.started = time.monotonic()

    def stopped_and_resumed(
        self, source: ParquetSource, family: str, stops: tuple[int, ...], executor: Any
    ) -> ScaleFitResult:
        path = self.root / f"{family}.db"
        for after in stops:
            store = Interrupting(path, after=after)
            with self.assertRaises(Interruption):
                fit(source, family, checkpoint=store, resume=path.exists(), executor=executor)
        return fit(source, family, checkpoint=path, resume=True, executor=executor)

    def test_a_one_pass_fit_over_millions_of_rows_resumes_three_times(self) -> None:
        rng = np.random.default_rng(31)
        source = parquet_dataset(
            self.root / "data", rng.lognormal(1.0, 0.5, ONE_PASS_ROWS), FILES, ROW_GROUPS
        )
        base = fit(source, "lognormal")
        self.assertEqual(base.scale.row_count, ONE_PASS_ROWS)
        for executor in (None, ThreadExecutor(4)):
            with self.subTest(executor=executor):
                result = self.stopped_and_resumed(source, "lognormal", (2, 3, 4), executor)
                self.assertTrue(same_as(result, base))
                self.assertEqual(bits(result), bits(base))
                self.assertEqual(result.scale.state, base.scale.state)
                assert result.scale.checkpoint is not None
                self.assertEqual(result.scale.checkpoint.reused_partitions, 2 + 3 + 4)
                self.assertEqual(result.scale.checkpoint.read_partitions, PARTITIONS - 9)
                (self.root / "lognormal.db").unlink()
        self.assertLess(time.monotonic() - self.started, LIMIT_SECONDS)

    def test_a_multi_pass_fit_resumes_three_times_inside_and_across_passes(self) -> None:
        source = parquet_dataset(
            self.root / "data", weibull_sample(1.7, 40.0, MULTI_PASS_ROWS, 32), FILES, ROW_GROUPS
        )
        base = fit(source, "weibull_min")
        for executor in (None, ThreadExecutor(4)):
            with self.subTest(executor=executor):
                result = self.stopped_and_resumed(
                    source, "weibull_min", (5, PARTITIONS, PARTITIONS + 7), executor
                )
                self.assertTrue(same_as(result, base))
                self.assertEqual(bits(result), bits(base))
                self.assertEqual(result.scale.state, base.scale.state)
                assert result.scale.checkpoint is not None
                self.assertGreaterEqual(result.scale.checkpoint.replayed_passes, 2)
                (self.root / "weibull_min.db").unlink()
        self.assertLess(time.monotonic() - self.started, LIMIT_SECONDS)
