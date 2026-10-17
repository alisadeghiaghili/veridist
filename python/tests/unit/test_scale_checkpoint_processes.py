"""One real process pool: a fit interrupted in the parent and resumed with worker processes.

Workers return state bytes and the parent is the only writer of the store; everything else
about checkpoints is tested without starting processes (``test_scale_checkpoint_fit``).
"""

from __future__ import annotations

import multiprocessing
import tempfile
from pathlib import Path

import numpy as np

from tests.unit.scale_checkpoint_fixtures import Interrupting, Interruption, same_as
from tests.unit.scale_executor_fixtures import NoLeaksTestCase, parquet_dataset
from veridist.scale._executors import ProcessExecutor
from veridist.scale._fit import fit


class ProcessCheckpointTests(NoLeaksTestCase):
    real_processes = True

    def test_workers_return_states_and_the_parent_commits_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.random.default_rng(3).exponential(3.0, 600)
            source = parquet_dataset(root / "data", values)
            base = fit(source, "exponential")
            path = root / "job.db"
            store = Interrupting(path, after=3)
            with self.assertRaises(Interruption):
                fit(source, "exponential", checkpoint=store, executor=ProcessExecutor(2))
            self.assertEqual(multiprocessing.active_children(), [])
            result = fit(
                source,
                "exponential",
                checkpoint=path,
                resume=True,
                executor=ProcessExecutor(2),
            )
            self.assertTrue(same_as(result, base))
            assert result.scale.checkpoint is not None and result.scale.execution is not None
            self.assertEqual(result.scale.checkpoint.reused_partitions, 3)
            self.assertEqual(result.scale.checkpoint.read_partitions, 3)
            (record,) = result.scale.execution.passes
            self.assertEqual((record.executor, record.tasks), ("processes", 3))
