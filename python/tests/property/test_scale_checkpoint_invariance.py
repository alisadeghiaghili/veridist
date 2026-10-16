"""Seeded sweep: any interruption, any layout, any executor, the same result as no interruption.

For random families, partition layouts, batch caps and executors, a fit that is interrupted at
a random commit (once, or twice) and resumed is bit for bit the fit that was never
interrupted, and reads exactly the partitions that were not committed.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

from tests.unit.scale_checkpoint_fixtures import (
    Interrupting,
    Interruption,
    ReadLog,
    same_as,
    values_for,
)
from tests.unit.scale_executor_fixtures import NoLeaksTestCase
from veridist.scale._executors import ThreadExecutor
from veridist.scale._fit import fit
from veridist.scale._sources import NumpySource

FAMILIES = ("exponential", "normal", "gamma", "lognormal", "weibull_min", "gumbel_right")
CASES = 10
SEED = 20261016


class CheckpointInvarianceTests(NoLeaksTestCase):
    def test_random_interruptions_resume_to_the_uninterrupted_fit(self) -> None:
        draw = random.Random(SEED)
        for case in range(CASES):
            family = draw.choice(FAMILIES)
            parts = draw.randint(2, 6)
            size = parts * draw.randint(20, 60)
            max_rows = draw.choice((7, 16, 64, 1000))
            executor = draw.choice((None, ThreadExecutor(2), ThreadExecutor(3)))
            values = values_for(family, size, seed=draw.randint(1, 10_000))
            source = ReadLog(NumpySource(values, partition_rows=size // parts))
            base = fit(source, family, max_rows=max_rows)
            with self.subTest(case=case, family=family, parts=parts, max_rows=max_rows):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "job.db"
                    counter = Interrupting(Path(directory) / "count.db")
                    complete = fit(source, family, max_rows=max_rows, checkpoint=counter)
                    total = counter.partition_commits
                    assert complete.scale.checkpoint is not None
                    self.assertEqual(complete.scale.checkpoint.read_partitions, total)
                    first = draw.randint(1, total - 1)
                    stopped = Interrupting(path, after=first)
                    with self.assertRaises(Interruption):
                        fit(
                            source, family, max_rows=max_rows, checkpoint=stopped, executor=executor
                        )
                    done = first
                    if draw.random() < 0.5 and total - first > 1:
                        more = draw.randint(1, total - first - 1)
                        again = Interrupting(path, after=more)
                        with self.assertRaises(Interruption):
                            fit(
                                source,
                                family,
                                max_rows=max_rows,
                                checkpoint=again,
                                resume=True,
                                executor=executor,
                            )
                        done += more
                    log = ReadLog(source.inner)
                    result = fit(
                        log,
                        family,
                        max_rows=max_rows,
                        checkpoint=path,
                        resume=True,
                        executor=executor,
                    )
                    self.assertTrue(same_as(result, base))
                    self.assertEqual(len(log.reads), total - done)
                    record = result.scale.checkpoint
                    assert record is not None
                    self.assertEqual(record.reused_partitions, done)
                    self.assertEqual(record.read_partitions, total - done)
