"""A fit whose process is killed part way resumes from what it had committed.

The child process dies with ``os._exit`` (no cleanup, no unwinding) at a chosen point: after k
partition commits, in the middle of a commit (the state written, the transaction never
committed) or between the last partition and the ledger row. The parent then resumes in
process, and the result must be the uninterrupted fit's, with the killed transaction absent.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from tests.unit.scale_checkpoint_fixtures import rows, same_as, values_for
from veridist.scale._fit import fit
from veridist.scale._partition_store import SQLitePartitionStore
from veridist.scale._sources import NumpySource

CHILD = textwrap.dedent(
    """
    import os, sqlite3, sys
    from veridist.scale._fit import fit
    from veridist.scale._partition_store import SQLitePartitionStore
    from veridist.scale._sources import NumpySource
    import numpy as np

    path, family, mode, point = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    values = np.load(sys.argv[5])

    class Dying(SQLitePartitionStore):
        commits = 0

        def commit_partition(self, *arguments):
            if mode == "after" and Dying.commits >= point:
                os._exit(3)
            Dying.commits += 1
            return super().commit_partition(*arguments)

        def commit_pass(self, *arguments, **options):
            if mode == "ledger" and point == 0:
                os._exit(4)
            return super().commit_pass(*arguments, **options)

        @staticmethod
        def _commit(connection):
            if mode == "midway" and Dying.commits == point:
                os._exit(5)
            connection.execute("COMMIT")

    store = Dying(path)
    source = NumpySource(values, partition_rows=len(values) // 4)
    fit(source, family, max_rows=64, checkpoint=store)
    os._exit(0)
    """
)


class KilledProcessTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "job.db"

    def kill(self, family: str, mode: str, point: int) -> int:
        import numpy as np

        data = self.root / "values.npy"
        np.save(data, values_for(family, 480))
        completed = subprocess.run(
            [sys.executable, "-c", CHILD, str(self.path), family, mode, str(point), str(data)],
            capture_output=True,
            timeout=120,
            check=False,
        )
        return completed.returncode

    def resume(self, family: str) -> None:
        source = NumpySource(values_for(family, 480), partition_rows=120)
        base = fit(source, family, max_rows=64)
        result = fit(source, family, max_rows=64, checkpoint=self.path, resume=True)
        self.assertTrue(same_as(result, base))

    def test_a_process_killed_after_some_partitions_is_resumed_from_them(self) -> None:
        for family, point in (("gamma", 1), ("weibull_min", 3)):
            with self.subTest(family=family, point=point):
                self.path = self.root / f"{family}-{point}.db"
                self.assertEqual(self.kill(family, "after", point), 3)
                held = rows(self.path, "SELECT COUNT(*) FROM partition_state")
                self.assertEqual(held, [(point,)])
                self.resume(family)

    def test_a_state_written_in_a_transaction_that_never_committed_is_absent(self) -> None:
        for family in ("gumbel_right",):
            with self.subTest(family=family):
                self.path = self.root / f"{family}.db"
                self.assertEqual(self.kill(family, "midway", 2), 5)
                # One transaction committed; the second was killed after its insert.
                self.assertEqual(rows(self.path, "SELECT COUNT(*) FROM partition_state"), [(1,)])
                self.assertEqual(SQLitePartitionStore(self.path).read_pass(0, "x" * 64), {})
                self.resume(family)

    def test_a_process_killed_between_the_last_partition_and_the_ledger_row(self) -> None:
        for family in ("weibull_min",):
            with self.subTest(family=family):
                self.path = self.root / f"{family}.db"
                self.assertEqual(self.kill(family, "ledger", 0), 4)
                self.assertEqual(rows(self.path, "SELECT COUNT(*) FROM partition_state"), [(4,)])
                self.assertEqual(rows(self.path, "SELECT COUNT(*) FROM pass_ledger"), [(0,)])
                self.resume(family)
