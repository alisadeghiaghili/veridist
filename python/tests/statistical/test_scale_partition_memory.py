"""The peak memory of a serial multi-pass fit does not grow with the number of partitions.

A pass used to keep the exact accumulator of every partition until it ended (about 0.8 MB each
for the eight candidates of a Weibull pass), so the peak resident set of a serial Weibull fit
grew by roughly 1.4 MB per partition. Each partition is now merged into one running value as
it is read. The fit runs in a fresh interpreter per number of partitions, because the peak of
a process never falls, over a synthetic source whose partitions are generated when they are
read and so cost nothing to hold. The numbers are printed (``pytest -s``); the test asserts
only that the growth per partition is far below the old one.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[2]
PARTITION_COUNTS = (16, 96, 500)
ROWS_PER_PARTITION = 2000
MAX_MEGABYTES_PER_PARTITION = 0.1
"""A tenth of what a held accumulator costs, with generous slack for the allocator."""

CHILD = """
import json
import sys

import numpy as np

from tools.process_memory import peak_rss_bytes
from veridist.scale._fit import fit
from veridist.scale._sources import FingerprintLevel, Partition


class Synthetic:
    def __init__(self, partitions, rows):
        self.count, self.rows = partitions, rows

    def partitions(self):
        return tuple(
            Partition(f"p{i}", i, self.rows, None, FingerprintLevel.NONE)
            for i in range(self.count)
        )

    def batches(self, partition, *, max_rows=1 << 20):
        rng = np.random.default_rng(1000 + partition.ordinal)
        yield 40.0 * rng.weibull(1.7, self.rows) + 0.5


partitions, rows = int(sys.argv[1]), int(sys.argv[2])
before = peak_rss_bytes()
result = fit(Synthetic(partitions, rows), "weibull_min", executor="serial")
after = peak_rss_bytes()
print(json.dumps({"before": before, "after": after, "rows": result.scale.row_count}))
"""


def measure(partitions: int) -> tuple[float, float]:
    """``(peak megabytes, growth in megabytes)`` of one fit in a fresh interpreter."""

    done = subprocess.run(
        [sys.executable, "-c", CHILD, str(partitions), str(ROWS_PER_PARTITION)],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    if done.returncode != 0:
        raise AssertionError(done.stderr)
    report = json.loads(done.stdout.strip().splitlines()[-1])
    if report["rows"] != partitions * ROWS_PER_PARTITION:
        raise AssertionError("the fit did not read every row")
    return report["after"] / 1e6, (report["after"] - report["before"]) / 1e6


class PartitionMemoryTests(unittest.TestCase):
    def test_peak_memory_does_not_grow_with_the_number_of_partitions(self) -> None:
        peaks = {count: measure(count) for count in PARTITION_COUNTS}
        for count, (peak, growth) in peaks.items():
            print(f"weibull_min, {count} partitions: peak {peak:.1f} MB, growth {growth:.1f} MB")
        smallest, largest = PARTITION_COUNTS[0], PARTITION_COUNTS[-1]
        extra = peaks[largest][0] - peaks[smallest][0]
        per_partition = extra / (largest - smallest)
        print(f"extra memory per partition: {per_partition:.4f} MB")
        self.assertLess(per_partition, MAX_MEGABYTES_PER_PARTITION)
        middle = PARTITION_COUNTS[1]
        self.assertLess(
            (peaks[middle][0] - peaks[smallest][0]) / (middle - smallest),
            MAX_MEGABYTES_PER_PARTITION,
        )


if __name__ == "__main__":
    unittest.main()
