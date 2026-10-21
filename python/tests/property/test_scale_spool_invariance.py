"""A spool changes no bit of a multi-pass fit, whatever the batching, layout or stream.

For ``weibull_min`` and ``gumbel_right`` the result bits, the canonical state digest and the
pass record with a spool are those without one, across batch caps, partition splits and
orders of a custom source, Parquet files, and a single-use stream cut into batches of any
size. The data are seeded; nothing is random between runs.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_fit_oracle import ListSource
from tests.unit.scale_multi_pass_oracle import gumbel_sample, weibull_sample
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._sources import NumpySource
from veridist.scale._spool import ScaleSpoolPolicy

SIZE = 120
FAMILIES = ("weibull_min", "gumbel_right")
SCHEMA = pa.schema([pa.field("x", pa.float64())])


def sample(family: str, seed: int, size: int = SIZE) -> np.ndarray:
    if family == "weibull_min":
        return weibull_sample(1.7, 1000.0, size, seed)
    return gumbel_sample(5.0, 2.0, size, seed)


def bits(result: ScaleFitResult) -> tuple[bytes, ...]:
    """Every number a result reports, as exact binary64 patterns."""

    success: Any = result.fit
    numbers = [*success.parameters.values(), success.log_likelihood]
    uncertainty = success.uncertainty()
    for row in uncertainty.covariance:
        numbers.extend(row)
    for lower, upper in uncertainty.confidence_intervals(0.95, "wald").values():
        numbers.extend((lower, upper))
    return tuple(struct.pack("<d", float(value)) for value in numbers)


def stream(values: np.ndarray, sizes: list[int]) -> ArrowSource:
    """A single-use source over ``values`` cut into batches of ``sizes`` (the last repeats)."""

    def generate() -> Iterator[Any]:
        start, index = 0, 0
        while start < len(values):
            size = sizes[min(index, len(sizes) - 1)]
            yield pa.record_batch([pa.array(values[start : start + size])], names=["x"])
            start, index = start + size, index + 1

    return ArrowSource(pa.RecordBatchReader.from_batches(SCHEMA, generate()), "x")


class SpoolInvarianceTests(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.directory = self.root / "spools"
        self.directory.mkdir()
        self.policy = ScaleSpoolPolicy(directory=self.directory, max_bytes=1 << 22)

    def assertSame(self, spooled: ScaleFitResult, base: ScaleFitResult, label: str) -> None:
        with self.subTest(layout=label):
            self.assertEqual(spooled.fit, base.fit)
            self.assertEqual(bits(spooled), bits(base))
            self.assertEqual(spooled.scale.state, base.scale.state)
            self.assertEqual(spooled.scale.passes, base.scale.passes)
            self.assertEqual(spooled.scale.row_count, base.scale.row_count)
            self.assertEqual(list(self.directory.iterdir()), [])

    def test_batch_caps_and_row_orders(self) -> None:
        for family in FAMILIES:
            values = sample(family, 31)
            base = fit(NumpySource(values), family)
            for max_rows in (1, 7, 119, 120, 121, 100000):
                self.assertSame(
                    fit(NumpySource(values), family, max_rows=max_rows, spool=self.policy),
                    base,
                    f"{family} max_rows={max_rows}",
                )
            shuffled = values[np.random.default_rng(97).permutation(SIZE)]
            self.assertSame(
                fit(NumpySource(shuffled), family, max_rows=89, spool=self.policy),
                base,
                f"{family} shuffled",
            )

    def test_partition_splits_and_orders_of_a_custom_source(self) -> None:
        rng = np.random.default_rng(55)
        for family in FAMILIES:
            values = sample(family, 32)
            base = fit(NumpySource(values), family)
            for trial in range(3):
                count = int(rng.integers(1, 9))
                cuts = np.sort(rng.choice(np.arange(1, SIZE), size=count, replace=False))
                parts = np.split(values[rng.permutation(SIZE)], cuts)
                parts = [parts[i] for i in rng.permutation(len(parts))]
                rows = int(rng.integers(1, 120))
                plain = fit(ListSource(parts), family, max_rows=rows)
                result = fit(ListSource(parts), family, max_rows=rows, spool=self.policy)
                self.assertSame(result, plain, f"{family} trial {trial}")
                self.assertSame(result, base, f"{family} trial {trial} against one array")
                self.assertEqual(replace(result.scale, spool=None), plain.scale)
                assert result.scale.spool is not None
                self.assertEqual(len(result.scale.spool.partitions), len(parts))

    def test_parquet_layouts(self) -> None:
        for family in FAMILIES:
            values = sample(family, 33)
            base = fit(NumpySource(values), family)
            root = self.root / family
            start = 0
            for index, size in enumerate((70, 30, 20)):
                write_parquet(
                    root / f"d{index}" / f"f{index}.parquet",
                    {"x": values[start : start + size]},
                    row_group_size=25,
                )
                start += size
            plain = fit(ParquetSource(root, "x"), family, max_rows=13)
            result = fit(ParquetSource(root, "x"), family, max_rows=13, spool=self.policy)
            self.assertSame(result, plain, f"{family} parquet")
            self.assertSame(result, base, f"{family} parquet against one array")

    def test_a_single_use_stream_cut_any_way_equals_a_re_readable_copy(self) -> None:
        for family in FAMILIES:
            values = sample(family, 35)
            base = fit(ArrowSource(pa.table({"x": values}), "x"), family)
            for sizes, rows in (([1], 1), ([7], 5), ([500], 100000), ([3, 50, 11], 40)):
                result = fit(stream(values, sizes), family, max_rows=rows, spool=self.policy)
                self.assertSame(result, base, f"{family} stream {sizes} rows {rows}")


class SpoolDigestTests(unittest.TestCase):
    def test_the_recorded_digest_depends_on_the_values_only(self) -> None:
        with tempfile.TemporaryDirectory() as holder:
            policy = ScaleSpoolPolicy(directory=holder, max_bytes=1 << 22)
            for family in FAMILIES:
                values = sample(family, 36)
                digests = set()
                for rows in (1, 13, 500):
                    result = fit(NumpySource(values), family, max_rows=rows, spool=policy)
                    assert result.scale.spool is not None
                    digests.add(result.scale.spool.partitions[0].digest_sha256)
                self.assertEqual(len(digests), 1)
                changed = values.copy()
                changed[7] = np.nextafter(changed[7], np.inf)
                other = fit(NumpySource(changed), family, spool=policy)
                assert other.scale.spool is not None
                self.assertNotIn(other.scale.spool.partitions[0].digest_sha256, digests)


if __name__ == "__main__":
    unittest.main()
