"""A spool makes a Polars or DuckDB source run once: the later passes read the local copy.

Without a spool a multi-pass fit runs a lazy plan or a query again for every pass. With one
the source is read exactly once and the result is the same, bit for bit.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_frame_fixtures import (
    connect,
    duckdb_relation,
    polars_frame,
    polars_lazy,
    sql_path,
)
from tests.unit.scale_multi_pass_oracle import gumbel_sample, weibull_sample
from veridist.scale._duckdb import DuckDBSource
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._polars import PolarsSource
from veridist.scale._sources import DEFAULT_MAX_ROWS, Partition
from veridist.scale._spool import SpoolPolicy

SIZE = 3000


def sample(family: str) -> np.ndarray:
    if family == "weibull_min":
        return weibull_sample(1.7, 1000.0, SIZE, 11)
    return gumbel_sample(50.0, 4.0, SIZE, 12)


def result_bits(result: ScaleFitResult) -> tuple[bytes, ...]:
    success: Any = result.fit
    numbers = [*success.parameters.values(), success.log_likelihood]
    uncertainty = success.uncertainty()
    for row in uncertainty.covariance:
        numbers.extend(row)
    return tuple(struct.pack("<d", float(value)) for value in numbers)


class Counting:
    """Delegates to a source and counts the calls of ``batches``."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.batch_calls = 0

    def partitions(self) -> tuple[Partition, ...]:
        partitions: tuple[Partition, ...] = self.inner.partitions()
        return partitions

    def batches(self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS) -> Iterator[Any]:
        self.batch_calls += 1
        batches: Iterator[Any] = self.inner.batches(partition, max_rows=max_rows)
        return batches


class FrameSpoolTests(unittest.TestCase):
    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.directory = self.root / "spools"
        self.directory.mkdir()
        self.policy = SpoolPolicy(directory=self.directory, max_bytes=1 << 24)

    def cases(self, values: np.ndarray) -> dict[str, Any]:
        table = pa.table({"x": values})
        path = self.root / "data.parquet"
        write_parquet(path, {"x": values}, row_group_size=1000)
        connection = connect()
        return {
            "polars frame": lambda: PolarsSource(polars_frame(table), "x"),
            "polars lazy frame": lambda: PolarsSource(polars_lazy(table), "x"),
            "polars scan": lambda: PolarsSource(pl.scan_parquet(path), "x"),
            "duckdb relation": lambda: DuckDBSource(duckdb_relation(table), "x"),
            "duckdb query": lambda: DuckDBSource(
                (connection, f"select x from read_parquet({sql_path(path)})"), "x"
            ),
        }

    def test_the_source_runs_once_with_a_spool_and_once_per_pass_without(self) -> None:
        for family in ("weibull_min", "gumbel_right"):
            values = sample(family)
            for name, make in self.cases(values).items():
                with self.subTest(family=family, source=name):
                    plain = Counting(make())
                    unspooled = fit(plain, family, max_rows=1024)
                    assert unspooled.scale.passes is not None
                    self.assertEqual(plain.batch_calls, unspooled.scale.passes.pass_count)
                    self.assertGreaterEqual(plain.batch_calls, 3)
                    once = Counting(make())
                    spooled = fit(once, family, max_rows=1024, spool=self.policy)
                    self.assertEqual(once.batch_calls, 1)
                    assert spooled.scale.passes is not None
                    self.assertEqual(spooled.scale.passes, unspooled.scale.passes)
                    self.assertEqual(result_bits(spooled), result_bits(unspooled))
                    self.assertEqual(spooled.scale.state, unspooled.scale.state)
                    assert spooled.scale.spool is not None
                    self.assertEqual(spooled.scale.spool.row_count, SIZE)
                    self.assertEqual(list(self.directory.iterdir()), [])

    def test_a_lazy_plan_with_a_side_effect_is_executed_once(self) -> None:
        executions: list[int] = []

        def observe(frame: Any) -> Any:
            executions.append(frame.height)
            return frame

        values = sample("weibull_min")
        lazy = pl.from_arrow(pa.table({"x": values})).lazy().map_batches(observe)
        fit(PolarsSource(lazy, "x"), "weibull_min", max_rows=1000, spool=self.policy)
        once = sum(executions)
        self.assertEqual(once, SIZE)
        executions.clear()
        result = fit(PolarsSource(lazy, "x"), "weibull_min", max_rows=1000)
        assert result.scale.passes is not None
        self.assertEqual(sum(executions), SIZE * result.scale.passes.pass_count)

    def test_a_query_with_a_side_effect_is_executed_once(self) -> None:
        values = sample("gumbel_right")
        connection = connect()
        calls: list[int] = []

        def observe(column: Any) -> Any:
            calls.append(len(column))
            return column

        connection.create_function("observe", observe, ["DOUBLE"], "DOUBLE", type="arrow")
        connection.from_arrow(pa.table({"x": values})).create("data")
        pair = (connection, "select observe(x) as x from data")
        fit(DuckDBSource(pair, "x"), "gumbel_right", max_rows=1000, spool=self.policy)
        self.assertEqual(sum(calls), SIZE)
        calls.clear()
        result = fit(DuckDBSource(pair, "x"), "gumbel_right", max_rows=1000)
        assert result.scale.passes is not None
        self.assertEqual(sum(calls), SIZE * result.scale.passes.pass_count)


if __name__ == "__main__":
    unittest.main()
