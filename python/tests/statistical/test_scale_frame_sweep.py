"""Millions of rows read through Polars and DuckDB give the Parquet state, byte for byte.

The data are seeded; nothing here is random between runs. The same multiset of
values is stored as multi-file Parquet (in a shuffled order), and every fold through
a Polars scan, a DuckDB scan or an in-memory Polars frame must produce the canonical
state bytes of the Parquet source and of the in-memory array fold.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_frame_fixtures import sql_path
from veridist.scale._arrow import ParquetSource
from veridist.scale._duckdb import DuckDBSource
from veridist.scale._one_pass import ExponentialState, NormalState
from veridist.scale._polars import PolarsSource
from veridist.scale._sources import NumpySource, fold

SIZE = 3_000_000
FILES = 4
STATES = (ExponentialState, NormalState)


def sweep_values() -> np.ndarray:
    rng = np.random.default_rng(20261013)
    return np.abs(rng.standard_normal(SIZE)) * np.exp2(rng.uniform(-20.0, 20.0, SIZE)) + 1e-30


class FrameSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = sweep_values()
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        shuffled = cls.values[np.random.default_rng(5).permutation(SIZE)]
        size = SIZE // FILES
        for index in range(FILES):
            write_parquet(
                cls.root / f"f{index}.parquet",
                {"x": shuffled[index * size : (index + 1) * size]},
                row_group_size=250_000,
            )
        cls.reference = {
            state_type: fold(ParquetSource(cls.root, "x"), state_type).to_bytes()
            for state_type in STATES
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def polars_scan(self) -> PolarsSource:
        return PolarsSource(pl.scan_parquet((self.root / "*.parquet").as_posix()), "x")

    def duckdb_scan(self) -> DuckDBSource:
        pattern = sql_path(self.root / "*.parquet")
        relation = duckdb.connect().sql(f"select x from read_parquet({pattern})")
        return DuckDBSource(relation, "x")

    def test_the_parquet_source_agrees_with_the_array(self) -> None:
        for state_type in STATES:
            expected = fold(NumpySource(self.values), state_type).to_bytes()
            self.assertEqual(self.reference[state_type], expected)

    def test_a_polars_scan_gives_the_parquet_state(self) -> None:
        source = self.polars_scan()
        for state_type in STATES:
            with self.subTest(state=state_type.__name__):
                self.assertEqual(fold(source, state_type).to_bytes(), self.reference[state_type])

    def test_a_duckdb_scan_gives_the_parquet_state(self) -> None:
        source = self.duckdb_scan()
        for state_type in STATES:
            with self.subTest(state=state_type.__name__):
                self.assertEqual(fold(source, state_type).to_bytes(), self.reference[state_type])

    def test_batch_size_does_not_matter_at_scale(self) -> None:
        for max_rows in (50_000, 1_000_000):
            for name, source in (("polars", self.polars_scan()), ("duckdb", self.duckdb_scan())):
                with self.subTest(source=name, max_rows=max_rows):
                    state = fold(source, NormalState, max_rows=max_rows)
                    self.assertEqual(state.to_bytes(), self.reference[NormalState])

    def test_an_in_memory_frame_gives_the_array_state(self) -> None:
        frame = pl.DataFrame({"x": self.values})
        expected = fold(NumpySource(self.values), NormalState).to_bytes()
        self.assertEqual(fold(PolarsSource(frame, "x"), NormalState).to_bytes(), expected)
        self.assertEqual(fold(PolarsSource(frame.lazy(), "x"), NormalState).to_bytes(), expected)


if __name__ == "__main__":
    unittest.main()
