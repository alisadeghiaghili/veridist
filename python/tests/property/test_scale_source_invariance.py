"""Seeded property tests: a folded state depends on the values, not on how they are stored."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import StreamExporter, write_parquet
from tests.unit.scale_source_oracle import ListSource
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._one_pass import ExponentialState, GammaState, LognormalState, NormalState
from veridist.scale._sources import NumpySource, Partition, fold, fold_partition

SIZE = 120
STATES = (ExponentialState, NormalState, GammaState, LognormalState)
MAX_ROWS = (7, 100_000)


def positive_values() -> np.ndarray:
    rng = np.random.default_rng(20261011)
    return np.exp2(rng.uniform(-30.0, 30.0, SIZE)) * rng.uniform(0.5, 1.5, SIZE)


def write_layout(
    root: Path, values: np.ndarray, sizes: list[int], row_group_size: int, nested: bool = False
) -> None:
    """Split ``values`` into consecutive files of the given sizes."""

    assert sum(sizes) == len(values)
    start = 0
    for index, size in enumerate(sizes):
        name = f"d{index % 3}/f{index:02d}.parquet" if nested else f"f{index:02d}.parquet"
        write_parquet(
            root / name, {"x": values[start : start + size]}, row_group_size=row_group_size
        )
        start += size


class RecordingSource:
    """Wraps a source and records every batch it hands out."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.seen: list[np.ndarray] = []

    def partitions(self) -> tuple[Partition, ...]:
        partitions: tuple[Partition, ...] = self.inner.partitions()
        return partitions

    def batches(self, partition: Partition, *, max_rows: int = 262144) -> Iterator[Any]:
        for batch in self.inner.batches(partition, max_rows=max_rows):
            self.seen.append(np.array(batch))
            yield batch


class LayoutInvarianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = positive_values()
        cls._directory = tempfile.TemporaryDirectory()
        base = Path(cls._directory.name)
        rng = np.random.default_rng(7)
        permuted = cls.values[rng.permutation(SIZE)]
        layouts: dict[str, tuple[np.ndarray, list[int], int, bool]] = {
            "one file one group": (cls.values, [SIZE], SIZE, False),
            "one file small groups": (cls.values, [SIZE], 7, False),
            "five files": (cls.values, [24] * 5, 10, False),
            "uneven files permuted": (permuted, [1, 50, 3, 66], 16, False),
            "descending nested": (np.sort(cls.values)[::-1].copy(), [40, 40, 40], 13, True),
            "tiny files and groups": (cls.values, [1, 2, 117], 20, True),
        }
        cls.layouts: dict[str, Path] = {}
        for index, (name, (data, sizes, group, nested)) in enumerate(layouts.items()):
            root = base / f"layout{index}"
            write_layout(root, data, sizes, group, nested)
            cls.layouts[name] = root

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def reference(self, state_type: type) -> bytes:
        return bytes(state_type.empty().update(self.values).to_bytes())

    def check(self, state_type: type, max_rows_options: tuple[int, ...]) -> None:
        expected = self.reference(state_type)
        self.assertEqual(fold(NumpySource(self.values), state_type).to_bytes(), expected)
        for name, root in self.layouts.items():
            for max_rows in max_rows_options:
                with self.subTest(layout=name, max_rows=max_rows):
                    state = fold(ParquetSource(root, "x"), state_type, max_rows=max_rows)
                    self.assertEqual(state.to_bytes(), expected)

    def test_exponential(self) -> None:
        self.check(ExponentialState, MAX_ROWS)

    def test_normal(self) -> None:
        self.check(NormalState, MAX_ROWS)

    def test_gamma(self) -> None:
        self.check(GammaState, MAX_ROWS)

    def test_lognormal(self) -> None:
        self.check(LognormalState, MAX_ROWS)

    def test_one_row_batches_give_the_same_state_for_every_family(self) -> None:
        root = self.layouts["five files"]
        for state_type in STATES:
            with self.subTest(state=state_type.__name__):
                state = fold(ParquetSource(root, "x"), state_type, max_rows=1)
                self.assertEqual(state.to_bytes(), self.reference(state_type))

    def test_the_fingerprint_level_does_not_change_the_state(self) -> None:
        expected = self.reference(NormalState)
        for name, root in self.layouts.items():
            with self.subTest(layout=name):
                source = ParquetSource(root, "x", content_hash=True)
                self.assertEqual(fold(source, NormalState, max_rows=64).to_bytes(), expected)

    def test_partition_states_merge_to_the_same_state_in_any_order(self) -> None:
        expected = self.reference(NormalState)
        source = ParquetSource(self.layouts["uneven files permuted"], "x")
        partitions = source.partitions()
        states = [fold_partition(source, p, NormalState, max_rows=9) for p in partitions]
        rng = np.random.default_rng(11)
        for _ in range(6):
            order = rng.permutation(len(states))
            merged = NormalState.empty()
            for index in order:
                merged = merged.merge(states[int(index)])
            self.assertEqual(merged.to_bytes(), expected)
        self.assertEqual(sum(state.count for state in states), SIZE)
        self.assertEqual(sum(p.rows or 0 for p in partitions), SIZE)

    def test_arrow_inputs_agree_with_the_array_and_parquet_sources(self) -> None:
        expected = self.reference(NormalState)
        table = pa.table({"x": self.values})
        pieces = pa.Table.from_batches(table.to_batches(max_chunksize=17))
        factories: dict[str, Callable[[], Any]] = {
            "table": lambda: table,
            "chunked table": lambda: pieces,
            "batch": lambda: table.to_batches()[0],
            "reader": lambda: pa.RecordBatchReader.from_stream(pieces),
            "exporter": lambda: StreamExporter(pieces),
        }
        for name, make in factories.items():
            for max_rows in (1, 13, 100_000):
                with self.subTest(input=name, max_rows=max_rows):
                    state = fold(ArrowSource(make(), "x"), NormalState, max_rows=max_rows)
                    self.assertEqual(state.to_bytes(), expected)

    def test_stored_types_that_widen_exactly_give_the_same_state_as_float64(self) -> None:
        rng = np.random.default_rng(3)
        ints = rng.integers(-(2**53), 2**53 + 1, SIZE)
        small = rng.integers(-(2**31), 2**31, SIZE).astype(np.int32)
        singles = rng.standard_normal(SIZE).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, data in (("i64", ints), ("i32", small), ("f32", singles)):
                write_parquet(root / f"{name}.parquet", {"x": data}, row_group_size=25)
                widened = data.astype(np.float64)
                expected = NormalState.empty().update(widened).to_bytes()
                with self.subTest(stored=name):
                    source = ParquetSource(root / f"{name}.parquet", "x")
                    self.assertEqual(fold(source, NormalState, max_rows=11).to_bytes(), expected)
                    self.assertEqual(fold(NumpySource(data), NormalState).to_bytes(), expected)

    def test_a_refused_batch_stops_the_fold_before_it_reaches_the_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_parquet(root / "f0.parquet", {"x": [1.0, 2.0, 3.0]}, row_group_size=3)
            write_parquet(root / "f1.parquet", {"x": [4.0, float("nan"), 6.0]}, row_group_size=3)
            write_parquet(root / "f2.parquet", {"x": [7.0, 8.0, 9.0]}, row_group_size=3)
            source = RecordingSource(ParquetSource(root, "x"))
            with self.assertRaises(ScaleSourceError) as context:
                fold(source, ExponentialState, max_rows=2)
            self.assertIs(context.exception.code, ScaleSourceErrorCode.NAN_VALUE)
            seen = [value for batch in source.seen for value in batch.tolist()]
            self.assertEqual(seen, [1.0, 2.0, 3.0])
            clean = fold(ParquetSource(root / "f0.parquet", "x"), ExponentialState)
            self.assertEqual(clean.count, 3)

    def test_a_listed_source_with_no_values_folds_to_the_empty_state(self) -> None:
        for state_type in STATES:
            self.assertEqual(fold(ListSource([[]]), state_type), state_type.empty())


if __name__ == "__main__":
    unittest.main()
