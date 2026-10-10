"""Millions of rows from multi-file Parquet in several layouts give one state, byte for byte.

The data are seeded; nothing here is random between runs. Every layout stores the
same multiset of values (one of them in a shuffled order) and every fold must
produce the identical canonical state bytes as the in-memory array fold.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.unit.scale_arrow_fixtures import write_parquet
from veridist.scale._arrow import ParquetSource
from veridist.scale._one_pass import ExponentialState, LognormalState, NormalState
from veridist.scale._sources import NumpySource, fold

SIZE = 4_000_000


def sweep_values() -> np.ndarray:
    rng = np.random.default_rng(20261012)
    return np.abs(rng.standard_normal(SIZE)) * np.exp2(rng.uniform(-20.0, 20.0, SIZE)) + 1e-30


def write_files(root: Path, values: np.ndarray, sizes: list[int], row_group_size: int) -> None:
    assert sum(sizes) == len(values)
    start = 0
    for index, size in enumerate(sizes):
        directory = root / f"part{index % 4}"
        write_parquet(
            directory / f"f{index:03d}.parquet",
            {"x": values[start : start + size]},
            row_group_size=row_group_size,
        )
        start += size


class ParquetLayoutSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = sweep_values()
        cls._directory = tempfile.TemporaryDirectory()
        base = Path(cls._directory.name)
        shuffled = cls.values[np.random.default_rng(5).permutation(SIZE)]
        cls.layouts = {
            "one file, groups of 1M": (base / "a", cls.values, [SIZE], 1_000_000),
            "three uneven files, groups of 250k": (
                base / "b",
                cls.values,
                [1_500_000, 1_000_000, 1_500_000],
                250_000,
            ),
            "eight files shuffled, groups of 131072": (
                base / "c",
                shuffled,
                [500_000] * 8,
                131_072,
            ),
        }
        for root, data, sizes, group in cls.layouts.values():
            write_files(root, data, sizes, group)
        cls.reference = {
            state_type: fold(NumpySource(cls.values), state_type).to_bytes()
            for state_type in (ExponentialState, NormalState)
        }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_every_layout_gives_the_array_state_for_two_families(self) -> None:
        for name, (root, _, _, _) in self.layouts.items():
            source = ParquetSource(root, "x")
            for state_type, expected in self.reference.items():
                with self.subTest(layout=name, state=state_type.__name__):
                    self.assertEqual(fold(source, state_type).to_bytes(), expected)

    def test_batch_size_does_not_matter_at_scale(self) -> None:
        root = self.layouts["three uneven files, groups of 250k"][0]
        source = ParquetSource(root, "x")
        for max_rows in (50_000, 250_000, 1_000_000):
            with self.subTest(max_rows=max_rows):
                state = fold(source, NormalState, max_rows=max_rows)
                self.assertEqual(state.to_bytes(), self.reference[NormalState])

    def test_a_log_family_agrees_across_a_layout_and_the_array(self) -> None:
        root = self.layouts["eight files shuffled, groups of 131072"][0]
        expected = fold(NumpySource(self.values), LognormalState).to_bytes()
        self.assertEqual(fold(ParquetSource(root, "x"), LognormalState).to_bytes(), expected)

    def test_partitions_cover_every_row_once_with_stable_fingerprints(self) -> None:
        for name, (root, _, _, _) in self.layouts.items():
            source = ParquetSource(root, "x")
            partitions = source.partitions()
            with self.subTest(layout=name):
                self.assertEqual(sum(p.rows or 0 for p in partitions), SIZE)
                self.assertEqual(len({p.id for p in partitions}), len(partitions))
                self.assertEqual(len({p.fingerprint for p in partitions}), len(partitions))
                self.assertEqual(source.partitions(), partitions)
                self.assertEqual([p.ordinal for p in partitions], list(range(len(partitions))))

    def test_content_hashing_a_multi_file_layout_changes_no_state(self) -> None:
        root = self.layouts["three uneven files, groups of 250k"][0]
        source = ParquetSource(root, "x", content_hash=True)
        partitions = source.partitions()
        self.assertEqual(len(partitions), 16)
        self.assertEqual(source.partitions(), partitions)
        state = fold(source, ExponentialState)
        self.assertEqual(state.to_bytes(), self.reference[ExponentialState])


if __name__ == "__main__":
    unittest.main()
