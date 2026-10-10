"""Larger seeded sweeps of the exact accumulator: size, chunking, shuffling and merge trees.

Every configuration must give the identical exact integer, the identical canonical
bytes and the identical rounded float, and that float must equal ``math.fsum``.
The data are seeded; nothing here is random between runs.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests.unit.scale_oracle import bits, exact_units, random_chunk_sizes
from veridist.scale._accumulator import AccumulatorLimits, ExactSum
from veridist.scale._one_pass import GammaState, NormalState

SIZE = 1_000_000
FORCED = AccumulatorLimits(block_size=1024, carry_interval=4096, lane_capacity=1 << 14)


def sweep_data() -> np.ndarray:
    rng = np.random.default_rng(20261010)
    mixed = rng.standard_normal(SIZE) * np.exp2(rng.uniform(-60.0, 60.0, SIZE))
    mixed[::997] = -mixed[1::997][: len(mixed[::997])]  # plenty of exact cancellation
    return mixed


def summed(values: np.ndarray, limits: AccumulatorLimits | None = None) -> ExactSum:
    accumulator = ExactSum() if limits is None else ExactSum(limits)
    accumulator.update(values)
    return accumulator


def observable(accumulator: ExactSum) -> tuple[int, bytes, int]:
    return accumulator.total_units(), accumulator.to_bytes(), bits(accumulator.total())


class MillionElementSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = sweep_data()
        cls.reference = observable(summed(cls.values))

    def test_total_equals_fsum_and_the_integer_oracle_on_a_prefix(self) -> None:
        self.assertEqual(self.reference[2], bits(math.fsum(self.values.tolist()) + 0.0))
        prefix = self.values[:100_000]
        self.assertEqual(summed(prefix).total_units(), exact_units(prefix.tolist()))

    def test_chunkings(self) -> None:
        rng = np.random.default_rng(1)
        layouts = {
            "tiny": [3] * 2000 + [SIZE - 6000],
            "fixed_65536": [65_536] * 15 + [SIZE - 15 * 65_536],
            "random_small": random_chunk_sizes(rng, SIZE, 1000),
            "random_large": random_chunk_sizes(rng, SIZE, 400_000),
        }
        for name, sizes in layouts.items():
            with self.subTest(layout=name):
                self.assertEqual(sum(sizes), SIZE)
                accumulator = ExactSum()
                start = 0
                for size in sizes:
                    accumulator.update(self.values[start : start + size])
                    start += size
                self.assertEqual(observable(accumulator), self.reference)

    def test_shuffles(self) -> None:
        for seed in (2, 3, 4):
            with self.subTest(seed=seed):
                order = np.random.default_rng(seed).permutation(SIZE)
                self.assertEqual(observable(summed(self.values[order])), self.reference)

    def test_merge_trees_over_random_partitions(self) -> None:
        for seed, parts in ((5, 3), (6, 16), (7, 64)):
            with self.subTest(parts=parts):
                rng = np.random.default_rng(seed)
                labels = rng.integers(0, parts, SIZE)
                states = [summed(self.values[labels == part]) for part in range(parts)]
                merged = ExactSum()
                for index in rng.permutation(parts):
                    merged = merged.merge(states[int(index)])
                self.assertEqual(observable(merged), self.reference)
                level = states
                for _ in range(parts):
                    if len(level) == 1:
                        break
                    level = [
                        level[i].merge(level[i + 1]) if i + 1 < len(level) else level[i]
                        for i in range(0, len(level), 2)
                    ]
                self.assertEqual(observable(level[0]), self.reference)

    def test_forced_carries_and_folds_at_scale(self) -> None:
        forced = summed(self.values, FORCED)
        self.assertEqual(observable(forced), self.reference)
        self.assertNotEqual(forced._spill, 0)
        halves = summed(self.values[: SIZE // 2], FORCED).merge(
            summed(self.values[SIZE // 2 :], FORCED)
        )
        self.assertEqual(observable(halves), self.reference)

    def test_full_range_magnitudes(self) -> None:
        rng = np.random.default_rng(8)
        wide = rng.standard_normal(200_000) * np.exp2(rng.uniform(-300.0, 300.0, 200_000))
        self.assertEqual(summed(wide).total_units(), exact_units(wide.tolist()))


class StateSweepTests(unittest.TestCase):
    def test_normal_state_over_a_million_values(self) -> None:
        rng = np.random.default_rng(9)
        values = rng.standard_normal(SIZE) * 1000.0 + 5.0
        whole = NormalState.empty().update(values)
        result = whole.finalize()
        self.assertEqual(result.sum_x.rounded, math.fsum(values.tolist()))
        self.assertEqual(result.sum_x_squared.rounded, math.fsum((values * values).tolist()))
        for seed in (1, 2):
            order = np.random.default_rng(seed).permutation(SIZE)
            pieces = np.array_split(values[order], 7)
            merged = NormalState.empty()
            for piece in pieces:
                merged = merged.merge(NormalState.empty().update(piece))
            self.assertEqual(merged, whole)
            self.assertEqual(merged.to_bytes(), whole.to_bytes())

    def test_gamma_state_with_logarithms(self) -> None:
        rng = np.random.default_rng(10)
        values = rng.gamma(2.5, 40.0, 300_000)
        whole = GammaState.empty().update(values)
        result = whole.finalize()
        self.assertEqual(result.sum_x.rounded, math.fsum(values.tolist()))
        self.assertEqual(result.sum_log_x.rounded, math.fsum(np.log(values).tolist()))
        shuffled = GammaState.empty()
        for piece in np.array_split(values[np.random.default_rng(3).permutation(300_000)], 5):
            shuffled = shuffled.merge(GammaState.empty().update(piece))
        self.assertEqual(shuffled.to_bytes(), whole.to_bytes())


if __name__ == "__main__":
    unittest.main()
