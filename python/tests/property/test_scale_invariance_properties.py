"""Seeded property tests: states depend on the multiset of observations only."""

from __future__ import annotations

import math
import unittest
import zlib

import numpy as np

from tests.unit.scale_oracle import (
    STATES,
    adversarial_sets,
    batch,
    bits,
    correctly_rounded,
    exact_units,
    random_chunk_sizes,
)
from veridist.scale._accumulator import AccumulatorLimits, ExactSum
from veridist.scale._one_pass import ExponentialState, GammaState, LognormalState, NormalState

SMALL = AccumulatorLimits(block_size=16, carry_interval=40, lane_capacity=300)


def accumulate(values: np.ndarray, limits: AccumulatorLimits) -> ExactSum:
    accumulator = ExactSum(limits)
    accumulator.update(values)
    return accumulator


def observable(accumulator: ExactSum) -> tuple[int, int, bytes, int]:
    return (
        accumulator.count,
        accumulator.total_units(),
        accumulator.to_bytes(),
        bits(accumulator.total()),
    )


def tree_merge(parts: list[ExactSum]) -> ExactSum:
    level = parts
    for _ in range(len(parts)):
        if len(level) == 1:
            break
        level = [
            level[i].merge(level[i + 1]) if i + 1 < len(level) else level[i]
            for i in range(0, len(level), 2)
        ]
    return level[0]


class AccumulatorInvarianceTests(unittest.TestCase):
    def check_invariance(self, limits: AccumulatorLimits, longest: int) -> None:
        for name, full in adversarial_sets().items():
            values = full[:longest]
            with self.subTest(name=name, block=limits.block_size):
                rng = np.random.default_rng(zlib.crc32(name.encode()))
                reference = observable(accumulate(values, limits))
                self.assertEqual(reference[1], exact_units(values.tolist()))
                self.assertEqual(reference[3], bits(correctly_rounded(reference[1])))
                total = len(values)
                for _ in range(2):
                    shuffled = values[rng.permutation(total)]
                    self.assertEqual(observable(accumulate(shuffled, limits)), reference)
                for sizes in (
                    [1] * min(total, 20) + [max(total - 20, 0)],
                    [7] * (total // 7) + [total % 7],
                    [total],
                    random_chunk_sizes(rng, total, 400),
                    random_chunk_sizes(rng, total, 5),
                ):
                    chunked = ExactSum(limits)
                    start = 0
                    for size in sizes:
                        chunked.update(values[start : start + size])
                        start += size
                    self.assertEqual(observable(chunked), reference)
                sizes = random_chunk_sizes(rng, total, max(total // 3, 1))
                edges = np.cumsum([0, *sizes])
                parts = [
                    accumulate(values[edges[i] : edges[i + 1]], limits) for i in range(len(sizes))
                ]
                order = [int(index) for index in rng.permutation(len(parts))]
                merged = ExactSum(limits)
                for index in order:
                    merged = merged.merge(parts[index])
                self.assertEqual(observable(merged), reference)
                self.assertEqual(observable(tree_merge(parts)), reference)
                self.assertEqual(observable(tree_merge([parts[i] for i in order])), reference)

    def test_order_chunking_and_merge_order_do_not_change_the_state(self) -> None:
        self.check_invariance(ExactSum().limits, 700)

    def test_the_same_holds_with_frequent_carries_and_folds(self) -> None:
        self.check_invariance(SMALL, 250)

    def test_interleaved_partitions_across_workers(self) -> None:
        values = adversarial_sets()["mixed_magnitude_sign_3"]
        reference = observable(accumulate(values, SMALL))
        for workers in (2, 3, 5, 8):
            labels = np.random.default_rng(workers).integers(0, workers, len(values))
            parts = [accumulate(values[labels == worker], SMALL) for worker in range(workers)]
            self.assertEqual(observable(tree_merge(parts)), reference)
            reverse = ExactSum(SMALL)
            for part in reversed(parts):
                reverse = reverse.merge(part)
            self.assertEqual(observable(reverse), reference)


class StateAlgebraTests(unittest.TestCase):
    def test_identity_commutativity_associativity_partition_and_permutation(self) -> None:
        for cls in STATES:
            rng = np.random.default_rng(100 + len(cls.__name__))
            for case in range(25):
                with self.subTest(state=cls.__name__, case=case):
                    sizes = [int(rng.integers(0, 40)) for _ in range(3)]
                    parts = [batch(cls, rng, size) for size in sizes]
                    a, b, c = (cls.empty().update(part) for part in parts)
                    joined = np.concatenate(parts)
                    whole = cls.empty().update(joined)
                    self.assertEqual(a.merge(cls.empty()), a)
                    self.assertEqual(cls.empty().merge(a), a)
                    self.assertEqual(a.merge(b), b.merge(a))
                    self.assertEqual(a.merge(b).merge(c), a.merge(b.merge(c)))
                    self.assertEqual(a.merge(b).merge(c), whole)
                    self.assertEqual(c.merge(a).merge(b), whole)
                    sequential = cls.empty().update(parts[0]).update(parts[1]).update(parts[2])
                    self.assertEqual(sequential, whole)
                    shuffled = cls.empty().update(joined[rng.permutation(len(joined))])
                    self.assertEqual(shuffled, whole)
                    self.assertEqual(shuffled.to_bytes(), whole.to_bytes())
                    self.assertEqual(whole.count, len(joined))
                    self.assertEqual(cls.from_bytes(whole.to_bytes()), whole)

    def test_finalized_sums_equal_fsum_of_the_in_memory_terms(self) -> None:
        for seed in range(6):
            rng = np.random.default_rng(seed)
            size = int(rng.integers(1, 200))
            plain = batch(NormalState, rng, size)
            positive = batch(GammaState, rng, size)
            exponential = batch(ExponentialState, rng, size)
            logs = np.log(positive)
            with self.subTest(seed=seed):
                normal = NormalState.empty().update(plain).finalize()
                self.assertEqual(normal.sum_x.rounded, math.fsum(plain.tolist()))
                self.assertEqual(normal.sum_x_squared.rounded, math.fsum((plain * plain).tolist()))
                self.assertEqual(
                    ExponentialState.empty().update(exponential).finalize().total_time.rounded,
                    math.fsum(exponential.tolist()),
                )
                gamma = GammaState.empty().update(positive).finalize()
                self.assertEqual(gamma.sum_x.rounded, math.fsum(positive.tolist()))
                self.assertEqual(gamma.sum_log_x.rounded, math.fsum(logs.tolist()))
                lognormal = LognormalState.empty().update(positive).finalize()
                self.assertEqual(lognormal.sum_log_x.rounded, math.fsum(logs.tolist()))
                self.assertEqual(
                    lognormal.sum_log_x_squared.rounded, math.fsum((logs * logs).tolist())
                )


if __name__ == "__main__":
    unittest.main()
