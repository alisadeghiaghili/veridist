"""A scale fit does not depend on the executor, the worker count or the order results arrive in.

Threads read seeded data split into many partitions, with seeded delays that shuffle the order
in which the partitions finish. The result bits, the state digest and the record of the passes
are those of the serial read for every family, split and thread count.
"""

from __future__ import annotations

import struct
import time
from dataclasses import replace
from typing import Any

import numpy as np

from tests.unit.scale_executor_fixtures import NoLeaksTestCase, PartSource, cores
from tests.unit.scale_multi_pass_oracle import gumbel_sample, weibull_sample
from veridist.scale._executors import ThreadExecutor
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._sources import NumpySource

FAMILIES = ("exponential", "normal", "gamma", "lognormal", "weibull_min", "gumbel_right")
SIZE = 700


def values_of(family: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "exponential":
        return rng.exponential(3.0, SIZE)
    if family == "normal":
        return rng.normal(10.0, 3.0, SIZE)
    if family == "gamma":
        return rng.gamma(3.0, 2.0, SIZE)
    if family == "lognormal":
        return rng.lognormal(1.0, 0.5, SIZE)
    if family == "weibull_min":
        return weibull_sample(1.7, 40.0, SIZE, seed)
    return gumbel_sample(5.0, 2.0, SIZE, seed)


def bits(result: ScaleFitResult) -> tuple[bytes, ...]:
    success: Any = result.fit
    numbers = [*success.parameters.values(), success.log_likelihood]
    for row in success.uncertainty().covariance:
        numbers.extend(row)
    return tuple(struct.pack("<d", float(value)) for value in numbers)


class Delayed(PartSource):
    """A declared source whose partitions finish in a seeded, shuffled order."""

    def __init__(self, parts: list[np.ndarray], seed: int) -> None:
        super().__init__(parts)
        delays = np.random.default_rng(seed).uniform(0.0, 0.004, len(parts))
        self.hooks = {
            ordinal: (lambda _number, wait=float(delay): time.sleep(wait))
            for ordinal, delay in enumerate(delays)
        }


class ExecutorInvarianceTests(NoLeaksTestCase):
    def assertSame(self, result: ScaleFitResult, base: ScaleFitResult, label: str) -> None:
        with self.subTest(label=label):
            self.assertEqual(result.fit, base.fit)
            self.assertEqual(bits(result), bits(base))
            self.assertEqual(result.scale.state, base.scale.state)
            self.assertEqual(replace(result.scale, execution=None).passes, base.scale.passes)

    def test_numpy_ranges_of_every_size_on_every_worker_count(self) -> None:
        for family in FAMILIES:
            values = values_of(family, 5)
            base = fit(NumpySource(values), family)
            for rows in (1 + SIZE // 40, SIZE // 7, SIZE // 3, SIZE):
                for workers in (1, 2, 5):
                    with cores(16):
                        result = fit(
                            NumpySource(values, partition_rows=rows),
                            family,
                            executor=ThreadExecutor(workers),
                        )
                    self.assertSame(result, base, f"{family} rows={rows} workers={workers}")

    def test_partitions_that_finish_in_any_order(self) -> None:
        for family in FAMILIES:
            values = values_of(family, 9)
            for seed in range(3):
                arrays = np.array_split(values, 9)
                base = fit(PartSource(arrays), family)
                with cores(16):
                    result = fit(Delayed(arrays, seed), family, executor=ThreadExecutor(4))
                self.assertSame(result, base, f"{family} delays={seed}")

    def test_a_different_split_of_the_same_rows_gives_the_same_state(self) -> None:
        values = values_of("weibull_min", 3)
        digests = set()
        for pieces in (1, 2, 5, 11):
            with cores(16):
                result = fit(
                    PartSource(np.array_split(values, pieces)),
                    "weibull_min",
                    executor=ThreadExecutor(3),
                )
            assert result.scale.state is not None
            digests.add(result.scale.state.digest_sha256)
        self.assertEqual(len(digests), 1)

    def test_shuffled_rows_and_shuffled_partitions(self) -> None:
        values = values_of("gamma", 4)
        base = fit(NumpySource(values), "gamma")
        rng = np.random.default_rng(1)
        for _ in range(3):
            shuffled = rng.permutation(values)
            pieces = [shuffled[start : start + 90] for start in range(0, SIZE, 90)]
            order = rng.permutation(len(pieces))
            with cores(16):
                result = fit(
                    PartSource([pieces[index] for index in order]),
                    "gamma",
                    executor=ThreadExecutor(4),
                )
            self.assertSame(result, base, "shuffled")
