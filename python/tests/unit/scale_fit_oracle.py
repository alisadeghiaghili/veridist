"""Independent oracles and shared data for the one-pass scale fit tests.

Nothing here uses the code under test. The point estimates are rebuilt from the
statistical definition with ``Fraction`` (exact sums of the binary64 values, the rounded
squares the states hold) and, for the gamma shape, from ``mpmath`` at 60 digits.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from fractions import Fraction
from typing import Any

import mpmath
import numpy as np

from veridist.domain.lifetimes import ExactLifetime
from veridist.domain.values import ExactValue
from veridist.families.registry import FamilyId
from veridist.scale._sources import DEFAULT_MAX_ROWS, FingerprintLevel, Partition

FAMILIES = (
    FamilyId.EXPONENTIAL,
    FamilyId.NORMAL,
    FamilyId.GAMMA,
    FamilyId.LOGNORMAL,
)

mpmath.mp.dps = 60


def dataset(family: FamilyId, size: int, seed: int) -> np.ndarray:
    """Seeded exact data for ``family`` with moderate conditioning."""

    rng = np.random.default_rng(seed)
    if family is FamilyId.EXPONENTIAL:
        return rng.exponential(3.0, size)
    if family is FamilyId.NORMAL:
        return rng.normal(10.0, 3.0, size)
    if family is FamilyId.GAMMA:
        return rng.gamma(3.0, 2.0, size)
    return rng.lognormal(1.0, 0.5, size)


def observations(family: FamilyId, values: np.ndarray) -> list[ExactValue] | list[ExactLifetime]:
    """The in-memory observation objects of ``values`` for ``family``."""

    if family is FamilyId.NORMAL:
        return [ExactValue(float(value)) for value in values]
    return [ExactLifetime(float(value)) for value in values]


def exact_sum(values: Sequence[float]) -> Fraction:
    return sum((Fraction(float(value)) for value in values), Fraction(0))


def exponential_rate(values: np.ndarray) -> float:
    """``n / sum t`` with the sum rounded once."""

    return len(values) / float(exact_sum(values.tolist()))


def gaussian_estimates(values: np.ndarray) -> tuple[float, float]:
    """``(mean, sd)`` from the exact sums of ``x`` and of the true squares ``x**2``.

    The mean is the exact mean rounded once and the standard deviation the square root of
    the exact variance at 60 digits, rounded once.
    """

    count = len(values)
    total = exact_sum(values.tolist())
    squares = sum((Fraction(float(value)) ** 2 for value in values), Fraction(0))
    variance = (squares - total * total / count) / count
    root = mpmath.sqrt(mpmath.mpf(variance.numerator) / variance.denominator)
    return float(total / count), float(root)


def lognormal_estimates(values: np.ndarray) -> tuple[float, float]:
    """The same on the platform logarithms of ``values``."""

    return gaussian_estimates(np.log(values))


def gamma_estimates(values: np.ndarray) -> tuple[float, float]:
    """``(shape, scale)``: the root of ``log k - digamma k = log mean - mean log`` at 60 digits.

    The means are exact over the values and over their platform logarithms, the same
    statistics the state holds, so only the root finding is independent.
    """

    count = len(values)
    mean = exact_sum(values.tolist()) / count
    mean_log = exact_sum(np.log(values).tolist()) / count
    gap = mpmath.log(mpmath.mpf(mean.numerator) / mean.denominator) - (
        mpmath.mpf(mean_log.numerator) / mean_log.denominator
    )
    start = 1.0 / (2.0 * float(gap)) if gap > 0 else 1.0
    shape = mpmath.findroot(lambda k: mpmath.log(k) - mpmath.digamma(k) - gap, start)
    scale = (mpmath.mpf(mean.numerator) / mean.denominator) / shape
    return float(shape), float(scale)


class ListSource:
    """A multi-partition in-memory source that counts what is asked of it."""

    def __init__(self, parts: list[np.ndarray]) -> None:
        self.parts = parts
        self.partition_calls = 0
        self.batch_calls: list[int] = []
        self.batch_sizes: list[int] = []

    def partitions(self) -> tuple[Partition, ...]:
        self.partition_calls += 1
        return tuple(
            Partition(f"p{i}", i, len(part), None, FingerprintLevel.NONE)
            for i, part in enumerate(self.parts)
        )

    def batches(self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS) -> Iterator[Any]:
        self.batch_calls.append(partition.ordinal)
        part = self.parts[partition.ordinal]
        for start in range(0, len(part), max_rows):
            batch = part[start : start + max_rows]
            self.batch_sizes.append(len(batch))
            yield batch
