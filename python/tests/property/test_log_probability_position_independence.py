"""CENS-SCALE-02: the log-probability kernels are position independent.

An element's result is its single-element result, bit for bit, whatever the batch length, the
offset into a larger array, the stride, the memory order, the neighbours and the permutation of
the batch.  The gamma kernel iterates each element on its own and drops converged lanes, so a
neighbour that needs 500 iterations must not change an element that needs 2.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from typing import Any

import numpy as np

from veridist.statistics import _loggamma as gamma_kernel
from veridist.statistics import _logprob as logprob

PARAMETERS: dict[str, dict[str, float]] = {
    "exponential": {"rate": 0.01},
    "weibull_min": {"shape": 1.7, "scale": 100.0},
    "gumbel_right": {"location": 10.0, "scale": 3.0},
    "normal": {"mu": 10.0, "sigma": 3.0},
    "lognormal": {"mu_log": 2.0, "sigma_log": 1.1},
    "gamma": {"shape": 3.5, "scale": 2.0},
}
LOCATION = ("normal", "gumbel_right")
SIZE = 96


def _bits(values: Any) -> Any:
    return np.ascontiguousarray(values).view(np.int64)


def _points(family: str, rng: np.random.Generator, size: int) -> Any:
    if family in LOCATION:
        return rng.normal(10.0, 12.0, size)
    return np.exp(rng.normal(2.0, 3.0, size))


def _windows(family: str, rng: np.random.Generator, size: int) -> tuple[Any, Any]:
    lower = _points(family, rng, size)
    width = np.abs(rng.normal(0.0, 1.0, size)) * 10.0 ** rng.uniform(-10.0, 1.0, size)
    return lower, lower + width * (1.0 + np.abs(lower)) + 1e-300


def _kernels(family: str) -> dict[str, Callable[..., Any]]:
    parameters = PARAMETERS[family]
    return {
        "log_cdf": lambda x: logprob.log_cdf(family, x, **parameters),
        "log_sf": lambda x: logprob.log_sf(family, x, **parameters),
        "log_window": lambda a, b: logprob.log_window(family, a, b, **parameters),
    }


class PositionIndependenceTests(unittest.TestCase):
    def _arguments(self, family: str, name: str, seed: int) -> tuple[Any, ...]:
        rng = np.random.default_rng(seed)
        if name == "log_window":
            return _windows(family, rng, SIZE)
        return (_points(family, rng, SIZE),)

    def _check_family(self, family: str) -> None:
        for name, kernel in _kernels(family).items():
            arguments = self._arguments(family, name, 20261025)
            batch = kernel(*arguments)
            with self.subTest(family=family, kernel=name, what="single element"):
                single = np.array(
                    [kernel(*[a[i : i + 1] for a in arguments])[0] for i in range(SIZE)]
                )
                np.testing.assert_array_equal(_bits(single), _bits(batch))
            with self.subTest(family=family, kernel=name, what="prefix, suffix and halves"):
                for lo, hi in ((0, 1), (0, 17), (3, SIZE), (SIZE // 2, SIZE), (0, SIZE // 2)):
                    part = kernel(*[a[lo:hi] for a in arguments])
                    np.testing.assert_array_equal(_bits(part), _bits(batch[lo:hi]))
            with self.subTest(family=family, kernel=name, what="offset in a larger array"):
                for offset in (1, 3, 7, 31):
                    padded = []
                    for position, a in enumerate(arguments):
                        filler = np.linspace(1.0, 2.0, offset) + 2.0 * position
                        padded.append(np.concatenate([filler, a, filler[::-1] + 5.0]))
                    result = kernel(*padded)
                    np.testing.assert_array_equal(
                        _bits(result[offset : offset + SIZE]), _bits(batch)
                    )
            with self.subTest(family=family, kernel=name, what="stride and reversed order"):
                strided = [np.repeat(a, 3)[::3] for a in arguments]
                np.testing.assert_array_equal(_bits(kernel(*strided)), _bits(batch))
                reverse = [a[::-1] for a in arguments]
                np.testing.assert_array_equal(_bits(kernel(*reverse)), _bits(batch[::-1]))
            with self.subTest(family=family, kernel=name, what="permutation"):
                order = np.random.default_rng(7).permutation(SIZE)
                shuffled = kernel(*[a[order] for a in arguments])
                np.testing.assert_array_equal(_bits(shuffled), _bits(batch[order]))
            with self.subTest(family=family, kernel=name, what="memory order"):
                fortran = [np.asfortranarray(a.reshape(8, SIZE // 8)) for a in arguments]
                result = kernel(*fortran)
                np.testing.assert_array_equal(_bits(result.reshape(-1)), _bits(batch))

    def test_exponential(self) -> None:
        self._check_family("exponential")

    def test_weibull_min(self) -> None:
        self._check_family("weibull_min")

    def test_gumbel_right(self) -> None:
        self._check_family("gumbel_right")

    def test_normal(self) -> None:
        self._check_family("normal")

    def test_lognormal(self) -> None:
        self._check_family("lognormal")

    def test_gamma(self) -> None:
        self._check_family("gamma")

    def test_log_phi_single_equals_batch(self) -> None:
        z = np.random.default_rng(4).normal(0.0, 8.0, SIZE)
        batch = logprob.log_phi(z)
        single = np.array([logprob.log_phi(z[i : i + 1])[0] for i in range(SIZE)])
        np.testing.assert_array_equal(_bits(single), _bits(batch))

    def test_log1mexp_is_elementwise(self) -> None:
        d = -(10.0 ** np.random.default_rng(5).uniform(-300.0, 3.0, SIZE))
        batch = logprob.log1mexp(d)
        single = np.array([logprob.log1mexp(d[i : i + 1])[0] for i in range(SIZE)])
        np.testing.assert_array_equal(_bits(single), _bits(batch))


class GammaNeighbourTests(unittest.TestCase):
    """Lane compaction: what else is in the batch never changes an element."""

    def test_an_element_is_unchanged_by_neighbours_that_need_many_iterations(self) -> None:
        shape = gamma_kernel.GAMMA_MAX_SHAPE
        slow_series = math.nextafter(shape + 1.0, 0.0)
        slow_fraction = shape + 1.0
        x = np.array(
            [1e-300, 0.5, shape * 0.5, shape * 0.9, shape, slow_series, slow_fraction, shape * 1.2]
        )
        neighbours = np.array([slow_series, slow_fraction, slow_series * 0.9999, 1e-5, 1e300])
        for kernel in (logprob.log_cdf, logprob.log_sf):
            alone = np.array(
                [kernel("gamma", x[i : i + 1], shape=shape, scale=1.0)[0] for i in range(x.size)]
            )
            for combination in (
                np.concatenate([x, neighbours]),
                np.concatenate([neighbours, x]),
                np.concatenate([neighbours, x, neighbours[::-1]]),
                np.concatenate([x[::-1], neighbours]),
            ):
                result = kernel("gamma", combination, shape=shape, scale=1.0)
                for index, value in enumerate(x):
                    positions = np.flatnonzero(combination == value)
                    for position in positions:
                        self.assertEqual(float(result[position]), float(alone[index]))

    def test_a_window_is_unchanged_by_neighbours(self) -> None:
        shape = 1500.0
        lower = np.array([1400.0, 1499.9, 1500.0, 1.0, 1700.0])
        upper = lower + np.array([50.0, 1e-3, 120.0, 10.0, 1e-9])
        alone = np.array(
            [
                logprob.log_window(
                    "gamma", lower[i : i + 1], upper[i : i + 1], shape=shape, scale=1.0
                )[0]
                for i in range(lower.size)
            ]
        )
        rng = np.random.default_rng(11)
        extra_lower = rng.uniform(1.0, 4000.0, 200)
        extra_upper = extra_lower + rng.uniform(1e-9, 400.0, 200)
        mixed_lower = np.concatenate([extra_lower[:100], lower, extra_lower[100:]])
        mixed_upper = np.concatenate([extra_upper[:100], upper, extra_upper[100:]])
        result = logprob.log_window("gamma", mixed_lower, mixed_upper, shape=shape, scale=1.0)
        np.testing.assert_array_equal(_bits(result[100:105]), _bits(alone))


if __name__ == "__main__":
    unittest.main()
