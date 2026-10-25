"""Seeded adversarial sweeps of the log-probability kernels against 50-digit mpmath.

Heavier than the reference tests (the oracle is evaluated live, a few thousand times per
family): every hard-coded constant of ``tests/reference`` is recomputed here, and each family is
swept over points and windows in the four regimes of ``log_probability_grids`` with the
bound of the design record, ``1e-12`` on ``|computed - reference| / max(1, |reference|)``.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from tests.reference import log_probability_constants as constants
from tests.reference import log_probability_grids as grids
from tests.reference import log_probability_oracle as oracle
from veridist.statistics import _logprob as logprob
from veridist.statistics.distributions import cdf

BOUND = 1.0e-12
#: ``(points, windows per regime)`` per case; the gamma oracle is the slow one.
SAMPLES = {"gamma": (70, 40)}
DEFAULT_SAMPLES = (150, 100)
SEED = 20261025
SUBNORMAL = (5e-324, 1e-323, 1e-320, 1e-319, 1e-317, 1e-315, 1e-310, 2.2250738585072014e-308)


def _keywords(family: str, parameters: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(oracle.PARAMETER_NAMES[family], parameters, strict=True))


def _worst(computed: Any, expected: list[float]) -> tuple[float, int]:
    """The largest scaled error and the index of the first one beyond the bound (or -1)."""

    errors = [oracle.scaled_error(float(c), e) for c, e in zip(computed, expected, strict=True)]
    beyond = [index for index, error in enumerate(errors) if not error <= BOUND]
    return (max(errors) if errors else 0.0), (beyond[0] if beyond else -1)


class ConstantsAreRecomputedTests(unittest.TestCase):
    def test_every_hard_coded_reference_equals_the_oracle(self) -> None:
        checked = 0
        for family, parameters, points, windows in constants.CASES:
            for x, log_cdf, log_sf in points:
                with self.subTest(family=family, x=x):
                    self.assertEqual(oracle.log_cdf(family, x, *parameters), log_cdf)
                    self.assertEqual(oracle.log_sf(family, x, *parameters), log_sf)
                checked += 1
            for lower, upper, value in windows:
                with self.subTest(family=family, lower=lower, upper=upper):
                    self.assertEqual(oracle.log_window(family, lower, upper, *parameters), value)
                checked += 1
        self.assertEqual(checked, 726 + 1347)


class SweepTests(unittest.TestCase):
    def _sweep(self, family: str) -> tuple[float, float]:
        """Sweep the cases of ``family``: the worst scaled error and the naive failure share."""

        rng = np.random.default_rng(SEED)
        points, windows = SAMPLES.get(family, DEFAULT_SAMPLES)
        worst = 0.0
        naive_failures = naive_total = 0
        for parameters in grids.CASES[family]:
            keywords = _keywords(family, parameters)
            x = grids.single_points(family, parameters, rng, points)
            for kernel, reference in (
                (logprob.log_cdf, oracle.log_cdf),
                (logprob.log_sf, oracle.log_sf),
            ):
                computed = kernel(family, x, **keywords)
                expected = [reference(family, float(v), *parameters) for v in x]
                error, first = _worst(computed, expected)
                worst = max(worst, error)
                with self.subTest(family=family, parameters=parameters, kernel=kernel.__name__):
                    self.assertEqual(first, -1, None if first < 0 else float(x[first]))
            for regime in grids.REGIMES:
                lower, upper = grids.window_points(family, parameters, regime, rng, windows)
                references = [
                    grids.reference_window(family, parameters, float(a), float(b))
                    for a, b in zip(lower, upper, strict=True)
                ]
                kept = [i for i, value in enumerate(references) if value is not None]
                lower, upper = lower[kept], upper[kept]
                expected = [float(references[i]) for i in kept]  # type: ignore[arg-type]
                computed = logprob.log_window(family, lower, upper, **keywords)
                error, first = _worst(computed, expected)
                worst = max(worst, error)
                with self.subTest(family=family, parameters=parameters, regime=regime):
                    self.assertEqual(
                        first, -1, None if first < 0 else (float(lower[first]), float(upper[first]))
                    )
                with np.errstate(all="ignore"):
                    naive = np.log(cdf(family, upper, **keywords) - cdf(family, lower, **keywords))
                lost = ~(np.abs(naive - np.array(expected)) <= 1e-9)
                naive_failures += int(lost.sum())
                naive_total += lost.size
        return worst, naive_failures / naive_total

    def _check(self, family: str, minimum_naive_failure_share: float) -> None:
        worst, share = self._sweep(family)
        self.assertLessEqual(worst, 1e-13)
        self.assertGreaterEqual(share, minimum_naive_failure_share)

    def test_exponential(self) -> None:
        self._check("exponential", 0.25)

    def test_weibull_min(self) -> None:
        self._check("weibull_min", 0.25)

    def test_gumbel_right(self) -> None:
        self._check("gumbel_right", 0.25)

    def test_normal(self) -> None:
        self._check("normal", 0.25)

    def test_lognormal(self) -> None:
        self._check("lognormal", 0.25)

    def test_gamma(self) -> None:
        self._check("gamma", 0.25)


class SubnormalAndExtremeInputTests(unittest.TestCase):
    def test_weibull_subnormal_inputs_and_windows(self) -> None:
        for shape, scale in ((0.05, 1.0), (0.3, 10.0), (0.3, 1e-300), (1.0, 1.0), (30.0, 2.5e-310)):
            parameters = (shape, scale)
            keywords = _keywords("weibull_min", parameters)
            points = np.array([*SUBNORMAL, 1e-300, 1e-200])
            lower, upper = points[:-1], points[1:]
            narrow_upper = np.where(
                lower > 1e-310, np.nextafter(lower, 1.0) * 1.0000001, lower * 2.0
            )
            for label, window_upper in (("adjacent", upper), ("narrow", narrow_upper)):
                computed = logprob.log_window("weibull_min", lower, window_upper, **keywords)
                expected = [
                    oracle.log_window("weibull_min", float(a), float(b), *parameters)
                    for a, b in zip(lower, window_upper, strict=True)
                ]
                error, first = _worst(computed, expected)
                with self.subTest(parameters=parameters, windows=label):
                    self.assertEqual(first, -1, None if first < 0 else float(lower[first]))
            for kernel, reference in (
                (logprob.log_cdf, oracle.log_cdf),
                (logprob.log_sf, oracle.log_sf),
            ):
                computed = kernel("weibull_min", points, **keywords)
                expected = [reference("weibull_min", float(v), *parameters) for v in points]
                with self.subTest(parameters=parameters, kernel=kernel.__name__):
                    self.assertEqual(_worst(computed, expected)[1], -1)

    def test_exponential_subnormal_inputs(self) -> None:
        for rate in (1.0, 1e5, 1e-300, 1e300):
            keywords = {"rate": rate}
            points = np.array([*SUBNORMAL, 1e-300])
            computed = logprob.log_cdf("exponential", points, **keywords)
            expected = [oracle.log_cdf("exponential", float(v), rate) for v in points]
            with self.subTest(rate=rate):
                self.assertEqual(_worst(computed, expected)[1], -1)

    def test_gamma_lower_tail_region_of_large_shapes(self) -> None:
        rng = np.random.default_rng(SEED)
        for shape in (8.0, 10.0, 30.0, 100.0, 1000.0, 3900.0):
            ratios = 10.0 ** rng.uniform(-300.0, -8.0, 40)
            x = shape * ratios
            keywords = {"shape": shape, "scale": 1.0}
            computed = logprob.log_cdf("gamma", x, **keywords)
            expected = [oracle.log_cdf("gamma", float(v), shape, 1.0) for v in x]
            with self.subTest(shape=shape):
                self.assertEqual(_worst(computed, expected)[1], -1)
            lower = x
            upper = x * (1.0 + 10.0 ** rng.uniform(-12.0, 0.0, x.size))
            computed = logprob.log_window("gamma", lower, upper, **keywords)
            expected = [
                oracle.log_window("gamma", float(a), float(b), shape, 1.0)
                for a, b in zip(lower, upper, strict=True)
            ]
            with self.subTest(shape=shape, kind="window"):
                self.assertEqual(_worst(computed, expected)[1], -1)

    def test_gumbel_far_tails(self) -> None:
        z = np.array([-709.0, -700.0, -100.0, 36.0, 39.5, 40.5, 100.0, 700.0, 1e4])
        keywords = {"location": 0.0, "scale": 1.0}
        for kernel, reference in (
            (logprob.log_cdf, oracle.log_cdf),
            (logprob.log_sf, oracle.log_sf),
        ):
            computed = kernel("gumbel_right", z, **keywords)
            expected = [reference("gumbel_right", float(v), 0.0, 1.0) for v in z]
            for value, wanted in zip(computed, expected, strict=True):
                if math.isfinite(wanted):
                    self.assertLessEqual(oracle.scaled_error(float(value), wanted), BOUND)


if __name__ == "__main__":
    unittest.main()
