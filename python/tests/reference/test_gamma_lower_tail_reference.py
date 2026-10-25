"""Gamma log-density, incomplete-gamma prefactor and Weibull log-ratio accuracy against mpmath.

Before the fix the scalar gamma log-density formed ``delta = x / (shape * scale) - 1`` and
rounded it to binary64.  Below a ratio of about ``1e-8`` that rounding keeps no information
about the ratio (it reads ``-1 + 1.1e-16 * m`` for a whole range of ratios), so shapes of 8 and
more returned one quantised value for many different observations, for instance
``-3540.1656`` for all three observations of ``SHAPE_100_EXAMPLES``.

Every expected value below is an mpmath result at 50 digits or more, never a library value:

* ``SHAPE_100_EXAMPLES`` and ``WEIBULL_EXAMPLES``: ``log_density_oracle.gamma`` /
  ``weibull_min`` at the exact binary inputs, printed with ``repr``.
* ``INCOMPLETE_GAMMA_EXAMPLES``: ``mpmath.gammainc(shape, 0, x, regularized=True)`` for ``P``
  and ``mpmath.gammainc(shape, x, mpmath.inf, regularized=True)`` for ``Q`` at 60 digits,
  printed to 20 digits.
* the seeded grids call the same oracle inside the test (about two seconds in all).
"""

from __future__ import annotations

import math
import random
import unittest

from tests.reference import log_density_oracle as oracle
from veridist.families.registry import FamilyId
from veridist.statistics.distributions import cdf, logpdf, sf
from veridist.statistics.log_density import LogDensitySuccess, evaluate_log_density

#: The accuracy bound of the gamma log-density: error divided by ``max(1, |true value|)``.
SCALED_BOUND = 1.0e-12

#: ``(observation, log-density)`` at ``shape=100, scale=1``; the quantised value was -3540.1656.
SHAPE_100_EXAMPLES = (
    (7.35e-15, -3580.9977374564833),
    (7.46e-15, -3579.5270824583818),
    (1.598e-14, -3504.1106123722584),
)

SHAPES = (0.003, 0.05, 0.5, 1.0, 3.0, 8.0, 10.0, 30.0, 100.0, 1000.0, 1.0e4)
SUBNORMAL_OBSERVATIONS = (5e-324, 1e-323, 1e-320, 1e-315, 1e-310, 2.2250738585072014e-308)

#: ``(shape, observation, scale, log-density)`` of the Weibull minimum.  The last three have a
#: ratio ``x / scale`` that is subnormal or whose logarithms are each a few hundred.
WEIBULL_EXAMPLES = (
    (30.0, 2.2250738585072014e-308, 2.5e-310, -3.034818215957271e58),
    (100.0, 2.2250738585072014e-308, 2.5e-310, -8.718510078322588e194),
    (1000.0, 1.5e-08, 1e-08, -1.2338405969060374e176),
    (100.0, 5e-324, 1.0, -73694.96195003076),
    (3.0, 1e-320, 1e-300, 599.7707142013614),
    (1000.0, 1.5e-300, 1e-300, -1.2338405969062417e176),
)

#: ``(shape, x, "P" or "Q", value)`` at ``scale=1``: the lower or the upper regularized ratio.
INCOMPLETE_GAMMA_EXAMPLES = (
    (1000.0, 700.0, "P", 1.0158583345333216375e-26),
    (1000.0, 950.0, "P", 0.055054686230738034495),
    (1000.0, 1050.0, "Q", 0.058671111377318077098),
    (1000.0, 1300.0, "Q", 1.8736155715785551242e-18),
    (2000.0, 1400.0, "P", 1.755313910422360811e-51),
    (2000.0, 1900.0, "P", 0.011695820381150113408),
    (2000.0, 2100.0, "Q", 0.013647467683370377295),
    (2000.0, 2600.0, "Q", 6.0281550200821681576e-35),
    (5000.0, 3500.0, "P", 1.6052658946905612452e-125),
    (5000.0, 4750.0, "P", 0.00016438616435882255554),
    (5000.0, 5250.0, "Q", 0.00024794736798936085205),
    (5000.0, 6500.0, "Q", 3.532553438542575214e-84),
)


def _scaled_error(computed: float, expected: float) -> float:
    return abs(computed - expected) / max(1.0, abs(expected))


def _gamma(observation: float, shape: float, scale: float = 1.0) -> float:
    result = evaluate_log_density(FamilyId.GAMMA, observation, shape=shape, scale=scale)
    assert isinstance(result, LogDensitySuccess)
    return result.log_density


class GammaLowerTailTests(unittest.TestCase):
    def test_the_shape_100_examples_are_no_longer_quantised(self) -> None:
        values = []
        for observation, expected in SHAPE_100_EXAMPLES:
            with self.subTest(observation=observation):
                value = _gamma(observation, 100.0)
                self.assertLessEqual(_scaled_error(value, expected), 1e-14)
                self.assertLess(abs(value - expected), 1e-10)
                values.append(value)
        self.assertEqual(len(set(values)), 3)

    def test_the_examples_agree_with_the_oracle_constants(self) -> None:
        for observation, expected in SHAPE_100_EXAMPLES:
            with self.subTest(observation=observation):
                self.assertEqual(oracle.gamma(observation, shape=100.0, scale=1.0), expected)

    def test_seeded_adversarial_grid_matches_mpmath_for_every_shape(self) -> None:
        rng = random.Random(20261024)
        worst = 0.0
        for shape in SHAPES:
            observations = [shape * 10 ** rng.uniform(-300.0, 2.5) for _ in range(40)]
            observations += [shape * ratio for ratio in (1e-17, 1e-16, 1e-12, 1e-8, 0.4999, 0.5)]
            observations += list(SUBNORMAL_OBSERVATIONS)
            for observation in observations:
                if not observation >= 5e-324:
                    continue
                with self.subTest(shape=shape, observation=observation):
                    expected = oracle.gamma(observation, shape=shape, scale=1.0)
                    error = _scaled_error(_gamma(observation, shape), expected)
                    worst = max(worst, error)
                    self.assertLessEqual(error, SCALED_BOUND)
        self.assertLessEqual(worst, 1e-13)

    def test_scales_other_than_one_keep_the_bound(self) -> None:
        rng = random.Random(7)
        for shape in (8.0, 30.0, 1000.0, 1.0e4):
            for scale in (1e-6, 0.37, 13.0, 1e6):
                for _ in range(8):
                    observation = shape * scale * 10 ** rng.uniform(-30.0, 1.5)
                    with self.subTest(shape=shape, scale=scale, observation=observation):
                        expected = oracle.gamma(observation, shape=shape, scale=scale)
                        error = _scaled_error(_gamma(observation, shape, scale), expected)
                        self.assertLessEqual(error, SCALED_BOUND)

    def test_the_array_path_gives_the_scalar_values(self) -> None:
        import numpy as np

        for shape in (10.0, 100.0, 1000.0):
            points = np.array([shape * ratio for ratio in (1e-16, 1e-10, 1e-5, 0.2, 0.7, 1.0, 3.0)])
            expected = [oracle.gamma(float(x), shape=shape, scale=1.0) for x in points]
            values = logpdf("gamma", points, shape=shape, scale=1.0)
            for point, value, truth in zip(points, values, expected, strict=True):
                with self.subTest(shape=shape, observation=float(point)):
                    self.assertLessEqual(_scaled_error(float(value), truth), SCALED_BOUND)

    def test_the_value_is_continuous_across_the_switch_point(self) -> None:
        # The deviance form switches at ``x / (shape * scale) - 1 == -0.5``.
        for shape in (8.0, 100.0, 1.0e4):
            switch = shape * 0.5
            neighbours = [switch]
            for _ in range(3):
                neighbours.insert(0, math.nextafter(neighbours[0], 0.0))
                neighbours.append(math.nextafter(neighbours[-1], math.inf))
            truths = [oracle.gamma(x, shape=shape, scale=1.0) for x in neighbours]
            values = [_gamma(x, shape) for x in neighbours]
            for x, value, truth in zip(neighbours, values, truths, strict=True):
                with self.subTest(shape=shape, observation=x):
                    self.assertLessEqual(_scaled_error(value, truth), 1e-14)
            for index in range(len(values) - 1):
                with self.subTest(shape=shape, step=index):
                    true_step = truths[index + 1] - truths[index]
                    self.assertLessEqual(
                        abs((values[index + 1] - values[index]) - true_step),
                        1e-14 * max(1.0, abs(truths[index])),
                    )

    def test_the_far_left_tail_below_binary64_resolution_of_the_ratio_is_exact(self) -> None:
        for shape in (8.0, 100.0):
            for scale in (1.0, 1e-300, 1e300):
                observation = 5e-324 if scale == 1e-300 else shape * scale * 1e-30
                with self.subTest(shape=shape, scale=scale):
                    expected = oracle.gamma(observation, shape=shape, scale=scale)
                    self.assertLessEqual(
                        _scaled_error(_gamma(observation, shape, scale), expected), SCALED_BOUND
                    )


class IncompleteGammaPrefactorTests(unittest.TestCase):
    def test_large_shape_cdf_and_sf_match_mpmath(self) -> None:
        for shape, x, which, expected in INCOMPLETE_GAMMA_EXAMPLES:
            with self.subTest(shape=shape, x=x, which=which):
                value = (cdf if which == "P" else sf)("gamma", x, shape=shape, scale=1.0)
                self.assertLessEqual(abs(value - expected) / expected, 5e-13)


class WeibullLogRatioTests(unittest.TestCase):
    def test_subnormal_and_extreme_ratios_match_mpmath(self) -> None:
        import numpy as np

        for shape, observation, scale, expected in WEIBULL_EXAMPLES:
            with self.subTest(shape=shape, observation=observation, scale=scale):
                result = evaluate_log_density(
                    FamilyId.WEIBULL_MIN, observation, shape=shape, scale=scale
                )
                self.assertIsInstance(result, LogDensitySuccess)
                assert isinstance(result, LogDensitySuccess)
                self.assertLessEqual(_scaled_error(result.log_density, expected), 5e-13)
                vector = logpdf("weibull_min", np.array([observation]), shape=shape, scale=scale)
                self.assertLessEqual(_scaled_error(float(vector[0]), expected), 5e-13)


if __name__ == "__main__":
    unittest.main()
