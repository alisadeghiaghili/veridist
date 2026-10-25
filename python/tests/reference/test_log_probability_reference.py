"""Log-probability kernels against 50-digit mpmath references (CENS-LL-01, -02, -05, -06).

* ``log_probability_constants.CASES``: hard-coded references on adversarial points and windows
  of every family (both tails, extreme parameters, widths ``1e-12`` to ``40`` in every regime,
  subnormal inputs, gamma shapes ``0.003`` to ``3900``, the lower-tail region of shapes 8 and
  above, windows straddling the quadrature switch); the sweep in ``tests/statistical``
  recomputes every one of them with the oracle.
* ``log Phi`` equals the in-memory scalar kernel bit for bit and the 50-digit values out to
  ``z = -40``.
* The naive ``log(cdf(b) - cdf(a))`` is wrong or non-finite where the kernel is within the
  bound (the negative control of the rejected form).

The bound is the one of the design record: ``|computed - reference| / max(1, |reference|)``
at most ``1e-12``.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from tests.reference import log_probability_constants as constants
from tests.reference import log_probability_oracle as oracle
from veridist.families.lognormal import _log_normal_sf
from veridist.statistics import _loggamma as gamma_kernel
from veridist.statistics import _logprob as logprob
from veridist.statistics.distributions import cdf

#: The accuracy bound of the kernels (scaled error).
BOUND = 1.0e-12
#: What the kernels actually achieve on these references (the worst is the normal tail).
ACHIEVED = 1.0e-13


def _scaled(computed: Any, expected: Any) -> Any:
    return np.abs(computed - expected) / np.maximum(1.0, np.abs(expected))


def _keywords(family: str, parameters: tuple[float, ...]) -> dict[str, float]:
    return dict(zip(oracle.PARAMETER_NAMES[family], parameters, strict=True))


class ConstantsAgainstMpmathTests(unittest.TestCase):
    def _worst(self, family: str) -> float:
        worst = 0.0
        for name, parameters, points, windows in constants.CASES:
            if name != family:
                continue
            keywords = _keywords(family, parameters)
            x = np.array([point[0] for point in points])
            window_lower = np.array([window[0] for window in windows])
            window_upper = np.array([window[1] for window in windows])
            checks = (
                ("log_cdf", logprob.log_cdf(family, x, **keywords), [p[1] for p in points], x),
                ("log_sf", logprob.log_sf(family, x, **keywords), [p[2] for p in points], x),
                (
                    "log_window",
                    logprob.log_window(family, window_lower, window_upper, **keywords),
                    [w[2] for w in windows],
                    window_lower,
                ),
            )
            for kernel, computed, expected, where in checks:
                errors = _scaled(computed, np.array(expected))
                worst = max(worst, float(errors.max()))
                bad = np.flatnonzero(~(errors <= BOUND))
                with self.subTest(family=family, parameters=parameters, kernel=kernel):
                    self.assertEqual(
                        bad.size, 0, [(float(where[i]), float(errors[i])) for i in bad[:3]]
                    )
        return worst

    def test_exponential(self) -> None:
        self.assertLessEqual(self._worst("exponential"), ACHIEVED)

    def test_weibull_min(self) -> None:
        self.assertLessEqual(self._worst("weibull_min"), ACHIEVED)

    def test_gumbel_right(self) -> None:
        self.assertLessEqual(self._worst("gumbel_right"), ACHIEVED)

    def test_normal(self) -> None:
        self.assertLessEqual(self._worst("normal"), ACHIEVED)

    def test_lognormal(self) -> None:
        self.assertLessEqual(self._worst("lognormal"), ACHIEVED)

    def test_gamma(self) -> None:
        self.assertLessEqual(self._worst("gamma"), ACHIEVED)

    def test_every_family_and_regime_is_represented(self) -> None:
        families = {case[0] for case in constants.CASES}
        self.assertEqual(families, set(oracle.PARAMETER_NAMES))
        self.assertGreaterEqual(min(len(case[2]) for case in constants.CASES), 14)
        self.assertGreaterEqual(min(len(case[3]) for case in constants.CASES), 17)
        widths = [
            (upper - lower) / max(abs(lower), 1e-300)
            for _, _, _, windows in constants.CASES
            for lower, upper, _ in windows
        ]
        self.assertLess(min(widths), 1e-9)
        self.assertGreater(max(widths), 1.0)
        self.assertEqual(sum(len(case[3]) for case in constants.CASES), 1347)
        self.assertEqual(sum(len(case[2]) for case in constants.CASES), 726)


#: ``(z, log Phi(z))`` from mpmath at 50 digits.
LOG_PHI = (
    (-40.0, -804.6084420137538),
    (-38.5, -745.695270290411),
    (-37.5, -707.6689893175072),
    (-37.0, -689.0305855768906),
    (-30.0, -454.3212439563432),
    (-10.0, -53.23128515051247),
    (-5.0, -15.064998393988725),
    (-1.0, -1.8410216450092636),
    (-1e-10, -0.6931471806397338),
    (0.0, -0.6931471805599453),
    (1.0, -0.17275377902344988),
    (5.0, -2.866516129637636e-07),
    (8.0, -6.220960574271786e-16),
    (38.0, 0.0),
)


class LogPhiTests(unittest.TestCase):
    def test_matches_mpmath_out_to_z_minus_40(self) -> None:
        z = np.array([point[0] for point in LOG_PHI])
        computed = logprob.log_phi(z)
        errors = _scaled(computed, np.array([point[1] for point in LOG_PHI]))
        self.assertLessEqual(float(errors.max()), 5e-14)

    def test_equals_the_scalar_kernel_bit_for_bit_over_the_sweep_grid(self) -> None:
        sweep = np.concatenate(
            [
                np.linspace(-40.0, 40.0, 16001),
                np.random.default_rng(3).uniform(-40.0, 40.0, 4000),
                np.random.default_rng(4).normal(0.0, 2.0, 4000),
                np.array([-1e5, -1e154, -1e200, 1e5, 1e200, 37.51, -37.51, 38.4, -38.4, 0.0]),
                np.array([-math.inf, math.inf]),
            ]
        )
        computed = logprob.log_phi(sweep)
        scalar = np.array([_log_normal_sf(-float(z)) for z in sweep])
        np.testing.assert_array_equal(computed.view(np.int64), scalar.view(np.int64))

    def test_the_survival_underflow_edge_uses_the_asymptotic_series_like_the_scalar_kernel(
        self,
    ) -> None:
        # erfc(z / sqrt 2) / 2 crosses 1e-300 near z = 37.0; both sides agree with mpmath
        edge = np.array([-37.0, -37.04, -37.08, -37.1, -37.2])
        expected = np.array([oracle.log_cdf("normal", float(z), 0.0, 1.0) for z in edge])
        np.testing.assert_array_less(_scaled(logprob.log_phi(edge), expected), 5e-14)

    def test_a_standard_normal_kernel_uses_it_in_both_tails(self) -> None:
        x = np.array([-39.0, -5.0, 0.0, 5.0, 39.0])
        parameters = {"mu": 0.0, "sigma": 1.0}
        np.testing.assert_array_equal(
            logprob.log_cdf("normal", x, **parameters), logprob.log_phi(x)
        )
        np.testing.assert_array_equal(
            logprob.log_sf("normal", x, **parameters), logprob.log_phi(-x)
        )


class GammaLowerTailRegionTests(unittest.TestCase):
    """CENS-LL-05: shape 8 and above with ``y = x / (shape scale)`` far below one."""

    SHAPES = (10.0, 100.0, 1000.0, 3900.0)
    RATIOS = (1e-300, 1e-100, 1e-30, 1e-17, 1e-12, 1e-8, 1e-3)

    def test_the_prefactor_of_every_shape_down_to_1e4_matches_mpmath(self) -> None:
        for shape in (10.0, 100.0, 1000.0, 1e4):
            y = np.array([shape * ratio for ratio in (*self.RATIOS, 0.4999, 1.0, 1.5, 7.0)])
            computed = gamma_kernel.log_prefactor(shape, y, np.log(y))
            expected = np.array([oracle.log_gamma_prefactor(shape, float(v)) for v in y])
            with self.subTest(shape=shape):
                self.assertLessEqual(float(_scaled(computed, expected).max()), 1e-13)

    def test_the_prefactor_is_not_quantised_where_the_deviance_rounds_to_minus_one(self) -> None:
        # delta = y / shape - 1 reads -1 to binary64 for these: three different values of y
        shape = 100.0
        y = np.array([7.35e-15, 7.46e-15, 1.598e-14])
        computed = gamma_kernel.log_prefactor(shape, y, np.log(y))
        self.assertEqual(len(set(computed.tolist())), 3)
        expected = np.array([oracle.log_gamma_prefactor(shape, float(v)) for v in y])
        self.assertLessEqual(float(_scaled(computed, expected).max()), 1e-14)

    def test_log_cdf_and_log_sf_in_the_lower_tail_match_mpmath(self) -> None:
        for shape in self.SHAPES:
            x = np.array([shape * ratio for ratio in self.RATIOS])
            lower = logprob.log_cdf("gamma", x, shape=shape, scale=1.0)
            upper = logprob.log_sf("gamma", x, shape=shape, scale=1.0)
            expected_lower = np.array([oracle.log_cdf("gamma", float(v), shape, 1.0) for v in x])
            expected_upper = np.array([oracle.log_sf("gamma", float(v), shape, 1.0) for v in x])
            with self.subTest(shape=shape):
                self.assertLessEqual(float(_scaled(lower, expected_lower).max()), 1e-13)
                self.assertLessEqual(float(_scaled(upper, expected_upper).max()), 1e-13)

    def test_windows_in_the_lower_tail_match_mpmath(self) -> None:
        for shape in self.SHAPES:
            lower = np.array([shape * ratio for ratio in (1e-100, 1e-30, 1e-17, 1e-12, 1e-8)])
            upper = lower * np.array([1.0 + 1e-10, 1.5, 2.0, 1.01, 3.0])
            computed = logprob.log_window("gamma", lower, upper, shape=shape, scale=1.0)
            expected = np.array(
                [
                    oracle.log_window("gamma", float(a), float(b), shape, 1.0)
                    for a, b in zip(lower, upper, strict=True)
                ]
            )
            with self.subTest(shape=shape):
                self.assertLessEqual(float(_scaled(computed, expected).max()), 1e-13)


class UnderflowingRatioTests(unittest.TestCase):
    """``x / scale`` underflows (or is subnormal) while ``log(x / scale)`` is an ordinary number."""

    CASES = (
        ("weibull_min", (0.3, 1e300), (5e-324, 1e-300, 1e-250, 1e-200, 1e-100)),
        ("weibull_min", (2.0, 1e300), (1e-200, 1e-150)),
        ("weibull_min", (50.0, 1e-300), (5e-324, 1e-320, 1e-310)),
        ("gamma", (10.0, 1e300), (5e-324, 1e-250, 1e-100, 1e-20)),
        ("gamma", (100.0, 1e300), (1e-200, 1e-100, 1e100)),
        ("exponential", (1e-300,), (5e-324, 1e-310, 1e-300)),
        ("exponential", (1e300,), (5e-324, 1e-320, 1e-300)),
    )

    def test_points_and_windows_match_mpmath(self) -> None:
        for family, parameters, points in self.CASES:
            keywords = _keywords(family, parameters)
            x = np.array(points)
            lower_expected = np.array([oracle.log_cdf(family, v, *parameters) for v in points])
            upper_expected = np.array([oracle.log_sf(family, v, *parameters) for v in points])
            window_expected = np.array(
                [
                    oracle.log_window(family, a, b, *parameters)
                    for a, b in zip(points[:-1], points[1:], strict=True)
                ]
            )
            with self.subTest(family=family, parameters=parameters):
                cdf_error = _scaled(logprob.log_cdf(family, x, **keywords), lower_expected)
                sf_error = _scaled(logprob.log_sf(family, x, **keywords), upper_expected)
                window = logprob.log_window(family, x[:-1], x[1:], **keywords)
                self.assertLessEqual(float(cdf_error.max()), ACHIEVED)
                self.assertLessEqual(float(sf_error.max()), ACHIEVED)
                self.assertLessEqual(float(_scaled(window, window_expected).max()), ACHIEVED)


#: ``(family, parameters, lower, upper, log(F(upper) - F(lower)))`` from mpmath at 50 digits, on
#: windows where ``log(cdf(upper) - cdf(lower))`` is lost.
NAIVE_FAILS = (
    ("exponential", (1.0,), 40.0, 40.5, -40.93275212956719),
    ("exponential", (1.0,), 1.0, 1.000000000001, -28.63093221929813),
    ("weibull_min", (1.7, 1000.0), 10000.0, 11000.0, -50.118871788467594),
    ("weibull_min", (1.7, 1000.0), 800.0, 800.0000000015999, -27.47095307491582),
    ("gumbel_right", (0.0, 1.0), 30.0, 31.0, -30.458675145387147),
    ("gumbel_right", (0.0, 1.0), 0.5, 0.500000000001, -28.73757389760619),
    ("normal", (0.0, 1.0), 10.0, 11.0, -53.23131022558312),
    ("normal", (0.0, 1.0), -2.0, -1.999999999999, -30.549870752501302),
    ("lognormal", (0.0, 1.0), 22026.465794806718, 59874.14171519782, -53.23131022558312),
    ("lognormal", (0.0, 1.0), 0.5, 0.5000000000005, -28.790097259461557),
    ("gamma", (3.0, 1.0), 80.0, 90.0, -71.90415376862244),
    ("gamma", (3.0, 1.0), 2.0, 2.000000000001, -28.937785038737683),
)


class NaiveFormulaNegativeControlTests(unittest.TestCase):
    def test_the_kernel_succeeds_where_log_of_the_cdf_difference_fails(self) -> None:
        failures = 0
        for family, parameters, lower, upper, expected in NAIVE_FAILS:
            keywords = _keywords(family, parameters)
            with self.subTest(family=family, lower=lower, upper=upper):
                point = logprob.log_window(family, np.array([lower]), np.array([upper]), **keywords)
                self.assertLessEqual(float(_scaled(point, np.array([expected]))[0]), BOUND)
                with np.errstate(all="ignore"):
                    naive = float(
                        np.log(
                            cdf(family, np.array([upper]), **keywords)
                            - cdf(family, np.array([lower]), **keywords)
                        )[0]
                    )
                lost = not math.isfinite(naive) or abs(naive - expected) > 1e-6
                self.assertTrue(lost, (naive, expected))
                failures += lost
        self.assertEqual(failures, len(NAIVE_FAILS))

    def test_the_reference_constants_agree_with_the_oracle(self) -> None:
        for family, parameters, lower, upper, expected in NAIVE_FAILS[:6]:
            with self.subTest(family=family, lower=lower):
                self.assertEqual(oracle.log_window(family, lower, upper, *parameters), expected)


if __name__ == "__main__":
    unittest.main()
