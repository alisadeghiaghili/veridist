"""Typed refusals of the log-probability kernels: never a silent NaN, never a guess."""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from typing import Any
from unittest import mock

import numpy as np

from veridist.engine.errors import VeridistError
from veridist.statistics import _loggamma as gamma_kernel
from veridist.statistics import _logprob as logprob
from veridist.statistics._logerrors import LogProbabilityError, LogProbabilityErrorCode

PARAMETERS: dict[str, dict[str, float]] = {
    "exponential": {"rate": 0.5},
    "weibull_min": {"shape": 1.7, "scale": 3.0},
    "gumbel_right": {"location": 1.0, "scale": 2.0},
    "normal": {"mu": 1.0, "sigma": 2.0},
    "lognormal": {"mu_log": 0.3, "sigma_log": 0.9},
    "gamma": {"shape": 2.5, "scale": 1.5},
}


def _array(*values: float) -> Any:
    return np.array(values, dtype=np.float64)


class ErrorTypeTests(unittest.TestCase):
    def test_the_codes_are_a_closed_locale_neutral_set(self) -> None:
        self.assertEqual(
            {code.value for code in LogProbabilityErrorCode},
            {"nonfinite_input", "invalid_window", "gamma_shape_refused", "gamma_nonconvergence"},
        )

    def test_the_error_carries_its_code_and_region(self) -> None:
        error = LogProbabilityError(LogProbabilityErrorCode.GAMMA_NONCONVERGENCE, "shape 8 to 1000")
        self.assertIsInstance(error, ArithmeticError)
        self.assertIsInstance(error, VeridistError)
        self.assertIs(error.code, LogProbabilityErrorCode.GAMMA_NONCONVERGENCE)
        self.assertEqual(error.region, "shape 8 to 1000")
        self.assertEqual(str(error), "gamma_nonconvergence (shape 8 to 1000)")
        plain = LogProbabilityError(LogProbabilityErrorCode.INVALID_WINDOW)
        self.assertIsNone(plain.region)
        self.assertEqual(str(plain), "invalid_window")

    def test_the_code_must_be_a_member_of_the_enumeration(self) -> None:
        with self.assertRaises(TypeError):
            LogProbabilityError("invalid_window")  # type: ignore[arg-type]


class InvalidWindowTests(unittest.TestCase):
    def test_a_window_with_lower_not_below_upper_is_refused_for_every_family(self) -> None:
        bad = (
            (_array(2.0), _array(1.0)),
            (_array(2.0), _array(2.0)),
            (_array(-math.inf), _array(-math.inf)),
            (_array(math.inf), _array(math.inf)),
            (_array(math.inf), _array(-math.inf)),
            (_array(0.5, 1.0, 7.0), _array(0.6, 1.0, 8.0)),
            (_array(0.5, 1.0, 7.0), _array(0.6, 2.0, 6.0)),
        )
        for family, parameters in PARAMETERS.items():
            for lower, upper in bad:
                with self.subTest(family=family, lower=lower.tolist(), upper=upper.tolist()):
                    with self.assertRaises(LogProbabilityError) as raised:
                        logprob.log_window(family, lower, upper, **parameters)
                    self.assertIs(raised.exception.code, LogProbabilityErrorCode.INVALID_WINDOW)
                    self.assertIsNone(raised.exception.region)

    def test_the_refusal_comes_before_anything_is_computed(self) -> None:
        with mock.patch.object(logprob._Normal, "log_cdf") as cdf:
            with self.assertRaises(LogProbabilityError):
                logprob.log_window("normal", _array(0.0, 3.0), _array(1.0, 2.0), mu=0.0, sigma=1.0)
        cdf.assert_not_called()

    def test_an_arbitrarily_ordered_pair_of_infinite_ends_is_not_a_window(self) -> None:
        with self.assertRaises(LogProbabilityError):
            logprob.log_window("normal", math.inf, -math.inf, mu=0.0, sigma=1.0)
        value = logprob.log_window("normal", -math.inf, math.inf, mu=0.0, sigma=1.0)
        self.assertEqual(float(value), 0.0)


class NonFiniteInputTests(unittest.TestCase):
    def test_nan_is_refused_in_every_entry_point(self) -> None:
        for family, parameters in PARAMETERS.items():
            calls: tuple[Callable[[], Any], ...] = (
                lambda f=family, p=parameters: logprob.log_cdf(f, _array(1.0, math.nan), **p),
                lambda f=family, p=parameters: logprob.log_sf(f, _array(math.nan), **p),
                lambda f=family, p=parameters: logprob.log_window(
                    f, _array(math.nan), _array(1.0), **p
                ),
                lambda f=family, p=parameters: logprob.log_window(
                    f, _array(0.5), _array(math.nan), **p
                ),
            )
            for call in calls:
                with self.subTest(family=family):
                    with self.assertRaises(LogProbabilityError) as raised:
                        call()
                    self.assertIs(raised.exception.code, LogProbabilityErrorCode.NONFINITE_INPUT)

    def test_infinities_are_points_of_the_extended_line(self) -> None:
        for family, parameters in PARAMETERS.items():
            with self.subTest(family=family):
                cdf = logprob.log_cdf(family, _array(-math.inf, math.inf), **parameters)
                sf = logprob.log_sf(family, _array(-math.inf, math.inf), **parameters)
                self.assertEqual(list(cdf), [-math.inf, 0.0])
                self.assertEqual(list(sf), [0.0, -math.inf])


class GammaRefusalTests(unittest.TestCase):
    def test_a_shape_above_the_limit_is_refused_by_every_entry_point(self) -> None:
        parameters = {"shape": 4000.0, "scale": 1.0}
        calls: tuple[Callable[[], Any], ...] = (
            lambda: logprob.log_cdf("gamma", _array(1.0), **parameters),
            lambda: logprob.log_sf("gamma", _array(1.0), **parameters),
            lambda: logprob.log_window("gamma", _array(1.0), _array(2.0), **parameters),
            lambda: logprob.log_window("gamma", _array(-1.0), _array(math.inf), **parameters),
        )
        for call in calls:
            with self.assertRaises(LogProbabilityError) as raised:
                call()
            self.assertIs(raised.exception.code, LogProbabilityErrorCode.GAMMA_SHAPE_REFUSED)
            self.assertEqual(raised.exception.region, "shape above 3900")

    def test_the_limit_itself_is_served_including_its_slowest_argument(self) -> None:
        shape = gamma_kernel.GAMMA_MAX_SHAPE
        worst = math.nextafter(shape + 1.0, 0.0)  # the series needs 509 of its 512 iterations
        x = _array(1e-300, 1.0, shape, worst, shape + 1.0, 2.0 * shape, 1e5)
        lower = logprob.log_cdf("gamma", x, shape=shape, scale=1.0)
        upper = logprob.log_sf("gamma", x, shape=shape, scale=1.0)
        self.assertTrue(np.isfinite(lower).all())
        self.assertTrue(np.isfinite(upper).all())

    def test_a_series_that_does_not_converge_is_a_typed_failure_naming_the_region(self) -> None:
        for shape, region in ((0.5, "shape 0.01 to 8"), (500.0, "shape 8 to 1000")):
            with mock.patch.object(gamma_kernel, "GAMMA_MAX_ITERATIONS", 3):
                with self.assertRaises(LogProbabilityError) as raised:
                    logprob.log_cdf("gamma", _array(shape * 0.9), shape=shape, scale=1.0)
            self.assertIs(raised.exception.code, LogProbabilityErrorCode.GAMMA_NONCONVERGENCE)
            self.assertEqual(raised.exception.region, region)

    def test_a_fraction_that_does_not_converge_is_a_typed_failure_naming_the_region(self) -> None:
        with mock.patch.object(gamma_kernel, "GAMMA_MAX_ITERATIONS", 3):
            with self.assertRaises(LogProbabilityError) as raised:
                logprob.log_sf("gamma", _array(2000.0), shape=1500.0, scale=1.0)
        self.assertIs(raised.exception.code, LogProbabilityErrorCode.GAMMA_NONCONVERGENCE)
        self.assertEqual(raised.exception.region, "shape 1000 to 3900")

    def test_one_failing_element_fails_the_whole_call_never_a_nan(self) -> None:
        x = _array(1.0, 400.0, 2.0)
        with mock.patch.object(gamma_kernel, "GAMMA_MAX_ITERATIONS", 30):
            with self.assertRaises(LogProbabilityError):
                logprob.log_cdf("gamma", x, shape=400.0, scale=1.0)
            fine = logprob.log_cdf("gamma", _array(1.0, 2.0), shape=400.0, scale=1.0)
        self.assertTrue(np.isfinite(fine).all())


class NoSilentNanTests(unittest.TestCase):
    def test_extreme_arguments_give_numbers_or_minus_infinity_never_nan(self) -> None:
        extreme = _array(
            -1e308, -1e5, -745.0, -1.0, 0.0, 5e-324, 1e-310, 1e-300, 1e-20, 0.5,
            1.0, 40.0, 709.0, 745.0, 1e5, 1e150, 1e300, 1.7e308, math.inf, -math.inf,
        )  # fmt: skip
        parameter_sets = {
            "exponential": ({"rate": 1e-300}, {"rate": 1.0}, {"rate": 1e300}),
            "weibull_min": (
                {"shape": 0.05, "scale": 1e-300},
                {"shape": 1.0, "scale": 1.0},
                {"shape": 1000.0, "scale": 1e300},
            ),
            "gumbel_right": (
                {"location": 0.0, "scale": 1e-300},
                {"location": 0.0, "scale": 1.0},
                {"location": 1e300, "scale": 1e300},
            ),
            "normal": (
                {"mu": 0.0, "sigma": 1e-300},
                {"mu": 0.0, "sigma": 1.0},
                {"mu": 1e300, "sigma": 1e300},
            ),
            "lognormal": (
                {"mu_log": 0.0, "sigma_log": 1e-3},
                {"mu_log": 0.0, "sigma_log": 1.0},
                {"mu_log": 700.0, "sigma_log": 100.0},
            ),
            "gamma": (
                {"shape": 0.003, "scale": 1e-300},
                {"shape": 1.0, "scale": 1.0},
                {"shape": 3900.0, "scale": 1e300},
            ),
        }
        lower = np.repeat(extreme, extreme.size)
        upper = np.tile(extreme, extreme.size)
        keep = lower < upper
        for family, sets in parameter_sets.items():
            for parameters in sets:
                with self.subTest(family=family, parameters=parameters):
                    self.assertFalse(np.isnan(logprob.log_cdf(family, extreme, **parameters)).any())
                    self.assertFalse(np.isnan(logprob.log_sf(family, extreme, **parameters)).any())
                    window = logprob.log_window(family, lower[keep], upper[keep], **parameters)
                    self.assertFalse(np.isnan(window).any())
                    self.assertTrue((window <= 0.0).all())


if __name__ == "__main__":
    unittest.main()
