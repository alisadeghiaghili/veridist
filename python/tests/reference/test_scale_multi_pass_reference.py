"""Multi-pass scale fits against 50-digit roots and against the in-memory fits.

The reference of each case is the root of the profile equation computed with ``mpmath`` at
50 digits from the exact logarithms (Weibull) or values (Gumbel) of the same binary64 data;
the scale fit must agree with it to better than 1e-13 relative (measured: below 1e-15 on
every case below). An extreme sample is allowed the rounding of its platform logarithms,
which is relative to the spread of the logarithms (a sample of relative spread 1e-5 has its
shape known to about 1e-11, whatever the method), and a result that is a subnormal or a
location of large magnitude one unit in the last place.

The in-memory fits stop a golden-section search whose objective is flat at the optimum, so
their parameters carry an error of about the square root of the machine epsilon; measured
differences are below 3e-8 relative over the cases here (and the log-likelihoods agree
to 5e-16), and the bound pinned is 1e-6.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from tests.unit.scale_multi_pass_oracle import (
    gumbel_sample,
    mp_gumbel,
    mp_weibull,
    relative,
    weibull_sample,
)
from veridist.domain.lifetimes import ExactLifetime
from veridist.domain.values import ExactValue
from veridist.families.gumbel import fit_gumbel_right
from veridist.families.weibull import fit_weibull
from veridist.scale._fit import fit
from veridist.scale._sources import NumpySource

MP_TOLERANCE = 1e-13
IN_MEMORY_TOLERANCE = 1e-6
LIKELIHOOD_TOLERANCE = 1e-13

WEIBULL_CASES = (
    (0.3, 10),
    (0.3, 100),
    (1.7, 10),
    (1.7, 100),
    (1.7, 300),
    (20.0, 10),
    (20.0, 100),
)
GUMBEL_CASES = (
    (0.0, 1.0, 10),
    (0.0, 1.0, 300),
    (1e6, 3.0, 100),
    (-50.0, 0.01, 100),
    (5.0, 50.0, 10),
)


class WeibullReferenceTests(unittest.TestCase):
    def test_the_fit_is_the_high_precision_root(self) -> None:
        for shape, size in WEIBULL_CASES:
            values = weibull_sample(shape, 1000.0, size, 40 + size)
            result: Any = fit(NumpySource(values), "weibull_min").fit
            reference_shape, reference_scale, reference_likelihood = mp_weibull(
                values, result.shape
            )
            with self.subTest(shape=shape, size=size):
                self.assertLess(relative(result.shape, reference_shape), MP_TOLERANCE)
                self.assertLess(relative(result.scale, reference_scale), MP_TOLERANCE)
                self.assertLess(relative(result.log_likelihood, reference_likelihood), MP_TOLERANCE)

    def test_the_fit_agrees_with_the_in_memory_fit_within_its_resolution(self) -> None:
        for shape, size in WEIBULL_CASES:
            values = weibull_sample(shape, 1000.0, size, 40 + size)
            ours: Any = fit(NumpySource(values), "weibull_min").fit
            theirs: Any = fit_weibull([ExactLifetime(float(v)) for v in values])
            with self.subTest(shape=shape, size=size):
                self.assertLess(abs(ours.shape - theirs.shape) / theirs.shape, IN_MEMORY_TOLERANCE)
                self.assertLess(abs(ours.scale - theirs.scale) / theirs.scale, IN_MEMORY_TOLERANCE)
                self.assertLess(
                    abs(ours.log_likelihood - theirs.log_likelihood) / abs(theirs.log_likelihood),
                    LIKELIHOOD_TOLERANCE,
                )
                self.assertGreaterEqual(ours.log_likelihood, theirs.log_likelihood - 1e-9)

    def test_extreme_samples_agree_with_the_high_precision_root(self) -> None:
        rng = np.random.default_rng(1)
        cases = {
            "twelve decades": np.array([1e-300, 1e300, 1.0, 1e100, 1e-100]),
            "tight": 1000.0 * (1.0 + 1e-5 * rng.standard_normal(40)),
            "tighter": 1000.0 * (1.0 + 2e-8 * rng.standard_normal(40)),
            "subnormal": np.array([5e-324, 1e-323, 2e-323, 3e-323]),
        }
        for name, values in cases.items():
            result: Any = fit(NumpySource(values), "weibull_min").fit
            reference_shape, reference_scale, _ = mp_weibull(values, result.shape)
            logs = np.log(values)
            limit = max(
                MP_TOLERANCE, 8.0 * 2.0**-52 * float(np.abs(logs).max() / (logs.max() - logs.min()))
            )
            with self.subTest(case=name):
                self.assertLess(relative(result.shape, reference_shape), limit)
                self.assertLessEqual(
                    float(abs(result.scale - reference_scale)),
                    float(np.spacing(result.scale)) + limit * float(reference_scale),
                )


class GumbelReferenceTests(unittest.TestCase):
    def test_the_fit_is_the_high_precision_root(self) -> None:
        for location, scale, size in GUMBEL_CASES:
            values = gumbel_sample(location, scale, size, 70 + size)
            result: Any = fit(NumpySource(values), "gumbel_right").fit
            reference_location, reference_scale, reference_likelihood, _ = mp_gumbel(
                values, result.scale
            )
            with self.subTest(location=location, scale=scale, size=size):
                self.assertLess(relative(result.scale, reference_scale), MP_TOLERANCE)
                rounding = 2.0 * float(np.spacing(abs(result.location)))
                self.assertLessEqual(
                    float(abs(result.location - reference_location)),
                    rounding + MP_TOLERANCE * float(reference_scale),
                )
                self.assertLess(relative(result.log_likelihood, reference_likelihood), MP_TOLERANCE)

    def test_the_fit_agrees_with_the_in_memory_fit_within_its_resolution(self) -> None:
        for location, scale, size in GUMBEL_CASES:
            values = gumbel_sample(location, scale, size, 70 + size)
            ours: Any = fit(NumpySource(values), "gumbel_right").fit
            theirs: Any = fit_gumbel_right([ExactValue(float(v)) for v in values])
            with self.subTest(location=location, scale=scale, size=size):
                self.assertLess(abs(ours.scale - theirs.scale) / theirs.scale, IN_MEMORY_TOLERANCE)
                self.assertLess(
                    abs(ours.location - theirs.location) / theirs.scale, IN_MEMORY_TOLERANCE
                )
                self.assertLess(
                    abs(ours.log_likelihood - theirs.log_likelihood) / abs(theirs.log_likelihood),
                    LIKELIHOOD_TOLERANCE,
                )

    def test_extreme_samples_agree_with_the_high_precision_root(self) -> None:
        rng = np.random.default_rng(2)
        cases = {
            "huge": rng.gumbel(1e300, 1e290, 30),
            "ulp apart": np.array([1e6, np.nextafter(1e6, 2e6), 1e6]),
            "tiny": np.array([1e-320, 3e-320, 2e-320, 5e-320]),
            "wide": np.array([-1e308, 1e308, 0.0, 5e307]),
        }
        for name, values in cases.items():
            result: Any = fit(NumpySource(values), "gumbel_right").fit
            _, reference_scale, _, _ = mp_gumbel(values, result.scale)
            with self.subTest(case=name):
                self.assertLessEqual(
                    float(abs(result.scale - reference_scale)),
                    float(np.spacing(result.scale)) + MP_TOLERANCE * float(reference_scale),
                )


if __name__ == "__main__":
    unittest.main()
