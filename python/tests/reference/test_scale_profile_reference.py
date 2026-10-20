"""Profile-likelihood intervals of the multi-pass scale fits against the definition at 40 digits.

The references are rebuilt with ``mpmath`` from the exact binary64 data and share no code with the
implementation: the interval of the shape is the pair of roots of the profile deviance and the
interval of every other quantity is the range of that quantity over the likelihood region, found as
the extremes of its value along the boundary of the region. The agreement asserted is a relative
``1e-10``; the intervals of the tests agree to about ``1e-15``, and the allowance of the
implementation for rounding of the exact sums of ``n`` rows is ``n * 2**-40`` of deviance, which
moves an end by a relative amount far below ``1e-10`` at these sizes.
"""

from __future__ import annotations

import unittest
from typing import Any

import mpmath

from tests.unit.scale_inference_oracle import (
    Likelihood,
    chi_square_quantile,
    gumbel_data,
    relative,
    weibull_data,
)
from veridist.scale import _profile_intervals as pi
from veridist.scale._fit import fit
from veridist.scale._sources import NumpySource


class ProfileReferenceTests(unittest.TestCase):
    """The definition costs ``rows * iterations`` operations at 40 digits per end, so these
    tests use few rows; the same checks at 200 rows and 60 steps are in
    ``tests/statistical/test_scale_inference_full_size.py``."""

    ROWS = 40
    ITERATIONS = 30

    def check_weibull(self, requested: list[pi.Quantity], names: tuple[str, ...]) -> None:
        values = weibull_data(1.7, 40.0, self.ROWS, 4)
        result = fit(NumpySource(values), "weibull_min")
        out = pi.profile_intervals(NumpySource(values), result, quantities=requested)
        likelihood = Likelihood(values, "weibull")
        delta = chi_square_quantile(0.95)
        shape = out.parameters["shape"]
        low, high, maximum = likelihood.interval_of_k(
            result.fit.shape,
            delta,
            shape.lower,
            shape.upper,  # type: ignore[union-attr]
        )
        self.assertLess(relative(shape.lower, low), 1e-10)
        self.assertLess(relative(shape.upper, high), 1e-10)
        quantile_level = -mpmath.log(1 - mpmath.mpf("0.1"))
        definitions: dict[str, Any] = {
            "scale": lambda k, s: s,
            "quantile:0.1": lambda k, s: s * quantile_level ** (1 / k),
            "mean": lambda k, s: s * mpmath.gamma(1 + 1 / k),
            "survival:30.0": lambda k, s: mpmath.exp(-((mpmath.mpf(30) / s) ** k)),
        }
        for name in names:
            got = out.parameters["scale"] if name == "scale" else out.derived[name]
            for upper in (False, True):
                expected = likelihood.extreme(
                    definitions[name], low, high, maximum, delta, upper, upper, self.ITERATIONS
                )
                reported = got.upper if upper else got.lower
                with self.subTest(name=name, upper=upper):
                    self.assertLess(relative(reported, expected), 1e-10)

    def test_the_weibull_shape_and_scale_equal_the_definition_at_forty_digits(self) -> None:
        self.check_weibull([], ("scale",))

    def test_the_weibull_quantile_equals_the_definition_at_forty_digits(self) -> None:
        self.check_weibull([pi.quantile(0.1)], ("quantile:0.1",))

    def test_the_weibull_mean_equals_the_definition_at_forty_digits(self) -> None:
        self.check_weibull([pi.mean()], ("mean",))

    def test_the_weibull_survival_equals_the_definition_at_forty_digits(self) -> None:
        self.check_weibull([pi.survival(30.0)], ("survival:30.0",))

    def test_the_gumbel_intervals_equal_the_definition_at_forty_digits(self) -> None:
        values = gumbel_data(5.0, 2.0, self.ROWS, 4)
        result = fit(NumpySource(values), "gumbel_right")
        out = pi.profile_intervals(NumpySource(values), result)
        likelihood = Likelihood(values, "gumbel")
        unit = likelihood.unit
        delta = chi_square_quantile(0.95)
        scale = out.parameters["scale"]
        low, high, maximum = likelihood.interval_of_k(
            result.fit.scale / float(unit),  # type: ignore[union-attr]
            delta,
            scale.lower / float(unit),
            scale.upper / float(unit),
        )
        self.assertLess(relative(scale.lower, low * unit), 1e-10)
        self.assertLess(relative(scale.upper, high * unit), 1e-10)
        location = out.parameters["location"]
        for upper in (False, True):
            expected = likelihood.extreme(
                lambda k, m: m, low, high, maximum, delta, upper, upper, self.ITERATIONS
            )
            reported = location.upper if upper else location.lower
            self.assertLess(relative(reported, expected * unit), 1e-10)


if __name__ == "__main__":
    unittest.main()
