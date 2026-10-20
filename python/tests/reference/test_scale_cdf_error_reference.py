"""The declared evaluation error of the fitted distribution function against ``mpmath``.

The large-n goodness-of-fit report is certified up to ``cdf_error_bound``. For each family the
distribution function is evaluated by the library on a grid of parameters, at points chosen from
the bulk and from both tails (probabilities from ``1e-14`` to ``1 - 1e-14``), and compared with the
exact value at 40 digits for the same binary64 point and parameters. The measured error must be at
most a quarter of the declared bound, so that the bound is never the sharpest figure that holds.
"""

from __future__ import annotations

import unittest
from typing import Any

import mpmath
import numpy as np

from veridist.families.registry import FamilyId
from veridist.scale._gof import cdf_error_bound
from veridist.statistics.distributions import cdf, ppf

mpmath.mp.dps = 40

POINTS = 60
MARGIN = 4.0
QUICK_POINTS = 12

CASES: dict[str, list[dict[str, float]]] = {
    "exponential": [{"rate": r} for r in (1e-6, 1e-2, 1.0, 50.0, 1e6)],
    "weibull_min": [
        {"shape": k, "scale": s}
        for k in (0.1, 0.5, 1.7, 10.0, 100.0, 1000.0, 1e4)
        for s in (1.0, 1e3)
    ],
    "gumbel_right": [
        {"location": m, "scale": s} for m in (0.0, 5.0, 1e3, 1e6) for s in (1e-3, 1.0, 100.0)
    ],
    "normal": [{"mu": m, "sigma": s} for m in (0.0, 5.0, 1e3, 1e6) for s in (1e-3, 1.0, 100.0)],
    "lognormal": [
        {"mu_log": m, "sigma_log": s}
        for m in (0.0, 5.0, 20.0, -20.0)
        for s in (1e-3, 0.1, 1.0, 3.0)
    ],
    "gamma": [
        {"shape": a, "scale": s}
        for a in (0.1, 0.5, 1.0, 3.0, 30.0, 300.0, 1000.0)
        for s in (1.0, 50.0)
    ],
}


# The parameters the collected test checks: for each family the ends of the grid above and one
# value in the middle of it. The whole grid, at 60 points each, is in
# ``tests/statistical/test_scale_inference_full_size.py``.
QUICK_CASES: dict[str, list[dict[str, float]]] = {
    "exponential": [{"rate": r} for r in (1e-6, 1.0, 1e6)],
    "weibull_min": [
        {"shape": 0.1, "scale": 1.0},
        {"shape": 1.7, "scale": 1e3},
        {"shape": 1000.0, "scale": 1.0},
        {"shape": 1e4, "scale": 1e3},
    ],
    "gumbel_right": [
        {"location": 0.0, "scale": 1e-3},
        {"location": 5.0, "scale": 1.0},
        {"location": 1e6, "scale": 100.0},
    ],
    "normal": [
        {"mu": 0.0, "sigma": 1e-3},
        {"mu": 5.0, "sigma": 1.0},
        {"mu": 1e6, "sigma": 100.0},
    ],
    "lognormal": [
        {"mu_log": 0.0, "sigma_log": 1e-3},
        {"mu_log": 5.0, "sigma_log": 1.0},
        {"mu_log": -20.0, "sigma_log": 3.0},
    ],
    "gamma": [
        {"shape": 0.1, "scale": 1.0},
        {"shape": 3.0, "scale": 50.0},
        {"shape": 1000.0, "scale": 1.0},
    ],
}


def exact_cdf(family: str, x: float, p: dict[str, float]) -> Any:
    point = mpmath.mpf(x)
    if family == "exponential":
        return 1 - mpmath.exp(-mpmath.mpf(p["rate"]) * point)
    if family == "weibull_min":
        return 1 - mpmath.exp(-((point / mpmath.mpf(p["scale"])) ** mpmath.mpf(p["shape"])))
    if family == "gumbel_right":
        return mpmath.exp(
            -mpmath.exp(-(point - mpmath.mpf(p["location"])) / mpmath.mpf(p["scale"]))
        )
    if family == "normal":
        return mpmath.ncdf(point, mpmath.mpf(p["mu"]), mpmath.mpf(p["sigma"]))
    if family == "lognormal":
        return mpmath.ncdf(mpmath.log(point), mpmath.mpf(p["mu_log"]), mpmath.mpf(p["sigma_log"]))
    return mpmath.gammainc(
        mpmath.mpf(p["shape"]), 0, point / mpmath.mpf(p["scale"]), regularized=True
    )


class CdfErrorReferenceTests(unittest.TestCase):
    CASES = QUICK_CASES
    POINTS = QUICK_POINTS

    def test_the_measured_error_is_below_a_quarter_of_the_declared_bound(self) -> None:
        rng = np.random.default_rng(7)
        for family, grid in self.CASES.items():
            for parameters in grid:
                with self.subTest(family=family, parameters=parameters):
                    probabilities = np.concatenate(
                        [
                            rng.uniform(1e-9, 1 - 1e-9, self.POINTS),
                            10 ** rng.uniform(-14, -3, self.POINTS // 3),
                            1 - 10 ** rng.uniform(-14, -3, self.POINTS // 3),
                        ]
                    )
                    points = np.asarray(ppf(family, probabilities, **parameters))
                    computed = cdf(family, points, **parameters)
                    worst = max(
                        abs(mpmath.mpf(float(got)) - exact_cdf(family, float(x), parameters))
                        for got, x in zip(computed, points, strict=True)
                    )
                    bound = cdf_error_bound(FamilyId(family), parameters)
                    self.assertLess(float(worst) * MARGIN, bound)

    def test_the_bound_grows_with_what_amplifies_the_error(self) -> None:
        weibull = [
            cdf_error_bound(FamilyId.WEIBULL_MIN, {"shape": k, "scale": 1.0})
            for k in (1.0, 10.0, 1000.0)
        ]
        self.assertEqual(weibull, sorted(weibull))
        sigmas = [
            cdf_error_bound(FamilyId.LOGNORMAL, {"mu_log": 3.0, "sigma_log": s})
            for s in (1.0, 0.1, 0.001)
        ]
        self.assertEqual(sigmas, sorted(sigmas))
        shapes = [
            cdf_error_bound(FamilyId.GAMMA, {"shape": a, "scale": 1.0}) for a in (1.0, 100.0, 1e3)
        ]
        self.assertEqual(shapes, sorted(shapes))


if __name__ == "__main__":
    unittest.main()
