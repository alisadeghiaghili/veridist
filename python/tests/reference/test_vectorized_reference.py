"""Array evaluation of every family against mpmath, through one broadcast call per case.

The mpmath oracle and the parameter grid are the ones the scalar reference tests use,
so the array path is held to the same accuracy as the scalar path.  Elements whose
scalar call would raise an overflow error (``exp``/``pow`` of a huge argument) are
checked separately against the exact limit.
"""

from __future__ import annotations

import unittest
from math import isfinite

import numpy as np

from tests.reference.test_unified_distribution_reference import (
    GRID,
    PROBABILITIES,
    _close,
    _oracle,
)
from veridist.families.registry import FamilyId
from veridist.statistics.distributions import cdf, logpdf, ppf, sf


def _flatten(family: FamilyId) -> tuple[np.ndarray, dict[str, np.ndarray], list[dict[str, float]]]:
    """Return one element per grid point: the points, the parameter arrays and the dicts."""

    points: list[float] = []
    rows: list[dict[str, float]] = []
    for parameters, grid_points in GRID[family]:
        for x in grid_points:
            points.append(x)
            rows.append(parameters)
    names = list(rows[0])
    return (
        np.array(points),
        {name: np.array([row[name] for row in rows]) for name in names},
        rows,
    )


class VectorizedReferenceTests(unittest.TestCase):
    def test_logpdf_cdf_and_sf_arrays_match_mpmath(self) -> None:
        for family in GRID:
            points, parameters, rows = _flatten(family)
            got = {
                "logpdf": logpdf(family, points, **parameters),
                "cdf": cdf(family, points, **parameters),
                "sf": sf(family, points, **parameters),
            }
            for index, row in enumerate(rows):
                oracle = _oracle(family, row)
                x = float(points[index])
                with self.subTest(family=family.value, row=row, x=x):
                    self.assertTrue(_close(got["logpdf"][index], oracle.logpdf(x), 1e-12, 1e-13))
                    self.assertTrue(_close(got["cdf"][index], oracle.cdf(x), 1e-12, 1e-300))
                    self.assertTrue(_close(got["sf"][index], oracle.sf(x), 1e-12, 1e-300))

    def test_ppf_arrays_invert_the_oracle_cdf_and_sf(self) -> None:
        probabilities = np.array(PROBABILITIES).reshape(-1, 1)
        for family, cases in GRID.items():
            names = list(cases[0][0])
            parameters = {name: np.array([case[0][name] for case in cases]) for name in names}
            quantiles = ppf(family, probabilities, **parameters)
            self.assertEqual(quantiles.shape, (len(PROBABILITIES), len(cases)))
            for i, probability in enumerate(PROBABILITIES):
                for j, (case, _) in enumerate(cases):
                    oracle = _oracle(family, case)
                    quantile = float(quantiles[i, j])
                    with self.subTest(family=family.value, case=case, q=probability):
                        self.assertTrue(isfinite(quantile))
                        if probability <= 0.5:
                            self.assertTrue(_close(probability, oracle.cdf(quantile), 1e-9))
                        else:
                            self.assertTrue(_close(1.0 - probability, oracle.sf(quantile), 1e-7))

    def test_dense_native_grids_match_mpmath(self) -> None:
        """Dense sweeps of the numpy-native kernels: both tails, and ``x`` close to the scale."""

        grid = np.linspace(-6.0, 30.0, 181)
        gumbel = {"location": np.full(grid.shape, 1.25), "scale": np.full(grid.shape, 0.8)}
        x = 1.25 + 0.8 * grid
        got = logpdf("gumbel_right", x, **gumbel)
        oracle = _oracle(FamilyId.GUMBEL_RIGHT, {"location": 1.25, "scale": 0.8})
        for index, value in enumerate(x):
            with self.subTest(family="gumbel_right", x=float(value)):
                self.assertTrue(_close(got[index], oracle.logpdf(float(value)), 1e-12, 1e-13))
        # Up to twice the scale: beyond it the survival probability of the largest shape
        # is subnormal, where relative accuracy is not meaningful.
        ratios = np.concatenate((np.linspace(0.02, 2.0, 150), 1.0 + np.linspace(-1e-6, 1e-6, 21)))
        for shape in (0.7, 1.0, 2.5, 6.0):
            weibull = {"shape": np.full(ratios.shape, shape), "scale": np.full(ratios.shape, 3.0)}
            x = 3.0 * ratios
            logpdf_values = logpdf("weibull_min", x, **weibull)
            cdf_values = cdf("weibull_min", x, **weibull)
            sf_values = sf("weibull_min", x, **weibull)
            oracle = _oracle(FamilyId.WEIBULL_MIN, {"shape": shape, "scale": 3.0})
            for index, value in enumerate(x):
                with self.subTest(family="weibull_min", shape=shape, x=float(value)):
                    self.assertTrue(
                        _close(logpdf_values[index], oracle.logpdf(float(value)), 1e-12, 1e-13)
                    )
                    self.assertTrue(_close(cdf_values[index], oracle.cdf(float(value)), 1e-12))
                    self.assertTrue(_close(sf_values[index], oracle.sf(float(value)), 1e-12))

    def test_saturated_cdf_and_sf_are_the_exact_limits_where_exp_and_pow_overflow(self) -> None:
        far_left = np.array([-800.0, -1e5])
        self.assertTrue(np.all(cdf("gumbel_right", far_left, location=0.0, scale=1.0) == 0.0))
        self.assertTrue(np.all(sf("gumbel_right", far_left, location=0.0, scale=1.0) == 1.0))
        far_right = np.array([1e200, 1e300])
        self.assertTrue(np.all(cdf("weibull_min", far_right, shape=3.0, scale=1.0) == 1.0))
        self.assertTrue(np.all(sf("weibull_min", far_right, shape=3.0, scale=1.0) == 0.0))
        self.assertTrue(np.all(cdf("exponential", far_right, rate=1e10) == 1.0))
        self.assertTrue(np.all(sf("exponential", far_right, rate=1e10) == 0.0))


if __name__ == "__main__":
    unittest.main()
