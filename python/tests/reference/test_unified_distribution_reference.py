"""Every family's scalar operations against mpmath, through the unified calling form.

The oracle is written directly from each family's definition with mpmath at 60
digits, independently of the code under test.  ``ppf`` is checked by feeding the
returned quantile back through the oracle CDF/SF.
"""

from __future__ import annotations

import unittest
from collections.abc import Callable
from math import isfinite

import mpmath
import numpy as np

from veridist.families.registry import FAMILY_REGISTRY, FamilyId, Operation
from veridist.statistics.distributions import cdf, logpdf, ppf, sample, sf

mpmath.mp.dps = 60
mp = mpmath.mp

_PI = mp.pi


def _normal_cdf(x: float, mu: float, sigma: float) -> mpmath.mpf:
    return mp.erfc(-(mp.mpf(x) - mu) / (sigma * mp.sqrt(2))) / 2


def _normal_sf(x: float, mu: float, sigma: float) -> mpmath.mpf:
    return mp.erfc((mp.mpf(x) - mu) / (sigma * mp.sqrt(2))) / 2


class _Oracle:
    """One family's mpmath definition: log-density, CDF and SF at a point."""

    def __init__(
        self,
        logpdf: Callable[[float], mpmath.mpf],
        cdf: Callable[[float], mpmath.mpf],
        sf: Callable[[float], mpmath.mpf],
    ) -> None:
        self.logpdf, self.cdf, self.sf = logpdf, cdf, sf


def _oracle(family: FamilyId, p: dict[str, float]) -> _Oracle:
    if family is FamilyId.NORMAL:
        mu, sigma = p["mu"], p["sigma"]
        return _Oracle(
            lambda x: (
                -((mp.mpf(x) - mu) ** 2) / (2 * sigma**2) - mp.log(sigma) - mp.log(2 * _PI) / 2
            ),
            lambda x: _normal_cdf(x, mu, sigma),
            lambda x: _normal_sf(x, mu, sigma),
        )
    if family is FamilyId.GAMMA:
        k, theta = mp.mpf(p["shape"]), mp.mpf(p["scale"])
        return _Oracle(
            lambda x: (k - 1) * mp.log(x) - mp.mpf(x) / theta - mp.loggamma(k) - k * mp.log(theta),
            lambda x: mp.gammainc(k, 0, mp.mpf(x) / theta, regularized=True),
            lambda x: mp.gammainc(k, mp.mpf(x) / theta, mp.inf, regularized=True),
        )
    if family is FamilyId.WEIBULL_MIN:
        k, s = mp.mpf(p["shape"]), mp.mpf(p["scale"])
        return _Oracle(
            lambda x: mp.log(k / s) + (k - 1) * mp.log(mp.mpf(x) / s) - (mp.mpf(x) / s) ** k,
            lambda x: 1 - mp.exp(-((mp.mpf(x) / s) ** k)),
            lambda x: mp.exp(-((mp.mpf(x) / s) ** k)),
        )
    if family is FamilyId.LOGNORMAL:
        mu, sigma = mp.mpf(p["mu_log"]), mp.mpf(p["sigma_log"])
        return _Oracle(
            lambda x: (
                -mp.log(x)
                - mp.log(sigma)
                - mp.log(2 * _PI) / 2
                - (mp.log(x) - mu) ** 2 / (2 * sigma**2)
            ),
            lambda x: _normal_cdf(float(mp.log(x)), mu, sigma),
            lambda x: _normal_sf(float(mp.log(x)), mu, sigma),
        )
    if family is FamilyId.GUMBEL_RIGHT:
        loc, scale = mp.mpf(p["location"]), mp.mpf(p["scale"])
        return _Oracle(
            lambda x: (
                -mp.log(scale) - (mp.mpf(x) - loc) / scale - mp.exp(-(mp.mpf(x) - loc) / scale)
            ),
            lambda x: mp.exp(-mp.exp(-(mp.mpf(x) - loc) / scale)),
            lambda x: 1 - mp.exp(-mp.exp(-(mp.mpf(x) - loc) / scale)),
        )
    rate = mp.mpf(p["rate"])
    return _Oracle(
        lambda x: mp.log(rate) - rate * mp.mpf(x),
        lambda x: 1 - mp.exp(-rate * mp.mpf(x)),
        lambda x: mp.exp(-rate * mp.mpf(x)),
    )


GRID: dict[FamilyId, list[tuple[dict[str, float], tuple[float, ...]]]] = {
    FamilyId.NORMAL: [
        ({"mu": 0.0, "sigma": 1.0}, (-6.0, -1.5, 0.0, 0.7, 4.0)),
        ({"mu": -3.5, "sigma": 0.25}, (-4.5, -3.5, -3.0)),
    ],
    FamilyId.GAMMA: [
        ({"shape": 0.5, "scale": 2.0}, (0.01, 0.5, 3.0, 20.0)),
        ({"shape": 2.0, "scale": 3.0}, (0.1, 1.0, 6.0, 40.0)),
        ({"shape": 9.5, "scale": 0.4}, (1.0, 3.8, 9.0)),
    ],
    FamilyId.WEIBULL_MIN: [
        ({"shape": 1.5, "scale": 4.0}, (0.05, 1.0, 4.0, 12.0)),
        ({"shape": 0.7, "scale": 0.3}, (0.01, 0.3, 2.0)),
        ({"shape": 6.0, "scale": 10.0}, (3.0, 9.0, 14.0)),
    ],
    FamilyId.LOGNORMAL: [
        ({"mu_log": 0.5, "sigma_log": 0.75}, (0.05, 1.0, 1.7, 30.0)),
        ({"mu_log": -2.0, "sigma_log": 2.0}, (0.01, 0.5, 9.0)),
    ],
    FamilyId.GUMBEL_RIGHT: [
        ({"location": 0.0, "scale": 2.0}, (-6.0, -1.0, 0.0, 3.0, 12.0)),
        ({"location": 10.0, "scale": 0.5}, (9.0, 10.0, 12.5)),
    ],
    FamilyId.EXPONENTIAL: [
        ({"rate": 2.0}, (0.01, 0.5, 3.0, 20.0)),
        ({"rate": 0.003}, (1.0, 300.0, 5000.0)),
    ],
}

PROBABILITIES = (1e-9, 1e-4, 0.025, 0.3, 0.5, 0.9, 0.999, 1.0 - 1e-9)


def _close(got: float, want: mpmath.mpf, relative: float, absolute: float = 0.0) -> bool:
    return abs(mp.mpf(got) - want) <= relative * abs(want) + absolute


class UnifiedOperationReferenceTests(unittest.TestCase):
    def test_every_family_is_on_the_grid_and_advertises_every_operation(self) -> None:
        self.assertEqual(set(GRID), set(FamilyId))
        for spec in FAMILY_REGISTRY.list():
            for operation in Operation:
                self.assertTrue(spec.supports(operation), (spec.id, operation))

    def test_logpdf_cdf_and_sf_match_mpmath(self) -> None:
        for family, cases in GRID.items():
            for parameters, points in cases:
                oracle = _oracle(family, parameters)
                for x in points:
                    with self.subTest(family=family, parameters=parameters, x=x):
                        self.assertTrue(
                            _close(logpdf(family, x, **parameters), oracle.logpdf(x), 1e-12, 1e-13)
                        )
                        self.assertTrue(
                            _close(cdf(family, x, **parameters), oracle.cdf(x), 1e-12, 1e-300)
                        )
                        self.assertTrue(
                            _close(sf(family, x, **parameters), oracle.sf(x), 1e-12, 1e-300)
                        )

    def test_ppf_inverts_the_oracle_cdf_and_sf(self) -> None:
        for family, cases in GRID.items():
            for parameters, _ in cases:
                oracle = _oracle(family, parameters)
                for probability in PROBABILITIES:
                    with self.subTest(family=family, parameters=parameters, q=probability):
                        quantile = ppf(family, probability, **parameters)
                        self.assertTrue(isfinite(quantile))
                        if probability <= 0.5:
                            self.assertTrue(_close(probability, oracle.cdf(quantile), 1e-9))
                        else:
                            self.assertTrue(
                                _close(1.0 - probability, oracle.sf(quantile), 1e-7),
                                (family, parameters, probability, quantile),
                            )

    def test_cdf_is_zero_and_sf_is_one_at_and_below_the_origin_for_fixed_location_families(
        self,
    ) -> None:
        for family, cases in GRID.items():
            if FAMILY_REGISTRY.families[family].fixed_location != 0.0:
                continue
            parameters = cases[0][0]
            for x in (-5.0, -1e-300, 0.0):
                with self.subTest(family=family, x=x):
                    self.assertEqual(cdf(family, x, **parameters), 0.0)
                    self.assertEqual(sf(family, x, **parameters), 1.0)

    def test_sampling_has_the_requested_size_and_support(self) -> None:
        for family, cases in GRID.items():
            parameters = cases[0][0]
            draws = sample(family, 200, rng=np.random.default_rng(3), **parameters)
            with self.subTest(family=family):
                self.assertEqual(draws.shape, (200,))
                self.assertTrue(np.isfinite(draws).all())
                if FAMILY_REGISTRY.families[family].fixed_location == 0.0:
                    self.assertTrue((draws >= 0.0).all())


if __name__ == "__main__":
    unittest.main()
