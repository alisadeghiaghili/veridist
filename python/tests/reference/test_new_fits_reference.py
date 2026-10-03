"""Normal, gamma and right-Gumbel fits against independent reference optima.

Provenance of every constant below: scipy 1.18.1 with numpy 2.5.3.  The data are
frozen literals (drawn once with ``numpy.random.default_rng(seed)`` and rounded to
three decimals; the seed is given at each dataset), so these tests do not depend
on any random-number stream.

* Uncensored references: ``scipy.stats.norm.fit``, ``scipy.stats.gumbel_r.fit`` and
  ``scipy.stats.gamma.fit(data, floc=0)``.
* Right-censored references: ``scipy.optimize.minimize`` (BFGS, then Nelder-Mead
  polish to ``xatol=1e-12``) of the negative log-likelihood written from
  ``scipy.stats.<dist>.logpdf`` for the events plus ``logsf`` at the censoring
  point for each censored observation.  Observations at or above the censoring
  point are replaced by a right-censored observation at that point.

The acceptance tolerance is a relative 1e-6, far above what the golden-section
search delivers (about 1e-8) and far below any estimate difference that matters.
"""

from __future__ import annotations

import unittest

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue, RightCensoredValue
from veridist.families.gamma import GammaFitSuccess, fit_gamma
from veridist.families.gumbel import GumbelFitSuccess, fit_gumbel_right
from veridist.families.normal import NormalFitSuccess, fit_normal
from veridist.families.registry import FamilyId
from veridist.statistics.lifetime_log_likelihood import (
    reduce_lifetime_log_likelihood_chunks,
    reduce_value_log_likelihood_chunks,
)
from veridist.statistics.log_likelihood import LogLikelihoodSuccess, reduce_log_likelihood_chunks

REL = 1e-6
REDUCER_REL = 1e-12

# numpy.random.default_rng(20260901).normal(100.0, 15.0, size=25), rounded to 3 decimals.
NORMAL_DATA = (
    99.455, 106.535, 111.712, 109.871, 96.311, 100.971, 111.878, 91.322, 91.11, 108.838,
    124.15, 53.405, 98.593, 88.337, 103.349, 93.884, 97.609, 105.417, 102.518, 110.046,
    92.529, 85.801, 74.256, 98.035, 96.2,
)  # fmt: skip
NORMAL_CENSORING_POINT = 110.0
# scipy.stats.norm.fit(NORMAL_DATA): the maximum-likelihood sigma, divisor n.
NORMAL_UNCENSORED = (98.08527999999998, 13.558456999290147, -100.64872546573645)
# scipy.optimize.minimize on the censored log-likelihood.
NORMAL_CENSORED = (98.69152298764182, 14.423128211017762, -89.8389202336652)

# numpy.random.default_rng(20260902).gumbel(50.0, 7.0, size=25), rounded to 3 decimals.
GUMBEL_DATA = (
    43.41, 50.124, 61.235, 42.231, 46.243, 52.158, 50.865, 44.28, 60.041, 48.633,
    49.49, 59.602, 47.596, 66.214, 51.48, 51.615, 53.119, 58.585, 37.848, 56.79,
    44.14, 59.934, 52.655, 54.972, 76.959,
)  # fmt: skip
GUMBEL_CENSORING_POINT = 55.0
# scipy.stats.gumbel_r.fit(GUMBEL_DATA).
GUMBEL_UNCENSORED = (48.9457225126579, 6.968165090920796, -87.39339202701417)
GUMBEL_CENSORED = (48.96001384037929, 7.0477399730497305, -63.48565093223071)

# numpy.random.default_rng(20260903).gamma(2.5, 4.0, size=25), rounded to 3 decimals.
GAMMA_DATA = (
    8.504, 18.042, 10.226, 3.112, 8.333, 2.251, 8.956, 15.848, 7.33, 9.592,
    12.179, 0.416, 15.786, 18.654, 7.692, 1.47, 24.849, 5.343, 2.069, 6.589,
    9.366, 6.347, 7.002, 3.888, 10.515,
)  # fmt: skip
GAMMA_CENSORING_POINT = 14.0
# scipy.stats.gamma.fit(GAMMA_DATA, floc=0).
GAMMA_UNCENSORED = (1.8736867126088326, 4.789680120805535, -77.4199074339232)
GAMMA_CENSORED = (1.732450866981071, 5.387995142977211, -64.60926965520174)


def _values(data: tuple[float, ...], censoring_point: float | None) -> list[object]:
    return [
        ExactValue(x)
        if censoring_point is None or x < censoring_point
        else RightCensoredValue(censoring_point)
        for x in data
    ]


def _lifetimes(data: tuple[float, ...], censoring_point: float | None) -> list[object]:
    return [
        ExactLifetime(x)
        if censoring_point is None or x < censoring_point
        else RightCensoredLifetime(censoring_point)
        for x in data
    ]


class NewFitReferenceTests(unittest.TestCase):
    def assertRelative(self, got: float, want: float, tolerance: float = REL) -> None:
        self.assertLessEqual(abs(got - want), tolerance * abs(want), (got, want))

    def test_normal_uncensored_is_the_closed_form_with_the_maximum_likelihood_sigma(self) -> None:
        result = fit_normal(_values(NORMAL_DATA, None))  # type: ignore[arg-type]
        assert isinstance(result, NormalFitSuccess)
        mu, sigma, log_likelihood = NORMAL_UNCENSORED
        self.assertRelative(result.mu, mu, 1e-12)
        self.assertRelative(result.sigma, sigma, 1e-12)
        self.assertRelative(result.log_likelihood, log_likelihood, 1e-12)
        # The divisor is n, not n - 1.
        n = len(NORMAL_DATA)
        mean = sum(NORMAL_DATA) / n
        mle = (sum((x - mean) ** 2 for x in NORMAL_DATA) / n) ** 0.5
        unbiased = (sum((x - mean) ** 2 for x in NORMAL_DATA) / (n - 1)) ** 0.5
        self.assertRelative(result.sigma, mle, 1e-12)
        self.assertGreater(abs(result.sigma - unbiased), 1e-3)
        self.assertEqual((result.observation_count, result.event_count), (25, 25))

    def test_normal_right_censored_matches_the_reference_optimum(self) -> None:
        result = fit_normal(_values(NORMAL_DATA, NORMAL_CENSORING_POINT))  # type: ignore[arg-type]
        assert isinstance(result, NormalFitSuccess)
        mu, sigma, log_likelihood = NORMAL_CENSORED
        self.assertRelative(result.mu, mu)
        self.assertRelative(result.sigma, sigma)
        self.assertRelative(result.log_likelihood, log_likelihood)
        self.assertEqual((result.event_count, result.censored_count), (21, 4))

    def test_gumbel_uncensored_matches_scipy(self) -> None:
        result = fit_gumbel_right(_values(GUMBEL_DATA, None))  # type: ignore[arg-type]
        assert isinstance(result, GumbelFitSuccess)
        location, scale, log_likelihood = GUMBEL_UNCENSORED
        self.assertRelative(result.location, location)
        self.assertRelative(result.scale, scale)
        self.assertRelative(result.log_likelihood, log_likelihood)

    def test_gumbel_right_censored_matches_the_reference_optimum(self) -> None:
        result = fit_gumbel_right(_values(GUMBEL_DATA, GUMBEL_CENSORING_POINT))  # type: ignore[arg-type]
        assert isinstance(result, GumbelFitSuccess)
        location, scale, log_likelihood = GUMBEL_CENSORED
        self.assertRelative(result.location, location)
        self.assertRelative(result.scale, scale)
        self.assertRelative(result.log_likelihood, log_likelihood)
        self.assertEqual((result.event_count, result.censored_count), (17, 8))

    def test_gamma_uncensored_matches_scipy(self) -> None:
        result = fit_gamma(_lifetimes(GAMMA_DATA, None))  # type: ignore[arg-type]
        assert isinstance(result, GammaFitSuccess)
        shape, scale, log_likelihood = GAMMA_UNCENSORED
        self.assertRelative(result.shape, shape)
        self.assertRelative(result.scale, scale)
        self.assertRelative(result.log_likelihood, log_likelihood)

    def test_gamma_right_censored_matches_the_reference_optimum(self) -> None:
        result = fit_gamma(_lifetimes(GAMMA_DATA, GAMMA_CENSORING_POINT))  # type: ignore[arg-type]
        assert isinstance(result, GammaFitSuccess)
        shape, scale, log_likelihood = GAMMA_CENSORED
        self.assertRelative(result.shape, shape)
        self.assertRelative(result.scale, scale)
        self.assertRelative(result.log_likelihood, log_likelihood)
        self.assertEqual((result.event_count, result.censored_count), (20, 5))

    def test_the_reducers_at_the_fitted_parameters_reproduce_each_log_likelihood(self) -> None:
        cases = (
            (FamilyId.NORMAL, fit_normal, _values(NORMAL_DATA, NORMAL_CENSORING_POINT)),
            (FamilyId.NORMAL, fit_normal, _values(NORMAL_DATA, None)),
            (FamilyId.GUMBEL_RIGHT, fit_gumbel_right, _values(GUMBEL_DATA, GUMBEL_CENSORING_POINT)),
            (FamilyId.GUMBEL_RIGHT, fit_gumbel_right, _values(GUMBEL_DATA, None)),
            (FamilyId.GAMMA, fit_gamma, _lifetimes(GAMMA_DATA, GAMMA_CENSORING_POINT)),
            (FamilyId.GAMMA, fit_gamma, _lifetimes(GAMMA_DATA, None)),
        )
        for family, fit, observations in cases:
            result = fit(observations)  # type: ignore[operator]
            with self.subTest(family=family, observations=len(observations)):
                reducer = (
                    reduce_value_log_likelihood_chunks
                    if family in (FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT)
                    else reduce_lifetime_log_likelihood_chunks
                )
                total = reducer(family, [observations], **result.parameters)
                assert isinstance(total, LogLikelihoodSuccess)
                self.assertRelative(total.total_log_likelihood, result.log_likelihood, REDUCER_REL)
                self.assertEqual(total.observation_count, result.observation_count)

    def test_uncensored_exact_values_also_match_the_density_reducer(self) -> None:
        values = _values(NORMAL_DATA, None)
        result = fit_normal(values)  # type: ignore[arg-type]
        density = reduce_log_likelihood_chunks(
            FamilyId.NORMAL,
            [list(NORMAL_DATA)],
            **result.parameters,  # type: ignore[union-attr]
        )
        assert isinstance(density, LogLikelihoodSuccess)
        self.assertRelative(density.total_log_likelihood, result.log_likelihood, REDUCER_REL)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
