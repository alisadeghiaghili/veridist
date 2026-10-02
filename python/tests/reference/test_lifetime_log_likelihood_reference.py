"""Independent mpmath references for the exact/right-censored lifetime reducer.

Every right-censored term of the reducer is checked against a 60-digit
``mpmath`` evaluation of the family's log-survival at the same binary64
inputs. The reducer is also checked, at the fitted parameters, against the
``log_likelihood`` the Weibull and lognormal fits report.

Tolerances follow what each evaluator can deliver, not what a chosen dataset
happens to satisfy:

* Weibull: the term is ``-exp(shape * log(t / scale))``, whose relative error
  is about ``|shape * log(t / scale)| * eps`` (at most ``709 * eps``).
* Lognormal: ``z`` is formed in binary64 and the tail uses an asymptotic
  expansion; the documented relative bound is ``1e-12``.
* Gamma: the log-domain evaluation sums terms of size
  ``max(t / scale, shape * |log(t / scale)|, |lgamma(shape)|)``, so its
  absolute error scales with that magnitude (see ``_gamma_budget``).
"""

from __future__ import annotations

import random
import unittest
from collections.abc import Callable, Sequence
from math import log, nextafter, ulp

import mpmath

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime
from veridist.families.lognormal import LognormalFitSuccess, fit_lognormal
from veridist.families.registry import FamilyId
from veridist.families.weibull import WeibullFitSuccess, fit_weibull
from veridist.statistics.lifetime_log_likelihood import reduce_lifetime_log_likelihood_chunks
from veridist.statistics.log_likelihood import LogLikelihoodSuccess

mpmath.mp.dps = 60

_EPS = 2.0**-52


def _censored_total(family: FamilyId, time: float, **parameters: float) -> float:
    result = reduce_lifetime_log_likelihood_chunks(
        family, [[RightCensoredLifetime(time)]], **parameters
    )
    assert isinstance(result, LogLikelihoodSuccess), result
    return result.total_log_likelihood


def _mp(value: float) -> mpmath.mpf:
    return mpmath.mpf(value)


class WeibullLogSurvivalReferenceTests(unittest.TestCase):
    def test_matches_mpmath_across_shape_scale_and_ratio_grid(self) -> None:
        worst = 0.0
        for shape in (0.1, 0.5, 1.0, 2.5, 10.0, 50.0):
            for scale in (1e-3, 1.0, 1e3):
                for ratio in (1e-6, 1e-3, 0.5, 1.0, 1.0 + 2.0**-52, 2.0, 5.0, 20.0):
                    time = ratio * scale
                    reference = -((_mp(time) / _mp(scale)) ** _mp(shape))
                    if float(abs(reference)) > 1.7e308:
                        continue
                    observed = _censored_total(
                        FamilyId.WEIBULL_MIN, time, shape=shape, scale=scale
                    )
                    error = abs(_mp(observed) - reference)
                    relative = float(error / abs(reference))
                    worst = max(worst, relative)
                    with self.subTest(shape=shape, scale=scale, ratio=ratio):
                        self.assertLessEqual(relative, 1e-12)
        self.assertLessEqual(worst, 1e-12)

    def test_survival_deep_in_the_tail_stays_finite_until_it_leaves_binary64(self) -> None:
        # exp(700) is representable; the term is about -1.0e304.
        shape, scale = 1.0, 1.0
        time = float(mpmath.exp(700))
        observed = _censored_total(FamilyId.WEIBULL_MIN, time, shape=shape, scale=scale)
        reference = -_mp(time)
        self.assertLessEqual(float(abs(_mp(observed) - reference) / abs(reference)), 1e-12)

    def test_survival_that_underflows_to_zero_is_exactly_zero(self) -> None:
        observed = _censored_total(FamilyId.WEIBULL_MIN, 1e-300, shape=50.0, scale=1.0)
        self.assertEqual(observed, -0.0)


_Z_TARGETS = (-5.0, -1.0, 0.0, 0.5, 1.0, 3.0, 10.0, 20.0, 36.0, 37.5, 40.0, 100.0)


class LognormalLogSurvivalReferenceTests(unittest.TestCase):
    def test_matches_mpmath_across_mu_sigma_and_z_grid(self) -> None:
        for mu in (-2.0, 0.0, 3.0):
            for sigma in (0.1, 0.5, 1.0, 3.0):
                for z_target in _Z_TARGETS:
                    log_time = mu + z_target * sigma
                    if abs(log_time) > 700.0:
                        continue
                    time = float(mpmath.exp(_mp(log_time)))
                    z = (mpmath.log(_mp(time)) - _mp(mu)) / _mp(sigma)
                    reference = mpmath.log(mpmath.ncdf(-z))
                    observed = _censored_total(
                        FamilyId.LOGNORMAL, time, mu_log=mu, sigma_log=sigma
                    )
                    error = float(abs(_mp(observed) - reference))
                    with self.subTest(mu=mu, sigma=sigma, z=z_target):
                        self.assertLessEqual(error, 1e-12 * max(1.0, float(abs(reference))))

    def test_cancellation_sensitive_inputs_use_the_bounded_decimal_path(self) -> None:
        # With sigma this small, a binary64 `log(t) - mu` is pure rounding
        # noise; the decimal path recovers the true standardized value.
        mu, sigma = 100.0, 1e-15
        time = float(mpmath.exp(_mp(mu)))
        for _ in range(12):
            time = nextafter(time, float("inf"))
        z = (mpmath.log(_mp(time)) - _mp(mu)) / _mp(sigma)
        self.assertGreater(float(z), 0.5)
        reference = mpmath.log(mpmath.ncdf(-z))
        observed = _censored_total(FamilyId.LOGNORMAL, time, mu_log=mu, sigma_log=sigma)
        self.assertLessEqual(
            float(abs(_mp(observed) - reference)), 1e-12 * float(abs(reference))
        )
        # The plain binary64 expression is not merely imprecise but wrong here.
        naive_z = (log(time) - mu) / sigma
        self.assertGreater(abs(naive_z - float(z)), 1e-3 * abs(float(z)))


def _gamma_budget(shape: float, ratio: float, reference: mpmath.mpf) -> float:
    """Error allowance: 1e-12 relative, plus 64 eps of the summed term magnitudes."""

    log_ratio = float(mpmath.log(_mp(ratio)))
    magnitude = max(ratio, shape * abs(log_ratio), abs(float(mpmath.loggamma(_mp(shape)))), 1.0)
    return 1e-12 * float(abs(reference)) + 1e-15 + 64.0 * _EPS * magnitude


class GammaLogSurvivalReferenceTests(unittest.TestCase):
    SHAPES = (
        1e-100, 1e-20, 1e-10, 1e-5, 0.005, 0.00999, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0,
        7.5, 40.0, 300.0, 1000.0, 20000.0,
    )  # fmt: skip

    def _ratios(self, shape: float) -> tuple[float, ...]:
        anchors = (1e-300, 1e-20, 1e-5, 0.01, 0.5, 1.0, 2.0)
        # Around the mode/regime switch (`shape + 1`); below shape 1 the
        # switch sits within the anchors already.
        around = (
            (shape * 0.5, shape * 0.9, shape, shape + 0.5, shape + 1.0, shape * 1.1)
            + (shape * 2.0, shape * 5.0)
            if shape >= 1.0
            else (shape + 0.5, shape + 1.0)
        )
        far = (700.0, 5000.0, 1e5, 1e6)
        return tuple(sorted({value for value in anchors + around + far if value > 0.0}))

    def test_matches_mpmath_across_shape_and_ratio_grid(self) -> None:
        checked = 0
        for shape in self.SHAPES:
            for ratio in self._ratios(shape):
                # `shape * 0.5` and `shape * 2.0` leave the series budget for
                # the largest shapes; those are covered by the typed-failure
                # tests rather than skipped silently.
                if shape >= 20000.0 and shape * 0.9 <= ratio <= shape + 1.0:
                    continue
                reference = mpmath.log(
                    mpmath.gammainc(_mp(shape), _mp(ratio), mpmath.inf, regularized=True)
                )
                observed = _censored_total(FamilyId.GAMMA, ratio, shape=shape, scale=1.0)
                with self.subTest(shape=shape, ratio=ratio):
                    self.assertLessEqual(
                        float(abs(_mp(observed) - reference)),
                        _gamma_budget(shape, ratio, reference),
                    )
                checked += 1
        self.assertGreater(checked, 150)

    def test_scale_enters_only_through_the_exact_ratio(self) -> None:
        for scale in (1e-3, 0.7, 1e3):
            for shape in (0.3, 2.0, 25.0):
                for ratio in (0.2, shape, 3.0 * shape + 5.0):
                    time = ratio * scale
                    exact_ratio = _mp(time) / _mp(scale)
                    reference = mpmath.log(
                        mpmath.gammainc(_mp(shape), exact_ratio, mpmath.inf, regularized=True)
                    )
                    observed = _censored_total(FamilyId.GAMMA, time, shape=shape, scale=scale)
                    with self.subTest(scale=scale, shape=shape, ratio=ratio):
                        self.assertLessEqual(
                            float(abs(_mp(observed) - reference)),
                            _gamma_budget(shape, float(exact_ratio), reference),
                        )

    def test_far_tail_below_the_smallest_binary64_stays_finite(self) -> None:
        # Q(3, 800) is about 1e-340, which underflows binary64; its logarithm
        # (about -792) is an ordinary finite number.
        for shape, ratio in ((3.0, 800.0), (0.5, 2000.0), (50.0, 5000.0), (1e-5, 1e4)):
            reference = mpmath.log(
                mpmath.gammainc(_mp(shape), _mp(ratio), mpmath.inf, regularized=True)
            )
            observed = _censored_total(FamilyId.GAMMA, ratio, shape=shape, scale=1.0)
            with self.subTest(shape=shape, ratio=ratio):
                self.assertLess(observed, -700.0)
                self.assertLessEqual(
                    float(abs(_mp(observed) - reference)),
                    _gamma_budget(shape, ratio, reference),
                )

    def test_ratio_of_astronomical_size_remains_finite(self) -> None:
        shape, ratio = 2.0, 1e300
        observed = _censored_total(FamilyId.GAMMA, ratio, shape=shape, scale=1.0)
        reference = -_mp(ratio) + (_mp(shape) - 1) * mpmath.log(_mp(ratio))
        self.assertLessEqual(float(abs(_mp(observed) - reference) / abs(reference)), 1e-14)

    def test_small_shape_lgamma_matches_the_taylor_reference(self) -> None:
        from veridist.statistics.distributions import _lgamma_one_plus_small

        for shape in (5e-324, 1e-300, 1e-20, 1e-8, 1e-5, 1e-3, 0.005, 0.00999):
            # `1 + shape` is not representable at the default precision for the
            # smallest shapes, so the reference is evaluated far above it.
            with mpmath.workdps(360):
                reference = mpmath.loggamma(1 + _mp(shape))
            observed = _lgamma_one_plus_small(shape)
            with self.subTest(shape=shape):
                self.assertLessEqual(
                    float(abs(_mp(observed) - reference)), 4.0 * ulp(float(reference))
                )


def _sample(
    seed: int, count: int, censored_fraction: float, draw: Callable[[random.Random], float]
) -> tuple[LifetimeObservation, ...]:
    rng = random.Random(seed)
    observations: list[LifetimeObservation] = []
    for _ in range(count):
        time = draw(rng)
        if rng.random() < censored_fraction:
            observations.append(RightCensoredLifetime(time))
        else:
            observations.append(ExactLifetime(time))
    return tuple(observations)


class FitAgreementTests(unittest.TestCase):
    """The reducer at the fitted parameters reproduces each fit's reported likelihood.

    The two sides sum the same terms in different arrangements (the Weibull fit
    works in geometric-mean-scaled units and the lognormal fit in a hoisted
    quadratic), so they agree to the conditioning of the sum, not to the last
    bit: the allowance is ``1e-12 * (|sum| + sum of absolute terms)``, and the
    observed ratio is asserted to stay far below it.
    """

    def _terms_scale(
        self, family: FamilyId, observations: Sequence[LifetimeObservation], **parameters: float
    ) -> float:
        scale = 0.0
        for observation in observations:
            term = reduce_lifetime_log_likelihood_chunks(family, [[observation]], **parameters)
            assert isinstance(term, LogLikelihoodSuccess)
            scale += abs(term.total_log_likelihood)
        return scale

    def _check(
        self,
        family: FamilyId,
        observations: Sequence[LifetimeObservation],
        fit_value: float,
        **parameters: float,
    ) -> None:
        result = reduce_lifetime_log_likelihood_chunks(family, [observations], **parameters)
        assert isinstance(result, LogLikelihoodSuccess)
        allowance = 1e-12 * (abs(fit_value) + self._terms_scale(family, observations, **parameters))
        self.assertLessEqual(abs(result.total_log_likelihood - fit_value), allowance)
        # And in the plain relative sense required of well-conditioned data.
        self.assertLessEqual(
            abs(result.total_log_likelihood - fit_value), 1e-12 * abs(fit_value)
        )

    def test_weibull_censored_fits(self) -> None:
        for seed, count, fraction in ((1, 40, 0.3), (2, 120, 0.5), (3, 25, 0.1), (4, 300, 0.7)):
            observations = _sample(
                seed, count, fraction, lambda rng: 10.0 * rng.weibullvariate(1.0, 1.7)
            )
            fit = fit_weibull(observations)
            assert isinstance(fit, WeibullFitSuccess)
            with self.subTest(seed=seed):
                self._check(
                    FamilyId.WEIBULL_MIN,
                    observations,
                    fit.log_likelihood,
                    shape=fit.shape,
                    scale=fit.scale,
                )

    def test_weibull_fixed_shape_fit(self) -> None:
        observations = _sample(9, 60, 0.4, lambda rng: 5.0 * rng.weibullvariate(1.0, 0.8))
        fit = fit_weibull(observations, fixed_shape=1.3)
        assert isinstance(fit, WeibullFitSuccess)
        self._check(
            FamilyId.WEIBULL_MIN, observations, fit.log_likelihood, shape=fit.shape, scale=fit.scale
        )

    def test_lognormal_censored_fits(self) -> None:
        for seed, count, fraction in ((11, 40, 0.3), (12, 120, 0.5), (13, 25, 0.1), (14, 300, 0.6)):
            observations = _sample(
                seed, count, fraction, lambda rng: rng.lognormvariate(1.0, 0.6)
            )
            fit = fit_lognormal(observations)
            assert isinstance(fit, LognormalFitSuccess)
            with self.subTest(seed=seed):
                self._check(
                    FamilyId.LOGNORMAL,
                    observations,
                    fit.log_likelihood,
                    mu_log=fit.mu_log,
                    sigma_log=fit.sigma_log,
                )

    def test_lognormal_uncensored_closed_form_fit(self) -> None:
        observations = _sample(21, 80, 0.0, lambda rng: rng.lognormvariate(0.2, 0.9))
        fit = fit_lognormal(observations)
        assert isinstance(fit, LognormalFitSuccess)
        self._check(
            FamilyId.LOGNORMAL,
            observations,
            fit.log_likelihood,
            mu_log=fit.mu_log,
            sigma_log=fit.sigma_log,
        )


class GammaSampleReferenceTests(unittest.TestCase):
    def test_mixed_sample_matches_a_high_precision_sum(self) -> None:
        shape, scale = 2.7, 3.1
        rng = random.Random(77)
        observations: list[LifetimeObservation] = []
        reference = mpmath.mpf(0)
        for index in range(60):
            time = rng.gammavariate(shape, scale)
            if index % 3 == 0:
                observations.append(RightCensoredLifetime(time))
                ratio = _mp(time) / _mp(scale)
                reference += mpmath.log(
                    mpmath.gammainc(_mp(shape), ratio, mpmath.inf, regularized=True)
                )
            else:
                observations.append(ExactLifetime(time))
                reference += (
                    (_mp(shape) - 1) * mpmath.log(_mp(time))
                    - _mp(time) / _mp(scale)
                    - mpmath.loggamma(_mp(shape))
                    - _mp(shape) * mpmath.log(_mp(scale))
                )
        result = reduce_lifetime_log_likelihood_chunks(
            FamilyId.GAMMA, [observations], shape=shape, scale=scale
        )
        assert isinstance(result, LogLikelihoodSuccess)
        self.assertLessEqual(
            float(abs(_mp(result.total_log_likelihood) - reference) / abs(reference)), 1e-12
        )

