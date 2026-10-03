"""The censored reducers for the real-line families and the exponential, against mpmath."""

from __future__ import annotations

import unittest
from math import log

import mpmath

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue, RightCensoredValue
from veridist.families.registry import FamilyId
from veridist.statistics.lifetime_log_likelihood import (
    SUPPORTED_LIFETIME_FAMILIES,
    SUPPORTED_VALUE_FAMILIES,
    reduce_lifetime_log_likelihood_chunks,
    reduce_value_log_likelihood_chunks,
)
from veridist.statistics.log_density import LogDensityErrorCode
from veridist.statistics.log_likelihood import (
    LogLikelihoodErrorCode,
    LogLikelihoodFailure,
    LogLikelihoodSuccess,
)

mpmath.mp.dps = 60
mp = mpmath.mp


def _success(result: object) -> LogLikelihoodSuccess:
    assert isinstance(result, LogLikelihoodSuccess), result
    return result


def _failure(result: object) -> LogLikelihoodFailure:
    assert isinstance(result, LogLikelihoodFailure), result
    return result


def _normal_log_sf(x: float, mu: float, sigma: float) -> mpmath.mpf:
    z = (mp.mpf(x) - mu) / sigma
    return mp.log(mp.erfc(z / mp.sqrt(2)) / 2)


def _gumbel_log_sf(x: float, location: float, scale: float) -> mpmath.mpf:
    z = (mp.mpf(x) - location) / scale
    return mp.log(-mp.expm1(-mp.exp(-z)))


class CensoredTermReferenceTests(unittest.TestCase):
    def test_normal_log_survival_matches_mpmath_across_both_tails(self) -> None:
        for mu, sigma in ((0.0, 1.0), (3.0, 0.5), (-10.0, 25.0)):
            for z in (-40.0, -5.0, -1.0, 0.0, 0.5, 3.0, 8.0, 20.0, 36.0, 40.0, 100.0, 1e4):
                x = mu + z * sigma
                total = _success(
                    reduce_value_log_likelihood_chunks(
                        FamilyId.NORMAL, [[RightCensoredValue(x)]], mu=mu, sigma=sigma
                    )
                ).total_log_likelihood
                want = _normal_log_sf(x, mu, sigma)
                with self.subTest(mu=mu, sigma=sigma, z=z):
                    self.assertLessEqual(abs(mp.mpf(total) - want), 1e-11 * abs(want) + 1e-15)

    def test_gumbel_log_survival_matches_mpmath_in_every_branch(self) -> None:
        # z < -700 (exactly one), tail > 0.7, the middle range, and tail < 1e-300.
        for location, scale in ((0.0, 1.0), (5.0, 0.25), (-2.0, 40.0)):
            for z in (
                -900.0,
                -700.0,
                -30.0,
                -3.0,
                -0.5,
                0.0,
                0.3,
                1.0,
                5.0,
                40.0,
                600.0,
                745.0,
                1000.0,
            ):
                x = location + z * scale
                total = _success(
                    reduce_value_log_likelihood_chunks(
                        FamilyId.GUMBEL_RIGHT,
                        [[RightCensoredValue(x)]],
                        location=location,
                        scale=scale,
                    )
                ).total_log_likelihood
                want = _gumbel_log_sf(x, location, scale)
                with self.subTest(location=location, scale=scale, z=z):
                    self.assertLessEqual(abs(mp.mpf(total) - want), 1e-12 * abs(want) + 1e-300)

    def test_exponential_log_survival_is_minus_rate_times_time(self) -> None:
        for rate, time in ((2.0, 0.5), (1e-3, 4e5), (7.5, 1e-9)):
            total = _success(
                reduce_lifetime_log_likelihood_chunks(
                    FamilyId.EXPONENTIAL, [[RightCensoredLifetime(time)]], rate=rate
                )
            ).total_log_likelihood
            want = -mp.mpf(rate) * mp.mpf(time)
            with self.subTest(rate=rate, time=time):
                self.assertLessEqual(abs(mp.mpf(total) - want), 1e-15 * abs(want))

    def test_exact_terms_use_the_log_density(self) -> None:
        total = _success(
            reduce_value_log_likelihood_chunks(
                FamilyId.NORMAL, [[ExactValue(1.5)]], mu=1.0, sigma=2.0
            )
        ).total_log_likelihood
        want = -(mp.mpf("0.25") ** 2) / 2 - mp.log(2) - mp.log(2 * mp.pi) / 2
        self.assertLessEqual(abs(mp.mpf(total) - want), 1e-15)
        exponential = _success(
            reduce_lifetime_log_likelihood_chunks(
                FamilyId.EXPONENTIAL, [[ExactLifetime(3.0)]], rate=0.5
            )
        ).total_log_likelihood
        self.assertLessEqual(abs(mp.mpf(exponential) - (mp.log(0.5) - mp.mpf(1.5))), 1e-15)


class ReducerContractTests(unittest.TestCase):
    def test_family_sets_are_disjoint_and_cover_all_families(self) -> None:
        self.assertEqual(SUPPORTED_VALUE_FAMILIES, {FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT})
        self.assertFalse(SUPPORTED_VALUE_FAMILIES & SUPPORTED_LIFETIME_FAMILIES)
        self.assertEqual(SUPPORTED_VALUE_FAMILIES | SUPPORTED_LIFETIME_FAMILIES, set(FamilyId))

    def test_each_reducer_rejects_the_other_families_before_iterating(self) -> None:
        consumed: list[int] = []

        def chunks() -> object:
            consumed.append(1)
            yield []

        for family, parameters in (
            (FamilyId.NORMAL, {"mu": 0.0, "sigma": 1.0}),
            (FamilyId.GUMBEL_RIGHT, {"location": 0.0, "scale": 1.0}),
        ):
            with self.assertRaises(ValueError):
                reduce_lifetime_log_likelihood_chunks(family, chunks(), **parameters)
        for family, parameters in (
            (FamilyId.EXPONENTIAL, {"rate": 1.0}),
            (FamilyId.WEIBULL_MIN, {"shape": 1.0, "scale": 1.0}),
            (FamilyId.GAMMA, {"shape": 1.0, "scale": 1.0}),
            (FamilyId.LOGNORMAL, {"mu_log": 0.0, "sigma_log": 1.0}),
        ):
            with self.assertRaises(ValueError):
                reduce_value_log_likelihood_chunks(family, chunks(), **parameters)
        self.assertEqual(consumed, [])

    def test_the_value_reducer_types_its_observation_failures(self) -> None:
        parameters = {"mu": 0.0, "sigma": 1.0}
        for bad in (ExactLifetime(1.0), RightCensoredLifetime(1.0), 1.0, None):
            failure = _failure(
                reduce_value_log_likelihood_chunks(
                    FamilyId.NORMAL, [[ExactValue(0.0), bad]], **parameters
                )
            )
            self.assertIs(failure.code, LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE)
            self.assertIs(failure.scalar_error_code, LogDensityErrorCode.NONFINITE_OBSERVATION)
            self.assertEqual(failure.processed_count, 1)
        # And the lifetime reducer refuses the real-valued types in the same way.
        failure = _failure(
            reduce_lifetime_log_likelihood_chunks(
                FamilyId.EXPONENTIAL, [[ExactValue(1.0)]], rate=1.0
            )
        )
        self.assertIs(failure.scalar_error_code, LogDensityErrorCode.NONFINITE_OBSERVATION)

    def test_negative_values_are_valid_for_the_real_line_only(self) -> None:
        self.assertEqual(
            _success(
                reduce_value_log_likelihood_chunks(
                    FamilyId.GUMBEL_RIGHT,
                    [[ExactValue(-5.0), RightCensoredValue(-7.0)]],
                    location=0.0,
                    scale=1.0,
                )
            ).observation_count,
            2,
        )
        for family, parameters in (
            (FamilyId.EXPONENTIAL, {"rate": 1.0}),
            (FamilyId.GAMMA, {"shape": 1.0, "scale": 1.0}),
        ):
            for observation in (ExactLifetime(1.0), RightCensoredLifetime(1.0)):
                self.assertIsInstance(
                    reduce_lifetime_log_likelihood_chunks(family, [[observation]], **parameters),
                    LogLikelihoodSuccess,
                )

    def test_zero_times_are_valid_for_the_exponential_and_not_for_the_open_families(self) -> None:
        for kind in (ExactLifetime, RightCensoredLifetime):
            total = _success(
                reduce_lifetime_log_likelihood_chunks(FamilyId.EXPONENTIAL, [[kind(0.0)]], rate=2.5)
            ).total_log_likelihood
            self.assertEqual(total, log(2.5) if kind is ExactLifetime else 0.0)
            for family, parameters in (
                (FamilyId.GAMMA, {"shape": 1.0, "scale": 1.0}),
                (FamilyId.WEIBULL_MIN, {"shape": 1.0, "scale": 1.0}),
                (FamilyId.LOGNORMAL, {"mu_log": 0.0, "sigma_log": 1.0}),
            ):
                with self.subTest(family=family, kind=kind.__name__):
                    failure = _failure(
                        reduce_lifetime_log_likelihood_chunks(family, [[kind(0.0)]], **parameters)
                    )
                    self.assertIs(failure.scalar_error_code, LogDensityErrorCode.SUPPORT_VIOLATION)

    def test_exponential_overflow_is_typed(self) -> None:
        failure = _failure(
            reduce_lifetime_log_likelihood_chunks(
                FamilyId.EXPONENTIAL, [[RightCensoredLifetime(1e300)]], rate=1e300
            )
        )
        self.assertIs(failure.scalar_error_code, LogDensityErrorCode.NUMERICAL_OVERFLOW)

    def test_fingerprints_separate_the_two_reducers_and_the_density_reducer(self) -> None:
        parameters = {"mu": 0.0, "sigma": 1.0}
        value = _success(
            reduce_value_log_likelihood_chunks(FamilyId.NORMAL, [[ExactValue(1.0)]], **parameters)
        )
        again = _success(
            reduce_value_log_likelihood_chunks(
                FamilyId.NORMAL, [[RightCensoredValue(4.0)]], **parameters
            )
        )
        other = _success(
            reduce_value_log_likelihood_chunks(
                FamilyId.NORMAL, [[ExactValue(1.0)]], mu=0.0, sigma=2.0
            )
        )
        self.assertEqual(value.parameter_fingerprint, again.parameter_fingerprint)
        self.assertNotEqual(value.parameter_fingerprint, other.parameter_fingerprint)
        exponential = _success(
            reduce_lifetime_log_likelihood_chunks(
                FamilyId.EXPONENTIAL, [[ExactLifetime(1.0)]], rate=1.0
            )
        )
        self.assertEqual(len(exponential.parameter_fingerprint), 64)

    def test_chunks_must_be_iterable_and_a_single_pass_is_enforced_by_a_source(self) -> None:
        with self.assertRaises(TypeError):
            reduce_value_log_likelihood_chunks(
                FamilyId.NORMAL,
                [5],
                mu=0.0,
                sigma=1.0,  # type: ignore[list-item]
            )
        empty = _success(reduce_value_log_likelihood_chunks(FamilyId.NORMAL, [], mu=0.0, sigma=1.0))
        self.assertEqual((empty.observation_count, empty.total_log_likelihood), (0, 0.0))


if __name__ == "__main__":
    unittest.main()
