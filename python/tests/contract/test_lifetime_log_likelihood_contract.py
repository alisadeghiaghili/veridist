"""Contracts for the exact/right-censored lifetime log-likelihood reducer."""

from __future__ import annotations

import random
import unittest
from decimal import Decimal
from math import isfinite
from unittest.mock import patch

import veridist
import veridist.statistics as statistics
from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime
from veridist.engine.data_source import DataSourceMetadata, Replayability
from veridist.engine.streaming import IterableDataSource, StreamSourceError
from veridist.families.registry import FamilyId
from veridist.statistics.lifetime_log_likelihood import (
    SUPPORTED_LIFETIME_FAMILIES,
    reduce_lifetime_log_likelihood_chunks,
)
from veridist.statistics.log_density import LogDensityErrorCode
from veridist.statistics.log_likelihood import (
    LogLikelihoodErrorCode,
    LogLikelihoodFailure,
    LogLikelihoodSuccess,
    _ExactAccumulator,
    _ObservationLimitExceeded,
    reduce_log_likelihood_chunks,
)

PARAMETERS: dict[FamilyId, dict[str, float]] = {
    FamilyId.WEIBULL_MIN: {"shape": 1.6, "scale": 4.0},
    FamilyId.LOGNORMAL: {"mu_log": 1.0, "sigma_log": 0.7},
    FamilyId.GAMMA: {"shape": 2.5, "scale": 3.0},
}


def _failure(result: object) -> LogLikelihoodFailure:
    assert isinstance(result, LogLikelihoodFailure), result
    return result


def _success(result: object) -> LogLikelihoodSuccess:
    assert isinstance(result, LogLikelihoodSuccess), result
    return result


class LifetimeReducerSurfaceTests(unittest.TestCase):
    def test_it_is_a_statistics_export_but_not_a_top_level_one(self) -> None:
        self.assertIn("reduce_lifetime_log_likelihood_chunks", statistics.__all__)
        self.assertIs(
            statistics.reduce_lifetime_log_likelihood_chunks,
            reduce_lifetime_log_likelihood_chunks,
        )
        self.assertNotIn("reduce_lifetime_log_likelihood_chunks", veridist.__all__)
        self.assertFalse(hasattr(veridist, "reduce_lifetime_log_likelihood_chunks"))

    def test_the_admitted_families_are_exactly_the_three_lifetime_families(self) -> None:
        self.assertEqual(
            SUPPORTED_LIFETIME_FAMILIES,
            frozenset({FamilyId.WEIBULL_MIN, FamilyId.LOGNORMAL, FamilyId.GAMMA}),
        )


class LifetimeReducerAccumulationTests(unittest.TestCase):
    def _observations(self, seed: int) -> list[LifetimeObservation]:
        rng = random.Random(seed)
        values: list[LifetimeObservation] = []
        for _ in range(97):
            time = rng.uniform(0.2, 30.0)
            censored = rng.random() < 0.4
            values.append(RightCensoredLifetime(time) if censored else ExactLifetime(time))
        return values

    def test_total_is_independent_of_chunking_and_order(self) -> None:
        for family, parameters in PARAMETERS.items():
            observations = self._observations(5)
            baseline = _success(
                reduce_lifetime_log_likelihood_chunks(family, [observations], **parameters)
            )
            rng = random.Random(8)
            for _ in range(5):
                shuffled = observations[:]
                rng.shuffle(shuffled)
                cuts = sorted(rng.sample(range(1, len(shuffled)), 6))
                chunks = [shuffled[a:b] for a, b in zip([0, *cuts], [*cuts, len(shuffled)])]
                result = _success(
                    reduce_lifetime_log_likelihood_chunks(family, chunks, **parameters)
                )
                with self.subTest(family=family):
                    self.assertEqual(result, baseline)
            self.assertEqual(baseline.observation_count, 97)

    def test_exact_only_data_equals_the_density_reducer_bit_for_bit(self) -> None:
        for family, parameters in PARAMETERS.items():
            times = [ExactLifetime(t) for t in (0.4, 1.1, 2.5, 7.0, 19.5)]
            lifetime = _success(
                reduce_lifetime_log_likelihood_chunks(family, [times], **parameters)
            )
            density = _success(
                reduce_log_likelihood_chunks(
                    family, [[observation.time for observation in times]], **parameters
                )
            )
            with self.subTest(family=family):
                self.assertEqual(lifetime.total_log_likelihood, density.total_log_likelihood)
                self.assertEqual(lifetime.observation_count, density.observation_count)

    def test_fingerprint_is_stable_parameter_bound_and_distinct_from_the_density_reducer(
        self,
    ) -> None:
        family = FamilyId.WEIBULL_MIN
        parameters = PARAMETERS[family]
        one = _success(
            reduce_lifetime_log_likelihood_chunks(family, [[ExactLifetime(1.0)]], **parameters)
        )
        two = _success(
            reduce_lifetime_log_likelihood_chunks(
                family, [[RightCensoredLifetime(9.0)]], **parameters
            )
        )
        other = _success(
            reduce_lifetime_log_likelihood_chunks(
                family, [[ExactLifetime(1.0)]], shape=1.6, scale=4.5
            )
        )
        density = _success(reduce_log_likelihood_chunks(family, [[1.0]], **parameters))
        self.assertEqual(one.parameter_fingerprint, two.parameter_fingerprint)
        self.assertNotEqual(one.parameter_fingerprint, other.parameter_fingerprint)
        self.assertNotEqual(one.parameter_fingerprint, density.parameter_fingerprint)

    def test_empty_input_is_a_zero_count_success(self) -> None:
        for family, parameters in PARAMETERS.items():
            for chunks in ([], [[]], [[], []]):
                result = _success(
                    reduce_lifetime_log_likelihood_chunks(family, chunks, **parameters)
                )
                self.assertEqual((result.observation_count, result.total_log_likelihood), (0, 0.0))

    def test_stream_sources_enforce_a_single_pass(self) -> None:
        metadata = DataSourceMetadata(
            source_id="lifetimes",
            schema_version="1",
            provenance_schema_version="1",
            replayability=Replayability.SINGLE_PASS,
            redaction_reason="test",
        )
        source = IterableDataSource(([ExactLifetime(1.0), RightCensoredLifetime(2.0)],), metadata)
        parameters = PARAMETERS[FamilyId.GAMMA]
        first = _success(
            reduce_lifetime_log_likelihood_chunks(FamilyId.GAMMA, source, **parameters)
        )
        self.assertEqual(first.observation_count, 2)
        with self.assertRaises(StreamSourceError):
            reduce_lifetime_log_likelihood_chunks(FamilyId.GAMMA, source, **parameters)

    def test_observation_count_cap_and_unrepresentable_total_are_typed(self) -> None:
        parameters = PARAMETERS[FamilyId.GAMMA]
        with patch.object(_ExactAccumulator, "add", side_effect=_ObservationLimitExceeded):
            capped = _failure(
                reduce_lifetime_log_likelihood_chunks(
                    FamilyId.GAMMA, [[ExactLifetime(1.0)]], **parameters
                )
            )
        self.assertIs(capped.code, LogLikelihoodErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        self.assertEqual(capped.processed_count, 0)
        self.assertIsNone(capped.scalar_error_code)

        # Two terms of about -1.7e308 are each finite; their sum is not.
        huge = 1.7e308
        total = _failure(
            reduce_lifetime_log_likelihood_chunks(
                FamilyId.WEIBULL_MIN,
                [[RightCensoredLifetime(huge), RightCensoredLifetime(huge)]],
                shape=1.0,
                scale=1.0,
            )
        )
        self.assertIs(total.code, LogLikelihoodErrorCode.FINAL_TOTAL_NOT_REPRESENTABLE)
        self.assertEqual(total.processed_count, 2)


class LifetimeReducerFailureTests(unittest.TestCase):
    def _scalar_failure(
        self, family: FamilyId, observations: list[object], **parameters: float
    ) -> LogLikelihoodFailure:
        result = _failure(
            reduce_lifetime_log_likelihood_chunks(family, [observations], **parameters)
        )
        self.assertIs(result.code, LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE)
        return result

    def test_invalid_observation_types_are_typed_failures(self) -> None:
        for bad in (1.0, 3, "1.0", None, Decimal("1"), (1.0,), object(), True):
            for family, parameters in PARAMETERS.items():
                with self.subTest(family=family, bad=repr(bad)):
                    failure = self._scalar_failure(
                        family, [ExactLifetime(1.0), RightCensoredLifetime(2.0), bad], **parameters
                    )
                    self.assertIs(
                        failure.scalar_error_code, LogDensityErrorCode.NONFINITE_OBSERVATION
                    )
                    self.assertEqual(failure.processed_count, 2)

    def test_lifetime_subclass_lookalikes_are_not_trusted(self) -> None:
        class Impostor(ExactLifetime):
            pass

        failure = self._scalar_failure(
            FamilyId.GAMMA, [Impostor(1.0)], **PARAMETERS[FamilyId.GAMMA]
        )
        self.assertIs(failure.scalar_error_code, LogDensityErrorCode.NONFINITE_OBSERVATION)

    def test_zero_time_is_a_support_violation_for_exact_and_censored_terms(self) -> None:
        for family, parameters in PARAMETERS.items():
            for observation in (ExactLifetime(0.0), RightCensoredLifetime(0.0)):
                with self.subTest(family=family, kind=type(observation).__name__):
                    failure = self._scalar_failure(family, [observation], **parameters)
                    self.assertIs(
                        failure.scalar_error_code, LogDensityErrorCode.SUPPORT_VIOLATION
                    )
                    self.assertEqual(failure.processed_count, 0)

    def test_terms_below_binary64_are_numerical_overflow(self) -> None:
        cases: list[tuple[FamilyId, float, dict[str, float]]] = [
            (FamilyId.WEIBULL_MIN, 1e300, {"shape": 5.0, "scale": 1.0}),
            (FamilyId.GAMMA, 1e300, {"shape": 2.0, "scale": 1e-10}),
            (FamilyId.LOGNORMAL, 1e300, {"mu_log": 0.0, "sigma_log": 1e-300}),
        ]
        for family, time, parameters in cases:
            with self.subTest(family=family):
                failure = self._scalar_failure(
                    family, [RightCensoredLifetime(time)], **parameters
                )
                self.assertIs(failure.scalar_error_code, LogDensityErrorCode.NUMERICAL_OVERFLOW)

    def test_non_converging_or_unrepresentable_gamma_terms_are_typed(self) -> None:
        # A shape this large exhausts the expansion budget near the mode.
        failure = self._scalar_failure(
            FamilyId.GAMMA,
            [RightCensoredLifetime(1e6)],
            shape=1e6,
            scale=1.0,
        )
        self.assertIs(failure.scalar_error_code, LogDensityErrorCode.NONFINITE_LOG_DENSITY)
        # A subnormal shape leaves a survival probability below binary64.
        tiny = self._scalar_failure(
            FamilyId.GAMMA, [RightCensoredLifetime(0.5)], shape=5e-324, scale=1.0
        )
        self.assertIs(tiny.scalar_error_code, LogDensityErrorCode.NONFINITE_LOG_DENSITY)

    def test_no_censored_term_is_ever_infinite_or_nan(self) -> None:
        rng = random.Random(3)
        for family in SUPPORTED_LIFETIME_FAMILIES:
            for _ in range(200):
                parameters = _random_parameters(family, rng)
                time = 10.0 ** rng.uniform(-320.0, 308.0)
                result = reduce_lifetime_log_likelihood_chunks(
                    family, [[RightCensoredLifetime(time)]], **parameters
                )
                if isinstance(result, LogLikelihoodSuccess):
                    self.assertTrue(isfinite(result.total_log_likelihood))
                    self.assertLessEqual(result.total_log_likelihood, 0.0)
                else:
                    self.assertIs(result.code, LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE)

    def test_failure_stops_the_reduction_at_the_first_bad_term(self) -> None:
        failure = self._scalar_failure(
            FamilyId.WEIBULL_MIN,
            [ExactLifetime(1.0), RightCensoredLifetime(0.0), ExactLifetime(2.0)],
            **PARAMETERS[FamilyId.WEIBULL_MIN],
        )
        self.assertEqual(failure.processed_count, 1)


def _random_parameters(family: FamilyId, rng: random.Random) -> dict[str, float]:
    if family is FamilyId.WEIBULL_MIN:
        return {"shape": 10.0 ** rng.uniform(-2, 2), "scale": 10.0 ** rng.uniform(-5, 5)}
    if family is FamilyId.LOGNORMAL:
        return {"mu_log": rng.uniform(-20, 20), "sigma_log": 10.0 ** rng.uniform(-3, 1)}
    return {"shape": 10.0 ** rng.uniform(-4, 4), "scale": 10.0 ** rng.uniform(-5, 5)}


class LifetimeReducerMisuseTests(unittest.TestCase):
    def test_unsupported_families_are_rejected_before_iteration(self) -> None:
        consumed: list[int] = []

        def chunks() -> object:
            consumed.append(1)
            yield [ExactLifetime(1.0)]

        for family, parameters in (
            (FamilyId.NORMAL, {"mu": 0.0, "sigma": 1.0}),
            (FamilyId.GUMBEL_RIGHT, {"location": 0.0, "scale": 1.0}),
        ):
            with self.subTest(family=family), self.assertRaises(ValueError):
                reduce_lifetime_log_likelihood_chunks(family, chunks(), **parameters)
        self.assertEqual(consumed, [])

    def test_family_identity_and_parameters_are_validated_first(self) -> None:
        with self.assertRaises(TypeError):
            reduce_lifetime_log_likelihood_chunks("gamma", [], shape=1.0, scale=1.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            reduce_lifetime_log_likelihood_chunks(FamilyId.GAMMA, [], shape=1.0)
        with self.assertRaises(ValueError):
            reduce_lifetime_log_likelihood_chunks(FamilyId.GAMMA, [], shape=-1.0, scale=1.0)
        with self.assertRaises(ValueError):
            reduce_lifetime_log_likelihood_chunks(
                FamilyId.LOGNORMAL, [], mu_log=0.0, sigma_log=float("inf")
            )

    def test_a_chunk_must_be_iterable(self) -> None:
        with self.assertRaises(TypeError):
            reduce_lifetime_log_likelihood_chunks(
                FamilyId.GAMMA, [5], shape=1.0, scale=1.0  # type: ignore[list-item]
            )


if __name__ == "__main__":
    unittest.main()
