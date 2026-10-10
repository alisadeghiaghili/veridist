"""Independent references for the exact accumulator and the exact-sum states.

The scalar log-likelihood reducer keeps the same integer (the exact sum in units
of 2**-1074) one value at a time; the accumulator must reproduce it bit for bit.
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

import mpmath
import numpy as np

from tests.unit.scale_oracle import (
    MAX_FLOAT,
    MAX_MANTISSA,
    adversarial_sets,
    as_array,
    batch,
    exact_units,
)
from veridist.families.registry import FamilyId
from veridist.scale._accumulator import ExactSum
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._one_pass import GammaState, LognormalState, NormalState
from veridist.statistics.log_likelihood import (
    MAX_OBSERVATION_COUNT,
    LogLikelihoodState,
    _FinalTotalNotRepresentable,
    _ObservationLimitExceeded,
)


def scalar_state(values: list[float]) -> LogLikelihoodState:
    state = LogLikelihoodState.empty(FamilyId.NORMAL, mu=0.0, sigma=1.0)
    for value in values:
        state = state.add_log_density(value)
    return state


class LogLikelihoodStateInteroperabilityTests(unittest.TestCase):
    def test_total_units_and_count_equal_the_scalar_reducer_state(self) -> None:
        for name, values in adversarial_sets().items():
            sample = values[:120].tolist()
            with self.subTest(name=name):
                scalar = scalar_state(sample)
                accumulator = ExactSum()
                accumulator.update(as_array(sample))
                self.assertEqual(accumulator.total_units(), scalar.total_units)
                self.assertEqual(accumulator.count, scalar.observation_count)
                self.assertEqual(accumulator.total(), scalar.finalize())

    def test_merge_of_halves_equals_the_scalar_merge(self) -> None:
        values = adversarial_sets()["mixed_magnitude_sign_1"][:100].tolist()
        scalar = scalar_state(values[:40]).merge(scalar_state(values[40:]))
        left, right = ExactSum(), ExactSum()
        left.update(as_array(values[:40]))
        right.update(as_array(values[40:]))
        merged = left.merge(right)
        self.assertEqual(merged.total_units(), scalar.total_units)
        self.assertEqual(merged.count, scalar.observation_count)

    def test_restored_scalar_state_equals_restored_accumulator(self) -> None:
        restored = LogLikelihoodState.restore(
            FamilyId.NORMAL, 7, 12345, mu=0.0, sigma=1.0
        )
        scalar = restored.add_log_density(0.5)
        accumulator = ExactSum.from_bytes(_exact_frame(7, 12345))
        accumulator.update(as_array([0.5]))
        self.assertEqual(accumulator.total_units(), scalar.total_units)
        self.assertEqual(accumulator.count, scalar.observation_count)

    def test_final_overflow_agrees(self) -> None:
        scalar = scalar_state([MAX_FLOAT, MAX_FLOAT])
        accumulator = ExactSum()
        accumulator.update(as_array([MAX_FLOAT, MAX_FLOAT]))
        self.assertEqual(accumulator.total_units(), scalar.total_units)
        with self.assertRaises(_FinalTotalNotRepresentable):
            scalar.finalize()
        with self.assertRaises(ScaleStateError) as caught:
            accumulator.total()
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.TOTAL_NOT_REPRESENTABLE)

    def test_observation_cap_agrees(self) -> None:
        capped = LogLikelihoodState.restore(
            FamilyId.NORMAL, MAX_OBSERVATION_COUNT, 0, mu=0.0, sigma=1.0
        )
        with self.assertRaises(_ObservationLimitExceeded):
            capped.add_log_density(0.0)
        accumulator = ExactSum.from_bytes(_exact_frame(MAX_OBSERVATION_COUNT, 0))
        with self.assertRaises(ScaleStateError) as caught:
            accumulator.update(as_array([0.0]))
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)

    def test_signed_zero_total_is_positive_in_both(self) -> None:
        scalar = scalar_state([-0.0, 0.0, -0.0])
        accumulator = ExactSum()
        accumulator.update(as_array([-0.0, 0.0, -0.0]))
        self.assertEqual(math.copysign(1.0, scalar.finalize()), 1.0)
        self.assertEqual(math.copysign(1.0, accumulator.total()), 1.0)


class FractionReferenceTests(unittest.TestCase):
    def test_exact_total_equals_the_fraction_sum_on_adversarial_data(self) -> None:
        for name, values in adversarial_sets().items():
            sample = values[:200].tolist()
            with self.subTest(name=name):
                accumulator = ExactSum()
                accumulator.update(as_array(sample))
                expected = sum((Fraction(value) for value in sample), Fraction(0))
                self.assertEqual(accumulator.exact_total(), expected)
                self.assertEqual(accumulator.total(), float(expected))

    def test_max_mantissa_extremes_against_fractions(self) -> None:
        values = [MAX_MANTISSA] * 500 + [-MAX_MANTISSA * 0.5] * 123 + [2.0**-1074] * 77
        accumulator = ExactSum()
        accumulator.update(as_array(values))
        self.assertEqual(accumulator.total_units(), exact_units(values))
        self.assertEqual(
            accumulator.exact_total(), sum((Fraction(value) for value in values), Fraction(0))
        )


class MpmathReferenceTests(unittest.TestCase):
    def test_log_sums_agree_with_high_precision_logarithms(self) -> None:
        rng = np.random.default_rng(12)
        values = batch(GammaState, rng, 60)
        results = (
            GammaState.empty().update(values).finalize().sum_log_x,
            LognormalState.empty().update(values).finalize().sum_log_x,
        )
        with mpmath.workdps(60):
            logs = [mpmath.log(mpmath.mpf(value)) for value in values.tolist()]
            reference = mpmath.fsum(logs)
            magnitude = mpmath.fsum(abs(term) for term in logs)
            for result in results:
                # numpy.log is within one unit in the last place per element; the sum is exact.
                exact = mpmath.mpf(result.exact.numerator) / result.exact.denominator
                self.assertLessEqual(abs(exact - reference), magnitude * mpmath.mpf(2) ** -52)

    def test_normal_moments_against_mpmath(self) -> None:
        values = np.array([1.0e7 + offset for offset in range(10)])  # squares are exact
        result = NormalState.empty().update(values).finalize()
        assert result.centered_sum_of_squares is not None
        value = result.centered_sum_of_squares.exact
        self.assertEqual(float(value), 82.5)
        with mpmath.workdps(80):
            exact = [mpmath.mpf(item) for item in values.tolist()]
            mean = mpmath.fsum(exact) / 10
            centered = mpmath.fsum((item - mean) ** 2 for item in exact)
            self.assertLess(abs(mpmath.mpf(value.numerator) / value.denominator - centered), 1e-30)


def _exact_frame(count: int, units: int) -> bytes:
    from veridist.scale import _accumulator, _codec

    return _codec.encode_frame(_accumulator.STATE_TAG, _accumulator.SCHEMA_VERSION, count, (units,))


if __name__ == "__main__":
    unittest.main()
