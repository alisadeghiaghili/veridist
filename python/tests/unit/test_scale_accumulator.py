"""Exactness, bounds and edge cases of the exact accumulator."""

from __future__ import annotations

import math
import unittest
from copy import deepcopy
from fractions import Fraction

import numpy as np

from tests.unit.scale_oracle import (
    MAX_FLOAT,
    MAX_MANTISSA,
    adversarial_sets,
    as_array,
    bits,
    correctly_rounded,
    exact_fraction,
    exact_units,
)
from veridist.scale import _accumulator as acc
from veridist.scale._accumulator import (
    DEFAULT_LIMITS,
    AccumulatorLimits,
    ExactSum,
    units_to_float,
    units_to_fraction,
)
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.statistics.log_likelihood import MAX_OBSERVATION_COUNT, MAX_TOTAL_UNITS

SMALL = AccumulatorLimits(block_size=4, carry_interval=10, lane_capacity=50)


def summed(values: np.ndarray, limits: AccumulatorLimits = DEFAULT_LIMITS) -> ExactSum:
    accumulator = ExactSum(limits)
    accumulator.update(values)
    return accumulator


class ConstantTests(unittest.TestCase):
    def test_documented_layout_constants(self) -> None:
        self.assertEqual(acc.BIN_COUNT, 2048)
        self.assertEqual(acc.LOW_BITS, 27)
        self.assertEqual(acc.UNITS_EXPONENT, 1074)
        self.assertEqual(acc.DEFAULT_BLOCK_SIZE, 2**15)
        self.assertEqual(acc.MAX_BLOCK_SIZE, 2**26)
        self.assertEqual(acc.DEFAULT_CARRY_INTERVAL, 2**34)
        self.assertEqual(acc.MAX_CARRY_INTERVAL, 2**36 - 1)
        self.assertEqual(acc.DEFAULT_LANE_CAPACITY, 2**36)
        self.assertEqual(acc.MAX_LANE_CAPACITY, 2**36)
        self.assertEqual(acc.SCHEMA_VERSION, 1)
        self.assertEqual(acc.STATE_TAG, "scale.exact_sum")
        self.assertEqual(DEFAULT_LIMITS, AccumulatorLimits(2**15, 2**34, 2**36))

    def test_element_bound_matches_the_log_likelihood_state(self) -> None:
        self.assertEqual(acc.MAX_ELEMENT_UNITS, exact_units([MAX_FLOAT]))
        self.assertEqual(MAX_OBSERVATION_COUNT * acc.MAX_ELEMENT_UNITS, MAX_TOTAL_UNITS)

    def test_documented_overflow_bounds_hold_with_room(self) -> None:
        # A low lane holds at most unnormalised * (2**27 - 1) and a high lane at most
        # lane_load * (2**26 + 1); both must stay inside int64 at the largest limits.
        int64_limit = 2**63
        self.assertLess(acc.MAX_CARRY_INTERVAL * (2**27 - 1), int64_limit)
        self.assertGreater(2 * acc.MAX_CARRY_INTERVAL * (2**27 - 1), int64_limit)
        self.assertLess(acc.MAX_LANE_CAPACITY * (2**26 + 1), int64_limit)
        self.assertGreater(2 * acc.MAX_LANE_CAPACITY * (2**26 + 1), int64_limit)
        # One block sums at most 2**26 values: both weight sums stay below 2**53.
        self.assertLess(acc.MAX_BLOCK_SIZE * (2**27 - 1), 2**53)
        self.assertLessEqual(acc.MAX_BLOCK_SIZE * 2**26, 2**52)

    def test_default_limits_satisfy_the_documented_bounds(self) -> None:
        self.assertLess(acc.DEFAULT_BLOCK_SIZE, acc.DEFAULT_CARRY_INTERVAL)
        self.assertLessEqual(acc.DEFAULT_CARRY_INTERVAL, acc.MAX_CARRY_INTERVAL)
        self.assertLessEqual(acc.DEFAULT_LANE_CAPACITY, acc.MAX_LANE_CAPACITY)


class LimitsValidationTests(unittest.TestCase):
    def test_accepted_boundaries(self) -> None:
        AccumulatorLimits(1, 2, 1)
        AccumulatorLimits(2**26, 2**36 - 1, 2**36)
        AccumulatorLimits(5, 6, 5)

    def test_every_violation_is_rejected(self) -> None:
        bad: list[tuple[int, int, int]] = [
            (0, 10, 50),
            (-1, 10, 50),
            (2**26 + 1, 2**36 - 1, 2**36),
            (4, 4, 50),
            (4, 3, 50),
            (4, 2**36, 50),
            (4, 10, 3),
            (4, 10, 2**36 + 1),
            (4, 10, 0),
        ]
        for block, interval, capacity in bad:
            with self.subTest(block=block, interval=interval, capacity=capacity):
                with self.assertRaises(ValueError):
                    AccumulatorLimits(block, interval, capacity)

    def test_non_integers_are_type_errors(self) -> None:
        for bad in (4.0, True, "4", None):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(TypeError):
                    AccumulatorLimits(bad, 10, 50)  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    AccumulatorLimits(4, bad, 50)  # type: ignore[arg-type]
                with self.assertRaises(TypeError):
                    AccumulatorLimits(4, 10, bad)  # type: ignore[arg-type]

    def test_accumulator_requires_limits_object(self) -> None:
        with self.assertRaises(TypeError):
            ExactSum((4, 10, 50))  # type: ignore[arg-type]
        self.assertIs(ExactSum().limits, DEFAULT_LIMITS)
        self.assertIs(ExactSum(SMALL).limits, SMALL)


class RoundingTests(unittest.TestCase):
    def test_units_are_exact_integers_for_known_values(self) -> None:
        self.assertEqual(summed(as_array([5e-324])).total_units(), 1)
        self.assertEqual(summed(as_array([-5e-324])).total_units(), -1)
        self.assertEqual(summed(as_array([1.0])).total_units(), 1 << 1074)
        self.assertEqual(summed(as_array([0.5, 0.25])).total_units(), 3 << 1072)
        self.assertEqual(summed(as_array([MAX_FLOAT])).total_units(), acc.MAX_ELEMENT_UNITS)
        self.assertEqual(summed(as_array([2.2250738585072014e-308])).total_units(), 1 << 52)
        self.assertEqual(summed(as_array([2.225073858507201e-308])).total_units(), (1 << 52) - 1)

    def test_units_to_float_rounds_ties_to_even_once(self) -> None:
        one = 1 << 1074
        ulp = 1 << 1022  # 2**-52 in units
        half = ulp >> 1
        self.assertEqual(units_to_float(one + half), 1.0)  # exactly half an ulp: to even
        self.assertEqual(units_to_float(one + half + 1), 1.0 + 2.0**-52)
        self.assertEqual(units_to_float(one + half - 1), 1.0)
        odd = one + ulp  # 1 + 2**-52 has an odd last bit, so its half-way point rounds up
        self.assertEqual(units_to_float(odd + half), 1.0 + 2.0 * 2.0**-52)
        self.assertEqual(units_to_float(odd + half - 1), 1.0 + 2.0**-52)
        self.assertEqual(units_to_float(-(one + half)), -1.0)
        self.assertEqual(units_to_float(-(one + half + 1)), -(1.0 + 2.0**-52))
        self.assertEqual(units_to_float(0), 0.0)
        self.assertEqual(math.copysign(1.0, units_to_float(0)), 1.0)

    def test_units_to_float_in_the_subnormal_range(self) -> None:
        for units in (1, 2, 3, -1, -7, (1 << 52) - 1, 1 << 52, (1 << 52) + 1):
            with self.subTest(units=units):
                self.assertEqual(units_to_float(units), correctly_rounded(units))
                self.assertEqual(units_to_float(units), units * 5e-324)

    def test_units_to_float_overflow_boundary(self) -> None:
        maximum = acc.MAX_ELEMENT_UNITS
        half_ulp = 1 << (1074 + 970)
        self.assertEqual(units_to_float(maximum), MAX_FLOAT)
        self.assertEqual(units_to_float(maximum + half_ulp - 1), MAX_FLOAT)
        for units in (maximum + half_ulp, maximum + 2 * half_ulp, -(maximum + half_ulp)):
            with self.subTest(units=units):
                with self.assertRaises(ScaleStateError) as caught:
                    units_to_float(units)
                self.assertEqual(caught.exception.code, ScaleStateErrorCode.TOTAL_NOT_REPRESENTABLE)
                self.assertIsInstance(caught.exception.__cause__, OverflowError)

    def test_units_to_fraction_is_exact(self) -> None:
        self.assertEqual(units_to_fraction(3), Fraction(3, 1 << 1074))
        self.assertEqual(units_to_fraction(-(1 << 1074)), Fraction(-1))
        self.assertEqual(units_to_fraction(0), Fraction(0))


class ExactnessTests(unittest.TestCase):
    def test_every_adversarial_set_is_exact_with_default_and_tiny_limits(self) -> None:
        for name, full in adversarial_sets().items():
            for limits in (DEFAULT_LIMITS, SMALL, AccumulatorLimits(1, 2, 1)):
                values = full[: {2**15: len(full), 4: 800, 1: 150}[limits.block_size]]
                with self.subTest(name=name, block=limits.block_size):
                    expected_units = exact_units(values.tolist())
                    accumulator = summed(values, limits)
                    self.assertEqual(accumulator.total_units(), expected_units)
                    self.assertEqual(accumulator.count, len(values))
                    self.assertEqual(accumulator.exact_total(), Fraction(expected_units, 1 << 1074))

    def test_totals_are_correctly_rounded_and_equal_fsum(self) -> None:
        for name, values in adversarial_sets().items():
            with self.subTest(name=name):
                expected = correctly_rounded(exact_units(values.tolist()))
                total = summed(values).total()
                self.assertEqual(bits(total), bits(expected))
                if name != "every_power_of_two":  # its partial sums overflow math.fsum
                    self.assertEqual(bits(total), bits(math.fsum(values.tolist()) + 0.0))

    def test_exact_against_fractions_on_a_mixed_sample(self) -> None:
        values = adversarial_sets()["mixed_magnitude_sign_1"][:200]
        self.assertEqual(summed(values).exact_total(), exact_fraction(values.tolist()))

    def test_one_rounding_cases(self) -> None:
        sets = adversarial_sets()
        self.assertEqual(summed(sets["tie_to_even_down"]).total(), 1.0)
        self.assertEqual(summed(sets["tie_to_even_up"]).total(), 1.0 + 2.0**-51)
        self.assertEqual(summed(sets["just_above_tie"]).total(), 1.0 + 2.0**-52)
        self.assertEqual(summed(sets["cancellation_to_zero"]).total_units(), 0)

    def test_plain_float_sums_are_not_exact_here(self) -> None:
        # A guard that the data really are adversarial: naive summation is wrong.
        values = adversarial_sets()["cancellation"]
        self.assertNotEqual(float(np.sum(values)), summed(values).total())
        self.assertNotEqual(sum(values.tolist()), summed(values).total())

    def test_every_exponent_bin_is_reachable(self) -> None:
        values = adversarial_sets()["every_power_of_two"]
        positives = values[values > 0]
        accumulator = summed(positives)
        self.assertEqual(accumulator.total_units(), exact_units(positives.tolist()))
        self.assertEqual(int(np.count_nonzero(accumulator._high | accumulator._low)), 2046)

    def test_subnormal_and_smallest_normal_share_a_bin_consistently(self) -> None:
        values = as_array([2.2250738585072014e-308, 2.225073858507201e-308, 5e-324])
        self.assertEqual(summed(values).total_units(), (1 << 52) + (1 << 52) - 1 + 1)

    def test_values_near_the_maximum(self) -> None:
        accumulator = summed(as_array([MAX_FLOAT, -MAX_FLOAT, MAX_FLOAT]))
        self.assertEqual(accumulator.total(), MAX_FLOAT)
        self.assertEqual(accumulator.total_units(), acc.MAX_ELEMENT_UNITS)
        accumulator = summed(as_array([MAX_FLOAT, 2.0**969]))
        self.assertEqual(accumulator.total(), MAX_FLOAT)

    def test_max_mantissa_repeated_exercises_the_widest_lane_sums(self) -> None:
        values = np.full(5000, MAX_MANTISSA)
        accumulator = summed(values)
        self.assertEqual(accumulator.total_units(), 5000 * ((1 << 1075) - (1 << 1022)))
        self.assertEqual(accumulator.total(), 5000 * MAX_MANTISSA)

    def test_signed_zeros_count_but_add_nothing(self) -> None:
        accumulator = summed(as_array([0.0, -0.0, -0.0]))
        self.assertEqual(accumulator.count, 3)
        self.assertEqual(accumulator.total_units(), 0)
        self.assertEqual(bits(accumulator.total()), bits(0.0))

    def test_negative_values_are_exact_in_both_lane_parts(self) -> None:
        values = as_array([-1.0, -MAX_MANTISSA, -(2.0**-1074), -3.5e-310, -1e300])
        self.assertEqual(summed(values).total_units(), exact_units(values.tolist()))


class EmptyAndEdgeTests(unittest.TestCase):
    def test_empty_accumulator(self) -> None:
        accumulator = ExactSum()
        self.assertEqual((accumulator.count, accumulator.total_units()), (0, 0))
        self.assertEqual(accumulator.total(), 0.0)
        self.assertEqual(accumulator.exact_total(), 0)

    def test_empty_batch_changes_nothing(self) -> None:
        accumulator = summed(as_array([1.0, 2.0]))
        before = accumulator.to_bytes()
        accumulator.update(np.empty(0, dtype=np.float64))
        self.assertEqual(accumulator.to_bytes(), before)
        empty = ExactSum()
        empty.update(np.empty(0, dtype=np.float64))
        self.assertEqual(empty.to_bytes(), ExactSum().to_bytes())

    def test_single_element(self) -> None:
        for value in (1.0, -1.0, 5e-324, MAX_FLOAT, -MAX_FLOAT, 0.1, 1e-310):
            with self.subTest(value=value):
                accumulator = summed(as_array([value]))
                self.assertEqual(accumulator.count, 1)
                self.assertEqual(accumulator.total(), value)

    def test_non_finite_values_are_refused_and_leave_the_state_unchanged(self) -> None:
        for bad in (math.nan, math.inf, -math.inf):
            for limits in (DEFAULT_LIMITS, SMALL):
                with self.subTest(bad=bad, block=limits.block_size):
                    accumulator = summed(as_array([1.0, 2.0, 3.0]), limits)
                    before = (accumulator.count, accumulator.total_units(), accumulator.to_bytes())
                    for batch in ([bad], [1.0, 2.0, 3.0, 4.0, 5.0, bad], [bad, 1.0, 2.0, 3.0, 4.0]):
                        with self.assertRaises(ScaleStateError) as caught:
                            accumulator.update(as_array(batch))
                        self.assertEqual(
                            caught.exception.code, ScaleStateErrorCode.NON_FINITE_VALUE
                        )
                    after = (accumulator.count, accumulator.total_units(), accumulator.to_bytes())
                    self.assertEqual(before, after)

    def test_error_text_carries_no_value(self) -> None:
        with self.assertRaises(ScaleStateError) as caught:
            ExactSum().update(as_array([123456.789, math.nan]))
        self.assertEqual(str(caught.exception), "NON_FINITE_VALUE")
        self.assertEqual(caught.exception.args, ("NON_FINITE_VALUE",))

    def test_wrong_container_dtype_and_shape_are_refused(self) -> None:
        wrong_type = [
            [1.0, 2.0],
            (1.0, 2.0),
            1.0,
            "1.0",
            None,
            np.float64(1.0),
            np.ma.masked_array([1.0, 2.0], mask=[False, True]),
        ]
        for value in wrong_type:
            with self.subTest(value=repr(value)):
                with self.assertRaises(TypeError):
                    ExactSum().update(value)
        wrong_dtype = [
            np.array([1.0], dtype=np.float32),
            np.array([1], dtype=np.int64),
            np.array([True]),
            np.array([1.0], dtype=np.float16),
            np.array([1.0], dtype=">f8"),
            np.array([1.0 + 0j]),
            np.array([1.0], dtype=object),
        ]
        for value in wrong_dtype:
            with self.subTest(dtype=str(value.dtype)):
                with self.assertRaises(TypeError):
                    ExactSum().update(value)
        for shape_value in (np.array(1.0), np.ones((2, 2)), np.ones((1, 3)), np.ones((0, 2))):
            with self.subTest(shape=shape_value.shape):
                with self.assertRaises(ValueError):
                    ExactSum().update(shape_value)

    def test_strided_and_read_only_views_are_accepted(self) -> None:
        base = np.random.default_rng(3).standard_normal(100)
        strided = base[::3]
        self.assertFalse(strided.flags.c_contiguous)
        self.assertEqual(summed(strided).total_units(), exact_units(strided.tolist()))
        frozen = base.copy()
        frozen.flags.writeable = False
        self.assertEqual(summed(frozen).total_units(), exact_units(base.tolist()))
        offset = base[1:]
        self.assertEqual(summed(offset).total_units(), exact_units(offset.tolist()))

    def test_the_input_array_is_not_modified(self) -> None:
        values = np.random.default_rng(4).standard_normal(50)
        copy = values.copy()
        summed(values)
        np.testing.assert_array_equal(values, copy)

    def test_equality_is_by_count_and_exact_total(self) -> None:
        left = summed(as_array([1.0, 3.0]))
        self.assertEqual(left, summed(as_array([2.0, 2.0])))
        self.assertNotEqual(left, summed(as_array([2.0, 2.0, 0.0])))
        self.assertNotEqual(left, summed(as_array([2.0, 3.0])))
        self.assertNotEqual(left, "not an accumulator")
        self.assertIs(ExactSum.__hash__, None)  # type: ignore[comparison-overlap]


class CarryAndFoldTests(unittest.TestCase):
    def assert_invariants(self, accumulator: ExactSum) -> None:
        limits = accumulator.limits
        low = int(accumulator._low.max())
        high = int(np.abs(accumulator._high).max())
        self.assertGreaterEqual(int(accumulator._low.min()), 0)
        self.assertLessEqual(accumulator._unnormalised, limits.carry_interval)
        self.assertLessEqual(accumulator._lane_load, limits.lane_capacity)
        self.assertLessEqual(low, accumulator._unnormalised * (2**27 - 1))
        self.assertLessEqual(high, accumulator._lane_load * (2**26 + 1))

    def test_carry_and_fold_keep_the_bounds_and_the_value(self) -> None:
        rng = np.random.default_rng(11)
        for sign in (1.0, -1.0):
            with self.subTest(sign=sign):
                accumulator = ExactSum(SMALL)
                expected = 0
                for _ in range(120):
                    batch = np.full(4, sign * MAX_MANTISSA) * np.exp2(rng.integers(-3, 3, 4))
                    accumulator.update(batch)
                    expected += exact_units(batch.tolist())
                    self.assert_invariants(accumulator)
                    self.assertEqual(accumulator.total_units(), expected)
                self.assertNotEqual(accumulator._spill, 0)

    def test_carry_happens_exactly_when_the_interval_would_be_exceeded(self) -> None:
        accumulator = ExactSum(
            AccumulatorLimits(block_size=4, carry_interval=10, lane_capacity=1000)
        )
        batch = np.full(4, MAX_MANTISSA)
        accumulator.update(batch)
        accumulator.update(batch)
        self.assertEqual(accumulator._unnormalised, 8)
        self.assertEqual(int(accumulator._low.max()), 8 * (2**27 - 1))
        accumulator.update(np.full(2, MAX_MANTISSA))
        self.assertEqual(accumulator._unnormalised, 10)  # exactly at the interval: no carry
        self.assertEqual(int(accumulator._low.max()), 10 * (2**27 - 1))
        accumulator.update(np.full(1, MAX_MANTISSA))
        self.assertEqual(accumulator._unnormalised, 2)  # carried (1) then added one
        self.assertLess(int(accumulator._low.max()), 2 * 2**27)
        self.assertEqual(accumulator._lane_load, 11)
        self.assertEqual(accumulator.total_units(), exact_units([MAX_MANTISSA] * 11))

    def test_fold_happens_exactly_when_the_capacity_would_be_exceeded(self) -> None:
        accumulator = ExactSum(AccumulatorLimits(block_size=4, carry_interval=40, lane_capacity=8))
        batch = np.full(4, MAX_MANTISSA)
        accumulator.update(batch)
        accumulator.update(batch)
        self.assertEqual((accumulator._lane_load, accumulator._spill), (8, 0))  # at capacity
        accumulator.update(np.full(1, 1.0))
        self.assertEqual((accumulator._lane_load, accumulator._unnormalised), (1, 1))
        self.assertEqual(accumulator._spill, exact_units([MAX_MANTISSA] * 8))
        self.assertEqual(accumulator.total_units(), exact_units([MAX_MANTISSA] * 8 + [1.0]))

    def test_forced_carry_changes_only_the_representation(self) -> None:
        values = np.full(30, MAX_MANTISSA)
        carried = ExactSum(AccumulatorLimits(5, 6, 1000))
        carried.update(values)
        plain = summed(values)
        self.assertEqual(carried.total_units(), plain.total_units())
        self.assertEqual(carried.to_bytes(), plain.to_bytes())
        before = carried.total_units()
        carried._carry()
        self.assertEqual(carried.total_units(), before)
        self.assertEqual(carried._unnormalised, 1)
        self.assertLess(int(carried._low.max()), 2**27)

    def test_block_bookkeeping_at_the_fold_boundary(self) -> None:
        accumulator = ExactSum(AccumulatorLimits(block_size=4, carry_interval=100, lane_capacity=8))
        block = np.full(4, MAX_MANTISSA)
        accumulator._add_block(block)
        accumulator._add_block(block)
        # Exactly at the capacity: nothing is folded yet.
        self.assertEqual((accumulator._lane_load, accumulator._spill), (8, 0))
        self.assertEqual(accumulator.count, 8)
        accumulator._add_block(np.full(1, MAX_MANTISSA))
        self.assertEqual(
            (accumulator._lane_load, accumulator._spill), (1, exact_units([MAX_MANTISSA] * 8))
        )
        self.assertEqual(accumulator.total_units(), exact_units([MAX_MANTISSA] * 9))
        accumulator._add_block(block)
        self.assertEqual(accumulator._lane_load, 5)
        accumulator._add_block(np.full(4, 1.0))
        self.assertEqual(
            (accumulator._lane_load, accumulator._spill), (4, exact_units([MAX_MANTISSA] * 13))
        )

    def test_block_bookkeeping_at_the_carry_boundary(self) -> None:
        accumulator = ExactSum(
            AccumulatorLimits(block_size=4, carry_interval=10, lane_capacity=1000)
        )
        block = np.full(4, MAX_MANTISSA)
        accumulator._add_block(block)
        accumulator._add_block(block)
        accumulator._add_block(np.full(2, MAX_MANTISSA))
        # Exactly at the interval: no carry yet, so the low lane holds the whole mass.
        self.assertEqual(accumulator._unnormalised, 10)
        self.assertEqual(int(accumulator._low.max()), 10 * (2**27 - 1))
        accumulator._add_block(np.full(1, MAX_MANTISSA))
        self.assertEqual(accumulator._unnormalised, 2)
        self.assertLess(int(accumulator._low.max()), 2 * 2**27)
        self.assertEqual(accumulator._lane_load, 11)
        self.assertEqual(accumulator.total_units(), exact_units([MAX_MANTISSA] * 11))
        # A block that fits exactly after a carry does not carry again.
        accumulator._add_block(np.full(4, MAX_MANTISSA))
        accumulator._add_block(np.full(4, MAX_MANTISSA))
        self.assertEqual(accumulator._unnormalised, 10)
        accumulator._add_block(np.full(2, MAX_MANTISSA))
        self.assertEqual(accumulator._unnormalised, 3)

    def test_total_is_unbounded_beyond_the_lane_capacity(self) -> None:
        accumulator = ExactSum(AccumulatorLimits(1, 2, 1))
        for _ in range(40):
            accumulator.update(np.array([MAX_FLOAT]))
        self.assertEqual(accumulator.total_units(), 40 * acc.MAX_ELEMENT_UNITS)
        self.assertGreater(accumulator.total_units(), 2**1074 * 2**1024)
        with self.assertRaises(ScaleStateError):
            accumulator.total()
        accumulator.update(np.array([-MAX_FLOAT] * 39))
        self.assertEqual(accumulator.total(), MAX_FLOAT)

    def test_a_large_single_batch_spans_blocks_carries_and_folds(self) -> None:
        values = np.random.default_rng(5).standard_normal(400) * 1e100
        accumulator = summed(values, AccumulatorLimits(3, 7, 20))
        self.assertEqual(accumulator.total_units(), exact_units(values.tolist()))
        self.assertEqual(accumulator.count, 400)


class MergeTests(unittest.TestCase):
    def test_merge_is_exact_and_leaves_operands_unchanged(self) -> None:
        values = adversarial_sets()["mixed_magnitude_sign_2"]
        left = summed(values[:700])
        right = summed(values[700:])
        left_before, right_before = left.to_bytes(), right.to_bytes()
        merged = left.merge(right)
        self.assertEqual(merged.total_units(), exact_units(values.tolist()))
        self.assertEqual(merged.count, len(values))
        self.assertEqual((left.to_bytes(), right.to_bytes()), (left_before, right_before))
        self.assertIsNot(merged, left)

    def test_identity_commutativity_associativity(self) -> None:
        rng = np.random.default_rng(6)
        parts = [rng.standard_normal(40) * np.exp2(rng.integers(-60, 60, 40)) for _ in range(3)]
        a, b, c = (summed(part) for part in parts)
        empty = ExactSum()
        self.assertEqual(a.merge(empty), a)
        self.assertEqual(empty.merge(a), a)
        self.assertEqual(a.merge(b), b.merge(a))
        self.assertEqual(a.merge(b).merge(c), a.merge(b.merge(c)))
        self.assertEqual(a.merge(b).merge(c).to_bytes(), c.merge(a).merge(b).to_bytes())

    def test_merge_triggers_fold_when_loads_exceed_the_capacity(self) -> None:
        limits = AccumulatorLimits(4, 100, 12)
        left = summed(np.full(8, MAX_MANTISSA), limits)
        right = summed(np.full(8, -MAX_MANTISSA * 0.75), limits)
        merged = left.merge(right)
        self.assertEqual(merged._lane_load, 8)  # folded, then took the other's lanes
        self.assertEqual(merged._spill, left.total_units())
        self.assertEqual(
            merged.total_units(), exact_units([MAX_MANTISSA] * 8 + [-MAX_MANTISSA * 0.75] * 8)
        )
        exactly = summed(np.full(8, 1.0), AccumulatorLimits(4, 100, 16))
        both = exactly.merge(exactly)
        self.assertEqual((both._lane_load, both._spill), (16, 0))  # at capacity: no fold

    def test_merge_carries_when_the_intervals_would_overflow(self) -> None:
        limits = AccumulatorLimits(4, 10, 1000)
        left = summed(np.full(8, MAX_MANTISSA), limits)
        right = summed(np.full(8, MAX_MANTISSA), limits)
        self.assertEqual((left._unnormalised, right._unnormalised), (8, 8))
        merged = left.merge(right)
        self.assertEqual(merged._unnormalised, 2)
        self.assertEqual(merged._lane_load, 16)
        self.assertLess(int(merged._low.max()), 2 * 2**27)
        self.assertEqual(merged.total_units(), exact_units([MAX_MANTISSA] * 16))
        self.assertEqual((left._unnormalised, int(left._low.max())), (8, 8 * (2**27 - 1)))
        fits = summed(np.full(2, MAX_MANTISSA), limits).merge(
            summed(np.full(8, MAX_MANTISSA), limits)
        )
        self.assertEqual(fits._unnormalised, 10)  # exactly at the interval: plain lane addition
        self.assertEqual(int(fits._low.max()), 10 * (2**27 - 1))

    def test_merge_with_an_empty_lane_set_adds_only_count_and_spill(self) -> None:
        folded = ExactSum(AccumulatorLimits(1, 2, 1))
        for _ in range(3):
            folded.update(np.array([2.0]))
        restored = ExactSum.from_bytes(folded.to_bytes(), folded.limits)
        merged = ExactSum(folded.limits).merge(restored)
        self.assertEqual(merged.total_units(), 3 * (2 << 1074))
        self.assertEqual((merged._lane_load, merged._unnormalised), (0, 0))
        self.assertEqual(merged.count, 3)

    def test_merge_refusals(self) -> None:
        with self.assertRaises(TypeError):
            ExactSum().merge(5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ExactSum().merge(ExactSum(SMALL))

    def test_merge_observation_cap(self) -> None:
        near = ExactSum.from_bytes(_frame(MAX_OBSERVATION_COUNT - 1, 0))
        one = summed(as_array([1.0]))
        merged = near.merge(one)
        self.assertEqual(merged.count, MAX_OBSERVATION_COUNT)
        with self.assertRaises(ScaleStateError) as caught:
            merged.merge(one)
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        with self.assertRaises(ScaleStateError):
            near.merge(summed(as_array([1.0, 2.0])))
        self.assertEqual(near.count, MAX_OBSERVATION_COUNT - 1)

    def test_update_observation_cap_is_atomic(self) -> None:
        accumulator = ExactSum.from_bytes(_frame(MAX_OBSERVATION_COUNT - 2, 5))
        with self.assertRaises(ScaleStateError) as caught:
            accumulator.update(as_array([1.0, 2.0, 3.0]))
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        self.assertEqual(
            (accumulator.count, accumulator.total_units()), (MAX_OBSERVATION_COUNT - 2, 5)
        )
        accumulator.update(as_array([1.0, 2.0]))
        self.assertEqual(accumulator.count, MAX_OBSERVATION_COUNT)
        accumulator.update(np.empty(0))
        with self.assertRaises(ScaleStateError):
            accumulator.update(as_array([0.0]))

    def test_copy_is_independent_and_equal(self) -> None:
        original = summed(as_array([1.0, 2.0, 3.0]), SMALL)
        clone = original.copy()
        self.assertEqual(clone, original)
        self.assertEqual(clone.limits, SMALL)
        clone.update(as_array([4.0]))
        self.assertEqual(original.total(), 6.0)
        self.assertEqual(clone.total(), 10.0)
        self.assertEqual(original.count, 3)
        self.assertIsNot(clone._high, original._high)
        self.assertIsNot(clone._low, original._low)
        self.assertEqual(deepcopy(original), original)


class CapAtTheTotalLevelTests(unittest.TestCase):
    def test_a_total_at_the_declared_maximum_state_is_representable_as_an_integer(self) -> None:
        state = ExactSum.from_bytes(_frame(MAX_OBSERVATION_COUNT, MAX_TOTAL_UNITS))
        self.assertEqual(state.total_units(), MAX_TOTAL_UNITS)
        with self.assertRaises(ScaleStateError) as caught:
            state.total()
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.TOTAL_NOT_REPRESENTABLE)
        self.assertEqual(state.total_units(), MAX_TOTAL_UNITS)  # still valid


class RefusalMessageTests(unittest.TestCase):
    """The programmer-error messages are part of the contract and carry no values."""

    def message(self, error_type: type[Exception], call: object) -> str:
        with self.assertRaises(error_type) as caught:
            call()  # type: ignore[operator]
        return str(caught.exception)

    def test_limit_messages(self) -> None:
        self.assertEqual(
            self.message(TypeError, lambda: AccumulatorLimits(4.0, 10, 50)),  # type: ignore[arg-type]
            "accumulator limits must be built-in integers",
        )
        self.assertEqual(
            self.message(ValueError, lambda: AccumulatorLimits(0, 10, 50)),
            "block_size must be between 1 and 2**26",
        )
        self.assertEqual(
            self.message(ValueError, lambda: AccumulatorLimits(4, 4, 50)),
            "carry_interval must exceed block_size and be below 2**36",
        )
        self.assertEqual(
            self.message(ValueError, lambda: AccumulatorLimits(4, 10, 3)),
            "lane_capacity must be between block_size and 2**36",
        )

    def test_accumulator_messages(self) -> None:
        self.assertEqual(
            self.message(TypeError, lambda: ExactSum(5)),  # type: ignore[arg-type]
            "limits must be AccumulatorLimits",
        )
        self.assertEqual(
            self.message(TypeError, lambda: ExactSum().merge(5)),  # type: ignore[arg-type]
            "other must be an ExactSum",
        )
        self.assertEqual(
            self.message(ValueError, lambda: ExactSum().merge(ExactSum(SMALL))),
            "accumulators with different limits cannot be merged",
        )

    def test_batch_messages(self) -> None:
        self.assertEqual(
            self.message(TypeError, lambda: ExactSum().update([1.0])),
            "values must be a numpy.ndarray",
        )
        self.assertEqual(
            self.message(TypeError, lambda: ExactSum().update(np.ones(2, dtype=np.float32))),
            "values must have dtype float64",
        )
        self.assertEqual(
            self.message(ValueError, lambda: ExactSum().update(np.ones((2, 2)))),
            "values must be one-dimensional",
        )

    def test_serialized_input_message(self) -> None:
        self.assertEqual(
            self.message(TypeError, lambda: ExactSum.from_bytes("text")),
            "state data must be bytes",
        )


def _frame(count: int, units: int) -> bytes:
    from veridist.scale import _codec

    return _codec.encode_frame(acc.STATE_TAG, acc.SCHEMA_VERSION, count, (units,))


if __name__ == "__main__":
    unittest.main()
