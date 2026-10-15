"""The exact sum of squares: against rational arithmetic over every binary64 magnitude."""

from __future__ import annotations

import unittest
from fractions import Fraction
from unittest import mock

import numpy as np

from tests.unit.scale_oracle import MAX_FLOAT, exact_square_units
from veridist.scale import _squares
from veridist.scale._accumulator import MAX_ELEMENT_UNITS
from veridist.scale._squares import MAX_SQUARE_UNITS, SQUARE_UNITS_EXPONENT, squares_units

TINY = 5e-324
LOW_EDGE = 2.0**-480
HIGH_EDGE = 2.0**500


def arr(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def below(value: float) -> float:
    return float(np.nextafter(value, 0.0))


def above(value: float) -> float:
    return float(np.nextafter(value, np.inf))


class ConstantTests(unittest.TestCase):
    def test_the_unit_is_the_square_of_the_accumulator_unit(self) -> None:
        self.assertEqual(SQUARE_UNITS_EXPONENT, 2148)
        self.assertEqual(squares_units(arr([TINY])), 1)
        self.assertEqual(squares_units(arr([1.0])), 1 << 2148)

    def test_the_element_bound_is_the_square_of_the_largest_float(self) -> None:
        self.assertEqual(squares_units(arr([MAX_FLOAT])), MAX_SQUARE_UNITS)
        self.assertEqual(squares_units(arr([-MAX_FLOAT])), MAX_SQUARE_UNITS)
        self.assertEqual(MAX_ELEMENT_UNITS, ((1 << 53) - 1) << 2045)
        self.assertEqual(MAX_SQUARE_UNITS, (((1 << 53) - 1) ** 2) << 4090)


class ExactnessTests(unittest.TestCase):
    def assertExact(self, values: np.ndarray) -> None:
        self.assertEqual(squares_units(values), exact_square_units(values.tolist()))

    def test_empty_and_trivial_batches(self) -> None:
        self.assertEqual(squares_units(np.empty(0)), 0)
        self.assertEqual(squares_units(arr([0.0])), 0)
        self.assertEqual(squares_units(arr([0.0, -0.0, 0.0])), 0)
        self.assertEqual(squares_units(arr([3.0])), 9 << 2148)
        self.assertEqual(squares_units(arr([-3.0, 4.0])), 25 << 2148)
        self.assertEqual(squares_units(arr([0.5])), 1 << 2146)

    def test_inexact_products_keep_their_rounding_error(self) -> None:
        value = 1.0 + 2.0**-52
        exact = (((1 << 52) + 1) ** 2) << (2148 - 104)
        self.assertEqual(squares_units(arr([value])), exact)
        rounded = int(Fraction(value * value) * (1 << 2148))
        self.assertNotEqual(exact, rounded)
        self.assertExact(arr([3.0 + 2.0**-51, 0.1, 1e9 + 0.3]))

    def chunked_exactness(self, values: list[float], size: int = 12) -> list[int]:
        """Start indices of the chunks whose exact sum of squares is wrong."""

        return [
            start
            for start in range(0, len(values), size)
            if squares_units(arr(values[start : start + size]))
            != exact_square_units(values[start : start + size])
        ]

    def test_every_power_of_two(self) -> None:
        values = [2.0**exponent for exponent in range(-1074, 1024)]
        self.assertEqual(self.chunked_exactness(values), [])
        self.assertEqual(squares_units(arr(values)), exact_square_units(values))

    def test_the_largest_mantissa_at_every_exponent(self) -> None:
        values = [below(2.0**exponent) for exponent in range(-1022, 1024)]
        self.assertEqual(self.chunked_exactness(values), [])
        self.assertEqual(self.chunked_exactness([-value for value in values], 5), [])

    def test_random_mantissas_across_the_whole_range(self) -> None:
        rng = np.random.default_rng(11)
        values: list[float] = []
        for exponent in range(-1074, 1024, 3):
            mantissas = rng.integers(1, 2**53, 4).astype(np.float64)
            sign = -1.0 if exponent % 2 else 1.0
            for value in np.ldexp(mantissas, exponent - 52):
                if np.isfinite(value) and value != 0.0:
                    values.append(sign * float(value))
        self.assertEqual(self.chunked_exactness(values, 7), [])
        self.assertEqual(squares_units(arr(values)), exact_square_units(values))

    def test_values_on_both_sides_of_every_class_edge(self) -> None:
        for edge in (LOW_EDGE, HIGH_EDGE):
            values = [edge, above(edge), below(edge), -edge, 1.5 * edge, 0.75 * edge,
                      above(above(edge)), below(below(edge))]
            with self.subTest(edge=edge):
                self.assertExact(arr(values))
                for value in values:
                    self.assertExact(arr([value]))

    def test_subnormals_and_the_smallest_normals(self) -> None:
        values = [TINY, 2 * TINY, 3 * TINY, 2.0**-1022, below(2.0**-1022), 1e-310, 1e-320]
        self.assertExact(arr(values))
        for value in values:
            self.assertEqual(squares_units(arr([value])), exact_square_units([value]))

    def test_largest_floats(self) -> None:
        values = [
            MAX_FLOAT,
            below(MAX_FLOAT),
            2.0**1023,
            1e308,
            -1.5e308,
            2.0**512,
            1.3407807929942596e154,
        ]
        self.assertExact(arr(values))

    def test_a_batch_mixing_every_class(self) -> None:
        values = arr(
            [0.0, 1.0, TINY, 1e-200, 1e-100, 3.7, 1e100, 1e200, 1e300, -MAX_FLOAT, -2e-300]
        )
        self.assertExact(values)

    def test_random_batches_with_many_magnitudes(self) -> None:
        for seed in range(8):
            rng = np.random.default_rng(seed)
            values = rng.standard_normal(500) * np.exp2(rng.uniform(-1000, 1000, 500))
            values = values[np.isfinite(values)]
            with self.subTest(seed=seed):
                self.assertExact(values)

    def test_normal_data_far_from_zero_relative_to_their_spread(self) -> None:
        values = 1e9 + np.arange(8.0)
        total = sum((Fraction(float(v)) for v in values), Fraction(0))
        centered = Fraction(squares_units(values), 1 << 2148) - total * total / 8
        self.assertEqual(centered, 42)


class StructureTests(unittest.TestCase):
    def test_the_sum_is_additive_over_any_split_and_order(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.standard_normal(400) * np.exp2(rng.uniform(-1070, 1000, 400))
        values = values[np.isfinite(values)]
        whole = squares_units(values)
        for cut in (1, 7, 200, len(values) - 1):
            self.assertEqual(squares_units(values[:cut]) + squares_units(values[cut:]), whole)
        self.assertEqual(squares_units(values[::-1].copy()), whole)
        self.assertEqual(squares_units(values[rng.permutation(len(values))]), whole)

    def test_the_result_does_not_depend_on_a_values_position(self) -> None:
        rng = np.random.default_rng(6)
        values = rng.standard_normal(64) * np.exp2(rng.uniform(-1000, 1000, 64))
        values = values[np.isfinite(values)]
        expected = [squares_units(values[i : i + 1]) for i in range(len(values))]
        self.assertEqual(sum(expected), squares_units(values))

    def test_the_input_is_not_modified(self) -> None:
        values = arr([1e-200, 3.0, 1e200, -2.5])
        before = values.copy()
        squares_units(values)
        np.testing.assert_array_equal(values, before)

    def test_a_non_contiguous_batch_gives_the_same_sum(self) -> None:
        base = arr([3.0, 1e-200, 5.0, 1e200, 7.0, 0.1])
        self.assertEqual(squares_units(base[::2]), exact_square_units([3.0, 5.0, 7.0]))
        self.assertEqual(squares_units(base[1::2]), exact_square_units([1e-200, 1e200, 0.1]))

    def test_the_fast_path_does_not_split_a_batch(self) -> None:
        calls = []
        original = _squares._window_units

        def spy(values: np.ndarray) -> int:
            calls.append(len(values))
            return original(values)

        with mock.patch.object(_squares, "_window_units", spy):
            squares_units(arr([1.0, 2.0, 3.0]))
            self.assertEqual(calls, [3])
            calls.clear()
            squares_units(arr([1.0, 1e300, 1e-300, 0.0]))
            self.assertEqual(sorted(calls), [1, 1, 2])


if __name__ == "__main__":
    unittest.main()
