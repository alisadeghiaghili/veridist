"""``lifetimes_from_arrays`` and ``values_from_arrays`` build exactly what a loop would."""

from __future__ import annotations

import unittest
from decimal import Decimal
from typing import Any

import numpy as np

import veridist.domain as domain
from veridist.domain import (
    ExactLifetime,
    ExactValue,
    RightCensoredLifetime,
    RightCensoredValue,
    lifetimes_from_arrays,
    values_from_arrays,
)

ROWS = 100_000


def _lifetime_loop(times: list[float], events: list[bool]) -> tuple[Any, ...]:
    return tuple(
        ExactLifetime(time) if event else RightCensoredLifetime(time)
        for time, event in zip(times, events, strict=True)
    )


def _value_loop(values: list[float], events: list[bool]) -> tuple[Any, ...]:
    return tuple(
        ExactValue(value) if event else RightCensoredValue(value)
        for value, event in zip(values, events, strict=True)
    )


class EquivalenceWithALoopTests(unittest.TestCase):
    def test_lifetimes_equal_the_loop_construction_on_a_hundred_thousand_rows(self) -> None:
        rng = np.random.default_rng(1)
        times = rng.exponential(3.0, ROWS)
        times[::97] = 0.0  # zero is a valid time
        events = rng.random(ROWS) < 0.6
        built = lifetimes_from_arrays(times, events)
        self.assertIs(type(built), tuple)
        self.assertEqual(len(built), ROWS)
        self.assertEqual(built, _lifetime_loop(times.tolist(), events.tolist()))
        self.assertTrue(all(type(item.time) is float for item in built))
        self.assertEqual(
            sum(type(item) is ExactLifetime for item in built), int(np.count_nonzero(events))
        )

    def test_values_equal_the_loop_construction_on_a_hundred_thousand_rows(self) -> None:
        rng = np.random.default_rng(2)
        values = rng.normal(0.0, 5.0, ROWS)
        events = rng.integers(0, 2, ROWS)
        built = values_from_arrays(values, events)
        self.assertEqual(len(built), ROWS)
        self.assertEqual(built, _value_loop(values.tolist(), events.astype(bool).tolist()))
        self.assertTrue(any(item.value < 0.0 for item in built))

    def test_event_may_be_boolean_or_zero_one_integers_of_any_width(self) -> None:
        times = [1.0, 2.0, 3.0, 4.0]
        flags = [True, False, False, True]
        expected = _lifetime_loop(times, flags)
        candidates: list[Any] = [
            flags,
            np.array(flags),
            [1, 0, 0, 1],
            np.array([1, 0, 0, 1], dtype=np.int8),
            np.array([1, 0, 0, 1], dtype=np.uint64),
            np.array([True, 0, 0, 1]),
            (np.True_, np.False_, np.False_, np.True_),
        ]
        for event in candidates:
            with self.subTest(event=repr(event)):
                self.assertEqual(lifetimes_from_arrays(times, event), expected)

    def test_time_may_be_any_real_array_like_of_at_most_64_bits(self) -> None:
        expected = _lifetime_loop([1.0, 2.0, 3.0], [True, False, True])
        for time in (
            [1, 2, 3],
            (1.0, 2.0, 3.0),
            np.array([1, 2, 3], dtype=np.uint8),
            np.array([1, 2, 3], dtype=np.int64),
            np.array([1, 2, 3], dtype=np.uint64),
            np.array([1, 2, 3], dtype=np.float32),
            np.array([1, 2, 3], dtype=np.float16),
        ):
            with self.subTest(time=repr(time)):
                self.assertEqual(lifetimes_from_arrays(time, [True, False, True]), expected)

    def test_empty_columns_give_an_empty_tuple(self) -> None:
        self.assertEqual(lifetimes_from_arrays([], []), ())
        self.assertEqual(values_from_arrays(np.empty(0), np.empty(0, dtype=bool)), ())
        self.assertEqual(lifetimes_from_arrays(np.empty(0, dtype=np.int64), np.empty(0, int)), ())

    def test_the_functions_are_exported_from_the_domain_package(self) -> None:
        self.assertIn("lifetimes_from_arrays", domain.__all__)
        self.assertIn("values_from_arrays", domain.__all__)

    def test_inputs_are_not_modified(self) -> None:
        times = np.array([1.0, 2.0])
        events = np.array([1, 0])
        lifetimes_from_arrays(times, events)
        np.testing.assert_array_equal(times, [1.0, 2.0])
        np.testing.assert_array_equal(events, [1, 0])


class RejectionTests(unittest.TestCase):
    def test_bad_times_name_the_first_bad_row(self) -> None:
        events = [True] * 5
        cases: list[tuple[Any, str]] = [
            ([1.0, 2.0, -0.5, np.nan, 1.0], "row 2"),
            ([1.0, np.nan, 2.0, -1.0, 1.0], "row 1"),
            ([1.0, 2.0, 3.0, np.inf, 1.0], "row 3"),
            ([1.0, 2.0, 3.0, 4.0, -np.inf], "row 4"),
            ([-1, 2, 3, 4, 5], "row 0"),
            (np.array([1, 2, 3, 4, 5], dtype=np.float64) * np.array([1, 1, 1, 1, np.inf]), "row 4"),
        ]
        for times, location in cases:
            with self.subTest(times=repr(times)):
                with self.assertRaisesRegex(ValueError, rf"first bad {location}\)$"):
                    lifetimes_from_arrays(times, events)

    def test_a_negative_value_is_valid_for_values_but_not_for_lifetimes(self) -> None:
        values = values_from_arrays([-2.5, 0.0, 3.0], [True, True, False])
        self.assertEqual(values[0], ExactValue(-2.5))
        self.assertEqual(values[2], RightCensoredValue(3.0))
        with self.assertRaises(ValueError):
            lifetimes_from_arrays([-2.5, 0.0, 3.0], [True, True, False])

    def test_non_finite_values_name_the_first_bad_row(self) -> None:
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(
                    ValueError, r"^value must be finite .*first bad row 2\)$"
                ):
                    values_from_arrays([0.0, 1.0, bad, np.nan], [True] * 4)

    def test_events_other_than_boolean_or_zero_one_are_rejected(self) -> None:
        times = [1.0, 2.0, 3.0]
        for bad_event in ([1, 0, 2], [1, -1, 0], np.array([0, 1, 255], dtype=np.uint8)):
            with self.subTest(event=repr(bad_event)):
                with self.assertRaisesRegex(
                    ValueError, r"^event must be boolean or 0/1 .*row \d\)$"
                ):
                    lifetimes_from_arrays(times, bad_event)
        with self.assertRaisesRegex(ValueError, r"first bad row 1\)$"):
            values_from_arrays(times, [1, 3, 7])
        for bad_event in (
            [1.0, 0.0, 1.0],
            ["a", "b", "c"],
            [None, True, False],
            np.array([1 + 0j, 0, 1]),
            [object()] * 3,
        ):
            with self.subTest(event=repr(bad_event)), self.assertRaises(TypeError):
                lifetimes_from_arrays(times, bad_event)

    def test_times_of_the_wrong_kind_are_type_errors(self) -> None:
        for bad_time in (
            [True, False, True],
            np.array([True, False, True]),
            ["1", "2", "3"],
            [None, 1.0, 2.0],
            np.array([1 + 0j, 2, 3]),
            [object()] * 3,
            [Decimal(1), Decimal(2), Decimal(3)],
        ):
            with self.subTest(time=repr(bad_time)), self.assertRaises(TypeError):
                lifetimes_from_arrays(bad_time, [True, False, True])
            with self.subTest(value=repr(bad_time)), self.assertRaises(TypeError):
                values_from_arrays(bad_time, [True, False, True])

    def test_wide_floats_are_rejected_instead_of_being_silently_narrowed(self) -> None:
        wide = np.array([1.0, 2.0], dtype=np.longdouble)
        if np.dtype(np.longdouble).itemsize > 8:
            with self.assertRaises(TypeError):
                lifetimes_from_arrays(wide, [True, False])
        else:  # a platform where long double is binary64: nothing is narrowed
            self.assertEqual(
                lifetimes_from_arrays(wide, [True, False]),
                _lifetime_loop([1.0, 2.0], [True, False]),
            )

    def test_shape_and_length_mismatches_are_value_errors(self) -> None:
        for time, event in (
            ([1.0, 2.0], [True]),
            ([1.0], [True, False]),
            ([], [True]),
            (np.ones((2, 2)), [True, False]),
            ([1.0, 2.0], np.ones((2, 1), dtype=bool)),
            (np.float64(1.0), [True]),
            ([1.0], np.True_),
        ):
            with self.subTest(time=repr(time), event=repr(event)), self.assertRaises(ValueError):
                lifetimes_from_arrays(time, event)
            with self.subTest(value=repr(time), event=repr(event)), self.assertRaises(ValueError):
                values_from_arrays(time, event)


if __name__ == "__main__":
    unittest.main()
