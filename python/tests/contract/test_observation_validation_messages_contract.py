"""Rejected observations say which input is wrong and why.

The messages are the public explanation of a rejected observation or column: the
name of the offending input, what it must be and, for a column, the first bad row.
Each case checks the exception type together with its whole message.
"""

from __future__ import annotations

import unittest
from decimal import Decimal
from typing import Any

from veridist.domain import (
    ExactLifetime,
    ExactValue,
    RightCensoredLifetime,
    RightCensoredValue,
    lifetimes_from_arrays,
    values_from_arrays,
)

HUGE_INTEGER = 10**400  # an int that float() cannot represent


class ScalarObservationMessageTests(unittest.TestCase):
    def check(self, build: Any, argument: Any, error: type[Exception], message: str) -> None:
        with self.assertRaises(error) as captured:
            build(argument)
        self.assertIs(type(captured.exception), error)
        self.assertEqual(str(captured.exception), message)

    def test_a_lifetime_rejects_a_non_number_by_name(self) -> None:
        for build in (ExactLifetime, RightCensoredLifetime):
            for argument in ("1.0", None, True, 1 + 2j):
                with self.subTest(build=build.__name__, argument=argument):
                    self.check(build, argument, TypeError, "time must be a finite numeric value")

    def test_a_lifetime_rejects_values_that_are_not_finite_and_non_negative(self) -> None:
        message = "time must be finite and non-negative"
        for build in (ExactLifetime, RightCensoredLifetime):
            for argument in (-1.0, float("nan"), float("inf"), HUGE_INTEGER, -HUGE_INTEGER):
                with self.subTest(build=build.__name__, argument=argument):
                    self.check(build, argument, ValueError, message)

    def test_a_positive_lifetime_that_underflows_to_zero_is_rejected(self) -> None:
        for build in (ExactLifetime, RightCensoredLifetime):
            for argument in (Decimal("1e-400"), Decimal("4e-324") / 10):
                with self.subTest(build=build.__name__, argument=argument):
                    self.check(
                        build, argument, ValueError, "positive time must remain representable"
                    )

    def test_a_value_rejects_a_non_number_by_name(self) -> None:
        for build in (ExactValue, RightCensoredValue):
            for argument in ("1.0", None, True, 1 + 2j):
                with self.subTest(build=build.__name__, argument=argument):
                    self.check(build, argument, TypeError, "value must be a finite real number")

    def test_a_value_rejects_values_that_are_not_finite(self) -> None:
        for build in (ExactValue, RightCensoredValue):
            for argument in (float("nan"), float("-inf"), HUGE_INTEGER, -HUGE_INTEGER):
                with self.subTest(build=build.__name__, argument=argument):
                    self.check(build, argument, ValueError, "value must be finite")


class ColumnMessageTests(unittest.TestCase):
    def check(self, call: Any, error: type[Exception], message: str, *columns: Any) -> None:
        with self.assertRaises(error) as captured:
            call(*columns)
        self.assertIs(type(captured.exception), error)
        self.assertEqual(str(captured.exception), message)

    def test_a_column_of_the_wrong_kind_is_a_type_error_naming_the_column(self) -> None:
        wrong_kind = ["a", "b"]
        self.check(
            lifetimes_from_arrays,
            TypeError,
            "time must be an array of real numbers of at most 64 bits",
            wrong_kind,
            [True, False],
        )
        self.check(
            values_from_arrays,
            TypeError,
            "value must be an array of real numbers of at most 64 bits",
            wrong_kind,
            [True, False],
        )

    def test_an_event_column_of_the_wrong_kind_is_a_type_error(self) -> None:
        message = "event must be a boolean array or an integer array of 0 and 1"
        for events in ([1.0, 0.0], ["1", "0"], [1.5, 0.5]):
            with self.subTest(events=events):
                self.check(lifetimes_from_arrays, TypeError, message, [1.0, 2.0], events)
                self.check(values_from_arrays, TypeError, message, [1.0, 2.0], events)

    def test_a_column_that_is_not_one_dimensional_is_a_value_error_naming_it(self) -> None:
        self.check(
            lifetimes_from_arrays, ValueError, "time must be one-dimensional", [[1.0]], [True]
        )
        self.check(
            values_from_arrays, ValueError, "value must be one-dimensional", [[1.0]], [True]
        )
        for events in ([[True]], [[1]]):
            with self.subTest(events=events):
                self.check(
                    lifetimes_from_arrays,
                    ValueError,
                    "event must be one-dimensional",
                    [1.0],
                    events,
                )
                self.check(
                    values_from_arrays, ValueError, "event must be one-dimensional", [1.0], events
                )

    def test_columns_of_different_length_are_a_value_error(self) -> None:
        message = "the columns must have the same length"
        self.check(lifetimes_from_arrays, ValueError, message, [1.0, 2.0], [True])
        self.check(values_from_arrays, ValueError, message, [1.0], [True, False])

    def test_a_bad_time_names_the_column_the_requirement_and_the_first_bad_row(self) -> None:
        message = "time must be finite and non-negative (first bad row 2)"
        for times in ([1.0, 2.0, -1.0, float("nan")], [1.0, 2.0, float("inf"), -3.0]):
            with self.subTest(times=times):
                self.check(lifetimes_from_arrays, ValueError, message, times, [1, 0, 1, 0])

    def test_a_bad_value_names_the_column_and_the_first_bad_row(self) -> None:
        message = "value must be finite (first bad row 1)"
        for values in ([-1.0, float("nan"), float("inf")], [-1.0, float("-inf"), 0.0]):
            with self.subTest(values=values):
                self.check(values_from_arrays, ValueError, message, values, [1, 0, 1])

    def test_a_negative_value_is_not_a_bad_value(self) -> None:
        built = values_from_arrays([-1.5, 2.0], [True, False])
        self.assertEqual(built, (ExactValue(-1.5), RightCensoredValue(2.0)))

    def test_a_bad_event_integer_names_the_first_bad_row(self) -> None:
        message = "event must be boolean or 0/1 (first bad row 1)"
        self.check(lifetimes_from_arrays, ValueError, message, [1.0, 2.0, 3.0], [1, 2, -1])
        self.check(values_from_arrays, ValueError, message, [1.0, 2.0, 3.0], [1, 2, -1])


if __name__ == "__main__":
    unittest.main()
