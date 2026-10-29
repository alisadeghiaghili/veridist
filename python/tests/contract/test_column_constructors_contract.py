"""The multi-column constructors ``lifetimes_from_columns`` and ``values_from_columns``.

Evidence CENS-OBS-01.

An explicit status column says what each row is (0 right-censored, 1 exact, 2 left-censored,
3 interval-censored); ``upper`` is the interval upper bound; the truncation columns hold one
number per row, with negative infinity (lower) and positive infinity (upper) meaning "not
truncated".  Nulls and NaN are always refused, every error names the first bad row, and the
result equals building the tuple row by row with the constructors.
"""

from __future__ import annotations

import unittest
from math import inf, nan
from typing import Any

import numpy as np

from veridist import domain
from veridist.domain import (
    ExactLifetime,
    ExactValue,
    IntervalCensoredLifetime,
    IntervalCensoredValue,
    LeftCensoredLifetime,
    LeftCensoredValue,
    RightCensoredLifetime,
    RightCensoredValue,
    lifetimes_from_columns,
    values_from_columns,
)

NAN = nan
BUILDERS = (
    (
        lifetimes_from_columns,
        (RightCensoredLifetime, ExactLifetime, LeftCensoredLifetime, IntervalCensoredLifetime),
        True,
    ),
    (
        values_from_columns,
        (RightCensoredValue, ExactValue, LeftCensoredValue, IntervalCensoredValue),
        False,
    ),
)


class ColumnTestCase(unittest.TestCase):
    def refuse(
        self, error: type[Exception], message: str, call: Any, *args: Any, **kw: Any
    ) -> None:
        with self.assertRaises(error) as captured:
            call(*args, **kw)
        self.assertIs(type(captured.exception), error)
        self.assertEqual(str(captured.exception), message)


class ConstructionTests(ColumnTestCase):
    def test_every_status_builds_its_kind_and_equals_the_row_by_row_tuple(self) -> None:
        for build, (right, exact, left, interval), _ in BUILDERS:
            with self.subTest(build=build.__name__):
                result = build(
                    [1.0, 2.0, 3.0, 4.0, 5.0],
                    [1, 0, 2, 3, 1],
                    upper=[NAN, NAN, NAN, 6.5, NAN],
                    truncation_lower=[-inf, 0.5, 1.0, 2.0, 5.0],
                    truncation_upper=[inf, 9.0, 3.5, inf, 5.5],
                )
                expected = (
                    exact(1.0),
                    right(2.0, truncation_lower=0.5, truncation_upper=9.0),
                    left(3.0, truncation_lower=1.0, truncation_upper=3.5),
                    interval(4.0, 6.5, truncation_lower=2.0),
                    exact(5.0, truncation_lower=5.0, truncation_upper=5.5),
                )
                self.assertEqual(result, expected)
                self.assertIsInstance(result, tuple)
                self.assertEqual([type(row) for row in result], [type(row) for row in expected])

    def test_status_accepts_booleans_and_every_integer_width(self) -> None:
        for build, (right, exact, _left, _interval), _ in BUILDERS:
            with self.subTest(build=build.__name__):
                expected = (exact(1.0), right(2.0), exact(3.0))
                self.assertEqual(build([1.0, 2.0, 3.0], [True, False, True]), expected)
                for dtype in (np.int8, np.int32, np.int64, np.uint8, np.uint64):
                    status = np.array([1, 0, 1], dtype=dtype)
                    self.assertEqual(build([1.0, 2.0, 3.0], status), expected)

    def test_values_may_be_integers_and_status_one_is_event_equal(self) -> None:
        from veridist import lifetimes_from_arrays, values_from_arrays

        self.assertEqual(
            lifetimes_from_columns([1, 2, 3], [1, 0, 1]),
            lifetimes_from_arrays([1, 2, 3], [1, 0, 1]),
        )
        self.assertEqual(
            values_from_columns([-1, 2, -3], [True, False, True]),
            values_from_arrays([-1, 2, -3], [True, False, True]),
        )

    def test_columns_may_be_empty_and_optional_columns_may_be_left_out(self) -> None:
        for build, _, _ in BUILDERS:
            self.assertEqual(build([], []), ())
            self.assertEqual(build([], [], upper=[], truncation_lower=[], truncation_upper=[]), ())
            self.assertEqual(build(np.array([]), np.array([], dtype=np.int8)), ())

    def test_infinities_mean_untruncated_and_a_lifetime_lower_zero_too(self) -> None:
        plain = lifetimes_from_columns([5.0], [1])
        self.assertEqual(
            lifetimes_from_columns([5.0], [1], truncation_lower=[0.0], truncation_upper=[inf]),
            plain,
        )
        self.assertEqual(
            lifetimes_from_columns([5.0], [1], truncation_lower=[-inf], truncation_upper=[inf]),
            plain,
        )
        self.assertEqual(plain[0].truncation_lower, None)

    def test_the_result_does_not_depend_on_the_position_of_a_row(self) -> None:
        value = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        status = np.array([1, 0, 2, 3, 1, 0])
        upper = np.array([NAN, NAN, NAN, 8.5, NAN, NAN])
        lower_end = np.array([-inf, 1.0, 2.0, 3.0, 0.5, -inf])
        upper_end = np.array([inf, inf, 9.0, 9.0, inf, 10.0])
        reference = lifetimes_from_columns(
            value, status, upper=upper, truncation_lower=lower_end, truncation_upper=upper_end
        )
        for order in ([5, 4, 3, 2, 1, 0], [2, 0, 5, 1, 4, 3], [3, 3, 0, 0, 5, 1]):
            index = np.array(order)
            shuffled = lifetimes_from_columns(
                value[index],
                status[index],
                upper=upper[index],
                truncation_lower=lower_end[index],
                truncation_upper=upper_end[index],
            )
            self.assertEqual(shuffled, tuple(reference[i] for i in order))
        for row in range(6):
            single = lifetimes_from_columns(
                value[row : row + 1],
                status[row : row + 1],
                upper=upper[row : row + 1],
                truncation_lower=lower_end[row : row + 1],
                truncation_upper=upper_end[row : row + 1],
            )
            self.assertEqual(single, (reference[row],))

    def test_the_constructors_are_exported(self) -> None:
        import veridist

        for name in ("lifetimes_from_columns", "values_from_columns"):
            self.assertIn(name, domain.__all__)
            self.assertIn(name, veridist.__all__)
            self.assertIs(getattr(veridist, name), getattr(domain, name))


class RefusalTests(ColumnTestCase):
    def test_a_column_of_the_wrong_kind_is_a_type_error_naming_it(self) -> None:
        for build, _, _ in BUILDERS:
            with self.subTest(build=build.__name__):
                self.refuse(
                    TypeError,
                    "value must be an array of real numbers of at most 64 bits",
                    build,
                    ["a", "b"],
                    [1, 0],
                )
                self.refuse(
                    TypeError,
                    "value must be an array of real numbers of at most 64 bits",
                    build,
                    [True, False],
                    [1, 0],
                )
                for name in ("upper", "truncation_lower", "truncation_upper"):
                    self.refuse(
                        TypeError,
                        f"{name} must be an array of real numbers of at most 64 bits",
                        build,
                        [1.0, 2.0],
                        [1, 0],
                        **{name: ["a", "b"]},
                    )

    def test_a_status_of_the_wrong_kind_or_value_is_refused(self) -> None:
        message = "status must be a boolean array or an integer array of 0, 1, 2 and 3"
        for build, _, _ in BUILDERS:
            for status in ([1.0, 0.0], ["1", "0"], [1.5, 0.5]):
                with self.subTest(build=build.__name__, status=status):
                    self.refuse(TypeError, message, build, [1.0, 2.0], status)
            for status, row in (([1, 4], 1), ([-1, 1], 0), ([1, 0, 7], 2)):
                with self.subTest(build=build.__name__, status=status):
                    self.refuse(
                        ValueError,
                        f"status must be boolean or 0, 1, 2 or 3 (first bad row {row})",
                        build,
                        [1.0] * len(status),
                        status,
                    )

    def test_shapes_and_lengths_are_checked(self) -> None:
        for build, _, _ in BUILDERS:
            with self.subTest(build=build.__name__):
                self.refuse(ValueError, "value must be one-dimensional", build, [[1.0]], [1])
                self.refuse(ValueError, "status must be one-dimensional", build, [1.0], [[1]])
                length = "the columns must have the same length"
                self.refuse(ValueError, length, build, [1.0, 2.0], [1])
                self.refuse(ValueError, length, build, [1.0], [1, 0])
                for name in ("upper", "truncation_lower", "truncation_upper"):
                    self.refuse(ValueError, length, build, [1.0, 2.0], [1, 0], **{name: [1.0]})
                    self.refuse(
                        ValueError,
                        f"{name} must be one-dimensional",
                        build,
                        [1.0],
                        [1],
                        **{name: [[1.0]]},
                    )

    def test_value_is_finite_and_a_lifetime_value_is_non_negative(self) -> None:
        for bad in (NAN, inf, -inf):
            self.refuse(
                ValueError,
                "value must be finite and non-negative (first bad row 1)",
                lifetimes_from_columns,
                [1.0, bad],
                [1, 1],
            )
            self.refuse(
                ValueError,
                "value must be finite (first bad row 1)",
                values_from_columns,
                [1.0, bad],
                [1, 1],
            )
        self.refuse(
            ValueError,
            "value must be finite and non-negative (first bad row 2)",
            lifetimes_from_columns,
            [1.0, 2.0, -0.5],
            [1, 0, 2],
        )
        self.assertEqual(values_from_columns([-0.5], [2]), (LeftCensoredValue(-0.5),))

    def test_upper_is_required_finite_on_interval_rows_and_nan_elsewhere(self) -> None:
        for build, _, _ in BUILDERS:
            with self.subTest(build=build.__name__):
                self.refuse(
                    ValueError,
                    "upper is required for interval-censored rows (first bad row 1)",
                    build,
                    [1.0, 2.0],
                    [1, 3],
                )
                for bad in (NAN, inf, -inf):
                    self.refuse(
                        ValueError,
                        "upper must be finite on interval-censored rows (first bad row 1)",
                        build,
                        [1.0, 2.0],
                        [1, 3],
                        upper=[NAN, bad],
                    )
                for status in (0, 1, 2):
                    self.refuse(
                        ValueError,
                        "upper must be NaN on rows that are not interval-censored "
                        "(first bad row 1)",
                        build,
                        [1.0, 2.0, 3.0],
                        [3, status, 3],
                        upper=[2.0, 5.0, 4.0],
                    )
                self.assertEqual(
                    len(build([1.0], [3], upper=[2.0])),
                    1,
                )

    def test_a_row_the_constructors_refuse_names_its_index(self) -> None:
        cases = (
            (lifetimes_from_columns, [1.0, 0.0], [1, 2], {}, "time must be finite and positive"),
            (
                values_from_columns,
                [1.0, 2.0],
                [1, 3],
                {"upper": [NAN, 2.0]},
                "lower and upper must differ: an interval with equal bounds is an exact "
                "observation, use ExactValue",
            ),
            (
                lifetimes_from_columns,
                [1.0, 3.0],
                [1, 3],
                {"upper": [NAN, 2.0]},
                "upper must be greater than lower",
            ),
            (
                values_from_columns,
                [1.0, 2.0],
                [1, 3],
                {"upper": [NAN, 1.5]},
                "upper must be greater than lower",
            ),
        )
        for build, value, status, keywords, message in cases:
            with self.subTest(message=message):
                self.refuse(
                    ValueError, f"{message} (first bad row 1)", build, value, status, **keywords
                )

    def test_a_truncation_column_is_finite_or_its_own_open_infinity(self) -> None:
        for build, _, _ in BUILDERS:
            for bad in (NAN, inf):
                self.refuse(
                    ValueError,
                    "truncation_lower must be finite or negative infinity (first bad row 1)",
                    build,
                    [5.0, 5.0],
                    [1, 1],
                    truncation_lower=[1.0, bad],
                )
            for bad in (NAN, -inf):
                self.refuse(
                    ValueError,
                    "truncation_upper must be finite or positive infinity (first bad row 1)",
                    build,
                    [5.0, 5.0],
                    [1, 1],
                    truncation_upper=[9.0, bad],
                )

    def test_a_lifetime_window_is_in_the_non_negative_half_line(self) -> None:
        self.refuse(
            ValueError,
            "truncation_lower must be non-negative (first bad row 1)",
            lifetimes_from_columns,
            [5.0, 5.0],
            [1, 1],
            truncation_lower=[1.0, -1.0],
        )
        for bad in (0.0, -2.0):
            self.refuse(
                ValueError,
                "truncation_upper must be positive (first bad row 1)",
                lifetimes_from_columns,
                [5.0, 5.0],
                [1, 1],
                truncation_upper=[9.0, bad],
            )
        self.assertEqual(
            values_from_columns([-5.0], [1], truncation_lower=[-9.0], truncation_upper=[-1.0])[
                0
            ].truncation_upper,
            -1.0,
        )

    def test_a_row_outside_its_window_is_refused_and_never_clipped(self) -> None:
        cases = (
            (
                [5.0, 5.0],
                [1, 1],
                {"truncation_lower": [1.0, 6.0]},
                "truncation_lower must not be after the exact observation",
            ),
            (
                [5.0, 5.0],
                [1, 1],
                {"truncation_upper": [9.0, 4.0]},
                "truncation_upper must not be before the exact observation",
            ),
            (
                [5.0, 5.0],
                [0, 0],
                {"truncation_upper": [9.0, 5.0]},
                "truncation_upper must be after the censoring point",
            ),
            (
                [5.0, 5.0],
                [2, 2],
                {"truncation_lower": [1.0, 5.0]},
                "truncation_lower must be before the censoring point",
            ),
            (
                [5.0, 5.0],
                [2, 2],
                {"truncation_upper": [9.0, 4.0]},
                "truncation_upper must not be before the censoring point",
            ),
            (
                [5.0, 5.0],
                [3, 3],
                {"upper": [6.0, 6.0], "truncation_lower": [1.0, 5.5]},
                "truncation_lower must not be after the lower bound",
            ),
            (
                [5.0, 5.0],
                [3, 3],
                {"upper": [6.0, 6.0], "truncation_upper": [9.0, 5.5]},
                "truncation_upper must not be before the upper bound",
            ),
        )
        for value, status, keywords, message in cases:
            for build, _, _ in BUILDERS:
                with self.subTest(build=build.__name__, message=message):
                    self.refuse(
                        ValueError, f"{message} (first bad row 1)", build, value, status, **keywords
                    )

    def test_the_first_bad_row_is_reported_when_several_rows_are_bad(self) -> None:
        self.refuse(
            ValueError,
            "truncation_lower must not be after the exact observation (first bad row 1)",
            lifetimes_from_columns,
            [5.0, 5.0, 5.0],
            [1, 1, 1],
            truncation_lower=[1.0, 6.0, 7.0],
        )
        self.refuse(
            ValueError,
            "value must be finite and non-negative (first bad row 0)",
            lifetimes_from_columns,
            [-1.0, NAN],
            [1, 1],
        )


if __name__ == "__main__":
    unittest.main()
