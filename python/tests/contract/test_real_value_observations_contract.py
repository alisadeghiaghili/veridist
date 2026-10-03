"""Contracts for the real-valued observation types."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
from math import inf, nan

import veridist.domain as domain
from veridist.domain import ExactValue, RightCensoredValue
from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import RealObservation

KINDS = (ExactValue, RightCensoredValue)


class RealValueObservationTests(unittest.TestCase):
    def test_finite_reals_are_stored_as_built_in_floats(self) -> None:
        for kind in KINDS:
            for given, expected in (
                (1.5, 1.5),
                (-3, -3.0),
                (0, 0.0),
                (-0.0, -0.0),
                (Decimal("2.25"), 2.25),
                (Fraction(1, 4), 0.25),
                (10**300, 1e300),
            ):
                with self.subTest(kind=kind.__name__, given=given):
                    value = kind(given).value
                    self.assertIs(type(value), float)
                    self.assertEqual(value, expected)

    def test_negative_values_are_valid_because_the_support_is_the_real_line(self) -> None:
        self.assertEqual(ExactValue(-1e300).value, -1e300)
        self.assertEqual(RightCensoredValue(-5).value, -5.0)

    def test_non_finite_values_are_rejected_with_value_error(self) -> None:
        for kind in KINDS:
            for bad in (nan, inf, -inf, Decimal("NaN"), Decimal("Infinity"), 10**400):
                with self.subTest(kind=kind.__name__, bad=bad), self.assertRaises(ValueError):
                    kind(bad)

    def test_bool_and_non_real_values_are_rejected_with_type_error(self) -> None:
        for kind in KINDS:
            for bad in (True, False, "1.0", None, 1 + 2j, b"1", [1.0], object()):
                with self.subTest(kind=kind.__name__, bad=bad), self.assertRaises(TypeError):
                    kind(bad)  # type: ignore[arg-type]

    def test_instances_are_immutable_hashable_and_compare_by_value(self) -> None:
        exact = ExactValue(2.0)
        with self.assertRaises(FrozenInstanceError):
            exact.value = 3.0  # type: ignore[misc]
        self.assertEqual(ExactValue(2), ExactValue(2.0))
        self.assertEqual(hash(ExactValue(2)), hash(ExactValue(2.0)))
        self.assertNotEqual(ExactValue(2.0), RightCensoredValue(2.0))
        self.assertNotEqual(ExactValue(2.0), ExactValue(2.5))

    def test_real_and_lifetime_types_are_distinct_and_exported_together(self) -> None:
        self.assertNotEqual(ExactValue(1.0), ExactLifetime(1.0))
        self.assertNotEqual(RightCensoredValue(1.0), RightCensoredLifetime(1.0))
        for name in (
            "ExactValue",
            "RightCensoredValue",
            "RealObservation",
            "ExactLifetime",
            "RightCensoredLifetime",
            "LifetimeObservation",
        ):
            self.assertIn(name, domain.__all__)
            self.assertTrue(hasattr(domain, name))
        self.assertEqual(set(RealObservation.__args__), set(KINDS))  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
