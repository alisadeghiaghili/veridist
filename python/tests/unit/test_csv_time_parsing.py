"""CSV time literals convert with ``float`` exactly as a ``Decimal`` round trip would."""

from __future__ import annotations

import random
import unittest
from decimal import Decimal
from math import isfinite

from veridist.adapters.csv_lifetimes import _TIME, CsvLifetimeAdapter, CsvLifetimeAdapterError
from veridist.engine.errors import FailureCode

_EDGE_LITERALS = (
    "0",
    "0.0",
    "0.0e5",
    "0e0",
    "0.000e-9",
    "00",
    "01",
    "1",
    "1e-400",
    "1e400",
    "1e308",
    "1.7976931348623157e308",
    "1.7976931348623159e308",
    "1.797693134862315807e308",
    "4.9e-324",
    "5e-324",
    "2.4703282292062328e-324",
    "2.4703282292062327e-324",
    "2.2250738585072014e-308",
    "0." + "0" * 323 + "1",
    "0." + "0" * 324 + "1",
    "9" * 400,
    "1e",
    "1e+",
    ".5",
    "5.",
    "-1",
    "+1",
    "inf",
    "nan",
    "",
    " 1",
    "1_0",
)


def _reference(value: str) -> float | None:
    """The previous conversion: grammar check, ``Decimal``, then ``float``."""

    if _TIME.fullmatch(value) is None:
        return None
    decimal = Decimal(value)
    converted = float(decimal)
    if not isfinite(converted) or (decimal > 0 and converted == 0.0):
        return None
    return converted


def _parse(value: str) -> float | None:
    try:
        parsed = CsvLifetimeAdapter._parse_time(value, 7)
    except CsvLifetimeAdapterError as error:
        assert error.code is FailureCode.SOURCE_ROW_INVALID
        assert dict(error.context) == {"reason": "invalid_time", "record_offset": 7}
        return None
    assert type(parsed) is float
    return parsed


def _random_literal(rng: random.Random) -> str:
    integer = "0" if rng.random() < 0.25 else str(rng.randint(1, 9)) + "".join(
        rng.choice("0123456789") for _ in range(rng.randint(0, 18))
    )
    fraction = (
        ""
        if rng.random() < 0.3
        else "." + "".join(rng.choice("0123456789") for _ in range(rng.randint(1, 25)))
    )
    pick = rng.random()
    if pick < 0.2:
        exponent = ""
    elif pick < 0.45:
        exponent = rng.choice("eE") + rng.choice(("", "+", "-")) + str(rng.randint(0, 30))
    elif pick < 0.7:
        exponent = rng.choice("eE") + "-" + str(rng.randint(295, 340))
    elif pick < 0.85:
        exponent = rng.choice("eE") + rng.choice(("", "+")) + str(rng.randint(295, 330))
    else:
        exponent = rng.choice("eE") + rng.choice(("", "+", "-")) + str(rng.randint(0, 5000))
    if rng.random() < 0.1:
        integer = "0"
        fraction = "." + "0" * rng.randint(0, 330) + rng.choice(("", "1", "49", "5", "25", "3"))
    return integer + fraction + exponent


class CsvTimeParsingTests(unittest.TestCase):
    def test_named_edge_literals(self) -> None:
        expected: dict[str, float | None] = {
            "0": 0.0,
            "0.0e5": 0.0,
            "00": None,
            "1e-400": None,
            "1e400": None,
            "4.9e-324": 5e-324,
            "2.4703282292062328e-324": 5e-324,
            "2.4703282292062327e-324": None,
            "1.7976931348623157e308": 1.7976931348623157e308,
            "1.7976931348623159e308": None,
        }
        for literal, outcome in expected.items():
            with self.subTest(literal=literal):
                self.assertEqual(_parse(literal), outcome)

    def test_edge_literals_match_the_decimal_reference(self) -> None:
        for literal in _EDGE_LITERALS:
            with self.subTest(literal=literal):
                self.assertEqual(_parse(literal), _reference(literal))

    def test_randomized_literals_match_the_decimal_reference(self) -> None:
        rng = random.Random(20260930)
        accepted = 0
        rejected = 0
        for _ in range(2000):
            literal = _random_literal(rng)
            outcome = _parse(literal)
            self.assertEqual(outcome, _reference(literal), literal)
            if outcome is None:
                rejected += 1
            else:
                accepted += 1
        # Both outcomes are exercised, not just the easy accepting path.
        self.assertGreater(accepted, 500)
        self.assertGreater(rejected, 50)

    def test_zero_with_any_exponent_is_zero_but_a_nonzero_mantissa_may_underflow(self) -> None:
        for literal in ("0e5", "0.000E+7", "0e-400", "0.0e-9999"):
            with self.subTest(literal=literal):
                self.assertEqual(_parse(literal), 0.0)
        for literal in ("1e-400", "0.1e-324", "9e-9999"):
            with self.subTest(literal=literal):
                self.assertIsNone(_parse(literal))

    def test_astronomical_exponents_are_typed_not_raw_errors(self) -> None:
        # `Decimal` itself raises `InvalidOperation` beyond roughly 1e18 digits
        # of exponent; `float` does not, so these are ordinary typed outcomes.
        self.assertIsNone(_parse("1e9999999999999999999999"))
        self.assertIsNone(_parse("7e-99999999999999999999"))
        self.assertEqual(_parse("0e9999999999999999999999"), 0.0)
