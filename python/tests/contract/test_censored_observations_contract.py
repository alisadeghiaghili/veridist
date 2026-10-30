"""The observation model of censored and truncated data (CENS-OBS-01).

Eight frozen observation types: exact, right-, left- and interval-censored, for lifetimes and for
real values, each with an optional truncation window ``(truncation_lower, truncation_upper]``.
Every rule is enforced at construction with ``TypeError`` or ``ValueError`` and a message that
names the attribute and never a data value.  Observations without a window keep the equality,
hash, ``repr`` and pickle of the types before truncation existed.
"""

from __future__ import annotations

import dataclasses
import pickle
import unittest
from decimal import Decimal
from math import inf, nan
from typing import Any

import numpy as np

from veridist.domain import (
    ExactLifetime,
    ExactValue,
    IntervalCensoredLifetime,
    IntervalCensoredValue,
    LeftCensoredLifetime,
    LeftCensoredValue,
    RightCensoredLifetime,
    RightCensoredValue,
)

HUGE_INTEGER = 10**400  # an int that float() cannot represent

LIFETIME_POINTS = (ExactLifetime, RightCensoredLifetime, LeftCensoredLifetime)
VALUE_POINTS = (ExactValue, RightCensoredValue, LeftCensoredValue)
POINT_TYPES = LIFETIME_POINTS + VALUE_POINTS
INTERVAL_TYPES = (IntervalCensoredLifetime, IntervalCensoredValue)
ALL_TYPES = POINT_TYPES + INTERVAL_TYPES

#: Pickles of the four types of the earlier releases, protocol 4, as those releases wrote them.
OLD_PICKLES = (
    (
        ExactLifetime(2.5),
        "8004953f000000000000008c1976657269646973742e646f6d61696e2e6c69666574696d6573948c0d"
        "45786163744c69666574696d659493942981945d9447400400000000000061622e",
    ),
    (
        RightCensoredLifetime(3.0),
        "80049547000000000000008c1976657269646973742e646f6d61696e2e6c69666574696d6573948c15"
        "526967687443656e736f7265644c69666574696d659493942981945d9447400800000000000061622e",
    ),
    (
        ExactValue(-1.5),
        "80049539000000000000008c1676657269646973742e646f6d61696e2e76616c756573948c0a457861"
        "637456616c75659493942981945d9447bff800000000000061622e",
    ),
    (
        RightCensoredValue(0.25),
        "80049541000000000000008c1676657269646973742e646f6d61696e2e76616c756573948c12526967"
        "687443656e736f72656456616c75659493942981945d94473fd000000000000061622e",
    ),
)


def build(kind: type, point: float, **window: Any) -> Any:
    """An observation of ``kind`` at ``point`` (an interval ``(point, point + 1]``)."""

    if kind in INTERVAL_TYPES:
        return kind(point, point + 1.0, **window)
    return kind(point, **window)


class MessageTestCase(unittest.TestCase):
    def refuse(
        self, error: type[Exception], message: str, build_it: Any, *args: Any, **kw: Any
    ) -> None:
        with self.assertRaises(error) as captured:
            build_it(*args, **kw)
        self.assertIs(type(captured.exception), error)
        self.assertEqual(str(captured.exception), message)


class UntruncatedObservationsAreUnchangedTests(unittest.TestCase):
    """Equality, hash, repr and pickling of the four earlier types without a window."""

    def test_repr_hash_and_equality_are_those_of_the_earlier_releases(self) -> None:
        cases = (
            (ExactLifetime(2.5), "ExactLifetime(time=2.5)", 2.5),
            (RightCensoredLifetime(3), "RightCensoredLifetime(time=3.0)", 3.0),
            (ExactValue(-1.5), "ExactValue(value=-1.5)", -1.5),
            (RightCensoredValue(Decimal("0.25")), "RightCensoredValue(value=0.25)", 0.25),
        )
        for observation, text, number in cases:
            with self.subTest(text=text):
                self.assertEqual(repr(observation), text)
                self.assertEqual(hash(observation), hash((number,)))
                self.assertEqual(observation, type(observation)(number))
                self.assertNotEqual(observation, type(observation)(number + 1.0))
                self.assertEqual({observation, type(observation)(number)}, {observation})

    def test_the_pickles_of_the_earlier_releases_load_and_the_new_ones_are_identical(self) -> None:
        for observation, hexadecimal in OLD_PICKLES:
            with self.subTest(observation=observation):
                old = bytes.fromhex(hexadecimal)
                self.assertEqual(pickle.dumps(observation, protocol=4), old)
                loaded = pickle.loads(old)
                self.assertEqual(loaded, observation)
                self.assertIsNone(loaded.truncation_lower)
                self.assertIsNone(loaded.truncation_upper)

    def test_different_kinds_with_the_same_number_are_not_equal(self) -> None:
        self.assertNotEqual(ExactLifetime(1.0), RightCensoredLifetime(1.0))
        self.assertNotEqual(LeftCensoredLifetime(1.0), RightCensoredLifetime(1.0))
        self.assertNotEqual(ExactValue(1.0), LeftCensoredValue(1.0))


class ConstructionTests(MessageTestCase):
    def test_every_type_is_frozen_and_slotted(self) -> None:
        for kind in ALL_TYPES:
            observation = build(kind, 2.0)
            with self.subTest(kind=kind.__name__):
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    observation.truncation_lower = 1.0
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    observation.truncation_upper = 1.0
                self.assertFalse(hasattr(observation, "__dict__"))
                self.assertIsNone(observation.truncation_lower)
                self.assertIsNone(observation.truncation_upper)

    def test_the_truncation_attributes_are_keyword_only(self) -> None:
        for kind in POINT_TYPES:
            with self.subTest(kind=kind.__name__):
                with self.assertRaises(TypeError):
                    kind(2.0, 1.0)
        for kind in INTERVAL_TYPES:
            with self.subTest(kind=kind.__name__):
                with self.assertRaises(TypeError):
                    kind(2.0, 3.0, 1.0)

    def test_numbers_are_stored_as_floats_from_every_real_spelling(self) -> None:
        for spelling in (2, 2.0, Decimal("2"), np.int64(2), np.float32(2.0), np.float64(2.0)):
            with self.subTest(spelling=repr(spelling)):
                self.assertEqual(ExactLifetime(spelling).time, 2.0)
                self.assertEqual(IntervalCensoredLifetime(spelling, 3).lower, 2.0)
                self.assertEqual(LeftCensoredValue(spelling).value, 2.0)
                self.assertEqual(IntervalCensoredValue(1, spelling).upper, 2.0)
                self.assertIs(type(ExactLifetime(spelling).time), float)

    def test_a_lifetime_point_is_a_finite_number(self) -> None:
        for kind in LIFETIME_POINTS:
            for argument in ("1.0", None, True, 1 + 2j):
                with self.subTest(kind=kind.__name__, argument=argument):
                    self.refuse(TypeError, "time must be a finite numeric value", kind, argument)
            for argument in (float("nan"), inf, -inf, HUGE_INTEGER):
                with self.subTest(kind=kind.__name__, argument=argument):
                    self.refuse(
                        ValueError,
                        "time must be finite and "
                        + ("positive" if kind is LeftCensoredLifetime else "non-negative"),
                        kind,
                        argument,
                    )

    def test_a_lifetime_is_never_negative_and_a_left_censoring_point_is_positive(self) -> None:
        for kind in (ExactLifetime, RightCensoredLifetime):
            self.refuse(ValueError, "time must be finite and non-negative", kind, -1.0)
            self.assertEqual(kind(0.0).time, 0.0)
        for argument in (0.0, -1.0, Decimal("-1e-3")):
            self.refuse(
                ValueError, "time must be finite and positive", LeftCensoredLifetime, argument
            )
        self.assertEqual(LeftCensoredLifetime(5e-324).time, 5e-324)

    def test_a_positive_lifetime_that_underflows_to_zero_is_refused(self) -> None:
        for kind in LIFETIME_POINTS:
            self.refuse(
                ValueError, "positive time must remain representable", kind, Decimal("1e-400")
            )
        self.refuse(
            ValueError,
            "positive lower must remain representable",
            IntervalCensoredLifetime,
            Decimal("1e-400"),
            1.0,
        )
        self.refuse(
            ValueError,
            "positive upper must remain representable",
            IntervalCensoredLifetime,
            0.0,
            Decimal("1e-400"),
        )

    def test_a_value_point_is_a_finite_real_number_of_any_sign(self) -> None:
        for kind in VALUE_POINTS:
            for argument in ("1.0", None, True, 1 + 2j):
                with self.subTest(kind=kind.__name__, argument=argument):
                    self.refuse(TypeError, "value must be a finite real number", kind, argument)
            for argument in (float("nan"), inf, -inf, HUGE_INTEGER, -HUGE_INTEGER):
                with self.subTest(kind=kind.__name__, argument=argument):
                    self.refuse(ValueError, "value must be finite", kind, argument)
            self.assertEqual(kind(-3.5).value, -3.5)
            self.assertEqual(kind(0).value, 0.0)

    def test_interval_bounds_are_finite_numbers_in_the_domain(self) -> None:
        for name, position in (("lower", 0), ("upper", 1)):
            args = [1.0, 2.0]
            for bad in ("1.0", None, True, 1 + 2j):
                args[position] = bad
                self.refuse(
                    TypeError,
                    f"{name} must be a finite numeric value",
                    IntervalCensoredLifetime,
                    *args,
                )
                self.refuse(
                    TypeError, f"{name} must be a finite real number", IntervalCensoredValue, *args
                )
            for bad in (nan, inf, -inf):
                args[position] = bad
                sign = "non-negative" if name == "lower" else "positive"
                self.refuse(
                    ValueError,
                    f"{name} must be finite and {sign}",
                    IntervalCensoredLifetime,
                    *args,
                )
                self.refuse(ValueError, f"{name} must be finite", IntervalCensoredValue, *args)
        self.refuse(
            ValueError, "lower must be finite and non-negative", IntervalCensoredLifetime, -1.0, 2.0
        )
        self.refuse(
            ValueError, "upper must be finite and positive", IntervalCensoredLifetime, 0.0, 0.0
        )
        self.assertEqual(IntervalCensoredLifetime(0.0, 1.0).lower, 0.0)
        self.assertEqual(IntervalCensoredValue(-2.0, -1.0).lower, -2.0)

    def test_equal_bounds_are_refused_and_point_to_the_exact_type(self) -> None:
        equal = (
            "lower and upper must differ: an interval with equal bounds is an exact observation, "
            "use {}"
        )
        self.refuse(ValueError, equal.format("ExactLifetime"), IntervalCensoredLifetime, 2.0, 2.0)
        self.refuse(ValueError, equal.format("ExactValue"), IntervalCensoredValue, 2.0, 2.0)
        self.refuse(ValueError, equal.format("ExactValue"), IntervalCensoredValue, Decimal("2"), 2)

    def test_reversed_bounds_are_refused(self) -> None:
        message = "upper must be greater than lower"
        self.refuse(ValueError, message, IntervalCensoredLifetime, 3.0, 2.0)
        self.refuse(ValueError, message, IntervalCensoredValue, 3.0, -2.0)

    def test_the_interval_is_half_open_and_its_bounds_are_kept(self) -> None:
        observation = IntervalCensoredLifetime(200, 300)
        self.assertEqual((observation.lower, observation.upper), (200.0, 300.0))
        self.assertEqual(
            IntervalCensoredLifetime(1.0, 2.0), IntervalCensoredLifetime(1.0, Decimal(2))
        )
        self.assertNotEqual(IntervalCensoredLifetime(1.0, 2.0), IntervalCensoredLifetime(1.0, 3.0))


class TruncationWindowTests(MessageTestCase):
    def test_a_window_is_two_optional_ends_and_is_kept(self) -> None:
        for kind in ALL_TYPES:
            with self.subTest(kind=kind.__name__):
                both = build(kind, 5.0, truncation_lower=2, truncation_upper=Decimal(9))
                self.assertEqual((both.truncation_lower, both.truncation_upper), (2.0, 9.0))
                left = build(kind, 5.0, truncation_lower=2.0)
                self.assertEqual((left.truncation_lower, left.truncation_upper), (2.0, None))
                right = build(kind, 5.0, truncation_upper=np.float64(9.0))
                self.assertEqual((right.truncation_lower, right.truncation_upper), (None, 9.0))

    def test_the_open_infinity_of_a_side_is_the_same_as_no_window_there(self) -> None:
        for kind in ALL_TYPES:
            with self.subTest(kind=kind.__name__):
                plain = build(kind, 5.0)
                self.assertEqual(build(kind, 5.0, truncation_lower=-inf), plain)
                self.assertEqual(build(kind, 5.0, truncation_upper=inf), plain)
                self.assertEqual(
                    build(kind, 5.0, truncation_lower=-inf, truncation_upper=inf), plain
                )
                self.assertEqual(
                    hash(build(kind, 5.0, truncation_lower=-inf, truncation_upper=inf)),
                    hash(plain),
                )

    def test_a_lifetime_lower_end_of_zero_is_the_same_as_none(self) -> None:
        for kind in (*LIFETIME_POINTS, IntervalCensoredLifetime):
            with self.subTest(kind=kind.__name__):
                observation = build(kind, 5.0, truncation_lower=0, truncation_upper=9)
                self.assertIsNone(observation.truncation_lower)
                self.assertEqual(observation, build(kind, 5.0, truncation_upper=9.0))
        self.assertEqual(ExactValue(0.0, truncation_lower=0.0).truncation_lower, 0.0)

    def test_windows_that_differ_make_observations_that_differ(self) -> None:
        for kind in ALL_TYPES:
            with self.subTest(kind=kind.__name__):
                a = build(kind, 5.0, truncation_lower=2.0)
                b = build(kind, 5.0, truncation_lower=3.0)
                c = build(kind, 5.0, truncation_upper=9.0)
                self.assertNotEqual(a, b)
                self.assertNotEqual(a, c)
                self.assertNotEqual(a, build(kind, 5.0))
                self.assertEqual(len({a, b, c, build(kind, 5.0)}), 4)
                self.assertEqual(a, build(kind, 5.0, truncation_lower=2))

    def test_an_end_must_be_a_real_number_or_none(self) -> None:
        for kind in (ExactLifetime, ExactValue):
            for name in ("truncation_lower", "truncation_upper"):
                for bad in ("1.0", True, 1 + 2j, [1.0]):
                    with self.subTest(kind=kind.__name__, name=name, bad=bad):
                        self.refuse(
                            TypeError,
                            f"{name} must be a finite real number or None",
                            kind,
                            5.0,
                            **{name: bad},
                        )

    def test_an_end_must_be_finite_or_the_open_infinity_of_its_side(self) -> None:
        for kind in (ExactLifetime, ExactValue):
            cases = (
                ("truncation_lower", nan),
                ("truncation_lower", inf),
                ("truncation_lower", HUGE_INTEGER),
                ("truncation_upper", nan),
                ("truncation_upper", -inf),
                ("truncation_upper", -HUGE_INTEGER),
            )
            for name, bad in cases:
                with self.subTest(kind=kind.__name__, name=name, bad=bad):
                    self.refuse(
                        ValueError, f"{name} must be finite or None", kind, 5.0, **{name: bad}
                    )
        self.refuse(
            ValueError,
            "nonzero truncation_lower must remain representable",
            ExactValue,
            5.0,
            truncation_lower=Decimal("1e-400"),
        )

    def test_a_lifetime_window_is_within_the_non_negative_half_line(self) -> None:
        for kind in (ExactLifetime, IntervalCensoredLifetime):
            self.refuse(
                ValueError,
                "truncation_lower must be non-negative",
                build,
                kind,
                5.0,
                truncation_lower=-1.0,
            )
            for bad in (0.0, -1.0):
                self.refuse(
                    ValueError,
                    "truncation_upper must be positive",
                    build,
                    kind,
                    5.0,
                    truncation_upper=bad,
                )
        self.assertEqual(ExactValue(-5.0, truncation_lower=-9.0).truncation_lower, -9.0)
        self.assertEqual(ExactValue(-5.0, truncation_upper=-1.0).truncation_upper, -1.0)

    def test_a_window_is_ordered_and_not_degenerate(self) -> None:
        message = "truncation_lower must be less than truncation_upper"
        for kind in (ExactLifetime, ExactValue):
            for lower, upper in ((5.0, 5.0), (6.0, 5.0)):
                with self.subTest(kind=kind.__name__, lower=lower, upper=upper):
                    self.refuse(
                        ValueError,
                        message,
                        kind,
                        5.0,
                        truncation_lower=lower,
                        truncation_upper=upper,
                    )


class ConsistencyTests(MessageTestCase):
    """An observation outside its window is refused, never clipped."""

    def test_an_exact_observation_lies_in_the_closed_window(self) -> None:
        for kind in (ExactLifetime, ExactValue):
            before = "truncation_lower must not be after the exact observation"
            after = "truncation_upper must not be before the exact observation"
            self.refuse(ValueError, before, kind, 5.0, truncation_lower=5.5)
            self.refuse(ValueError, after, kind, 5.0, truncation_upper=4.5)
            self.refuse(ValueError, before, kind, 5.0, truncation_lower=5.5, truncation_upper=9.0)
            self.refuse(ValueError, after, kind, 5.0, truncation_lower=1.0, truncation_upper=4.5)
            self.assertEqual(kind(5.0, truncation_lower=5.0).truncation_lower, 5.0)
            self.assertEqual(kind(5.0, truncation_upper=5.0).truncation_upper, 5.0)
            self.assertEqual(
                kind(5.0, truncation_lower=5.0, truncation_upper=5.5).truncation_upper, 5.5
            )

    def test_a_right_censoring_point_needs_a_at_most_c_below_b(self) -> None:
        for kind in (RightCensoredLifetime, RightCensoredValue):
            before = "truncation_lower must not be after the censoring point"
            after = "truncation_upper must be after the censoring point"
            self.refuse(ValueError, before, kind, 5.0, truncation_lower=5.5)
            self.refuse(ValueError, after, kind, 5.0, truncation_upper=5.0)
            self.refuse(ValueError, after, kind, 5.0, truncation_upper=4.0)
            self.assertEqual(kind(5.0, truncation_lower=5.0).truncation_lower, 5.0)
            self.assertEqual(kind(5.0, truncation_upper=5.000001).truncation_upper, 5.000001)

    def test_a_left_censoring_point_needs_a_below_u_at_most_b(self) -> None:
        for kind in (LeftCensoredLifetime, LeftCensoredValue):
            before = "truncation_lower must be before the censoring point"
            after = "truncation_upper must not be before the censoring point"
            self.refuse(ValueError, before, kind, 5.0, truncation_lower=5.0)
            self.refuse(ValueError, before, kind, 5.0, truncation_lower=6.0)
            self.refuse(ValueError, after, kind, 5.0, truncation_upper=4.0)
            self.assertEqual(kind(5.0, truncation_lower=4.999999).truncation_lower, 4.999999)
            self.assertEqual(kind(5.0, truncation_upper=5.0).truncation_upper, 5.0)

    def test_an_interval_needs_a_at_most_lower_and_upper_at_most_b(self) -> None:
        for kind in INTERVAL_TYPES:
            before = "truncation_lower must not be after the lower bound"
            after = "truncation_upper must not be before the upper bound"
            self.refuse(ValueError, before, kind, 5.0, 6.0, truncation_lower=5.5)
            self.refuse(ValueError, after, kind, 5.0, 6.0, truncation_upper=5.5)
            self.assertEqual(kind(5.0, 6.0, truncation_lower=5.0).truncation_lower, 5.0)
            self.assertEqual(kind(5.0, 6.0, truncation_upper=6.0).truncation_upper, 6.0)

    def test_the_window_is_judged_after_the_value_checks(self) -> None:
        self.refuse(
            ValueError,
            "time must be finite and non-negative",
            ExactLifetime,
            -1.0,
            truncation_lower=0.5,
        )
        self.refuse(ValueError, "value must be finite", ExactValue, nan, truncation_upper=1.0)
        self.refuse(
            ValueError,
            "upper must be greater than lower",
            IntervalCensoredValue,
            2.0,
            1.0,
            truncation_lower=0.0,
        )


class StateTests(unittest.TestCase):
    def test_a_truncated_observation_shows_its_window_in_repr(self) -> None:
        self.assertEqual(
            repr(ExactLifetime(5.0, truncation_upper=9.0)),
            "ExactLifetime(time=5.0, truncation_upper=9.0)",
        )
        self.assertEqual(
            repr(IntervalCensoredValue(1.0, 2.0, truncation_lower=0.5, truncation_upper=3.0)),
            "IntervalCensoredValue(lower=1.0, upper=2.0, truncation_lower=0.5, "
            "truncation_upper=3.0)",
        )
        self.assertEqual(
            repr(LeftCensoredLifetime(5.0, truncation_lower=2.0)),
            "LeftCensoredLifetime(time=5.0, truncation_lower=2.0)",
        )
        self.assertEqual(
            repr(IntervalCensoredLifetime(1, 2)), "IntervalCensoredLifetime(lower=1.0, upper=2.0)"
        )

    def test_every_type_survives_pickle_and_copy_with_and_without_a_window(self) -> None:
        import copy

        windows: tuple[dict[str, float], ...] = (
            {},
            {"truncation_lower": 0.5},
            {"truncation_upper": 99.0},
            {"truncation_lower": 0.5, "truncation_upper": 99.0},
        )
        for kind in ALL_TYPES:
            for window in windows:
                observation = build(kind, 5.0, **window)
                with self.subTest(kind=kind.__name__, window=window):
                    for protocol in range(2, pickle.HIGHEST_PROTOCOL + 1):
                        loaded = pickle.loads(pickle.dumps(observation, protocol=protocol))
                        self.assertEqual(loaded, observation)
                        self.assertEqual(hash(loaded), hash(observation))
                        self.assertEqual(repr(loaded), repr(observation))
                    self.assertEqual(copy.deepcopy(observation), observation)
                    self.assertEqual(copy.copy(observation), observation)

    def test_the_untruncated_state_has_no_truncation_entries(self) -> None:
        for kind in ALL_TYPES:
            plain = build(kind, 5.0)
            truncated = build(kind, 5.0, truncation_upper=99.0)
            members = len(dataclasses.fields(plain))
            self.assertEqual(members, 4 if kind in INTERVAL_TYPES else 3)
            self.assertEqual(len(plain.__getstate__()), members - 2)
            self.assertEqual(len(truncated.__getstate__()), members)

    def test_the_new_types_are_exported_where_the_old_ones_are(self) -> None:
        import veridist
        from veridist import domain

        for kind in ALL_TYPES:
            self.assertIn(kind.__name__, domain.__all__)
            self.assertIn(kind.__name__, veridist.__all__)
            self.assertIs(getattr(veridist, kind.__name__), kind)


if __name__ == "__main__":
    unittest.main()
