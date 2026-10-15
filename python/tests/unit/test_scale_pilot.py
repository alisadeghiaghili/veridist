"""The first-pass states: exact sum, extremes, merge, canonical bytes and refusals."""

from __future__ import annotations

import struct
import unittest
from fractions import Fraction

import numpy as np

from veridist.scale._accumulator import MAX_ELEMENT_UNITS
from veridist.scale._codec import encode_frame
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._pilot import GumbelPilotState, WeibullPilotState
from veridist.scale._sources import NumpySource, fold
from veridist.scale._state import MergeableState


def batch(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float64)


def bits(value: float) -> int:
    return int(struct.unpack("<q", struct.pack("<d", value))[0])


class GumbelPilotTests(unittest.TestCase):
    def test_count_exact_sum_and_extremes(self) -> None:
        state = GumbelPilotState.empty().update(batch([0.1, -3.5, 2.0, 0.2]))
        self.assertEqual(state.count, 4)
        self.assertEqual((state.minimum, state.maximum), (-3.5, 2.0))
        statistics = state.finalize()
        exact = sum((Fraction(v) for v in (0.1, -3.5, 2.0, 0.2)), Fraction(0))
        self.assertEqual(statistics.total.exact, exact)
        self.assertEqual(statistics.total.rounded, float(exact))
        self.assertEqual((statistics.count, statistics.minimum, statistics.maximum), (4, -3.5, 2.0))

    def test_the_identity_and_the_protocol(self) -> None:
        empty = GumbelPilotState.empty()
        self.assertEqual(
            (empty.count, empty.total_units, empty.minimum, empty.maximum), (0, 0, None, None)
        )
        self.assertIsInstance(empty, MergeableState)
        state = empty.update(batch([1.0, 2.0]))
        self.assertEqual(state.merge(empty), state)
        self.assertEqual(empty.merge(state), state)
        self.assertEqual(empty.update(batch([])), empty)
        self.assertEqual(empty.finalize().minimum, None)

    def test_merge_is_exact_commutative_and_independent_of_the_split(self) -> None:
        values = [0.1, 0.7, -2.25, 5.0, 3.5, -0.0, 0.3]
        whole = GumbelPilotState.empty().update(batch(values))
        left = GumbelPilotState.empty().update(batch(values[:3]))
        right = GumbelPilotState.empty().update(batch(values[3:]))
        self.assertEqual(left.merge(right), whole)
        self.assertEqual(right.merge(left), whole)
        self.assertEqual(left.merge(right).to_bytes(), whole.to_bytes())

    def test_negative_zero_is_stored_as_zero(self) -> None:
        state = GumbelPilotState.empty().update(batch([-0.0, 0.0]))
        self.assertEqual(bits(state.minimum or 0.0), 0)
        self.assertEqual(bits(state.maximum or 0.0), 0)
        one = GumbelPilotState.empty().update(batch([0.0, -0.0]))
        self.assertEqual(one.to_bytes(), state.to_bytes())

    def test_a_non_finite_value_is_refused_and_a_wrong_batch_is_a_type_error(self) -> None:
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad), self.assertRaises(ScaleStateError) as caught:
                GumbelPilotState.empty().update(batch([1.0, bad]))
            self.assertIs(caught.exception.code, ScaleStateErrorCode.NON_FINITE_VALUE)
        with self.assertRaises(TypeError):
            GumbelPilotState.empty().update(np.array([1, 2]))
        with self.assertRaises(ValueError):
            GumbelPilotState.empty().update(np.zeros((2, 2)))

    def test_the_observation_limit_is_enforced_before_the_batch_is_read(self) -> None:
        full = GumbelPilotState(2**64 - 1, 0, -1.0, 1.0)
        with self.assertRaises(ScaleStateError) as caught:
            full.update(batch([1.0]))
        self.assertIs(caught.exception.code, ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        with self.assertRaises(ScaleStateError) as merged:
            full.merge(GumbelPilotState(1, 0, 0.0, 0.0))
        self.assertIs(merged.exception.code, ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)

    def test_states_of_other_types_do_not_merge(self) -> None:
        a = GumbelPilotState.empty().update(batch([1.0, 2.0]))
        b = WeibullPilotState.empty().update(batch([1.0, 2.0]))
        with self.assertRaises(ScaleStateError) as caught:
            a.merge(b)  # type: ignore[arg-type]
        self.assertIs(caught.exception.code, ScaleStateErrorCode.INCOMPATIBLE_STATE)
        with self.assertRaises(TypeError):
            a.merge("state")  # type: ignore[arg-type]

    def test_fold_over_a_source_is_the_state_of_all_values(self) -> None:
        values = np.random.default_rng(4).normal(3.0, 2.0, 1000)
        state = fold(NumpySource(values), GumbelPilotState, max_rows=77)
        self.assertEqual(state, GumbelPilotState.empty().update(values))
        self.assertEqual(state.minimum, float(values.min()))
        self.assertEqual(state.maximum, float(values.max()))


class WeibullPilotTests(unittest.TestCase):
    def test_the_sum_is_that_of_the_platform_logarithms(self) -> None:
        values = batch([0.5, 2.0, 7.0, 1e-300, 1e300])
        state = WeibullPilotState.empty().update(values)
        logs = np.log(values)
        exact = sum((Fraction(float(v)) for v in logs), Fraction(0))
        self.assertEqual(state.finalize().total.exact, exact)
        self.assertEqual((state.minimum, state.maximum), (1e-300, 1e300))

    def test_a_value_that_is_not_positive_adds_nothing_and_shows_in_the_minimum(self) -> None:
        for bad in (0.0, -1.0):
            state = WeibullPilotState.empty().update(batch([2.0, bad, 3.0]))
            self.assertEqual(state.count, 3)
            self.assertEqual(state.total_units, 0)
            self.assertEqual(state.minimum, min(bad, 2.0))
        merged = (
            WeibullPilotState.empty()
            .update(batch([2.0, 3.0]))
            .merge(WeibullPilotState.empty().update(batch([-4.0])))
        )
        self.assertEqual((merged.count, merged.minimum), (3, -4.0))

    def test_the_states_have_their_own_tags(self) -> None:
        self.assertEqual(WeibullPilotState.STATE_TAG, "scale.weibull_pilot")
        self.assertEqual(GumbelPilotState.STATE_TAG, "scale.gumbel_pilot")
        self.assertEqual(WeibullPilotState.SCHEMA_VERSION, 1)
        self.assertEqual(GumbelPilotState.SCHEMA_VERSION, 1)


class SerializationTests(unittest.TestCase):
    def test_round_trip_and_canonical_bytes(self) -> None:
        for state_type in (GumbelPilotState, WeibullPilotState):
            state = state_type.empty().update(batch([0.5, 4.0, 1.5]))
            data = state.to_bytes()
            with self.subTest(state=state_type.__name__):
                self.assertEqual(state_type.from_bytes(data), state)
                self.assertEqual(state_type.from_bytes(data).to_bytes(), data)
                self.assertEqual(
                    state_type.from_bytes(state_type.empty().to_bytes()), state_type.empty()
                )

    def test_the_frame_holds_the_count_the_sum_and_the_extremes_bit_patterns(self) -> None:
        state = GumbelPilotState.empty().update(batch([1.5, -2.0]))
        expected = encode_frame(
            "scale.gumbel_pilot", 1, 2, (state.total_units, bits(-2.0), bits(1.5))
        )
        self.assertEqual(state.to_bytes(), expected)
        self.assertEqual(
            GumbelPilotState.empty().to_bytes(), encode_frame("scale.gumbel_pilot", 1, 0, (0, 0, 0))
        )

    def test_a_frame_of_another_tag_or_version_is_refused(self) -> None:
        data = WeibullPilotState.empty().update(batch([1.0, 2.0])).to_bytes()
        with self.assertRaises(ScaleStateError) as tag:
            GumbelPilotState.from_bytes(data)
        self.assertIs(tag.exception.code, ScaleStateErrorCode.STATE_TAG_MISMATCH)
        old = encode_frame("scale.gumbel_pilot", 2, 0, (0, 0, 0))
        with self.assertRaises(ScaleStateError) as version:
            GumbelPilotState.from_bytes(old)
        self.assertIs(version.exception.code, ScaleStateErrorCode.STATE_VERSION_UNSUPPORTED)
        with self.assertRaises(TypeError):
            GumbelPilotState.from_bytes("text")

    def test_invalid_frames_are_refused(self) -> None:
        cases = {
            "empty with a sum": (0, (5, 0, 0)),
            "empty with an extreme": (0, (0, bits(1.0), 0)),
            "empty with a high extreme": (0, (0, 0, bits(1.0))),
            "minimum above maximum": (2, (0, bits(2.0), bits(1.0))),
            "not finite": (1, (0, bits(float("inf")), bits(float("inf")))),
            "nan": (1, (0, bits(float("nan")), bits(float("nan")))),
            "negative zero": (1, (0, bits(-0.0), bits(1.0))),
            "outside 64 bits": (1, (0, 1 << 63, 0)),
            "outside 64 bits low": (1, (0, 0, -(1 << 63) - 1)),
            "sum beyond the count": (1, (MAX_ELEMENT_UNITS + 1, bits(1.0), bits(1.0))),
        }
        for name, (count, integers) in cases.items():
            frame = encode_frame("scale.gumbel_pilot", 1, count, integers)
            with self.subTest(case=name), self.assertRaises(ScaleStateError) as caught:
                GumbelPilotState.from_bytes(frame)
            self.assertIs(caught.exception.code, ScaleStateErrorCode.STATE_BYTES_INVALID)

    def test_the_constructor_validates(self) -> None:
        with self.assertRaises(TypeError):
            GumbelPilotState(1.0, 0, 0.0, 0.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            GumbelPilotState(1, 0.0, 0.0, 0.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            GumbelPilotState(1, 0, 0, 0.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            GumbelPilotState(-1, 0, None, None)
        with self.assertRaises(ValueError):
            GumbelPilotState(2**64, 0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            GumbelPilotState(1, 0, 1.0, 0.0)
        with self.assertRaises(ValueError):
            GumbelPilotState(0, 1, None, None)
        with self.assertRaises(ValueError):
            GumbelPilotState(0, 0, 1.0, None)


if __name__ == "__main__":
    unittest.main()
