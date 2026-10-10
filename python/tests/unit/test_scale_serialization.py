"""Canonical bytes, round trips and strict rejection for exact states."""

from __future__ import annotations

import unittest
from collections.abc import Callable

import numpy as np

from tests.unit.scale_oracle import (
    MAX_FLOAT,
    MAX_MANTISSA,
    adversarial_sets,
    as_array,
    frame_body,
    seal,
)
from veridist.scale import _accumulator as acc
from veridist.scale import _codec
from veridist.scale._accumulator import AccumulatorLimits, ExactSum
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._one_pass import (
    ExponentialState,
    GammaState,
    LognormalState,
    NormalState,
)
from veridist.statistics.log_likelihood import MAX_OBSERVATION_COUNT

TAG = b"scale.exact_sum"


def code_of(data: object, loader: Callable[[object], object] = ExactSum.from_bytes) -> object:
    try:
        loader(data)
    except ScaleStateError as error:
        return error.code
    raise AssertionError("expected a ScaleStateError")


class CodecLayoutTests(unittest.TestCase):
    def test_layout_is_exactly_as_documented(self) -> None:
        state = ExactSum()
        state.update(as_array([1.0, 2.5]))
        units = 7 << 1073
        magnitude = units.to_bytes(135, "big")
        expected = seal(
            b"VDSS"
            + b"\x0f"
            + TAG
            + b"\x00\x01"
            + (2).to_bytes(8, "big")
            + b"\x01"
            + b"\x00"
            + (135).to_bytes(4, "big")
            + magnitude
        )
        self.assertEqual(state.to_bytes(), expected)
        self.assertEqual(
            expected[-32:].hex(), "fd678b28732b8ca3166f358c1689176788ff42d1634faa4079949c6168d8ce0d"
        )

    def test_empty_state_frames_are_frozen(self) -> None:
        self.assertEqual(
            NormalState.empty().to_bytes().hex(),
            "56445353"  # magic
            "0c7363616c652e6e6f726d616c"  # tag length and tag "scale.normal"
            "0001"  # schema version
            "0000000000000000"  # observation count
            "02"  # two integers
            "00"
            "00000000"  # sum x: sign and empty magnitude
            "00"
            "00000000"  # sum x*x: sign and empty magnitude
            "c1ee2f84cecd965ef6425f9716fd1cc5c85ef1617d8cf099f75f546d5dbaf4ad",  # SHA-256
        )

    def test_negative_sum_uses_the_sign_byte_and_a_positive_magnitude(self) -> None:
        state = ExactSum()
        state.update(as_array([-1.0]))
        body = frame_body(count=1, integers=((1, (1 << 1074).to_bytes(135, "big")),))
        self.assertEqual(state.to_bytes(), seal(body))
        self.assertEqual(ExactSum.from_bytes(seal(body)).total(), -1.0)

    def test_constants(self) -> None:
        self.assertEqual(_codec.MAGIC, b"VDSS")
        self.assertEqual(_codec.CHECKSUM_BYTES, 32)

    def test_encode_decode_round_trip_of_arbitrary_integers(self) -> None:
        for integers in ((), (0,), (1,), (-1,), (255,), (256,), (-(2**2000), 2**64, 0)):
            with self.subTest(integers=integers):
                frame = _codec.encode_frame("t.x", 3, 12345, integers)
                self.assertEqual(
                    _codec.decode_frame(frame, tag="t.x", version=3, integer_count=len(integers)),
                    (12345, integers),
                )


class RoundTripTests(unittest.TestCase):
    def test_exact_sum_round_trip_preserves_everything_observable(self) -> None:
        for name, values in adversarial_sets().items():
            with self.subTest(name=name):
                state = ExactSum()
                state.update(values)
                data = state.to_bytes()
                restored = ExactSum.from_bytes(data)
                self.assertEqual(restored, state)
                self.assertEqual(restored.count, state.count)
                self.assertEqual(restored.total_units(), state.total_units())
                self.assertEqual(restored.to_bytes(), data)
                self.assertIsInstance(data, bytes)

    def test_bytearray_is_accepted_and_limits_are_applied(self) -> None:
        state = ExactSum()
        state.update(as_array([1.0, 2.0]))
        small = AccumulatorLimits(4, 10, 50)
        restored = ExactSum.from_bytes(bytearray(state.to_bytes()), small)
        self.assertEqual(restored.limits, small)
        self.assertEqual(restored, state)
        restored.update(as_array([3.0]))
        self.assertEqual(restored.total(), 6.0)

    def test_equal_states_have_equal_bytes_whatever_the_history(self) -> None:
        values = adversarial_sets()["mixed_magnitude_sign_1"][:300]
        reference = ExactSum()
        reference.update(values)
        histories = []
        for limits in (
            AccumulatorLimits(1, 2, 1),
            AccumulatorLimits(7, 20, 30),
            acc.DEFAULT_LIMITS,
        ):
            state = ExactSum(limits)
            state.update(values[::-1])
            histories.append(state)
        halves = ExactSum().merge(ExactSum())
        for chunk in np.array_split(values, 5):
            piece = ExactSum()
            piece.update(chunk)
            halves = halves.merge(piece)
        histories.append(halves)
        histories.append(ExactSum.from_bytes(reference.to_bytes()))
        for state in histories:
            self.assertEqual(state.to_bytes(), reference.to_bytes())

    def test_states_with_the_same_count_and_exact_sum_are_equal(self) -> None:
        left, right = ExactSum(), ExactSum()
        left.update(as_array([1.0, 3.0]))
        right.update(as_array([2.0, 2.0]))
        self.assertEqual(left.to_bytes(), right.to_bytes())
        self.assertEqual(left, right)

    def test_different_states_have_different_bytes(self) -> None:
        base = ExactSum()
        base.update(as_array([1.0, 2.0]))
        distinct = []
        for values in ([1.0, 2.0, 0.0], [1.0, 2.0000000000000004], [-1.0, -2.0], [1.0], []):
            other = ExactSum()
            other.update(as_array(values))
            distinct.append(other.to_bytes())
        self.assertEqual(len(set(distinct + [base.to_bytes()])), 6)

    def test_family_states_round_trip(self) -> None:
        rng = np.random.default_rng(1)
        positive = np.exp(rng.standard_normal(60) * 5.0)
        cases = (
            (ExponentialState, positive),
            (NormalState, rng.standard_normal(60) * 1e5),
            (GammaState, positive),
            (LognormalState, positive),
        )
        for cls, values in cases:
            with self.subTest(state=cls.__name__):
                state = cls.empty().update(values)
                data = state.to_bytes()
                self.assertEqual(cls.from_bytes(data), state)
                self.assertEqual(cls.from_bytes(data).to_bytes(), data)
                self.assertEqual(cls.from_bytes(cls.empty().to_bytes()), cls.empty())

    def test_negative_sums_round_trip(self) -> None:
        state = NormalState.empty().update(as_array([-5.0, -MAX_MANTISSA, 1e-300]))
        self.assertTrue(state.sums[0] < 0)
        self.assertEqual(NormalState.from_bytes(state.to_bytes()), state)


class RejectionTests(unittest.TestCase):
    def setUp(self) -> None:
        state = ExactSum()
        state.update(as_array([1.0, 2.5]))
        self.good = state.to_bytes()

    def test_non_bytes_is_a_type_error(self) -> None:
        for bad in ("text", 5, None, [1, 2], memoryview(b"abc"), 1.5):
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(TypeError):
                    ExactSum.from_bytes(bad)

    def test_every_truncation_is_rejected(self) -> None:
        for length in range(len(self.good)):
            with self.subTest(length=length):
                self.assertRaises(ScaleStateError, ExactSum.from_bytes, self.good[:length])

    def test_every_single_byte_change_is_rejected(self) -> None:
        for position in range(len(self.good)):
            altered = bytearray(self.good)
            altered[position] ^= 0x01
            with self.subTest(position=position):
                self.assertRaises(ScaleStateError, ExactSum.from_bytes, bytes(altered))

    def test_trailing_and_leading_bytes_are_rejected(self) -> None:
        for data in (self.good + b"\x00", b"\x00" + self.good, self.good + self.good):
            with self.subTest(length=len(data)):
                self.assertEqual(code_of(data), ScaleStateErrorCode.STATE_BYTES_INVALID)

    def test_empty_and_garbage(self) -> None:
        for data in (b"", b"VDSS", b"\x00" * 200, b"x" * 3):
            with self.subTest(data=data[:8]):
                self.assertEqual(code_of(data), ScaleStateErrorCode.STATE_BYTES_INVALID)

    def test_wrong_magic_with_a_valid_checksum(self) -> None:
        self.assertEqual(
            code_of(seal(frame_body(magic=b"VDSX"))), ScaleStateErrorCode.STATE_BYTES_INVALID
        )

    def test_checksum_mismatch(self) -> None:
        altered = bytearray(self.good)
        altered[-1] ^= 0xFF
        self.assertEqual(code_of(bytes(altered)), ScaleStateErrorCode.STATE_BYTES_INVALID)

    def test_unknown_version_is_rejected_without_migration(self) -> None:
        for version in (0, 2, 255, 256, 65535):
            with self.subTest(version=version):
                self.assertEqual(
                    code_of(seal(frame_body(version=version))),
                    ScaleStateErrorCode.STATE_VERSION_UNSUPPORTED,
                )

    def test_other_state_type_is_a_tag_mismatch(self) -> None:
        for tag in (b"scale.normal", b"scale.exact_sux", b"scale.exact_sum2", b"", b"x"):
            with self.subTest(tag=tag):
                self.assertEqual(
                    code_of(seal(frame_body(tag=tag))), ScaleStateErrorCode.STATE_TAG_MISMATCH
                )
        gamma = GammaState.empty().update(as_array([1.0, 2.0])).to_bytes()
        for loader in (
            NormalState.from_bytes,
            ExponentialState.from_bytes,
            LognormalState.from_bytes,
        ):
            with self.subTest(loader=loader.__qualname__):
                self.assertEqual(code_of(gamma, loader), ScaleStateErrorCode.STATE_TAG_MISMATCH)
        self.assertEqual(code_of(gamma), ScaleStateErrorCode.STATE_TAG_MISMATCH)
        self.assertEqual(
            code_of(self.good, GammaState.from_bytes), ScaleStateErrorCode.STATE_TAG_MISMATCH
        )

    def test_structural_defects_with_a_valid_checksum(self) -> None:
        overrun = frame_body(integers=(), declared=1) + b"\x00" + (5).to_bytes(4, "big") + b"\x07"
        defects = {
            "wrong integer count": frame_body(declared=2),
            "zero integers declared": frame_body(declared=0),
            "sign above one": frame_body(integers=((2, b"\x07"),)),
            "negative zero": frame_body(integers=((1, b""),)),
            "leading zero byte": frame_body(integers=((0, b"\x00\x07"),)),
            "zero as one byte": frame_body(integers=((0, b"\x00"),)),
            "magnitude overruns frame": overrun,
            "trailing byte in frame": frame_body(extra=b"\x00"),
            "declared integer missing": frame_body(integers=(), declared=1),
        }
        for name, body in defects.items():
            with self.subTest(name=name):
                self.assertEqual(code_of(seal(body)), ScaleStateErrorCode.STATE_BYTES_INVALID)

    def test_huge_declared_length_does_not_allocate_or_pass(self) -> None:
        body = frame_body(integers=())
        body = body[:-1] + b"\x01" + b"\x00" + (2**32 - 1).to_bytes(4, "big") + b"\x07"
        self.assertEqual(code_of(seal(body)), ScaleStateErrorCode.STATE_BYTES_INVALID)

    def test_out_of_range_values_are_rejected(self) -> None:
        maximum = acc.MAX_ELEMENT_UNITS

        def unit_frame(count: int, units: int) -> bytes:
            magnitude = abs(units).to_bytes((abs(units).bit_length() + 7) // 8, "big")
            return seal(frame_body(count=count, integers=((int(units < 0), magnitude),)))

        self.assertEqual(ExactSum.from_bytes(unit_frame(1, maximum)).total(), MAX_FLOAT)
        self.assertEqual(ExactSum.from_bytes(unit_frame(1, -maximum)).total(), -MAX_FLOAT)
        self.assertEqual(
            ExactSum.from_bytes(unit_frame(MAX_OBSERVATION_COUNT, 0)).count, MAX_OBSERVATION_COUNT
        )
        for count, units in (
            (1, maximum + 1),
            (1, -maximum - 1),
            (0, 1),
            (0, -1),
            (3, 3 * maximum + 1),
        ):
            with self.subTest(count=count, units=units):
                self.assertEqual(
                    code_of(unit_frame(count, units)), ScaleStateErrorCode.STATE_BYTES_INVALID
                )

    def test_family_state_bounds_are_checked_per_sum(self) -> None:
        maximum = acc.MAX_ELEMENT_UNITS

        def pair(count: int, first: int, second: int) -> bytes:
            def part(value: int) -> tuple[int, bytes]:
                return int(value < 0), abs(value).to_bytes(
                    (abs(value).bit_length() + 7) // 8, "big"
                )

            return seal(
                frame_body(tag=b"scale.normal", count=count, integers=(part(first), part(second)))
            )

        self.assertEqual(NormalState.from_bytes(pair(2, 2 * maximum, -2 * maximum)).count, 2)
        for first, second in ((2 * maximum + 1, 0), (0, -2 * maximum - 1)):
            with self.subTest(first=first, second=second):
                self.assertEqual(
                    code_of(pair(2, first, second), NormalState.from_bytes),
                    ScaleStateErrorCode.STATE_BYTES_INVALID,
                )

    def test_error_text_carries_no_content(self) -> None:
        with self.assertRaises(ScaleStateError) as caught:
            ExactSum.from_bytes(self.good[:-5])
        self.assertEqual(str(caught.exception), "STATE_BYTES_INVALID")


if __name__ == "__main__":
    unittest.main()
