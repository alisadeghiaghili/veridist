"""Contract of the mergeable exact states: errors, finalization, bounds and typing."""

from __future__ import annotations

import math
import pickle
import unittest
from dataclasses import FrozenInstanceError, dataclass
from fractions import Fraction
from typing import ClassVar

import numpy as np

from tests.unit.scale_oracle import (
    MAX_FLOAT,
    STATES,
    batch,
    exact_fraction,
    exact_square_units,
    expected_sums,
    frame_body,
    seal,
)
from veridist.engine.errors import VeridistError
from veridist.scale import _kernels
from veridist.scale._accumulator import MAX_ELEMENT_UNITS
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._one_pass import (
    ExactStatistic,
    ExponentialState,
    GammaState,
    LognormalState,
    NormalState,
)
from veridist.scale._state import MergeableState
from veridist.statistics.log_likelihood import MAX_OBSERVATION_COUNT

CODES = ScaleStateErrorCode


@dataclass(frozen=True, slots=True)
class _OtherTag(ExponentialState):
    STATE_TAG: ClassVar[str] = "scale.other"


@dataclass(frozen=True, slots=True)
class _NextVersion(ExponentialState):
    SCHEMA_VERSION: ClassVar[int] = 2


def refused(case: unittest.TestCase, code: ScaleStateErrorCode, call: object) -> ScaleStateError:
    with case.assertRaises(ScaleStateError) as caught:
        call()  # type: ignore[operator]
    case.assertEqual(caught.exception.code, code)
    return caught.exception


class ErrorTypeTests(unittest.TestCase):
    def test_hierarchy_and_text(self) -> None:
        error = ScaleStateError(CODES.NON_FINITE_VALUE)
        self.assertIsInstance(error, VeridistError)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(str(error), "NON_FINITE_VALUE")
        self.assertIs(error.code, CODES.NON_FINITE_VALUE)
        self.assertIs(ScaleStateError("STATE_BYTES_INVALID").code, CODES.STATE_BYTES_INVALID)
        with self.assertRaises(ValueError):
            ScaleStateError("NOT_A_CODE")

    def test_codes_are_a_closed_documented_set(self) -> None:
        self.assertEqual(
            {code.value for code in CODES},
            {
                "NON_FINITE_VALUE",
                "VALUE_OUT_OF_SUPPORT",
                "OBSERVATION_LIMIT_EXCEEDED",
                "TOTAL_NOT_REPRESENTABLE",
                "INCOMPATIBLE_STATE",
                "STATE_BYTES_INVALID",
                "STATE_TAG_MISMATCH",
                "STATE_VERSION_UNSUPPORTED",
            },
        )
        for code in CODES:
            self.assertEqual(code.value, code.name)

    def test_the_error_survives_pickling(self) -> None:
        for code in CODES:
            restored = pickle.loads(pickle.dumps(ScaleStateError(code)))
            self.assertIs(type(restored), ScaleStateError)
            self.assertIs(restored.code, code)


class IdentityAndTypingTests(unittest.TestCase):
    def test_tags_versions_and_sum_counts(self) -> None:
        expected = {
            ExponentialState: ("scale.exponential", 1, 1),
            NormalState: ("scale.normal", 2, 2),
            GammaState: ("scale.gamma", 1, 2),
            LognormalState: ("scale.lognormal", 2, 2),
        }
        for cls, (tag, version, sums) in expected.items():
            with self.subTest(state=cls.__name__):
                self.assertEqual(
                    (cls.STATE_TAG, cls.SCHEMA_VERSION, cls.SUM_COUNT), (tag, version, sums)
                )
        self.assertEqual(len({cls.STATE_TAG for cls in STATES}), 4)

    def test_every_state_satisfies_the_contract_protocol(self) -> None:
        for cls in STATES:
            with self.subTest(state=cls.__name__):
                self.assertIsInstance(cls.empty(), MergeableState)
        self.assertNotIsInstance(object(), MergeableState)
        self.assertNotIsInstance(5, MergeableState)

    def test_empty_is_the_identity_and_has_no_observation(self) -> None:
        rng = np.random.default_rng(1)
        for cls in STATES:
            with self.subTest(state=cls.__name__):
                empty = cls.empty()
                self.assertEqual((empty.count, empty.sums), (0, (0,) * cls.SUM_COUNT))
                state = empty.update(batch(cls, rng, 17))
                self.assertEqual(state.merge(empty), state)
                self.assertEqual(empty.merge(state), state)
                self.assertEqual(empty.merge(empty), empty)
                self.assertEqual(empty.update(np.empty(0)), empty)

    def test_states_are_immutable_hashable_value_objects(self) -> None:
        rng = np.random.default_rng(2)
        for cls in STATES:
            with self.subTest(state=cls.__name__):
                state = cls.empty().update(batch(cls, rng, 9))
                with self.assertRaises(FrozenInstanceError):
                    state.count = 1  # type: ignore[misc]
                with self.assertRaises(AttributeError):
                    state.__dict__  # noqa: B018
                self.assertEqual(hash(state), hash(cls(state.count, state.sums)))
                self.assertEqual(len({state, cls(state.count, state.sums)}), 1)
                self.assertNotIn("sums", repr(state))
                self.assertIn(f"count={state.count}", repr(state))

    def test_states_of_different_types_are_never_equal(self) -> None:
        self.assertNotEqual(GammaState.empty(), LognormalState.empty())
        self.assertNotEqual(ExponentialState.empty(), _OtherTag.empty())
        self.assertNotEqual(NormalState.empty(), (0, (0, 0)))

    def test_results_have_the_state_type(self) -> None:
        rng = np.random.default_rng(3)
        for cls in STATES:
            with self.subTest(state=cls.__name__):
                state = cls.empty().update(batch(cls, rng, 5))
                self.assertIs(type(state), cls)
                self.assertIs(type(state.merge(state)), cls)
                self.assertIs(type(cls.from_bytes(state.to_bytes())), cls)


class ConstructionTests(unittest.TestCase):
    def test_boundaries_are_accepted(self) -> None:
        for cls in STATES:
            with self.subTest(state=cls.__name__):
                cls(
                    MAX_OBSERVATION_COUNT,
                    (MAX_OBSERVATION_COUNT * MAX_ELEMENT_UNITS,) * cls.SUM_COUNT,
                )
                cls(1, (-MAX_ELEMENT_UNITS,) * cls.SUM_COUNT)
                cls(0, (0,) * cls.SUM_COUNT)

    def test_invalid_facts_are_rejected(self) -> None:
        for cls in STATES:
            zeros = (0,) * cls.SUM_COUNT
            with self.subTest(state=cls.__name__):
                for count in (True, 1.0, "1", None):
                    with self.assertRaises(TypeError):
                        cls(count, zeros)  # type: ignore[arg-type]
                for count in (-1, MAX_OBSERVATION_COUNT + 1):
                    with self.assertRaises(ValueError):
                        cls(count, zeros)
                for sums in ([0] * cls.SUM_COUNT, (), zeros + (0,), None, 0):
                    with self.assertRaises(ValueError):
                        cls(0, sums)  # type: ignore[arg-type]
                for element in (0.0, True, "0", None):
                    with self.assertRaises(TypeError):
                        cls(0, (element,) * cls.SUM_COUNT)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    cls(0, (1,) * cls.SUM_COUNT)
                with self.assertRaises(ValueError):
                    cls(1, (MAX_ELEMENT_UNITS + 1,) * cls.SUM_COUNT)
                with self.assertRaises(ValueError):
                    cls(1, (-MAX_ELEMENT_UNITS - 1,) * cls.SUM_COUNT)
                with self.assertRaises(ValueError):
                    cls(3, (3 * MAX_ELEMENT_UNITS + 1,) + (0,) * (cls.SUM_COUNT - 1))

    def test_the_abstract_base_cannot_be_instantiated(self) -> None:
        from veridist.scale._state import ExactSumsState

        with self.assertRaises(TypeError):
            ExactSumsState(0, ())  # type: ignore[abstract]


class UpdateRefusalTests(unittest.TestCase):
    def test_non_finite_values_are_refused_before_any_domain_check(self) -> None:
        for cls in STATES:
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(state=cls.__name__, bad=bad):
                    refused(
                        self,
                        CODES.NON_FINITE_VALUE,
                        lambda: cls.empty().update(np.array([1.0, bad])),
                    )

    def test_support_violations(self) -> None:
        refused(
            self,
            CODES.VALUE_OUT_OF_SUPPORT,
            lambda: ExponentialState.empty().update(np.array([1.0, -1e-300])),
        )
        for cls in (GammaState, LognormalState):
            for bad in (0.0, -0.0, -1.0, -5e-324):
                with self.subTest(state=cls.__name__, bad=bad):
                    refused(
                        self,
                        CODES.VALUE_OUT_OF_SUPPORT,
                        lambda: cls.empty().update(np.array([2.0, bad])),
                    )
        ExponentialState.empty().update(np.array([0.0, -0.0, 5e-324]))
        NormalState.empty().update(np.array([-1e150, 0.0, -0.0, 1e-300]))

    def test_support_boundaries_are_accepted(self) -> None:
        for cls in (GammaState, LognormalState):
            state = cls.empty().update(np.array([5e-324, MAX_FLOAT, 1.0]))
            self.assertEqual(state.count, 3)

    def test_a_square_beyond_the_float_range_is_summed_exactly_not_refused(self) -> None:
        for values in (
            np.array([1.0, 1.5e154]),
            np.array([-MAX_FLOAT]),
            np.array([1e154, -1e154]),
            np.array([5e-324, 1e-200]),
        ):
            state = NormalState.empty().update(values)
            self.assertEqual(state.sums[1], exact_square_units(values.tolist()))
            self.assertEqual(state.count, len(values))

    def test_a_refused_update_leaves_the_state_unchanged(self) -> None:
        rng = np.random.default_rng(4)
        for cls in STATES:
            with self.subTest(state=cls.__name__):
                state = cls.empty().update(batch(cls, rng, 6))
                before = (state.count, state.sums, state.to_bytes())
                for bad in (np.array([1.0, np.nan]), np.array([np.inf])):
                    with self.assertRaises(ScaleStateError):
                        state.update(bad)
                self.assertEqual(before, (state.count, state.sums, state.to_bytes()))

    def test_wrong_container_dtype_shape(self) -> None:
        for cls in STATES:
            with self.subTest(state=cls.__name__):
                for bad in (
                    [1.0],
                    (1.0,),
                    1.0,
                    None,
                    np.array([1.0], dtype=np.float32),
                    np.array([1]),
                ):
                    with self.assertRaises(TypeError):
                        cls.empty().update(bad)
                for bad in (np.array(1.0), np.ones((2, 2))):
                    with self.assertRaises(ValueError):
                        cls.empty().update(bad)

    def test_observation_count_cap_on_update_and_merge(self) -> None:
        for cls in STATES:
            with self.subTest(state=cls.__name__):
                zeros = (0,) * cls.SUM_COUNT
                near = cls(MAX_OBSERVATION_COUNT - 1, zeros)
                refused(
                    self,
                    CODES.OBSERVATION_LIMIT_EXCEEDED,
                    lambda: near.update(np.array([1.0, 2.0])),
                )
                full = near.update(np.array([1.0]))
                self.assertEqual(full.count, MAX_OBSERVATION_COUNT)
                self.assertEqual(full.update(np.empty(0)), full)
                refused(
                    self, CODES.OBSERVATION_LIMIT_EXCEEDED, lambda: full.update(np.array([1.0]))
                )
                one = cls(1, zeros)
                refused(self, CODES.OBSERVATION_LIMIT_EXCEEDED, lambda: full.merge(one))
                refused(self, CODES.OBSERVATION_LIMIT_EXCEEDED, lambda: one.merge(full))
                self.assertEqual(near.merge(one).count, MAX_OBSERVATION_COUNT)
                self.assertEqual(near.merge(cls.empty()).count, MAX_OBSERVATION_COUNT - 1)


class MergeRefusalTests(unittest.TestCase):
    def test_different_family_or_version_is_incompatible(self) -> None:
        for left in STATES:
            for right in STATES:
                if left is not right:
                    with self.subTest(left=left.__name__, right=right.__name__):
                        refused(
                            self,
                            CODES.INCOMPATIBLE_STATE,
                            lambda: left.empty().merge(right.empty()),
                        )  # type: ignore[arg-type]
        refused(
            self,
            CODES.INCOMPATIBLE_STATE,
            lambda: ExponentialState.empty().merge(_OtherTag.empty()),
        )
        refused(
            self,
            CODES.INCOMPATIBLE_STATE,
            lambda: ExponentialState.empty().merge(_NextVersion.empty()),
        )
        refused(
            self,
            CODES.INCOMPATIBLE_STATE,
            lambda: _NextVersion.empty().merge(ExponentialState.empty()),
        )
        self.assertEqual(_NextVersion.empty().merge(_NextVersion.empty()), _NextVersion.empty())

    def test_a_non_state_is_a_type_error(self) -> None:
        for bad in (None, 5, (0, (0,)), "state"):
            with self.assertRaises(TypeError):
                ExponentialState.empty().merge(bad)  # type: ignore[arg-type]


class SerializationContractTests(unittest.TestCase):
    def test_from_bytes_errors_by_state_type(self) -> None:
        state = NormalState.empty().update(np.array([1.0, 2.0]))
        data = state.to_bytes()
        refused(self, CODES.STATE_TAG_MISMATCH, lambda: GammaState.from_bytes(data))
        refused(
            self,
            CODES.STATE_TAG_MISMATCH,
            lambda: _OtherTag.from_bytes(ExponentialState.empty().to_bytes()),
        )
        refused(
            self,
            CODES.STATE_VERSION_UNSUPPORTED,
            lambda: _NextVersion.from_bytes(ExponentialState.empty().to_bytes()),
        )
        refused(self, CODES.STATE_BYTES_INVALID, lambda: NormalState.from_bytes(data[:-1]))
        with self.assertRaises(TypeError):
            NormalState.from_bytes("text")

    def test_bytes_that_decode_to_an_invalid_state_are_invalid_bytes(self) -> None:
        zero = ((0, b""),)
        body = frame_body(tag=b"scale.exponential", count=0, integers=((0, b"\x01"),))
        refused(self, CODES.STATE_BYTES_INVALID, lambda: ExponentialState.from_bytes(seal(body)))
        self.assertEqual(
            ExponentialState.from_bytes(
                seal(frame_body(tag=b"scale.exponential", count=0, integers=zero))
            ),
            ExponentialState.empty(),
        )


class FinalizeTests(unittest.TestCase):
    def test_empty_states(self) -> None:
        exponential = ExponentialState.empty().finalize()
        self.assertEqual(
            (exponential.count, exponential.total_time), (0, ExactStatistic(Fraction(0), 0.0))
        )
        normal = NormalState.empty().finalize()
        self.assertEqual(normal.count, 0)
        self.assertEqual(normal.sum_x, ExactStatistic(Fraction(0), 0.0))
        self.assertEqual(normal.sum_x_squared, ExactStatistic(Fraction(0), 0.0))
        self.assertIsNone(normal.mean)
        self.assertIsNone(normal.centered_sum_of_squares)
        gamma = GammaState.empty().finalize()
        self.assertEqual((gamma.sum_x.rounded, gamma.sum_log_x.rounded), (0.0, 0.0))
        self.assertIsNone(gamma.mean_x)
        self.assertIsNone(gamma.mean_log_x)
        lognormal = LognormalState.empty().finalize()
        self.assertIsNone(lognormal.mean_log_x)
        self.assertIsNone(lognormal.centered_sum_of_squares_log_x)

    def test_exponential_sufficient_statistics(self) -> None:
        values = np.array([0.5, 1.5, 0.0, 2.25])
        result = ExponentialState.empty().update(values).finalize()
        self.assertEqual(result.count, 4)
        self.assertEqual(result.total_time, ExactStatistic(Fraction(17, 4), 4.25))

    def test_normal_statistics_are_exact_rationals(self) -> None:
        result = NormalState.empty().update(np.array([1.0, 2.0, 4.0])).finalize()
        self.assertEqual(result.count, 3)
        self.assertEqual(result.sum_x, ExactStatistic(Fraction(7), 7.0))
        self.assertEqual(result.sum_x_squared, ExactStatistic(Fraction(21), 21.0))
        assert result.mean is not None and result.centered_sum_of_squares is not None
        self.assertEqual(result.mean.exact, Fraction(7, 3))
        self.assertEqual(result.mean.rounded, 7 / 3)
        self.assertEqual(result.centered_sum_of_squares.exact, Fraction(14, 3))
        self.assertEqual(result.centered_sum_of_squares.rounded, 14 / 3)

    def test_normal_variance_numerator_is_exact(self) -> None:
        # Data far from zero relative to their spread: the squares are summed exactly, so
        # the centered sum is the exact one, 8 * 5.25 for eight consecutive integers.
        values = 1.0e9 + np.arange(0.0, 8.0)
        result = NormalState.empty().update(values).finalize()
        assert result.centered_sum_of_squares is not None
        self.assertEqual(result.centered_sum_of_squares.exact, Fraction(42))
        self.assertEqual(result.centered_sum_of_squares.rounded, 42.0)
        self.assertEqual(
            result.sum_x_squared.exact,
            sum((Fraction(float(v)) ** 2 for v in values), Fraction(0)),
        )

    def test_the_centered_sum_is_zero_exactly_when_all_values_are_equal(self) -> None:
        for value in (0.1, 1e9 + 0.3, 1e200, 5e-300, 123456.789):
            same = NormalState.empty().update(np.full(7, value)).finalize()
            assert same.centered_sum_of_squares is not None
            self.assertEqual(same.centered_sum_of_squares.exact, 0, value)
            bumped = np.array([value] * 6 + [np.nextafter(value, 2 * value + 1)])
            other = NormalState.empty().update(bumped)
            statistics = other.finalize()
            assert statistics.centered_sum_of_squares is not None
            self.assertGreater(statistics.centered_sum_of_squares.exact, 0)

    def test_a_sum_of_squares_beyond_the_float_range_saturates_but_stays_exact(self) -> None:
        values = np.array([1.2e154, 1.2e154])
        statistics = NormalState.empty().update(values).finalize()
        self.assertEqual(statistics.sum_x_squared.exact, 2 * Fraction(1.2e154) ** 2)
        self.assertEqual(statistics.sum_x_squared.rounded, math.inf)
        assert statistics.centered_sum_of_squares is not None
        self.assertEqual(statistics.centered_sum_of_squares.exact, 0)
        huge = NormalState.empty().update(np.array([1e200, -1e200])).finalize()
        assert huge.centered_sum_of_squares is not None
        self.assertEqual(huge.centered_sum_of_squares.rounded, math.inf)
        self.assertEqual(huge.centered_sum_of_squares.exact, 2 * Fraction(1e200) ** 2)

    def test_sums_match_independent_exact_oracles(self) -> None:
        rng = np.random.default_rng(5)
        for cls in STATES:
            with self.subTest(state=cls.__name__):
                values = batch(cls, rng, 40)
                state = cls.empty().update(values)
                self.assertEqual(state.sums, expected_sums(cls, values))
                self.assertEqual(state.count, 40)

    def test_gamma_and_lognormal_statistics(self) -> None:
        rng = np.random.default_rng(6)
        values = batch(GammaState, rng, 50)
        logs = np.log(values)
        gamma = GammaState.empty().update(values).finalize()
        total, total_logs = exact_fraction(values.tolist()), exact_fraction(logs.tolist())
        self.assertEqual(gamma.sum_x.exact, total)
        self.assertEqual(gamma.sum_log_x.exact, total_logs)
        assert gamma.mean_x is not None and gamma.mean_log_x is not None
        self.assertEqual(gamma.mean_x.exact, total / 50)
        self.assertEqual(gamma.mean_log_x.exact, total_logs / 50)
        self.assertEqual(gamma.mean_log_x.rounded, float(total_logs / 50))
        lognormal = LognormalState.empty().update(values).finalize()
        squares = sum((Fraction(float(v)) ** 2 for v in logs), Fraction(0))
        self.assertEqual(lognormal.sum_log_x.exact, total_logs)
        self.assertEqual(lognormal.sum_log_x_squared.exact, squares)
        assert lognormal.mean_log_x is not None
        assert lognormal.centered_sum_of_squares_log_x is not None
        self.assertEqual(lognormal.mean_log_x.exact, total_logs / 50)
        self.assertEqual(
            lognormal.centered_sum_of_squares_log_x.exact, squares - total_logs * total_logs / 50
        )

    def test_final_overflow_is_typed_and_leaves_the_state_valid(self) -> None:
        state = ExponentialState.empty().update(np.array([MAX_FLOAT, MAX_FLOAT]))
        refused(self, CODES.TOTAL_NOT_REPRESENTABLE, state.finalize)
        self.assertEqual(state.sums, (2 * MAX_ELEMENT_UNITS,))
        self.assertEqual(state.update(np.array([0.0])).count, 3)
        normal = NormalState.empty().update(np.array([MAX_FLOAT, MAX_FLOAT]))
        refused(self, CODES.TOTAL_NOT_REPRESENTABLE, normal.finalize)

    def test_exact_statistic_rounds_once_and_refuses_overflow(self) -> None:
        third = ExactStatistic.of(Fraction(1, 3))
        self.assertEqual((third.exact, third.rounded), (Fraction(1, 3), 1 / 3))
        self.assertEqual(ExactStatistic.of(Fraction(-1, 3)).rounded, -1 / 3)
        tiny = ExactStatistic.of(Fraction(1, 10**400))
        self.assertEqual(tiny.rounded, 0.0)
        self.assertEqual(tiny.exact, Fraction(1, 10**400))
        error = refused(
            self, CODES.TOTAL_NOT_REPRESENTABLE, lambda: ExactStatistic.of(Fraction(10**400))
        )
        self.assertIsInstance(error.__cause__, OverflowError)
        self.assertEqual(
            ExactStatistic.of(Fraction(MAX_FLOAT).limit_denominator(1)).rounded, MAX_FLOAT
        )


class SaturatingStatisticTests(unittest.TestCase):
    def test_in_range_values_round_once_like_the_plain_constructor(self) -> None:
        for value in (Fraction(1, 3), Fraction(-1, 3), Fraction(0), Fraction(10**300)):
            self.assertEqual(ExactStatistic.saturating(value), ExactStatistic.of(value))

    def test_values_beyond_the_range_saturate_with_their_sign(self) -> None:
        huge = Fraction(10**400)
        self.assertEqual(ExactStatistic.saturating(huge), ExactStatistic(huge, math.inf))
        self.assertEqual(ExactStatistic.saturating(-huge), ExactStatistic(-huge, -math.inf))


class KernelTests(unittest.TestCase):
    def test_batch_validation_returns_the_same_array_when_contiguous(self) -> None:
        values = np.arange(5.0)
        self.assertIs(_kernels.float64_batch(values), values)
        strided = np.arange(10.0)[::2]
        converted = _kernels.float64_batch(strided)
        self.assertTrue(converted.flags.c_contiguous)
        np.testing.assert_array_equal(converted, strided)

    def test_numpy_is_loaded_lazily_through_one_helper(self) -> None:
        self.assertIs(_kernels.numpy_module(), np)

    def test_require_finite_and_domain_checks(self) -> None:
        _kernels.require_finite(np.array([1.0, -1.0, 0.0]))
        _kernels.require_finite(np.empty(0))
        _kernels.require_non_negative(np.array([0.0, -0.0, 2.0]))
        _kernels.require_positive(np.array([5e-324, 2.0]))
        for check, bad in (
            (_kernels.require_finite, np.array([1.0, np.nan])),
            (_kernels.require_non_negative, np.array([1.0, -5e-324])),
            (_kernels.require_positive, np.array([1.0, 0.0])),
        ):
            with self.assertRaises(ScaleStateError):
                check(bad)

    def test_kernel_results_do_not_depend_on_position_batch_or_alignment(self) -> None:
        rng = np.random.default_rng(7)
        buffer = np.exp2(rng.uniform(-300, 300, 700)) * rng.uniform(0.5, 2.0, 700)
        reference_log = np.array([_kernels.logarithms(buffer[i : i + 1])[0] for i in range(700)])
        for offset in (0, 1, 2, 3, 5, 7):
            for length in (1, 2, 3, 4, 7, 8, 9, 15, 16, 17, 31, 33, 100, 693):
                window = slice(offset, offset + length)
                with self.subTest(offset=offset, length=length):
                    np.testing.assert_array_equal(
                        _kernels.logarithms(buffer[window]), reference_log[window]
                    )

    def test_logarithm_is_numpy_log(self) -> None:
        values = np.array([1e-300, 0.1, 1.0, 3.0, 1e300])
        np.testing.assert_array_equal(_kernels.logarithms(values), np.log(values))
        self.assertEqual(float(_kernels.logarithms(np.array([1.0]))[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
