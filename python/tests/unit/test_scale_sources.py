"""Partition descriptor, fingerprint encoding, value policy, array source and fold."""

from __future__ import annotations

import hashlib
import pickle
import re
import unittest
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from unittest import mock

import numpy as np

from tests.unit.scale_source_oracle import (
    ListSource,
    TraceState,
    numpy_fingerprint,
    reference_digest,
    unresumable,
)
from veridist.engine.errors import VeridistError
from veridist.scale import _sources as src
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode, ScaleStateError
from veridist.scale._one_pass import ExponentialState, NormalState
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    FingerprintLevel,
    NumpySource,
    Partition,
    canonical_digest,
    check_max_rows,
    float64_values,
    fold,
    fold_partition,
)

CODES = ScaleSourceErrorCode
GOOD = "sha256:" + "0" * 64


class ConstantTests(unittest.TestCase):
    def test_documented_constants(self) -> None:
        self.assertEqual(DEFAULT_MAX_ROWS, 262144)
        self.assertEqual(DEFAULT_MAX_ROWS, 2**18)
        self.assertEqual(src.MAX_EXACT_INTEGER, 2**53)
        self.assertEqual(src.FINGERPRINT_ALGORITHM, "sha256")
        self.assertEqual(src.HASH_BLOCK_ROWS, 2**16)

    def test_fingerprint_levels(self) -> None:
        self.assertEqual(
            [level.value for level in FingerprintLevel], ["none", "metadata", "content"]
        )


class ErrorTests(unittest.TestCase):
    def test_error_carries_only_its_code_without_a_hint(self) -> None:
        error = ScaleSourceError(CODES.NULL_VALUE)
        self.assertEqual(str(error), "NULL_VALUE")
        self.assertIs(error.code, CODES.NULL_VALUE)
        self.assertIsNone(error.hint)
        self.assertEqual(error.args, ("NULL_VALUE",))

    def test_error_with_a_hint_appends_it_after_the_code(self) -> None:
        error = ScaleSourceError("COLUMN_NOT_FOUND", "do this")
        self.assertEqual(str(error), "COLUMN_NOT_FOUND: do this")
        self.assertEqual(error.hint, "do this")
        self.assertIs(error.code, CODES.COLUMN_NOT_FOUND)

    def test_error_is_a_catchable_value_error_and_rejects_unknown_codes(self) -> None:
        self.assertIsInstance(ScaleSourceError(CODES.LOSSY_CAST), VeridistError)
        self.assertIsInstance(ScaleSourceError(CODES.LOSSY_CAST), ValueError)
        with self.assertRaises(ValueError):
            ScaleSourceError("NOT_A_CODE")


class PartitionTests(unittest.TestCase):
    def make(self, **overrides: object) -> Partition:
        fields: dict[str, object] = {
            "id": "a",
            "ordinal": 3,
            "rows": 5,
            "fingerprint": GOOD,
            "fingerprint_level": FingerprintLevel.METADATA,
        }
        fields.update(overrides)
        return Partition(**fields)  # type: ignore[arg-type]

    def test_fields_and_resumability(self) -> None:
        partition = self.make()
        self.assertEqual(
            (partition.id, partition.ordinal, partition.rows, partition.fingerprint),
            ("a", 3, 5, GOOD),
        )
        self.assertIs(partition.fingerprint_level, FingerprintLevel.METADATA)
        self.assertTrue(partition.resumable)
        self.assertFalse(unresumable("s").resumable)

    def test_unknown_rows_and_zero_values_are_allowed(self) -> None:
        self.assertIsNone(self.make(rows=None).rows)
        zero = self.make(rows=0, ordinal=0)
        self.assertEqual((zero.rows, zero.ordinal), (0, 0))

    def test_is_immutable_hashable_and_picklable(self) -> None:
        partition = self.make()
        with self.assertRaises(FrozenInstanceError):
            partition.id = "b"  # type: ignore[misc]
        self.assertEqual(hash(partition), hash(self.make()))
        self.assertEqual(pickle.loads(pickle.dumps(partition)), partition)
        self.assertEqual(len({partition, self.make()}), 1)
        self.assertNotEqual(partition, self.make(ordinal=4))

    def test_invalid_fields_are_refused(self) -> None:
        id_message = "id must be a non-empty string"
        ordinal_type = "ordinal must be a built-in integer"
        rows_type = "rows must be a built-in integer or None"
        fingerprint_message = "fingerprint must be 'sha256:' and 64 lowercase hex digits"
        for overrides, error, message in (
            ({"id": ""}, TypeError, id_message),
            ({"id": 5}, TypeError, id_message),
            ({"ordinal": True}, TypeError, ordinal_type),
            ({"ordinal": 1.0}, TypeError, ordinal_type),
            ({"ordinal": -1}, ValueError, "ordinal must not be negative"),
            ({"rows": True}, TypeError, rows_type),
            ({"rows": 2.0}, TypeError, rows_type),
            ({"rows": -1}, ValueError, "rows must not be negative"),
            (
                {"fingerprint_level": "metadata"},
                TypeError,
                "fingerprint_level must be a FingerprintLevel",
            ),
            ({"fingerprint": None}, ValueError, fingerprint_message),
            ({"fingerprint": 5}, ValueError, fingerprint_message),
            ({"fingerprint": "sha256:" + "0" * 63}, ValueError, fingerprint_message),
            ({"fingerprint": "sha256:" + "0" * 65}, ValueError, fingerprint_message),
            ({"fingerprint": "sha256:" + "G" * 64}, ValueError, fingerprint_message),
            ({"fingerprint": "sha256:" + "A" * 64}, ValueError, fingerprint_message),
            ({"fingerprint": "md5:" + "0" * 64}, ValueError, fingerprint_message),
            ({"fingerprint": "x" + GOOD}, ValueError, fingerprint_message),
            ({"fingerprint": GOOD + "x"}, ValueError, fingerprint_message),
            (
                {"fingerprint_level": FingerprintLevel.NONE},
                ValueError,
                "a partition without a fingerprint level has no fingerprint",
            ),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(error, f"^{re.escape(message)}$"),
            ):
                self.make(**overrides)

    def test_level_none_requires_no_fingerprint(self) -> None:
        partition = self.make(fingerprint=None, fingerprint_level=FingerprintLevel.NONE)
        self.assertIsNone(partition.fingerprint)
        for level in (FingerprintLevel.METADATA, FingerprintLevel.CONTENT):
            with self.subTest(level=level), self.assertRaises(ValueError):
                self.make(fingerprint=None, fingerprint_level=level)
            self.assertIs(self.make(fingerprint_level=level).fingerprint_level, level)


class CanonicalDigestTests(unittest.TestCase):
    def test_pinned_vectors(self) -> None:
        self.assertEqual(
            canonical_digest(),
            "sha256:eba279699b0e698aab59063192bf881ed0a317a82a5a9736ccdacca393250024",
        )
        self.assertEqual(
            canonical_digest("a", -12, None, b"x", (1, "b", (None,)), ()),
            "sha256:b375569ff71d53eae4c1bc104f1855d2a2a4e3a6985eac9ef75bbbe3992591b2",
        )

    def test_matches_the_independent_encoding(self) -> None:
        for items in (
            (),
            (None,),
            (0,),
            (10**30, -7),
            ("", "é"),
            (b"", b"\x00\xff"),
            ((), ((),), (1, 2)),
            ("a", ("b", ("c", None))),
        ):
            with self.subTest(items=items):
                self.assertEqual(canonical_digest(*items), reference_digest(*items))

    def test_distinct_records_never_collide(self) -> None:
        records = [
            (),
            (None,),
            (0,),
            ("0",),
            (b"0",),
            ((0,),),
            (0, 0),
            (1,),
            ("ab",),
            ("a", "b"),
            (("a", "b"),),
            (("a",), "b"),
            ("a", ("b",)),
            (None, None),
            ((None,),),
            ((),),
            ((), ()),
        ]
        digests = {canonical_digest(*record) for record in records}
        self.assertEqual(len(digests), len(records))

    def test_format_and_unsupported_items(self) -> None:
        digest = canonical_digest("x")
        self.assertRegex(digest, r"^sha256:[0-9a-f]{64}$")
        for bad in (True, 1.5, [1], {"a": 1}, {1}, np.int64(1), object()):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                canonical_digest(bad)
        message = "a fingerprint record holds None, int, str, bytes and tuples only"
        with self.assertRaisesRegex(TypeError, f"^{re.escape(message)}$"):
            canonical_digest(("fine", 2.5))


class MaxRowsTests(unittest.TestCase):
    def test_accepts_positive_integers(self) -> None:
        self.assertEqual(check_max_rows(1), 1)
        self.assertEqual(check_max_rows(10**9), 10**9)

    def test_refuses_others(self) -> None:
        small, kind = "max_rows must be at least 1", "max_rows must be a built-in integer"
        for bad, error, message in (
            (0, ValueError, small),
            (-3, ValueError, small),
            (True, TypeError, kind),
            (2.0, TypeError, kind),
            ("2", TypeError, kind),
            (None, TypeError, kind),
            (np.int64(4), TypeError, kind),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(error, f"^{re.escape(message)}$"):
                check_max_rows(bad)


def code_of(call: Callable[[], object]) -> ScaleSourceErrorCode:
    try:
        call()
    except ScaleSourceError as error:
        return error.code
    raise AssertionError("the call was not refused")


class ValuePolicyTests(unittest.TestCase):
    def test_float64_is_returned_unchanged_and_without_a_copy(self) -> None:
        values = np.array([1.5, -2.25, 0.0, -0.0, 1e300, 5e-324])
        result = float64_values(values)
        self.assertIs(result, values)
        self.assertEqual(result.dtype, np.float64)

    def test_float64_views_stay_views(self) -> None:
        base = np.arange(10.0)
        view = float64_values(base[2:7])
        self.assertTrue(np.shares_memory(view, base))
        np.testing.assert_array_equal(view, [2.0, 3.0, 4.0, 5.0, 6.0])

    def test_non_contiguous_and_foreign_endian_float64_become_native_contiguous(self) -> None:
        strided = np.arange(10.0)[::2]
        result = float64_values(strided)
        self.assertTrue(result.flags["C_CONTIGUOUS"])
        np.testing.assert_array_equal(result, [0.0, 2.0, 4.0, 6.0, 8.0])
        swapped = np.array([1.0, 2.0, -3.5], dtype=">f8")
        result = float64_values(swapped)
        self.assertEqual(result.dtype, np.dtype("<f8"))
        np.testing.assert_array_equal(result, [1.0, 2.0, -3.5])

    def test_float32_is_widened_exactly(self) -> None:
        values = np.array([0.1, -3.4028235e38, 1.0e-45, 16777217.0], dtype=np.float32)
        result = float64_values(values)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.tolist(), [float(v) for v in values])
        self.assertEqual(result[0], 0.10000000149011612)

    def test_small_integers_are_widened_without_a_range_check(self) -> None:
        for dtype in (np.int8, np.int16, np.int32, np.uint8, np.uint16, np.uint32):
            info = np.iinfo(dtype)
            values = np.array([info.min, 0, info.max], dtype=dtype)
            with self.subTest(dtype=dtype):
                result = float64_values(values)
                self.assertEqual(result.dtype, np.float64)
                self.assertEqual(result.tolist(), [float(info.min), 0.0, float(info.max)])

    def test_wide_integers_are_accepted_up_to_two_to_the_53(self) -> None:
        limit = 2**53
        for dtype in (np.int64, np.uint64):
            low = -limit if dtype is np.int64 else 0
            values = np.array([low, 1, limit - 1, limit], dtype=dtype)
            with self.subTest(dtype=dtype):
                expected = [float(v) for v in values.tolist()]
                self.assertEqual(float64_values(values).tolist(), expected)

    def test_wide_integers_beyond_two_to_the_53_are_refused_on_either_side(self) -> None:
        limit = 2**53
        cases = [
            np.array([limit + 1], dtype=np.int64),
            np.array([-limit - 1], dtype=np.int64),
            np.array([-limit - 1, 0], dtype=np.int64),
            np.array([0, limit + 1], dtype=np.int64),
            np.array([0, 5, limit + 1, 7], dtype=np.int64),
            np.array([limit + 1], dtype=np.uint64),
            np.array([2**64 - 1], dtype=np.uint64),
            np.array([np.iinfo(np.int64).min], dtype=np.int64),
        ]
        for values in cases:
            with self.subTest(values=values.tolist()):
                self.assertIs(code_of(lambda v=values: float64_values(v)), CODES.LOSSY_CAST)

    def test_empty_arrays_of_every_accepted_dtype(self) -> None:
        for dtype in (np.float64, np.float32, np.int8, np.int64, np.uint64):
            with self.subTest(dtype=dtype):
                result = float64_values(np.array([], dtype=dtype))
                self.assertEqual((result.dtype, result.shape), (np.float64, (0,)))

    def test_nan_and_infinities_are_refused_with_their_own_codes(self) -> None:
        for values, code in (
            ([1.0, np.nan], CODES.NAN_VALUE),
            ([np.nan], CODES.NAN_VALUE),
            ([np.inf, 1.0], CODES.NON_FINITE_VALUE),
            ([-np.inf], CODES.NON_FINITE_VALUE),
            ([np.inf, np.nan], CODES.NAN_VALUE),
            ([-np.inf, np.nan, np.inf], CODES.NAN_VALUE),
        ):
            for dtype in (np.float64, np.float32):
                with self.subTest(values=values, dtype=dtype):
                    array = np.array(values, dtype=dtype)
                    self.assertIs(code_of(lambda a=array: float64_values(a)), code)

    def test_other_dtypes_are_unsupported(self) -> None:
        for values in (
            np.array([1.0], dtype=np.float16),
            np.array([True]),
            np.array([1 + 2j]),
            np.array(["1.0"]),
            np.array([1.0], dtype=object),
            np.array(["2020-01-01"], dtype="datetime64[D]"),
            np.array([1], dtype="timedelta64[s]"),
        ):
            with self.subTest(dtype=values.dtype):
                self.assertIs(code_of(lambda v=values: float64_values(v)), CODES.UNSUPPORTED_DTYPE)

    def test_a_refusal_message_is_the_code_and_never_a_value(self) -> None:
        for values in (np.array([123456.789, np.nan]), np.array([2**60 + 12345], dtype=np.int64)):
            with self.assertRaises(ScaleSourceError) as context:
                float64_values(values)
            self.assertIn(str(context.exception), {"NAN_VALUE", "LOSSY_CAST"})
            self.assertEqual(context.exception.args, (str(context.exception),))


class NumpySourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
        self.source = NumpySource(self.values)

    def test_partitions_describe_one_content_fingerprinted_partition(self) -> None:
        (partition,) = self.source.partitions()
        self.assertEqual(partition.id, "array")
        self.assertEqual(partition.ordinal, 0)
        self.assertEqual(partition.rows, 7)
        self.assertIs(partition.fingerprint_level, FingerprintLevel.CONTENT)
        self.assertEqual(partition.fingerprint, numpy_fingerprint(self.values))
        self.assertTrue(partition.resumable)
        self.assertEqual(self.source.partitions(), (partition,))

    def test_the_fingerprint_depends_on_bytes_dtype_and_length(self) -> None:
        def fingerprint(values: np.ndarray) -> str | None:
            return NumpySource(values).partitions()[0].fingerprint

        base = fingerprint(self.values)
        self.assertEqual(fingerprint(self.values.copy()), base)
        changed = self.values.copy()
        changed[3] = np.nextafter(changed[3], 10.0)
        self.assertNotEqual(fingerprint(changed), base)
        self.assertNotEqual(fingerprint(self.values[:-1]), base)
        self.assertNotEqual(fingerprint(self.values.astype(np.float32)), base)
        self.assertNotEqual(fingerprint(self.values.astype(">f8")), base)
        self.assertNotEqual(
            fingerprint(np.array([0.0], dtype=np.float64)),
            fingerprint(np.array([-0.0], dtype=np.float64)),
        )
        self.assertNotEqual(
            fingerprint(np.arange(3, dtype=np.int64)), fingerprint(np.arange(3, dtype=np.uint64))
        )

    def test_the_fingerprint_of_a_strided_view_equals_that_of_its_copy(self) -> None:
        strided = self.values[::2]
        self.assertEqual(
            NumpySource(strided).partitions()[0].fingerprint,
            NumpySource(strided.copy()).partitions()[0].fingerprint,
        )

    def test_hashing_is_blockwise_and_block_size_does_not_matter(self) -> None:
        expected = numpy_fingerprint(self.values)
        for block in (1, 2, 3, 7, 8, 100):
            with (
                self.subTest(block=block),
                mock.patch.object(src, "HASH_BLOCK_ROWS", block),
            ):
                self.assertEqual(self.source.partitions()[0].fingerprint, expected)

    def test_an_empty_array_has_a_fingerprint_and_no_batches(self) -> None:
        empty = NumpySource(np.array([], dtype=np.float64))
        (partition,) = empty.partitions()
        self.assertEqual(partition.rows, 0)
        self.assertEqual(partition.fingerprint, numpy_fingerprint(np.array([], dtype=np.float64)))
        self.assertEqual(list(empty.batches(partition)), [])

    def test_batches_are_bounded_ordered_and_cover_every_row_once(self) -> None:
        (partition,) = self.source.partitions()
        for max_rows, sizes in ((1, [1] * 7), (3, [3, 3, 1]), (7, [7]), (8, [7]), (100, [7])):
            with self.subTest(max_rows=max_rows):
                batches = list(self.source.batches(partition, max_rows=max_rows))
                self.assertEqual([len(batch) for batch in batches], sizes)
                self.assertEqual(np.concatenate(batches).tolist(), self.values.tolist())

    def test_the_default_bound_is_the_documented_default(self) -> None:
        values = np.zeros(DEFAULT_MAX_ROWS + 5)
        source = NumpySource(values)
        sizes = [len(b) for b in source.batches(source.partitions()[0])]
        self.assertEqual(sizes, [DEFAULT_MAX_ROWS, 5])

    def test_float64_batches_are_zero_copy_views(self) -> None:
        (partition,) = self.source.partitions()
        batches = list(self.source.batches(partition, max_rows=3))
        for batch in batches:
            self.assertTrue(np.shares_memory(batch, self.values))
            self.assertFalse(batch.flags["OWNDATA"])
        batches[0][0] = 99.0  # a view: the source array is the same memory
        self.assertEqual(self.values[0], 99.0)

    def test_other_dtypes_are_widened_one_batch_at_a_time(self) -> None:
        for dtype in (np.float32, np.int32, np.int64, np.uint16):
            values = np.array([1, 2, 3, 4, 5], dtype=dtype)
            source = NumpySource(values)
            with self.subTest(dtype=dtype):
                batches = list(source.batches(source.partitions()[0], max_rows=2))
                self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
                self.assertTrue(all(batch.dtype == np.float64 for batch in batches))
                self.assertFalse(any(np.shares_memory(batch, values) for batch in batches))
                self.assertEqual(np.concatenate(batches).tolist(), [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_a_refused_value_surfaces_at_its_batch_after_earlier_ones(self) -> None:
        source = NumpySource(np.array([1.0, 2.0, np.nan, 4.0]))
        iterator = source.batches(source.partitions()[0], max_rows=2)
        self.assertEqual(next(iterator).tolist(), [1.0, 2.0])
        with self.assertRaises(ScaleSourceError) as context:
            next(iterator)
        self.assertIs(context.exception.code, CODES.NAN_VALUE)

    def test_integers_beyond_exact_range_are_refused_when_read(self) -> None:
        source = NumpySource(np.array([1, 2**60], dtype=np.int64))
        iterator = source.batches(source.partitions()[0])
        self.assertIs(code_of(lambda: next(iterator)), CODES.LOSSY_CAST)

    def test_arguments_are_validated_when_batches_is_called_not_when_iterated(self) -> None:
        (partition,) = self.source.partitions()
        for bad, error in ((0, ValueError), (True, TypeError), (1.5, TypeError)):
            with self.subTest(max_rows=bad), self.assertRaises(error):
                self.source.batches(partition, max_rows=bad)  # type: ignore[arg-type]
        for foreign in (unresumable("other"), unresumable("arrays"), "array", None):
            with (
                self.subTest(partition=foreign),
                self.assertRaisesRegex(ValueError, "^partition does not belong to this source$"),
            ):
                self.source.batches(foreign)  # type: ignore[arg-type]

    def test_constructor_refuses_unsuitable_containers_and_dtypes(self) -> None:
        container, shape = "values must be a numpy.ndarray", "values must be one-dimensional"
        for bad, error, message in (
            ([1.0, 2.0], TypeError, container),
            (5.0, TypeError, container),
            (None, TypeError, container),
            (np.ma.masked_array([1.0, 2.0], mask=[False, True]), TypeError, container),
            (np.zeros((2, 2)), ValueError, shape),
            (np.float64(1.0), TypeError, container),
            (np.zeros(()), ValueError, shape),
        ):
            with self.subTest(bad=bad), self.assertRaisesRegex(error, f"^{re.escape(message)}$"):
                NumpySource(bad)
        for values in (
            np.zeros(2, dtype=np.float16),
            np.zeros(2, dtype=bool),
            np.zeros(2, dtype=complex),
            np.zeros(2, dtype=object),
            np.array(["a"]),
        ):
            with self.subTest(dtype=values.dtype):
                self.assertIs(code_of(lambda v=values: NumpySource(v)), CODES.UNSUPPORTED_DTYPE)

    def test_a_non_finite_array_is_constructible_but_not_readable(self) -> None:
        source = NumpySource(np.array([np.inf]))
        self.assertEqual(len(source.partitions()), 1)
        iterator = source.batches(source.partitions()[0])
        self.assertIs(code_of(lambda: next(iterator)), CODES.NON_FINITE_VALUE)

    def test_the_fingerprint_digest_is_a_sha256_of_the_documented_record(self) -> None:
        values = np.array([1.0, 2.0])
        content = hashlib.sha256(values.tobytes()).hexdigest()
        expected = reference_digest("numpy", "<f8", (2,), content)
        self.assertEqual(NumpySource(values).partitions()[0].fingerprint, expected)


class FoldTests(unittest.TestCase):
    def test_fold_merges_partition_states_in_canonical_order(self) -> None:
        source = ListSource([[np.zeros(2), np.zeros(3)], [], [np.zeros(1)]])
        state = fold(source, TraceState)
        self.assertEqual(
            state.events,
            (
                ("merge", 2),
                ("update", 2),
                ("update", 3),
                ("merge", 0),
                ("merge", 1),
                ("update", 1),
            ),
        )

    def test_fold_partition_updates_with_every_batch_in_order(self) -> None:
        source = ListSource([[np.zeros(2), np.zeros(3)]])
        (partition,) = source.partitions()
        state = fold_partition(source, partition, TraceState)
        self.assertEqual(state.events, (("update", 2), ("update", 3)))

    def test_max_rows_is_forwarded_and_defaults_to_the_documented_bound(self) -> None:
        source = ListSource([[np.zeros(1)], [np.zeros(1)]])
        fold(source, TraceState)
        self.assertEqual(source.max_rows_seen, [DEFAULT_MAX_ROWS, DEFAULT_MAX_ROWS])
        source = ListSource([[np.zeros(1)], [np.zeros(1)]])
        fold(source, TraceState, max_rows=17)
        self.assertEqual(source.max_rows_seen, [17, 17])
        source = ListSource([[np.zeros(1)]])
        fold_partition(source, source.partitions()[0], TraceState, max_rows=5)
        self.assertEqual(source.max_rows_seen, [5])

    def test_an_invalid_bound_is_refused_before_the_source_is_touched(self) -> None:
        source = ListSource([[np.zeros(1)]])
        for bad, error in ((0, ValueError), (True, TypeError)):
            with self.subTest(bad=bad), self.assertRaises(error):
                fold(source, TraceState, max_rows=bad)  # type: ignore[arg-type]
        self.assertEqual(source.partition_calls, 0)

    def test_a_source_without_partitions_folds_to_the_empty_state(self) -> None:
        self.assertEqual(fold(ListSource([]), ExponentialState), ExponentialState.empty())

    def test_fold_equals_a_direct_update_with_the_concatenated_values(self) -> None:
        values = np.array([0.5, 2.0, 3.25, 1e10, 7.0, 0.0, 9.5])
        direct = NormalState.empty().update(values)
        layout = [[values[:2], values[2:3]], [values[3:5]], [], [values[5:]]]
        self.assertEqual(fold(ListSource(layout), NormalState).to_bytes(), direct.to_bytes())
        self.assertEqual(
            fold(NumpySource(values), NormalState, max_rows=2).to_bytes(), direct.to_bytes()
        )

    def test_a_refused_batch_raises_and_returns_nothing(self) -> None:
        good, bad = np.array([1.0, 2.0]), np.array([1.0, np.nan])
        source = ListSource([[good], [good, bad], [good]])
        with self.assertRaises(ScaleStateError):
            fold(source, ExponentialState)
        # The sources hand out the same arrays again: a fresh fold is unaffected.
        clean = ListSource([[good], [good], [good]])
        self.assertEqual(fold(clean, ExponentialState).count, 6)

    def test_fold_of_an_array_source_reaches_the_policy_refusal(self) -> None:
        source = NumpySource(np.array([1.0, 2.0, np.inf, 3.0]))
        with self.assertRaises(ScaleSourceError) as context:
            fold(source, ExponentialState, max_rows=2)
        self.assertIs(context.exception.code, CODES.NON_FINITE_VALUE)


if __name__ == "__main__":
    unittest.main()
