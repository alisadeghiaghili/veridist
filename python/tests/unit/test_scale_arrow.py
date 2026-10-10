"""Arrow value policy, the Arrow source and its single-use rule."""

from __future__ import annotations

import re
import unittest
from collections.abc import Callable
from typing import Any

import numpy as np
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import StreamExporter, failing_reader, float_column
from veridist.scale._arrow import ArrowSource, arrow_float64
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._one_pass import ExponentialState, NormalState
from veridist.scale._sources import DEFAULT_MAX_ROWS, FingerprintLevel, NumpySource, Partition, fold

CODES = ScaleSourceErrorCode


def code_of(call: Callable[[], object]) -> ScaleSourceErrorCode:
    try:
        call()
    except ScaleSourceError as error:
        return error.code
    raise AssertionError("the call was not refused")


def all_batches(source: ArrowSource, max_rows: int = DEFAULT_MAX_ROWS) -> list[np.ndarray]:
    (partition,) = source.partitions()
    return list(source.batches(partition, max_rows=max_rows))


class ArrowValuePolicyTests(unittest.TestCase):
    def test_float64_without_nulls_is_a_read_only_zero_copy_view(self) -> None:
        array = pa.array(np.array([1.5, -2.0, 0.25]))
        result = arrow_float64(array)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.tolist(), [1.5, -2.0, 0.25])
        self.assertTrue(np.shares_memory(result, array.to_numpy(zero_copy_only=True)))
        self.assertFalse(result.flags["OWNDATA"])
        self.assertFalse(result.flags["WRITEABLE"])

    def test_a_sliced_array_is_a_view_of_the_parent_buffer(self) -> None:
        parent = pa.array(np.arange(10.0))
        result = arrow_float64(parent.slice(3, 4))
        self.assertEqual(result.tolist(), [3.0, 4.0, 5.0, 6.0])
        self.assertTrue(np.shares_memory(result, parent.to_numpy(zero_copy_only=True)))

    def test_a_validity_buffer_without_nulls_is_not_a_null(self) -> None:
        array = pa.array(np.arange(3.0), mask=np.array([False, False, False]))
        self.assertEqual(array.null_count, 0)
        self.assertEqual(arrow_float64(array).tolist(), [0.0, 1.0, 2.0])

    def test_float32_is_widened_exactly(self) -> None:
        array = pa.array(np.array([0.1, -3.4028235e38, 1.0e-45], dtype=np.float32))
        result = arrow_float64(array)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result[0], 0.10000000149011612)
        self.assertEqual(result.tolist(), [float(v) for v in array.to_numpy()])

    def test_integers_widen_when_exact(self) -> None:
        limit = 2**53
        for dtype, values in (
            (pa.int8(), [-128, 0, 127]),
            (pa.uint8(), [0, 255]),
            (pa.int16(), [-32768, 32767]),
            (pa.uint16(), [0, 65535]),
            (pa.int32(), [-(2**31), 2**31 - 1]),
            (pa.uint32(), [0, 2**32 - 1]),
            (pa.int64(), [-limit, 0, limit]),
            (pa.uint64(), [0, limit]),
        ):
            with self.subTest(dtype=str(dtype)):
                result = arrow_float64(pa.array(values, dtype))
                self.assertEqual(result.dtype, np.float64)
                self.assertEqual(result.tolist(), [float(v) for v in values])

    def test_integers_beyond_two_to_the_53_are_a_lossy_cast(self) -> None:
        limit = 2**53
        for dtype, values in (
            (pa.int64(), [limit + 1]),
            (pa.int64(), [-limit - 1, 0]),
            (pa.int64(), [0, limit + 1, 3]),
            (pa.uint64(), [limit + 1]),
            (pa.uint64(), [2**64 - 1]),
        ):
            with self.subTest(dtype=str(dtype), values=values):
                array = pa.array(values, dtype)
                self.assertIs(code_of(lambda a=array: arrow_float64(a)), CODES.LOSSY_CAST)

    def test_a_null_is_refused_for_every_accepted_type_and_never_dropped(self) -> None:
        for dtype in (pa.float64(), pa.float32(), pa.int8(), pa.int64(), pa.uint64()):
            with self.subTest(dtype=str(dtype)):
                array = pa.array([1, None, 3], dtype)
                self.assertIs(code_of(lambda a=array: arrow_float64(a)), CODES.NULL_VALUE)

    def test_nan_and_infinities_use_their_own_codes(self) -> None:
        for values, code in (
            ([1.0, float("nan")], CODES.NAN_VALUE),
            ([float("inf")], CODES.NON_FINITE_VALUE),
            ([-float("inf"), 2.0], CODES.NON_FINITE_VALUE),
            ([float("inf"), float("nan")], CODES.NAN_VALUE),
        ):
            for dtype in (pa.float64(), pa.float32()):
                with self.subTest(values=values, dtype=str(dtype)):
                    array = pa.array(values, dtype)
                    self.assertIs(code_of(lambda a=array: arrow_float64(a)), code)

    def test_every_other_type_is_unsupported(self) -> None:
        for array in (
            pa.array(["1.0"]),
            pa.array([True, False]),
            pa.array(np.array([1.0], dtype=np.float16)),
            pa.array([1], pa.decimal128(10, 2)),
            pa.array([1], pa.timestamp("s")),
            pa.array([1], pa.date32()),
            pa.array([1.0]).dictionary_encode(),
            pa.array([[1.0]]),
            pa.array([None, None]),
            pa.array([b"1"]),
        ):
            with self.subTest(dtype=str(array.type)):
                self.assertIs(code_of(lambda a=array: arrow_float64(a)), CODES.UNSUPPORTED_DTYPE)

    def test_the_type_is_judged_before_the_nulls(self) -> None:
        array = pa.array([None, "a"])
        self.assertIs(code_of(lambda: arrow_float64(array)), CODES.UNSUPPORTED_DTYPE)

    def test_a_refusal_message_is_the_bare_code(self) -> None:
        with self.assertRaises(ScaleSourceError) as context:
            arrow_float64(pa.array([123456.789, None]))
        self.assertEqual(str(context.exception), "NULL_VALUE")

    def test_empty_arrays_are_accepted(self) -> None:
        for dtype in (pa.float64(), pa.float32(), pa.int64()):
            with self.subTest(dtype=str(dtype)):
                self.assertEqual(arrow_float64(pa.array([], dtype)).shape, (0,))


class ArrowSourceInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
        self.table = pa.table({"x": self.values, "y": np.arange(10)})

    def test_every_supported_input_gives_the_same_values(self) -> None:
        inputs: dict[str, Any] = {
            "table": self.table,
            "batch": self.table.to_batches(max_chunksize=100)[0],
            "reader": pa.RecordBatchReader.from_stream(self.table),
            "exporter": StreamExporter(self.table),
        }
        for name, data in inputs.items():
            with self.subTest(input=name):
                batches = all_batches(ArrowSource(data, "x"), max_rows=4)
                self.assertEqual([len(b) for b in batches], [4, 4, 2])
                self.assertEqual(np.concatenate(batches).tolist(), self.values.tolist())

    def test_the_projected_column_is_the_one_named(self) -> None:
        batches = all_batches(ArrowSource(self.table, "y"))
        self.assertEqual(np.concatenate(batches).tolist(), [float(i) for i in range(10)])

    def test_the_column_is_found_in_any_position(self) -> None:
        table = pa.table({"a": [1.0], "b": [2.0], "c": [3.0]})
        self.assertEqual(all_batches(ArrowSource(table, "c"))[0].tolist(), [3.0])
        self.assertEqual(all_batches(ArrowSource(table, "b"))[0].tolist(), [2.0])
        self.assertEqual(all_batches(ArrowSource(table, "a"))[0].tolist(), [1.0])

    def test_the_partition_is_one_non_resumable_partition(self) -> None:
        (partition,) = ArrowSource(self.table, "x").partitions()
        self.assertEqual(partition, Partition("arrow", 0, 10, None, FingerprintLevel.NONE))
        self.assertFalse(partition.resumable)
        (batch_partition,) = ArrowSource(self.table.to_batches()[0], "x").partitions()
        self.assertEqual(batch_partition.rows, 10)

    def test_the_row_count_of_a_stream_is_unknown(self) -> None:
        for data in (pa.RecordBatchReader.from_stream(self.table), StreamExporter(self.table)):
            with self.subTest(kind=type(data).__name__):
                (partition,) = ArrowSource(data, "x").partitions()
                self.assertIsNone(partition.rows)
                self.assertIsNone(partition.fingerprint)
                self.assertIs(partition.fingerprint_level, FingerprintLevel.NONE)

    def test_a_table_and_a_batch_can_be_read_again_but_a_stream_cannot(self) -> None:
        for data in (self.table, self.table.to_batches()[0]):
            source = ArrowSource(data, "x")
            first, second = all_batches(source), all_batches(source)
            self.assertEqual(np.concatenate(first).tolist(), np.concatenate(second).tolist())
        for stream in (pa.RecordBatchReader.from_stream(self.table), StreamExporter(self.table)):
            source = ArrowSource(stream, "x")
            (partition,) = source.partitions()
            self.assertEqual(len(list(source.batches(partition))), 1)
            for _ in range(2):
                refused = code_of(lambda s=source, p=partition: s.batches(p))
                self.assertIs(refused, CODES.SOURCE_CONSUMED)

    def test_a_stream_is_marked_consumed_when_batches_is_called(self) -> None:
        source = ArrowSource(pa.RecordBatchReader.from_stream(self.table), "x")
        (partition,) = source.partitions()
        iterator = source.batches(partition)
        self.assertIs(code_of(lambda: source.batches(partition)), CODES.SOURCE_CONSUMED)
        self.assertEqual(len(list(iterator)), 1)

    def test_the_stream_object_is_opened_once_at_construction(self) -> None:
        exporter = StreamExporter(self.table)
        source = ArrowSource(exporter, "x")
        self.assertEqual(exporter.calls, 1)
        all_batches(source)
        self.assertEqual(exporter.calls, 1)

    def test_batches_are_zero_copy_slices_of_the_stored_column(self) -> None:
        stored = self.table.column("x").chunk(0).to_numpy(zero_copy_only=True)
        for max_rows in (3, 100):
            for batch in all_batches(ArrowSource(self.table, "x"), max_rows=max_rows):
                with self.subTest(max_rows=max_rows):
                    self.assertTrue(np.shares_memory(batch, stored))
                    self.assertFalse(batch.flags["OWNDATA"])

    def test_chunk_layout_and_max_rows_decide_the_batch_sizes(self) -> None:
        chunked = pa.concat_tables(
            [
                pa.table({"x": float_column(np.arange(6.0))}),
                pa.table({"x": float_column(np.arange(6.0, 10.0))}),
            ]
        )
        self.assertEqual(chunked.column("x").num_chunks, 2)
        for max_rows, sizes in (
            (4, [4, 2, 4]),
            (6, [6, 4]),
            (10, [6, 4]),
            (1, [1] * 10),
            (3, [3, 3, 3, 1]),
        ):
            with self.subTest(max_rows=max_rows):
                batches = all_batches(ArrowSource(chunked, "x"), max_rows=max_rows)
                self.assertEqual([len(b) for b in batches], sizes)
                self.assertEqual(np.concatenate(batches).tolist(), [float(i) for i in range(10)])

    def test_empty_inputs_yield_no_batches(self) -> None:
        empty = pa.table({"x": pa.array([], pa.float64())})
        empty_batch = pa.record_batch([pa.array([], pa.float64())], names=["x"])
        self.assertEqual(all_batches(ArrowSource(empty, "x")), [])
        self.assertEqual(all_batches(ArrowSource(empty_batch, "x")), [])
        self.assertEqual(ArrowSource(empty, "x").partitions()[0].rows, 0)
        self.assertEqual(ArrowSource(empty_batch, "x").partitions()[0].rows, 0)

    def test_the_default_bound_applies(self) -> None:
        table = pa.table({"x": np.zeros(DEFAULT_MAX_ROWS + 1)})
        sizes = [len(b) for b in all_batches(ArrowSource(table, "x"))]
        self.assertEqual(sizes, [DEFAULT_MAX_ROWS, 1])

    def test_a_missing_or_ambiguous_column_is_refused_at_construction(self) -> None:
        duplicated = pa.Table.from_arrays([pa.array([1.0]), pa.array([2.0])], names=["x", "x"])
        stream = pa.RecordBatchReader.from_stream(self.table)
        for data in (self.table, duplicated, stream, StreamExporter(self.table)):
            with self.subTest(kind=type(data).__name__):
                refused = code_of(lambda d=data: ArrowSource(d, "missing"))
                self.assertIs(refused, CODES.COLUMN_NOT_FOUND)
        self.assertIs(code_of(lambda: ArrowSource(duplicated, "x")), CODES.COLUMN_NOT_FOUND)
        self.assertIs(code_of(lambda: ArrowSource(self.table, "X")), CODES.COLUMN_NOT_FOUND)

    def test_programmer_errors(self) -> None:
        data_message = (
            "data must be a pyarrow Table, RecordBatch or RecordBatchReader, "
            "or expose __arrow_c_stream__"
        )
        for bad in ("x.parquet", [1.0], {"x": [1.0]}, None, 5, object()):
            with (
                self.subTest(bad=type(bad).__name__),
                self.assertRaisesRegex(TypeError, f"^{re.escape(data_message)}$"),
            ):
                ArrowSource(bad, "x")
        for column in ("", None, 3, b"x"):
            with (
                self.subTest(column=column),
                self.assertRaisesRegex(TypeError, "^column must be a non-empty string$"),
            ):
                ArrowSource(self.table, column)  # type: ignore[arg-type]

    def test_a_stream_exporter_that_fails_is_unreadable(self) -> None:
        class Broken:
            def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
                raise pa.ArrowInvalid("secret detail")

        with self.assertRaises(ScaleSourceError) as context:
            ArrowSource(Broken(), "x")
        self.assertIs(context.exception.code, CODES.SOURCE_UNREADABLE)
        self.assertNotIn("secret", str(context.exception))
        self.assertIsInstance(context.exception.__cause__, pa.ArrowInvalid)

    def test_arguments_are_validated_when_batches_is_called(self) -> None:
        source = ArrowSource(self.table, "x")
        (partition,) = source.partitions()
        for bad, error in ((0, ValueError), (True, TypeError), (2.5, TypeError)):
            with self.subTest(max_rows=bad), self.assertRaises(error):
                source.batches(partition, max_rows=bad)  # type: ignore[arg-type]
        stream = ArrowSource(pa.RecordBatchReader.from_stream(self.table), "x")
        (stream_partition,) = stream.partitions()
        with self.assertRaises(ValueError):
            stream.batches(stream_partition, max_rows=0)
        # A refused call does not consume the stream.
        self.assertEqual(len(list(stream.batches(stream_partition))), 1)
        for foreign in (Partition("other", 0, None, None, FingerprintLevel.NONE), "arrow", None):
            with (
                self.subTest(partition=foreign),
                self.assertRaisesRegex(ValueError, "^partition does not belong to this source$"),
            ):
                source.batches(foreign)  # type: ignore[arg-type]


class ArrowSourceFailureTests(unittest.TestCase):
    schema = pa.schema([("x", pa.float64())])

    def batch(self, values: list[Any], dtype: Any = None) -> pa.RecordBatch:
        return pa.record_batch([pa.array(values, dtype or pa.float64())], names=["x"])

    def test_a_refusal_in_the_second_batch_comes_after_the_first(self) -> None:
        table = pa.Table.from_batches([self.batch([1.0, 2.0]), self.batch([3.0, None])])
        source = ArrowSource(table, "x")
        (partition,) = source.partitions()
        iterator = source.batches(partition)
        self.assertEqual(next(iterator).tolist(), [1.0, 2.0])
        self.assertIs(code_of(lambda: next(iterator)), CODES.NULL_VALUE)

    def test_a_failing_stream_is_unreadable_and_keeps_no_detail(self) -> None:
        reader = failing_reader(self.schema, [self.batch([1.0, 2.0])])
        source = ArrowSource(reader, "x")
        (partition,) = source.partitions()
        iterator = source.batches(partition)
        self.assertEqual(next(iterator).tolist(), [1.0, 2.0])
        with self.assertRaises(ScaleSourceError) as context:
            next(iterator)
        self.assertIs(context.exception.code, CODES.SOURCE_UNREADABLE)
        self.assertEqual(str(context.exception), "SOURCE_UNREADABLE")
        self.assertIsInstance(context.exception.__cause__, pa.ArrowException)

    def test_a_policy_refusal_is_not_rewrapped_as_unreadable(self) -> None:
        source = ArrowSource(pa.table({"x": ["a", "b"]}), "x")
        (partition,) = source.partitions()
        iterator = source.batches(partition)
        self.assertIs(code_of(lambda: next(iterator)), CODES.UNSUPPORTED_DTYPE)

    def test_a_refused_stream_batch_never_reaches_the_state(self) -> None:
        good = self.batch([1.0, 2.0])
        bad = self.batch([float("nan")])
        table = pa.Table.from_batches([good, bad, good])
        with self.assertRaises(ScaleSourceError) as context:
            fold(ArrowSource(table, "x"), ExponentialState, max_rows=1)
        self.assertIs(context.exception.code, CODES.NAN_VALUE)

    def test_the_fold_of_every_input_equals_the_array_fold(self) -> None:
        values = np.array([0.0, 1.5, 2.5, 1e10, 3.25, 7.0, 0.125, 9.0, 1e-300])
        table = pa.table({"x": values})
        reference = fold(NumpySource(values), NormalState).to_bytes()
        factories: dict[str, Callable[[], Any]] = {
            "table": lambda: table,
            "batch": lambda: table.to_batches()[0],
            "reader": lambda: pa.RecordBatchReader.from_stream(table),
            "exporter": lambda: StreamExporter(table),
        }
        for name, make in factories.items():
            for max_rows in (1, 4, 100):
                with self.subTest(kind=name, max_rows=max_rows):
                    state = fold(ArrowSource(make(), "x"), NormalState, max_rows=max_rows)
                    self.assertEqual(state.to_bytes(), reference)


if __name__ == "__main__":
    unittest.main()
