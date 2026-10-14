"""The value policy of every frame adapter is the Arrow value policy, case by case."""

from __future__ import annotations

import unittest
from typing import Any

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa

from tests.unit.scale_frame_fixtures import (
    FRAME_SOURCES,
    all_batches,
    code_of,
    values_of,
)
from veridist.scale._duckdb import DuckDBSource
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._one_pass import ExponentialState
from veridist.scale._polars import PolarsSource
from veridist.scale._sources import fold

CODES = ScaleSourceErrorCode
LIMIT = 2**53
NAN = float("nan")
INF = float("inf")


def column(values: list[Any], dtype: Any) -> pa.Table:
    return pa.table({"x": pa.array(values, dtype)})


ACCEPTED: list[tuple[str, pa.Table, list[float]]] = [
    ("float64", column([1.5, -2.0, 0.25], pa.float64()), [1.5, -2.0, 0.25]),
    (
        "float32 is widened exactly",
        column([0.1, -3.4028235e38, 1.0e-45], pa.float32()),
        [float(v) for v in np.array([0.1, -3.4028235e38, 1.0e-45], dtype=np.float32)],
    ),
    ("int8", column([-128, 0, 127], pa.int8()), [-128.0, 0.0, 127.0]),
    ("uint8", column([0, 255], pa.uint8()), [0.0, 255.0]),
    ("int16", column([-32768, 32767], pa.int16()), [-32768.0, 32767.0]),
    ("uint16", column([0, 65535], pa.uint16()), [0.0, 65535.0]),
    ("int32", column([-(2**31), 2**31 - 1], pa.int32()), [-(2.0**31), 2.0**31 - 1]),
    ("uint32", column([0, 2**32 - 1], pa.uint32()), [0.0, 2.0**32 - 1]),
    ("int64 at the limit", column([-LIMIT, 0, LIMIT], pa.int64()), [-(2.0**53), 0.0, 2.0**53]),
    ("uint64 at the limit", column([0, LIMIT], pa.uint64()), [0.0, 2.0**53]),
    ("empty float64", column([], pa.float64()), []),
    ("empty int64", column([], pa.int64()), []),
]

REFUSED: list[tuple[str, pa.Table, ScaleSourceErrorCode]] = [
    ("int64 above the limit", column([LIMIT + 1], pa.int64()), CODES.LOSSY_CAST),
    ("int64 below the limit", column([-LIMIT - 1, 0], pa.int64()), CODES.LOSSY_CAST),
    ("int64 late above the limit", column([0, LIMIT + 1, 3], pa.int64()), CODES.LOSSY_CAST),
    ("uint64 above the limit", column([LIMIT + 1], pa.uint64()), CODES.LOSSY_CAST),
    ("uint64 maximum", column([2**64 - 1], pa.uint64()), CODES.LOSSY_CAST),
    ("float64 null", column([1.0, None, 3.0], pa.float64()), CODES.NULL_VALUE),
    ("float32 null", column([1.0, None], pa.float32()), CODES.NULL_VALUE),
    ("int8 null", column([1, None], pa.int8()), CODES.NULL_VALUE),
    ("int64 null", column([None, 2], pa.int64()), CODES.NULL_VALUE),
    ("uint64 null", column([None], pa.uint64()), CODES.NULL_VALUE),
    ("float64 NaN", column([1.0, NAN], pa.float64()), CODES.NAN_VALUE),
    ("float32 NaN", column([NAN], pa.float32()), CODES.NAN_VALUE),
    ("float64 infinity", column([INF], pa.float64()), CODES.NON_FINITE_VALUE),
    ("float64 negative infinity", column([-INF, 2.0], pa.float64()), CODES.NON_FINITE_VALUE),
    ("float32 infinity", column([INF], pa.float32()), CODES.NON_FINITE_VALUE),
    ("string", column(["1.0"], pa.string()), CODES.UNSUPPORTED_DTYPE),
    ("boolean", column([True, False], pa.bool_()), CODES.UNSUPPORTED_DTYPE),
    ("decimal", column([1], pa.decimal128(10, 2)), CODES.UNSUPPORTED_DTYPE),
    ("timestamp", column([1], pa.timestamp("s")), CODES.UNSUPPORTED_DTYPE),
    ("date", column([1], pa.date32()), CODES.UNSUPPORTED_DTYPE),
    ("list", column([[1.0]], pa.list_(pa.float64())), CODES.UNSUPPORTED_DTYPE),
    ("binary", column([b"1"], pa.binary()), CODES.UNSUPPORTED_DTYPE),
]


class FrameValuePolicyTests(unittest.TestCase):
    def test_accepted_columns_give_exact_float64_values_through_every_adapter(self) -> None:
        for adapter, make in FRAME_SOURCES.items():
            for name, table, expected in ACCEPTED:
                with self.subTest(adapter=adapter, case=name):
                    source = make(table, "x")
                    batches = all_batches(source)
                    self.assertTrue(all(b.dtype == np.float64 and b.ndim == 1 for b in batches))
                    self.assertEqual(values_of(source), expected)

    def test_refused_columns_use_the_arrow_codes_through_every_adapter(self) -> None:
        for adapter, make in FRAME_SOURCES.items():
            for name, table, code in REFUSED:
                with self.subTest(adapter=adapter, case=name):
                    source = make(table, "x")
                    (partition,) = source.partitions()
                    refused = code_of(lambda s=source, p=partition: list(s.batches(p)))
                    self.assertIs(refused, code)

    def test_nan_is_judged_before_infinity_within_one_batch(self) -> None:
        table = column([INF, NAN], pa.float64())
        for adapter in ("polars frame", "duckdb relation"):
            with self.subTest(adapter=adapter):
                source = FRAME_SOURCES[adapter](table, "x")
                (partition,) = source.partitions()
                refused = code_of(lambda s=source, p=partition: list(s.batches(p)))
                self.assertIs(refused, CODES.NAN_VALUE)

    def test_a_refusal_message_is_the_bare_code_and_carries_no_value(self) -> None:
        table = column([123456.789, None], pa.float64())
        for adapter, make in FRAME_SOURCES.items():
            with self.subTest(adapter=adapter):
                source = make(table, "x")
                (partition,) = source.partitions()
                with self.assertRaises(ScaleSourceError) as context:
                    list(source.batches(partition))
                self.assertEqual(str(context.exception), "NULL_VALUE")
                self.assertIsNone(context.exception.hint)

    def test_a_missing_column_is_refused_at_construction_through_every_adapter(self) -> None:
        table = pa.table({"x": [1.0], "y": [2.0]})
        for adapter, make in FRAME_SOURCES.items():
            for name in ("missing", "X", "x ", "xy"):
                with self.subTest(adapter=adapter, column=name):
                    refused = code_of(lambda m=make, n=name: m(table, n))
                    self.assertIs(refused, CODES.COLUMN_NOT_FOUND)

    def test_the_column_is_found_in_any_position(self) -> None:
        table = pa.table({"a": [1.0], "b": [2.0], "c": [3.0]})
        for adapter, make in FRAME_SOURCES.items():
            with self.subTest(adapter=adapter):
                self.assertEqual(values_of(make(table, "a")), [1.0])
                self.assertEqual(values_of(make(table, "b")), [2.0])
                self.assertEqual(values_of(make(table, "c")), [3.0])

    def test_only_the_projected_column_is_judged(self) -> None:
        table = pa.table({"bad": ["a", "b"], "x": [1.0, 2.0], "worse": [None, None]})
        for adapter, make in FRAME_SOURCES.items():
            with self.subTest(adapter=adapter):
                self.assertEqual(values_of(make(table, "x")), [1.0, 2.0])

    def test_names_that_look_like_patterns_or_sql_are_taken_literally(self) -> None:
        names = ["^a$", "a", "a.b", 'x"y', "select", "A b", "[x]", ".*", "tab.col"]
        table = pa.table({name: [float(index)] for index, name in enumerate(names)})
        for adapter, make in FRAME_SOURCES.items():
            for index, name in enumerate(names):
                with self.subTest(adapter=adapter, column=name):
                    self.assertEqual(values_of(make(table, name)), [float(index)])

    def test_a_refused_batch_never_reaches_the_state(self) -> None:
        table = column([1.0, 2.0, NAN, 4.0], pa.float64())
        for adapter, make in FRAME_SOURCES.items():
            with self.subTest(adapter=adapter):
                with self.assertRaises(ScaleSourceError) as context:
                    fold(make(table, "x"), ExponentialState, max_rows=1)
                self.assertIs(context.exception.code, CODES.NAN_VALUE)

    def test_hugeint_and_decimal_columns_of_a_query_are_unsupported(self) -> None:
        connection = duckdb.connect()
        for sql in (
            "select 1::hugeint as x",
            "select 1.5::decimal(10, 2) as x",
            "select 'a' as x",
            "select true as x",
        ):
            with self.subTest(sql=sql):
                source = DuckDBSource((connection, sql), "x")
                (partition,) = source.partitions()
                iterator = source.batches(partition)
                self.assertIs(code_of(lambda i=iterator: next(i)), CODES.UNSUPPORTED_DTYPE)

    def test_a_query_column_of_a_float_or_an_integer_type_is_accepted(self) -> None:
        connection = duckdb.connect()
        for sql in (
            "select 1.5::double as x",
            "select 1.5::float as x",
            "select 3::tinyint as x",
            "select 3::utinyint as x",
            "select 3::bigint as x",
            "select 3::ubigint as x",
        ):
            with self.subTest(sql=sql):
                self.assertEqual(len(values_of(DuckDBSource((connection, sql), "x"))), 1)

    def test_a_polars_null_is_not_a_nan_and_a_nan_is_not_a_null(self) -> None:
        with_null = pl.DataFrame({"x": [1.0, None]})
        with_nan = pl.DataFrame({"x": [1.0, float("nan")]})
        for frame_of in (lambda f: f, lambda f: f.lazy()):
            with self.subTest(frame=frame_of(with_null).__class__.__name__):
                for data, code in ((with_null, CODES.NULL_VALUE), (with_nan, CODES.NAN_VALUE)):
                    source = PolarsSource(frame_of(data), "x")
                    (partition,) = source.partitions()
                    refused = code_of(lambda s=source, p=partition: list(s.batches(p)))
                    self.assertIs(refused, code)


if __name__ == "__main__":
    unittest.main()
