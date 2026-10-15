"""The DuckDB source: inputs, partition, batching, re-reading and failures."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import duckdb
import numpy as np
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_frame_fixtures import (
    all_batches,
    code_of,
    connect,
    sql_path,
    values_of,
)
from veridist.scale import _duckdb
from veridist.scale._arrow import ParquetSource
from veridist.scale._duckdb import DuckDBSource, _quoted, to_arrow_reader
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._one_pass import NormalState
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    FingerprintLevel,
    NumpySource,
    Partition,
    fold,
)

CODES = ScaleSourceErrorCode
VALUES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
SELECT = "select (range + 0.5)::double as x, range::bigint as y, 'a' as s from range(10)"


class DuckDBInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = connect()
        self.addCleanup(self.connection.close)

    def test_a_relation_and_a_connection_with_sql_give_the_same_values(self) -> None:
        inputs: dict[str, Any] = {
            "relation": self.connection.sql(SELECT),
            "pair": (self.connection, SELECT),
        }
        for name, source in inputs.items():
            with self.subTest(input=name):
                batches = all_batches(DuckDBSource(source, "x"), max_rows=4)
                self.assertEqual([len(b) for b in batches], [4, 4, 2])
                self.assertEqual(np.concatenate(batches).tolist(), VALUES)

    def test_the_projected_column_is_the_one_named(self) -> None:
        source = DuckDBSource(self.connection.sql(SELECT), "y")
        self.assertEqual(values_of(source), [float(i) for i in range(10)])

    def test_the_partition_is_one_non_resumable_partition_of_unknown_size(self) -> None:
        (partition,) = DuckDBSource(self.connection.sql(SELECT), "x").partitions()
        self.assertEqual(partition, Partition("duckdb", 0, None, None, FingerprintLevel.NONE))
        self.assertFalse(partition.resumable)

    def test_an_empty_result_yields_no_batches(self) -> None:
        source = DuckDBSource((self.connection, "select 1.5::double as x where false"), "x")
        self.assertEqual(all_batches(source), [])

    def test_a_relation_can_be_read_again(self) -> None:
        source = DuckDBSource(self.connection.sql(SELECT), "x")
        for _ in range(3):
            self.assertEqual(values_of(source, max_rows=3), VALUES)

    def test_a_relation_is_run_again_for_every_read(self) -> None:
        self.connection.sql("create table counter(v double)")
        source = DuckDBSource(self.connection.sql("select v as x from counter"), "x")
        self.assertEqual(values_of(source), [])
        self.connection.sql("insert into counter values (1.5), (2.5)")
        self.assertEqual(sorted(values_of(source)), [1.5, 2.5])

    def test_the_bound_caps_every_batch(self) -> None:
        relation = self.connection.sql("select range::double as x from range(1000)")
        for max_rows in (1, 7, 100, 999, 1000, 5000):
            with self.subTest(max_rows=max_rows):
                batches = all_batches(DuckDBSource(relation, "x"), max_rows=max_rows)
                self.assertLessEqual(max(len(b) for b in batches), max_rows)
                self.assertEqual(
                    sorted(np.concatenate(batches).tolist()), np.arange(1000.0).tolist()
                )

    def test_the_default_bound_applies(self) -> None:
        total = DEFAULT_MAX_ROWS + 1
        relation = self.connection.sql(f"select 0.0::double as x from range({total})")
        sizes = [len(b) for b in all_batches(DuckDBSource(relation, "x"))]
        self.assertEqual(sum(sizes), total)
        self.assertLessEqual(max(sizes), DEFAULT_MAX_ROWS)

    def test_the_bound_is_passed_to_the_reader(self) -> None:
        recorded: list[int] = []

        class Spy:
            def to_arrow_reader(self, batch_size: int) -> Any:
                recorded.append(batch_size)
                return iter(())

        self.assertEqual(list(to_arrow_reader(Spy(), 11)), [])
        self.assertEqual(recorded, [11])

    def test_the_reader_is_asked_for_batches_of_the_bound(self) -> None:
        with mock.patch.object(_duckdb, "to_arrow_reader", wraps=to_arrow_reader) as spy:
            values_of(DuckDBSource(self.connection.sql(SELECT), "x"), max_rows=5)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.args[1], 5)

    def test_a_scan_is_projected_before_it_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_parquet(Path(directory) / "a.parquet", {"x": [1.0, 2.0], "s": ["a", "b"]})
            relation = self.connection.sql(f"select * from read_parquet({sql_path(path)})")
            source = DuckDBSource(relation, "x")
            self.assertEqual(values_of(source), [1.0, 2.0])
            plan = relation.project(_quoted("x")).explain()
            self.assertIn("Projections: x", plan)

    def test_a_filter_and_a_join_of_a_query_are_applied_before_reading(self) -> None:
        sql = (
            "select a.x from (values (1, 10.0::double), (2, 20.0::double), "
            "(3, 30.0::double), (4, 40.0::double)) as a(k, x) "
            "join (values (2), (4)) as b(k) on a.k = b.k where a.x > 25.0::double"
        )
        self.assertEqual(values_of(DuckDBSource((self.connection, sql), "x")), [40.0])

    def test_the_quoted_identifier_doubles_quotes(self) -> None:
        self.assertEqual(_quoted("x"), '"x"')
        self.assertEqual(_quoted('a"b'), '"a""b"')
        self.assertEqual(_quoted(""), '""')

    def test_programmer_errors(self) -> None:
        source_message = (
            "source must be a duckdb relation or a (connection, sql) pair "
            "of a duckdb connection and a query string"
        )
        wrong_pairs: list[Any] = [
            (self.connection,),
            (self.connection, SELECT, 1),
            (SELECT, self.connection),
            (self.connection, b"select 1"),
            [self.connection, SELECT],
            ("connection", SELECT),
        ]
        for bad in ["x.parquet", [1.0], None, 5, object(), pa.table({"x": [1.0]}), *wrong_pairs]:
            with (
                self.subTest(bad=repr(bad)[:30]),
                self.assertRaisesRegex(TypeError, f"^{re.escape(source_message)}$"),
            ):
                DuckDBSource(bad, "x")
        relation = self.connection.sql(SELECT)
        for column in ("", None, 3, b"x"):
            with (
                self.subTest(column=column),
                self.assertRaisesRegex(TypeError, "^column must be a non-empty string$"),
            ):
                DuckDBSource(relation, column)  # type: ignore[arg-type]

    def test_a_statement_that_returns_no_rows_is_a_programmer_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "^sql must be a query that returns rows$"):
            DuckDBSource((self.connection, "create table t(a double)"), "a")

    def test_arguments_are_validated_when_batches_is_called(self) -> None:
        source = DuckDBSource(self.connection.sql(SELECT), "x")
        (partition,) = source.partitions()
        for bad, error in ((0, ValueError), (True, TypeError), (2.5, TypeError), (-1, ValueError)):
            with self.subTest(max_rows=bad), self.assertRaises(error):
                source.batches(partition, max_rows=bad)  # type: ignore[arg-type]
        for foreign in (Partition("other", 0, None, None, FingerprintLevel.NONE), "duckdb", None):
            with (
                self.subTest(partition=foreign),
                self.assertRaisesRegex(ValueError, "^partition does not belong to this source$"),
            ):
                source.batches(foreign)  # type: ignore[arg-type]

    def test_a_missing_or_ambiguous_column_is_refused_at_construction(self) -> None:
        relation = self.connection.sql(SELECT)
        for name in ("missing", "X", "xx", "x "):
            with self.subTest(column=name):
                refused = code_of(lambda n=name: DuckDBSource(relation, n))
                self.assertIs(refused, CODES.COLUMN_NOT_FOUND)
        duplicated = self.connection.sql("select 1.0 as x, 2.0 as x")
        self.assertEqual(duplicated.columns, ["x", "x"])
        self.assertIs(code_of(lambda: DuckDBSource(duplicated, "x")), CODES.COLUMN_NOT_FOUND)

    def test_the_fold_of_every_input_equals_the_array_fold(self) -> None:
        values = np.array([0.0, 1.5, 2.5, 1e10, 3.25, 7.0, 0.125, 9.0, 1e-300])
        reference = fold(NumpySource(values), NormalState).to_bytes()
        relation = self.connection.from_arrow(pa.table({"x": values}))
        for max_rows in (1, 4, 100):
            with self.subTest(max_rows=max_rows):
                state = fold(DuckDBSource(relation, "x"), NormalState, max_rows=max_rows)
                self.assertEqual(state.to_bytes(), reference)

    def test_a_scan_of_a_parquet_file_folds_to_the_parquet_source_state(self) -> None:
        values = np.array([0.0, 1.5, 2.5, 1e10, 3.25, 7.0, 0.125, 9.0, 1e-300, 4.75])
        with tempfile.TemporaryDirectory() as directory:
            path = write_parquet(Path(directory) / "a.parquet", {"x": values}, row_group_size=3)
            reference = fold(ParquetSource(path, "x"), NormalState).to_bytes()
            self.assertEqual(reference, fold(NumpySource(values), NormalState).to_bytes())
            scan = f"read_parquet({sql_path(path)})"
            inputs: dict[str, Any] = {
                "read_parquet relation": self.connection.read_parquet(path.as_posix()),
                "sql relation": self.connection.sql(f"select * from {scan}"),
                "pair": (self.connection, f"select x from {scan} where x > -1.0"),
            }
            for name, data in inputs.items():
                for max_rows in (1, 4, 100):
                    with self.subTest(input=name, max_rows=max_rows):
                        state = fold(DuckDBSource(data, "x"), NormalState, max_rows=max_rows)
                        self.assertEqual(state.to_bytes(), reference)

    def test_the_order_of_the_rows_does_not_change_the_state(self) -> None:
        values = [0.5, 8.25, 3.0, 1e6, 2.0, 7.125]
        listing = ", ".join(f"({value!r}::double)" for value in values)
        forward = DuckDBSource((self.connection, f"select * from (values {listing}) t(x)"), "x")
        backward = DuckDBSource(
            (self.connection, f"select * from (values {listing}) t(x) order by x desc"), "x"
        )
        self.assertEqual(
            fold(forward, NormalState).to_bytes(), fold(backward, NormalState).to_bytes()
        )


class DuckDBFailureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.connection = connect()
        self.addCleanup(self.connection.close)

    def assert_unreadable(self, error: ScaleSourceError, cause: type[BaseException]) -> None:
        self.assertIs(error.code, CODES.SOURCE_UNREADABLE)
        self.assertEqual(str(error), "SOURCE_UNREADABLE")
        self.assertIsNone(error.hint)
        self.assertIsInstance(error.__cause__, cause)

    def test_a_query_that_does_not_parse_is_unreadable_at_construction(self) -> None:
        with self.assertRaises(ScaleSourceError) as context:
            DuckDBSource((self.connection, "selec nothing"), "x")
        self.assert_unreadable(context.exception, duckdb.Error)

    def test_a_query_of_a_missing_table_is_unreadable_at_construction(self) -> None:
        with self.assertRaises(ScaleSourceError) as context:
            DuckDBSource((self.connection, "select x from no_such_table"), "x")
        self.assert_unreadable(context.exception, duckdb.Error)

    def test_a_closed_connection_is_unreadable(self) -> None:
        other = connect()
        relation = other.sql(SELECT)
        source = DuckDBSource(relation, "x")
        (partition,) = source.partitions()
        other.close()
        with self.assertRaises(ScaleSourceError) as context:
            list(source.batches(partition))
        self.assert_unreadable(context.exception, duckdb.Error)
        with self.assertRaises(ScaleSourceError) as pair_context:
            DuckDBSource((other, SELECT), "x")
        self.assert_unreadable(pair_context.exception, duckdb.Error)

    def test_a_query_that_fails_while_running_is_unreadable_and_keeps_no_detail(self) -> None:
        sql = (
            "select case when range = 50 then error('secret detail') else 1.0 end as x "
            "from range(100)"
        )
        source = DuckDBSource((self.connection, sql), "x")
        (partition,) = source.partitions()
        with self.assertRaises(ScaleSourceError) as context:
            list(source.batches(partition))
        self.assert_unreadable(context.exception, duckdb.Error)
        self.assertNotIn("secret", str(context.exception))

    def test_an_arrow_failure_is_unreadable(self) -> None:
        source = DuckDBSource(self.connection.sql(SELECT), "x")
        (partition,) = source.partitions()
        with mock.patch.object(_duckdb, "to_arrow_reader", side_effect=pa.ArrowInvalid("detail")):
            with self.assertRaises(ScaleSourceError) as context:
                list(source.batches(partition))
        self.assert_unreadable(context.exception, pa.ArrowException)

    def test_a_missing_file_is_unreadable_when_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_parquet(Path(directory) / "a.parquet", {"x": [1.0, 2.0]})
            relation = self.connection.sql(f"select * from read_parquet({sql_path(path)})")
            source = DuckDBSource(relation, "x")
            (partition,) = source.partitions()
            path.unlink()
            with self.assertRaises(ScaleSourceError) as context:
                list(source.batches(partition))
        self.assert_unreadable(context.exception, duckdb.Error)

    def test_a_policy_refusal_is_not_rewrapped_as_unreadable(self) -> None:
        source = DuckDBSource(self.connection.sql("select 'a' as x"), "x")
        (partition,) = source.partitions()
        iterator = source.batches(partition)
        self.assertIs(code_of(lambda: next(iterator)), CODES.UNSUPPORTED_DTYPE)

    def test_a_refusal_in_a_later_batch_comes_after_the_earlier_ones(self) -> None:
        relation = self.connection.sql(
            "select case when range = 2 then 'nan'::double else range::double end as x "
            "from range(5)"
        )
        source = DuckDBSource(relation, "x")
        (partition,) = source.partitions()
        iterator = source.batches(partition, max_rows=2)
        self.assertEqual(next(iterator).tolist(), [0.0, 1.0])
        self.assertIs(code_of(lambda: next(iterator)), CODES.NAN_VALUE)


if __name__ == "__main__":
    unittest.main()
