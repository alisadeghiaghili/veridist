"""The Polars source: inputs, partition, batching, re-reading and failures."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import polars as pl
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_frame_fixtures import all_batches, code_of, values_of
from veridist.scale import _polars
from veridist.scale._arrow import ParquetSource
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._one_pass import ExponentialState, NormalState
from veridist.scale._polars import PolarsSource, collect_batches
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    FingerprintLevel,
    NumpySource,
    Partition,
    fold,
)

CODES = ScaleSourceErrorCode
VALUES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]


def frame() -> Any:
    return pl.DataFrame({"x": VALUES, "y": list(range(10))})


class PolarsInputTests(unittest.TestCase):
    def test_a_frame_and_a_lazy_frame_give_the_same_values(self) -> None:
        for name, data in (("frame", frame()), ("lazy frame", frame().lazy())):
            with self.subTest(input=name):
                batches = all_batches(PolarsSource(data, "x"), max_rows=4)
                self.assertTrue(all(len(b) <= 4 for b in batches))
                self.assertEqual(np.concatenate(batches).tolist(), VALUES)

    def test_the_projected_column_is_the_one_named(self) -> None:
        self.assertEqual(values_of(PolarsSource(frame(), "y")), [float(i) for i in range(10)])
        lazy = PolarsSource(frame().lazy(), "y")
        self.assertEqual(values_of(lazy), [float(i) for i in range(10)])

    def test_the_partition_is_one_non_resumable_partition(self) -> None:
        (partition,) = PolarsSource(frame(), "x").partitions()
        self.assertEqual(partition, Partition("polars", 0, 10, None, FingerprintLevel.NONE))
        self.assertFalse(partition.resumable)

    def test_the_row_count_of_a_plan_is_unknown(self) -> None:
        (partition,) = PolarsSource(frame().lazy(), "x").partitions()
        self.assertEqual(partition, Partition("polars", 0, None, None, FingerprintLevel.NONE))

    def test_an_empty_frame_has_zero_rows_and_no_batches(self) -> None:
        empty = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
        self.assertEqual(PolarsSource(empty, "x").partitions()[0].rows, 0)
        self.assertEqual(all_batches(PolarsSource(empty, "x")), [])
        self.assertEqual(all_batches(PolarsSource(empty.lazy(), "x")), [])

    def test_a_frame_and_a_plan_can_be_read_again(self) -> None:
        for data in (frame(), frame().lazy()):
            source = PolarsSource(data, "x")
            for _ in range(3):
                with self.subTest(kind=type(data).__name__):
                    self.assertEqual(values_of(source, max_rows=3), VALUES)

    def test_a_plan_is_run_again_for_every_read(self) -> None:
        calls: list[int] = []

        def count(series: Any) -> Any:
            calls.append(series.len())
            return series

        plan = frame().lazy().with_columns(pl.col("x").map_batches(count, return_dtype=pl.Float64))
        source = PolarsSource(plan, "x")
        self.assertEqual(calls, [])
        values_of(source)
        first = len(calls)
        self.assertGreater(first, 0)
        values_of(source)
        self.assertGreater(len(calls), first)

    def test_a_scan_is_projected_before_it_is_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_parquet(Path(directory) / "a.parquet", {"x": [1.0, 2.0], "s": ["a", "b"]})
            source = PolarsSource(pl.scan_parquet(path), "x")
            self.assertEqual(values_of(source), [1.0, 2.0])
            plan = pl.scan_parquet(path).select(pl.nth(0)).explain()
            self.assertIn("PROJECT 1/2 COLUMNS", plan)

    def test_batches_are_zero_copy_slices_of_the_stored_column(self) -> None:
        data = pl.DataFrame({"x": np.arange(10.0)})
        stored = data.get_column("x").to_numpy()
        for max_rows in (3, 100):
            for batch in all_batches(PolarsSource(data, "x"), max_rows=max_rows):
                with self.subTest(max_rows=max_rows):
                    self.assertTrue(np.shares_memory(batch, stored))
                    self.assertFalse(batch.flags["OWNDATA"])
                    self.assertFalse(batch.flags["WRITEABLE"])

    def test_the_bound_decides_the_batch_sizes_of_a_frame(self) -> None:
        for max_rows, sizes in ((4, [4, 4, 2]), (10, [10]), (1, [1] * 10), (3, [3, 3, 3, 1])):
            with self.subTest(max_rows=max_rows):
                batches = all_batches(PolarsSource(frame(), "x"), max_rows=max_rows)
                self.assertEqual([len(b) for b in batches], sizes)

    def test_a_frame_of_several_chunks_is_read_chunk_by_chunk(self) -> None:
        first, second = pl.DataFrame({"x": [0.0, 1.0, 2.0]}), pl.DataFrame({"x": [3.0, 4.0]})
        chunked = pl.concat([first, second], rechunk=False)
        self.assertEqual(chunked.n_chunks(), 2)
        for max_rows, sizes in ((2, [2, 1, 2]), (10, [3, 2])):
            with self.subTest(max_rows=max_rows):
                batches = all_batches(PolarsSource(chunked, "x"), max_rows=max_rows)
                self.assertEqual([len(b) for b in batches], sizes)
                self.assertEqual(np.concatenate(batches).tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_a_plan_never_hands_out_more_than_the_bound(self) -> None:
        data = pl.DataFrame({"x": np.arange(1000.0)}).lazy()
        for max_rows in (1, 7, 100, 999, 1000, 5000):
            with self.subTest(max_rows=max_rows):
                batches = all_batches(PolarsSource(data, "x"), max_rows=max_rows)
                self.assertLessEqual(max(len(b) for b in batches), max_rows)
                self.assertEqual(np.concatenate(batches).tolist(), np.arange(1000.0).tolist())

    def test_the_default_bound_applies(self) -> None:
        data = pl.DataFrame({"x": np.zeros(DEFAULT_MAX_ROWS + 1)})
        sizes = [len(b) for b in all_batches(PolarsSource(data, "x"))]
        self.assertEqual(sizes, [DEFAULT_MAX_ROWS, 1])
        lazy_sizes = [len(b) for b in all_batches(PolarsSource(data.lazy(), "x"))]
        self.assertEqual(sum(lazy_sizes), DEFAULT_MAX_ROWS + 1)
        self.assertLessEqual(max(lazy_sizes), DEFAULT_MAX_ROWS)

    def test_a_filter_and_a_join_of_a_plan_are_applied_before_reading(self) -> None:
        left = pl.DataFrame({"k": [1, 2, 3, 4], "x": [10.0, 20.0, 30.0, 40.0]}).lazy()
        right = pl.DataFrame({"k": [2, 4]}).lazy()
        plan = left.join(right, on="k").filter(pl.col("x") > 25.0).select("x")
        self.assertEqual(values_of(PolarsSource(plan, "x")), [40.0])

    def test_programmer_errors(self) -> None:
        frame_message = "frame must be a polars LazyFrame or DataFrame"
        for bad in ("x.parquet", [1.0], {"x": [1.0]}, None, 5, object(), pa.table({"x": [1.0]})):
            with (
                self.subTest(bad=type(bad).__name__),
                self.assertRaisesRegex(TypeError, f"^{re.escape(frame_message)}$"),
            ):
                PolarsSource(bad, "x")
        for column in ("", None, 3, b"x"):
            with (
                self.subTest(column=column),
                self.assertRaisesRegex(TypeError, "^column must be a non-empty string$"),
            ):
                PolarsSource(frame(), column)  # type: ignore[arg-type]

    def test_arguments_are_validated_when_batches_is_called(self) -> None:
        source = PolarsSource(frame(), "x")
        (partition,) = source.partitions()
        for bad, error in ((0, ValueError), (True, TypeError), (2.5, TypeError), (-1, ValueError)):
            with self.subTest(max_rows=bad), self.assertRaises(error):
                source.batches(partition, max_rows=bad)  # type: ignore[arg-type]
        for foreign in (Partition("other", 0, None, None, FingerprintLevel.NONE), "polars", None):
            with (
                self.subTest(partition=foreign),
                self.assertRaisesRegex(ValueError, "^partition does not belong to this source$"),
            ):
                source.batches(foreign)  # type: ignore[arg-type]

    def test_a_missing_column_is_refused_at_construction(self) -> None:
        for data in (frame(), frame().lazy()):
            for name in ("missing", "X"):
                refused = code_of(lambda d=data, n=name: PolarsSource(d, n))
                self.assertIs(refused, CODES.COLUMN_NOT_FOUND)

    def test_the_fold_of_every_input_equals_the_array_fold(self) -> None:
        values = np.array([0.0, 1.5, 2.5, 1e10, 3.25, 7.0, 0.125, 9.0, 1e-300])
        reference = fold(NumpySource(values), NormalState).to_bytes()
        data = pl.DataFrame({"x": values})
        for name, make in (("frame", lambda: data), ("lazy frame", lambda: data.lazy())):
            for max_rows in (1, 4, 100):
                with self.subTest(kind=name, max_rows=max_rows):
                    state = fold(PolarsSource(make(), "x"), NormalState, max_rows=max_rows)
                    self.assertEqual(state.to_bytes(), reference)

    def test_a_scan_of_a_parquet_file_folds_to_the_parquet_source_state(self) -> None:
        values = np.array([0.0, 1.5, 2.5, 1e10, 3.25, 7.0, 0.125, 9.0, 1e-300, 4.75])
        with tempfile.TemporaryDirectory() as directory:
            path = write_parquet(Path(directory) / "a.parquet", {"x": values}, row_group_size=3)
            reference = fold(ParquetSource(path, "x"), NormalState).to_bytes()
            self.assertEqual(reference, fold(NumpySource(values), NormalState).to_bytes())
            for name, data in (
                ("scan", pl.scan_parquet(path)),
                ("read", pl.read_parquet(path)),
                ("scan then filter", pl.scan_parquet(path).filter(pl.col("x") > -1.0)),
            ):
                for max_rows in (1, 4, 100):
                    with self.subTest(input=name, max_rows=max_rows):
                        state = fold(PolarsSource(data, "x"), NormalState, max_rows=max_rows)
                        self.assertEqual(state.to_bytes(), reference)


class PolarsFailureTests(unittest.TestCase):
    def assert_unreadable(self, error: ScaleSourceError, cause: type[BaseException]) -> None:
        self.assertIs(error.code, CODES.SOURCE_UNREADABLE)
        self.assertEqual(str(error), "SOURCE_UNREADABLE")
        self.assertIsNone(error.hint)
        self.assertIsInstance(error.__cause__, cause)

    def test_a_scan_of_a_missing_file_is_unreadable_at_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = pl.scan_parquet(Path(directory) / "missing.parquet")
            with self.assertRaises(ScaleSourceError) as context:
                PolarsSource(plan, "x")
        self.assert_unreadable(context.exception, OSError)

    def test_a_plan_that_cannot_be_resolved_is_unreadable_at_construction(self) -> None:
        plan = pl.LazyFrame({"x": [1.0]}).select(pl.col("nope"))
        with self.assertRaises(ScaleSourceError) as context:
            PolarsSource(plan, "x")
        self.assert_unreadable(context.exception, pl.exceptions.PolarsError)

    def test_a_file_that_disappears_is_unreadable_when_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_parquet(Path(directory) / "a.parquet", {"x": [1.0, 2.0]})
            source = PolarsSource(pl.scan_parquet(path), "x")
            path.unlink()
            (partition,) = source.partitions()
            with self.assertRaises(ScaleSourceError) as context:
                list(source.batches(partition))
        self.assert_unreadable(context.exception, OSError)

    def test_a_plan_that_fails_while_running_is_unreadable_and_keeps_no_detail(self) -> None:
        plan = pl.LazyFrame({"x": [1.0, 123456.789]}).select(pl.col("x").cast(pl.Int8))
        source = PolarsSource(plan, "x")
        (partition,) = source.partitions()
        with self.assertRaises(ScaleSourceError) as context:
            list(source.batches(partition))
        self.assert_unreadable(context.exception, pl.exceptions.PolarsError)
        self.assertNotIn("123456", str(context.exception))

    def test_a_malformed_file_is_unreadable_and_no_value_is_dropped_silently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.csv"
            rows = "\n".join(str(index) for index in range(3000))
            path.write_text(f"x\n{rows}\nnot a number\n", encoding="utf-8")
            plan = pl.scan_csv(path, schema_overrides={"x": pl.Float64})
            source = PolarsSource(plan, "x")
            (partition,) = source.partitions()
            with self.assertRaises(ScaleSourceError) as context:
                fold(source, ExponentialState, max_rows=100)
            self.assertIs(context.exception.code, CODES.SOURCE_UNREADABLE)
            with self.assertRaises(ScaleSourceError):
                list(source.batches(partition))

    def test_an_arrow_conversion_failure_is_unreadable(self) -> None:
        source = PolarsSource(frame(), "x")
        (partition,) = source.partitions()
        with mock.patch.object(pl.DataFrame, "to_arrow", side_effect=pa.ArrowInvalid("detail")):
            with self.assertRaises(ScaleSourceError) as context:
                list(source.batches(partition))
        self.assert_unreadable(context.exception, pa.ArrowException)

    def test_a_policy_refusal_is_not_rewrapped_as_unreadable(self) -> None:
        for data in (pl.DataFrame({"x": ["a"]}), pl.DataFrame({"x": ["a"]}).lazy()):
            source = PolarsSource(data, "x")
            (partition,) = source.partitions()
            iterator = source.batches(partition)
            self.assertIs(code_of(lambda i=iterator: next(i)), CODES.UNSUPPORTED_DTYPE)

    def test_a_refusal_in_a_later_batch_comes_after_the_earlier_ones(self) -> None:
        source = PolarsSource(pl.DataFrame({"x": [1.0, 2.0, float("nan")]}), "x")
        (partition,) = source.partitions()
        iterator = source.batches(partition, max_rows=2)
        self.assertEqual(next(iterator).tolist(), [1.0, 2.0])
        self.assertIs(code_of(lambda: next(iterator)), CODES.NAN_VALUE)


class CollectBatchesTests(unittest.TestCase):
    def test_the_streaming_call_is_made_with_the_bound_as_chunk_size(self) -> None:
        class Plan:
            def __init__(self) -> None:
                self.calls: list[dict[str, Any]] = []
                self.chunks = iter(["a", "b"])

            def collect_batches(self, **options: Any) -> Any:
                self.calls.append(options)
                return self.chunks

        plan = Plan()
        result = collect_batches(plan, 7)
        self.assertEqual(plan.calls, [{"chunk_size": 7}])
        self.assertIs(result, plan.chunks)
        self.assertEqual(list(result), ["a", "b"])

    def test_the_real_call_streams_frames_of_the_projected_column(self) -> None:
        chunks = list(collect_batches(pl.LazyFrame({"x": [1.0, 2.0, 3.0]}), 2))
        self.assertTrue(all(isinstance(chunk, pl.DataFrame) for chunk in chunks))
        values = [value for chunk in chunks for value in chunk["x"].to_list()]
        self.assertEqual(values, [1.0, 2.0, 3.0])

    def test_the_streaming_call_is_used_for_a_plan_only(self) -> None:
        with mock.patch.object(_polars, "collect_batches", wraps=collect_batches) as spy:
            values_of(PolarsSource(frame(), "x"))
            self.assertEqual(spy.call_count, 0)
            values_of(PolarsSource(frame().lazy(), "x"), max_rows=5)
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(spy.call_args.args[1], 5)


if __name__ == "__main__":
    unittest.main()
