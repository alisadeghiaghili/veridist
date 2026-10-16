"""``fit(..., executor=...)``: identical results, the record of what ran, downgrades, errors."""

from __future__ import annotations

import struct
import tempfile
import threading
import unittest
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa

from tests.unit.scale_executor_fixtures import (
    NoLeaksTestCase,
    PartSource,
    PlainSource,
    cores,
    parquet_dataset,
)
from tests.unit.scale_multi_pass_oracle import gumbel_sample, weibull_sample
from veridist.engine.provenance import (
    ScaleExecutionPassProvenance,
    scale_provenance_to_canonical_json_bytes,
)
from veridist.families.gamma import GammaFitFailure, GammaFitFailureCode
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._duckdb import DuckDBSource
from veridist.scale._errors import (
    ScaleSourceError,
    ScaleSourceErrorCode,
    ScaleStateError,
    ScaleStateErrorCode,
)
from veridist.scale._executors import (
    ProcessExecutor,
    SerialExecutor,
    ThreadExecutor,
)
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._polars import PolarsSource
from veridist.scale._sources import NumpySource, Partition

ONE_PASS = ("exponential", "normal", "gamma", "lognormal")
MULTI_PASS = ("weibull_min", "gumbel_right")


def data_for(family: str, size: int = 900) -> np.ndarray:
    rng = np.random.default_rng(7)
    if family == "exponential":
        return rng.exponential(3.0, size)
    if family == "normal":
        return rng.normal(10.0, 3.0, size)
    if family == "gamma":
        return rng.gamma(3.0, 2.0, size)
    if family == "lognormal":
        return rng.lognormal(1.0, 0.5, size)
    if family == "weibull_min":
        return weibull_sample(1.7, 40.0, size, 7)
    return gumbel_sample(5.0, 2.0, size, 7)


def bits(result: ScaleFitResult) -> tuple[bytes, ...]:
    success: Any = result.fit
    numbers = [*success.parameters.values(), success.log_likelihood]
    uncertainty = success.uncertainty()
    for row in uncertainty.covariance:
        numbers.extend(row)
    return tuple(struct.pack("<d", float(value)) for value in numbers)


def same_but_execution(result: ScaleFitResult, base: ScaleFitResult) -> bool:
    return replace(result.scale, execution=None) == base.scale


class IdenticalResultTests(NoLeaksTestCase):
    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_every_family_gives_the_same_bits_digest_and_record_with_any_thread_count(self) -> None:
        for family in (*ONE_PASS, *MULTI_PASS):
            source = parquet_dataset(self.root / family, data_for(family))
            base = fit(source, family)
            assert base.scale.state is not None
            for executor in (
                None,
                "serial",
                SerialExecutor(),
                *(ThreadExecutor(n) for n in (1, 2, 4)),
            ):
                with self.subTest(family=family, executor=repr(executor)), cores(16):
                    result = fit(source, family, executor=executor)
                    self.assertEqual(result.fit, base.fit)
                    self.assertEqual(bits(result), bits(base))
                    self.assertEqual(result.scale.state, base.scale.state)
                    self.assertTrue(same_but_execution(result, base))

    def test_the_state_bytes_are_canonical_whatever_the_executor(self) -> None:
        source = parquet_dataset(self.root / "d", data_for("exponential"), files=4, row_groups=3)
        digests = set()
        with cores(16):
            for executor in (None, ThreadExecutor(1), ThreadExecutor(3), ThreadExecutor(8)):
                result = fit(source, "exponential", executor=executor)
                assert result.scale.state is not None
                digests.add(result.scale.state.digest_sha256)
        self.assertEqual(len(digests), 1)

    def test_the_numpy_ranges_are_read_on_threads_with_the_same_result(self) -> None:
        for family in ("exponential", "weibull_min"):
            values = data_for(family)
            base = fit(NumpySource(values), family)
            source = NumpySource(values, partition_rows=130)
            with self.subTest(family=family), cores(16):
                result = fit(source, family, executor=ThreadExecutor(4))
                self.assertEqual(result.fit, base.fit)
                assert result.scale.state is not None and base.scale.state is not None
                self.assertEqual(result.scale.state, base.scale.state)
                assert result.scale.execution is not None
                self.assertEqual(result.scale.execution.effective_workers, 4)
                self.assertIsNone(result.scale.execution.downgrade)

    def test_a_custom_source_that_declares_itself_is_read_on_threads(self) -> None:
        values = data_for("exponential")
        arrays = np.array_split(values, 5)
        base = fit(PlainSource(arrays), "exponential")
        with cores(16):
            result = fit(PartSource(arrays), "exponential", executor=ThreadExecutor(3))
        self.assertEqual(result.fit, base.fit)
        assert result.scale.execution is not None
        self.assertEqual(result.scale.execution.passes[0].executor, "threads")


class RecordTests(NoLeaksTestCase):
    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_no_executor_leaves_no_record(self) -> None:
        source = parquet_dataset(self.root, data_for("exponential"))
        self.assertIsNone(fit(source, "exponential").scale.execution)
        self.assertNotIn(
            b"execution", scale_provenance_to_canonical_json_bytes(fit(source, "exponential").scale)
        )

    def test_a_one_pass_fit_has_one_record(self) -> None:
        source = parquet_dataset(self.root, data_for("exponential"))
        with cores(16):
            execution = fit(source, "exponential", executor=ThreadExecutor(4)).scale.execution
        assert execution is not None
        self.assertEqual(execution.executor, "threads")
        self.assertEqual(execution.requested_workers, 4)
        self.assertEqual(
            execution.passes, (ScaleExecutionPassProvenance("threads", 4, 6, 0, None),)
        )
        self.assertEqual((execution.effective_workers, execution.retries), (4, 0))
        self.assertIsNone(execution.downgrade)

    def test_the_serial_executor_is_recorded(self) -> None:
        source = parquet_dataset(self.root, data_for("exponential"))
        execution = fit(source, "exponential", executor="serial").scale.execution
        assert execution is not None
        self.assertEqual(
            (execution.executor, execution.requested_workers, execution.effective_workers),
            ("serial", 1, 1),
        )
        self.assertEqual(execution.passes, (ScaleExecutionPassProvenance("serial", 1, 6, 0, None),))

    def test_a_multi_pass_fit_has_one_record_per_pass(self) -> None:
        source = parquet_dataset(self.root, data_for("weibull_min"))
        with cores(16):
            result = fit(source, "weibull_min", executor=ThreadExecutor(2))
        assert result.scale.execution is not None and result.scale.passes is not None
        records = result.scale.execution.passes
        self.assertEqual(len(records), result.scale.passes.pass_count)
        self.assertGreaterEqual(len(records), 3)
        for record in records:
            self.assertEqual(record, ScaleExecutionPassProvenance("threads", 2, 6, 0, None))

    def test_the_record_is_in_the_canonical_json(self) -> None:
        source = parquet_dataset(self.root, data_for("exponential"))
        with cores(16):
            scale = fit(source, "exponential", executor=ThreadExecutor(3)).scale
        encoded = scale_provenance_to_canonical_json_bytes(scale)
        self.assertIn(
            b'"execution":{"downgrade":null,"effective_workers":3,"executor":"threads",'
            b'"passes":[{"downgrade":null,"executor":"threads","retries":0,"tasks":6,"workers":3}],'
            b'"requested_workers":3,"retries":0}',
            encoded,
        )


class DowngradeTests(NoLeaksTestCase):
    def last(self, result: ScaleFitResult) -> ScaleExecutionPassProvenance:
        assert result.scale.execution is not None
        return result.scale.execution.passes[-1]

    def test_each_source_that_cannot_be_split_is_read_serially_with_its_reason(self) -> None:
        values = data_for("exponential", 300)
        table = pa.table({"x": values})
        connection = duckdb.connect()
        self.addCleanup(connection.close)

        def stream() -> ArrowSource:
            reader = pa.RecordBatchReader.from_batches(
                table.schema, table.to_batches(max_chunksize=64)
            )
            return ArrowSource(reader, "x")

        cases: list[tuple[str, Callable[[], Any], int, str, str]] = [
            (
                "arrow table",
                lambda: ArrowSource(table, "x"),
                1,
                "single_partition",
                "in_memory_data",
            ),
            ("arrow stream", stream, 1, "single_use_source", "single_use_source"),
            ("numpy", lambda: NumpySource(values), 1, "single_partition", "in_memory_data"),
            (
                "polars frame",
                lambda: PolarsSource(pl.DataFrame({"x": values}), "x"),
                1,
                "engine_managed",
                "engine_managed",
            ),
            (
                "polars plan",
                lambda: PolarsSource(pl.LazyFrame({"x": values}), "x"),
                1,
                "engine_managed",
                "engine_managed",
            ),
            (
                "duckdb",
                lambda: DuckDBSource(
                    (connection, "select unnest(range(1, 301))::double as x"), "x"
                ),
                1,
                "engine_managed",
                "engine_managed",
            ),
            (
                "custom",
                lambda: PlainSource(np.array_split(values, 3)),
                3,
                "undeclared_source",
                "undeclared_source",
            ),
        ]
        for make_executor in (ThreadExecutor, ProcessExecutor):
            for name, make, tasks, threads_reason, processes_reason in cases:
                reason = threads_reason if make_executor is ThreadExecutor else processes_reason
                with self.subTest(source=name, kind=make_executor.__name__), cores(16):
                    result = fit(make(), "exponential", executor=make_executor(3))
                    self.assertEqual(
                        self.last(result),
                        ScaleExecutionPassProvenance("serial", 1, tasks, 0, reason),
                    )
                    assert result.scale.execution is not None
                    self.assertEqual(result.scale.execution.downgrade, reason)
                    self.assertEqual(result.scale.execution.effective_workers, 1)
                    self.assertEqual(result.scale.execution.requested_workers, 3)

    def test_a_downgraded_fit_has_the_result_of_the_plain_one(self) -> None:
        values = data_for("gamma", 300)
        plain = fit(PolarsSource(pl.DataFrame({"x": values}), "x"), "gamma")
        with cores(16):
            result = fit(
                PolarsSource(pl.DataFrame({"x": values}), "x"), "gamma", executor=ThreadExecutor(4)
            )
        self.assertEqual(result.fit, plain.fit)
        self.assertEqual(bits(result), bits(plain))
        self.assertTrue(same_but_execution(result, plain))

    def test_processes_cannot_have_numpy_ranges_and_say_why(self) -> None:
        source = NumpySource(data_for("exponential", 400), partition_rows=100)
        with cores(16):
            result = fit(source, "exponential", executor=ProcessExecutor(4))
        self.assertEqual(
            self.last(result), ScaleExecutionPassProvenance("serial", 1, 4, 0, "in_memory_data")
        )

    def test_one_row_group_is_not_split(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = parquet_dataset(
                Path(directory), data_for("exponential", 200), files=1, row_groups=1
            )
            with cores(16):
                result = fit(source, "exponential", executor=ThreadExecutor(4))
        self.assertEqual(
            self.last(result), ScaleExecutionPassProvenance("serial", 1, 1, 0, "single_partition")
        )


class ArgumentTests(unittest.TestCase):
    class Untouched:
        def partitions(self) -> tuple[Partition, ...]:
            raise AssertionError("read before the arguments were checked")

        def batches(self, partition: Partition, *, max_rows: int = 1) -> Iterator[Any]:
            raise AssertionError("read before the arguments were checked")

    def test_a_bad_executor_is_refused_before_anything_is_read(self) -> None:
        for bad in ("threads", "processes", ""):
            with self.subTest(executor=bad), self.assertRaises(ValueError):
                fit(self.Untouched(), "exponential", executor=bad)
        for wrong in (4, b"serial", ThreadExecutor, object(), 2.5):
            with self.subTest(executor=repr(wrong)), self.assertRaises(TypeError):
                fit(self.Untouched(), "weibull_min", executor=wrong)  # type: ignore[arg-type]

    def test_the_executor_is_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            fit(NumpySource(np.ones(3)), "exponential", 262144, 8, None, "serial")  # type: ignore[misc]


class ErrorTests(NoLeaksTestCase):
    def setUp(self) -> None:
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def damaged(self, family: str, edits: dict[int, tuple[int, float]]) -> ParquetSource:
        """Six partitions (three files, two row groups); each edit sets one row of a file."""

        values = data_for(family, 600).copy()
        pieces = np.array_split(values, 3)
        for file, (row, value) in edits.items():
            pieces[file][row] = value
        return parquet_dataset(self.root, np.concatenate(pieces))

    def test_a_refused_value_is_the_same_typed_error_on_threads(self) -> None:
        source = self.damaged("exponential", {1: (5, float("nan"))})
        with self.assertRaises(ScaleSourceError) as serial:
            fit(source, "exponential")
        with cores(16), self.assertRaises(ScaleSourceError) as threaded:
            fit(source, "exponential", executor=ThreadExecutor(4))
        self.assertIs(threaded.exception.code, serial.exception.code)
        self.assertIs(threaded.exception.code, ScaleSourceErrorCode.NAN_VALUE)

    def test_the_first_failure_in_canonical_order_is_the_one_raised(self) -> None:
        source = self.damaged("exponential", {0: (3, -1.0), 2: (4, float("nan"))})
        for _ in range(8):
            with cores(16), self.assertRaises(ScaleStateError) as caught:
                fit(source, "exponential", executor=ThreadExecutor(4))
            self.assertIs(caught.exception.code, ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)

    def test_a_value_outside_the_support_is_the_same_failure_with_the_same_counts(self) -> None:
        source = self.damaged("gamma", {2: (7, -2.0)})
        serial = fit(source, "gamma")
        assert isinstance(serial.fit, GammaFitFailure)
        for executor in ("serial", ThreadExecutor(1), ThreadExecutor(4)):
            with self.subTest(executor=repr(executor)), cores(16):
                result = fit(source, "gamma", executor=executor)
                self.assertEqual(result.fit, serial.fit)
                self.assertIs(result.fit.code, GammaFitFailureCode.INVALID_SUPPORT)  # type: ignore[union-attr]
                self.assertEqual(result.scale.row_count, 600)
                self.assertTrue(same_but_execution(result, serial))
                assert result.scale.execution is not None
                self.assertEqual(len(result.scale.execution.passes), 1)

    def test_a_multi_pass_family_reports_its_support_failure_the_same_way(self) -> None:
        source = self.damaged("weibull_min", {1: (2, -3.0)})
        serial = fit(source, "weibull_min")
        with cores(16):
            result = fit(source, "weibull_min", executor=ThreadExecutor(3))
        self.assertEqual(result.fit, serial.fit)
        self.assertTrue(same_but_execution(result, serial))

    def test_rows_that_change_between_passes_are_partition_changed_on_threads(self) -> None:
        class Shrinking(PartSource):
            def batches(self, partition: Partition, *, max_rows: int = 100) -> Iterator[Any]:
                batches = super().batches(partition, max_rows=max_rows)
                if partition.ordinal == 1 and self.reads.count(1) > 1:
                    return (batch[:-1] for batch in batches)
                return batches

        arrays = np.array_split(data_for("weibull_min", 120), 4)
        for executor in (None, ThreadExecutor(3)):
            with self.subTest(executor=repr(executor)), cores(16):
                with self.assertRaises(ScaleSourceError) as caught:
                    fit(Shrinking(arrays), "weibull_min", executor=executor)
                self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)

    def test_a_transient_read_error_is_read_again_and_recorded(self) -> None:
        class Flaky:
            """Delegates to a Parquet source; the first read of partition 2 is a read error."""

            def __init__(self, inner: ParquetSource, cause: BaseException) -> None:
                self.inner = inner
                self.cause = cause
                self.failed = False
                self._lock = threading.Lock()

            def partitions(self) -> tuple[Partition, ...]:
                return self.inner.partitions()

            def parallel_obstacle(self, kind: Any, partitions: Any) -> Any:
                return self.inner.parallel_obstacle(kind, partitions)

            def batches(self, partition: Partition, *, max_rows: int = 100) -> Iterator[Any]:
                with self._lock:
                    first = partition.ordinal == 2 and not self.failed
                    self.failed = self.failed or first
                if first:
                    error = ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE)
                    raise error from self.cause
                return self.inner.batches(partition, max_rows=max_rows)

        source = parquet_dataset(self.root, data_for("exponential"))
        base = fit(source, "exponential")
        for executor in (SerialExecutor(), ThreadExecutor(3)):
            with self.subTest(executor=repr(executor)), cores(16):
                result = fit(
                    Flaky(source, ConnectionResetError()), "exponential", executor=executor
                )
                self.assertEqual(result.fit, base.fit)
                assert result.scale.execution is not None
                self.assertEqual(result.scale.execution.retries, 1)
                self.assertEqual(result.scale.state, base.scale.state)
        for cause in (FileNotFoundError("gone"), PermissionError("no"), ValueError("bad")):
            with self.subTest(cause=type(cause).__name__), self.assertRaises(ScaleSourceError):
                fit(Flaky(source, cause), "exponential", executor="serial")
        with self.assertRaises(ScaleSourceError):
            fit(Flaky(source, ConnectionResetError()), "exponential")
