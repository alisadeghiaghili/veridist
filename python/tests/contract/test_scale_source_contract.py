"""Contract of the source layer: protocol, closed error codes and the optional extra."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import polars as pl
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_frame_fixtures import connect
from tests.unit.scale_source_oracle import ListSource
from veridist.engine.errors import VeridistError
from veridist.scale import _arrow, _duckdb, _polars
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._duckdb import DuckDBSource
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._one_pass import ExponentialState
from veridist.scale._polars import PolarsSource
from veridist.scale._sources import (
    FingerprintLevel,
    NumpySource,
    Partition,
    Source,
    fold,
)

EXPECTED_HINT = 'install the optional "arrow" extra with: pip install "veridist[arrow]"'
POLARS_HINT = 'install the optional "polars" extra with: pip install "veridist[polars]"'
DUCKDB_HINT = 'install the optional "duckdb" extra with: pip install "veridist[duckdb]"'
BLOCK_PYARROW = {"pyarrow": None, "pyarrow.parquet": None}


class ErrorCodeContractTests(unittest.TestCase):
    def test_the_codes_are_closed_and_named_after_their_values(self) -> None:
        self.assertEqual(
            sorted(code.value for code in ScaleSourceErrorCode),
            [
                "COLUMN_NOT_FOUND",
                "LOSSY_CAST",
                "NAN_VALUE",
                "NON_FINITE_VALUE",
                "NULL_VALUE",
                "OPTIONAL_DEPENDENCY_MISSING",
                "PARTITION_CHANGED",
                "SOURCE_CONSUMED",
                "SOURCE_NOT_FOUND",
                "SOURCE_UNREADABLE",
                "SPOOL_CHANGED",
                "SPOOL_LIMIT_EXCEEDED",
                "SPOOL_REQUIRED",
                "SPOOL_UNAVAILABLE",
                "UNSUPPORTED_DTYPE",
            ],
        )
        for code in ScaleSourceErrorCode:
            self.assertEqual(code.name, code.value)

    def test_the_error_is_a_veridist_error_and_a_value_error(self) -> None:
        self.assertTrue(issubclass(ScaleSourceError, VeridistError))
        self.assertTrue(issubclass(ScaleSourceError, ValueError))


class ProtocolTests(unittest.TestCase):
    def test_every_source_provides_partitions_and_batches(self) -> None:
        table = pa.table({"x": [1.0]})
        with tempfile.TemporaryDirectory() as directory:
            write_parquet(Path(directory) / "a.parquet", {"x": [1.0]})
            sources: list[Any] = [
                NumpySource(np.array([1.0])),
                ArrowSource(table, "x"),
                ParquetSource(directory, "x"),
                PolarsSource(pl.DataFrame({"x": [1.0]}), "x"),
                PolarsSource(pl.LazyFrame({"x": [1.0]}), "x"),
                DuckDBSource((connect(), "select 1.0::double as x"), "x"),
                ListSource([[np.array([1.0])]]),
            ]
            for source in sources:
                with self.subTest(kind=type(source).__name__):
                    self.assertIsInstance(source, Source)
                    partitions = source.partitions()
                    self.assertIsInstance(partitions, tuple)
                    self.assertTrue(all(isinstance(p, Partition) for p in partitions))
                    self.assertEqual([p.ordinal for p in partitions], list(range(len(partitions))))
                    self.assertEqual(len({p.id for p in partitions}), len(partitions))
                    for partition in partitions:
                        batches = list(source.batches(partition, max_rows=1))
                        self.assertTrue(all(b.dtype == np.float64 and b.ndim == 1 for b in batches))

    def test_objects_without_the_two_calls_are_not_sources(self) -> None:
        for other in (
            object(),
            np.array([1.0]),
            pa.table({"x": [1.0]}),
            pl.DataFrame({"x": [1.0]}),
            "x",
            None,
        ):
            with self.subTest(kind=type(other).__name__):
                self.assertNotIsInstance(other, Source)

    def test_fold_accepts_any_source_and_keeps_the_array_result(self) -> None:
        values = np.array([0.5, 1.5, 4.0])
        expected = ExponentialState.empty().update(values).to_bytes()
        table = pa.table({"x": values})
        sources = (
            NumpySource(values),
            ArrowSource(table, "x"),
            PolarsSource(pl.DataFrame({"x": values}), "x"),
            PolarsSource(pl.LazyFrame({"x": values}), "x"),
            DuckDBSource(connect().from_arrow(table), "x"),
            ListSource([[values]]),
        )
        for source in sources:
            with self.subTest(kind=type(source).__name__):
                self.assertEqual(fold(source, ExponentialState).to_bytes(), expected)

    def test_non_resumable_means_no_fingerprint_and_level_none(self) -> None:
        (partition,) = ArrowSource(pa.table({"x": [1.0]}), "x").partitions()
        self.assertIsNone(partition.fingerprint)
        self.assertIs(partition.fingerprint_level, FingerprintLevel.NONE)
        self.assertFalse(partition.resumable)
        (array,) = NumpySource(np.array([1.0])).partitions()
        self.assertIs(array.fingerprint_level, FingerprintLevel.CONTENT)
        self.assertTrue(array.resumable)

    def test_polars_and_duckdb_partitions_are_not_resumable(self) -> None:
        sources: list[Any] = [
            PolarsSource(pl.DataFrame({"x": [1.0]}), "x"),
            PolarsSource(pl.LazyFrame({"x": [1.0]}), "x"),
            DuckDBSource((connect(), "select 1.0::double as x"), "x"),
        ]
        for source in sources:
            with self.subTest(kind=type(source).__name__):
                (partition,) = source.partitions()
                self.assertIsNone(partition.fingerprint)
                self.assertIs(partition.fingerprint_level, FingerprintLevel.NONE)
                self.assertFalse(partition.resumable)


class MissingExtraTests(unittest.TestCase):
    def assert_missing_extra(self, call: Any) -> None:
        with self.assertRaises(ScaleSourceError) as context:
            call()
        error = context.exception
        self.assertIs(error.code, ScaleSourceErrorCode.OPTIONAL_DEPENDENCY_MISSING)
        self.assertEqual(error.hint, EXPECTED_HINT)
        self.assertEqual(str(error), f"OPTIONAL_DEPENDENCY_MISSING: {EXPECTED_HINT}")
        self.assertIsInstance(error.__cause__, ImportError)

    def test_the_loaders_name_the_extra(self) -> None:
        with mock.patch.dict(sys.modules, BLOCK_PYARROW):
            self.assert_missing_extra(_arrow.pyarrow_module)
            self.assert_missing_extra(_arrow.pyarrow_parquet_module)

    def test_only_the_parquet_module_missing_is_reported_too(self) -> None:
        with mock.patch.dict(sys.modules, {"pyarrow.parquet": None}):
            self.assert_missing_extra(_arrow.pyarrow_parquet_module)
            self.assertIsNotNone(_arrow.pyarrow_module())

    def test_an_import_error_of_any_kind_is_reported_the_same_way(self) -> None:
        with mock.patch.object(_arrow, "import_module", side_effect=ImportError("broken lib")):
            self.assert_missing_extra(_arrow.pyarrow_module)

    def test_the_arrow_and_parquet_sources_need_the_extra_at_first_use(self) -> None:
        table = pa.table({"x": [1.0, 2.0]})
        with tempfile.TemporaryDirectory() as directory:
            write_parquet(Path(directory) / "a.parquet", {"x": [1.0, 2.0]})
            parquet = ParquetSource(directory, "x")
            (partition,) = parquet.partitions()
            with mock.patch.dict(sys.modules, BLOCK_PYARROW):
                self.assert_missing_extra(lambda: ArrowSource(table, "x"))
                self.assert_missing_extra(parquet.partitions)
                self.assert_missing_extra(lambda: parquet.batches(partition))
                self.assert_missing_extra(lambda: _arrow.arrow_float64(pa.array([1.0])))

    def assert_missing(self, call: Any, hint: str) -> None:
        with self.assertRaises(ScaleSourceError) as context:
            call()
        error = context.exception
        self.assertIs(error.code, ScaleSourceErrorCode.OPTIONAL_DEPENDENCY_MISSING)
        self.assertEqual(error.hint, hint)
        self.assertEqual(str(error), f"OPTIONAL_DEPENDENCY_MISSING: {hint}")
        self.assertIsInstance(error.__cause__, ImportError)

    def test_the_loader_names_the_extra_it_is_given(self) -> None:
        with mock.patch.dict(sys.modules, {"polars": None, "duckdb": None}):
            self.assert_missing(lambda: _arrow.load_optional("polars", "polars"), POLARS_HINT)
            self.assert_missing(lambda: _arrow.load_optional("duckdb", "duckdb"), DUCKDB_HINT)
            self.assert_missing(lambda: _arrow.load_optional("duckdb", "arrow"), EXPECTED_HINT)

    def test_the_polars_source_needs_polars_and_pyarrow_and_names_its_extra(self) -> None:
        data = pl.DataFrame({"x": [1.0, 2.0]})
        source = PolarsSource(data, "x")
        (partition,) = source.partitions()
        with mock.patch.dict(sys.modules, {"polars": None}):
            self.assert_missing(_polars.polars_module, POLARS_HINT)
            self.assert_missing(lambda: PolarsSource(data, "x"), POLARS_HINT)
            self.assert_missing(lambda: source.batches(partition), POLARS_HINT)
        with mock.patch.dict(sys.modules, {"pyarrow": None}):
            self.assert_missing(lambda: PolarsSource(data, "x"), POLARS_HINT)
            self.assert_missing(lambda: source.batches(partition), POLARS_HINT)
        with mock.patch.dict(sys.modules, BLOCK_PYARROW):
            self.assert_missing(lambda: PolarsSource(data.lazy(), "x"), POLARS_HINT)

    def test_the_duckdb_source_needs_duckdb_and_pyarrow_and_names_its_extra(self) -> None:
        connection = connect()
        pair = (connection, "select 1.0::double as x")
        source = DuckDBSource(pair, "x")
        (partition,) = source.partitions()
        with mock.patch.dict(sys.modules, {"duckdb": None}):
            self.assert_missing(_duckdb.duckdb_module, DUCKDB_HINT)
            self.assert_missing(lambda: DuckDBSource(pair, "x"), DUCKDB_HINT)
            self.assert_missing(lambda: source.batches(partition), DUCKDB_HINT)
        with mock.patch.dict(sys.modules, {"pyarrow": None}):
            self.assert_missing(lambda: DuckDBSource(pair, "x"), DUCKDB_HINT)
            self.assert_missing(lambda: source.batches(partition), DUCKDB_HINT)

    def test_an_import_error_of_any_kind_is_reported_for_polars_and_duckdb_too(self) -> None:
        with mock.patch.object(_arrow, "import_module", side_effect=ImportError("broken lib")):
            self.assert_missing(_polars.polars_module, POLARS_HINT)
            self.assert_missing(_duckdb.duckdb_module, DUCKDB_HINT)

    def test_the_array_source_and_fold_do_not_need_the_extra(self) -> None:
        values = np.array([1.0, 2.0, 3.0])
        with mock.patch.dict(sys.modules, BLOCK_PYARROW):
            state = fold(NumpySource(values), ExponentialState, max_rows=2)
        self.assertEqual(state.count, 3)


if __name__ == "__main__":
    unittest.main()
