"""The scale fit entry point: arguments, results, provenance and refusals."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_fit_oracle import FAMILIES, ListSource, dataset
from veridist.engine.provenance import (
    SCALE_PROVENANCE_SCHEMA_VERSION,
    ScaleProvenance,
    scale_provenance_to_canonical_json_bytes,
)
from veridist.families.exponential import (
    ExponentialFitFailure,
    ExponentialFitFailureCode,
    ExponentialFitSuccess,
)
from veridist.families.gamma import GammaFitFailure, GammaFitFailureCode, GammaFitSuccess
from veridist.families.lognormal import (
    LognormalFitFailure,
    LognormalFitFailureCode,
    LognormalFitSuccess,
)
from veridist.families.normal import NormalFitFailure, NormalFitFailureCode, NormalFitSuccess
from veridist.families.registry import FamilyId
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._duckdb import DuckDBSource
from veridist.scale._environment import numeric_environment_digest, numpy_version
from veridist.scale._errors import (
    ScaleSourceError,
    ScaleSourceErrorCode,
    ScaleStateError,
    ScaleStateErrorCode,
)
from veridist.scale._fit import (
    CUSTOM_SOURCE_KIND,
    SOURCE_KINDS,
    ScaleFitResult,
    fit,
)
from veridist.scale._one_pass import (
    ExponentialState,
    GammaState,
    LognormalState,
    NormalState,
)
from veridist.scale._polars import PolarsSource
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    NumpySource,
    Partition,
    fold,
)

SUCCESS = {
    FamilyId.EXPONENTIAL: ExponentialFitSuccess,
    FamilyId.NORMAL: NormalFitSuccess,
    FamilyId.GAMMA: GammaFitSuccess,
    FamilyId.LOGNORMAL: LognormalFitSuccess,
}
STATE = {
    FamilyId.EXPONENTIAL: ExponentialState,
    FamilyId.NORMAL: NormalState,
    FamilyId.GAMMA: GammaState,
    FamilyId.LOGNORMAL: LognormalState,
}


class ResultTests(unittest.TestCase):
    def test_every_family_returns_its_in_memory_success_type(self) -> None:
        for family in FAMILIES:
            result = fit(NumpySource(dataset(family, 200, 1)), family)
            with self.subTest(family=family):
                self.assertIsInstance(result, ScaleFitResult)
                self.assertIs(type(result.fit), SUCCESS[family])
                self.assertEqual(result.fit.family, family)
                self.assertEqual(result.fit.observation_count, 200)
                self.assertEqual(result.fit.event_count, 200)
                self.assertEqual(result.fit.censored_count, 0)
                self.assertTrue(result.fit.converged)

    def test_family_names_and_aliases_select_the_same_family(self) -> None:
        values = dataset(FamilyId.NORMAL, 50, 2)
        expected = fit(NumpySource(values), FamilyId.NORMAL).fit
        for name in ("normal", "gaussian"):
            self.assertEqual(fit(NumpySource(values), name).fit, expected)

    def test_uncertainty_is_available_on_every_family(self) -> None:
        for family in FAMILIES:
            result = fit(NumpySource(dataset(family, 300, 3)), family).fit
            assert isinstance(result, tuple(SUCCESS.values()))
            uncertainty = result.uncertainty()
            with self.subTest(family=family):
                self.assertEqual(set(uncertainty.standard_errors), set(result.parameters))  # type: ignore[union-attr]

    def test_the_result_validates_its_parts(self) -> None:
        good = fit(NumpySource(dataset(FamilyId.EXPONENTIAL, 10, 1)), "exponential")
        with self.assertRaises(TypeError):
            ScaleFitResult("not a fit", good.scale)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            ScaleFitResult(good.fit, None)  # type: ignore[arg-type]
        self.assertEqual(ScaleFitResult(good.fit, good.scale), good)


class ArgumentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = ListSource([np.array([1.0, 2.0, 3.0])])

    def test_an_unknown_family_is_a_value_error(self) -> None:
        with self.assertRaises(ValueError):
            fit(self.source, "no_such_family")

    def test_a_family_of_the_wrong_type_is_a_type_error(self) -> None:
        for bad in (3, None, 1.5, b"normal"):
            with self.subTest(family=bad), self.assertRaises(TypeError):
                fit(self.source, bad)  # type: ignore[arg-type]

    def test_the_pass_budget_is_validated_before_any_read(self) -> None:
        for family in ("normal", "weibull_min", "gumbel_right"):
            for bad in (0, -1):
                with self.subTest(family=family, pass_budget=bad), self.assertRaises(ValueError):
                    fit(self.source, family, pass_budget=bad)
            for bad in (True, 8.0, "8", None):
                with self.subTest(family=family, pass_budget=bad), self.assertRaises(TypeError):
                    fit(self.source, family, pass_budget=bad)  # type: ignore[arg-type]
        self.assertEqual(self.source.partition_calls, 0)

    def test_the_batch_cap_is_validated_before_any_read(self) -> None:
        for bad in (0, -5):
            with self.subTest(max_rows=bad), self.assertRaises(ValueError):
                fit(self.source, "normal", max_rows=bad)
        for bad in (True, 2.0, "8", None):
            with self.subTest(max_rows=bad), self.assertRaises(TypeError):
                fit(self.source, "normal", max_rows=bad)  # type: ignore[arg-type]
        self.assertEqual(self.source.partition_calls, 0)

    def test_a_source_must_follow_the_protocol(self) -> None:
        for bad in (object(), [1.0, 2.0], np.array([1.0, 2.0]), None):
            with self.subTest(source=type(bad).__name__), self.assertRaises(TypeError):
                fit(bad, "normal")  # type: ignore[arg-type]

    def test_the_default_pass_budget_is_eight(self) -> None:
        import inspect

        self.assertEqual(inspect.signature(fit).parameters["pass_budget"].default, 8)

    def test_the_default_cap_is_the_source_default(self) -> None:
        result = fit(self.source, "normal")
        self.assertEqual(result.scale.max_rows, DEFAULT_MAX_ROWS)
        self.assertEqual(DEFAULT_MAX_ROWS, 262144)


class ProvenanceTests(unittest.TestCase):
    def test_the_record_describes_a_numpy_fit(self) -> None:
        values = dataset(FamilyId.GAMMA, 123, 5)
        source = NumpySource(values)
        record = fit(source, "gamma", max_rows=50).scale
        (partition,) = source.partitions()
        self.assertIs(type(record), ScaleProvenance)
        self.assertEqual(record.schema_version, SCALE_PROVENANCE_SCHEMA_VERSION)
        self.assertEqual(record.source_kind, "numpy")
        self.assertEqual(record.partition_count, 1)
        self.assertEqual(record.row_count, 123)
        self.assertEqual(record.max_rows, 50)
        (item,) = record.partitions
        self.assertEqual(
            (item.partition_id, item.ordinal, item.rows, item.fingerprint),
            (partition.id, 0, 123, partition.fingerprint),
        )
        self.assertEqual(item.fingerprint_level, "content")

    def test_the_state_digest_is_the_sha256_of_the_canonical_state_bytes(self) -> None:
        for family in FAMILIES:
            values = dataset(family, 77, 6)
            record = fit(NumpySource(values), family).scale
            state = fold(NumpySource(values), STATE[family])
            assert record.state is not None
            with self.subTest(family=family):
                self.assertEqual(record.state.state_tag, state.STATE_TAG)
                self.assertEqual(record.state.state_schema_version, state.SCHEMA_VERSION)
                self.assertEqual(
                    record.state.digest_sha256, hashlib.sha256(state.to_bytes()).hexdigest()
                )

    def test_the_environment_is_the_numeric_environment(self) -> None:
        record = fit(NumpySource(dataset(FamilyId.NORMAL, 20, 1)), "normal").scale
        self.assertEqual(record.environment.numpy_version, numpy_version())
        self.assertEqual(record.environment.digest_sha256, numeric_environment_digest())

    def test_the_record_encodes_to_canonical_json(self) -> None:
        record = fit(NumpySource(dataset(FamilyId.NORMAL, 20, 1)), "normal").scale
        data = scale_provenance_to_canonical_json_bytes(record)
        self.assertIn(b'"source_kind":"numpy"', data)
        self.assertIn(record.environment.digest_sha256.encode(), data)

    def test_partitions_are_listed_once_and_each_is_read_once(self) -> None:
        source = ListSource([np.arange(1.0, 6.0), np.arange(6.0, 9.0), np.arange(9.0, 14.0)])
        record = fit(source, "exponential", max_rows=2).scale
        self.assertEqual(source.partition_calls, 1)
        self.assertEqual(source.batch_calls, [0, 1, 2])
        self.assertEqual(source.batch_sizes, [2, 2, 1, 2, 1, 2, 2, 1])
        self.assertEqual([item.partition_id for item in record.partitions], ["p0", "p1", "p2"])
        self.assertEqual([item.ordinal for item in record.partitions], [0, 1, 2])
        self.assertEqual([item.rows for item in record.partitions], [5, 3, 5])
        self.assertEqual({item.fingerprint for item in record.partitions}, {None})
        self.assertEqual({item.fingerprint_level for item in record.partitions}, {"none"})
        self.assertEqual(record.partition_count, 3)
        self.assertEqual(record.row_count, 13)

    def test_a_custom_source_is_named_custom(self) -> None:
        record = fit(ListSource([np.array([1.0, 2.0])]), "normal").scale
        self.assertEqual(record.source_kind, CUSTOM_SOURCE_KIND)
        self.assertEqual(CUSTOM_SOURCE_KIND, "custom")

    def test_the_bundled_sources_have_names(self) -> None:
        self.assertEqual(
            SOURCE_KINDS,
            {
                NumpySource: "numpy",
                ArrowSource: "arrow",
                ParquetSource: "parquet",
                PolarsSource: "polars",
                DuckDBSource: "duckdb",
            },
        )

    def test_a_polars_and_a_duckdb_fit_record_their_own_source_kind(self) -> None:
        values = np.array([1.0, 2.0, 4.0, 8.0])
        frame = pl.DataFrame({"x": values})
        self.assertEqual(fit(PolarsSource(frame, "x"), "gamma").scale.source_kind, "polars")
        connection = duckdb.connect()
        try:
            query = "select unnest([1.0, 2.0, 4.0, 8.0])::double as x"
            source = DuckDBSource((connection, query), "x")
            self.assertEqual(fit(source, "gamma").scale.source_kind, "duckdb")
        finally:
            connection.close()

    def test_an_arrow_source_is_not_resumable(self) -> None:
        table = pa.table({"x": np.array([1.0, 2.0, 4.0])})
        record = fit(ArrowSource(table, "x"), "gamma").scale
        self.assertEqual(record.source_kind, "arrow")
        (item,) = record.partitions
        self.assertEqual((item.fingerprint, item.fingerprint_level), (None, "none"))
        self.assertEqual(item.rows, 3)

    def test_a_parquet_source_records_every_row_group(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_parquet(root / "a.parquet", {"x": np.arange(1.0, 8.0)}, row_group_size=3)
            write_parquet(root / "b.parquet", {"x": np.arange(8.0, 11.0)})
            source = ParquetSource(root, "x")
            expected = source.partitions()
            record = fit(source, "exponential").scale
        self.assertEqual(record.source_kind, "parquet")
        self.assertEqual(record.partition_count, 4)
        self.assertEqual(record.row_count, 10)
        self.assertEqual(
            [(i.partition_id, i.ordinal, i.rows, i.fingerprint_level) for i in record.partitions],
            [(p.id, p.ordinal, p.rows, p.fingerprint_level.value) for p in expected],
        )
        self.assertEqual(
            [i.fingerprint for i in record.partitions], [p.fingerprint for p in expected]
        )
        self.assertEqual({i.fingerprint_level for i in record.partitions}, {"metadata"})

    def test_an_empty_source_has_an_empty_state_record(self) -> None:
        record = fit(NumpySource(np.empty(0)), "normal").scale
        self.assertEqual(record.row_count, 0)
        assert record.state is not None
        self.assertEqual(
            record.state.digest_sha256,
            hashlib.sha256(NormalState.empty().to_bytes()).hexdigest(),
        )
        (item,) = record.partitions
        self.assertEqual(item.rows, 0)


class FailureTests(unittest.TestCase):
    def test_an_empty_sample_is_the_in_memory_failure_of_each_family(self) -> None:
        empty = NumpySource(np.empty(0))
        self.assertEqual(
            fit(empty, "exponential").fit,
            ExponentialFitFailure(ExponentialFitFailureCode.EMPTY_SAMPLE, 0, 0, 0.0),
        )
        self.assertEqual(
            fit(empty, "normal").fit, NormalFitFailure(NormalFitFailureCode.EMPTY_SAMPLE, 0, 0, 0)
        )
        self.assertEqual(
            fit(empty, "gamma").fit, GammaFitFailure(GammaFitFailureCode.EMPTY_SAMPLE, 0, 0, 0)
        )
        self.assertEqual(
            fit(empty, "lognormal").fit,
            LognormalFitFailure(LognormalFitFailureCode.EMPTY_SAMPLE, 0, 0, 0),
        )

    def test_degenerate_samples_are_the_in_memory_failures(self) -> None:
        same = NumpySource(np.full(6, 0.3))
        self.assertEqual(
            fit(same, "normal").fit,
            NormalFitFailure(NormalFitFailureCode.DEGENERATE_SAMPLE, 6, 6, 0),
        )
        self.assertEqual(
            fit(same, "gamma").fit,
            GammaFitFailure(GammaFitFailureCode.DEGENERATE_SAMPLE, 6, 6, 0),
        )
        self.assertEqual(
            fit(same, "lognormal").fit,
            LognormalFitFailure(LognormalFitFailureCode.OPTIMIZER_EXHAUSTED, 6, 6, 0),
        )
        self.assertEqual(
            fit(NumpySource(np.zeros(4)), "exponential").fit,
            ExponentialFitFailure(ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD, 4, 4, 0.0),
        )

    def test_a_failure_still_carries_provenance_of_its_state(self) -> None:
        record = fit(NumpySource(np.full(6, 0.3)), "normal").scale
        assert record.state is not None
        self.assertEqual(record.row_count, 6)
        self.assertEqual(record.state.state_tag, "scale.normal")

    def test_non_positive_values_are_invalid_support_for_the_lifetime_families(self) -> None:
        values = np.array([2.0, 0.0, 3.0, 5.0])
        gamma = fit(NumpySource(values), "gamma")
        self.assertEqual(
            gamma.fit, GammaFitFailure(GammaFitFailureCode.INVALID_SUPPORT, 4, 4, 0)
        )
        lognormal = fit(NumpySource(values), "lognormal")
        self.assertEqual(
            lognormal.fit, LognormalFitFailure(LognormalFitFailureCode.INVALID_SUPPORT, 4, 4, 0)
        )
        for result in (gamma, lognormal):
            self.assertIsNone(result.scale.state)
            self.assertEqual(result.scale.row_count, 4)
            self.assertEqual(result.scale.partition_count, 1)

    def test_invalid_support_counts_every_row_of_every_partition(self) -> None:
        source = ListSource([np.array([-1.0, 2.0, 3.0]), np.array([4.0, 5.0]), np.array([6.0])])
        result = fit(source, "gamma", max_rows=2)
        self.assertEqual(result.fit, GammaFitFailure(GammaFitFailureCode.INVALID_SUPPORT, 6, 6, 0))
        self.assertEqual(result.scale.row_count, 6)
        self.assertEqual(source.partition_calls, 1)
        self.assertEqual(max(source.batch_sizes), 2)

    def test_the_support_failure_is_found_in_a_later_partition(self) -> None:
        source = ListSource([np.array([1.0, 2.0]), np.array([3.0, -4.0])])
        result = fit(source, "lognormal")
        self.assertEqual(
            result.fit, LognormalFitFailure(LognormalFitFailureCode.INVALID_SUPPORT, 4, 4, 0)
        )

    def test_negative_times_are_refused_for_the_exponential_not_reported(self) -> None:
        with self.assertRaises(ScaleStateError) as caught:
            fit(NumpySource(np.array([1.0, -2.0])), "exponential")
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)

    def test_other_state_refusals_propagate_unchanged(self) -> None:
        bypass = ListSource([np.array([1.0, np.inf])])
        for family in ("gamma", "lognormal", "exponential", "normal"):
            with self.subTest(family=family), self.assertRaises(ScaleStateError) as again:
                fit(bypass, family)
            self.assertEqual(again.exception.code, ScaleStateErrorCode.NON_FINITE_VALUE)

    def test_value_policy_refusals_propagate_unchanged(self) -> None:
        for family in FAMILIES:
            for data, code in (
                (np.array([1.0, np.nan]), ScaleSourceErrorCode.NAN_VALUE),
                (np.array([1.0, np.inf]), ScaleSourceErrorCode.NON_FINITE_VALUE),
                (np.array([1, 2**60]), ScaleSourceErrorCode.LOSSY_CAST),
            ):
                with self.subTest(family=family, code=code), self.assertRaises(
                    ScaleSourceError
                ) as caught:
                    fit(NumpySource(data), family)
                self.assertEqual(caught.exception.code, code)

    def test_nulls_in_arrow_data_are_refused(self) -> None:
        table = pa.table({"x": pa.array([1.0, None, 3.0])})
        with self.assertRaises(ScaleSourceError) as caught:
            fit(ArrowSource(table, "x"), "normal")
        self.assertEqual(caught.exception.code, ScaleSourceErrorCode.NULL_VALUE)

    def test_a_refusal_in_the_counting_pass_propagates(self) -> None:
        class Flaky(ListSource):
            def batches(
                self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
            ) -> Iterator[Any]:
                yield from super().batches(partition, max_rows=max_rows)
                if len(self.batch_calls) > 1:
                    raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE)

        with self.assertRaises(ScaleSourceError) as caught:
            fit(Flaky([np.array([-1.0, 2.0])]), "gamma")
        self.assertEqual(caught.exception.code, ScaleSourceErrorCode.SOURCE_UNREADABLE)


if __name__ == "__main__":
    unittest.main()
