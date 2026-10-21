"""The public contract of ``veridist.scale``: the curated names, signatures and error codes.

Every name in ``veridist.scale.__all__`` is documented public API, so the set is pinned here
exactly: adding or removing a name is a deliberate, reviewed change. The behavioural checks
below use only the public names, as a user does.
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
import tempfile
import unittest
from enum import StrEnum
from pathlib import Path

import numpy as np
import pyarrow as pa

import veridist
from veridist import scale
from veridist.engine.errors import EngineContractError, VeridistError
from veridist.families.results import FitFailure, FitSuccess

PUBLIC_NAMES = {
    "ArrowSource",
    "DuckDBSource",
    "GofDecision",
    "LargeNGof",
    "NumpySource",
    "ParquetSource",
    "PassBudgetError",
    "PolarsSource",
    "ProcessExecutor",
    "ProfileIntervals",
    "ProfileResolution",
    "ProfileUnavailable",
    "ProfileUnavailableReason",
    "Quantity",
    "QuantityKind",
    "SQLitePartitionStore",
    "ScaleCheckpointError",
    "ScaleCheckpointErrorCode",
    "ScaleCheckpointPassProvenance",
    "ScaleCheckpointProvenance",
    "ScaleEnvironmentProvenance",
    "ScaleExecutionPassProvenance",
    "ScaleExecutionProvenance",
    "ScaleFitResult",
    "ScaleInferenceError",
    "ScaleInferenceErrorCode",
    "ScalePartitionProvenance",
    "ScalePassProvenance",
    "ScalePassRecord",
    "ScaleProvenance",
    "ScaleSourceError",
    "ScaleSourceErrorCode",
    "ScaleSpoolPartitionProvenance",
    "ScaleSpoolProvenance",
    "ScaleStateError",
    "ScaleStateErrorCode",
    "ScaleStateProvenance",
    "ScaleSpoolPolicy",
    "ThreadExecutor",
    "fit",
    "large_n_gof",
    "mean",
    "profile_intervals",
    "quantile",
    "survival",
}

CODES = {
    scale.ScaleSourceErrorCode: [
        "OPTIONAL_DEPENDENCY_MISSING",
        "COLUMN_NOT_FOUND",
        "UNSUPPORTED_DTYPE",
        "LOSSY_CAST",
        "NULL_VALUE",
        "NAN_VALUE",
        "NON_FINITE_VALUE",
        "SOURCE_NOT_FOUND",
        "SOURCE_UNREADABLE",
        "SOURCE_CONSUMED",
        "PARTITION_CHANGED",
        "SPOOL_REQUIRED",
        "SPOOL_LIMIT_EXCEEDED",
        "SPOOL_CHANGED",
        "SPOOL_UNAVAILABLE",
    ],
    scale.ScaleStateErrorCode: [
        "NON_FINITE_VALUE",
        "VALUE_OUT_OF_SUPPORT",
        "OBSERVATION_LIMIT_EXCEEDED",
        "TOTAL_NOT_REPRESENTABLE",
        "INCOMPATIBLE_STATE",
        "STATE_BYTES_INVALID",
        "STATE_TAG_MISMATCH",
        "STATE_VERSION_UNSUPPORTED",
    ],
    scale.ScaleCheckpointErrorCode: [
        "SOURCE_REVISION_UNAVAILABLE",
        "INCREMENTAL_UNSUPPORTED",
        "JOB_EXISTS",
        "JOB_MISMATCH",
        "PARTITION_MISSING",
        "PARTITION_ADDED",
        "PARTITION_STATE_CONFLICT",
        "LEDGER_MISMATCH",
        "STORE_FORMAT_UNSUPPORTED",
        "STORE_DECODE_FAILED",
        "STORE_DIGEST_MISMATCH",
        "STORE_STORAGE_FAILED",
        "STORE_COMMIT_UNCERTAIN",
    ],
    scale.ScaleInferenceErrorCode: [
        "FIT_NOT_PROFILABLE",
        "PROFILE_NOT_CONVERGED",
        "PROFILE_UNBOUNDED",
        "PROFILE_NOT_COMPUTABLE",
        "GOF_FIT_NOT_AVAILABLE",
        "GOF_NOT_REPRESENTABLE",
        "GOF_RESOLUTION_UNSUPPORTED",
    ],
    scale.GofDecision: ["PASS", "FAIL", "INCONCLUSIVE"],
    scale.ProfileUnavailableReason: ["NOT_IMPLEMENTED", "NOT_REPRESENTABLE"],
    scale.QuantityKind: ["mean", "quantile", "survival"],
}


def parameters(function: object) -> list[tuple[str, str, object]]:
    return [
        (name, item.kind.name, item.default if item.default is not item.empty else "<required>")
        for name, item in inspect.signature(function).parameters.items()  # type: ignore[arg-type]
    ]


def weibull_quantiles(count: int, shape: float, scale_: float) -> np.ndarray:
    probability = (np.arange(count) + 0.5) / count
    return scale_ * (-np.log1p(-probability)) ** (1.0 / shape)


class PublicNamesTests(unittest.TestCase):
    def test_the_public_names_are_exactly_these(self) -> None:
        self.assertEqual(set(scale.__all__), PUBLIC_NAMES)

    def test_every_name_is_listed_once_sorted_and_importable(self) -> None:
        self.assertEqual(len(scale.__all__), len(set(scale.__all__)))
        self.assertEqual(scale.__all__, sorted(scale.__all__))
        for name in scale.__all__:
            with self.subTest(name=name):
                self.assertIsNotNone(getattr(scale, name))

    def test_every_name_is_the_object_its_defining_module_holds(self) -> None:
        for name in scale.__all__:
            with self.subTest(name=name):
                item = getattr(scale, name)
                module = importlib.import_module(item.__module__)
                self.assertIs(getattr(module, item.__name__), item)

    def test_a_star_import_brings_in_exactly_the_public_names(self) -> None:
        namespace: dict[str, object] = {}
        exec("from veridist.scale import *", namespace)  # noqa: S102
        namespace.pop("__builtins__")
        self.assertEqual(set(namespace), PUBLIC_NAMES)

    def test_the_package_no_longer_calls_itself_internal(self) -> None:
        text = (scale.__doc__ or "").casefold()
        self.assertNotIn("internal", text)
        self.assertNotIn("unstable", text)
        self.assertIn("deprecat", text)
        self.assertIn("__all__", text)

    def test_the_top_level_package_does_not_export_the_scale_names(self) -> None:
        self.assertFalse((PUBLIC_NAMES - {"fit"}) & set(veridist.__all__))
        self.assertIsNot(veridist.fit, scale.fit)

    def test_importing_the_package_loads_no_array_or_optional_library(self) -> None:
        code = (
            "import sys, veridist.scale; "
            "print(sorted(m for m in ('numpy', 'pyarrow', 'polars', 'duckdb') if m in sys.modules))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
        )
        self.assertEqual(completed.stdout.strip(), "[]")

    def test_importing_the_top_level_package_does_not_import_the_scale_package(self) -> None:
        code = "import sys, veridist; print('veridist.scale' in sys.modules)"
        completed = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=120
        )
        self.assertEqual(completed.stdout.strip(), "False")


class SignatureTests(unittest.TestCase):
    def test_fit(self) -> None:
        self.assertEqual(
            parameters(scale.fit),
            [
                ("source", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("family", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("max_rows", "KEYWORD_ONLY", 262144),
                ("pass_budget", "KEYWORD_ONLY", 8),
                ("spool", "KEYWORD_ONLY", None),
                ("executor", "KEYWORD_ONLY", None),
                ("checkpoint", "KEYWORD_ONLY", None),
                ("resume", "KEYWORD_ONLY", False),
                ("incremental", "KEYWORD_ONLY", False),
                ("recompute_changed", "KEYWORD_ONLY", False),
                ("allow_removed", "KEYWORD_ONLY", False),
            ],
        )

    def test_profile_intervals(self) -> None:
        self.assertEqual(
            parameters(scale.profile_intervals),
            [
                ("source", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("result", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("level", "KEYWORD_ONLY", 0.95),
                ("quantities", "KEYWORD_ONLY", ()),
                ("inference_pass_budget", "KEYWORD_ONLY", 6),
                ("max_rows", "KEYWORD_ONLY", 262144),
                ("spool", "KEYWORD_ONLY", None),
                ("executor", "KEYWORD_ONLY", None),
                ("checkpoint", "KEYWORD_ONLY", None),
                ("resume", "KEYWORD_ONLY", False),
                ("recompute_changed", "KEYWORD_ONLY", False),
            ],
        )

    def test_large_n_gof(self) -> None:
        self.assertEqual(
            parameters(scale.large_n_gof),
            [
                ("source", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("result", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("bin_exponent", "KEYWORD_ONLY", 12),
                ("threshold", "KEYWORD_ONLY", None),
                ("inference_pass_budget", "KEYWORD_ONLY", 1),
                ("max_rows", "KEYWORD_ONLY", 262144),
                ("executor", "KEYWORD_ONLY", None),
                ("checkpoint", "KEYWORD_ONLY", None),
                ("resume", "KEYWORD_ONLY", False),
                ("recompute_changed", "KEYWORD_ONLY", False),
            ],
        )

    def test_the_sources(self) -> None:
        self.assertEqual(
            parameters(scale.NumpySource),
            [
                ("values", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("partition_rows", "KEYWORD_ONLY", None),
            ],
        )
        self.assertEqual(
            parameters(scale.ParquetSource),
            [
                ("paths", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("column", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("root", "KEYWORD_ONLY", None),
                ("content_hash", "KEYWORD_ONLY", False),
            ],
        )
        self.assertEqual(
            parameters(scale.ArrowSource),
            [
                ("data", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("column", "POSITIONAL_OR_KEYWORD", "<required>"),
            ],
        )
        self.assertEqual(
            parameters(scale.PolarsSource),
            [
                ("frame", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("column", "POSITIONAL_OR_KEYWORD", "<required>"),
            ],
        )
        self.assertEqual(
            parameters(scale.DuckDBSource),
            [
                ("source", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("column", "POSITIONAL_OR_KEYWORD", "<required>"),
            ],
        )

    def test_the_executors_the_spool_and_the_store(self) -> None:
        for executor in (scale.ThreadExecutor, scale.ProcessExecutor):
            with self.subTest(executor=executor.__name__):
                self.assertEqual(
                    parameters(executor)[0], ("workers", "POSITIONAL_OR_KEYWORD", "<required>")
                )
        self.assertEqual(
            parameters(scale.ScaleSpoolPolicy),
            [
                ("max_bytes", "KEYWORD_ONLY", "<required>"),
                ("directory", "KEYWORD_ONLY", None),
                ("allow_system_temp", "KEYWORD_ONLY", False),
            ],
        )
        self.assertEqual(
            parameters(scale.SQLitePartitionStore),
            [
                ("path", "POSITIONAL_OR_KEYWORD", "<required>"),
                ("timeout", "KEYWORD_ONLY", 5.0),
            ],
        )


class ErrorAndCodeTests(unittest.TestCase):
    def test_the_closed_code_sets(self) -> None:
        for enum, values in CODES.items():
            with self.subTest(enum=enum.__name__):
                self.assertTrue(issubclass(enum, StrEnum))
                self.assertEqual([member.value for member in enum], values)
                self.assertEqual([member.name for member in enum], [v.upper() for v in values])

    def test_the_errors_are_typed_value_errors_of_one_hierarchy(self) -> None:
        for error in (
            scale.ScaleSourceError,
            scale.ScaleStateError,
            scale.ScaleCheckpointError,
            scale.ScaleInferenceError,
        ):
            with self.subTest(error=error.__name__):
                self.assertTrue(issubclass(error, VeridistError))
                self.assertTrue(issubclass(error, ValueError))
        self.assertTrue(issubclass(scale.PassBudgetError, EngineContractError))

    def test_an_error_carries_its_code_and_nothing_else(self) -> None:
        error = scale.ScaleInferenceError("FIT_NOT_PROFILABLE")
        self.assertIs(error.code, scale.ScaleInferenceErrorCode.FIT_NOT_PROFILABLE)
        self.assertEqual(str(error), "FIT_NOT_PROFILABLE")
        with self.assertRaises(ValueError):
            scale.ScaleInferenceError("NOT_A_CODE")


class PublicFlowTests(unittest.TestCase):
    """Fit, execute, resume, bound and test a fit using only the public names."""

    def test_every_family_fits_through_the_public_entry_point(self) -> None:
        values = weibull_quantiles(2_000, 1.6, 40.0)
        source = scale.NumpySource(values, partition_rows=500)
        for family in (
            "exponential",
            "normal",
            "gamma",
            "lognormal",
            "weibull_min",
            "gumbel_right",
        ):
            with self.subTest(family=family):
                result = scale.fit(source, family)
                self.assertIsInstance(result, scale.ScaleFitResult)
                self.assertIsInstance(result.fit, FitSuccess)
                self.assertIsInstance(result.scale, scale.ScaleProvenance)
                self.assertEqual(result.scale.row_count, 2_000)
                self.assertEqual(result.scale.partition_count, 4)
                self.assertEqual(result.scale.source_kind, "numpy")
                multi = family in ("weibull_min", "gumbel_right")
                self.assertEqual(result.scale.passes is not None, multi)

    def test_a_failed_fit_is_a_value_and_a_refusal_is_raised(self) -> None:
        negative = scale.NumpySource(np.array([-1.0, 2.0, 3.0]))
        failure = scale.fit(negative, "gamma").fit
        self.assertIsInstance(failure, FitFailure)
        with self.assertRaises(scale.PassBudgetError) as caught:
            scale.fit(scale.NumpySource(np.arange(1.0, 50.0)), "weibull_min", pass_budget=2)
        self.assertEqual(caught.exception.code.value, "PASS_BUDGET_INSUFFICIENT")
        with self.assertRaises(scale.ScaleStateError) as refused:
            scale.fit(scale.NumpySource(np.array([-1.0, 2.0])), "exponential")
        self.assertIs(refused.exception.code, scale.ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)
        with self.assertRaises(scale.ScaleSourceError) as dtype:
            scale.NumpySource(np.array(["a"]))
        self.assertIs(dtype.exception.code, scale.ScaleSourceErrorCode.UNSUPPORTED_DTYPE)

    def test_an_executor_changes_neither_the_result_nor_the_state(self) -> None:
        source = scale.NumpySource(weibull_quantiles(3_000, 1.3, 25.0), partition_rows=700)
        serial = scale.fit(source, "weibull_min")
        threaded = scale.fit(source, "weibull_min", executor=scale.ThreadExecutor(3))
        self.assertEqual(threaded.fit.parameters, serial.fit.parameters)
        self.assertEqual(threaded.scale.state, serial.scale.state)
        assert threaded.scale.execution is not None
        self.assertEqual(threaded.scale.execution.executor, "threads")
        named = scale.fit(source, "weibull_min", executor="serial")
        assert named.scale.execution is not None
        self.assertEqual(named.scale.execution.executor, "serial")

    def test_a_checkpoint_resumes_and_is_refused_when_it_does_not_apply(self) -> None:
        source = scale.NumpySource(weibull_quantiles(1_200, 1.4, 10.0), partition_rows=300)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "job.sqlite"
            first = scale.fit(source, "normal", checkpoint=path, resume=True)
            again = scale.fit(
                source, "normal", checkpoint=scale.SQLitePartitionStore(path), resume=True
            )
            self.assertEqual(again.fit.parameters, first.fit.parameters)
            assert first.scale.checkpoint is not None and again.scale.checkpoint is not None
            self.assertEqual(
                (again.scale.checkpoint.resumed, again.scale.checkpoint.read_partitions),
                (True, 0),
            )
            with self.assertRaises(scale.ScaleCheckpointError) as exists:
                scale.fit(source, "normal", checkpoint=path)
            self.assertIs(exists.exception.code, scale.ScaleCheckpointErrorCode.JOB_EXISTS)
            with self.assertRaises(scale.ScaleCheckpointError) as other:
                scale.fit(source, "gamma", checkpoint=path, resume=True)
            self.assertIs(other.exception.code, scale.ScaleCheckpointErrorCode.JOB_MISMATCH)
            self.assertIn("family", other.exception.components)

    def test_a_single_use_source_needs_a_spool_for_a_multi_pass_family(self) -> None:
        table = pa.table({"x": weibull_quantiles(900, 1.5, 12.0)})
        reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches(300))
        source = scale.ArrowSource(reader, "x")
        with self.assertRaises(scale.ScaleSourceError) as refused:
            scale.fit(source, "weibull_min")
        self.assertIs(refused.exception.code, scale.ScaleSourceErrorCode.SPOOL_REQUIRED)
        with tempfile.TemporaryDirectory() as directory:
            policy = scale.ScaleSpoolPolicy(max_bytes=1_000_000, directory=directory)
            result = scale.fit(source, "weibull_min", spool=policy)
        assert result.scale.spool is not None
        self.assertEqual(result.scale.spool.row_count, 900)

    def test_intervals_and_the_large_n_report_through_the_public_names(self) -> None:
        source = scale.NumpySource(weibull_quantiles(4_000, 1.5, 100.0), partition_rows=1_000)
        result = scale.fit(source, "weibull_min")
        intervals = scale.profile_intervals(
            source, result, quantities=[scale.mean(), scale.quantile(0.1), scale.survival(50.0)]
        )
        self.assertIsInstance(intervals, scale.ProfileIntervals)
        self.assertEqual(sorted(intervals.parameters), ["scale", "shape"])
        self.assertEqual(sorted(intervals.derived), ["mean", "quantile:0.1", "survival:50.0"])
        shape = intervals.parameters["shape"]
        self.assertLess(shape.lower, result.fit.parameters["shape"])  # type: ignore[attr-defined]
        self.assertGreater(shape.upper, result.fit.parameters["shape"])  # type: ignore[attr-defined]
        report = scale.large_n_gof(source, result, threshold=0.05)
        self.assertIsInstance(report, scale.LargeNGof)
        self.assertLessEqual(report.ks_lower, report.ks_upper)
        self.assertIs(report.decision, scale.GofDecision.PASS)
        with self.assertRaises(scale.ScaleInferenceError) as refused:
            scale.profile_intervals(source, scale.fit(source, "normal"))
        self.assertIs(refused.exception.code, scale.ScaleInferenceErrorCode.FIT_NOT_PROFILABLE)


if __name__ == "__main__":
    unittest.main()
