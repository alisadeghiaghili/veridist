"""Public CSV/exponential API contract; legacy names never enter this surface."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import veridist
from veridist.engine.outcome import CompleteOutcome, FailedOutcome
from veridist.families import ExponentialFitSuccess


class PublicCsvApiTests(unittest.TestCase):
    def test_public01_exports_only_the_narrow_csv_vertical_contract(self) -> None:
        expected = {
            "__version__",
            "CsvLifetimeLimits",
            "CsvLifetimeSchema",
            "DataSourceMetadata",
            "ExponentialSourceFitResult",
            "FamilyId",
            "IterableDataSource",
            "PublicSourceId",
            "Replayability",
            "StreamSource",
            "StreamSourceError",
            "fit_exponential_csv",
            "fit_exponential_checkpointed_chunks",
            "reduce_log_likelihood_chunks",
        }
        self.assertEqual(set(veridist.__all__), expected)
        signature = inspect.signature(veridist.fit_exponential_csv)
        self.assertEqual(tuple(signature.parameters), ("path", "schema", "source_id", "limits"))
        self.assertEqual(signature.parameters["schema"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_public02_fits_strict_csv_in_one_complete_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
            result = veridist.fit_exponential_csv(
                source,
                schema=veridist.CsvLifetimeSchema("time", "event_observed"),
                source_id=veridist.PublicSourceId("src_0123456789abcdef0123456789abcdef"),
                limits=veridist.CsvLifetimeLimits(4096, 4096),
            )
        self.assertIsInstance(result, veridist.ExponentialSourceFitResult)
        self.assertIsInstance(result.execution.outcome, CompleteOutcome)
        self.assertIsInstance(result.fit, ExponentialFitSuccess)
        assert isinstance(result.fit, ExponentialFitSuccess)
        self.assertEqual(result.fit.rate, 0.5)
        self.assertEqual(result.execution.provenance.execution.passes.actual_pass_count, 1)

    def test_public02b_fits_strict_csv_from_a_str_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
            result = veridist.fit_exponential_csv(
                str(source),
                schema=veridist.CsvLifetimeSchema("time", "event_observed"),
                source_id=veridist.PublicSourceId("src_0123456789abcdef0123456789abcdef"),
                limits=veridist.CsvLifetimeLimits(4096, 4096),
            )
        self.assertIsInstance(result.fit, ExponentialFitSuccess)
        assert isinstance(result.fit, ExponentialFitSuccess)
        self.assertEqual(result.fit.rate, 0.5)

    def test_public03_csv_failure_is_typed_result_without_a_path_leak(self) -> None:
        result = veridist.fit_exponential_csv(
            Path("this-path-must-not-appear.csv"),
            schema=veridist.CsvLifetimeSchema("time", "event_observed"),
            source_id=veridist.PublicSourceId("src_0123456789abcdef0123456789abcdef"),
            limits=veridist.CsvLifetimeLimits(4096, 4096),
        )
        self.assertIsNone(result.fit)
        self.assertIsInstance(result.execution.outcome, FailedOutcome)
        self.assertNotIn("this-path-must-not-appear", repr(result.execution))


if __name__ == "__main__":
    unittest.main()
