"""Public CSV/exponential API contract; legacy names never enter this surface."""

from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import veridist
from veridist.engine.outcome import CompleteOutcome, FailedOutcome
from veridist.families import ExponentialFitSuccess


class PublicCsvApiTests(unittest.TestCase):
    def test_public01_exports_the_documented_user_api(self) -> None:
        expected = {
            "__version__",
            "CapabilityError",
            "CsvLifetimeLimits",
            "CsvLifetimeSchema",
            "DataSourceMetadata",
            "EngineContractError",
            "ExactLifetime",
            "ExactValue",
            "ExponentialSourceFitResult",
            "FamilyId",
            "FitFailure",
            "FitSuccess",
            "IterableDataSource",
            "LifetimeObservation",
            "PublicSourceId",
            "RealObservation",
            "Replayability",
            "RightCensoredLifetime",
            "RightCensoredValue",
            "StreamSource",
            "StreamSourceError",
            "VeridistError",
            "cdf",
            "create_checkpointed_csv_store",
            "fit",
            "fit_exponential",
            "fit_exponential_checkpointed_chunks",
            "fit_exponential_checkpointed_csv",
            "fit_exponential_csv",
            "fit_gamma",
            "fit_gumbel_right",
            "fit_lognormal",
            "fit_normal",
            "fit_weibull",
            "lifetimes_from_arrays",
            "logpdf",
            "ppf",
            "reduce_lifetime_log_likelihood_chunks",
            "reduce_log_likelihood_chunks",
            "reduce_value_log_likelihood_chunks",
            "sample",
            "sf",
            "values_from_arrays",
        }
        self.assertEqual(set(veridist.__all__), expected)
        signature = inspect.signature(veridist.fit_exponential_csv)
        self.assertEqual(tuple(signature.parameters), ("path", "schema", "source_id", "limits"))
        self.assertEqual(signature.parameters["schema"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_public01b_every_exported_name_is_importable_and_listed_once(self) -> None:
        self.assertEqual(len(veridist.__all__), len(set(veridist.__all__)))
        for name in veridist.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(veridist, name))
                self.assertIsNotNone(getattr(veridist, name))

    def test_public01c_error_types_form_one_hierarchy(self) -> None:
        self.assertTrue(issubclass(veridist.EngineContractError, veridist.VeridistError))
        self.assertTrue(issubclass(veridist.CapabilityError, veridist.VeridistError))
        self.assertTrue(issubclass(veridist.StreamSourceError, veridist.EngineContractError))

    def test_public01d_top_level_import_does_not_load_numpy(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-c", "import sys, veridist; print('numpy' in sys.modules)"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        self.assertEqual(completed.stdout.strip(), "False")

    def test_public01e_engine_exports_capability_error(self) -> None:
        import veridist.engine as engine

        self.assertIn("CapabilityError", engine.__all__)
        self.assertIs(engine.CapabilityError, veridist.CapabilityError)

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
