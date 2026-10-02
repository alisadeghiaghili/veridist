"""Fail-closed contracts for exact-state likelihood scale evidence."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from veridist import DataSourceMetadata, IterableDataSource, Replayability
from veridist.families.registry import FamilyId

REPO = Path(__file__).parents[3]
TOOLS = REPO / "python" / "tools"
CHECKER = TOOLS / "check_log_likelihood_scale_evidence.py"
RUNNER = TOOLS / "run_log_likelihood_scale_evidence.py"
SPEC = importlib.util.spec_from_file_location("likelihood_scale_checker", CHECKER)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
RUNNER_SPEC = importlib.util.spec_from_file_location("likelihood_scale_runner", RUNNER)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)


def _head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


class LogLikelihoodScaleEvidenceTests(unittest.TestCase):
    def _value(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": "5",
            "run": {
                "git_sha": _head(),
                "candidate_git_sha": _head(),
                "git_dirty": False,
                "generator": "fixed-supported-family-v1",
                "source_contract": "public-iterable-data-source-v1",
                "python": {"implementation": "CPython", "version": "3.11"},
                "platform": "test-platform",
                "measurement_workers": 1,
                "methodology": dict(MODULE.METHODOLOGY),
            },
            "cells": [],
        }
        for family in MODULE.FAMILIES:
            for rows in MODULE.ROWS:
                for budget in MODULE.BUDGETS:
                    units = MODULE._units(rows, family)
                    expected = float(__import__("fractions").Fraction(units, 1 << 1074))
                    value["cells"].append(
                        {
                            "family": family,
                            "rows": rows,
                            "chunk_size": budget,
                            "one_pass": {"iterator_acquisitions": 1, "observation_yields": rows},
                            "oracle": {
                                "oracle_total_units": units,
                                "oracle_total_units_bit_length": abs(units).bit_length(),
                                "bound_bits": 2162,
                            },
                            "actual": {
                                "observation_count": rows,
                                "total_log_likelihood": expected,
                                "total_log_likelihood_hex": expected.hex(),
                            },
                            "elapsed_seconds": 1.0,
                            "throughput_rows_per_second": float(rows),
                            "memory": {
                                "tracemalloc_peak_bytes": 0,
                                "rss_peak_bytes": 0,
                                "rss_delta_bytes": 0,
                            },
                        }
                    )
        value["artifact_sha256"] = MODULE._digest(value)
        return value

    def test_runner_rejects_a_correct_count_with_a_wrong_returned_total(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodSuccess

        def wrong_total(
            family: FamilyId, chunks: object, /, **_parameters: object
        ) -> LogLikelihoodSuccess:
            from veridist.engine.streaming import iter_stream

            count = 0
            for chunk in iter_stream(chunks):
                for _ in chunk:
                    count += 1
            return LogLikelihoodSuccess(family, "0" * 64, count, 0.0)

        with patch.object(RUNNER_MODULE, "reduce_log_likelihood_chunks", side_effect=wrong_total):
            with self.assertRaisesRegex(
                RuntimeError, "returned total does not match independent exact oracle"
            ):
                RUNNER_MODULE._cell(10, 1)

    def test_runner_cell_uses_public_single_pass_source_and_exact_result(self) -> None:
        observed: list[object] = []

        class RecordingSource:
            def __init__(self, chunks: object, metadata: DataSourceMetadata) -> None:
                observed.append(metadata)
                self._delegate = IterableDataSource(chunks, metadata)

            def iter_chunks(self):
                return self._delegate.iter_chunks()

        with patch.object(RUNNER_MODULE, "IterableDataSource", RecordingSource):
            cell = RUNNER_MODULE._cell(10, 3)
        expected = DataSourceMetadata(
            source_id="scale-normal-10-3",
            schema_version="1",
            provenance_schema_version="1",
            replayability=Replayability.SINGLE_PASS,
            redaction_reason="generated",
        )
        # One single-pass source for the untraced timing pass and a fresh one for the memory pass.
        self.assertEqual(observed, [expected, expected])
        self.assertEqual(cell["one_pass"], {"iterator_acquisitions": 1, "observation_yields": 10})
        self.assertEqual(cell["actual"]["observation_count"], 10)

    def test_runner_rejects_a_second_outer_iterator_acquisition(self) -> None:
        from veridist.statistics.log_likelihood import LogLikelihoodSuccess

        def second_pass(
            family: FamilyId, chunks: object, /, **_parameters: object
        ) -> LogLikelihoodSuccess:
            iter_chunks = getattr(chunks, "iter_chunks")
            next(iter_chunks())
            next(iter_chunks())
            return LogLikelihoodSuccess(family, "0" * 64, 10, -1.0)

        with patch.object(RUNNER_MODULE, "reduce_log_likelihood_chunks", side_effect=second_pass):
            with self.assertRaisesRegex(Exception, "PASS_BUDGET_EXCEEDED"):
                RUNNER_MODULE._cell(10, 1)

    def test_checker_rejects_parallel_workers_and_unknown_methodology(self) -> None:
        self.assertEqual(
            MODULE.validate(self._value(), expected_git_sha=_head(), repo_root=REPO), []
        )
        for workers, message in (
            (2, "exactly one measurement worker"),
            (0, "must be positive"),
            (True, "must be positive"),
        ):
            with self.subTest(workers=workers):
                value = self._value()
                value["run"]["measurement_workers"] = workers
                value["artifact_sha256"] = MODULE._digest(value)
                errors = MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO)
                self.assertTrue(any(message in error for error in errors), errors)
        for methodology in (
            {"passes": "traced-single-pass", "rss": "process-peak-v1"},
            {"passes": "elapsed-untraced-then-memory-traced-v1", "rss": "current"},
            {},
        ):
            with self.subTest(methodology=methodology):
                value = self._value()
                value["run"]["methodology"] = methodology
                value["artifact_sha256"] = MODULE._digest(value)
                self.assertIn(
                    "run measurement methodology is not supported",
                    MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO),
                )

    def test_checker_rejects_the_previous_schema(self) -> None:
        value = self._value()
        value["schema_version"] = "4"
        value["artifact_sha256"] = MODULE._digest(value)
        self.assertIn(
            "artifact version or digest invalid",
            MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO),
        )

    def test_runner_times_an_untraced_pass_then_measures_memory_in_a_separate_pass(self) -> None:
        import tracemalloc

        tracing: list[bool] = []
        real_reduce = RUNNER_MODULE.reduce_log_likelihood_chunks

        def spy(*arguments: object, **keywords: object) -> object:
            tracing.append(tracemalloc.is_tracing())
            return real_reduce(*arguments, **keywords)

        with patch.object(RUNNER_MODULE, "reduce_log_likelihood_chunks", side_effect=spy):
            cell = RUNNER_MODULE._cell(10, 3)
        self.assertEqual(tracing, [False, True])
        self.assertFalse(tracemalloc.is_tracing())
        self.assertGreater(cell["memory"]["tracemalloc_peak_bytes"], 0)
        self.assertGreater(cell["memory"]["rss_peak_bytes"], 0)
        self.assertEqual(cell["one_pass"], {"iterator_acquisitions": 1, "observation_yields": 10})

    def test_runner_refuses_a_memory_pass_that_differs_from_the_timed_reduction(self) -> None:
        real_reduce = RUNNER_MODULE.reduce_log_likelihood_chunks
        calls: list[object] = []

        def drifting(family: FamilyId, chunks: object, /, **parameters: object) -> object:
            calls.append(chunks)
            if len(calls) == 1:
                return real_reduce(family, chunks, **parameters)
            return real_reduce(family, chunks, **{**parameters, "mu": 1.0})

        with patch.object(RUNNER_MODULE, "reduce_log_likelihood_chunks", side_effect=drifting):
            with self.assertRaisesRegex(RuntimeError, "memory pass did not reproduce"):
                RUNNER_MODULE._cell(10, 3)

    def test_checker_rejects_tampered_actual_returned_total(self) -> None:
        value = self._value()
        cell = value["cells"][0]
        assert isinstance(cell, dict)
        actual = cell["actual"]
        assert isinstance(actual, dict)
        actual["total_log_likelihood"] = 0.0
        actual["total_log_likelihood_hex"] = (0.0).hex()
        value["artifact_sha256"] = MODULE._digest(value)
        self.assertIn(
            "actual returned total does not match independent exact oracle",
            MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO),
        )

    def test_checker_rejects_tampered_exact_oracle(self) -> None:
        value = self._value()
        self.assertEqual(MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO), [])
        cell = value["cells"][0]
        assert isinstance(cell, dict)
        oracle = cell["oracle"]
        assert isinstance(oracle, dict)
        oracle["oracle_total_units"] += 1
        value["artifact_sha256"] = MODULE._digest(value)
        self.assertIn(
            "independent exact oracle mismatch",
            MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO),
        )

    def test_checker_rejects_transplanted_candidate_sha(self) -> None:
        value = self._value()
        run = value["run"]
        assert isinstance(run, dict)
        run["candidate_git_sha"] = "0" * 40
        value["artifact_sha256"] = MODULE._digest(value)
        self.assertIn(
            "candidate Git SHA mismatch",
            MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO),
        )

    def test_checker_rejects_missing_runtime_measurement_facts(self) -> None:
        value = self._value()
        run = value["run"]
        assert isinstance(run, dict)
        del run["platform"]
        cell = value["cells"][0]
        assert isinstance(cell, dict)
        del cell["throughput_rows_per_second"]
        value["artifact_sha256"] = MODULE._digest(value)
        errors = MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO)
        self.assertIn("run schema invalid", errors)

    def test_checker_rejects_incomplete_family_matrix(self) -> None:
        value = self._value()
        cells = value["cells"]
        assert isinstance(cells, list)
        value["cells"] = [cell for cell in cells if cell["family"] != "gamma"]
        value["artifact_sha256"] = MODULE._digest(value)
        self.assertIn(
            "full five-family 10k/100k/1m by three-chunk matrix required",
            MODULE.validate(value, expected_git_sha=_head(), repo_root=REPO),
        )

    def test_runner_smoke_output_declares_current_public_source_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            command = [str(RUNNER), "--output", str(output)]
            with (
                patch.object(RUNNER_MODULE, "ROWS", (10,)),
                patch.object(RUNNER_MODULE, "BUDGETS", (1,)),
                patch.object(RUNNER_MODULE, "_head", return_value="a" * 40),
                patch.object(sys, "argv", command),
            ):
                self.assertEqual(RUNNER_MODULE.main(), 0)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["schema_version"], "5")
            self.assertEqual(artifact["run"]["candidate_git_sha"], "a" * 40)
            self.assertEqual(artifact["run"]["measurement_workers"], 1)
            self.assertEqual(artifact["run"]["methodology"], MODULE.METHODOLOGY)
            self.assertEqual(artifact["run"]["source_contract"], "public-iterable-data-source-v1")
            self.assertEqual(len(artifact["cells"]), len(RUNNER_MODULE.FAMILY_CASES))
