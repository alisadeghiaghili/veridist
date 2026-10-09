"""Adversarial contracts for the retained CSV/exponential scale evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).parents[2] / "tools"
CHECKER = TOOLS / "check_scale_csv_exponential_evidence.py"
RUNNER = TOOLS / "run_scale_csv_exponential_evidence.py"
REPO = Path(__file__).parents[3]
CHECKER_SPEC = importlib.util.spec_from_file_location("scale_checker", CHECKER)
assert CHECKER_SPEC is not None and CHECKER_SPEC.loader is not None
CHECKER_MODULE = importlib.util.module_from_spec(CHECKER_SPEC)
CHECKER_SPEC.loader.exec_module(CHECKER_MODULE)
RUNNER_SPEC = importlib.util.spec_from_file_location("scale_runner", RUNNER)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
RUNNER_MODULE = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER_MODULE)


def _head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _seal(value: dict[str, object]) -> None:
    body = dict(value)
    body.pop("artifact_sha256", None)
    value["artifact_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _cell(rows: int, budget: int) -> dict[str, object]:
    events, total, byte_count, source_hash = CHECKER_MODULE._fixture_facts(rows, "1")
    expected_rate = Decimal(events) / total
    rate = float(expected_rate)
    actual_rate = Decimal(str(rate))
    absolute = abs(actual_rate - expected_rate)
    relative = absolute / abs(expected_rate)
    return {
        "rows": rows,
        "chunk_bytes": budget,
        "max_inflight_bytes": budget,
        "source": {"bytes": byte_count, "sha256": source_hash},
        "observed": {
            "actual_pass_count": 1,
            "max_passes": 1,
            "accepted_chunk_count": budget // 1024,
            "processed_row_count": rows,
            "peak_inflight_bytes": min(128, budget),
            "largest_retained_chunk_bytes": min(128, budget),
            "backpressure_event_count": 0,
        },
        "fit": {
            "observation_count": rows,
            "event_count": events,
            "total_time": float(total),
            "rate": rate,
            "expected_event_count": events,
            "expected_total_time": str(total),
            "expected_rate": str(expected_rate),
            "absolute_rate_error": float(absolute),
            "relative_rate_error": float(relative),
        },
        "memory": {"tracemalloc_peak_bytes": 1, "rss_peak_bytes": 1, "rss_delta_bytes": 0},
        "elapsed_seconds": 0.01,
        "throughput_rows_per_second": rows / 0.01,
    }


def _smoke_artifact() -> dict[str, object]:
    row, budgets = 100, (2048, 4096, 8192)
    cells = [_cell(row, budget) for budget in budgets]
    value: dict[str, object] = {
        "schema_version": "3",
        "run": {
            "git_sha": _head(),
            "candidate_git_sha": _head(),
            "git_dirty": False,
            "utc_started": "2026-08-27T00:00:00Z",
            "python": {"implementation": "CPython", "version": "3.11.0"},
            "platform": "test-platform",
            "measurement_workers": 1,
            "timing": {"clock": "time.perf_counter_ns", "preflight": "paired-monotonic-wall-v2"},
            "methodology": dict(CHECKER_MODULE.METHODOLOGY),
        },
        "generator": {"formula_version": "1", "temporary_root": "redacted"},
        "cells": cells,
        "operation_evidence": {
            "rows": [row],
            "accepted_chunks": [cells[0]["observed"]["accepted_chunk_count"]],
        },
    }
    _seal(value)
    return value


def _full_artifact() -> dict[str, object]:
    rows, budgets = (10_000, 100_000, 1_000_000), (32_768, 65_536, 131_072)
    cells = [_cell(row, budget) for row in rows for budget in budgets]
    value: dict[str, object] = {
        "schema_version": "3",
        "run": {
            "git_sha": _head(),
            "candidate_git_sha": _head(),
            "git_dirty": False,
            "utc_started": "2026-08-27T00:00:00Z",
            "python": {"implementation": "CPython", "version": "3.11.0"},
            "platform": "test-platform",
            "measurement_workers": 1,
            "timing": {"clock": "time.perf_counter_ns", "preflight": "paired-monotonic-wall-v2"},
            "methodology": dict(CHECKER_MODULE.METHODOLOGY),
        },
        "generator": {"formula_version": "1", "temporary_root": "redacted"},
        "cells": cells,
        "operation_evidence": {
            "rows": list(rows),
            "accepted_chunks": [
                next(
                    cell["observed"]["accepted_chunk_count"]
                    for cell in cells
                    if cell["rows"] == row and cell["chunk_bytes"] == budgets[0]
                )
                for row in rows
            ],
        },
    }
    _seal(value)
    return value


class ScaleCsvExponentialEvidenceTests(unittest.TestCase):
    @staticmethod
    def _environment() -> dict[str, str]:
        environment = dict(os.environ)
        source = str(Path(__file__).parents[2] / "src")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        return environment

    def _check(
        self,
        artifact: dict[str, object],
        *,
        expected_sha: str | None = None,
        smoke: bool = True,
        bind_repository: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            command = [
                sys.executable,
                str(CHECKER),
                "--artifact",
                str(path),
                "--expected-git-sha",
                expected_sha or _head(),
            ]
            if bind_repository:
                command.extend(["--repo-root", str(REPO)])
            if smoke:
                command.append("--smoke")
            return subprocess.run(
                command, check=False, capture_output=True, text=True, env=self._environment()
            )

    def test_scale01_checker_accepts_explicitly_locked_smoke_matrix(self) -> None:
        result = self._check(_smoke_artifact())
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_scale02_rejects_wrong_expected_sha_and_nonancestor(self) -> None:
        artifact = _smoke_artifact()
        result = self._check(artifact, expected_sha="0" * 40)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frozen expected SHA", result.stderr)
        self.assertIn("existing ancestor", result.stderr)

    def test_scale03_rejects_matrix_cell_count_duplicate_and_wrong_operation_crosslink(
        self,
    ) -> None:
        artifact = _smoke_artifact()
        cells = artifact["cells"]
        assert isinstance(cells, list)
        cells.append(dict(cells[0]))
        artifact["operation_evidence"]["accepted_chunks"] = [999]
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly three", result.stderr)
        self.assertIn("duplicate", result.stderr)
        self.assertIn("cross-link", result.stderr)

    def test_scale04_rejects_zero_source_sha_and_fake_counts(self) -> None:
        artifact = _smoke_artifact()
        cells = artifact["cells"]
        assert isinstance(cells, list)
        cells[0]["source"]["sha256"] = "0" * 64
        cells[0]["fit"]["event_count"] = 1
        cells[0]["fit"]["expected_event_count"] = 1
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source bytes or SHA", result.stderr)
        self.assertIn("observation/event", result.stderr)

    def test_scale05_rejects_fake_total_and_rate_errors(self) -> None:
        artifact = _smoke_artifact()
        cell = artifact["cells"][0]
        cell["fit"]["total_time"] = 999999
        cell["fit"]["expected_total_time"] = "999999"
        cell["fit"]["absolute_rate_error"] = 999
        cell["fit"]["relative_rate_error"] = 999
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("total time", result.stderr)
        self.assertIn("rate errors", result.stderr)

    def test_scale06_rejects_extra_schema_key_and_path_key(self) -> None:
        artifact = _smoke_artifact()
        artifact["run"]["platform"] = "C:/secret"
        artifact["cells"][0]["fit"]["extra"] = 1
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("schema keys", result.stderr)
        self.assertIn("path", result.stderr)

    def test_scale07_rejects_bad_worker_and_dirty_run(self) -> None:
        artifact = _smoke_artifact()
        artifact["run"]["measurement_workers"] = 0
        artifact["run"]["git_dirty"] = True
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("workers", result.stderr)
        self.assertIn("dirty", result.stderr)

    def test_scale07a_rejects_timing_evidence_from_parallel_workers(self) -> None:
        for smoke, build in ((True, _smoke_artifact), (False, _full_artifact)):
            for workers in (2, 3):
                with self.subTest(smoke=smoke, workers=workers):
                    artifact = build()
                    artifact["run"]["measurement_workers"] = workers
                    _seal(artifact)
                    result = self._check(artifact, smoke=smoke)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("exactly one measurement worker", result.stderr)
        accepted = _full_artifact()
        self.assertEqual(accepted["run"]["measurement_workers"], 1)
        self.assertEqual(self._check(accepted, smoke=False).returncode, 0)

    def test_scale07b_rejects_missing_or_unknown_measurement_methodology(self) -> None:
        missing = _smoke_artifact()
        del missing["run"]["methodology"]
        _seal(missing)
        result = self._check(missing)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("schema keys", result.stderr)
        for methodology in (
            {"passes": "traced-single-pass", "rss": "process-peak-v1"},
            {"passes": "elapsed-untraced-then-memory-traced-v1", "rss": "current"},
        ):
            with self.subTest(methodology=methodology):
                artifact = _smoke_artifact()
                artifact["run"]["methodology"] = methodology
                _seal(artifact)
                result = self._check(artifact)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("methodology is not supported", result.stderr)

    def test_scale08_runner_produces_checker_accepted_smoke_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            command = [
                sys.executable,
                str(RUNNER),
                "--output",
                str(output),
                "--rows",
                "100",
                "--chunk-bytes",
                "2048,4096,8192",
            ]
            generated = subprocess.run(
                command, check=False, capture_output=True, text=True, env=self._environment()
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["run"]["measurement_workers"], 1)
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--artifact",
                    str(output),
                    "--expected-git-sha",
                    _head(),
                    "--repo-root",
                    str(REPO),
                    "--smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=self._environment(),
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_scale08b_checker_rejects_runner_output_from_parallel_workers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            generated = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    "--output",
                    str(output),
                    "--rows",
                    "100",
                    "--chunk-bytes",
                    "2048,4096,8192",
                    "--workers",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=self._environment(),
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["run"]["measurement_workers"], 2)
            result = self._check(artifact)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("exactly one measurement worker", result.stderr)

    def test_scale08c_elapsed_pass_is_untraced_and_memory_pass_is_separate(self) -> None:
        import tracemalloc

        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "fixture.csv"
            events, total = RUNNER_MODULE._generate(fixture, 50)
            tracing: list[bool] = []
            real_fit = RUNNER_MODULE._fit

            def spy(*arguments: object) -> object:
                tracing.append(tracemalloc.is_tracing())
                return real_fit(*arguments)

            with patch.object(RUNNER_MODULE, "_fit", side_effect=spy):
                cell = RUNNER_MODULE._cell(
                    fixture,
                    rows=50,
                    chunk_bytes=2048,
                    expected_events=events,
                    expected_total=total,
                )
        self.assertEqual(tracing, [False, True])
        self.assertFalse(tracemalloc.is_tracing())
        self.assertGreater(cell["memory"]["tracemalloc_peak_bytes"], 0)
        self.assertGreater(cell["memory"]["rss_peak_bytes"], 0)
        self.assertGreater(cell["elapsed_seconds"], 0)

    def test_scale08d_runner_refuses_a_memory_pass_that_differs_from_the_timed_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "fixture.csv"
            events, total = RUNNER_MODULE._generate(fixture, 50)
            real_fit = RUNNER_MODULE._fit
            shorter_fixture = Path(temporary) / "shorter.csv"
            RUNNER_MODULE._generate(shorter_fixture, 40)
            shorter = real_fit(shorter_fixture, 40, 2048)
            calls: list[object] = []

            def drifting(*arguments: object) -> object:
                calls.append(arguments)
                return real_fit(*arguments) if len(calls) == 1 else shorter

            with patch.object(RUNNER_MODULE, "_fit", side_effect=drifting):
                with self.assertRaisesRegex(RuntimeError, "memory pass did not reproduce"):
                    RUNNER_MODULE._cell(
                        fixture,
                        rows=50,
                        chunk_bytes=2048,
                        expected_events=events,
                        expected_total=total,
                    )

    def test_scale08a_runner_default_uses_a_single_measurement_worker(self) -> None:
        class InlineExecutor:
            def __init__(self, *, max_workers: int) -> None:
                self.max_workers = max_workers

            def __enter__(self) -> InlineExecutor:
                return self

            def __exit__(self, *unused: object) -> None:
                return None

            def map(self, function: object, values: object) -> object:
                assert callable(function)
                return [function(value) for value in values]  # type: ignore[union-attr]

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            command = [
                str(RUNNER),
                "--output",
                str(output),
                "--rows",
                "10",
                "--chunk-bytes",
                "2048,4096,8192",
            ]
            cell = {"observed": {"accepted_chunk_count": 1}}
            with (
                patch.object(RUNNER_MODULE, "_clean_checkout_sha", return_value="a" * 40),
                patch.object(RUNNER_MODULE, "_cell_request", return_value=cell),
                patch.object(RUNNER_MODULE, "ProcessPoolExecutor", InlineExecutor),
                patch.object(sys, "argv", command),
            ):
                self.assertEqual(RUNNER_MODULE.main(), 0)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["run"]["measurement_workers"], 1)
            self.assertEqual(artifact["schema_version"], "3")
            self.assertEqual(
                artifact["run"]["timing"],
                {"clock": "time.perf_counter_ns", "preflight": "paired-monotonic-wall-v2"},
            )
            self.assertEqual(artifact["run"]["candidate_git_sha"], "a" * 40)
            self.assertEqual(
                artifact["run"]["methodology"],
                {"passes": "elapsed-untraced-then-memory-traced-v1", "rss": "process-peak-v1"},
            )
            self.assertEqual(artifact["run"]["methodology"], CHECKER_MODULE.METHODOLOGY)

    def test_scale09_runner_smoke_is_concurrent_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary) / f"artifact-{index}.json" for index in range(2)]
            processes = [
                subprocess.Popen(
                    [
                        sys.executable,
                        str(RUNNER),
                        "--output",
                        str(output),
                        "--rows",
                        "100",
                        "--chunk-bytes",
                        "2048,4096,8192",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=self._environment(),
                )
                for output in outputs
            ]
            results = [process.communicate() for process in processes]
            self.assertTrue(all(process.returncode == 0 for process in processes), results)
            self.assertTrue(all(output.exists() for output in outputs))

    def test_scale10_full_contract_rejects_twelve_cells(self) -> None:
        artifact = _full_artifact()
        artifact["cells"].extend(artifact["cells"][:3])
        _seal(artifact)
        result = self._check(artifact, smoke=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exactly nine", result.stderr)
        self.assertIn("duplicate", result.stderr)

    def test_scale11_runner_refuses_output_when_head_changes_mid_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            facts = iter([_head(), "f" * 40])
            command = [
                str(RUNNER),
                "--output",
                str(output),
                "--rows",
                "10",
                "--chunk-bytes",
                "2048,4096,8192",
                "--workers",
                "1",
            ]
            with (
                patch.object(
                    RUNNER_MODULE, "_clean_checkout_sha", side_effect=lambda _root: next(facts)
                ),
                patch.object(sys, "argv", command),
            ):
                with self.assertRaises(SystemExit) as failure:
                    RUNNER_MODULE.main()
            self.assertIn("HEAD changed", str(failure.exception))
            self.assertFalse(output.exists())

    def test_scale12_previous_schema_artifact_is_rejected_by_current_checker(self) -> None:
        artifact = _smoke_artifact()
        artifact["schema_version"] = "1"
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("current evidence requires schema version 3", result.stderr)

    def test_scale13_rejects_large_rate_with_truthful_large_error_facts(self) -> None:
        artifact = _smoke_artifact()
        fit = artifact["cells"][0]["fit"]
        expected_rate = Decimal(str(fit["expected_rate"]))
        actual_rate = Decimal("999")
        absolute = abs(actual_rate - expected_rate)
        fit["rate"] = 999
        fit["absolute_rate_error"] = float(absolute)
        fit["relative_rate_error"] = float(absolute / abs(expected_rate))
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rate error exceeds frozen tolerance", result.stderr)

    def test_scale14_cli_rejects_unbound_zero_sha_smoke_artifact(self) -> None:
        artifact = _smoke_artifact()
        artifact["run"]["git_sha"] = "0" * 40
        _seal(artifact)
        result = self._check(
            artifact,
            expected_sha="0" * 40,
            bind_repository=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--repo-root", result.stderr)

    def test_scale15_rejects_zero_accepted_chunks_in_a_64kib_full_cell(self) -> None:
        artifact = _full_artifact()
        target = next(
            cell
            for cell in artifact["cells"]
            if cell["rows"] == 10_000 and cell["chunk_bytes"] == 65_536
        )
        target["observed"]["accepted_chunk_count"] = 0
        _seal(artifact)
        result = self._check(artifact, smoke=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("accepted_chunk_count must be positive", result.stderr)

    def test_scale16_elapsed_clock_pair_rejects_cross_domain_disagreement(self) -> None:
        self.assertEqual(
            RUNNER_MODULE._paired_elapsed_seconds(1_000, 1_100, 8_000, 8_100),
            0.0000001,
        )
        with self.assertRaisesRegex(RuntimeError, "clocks disagree"):
            RUNNER_MODULE._paired_elapsed_seconds(
                1_000,
                1_000_000_000,
                8_000,
                72_008_000_000_000,
            )

    def test_scale16a_elapsed_uses_monotonic_clock_when_wall_clock_is_coarse(self) -> None:
        # A short cell can finish within one wall-clock tick (the wall clock
        # advances in coarse steps on some platforms), so the persisted
        # duration comes from the monotonic clock.
        self.assertEqual(
            RUNNER_MODULE._paired_elapsed_seconds(5_000, 5_000, 8_000, 8_250),
            0.00000025,
        )
        with self.assertRaisesRegex(RuntimeError, "below the monotonic clock resolution"):
            RUNNER_MODULE._paired_elapsed_seconds(5_000, 5_000, 8_000, 8_000)

    def test_scale17_v2_rejects_missing_timing_provenance(self) -> None:
        artifact = _smoke_artifact()
        del artifact["run"]["timing"]
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run schema keys invalid", result.stderr)

    def test_scale17a_rejects_missing_process_memory_and_throughput(self) -> None:
        artifact = _smoke_artifact()
        cell = artifact["cells"][0]
        cell["memory"]["rss_peak_bytes"] = None
        del cell["throughput_rows_per_second"]
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("schema keys invalid", result.stderr)

    def test_scale18_v2_rejects_transplanted_candidate_commit(self) -> None:
        artifact = _smoke_artifact()
        artifact["run"]["candidate_git_sha"] = "0" * 40
        _seal(artifact)
        result = self._check(artifact)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("candidate git SHA does not match frozen expected SHA", result.stderr)


if __name__ == "__main__":
    unittest.main()
