"""Adversarial unit contracts for the fail-closed v1 release-evidence gates."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import collect_v1_execution_evidence as collector
from tools.assemble_v1_execution_evidence import assemble
from tools.check_v1_execution_evidence import validate as validate_execution
from tools.check_v1_execution_evidence import validate_raw_fragment
from tools.collect_v1_execution_evidence import _run_scenario

SHA = "a" * 40
PLATFORMS = ("linux", "macos", "windows")
ROW_COUNTS = (10_000, 100_000, 1_000_000)
SCENARIO_NAMES = (
    "complete",
    "retry_resume",
    "cancel",
    "source_mutated",
    "chunk_replay",
    "process_killed",
)
PARTIAL = {"retry_resume", "cancel", "source_mutated", "process_killed"}
COMPLETED = {"complete", "retry_resume", "chunk_replay", "process_killed"}


def _cell(platform: str, rows: int, scenario: str) -> dict[str, object]:
    cell: dict[str, object] = {
        "platform": platform,
        "rows": rows,
        "scenario": scenario,
        "candidate_git_sha": SHA,
        "attempt_count": 1 if scenario in {"complete", "cancel"} else 2,
        "process_peak_rss_bytes": 1,
        "interrupted_cursor": 5 if scenario in PARTIAL else 0,
        "final_cursor": rows if scenario in COMPLETED else 5,
        "source_bytes": 1,
        "source_sha256": "b" * 64,
        "result_sha256": "c" * 64 if scenario in COMPLETED else None,
        "result_code": "COMPLETE" if scenario in COMPLETED else "CANCELLED",
        "retry_initial_code": None,
        "canonical_result_equal": True,
        "cancellation_observed": scenario in {"retry_resume", "cancel", "source_mutated"},
        "resources_released": scenario == "cancel",
    }
    if scenario == "retry_resume":
        cell["retry_initial_code"] = "CANCELLED"
    if scenario == "source_mutated":
        cell.update(
            result_code="SOURCE_REVISION_MISMATCH",
            retry_initial_code="CANCELLED",
            rebound_revision_code="SOURCE_REVISION_MISMATCH",
            mutated_source_sha256="d" * 64,
            checkpoint_unchanged=True,
            generation_before=3,
            generation_after=3,
        )
    if scenario == "chunk_replay":
        cell.update(
            checkpoint_unchanged=True,
            generation_before=10,
            generation_after=10,
            replayed_chunk_count=10,
        )
    if scenario == "process_killed":
        cell.update(
            retry_initial_code="PROCESS_TERMINATED", child_exit_code=1, kill_attempts=1
        )
    return cell


def _execution_payload() -> dict[str, object]:
    cells = [
        _cell(platform, rows, scenario)
        for platform in PLATFORMS
        for rows in ROW_COUNTS
        for scenario in SCENARIO_NAMES
    ]
    return {"schema_version": 2, "candidate_git_sha": SHA, "cells": cells}


def _raw_payload(platform: str) -> dict[str, object]:
    return {
        "schema_version": 2,
        "artifact_kind": "v1-execution-raw",
        "candidate_git_sha": SHA,
        "platform": platform,
        "host_platform": "fixture",
        "python_version": "3.11.0",
        "numpy_version": "2.0.0",
        "veridist_version": "1.0.0",
        "collected_at": "2026-09-10T00:00:00Z",
        "cells": [
            _cell(platform, rows, scenario) for rows in ROW_COUNTS for scenario in SCENARIO_NAMES
        ],
    }


def _raw_cell(payload: dict[str, object], scenario: str, rows: int = 10_000) -> dict[str, object]:
    cells = payload["cells"]
    assert isinstance(cells, list)
    return next(cell for cell in cells if cell["scenario"] == scenario and cell["rows"] == rows)


class V1EvidenceCheckerTests(unittest.TestCase):
    def test_matrix_has_six_scenarios_per_row_count_and_platform(self) -> None:
        self.assertEqual(collector.SCENARIOS, SCENARIO_NAMES)
        payload = _execution_payload()
        self.assertEqual(len(payload["cells"]), 3 * 3 * 6)  # type: ignore[arg-type]
        self.assertEqual(len(_raw_payload("linux")["cells"]), 3 * 6)  # type: ignore[arg-type]

    def test_execution_evidence_requires_the_complete_candidate_bound_matrix(self) -> None:
        payload = _execution_payload()
        self.assertEqual(validate_execution(payload, SHA), [])
        cells = payload["cells"]
        assert isinstance(cells, list)
        cells.pop()
        self.assertIn("matrix", " ".join(validate_execution(payload, SHA)))

    def test_execution_evidence_rejects_a_matrix_without_the_new_scenarios(self) -> None:
        payload = _execution_payload()
        cells = payload["cells"]
        assert isinstance(cells, list)
        payload["cells"] = [
            cell
            for cell in cells
            if cell["scenario"] in {"complete", "retry_resume", "cancel"}
        ]
        self.assertIn("matrix", " ".join(validate_execution(payload, SHA)))

    def test_execution_evidence_rejects_the_previous_schema_version(self) -> None:
        payload = _execution_payload()
        payload["schema_version"] = 1
        self.assertIn("schema", " ".join(validate_execution(payload, SHA)))
        raw = _raw_payload("linux")
        raw["schema_version"] = 1
        self.assertIn("schema", " ".join(validate_raw_fragment(raw, SHA)))

    def test_execution_evidence_requires_the_facts_of_each_new_scenario(self) -> None:
        for scenario, field, value, message in (
            ("source_mutated", "checkpoint_unchanged", False, "unchanged-checkpoint"),
            ("source_mutated", "result_code", "COMPLETE", "refusal"),
            ("chunk_replay", "checkpoint_unchanged", None, "unchanged-checkpoint"),
            ("chunk_replay", "canonical_result_equal", False, "baseline equality"),
            ("process_killed", "canonical_result_equal", False, "baseline equality"),
        ):
            with self.subTest(scenario=scenario, field=field):
                payload = _execution_payload()
                _raw_cell(payload, scenario)[field] = value
                self.assertIn(message, " ".join(validate_execution(payload, SHA)))

    def test_raw_platform_fragments_require_executed_retry_and_cleanup_facts(self) -> None:
        payload = _raw_payload("linux")
        self.assertEqual(validate_raw_fragment(payload, SHA), [])
        cancellation = _raw_cell(payload, "cancel")
        cancellation["resources_released"] = False
        self.assertIn("cleanup", " ".join(validate_raw_fragment(payload, SHA)))

    def test_raw_fragment_rejects_each_tampered_new_scenario_fact(self) -> None:
        tampering: tuple[tuple[str, str, object, str], ...] = (
            ("source_mutated", "result_code", "COMPLETE", "source-mutation refusal"),
            ("source_mutated", "rebound_revision_code", "COMPLETE", "source-mutation refusal"),
            ("source_mutated", "mutated_source_sha256", "b" * 64, "mutated-source digest"),
            ("source_mutated", "mutated_source_sha256", None, "mutated-source digest"),
            ("source_mutated", "checkpoint_unchanged", False, "unchanged-checkpoint"),
            ("source_mutated", "generation_after", 4, "unchanged-checkpoint"),
            ("source_mutated", "result_sha256", "c" * 64, "reports a result"),
            ("source_mutated", "final_cursor", 6, "changed cursor"),
            ("source_mutated", "interrupted_cursor", 0, "partial checkpoint"),
            ("source_mutated", "attempt_count", 1, "attempt count"),
            ("chunk_replay", "checkpoint_unchanged", False, "unchanged-checkpoint"),
            ("chunk_replay", "generation_after", 11, "unchanged-checkpoint"),
            ("chunk_replay", "generation_before", True, "unchanged-checkpoint"),
            ("chunk_replay", "replayed_chunk_count", 0, "chunk-replay"),
            ("chunk_replay", "canonical_result_equal", False, "chunk-replay"),
            ("chunk_replay", "result_code", "CANCELLED", "chunk-replay"),
            ("chunk_replay", "result_sha256", None, "result digest"),
            ("chunk_replay", "final_cursor", 1, "final cursor"),
            ("process_killed", "child_exit_code", 0, "process-termination"),
            ("process_killed", "child_exit_code", True, "process-termination"),
            ("process_killed", "kill_attempts", 0, "process-termination"),
            ("process_killed", "kill_attempts", 4, "process-termination"),
            ("process_killed", "retry_initial_code", "CANCELLED", "process-termination"),
            ("process_killed", "canonical_result_equal", False, "process-termination"),
            ("process_killed", "result_code", "CANCELLED", "process-termination"),
            ("process_killed", "interrupted_cursor", 10_000, "partial checkpoint"),
            ("process_killed", "final_cursor", 5, "final cursor"),
            ("process_killed", "result_sha256", "short", "result digest"),
            ("process_killed", "attempt_count", 1, "attempt count"),
        )
        for scenario, field, value, message in tampering:
            with self.subTest(scenario=scenario, field=field, value=value):
                payload = _raw_payload("linux")
                _raw_cell(payload, scenario)[field] = value
                self.assertIn(message, " ".join(validate_raw_fragment(payload, SHA)))

    def test_raw_fragment_rejects_a_missing_new_scenario_cell(self) -> None:
        payload = _raw_payload("linux")
        cells = payload["cells"]
        assert isinstance(cells, list)
        payload["cells"] = [cell for cell in cells if cell["scenario"] != "process_killed"]
        self.assertIn("matrix", " ".join(validate_raw_fragment(payload, SHA)))

    def test_assembler_accepts_only_all_executed_platform_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths: list[Path] = []
            for platform in PLATFORMS:
                path = root / f"{platform}.json"
                path.write_text(json.dumps(_raw_payload(platform)), encoding="utf-8")
                paths.append(path)
            value, errors = assemble(paths, SHA)
            self.assertEqual(errors, [])
            assert value is not None
            self.assertEqual(value["schema_version"], 2)
            self.assertEqual(len(value["cells"]), 54)  # type: ignore[arg-type]
            value, errors = assemble(paths[:2], SHA)
            self.assertIsNone(value)
            self.assertIn("matrix", " ".join(errors))

    def test_assembler_rejects_cross_platform_result_drift(self) -> None:
        for scenario in ("complete", "chunk_replay", "process_killed"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                paths: list[Path] = []
                for platform in PLATFORMS:
                    payload = _raw_payload(platform)
                    if platform == "windows":
                        _raw_cell(payload, scenario)["result_sha256"] = "d" * 64
                    path = root / f"{platform}.json"
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    paths.append(path)
                value, errors = assemble(paths, SHA)
                self.assertIsNone(value)
                self.assertIn("result digests differ", " ".join(errors))

    def test_assembler_rejects_a_fragment_from_the_previous_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths: list[Path] = []
            for platform in PLATFORMS:
                payload = _raw_payload(platform)
                payload["schema_version"] = 1
                path = root / f"{platform}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths.append(path)
            value, errors = assemble(paths, SHA)
            self.assertIsNone(value)
            self.assertIn("unsupported raw execution evidence schema", " ".join(errors))

    def test_local_collector_observes_real_retry_and_cancel_paths(self) -> None:
        retry = _run_scenario(3, "retry_resume")
        cancellation = _run_scenario(3, "cancel")
        self.assertTrue(retry["canonical_result_equal"])
        self.assertEqual(retry["retry_initial_code"], "CANCELLED")
        self.assertGreater(retry["interrupted_cursor"], 0)
        self.assertEqual(retry["final_cursor"], 3)
        self.assertEqual(cancellation["result_code"], "CANCELLED")
        self.assertGreater(cancellation["interrupted_cursor"], 0)
        self.assertTrue(cancellation["resources_released"])

    def test_local_collector_refuses_a_resume_against_a_mutated_source(self) -> None:
        cell = _run_scenario(40, "source_mutated")
        self.assertEqual(cell["result_code"], "SOURCE_REVISION_MISMATCH")
        self.assertEqual(cell["rebound_revision_code"], "SOURCE_REVISION_MISMATCH")
        self.assertEqual(cell["retry_initial_code"], "CANCELLED")
        self.assertTrue(cell["checkpoint_unchanged"])
        self.assertEqual(cell["generation_before"], cell["generation_after"])
        self.assertEqual(cell["interrupted_cursor"], 20)
        self.assertEqual(cell["final_cursor"], 20)
        self.assertIsNone(cell["result_sha256"])
        self.assertNotEqual(cell["mutated_source_sha256"], cell["source_sha256"])
        self.assertEqual(cell["attempt_count"], 2)

    def test_local_collector_replays_committed_chunks_without_changing_state(self) -> None:
        cell = _run_scenario(95, "chunk_replay")
        self.assertTrue(cell["checkpoint_unchanged"])
        self.assertEqual(cell["generation_before"], cell["generation_after"])
        self.assertEqual(cell["generation_before"], 10)
        self.assertEqual(cell["replayed_chunk_count"], 10)
        self.assertEqual(cell["final_cursor"], 95)
        self.assertEqual(cell["result_code"], "COMPLETE")
        self.assertTrue(cell["canonical_result_equal"])
        complete = _run_scenario(95, "complete")
        self.assertEqual(cell["result_sha256"], complete["result_sha256"])

    def test_local_collector_resumes_a_terminated_child_to_the_baseline_result(self) -> None:
        cell = _run_scenario(50_000, "process_killed")
        self.assertEqual(cell["result_code"], "COMPLETE")
        self.assertEqual(cell["retry_initial_code"], "PROCESS_TERMINATED")
        self.assertGreater(cell["interrupted_cursor"], 0)
        self.assertLess(cell["interrupted_cursor"], 50_000)
        self.assertEqual(cell["final_cursor"], 50_000)
        self.assertNotEqual(cell["child_exit_code"], 0)
        self.assertIn(cell["kill_attempts"], {1, 2, 3})
        self.assertTrue(cell["canonical_result_equal"])
        complete = _run_scenario(50_000, "complete")
        self.assertEqual(cell["result_sha256"], complete["result_sha256"])

    def test_child_entry_point_completes_a_checkpointed_fit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "lifetimes.csv"
            revision, _ = collector._write_fixture(source, 100)
            store = collector._initial_store(
                root / "child.sqlite3", revision, collector._source_id(100)
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(collector.__file__).resolve()),
                    "--child-checkpointed-fit",
                    "--csv",
                    str(source),
                    "--store",
                    str(store.path),
                    "--rows",
                    "100",
                    "--revision",
                    revision,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(store.read().cursor, 100)
            stale = subprocess.run(
                [
                    sys.executable,
                    str(Path(collector.__file__).resolve()),
                    "--child-checkpointed-fit",
                    "--csv",
                    str(source),
                    "--store",
                    str(store.path),
                    "--rows",
                    "100",
                    "--revision",
                    "0" * 64,
                ],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
            self.assertEqual(stale.returncode, 3)


GATED_CHILD = """
import sys, time
sys.path.insert(0, {tools!r})
from collect_v1_execution_evidence import (
    _KILL_LIMITS, _SCHEMA, _source_id, fit_exponential_checkpointed_csv,
)
from veridist.engine.checkpoint import SQLiteCheckpointStore
rows, path, store, revision = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
fit_exponential_checkpointed_csv(
    path=path, schema=_SCHEMA, source_id=_source_id(rows), limits=_KILL_LIMITS,
    store=SQLiteCheckpointStore(store), source_revision=revision,
    # after the first batch has been committed, wait to be terminated
    cancel=lambda cursor: time.sleep(60) if SQLiteCheckpointStore(store).read().cursor else False,
)
"""


class TerminateChildTests(unittest.TestCase):
    def _gated_command(self, root: Path) -> tuple[list[str], Path]:
        source = root / "lifetimes.csv"
        revision, _ = collector._write_fixture(source, 500)
        store = collector._initial_store(
            root / "gated.sqlite3", revision, collector._source_id(500)
        )
        tools = str(Path(collector.__file__).resolve().parent)
        command = [
            sys.executable,
            "-c",
            GATED_CHILD.format(tools=tools),
            "500",
            str(source),
            str(store.path),
            revision,
        ]
        return command, store.path

    def test_terminates_a_child_after_its_first_commit_is_observed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command, store_path = self._gated_command(Path(directory))
            observed, exit_code = collector._terminate_child_after_first_commit(
                command, store_path, first_commit_timeout=60, process_timeout=30
            )
            self.assertGreater(observed, 0)
            self.assertNotEqual(exit_code, 0)
            cursor = collector.SQLiteCheckpointStore(store_path).read().cursor
            self.assertGreaterEqual(cursor, observed)
            self.assertLess(cursor, 500)

    def test_reports_a_child_that_exits_before_committing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "unused.sqlite3"
            with self.assertRaisesRegex(RuntimeError, "exited before its first commit"):
                collector._terminate_child_after_first_commit(
                    [sys.executable, "-c", "pass"],
                    store_path,
                    first_commit_timeout=60,
                    process_timeout=30,
                )

    def test_bounds_the_wait_for_a_child_that_never_commits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store_path = Path(directory) / "unused.sqlite3"
            with self.assertRaisesRegex(RuntimeError, "did not commit within"):
                collector._terminate_child_after_first_commit(
                    [sys.executable, "-c", "import time; time.sleep(60)"],
                    store_path,
                    first_commit_timeout=0.5,
                    process_timeout=30,
                )

    def test_a_child_that_finishes_before_the_kill_is_never_recorded_as_killed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "lifetimes.csv"
            revision, _ = collector._write_fixture(source, 50)
            with patch.object(
                collector, "_terminate_child_after_first_commit", return_value=(1, 0)
            ) as terminate:
                with self.assertRaisesRegex(RuntimeError, "completed before it could be"):
                    collector._kill_child_mid_run(root, source, revision, 50)
            self.assertEqual(terminate.call_count, collector._KILL_ATTEMPTS)


if __name__ == "__main__":
    unittest.main()
