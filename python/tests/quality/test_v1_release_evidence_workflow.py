"""Contracts for the cross-platform v1 release-evidence workflow."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from tools import check_v1_execution_evidence as checker

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[3] / ".github" / "workflows" / "v1-release-evidence.yml"
)
EXPECTED_CELLS = len(checker._PLATFORMS) * len(checker._ROWS) * len(checker._SCENARIOS)


def _workflow() -> dict[str, Any]:
    document = yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def _step(job: dict[str, Any], name: str) -> dict[str, Any]:
    return next(step for step in job["steps"] if step.get("name") == name)


class V1ReleaseEvidenceWorkflowTests(unittest.TestCase):
    def test_collection_runs_every_scenario_on_the_three_platforms(self) -> None:
        job = _workflow()["jobs"]["collect-execution-evidence"]
        entries = job["strategy"]["matrix"]["include"]
        self.assertEqual(
            {(entry["platform"], entry["runner"]) for entry in entries},
            {
                ("linux", "ubuntu-latest"),
                ("macos", "macos-latest"),
                ("windows", "windows-latest"),
            },
        )
        self.assertIs(job["strategy"]["fail-fast"], False)
        self.assertIsInstance(job["timeout-minutes"], int)
        self.assertLessEqual(job["timeout-minutes"], 120)
        collect = _step(job, "Collect complete retry-resume and cancellation facts")
        self.assertIn("tools/collect_v1_execution_evidence.py", collect["run"])
        self.assertIn('--platform "${{ matrix.platform }}"', collect["run"])

    def test_the_scenario_set_is_the_documented_six(self) -> None:
        self.assertEqual(
            checker._SCENARIOS,
            frozenset(
                {
                    "complete",
                    "retry_resume",
                    "cancel",
                    "source_mutated",
                    "chunk_replay",
                    "process_killed",
                }
            ),
        )
        self.assertEqual(EXPECTED_CELLS, 54)

    def test_assembly_consumes_one_fragment_per_platform_and_rechecks_the_matrix(self) -> None:
        job = _workflow()["jobs"]["validate-collected-execution"]
        assemble = _step(job, "Assemble and validate the executed cross-platform matrix")["run"]
        for platform in ("linux", "macos", "windows"):
            self.assertIn(f"--input ../raw-execution/v1-execution-{platform}.json", assemble)
        self.assertIn("tools/assemble_v1_execution_evidence.py", assemble)
        self.assertIn("tools/check_v1_execution_evidence.py", assemble)
        self.assertNotIn("continue-on-error", WORKFLOW_PATH.read_text(encoding="utf-8"))

    def _size_command(self) -> list[str]:
        job = _workflow()["jobs"]["validate-collected-execution"]
        run = _step(job, "Require every scenario on every platform and row count")["run"]
        words = shlex.split(run.strip())
        self.assertEqual(words[:2], ["python", "-c"])
        self.assertEqual(len(words), 3)
        self.assertIn(f"len(cells) == {EXPECTED_CELLS}", words[2])
        self.assertIn(f"== {len(checker._SCENARIOS)}", words[2])
        return [sys.executable, *words[1:]]

    def _run_size_command(self, cells: list[dict[str, str]]) -> int:
        command = self._size_command()
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "v1-execution-collected.json").write_text(
                json.dumps({"cells": cells}), encoding="utf-8"
            )
            return subprocess.run(
                command, cwd=directory, capture_output=True, timeout=60, check=False
            ).returncode

    def test_the_matrix_size_step_accepts_only_the_full_matrix(self) -> None:
        scenarios = sorted(checker._SCENARIOS)
        full = [{"scenario": scenarios[index % len(scenarios)]} for index in range(EXPECTED_CELLS)]
        self.assertEqual(self._run_size_command(full), 0)
        self.assertNotEqual(self._run_size_command(full[:-1]), 0)
        fewer_scenarios = [{"scenario": scenarios[index % 5]} for index in range(EXPECTED_CELLS)]
        self.assertNotEqual(self._run_size_command(fewer_scenarios), 0)

    def test_the_candidate_is_never_expanded_from_event_payloads_in_scripts(self) -> None:
        workflow = _workflow()
        for name, job in workflow["jobs"].items():
            for step in job.get("steps", []):
                with self.subTest(job=name, step=step.get("name")):
                    self.assertNotIn("github.event.", step.get("run", ""))


if __name__ == "__main__":
    unittest.main()
