"""Every run step that can execute on a Windows runner uses bash explicitly.

The default shell on a Windows runner is PowerShell, where ``$NAME`` is not an
environment variable reference. Workflow scripts receive caller-controlled
values through ``env:`` and read them as ``$NAME``, so a Windows step without
an explicit ``bash`` shell silently sees empty values.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"

# The legacy workflow is validated byte-for-byte by tools/check_legacy_release_safety.py.
EXEMPT_WORKFLOWS = frozenset({"ci.yml"})


def _strings(value: Any) -> list[str]:
    """Return every scalar in a nested structure as a lowercase string."""
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return [str(value).lower()]


def _can_run_on_windows(job: dict[str, Any]) -> bool:
    runs_on = job.get("runs-on", "")
    if any("windows" in text for text in _strings(runs_on)):
        return True
    matrix = (job.get("strategy") or {}).get("matrix") or {}
    # An expression runs-on (for example ${{ matrix.os }}) is resolved from the
    # matrix, so any matrix value naming Windows makes the job possibly Windows.
    if "${{" in str(runs_on):
        return any("windows" in text for text in _strings(matrix))
    return False


def _default_shell(container: dict[str, Any]) -> str | None:
    return ((container.get("defaults") or {}).get("run") or {}).get("shell")


def _violations(document: dict[str, Any]) -> list[str]:
    """Name every run step on a possibly-Windows job whose shell is not bash."""
    problems = []
    workflow_shell = _default_shell(document)
    for job_name, job in document["jobs"].items():
        if not _can_run_on_windows(job):
            continue
        job_shell = _default_shell(job)
        for index, step in enumerate(job.get("steps") or []):
            if "run" not in step:
                continue
            shell = step.get("shell") or job_shell or workflow_shell
            if shell != "bash":
                label = step.get("name", f"step {index}")
                problems.append(f"{job_name}: {label} (shell: {shell or 'default'})")
    return problems


def _load(text: str) -> dict[str, Any]:
    document = yaml.safe_load(text)
    assert isinstance(document, dict)
    return document


_WINDOWS_WITHOUT_BASH = """
jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
    steps:
      - name: Read an environment variable
        env:
          VALUE: x
        run: python tool.py --value "$VALUE"
"""

_WINDOWS_WITH_STEP_BASH = _WINDOWS_WITHOUT_BASH.replace(
    "        run: python", "        shell: bash\n        run: python"
)

_WINDOWS_WITH_JOB_BASH = _WINDOWS_WITHOUT_BASH.replace(
    "    steps:", "    defaults:\n      run:\n        shell: bash\n    steps:"
)

_WINDOWS_WITH_WORKFLOW_BASH = "defaults:\n  run:\n    shell: bash\n" + _WINDOWS_WITHOUT_BASH

_WINDOWS_WITH_POWERSHELL = _WINDOWS_WITHOUT_BASH.replace(
    "        run: python", "        shell: pwsh\n        run: python"
)

_INCLUDE_ONLY_WINDOWS = """
jobs:
  build:
    runs-on: ${{ matrix.runner }}
    strategy:
      matrix:
        include:
          - runner: ubuntu-latest
          - runner: windows-2022
    steps:
      - run: echo "$HOME"
"""

_LITERAL_WINDOWS = """
jobs:
  build:
    runs-on: windows-latest
    steps:
      - run: echo "$HOME"
"""

_LINUX_ONLY = """
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - run: echo "$HOME"
  other:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
    steps:
      - run: echo "$HOME"
"""


class WorkflowShellTests(unittest.TestCase):
    def test_every_windows_capable_run_step_uses_bash(self) -> None:
        checked = 0
        for path in sorted(WORKFLOWS.glob("*.yml")):
            if path.name in EXEMPT_WORKFLOWS:
                continue
            document = _load(path.read_text(encoding="utf-8"))
            with self.subTest(workflow=path.name):
                self.assertEqual(_violations(document), [])
            checked += sum(_can_run_on_windows(job) for job in document["jobs"].values())
        # The guard is only meaningful while it still sees Windows-capable jobs.
        self.assertGreaterEqual(checked, 3)

    def test_the_release_evidence_collector_step_reads_its_sha_through_bash(self) -> None:
        document = _load((WORKFLOWS / "v1-release-evidence.yml").read_text(encoding="utf-8"))
        job = document["jobs"]["collect-execution-evidence"]
        step = next(
            step
            for step in job["steps"]
            if step.get("name") == "Collect complete retry-resume and cancellation facts"
        )
        self.assertEqual(step.get("shell") or _default_shell(job), "bash")
        self.assertIn('"$CANDIDATE_SHA"', step["run"])
        self.assertNotIn("${{ env.CANDIDATE_SHA }}", step["run"])

    def test_windows_step_without_bash_is_flagged(self) -> None:
        problems = _violations(_load(_WINDOWS_WITHOUT_BASH))
        self.assertEqual(problems, ["build: Read an environment variable (shell: default)"])

    def test_windows_step_with_another_shell_is_flagged(self) -> None:
        problems = _violations(_load(_WINDOWS_WITH_POWERSHELL))
        self.assertEqual(problems, ["build: Read an environment variable (shell: pwsh)"])

    def test_windows_runner_named_only_in_an_include_entry_is_flagged(self) -> None:
        self.assertEqual(len(_violations(_load(_INCLUDE_ONLY_WINDOWS))), 1)

    def test_literal_windows_runner_is_flagged(self) -> None:
        self.assertEqual(len(_violations(_load(_LITERAL_WINDOWS))), 1)

    def test_bash_from_step_job_or_workflow_defaults_is_accepted(self) -> None:
        for fixture in (
            _WINDOWS_WITH_STEP_BASH,
            _WINDOWS_WITH_JOB_BASH,
            _WINDOWS_WITH_WORKFLOW_BASH,
        ):
            with self.subTest(fixture=fixture):
                self.assertEqual(_violations(_load(fixture)), [])

    def test_step_shell_overrides_a_bash_default(self) -> None:
        override = _WINDOWS_WITH_JOB_BASH.replace(
            "        run: python", "        shell: pwsh\n        run: python"
        )
        self.assertEqual(len(_violations(_load(override))), 1)

    def test_jobs_that_never_run_on_windows_are_not_checked(self) -> None:
        self.assertEqual(_violations(_load(_LINUX_ONLY)), [])

    def test_uses_steps_are_not_run_steps(self) -> None:
        document = _load(
            _WINDOWS_WITHOUT_BASH.replace(
                'run: python tool.py --value "$VALUE"', "uses: actions/checkout@v4"
            )
        )
        self.assertEqual(_violations(document), [])


if __name__ == "__main__":
    unittest.main()
