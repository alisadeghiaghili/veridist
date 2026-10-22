"""The scale evidence retained in the repository must satisfy its current checkers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import unittest
from pathlib import Path
from types import ModuleType

PYTHON_ROOT = Path(__file__).parents[2]
REPO = Path(__file__).parents[3]
EVIDENCE = PYTHON_ROOT / "evidence"
TOOLS = PYTHON_ROOT / "tools"
CANDIDATE = "e316d125e0ae434beaf0c69ebeee3238c1076f6b"
RELEASE = "2.2.0"
PLATFORMS = ("linux", "windows")
KINDS = ("engine", "csv-exponential", "log-likelihood")


def _load(name: str, filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ENGINE_CHECKER = _load("retained_engine_checker", "check_scale_engine_evidence.py")
CSV_CHECKER = _load("retained_csv_checker", "check_scale_csv_exponential_evidence.py")
LIKELIHOOD_CHECKER = _load("retained_likelihood_checker", "check_log_likelihood_scale_evidence.py")


def _candidate_is_available() -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "-e", f"{CANDIDATE}^{{commit}}"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


class RetainedScaleEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _candidate_is_available():
            self.skipTest("the evidence candidate commit is not in this checkout's history")

    @staticmethod
    def _load_artifact(name: str) -> object:
        return json.loads((EVIDENCE / name).read_text(encoding="utf-8"))

    def test_directory_holds_exactly_the_current_evidence_files(self) -> None:
        expected = sorted(
            f"scale-{kind}-{RELEASE}-{platform}.json" for kind in KINDS for platform in PLATFORMS
        )
        self.assertEqual(sorted(path.name for path in EVIDENCE.iterdir()), expected)

    def test_engine_evidence_passes_its_checker(self) -> None:
        # The Linux ladder includes the 1e9-row rung; the Windows ladder does not.
        for platform in PLATFORMS:
            with self.subTest(platform=platform):
                artifact = self._load_artifact(f"scale-engine-{RELEASE}-{platform}.json")
                errors = ENGINE_CHECKER.validate(
                    artifact,
                    expected_git_sha=CANDIDATE,
                    top_rung=platform == "linux",
                    repo_root=REPO,
                )
                self.assertEqual(errors, [])

    def test_csv_exponential_evidence_passes_its_checker(self) -> None:
        for platform in PLATFORMS:
            with self.subTest(platform=platform):
                artifact = self._load_artifact(f"scale-csv-exponential-{RELEASE}-{platform}.json")
                errors = CSV_CHECKER.validate(artifact, expected_git_sha=CANDIDATE, repo_root=REPO)
                self.assertEqual(errors, [])

    def test_log_likelihood_evidence_passes_its_checker(self) -> None:
        for platform in PLATFORMS:
            with self.subTest(platform=platform):
                artifact = self._load_artifact(f"scale-log-likelihood-{RELEASE}-{platform}.json")
                errors = LIKELIHOOD_CHECKER.validate(
                    artifact, expected_git_sha=CANDIDATE, repo_root=REPO
                )
                self.assertEqual(errors, [])

    def test_evidence_runs_were_measured_on_the_platform_in_the_file_name(self) -> None:
        markers = {"linux": "Linux", "windows": "Windows"}
        for kind in KINDS:
            for platform in PLATFORMS:
                with self.subTest(kind=kind, platform=platform):
                    artifact = self._load_artifact(f"scale-{kind}-{RELEASE}-{platform}.json")
                    assert isinstance(artifact, dict)
                    self.assertTrue(artifact["run"]["platform"].startswith(markers[platform]))


if __name__ == "__main__":
    unittest.main()
