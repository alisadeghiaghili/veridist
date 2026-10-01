"""TDD specifications for the deterministic coverage JSON gate."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

from tools.check_coverage import validate

CHECKER = Path(__file__).parents[2] / "tools" / "check_coverage.py"


def _summary(
    *,
    statements: int = 100,
    covered_lines: int = 100,
    branches: int = 20,
    covered_branches: int = 20,
) -> dict[str, int]:
    return {
        "num_statements": statements,
        "covered_lines": covered_lines,
        "num_branches": branches,
        "covered_branches": covered_branches,
    }


def _manifest(files: list[str]) -> dict[str, object]:
    return {
        "production_root": "src/veridist",
        "production_files": files,
        "critical_modules": ["domain", "statistics", "families", "engine"],
        "expected_denominators": {
            path: {"statements": 100, "branches": 20} for path in files
        },
        "accepted_exceptions": [],
    }


def _coverage(files: list[str]) -> dict[str, object]:
    return {"files": {path: {"summary": _summary()} for path in files}}


class CoverageGateTests(unittest.TestCase):
    """The checker must reject missing, weak and gameable coverage evidence."""

    def _run(
        self,
        manifest: dict[str, object],
        coverage: dict[str, object],
        source_files: list[str],
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for relative_path in source_files:
                target = root / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("# fixture\n", encoding="utf-8")
            manifest_path = root / "manifest.json"
            coverage_path = root / "coverage.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            coverage_path.write_text(json.dumps(coverage), encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--project-root",
                    str(root),
                    "--manifest",
                    str(manifest_path),
                    "--coverage-json",
                    str(coverage_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

    def test_accepts_complete_coverage_evidence(self) -> None:
        files = [
            "src/veridist/domain/model.py",
            "src/veridist/statistics/fit.py",
            "src/veridist/families/normal.py",
            "src/veridist/engine/run.py",
            "src/veridist/result.py",
        ]
        result = self._run(_manifest(files), _coverage(files), files)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_rejects_global_line_or_branch_failure(self) -> None:
        files = [
            "src/veridist/domain/model.py",
            "src/veridist/statistics/fit.py",
            "src/veridist/families/normal.py",
            "src/veridist/engine/run.py",
            "src/veridist/result.py",
        ]
        coverage = _coverage(files)
        for path in files:
            coverage["files"][path]["summary"] = _summary(
                covered_lines=94,
                covered_branches=18,
            )
        result = self._run(_manifest(files), coverage, files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("global line", result.stderr)
        self.assertIn("global branch", result.stderr)

    def test_rejects_critical_and_file_floor_failures(self) -> None:
        files = [
            "src/veridist/domain/model.py",
            "src/veridist/statistics/fit.py",
            "src/veridist/families/normal.py",
            "src/veridist/engine/run.py",
            "src/veridist/result.py",
        ]
        coverage = _coverage(files)
        coverage["files"]["src/veridist/domain/model.py"]["summary"] = _summary(
            covered_lines=97,
            covered_branches=19,
        )
        coverage["files"]["src/veridist/result.py"]["summary"] = _summary(
            covered_lines=89,
            covered_branches=17,
        )
        result = self._run(_manifest(files), coverage, files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("critical line", result.stderr)
        self.assertIn("file line", result.stderr)

    def test_rejects_missing_files_metrics_denominator_drift_and_unlisted_modules(self) -> None:
        files = [
            "src/veridist/domain/model.py",
            "src/veridist/statistics/fit.py",
            "src/veridist/families/normal.py",
            "src/veridist/engine/run.py",
        ]
        manifest = _manifest(files)
        coverage = _coverage(files)
        del coverage["files"]["src/veridist/domain/model.py"]["summary"]["covered_branches"]
        coverage["files"]["src/veridist/statistics/fit.py"]["summary"]["num_statements"] = 99
        coverage["files"]["src/veridist/unlisted.py"] = {"summary": _summary()}
        source_files = [*files, "src/veridist/unlisted.py"]
        result = self._run(manifest, coverage, source_files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing metric", result.stderr)
        self.assertIn("denominator drift", result.stderr)
        self.assertIn("unlisted production file", result.stderr)

    def _exception_project(
        self, repository_root: Path, *, adr: str, expiry: str, create_adr: bool
    ) -> tuple[dict[str, object], dict[str, object]]:
        project_root = repository_root / "python"
        files = [
            "src/veridist/domain/model.py",
            "src/veridist/statistics/fit.py",
            "src/veridist/families/normal.py",
            "src/veridist/engine/run.py",
            "src/veridist/result.py",
        ]
        excepted = files[-1]
        for relative_path in files:
            target = project_root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# fixture\n", encoding="utf-8")
        if create_adr:
            adr_dir = repository_root / "docs" / "adr"
            adr_dir.mkdir(parents=True, exist_ok=True)
            (adr_dir / f"{adr}-coverage-exception.md").write_text("# fixture\n", encoding="utf-8")
        manifest = _manifest(files)
        manifest["accepted_exceptions"] = [
            {
                "path": excepted,
                "owner": "platform",
                "reason": "legacy adapter shim pending removal",
                "expiry": expiry,
                "adr": adr,
            }
        ]
        coverage = _coverage(files)
        coverage["files"][excepted]["summary"] = _summary(covered_lines=89, covered_branches=17)
        return manifest, coverage

    def test_rejects_expired_exception(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository_root = Path(temp_dir)
            manifest, coverage = self._exception_project(
                repository_root, adr="ADR-0001", expiry="2000-01-01", create_adr=True
            )
            (repository_root / "python" / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (repository_root / "python" / "coverage.json").write_text(
                json.dumps(coverage), encoding="utf-8"
            )
            errors = validate(
                repository_root / "python",
                repository_root / "python" / "manifest.json",
                repository_root / "python" / "coverage.json",
                today=date(2026, 1, 1),
            )
            self.assertTrue(any("exception expired" in error for error in errors), errors)

    def test_accepts_exception_exactly_on_its_expiry_date(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository_root = Path(temp_dir)
            manifest, coverage = self._exception_project(
                repository_root, adr="ADR-0001", expiry="2026-06-15", create_adr=True
            )
            (repository_root / "python" / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (repository_root / "python" / "coverage.json").write_text(
                json.dumps(coverage), encoding="utf-8"
            )
            errors = validate(
                repository_root / "python",
                repository_root / "python" / "manifest.json",
                repository_root / "python" / "coverage.json",
                today=date(2026, 6, 15),
            )
            self.assertEqual(errors, [])

    def test_rejects_exception_whose_adr_does_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository_root = Path(temp_dir)
            manifest, coverage = self._exception_project(
                repository_root, adr="ADR-9999", expiry="2099-01-01", create_adr=False
            )
            (repository_root / "python" / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (repository_root / "python" / "coverage.json").write_text(
                json.dumps(coverage), encoding="utf-8"
            )
            errors = validate(
                repository_root / "python",
                repository_root / "python" / "manifest.json",
                repository_root / "python" / "coverage.json",
                today=date(2026, 1, 1),
            )
            self.assertTrue(
                any("exception ADR does not exist" in error for error in errors), errors
            )

    def test_rejects_weak_exception_manifest(self) -> None:
        files = [
            "src/veridist/domain/model.py",
            "src/veridist/statistics/fit.py",
            "src/veridist/families/normal.py",
            "src/veridist/engine/run.py",
        ]
        manifest = _manifest(files)
        manifest["accepted_exceptions"] = [{"path": files[0]}]
        coverage = _coverage(files)
        coverage["files"][files[0]]["summary"] = _summary(covered_lines=89)
        result = self._run(manifest, coverage, files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("exception", result.stderr)


if __name__ == "__main__":
    unittest.main()
