"""TDD specifications for the deterministic coverage JSON gate."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

from tools.check_coverage import count_pragmas, validate

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
        "pragma_budget": {path: {"no_branch": 0, "no_cover": 0} for path in files},
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
        quality_dir = project_root / "quality"
        quality_dir.mkdir(parents=True, exist_ok=True)
        index = {"ADR-0002": "Statistical correctness and capability matrix"}
        if create_adr:
            index[adr] = "Coverage exception fixture"
        (quality_dir / "adr-index.json").write_text(json.dumps(index), encoding="utf-8")
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

    def _validate_with_index(self, index_text: str | None) -> list[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository_root = Path(temp_dir)
            manifest, coverage = self._exception_project(
                repository_root, adr="ADR-0001", expiry="2099-01-01", create_adr=True
            )
            index_path = repository_root / "python" / "quality" / "adr-index.json"
            if index_text is None:
                index_path.unlink()
            else:
                index_path.write_text(index_text, encoding="utf-8")
            (repository_root / "python" / "manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (repository_root / "python" / "coverage.json").write_text(
                json.dumps(coverage), encoding="utf-8"
            )
            return validate(
                repository_root / "python",
                repository_root / "python" / "manifest.json",
                repository_root / "python" / "coverage.json",
                today=date(2026, 1, 1),
            )

    def test_accepts_exception_whose_adr_is_in_the_index(self) -> None:
        self.assertEqual(self._validate_with_index(json.dumps({"ADR-0001": "Title"})), [])

    def test_rejects_exception_when_the_adr_index_is_missing(self) -> None:
        errors = self._validate_with_index(None)
        self.assertTrue(any("cannot read ADR index" in error for error in errors), errors)
        self.assertTrue(any("exception ADR does not exist" in error for error in errors), errors)

    def test_rejects_a_malformed_adr_index(self) -> None:
        for index in (
            json.dumps({"ADR-0001": ""}),
            json.dumps({"ADR-0001": 7}),
            json.dumps({"adr-1": "Title", "ADR-0001": "Title"}),
            json.dumps(["ADR-0001"]),
            "not json",
        ):
            with self.subTest(index=index):
                errors = self._validate_with_index(index)
                self.assertTrue(errors, index)
                self.assertTrue(
                    any("exception ADR does not exist" in error for error in errors), errors
                )

    def _pragma_project(
        self, root: Path, source: str, budget: dict[str, int] | None
    ) -> tuple[Path, Path, Path]:
        files = [
            "src/veridist/domain/model.py",
            "src/veridist/statistics/fit.py",
            "src/veridist/families/normal.py",
            "src/veridist/engine/run.py",
            "src/veridist/result.py",
        ]
        for relative_path in files:
            target = root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("# fixture\n", encoding="utf-8")
        (root / files[0]).write_text(source, encoding="utf-8")
        manifest = _manifest(files)
        budgets = manifest["pragma_budget"]
        assert isinstance(budgets, dict)
        if budget is None:
            del budgets[files[0]]
        else:
            budgets[files[0]] = budget
        manifest_path = root / "manifest.json"
        coverage_path = root / "coverage.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        coverage_path.write_text(json.dumps(_coverage(files)), encoding="utf-8")
        return root, manifest_path, coverage_path

    def _pragma_errors(self, source: str, budget: dict[str, int] | None) -> list[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, manifest_path, coverage_path = self._pragma_project(
                Path(temp_dir), source, budget
            )
            return validate(root, manifest_path, coverage_path, today=date(2026, 1, 1))

    def test_counts_both_pragma_kinds_by_regex(self) -> None:
        source = (
            "a = 1  # pragma: no cover\n"
            "b = 2  #pragma:no cover - reason\n"
            "if c:  # pragma: no branch\n"
            "d = 'text mentioning pragma and cover'\n"
            "e = 3  # PRAGMA: NO COVER\n"
            "f = 4  # pragma: nocover\n"
        )
        self.assertEqual(count_pragmas(source), {"no_branch": 1, "no_cover": 4})
        self.assertEqual(count_pragmas("# fixture\n"), {"no_branch": 0, "no_cover": 0})

    def test_accepts_a_file_exactly_at_its_pragma_budget(self) -> None:
        source = "a = 1  # pragma: no cover\nif b:  # pragma: no branch\n    pass\n"
        self.assertEqual(self._pragma_errors(source, {"no_branch": 1, "no_cover": 1}), [])

    def test_accepts_a_file_under_its_pragma_budget(self) -> None:
        source = "a = 1  # pragma: no cover\n"
        self.assertEqual(self._pragma_errors(source, {"no_branch": 3, "no_cover": 2}), [])
        self.assertEqual(self._pragma_errors("x = 1\n", {"no_branch": 1, "no_cover": 1}), [])

    def test_rejects_a_file_over_its_pragma_budget(self) -> None:
        source = "a = 1  # pragma: no cover\nb = 2  # pragma: no cover\n"
        errors = self._pragma_errors(source, {"no_branch": 0, "no_cover": 1})
        self.assertEqual(
            errors,
            ["pragma budget exceeded for src/veridist/domain/model.py: no cover 2 > 1"],
        )
        errors = self._pragma_errors(
            "if a:  # pragma: no branch\n    pass\n", {"no_branch": 0, "no_cover": 5}
        )
        self.assertEqual(
            errors,
            ["pragma budget exceeded for src/veridist/domain/model.py: no branch 1 > 0"],
        )

    def test_rejects_a_pragma_in_a_file_the_budget_does_not_list(self) -> None:
        errors = self._pragma_errors("a = 1  # pragma: no cover\n", None)
        self.assertEqual(
            errors, ["unlisted file contains coverage pragmas: src/veridist/domain/model.py"]
        )
        self.assertEqual(self._pragma_errors("a = 1\n", None), [])

    def test_rejects_malformed_pragma_budgets(self) -> None:
        files = ["src/veridist/domain/model.py"]
        malformed_entries: dict[str, object] = {
            "wrong keys": {"no_cover": 0},
            "extra keys": {"no_branch": 0, "no_cover": 0, "other": 0},
            "negative": {"no_branch": -1, "no_cover": 0},
            "boolean": {"no_branch": True, "no_cover": 0},
            "float": {"no_branch": 0.0, "no_cover": 0},
            "not an object": 3,
        }
        for name, entry in malformed_entries.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                target = root / files[0]
                target.parent.mkdir(parents=True)
                target.write_text("# fixture\n", encoding="utf-8")
                manifest = _manifest(files)
                manifest["pragma_budget"] = {files[0]: entry}
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (root / "coverage.json").write_text(
                    json.dumps(_coverage(files)), encoding="utf-8"
                )
                errors = validate(root, root / "manifest.json", root / "coverage.json")
                self.assertIn(f"invalid pragma budget for {files[0]}", errors)

    def test_rejects_budget_structure_errors(self) -> None:
        files = ["src/veridist/domain/model.py"]
        for name, budget, expected in (
            ("not an object", [], "manifest pragma_budget must be an object"),
            (
                "unlisted path",
                {"src/veridist/ghost.py": {"no_branch": 0, "no_cover": 0}},
                "pragma budget names an unlisted file: src/veridist/ghost.py",
            ),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                target = root / files[0]
                target.parent.mkdir(parents=True)
                target.write_text("# fixture\n", encoding="utf-8")
                manifest = _manifest(files)
                manifest["pragma_budget"] = budget
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (root / "coverage.json").write_text(
                    json.dumps(_coverage(files)), encoding="utf-8"
                )
                errors = validate(root, root / "manifest.json", root / "coverage.json")
                self.assertIn(expected, errors)

    def test_rejects_a_manifest_without_a_pragma_budget(self) -> None:
        files = ["src/veridist/domain/model.py"]
        manifest = _manifest(files)
        del manifest["pragma_budget"]
        result = self._run(manifest, _coverage(files), files)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("manifest missing required field: pragma_budget", result.stderr)

    def test_reports_an_unreadable_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root, manifest_path, coverage_path = self._pragma_project(
                Path(temp_dir), "# fixture\n", {"no_branch": 0, "no_cover": 0}
            )
            (root / "src/veridist/domain/model.py").write_bytes(b"\xff\xfe\x00bad")
            errors = validate(root, manifest_path, coverage_path, today=date(2026, 1, 1))
            expected = "cannot read src/veridist/domain/model.py for the pragma budget"
            self.assertTrue(any(error.startswith(expected) for error in errors), errors)

    def test_repository_pragma_budget_matches_the_source_tree_exactly(self) -> None:
        root = Path(__file__).parents[2]
        manifest = json.loads((root / "quality/coverage-manifest.json").read_text("utf-8"))
        budget = manifest["pragma_budget"]
        self.assertEqual(list(budget), sorted(budget))
        self.assertEqual(set(budget), set(manifest["production_files"]))
        for path, recorded in budget.items():
            with self.subTest(path=path):
                self.assertEqual(
                    recorded, count_pragmas((root / path).read_text(encoding="utf-8"))
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
