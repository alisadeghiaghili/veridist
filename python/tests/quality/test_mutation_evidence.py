"""TDD contracts for cache-bound mutation evidence v2."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PYTHON_ROOT / "tools"))
from mutation_evidence import (  # noqa: E402
    CRITICAL_MODULES,
    MUTMUT_WHEEL_SHA256,
    config_digest,
    module_minimum_scores,
    mutation_manifest,
    official_status,
    score_excluding_type_check,
    scoring_status,
    source_tree_digest,
)

CHECKER = PYTHON_ROOT / "tools" / "check_mutation_evidence.py"
RUNNER = PYTHON_ROOT / "tools" / "run_mutation.py"
COUNT_KEYS = ("generated", "killed", "survived", "unresolved")
REPORT_COUNT_KEYS = (*COUNT_KEYS, "type_check")
MUTATION_SELECTION = [
    "tests/contract",
    "tests/reference",
    "tests/unit",
    "tests/conformance",
    "tests/property",
]
MUTATION_COPY = [
    "tools",
    "src/veridist/__init__.py",
    "src/veridist/execution.py",
    "src/veridist/inference.py",
    "src/veridist/py.typed",
    "src/veridist/adapters",
    "src/veridist/reporting",
]


def command(exit_code: int = 0) -> dict[str, object]:
    return {
        "state": "passed" if exit_code == 0 else "failed",
        "command": ["python", "-m", "pytest", "tests"],
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:01Z",
        "exit_code": exit_code,
        "log_path": "fixture.log",
        "log_sha256": "0" * 64,
    }


def fixture(root: Path) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    for module in CRITICAL_MODULES:
        name = f"src/veridist/{module}/sample.py"
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def value():\n    return 1\n", encoding="utf-8")
        reports.append(
            {
                "path": name,
                "generated": 1,
                "killed": 1,
                "survived": 0,
                "unresolved": 0,
                "type_check": 0,
                "mutants": [
                    {
                        "id": f"{name}::value__mutmut_1",
                        "cache_key": "value__mutmut_1",
                        "exit_code": 1,
                        "official_status": "killed",
                        "scoring_status": "killed",
                    }
                ],
                "function_hashes": {},
                "type_check_errors": {},
                "durations": {},
                "estimated_durations": {},
            }
        )
    (root / "pyproject.toml").write_text(
        "[tool.mutmut]\n"
        'source_paths = ["src/veridist/domain", "src/veridist/statistics", '
        '"src/veridist/families", "src/veridist/engine", "src/veridist/scale"]\n'
        'pytest_add_cli_args_test_selection = ["tests/contract", "tests/reference", "tests/unit", '
        '"tests/conformance", "tests/property"]\n'
        'also_copy = ["tools", "src/veridist/__init__.py", "src/veridist/execution.py", '
        '"src/veridist/inference.py", "src/veridist/py.typed", "src/veridist/adapters", '
        '"src/veridist/reporting"]\n'
        'mutate_only_covered_lines = false\n',
        encoding="utf-8",
    )
    (root / "quality").mkdir()
    (root / "quality" / "mutation-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "production_root": "src/veridist",
                "critical_modules": list(CRITICAL_MODULES),
                "mutmut_version": "3.7.0",
                "minimum_score": 0.8,
                "module_minimum_scores": {module: 0.5 for module in CRITICAL_MODULES},
                "pytest_selection": MUTATION_SELECTION,
            }
        ),
        encoding="utf-8",
    )
    return {
        "schema_version": 3,
        "source": {
            "commit": "fixture",
            "tree_sha256": source_tree_digest(root),
            "cache_sha256": "fixture",
        },
        "config": {
            "mutmut_version": "3.7.0",
            "wheel_sha256": MUTMUT_WHEEL_SHA256,
            "config_sha256": config_digest(root),
            "source_paths": [f"src/veridist/{name}" for name in CRITICAL_MODULES],
            "pytest_selection": MUTATION_SELECTION,
            "also_copy": MUTATION_COPY,
        },
        "environment": {"python": "fixture", "platform": "fixture"},
        "provenance": {
            "inputs": {},
            "input_digest": "fixture",
            "pre_input_digest": "fixture",
            "post_input_digest": "fixture",
        },
        "baseline": command(),
        "mutation": {**command(), "command": ["mutmut", "run"]},
        "files": reports,
        "modules": [
            {
                "module": module,
                **{key: 1 if key in {"generated", "killed"} else 0 for key in REPORT_COUNT_KEYS},
            }
            for module in CRITICAL_MODULES
        ],
        "totals": {
            key: len(CRITICAL_MODULES) if key in {"generated", "killed"} else 0
            for key in REPORT_COUNT_KEYS
        },
        "score": 1.0,
        "score_excluding_type_check": 1.0,
    }


def check(root: Path, payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
    evidence = root / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--project-root",
            str(root),
            "--evidence",
            str(evidence),
            "--fixture",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def mutant(name: str, key: str, exit_code: int) -> dict[str, object]:
    return {
        "id": f"{name}::{key}",
        "cache_key": key,
        "exit_code": exit_code,
        "official_status": official_status(exit_code),
        "scoring_status": scoring_status(exit_code),
    }


def with_type_check_mutants(payload: dict[str, object]) -> dict[str, object]:
    """Give the first file 5 test kills, 4 type-check kills and 1 survivor."""
    report = payload["files"][0]  # type: ignore[index]
    name = report["path"]
    report["mutants"] = [
        *(mutant(name, f"killed_{index}", 1) for index in range(5)),
        *(mutant(name, f"typed_{index}", 37) for index in range(4)),
        mutant(name, "survivor", 0),
    ]
    report.update(generated=10, killed=9, survived=1, type_check=4)
    module = payload["modules"][0]  # type: ignore[index]
    module.update(generated=10, killed=9, survived=1, type_check=4)
    others = len(CRITICAL_MODULES) - 1  # one perfect mutant in each remaining module
    payload["totals"] = {
        "generated": 10 + others,
        "killed": 9 + others,
        "survived": 1,
        "unresolved": 0,
        "type_check": 4,
    }
    payload["score"] = (9 + others) / (10 + others)
    payload["score_excluding_type_check"] = (5 + others) / (6 + others)
    return payload


def write_manifest(root: Path, **changes: object) -> None:
    """Rewrite the fixture's mutation manifest, replacing keys (``None`` removes a key)."""

    path = root / "quality" / "mutation-manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is None:
            del manifest[key]
        else:
            manifest[key] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")


def with_module_counts(
    payload: dict[str, object], module: str, killed: int, survived: int
) -> dict[str, object]:
    """Make ``module`` report ``killed`` kills and ``survived`` survivors in one file."""

    report = next(
        item
        for item in payload["files"]  # type: ignore[attr-defined]
        if f"/{module}/" in item["path"]
    )
    name = report["path"]
    report["mutants"] = [
        *(mutant(name, f"killed_{index}", 1) for index in range(killed)),
        *(mutant(name, f"survived_{index}", 0) for index in range(survived)),
    ]
    report.update(generated=killed + survived, killed=killed, survived=survived)
    entry = next(
        item
        for item in payload["modules"]  # type: ignore[attr-defined]
        if item["module"] == module
    )
    entry.update(generated=killed + survived, killed=killed, survived=survived)
    totals = {key: 0 for key in REPORT_COUNT_KEYS}
    for item in payload["files"]:  # type: ignore[attr-defined]
        for key in REPORT_COUNT_KEYS:
            totals[key] += item[key]
    payload["totals"] = totals
    payload["score"] = totals["killed"] / (totals["killed"] + totals["survived"])
    payload["score_excluding_type_check"] = payload["score"]
    return payload


class MutationEvidenceTests(unittest.TestCase):
    def test_type_check_kills_are_counted_separately_and_scored_both_ways(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = with_type_check_mutants(fixture(root))
            result = check(root, payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotEqual(payload["score"], payload["score_excluding_type_check"])

    def test_rejects_type_check_accounting_and_score_tampering(self) -> None:
        def file_count(payload: dict[str, object]) -> None:
            payload["files"][0]["type_check"] = 3  # type: ignore[index]

        def total_count(payload: dict[str, object]) -> None:
            payload["totals"]["type_check"] = 5  # type: ignore[index]

        def module_count(payload: dict[str, object]) -> None:
            payload["modules"][0]["type_check"] = 0  # type: ignore[index]

        def hidden_type_checks(payload: dict[str, object]) -> None:
            for report in payload["files"]:  # type: ignore[attr-defined]
                report["type_check"] = 0
            payload["totals"]["type_check"] = 0  # type: ignore[index]
            payload["modules"][0]["type_check"] = 0  # type: ignore[index]

        def plain_score_reused(payload: dict[str, object]) -> None:
            payload["score_excluding_type_check"] = payload["score"]

        def missing_score(payload: dict[str, object]) -> None:
            del payload["score_excluding_type_check"]

        def boolean_score(payload: dict[str, object]) -> None:
            payload["score_excluding_type_check"] = True

        def gate_score_changed(payload: dict[str, object]) -> None:
            payload["score"] = 8 / 9

        def schema_v2(payload: dict[str, object]) -> None:
            payload["schema_version"] = 2

        for name, tamper in (
            ("file count", file_count),
            ("total count", total_count),
            ("module count", module_count),
            ("hidden type-check kills", hidden_type_checks),
            ("plain score reused", plain_score_reused),
            ("missing score", missing_score),
            ("boolean score", boolean_score),
            ("gate score changed", gate_score_changed),
            ("old schema", schema_v2),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = with_type_check_mutants(fixture(root))
                tamper(payload)
                self.assertNotEqual(check(root, payload).returncode, 0)

    def test_rejects_evidence_without_the_type_check_fields(self) -> None:
        for name in ("file", "total", "module"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = fixture(root)
                if name == "file":
                    del payload["files"][0]["type_check"]  # type: ignore[index]
                elif name == "total":
                    del payload["totals"]["type_check"]  # type: ignore[index]
                else:
                    del payload["modules"][0]["type_check"]  # type: ignore[index]
                self.assertNotEqual(check(root, payload).returncode, 0)

    def test_score_excluding_type_check_removes_those_mutants_entirely(self) -> None:
        self.assertEqual(score_excluding_type_check(12, 1, 4), 8 / 9)
        self.assertEqual(score_excluding_type_check(12, 1, 0), 12 / 13)
        self.assertEqual(score_excluding_type_check(4, 0, 4), 0.0)
        self.assertEqual(score_excluding_type_check(0, 0, 0), 0.0)

    def test_runner_report_counts_type_check_mutants_inside_killed(self) -> None:
        import run_mutation

        meta = {
            "exit_code_by_key": {"a": 1, "b": 37, "c": 37, "d": 0, "e": 36},
            "hash_by_function_name": {},
            "type_check_error_by_key": {"b": "error", "c": None},
            "durations_by_key": {},
            "estimated_durations_by_key": {},
        }
        report = run_mutation.report_for("src/veridist/domain/x.py", meta)
        self.assertEqual(
            {key: report[key] for key in (*COUNT_KEYS, "type_check")},
            {"generated": 5, "killed": 3, "survived": 1, "unresolved": 1, "type_check": 2},
        )
        empty = run_mutation.report_for("src/veridist/domain/y.py", None)
        self.assertEqual(empty["type_check"], 0)

    def test_accepts_complete_deterministic_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = check(root, fixture(root))
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_status_score_and_schema_tampering(self) -> None:
        for change in ("bool", "wrong-status", "extra-key", "unresolved", "score"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = fixture(root)
                report = payload["files"][0]  # type: ignore[index]
                if change == "bool":
                    report["killed"] = True
                elif change == "wrong-status":
                    report["mutants"][0]["official_status"] = "survived"
                elif change == "extra-key":
                    report["extra"] = 1
                elif change == "unresolved":
                    report["mutants"][0]["exit_code"] = None
                    report["mutants"][0]["official_status"] = "not_checked"
                    report["mutants"][0]["scoring_status"] = "unresolved"
                    report["killed"], report["unresolved"] = 0, 1
                    payload["totals"]["killed"], payload["totals"]["unresolved"] = 3, 1
                    payload["modules"][0]["killed"], payload["modules"][0]["unresolved"] = 0, 1
                else:
                    payload["score"] = True
                self.assertNotEqual(check(root, payload).returncode, 0)

    def test_runner_refuses_native_windows_before_mutmut_execution(self) -> None:
        launcher = (
            "import platform, runpy, sys; "
            "platform.system = lambda: 'Windows'; "
            f"sys.path.insert(0, {str(PYTHON_ROOT / 'tools')!r}); "
            f"sys.argv = [{str(RUNNER)!r}, '--project-root', {str(PYTHON_ROOT)!r}, "
            "'--output', 'ignored.json']; "
            f"runpy.run_path({str(RUNNER)!r}, run_name='__main__')"
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                launcher,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires POSIX/fork", result.stderr)


class ModuleMinimumScoreTests(unittest.TestCase):
    def test_the_manifest_declares_a_floor_for_every_critical_module(self) -> None:
        manifest = mutation_manifest(PYTHON_ROOT)
        floors = module_minimum_scores(manifest["module_minimum_scores"])
        self.assertEqual(tuple(floors), CRITICAL_MODULES)
        for module, floor in floors.items():
            with self.subTest(module=module):
                self.assertGreater(floor, 0.0)
                self.assertLessEqual(floor, manifest["minimum_score"])

    def test_a_module_below_its_floor_fails_even_when_the_global_score_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # domain: 9 killed and 11 survived (0.45); every other module is perfect.
            payload = with_module_counts(fixture(root), "domain", 9, 11)
            for module in ("statistics", "families", "engine", "scale"):
                payload = with_module_counts(payload, module, 400, 0)
            write_manifest(
                root, module_minimum_scores={module: 0.5 for module in CRITICAL_MODULES}
            )
            self.assertGreaterEqual(payload["score"], 0.8)  # type: ignore[operator]
            result = check(root, payload)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn(
                "module domain mutation score 0.4500 is below its minimum 0.50", result.stderr
            )

    def test_a_module_exactly_at_its_floor_passes_and_scores_are_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = with_module_counts(fixture(root), "domain", 9, 11)
            for module in ("statistics", "families", "engine", "scale"):
                payload = with_module_counts(payload, module, 400, 0)
            write_manifest(
                root,
                module_minimum_scores={
                    "domain": 0.45,
                    "statistics": 0.8,
                    "families": 0.8,
                    "engine": 0.8,
                    "scale": 0.8,
                },
            )
            result = check(root, payload)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("module domain: mutation score 0.4500", result.stdout)
            self.assertIn("module engine: mutation score 1.0000", result.stdout)
            self.assertIn("MUTATION EVIDENCE PASS", result.stdout)

    def test_the_manifest_floors_are_validated_strictly(self) -> None:
        good = {module: 0.5 for module in CRITICAL_MODULES}
        bad_floors: dict[str, object] = {
            "missing module": {k: v for k, v in good.items() if k != "engine"},
            "unknown module": {**good, "adapters": 0.5},
            "not a mapping": [0.5, 0.5, 0.5, 0.5],
            "boolean floor": {**good, "domain": True},
            "string floor": {**good, "domain": "0.5"},
            "zero floor": {**good, "domain": 0.0},
            "negative floor": {**good, "domain": -0.1},
            "floor above one": {**good, "domain": 1.01},
            "not a number": {**good, "domain": float("nan")},
            "infinite": {**good, "domain": float("inf")},
        }
        for name, floors in bad_floors.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = fixture(root)
                write_manifest(root, module_minimum_scores=floors)
                result = check(root, payload)
                self.assertEqual(result.returncode, 1)
                self.assertIn("MUTATION EVIDENCE FAIL", result.stderr)
                with self.assertRaises(ValueError):
                    mutation_manifest(root)

    def test_the_manifest_without_floors_or_with_an_old_schema_is_rejected(self) -> None:
        for name, changes in (
            ("no floors", {"module_minimum_scores": None}),
            ("old schema", {"schema_version": 2}),
            ("extra key", {"module_target_scores": {}}),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = fixture(root)
                write_manifest(root, **changes)
                self.assertEqual(check(root, payload).returncode, 1)

    def test_the_global_minimum_stays_at_eight_tenths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root)
            for score in (0.79, 0.81, True):
                write_manifest(root, minimum_score=score)
                with self.subTest(score=score), self.assertRaises(ValueError):
                    mutation_manifest(root)
