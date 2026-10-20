"""TDD contracts for the cache-bound evidence of one mutation shard (schema 4)."""

from __future__ import annotations

import hashlib
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
    module_of,
    mutation_manifest,
    official_status,
    score_excluding_type_check,
    scoring_status,
    shard_config_digest,
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
# The fixture package: one file per critical module, plus a second ``scale`` file so that the
# ``scale`` module is split over two shards, exactly like the production shard map.
FIXTURE_SHARDS: dict[str, list[str]] = {
    "domain": ["src/veridist/domain/sample.py"],
    "statistics": ["src/veridist/statistics/sample.py"],
    "families": ["src/veridist/families/sample.py"],
    "engine": ["src/veridist/engine/sample.py"],
    "scale-a": ["src/veridist/scale/other.py"],
    "scale-b": ["src/veridist/scale/sample.py"],
}
PYPROJECT = (
    "[tool.mutmut]\n"
    'source_paths = ["src/veridist/domain", "src/veridist/statistics", '
    '"src/veridist/families", "src/veridist/engine", "src/veridist/scale"]\n'
    'pytest_add_cli_args_test_selection = ["tests/contract", "tests/reference", "tests/unit", '
    '"tests/conformance", "tests/property"]\n'
    'also_copy = ["tools", "src/veridist/__init__.py", "src/veridist/execution.py", '
    '"src/veridist/inference.py", "src/veridist/py.typed", "src/veridist/adapters", '
    '"src/veridist/reporting"]\n'
    "mutate_only_covered_lines = false\n"
)


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


def mutant(name: str, key: str, exit_code: int) -> dict[str, object]:
    return {
        "id": f"{name}::{key}",
        "cache_key": key,
        "exit_code": exit_code,
        "official_status": official_status(exit_code),
        "scoring_status": scoring_status(exit_code),
    }


def write_tree(root: Path, shards: dict[str, list[str]] | None = None) -> None:
    """Write the fixture sources, ``[tool.mutmut]`` configuration and manifest into ``root``."""

    shards = FIXTURE_SHARDS if shards is None else shards
    for files in FIXTURE_SHARDS.values():
        for name in files:
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("def value():\n    return 1\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    (root / "quality").mkdir(exist_ok=True)
    (root / "quality" / "mutation-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 4,
                "production_root": "src/veridist",
                "critical_modules": list(CRITICAL_MODULES),
                "mutmut_version": "3.7.0",
                "minimum_score": 0.8,
                "module_minimum_scores": {module: 0.5 for module in CRITICAL_MODULES},
                "pytest_selection": MUTATION_SELECTION,
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )


def shard_evidence(root: Path, shard: str) -> dict[str, object]:
    """Return valid fixture-mode evidence for ``shard``: one killed mutant in each of its files."""

    files = FIXTURE_SHARDS[shard]
    reports: list[dict[str, object]] = []
    for name in files:
        reports.append(
            {
                "path": name,
                "generated": 1,
                "killed": 1,
                "survived": 0,
                "unresolved": 0,
                "type_check": 0,
                "mutants": [mutant(name, "value__mutmut_1", 1)],
                "function_hashes": {},
                "type_check_errors": {},
                "durations": {},
                "estimated_durations": {},
            }
        )
    modules = sorted({module_of(name) for name in files}, key=CRITICAL_MODULES.index)
    count = len(files)
    return {
        "schema_version": 4,
        "shard": {
            "name": shard,
            "files": [
                {
                    "path": name,
                    "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest(),
                }
                for name in files
            ],
        },
        "source": {
            "commit": "fixture",
            "tree_sha256": source_tree_digest(root, files),
            "cache_sha256": "fixture",
        },
        "config": {
            "mutmut_version": "3.7.0",
            "wheel_sha256": MUTMUT_WHEEL_SHA256,
            "config_sha256": config_digest(root),
            "source_paths": [f"src/veridist/{name}" for name in CRITICAL_MODULES],
            "pytest_selection": MUTATION_SELECTION,
            "also_copy": MUTATION_COPY,
            "only_mutate": list(files),
            "effective_config_sha256": shard_config_digest(root, files),
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
                **{
                    key: sum(1 for name in files if module_of(name) == module)
                    if key in {"generated", "killed"}
                    else 0
                    for key in REPORT_COUNT_KEYS
                },
            }
            for module in modules
        ],
        "totals": {
            key: count if key in {"generated", "killed"} else 0 for key in REPORT_COUNT_KEYS
        },
        "score": 1.0,
        "score_excluding_type_check": 1.0,
    }


def fixture(root: Path, shard: str = "domain") -> dict[str, object]:
    write_tree(root)
    return shard_evidence(root, shard)


def check(
    root: Path, payload: dict[str, object], shard: str | None = None
) -> subprocess.CompletedProcess[str]:
    evidence = root / "evidence.json"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    if shard is None:
        shard = str(payload["shard"]["name"])  # type: ignore[index]
    return subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--project-root",
            str(root),
            "--shard",
            shard,
            "--evidence",
            str(evidence),
            "--fixture",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def with_counts(
    payload: dict[str, object], killed: int, survived: int, type_check: int = 0
) -> dict[str, object]:
    """Give the first file of a one-file shard ``killed`` test kills, type-check kills, survivors.

    ``killed`` counts every killed mutant, including the ``type_check`` subset.
    """

    report = payload["files"][0]  # type: ignore[index]
    name = report["path"]
    report["mutants"] = [
        *(mutant(name, f"killed_{index}", 1) for index in range(killed - type_check)),
        *(mutant(name, f"typed_{index}", 37) for index in range(type_check)),
        *(mutant(name, f"survived_{index}", 0) for index in range(survived)),
    ]
    report.update(
        generated=killed + survived, killed=killed, survived=survived, type_check=type_check
    )
    module = payload["modules"][0]  # type: ignore[index]
    module.update(generated=killed + survived, killed=killed, survived=survived)
    module["type_check"] = type_check
    payload["totals"] = {
        "generated": killed + survived,
        "killed": killed,
        "survived": survived,
        "unresolved": 0,
        "type_check": type_check,
    }
    payload["score"] = killed / (killed + survived)
    payload["score_excluding_type_check"] = score_excluding_type_check(
        killed, survived, type_check
    )
    return payload


def with_type_check_mutants(payload: dict[str, object]) -> dict[str, object]:
    """Give the shard's file 5 test kills, 4 type-check kills and 1 survivor."""

    return with_counts(payload, killed=9, survived=1, type_check=4)


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
                    payload["totals"]["killed"], payload["totals"]["unresolved"] = 0, 1
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
            "'--shard', 'domain', '--output', 'ignored.json']; "
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
            ("old schema", {"schema_version": 3}),
            ("no shards", {"shards": None}),
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


class ShardEvidenceTests(unittest.TestCase):
    def test_every_fixture_shard_is_accepted_and_reports_only_its_own_modules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_tree(root)
            for shard in FIXTURE_SHARDS:
                with self.subTest(shard=shard):
                    result = check(root, shard_evidence(root, shard))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"shard {shard} module", result.stdout)

    def test_a_shard_below_the_module_floor_is_left_to_the_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = with_counts(fixture(root, "scale-a"), killed=1, survived=9)
            result = check(root, payload)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_shard_identity_and_file_binding_tampering(self) -> None:
        def other_shard(payload: dict[str, object]) -> None:
            payload["shard"]["name"] = "engine"  # type: ignore[index]

        def unknown_shard(payload: dict[str, object]) -> None:
            payload["shard"]["name"] = "scale"  # type: ignore[index]

        def extra_file(payload: dict[str, object]) -> None:
            payload["shard"]["files"].append(  # type: ignore[index]
                {"path": "src/veridist/engine/sample.py", "sha256": "0" * 64}
            )

        def missing_file_list(payload: dict[str, object]) -> None:
            payload["shard"]["files"] = []  # type: ignore[index]

        def file_hash(payload: dict[str, object]) -> None:
            payload["shard"]["files"][0]["sha256"] = "1" * 64  # type: ignore[index]

        def upper_hash(payload: dict[str, object]) -> None:
            entry = payload["shard"]["files"][0]  # type: ignore[index]
            entry["sha256"] = entry["sha256"].upper()

        def shard_key(payload: dict[str, object]) -> None:
            payload["shard"]["extra"] = 1  # type: ignore[index]

        def no_shard(payload: dict[str, object]) -> None:
            del payload["shard"]

        def foreign_report(payload: dict[str, object]) -> None:
            payload["files"][0]["path"] = "src/veridist/engine/sample.py"  # type: ignore[index]

        def foreign_module(payload: dict[str, object]) -> None:
            payload["modules"][0]["module"] = "engine"  # type: ignore[index]

        def only_mutate_drift(payload: dict[str, object]) -> None:
            payload["config"]["only_mutate"] = []  # type: ignore[index]

        def only_mutate_widened(payload: dict[str, object]) -> None:
            payload["config"]["only_mutate"].append("src/veridist/engine/sample.py")  # type: ignore[index]

        def effective_digest(payload: dict[str, object]) -> None:
            payload["config"]["effective_config_sha256"] = "2" * 64  # type: ignore[index]

        def no_only_mutate(payload: dict[str, object]) -> None:
            del payload["config"]["only_mutate"]  # type: ignore[attr-defined]

        for name, tamper in (
            ("other shard", other_shard),
            ("unknown shard", unknown_shard),
            ("extra file", extra_file),
            ("empty file list", missing_file_list),
            ("file hash", file_hash),
            ("upper-case hash", upper_hash),
            ("shard key", shard_key),
            ("no shard block", no_shard),
            ("foreign file report", foreign_report),
            ("foreign module report", foreign_module),
            ("only_mutate drift", only_mutate_drift),
            ("only_mutate widened", only_mutate_widened),
            ("effective config digest", effective_digest),
            ("no only_mutate", no_only_mutate),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = fixture(root, "domain")
                tamper(payload)
                result = check(root, payload, shard="domain")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("MUTATION EVIDENCE FAIL", result.stderr)

    def test_a_changed_source_file_is_rejected_by_its_recorded_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = fixture(root, "domain")
            (root / "src/veridist/domain/sample.py").write_text("x = 2\n", encoding="utf-8")
            self.assertEqual(check(root, payload).returncode, 1)

    def test_evidence_for_another_shard_is_rejected_by_the_shard_argument(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = fixture(root, "engine")
            result = check(root, payload, shard="domain")
            self.assertEqual(result.returncode, 1)
            self.assertIn("evidence is for shard engine, not domain", result.stderr)

    def test_a_shard_with_an_unresolved_or_unscored_mutant_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = fixture(root, "domain")
            report = payload["files"][0]  # type: ignore[index]
            report["mutants"][0].update(
                exit_code=36, official_status="timeout", scoring_status="unresolved"
            )
            report.update(killed=0, unresolved=1)
            payload["modules"][0].update(killed=0, unresolved=1)  # type: ignore[index]
            payload["totals"].update(killed=0, unresolved=1)  # type: ignore[attr-defined]
            result = check(root, payload)
            self.assertEqual(result.returncode, 1)
            self.assertIn("unresolved or zero-scored evidence", result.stderr)

    def test_the_shard_argument_is_required(self) -> None:
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--project-root", ".", "--evidence", "e.json"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--shard", result.stderr)
