"""Shared, fail-closed primitives for formal mutmut 3.7 evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
CRITICAL_MODULES = ("domain", "statistics", "families", "engine")
MUTATION_TEST_SELECTION = (
    "tests/contract",
    "tests/reference",
    "tests/unit",
    "tests/conformance",
    "tests/property",
)
MUTMUT_VERSION = "3.7.0"
MUTMUT_WHEEL_SHA256 = "1d2f9a1bfa4a474b2213df6b17223150b492bf4a85af0eda4fb322297337fb32"
META_KEYS = frozenset(
    {
        "exit_code_by_key",
        "hash_by_function_name",
        "type_check_error_by_key",
        "durations_by_key",
        "estimated_durations_by_key",
    }
)


def official_status(exit_code: object) -> str:
    """Return the mutmut 3.7 outcome represented by one cached exit code."""
    if exit_code is None:
        return "not_checked"
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        return "unknown"
    return {
        0: "survived",
        1: "killed",
        3: "killed",
        5: "no_tests",
        33: "no_tests",
        2: "interrupted",
        34: "skipped",
        35: "suspicious",
        36: "timeout",
        37: "type_check",
        -24: "timeout",
        24: "timeout",
        152: "timeout",
        255: "timeout",
        -11: "segfault",
        -9: "segfault",
    }.get(exit_code, "suspicious")


def scoring_status(exit_code: object) -> str:
    outcome = official_status(exit_code)
    if outcome in {"killed", "type_check"}:
        return "killed"
    if outcome == "survived":
        return "survived"
    return "unresolved"


def canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def source_files(project_root: Path) -> list[str]:
    root = project_root / "src" / "veridist"
    return sorted(
        path.relative_to(project_root).as_posix()
        for module in CRITICAL_MODULES
        for path in (root / module).rglob("*.py")
        if path.is_file()
    )


def tree_files(project_root: Path, relative: str) -> list[str]:
    base = project_root / relative
    return sorted(
        path.relative_to(project_root).as_posix()
        for path in base.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and not path.suffix == ".pyc"
    )


def tree_digest(project_root: Path, files: list[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(files):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256((project_root / name).read_bytes()).digest())
    return digest.hexdigest()


def source_tree_digest(project_root: Path, files: list[str] | None = None) -> str:
    return tree_digest(project_root, source_files(project_root) if files is None else files)


def strict_json(source: str) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(source, object_pairs_hook=pairs, parse_constant=constant)


def mutation_manifest(project_root: Path) -> dict[str, Any]:
    value = strict_json(
        (project_root / "quality" / "mutation-manifest.json").read_text(encoding="utf-8")
    )
    wanted = {
        "schema_version",
        "production_root",
        "critical_modules",
        "mutmut_version",
        "minimum_score",
        "pytest_selection",
    }
    if not isinstance(value, dict) or set(value) != wanted:
        raise ValueError("invalid mutation manifest schema")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["production_root"] != "src/veridist"
        or value["critical_modules"] != list(CRITICAL_MODULES)
        or value["mutmut_version"] != MUTMUT_VERSION
        or value["pytest_selection"] != list(MUTATION_TEST_SELECTION)
    ):
        raise ValueError("mutation manifest scope/version drift")
    score = value["minimum_score"]
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(score)
        or score != 0.8
    ):
        raise ValueError("mutation manifest minimum score must be finite 0.8")
    return value


def mutation_config(project_root: Path) -> dict[str, Any]:
    import tomllib

    project = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    config = project.get("tool", {}).get("mutmut")
    if not isinstance(config, dict):
        raise ValueError("[tool.mutmut] is required")
    allowed = {
        "source_paths",
        "pytest_add_cli_args_test_selection",
        "also_copy",
        "mutate_only_covered_lines",
    }
    if (
        set(config) != allowed
        or config.get("source_paths") != [f"src/veridist/{item}" for item in CRITICAL_MODULES]
        or config.get("pytest_add_cli_args_test_selection")
        != list(MUTATION_TEST_SELECTION)
        or config.get("also_copy")
        != [
            "tools",
            "src/veridist/__init__.py",
            "src/veridist/execution.py",
            "src/veridist/inference.py",
            "src/veridist/py.typed",
            "src/veridist/adapters",
            "src/veridist/reporting",
        ]
        or config.get("mutate_only_covered_lines") is not False
    ):
        raise ValueError("invalid mutmut configuration")
    return config


def reject_mutation_pragmas(project_root: Path) -> None:
    pattern = re.compile(r"pragma\s*:\s*no\s*mutate", re.IGNORECASE)
    for name in source_files(project_root):
        if pattern.search((project_root / name).read_text(encoding="utf-8")):
            raise ValueError(f"mutation exclusion pragma is forbidden: {name}")


def config_digest(project_root: Path) -> str:
    return sha256_text(canonical_json(mutation_config(project_root)))


def input_files(project_root: Path) -> list[str]:
    candidates = source_files(project_root) + tree_files(project_root, "tests")
    candidates += [
        "pyproject.toml",
        "quality/mutation-manifest.json",
        "tools/mutation_evidence.py",
        "tools/run_mutation.py",
        "tools/check_mutation_evidence.py",
        "../.github/workflows/mutation.yml",
    ]
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *candidates],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError("cannot enumerate tracked mutation inputs")
    files = sorted(item for item in result.stdout.decode("utf-8").split("\0") if item)
    if set(files) != set(candidates) - {
        name for name in candidates if not (project_root / name).is_file()
    }:
        raise ValueError("mutation input manifest contains untracked or missing file")
    return files


def input_digest(project_root: Path) -> tuple[dict[str, str], str]:
    records = {
        name: sha256_bytes((project_root / name).resolve().read_bytes())
        for name in input_files(project_root)
    }
    return records, sha256_text(canonical_json(records))
