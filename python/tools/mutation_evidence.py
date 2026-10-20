"""Shared, fail-closed primitives for formal mutmut 3.7 evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4  # mutation manifest (adds the shard map); evidence has its own schema version
CRITICAL_MODULES = ("domain", "statistics", "families", "engine", "scale")
MUTATION_TEST_SELECTION = (
    "tests/contract",
    "tests/reference",
    "tests/unit",
    "tests/conformance",
    "tests/property",
)
MUTMUT_VERSION = "3.7.0"
MUTMUT_WHEEL_SHA256 = "1d2f9a1bfa4a474b2213df6b17223150b492bf4a85af0eda4fb322297337fb32"
SHARD_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*")
# ``only_mutate`` entries are fnmatch patterns; a production path must never contain one.
GLOB_CHARACTERS = frozenset("*?[]\\")
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


def score_excluding_type_check(killed: int, survived: int, type_check: int) -> float:
    """Return the mutation score with type-check-rejected mutants removed entirely.

    ``killed`` already counts every mutant that mutmut rejected through its type
    checker (exit code 37); ``type_check`` is the size of that subset. Dropping
    those mutants from numerator and denominator reports how many mutants were
    killed by the test suite itself. The gate score is unchanged and still counts them.
    """

    denominator = killed + survived - type_check
    return 0.0 if denominator == 0 else (killed - type_check) / denominator


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


def module_of(path: str) -> str:
    """Return the critical module (``domain``, ``scale``, ...) that owns a production file."""

    prefix = "src/veridist/"
    if not path.startswith(prefix):
        raise ValueError(f"not a critical production path: {path}")
    return path[len(prefix) :].split("/", 1)[0]


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


def module_minimum_scores(value: object) -> dict[str, float]:
    """Validate the per-module mutation floors: exactly the critical modules, each in (0, 1]."""

    if not isinstance(value, dict) or set(value) != set(CRITICAL_MODULES):
        raise ValueError("module_minimum_scores must name exactly the critical modules")
    floors: dict[str, float] = {}
    for module in CRITICAL_MODULES:
        floor = value[module]
        if (
            isinstance(floor, bool)
            or not isinstance(floor, int | float)
            or not math.isfinite(floor)
            or not 0.0 < floor <= 1.0
        ):
            raise ValueError(f"module minimum score for {module} must be a finite number in (0, 1]")
        floors[module] = float(floor)
    return floors


def mutation_shards(project_root: Path, declared: object) -> dict[str, list[str]]:
    """Validate the declared shard map against the production files on disk.

    Every production file of a critical module must belong to exactly one shard, a shard
    may only name such files, and file lists must be canonical (sorted, no repeats, no
    glob characters, because mutmut matches ``only_mutate`` entries with ``fnmatch``).
    """

    if not isinstance(declared, dict) or not declared:
        raise ValueError("mutation manifest shards must be a non-empty mapping")
    expected = source_files(project_root)
    known = set(expected)
    owner: dict[str, str] = {}
    shards: dict[str, list[str]] = {}
    for name, files in declared.items():
        if not isinstance(name, str) or SHARD_NAME.fullmatch(name) is None:
            raise ValueError(f"invalid mutation shard name: {name!r}")
        if (
            not isinstance(files, list)
            or not files
            or any(not isinstance(item, str) for item in files)
        ):
            raise ValueError(f"mutation shard {name} must list at least one file")
        if files != sorted(set(files)):
            raise ValueError(f"mutation shard {name} files must be sorted and unique")
        for path in files:
            if GLOB_CHARACTERS & set(path):
                raise ValueError(f"mutation shard {name} lists a glob character: {path}")
            if path not in known:
                raise ValueError(
                    f"mutation shard {name} lists a file outside the critical scope: {path}"
                )
            if path in owner:
                raise ValueError(
                    f"production file is assigned to more than one mutation shard: {path} "
                    f"({owner[path]}, {name})"
                )
            owner[path] = name
        shards[name] = list(files)
    unassigned = [path for path in expected if path not in owner]
    if unassigned:
        raise ValueError(
            "production file is not assigned to a mutation shard: " + ", ".join(unassigned)
        )
    return shards


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
        "module_minimum_scores",
        "pytest_selection",
        "shards",
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
    module_minimum_scores(value["module_minimum_scores"])
    mutation_shards(project_root, value["shards"])
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


def shard_config(config: dict[str, Any], files: list[str]) -> dict[str, Any]:
    """Return the mutmut configuration one shard runs: the binding config plus ``only_mutate``."""

    return {**config, "only_mutate": list(files)}


def shard_config_digest(project_root: Path, files: list[str]) -> str:
    return sha256_text(canonical_json(shard_config(mutation_config(project_root), files)))


_MUTMUT_TABLE = re.compile(r"^\[tool\.mutmut\][ \t]*(?:#[^\r\n]*)?\r?$", re.MULTILINE)


def overlay_only_mutate(pyproject: str, files: list[str]) -> str:
    """Return ``pyproject`` with ``only_mutate = files`` added to ``[tool.mutmut]``.

    mutmut 3.7 reads ``only_mutate`` (fnmatch patterns over source paths such as
    ``src/veridist/scale/_fit.py``) from ``[tool.mutmut]`` and mutates only matching files;
    every other file under ``source_paths`` is copied unmutated and gets no ``.meta`` file.
    The checked-in configuration must not carry the key, so the shard runner applies this
    overlay for the duration of ``mutmut run`` only and restores the tracked bytes after.
    """

    import tomllib

    matches = list(_MUTMUT_TABLE.finditer(pyproject))
    if len(matches) != 1:
        raise ValueError("pyproject.toml must contain exactly one [tool.mutmut] table")
    original = tomllib.loads(pyproject)
    table = original["tool"]["mutmut"]
    if "only_mutate" in table:
        raise ValueError("the tracked [tool.mutmut] table must not set only_mutate")
    newline = pyproject.find("\n", matches[0].end())
    if newline < 0:
        raise ValueError("pyproject.toml ends inside the [tool.mutmut] header")
    eol = "\r\n" if pyproject[newline - 1 : newline] == "\r" else "\n"
    line = f"only_mutate = {json.dumps(list(files))}{eol}"
    result = pyproject[: newline + 1] + line + pyproject[newline + 1 :]
    expected = {**original, "tool": {**original["tool"], "mutmut": shard_config(table, files)}}
    if tomllib.loads(result) != expected:
        raise ValueError("only_mutate overlay changed more than the [tool.mutmut] table")
    return result


REQUIRED_NON_TEST_FILES = (
    "pyproject.toml",
    "quality/mutation-manifest.json",
    "tools/mutation_evidence.py",
    "tools/run_mutation.py",
    "tools/check_mutation_evidence.py",
    "tools/merge_mutation_evidence.py",
    "../.github/workflows/mutation.yml",
)


def input_files(project_root: Path) -> list[str]:
    missing = [name for name in REQUIRED_NON_TEST_FILES if not (project_root / name).is_file()]
    if missing:
        raise ValueError(f"mutation input file is missing: {', '.join(sorted(missing))}")
    candidates = (
        source_files(project_root)
        + tree_files(project_root, "tests")
        + list(REQUIRED_NON_TEST_FILES)
    )
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", *candidates],
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError("cannot enumerate tracked mutation inputs")
    files = sorted(item for item in result.stdout.decode("utf-8").split("\0") if item)
    if set(files) != set(candidates):
        raise ValueError("mutation input manifest contains untracked or missing file")
    return files


def input_digest(project_root: Path) -> tuple[dict[str, str], str]:
    records = {
        name: sha256_bytes((project_root / name).resolve().read_bytes())
        for name in input_files(project_root)
    }
    return records, sha256_text(canonical_json(records))
