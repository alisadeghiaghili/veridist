"""Fail-closed validator for cache-bound formal mutation evidence v2."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from mutation_evidence import (
    CRITICAL_MODULES,
    META_KEYS,
    MUTATION_TEST_SELECTION,
    MUTMUT_VERSION,
    config_digest,
    input_digest,
    mutation_manifest,
    official_status,
    reject_mutation_pragmas,
    scoring_status,
    sha256_bytes,
    source_files,
    source_tree_digest,
    strict_json,
)

SCHEMA_VERSION = 2
TOP_KEYS = frozenset(
    {
        "schema_version",
        "source",
        "config",
        "environment",
        "provenance",
        "baseline",
        "mutation",
        "files",
        "modules",
        "totals",
        "score",
    }
)
COUNT_KEYS = ("generated", "killed", "survived", "unresolved")


def load_json(source: str) -> object:
    return strict_json(source)


def fail(message: str) -> None:
    raise ValueError(message)


def exact(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        fail(f"{label} has missing or extra keys")
    return cast(dict[str, Any], value)


def integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        fail(f"{label} must be a non-negative integer (not boolean)")
    return cast(int, value)


def finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        fail(f"{label} must be finite non-boolean number")
    return float(cast(float, value))


def utc_timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        fail(f"{label} must be an RFC3339 UTC timestamp")
    try:
        return datetime.fromisoformat(cast(str, value)[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{label} must be an RFC3339 UTC timestamp") from error


def sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        fail(f"{label} must be lower-case SHA-256")
    return cast(str, value)


def phase(
    value: object,
    name: str,
    command: list[str],
    fixture: bool,
    logs_root: Path | None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{name} has invalid state")
    data = cast(dict[str, Any], value)
    if data.get("state") not in {"passed", "failed", "not_run"}:
        fail(f"{name} has invalid state")
    state = data["state"]
    keys = (
        {"state", "command"}
        if state == "not_run"
        else {"state", "command", "started_at", "ended_at", "exit_code", "log_path", "log_sha256"}
    )
    record = exact(data, keys, name)
    if record["command"] != command:
        fail(f"{name} command drift")
    if state == "not_run":
        return record
    started = utc_timestamp(record["started_at"], f"{name}:started_at")
    ended = utc_timestamp(record["ended_at"], f"{name}:ended_at")
    if started > ended:
        fail(f"{name} ended before it started")
    exit_code = record["exit_code"]
    if isinstance(exit_code, bool) or not isinstance(exit_code, int):
        fail(f"{name} exit code invalid")
    if (state == "passed") != (exit_code == 0):
        fail(f"{name} state/exit-code mismatch")
    digest = sha256(record["log_sha256"], f"{name}:log_sha256")
    if not isinstance(record["log_path"], str) or not record["log_path"]:
        fail(f"{name} log path invalid")
    if not fixture:
        if logs_root is None:
            fail("--logs-root is required for non-fixture evidence")
        relative = Path(record["log_path"])
        if relative.is_absolute() or ".." in relative.parts:
            fail(f"{name} log path must be safe and relative")
        path = (cast(Path, logs_root) / relative).resolve()
        if (
            logs_root not in path.parents
            or not path.is_file()
            or sha256_bytes(path.read_bytes()) != digest
        ):
            fail(f"{name} log binding mismatch")
    return record


def commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode:
        fail("cannot determine checked-out git commit")
    return result.stdout.strip()


def ensure_clean_relevant_tree(project_root: Path) -> None:
    """Reject tracked or untracked evidence inputs changed after the mutation run."""
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        fail("cannot inspect evidence checkout cleanliness")
    if result.stdout.strip():
        fail("evidence checkout is dirty")


def file_meta(cache: Path, source: str) -> tuple[dict[str, Any], bytes]:
    path = cache / f"{source}.meta"
    if not path.is_file():
        fail(f"missing raw mutmut meta: {source}")
    raw = path.read_bytes()
    value = strict_json(raw.decode("utf-8"))
    meta = exact(value, set(META_KEYS), f"raw meta {source}")
    for key in META_KEYS:
        if not isinstance(meta[key], dict) or any(not isinstance(name, str) for name in meta[key]):
            fail(f"raw meta {source}:{key} invalid")
    for key, code in meta["exit_code_by_key"].items():
        if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
            fail(f"raw meta {source}:{key} exit code invalid")
    if any(not isinstance(value, str) for value in meta["hash_by_function_name"].values()):
        fail(f"raw meta {source}: function hash invalid")
    if not set(meta["type_check_error_by_key"]).issubset(meta["exit_code_by_key"]):
        fail(f"raw meta {source}: type-check identity drift")
    for key, error in meta["type_check_error_by_key"].items():
        if error is not None and not isinstance(error, str):
            fail(f"raw meta {source}: type-check error invalid")
        if meta["exit_code_by_key"][key] != 37:
            fail(f"raw meta {source}: type-check exit-code drift")
    if any(
        code == 37 and key not in meta["type_check_error_by_key"]
        for key, code in meta["exit_code_by_key"].items()
    ):
        fail(f"raw meta {source}: missing type-check result")
    for key in ("durations_by_key", "estimated_durations_by_key"):
        if not set(meta[key]).issubset(meta["exit_code_by_key"]):
            fail(f"raw meta {source}: duration identity drift")
        for value in meta[key].values():
            if finite(value, f"raw meta {source}:{key}") < 0:
                fail(f"raw meta {source}:{key} duration must be non-negative")
    return meta, raw


def check(
    payload: dict[str, Any],
    project_root: Path,
    fixture: bool = False,
    mutants_root: Path | None = None,
    logs_root: Path | None = None,
) -> None:
    exact(payload, set(TOP_KEYS), "evidence")
    if payload["schema_version"] != SCHEMA_VERSION:
        fail("unsupported mutation evidence schema")
    files = source_files(project_root)
    if not files:
        fail("critical mutation scope is empty")
    source = exact(payload["source"], {"commit", "tree_sha256", "cache_sha256"}, "source")
    if source["commit"] != ("fixture" if fixture else commit(project_root)) or source[
        "tree_sha256"
    ] != source_tree_digest(project_root, files):
        fail("stale source binding")
    config = exact(
        payload["config"],
        {
            "mutmut_version",
            "wheel_sha256",
            "config_sha256",
            "source_paths",
            "pytest_selection",
            "also_copy",
        },
        "config",
    )
    if (
        config["mutmut_version"] != MUTMUT_VERSION
        or config["wheel_sha256"]
        != "1d2f9a1bfa4a474b2213df6b17223150b492bf4a85af0eda4fb322297337fb32"
        or config["config_sha256"] != config_digest(project_root)
        or config["source_paths"] != [f"src/veridist/{name}" for name in CRITICAL_MODULES]
        or config["pytest_selection"] != list(MUTATION_TEST_SELECTION)
        or config["also_copy"]
        != [
            "tools",
            "src/veridist/__init__.py",
            "src/veridist/execution.py",
            "src/veridist/inference.py",
            "src/veridist/py.typed",
            "src/veridist/adapters",
            "src/veridist/reporting",
        ]
    ):
        fail("mutation configuration drift")
    env = exact(payload["environment"], {"python", "platform"}, "environment")
    if not all(isinstance(env[key], str) and env[key] for key in env):
        fail("invalid environment")
    provenance = exact(
        payload["provenance"],
        {"inputs", "input_digest", "pre_input_digest", "post_input_digest"},
        "provenance",
    )
    if not isinstance(provenance["inputs"], dict) or not isinstance(
        provenance["input_digest"], str
    ):
        fail("invalid provenance")
    if not fixture:
        ensure_clean_relevant_tree(project_root)
        current_inputs, current_digest = input_digest(project_root)
        if provenance != {
            "inputs": current_inputs,
            "input_digest": current_digest,
            "pre_input_digest": current_digest,
            "post_input_digest": current_digest,
        }:
            fail("input provenance drift")
    mutation_manifest(project_root)
    reject_mutation_pragmas(project_root)
    baseline = phase(
        payload["baseline"],
        "baseline",
        ["python", "-m", "pytest", "tests"],
        fixture,
        logs_root,
    )
    mutation = phase(payload["mutation"], "mutation", ["mutmut", "run"], fixture, logs_root)
    if baseline["state"] != "passed":
        fail("baseline did not pass")
    if mutation["state"] != "passed":
        fail("mutation run did not complete")
    reports = payload["files"]
    if not isinstance(reports, list):
        fail("files must be a list")
    by_path: dict[str, dict[str, Any]] = {}
    for report in reports:
        if (
            not isinstance(report, dict)
            or not isinstance(report.get("path"), str)
            or report["path"] in by_path
        ):
            fail("invalid or duplicate file report")
        by_path[report["path"]] = report
    if set(by_path) != set(files):
        fail("missing or extra source file report")
    total = {key: 0 for key in COUNT_KEYS}
    seen: set[str] = set()
    cache_parts: list[bytes] = []
    for path in files:
        report = by_path[path]
        exact(
            report,
            {
                "path",
                *COUNT_KEYS,
                "mutants",
                "function_hashes",
                "type_check_errors",
                "durations",
                "estimated_durations",
            },
            path,
        )
        counts = {key: integer(report[key], f"{path}:{key}") for key in COUNT_KEYS}
        mutants = report["mutants"]
        if not isinstance(mutants, list) or counts["generated"] != len(mutants):
            fail(f"{path}: mutant count drift")
        meta: dict[str, Any] | None = None
        if not fixture:
            assert mutants_root is not None
            meta, raw = file_meta(mutants_root, path)
            cache_parts += [path.encode("utf-8"), b"\0", raw]
            for evidence_key, meta_key in (
                ("function_hashes", "hash_by_function_name"),
                ("type_check_errors", "type_check_error_by_key"),
                ("durations", "durations_by_key"),
                ("estimated_durations", "estimated_durations_by_key"),
            ):
                if report[evidence_key] != meta[meta_key]:
                    fail(f"{path}: {evidence_key} does not bind raw cache")
        observed = {key: 0 for key in COUNT_KEYS}
        raw_keys = set() if meta is None else set(meta["exit_code_by_key"])
        for mutant in mutants:
            data = exact(
                mutant,
                {"id", "cache_key", "exit_code", "official_status", "scoring_status"},
                f"{path} mutant",
            )
            identifier, key, code = data["id"], data["cache_key"], data["exit_code"]
            if (
                not isinstance(identifier, str)
                or not identifier.startswith(f"{path}::")
                or identifier != f"{path}::{key}"
                or identifier in seen
                or not isinstance(key, str)
            ):
                fail(f"{path}: invalid mutant identity prefix")
            if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
                fail(f"{path}: invalid mutant exit code")
            if data["official_status"] != official_status(code) or data[
                "scoring_status"
            ] != scoring_status(code):
                fail(f"{path}: status not derived from exit code")
            observed[
                data["scoring_status"]
                if data["scoring_status"] in {"killed", "survived"}
                else "unresolved"
            ] += 1
            seen.add(identifier)
            if meta is not None and (key not in raw_keys or meta["exit_code_by_key"][key] != code):
                fail(f"{path}: raw cache identity/exit-code drift")
        if meta is not None and {item["cache_key"] for item in mutants} != raw_keys:
            fail(f"{path}: omitted/fabricated cache identities")
        if any(counts[key] != observed[key] for key in ("killed", "survived", "unresolved")):
            fail(f"{path}: declared counts mismatch")
        for key in COUNT_KEYS:
            total[key] += counts[key]
    if not fixture and source["cache_sha256"] != sha256_bytes(b"".join(cache_parts)):
        fail("raw cache digest drift")
    declared = exact(payload["totals"], set(COUNT_KEYS), "totals")
    if (
        any(integer(declared[key], f"totals:{key}") != total[key] for key in COUNT_KEYS)
        or total["generated"] != total["killed"] + total["survived"] + total["unresolved"]
    ):
        fail("totals drift")
    modules = payload["modules"]
    if not isinstance(modules, list) or len(modules) != len(CRITICAL_MODULES):
        fail("invalid module reports")
    module_map = {entry.get("module"): entry for entry in modules if isinstance(entry, dict)}
    if set(module_map) != set(CRITICAL_MODULES):
        fail("missing/duplicate module reports")
    for module, report in module_map.items():
        exact(report, {"module", *COUNT_KEYS}, f"module {module}")
        for key in COUNT_KEYS:
            if integer(report[key], f"{module}:{key}") != sum(
                by_path[name][key] for name in files if f"/{module}/" in name
            ):
                fail("module total drift")
    denominator = total["killed"] + total["survived"]
    if denominator == 0 or total["unresolved"]:
        fail("unresolved or zero-scored evidence")
    score = finite(payload["score"], "score")
    if score != total["killed"] / denominator or score < 0.8:
        fail("invalid/insufficient mutation score")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--mutants-root", type=Path)
    parser.add_argument("--logs-root", type=Path)
    args = parser.parse_args()
    try:
        if not args.fixture and (args.mutants_root is None or args.logs_root is None):
            fail("--mutants-root and --logs-root are required for non-fixture evidence")
        payload = load_json(args.evidence.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            fail("evidence root must be object")
        check(
            cast(dict[str, Any], payload),
            args.project_root.resolve(),
            args.fixture,
            args.mutants_root.resolve() if args.mutants_root else None,
            args.logs_root.resolve() if args.logs_root else None,
        )
    except (OSError, ValueError) as error:
        print(f"MUTATION EVIDENCE FAIL: {error}", file=sys.stderr)
        return 1
    print("MUTATION EVIDENCE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
