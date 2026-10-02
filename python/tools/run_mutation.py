"""Run pinned mutmut on Linux and export cache-bound evidence without guessing."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from mutation_evidence import (
    CRITICAL_MODULES,
    MUTATION_TEST_SELECTION,
    MUTMUT_VERSION,
    MUTMUT_WHEEL_SHA256,
    config_digest,
    input_digest,
    mutation_config,
    mutation_manifest,
    official_status,
    reject_mutation_pragmas,
    score_excluding_type_check,
    scoring_status,
    sha256_bytes,
    source_files,
    source_tree_digest,
    strict_json,
)


def fail(message: str) -> None:
    raise RuntimeError(message)


def git(project_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=project_root, capture_output=True, text=True, check=False
    )
    if result.returncode:
        fail(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def ensure_clean_tree(project_root: Path) -> str:
    # This executes `git status --porcelain` with all untracked paths included.
    if git(project_root, "status", "--porcelain", "--untracked-files=all"):
        fail("refusing mutation run: repository is dirty")
    return git(project_root, "rev-parse", "HEAD")


def now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def run_logged(
    command: list[str], execution: list[str], project_root: Path, log_path: Path, logs_root: Path
) -> dict[str, Any]:
    started = now()
    try:
        result = subprocess.run(
            execution, cwd=project_root, capture_output=True, text=True, check=False
        )
        exit_code = result.returncode
        content = result.stdout + result.stderr
    except OSError as error:
        exit_code = 127
        content = f"process start failed: {error}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(content, encoding="utf-8")
    return {
        "state": "passed" if exit_code == 0 else "failed",
        "command": command,
        "started_at": started,
        "ended_at": now(),
        "exit_code": exit_code,
        "log_path": log_path.relative_to(logs_root).as_posix(),
        "log_sha256": sha256_bytes(log_path.read_bytes()),
    }


def read_meta(cache_root: Path, source: str) -> tuple[dict[str, Any], bytes] | None:
    path = cache_root / f"{source}.meta"
    if not path.is_file():
        return None
    raw = path.read_bytes()
    value = strict_json(raw.decode("utf-8"))
    if not isinstance(value, dict):
        fail(f"invalid raw cache: {source}")
    return cast(dict[str, Any], value), raw


def report_for(path: str, meta: dict[str, Any] | None) -> dict[str, Any]:
    mutants: list[dict[str, Any]] = []
    if meta is not None:
        exits = meta.get("exit_code_by_key")
        if not isinstance(exits, dict):
            fail(f"missing raw exit codes: {path}")
        for key, code in sorted(cast(dict[str, object], exits).items()):
            if not isinstance(key, str):
                fail(f"invalid raw cache identity: {path}")
            mutants.append(
                {
                    "id": f"{path}::{key}",
                    "cache_key": key,
                    "exit_code": code,
                    "official_status": official_status(code),
                    "scoring_status": scoring_status(code),
                }
            )
    counts = {
        "generated": len(mutants),
        "killed": 0,
        "survived": 0,
        "unresolved": 0,
        "type_check": 0,
    }
    for item in mutants:
        key = item["scoring_status"]
        counts[key if key in {"killed", "survived"} else "unresolved"] += 1
        if item["official_status"] == "type_check":
            counts["type_check"] += 1
    return {
        "path": path,
        **counts,
        "mutants": mutants,
        "function_hashes": {} if meta is None else meta.get("hash_by_function_name", {}),
        "type_check_errors": {} if meta is None else meta.get("type_check_error_by_key", {}),
        "durations": {} if meta is None else meta.get("durations_by_key", {}),
        "estimated_durations": {} if meta is None else meta.get("estimated_durations_by_key", {}),
    }


def export(
    project_root: Path,
    output: Path,
    commit: str,
    wheel: Path,
    baseline: dict[str, Any],
    mutation: dict[str, Any],
    cache_root: Path,
    records: dict[str, str],
    pre_digest: str,
    post_digest: str,
) -> None:
    files = source_files(project_root)
    cache_parts: list[bytes] = []
    reports: list[dict[str, Any]] = []
    for path in files:
        pair = read_meta(cache_root, path)
        reports.append(report_for(path, None if pair is None else pair[0]))
        if pair is not None:
            cache_parts.extend([path.encode("utf-8"), b"\0", pair[1]])
    totals = {
        key: sum(int(report[key]) for report in reports)
        for key in ("generated", "killed", "survived", "unresolved", "type_check")
    }
    modules = [
        {
            "module": module,
            **{
                key: sum(int(report[key]) for report in reports if f"/{module}/" in report["path"])
                for key in totals
            },
        }
        for module in CRITICAL_MODULES
    ]
    denominator = totals["killed"] + totals["survived"]
    payload = {
        "schema_version": 3,
        "source": {
            "commit": commit,
            "tree_sha256": source_tree_digest(project_root, files),
            "cache_sha256": sha256_bytes(b"".join(cache_parts)),
        },
        "config": {
            "mutmut_version": MUTMUT_VERSION,
            "wheel_sha256": sha256_bytes(wheel.read_bytes()),
            "config_sha256": config_digest(project_root),
            "source_paths": [f"src/veridist/{name}" for name in CRITICAL_MODULES],
            "pytest_selection": list(MUTATION_TEST_SELECTION),
            "also_copy": [
                "tools",
                "src/veridist/__init__.py",
                "src/veridist/execution.py",
                "src/veridist/inference.py",
                "src/veridist/py.typed",
                "src/veridist/adapters",
                "src/veridist/reporting",
            ],
        },
        "environment": {"python": sys.version, "platform": platform.platform()},
        "provenance": {
            "inputs": records,
            "input_digest": pre_digest,
            "pre_input_digest": pre_digest,
            "post_input_digest": post_digest,
        },
        "baseline": baseline,
        "mutation": mutation,
        "files": reports,
        "modules": modules,
        "totals": totals,
        "score": 0.0 if denominator == 0 else totals["killed"] / denominator,
        "score_excluding_type_check": score_excluding_type_check(
            totals["killed"], totals["survived"], totals["type_check"]
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".tmp")
    staging.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8"
    )
    if (
        pre_digest != post_digest
        or ensure_clean_tree(project_root) != commit
        or input_digest(project_root)[1] != post_digest
    ):
        staging.unlink(missing_ok=True)
        fail("repository/input drift before evidence publication")
    staging.replace(output)


def verify_mutmut(wheel: Path) -> None:
    if not wheel.is_file() or sha256_bytes(wheel.read_bytes()) != MUTMUT_WHEEL_SHA256:
        fail("mutmut wheel SHA-256 mismatch")
    if importlib.metadata.version("mutmut") != MUTMUT_VERSION:
        fail("installed mutmut version is not 3.7.0")


def publish_diagnostic(output: Path, error: Exception) -> None:
    """Persist an explicitly incomplete artifact when a started run cannot export evidence."""
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".tmp")
    staging.write_text(
        json.dumps({"schema_version": 3, "state": "incomplete", "error": str(error)}) + "\n",
        encoding="utf-8",
    )
    staging.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mutants-root", type=Path, default=Path("mutants"))
    parser.add_argument("--logs-dir", type=Path, default=Path("quality/mutation-logs"))
    parser.add_argument("--mutmut-wheel", type=Path)
    args = parser.parse_args()
    if sys.platform == "win32" or platform.system() == "Windows":
        print(
            "MUTATION RUN REFUSED: mutmut 3.7.0 requires POSIX/fork; use GitHub Linux.",
            file=sys.stderr,
        )
        return 2
    root = args.project_root.resolve()
    cache = (root / args.mutants_root).resolve()
    logs = (root / args.logs_dir).resolve()
    wheel = args.mutmut_wheel.resolve() if args.mutmut_wheel else None
    baseline: dict[str, Any] | None = None
    mutation: dict[str, Any] | None = None
    started = False
    try:
        if wheel is None:
            fail("--mutmut-wheel is required")
        mutation_manifest(root)
        mutation_config(root)
        reject_mutation_pragmas(root)
        commit = ensure_clean_tree(root)
        verify_mutmut(wheel)
        records, pre_digest = input_digest(root)
        baseline = run_logged(
            ["python", "-m", "pytest", "tests"],
            [sys.executable, "-m", "pytest", "tests"],
            root,
            logs / "baseline.log",
            logs,
        )
        started = True
        if baseline["state"] == "passed":
            mutation = run_logged(
                ["mutmut", "run"], ["mutmut", "run"], root, logs / "mutmut.log", logs
            )
        else:
            mutation = {"state": "not_run", "command": ["mutmut", "run"]}
        post_digest = input_digest(root)[1]
        export(
            root,
            args.output.resolve(),
            commit,
            wheel,
            baseline,
            mutation,
            cache,
            records,
            pre_digest,
            post_digest,
        )
        return 0 if baseline["state"] == mutation["state"] == "passed" else 1
    except (OSError, RuntimeError, ValueError, importlib.metadata.PackageNotFoundError) as error:
        if started:
            try:
                publish_diagnostic(args.output.resolve(), error)
            except OSError:
                pass
        print(f"MUTATION RUN FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
