"""Fail closed on retained SCALE-CSV-EXP evidence artifacts.

The checker recomputes deterministic fixture facts without importing the
production fitter. Acceptance is tied to an explicit frozen Git revision.

Timing evidence is accepted only from a single measurement worker, and only when the
artifact declares the measurement methodology this checker understands (elapsed time
from an untraced pass, memory from a separate traced pass). ``artifact_sha256`` is an
integrity digest of the canonical JSON body: it detects accidental damage but is not a
signature and does not prove who produced the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from functools import cache
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "3"
FULL_ROWS = (10_000, 100_000, 1_000_000)
FULL_BUDGETS = (32_768, 65_536, 131_072)
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")

ARTIFACT_KEYS = {
    "schema_version",
    "run",
    "generator",
    "cells",
    "operation_evidence",
    "artifact_sha256",
}
RUN_KEYS = {
    "git_sha",
    "candidate_git_sha",
    "git_dirty",
    "utc_started",
    "python",
    "platform",
    "measurement_workers",
    "timing",
    "methodology",
}
TIMING_KEYS = {"clock", "preflight"}
METHODOLOGY_KEYS = {"passes", "rss"}
METHODOLOGY = {"passes": "elapsed-untraced-then-memory-traced-v1", "rss": "process-peak-v1"}
PYTHON_KEYS = {"implementation", "version"}
GENERATOR_KEYS = {"formula_version", "temporary_root"}
CELL_KEYS = {
    "rows",
    "chunk_bytes",
    "max_inflight_bytes",
    "source",
    "observed",
    "fit",
    "memory",
    "elapsed_seconds",
    "throughput_rows_per_second",
}
SOURCE_KEYS = {"bytes", "sha256"}
OBSERVED_KEYS = {
    "actual_pass_count",
    "max_passes",
    "accepted_chunk_count",
    "processed_row_count",
    "peak_inflight_bytes",
    "largest_retained_chunk_bytes",
    "backpressure_event_count",
}
FIT_KEYS = {
    "observation_count",
    "event_count",
    "total_time",
    "rate",
    "expected_event_count",
    "expected_total_time",
    "expected_rate",
    "absolute_rate_error",
    "relative_rate_error",
}
MEMORY_KEYS = {"tracemalloc_peak_bytes", "rss_peak_bytes", "rss_delta_bytes"}
OPERATION_KEYS = {"rows", "accepted_chunks"}
# Formula v1 emits the fitted rate as a JSON binary64 value. Ten IEEE-754
# double-precision epsilons is the frozen allowance for that conversion and
# the reducer's binary64 accumulation; it is not a performance tolerance.
FORMULA_V1_RATE_RELATIVE_TOLERANCE = Decimal("2.220446049250313e-15")


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def _integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _contains_path(value: object) -> bool:
    if isinstance(value, str):
        return "/" in value or "\\" in value or re.search(r"[A-Za-z]:", value) is not None
    if isinstance(value, list):
        return any(_contains_path(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_path(key) or _contains_path(item) for key, item in value.items())
    return False


def _digest(value: dict[str, object]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _exact_keys(value: object, expected: set[str], label: str, errors: list[str]) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return False
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        errors.append(f"{label} schema keys invalid: missing={missing}, extra={extra}")
        return False
    return True


@cache
def _fixture_facts(rows: int, formula_version: str) -> tuple[int, Decimal, int, str]:
    """Independently reproduce the v1 bytes and Decimal sufficient statistics."""

    if formula_version != "1":
        raise ValueError("unsupported deterministic generator formula")
    digest = hashlib.sha256()
    header = b"time,event_observed\n"
    digest.update(header)
    byte_count, events, total = len(header), 0, Decimal(0)
    for index in range(rows):
        time_value = Decimal((index % 997) + 1) / Decimal(1000)
        event = 0 if index % 3 == 0 else 1
        encoded = f"{time_value:f},{event}\n".encode()
        digest.update(encoded)
        byte_count += len(encoded)
        events += event
        total += time_value
    return events, total, byte_count, digest.hexdigest()


def _git_commit_is_ancestor(repo_root: Path, sha: str) -> bool:
    """Return whether ``sha`` is an existing ancestor of the bound repository."""

    if not (repo_root / ".git").exists():
        return False
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "-e", f"{sha}^{{commit}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", sha, "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return True


def _cell_key(cell: dict[str, Any]) -> tuple[int, int] | None:
    rows, chunk = cell.get("rows"), cell.get("chunk_bytes")
    if not _integer(rows) or not _integer(chunk) or chunk == 0:
        return None
    return rows, chunk


def validate(
    value: object, *, expected_git_sha: str, smoke: bool = False, repo_root: Path | None = None
) -> list[str]:
    """Return all violations; malformed evidence never partly passes."""

    errors: list[str] = []
    if GIT_SHA.fullmatch(expected_git_sha) is None:
        return ["expected git SHA is invalid"]
    if not _exact_keys(value, ARTIFACT_KEYS, "artifact", errors):
        return errors
    assert isinstance(value, dict)
    schema_version = value["schema_version"]
    if schema_version != SCHEMA_VERSION:
        errors.append("current evidence requires schema version 3")
    if not isinstance(value["artifact_sha256"], str) or value["artifact_sha256"] != _digest(value):
        errors.append("artifact digest mismatch")
    if _contains_path(value):
        errors.append("path leaked into evidence artifact")
    run = value["run"]
    if not _exact_keys(run, RUN_KEYS, "run", errors):
        return errors
    assert isinstance(run, dict)
    actual_sha = run["git_sha"]
    if (
        not isinstance(actual_sha, str)
        or GIT_SHA.fullmatch(actual_sha) is None
        or actual_sha != expected_git_sha
    ):
        errors.append("run git SHA does not match frozen expected SHA")
    if run["candidate_git_sha"] != expected_git_sha:
        errors.append("candidate git SHA does not match frozen expected SHA")
    if run["git_dirty"] is not False:
        errors.append("run is dirty")
    if repo_root is None:
        errors.append("repository binding is required")
    elif not _git_commit_is_ancestor(repo_root, expected_git_sha):
        errors.append("expected git SHA is not an existing ancestor of repository HEAD")
    if not _exact_keys(run["python"], PYTHON_KEYS, "run python", errors):
        return errors
    python = run["python"]
    assert isinstance(python, dict)
    if python["implementation"] != "CPython" or not isinstance(python["version"], str):
        errors.append("run Python metadata is invalid")
    if not isinstance(run["utc_started"], str) or not run["utc_started"].endswith("Z"):
        errors.append("run UTC timestamp is invalid")
    if not _integer(run["measurement_workers"]) or run["measurement_workers"] <= 0:
        errors.append("measurement workers must be positive")
    elif run["measurement_workers"] != 1:
        errors.append("timing evidence requires exactly one measurement worker")
    timing = run["timing"]
    if not _exact_keys(timing, TIMING_KEYS, "run timing", errors):
        return errors
    assert isinstance(timing, dict)
    if timing["clock"] != "time.time_ns" or timing["preflight"] != "paired-wall-monotonic-v1":
        errors.append("run timing provenance is invalid")
    methodology = run["methodology"]
    if not _exact_keys(methodology, METHODOLOGY_KEYS, "run methodology", errors):
        return errors
    if methodology != METHODOLOGY:
        errors.append("run measurement methodology is not supported")
    generator = value["generator"]
    if not _exact_keys(generator, GENERATOR_KEYS, "generator", errors):
        return errors
    assert isinstance(generator, dict)
    formula_version = generator["formula_version"]
    if formula_version != "1" or generator["temporary_root"] != "redacted":
        errors.append("generator contract is invalid")
    cells = value["cells"]
    if not isinstance(cells, list):
        return [*errors, "cells must be a list"]
    if not smoke and len(cells) != 9:
        errors.append("full matrix must contain exactly nine cells")
    if smoke and len(cells) != 3:
        errors.append("smoke matrix must contain exactly three cells")
    keys: list[tuple[int, int]] = []
    cells_by_key: dict[tuple[int, int], dict[str, Any]] = {}
    for index, cell in enumerate(cells):
        if not _exact_keys(cell, CELL_KEYS, f"cell {index}", errors):
            continue
        assert isinstance(cell, dict)
        key = _cell_key(cell)
        if key is None:
            errors.append(f"cell {index} has invalid rows/chunk")
            continue
        keys.append(key)
        cells_by_key[key] = cell
        rows, chunk = key
        if cell["max_inflight_bytes"] != chunk:
            errors.append(f"cell {key} max inflight must equal configured chunk")
        source = cell["source"]
        if not _exact_keys(source, SOURCE_KEYS, f"cell {key} source", errors):
            continue
        assert isinstance(source, dict)
        try:
            expected_events, expected_total, expected_bytes, expected_hash = _fixture_facts(
                rows, str(formula_version)
            )
        except ValueError:
            errors.append(f"cell {key} generator formula is unsupported")
            continue
        if source["bytes"] != expected_bytes or source["sha256"] != expected_hash:
            errors.append(f"cell {key} source bytes or SHA-256 mismatch")
        observed = cell["observed"]
        if not _exact_keys(observed, OBSERVED_KEYS, f"cell {key} observed", errors):
            continue
        assert isinstance(observed, dict)
        if observed["actual_pass_count"] != 1 or observed["max_passes"] != 1:
            errors.append(f"cell {key} pass count must be exactly one")
        if observed["processed_row_count"] != rows:
            errors.append(f"cell {key} observed rows mismatch")
        for metric in (
            "accepted_chunk_count",
            "peak_inflight_bytes",
            "largest_retained_chunk_bytes",
            "backpressure_event_count",
        ):
            if not _integer(observed[metric]):
                errors.append(f"cell {key} {metric} invalid")
        if observed["accepted_chunk_count"] == 0:
            errors.append(f"cell {key} accepted_chunk_count must be positive")
        if (
            isinstance(observed["peak_inflight_bytes"], int)
            and observed["peak_inflight_bytes"] > chunk
        ):
            errors.append(f"cell {key} inflight bound exceeded")
        if (
            isinstance(observed["largest_retained_chunk_bytes"], int)
            and observed["largest_retained_chunk_bytes"] > chunk
        ):
            errors.append(f"cell {key} retained chunk bound exceeded")
        fit = cell["fit"]
        if not _exact_keys(fit, FIT_KEYS, f"cell {key} fit", errors):
            continue
        assert isinstance(fit, dict)
        if (
            fit["observation_count"] != rows
            or fit["event_count"] != expected_events
            or fit["expected_event_count"] != expected_events
        ):
            errors.append(f"cell {key} fit observation/event count mismatch")
        try:
            actual_total = Decimal(str(fit["total_time"]))
            stored_expected_total = Decimal(str(fit["expected_total_time"]))
            actual_rate = Decimal(str(fit["rate"]))
            stored_expected_rate = Decimal(str(fit["expected_rate"]))
            stored_absolute = Decimal(str(fit["absolute_rate_error"]))
            stored_relative = Decimal(str(fit["relative_rate_error"]))
            recomputed_rate = Decimal(expected_events) / expected_total
            recomputed_absolute = abs(actual_rate - recomputed_rate)
            recomputed_relative = recomputed_absolute / abs(recomputed_rate)
            if actual_total != expected_total or stored_expected_total != expected_total:
                errors.append(f"cell {key} total time mismatch")
            if stored_expected_rate != recomputed_rate:
                errors.append(f"cell {key} expected rate mismatch")
            # The runner records errors as JSON floats; reproduce that deliberate
            # serialization boundary before requiring exact equality.
            serialized_absolute = Decimal(str(float(recomputed_absolute)))
            serialized_relative = Decimal(str(float(recomputed_relative)))
            if stored_absolute != serialized_absolute or stored_relative != serialized_relative:
                errors.append(f"cell {key} rate errors mismatch")
            maximum_absolute = FORMULA_V1_RATE_RELATIVE_TOLERANCE * max(
                Decimal(1), abs(recomputed_rate)
            )
            if (
                recomputed_absolute > maximum_absolute
                or recomputed_relative > FORMULA_V1_RATE_RELATIVE_TOLERANCE
            ):
                errors.append(f"cell {key} rate error exceeds frozen tolerance")
        except (InvalidOperation, ValueError, TypeError):
            errors.append(f"cell {key} fit numeric facts invalid")
        memory = cell["memory"]
        if not _exact_keys(memory, MEMORY_KEYS, f"cell {key} memory", errors):
            continue
        assert isinstance(memory, dict)
        if not _integer(memory["tracemalloc_peak_bytes"]):
            errors.append(f"cell {key} memory facts invalid")
        for metric in ("rss_peak_bytes", "rss_delta_bytes"):
            if not _integer(memory[metric]):
                errors.append(f"cell {key} {metric} invalid")
        if not _finite_nonnegative(cell["elapsed_seconds"]):
            errors.append(f"cell {key} elapsed time invalid")
        throughput = cell["throughput_rows_per_second"]
        if (
            not _finite_nonnegative(throughput)
            or throughput == 0
            or cell["elapsed_seconds"] == 0
            or abs(throughput - rows / cell["elapsed_seconds"]) > max(1e-9, throughput * 1e-12)
        ):
            errors.append(f"cell {key} throughput invalid")
    if len(keys) != len(set(keys)):
        errors.append("duplicate matrix cells")
    row_set, budget_set = {key[0] for key in keys}, {key[1] for key in keys}
    if smoke:
        if len(row_set) != 1 or len(budget_set) != 3 or len(keys) != 3:
            errors.append("smoke matrix must have one row size and exactly three chunk budgets")
    elif (
        row_set != set(FULL_ROWS)
        or budget_set != set(FULL_BUDGETS)
        or set(keys) != {(r, b) for r in FULL_ROWS for b in FULL_BUDGETS}
    ):
        errors.append("full matrix row/budget set is not exact")
    operation = value["operation_evidence"]
    if not _exact_keys(operation, OPERATION_KEYS, "operation evidence", errors):
        return errors
    assert isinstance(operation, dict)
    operation_rows = list(FULL_ROWS) if not smoke else sorted(row_set)
    if operation["rows"] != operation_rows:
        errors.append("operation evidence rows invalid")
    counts = operation["accepted_chunks"]
    if (
        not isinstance(counts, list)
        or len(counts) != len(operation_rows)
        or not all(_integer(item) and item > 0 for item in counts)
    ):
        errors.append("operation evidence counts invalid")
    elif len(budget_set) == 3:
        smallest = min(budget_set)
        for row, count in zip(operation_rows, counts, strict=True):
            cell = cells_by_key.get((row, smallest))
            if (
                cell is None
                or not isinstance(cell.get("observed"), dict)
                or count != cell["observed"].get("accepted_chunk_count")
            ):
                errors.append("operation evidence does not cross-link smallest-budget cells")
                break
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        value = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"FAIL: cannot read artifact: {error}", file=sys.stderr)
        return 1
    errors = validate(
        value, expected_git_sha=args.expected_git_sha, smoke=args.smoke, repo_root=args.repo_root
    )
    if errors:
        print("FAIL: scale evidence rejected", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("PASS: scale evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
