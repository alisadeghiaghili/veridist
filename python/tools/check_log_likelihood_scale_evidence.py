"""Fail closed on retained exact-state log-likelihood scale evidence.

Timing evidence is accepted only from a single measurement worker and only when the
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
from fractions import Fraction
from pathlib import Path

ROWS = (10_000, 100_000, 1_000_000)
BUDGETS = (1_024, 8_192, 65_536)
FAMILIES = ("normal", "gamma", "weibull_min", "lognormal", "gumbel_right")
CONTRIBUTIONS = {
    "normal": "-0x1.d67f1c864beb4p-1",
    "gamma": "-0x1.0000000000000p+0",
    "weibull_min": "-0x1.3a37a020b8c22p-2",
    "lognormal": "-0x1.d67f1c864beb4p-1",
    "gumbel_right": "-0x1.0000000000000p+0",
}
SCHEMA_VERSION = "5"
METHODOLOGY = {"passes": "elapsed-untraced-then-memory-traced-v1", "rss": "process-peak-v1"}
SHA = re.compile(r"[0-9a-f]{40}")
KEYS = {"schema_version", "run", "cells", "artifact_sha256"}
CELL_KEYS = {
    "family",
    "rows",
    "chunk_size",
    "one_pass",
    "oracle",
    "actual",
    "elapsed_seconds",
    "throughput_rows_per_second",
    "memory",
}
ONE_PASS_KEYS = {"iterator_acquisitions", "observation_yields"}
ORACLE_KEYS = {"oracle_total_units", "oracle_total_units_bit_length", "bound_bits"}
ACTUAL_KEYS = {"observation_count", "total_log_likelihood", "total_log_likelihood_hex"}
MEMORY_KEYS = {"tracemalloc_peak_bytes", "rss_peak_bytes", "rss_delta_bytes"}


def _digest(value: dict[str, object]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _units(rows: int, family: str) -> int:
    contribution = float.fromhex(CONTRIBUTIONS[family])
    numerator, denominator = contribution.as_integer_ratio()
    return rows * numerator * ((1 << 1074) // denominator)


def validate(value: object, *, expected_git_sha: str, repo_root: Path) -> list[str]:
    """Return all violations; malformed or non-reproducible evidence fails closed."""

    errors: list[str] = []
    if not isinstance(value, dict) or set(value) != KEYS:
        return ["artifact schema keys invalid"]
    if value["schema_version"] != SCHEMA_VERSION or value["artifact_sha256"] != _digest(value):
        errors.append("artifact version or digest invalid")
    run = value["run"]
    if not isinstance(run, dict) or set(run) != {
        "git_sha",
        "candidate_git_sha",
        "git_dirty",
        "generator",
        "source_contract",
        "python",
        "platform",
        "measurement_workers",
        "methodology",
    }:
        return [*errors, "run schema invalid"]
    workers = run["measurement_workers"]
    if type(workers) is not int or workers <= 0:
        errors.append("measurement workers must be positive")
    elif workers != 1:
        errors.append("timing evidence requires exactly one measurement worker")
    if run["methodology"] != METHODOLOGY:
        errors.append("run measurement methodology is not supported")
    if (
        run["git_sha"] != expected_git_sha
        or not isinstance(run["git_sha"], str)
        or SHA.fullmatch(run["git_sha"]) is None
    ):
        errors.append("frozen Git SHA mismatch")
    if run["candidate_git_sha"] != expected_git_sha:
        errors.append("candidate Git SHA mismatch")
    if (
        run["git_dirty"] is not False
        or run["generator"] != "fixed-supported-family-v1"
        or run["source_contract"] != "public-iterable-data-source-v1"
    ):
        errors.append("run metadata invalid")
    if (
        not isinstance(run["python"], dict)
        or set(run["python"]) != {"implementation", "version"}
        or not all(isinstance(value, str) and value for value in run["python"].values())
        or not isinstance(run["platform"], str)
        or not run["platform"]
    ):
        errors.append("runtime platform metadata invalid")
    try:
        subprocess.run(
            ["git", "-C", str(repo_root), "merge-base", "--is-ancestor", expected_git_sha, "HEAD"],
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError):
        errors.append("frozen Git SHA is not an ancestor")
    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != len(FAMILIES) * len(ROWS) * len(BUDGETS):
        return [*errors, "full five-family 10k/100k/1m by three-chunk matrix required"]
    seen: set[tuple[str, int, int]] = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != CELL_KEYS:
            errors.append("cell schema invalid")
            continue
        family, rows, chunk = cell["family"], cell["rows"], cell["chunk_size"]
        if (
            not isinstance(family, str)
            or family not in FAMILIES
            or type(rows) is not int
            or type(chunk) is not int
            or (family, rows, chunk) in seen
        ):
            errors.append("cell key invalid or duplicate")
            continue
        seen.add((family, rows, chunk))
        if rows not in ROWS or chunk not in BUDGETS:
            errors.append("cell scale or pass facts invalid")
        one_pass, oracle, actual, memory = (
            cell["one_pass"],
            cell["oracle"],
            cell["actual"],
            cell["memory"],
        )
        if not isinstance(one_pass, dict) or set(one_pass) != ONE_PASS_KEYS:
            errors.append("one-pass measurement schema invalid")
            continue
        if one_pass != {"iterator_acquisitions": 1, "observation_yields": rows}:
            errors.append("actual one-pass traversal mismatch")
        if not isinstance(oracle, dict) or set(oracle) != ORACLE_KEYS:
            errors.append("oracle schema invalid")
            continue
        if oracle != {
            "oracle_total_units": _units(rows, family),
            "oracle_total_units_bit_length": abs(_units(rows, family)).bit_length(),
            "bound_bits": 2162,
        }:
            errors.append("independent exact oracle mismatch")
        expected = float(Fraction(_units(rows, family), 1 << 1074))
        if not isinstance(actual, dict) or set(actual) != ACTUAL_KEYS:
            errors.append("actual result schema invalid")
            continue
        if (
            actual["observation_count"] != rows
            or type(actual["total_log_likelihood"]) is not float
            or actual["total_log_likelihood_hex"] != expected.hex()
            or actual["total_log_likelihood"].hex() != expected.hex()
        ):
            errors.append("actual returned total does not match independent exact oracle")
        if (
            not isinstance(memory, dict)
            or set(memory) != MEMORY_KEYS
            or any(type(memory[key]) is not int or memory[key] < 0 for key in MEMORY_KEYS)
        ):
            errors.append("descriptive memory fact invalid")
        if (
            not isinstance(cell["elapsed_seconds"], int | float)
            or isinstance(cell["elapsed_seconds"], bool)
            or not math.isfinite(cell["elapsed_seconds"])
            or cell["elapsed_seconds"] <= 0
        ):
            errors.append("descriptive elapsed fact invalid")
        throughput = cell["throughput_rows_per_second"]
        if (
            not isinstance(throughput, int | float)
            or isinstance(throughput, bool)
            or not math.isfinite(throughput)
            or throughput <= 0
            or abs(throughput - rows / cell["elapsed_seconds"]) > max(1e-9, throughput * 1e-12)
        ):
            errors.append("throughput fact invalid")
    if seen != {(family, row, budget) for family in FAMILIES for row in ROWS for budget in BUDGETS}:
        errors.append("matrix coverage invalid")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        value: object = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    errors = validate(value, expected_git_sha=args.expected_git_sha, repo_root=args.repo_root)
    if errors:
        print("FAIL: log-likelihood scale evidence rejected", file=sys.stderr)
        print(*[f"- {error}" for error in errors], sep="\n", file=sys.stderr)
        return 1
    print("PASS: log-likelihood scale evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
