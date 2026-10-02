"""Fail closed for retained v1 retry/cancel/resume/replay/kill scale evidence."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_SHA = re.compile(r"^[0-9a-f]{40}$")
_PLATFORMS = frozenset({"linux", "macos", "windows"})
_ROWS = frozenset({10_000, 100_000, 1_000_000})
_SCENARIOS = frozenset(
    {
        "complete",
        "retry_resume",
        "cancel",
        "source_mutated",
        "chunk_replay",
        "process_killed",
    }
)
SCHEMA_VERSION = 2
_RAW_KIND = "v1-execution-raw"
# Scenarios that stop part-way and therefore must record a nonzero partial checkpoint.
_PARTIAL = frozenset({"retry_resume", "cancel", "source_mutated", "process_killed"})
# Scenarios whose final cursor is the last row, and whose result digest must therefore exist.
RESULT_SCENARIOS = frozenset({"complete", "retry_resume", "chunk_replay", "process_killed"})
_TWO_ATTEMPTS = frozenset({"retry_resume", "source_mutated", "chunk_replay", "process_killed"})
_REFUSED_CODE = "SOURCE_REVISION_MISMATCH"
_KILL_ATTEMPT_LIMIT = 3


def _valid_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _valid_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def validate(payload: object, expected_sha: str) -> list[str]:
    """Return schema and candidate-binding violations without accepting partial evidence."""

    errors: list[str] = []
    if _SHA.fullmatch(expected_sha) is None:
        return ["expected candidate SHA is invalid"]
    if not isinstance(payload, dict):
        return ["evidence root must be an object"]
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported execution evidence schema")
    if payload.get("candidate_git_sha") != expected_sha:
        errors.append("evidence candidate SHA does not match the reviewed candidate")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return [*errors, "evidence cells must be a list"]
    expected = {
        (platform, rows, scenario)
        for platform in _PLATFORMS
        for rows in _ROWS
        for scenario in _SCENARIOS
    }
    observed: set[tuple[object, object, object]] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"cell {index} must be an object")
            continue
        key = (cell.get("platform"), cell.get("rows"), cell.get("scenario"))
        if key in observed:
            errors.append(f"duplicate evidence cell: {key!r}")
        observed.add(key)
        if cell.get("candidate_git_sha") != expected_sha:
            errors.append(f"cell {index} is not bound to the reviewed candidate")
        if not _valid_positive_int(cell.get("attempt_count")):
            errors.append(f"cell {index} has no measured attempt count")
        if not _valid_positive_int(cell.get("process_peak_rss_bytes")):
            errors.append(f"cell {index} has no measured process peak RSS")
        if (
            cell.get("scenario") == "retry_resume"
            and cell.get("canonical_result_equal") is not True
        ):
            errors.append(f"cell {index} lacks retry/resume equality evidence")
        if cell.get("scenario") == "cancel" and cell.get("resources_released") is not True:
            errors.append(f"cell {index} lacks cancellation cleanup evidence")
        if (
            cell.get("scenario") in {"source_mutated", "chunk_replay"}
            and cell.get("checkpoint_unchanged") is not True
        ):
            errors.append(f"cell {index} lacks unchanged-checkpoint evidence")
        if (
            cell.get("scenario") == "source_mutated"
            and cell.get("result_code") != _REFUSED_CODE
        ):
            errors.append(f"cell {index} lacks source-mutation refusal evidence")
        if (
            cell.get("scenario") in {"chunk_replay", "process_killed"}
            and cell.get("canonical_result_equal") is not True
        ):
            errors.append(f"cell {index} lacks baseline equality evidence")
    if observed != expected:
        errors.append("execution evidence matrix is incomplete or contains unknown cells")
    return errors


def _source_mutated_errors(cell: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    if not (
        cell.get("result_code") == _REFUSED_CODE
        and cell.get("rebound_revision_code") == _REFUSED_CODE
        and cell.get("retry_initial_code") == "CANCELLED"
        and cell.get("cancellation_observed") is True
    ):
        errors.append(f"raw cell {index} lacks source-mutation refusal facts")
    mutated = cell.get("mutated_source_sha256")
    if not _valid_sha256(mutated) or mutated == cell.get("source_sha256"):
        errors.append(f"raw cell {index} lacks a distinct mutated-source digest")
    errors.extend(_unchanged_checkpoint_errors(cell, index))
    return errors


def _chunk_replay_errors(cell: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    if not (
        cell.get("result_code") == "COMPLETE"
        and cell.get("canonical_result_equal") is True
        and _valid_positive_int(cell.get("replayed_chunk_count"))
    ):
        errors.append(f"raw cell {index} lacks chunk-replay facts")
    errors.extend(_unchanged_checkpoint_errors(cell, index))
    return errors


def _process_killed_errors(cell: dict[str, Any], index: int) -> list[str]:
    errors: list[str] = []
    exit_code = cell.get("child_exit_code")
    kill_attempts = cell.get("kill_attempts")
    if not (
        cell.get("result_code") == "COMPLETE"
        and cell.get("retry_initial_code") == "PROCESS_TERMINATED"
        and cell.get("canonical_result_equal") is True
        and cell.get("cancellation_observed") is False
        and isinstance(exit_code, int)
        and not isinstance(exit_code, bool)
        and exit_code != 0
        and isinstance(kill_attempts, int)
        and not isinstance(kill_attempts, bool)
        and 1 <= kill_attempts <= _KILL_ATTEMPT_LIMIT
    ):
        errors.append(f"raw cell {index} lacks process-termination facts")
    return errors


def _unchanged_checkpoint_errors(cell: dict[str, Any], index: int) -> list[str]:
    before, after = cell.get("generation_before"), cell.get("generation_after")
    if not (
        cell.get("checkpoint_unchanged") is True
        and _valid_nonnegative_int(before)
        and before == after
    ):
        return [f"raw cell {index} lacks unchanged-checkpoint facts"]
    return []


def validate_raw_fragment(payload: object, expected_sha: str) -> list[str]:
    """Validate one actually executed platform fragment before matrix assembly.

    This stricter schema keeps a collector from turning an arbitrary JSON
    object into retained-looking evidence.  It deliberately accepts only one
    platform's eighteen real cells; :func:`validate` remains responsible for the
    final all-platform matrix.
    """

    errors: list[str] = []
    if _SHA.fullmatch(expected_sha) is None:
        return ["expected candidate SHA is invalid"]
    if not isinstance(payload, dict):
        return ["raw evidence root must be an object"]
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("artifact_kind") != _RAW_KIND:
        errors.append("unsupported raw execution evidence schema")
    platform = payload.get("platform")
    if platform not in _PLATFORMS:
        errors.append("raw evidence platform is invalid")
    if payload.get("candidate_git_sha") != expected_sha:
        errors.append("raw evidence candidate SHA does not match the reviewed candidate")
    if not isinstance(payload.get("host_platform"), str) or not payload["host_platform"].strip():
        errors.append("raw evidence lacks host-platform provenance")
    for field in ("python_version", "numpy_version", "veridist_version"):
        if not isinstance(payload.get(field), str) or not payload[field].strip():
            errors.append(f"raw evidence lacks {field.replace('_', '-')} provenance")
    collected_at = payload.get("collected_at")
    if not isinstance(collected_at, str) or not collected_at.endswith("Z"):
        errors.append("raw evidence lacks UTC collection provenance")
    cells = payload.get("cells")
    if not isinstance(cells, list):
        return [*errors, "raw evidence cells must be a list"]
    expected = (
        {(platform, rows, scenario) for rows in _ROWS for scenario in _SCENARIOS}
        if isinstance(platform, str)
        else set()
    )
    observed: set[tuple[object, object, object]] = set()
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            errors.append(f"raw cell {index} must be an object")
            continue
        key = (cell.get("platform"), cell.get("rows"), cell.get("scenario"))
        if not (isinstance(key[0], str) and isinstance(key[1], int) and isinstance(key[2], str)):
            errors.append(f"raw cell {index} has an invalid matrix key")
            continue
        if key in observed:
            errors.append(f"duplicate raw evidence cell: {key!r}")
        observed.add(key)
        if cell.get("candidate_git_sha") != expected_sha:
            errors.append(f"raw cell {index} is not bound to the reviewed candidate")
        if not _valid_positive_int(cell.get("source_bytes")):
            errors.append(f"raw cell {index} lacks a measured source size")
        source_sha = cell.get("source_sha256")
        if not _valid_sha256(source_sha):
            errors.append(f"raw cell {index} lacks a deterministic source digest")
        attempt_count = cell.get("attempt_count")
        scenario = cell.get("scenario")
        required_attempts = 2 if scenario in _TWO_ATTEMPTS else 1
        if attempt_count != required_attempts:
            errors.append(f"raw cell {index} has an invalid observed attempt count")
        if not _valid_positive_int(cell.get("process_peak_rss_bytes")):
            errors.append(f"raw cell {index} has no measured process peak RSS")
        rows = cell.get("rows")
        interrupted_cursor = cell.get("interrupted_cursor")
        final_cursor = cell.get("final_cursor")
        if scenario in _PARTIAL and not (
            isinstance(rows, int)
            and isinstance(interrupted_cursor, int)
            and 0 < interrupted_cursor < rows
        ):
            errors.append(f"raw cell {index} lacks a nonzero partial checkpoint")
        if scenario in RESULT_SCENARIOS and final_cursor != rows:
            errors.append(f"raw cell {index} lacks a complete final cursor")
        if scenario in {"cancel", "source_mutated"} and final_cursor != interrupted_cursor:
            errors.append(f"raw cell {index} changed cursor after cancellation")
        result_sha = cell.get("result_sha256")
        if scenario in RESULT_SCENARIOS and not _valid_sha256(result_sha):
            errors.append(f"raw cell {index} lacks a deterministic result digest")
        if scenario in {"cancel", "source_mutated"} and result_sha is not None:
            errors.append(f"raw cell {index} reports a result after cancellation")
        if scenario == "complete" and cell.get("result_code") != "COMPLETE":
            errors.append(f"raw cell {index} lacks a complete result")
        if scenario == "retry_resume" and not (
            cell.get("result_code") == "COMPLETE"
            and cell.get("retry_initial_code") == "CANCELLED"
            and cell.get("cancellation_observed") is True
            and cell.get("canonical_result_equal") is True
        ):
            errors.append(f"raw cell {index} lacks retry/resume facts")
        if scenario == "cancel" and not (
            cell.get("result_code") == "CANCELLED"
            and cell.get("cancellation_observed") is True
            and cell.get("resources_released") is True
        ):
            errors.append(f"raw cell {index} lacks cancellation cleanup facts")
        if scenario == "source_mutated":
            errors.extend(_source_mutated_errors(cell, index))
        if scenario == "chunk_replay":
            errors.extend(_chunk_replay_errors(cell, index))
        if scenario == "process_killed":
            errors.extend(_process_killed_errors(cell, index))
    if observed != expected:
        errors.append("raw execution evidence matrix is incomplete or contains unknown cells")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    args = parser.parse_args()
    try:
        payload: Any = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("FAIL: execution evidence artifact is unreadable", file=sys.stderr)
        return 1
    errors = validate(payload, args.expected_git_sha)
    if errors:
        print("FAIL: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("PASS: v1 execution evidence is complete and candidate-bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
