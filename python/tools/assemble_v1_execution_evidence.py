"""Assemble validated platform execution fragments into one ephemeral matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from tools.check_v1_execution_evidence import (
        RESULT_SCENARIOS,
        SCHEMA_VERSION,
        validate,
        validate_raw_fragment,
    )
except ModuleNotFoundError:  # Direct execution from the tools directory.
    from check_v1_execution_evidence import (
        RESULT_SCENARIOS,
        SCHEMA_VERSION,
        validate,
        validate_raw_fragment,
    )


def assemble(paths: list[Path], expected_sha: str) -> tuple[dict[str, object] | None, list[str]]:
    """Return a fully checked 54-cell object only when all raw fragments verify."""

    errors: list[str] = []
    cells: list[object] = []
    seen_platforms: set[object] = set()
    for path in paths:
        try:
            payload: Any = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"raw artifact is unreadable: {path.name}")
            continue
        if not isinstance(payload, dict):
            errors.append(f"raw artifact root is invalid: {path.name}")
            continue
        platform = payload.get("platform")
        if platform in seen_platforms:
            errors.append(f"duplicate raw platform artifact: {platform!r}")
        seen_platforms.add(platform)
        errors.extend(validate_raw_fragment(payload, expected_sha))
        fragment_cells = payload.get("cells")
        if isinstance(fragment_cells, list):
            cells.extend(fragment_cells)
    value: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_git_sha": expected_sha,
        "cells": cells,
    }
    source_digests: dict[int, set[object]] = {}
    result_digests: dict[int, set[object]] = {}
    for cell in cells:
        if not isinstance(cell, dict) or not isinstance(cell.get("rows"), int):
            continue
        rows = cell["rows"]
        source_digests.setdefault(rows, set()).add(cell.get("source_sha256"))
        if cell.get("scenario") in RESULT_SCENARIOS:
            result_digests.setdefault(rows, set()).add(cell.get("result_sha256"))
    if any(len(digests) != 1 for digests in source_digests.values()):
        errors.append("source digests differ across platforms")
    if any(len(digests) != 1 for digests in result_digests.values()):
        errors.append("result digests differ across complete, resumed, or platform runs")
    errors.extend(validate(value, expected_sha))
    return (None, errors) if errors else (value, [])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    args = parser.parse_args()
    value, errors = assemble(args.input, args.expected_git_sha)
    if errors or value is None:
        print("FAIL: " + "; ".join(errors), file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("PASS: assembled execution evidence from executed platform fragments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
