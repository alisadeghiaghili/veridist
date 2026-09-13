"""Classify repository paths for CI lanes and enforce aggregate gate states."""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable

_VERIDIST_PREFIXES = ("python/", "docs/adr/", "docs/evidence/", "docs/migration/")
_VERIDIST_PATHS = frozenset(
    {
        "README.md",
        "CHANGELOG.md",
        "README.fa.md",
        "README.de.md",
        "SECURITY.md",
        ".zenodo.json",
        "CITATION.cff",
        "conda-forge-recipe/meta.yaml",
        ".github/workflows/ci.yml",
        ".github/workflows/mutation.yml",
        ".github/workflows/scale-evidence.yml",
        ".github/workflows/veridist-release.yml",
        ".github/workflows/v1-ci.yml",
        ".github/workflows/v1-release-evidence.yml",
        ".github/workflows/pypi-publish.yml",
        "docs/adoption-and-reputation-strategy.md",
        "docs/capability-matrix.md",
        "docs/capability-matrix.fa.md",
        "docs/capability-matrix.de.md",
        "docs/competitive-evidence-policy.md",
        "docs/competitive-feature-matrix.csv",
        "docs/competitive-feature-matrix.md",
        "docs/competitive-landscape.md",
        "docs/conventions.md",
        "docs/readme-design-policy.md",
        "docs/citing-veridist.md",
        "docs/v1-readiness.md",
        "docs/v1-roadmap.md",
        "docs/v1-test-plan.md",
    }
)
_DATED_DECISION = re.compile(r"docs/decisions-\d{4}-\d{2}-\d{2}\.md")


def _validate_repository_path(path: str) -> None:
    if not path or path.startswith("/") or "\\" in path or "\0" in path:
        raise ValueError("invalid repository path")
    parts = path.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid repository path")


def _is_veridist_only(path: str) -> bool:
    return (
        path.startswith(_VERIDIST_PREFIXES)
        or path in _VERIDIST_PATHS
        or _DATED_DECISION.fullmatch(path) is not None
    )


def legacy_relevant(paths: Iterable[str]) -> bool:
    """Return whether any changed path can affect the legacy product lane."""

    found_path = False
    relevant = False
    for path in paths:
        found_path = True
        _validate_repository_path(path)
        if not _is_veridist_only(path):
            relevant = True
    return relevant or not found_path


def decode_nul_paths(payload: bytes) -> tuple[str, ...]:
    """Decode the unambiguous NUL-delimited output produced by ``git diff -z``."""

    if not payload:
        return ()
    if not payload.endswith(b"\0"):
        raise ValueError("incomplete path stream")
    encoded_paths = payload.split(b"\0")[:-1]
    if any(not path for path in encoded_paths):
        raise ValueError("empty path in stream")
    try:
        paths = tuple(path.decode("utf-8", errors="strict") for path in encoded_paths)
    except UnicodeDecodeError as error:
        raise ValueError("path stream is not UTF-8") from error
    for path in paths:
        _validate_repository_path(path)
    return paths


def legacy_gate_allows(relevant: bool, scope_result: str, test_result: str) -> bool:
    """Return whether the aggregate legacy gate represents a valid terminal state."""

    if type(relevant) is not bool or scope_result != "success":
        return False
    if relevant:
        return test_result == "success"
    return test_result == "skipped"


def _parse_boolean(value: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError("expected a boolean token")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    classify = commands.add_parser("classify", help="classify changed repository paths")
    classify.add_argument("--nul", action="store_true", required=True)

    gate = commands.add_parser("legacy-gate", help="enforce the aggregate legacy gate")
    gate.add_argument("--relevant", required=True)
    gate.add_argument("--scope-result", required=True)
    gate.add_argument("--test-result", required=True)
    return parser


def _main() -> int:
    arguments = _build_parser().parse_args()
    if arguments.command == "classify":
        paths = decode_nul_paths(sys.stdin.buffer.read())
        print("true" if legacy_relevant(paths) else "false")
        return 0

    relevant = _parse_boolean(arguments.relevant)
    if legacy_gate_allows(relevant, arguments.scope_result, arguments.test_result):
        return 0
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(_main())
    except ValueError:
        print("invalid CI scope input", file=sys.stderr)
        raise SystemExit(2) from None
