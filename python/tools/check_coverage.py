"""Deterministically enforce Veridist's v1 coverage contract from coverage JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

REQUIRED_SUMMARY_METRICS = {
    "num_statements",
    "covered_lines",
    "num_branches",
    "covered_branches",
}
REQUIRED_EXCEPTION_FIELDS = {"path", "owner", "reason", "expiry", "adr"}
ADR_PATTERN = re.compile(r"ADR-\d{4}$")
ADR_INDEX_PATH = Path("quality") / "adr-index.json"
PRAGMA_PATTERN = re.compile(r"#\s*pragma:\s*no\s*(cover|branch)", re.IGNORECASE)
PRAGMA_KINDS = ("no_branch", "no_cover")


@dataclass(frozen=True)
class Totals:
    """Line and branch totals extracted from a coverage.py summary."""

    statements: int
    covered_lines: int
    branches: int
    covered_branches: int

    @property
    def line_rate(self) -> float:
        return 1.0 if self.statements == 0 else self.covered_lines / self.statements

    @property
    def branch_rate(self) -> float:
        return 1.0 if self.branches == 0 else self.covered_branches / self.branches

    def plus(self, other: Totals) -> Totals:
        return Totals(
            self.statements + other.statements,
            self.covered_lines + other.covered_lines,
            self.branches + other.branches,
            self.covered_branches + other.covered_branches,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--coverage-json", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {label}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label} must be a JSON object")
        return {}
    return value


def _normalise_path(value: object, project_root: Path) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    try:
        resolved = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()
        return resolved.relative_to(project_root.resolve()).as_posix()
    except (OSError, ValueError):
        return None


def _integer_metric(summary: dict[str, Any], metric: str, path: str, errors: list[str]) -> int:
    value = summary.get(metric)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        errors.append(f"missing metric {metric!r} for {path}")
        return 0
    return value


def _load_adr_index(project_root: Path, errors: list[str]) -> frozenset[str]:
    """Return the ADR ids published in ``quality/adr-index.json``.

    The index maps ``ADR-NNNN`` to a non-empty title and carries nothing else;
    a missing or malformed index yields no ids, so every exception that names an
    ADR is rejected.
    """

    index = _load_json(project_root / ADR_INDEX_PATH, "ADR index", errors)
    valid = True
    for adr, title in index.items():
        if not ADR_PATTERN.fullmatch(adr) or not isinstance(title, str) or not title.strip():
            errors.append(f"ADR index entry is invalid: {adr!r}")
            valid = False
    return frozenset(index) if valid else frozenset()


def _validate_exception(
    exception: object,
    production_files: set[str],
    seen: set[str],
    errors: list[str],
    *,
    adr_ids: frozenset[str],
    today: date,
) -> str | None:
    if not isinstance(exception, dict):
        errors.append("exception must be an object")
        return None
    missing = REQUIRED_EXCEPTION_FIELDS.difference(exception)
    if missing:
        errors.append(f"exception missing required fields: {', '.join(sorted(missing))}")
        return None
    all_fields_are_non_empty_strings = all(
        isinstance(exception[field], str) and exception[field].strip()
        for field in REQUIRED_EXCEPTION_FIELDS
    )
    if not all_fields_are_non_empty_strings:
        errors.append("exception fields must be non-empty strings")
        return None
    path = exception["path"]
    assert isinstance(path, str)
    if path not in production_files:
        errors.append(f"exception path is not a production file: {path}")
        return None
    if path in seen:
        errors.append(f"exception is duplicated for {path}")
        return None
    adr = exception["adr"]
    expiry = exception["expiry"]
    assert isinstance(adr, str) and isinstance(expiry, str)
    if not ADR_PATTERN.fullmatch(adr):
        errors.append(f"exception ADR is invalid for {path}")
        return None
    if adr not in adr_ids:
        errors.append(f"exception ADR does not exist for {path}: {adr}")
        return None
    try:
        expiry_date = date.fromisoformat(expiry)
    except ValueError:
        errors.append(f"exception expiry is invalid for {path}")
        return None
    if expiry_date < today:
        errors.append(f"exception expired for {path} on {expiry_date.isoformat()}")
        return None
    seen.add(path)
    return path


def count_pragmas(source: str) -> dict[str, int]:
    """Count coverage-exclusion pragma comments in source text, keyed by kind."""

    counts = {kind: 0 for kind in PRAGMA_KINDS}
    for match in PRAGMA_PATTERN.finditer(source):
        counts["no_" + match.group(1).lower()] += 1
    return counts


def _validate_pragma_budget(
    project_root: Path,
    budget_value: object,
    production_files: set[str],
    discovered: set[str],
    errors: list[str],
) -> None:
    """Fail when a file holds more coverage pragmas than its recorded budget allows.

    The budget can only shrink: a file may carry fewer pragmas than recorded
    (an under-budget file passes), but any increase, and any pragma in a file
    the budget does not list, is rejected.
    """

    if not isinstance(budget_value, dict):
        errors.append("manifest pragma_budget must be an object")
        return
    budgets: dict[str, dict[str, int]] = {}
    for path, entry in budget_value.items():
        if path not in production_files:
            errors.append(f"pragma budget names an unlisted file: {path}")
            continue
        valid = (
            isinstance(entry, dict)
            and set(entry) == set(PRAGMA_KINDS)
            and all(
                isinstance(entry[kind], int)
                and not isinstance(entry[kind], bool)
                and entry[kind] >= 0
                for kind in PRAGMA_KINDS
            )
        )
        if not valid:
            errors.append(f"invalid pragma budget for {path}")
            continue
        budgets[path] = entry
    for path in sorted(discovered):
        try:
            counts = count_pragmas((project_root / path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            errors.append(f"cannot read {path} for the pragma budget: {exc}")
            continue
        budget = budgets.get(path)
        if budget is None:
            if any(counts.values()):
                errors.append(f"unlisted file contains coverage pragmas: {path}")
            continue
        for kind in PRAGMA_KINDS:
            if counts[kind] > budget[kind]:
                errors.append(
                    f"pragma budget exceeded for {path}: "
                    f"{kind.replace('_', ' ')} {counts[kind]} > {budget[kind]}"
                )


def _format_rate(rate: float) -> str:
    return f"{rate * 100:.2f}%"


def validate(
    project_root: Path,
    manifest_path: Path,
    coverage_path: Path,
    *,
    today: date | None = None,
) -> list[str]:
    """Return contract violations, or an empty list when evidence satisfies the gate.

    ``today`` defaults to :func:`datetime.date.today` and is compared against
    every accepted exception's ``expiry``: an exception whose expiry has
    passed is rejected rather than silently honored forever. Each exception's
    ``adr`` must also be listed in ``quality/adr-index.json`` under
    ``project_root``.
    """

    if today is None:
        today = date.today()

    errors: list[str] = []
    manifest = _load_json(manifest_path, "manifest", errors)
    coverage = _load_json(coverage_path, "coverage JSON", errors)
    required_manifest = {
        "production_root",
        "production_files",
        "critical_modules",
        "expected_denominators",
        "accepted_exceptions",
        "pragma_budget",
    }
    for field in sorted(required_manifest.difference(manifest)):
        errors.append(f"manifest missing required field: {field}")
    if errors:
        return errors

    production_root = _normalise_path(manifest["production_root"], project_root)
    listed_value = manifest["production_files"]
    expected_value = manifest["expected_denominators"]
    critical_value = manifest["critical_modules"]
    exceptions_value = manifest["accepted_exceptions"]
    if production_root is None:
        errors.append("manifest production_root is invalid")
        return errors
    listed_strings = isinstance(listed_value, list) and all(
        isinstance(item, str) for item in listed_value
    )
    if not listed_strings:
        errors.append("manifest production_files must be a list of paths")
        return errors
    if len(listed_value) != len(set(listed_value)):
        errors.append("manifest production_files contains duplicates")
    production_files = set(listed_value)
    if not isinstance(expected_value, dict):
        errors.append("manifest expected_denominators must be an object")
        return errors
    valid_critical_modules = isinstance(critical_value, list) and all(
        isinstance(item, str) and item for item in critical_value
    )
    if not valid_critical_modules:
        errors.append("manifest critical_modules must be a list of names")
        return errors
    if not isinstance(exceptions_value, list):
        errors.append("manifest accepted_exceptions must be a list")
        return errors

    root_path = project_root / production_root
    if not root_path.is_dir():
        errors.append(f"production root does not exist: {production_root}")
        return errors
    discovered = {
        path.relative_to(project_root).as_posix()
        for path in root_path.rglob("*.py")
        if "__pycache__" not in path.parts
    }
    for path in sorted(discovered.difference(production_files)):
        errors.append(f"unlisted production file: {path}")
    for path in sorted(production_files.difference(discovered)):
        errors.append(f"listed production file does not exist: {path}")

    _validate_pragma_budget(
        project_root, manifest["pragma_budget"], production_files, discovered, errors
    )

    adr_ids = _load_adr_index(project_root, errors) if exceptions_value else frozenset()
    exception_paths: set[str] = set()
    for exception in exceptions_value:
        _validate_exception(
            exception,
            production_files,
            exception_paths,
            errors,
            adr_ids=adr_ids,
            today=today,
        )

    coverage_files_value = coverage.get("files")
    if not isinstance(coverage_files_value, dict):
        errors.append("coverage JSON missing files object")
        return errors
    coverage_files: dict[str, dict[str, Any]] = {}
    for raw_path, record in coverage_files_value.items():
        path = _normalise_path(raw_path, project_root)
        if path is None:
            errors.append(f"coverage path is invalid: {raw_path!r}")
            continue
        if not isinstance(record, dict):
            errors.append(f"coverage record is invalid for {path}")
            continue
        coverage_files[path] = record
    for path in sorted(set(coverage_files).intersection(discovered).difference(production_files)):
        errors.append(f"unlisted production file: {path}")
    for path in sorted(production_files.difference(coverage_files)):
        errors.append(f"missing coverage file: {path}")

    totals_by_file: dict[str, Totals] = {}
    for path in sorted(production_files.intersection(coverage_files)):
        record = coverage_files[path]
        summary = record.get("summary")
        if not isinstance(summary, dict):
            errors.append(f"missing metric summary for {path}")
            continue
        values = {
            metric: _integer_metric(summary, metric, path, errors)
            for metric in REQUIRED_SUMMARY_METRICS
        }
        if values["covered_lines"] > values["num_statements"]:
            errors.append(f"invalid covered line count for {path}")
        if values["covered_branches"] > values["num_branches"]:
            errors.append(f"invalid covered branch count for {path}")
        expected = expected_value.get(path)
        if not isinstance(expected, dict):
            errors.append(f"missing expected denominators for {path}")
            continue
        metrics = (("statements", "num_statements"), ("branches", "num_branches"))
        for expected_metric, actual_metric in metrics:
            expected_number = expected.get(expected_metric)
            is_non_negative_integer = (
                not isinstance(expected_number, bool)
                and isinstance(expected_number, int)
                and expected_number >= 0
            )
            if not is_non_negative_integer:
                errors.append(f"invalid expected denominators for {path}")
            elif expected_number != values[actual_metric]:
                errors.append(f"denominator drift for {path}: {actual_metric}")
        totals_by_file[path] = Totals(
            values["num_statements"],
            values["covered_lines"],
            values["num_branches"],
            values["covered_branches"],
        )
    for path in sorted(set(expected_value).difference(production_files)):
        errors.append(f"expected denominators name an unlisted file: {path}")

    if errors:
        return errors
    all_totals = Totals(0, 0, 0, 0)
    for totals in totals_by_file.values():
        all_totals = all_totals.plus(totals)
    if all_totals.line_rate < 0.95:
        errors.append(
            f"global line coverage {_format_rate(all_totals.line_rate)} is below 95.00%"
        )
    if all_totals.branch_rate < 0.95:
        errors.append(
            f"global branch coverage {_format_rate(all_totals.branch_rate)} is below 95.00%"
        )

    for module in critical_value:
        prefix = f"{production_root}/{module}/"
        module_files = [
            totals for path, totals in totals_by_file.items() if path.startswith(prefix)
        ]
        if not module_files:
            errors.append(f"missing critical module report: {module}")
            continue
        module_totals = Totals(0, 0, 0, 0)
        for totals in module_files:
            module_totals = module_totals.plus(totals)
        if module_totals.line_rate < 0.98:
            errors.append(
                f"critical line coverage for {module} is {_format_rate(module_totals.line_rate)}"
            )
        if module_totals.branch_rate < 0.98:
            errors.append(
                "critical branch coverage for "
                f"{module} is {_format_rate(module_totals.branch_rate)}"
            )

    for path, totals in sorted(totals_by_file.items()):
        if path in exception_paths:
            continue
        if totals.line_rate < 0.90:
            errors.append(f"file line coverage for {path} is {_format_rate(totals.line_rate)}")
        if totals.branch_rate < 0.90:
            errors.append(f"file branch coverage for {path} is {_format_rate(totals.branch_rate)}")
    return errors


def main() -> int:
    """Run the coverage contract checker as a command-line program."""

    args = _parse_args()
    errors = validate(
        args.project_root.resolve(), args.manifest.resolve(), args.coverage_json.resolve()
    )
    if errors:
        print("FAIL: coverage gate rejected", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("PASS: coverage gate satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
