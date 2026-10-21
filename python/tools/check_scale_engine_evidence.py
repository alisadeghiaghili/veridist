"""Fail closed on retained evidence of the internal ``veridist.scale`` engine (schema 1).

The checker does not import the engine. It holds the frozen contract of the evidence: the row
ladder, the layouts of the data, the bounds of the pass planner, the tolerances of the references
and the 40-digit reference values of the fixed sample, and it recomputes from the artifact every
fact that can be recomputed (result digests, layout geometry, relative differences, the
goodness-of-fit evaluation error, the enclosure decision, the resume arithmetic and the sample
digest). Acceptance is tied to an explicit frozen Git revision.

Timing and memory facts are descriptive: they must be positive and finite and carry no other
threshold. Timing evidence is accepted only from a single measurement worker and only when the
artifact declares the measurement methodology this checker understands. ``artifact_sha256`` is an
integrity digest of the canonical JSON body: it detects accidental damage but is not a signature
and does not prove who produced the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
import sys
from collections.abc import Callable
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"
STANDARD_LADDER = (1_000_000, 10_000_000, 100_000_000)
TOP_RUNG = 1_000_000_000
SMOKE_MAX_ROWS = 1_000_000
SMOKE_MIN_ROWS = 1_000
PROFILE_MAX_ROWS = 100_000_000
MULTI_PASS_MAX_ROWS = 100_000_000

ONE_PASS_FAMILIES = ("exponential", "normal", "gamma", "lognormal")
MULTI_PASS_FAMILIES = ("weibull_min", "gumbel_right")
FAMILIES = ONE_PASS_FAMILIES + MULTI_PASS_FAMILIES
EXECUTORS = ("serial", "processes")
PARAMETER_NAMES = {
    "exponential": ("rate",),
    "normal": ("mu", "sigma"),
    "gamma": ("shape", "scale"),
    "lognormal": ("mu_log", "sigma_log"),
    "weibull_min": ("shape", "scale"),
    "gumbel_right": ("location", "scale"),
}
POSITIVE_PARAMETERS = {"rate", "sigma", "shape", "scale", "sigma_log"}

SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_SHA = re.compile(r"[0-9a-f]{40}")
TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")

METHODOLOGY = {
    "passes": "elapsed-untraced-then-memory-traced-v1",
    "rss": "process-peak-v1",
    "memory_process": "fresh-process-per-cell-v1",
    "rss_scope": "measuring-process-excluding-workers-v1",
}
TIMING = {"clock": "time.perf_counter_ns", "preflight": "paired-monotonic-wall-v2"}

# The data generator, frozen (formula version 1); see the runner for its description.
GENERATOR = {
    "formula_version": "1",
    "distribution": "exponential",
    "scale": 1000.0,
    "seed": 0x5CA1E5EED,
    "block_rows": 1 << 20,
    "column": "value",
    "compression": "snappy",
    "temporary_root": "redacted",
}
ROW_GROUP_CAP = 1 << 20
LAYOUT_A_GROUPS_PER_FILE = 16
LAYOUT_B_GROUPS_PER_FILE = 5
SUBSAMPLE_ROWS = 100_000

# The bounds of the pass planner (``veridist.scale._planner`` and ``_profile_intervals``); a test
# keeps them equal to the engine's constants.
MAX_ROWS_PER_BATCH = 262_144
MAX_WORKERS = 64
FIT_PASS_BUDGET = 8
FIT_MINIMUM_PASSES = 3
FIT_MAXIMUM_PASSES = 17
PROFILE_PASS_BUDGET = 6
PROFILE_MINIMUM_PASSES = 2
PROFILE_MAXIMUM_PASSES = 9
PROFILE_DEGREES = (8, 16, 32, 64, 128)
PROFILE_LEVEL = 0.95

GOF_BIN_EXPONENT = 12
GOF_THRESHOLD = 0.001
EXPONENTIAL_CDF_ERROR = 2.0**-48  # the engine's evaluation error of the exponential distribution

# Agreement with the in-memory fit of a subsample, relative, per parameter and for the
# log-likelihood. The one-pass families are exact functions of exact sums; the others carry the
# stopping tolerance of the in-memory search.
IN_MEMORY_PARAMETER_TOLERANCE = {
    "exponential": 1e-12,
    "normal": 1e-12,
    "gamma": 1e-6,
    "lognormal": 1e-12,
    "weibull_min": 1e-6,
    "gumbel_right": 1e-6,
}
IN_MEMORY_LIKELIHOOD_TOLERANCE = 1e-9

# Agreement with the 50-digit references of the fixed sample, relative.
MPMATH_TOLERANCE = {
    "exponential": 1e-14,
    "normal": 1e-14,
    "gamma": 1e-12,
    "lognormal": 1e-14,
    "weibull_min": 1e-12,
    "gumbel_right": 1e-12,
}
MPMATH_SAMPLE_ROWS = 2000
MPMATH_DIGITS = 50
# The maximum-likelihood estimates of the fixed sample to 30 digits (mpmath, 50-digit working
# precision); a test recomputes them.
MPMATH_REFERENCE = {
    "exponential": {
        "rate": "0.666942900469296483192947018302",
    },
    "normal": {
        "mu": "1.49937873136717826128005981445",
        "sigma": "0.419760008975983465419379018600",
    },
    "gamma": {
        "shape": "11.6893845649993001174768357017",
        "scale": "0.128268406521303291051068111397",
    },
    "lognormal": {
        "mu_log": "0.361667569005593386548835318884",
        "sigma_log": "0.304390997864365854788853808793",
    },
    "weibull_min": {
        "shape": "4.02012048520255157833033661781",
        "scale": "1.65592664110477352178279475344",
    },
    "gumbel_right": {
        "location": "1.28997093048026207955945698696",
        "scale": "0.395522598450925551285519672807",
    },
}

ARTIFACT_KEYS = {
    "schema_version",
    "run",
    "generator",
    "rungs",
    "mpmath_reference",
    "inference",
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
    "environment",
    "ladder",
    "limits",
}
ENVIRONMENT_KEYS = {
    "numpy_version",
    "pyarrow_version",
    "cpu_count",
    "process_workers",
    "numeric_environment_digest",
}
RUNG_KEYS = {"rows", "datasets", "cells", "invariance", "references"}
DATASET_KEYS = {"layout", "rows", "partitions", "files"}
LAYOUT_KEYS = {"row_group_rows", "groups_per_file", "order"}
CELL_KEYS = {
    "rows",
    "family",
    "group",
    "partitions",
    "max_rows_per_batch",
    "executor",
    "retries",
    "passes",
    "digests",
    "result",
    "elapsed_seconds",
    "throughput_rows_per_second",
    "memory",
}
EXECUTOR_KEYS = {"kind", "requested_workers", "effective_workers"}
PASSES_KEYS = {"data_passes", "minimum", "maximum", "budget", "roles"}
DIGEST_KEYS = {"state_sha256", "result_sha256", "passes_sha256"}
RESULT_KEYS = {"observation_count", "parameters_hex", "log_likelihood_hex"}
MEMORY_KEYS = {"tracemalloc_peak_bytes", "rss_baseline_bytes", "rss_peak_bytes"}
INVARIANCE_KEYS = {"family", "layout_b", "resumed"}
LAYOUT_B_KEYS = {"partitions", "requested_workers", "digests"}
RESUMED_KEYS = {
    "interrupted_after_commits",
    "partitions",
    "passes_after_resume",
    "replayed_passes",
    "reused_partitions",
    "read_partitions",
    "digests",
}
REFERENCE_KEYS = {"family", "subsample_rows", "subsample_sha256", "parameters", "log_likelihood"}
DIFFERENCE_KEYS = {"engine_hex", "reference_hex", "relative_difference"}
MPMATH_KEYS = {"sample", "digits", "families"}
MPMATH_SAMPLE_KEYS = {"rows", "formula_version", "sha256"}
MPMATH_FAMILY_KEYS = {"family", "parameters"}
MPMATH_PARAMETER_KEYS = {"engine_hex", "reference", "relative_difference"}
INFERENCE_KEYS = {"gof", "profile"}
GOF_KEYS = {
    "rows",
    "family",
    "executor",
    "fit_parameters_hex",
    "bin_exponent",
    "bins",
    "cdf_error",
    "ks_lower",
    "ks_upper",
    "tv_lower",
    "tv_upper",
    "strip_rows",
    "counts_sha256",
    "threshold",
    "decision",
    "passes",
    "elapsed_seconds",
}
PROFILE_KEYS = {
    "rows",
    "family",
    "level",
    "executor",
    "intervals",
    "resolution",
    "passes",
    "elapsed_seconds",
}
INTERVAL_KEYS = {"estimate_hex", "lower", "upper", "lower_unbounded", "upper_unbounded"}
RESOLUTION_KEYS = {"surrogate_degree", "surrogate_gap", "confirmation_residual", "tolerance"}


# ----------------------------------------------------------------------------- primitives


def _integer(value: object, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _finite(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _positive(value: object) -> bool:
    return _finite(value) and value > 0  # type: ignore[operator]


def _nonnegative(value: object) -> bool:
    return _finite(value) and value >= 0  # type: ignore[operator]


def _sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _hex_float(value: object) -> float | None:
    """The float named by a ``float.hex`` string, or ``None`` if it is not a finite one."""

    if not isinstance(value, str):
        return None
    try:
        parsed = float.fromhex(value)
    except ValueError:
        return None
    if not math.isfinite(parsed) or parsed.hex() != value:
        return None
    return parsed


def _contains_path(value: object) -> bool:
    if isinstance(value, str):
        return "/" in value or "\\" in value or re.search(r"[A-Za-z]:", value) is not None
    if isinstance(value, list):
        return any(_contains_path(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_path(key) or _contains_path(item) for key, item in value.items())
    return False


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _digest(value: dict[str, object]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return _canonical_digest(body)


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


# ------------------------------------------------------------------- the frozen contract


def layouts(rows: int) -> dict[str, dict[str, Any]]:
    """The two layouts of ``rows`` rows (row-group size, groups per file, block order)."""

    group_a = max(1, min(ROW_GROUP_CAP, rows // 16))
    group_b = group_a * 3 // 4 + 1
    return {
        "a": {
            "row_group_rows": group_a,
            "groups_per_file": LAYOUT_A_GROUPS_PER_FILE,
            "order": "ascending",
        },
        "b": {
            "row_group_rows": group_b,
            "groups_per_file": LAYOUT_B_GROUPS_PER_FILE,
            "order": "reversed",
        },
    }


def families_at(rows: int) -> tuple[str, ...]:
    """The families measured at a rung: the multi-pass ones only up to ``MULTI_PASS_MAX_ROWS``."""

    return FAMILIES if rows <= MULTI_PASS_MAX_ROWS else ONE_PASS_FAMILIES


def interruption_point(family: str, partitions: int) -> int:
    """Partition commits before the interruption: mid-pass-two for a multi-pass family."""

    if family in MULTI_PASS_FAMILIES:
        return partitions + partitions // 2
    return partitions // 2


def mpmath_sample_sha256() -> str:
    """The digest of the fixed sample, recomputed from its integer recipe."""

    chunks = []
    for index in range(MPMATH_SAMPLE_ROWS):
        first = (index * 2654435761 + 12345) % (1 << 32)
        second = (index * 1597334677 + 54321) % (1 << 32)
        chunks.append(struct.pack("<d", float((1 << 33) + 3 * first + 5 * second) / float(1 << 34)))
    return hashlib.sha256(b"".join(chunks)).hexdigest()


def decide(ks_lower: float, ks_upper: float, threshold: float) -> str:
    if ks_upper <= threshold:
        return "PASS"
    if ks_lower > threshold:
        return "FAIL"
    return "INCONCLUSIVE"


# ------------------------------------------------------------------------- the validators


def _check_run(
    run: object, expected_git_sha: str, repo_root: Path | None, errors: list[str]
) -> dict[str, Any] | None:
    if not _exact_keys(run, RUN_KEYS, "run", errors):
        return None
    assert isinstance(run, dict)
    if run["git_sha"] != expected_git_sha:
        errors.append("run git SHA does not match frozen expected SHA")
    if run["candidate_git_sha"] != expected_git_sha:
        errors.append("candidate git SHA does not match frozen expected SHA")
    if run["git_dirty"] is not False:
        errors.append("run is dirty")
    if repo_root is None:
        errors.append("repository binding is required")
    elif not _git_commit_is_ancestor(repo_root, expected_git_sha):
        errors.append("expected git SHA is not an existing ancestor of repository HEAD")
    python = run["python"]
    if (
        not _exact_keys(python, {"implementation", "version"}, "run python", errors)
        or python["implementation"] != "CPython"
        or not isinstance(python["version"], str)
        or not python["version"]
    ):
        errors.append("run Python metadata is invalid")
    if not isinstance(run["platform"], str) or not run["platform"]:
        errors.append("run platform is invalid")
    if not isinstance(run["utc_started"], str) or not run["utc_started"].endswith("Z"):
        errors.append("run UTC timestamp is invalid")
    if not _integer(run["measurement_workers"], 1):
        errors.append("measurement workers must be positive")
    elif run["measurement_workers"] != 1:
        errors.append("timing evidence requires exactly one measurement worker")
    if run["timing"] != TIMING:
        errors.append("run timing provenance is invalid")
    if run["methodology"] != METHODOLOGY:
        errors.append("run measurement methodology is not supported")
    environment = run["environment"]
    if _exact_keys(environment, ENVIRONMENT_KEYS, "run environment", errors):
        for key in ("numpy_version", "pyarrow_version"):
            if not isinstance(environment[key], str) or TOKEN.fullmatch(environment[key]) is None:
                errors.append(f"run environment {key} is invalid")
        if not _integer(environment["cpu_count"], 1) or not _integer(
            environment["process_workers"], 1
        ):
            errors.append("run environment core and worker counts must be positive")
        if not _sha256(environment["numeric_environment_digest"]):
            errors.append("run numeric environment digest is invalid")
    limits = run["limits"]
    if _exact_keys(
        limits,
        {"max_rows_per_batch", "pass_budget", "inference_pass_budget"},
        "run limits",
        errors,
    ) and limits != {
        "max_rows_per_batch": MAX_ROWS_PER_BATCH,
        "pass_budget": FIT_PASS_BUDGET,
        "inference_pass_budget": PROFILE_PASS_BUDGET,
    }:
        errors.append("run limits differ from the planner's defaults")
    return run


def _check_ladder(
    run: dict[str, Any], *, smoke: bool, top_rung: bool, errors: list[str]
) -> list[int] | None:
    ladder = run["ladder"]
    if not _exact_keys(ladder, {"rows", "top_rung"}, "run ladder", errors):
        return None
    rows = ladder["rows"]
    if (
        not isinstance(rows, list)
        or not rows
        or not all(_integer(item, 1) for item in rows)
        or ladder["top_rung"] is not (TOP_RUNG in rows)
    ):
        errors.append("run ladder is invalid")
        return None
    if smoke:
        if (
            top_rung
            or not 1 <= len(rows) <= 3
            or rows != sorted(set(rows))
            or rows[0] < SMOKE_MIN_ROWS
            or rows[-1] > SMOKE_MAX_ROWS
        ):
            errors.append("smoke ladder must be one to three small increasing rungs")
            return None
    else:
        expected = [*STANDARD_LADDER, *([TOP_RUNG] if top_rung else [])]
        if rows != expected:
            errors.append("row ladder is not the required one")
            return None
    environment = run["environment"]
    if isinstance(environment, dict) and _integer(environment.get("process_workers"), 1):
        if not smoke and environment["process_workers"] != environment.get("cpu_count"):
            errors.append("process executor must use the core count of the machine")
    return list(rows)


def _check_dataset(
    dataset: object, rows: int, name: str, expected: dict[str, Any], errors: list[str]
) -> int | None:
    label = f"rung {rows} dataset {name}"
    if not _exact_keys(dataset, DATASET_KEYS, label, errors):
        return None
    assert isinstance(dataset, dict)
    if not _exact_keys(dataset["layout"], LAYOUT_KEYS, f"{label} layout", errors):
        return None
    if dataset["layout"] != expected:
        errors.append(f"{label} layout is not the declared one")
    group = expected["row_group_rows"]
    partitions = -(-rows // group)
    if dataset["rows"] != rows or dataset["partitions"] != partitions:
        errors.append(f"{label} row or partition count mismatch")
    files = dataset["files"]
    wanted = -(-partitions // expected["groups_per_file"])
    if not isinstance(files, list) or len(files) != wanted:
        errors.append(f"{label} file count mismatch")
        return partitions
    total = 0
    for index, item in enumerate(files):
        if not _exact_keys(item, {"bytes", "sha256"}, f"{label} file {index}", errors):
            continue
        if not _integer(item["bytes"], 1) or not _sha256(item["sha256"]):
            errors.append(f"{label} file {index} bytes or SHA-256 invalid")
            continue
        total += item["bytes"]
    if total < rows * 4 or total > rows * 12 + 65_536 * wanted:
        errors.append(f"{label} byte count is implausible for the rows")
    return partitions


def _check_digests(digests: object, label: str, errors: list[str]) -> dict[str, str] | None:
    if not _exact_keys(digests, DIGEST_KEYS, label, errors):
        return None
    assert isinstance(digests, dict)
    if not all(_sha256(item) for item in digests.values()):
        errors.append(f"{label} must be SHA-256 digests")
        return None
    return {key: str(value) for key, value in digests.items()}


def _fit_roles_valid(roles: list[Any]) -> bool:
    """statistics, grid, any expansions, then iterations (at least one)."""

    if roles[:2] != ["statistics", "grid"]:
        return False
    rest = roles[2:]
    expansions = 0
    while expansions < len(rest) and rest[expansions] == "expansion":
        expansions += 1
    tail = rest[expansions:]
    return bool(tail) and all(item == "iteration" for item in tail)


def _gof_roles_valid(roles: list[Any]) -> bool:
    return roles == ["gof"]


def _profile_roles_valid(roles: list[Any]) -> bool:
    """The window pass, refinement passes, then the confirmation (the only one of its role)."""

    return (
        len(roles) >= 2
        and roles[0] == "profile_window"
        and roles[-1] == "profile_confirmation"
        and all(item == "profile_refinement" or item == "profile_window" for item in roles[1:-1])
    )


def _check_passes(
    passes: object,
    label: str,
    errors: list[str],
    *,
    minimum: int,
    maximum: int,
    budget: int,
    roles_valid: Callable[[list[Any]], bool] | None,
) -> None:
    """The passes of a computation: within the planner's bounds, in a valid sequence of roles.

    ``roles_valid`` is ``None`` for a one-pass family, which records no roles.
    """

    if not _exact_keys(passes, PASSES_KEYS, label, errors):
        return
    assert isinstance(passes, dict)
    count = passes["data_passes"]
    if (
        passes["minimum"] != minimum
        or passes["maximum"] != maximum
        or passes["budget"] != budget
        or not _integer(count, 1)
        or not minimum <= count <= min(maximum, budget)
    ):
        errors.append(f"{label} pass count is outside the planner's bounds")
        return
    roles = passes["roles"]
    if roles_valid is None:
        if roles != []:
            errors.append(f"{label} one-pass roles must be empty")
    elif not isinstance(roles, list) or len(roles) != count or not roles_valid(roles):
        errors.append(f"{label} pass roles are not a valid sequence")


def _check_executor(
    executor: object,
    label: str,
    kind: str,
    partitions: int,
    run: dict[str, Any],
    errors: list[str],
) -> None:
    if not _exact_keys(executor, EXECUTOR_KEYS, f"{label} executor", errors):
        return
    assert isinstance(executor, dict)
    environment = run["environment"]
    if kind == "serial":
        expected = {"kind": "serial", "requested_workers": 1, "effective_workers": 1}
    else:
        requested = environment["process_workers"]
        expected = {
            "kind": "processes",
            "requested_workers": requested,
            "effective_workers": min(requested, MAX_WORKERS, environment["cpu_count"], partitions),
        }
    if executor != expected:
        errors.append(f"{label} executor facts do not match the request")


def _check_result(
    result: object, family: str, rows: int, label: str, errors: list[str]
) -> dict[str, Any] | None:
    if not _exact_keys(result, RESULT_KEYS, f"{label} result", errors):
        return None
    assert isinstance(result, dict)
    parameters = result["parameters_hex"]
    if (
        result["observation_count"] != rows
        or not isinstance(parameters, dict)
        or set(parameters) != set(PARAMETER_NAMES[family])
        or _hex_float(result["log_likelihood_hex"]) is None
    ):
        errors.append(f"{label} result facts invalid")
        return None
    for name, text in parameters.items():
        parsed = _hex_float(text)
        if parsed is None or (name in POSITIVE_PARAMETERS and parsed <= 0.0):
            errors.append(f"{label} parameter {name} invalid")
            return None
    return result


def _result_digest(result: dict[str, Any], family: str) -> str:
    return _canonical_digest(
        {
            "family": family,
            "observation_count": result["observation_count"],
            "parameters_hex": result["parameters_hex"],
            "log_likelihood_hex": result["log_likelihood_hex"],
        }
    )


def _check_cell(
    cell: object, rows: int, partitions: int | None, run: dict[str, Any], errors: list[str]
) -> tuple[str, str] | None:
    if not _exact_keys(cell, CELL_KEYS, f"rung {rows} cell", errors):
        return None
    assert isinstance(cell, dict)
    family = cell["family"]
    executor = cell["executor"]
    kind = executor.get("kind") if isinstance(executor, dict) else None
    if family not in FAMILIES or kind not in EXECUTORS:
        errors.append(f"rung {rows} cell family or executor unknown")
        return None
    label = f"rung {rows} {family} {kind}"
    multi = family in MULTI_PASS_FAMILIES
    if cell["rows"] != rows or cell["group"] != ("multi_pass" if multi else "one_pass"):
        errors.append(f"{label} rows or group mismatch")
    if partitions is not None and cell["partitions"] != partitions:
        errors.append(f"{label} partition count mismatch")
    if cell["max_rows_per_batch"] != MAX_ROWS_PER_BATCH:
        errors.append(f"{label} batch cap differs from the default")
    _check_executor(executor, label, kind, cell["partitions"], run, errors)
    if not _integer(cell["retries"]):
        errors.append(f"{label} retries invalid")
    _check_passes(
        cell["passes"],
        f"{label} passes",
        errors,
        minimum=FIT_MINIMUM_PASSES if multi else 1,
        maximum=FIT_MAXIMUM_PASSES if multi else 1,
        budget=FIT_PASS_BUDGET if multi else 1,
        roles_valid=_fit_roles_valid if multi else None,
    )
    digests = _check_digests(cell["digests"], f"{label} digests", errors)
    result = _check_result(cell["result"], family, rows, label, errors)
    if digests is not None and result is not None:
        if digests["result_sha256"] != _result_digest(result, family):
            errors.append(f"{label} result digest does not match the recorded result")
    elapsed = cell["elapsed_seconds"]
    if not _positive(elapsed):
        errors.append(f"{label} elapsed time invalid")
    throughput = cell["throughput_rows_per_second"]
    if (
        not _positive(throughput)
        or not _positive(elapsed)
        or abs(throughput - rows / elapsed) > max(1e-9, throughput * 1e-12)
    ):
        errors.append(f"{label} throughput invalid")
    memory = cell["memory"]
    if _exact_keys(memory, MEMORY_KEYS, f"{label} memory", errors):
        if not all(_integer(memory[key], 1) for key in MEMORY_KEYS) or (
            memory["rss_peak_bytes"] < memory["rss_baseline_bytes"]
        ):
            errors.append(f"{label} memory facts invalid")
    return str(family), str(kind)


def _check_cells(
    rung: dict[str, Any], rows: int, partitions: int | None, run: dict[str, Any], errors: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    cells = rung["cells"]
    families = families_at(rows)
    found: dict[tuple[str, str], dict[str, Any]] = {}
    if not isinstance(cells, list) or len(cells) != len(families) * len(EXECUTORS):
        errors.append(f"rung {rows} must hold one cell per family and executor")
        return found
    for cell in cells:
        key = _check_cell(cell, rows, partitions, run, errors)
        if key is None:
            continue
        if key in found:
            errors.append(f"rung {rows} duplicate cell {key}")
            continue
        found[key] = cell
    if set(found) != {(f, e) for f in families for e in EXECUTORS}:
        errors.append(f"rung {rows} cell matrix is not complete")
    for family in families:
        serial, processes = found.get((family, "serial")), found.get((family, "processes"))
        if serial is None or processes is None:
            continue
        if (
            serial["digests"] != processes["digests"]
            or serial["result"] != processes["result"]
            or serial["passes"] != processes["passes"]
        ):
            errors.append(f"rung {rows} {family} serial and process results differ")
    return found


def _check_invariance(
    rung: dict[str, Any],
    rows: int,
    partitions: tuple[int | None, int | None],
    cells: dict[tuple[str, str], dict[str, Any]],
    run: dict[str, Any],
    errors: list[str],
) -> None:
    entries = rung["invariance"]
    families = families_at(rows)
    if not isinstance(entries, list) or len(entries) != len(families):
        errors.append(f"rung {rows} invariance must hold one entry per family")
        return
    for family, entry in zip(families, entries, strict=True):
        label = f"rung {rows} {family} invariance"
        if not _exact_keys(entry, INVARIANCE_KEYS, label, errors) or entry["family"] != family:
            errors.append(f"{label} entry invalid")
            continue
        reference = cells.get((family, "serial"))
        if reference is None:
            continue
        digests = reference["digests"]
        layout_b = entry["layout_b"]
        if _exact_keys(layout_b, LAYOUT_B_KEYS, f"{label} layout B", errors):
            _check_digests(layout_b["digests"], f"{label} layout B digests", errors)
            if (partitions[1] is not None and layout_b["partitions"] != partitions[1]) or layout_b[
                "requested_workers"
            ] != run["environment"]["process_workers"]:
                errors.append(f"{label} layout B run facts mismatch")
            if layout_b["digests"] != digests:
                errors.append(f"{label} digests differ between layouts A and B")
        resumed = entry["resumed"]
        if not _exact_keys(resumed, RESUMED_KEYS, f"{label} resumed", errors):
            continue
        _check_digests(resumed["digests"], f"{label} resumed digests", errors)
        count = reference["passes"]["data_passes"]
        total = reference["partitions"]
        if not _integer(total, 2):
            continue
        after = interruption_point(family, total)
        multi = family in MULTI_PASS_FAMILIES
        expected = {
            "interrupted_after_commits": after,
            "partitions": total,
            "passes_after_resume": count,
            "replayed_passes": 1 if multi else 0,
            "reused_partitions": after,
            "read_partitions": count * total - after,
        }
        if {key: resumed[key] for key in expected} != expected:
            errors.append(f"{label} resume arithmetic does not match the interruption")
        if resumed["digests"] != digests:
            errors.append(f"{label} digests differ between an interrupted run and a whole one")


def _relative(engine: float, reference: float) -> float:
    return abs(engine - reference) / abs(reference)


def _check_difference(item: object, label: str, tolerance: float, errors: list[str]) -> None:
    if not _exact_keys(item, DIFFERENCE_KEYS, label, errors):
        return
    assert isinstance(item, dict)
    engine, reference = _hex_float(item["engine_hex"]), _hex_float(item["reference_hex"])
    if engine is None or reference is None or reference == 0.0:
        errors.append(f"{label} values invalid")
        return
    recorded = item["relative_difference"]
    recomputed = _relative(engine, reference)
    if not _nonnegative(recorded) or not math.isclose(
        recorded, recomputed, rel_tol=1e-9, abs_tol=1e-300
    ):
        errors.append(f"{label} relative difference does not match its values")
    if recomputed > tolerance:
        errors.append(f"{label} differs from its reference beyond the declared tolerance")


def _check_references(rung: dict[str, Any], rows: int, errors: list[str]) -> None:
    entries = rung["references"]
    families = families_at(rows)
    if not isinstance(entries, list) or len(entries) != len(families):
        errors.append(f"rung {rows} references must hold one entry per family")
        return
    for family, entry in zip(families, entries, strict=True):
        label = f"rung {rows} {family} reference"
        if not _exact_keys(entry, REFERENCE_KEYS, label, errors) or entry["family"] != family:
            errors.append(f"{label} entry invalid")
            continue
        if entry["subsample_rows"] != min(rows, SUBSAMPLE_ROWS) or not _sha256(
            entry["subsample_sha256"]
        ):
            errors.append(f"{label} subsample facts invalid")
        parameters = entry["parameters"]
        if not isinstance(parameters, dict) or set(parameters) != set(PARAMETER_NAMES[family]):
            errors.append(f"{label} parameters invalid")
            continue
        for name, item in parameters.items():
            _check_difference(
                item, f"{label} {name}", IN_MEMORY_PARAMETER_TOLERANCE[family], errors
            )
        _check_difference(
            entry["log_likelihood"],
            f"{label} log-likelihood",
            IN_MEMORY_LIKELIHOOD_TOLERANCE,
            errors,
        )


def _check_mpmath(block: object, errors: list[str]) -> None:
    if not _exact_keys(block, MPMATH_KEYS, "mpmath reference", errors):
        return
    assert isinstance(block, dict)
    sample = block["sample"]
    if _exact_keys(sample, MPMATH_SAMPLE_KEYS, "mpmath sample", errors) and sample != {
        "rows": MPMATH_SAMPLE_ROWS,
        "formula_version": "1",
        "sha256": mpmath_sample_sha256(),
    }:
        errors.append("mpmath sample does not match the fixed sample")
    if block["digits"] != MPMATH_DIGITS:
        errors.append("mpmath precision is not the declared one")
    families = block["families"]
    if not isinstance(families, list) or len(families) != len(FAMILIES):
        errors.append("mpmath reference must hold one entry per family")
        return
    for family, entry in zip(FAMILIES, families, strict=True):
        label = f"mpmath {family}"
        if not _exact_keys(entry, MPMATH_FAMILY_KEYS, label, errors) or entry["family"] != family:
            errors.append(f"{label} entry invalid")
            continue
        parameters = entry["parameters"]
        if not isinstance(parameters, dict) or set(parameters) != set(PARAMETER_NAMES[family]):
            errors.append(f"{label} parameters invalid")
            continue
        for name, item in parameters.items():
            _check_mpmath_parameter(item, family, name, f"{label} {name}", errors)


def _check_mpmath_parameter(
    item: object, family: str, name: str, label: str, errors: list[str]
) -> None:
    if not _exact_keys(item, MPMATH_PARAMETER_KEYS, label, errors):
        return
    assert isinstance(item, dict)
    frozen = MPMATH_REFERENCE[family][name]
    if item["reference"] != frozen:
        errors.append(f"{label} reference value is not the frozen one")
        return
    engine = _hex_float(item["engine_hex"])
    if engine is None:
        errors.append(f"{label} engine value invalid")
        return
    try:
        with localcontext() as context:
            context.prec = 80
            reference = Decimal(frozen)
            difference = abs(Decimal(engine) - reference) / abs(reference)
            recomputed = float(difference)
    except (InvalidOperation, ValueError):
        errors.append(f"{label} difference cannot be recomputed")
        return
    recorded = item["relative_difference"]
    if not _nonnegative(recorded) or not math.isclose(
        recorded, recomputed, rel_tol=1e-6, abs_tol=1e-30
    ):
        errors.append(f"{label} relative difference does not match its values")
    if recomputed > MPMATH_TOLERANCE[family]:
        errors.append(f"{label} differs from its 50-digit reference beyond the declared tolerance")


def _check_inference_executor(
    executor: object, label: str, partitions: int | None, run: dict[str, Any], errors: list[str]
) -> None:
    _check_executor(executor, label, "processes", partitions or 1, run, errors)


def _check_gof(
    gof: object,
    rows: int,
    cells: dict[tuple[str, str], dict[str, Any]] | None,
    partitions: int | None,
    run: dict[str, Any],
    *,
    smoke: bool,
    errors: list[str],
) -> None:
    if not _exact_keys(gof, GOF_KEYS, "goodness of fit", errors):
        return
    assert isinstance(gof, dict)
    if gof["rows"] != rows or gof["family"] != "exponential":
        errors.append("goodness of fit is not for the exponential fit at the top rung")
    _check_inference_executor(gof["executor"], "goodness of fit", partitions, run, errors)
    parameters = gof["fit_parameters_hex"]
    rate = _hex_float(parameters.get("rate")) if isinstance(parameters, dict) else None
    if not isinstance(parameters, dict) or tuple(parameters) != ("rate",) or not rate or rate <= 0:
        errors.append("goodness of fit parameters invalid")
    elif cells is not None:
        bound = cells.get(("exponential", "processes"))
        if bound is not None and bound["result"]["parameters_hex"] != parameters:
            errors.append("goodness of fit was not computed at the measured exponential fit")
    bins = 1 << GOF_BIN_EXPONENT
    if gof["bin_exponent"] != GOF_BIN_EXPONENT or gof["bins"] != bins:
        errors.append("goodness of fit resolution is not the declared one")
    if gof["cdf_error"] != EXPONENTIAL_CDF_ERROR:
        errors.append("goodness of fit evaluation error is not the declared one")
    lower, upper = gof["ks_lower"], gof["ks_upper"]
    tv_lower, tv_upper = gof["tv_lower"], gof["tv_upper"]
    strips = gof["strip_rows"]
    if not (
        _finite(lower)
        and _finite(upper)
        and _finite(tv_lower)
        and _finite(tv_upper)
        and 0.0 <= lower <= upper <= 1.0
        and 0.0 <= tv_lower <= tv_upper <= 1.0
        and _integer(strips)
        and strips <= rows
    ):
        errors.append("goodness of fit enclosure is not a valid interval")
    else:
        # The interval is never wider than one bin plus the share of rows in the strips.
        if upper - lower > 1.0 / bins + 2.0 * strips / rows + 1e-12:
            errors.append("goodness of fit enclosure is wider than its stated bound")
    if not _sha256(gof["counts_sha256"]):
        errors.append("goodness of fit counts digest invalid")
    if gof["threshold"] != GOF_THRESHOLD:
        errors.append("goodness of fit threshold is not the declared one")
    elif (
        _finite(lower) and _finite(upper) and gof["decision"] != decide(lower, upper, GOF_THRESHOLD)
    ):
        errors.append("goodness of fit decision does not follow from its enclosure")
    if not smoke and gof["decision"] != "PASS":
        errors.append("the well-specified exponential fit must be certified within the threshold")
    _check_passes(
        gof["passes"],
        "goodness of fit passes",
        errors,
        minimum=1,
        maximum=1,
        budget=1,
        roles_valid=_gof_roles_valid,
    )
    if not _positive(gof["elapsed_seconds"]):
        errors.append("goodness of fit elapsed time invalid")


def _check_profile(
    profile: object,
    rows: int,
    cells: dict[tuple[str, str], dict[str, Any]] | None,
    partitions: int | None,
    run: dict[str, Any],
    errors: list[str],
) -> None:
    if not _exact_keys(profile, PROFILE_KEYS, "profile interval", errors):
        return
    assert isinstance(profile, dict)
    if profile["rows"] != rows or profile["family"] != "weibull_min":
        errors.append("profile interval is not for the Weibull fit at its rung")
    if profile["level"] != PROFILE_LEVEL:
        errors.append("profile interval level is not the declared one")
    _check_inference_executor(profile["executor"], "profile interval", partitions, run, errors)
    intervals = profile["intervals"]
    if not isinstance(intervals, dict) or set(intervals) != set(PARAMETER_NAMES["weibull_min"]):
        errors.append("profile interval parameters invalid")
    else:
        bound = None if cells is None else cells.get(("weibull_min", "processes"))
        for name, item in intervals.items():
            _check_profile_interval(item, name, rows, bound, errors)
    resolution = profile["resolution"]
    if _exact_keys(resolution, RESOLUTION_KEYS, "profile resolution", errors):
        if (
            resolution["surrogate_degree"] not in PROFILE_DEGREES
            or not _nonnegative(resolution["surrogate_gap"])
            or not _nonnegative(resolution["confirmation_residual"])
            or not _positive(resolution["tolerance"])
            or resolution["confirmation_residual"] > resolution["tolerance"]
        ):
            errors.append("profile resolution is invalid or beyond its tolerance")
    _check_passes(
        profile["passes"],
        "profile interval passes",
        errors,
        minimum=PROFILE_MINIMUM_PASSES,
        maximum=PROFILE_MAXIMUM_PASSES,
        budget=PROFILE_PASS_BUDGET,
        roles_valid=_profile_roles_valid,
    )
    if not _positive(profile["elapsed_seconds"]):
        errors.append("profile interval elapsed time invalid")


def _check_profile_interval(
    item: object, name: str, rows: int, bound: dict[str, Any] | None, errors: list[str]
) -> None:
    label = f"profile interval {name}"
    if not _exact_keys(item, INTERVAL_KEYS, label, errors):
        return
    assert isinstance(item, dict)
    estimate = _hex_float(item["estimate_hex"])
    lower, upper = item["lower"], item["upper"]
    if item["lower_unbounded"] is not False or item["upper_unbounded"] is not False:
        errors.append(f"{label} has an unbounded end")
    if (
        estimate is None
        or not (_finite(lower) and _finite(upper))
        or not (0.0 < lower < estimate < upper)
    ):
        errors.append(f"{label} does not enclose its estimate")
        return
    if bound is not None and bound["result"]["parameters_hex"][name] != item["estimate_hex"]:
        errors.append(f"{label} was not computed at the measured fit")
    if (upper - lower) / estimate > 20.0 / math.sqrt(rows):
        errors.append(f"{label} is wider than the sample size allows")


def validate(
    value: object,
    *,
    expected_git_sha: str,
    smoke: bool = False,
    top_rung: bool = False,
    repo_root: Path | None = None,
) -> list[str]:
    """Return all violations; malformed evidence never partly passes.

    ``smoke`` accepts the small ladders of the runner's own tests; ``top_rung`` requires the
    ladder that includes the largest rung.
    """

    try:
        return _validate(
            value,
            expected_git_sha=expected_git_sha,
            smoke=smoke,
            top_rung=top_rung,
            repo_root=repo_root,
        )
    except (AttributeError, IndexError, KeyError, OverflowError, TypeError, ValueError) as error:
        return [f"artifact structure is invalid ({type(error).__name__})"]


def _validate(
    value: object,
    *,
    expected_git_sha: str,
    smoke: bool,
    top_rung: bool,
    repo_root: Path | None,
) -> list[str]:
    errors: list[str] = []
    if GIT_SHA.fullmatch(expected_git_sha) is None:
        return ["expected git SHA is invalid"]
    if not _exact_keys(value, ARTIFACT_KEYS, "artifact", errors):
        return errors
    assert isinstance(value, dict)
    if value["schema_version"] != SCHEMA_VERSION:
        errors.append("current evidence requires schema version 1")
    if not isinstance(value["artifact_sha256"], str) or value["artifact_sha256"] != _digest(value):
        errors.append("artifact digest mismatch")
    if _contains_path(value):
        errors.append("path leaked into evidence artifact")
    run = _check_run(value["run"], expected_git_sha, repo_root, errors)
    if value["generator"] != GENERATOR:
        errors.append("generator contract is invalid")
    if run is None:
        return errors
    rows = _check_ladder(run, smoke=smoke, top_rung=top_rung, errors=errors)
    if rows is None:
        return errors
    rungs = value["rungs"]
    if not isinstance(rungs, list) or len(rungs) != len(rows):
        return [*errors, "rungs must hold one entry per ladder row count"]
    by_rows: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
    partitions_by_rows: dict[int, int | None] = {}
    for count, rung in zip(rows, rungs, strict=True):
        if not _exact_keys(rung, RUNG_KEYS, f"rung {count}", errors):
            continue
        assert isinstance(rung, dict)
        if rung["rows"] != count:
            errors.append(f"rung {count} row count mismatch")
            continue
        geometry = layouts(count)
        datasets = rung["datasets"]
        parts: dict[str, int | None] = {"a": None, "b": None}
        if _exact_keys(datasets, {"a", "b"}, f"rung {count} datasets", errors):
            for name in ("a", "b"):
                parts[name] = _check_dataset(datasets[name], count, name, geometry[name], errors)
            if datasets["a"].get("files") == datasets["b"].get("files"):
                errors.append(f"rung {count} layouts hold identical files")
        cells = _check_cells(rung, count, parts["a"], run, errors)
        by_rows[count] = cells
        partitions_by_rows[count] = parts["a"]
        _check_invariance(rung, count, (parts["a"], parts["b"]), cells, run, errors)
        _check_references(rung, count, errors)
    _check_mpmath(value["mpmath_reference"], errors)
    inference = value["inference"]
    if _exact_keys(inference, INFERENCE_KEYS, "inference", errors):
        top = rows[-1]
        profile_rows = max([item for item in rows if item <= PROFILE_MAX_ROWS] or [rows[0]])
        _check_gof(
            inference["gof"],
            top,
            by_rows.get(top),
            partitions_by_rows.get(top),
            run,
            smoke=smoke,
            errors=errors,
        )
        _check_profile(
            inference["profile"],
            profile_rows,
            by_rows.get(profile_rows),
            partitions_by_rows.get(profile_rows),
            run,
            errors,
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--top-rung",
        action="store_true",
        help="require the largest rung (the Linux ladder)",
    )
    args = parser.parse_args()
    try:
        value = json.loads(args.artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"FAIL: cannot read artifact: {error}", file=sys.stderr)
        return 1
    errors = validate(
        value,
        expected_git_sha=args.expected_git_sha,
        smoke=args.smoke,
        top_rung=args.top_rung,
        repo_root=args.repo_root,
    )
    if errors:
        print("FAIL: scale engine evidence rejected", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("PASS: scale engine evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
