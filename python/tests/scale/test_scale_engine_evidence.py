"""Contracts of the retained evidence for the internal scale engine.

The checker is exercised on synthetic artifacts that are valid by construction, one rejected
field at a time, and on the artifact of a real run of the runner at a tiny ladder. The runner's
orchestration is exercised with the measured parts replaced, and its parts (the data layouts, the
cell, the memory pass, the interrupted run) on small data.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from collections.abc import Callable
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.scale import _gof as scale_gof
from veridist.scale import _planner, _profile_intervals, _profile_math
from veridist.scale._executors import MAX_WORKERS
from veridist.scale._sources import DEFAULT_MAX_ROWS

REPO = Path(__file__).parents[3]
TOOLS = REPO / "python" / "tools"
CHECKER_PATH = TOOLS / "check_scale_engine_evidence.py"
RUNNER_PATH = TOOLS / "run_scale_engine_evidence.py"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHECKER = _load("scale_engine_checker", CHECKER_PATH)
RUNNER = _load("scale_engine_runner", RUNNER_PATH)


def _head() -> str:
    return subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip()


def _seal(value: dict[str, Any]) -> None:
    value["artifact_sha256"] = CHECKER._digest(value)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


PARAMETERS = {
    "exponential": {"rate": 0.001},
    "normal": {"mu": 1000.0, "sigma": 1000.0},
    "gamma": {"shape": 1.0, "scale": 1000.0},
    "lognormal": {"mu_log": 6.0, "sigma_log": 1.1},
    "weibull_min": {"shape": 1.0, "scale": 1000.0},
    "gumbel_right": {"location": 400.0, "scale": 600.0},
}
FIT_ROLES = ["statistics", "grid", "iteration", "iteration", "iteration", "iteration"]
CPU_COUNT = 4


def _result(family: str, rows: int) -> dict[str, Any]:
    return {
        "observation_count": rows,
        "parameters_hex": {name: value.hex() for name, value in PARAMETERS[family].items()},
        "log_likelihood_hex": (-5000.0).hex(),
    }


def _fit_passes(family: str) -> dict[str, Any]:
    if family in CHECKER.MULTI_PASS_FAMILIES:
        return {
            "data_passes": len(FIT_ROLES),
            "minimum": 3,
            "maximum": 17,
            "budget": 8,
            "roles": list(FIT_ROLES),
        }
    return {"data_passes": 1, "minimum": 1, "maximum": 1, "budget": 1, "roles": []}


def _digests(family: str, rows: int) -> dict[str, str]:
    return {
        "state_sha256": _sha(f"state-{family}-{rows}"),
        "result_sha256": CHECKER._result_digest(_result(family, rows), family),
        "passes_sha256": _sha(f"passes-{family}-{rows}"),
    }


def _executor(kind: str, partitions: int, workers: int) -> dict[str, Any]:
    if kind == "serial":
        return {"kind": "serial", "requested_workers": 1, "effective_workers": 1}
    return {
        "kind": "processes",
        "requested_workers": workers,
        "effective_workers": min(workers, MAX_WORKERS, CPU_COUNT, partitions),
    }


def _dataset(rows: int, name: str) -> dict[str, Any]:
    layout = CHECKER.layouts(rows)[name]
    partitions = -(-rows // layout["row_group_rows"])
    files = -(-partitions // layout["groups_per_file"])
    return {
        "layout": dict(layout),
        "rows": rows,
        "partitions": partitions,
        "files": [
            {"bytes": rows * 8 // files + 1000, "sha256": _sha(f"{name}-{rows}-{index}")}
            for index in range(files)
        ],
    }


def _cell(rows: int, family: str, kind: str, partitions: int, workers: int) -> dict[str, Any]:
    elapsed = 2.5
    return {
        "rows": rows,
        "family": family,
        "group": "multi_pass" if family in CHECKER.MULTI_PASS_FAMILIES else "one_pass",
        "partitions": partitions,
        "max_rows_per_batch": 262_144,
        "executor": _executor(kind, partitions, workers),
        "retries": 0,
        "passes": _fit_passes(family),
        "digests": _digests(family, rows),
        "result": _result(family, rows),
        "elapsed_seconds": elapsed,
        "throughput_rows_per_second": rows / elapsed,
        "memory": {
            "tracemalloc_peak_bytes": 4_000_000,
            "rss_baseline_bytes": 90_000_000,
            "rss_peak_bytes": 130_000_000,
        },
    }


def _difference(value: float) -> dict[str, Any]:
    return {
        "engine_hex": float(value).hex(),
        "reference_hex": float(value).hex(),
        "relative_difference": 0.0,
    }


def _rung(rows: int, workers: int) -> dict[str, Any]:
    families = CHECKER.families_at(rows)
    a, b = _dataset(rows, "a"), _dataset(rows, "b")
    cells = [
        _cell(rows, family, kind, a["partitions"], workers)
        for family in families
        for kind in CHECKER.EXECUTORS
    ]
    invariance = []
    references = []
    for family in families:
        count = _fit_passes(family)["data_passes"]
        after = CHECKER.interruption_point(family, a["partitions"])
        multi = family in CHECKER.MULTI_PASS_FAMILIES
        invariance.append(
            {
                "family": family,
                "layout_b": {
                    "partitions": b["partitions"],
                    "requested_workers": workers,
                    "digests": _digests(family, rows),
                },
                "resumed": {
                    "interrupted_after_commits": after,
                    "partitions": a["partitions"],
                    "passes_after_resume": count,
                    "replayed_passes": 1 if multi else 0,
                    "reused_partitions": after,
                    "read_partitions": count * a["partitions"] - after,
                    "digests": _digests(family, rows),
                },
            }
        )
        references.append(
            {
                "family": family,
                "subsample_rows": min(rows, CHECKER.SUBSAMPLE_ROWS),
                "subsample_sha256": _sha(f"sample-{rows}"),
                "parameters": {
                    name: _difference(value) for name, value in PARAMETERS[family].items()
                },
                "log_likelihood": _difference(-5000.0),
            }
        )
    return {
        "rows": rows,
        "datasets": {"a": a, "b": b},
        "cells": cells,
        "invariance": invariance,
        "references": references,
    }


def _mpmath() -> dict[str, Any]:
    families = []
    for family in CHECKER.FAMILIES:
        parameters = {}
        for name in CHECKER.PARAMETER_NAMES[family]:
            reference = CHECKER.MPMATH_REFERENCE[family][name]
            engine = float(Decimal(reference))
            difference = abs(Decimal(engine) - Decimal(reference)) / Decimal(reference)
            parameters[name] = {
                "engine_hex": engine.hex(),
                "reference": reference,
                "relative_difference": float(difference),
            }
        families.append({"family": family, "parameters": parameters})
    return {
        "sample": {
            "rows": CHECKER.MPMATH_SAMPLE_ROWS,
            "formula_version": "1",
            "sha256": CHECKER.mpmath_sample_sha256(),
        },
        "digits": CHECKER.MPMATH_DIGITS,
        "families": families,
    }


def _gof_record(rows: int, partitions: int, workers: int) -> dict[str, Any]:
    return {
        "rows": rows,
        "family": "exponential",
        "executor": _executor("processes", partitions, workers),
        "fit_parameters_hex": _result("exponential", rows)["parameters_hex"],
        "bin_exponent": 12,
        "bins": 4096,
        "cdf_error": 2.0**-48,
        "ks_lower": 0.0002,
        "ks_upper": 0.0004,
        "tv_lower": 0.1,
        "tv_upper": 0.1,
        "strip_rows": 0,
        "counts_sha256": _sha(f"counts-{rows}"),
        "threshold": 0.001,
        "decision": "PASS",
        "passes": {"data_passes": 1, "minimum": 1, "maximum": 1, "budget": 1, "roles": ["gof"]},
        "elapsed_seconds": 3.0,
    }


def _profile_record(rows: int, partitions: int, workers: int) -> dict[str, Any]:
    hexes = _result("weibull_min", rows)["parameters_hex"]
    intervals = {}
    half_width = 2.0 / math.sqrt(rows)
    for name, text in hexes.items():
        estimate = float.fromhex(text)
        intervals[name] = {
            "estimate_hex": text,
            "lower": estimate * (1.0 - half_width),
            "upper": estimate * (1.0 + half_width),
            "lower_unbounded": False,
            "upper_unbounded": False,
        }
    return {
        "rows": rows,
        "family": "weibull_min",
        "level": 0.95,
        "executor": _executor("processes", partitions, workers),
        "intervals": intervals,
        "resolution": {
            "surrogate_degree": 16,
            "surrogate_gap": 1e-15,
            "confirmation_residual": 1e-11,
            "tolerance": 1e-8,
        },
        "passes": {
            "data_passes": 3,
            "minimum": 2,
            "maximum": 9,
            "budget": 6,
            "roles": ["profile_window", "profile_refinement", "profile_confirmation"],
        },
        "elapsed_seconds": 5.0,
    }


def _artifact(*, rows: tuple[int, ...], workers: int, top_rung: bool = False) -> dict[str, Any]:
    rungs = [_rung(count, workers) for count in rows]
    by_rows = {rung["rows"]: rung for rung in rungs}
    profile_rows = max(count for count in rows if count <= CHECKER.PROFILE_MAX_ROWS)
    value: dict[str, Any] = {
        "schema_version": "1",
        "run": {
            "git_sha": _head(),
            "candidate_git_sha": _head(),
            "git_dirty": False,
            "utc_started": "2026-10-21T00:00:00Z",
            "python": {"implementation": "CPython", "version": "3.11.9"},
            "platform": "test-platform",
            "measurement_workers": 1,
            "timing": dict(CHECKER.TIMING),
            "methodology": dict(CHECKER.METHODOLOGY),
            "environment": {
                "numpy_version": "2.0.2",
                "pyarrow_version": "17.0.0",
                "cpu_count": CPU_COUNT,
                "process_workers": workers,
                "numeric_environment_digest": _sha("environment"),
            },
            "ladder": {"rows": list(rows), "top_rung": top_rung},
            "limits": {
                "max_rows_per_batch": 262_144,
                "pass_budget": 8,
                "inference_pass_budget": 6,
            },
        },
        "generator": dict(CHECKER.GENERATOR),
        "rungs": rungs,
        "mpmath_reference": _mpmath(),
        "inference": {
            "gof": _gof_record(rows[-1], by_rows[rows[-1]]["datasets"]["a"]["partitions"], workers),
            "profile": _profile_record(
                profile_rows, by_rows[profile_rows]["datasets"]["a"]["partitions"], workers
            ),
        },
    }
    _seal(value)
    return value


SMOKE_ROWS = (20_000,)
FULL_ROWS = CHECKER.STANDARD_LADDER
TOP_ROWS = (*CHECKER.STANDARD_LADDER, CHECKER.TOP_RUNG)
_CACHE: dict[str, dict[str, Any]] = {}


def _base(kind: str) -> dict[str, Any]:
    if kind not in _CACHE:
        if kind == "smoke":
            _CACHE[kind] = _artifact(rows=SMOKE_ROWS, workers=2)
        elif kind == "full":
            _CACHE[kind] = _artifact(rows=FULL_ROWS, workers=CPU_COUNT)
        else:
            _CACHE[kind] = _artifact(rows=TOP_ROWS, workers=CPU_COUNT, top_rung=True)
    return copy.deepcopy(_CACHE[kind])


def _find_cell(value: dict[str, Any], rows: int, family: str, kind: str) -> dict[str, Any]:
    rung = next(item for item in value["rungs"] if item["rows"] == rows)
    return next(
        cell
        for cell in rung["cells"]
        if cell["family"] == family and cell["executor"]["kind"] == kind
    )


def _find_rung(value: dict[str, Any], rows: int) -> dict[str, Any]:
    return next(item for item in value["rungs"] if item["rows"] == rows)


def _find_invariance(value: dict[str, Any], rows: int, family: str) -> dict[str, Any]:
    return next(item for item in _find_rung(value, rows)["invariance"] if item["family"] == family)


def _find_reference(value: dict[str, Any], rows: int, family: str) -> dict[str, Any]:
    return next(item for item in _find_rung(value, rows)["references"] if item["family"] == family)


def _mp_parameter(value: dict[str, Any], family: str, name: str) -> dict[str, Any]:
    entry = next(item for item in value["mpmath_reference"]["families"] if item["family"] == family)
    return entry["parameters"][name]


class CheckerAcceptanceTests(unittest.TestCase):
    def test_valid_artifacts_are_accepted_at_every_ladder(self) -> None:
        for kind, smoke, top in (
            ("smoke", True, False),
            ("full", False, False),
            ("top", False, True),
        ):
            with self.subTest(kind=kind):
                errors = CHECKER.validate(
                    _base(kind),
                    expected_git_sha=_head(),
                    smoke=smoke,
                    top_rung=top,
                    repo_root=REPO,
                )
                self.assertEqual(errors, [])

    def test_the_top_rung_holds_the_one_pass_families_only(self) -> None:
        artifact = _base("top")
        top = _find_rung(artifact, CHECKER.TOP_RUNG)
        self.assertEqual({cell["family"] for cell in top["cells"]}, set(CHECKER.ONE_PASS_FAMILIES))
        self.assertEqual(len(top["cells"]), 8)
        self.assertEqual(len(_find_rung(artifact, 100_000_000)["cells"]), 12)
        self.assertEqual(artifact["inference"]["gof"]["rows"], CHECKER.TOP_RUNG)
        self.assertEqual(artifact["inference"]["profile"]["rows"], 100_000_000)


class CheckerRejectionTests(unittest.TestCase):
    """Each case damages one field of a valid artifact and expects the named violation."""

    def reject(
        self,
        mutate: Callable[[dict[str, Any]], None],
        message: str,
        *,
        kind: str = "smoke",
        reseal: bool = True,
        expected_sha: str | None = None,
        repo_root: Path | None = REPO,
    ) -> None:
        artifact = _base(kind)
        mutate(artifact)
        if reseal:
            _seal(artifact)
        errors = CHECKER.validate(
            artifact,
            expected_git_sha=expected_sha or _head(),
            smoke=kind == "smoke",
            top_rung=kind == "top",
            repo_root=repo_root,
        )
        self.assertTrue(any(message in error for error in errors), errors)

    def reject_all(
        self, cases: dict[str, tuple[Callable[[dict[str, Any]], None], str]], **options: Any
    ) -> None:
        for name, (mutate, message) in cases.items():
            with self.subTest(case=name):
                self.reject(mutate, message, **options)

    # ---------------------------------------------------------------- artifact and run

    def test_artifact_envelope(self) -> None:
        self.reject(lambda v: v.update(schema_version="2"), "schema version 1")
        self.reject(lambda v: v.update(artifact_sha256="0" * 64), "digest mismatch", reseal=False)
        self.reject(lambda v: v["run"].update(platform="C:/runner"), "path leaked")
        self.reject(lambda v: v.update(extra=1), "schema keys invalid")
        self.reject(lambda v: v.pop("inference"), "schema keys invalid")
        self.reject(lambda v: v["generator"].update(seed=1), "generator contract")
        self.reject(lambda v: v["generator"].update(temporary_root="/tmp"), "generator contract")

    def test_a_body_edited_after_sealing_is_rejected(self) -> None:
        artifact = _base("smoke")
        artifact["rungs"][0]["cells"][0]["elapsed_seconds"] = 9.0
        errors = CHECKER.validate(artifact, expected_git_sha=_head(), smoke=True, repo_root=REPO)
        self.assertIn("artifact digest mismatch", errors)

    def test_candidate_binding(self) -> None:
        self.reject(lambda v: v["run"].update(git_sha="1" * 40), "run git SHA")
        self.reject(lambda v: v["run"].update(candidate_git_sha="1" * 40), "candidate git SHA")
        self.reject(lambda v: v["run"].update(git_dirty=True), "run is dirty")
        self.reject(lambda v: None, "not an existing ancestor", expected_sha="0" * 40)
        self.reject(lambda v: None, "repository binding is required", repo_root=None)
        errors = CHECKER.validate(_base("smoke"), expected_git_sha="abc", smoke=True)
        self.assertEqual(errors, ["expected git SHA is invalid"])

    def test_run_facts(self) -> None:
        self.reject_all(
            {
                "python": (lambda v: v["run"]["python"].update(implementation="PyPy"), "Python"),
                "python keys": (lambda v: v["run"]["python"].pop("version"), "run python"),
                "platform": (lambda v: v["run"].update(platform=""), "platform is invalid"),
                "timestamp": (lambda v: v["run"].update(utc_started="2026-10-21"), "timestamp"),
                "no workers": (
                    lambda v: v["run"].update(measurement_workers=0),
                    "workers must be positive",
                ),
                "parallel workers": (
                    lambda v: v["run"].update(measurement_workers=2),
                    "exactly one measurement worker",
                ),
                "clock": (lambda v: v["run"]["timing"].update(clock="time.time"), "timing"),
                "methodology": (
                    lambda v: v["run"]["methodology"].update(rss="current"),
                    "methodology is not supported",
                ),
                "methodology missing": (
                    lambda v: v["run"].pop("methodology"),
                    "run schema keys invalid",
                ),
                "numpy": (
                    lambda v: v["run"]["environment"].update(numpy_version=""),
                    "numpy_version",
                ),
                "pyarrow": (
                    lambda v: v["run"]["environment"].update(pyarrow_version=3),
                    "pyarrow_version",
                ),
                "cores": (
                    lambda v: v["run"]["environment"].update(cpu_count=0),
                    "core and worker counts",
                ),
                "environment digest": (
                    lambda v: v["run"]["environment"].update(numeric_environment_digest="x"),
                    "numeric environment digest",
                ),
                "limits": (
                    lambda v: v["run"]["limits"].update(pass_budget=9),
                    "planner's defaults",
                ),
                "batch cap": (
                    lambda v: v["run"]["limits"].update(max_rows_per_batch=1),
                    "planner's defaults",
                ),
            }
        )

    def test_ladders(self) -> None:
        self.reject(lambda v: v["run"]["ladder"].update(rows=[]), "ladder is invalid")
        self.reject(lambda v: v["run"]["ladder"].update(top_rung=True), "ladder is invalid")
        self.reject(
            lambda v: v["run"]["ladder"].update(rows=[20_000, 10_000]), "smoke ladder", kind="smoke"
        )
        self.reject(lambda v: v["run"]["ladder"].update(rows=[500]), "smoke ladder", kind="smoke")
        self.reject(
            lambda v: v["run"]["ladder"].update(rows=[2_000_000]), "smoke ladder", kind="smoke"
        )
        self.reject(
            lambda v: v["run"]["ladder"].update(rows=[10_000, 20_000, 30_000, 40_000]),
            "smoke ladder",
        )
        self.reject(
            lambda v: v["run"]["ladder"].update(rows=[1_000_000, 10_000_000]),
            "row ladder is not the required one",
            kind="full",
        )
        self.reject(
            lambda v: v["run"]["ladder"].update(rows=list(FULL_ROWS), top_rung=False),
            "row ladder is not the required one",
            kind="top",
        )
        self.reject(
            lambda v: v["run"]["ladder"].update(rows=[1_000_000, 100_000_000, 10_000_000]),
            "row ladder is not the required one",
            kind="full",
        )

    def test_a_linux_run_without_the_top_rung_is_rejected_when_it_is_required(self) -> None:
        errors = CHECKER.validate(
            _base("full"), expected_git_sha=_head(), top_rung=True, repo_root=REPO
        )
        self.assertIn("row ladder is not the required one", errors)

    def test_a_windows_run_with_the_top_rung_is_rejected_when_it_is_not_allowed(self) -> None:
        errors = CHECKER.validate(_base("top"), expected_git_sha=_head(), repo_root=REPO)
        self.assertIn("row ladder is not the required one", errors)

    def test_the_process_executor_must_use_the_core_count_in_a_full_run(self) -> None:
        def mutate(value: dict[str, Any]) -> None:
            value["run"]["environment"]["process_workers"] = 2

        self.reject(mutate, "core count of the machine", kind="full")

    # ------------------------------------------------------------------------ rungs

    def test_rung_envelope_and_datasets(self) -> None:
        rows = SMOKE_ROWS[0]
        self.reject_all(
            {
                "rung rows": (lambda v: v["rungs"][0].update(rows=1), "row count mismatch"),
                "rung keys": (lambda v: v["rungs"][0].update(extra=1), "schema keys invalid"),
                "rung count": (lambda v: v["rungs"].append(v["rungs"][0]), "one entry per ladder"),
                "layout a": (
                    lambda v: v["rungs"][0]["datasets"]["a"]["layout"].update(row_group_rows=7),
                    "layout is not the declared one",
                ),
                "layout b order": (
                    lambda v: v["rungs"][0]["datasets"]["b"]["layout"].update(order="ascending"),
                    "layout is not the declared one",
                ),
                "dataset rows": (
                    lambda v: v["rungs"][0]["datasets"]["a"].update(rows=rows + 1),
                    "row or partition count",
                ),
                "dataset partitions": (
                    lambda v: v["rungs"][0]["datasets"]["b"].update(partitions=3),
                    "row or partition count",
                ),
                "files": (
                    lambda v: v["rungs"][0]["datasets"]["a"]["files"].pop(),
                    "file count mismatch",
                ),
                "file digest": (
                    lambda v: v["rungs"][0]["datasets"]["a"]["files"][0].update(sha256="zz"),
                    "bytes or SHA-256 invalid",
                ),
                "file bytes": (
                    lambda v: v["rungs"][0]["datasets"]["a"]["files"][0].update(bytes=0),
                    "bytes or SHA-256 invalid",
                ),
                "file keys": (
                    lambda v: v["rungs"][0]["datasets"]["a"]["files"][0].update(path="x"),
                    "schema keys invalid",
                ),
                "implausible bytes": (
                    lambda v: v["rungs"][0]["datasets"]["a"]["files"][0].update(bytes=10),
                    "byte count is implausible",
                ),
                "layouts identical": (
                    lambda v: v["rungs"][0]["datasets"].update(
                        b={
                            **v["rungs"][0]["datasets"]["b"],
                            "files": v["rungs"][0]["datasets"]["a"]["files"],
                        }
                    ),
                    "layouts hold identical files",
                ),
            }
        )

    # ------------------------------------------------------------------------ cells

    def test_cell_facts(self) -> None:
        rows = SMOKE_ROWS[0]

        def cell(value: dict[str, Any], family: str = "normal", kind: str = "serial") -> Any:
            return _find_cell(value, rows, family, kind)

        self.reject_all(
            {
                "missing cell": (lambda v: v["rungs"][0]["cells"].pop(), "one cell per family"),
                "duplicate cell": (
                    lambda v: v["rungs"][0]["cells"].__setitem__(
                        0, dict(v["rungs"][0]["cells"][1])
                    ),
                    "duplicate cell",
                ),
                "unknown family": (
                    lambda v: cell(v).update(family="cauchy"),
                    "family or executor unknown",
                ),
                "unknown executor": (
                    lambda v: cell(v)["executor"].update(kind="threads"),
                    "family or executor unknown",
                ),
                "cell keys": (lambda v: cell(v).update(extra=1), "schema keys invalid"),
                "rows": (lambda v: cell(v).update(rows=1), "rows or group mismatch"),
                "group": (lambda v: cell(v).update(group="multi_pass"), "rows or group mismatch"),
                "partitions": (lambda v: cell(v).update(partitions=1), "partition count mismatch"),
                "batch cap": (lambda v: cell(v).update(max_rows_per_batch=1024), "batch cap"),
                "serial workers": (
                    lambda v: cell(v)["executor"].update(requested_workers=2),
                    "executor facts do not match",
                ),
                "process workers": (
                    lambda v: cell(v, kind="processes")["executor"].update(requested_workers=3),
                    "executor facts do not match",
                ),
                "effective workers": (
                    lambda v: cell(v, kind="processes")["executor"].update(effective_workers=1),
                    "executor facts do not match",
                ),
                "executor keys": (
                    lambda v: cell(v)["executor"].pop("effective_workers"),
                    "schema keys invalid",
                ),
                "retries": (lambda v: cell(v).update(retries=-1), "retries invalid"),
                "one-pass count": (
                    lambda v: cell(v)["passes"].update(data_passes=2),
                    "outside the planner's bounds",
                ),
                "one-pass roles": (
                    lambda v: cell(v)["passes"].update(roles=["statistics"]),
                    "one-pass roles must be empty",
                ),
                "multi-pass below minimum": (
                    lambda v: cell(v, "weibull_min")["passes"].update(data_passes=2),
                    "outside the planner's bounds",
                ),
                "multi-pass over budget": (
                    lambda v: cell(v, "weibull_min")["passes"].update(data_passes=9),
                    "outside the planner's bounds",
                ),
                "planner maximum": (
                    lambda v: cell(v, "gumbel_right")["passes"].update(maximum=12),
                    "outside the planner's bounds",
                ),
                "pass budget": (
                    lambda v: cell(v, "gumbel_right")["passes"].update(budget=100),
                    "outside the planner's bounds",
                ),
                "role count": (
                    lambda v: cell(v, "weibull_min")["passes"].update(roles=["statistics"]),
                    "pass roles are not a valid sequence",
                ),
                "role order": (
                    lambda v: cell(v, "weibull_min")["passes"].update(
                        roles=[
                            "grid",
                            "statistics",
                            "iteration",
                            "iteration",
                            "iteration",
                            "iteration",
                        ]
                    ),
                    "pass roles are not a valid sequence",
                ),
                "role without iteration": (
                    lambda v: cell(v, "weibull_min")["passes"].update(
                        roles=[
                            "statistics",
                            "grid",
                            "expansion",
                            "expansion",
                            "expansion",
                            "expansion",
                        ]
                    ),
                    "pass roles are not a valid sequence",
                ),
                "digest form": (
                    lambda v: cell(v)["digests"].update(state_sha256="abc"),
                    "must be SHA-256 digests",
                ),
                "result digest": (
                    lambda v: cell(v)["digests"].update(result_sha256="0" * 64),
                    "result digest does not match",
                ),
                "digest keys": (lambda v: cell(v)["digests"].pop("passes_sha256"), "schema keys"),
                "observation count": (
                    lambda v: cell(v)["result"].update(observation_count=1),
                    "result facts invalid",
                ),
                "parameter names": (
                    lambda v: cell(v)["result"]["parameters_hex"].pop("mu"),
                    "result facts invalid",
                ),
                "parameter sign": (
                    lambda v: cell(v)["result"]["parameters_hex"].update(sigma=(-1.0).hex()),
                    "parameter sigma invalid",
                ),
                "parameter form": (
                    lambda v: cell(v)["result"]["parameters_hex"].update(mu="1.0"),
                    "parameter mu invalid",
                ),
                "parameter non-finite": (
                    lambda v: cell(v)["result"]["parameters_hex"].update(mu="inf"),
                    "parameter mu invalid",
                ),
                "log-likelihood": (
                    lambda v: cell(v)["result"].update(log_likelihood_hex="nan"),
                    "result facts invalid",
                ),
                "elapsed zero": (
                    lambda v: cell(v).update(elapsed_seconds=0),
                    "elapsed time invalid",
                ),
                "elapsed bool": (
                    lambda v: cell(v).update(elapsed_seconds=True),
                    "elapsed time invalid",
                ),
                "elapsed string": (
                    lambda v: cell(v).update(elapsed_seconds="2.5"),
                    "elapsed time invalid",
                ),
                "throughput": (
                    lambda v: cell(v).update(throughput_rows_per_second=1.0),
                    "throughput invalid",
                ),
                "memory zero": (
                    lambda v: cell(v)["memory"].update(tracemalloc_peak_bytes=0),
                    "memory facts invalid",
                ),
                "memory order": (
                    lambda v: cell(v)["memory"].update(rss_peak_bytes=1),
                    "memory facts invalid",
                ),
                "memory keys": (
                    lambda v: cell(v)["memory"].update(rss_delta_bytes=0),
                    "schema keys invalid",
                ),
            }
        )

    def test_non_finite_timings_are_rejected(self) -> None:
        artifact = _base("smoke")
        _find_cell(artifact, SMOKE_ROWS[0], "normal", "serial")["elapsed_seconds"] = math.nan
        errors = CHECKER.validate(artifact, expected_git_sha=_head(), smoke=True, repo_root=REPO)
        self.assertTrue(errors)

    def test_the_serial_and_the_process_cell_of_a_family_must_agree(self) -> None:
        rows = SMOKE_ROWS[0]

        def state(value: dict[str, Any]) -> None:
            _find_cell(value, rows, "gamma", "processes")["digests"]["state_sha256"] = "1" * 64

        def passes(value: dict[str, Any]) -> None:
            _find_cell(value, rows, "weibull_min", "processes")["passes"]["data_passes"] = 7

        def result(value: dict[str, Any]) -> None:
            cell = _find_cell(value, rows, "normal", "processes")
            cell["result"]["parameters_hex"]["mu"] = (999.0).hex()
            cell["digests"]["result_sha256"] = CHECKER._result_digest(cell["result"], "normal")

        for mutate in (state, passes, result):
            with self.subTest(case=mutate.__name__):
                self.reject(mutate, "serial and process results differ")

    def test_multi_pass_cells_are_not_accepted_at_the_top_rung(self) -> None:
        def mutate(value: dict[str, Any]) -> None:
            top = _find_rung(value, CHECKER.TOP_RUNG)
            top["cells"].append(
                _cell(
                    CHECKER.TOP_RUNG, "weibull_min", "serial", top["datasets"]["a"]["partitions"], 4
                )
            )

        self.reject(mutate, "one cell per family and executor", kind="top")

    def test_one_pass_cells_are_required_at_the_top_rung(self) -> None:
        def mutate(value: dict[str, Any]) -> None:
            top = _find_rung(value, CHECKER.TOP_RUNG)
            top["cells"] = [cell for cell in top["cells"] if cell["family"] != "gamma"]

        self.reject(mutate, "one cell per family and executor", kind="top")

    # ------------------------------------------------------------------ invariance

    def test_invariance_proofs(self) -> None:
        rows = SMOKE_ROWS[0]

        def entry(value: dict[str, Any], family: str = "gamma") -> Any:
            return _find_invariance(value, rows, family)

        self.reject_all(
            {
                "missing entry": (
                    lambda v: v["rungs"][0]["invariance"].pop(),
                    "one entry per family",
                ),
                "entry family": (lambda v: entry(v).update(family="normal"), "entry invalid"),
                "entry keys": (lambda v: entry(v).update(extra=1), "schema keys invalid"),
                "layout b state": (
                    lambda v: entry(v)["layout_b"]["digests"].update(state_sha256="2" * 64),
                    "differ between layouts A and B",
                ),
                "layout b result": (
                    lambda v: entry(v)["layout_b"]["digests"].update(result_sha256="2" * 64),
                    "differ between layouts A and B",
                ),
                "layout b passes": (
                    lambda v: entry(v)["layout_b"]["digests"].update(passes_sha256="2" * 64),
                    "differ between layouts A and B",
                ),
                "layout b digest form": (
                    lambda v: entry(v)["layout_b"]["digests"].update(state_sha256="x"),
                    "must be SHA-256 digests",
                ),
                "layout b partitions": (
                    lambda v: entry(v)["layout_b"].update(partitions=1),
                    "layout B run facts mismatch",
                ),
                "layout b workers": (
                    lambda v: entry(v)["layout_b"].update(requested_workers=1),
                    "layout B run facts mismatch",
                ),
                "resumed state": (
                    lambda v: entry(v)["resumed"]["digests"].update(state_sha256="3" * 64),
                    "differ between an interrupted run and a whole one",
                ),
                "resumed result": (
                    lambda v: entry(v, "weibull_min")["resumed"]["digests"].update(
                        result_sha256="3" * 64
                    ),
                    "differ between an interrupted run and a whole one",
                ),
                "resumed point": (
                    lambda v: entry(v)["resumed"].update(interrupted_after_commits=1),
                    "resume arithmetic",
                ),
                "resumed reused": (
                    lambda v: entry(v)["resumed"].update(reused_partitions=0),
                    "resume arithmetic",
                ),
                "resumed read": (
                    lambda v: entry(v)["resumed"].update(read_partitions=1),
                    "resume arithmetic",
                ),
                "resumed passes": (
                    lambda v: entry(v, "weibull_min")["resumed"].update(passes_after_resume=7),
                    "resume arithmetic",
                ),
                "resumed replay": (
                    lambda v: entry(v, "weibull_min")["resumed"].update(replayed_passes=0),
                    "resume arithmetic",
                ),
                "one-pass replay": (
                    lambda v: entry(v)["resumed"].update(replayed_passes=1),
                    "resume arithmetic",
                ),
                "resumed keys": (lambda v: entry(v)["resumed"].pop("digests"), "schema keys"),
            }
        )

    # ------------------------------------------------------------------ references

    def test_in_memory_references(self) -> None:
        rows = SMOKE_ROWS[0]

        def reference(value: dict[str, Any], family: str = "gamma") -> Any:
            return _find_reference(value, rows, family)

        def moved(value: dict[str, Any], family: str, name: str, factor: float) -> None:
            item = reference(value, family)["parameters"][name]
            engine = float.fromhex(item["reference_hex"]) * factor
            item["engine_hex"] = engine.hex()
            item["relative_difference"] = abs(factor - 1.0)

        self.reject_all(
            {
                "tolerance": (
                    lambda v: moved(v, "normal", "mu", 1.0 + 1e-9),
                    "beyond the declared tolerance",
                ),
                "loose family": (
                    lambda v: moved(v, "gamma", "shape", 1.0 + 1e-3),
                    "beyond the declared tolerance",
                ),
                "recorded difference": (
                    lambda v: reference(v)["parameters"]["shape"].update(relative_difference=0.5),
                    "does not match its values",
                ),
                "log-likelihood": (
                    lambda v: reference(v)["log_likelihood"].update(
                        engine_hex=(-4999.0).hex(), relative_difference=2e-4
                    ),
                    "beyond the declared tolerance",
                ),
                "value form": (
                    lambda v: reference(v)["parameters"]["shape"].update(engine_hex="1.0"),
                    "values invalid",
                ),
                "zero reference": (
                    lambda v: reference(v)["parameters"]["shape"].update(reference_hex=(0.0).hex()),
                    "values invalid",
                ),
                "difference keys": (
                    lambda v: reference(v)["parameters"]["shape"].update(extra=1),
                    "schema keys invalid",
                ),
                "subsample rows": (
                    lambda v: reference(v).update(subsample_rows=5),
                    "subsample facts invalid",
                ),
                "subsample digest": (
                    lambda v: reference(v).update(subsample_sha256="g"),
                    "subsample facts invalid",
                ),
                "parameter names": (
                    lambda v: reference(v)["parameters"].pop("scale"),
                    "parameters invalid",
                ),
                "missing reference": (
                    lambda v: v["rungs"][0]["references"].pop(),
                    "one entry per family",
                ),
                "reference family": (
                    lambda v: reference(v).update(family="normal"),
                    "entry invalid",
                ),
            }
        )

    def test_mpmath_references(self) -> None:
        def moved(value: dict[str, Any], factor: str) -> None:
            item = _mp_parameter(value, "normal", "mu")
            engine = float(Decimal(item["reference"]) * Decimal(factor))
            item["engine_hex"] = engine.hex()
            item["relative_difference"] = float(
                abs(Decimal(engine) - Decimal(item["reference"])) / Decimal(item["reference"])
            )

        self.reject_all(
            {
                "sample": (
                    lambda v: v["mpmath_reference"]["sample"].update(sha256="0" * 64),
                    "does not match the fixed sample",
                ),
                "sample rows": (
                    lambda v: v["mpmath_reference"]["sample"].update(rows=10),
                    "does not match the fixed sample",
                ),
                "digits": (
                    lambda v: v["mpmath_reference"].update(digits=30),
                    "precision is not the declared one",
                ),
                "frozen value": (
                    lambda v: _mp_parameter(v, "gamma", "shape").update(
                        reference="11.6893845649993001174768357018"
                    ),
                    "reference value is not the frozen one",
                ),
                "tolerance": (
                    lambda v: moved(v, "1.000000000001"),
                    "differs from its 50-digit reference",
                ),
                "recorded difference": (
                    lambda v: _mp_parameter(v, "normal", "sigma").update(relative_difference=0.1),
                    "does not match its values",
                ),
                "engine form": (
                    lambda v: _mp_parameter(v, "normal", "sigma").update(engine_hex="x"),
                    "engine value invalid",
                ),
                "family count": (
                    lambda v: v["mpmath_reference"]["families"].pop(),
                    "one entry per family",
                ),
                "family order": (
                    lambda v: v["mpmath_reference"]["families"].reverse(),
                    "entry invalid",
                ),
                "parameter names": (
                    lambda v: v["mpmath_reference"]["families"][1]["parameters"].pop("mu"),
                    "parameters invalid",
                ),
                "block keys": (lambda v: v["mpmath_reference"].update(extra=1), "schema keys"),
            }
        )

    # -------------------------------------------------------------------- inference

    def test_goodness_of_fit(self) -> None:
        def gof(value: dict[str, Any]) -> dict[str, Any]:
            return value["inference"]["gof"]

        self.reject_all(
            {
                "keys": (lambda v: gof(v).update(extra=1), "schema keys invalid"),
                "rows": (lambda v: gof(v).update(rows=1), "not for the exponential fit"),
                "family": (
                    lambda v: gof(v).update(family="weibull_min"),
                    "not for the exponential",
                ),
                "executor": (
                    lambda v: gof(v)["executor"].update(kind="serial"),
                    "executor facts do not match",
                ),
                "parameters": (
                    lambda v: gof(v).update(fit_parameters_hex={"rate": (0.002).hex()}),
                    "not computed at the measured exponential fit",
                ),
                "parameter names": (
                    lambda v: gof(v).update(fit_parameters_hex={"shape": (1.0).hex()}),
                    "parameters invalid",
                ),
                "resolution": (
                    lambda v: gof(v).update(bins=2048),
                    "resolution is not the declared",
                ),
                "exponent": (
                    lambda v: gof(v).update(bin_exponent=11),
                    "resolution is not the declared",
                ),
                "evaluation error": (
                    lambda v: gof(v).update(cdf_error=1e-9),
                    "evaluation error is not the declared one",
                ),
                "inverted enclosure": (
                    lambda v: gof(v).update(ks_lower=0.5),
                    "not a valid interval",
                ),
                "enclosure above one": (
                    lambda v: gof(v).update(ks_upper=1.5),
                    "not a valid interval",
                ),
                "negative": (lambda v: gof(v).update(ks_lower=-0.1), "not a valid interval"),
                "tv inverted": (
                    lambda v: gof(v).update(tv_lower=0.5, tv_upper=0.1),
                    "not a valid interval",
                ),
                "strip rows": (
                    lambda v: gof(v).update(strip_rows=10**9),
                    "not a valid interval",
                ),
                "wide enclosure": (
                    lambda v: gof(v).update(ks_lower=0.0, ks_upper=0.0009),
                    "wider than its stated bound",
                ),
                "counts digest": (
                    lambda v: gof(v).update(counts_sha256="abc"),
                    "counts digest invalid",
                ),
                "threshold": (
                    lambda v: gof(v).update(threshold=0.01),
                    "threshold is not the declared",
                ),
                "decision": (
                    lambda v: gof(v).update(decision="FAIL"),
                    "decision does not follow",
                ),
                "passes": (
                    lambda v: gof(v)["passes"].update(data_passes=2),
                    "outside the planner's bounds",
                ),
                "roles": (
                    lambda v: gof(v)["passes"].update(roles=["grid"]),
                    "pass roles are not a valid sequence",
                ),
                "elapsed": (lambda v: gof(v).update(elapsed_seconds=0.0), "elapsed time invalid"),
            }
        )

    def test_a_full_run_must_certify_the_well_specified_fit(self) -> None:
        def mutate(value: dict[str, Any]) -> None:
            gof = value["inference"]["gof"]
            gof.update(ks_lower=0.0015, ks_upper=0.0017, decision="FAIL")

        self.reject(mutate, "must be certified within the threshold", kind="full")

        def inconclusive(value: dict[str, Any]) -> None:
            value["inference"]["gof"].update(
                ks_lower=0.0008, ks_upper=0.0011, decision="INCONCLUSIVE"
            )

        self.reject(inconclusive, "must be certified within the threshold", kind="full")

    def test_a_smoke_run_may_report_any_consistent_decision(self) -> None:
        artifact = _base("smoke")
        artifact["inference"]["gof"].update(ks_lower=0.0015, ks_upper=0.0017, decision="FAIL")
        _seal(artifact)
        errors = CHECKER.validate(artifact, expected_git_sha=_head(), smoke=True, repo_root=REPO)
        self.assertEqual(errors, [])

    def test_profile_interval(self) -> None:
        def profile(value: dict[str, Any]) -> dict[str, Any]:
            return value["inference"]["profile"]

        self.reject_all(
            {
                "keys": (lambda v: profile(v).update(extra=1), "schema keys invalid"),
                "rows": (lambda v: profile(v).update(rows=1), "not for the Weibull fit"),
                "family": (
                    lambda v: profile(v).update(family="gumbel_right"),
                    "not for the Weibull",
                ),
                "level": (lambda v: profile(v).update(level=0.9), "level is not the declared"),
                "executor": (
                    lambda v: profile(v)["executor"].update(requested_workers=1),
                    "executor facts do not match",
                ),
                "parameter names": (
                    lambda v: profile(v)["intervals"].pop("shape"),
                    "parameters invalid",
                ),
                "lower above estimate": (
                    lambda v: profile(v)["intervals"]["shape"].update(lower=1.5),
                    "does not enclose its estimate",
                ),
                "upper below estimate": (
                    lambda v: profile(v)["intervals"]["scale"].update(upper=1.0),
                    "does not enclose its estimate",
                ),
                "non-positive lower": (
                    lambda v: profile(v)["intervals"]["scale"].update(lower=0.0),
                    "does not enclose its estimate",
                ),
                "unbounded": (
                    lambda v: profile(v)["intervals"]["shape"].update(upper_unbounded=True),
                    "has an unbounded end",
                ),
                "other fit": (
                    lambda v: profile(v)["intervals"]["shape"].update(
                        estimate_hex=(1.0000001).hex(), lower=0.5, upper=1.5
                    ),
                    "not computed at the measured fit",
                ),
                "too wide": (
                    lambda v: profile(v)["intervals"]["shape"].update(lower=0.1, upper=5.0),
                    "wider than the sample size allows",
                ),
                "interval keys": (
                    lambda v: profile(v)["intervals"]["shape"].update(extra=1),
                    "schema keys invalid",
                ),
                "degree": (
                    lambda v: profile(v)["resolution"].update(surrogate_degree=12),
                    "resolution is invalid",
                ),
                "residual": (
                    lambda v: profile(v)["resolution"].update(confirmation_residual=1.0),
                    "beyond its tolerance",
                ),
                "tolerance": (
                    lambda v: profile(v)["resolution"].update(tolerance=0.0),
                    "resolution is invalid",
                ),
                "resolution keys": (
                    lambda v: profile(v)["resolution"].pop("surrogate_gap"),
                    "schema keys invalid",
                ),
                "passes below minimum": (
                    lambda v: profile(v)["passes"].update(
                        data_passes=1, roles=["profile_confirmation"]
                    ),
                    "outside the planner's bounds",
                ),
                "passes over budget": (
                    lambda v: profile(v)["passes"].update(data_passes=7),
                    "outside the planner's bounds",
                ),
                "roles": (
                    lambda v: profile(v)["passes"].update(
                        roles=["profile_confirmation", "profile_window", "profile_refinement"]
                    ),
                    "pass roles are not a valid sequence",
                ),
                "elapsed": (
                    lambda v: profile(v).update(elapsed_seconds=-1.0),
                    "elapsed time invalid",
                ),
            }
        )

    def test_profile_interval_of_a_full_ladder_is_at_the_rung_below_the_limit(self) -> None:
        def mutate(value: dict[str, Any]) -> None:
            value["inference"]["profile"]["rows"] = CHECKER.TOP_RUNG

        self.reject(mutate, "not for the Weibull fit", kind="top")

    # ------------------------------------------------------------------ robustness

    def test_malformed_artifacts_are_reported_and_never_raise(self) -> None:
        for garbage in (None, [], "text", 3, {}, {"schema_version": "1"}):
            with self.subTest(garbage=garbage):
                errors = CHECKER.validate(
                    garbage, expected_git_sha=_head(), smoke=True, repo_root=REPO
                )
                self.assertTrue(errors)

        def deep(path: tuple[Any, ...], replacement: object) -> None:
            artifact = _base("smoke")
            target: Any = artifact
            for step in path[:-1]:
                target = target[step]
            target[path[-1]] = replacement
            _seal(artifact)
            errors = CHECKER.validate(
                artifact, expected_git_sha=_head(), smoke=True, repo_root=REPO
            )
            self.assertTrue(errors, path)

        for path in (
            ("rungs", 0, "cells", 0, "executor"),
            ("rungs", 0, "cells", 0, "digests"),
            ("rungs", 0, "cells", 3, "passes"),
            ("rungs", 0, "datasets", "a"),
            ("rungs", 0, "datasets", "b", "files"),
            ("rungs", 0, "invariance", 0, "resumed"),
            ("rungs", 0, "references", 0, "parameters"),
            ("mpmath_reference", "families", 0),
            ("inference", "gof", "passes"),
            ("inference", "profile", "intervals"),
            ("run", "environment"),
            ("run", "ladder"),
        ):
            for replacement in (None, 7, "x", []):
                with self.subTest(path=path, replacement=replacement):
                    deep(path, replacement)

    def test_cli_accepts_and_rejects_by_exit_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            path.write_text(json.dumps(_base("smoke")), encoding="utf-8")
            command = [
                sys.executable,
                str(CHECKER_PATH),
                "--artifact",
                str(path),
                "--expected-git-sha",
                _head(),
                "--repo-root",
                str(REPO),
            ]
            accepted = subprocess.run(
                [*command, "--smoke"], check=False, capture_output=True, text=True
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn("PASS", accepted.stdout)
            rejected = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(rejected.returncode, 1)
            self.assertIn("row ladder is not the required one", rejected.stderr)
            with_top = subprocess.run(
                [*command, "--smoke", "--top-rung"], check=False, capture_output=True, text=True
            )
            self.assertEqual(with_top.returncode, 1)
            path.write_text("{", encoding="utf-8")
            unreadable = subprocess.run(command, check=False, capture_output=True, text=True)
            self.assertEqual(unreadable.returncode, 1)
            self.assertIn("cannot read artifact", unreadable.stderr)
            missing = subprocess.run(
                [*command[:3], str(Path(temporary) / "none.json"), *command[4:]],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(missing.returncode, 1)

    def test_cli_requires_a_repository_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "artifact.json"
            path.write_text(json.dumps(_base("smoke")), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER_PATH),
                    "--artifact",
                    str(path),
                    "--expected-git-sha",
                    _head(),
                    "--smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--repo-root", result.stderr)


class FrozenContractTests(unittest.TestCase):
    """The checker's frozen constants are the engine's and the runner's."""

    def test_the_planner_bounds_are_the_engines(self) -> None:
        self.assertEqual(CHECKER.FIT_PASS_BUDGET, _planner.PASS_BUDGET_DEFAULT)
        self.assertEqual(CHECKER.FIT_MINIMUM_PASSES, _planner.MINIMUM_PASSES)
        self.assertEqual(CHECKER.FIT_MAXIMUM_PASSES, _planner.MAXIMUM_PASSES)
        self.assertEqual(
            CHECKER.PROFILE_PASS_BUDGET, _profile_intervals.PROFILE_PASS_BUDGET_DEFAULT
        )
        self.assertEqual(CHECKER.PROFILE_MINIMUM_PASSES, _profile_intervals.PROFILE_MIN_PASSES)
        self.assertEqual(CHECKER.PROFILE_MAXIMUM_PASSES, _profile_intervals.PROFILE_MAX_PASSES)
        self.assertEqual(CHECKER.PROFILE_DEGREES, _profile_math.SURROGATE_DEGREES)
        self.assertEqual(CHECKER.MAX_ROWS_PER_BATCH, DEFAULT_MAX_ROWS)
        self.assertEqual(CHECKER.MAX_WORKERS, MAX_WORKERS)
        self.assertEqual(CHECKER.GOF_BIN_EXPONENT, scale_gof.DEFAULT_BIN_EXPONENT)
        self.assertEqual(
            CHECKER.EXPONENTIAL_CDF_ERROR, scale_gof.cdf_error_bound(FamilyId.EXPONENTIAL, {})
        )

    def test_the_parameter_names_are_the_registrys(self) -> None:
        for family in CHECKER.FAMILIES:
            spec = FAMILY_REGISTRY.lookup(family)
            self.assertEqual(
                CHECKER.PARAMETER_NAMES[family],
                tuple(parameter.name for parameter in spec.parameters),
                family,
            )

    def test_checker_and_runner_declare_the_same_contract(self) -> None:
        for name in (
            "STANDARD_LADDER",
            "TOP_RUNG",
            "PROFILE_MAX_ROWS",
            "MULTI_PASS_MAX_ROWS",
            "FAMILIES",
            "ONE_PASS_FAMILIES",
            "MULTI_PASS_FAMILIES",
            "EXECUTORS",
            "PARAMETER_NAMES",
            "GENERATOR",
            "METHODOLOGY",
            "SUBSAMPLE_ROWS",
            "GOF_BIN_EXPONENT",
            "GOF_THRESHOLD",
            "PROFILE_LEVEL",
            "MPMATH_SAMPLE_ROWS",
            "MPMATH_DIGITS",
        ):
            with self.subTest(name=name):
                self.assertEqual(getattr(CHECKER, name), getattr(RUNNER, name))
        self.assertEqual(
            CHECKER.TIMING, {"clock": RUNNER.ELAPSED_CLOCK, "preflight": RUNNER.ELAPSED_PREFLIGHT}
        )
        self.assertEqual(CHECKER.PROFILE_MAXIMUM_PASSES, _profile_intervals.PROFILE_MAX_PASSES)
        for rows in (1_000, 20_000, 999_999, 1_000_000, 10**8, 10**9):
            with self.subTest(rows=rows):
                self.assertEqual(CHECKER.layouts(rows), RUNNER.layouts(rows))
                self.assertEqual(CHECKER.families_at(rows), RUNNER.families_at(rows))

    def test_the_fixed_sample_and_its_references_are_independent_of_the_checker(self) -> None:
        self.assertEqual(CHECKER.mpmath_sample_sha256(), RUNNER.mpmath_sample_sha256())
        sample = RUNNER.mpmath_sample()
        self.assertEqual(len(sample), CHECKER.MPMATH_SAMPLE_ROWS)
        self.assertTrue(0.5 <= min(sample) and max(sample) < 2.5)
        self.assertEqual(
            hashlib.sha256(np.asarray(sample, dtype="<f8").tobytes()).hexdigest(),
            CHECKER.mpmath_sample_sha256(),
        )

    def test_the_frozen_reference_values_are_what_mpmath_computes(self) -> None:
        self.assertEqual(RUNNER.mpmath_references(), CHECKER.MPMATH_REFERENCE)

    def test_the_interruption_point_leaves_work_on_both_sides(self) -> None:
        for partitions in (2, 16, 96, 954):
            self.assertEqual(CHECKER.interruption_point("gamma", partitions), partitions // 2)
            self.assertEqual(
                CHECKER.interruption_point("weibull_min", partitions), partitions + partitions // 2
            )
            self.assertGreater(CHECKER.interruption_point("gamma", partitions), 0)
            self.assertLess(CHECKER.interruption_point("gamma", partitions), partitions)
            self.assertLess(CHECKER.interruption_point("gumbel_right", partitions), 2 * partitions)


class RunnerDataTests(unittest.TestCase):
    def test_layouts_differ_and_cover_the_same_rows(self) -> None:
        for rows in (1_000, 20_000, 1_000_000):
            geometry = RUNNER.layouts(rows)
            with self.subTest(rows=rows):
                self.assertNotEqual(geometry["a"], geometry["b"])
                self.assertNotEqual(
                    geometry["a"]["row_group_rows"], geometry["b"]["row_group_rows"]
                )
                self.assertEqual(geometry["a"]["order"], "ascending")
                self.assertEqual(geometry["b"]["order"], "reversed")

    def test_the_data_are_deterministic_and_independent_of_the_layout(self) -> None:
        rows = 20_000
        geometry = RUNNER.layouts(rows)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = RUNNER.write_layout(root / "a", rows, geometry["a"])
            second = RUNNER.write_layout(root / "a2", rows, geometry["a"])
            other = RUNNER.write_layout(root / "b", rows, geometry["b"])
            self.assertEqual(first, second)
            self.assertNotEqual(first["files"], other["files"])
            self.assertNotEqual(first["partitions"], other["partitions"])
            self.assertEqual(first["rows"], rows)
            self.assertEqual(sum(item["bytes"] for item in first["files"]) > rows * 4, True)
            for name, record in (("a", first), ("b", other)):
                files = sorted((root / name).glob("*.parquet"))
                self.assertEqual(len(files), len(record["files"]))
                groups = sum(pq.ParquetFile(path).metadata.num_row_groups for path in files)
                self.assertEqual(groups, record["partitions"])
            a = np.sort(
                np.concatenate(
                    [
                        pq.read_table(p).column(0).to_numpy()
                        for p in sorted((root / "a").glob("*.parquet"))
                    ]
                )
            )
            b = np.sort(
                np.concatenate(
                    [
                        pq.read_table(p).column(0).to_numpy()
                        for p in sorted((root / "b").glob("*.parquet"))
                    ]
                )
            )
            self.assertEqual(a.shape, (rows,))
            self.assertTrue(np.array_equal(a, b))
            self.assertTrue(np.all(a > 0.0) and np.all(np.isfinite(a)))
            self.assertLess(
                abs(a.mean() - RUNNER.DISTRIBUTION_SCALE) / RUNNER.DISTRIBUTION_SCALE, 0.05
            )

    def test_the_rung_data_differ_between_rungs(self) -> None:
        self.assertFalse(np.array_equal(RUNNER._block(20_000, 0), RUNNER._block(30_000, 0)))
        self.assertTrue(np.array_equal(RUNNER._block(20_000, 3), RUNNER._block(20_000, 3)))
        self.assertFalse(np.array_equal(RUNNER._block(20_000, 3), RUNNER._block(20_000, 4)))

    def test_layout_files_hold_no_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            record = RUNNER.write_layout(Path(temporary) / "a", 1_000, RUNNER.layouts(1_000)["a"])
        self.assertFalse(CHECKER._contains_path(record))

    def test_a_layout_is_not_written_over_an_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileExistsError):
                RUNNER.write_layout(Path(temporary), 1_000, RUNNER.layouts(1_000)["a"])

    def test_the_disk_is_checked_before_a_rung_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(RUNNER, "_free_bytes", return_value=1 << 20):
                with self.assertRaisesRegex(RuntimeError, "not enough free disk space"):
                    RUNNER._require_disk(Path(temporary), 10**9)
            with patch.object(RUNNER, "_free_bytes", return_value=10 * (1 << 30)):
                RUNNER._require_disk(Path(temporary), 10**9)

    def test_ladder_selection(self) -> None:
        self.assertEqual(RUNNER.ladder(None, top_rung=False), list(CHECKER.STANDARD_LADDER))
        self.assertEqual(
            RUNNER.ladder(None, top_rung=True), [*CHECKER.STANDARD_LADDER, CHECKER.TOP_RUNG]
        )
        self.assertEqual(RUNNER.ladder("10000,20000", top_rung=False), [10_000, 20_000])
        for bad in ("500", "20000,10000", "10000,10000", "10000,20000,30000,40000", "x"):
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                RUNNER.ladder(bad, top_rung=False)
        with self.assertRaises(ValueError):
            RUNNER.ladder("10000,20000", top_rung=True)

    def test_the_multi_pass_families_stop_before_the_top_rung(self) -> None:
        self.assertEqual(RUNNER.families_at(100_000_000), RUNNER.FAMILIES)
        self.assertEqual(RUNNER.families_at(100_000_001), RUNNER.ONE_PASS_FAMILIES)
        self.assertEqual(RUNNER.families_at(10**9), RUNNER.ONE_PASS_FAMILIES)


class RunnerClockTests(unittest.TestCase):
    def test_elapsed_clock_pair_rejects_cross_domain_disagreement(self) -> None:
        self.assertEqual(RUNNER._paired_elapsed_seconds(1_000, 1_100, 8_000, 8_100), 0.0000001)
        with self.assertRaisesRegex(RuntimeError, "clocks disagree"):
            RUNNER._paired_elapsed_seconds(1_000, 1_000_000_000, 8_000, 72_008_000_000_000)

    def test_elapsed_uses_the_monotonic_clock_when_the_wall_clock_is_coarse(self) -> None:
        self.assertEqual(RUNNER._paired_elapsed_seconds(5_000, 5_000, 8_000, 8_250), 0.00000025)
        with self.assertRaisesRegex(RuntimeError, "below the monotonic clock resolution"):
            RUNNER._paired_elapsed_seconds(5_000, 5_000, 8_000, 8_000)
        with self.assertRaisesRegex(RuntimeError, "moved backwards"):
            RUNNER._paired_elapsed_seconds(5_000, 4_000, 8_000, 9_000)


class RunnerMeasurementTests(unittest.TestCase):
    ROWS = 20_000

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        root = Path(cls._directory.name)
        cls.data = root / "a"
        cls.record = RUNNER.write_layout(cls.data, cls.ROWS, RUNNER.layouts(cls.ROWS)["a"])
        cls.root = root

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_the_timed_pass_is_untraced_and_the_memory_pass_is_separate(self) -> None:
        tracing: list[bool] = []
        real_fit = RUNNER._fit

        def spy(*arguments: Any, **options: Any) -> Any:
            tracing.append(tracemalloc.is_tracing())
            return real_fit(*arguments, **options)

        calls: list[dict[str, Any]] = []

        def memory(request: dict[str, Any]) -> dict[str, Any]:
            calls.append(request)
            result = real_fit(self.data, "normal", "serial", 1)
            facts = RUNNER._result_facts(result, self.ROWS, "normal")
            return {
                "tracemalloc_peak_bytes": 1,
                "rss_baseline_bytes": 10,
                "rss_peak_bytes": 20,
                "digests": facts["digests"],
            }

        with (
            patch.object(RUNNER, "_fit", side_effect=spy),
            patch.object(RUNNER, "_memory_pass", side_effect=memory),
        ):
            cell, result = RUNNER._cell(self.data, self.ROWS, "normal", "serial", 1)
        self.assertEqual(tracing, [False])
        self.assertFalse(tracemalloc.is_tracing())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["family"], "normal")
        self.assertEqual(calls[0]["executor"], "serial")
        self.assertEqual(
            cell["memory"],
            {"tracemalloc_peak_bytes": 1, "rss_baseline_bytes": 10, "rss_peak_bytes": 20},
        )
        self.assertEqual(cell["rows"], self.ROWS)
        self.assertEqual(cell["partitions"], self.record["partitions"])
        self.assertGreater(cell["elapsed_seconds"], 0)
        self.assertEqual(
            cell["executor"], {"kind": "serial", "requested_workers": 1, "effective_workers": 1}
        )
        self.assertEqual(cell["passes"]["data_passes"], 1)
        self.assertFalse(CHECKER._contains_path(cell))
        self.assertTrue(result.fit.converged)  # type: ignore[union-attr]

    def test_a_memory_pass_that_differs_from_the_timed_fit_is_refused(self) -> None:
        def memory(request: dict[str, Any]) -> dict[str, Any]:
            return {
                "tracemalloc_peak_bytes": 1,
                "rss_baseline_bytes": 10,
                "rss_peak_bytes": 20,
                "digests": {
                    "state_sha256": "0" * 64,
                    "result_sha256": "0" * 64,
                    "passes_sha256": "0" * 64,
                },
            }

        with patch.object(RUNNER, "_memory_pass", side_effect=memory):
            with self.assertRaisesRegex(RuntimeError, "memory pass did not reproduce"):
                RUNNER._cell(self.data, self.ROWS, "exponential", "serial", 1)

    def test_the_memory_child_reports_traced_and_resident_memory(self) -> None:
        request = {
            "directory": str(self.data),
            "family": "gamma",
            "executor": "serial",
            "workers": 1,
            "rows": self.ROWS,
        }
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            self.assertEqual(RUNNER._memory_child(request), 0)
        facts = json.loads(buffer.getvalue().strip().splitlines()[-1])
        self.assertGreater(facts["tracemalloc_peak_bytes"], 0)
        self.assertGreater(facts["rss_baseline_bytes"], 1_000_000)
        self.assertGreaterEqual(facts["rss_peak_bytes"], facts["rss_baseline_bytes"])
        timed = RUNNER._result_facts(
            RUNNER._fit(self.data, "gamma", "serial", 1), self.ROWS, "gamma"
        )
        self.assertEqual(facts["digests"], timed["digests"])
        self.assertFalse(tracemalloc.is_tracing())

    def test_the_memory_pass_runs_in_a_fresh_process(self) -> None:
        facts = RUNNER._memory_pass(
            {
                "directory": str(self.data),
                "family": "exponential",
                "executor": "serial",
                "workers": 1,
                "rows": self.ROWS,
            }
        )
        self.assertEqual(
            set(facts),
            {"tracemalloc_peak_bytes", "rss_baseline_bytes", "rss_peak_bytes", "digests"},
        )
        self.assertGreater(facts["rss_peak_bytes"], 0)

    def test_a_failing_memory_pass_is_reported(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "memory pass failed"):
            RUNNER._memory_pass(
                {
                    "directory": str(self.root / "missing"),
                    "family": "exponential",
                    "executor": "serial",
                    "workers": 1,
                    "rows": self.ROWS,
                }
            )

    def test_result_facts_refuse_a_fit_that_is_not_the_rung(self) -> None:
        result = RUNNER._fit(self.data, "exponential", "serial", 1)
        with self.assertRaisesRegex(RuntimeError, "did not read the rung"):
            RUNNER._result_facts(result, self.ROWS + 1, "exponential")
        with self.assertRaisesRegex(RuntimeError, "parameters are not the declared ones"):
            RUNNER._result_facts(result, self.ROWS, "gamma")

    def test_result_facts_are_what_the_checker_recomputes(self) -> None:
        for family in ("exponential", "weibull_min"):
            facts = RUNNER._result_facts(
                RUNNER._fit(self.data, family, "serial", 1), self.ROWS, family
            )
            with self.subTest(family=family):
                self.assertEqual(
                    facts["digests"]["result_sha256"],
                    CHECKER._result_digest(
                        {
                            "observation_count": facts["result"]["observation_count"],
                            "parameters_hex": facts["result"]["parameters_hex"],
                            "log_likelihood_hex": facts["result"]["log_likelihood_hex"],
                        },
                        family,
                    ),
                )
                multi = family == "weibull_min"
                self.assertEqual(facts["passes"]["minimum"], 3 if multi else 1)
                self.assertEqual(bool(facts["passes"]["roles"]), multi)

    def test_the_interrupting_store_stops_after_the_declared_commits(self) -> None:
        store = RUNNER._InterruptingStore(self.root / "store.db", after=2)
        arguments = (0, "k", "p", "sha256:" + "0" * 64, "metadata", b"x")
        with patch.object(
            RUNNER.SQLitePartitionStore, "commit_partition", return_value=True
        ) as commit:
            store.commit_partition(*arguments)
            store.commit_partition(*arguments)
            with self.assertRaises(RUNNER._Interrupted):
                store.commit_partition(*arguments)
        self.assertEqual(commit.call_count, 2)

    def test_an_interrupted_run_resumes_to_the_digests_of_the_whole_run(self) -> None:
        for family in ("lognormal", "gumbel_right"):
            whole = RUNNER._result_facts(
                RUNNER._fit(self.data, family, "serial", 1), self.ROWS, family
            )
            record = RUNNER._resumed(self.data, self.ROWS, family, 2, self.record["partitions"])
            with self.subTest(family=family):
                self.assertEqual(record["digests"], whole["digests"])
                multi = family in CHECKER.MULTI_PASS_FAMILIES
                after = CHECKER.interruption_point(family, self.record["partitions"])
                self.assertEqual(record["interrupted_after_commits"], after)
                self.assertEqual(record["reused_partitions"], after)
                self.assertEqual(record["replayed_passes"], 1 if multi else 0)
                self.assertEqual(
                    record["reused_partitions"] + record["read_partitions"],
                    record["passes_after_resume"] * self.record["partitions"],
                )
                self.assertEqual(record["passes_after_resume"], whole["passes"]["data_passes"])
        self.assertEqual(list(self.root.glob("*.db")), [])

    def test_an_interrupted_run_that_completes_is_an_error(self) -> None:
        with patch.object(RUNNER, "_InterruptingStore") as store:
            store.return_value = str(self.root / "unused.db")
            with self.assertRaisesRegex(RuntimeError, "interrupted run completed"):
                RUNNER._resumed(self.data, self.ROWS, "exponential", 2, self.record["partitions"])
        for leftover in self.root.glob("*.db"):
            leftover.unlink()

    def test_subsample_references_agree_with_the_in_memory_fits(self) -> None:
        records = RUNNER._subsample_references(self.ROWS, self.root, RUNNER.ONE_PASS_FAMILIES)
        self.assertEqual([item["family"] for item in records], list(RUNNER.ONE_PASS_FAMILIES))
        for item in records:
            with self.subTest(family=item["family"]):
                self.assertEqual(item["subsample_rows"], self.ROWS)
                tolerance = CHECKER.IN_MEMORY_PARAMETER_TOLERANCE[item["family"]]
                for difference in item["parameters"].values():
                    self.assertLessEqual(difference["relative_difference"], tolerance)
        self.assertFalse((self.root / "subsample").exists())

    def test_the_fixed_sample_agrees_with_the_references(self) -> None:
        block = RUNNER._mpmath_block(self.root)
        self.assertEqual(block["sample"]["sha256"], CHECKER.mpmath_sample_sha256())
        for family in block["families"]:
            for name, item in family["parameters"].items():
                with self.subTest(family=family["family"], name=name):
                    self.assertEqual(
                        item["reference"], CHECKER.MPMATH_REFERENCE[family["family"]][name]
                    )
                    self.assertLessEqual(
                        item["relative_difference"], CHECKER.MPMATH_TOLERANCE[family["family"]]
                    )


class RunnerMainTests(unittest.TestCase):
    """The orchestration of ``main`` with the measured parts replaced."""

    @staticmethod
    def _fake_rung(calls: list[dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        def fake(
            rows: int, root: Path, workers: int, *, gof: bool, profile: bool
        ) -> dict[str, Any]:
            calls.append({"rows": rows, "workers": workers, "gof": gof, "profile": profile})
            inference: dict[str, Any] = {}
            if gof:
                inference["gof"] = {"rows": rows}
            if profile:
                inference["profile"] = {"rows": rows}
            return {"rows": rows, "inference": inference}

        return fake

    def _run(self, *arguments: str, shas: tuple[str, ...] = ("a" * 40, "a" * 40)) -> dict[str, Any]:
        calls: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            command = ["runner", "--output", str(output), *arguments]
            facts = iter(shas)
            with (
                patch.object(RUNNER, "_clean_checkout_sha", side_effect=lambda _root: next(facts)),
                patch.object(RUNNER, "_rung", side_effect=self._fake_rung(calls)),
                patch.object(RUNNER, "_mpmath_block", return_value={"families": []}),
                patch.object(sys, "argv", command),
            ):
                self.assertEqual(RUNNER.main(), 0)
            artifact = json.loads(output.read_text(encoding="utf-8"))
        artifact["calls"] = calls
        return artifact

    def test_the_default_ladder_stops_before_the_top_rung(self) -> None:
        artifact = self._run()
        self.assertEqual(
            [call["rows"] for call in artifact["calls"]], list(CHECKER.STANDARD_LADDER)
        )
        self.assertFalse(artifact["run"]["ladder"]["top_rung"])
        self.assertEqual(
            [(call["rows"], call["gof"], call["profile"]) for call in artifact["calls"]],
            [(1_000_000, False, False), (10_000_000, False, False), (100_000_000, True, True)],
        )

    def test_the_top_rung_flag_adds_the_top_rung_and_moves_the_goodness_of_fit(self) -> None:
        artifact = self._run("--top-rung")
        self.assertEqual(
            [(call["rows"], call["gof"], call["profile"]) for call in artifact["calls"]],
            [
                (1_000_000, False, False),
                (10_000_000, False, False),
                (100_000_000, False, True),
                (1_000_000_000, True, False),
            ],
        )
        self.assertTrue(artifact["run"]["ladder"]["top_rung"])
        self.assertEqual(artifact["inference"]["gof"]["rows"], 1_000_000_000)
        self.assertEqual(artifact["inference"]["profile"]["rows"], 100_000_000)

    def test_the_process_executor_defaults_to_the_core_count(self) -> None:
        with patch.object(os, "cpu_count", return_value=6):
            artifact = self._run("--rows", "10000,20000")
        self.assertEqual({call["workers"] for call in artifact["calls"]}, {6})
        self.assertEqual(artifact["run"]["environment"]["cpu_count"], 6)
        self.assertEqual(artifact["run"]["environment"]["process_workers"], 6)
        artifact = self._run("--rows", "10000", "--process-workers", "2")
        self.assertEqual({call["workers"] for call in artifact["calls"]}, {2})

    def test_the_artifact_declares_its_methodology_and_candidate(self) -> None:
        artifact = self._run("--rows", "10000")
        run = artifact["run"]
        self.assertEqual(artifact["schema_version"], "1")
        self.assertEqual(run["measurement_workers"], 1)
        self.assertEqual(run["candidate_git_sha"], "a" * 40)
        self.assertEqual(run["git_sha"], "a" * 40)
        self.assertIs(run["git_dirty"], False)
        self.assertEqual(run["methodology"], CHECKER.METHODOLOGY)
        self.assertEqual(run["timing"], CHECKER.TIMING)
        self.assertEqual(artifact["generator"], CHECKER.GENERATOR)
        self.assertEqual(
            run["limits"],
            {"max_rows_per_batch": 262_144, "pass_budget": 8, "inference_pass_budget": 6},
        )
        self.assertEqual(
            artifact["artifact_sha256"],
            CHECKER._digest({key: value for key, value in artifact.items() if key != "calls"}),
        )
        environment = run["environment"]
        self.assertEqual(len(environment["numeric_environment_digest"]), 64)
        self.assertTrue(environment["pyarrow_version"])
        self.assertFalse(CHECKER._contains_path(artifact["run"]))

    def test_the_run_is_refused_from_a_dirty_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            with (
                patch.object(RUNNER, "_git", return_value=" M python/tools/x.py"),
                patch.object(sys, "argv", ["runner", "--output", str(output), "--rows", "10000"]),
            ):
                with self.assertRaises(SystemExit) as failure:
                    RUNNER.main()
            self.assertIn("dirty checkout", str(failure.exception))
            self.assertFalse(output.exists())

    def test_the_output_is_refused_when_head_changes_mid_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            facts = iter(["a" * 40, "f" * 40])
            with (
                patch.object(RUNNER, "_clean_checkout_sha", side_effect=lambda _root: next(facts)),
                patch.object(RUNNER, "_rung", side_effect=self._fake_rung([])),
                patch.object(RUNNER, "_mpmath_block", return_value={}),
                patch.object(sys, "argv", ["runner", "--output", str(output), "--rows", "10000"]),
            ):
                with self.assertRaises(SystemExit) as failure:
                    RUNNER.main()
            self.assertIn("HEAD changed", str(failure.exception))
            self.assertFalse(output.exists())

    def test_the_output_is_refused_when_the_checkout_becomes_dirty(self) -> None:
        def clean(_root: Path) -> str:
            if not calls:
                calls.append(1)
                return "a" * 40
            raise RuntimeError("refusing evidence run from a dirty checkout")

        calls: list[int] = []
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            with (
                patch.object(RUNNER, "_clean_checkout_sha", side_effect=clean),
                patch.object(RUNNER, "_rung", side_effect=self._fake_rung([])),
                patch.object(RUNNER, "_mpmath_block", return_value={}),
                patch.object(sys, "argv", ["runner", "--output", str(output), "--rows", "10000"]),
            ):
                with self.assertRaises(SystemExit) as failure:
                    RUNNER.main()
            self.assertIn("checkout changed", str(failure.exception))
            self.assertFalse(output.exists())

    def test_bad_arguments_are_refused(self) -> None:
        for arguments in (
            ["--rows", "10000", "--top-rung"],
            ["--rows", "500"],
            ["--rows", "10000", "--process-workers", "0"],
        ):
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "artifact.json"
                with (
                    patch.object(RUNNER, "_clean_checkout_sha", return_value="a" * 40),
                    patch.object(sys, "argv", ["runner", "--output", str(output), *arguments]),
                ):
                    with self.assertRaises(SystemExit):
                        RUNNER.main()
                self.assertFalse(output.exists())

    def test_a_rung_measures_the_declared_families_and_removes_its_data(self) -> None:
        rows = 20_000
        written: list[str] = []
        real_write = RUNNER.write_layout

        def write(directory: Path, count: int, layout: dict[str, Any]) -> dict[str, Any]:
            written.append(directory.name)
            return real_write(directory, count, layout)

        def cell(directory: Path, count: int, family: str, kind: str, workers: int) -> Any:
            fit = RUNNER._fit(directory, family, kind, workers)
            facts = RUNNER._result_facts(fit, count, family)
            return {
                "family": family,
                "executor": facts["executor"],
                "digests": facts["digests"],
                "elapsed_seconds": 1.0,
            }, fit

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                patch.object(RUNNER, "write_layout", side_effect=write),
                patch.object(RUNNER, "_cell", side_effect=cell),
                patch.object(RUNNER, "families_at", return_value=("exponential",)),
                patch.object(RUNNER, "_subsample_references", return_value=[]),
            ):
                rung = RUNNER._rung(rows, root, 2, gof=True, profile=False)
            self.assertEqual(list(root.iterdir()), [])
        self.assertEqual(written, [f"rung-{rows}-a", f"rung-{rows}-b"])
        self.assertEqual(set(rung["inference"]), {"gof"})
        self.assertEqual(rung["inference"]["gof"]["family"], "exponential")
        self.assertEqual(rung["inference"]["gof"]["rows"], rows)
        self.assertEqual([item["family"] for item in rung["invariance"]], ["exponential"])
        entry = rung["invariance"][0]
        digests = [cell_["digests"] for cell_ in rung["cells"]]
        self.assertEqual(entry["layout_b"]["digests"], digests[0])
        self.assertEqual(entry["resumed"]["digests"], digests[0])
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(
            {cell_["executor"]["kind"] for cell_ in rung["cells"]}, {"serial", "processes"}
        )
        self.assertNotEqual(rung["datasets"]["a"]["files"], rung["datasets"]["b"]["files"])


class RealRunTests(unittest.TestCase):
    """The runner as a script on a tiny ladder, the artifact checked by the checker CLI.

    The runner refuses a dirty checkout, so this test needs a clean tree (CI has one).
    """

    @staticmethod
    def _environment() -> dict[str, str]:
        environment = dict(os.environ)
        source = str(Path(__file__).parents[2] / "src")
        environment["PYTHONPATH"] = source + os.pathsep + environment.get("PYTHONPATH", "")
        return environment

    def test_the_runner_produces_an_artifact_the_checker_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "artifact.json"
            generated = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER_PATH),
                    "--output",
                    str(output),
                    "--rows",
                    "20000",
                    "--process-workers",
                    "2",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=self._environment(),
            )
            self.assertEqual(generated.returncode, 0, generated.stderr)
            artifact = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(artifact["run"]["measurement_workers"], 1)
            self.assertEqual(artifact["run"]["ladder"], {"rows": [20_000], "top_rung": False})
            self.assertEqual(len(artifact["rungs"]), 1)
            self.assertEqual(len(artifact["rungs"][0]["cells"]), 12)
            checked = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER_PATH),
                    "--artifact",
                    str(output),
                    "--expected-git-sha",
                    _head(),
                    "--repo-root",
                    str(REPO),
                    "--smoke",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertEqual(
                CHECKER.validate(artifact, expected_git_sha=_head(), smoke=True, repo_root=REPO), []
            )
            for cell in artifact["rungs"][0]["cells"]:
                self.assertGreater(cell["memory"]["rss_peak_bytes"], 1_000_000)
                self.assertGreater(cell["elapsed_seconds"], 0)
            invariance = artifact["rungs"][0]["invariance"]
            serial = {
                cell["family"]: cell["digests"]
                for cell in artifact["rungs"][0]["cells"]
                if cell["executor"]["kind"] == "serial"
            }
            for entry in invariance:
                self.assertEqual(entry["layout_b"]["digests"], serial[entry["family"]])
                self.assertEqual(entry["resumed"]["digests"], serial[entry["family"]])
                self.assertGreater(entry["resumed"]["reused_partitions"], 0)


if __name__ == "__main__":
    unittest.main()
