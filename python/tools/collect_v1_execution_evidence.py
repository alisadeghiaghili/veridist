"""Collect one platform's real checkpointed-CSV v1 execution evidence.

This command intentionally produces a *raw platform fragment*, never a
checked-in release assertion.  A workflow must collect fragments on all three
platforms and assemble them only after the matrix has actually run.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import platform as host_platform
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId
from veridist.engine.checkpoint import CheckpointRecord, SQLiteCheckpointStore
from veridist.execution import CheckpointedCsvFitResult, fit_exponential_checkpointed_csv
from veridist.families.exponential import ExponentialFitSuccess

ROWS = (10_000, 100_000, 1_000_000)
SCENARIOS = ("complete", "retry_resume", "cancel")
_SHA_LENGTH = 40
_SCHEMA = CsvLifetimeSchema("time", "event_observed")
_LIMITS = CsvLifetimeLimits(65_536, 65_536)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _clean_candidate_sha(root: Path, candidate_sha: str) -> str:
    """Bind the run to a clean checkout of the dispatch-selected commit."""

    if len(candidate_sha) != _SHA_LENGTH or any(
        char not in "0123456789abcdef" for char in candidate_sha
    ):
        raise RuntimeError("candidate SHA must be a lowercase full Git SHA")
    if _git(root, "status", "--porcelain"):
        raise RuntimeError("refusing execution evidence from a dirty checkout")
    head = _git(root, "rev-parse", "HEAD")
    if head != candidate_sha:
        raise RuntimeError("checkout HEAD does not match the requested candidate SHA")
    return head


def detected_platform() -> Literal["linux", "macos", "windows"]:
    """Return the only platform labels accepted by the evidence schema."""

    name = sys.platform
    if name.startswith("linux"):
        return "linux"
    if name == "darwin":
        return "macos"
    if name == "win32":
        return "windows"
    raise RuntimeError(f"unsupported evidence platform: {name}")


def _rss_bytes() -> int:
    """Read a process RSS fact, or fail rather than report a placeholder."""

    if sys.platform == "win32":

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        getter = ctypes.windll.psapi.GetProcessMemoryInfo
        getter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        getter.restype = ctypes.c_int
        if not getter(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise RuntimeError("cannot obtain Windows process RSS")
        return int(counters.PeakWorkingSetSize)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _source_id(rows: int) -> PublicSourceId:
    digest = hashlib.md5(str(rows).encode("ascii"), usedforsecurity=False).hexdigest()
    return PublicSourceId(f"src_{digest}")


def _write_fixture(path: Path, rows: int) -> tuple[str, int]:
    """Write a deterministic, synthetic lifetime source and return its revision."""

    with path.open("w", encoding="utf-8", newline="") as target:
        target.write("time,event_observed\n")
        for index in range(rows):
            target.write(f"{(index % 997 + 1) / 1000:.3f},{0 if index % 3 == 0 else 1}\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest, path.stat().st_size


def _initial_store(
    path: Path, source_revision: str, source_id: PublicSourceId
) -> SQLiteCheckpointStore:
    state = (
        b'{"compensation":"0x0.0p+0","event_count":0,"observation_count":0,"total_time":"0x0.0p+0"}'
    )
    record = CheckpointRecord.create(
        format_version=1,
        # Must equal the PublicSourceId passed to fit_exponential_checkpointed_csv
        # below (`_source_id(rows)`): the resumable CSV contract now rejects a
        # checkpoint whose recorded source_id does not match the caller's.
        source_id=source_id.value,
        source_schema="csv-lifetime-v1",
        source_revision=source_revision,
        reducer_id="exponential-reduction-v1",
        accumulator_schema="exponential-reduction-v1",
        plan_digest="v1-execution-evidence",
        cursor=0,
        committed_ranges=(),
        generation=0,
        operation_token=None,
        operation_digest=None,
        state=state,
    )
    return SQLiteCheckpointStore.create(path, record)


def _fit_digest(result: CheckpointedCsvFitResult) -> str:
    if result.code != "COMPLETE" or not isinstance(result.fit, ExponentialFitSuccess):
        raise RuntimeError("checkpointed evidence run did not complete")
    fit = result.fit
    value = {
        "event_count": fit.event_count,
        "observation_count": fit.observation_count,
        "rate": fit.rate,
        "total_time": fit.total_time,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _delete_and_verify(paths: tuple[Path, ...]) -> bool:
    """Prove cancellation returned with no open source or checkpoint handles."""

    for path in paths:
        if path.exists():
            path.unlink()
    return all(not path.exists() for path in paths)


def _run_scenario(
    rows: int, scenario: str, baseline_digest: str | None = None
) -> dict[str, object]:
    """Run one real complete, retry/resume, or cancellation observation."""

    if scenario not in SCENARIOS:
        raise ValueError("unknown execution evidence scenario")
    with tempfile.TemporaryDirectory(prefix="veridist-v1-execution-") as directory:
        root = Path(directory)
        source = root / "lifetimes.csv"
        revision, source_bytes = _write_fixture(source, rows)
        before_rss = _rss_bytes()
        resources_released = False
        attempt_count = 0
        interrupted_cursor = 0
        final_cursor = 0
        if scenario == "complete":
            store = _initial_store(root / "complete.sqlite3", revision, _source_id(rows))
            attempt_count += 1
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=_SCHEMA,
                source_id=_source_id(rows),
                limits=_LIMITS,
                store=store,
                source_revision=revision,
                cancel=None,
            )
            result_digest = _fit_digest(result)
            final_cursor = store.read().cursor
            cancellation_observed, retry_initial_code = False, None
        elif scenario == "retry_resume":
            if baseline_digest is None:
                baseline_store = _initial_store(
                    root / "baseline.sqlite3", revision, _source_id(rows)
                )
                baseline = fit_exponential_checkpointed_csv(
                    path=source,
                    schema=_SCHEMA,
                    source_id=_source_id(rows),
                    limits=_LIMITS,
                    store=baseline_store,
                    source_revision=revision,
                    cancel=None,
                )
                baseline_digest = _fit_digest(baseline)
            store = _initial_store(root / "retry.sqlite3", revision, _source_id(rows))
            interrupt_at = max(1, rows // 2)
            attempt_count += 1
            interrupted = fit_exponential_checkpointed_csv(
                path=source,
                schema=_SCHEMA,
                source_id=_source_id(rows),
                limits=_LIMITS,
                store=store,
                source_revision=revision,
                cancel=lambda cursor: cursor >= interrupt_at,
            )
            interrupted_cursor = store.read().cursor
            if interrupted.code != "CANCELLED" or interrupted_cursor != interrupt_at:
                raise RuntimeError("retry/resume interruption did not retain a nonzero checkpoint")
            attempt_count += 1
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=_SCHEMA,
                source_id=_source_id(rows),
                limits=_LIMITS,
                store=SQLiteCheckpointStore(store.path),
                source_revision=revision,
                cancel=None,
            )
            result_digest = _fit_digest(result)
            final_cursor = store.read().cursor
            cancellation_observed, retry_initial_code = True, interrupted.code
        else:
            store = _initial_store(root / "cancel.sqlite3", revision, _source_id(rows))
            interrupt_at = max(1, rows // 2)
            attempt_count += 1
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=_SCHEMA,
                source_id=_source_id(rows),
                limits=_LIMITS,
                store=store,
                source_revision=revision,
                cancel=lambda cursor: cursor >= interrupt_at,
            )
            interrupted_cursor = store.read().cursor
            if result.code != "CANCELLED" or interrupted_cursor != interrupt_at:
                raise RuntimeError("cancellation did not retain the expected nonzero checkpoint")
            result_digest = None
            final_cursor = interrupted_cursor
            cancellation_observed, retry_initial_code = True, None
            resources_released = _delete_and_verify(
                (
                    source,
                    store.path,
                    store.path.with_name(store.path.name + "-wal"),
                    store.path.with_name(store.path.name + "-shm"),
                )
            )
        peak_rss = _rss_bytes()
        if peak_rss < before_rss:
            raise RuntimeError("RSS measurement moved backwards")
        canonical_equal = scenario != "retry_resume" or result_digest == baseline_digest
        if scenario == "retry_resume" and result_digest != baseline_digest:
            raise RuntimeError("complete checkpointed result differs from baseline")
        cell: dict[str, object] = {
            "rows": rows,
            "scenario": scenario,
            "attempt_count": attempt_count,
            "process_peak_rss_bytes": peak_rss,
            "interrupted_cursor": interrupted_cursor,
            "final_cursor": final_cursor,
            "canonical_result_equal": canonical_equal,
            "cancellation_observed": cancellation_observed,
            "resources_released": resources_released,
            "result_code": result.code,
            "retry_initial_code": retry_initial_code,
            "source_bytes": source_bytes,
            "source_sha256": revision,
            "result_sha256": result_digest,
        }
        if scenario == "complete":
            assert result_digest is not None
            cell["_completed_result_digest"] = result_digest
        return cell


def collect_platform(
    platform: str, candidate_sha: str, rows: tuple[int, ...] = ROWS
) -> dict[str, object]:
    """Collect the complete scenario matrix for one asserted host platform."""

    if platform != detected_platform():
        raise RuntimeError("requested evidence platform does not match the host platform")
    if not rows or any(type(row) is not int or row <= 0 for row in rows):
        raise ValueError("rows must contain positive integers")
    cells: list[dict[str, object]] = []
    for row_count in rows:
        complete = _run_scenario(row_count, "complete")
        digest = complete.pop("_completed_result_digest")
        assert isinstance(digest, str)
        for scenario in SCENARIOS:
            cell = (
                complete
                if scenario == "complete"
                else _run_scenario(row_count, scenario, baseline_digest=digest)
            )
            cell["platform"] = platform
            cell["candidate_git_sha"] = candidate_sha
            cells.append(cell)
    return {
        "schema_version": 1,
        "artifact_kind": "v1-execution-raw",
        "candidate_git_sha": candidate_sha,
        "platform": platform,
        "host_platform": host_platform.platform(),
        "python_version": host_platform.python_version(),
        "numpy_version": importlib.metadata.version("numpy"),
        "veridist_version": importlib.metadata.version("veridist"),
        "collected_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "cells": cells,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-git-sha", required=True)
    parser.add_argument("--platform", choices=("linux", "macos", "windows"), required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    try:
        candidate_sha = _clean_candidate_sha(repository, args.candidate_git_sha)
        payload = collect_platform(args.platform, candidate_sha)
        if _clean_candidate_sha(repository, candidate_sha) != candidate_sha:
            raise RuntimeError("candidate changed during collection")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote raw {args.platform} execution evidence: {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
