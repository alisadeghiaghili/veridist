"""Generate deterministic retained evidence for the CSV exponential vertical.

Each cell is measured twice in the measuring process: first an untraced pass that
supplies ``elapsed_seconds`` (``tracemalloc`` slows allocation-heavy code and would
distort timing), then a separate traced pass that supplies the memory facts. Timing
evidence is only meaningful when cells run one at a time, so ``--workers`` defaults to
one, is recorded in the artifact, and the checker rejects any value above one.

``artifact_sha256`` is an integrity digest over the canonical JSON body. It detects
accidental edits and truncation; it is not a signature and does not authenticate who
produced the artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
import time
import tracemalloc
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from veridist.adapters.csv_lifetimes import CsvLifetimeLimits, CsvLifetimeSchema
from veridist.engine.provenance import PublicSourceId
from veridist.execution import ExponentialSourceFitResult, fit_exponential_csv
from veridist.families.exponential import ExponentialFitSuccess

try:  # direct script execution puts tools/ on sys.path
    from process_memory import peak_rss_bytes
except ImportError:  # imported as ``tools.<module>``
    from tools.process_memory import peak_rss_bytes

SCHEMA = CsvLifetimeSchema("time", "event_observed")

# Windows virtualized clocks can disagree after a worker migrates between
# timing domains.  Evidence must fail closed rather than preserve an elapsed
# value that cannot be tied to a trustworthy clock.
ELAPSED_CLOCK = "time.time_ns"
ELAPSED_PREFLIGHT = "paired-wall-monotonic-v1"

# Elapsed time comes from an untraced pass, memory from a separate traced pass, and RSS is
# the process high-water mark (``peak_rss_bytes``), so ``rss_delta_bytes`` is only the growth
# of that mark and is zero when an earlier pass in the process already reached it.
METHODOLOGY = {"passes": "elapsed-untraced-then-memory-traced-v1", "rss": "process-peak-v1"}
_CLOCK_ABSOLUTE_TOLERANCE_NS = 100_000_000
_CLOCK_RELATIVE_TOLERANCE = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _generate(path: Path, rows: int) -> tuple[int, Decimal]:
    """Write the fixture and independently calculate its Decimal sufficient facts."""

    events = 0
    total = Decimal(0)
    with path.open("w", encoding="utf-8", newline="") as target:
        target.write("time,event_observed\n")
        for index in range(rows):
            time_value = Decimal((index % 997) + 1) / Decimal(1000)
            event = 0 if index % 3 == 0 else 1
            target.write(f"{time_value:f},{event}\n")
            total += time_value
            events += event
    return events, total


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _clean_checkout_sha(root: Path) -> str:
    """Fail closed unless this exact checkout is clean, returning its HEAD."""

    if _git(root, "status", "--porcelain"):
        raise RuntimeError("refusing evidence run from a dirty checkout")
    return _git(root, "rev-parse", "HEAD")


def _canonical_digest(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _parse_budgets(value: str) -> list[int]:
    budgets = [int(item) for item in value.split(",")]
    if len(budgets) < 3 or len(set(budgets)) != len(budgets) or any(item <= 0 for item in budgets):
        raise ValueError("chunk budgets must contain at least three distinct positive integers")
    return budgets


def _paired_elapsed_seconds(
    wall_started_ns: int,
    wall_finished_ns: int,
    monotonic_started_ns: int,
    monotonic_finished_ns: int,
) -> float:
    """Return a wall duration only when an independent monotonic clock agrees.

    ``time.time_ns`` is the persisted wall-clock measurement.  The paired
    monotonic duration is not reported as a performance metric; it is a local
    provenance check that prevents a known bad timer domain from producing a
    deceptively precise retained value.
    """

    wall_elapsed = wall_finished_ns - wall_started_ns
    monotonic_elapsed = monotonic_finished_ns - monotonic_started_ns
    if wall_elapsed < 0 or monotonic_elapsed < 0:
        raise RuntimeError("elapsed clock moved backwards")
    tolerance = max(
        _CLOCK_ABSOLUTE_TOLERANCE_NS,
        int(max(wall_elapsed, monotonic_elapsed) * _CLOCK_RELATIVE_TOLERANCE),
    )
    if abs(wall_elapsed - monotonic_elapsed) > tolerance:
        raise RuntimeError("elapsed clocks disagree; refusing timing evidence")
    return wall_elapsed / 1_000_000_000


def _preflight_elapsed_clock() -> None:
    """Cheap per-worker guard before a scale cell is measured."""

    wall_started = time.time_ns()
    monotonic_started = time.perf_counter_ns()
    time.sleep(0.01)
    _paired_elapsed_seconds(
        wall_started,
        time.time_ns(),
        monotonic_started,
        time.perf_counter_ns(),
    )


def _fit(path: Path, rows: int, chunk_bytes: int) -> ExponentialSourceFitResult:
    return fit_exponential_csv(
        path,
        schema=SCHEMA,
        source_id=PublicSourceId(
            f"src_{hashlib.md5(str(rows).encode(), usedforsecurity=False).hexdigest()}"
        ),
        limits=CsvLifetimeLimits(chunk_bytes, chunk_bytes),
    )


def _cell(
    path: Path,
    *,
    rows: int,
    chunk_bytes: int,
    expected_events: int,
    expected_total: Decimal,
) -> dict[str, object]:
    _preflight_elapsed_clock()
    wall_started = time.time_ns()
    monotonic_started = time.perf_counter_ns()
    result = _fit(path, rows, chunk_bytes)
    elapsed = _paired_elapsed_seconds(
        wall_started,
        time.time_ns(),
        monotonic_started,
        time.perf_counter_ns(),
    )
    before_rss = peak_rss_bytes()
    tracemalloc.start()
    try:
        traced = _fit(path, rows, chunk_bytes)
        _, trace_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    after_rss = peak_rss_bytes()
    if not result.execution.outcome.complete or not isinstance(result.fit, ExponentialFitSuccess):
        raise RuntimeError("scale fixture did not produce a complete exponential fit")
    if not isinstance(traced.fit, ExponentialFitSuccess) or traced.fit != result.fit:
        raise RuntimeError("memory pass did not reproduce the timed fit")
    fit = result.fit
    expected_rate = Decimal(expected_events) / expected_total
    actual_rate = Decimal(str(fit.rate))
    absolute_error = abs(actual_rate - expected_rate)
    relative_error = absolute_error / abs(expected_rate)
    execution = result.execution.provenance.execution
    return {
        "rows": rows,
        "chunk_bytes": chunk_bytes,
        "max_inflight_bytes": chunk_bytes,
        "source": {"bytes": path.stat().st_size, "sha256": _sha256(path)},
        "observed": {
            "actual_pass_count": execution.passes.actual_pass_count,
            "max_passes": execution.passes.max_passes,
            "accepted_chunk_count": result.execution.outcome.coverage.accepted_chunk_count,
            "processed_row_count": result.execution.outcome.coverage.processed_row_count,
            "peak_inflight_bytes": execution.buffer.peak_inflight_bytes,
            "largest_retained_chunk_bytes": execution.buffer.largest_retained_chunk_bytes,
            "backpressure_event_count": execution.buffer.backpressure_event_count,
        },
        "fit": {
            "observation_count": fit.observation_count,
            "event_count": fit.event_count,
            "total_time": fit.total_time,
            "rate": fit.rate,
            "expected_event_count": expected_events,
            "expected_total_time": str(expected_total),
            "expected_rate": str(expected_rate),
            "absolute_rate_error": float(absolute_error),
            "relative_rate_error": float(relative_error),
        },
        "memory": {
            "tracemalloc_peak_bytes": trace_peak,
            "rss_peak_bytes": after_rss,
            "rss_delta_bytes": max(0, after_rss - before_rss),
        },
        "elapsed_seconds": elapsed,
        "throughput_rows_per_second": rows / elapsed if elapsed else float(rows),
    }


def _cell_request(request: tuple[str, int, int, int, Decimal]) -> dict[str, object]:
    """Pickle-safe worker boundary; paths stay process-local and unrecorded."""

    path, rows, budget, events, total = request
    return _cell(
        Path(path),
        rows=rows,
        chunk_bytes=budget,
        expected_events=events,
        expected_total=total,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", default="10000,100000,1000000")
    parser.add_argument("--chunk-bytes", default="32768,65536,131072")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="parallel measurement processes; timing evidence is only valid with 1",
    )
    parser.add_argument(
        "--temporary-root",
        type=Path,
        help="optional parent directory for one unique private run directory",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        preflight_sha = _clean_checkout_sha(root)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    rows = [int(item) for item in args.rows.split(",")]
    budgets = _parse_budgets(args.chunk_bytes)
    if not rows or any(item <= 0 for item in rows) or len(set(rows)) != len(rows):
        raise SystemExit("rows must contain distinct positive integers")
    if args.workers <= 0:
        raise SystemExit("workers must be positive")
    started = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    cells: list[dict[str, object]] = []
    chunks_by_row: dict[int, int] = {}
    temporary_parent = None if args.temporary_root is None else str(args.temporary_root)
    with tempfile.TemporaryDirectory(
        prefix="veridist-scale-csv-exp-", dir=temporary_parent
    ) as temporary:
        run_directory = Path(temporary)
        for row_count in rows:
            fixture = run_directory / f"scale-csv-exp-v1-{row_count}.csv"
            expected_events, expected_total = _generate(fixture, row_count)
            requests = [
                (str(fixture), row_count, budget, expected_events, expected_total)
                for budget in budgets
            ]
            if args.workers == 1:
                row_cells = [_cell_request(request) for request in requests]
            else:
                with ProcessPoolExecutor(max_workers=min(args.workers, len(requests))) as executor:
                    row_cells = list(executor.map(_cell_request, requests))
            cells.extend(row_cells)
            chunks_by_row[row_count] = int(row_cells[0]["observed"]["accepted_chunk_count"])
    value: dict[str, object] = {
        "schema_version": "3",
        "run": {
            "git_sha": preflight_sha,
            "candidate_git_sha": preflight_sha,
            "git_dirty": False,
            "utc_started": started,
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "platform": platform.platform(),
            "measurement_workers": args.workers,
            "timing": {"clock": ELAPSED_CLOCK, "preflight": ELAPSED_PREFLIGHT},
            "methodology": dict(METHODOLOGY),
        },
        "generator": {"formula_version": "1", "temporary_root": "redacted"},
        "cells": cells,
        "operation_evidence": {
            "rows": rows,
            "accepted_chunks": [chunks_by_row[row] for row in rows],
        },
    }
    value["artifact_sha256"] = _canonical_digest(value)
    try:
        end_sha = _clean_checkout_sha(root)
    except RuntimeError as error:
        raise SystemExit(f"refusing evidence output after checkout changed: {error}") from error
    if end_sha != preflight_sha:
        raise SystemExit("refusing evidence output after HEAD changed during measurement")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote retained evidence: {args.output.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
