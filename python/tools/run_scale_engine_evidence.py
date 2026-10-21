"""Generate retained evidence for the internal ``veridist.scale`` engine.

The artifact (schema 1) records, for a ladder of row counts, how the engine behaves on
deterministic Parquet data: elapsed time, peak memory, pass counts, partitions and executors for
the one-pass families (exponential, normal, gamma, lognormal) and the multi-pass families
(weibull_min, gumbel_right), proofs that the state and the result do not depend on the layout of
the data, the executor or an interruption, agreement with independent references, a large-n
goodness-of-fit enclosure and a profile-likelihood interval.

Data. One data set per rung: ``rows`` draws of an exponential distribution (scale 1000) built
from the raw output of a seeded ``PCG64`` generator, written to Parquet (one float64 column,
snappy, no dictionary). The exponential fit is therefore well specified and so is the Weibull fit
(shape 1); the other families are fitted to the same positive data as a computation, not as a
claim about the data. The draws are produced in blocks of 2**20 rows, each block from its own
seed sequence, so the rows do not depend on how they are grouped into files. Layout A
puts ``min(2**20, rows // 16)`` rows in a row group and 16 row groups in a file; layout B has
row groups of three quarters of that size plus one row, five per file, and lists the blocks and
the rows of each block in reverse. Both layouts hold the same rows. Only one layout is on disk at
a time, and the data live in a private temporary directory that is removed after the run; the
artifact records byte counts and SHA-256 digests of the files, never paths.

Ladder. Every rung measures the one-pass families. The multi-pass families (six passes of exact
sums per fit, about thirty times the cost of a one-pass fit) are measured at the rungs up to
``MULTI_PASS_MAX_ROWS``: a fit of ``weibull_min`` and ``gumbel_right`` at 1e9 rows, repeated for
timing, memory, the second layout and the resumed run, does not fit the time limit of a hosted
runner. The goodness-of-fit enclosure (exponential) is computed at the largest rung and the
profile interval (Weibull) at the largest rung that does not exceed ``PROFILE_MAX_ROWS``.

Measurement. Cells run strictly one after another (``measurement_workers`` is 1). Each cell is
read twice: an untraced pass in this process supplies ``elapsed_seconds``, and a separate pass
in a fresh process, under ``tracemalloc``, supplies the memory facts, so the peak resident set of
that process is the peak of the cell alone. The memory pass must reproduce the digests of the
timed fit. For the process executor the resident set is that of the measuring process; the
workers are not included. Timings are descriptive; the checker applies no threshold to them.

Proofs of invariance. The state digest, the digest of the result bits and the digest of the pass
record of a fit are recorded and must be identical across: serial and process executors (layout
A), layouts A and B, and an uninterrupted run and a run that was interrupted part way through
the checkpoint store and resumed.

``artifact_sha256`` is an integrity digest over the canonical JSON body. It detects accidental
edits and truncation; it is not a signature and does not authenticate who produced the artifact.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from veridist.domain.lifetimes import ExactLifetime
from veridist.domain.values import ExactValue
from veridist.families.exponential import fit_exponential
from veridist.families.gamma import fit_gamma
from veridist.families.gumbel import fit_gumbel_right
from veridist.families.lognormal import fit_lognormal
from veridist.families.normal import fit_normal
from veridist.families.results import FitSuccess
from veridist.families.weibull import fit_weibull
from veridist.scale._arrow import ParquetSource
from veridist.scale._environment import numeric_environment_digest, numpy_version
from veridist.scale._executors import ProcessExecutor
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._gof import large_n_gof
from veridist.scale._partition_store import SQLitePartitionStore
from veridist.scale._planner import PASS_BUDGET_DEFAULT
from veridist.scale._profile_intervals import PROFILE_PASS_BUDGET_DEFAULT, profile_intervals
from veridist.scale._sources import DEFAULT_MAX_ROWS

try:  # direct script execution puts tools/ on sys.path
    from process_memory import peak_rss_bytes
except ImportError:  # imported as ``tools.<module>``
    from tools.process_memory import peak_rss_bytes

SCHEMA_VERSION = "1"
MEASUREMENT_WORKERS = 1

# Windows virtualized clocks can disagree after a worker migrates between timing domains.
# Evidence must fail closed rather than preserve an elapsed value that cannot be tied to a
# trustworthy clock.
ELAPSED_CLOCK = "time.perf_counter_ns"
ELAPSED_PREFLIGHT = "paired-monotonic-wall-v2"
METHODOLOGY = {
    "passes": "elapsed-untraced-then-memory-traced-v1",
    "rss": "process-peak-v1",
    "memory_process": "fresh-process-per-cell-v1",
    "rss_scope": "measuring-process-excluding-workers-v1",
}
_CLOCK_ABSOLUTE_TOLERANCE_NS = 100_000_000
_CLOCK_RELATIVE_TOLERANCE = 0.05

STANDARD_LADDER = (1_000_000, 10_000_000, 100_000_000)
TOP_RUNG = 1_000_000_000
"""The largest rung, measured only when the caller asks for it (``--top-rung``)."""

PROFILE_MAX_ROWS = 100_000_000
"""The profile interval is computed at the largest rung that does not exceed this."""

MULTI_PASS_MAX_ROWS = 100_000_000
"""The multi-pass families are measured at the rungs that do not exceed this."""

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

# The data generator (formula version 1).
SEED = 0x5CA1E5EED
DISTRIBUTION_SCALE = 1000.0
BLOCK_ROWS = 1 << 20
COLUMN = "value"
ROW_GROUP_CAP = 1 << 20
LAYOUT_A_GROUPS_PER_FILE = 16
LAYOUT_B_GROUPS_PER_FILE = 5
DISK_MARGIN_BYTES = 1 << 30

GENERATOR = {
    "formula_version": "1",
    "distribution": "exponential",
    "scale": DISTRIBUTION_SCALE,
    "seed": SEED,
    "block_rows": BLOCK_ROWS,
    "column": COLUMN,
    "compression": "snappy",
    "temporary_root": "redacted",
}

SUBSAMPLE_ROWS = 100_000
PROFILE_LEVEL = 0.95
GOF_BIN_EXPONENT = 12
GOF_THRESHOLD = 0.001

# The fixed small sample of the mpmath reference: x_i = (2**33 + 3 h1_i + 5 h2_i) / 2**34 with two
# multiplicative congruential streams, so every value is exact in binary64 on every platform.
MPMATH_SAMPLE_ROWS = 2000
MPMATH_DIGITS = 50
MPMATH_REFERENCE_DIGITS = 30


# ------------------------------------------------------------------------------------ clock


def _paired_elapsed_seconds(
    wall_started_ns: int,
    wall_finished_ns: int,
    monotonic_started_ns: int,
    monotonic_finished_ns: int,
) -> float:
    """Return the monotonic duration only when the wall clock agrees with it.

    ``time.perf_counter_ns`` is the persisted measurement: it is monotonic and high resolution
    on every platform. The wall-clock duration is a provenance check that rejects a timer domain
    whose two clocks disagree beyond the tolerance.
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
    if monotonic_elapsed == 0:
        raise RuntimeError("elapsed time is below the monotonic clock resolution")
    return monotonic_elapsed / 1_000_000_000


def _preflight_elapsed_clock() -> None:
    """Cheap guard before a cell is measured."""

    wall_started = time.time_ns()
    monotonic_started = time.perf_counter_ns()
    time.sleep(0.01)
    _paired_elapsed_seconds(wall_started, time.time_ns(), monotonic_started, time.perf_counter_ns())


_STARTED = time.monotonic()


def _log(message: str) -> None:
    """One progress line on stderr, with the seconds since the start; no path, no data value."""

    print(f"[{time.monotonic() - _STARTED:9.1f}s] {message}", file=sys.stderr, flush=True)


def _timed(work: Callable[[], Any]) -> tuple[Any, float]:
    _preflight_elapsed_clock()
    wall_started = time.time_ns()
    monotonic_started = time.perf_counter_ns()
    value = work()
    elapsed = _paired_elapsed_seconds(
        wall_started, time.time_ns(), monotonic_started, time.perf_counter_ns()
    )
    return value, elapsed


# ------------------------------------------------------------------------------- checkout


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()


def _clean_checkout_sha(root: Path) -> str:
    """Fail closed unless this exact checkout is clean, returning its HEAD."""

    if _git(root, "status", "--porcelain"):
        raise RuntimeError("refusing evidence run from a dirty checkout")
    return _git(root, "rev-parse", "HEAD")


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------- the data


def layouts(rows: int) -> dict[str, dict[str, Any]]:
    """The two layouts of ``rows`` rows: row-group size, groups per file and block order."""

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
    """The families measured at a rung."""

    return FAMILIES if rows <= MULTI_PASS_MAX_ROWS else ONE_PASS_FAMILIES


def _block(rows: int, index: int) -> Any:
    """Block ``index`` of the data of a ``rows``-row rung: ``BLOCK_ROWS`` exponential draws."""

    import numpy as np

    bit_generator = np.random.PCG64(np.random.SeedSequence([SEED, rows, index]))
    raw = bit_generator.random_raw(BLOCK_ROWS)
    # 52 random bits and a half: exactly representable, strictly inside (0, 1)
    uniform = ((raw >> np.uint64(12)).astype(np.float64) + 0.5) * 2.0**-52
    return -DISTRIBUTION_SCALE * np.log1p(-uniform)


def _stream(rows: int, order: str) -> Any:
    """The rows of the rung as arrays, in the order of the layout (the same multiset)."""

    blocks = -(-rows // BLOCK_ROWS)
    indices = range(blocks) if order == "ascending" else range(blocks - 1, -1, -1)
    for index in indices:
        values = _block(rows, index)[: min(BLOCK_ROWS, rows - index * BLOCK_ROWS)]
        yield values if order == "ascending" else values[::-1]


def _free_bytes(directory: Path) -> int:
    return shutil.disk_usage(directory).free


def _require_disk(directory: Path, rows: int) -> None:
    needed = rows * 8 + DISK_MARGIN_BYTES
    if _free_bytes(directory) < needed:
        raise RuntimeError("not enough free disk space for the data of this rung")


def write_layout(directory: Path, rows: int, layout: dict[str, Any]) -> dict[str, Any]:
    """Write one layout of the rung's data and return its record (no path).

    The record holds the layout parameters, the number of partitions and, for every file in
    canonical order, its byte count and SHA-256.
    """

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    group = int(layout["row_group_rows"])
    per_file = int(layout["groups_per_file"])
    directory.mkdir(parents=True, exist_ok=False)
    schema = pa.schema([(COLUMN, pa.float64())])
    pending: list[Any] = []
    held = 0
    groups_in_file = 0
    files: list[Path] = []
    writer: Any = None

    def flush(values: Any) -> None:
        nonlocal writer, groups_in_file
        if writer is None:
            path = directory / f"part-{len(files):05d}.parquet"
            files.append(path)
            writer = pq.ParquetWriter(path, schema, compression="snappy", use_dictionary=False)
            groups_in_file = 0
        writer.write_table(pa.table({COLUMN: values}, schema=schema), row_group_size=len(values))
        groups_in_file += 1
        if groups_in_file == per_file:
            writer.close()
            writer = None

    try:
        for values in _stream(rows, str(layout["order"])):
            pending.append(values)
            held += len(values)
            while held >= group:
                merged = np.concatenate(pending) if len(pending) > 1 else pending[0]
                flush(merged[:group])
                rest = merged[group:]
                pending = [rest] if len(rest) else []
                held = len(rest)
        if held:
            flush(np.concatenate(pending) if len(pending) > 1 else pending[0])
    finally:
        if writer is not None:
            writer.close()
    partitions = -(-rows // group)
    return {
        "layout": {
            "row_group_rows": group,
            "groups_per_file": per_file,
            "order": str(layout["order"]),
        },
        "rows": rows,
        "partitions": partitions,
        "files": [{"bytes": path.stat().st_size, "sha256": _sha256_file(path)} for path in files],
    }


# ---------------------------------------------------------------------- what a fit leaves


def _result_facts(
    result: ScaleFitResult, rows: int, family: str, *, resumed: bool = False
) -> dict[str, Any]:
    """The checkable facts of one successful scale fit: digests, result bits and passes.

    A pass of a resumed run that had a single partition left to read is read serially by design;
    any other downgrade is refused.
    """

    outcome = result.fit
    scale = result.scale
    if not isinstance(outcome, FitSuccess) or not outcome.converged:
        raise RuntimeError(f"{family} fit did not succeed")
    if outcome.observation_count != rows or scale.row_count != rows or scale.state is None:
        raise RuntimeError(f"{family} fit did not read the rung")
    if scale.environment.numpy_version != numpy_version() or (
        scale.environment.digest_sha256 != numeric_environment_digest()
    ):
        raise RuntimeError("numeric environment changed during the run")
    parameters = {name: float(value).hex() for name, value in outcome.parameters.items()}
    if tuple(parameters) != PARAMETER_NAMES[family]:
        raise RuntimeError(f"{family} parameters are not the declared ones")
    result_body = {
        "family": family,
        "observation_count": outcome.observation_count,
        "parameters_hex": parameters,
        "log_likelihood_hex": float(outcome.log_likelihood).hex(),
    }
    execution = scale.execution
    if execution is None:
        raise RuntimeError("fit carries no execution record")
    if scale.passes is None:
        records: list[list[object]] = []
        roles: list[str] = []
        passes = {"data_passes": len(execution.passes), "minimum": 1, "maximum": 1, "budget": 1}
    else:
        records = [
            [
                item.role,
                item.points,
                None if item.log_low is None else item.log_low.hex(),
                None if item.log_high is None else item.log_high.hex(),
            ]
            for item in scale.passes.records
        ]
        roles = [item.role for item in scale.passes.records]
        passes = {
            "data_passes": len(execution.passes),
            "minimum": scale.passes.minimum_passes,
            "maximum": scale.passes.maximum_passes,
            "budget": scale.passes.pass_budget,
        }
        if scale.passes.pass_count != len(execution.passes):
            raise RuntimeError("pass record and execution record disagree")
    passes["roles"] = roles
    if any(
        item.downgrade is not None and not (resumed and item.downgrade == "single_partition")
        for item in execution.passes
    ):
        raise RuntimeError("a pass was read serially although more was requested")
    return {
        "partitions": scale.partition_count,
        "max_rows_per_batch": scale.max_rows,
        "executor": {
            "kind": execution.executor,
            "requested_workers": execution.requested_workers,
            "effective_workers": max(item.workers for item in execution.passes),
        },
        "retries": sum(item.retries for item in execution.passes),
        "passes": passes,
        "digests": {
            "state_sha256": scale.state.digest_sha256,
            "result_sha256": _canonical_digest(result_body),
            "passes_sha256": _canonical_digest(records),
        },
        "result": {
            "observation_count": outcome.observation_count,
            "parameters_hex": parameters,
            "log_likelihood_hex": result_body["log_likelihood_hex"],
        },
    }


def _executor(kind: str, workers: int) -> Any:
    return "serial" if kind == "serial" else ProcessExecutor(workers)


def _fit(directory: Path, family: str, kind: str, workers: int, **options: Any) -> ScaleFitResult:
    source = ParquetSource(directory, COLUMN)
    return fit(source, family, executor=_executor(kind, workers), **options)


# ------------------------------------------------------------------------------- the cells


def _memory_pass(request: dict[str, Any]) -> dict[str, Any]:
    """Run one fit in a fresh process under ``tracemalloc`` and return its memory facts."""

    command = [sys.executable, str(Path(__file__).resolve()), "--memory-cell", json.dumps(request)]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError("memory pass failed: " + completed.stderr.strip()[-400:])
    lines = completed.stdout.strip().splitlines()
    return dict(json.loads(lines[-1]))


def _memory_child(request: dict[str, Any]) -> int:
    """Entry of the memory pass: print the memory facts and the digests of one fit."""

    directory = Path(request["directory"])
    family, kind, workers, rows = (
        str(request["family"]),
        str(request["executor"]),
        int(request["workers"]),
        int(request["rows"]),
    )
    baseline = peak_rss_bytes()
    tracemalloc.start()
    try:
        result = _fit(directory, family, kind, workers)
        _, traced_peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    facts = _result_facts(result, rows, family)
    print(
        json.dumps(
            {
                "tracemalloc_peak_bytes": traced_peak,
                "rss_baseline_bytes": baseline,
                "rss_peak_bytes": max(baseline, peak_rss_bytes()),
                "digests": facts["digests"],
            }
        )
    )
    return 0


def _cell(
    directory: Path, rows: int, family: str, kind: str, workers: int
) -> tuple[dict[str, Any], ScaleFitResult]:
    """Measure one cell: a timed untraced fit, then the memory pass in a fresh process."""

    gc.collect()
    result, elapsed = _timed(lambda: _fit(directory, family, kind, workers))
    facts = _result_facts(result, rows, family)
    memory = _memory_pass(
        {
            "directory": str(directory),
            "family": family,
            "executor": kind,
            "workers": workers,
            "rows": rows,
        }
    )
    if memory["digests"] != facts["digests"]:
        raise RuntimeError("memory pass did not reproduce the timed fit")
    cell = {
        "rows": rows,
        "family": family,
        "group": "one_pass" if family in ONE_PASS_FAMILIES else "multi_pass",
        **facts,
        "elapsed_seconds": elapsed,
        "throughput_rows_per_second": rows / elapsed,
        "memory": {
            "tracemalloc_peak_bytes": int(memory["tracemalloc_peak_bytes"]),
            "rss_baseline_bytes": int(memory["rss_baseline_bytes"]),
            "rss_peak_bytes": int(memory["rss_peak_bytes"]),
        },
    }
    return cell, result


class _Interrupted(Exception):
    """Raised by the interrupting store: the run stops as if its process had been killed."""


class _InterruptingStore(SQLitePartitionStore):
    """A partition store that refuses the commit after the first ``after`` of them."""

    def __init__(self, path: Path, *, after: int) -> None:
        super().__init__(path)
        self._after = after
        self._commits = 0

    def commit_partition(self, *arguments: Any, **options: Any) -> bool:
        if self._commits >= self._after:
            raise _Interrupted
        self._commits += 1
        return super().commit_partition(*arguments, **options)


def _resumed(
    directory: Path, rows: int, family: str, workers: int, partitions: int
) -> dict[str, Any]:
    """An interrupted run of the checkpointed fit, then its resumption (the process executor)."""

    multi = family in MULTI_PASS_FAMILIES
    after = partitions + partitions // 2 if multi else partitions // 2
    database = directory.parent / f"{directory.name}-{family}.db"
    try:
        _fit(
            directory,
            family,
            "processes",
            workers,
            checkpoint=_InterruptingStore(database, after=after),
        )
    except _Interrupted:
        pass
    else:
        raise RuntimeError("the interrupted run completed")
    result = _fit(directory, family, "processes", workers, checkpoint=database, resume=True)
    database.unlink()
    facts = _result_facts(result, rows, family, resumed=True)
    record = result.scale.checkpoint
    if record is None or not record.resumed:
        raise RuntimeError("the second run did not resume the first")
    return {
        "interrupted_after_commits": after,
        "partitions": partitions,
        "passes_after_resume": facts["passes"]["data_passes"],
        "replayed_passes": sum(1 for item in record.passes if item.replayed),
        "reused_partitions": sum(item.reused for item in record.passes),
        "read_partitions": sum(item.read for item in record.passes),
        "digests": facts["digests"],
    }


# ---------------------------------------------------------------------- the references


def _in_memory(family: str, values: Any) -> Any:
    """The in-memory fit of ``family`` to ``values`` (the independent reference)."""

    items = [float(value) for value in values]
    if family == "normal":
        return fit_normal([ExactValue(item) for item in items])
    if family == "gumbel_right":
        return fit_gumbel_right([ExactValue(item) for item in items])
    lifetimes = [ExactLifetime(item) for item in items]
    fitter = {
        "exponential": fit_exponential,
        "gamma": fit_gamma,
        "lognormal": fit_lognormal,
        "weibull_min": fit_weibull,
    }[family]
    return fitter(lifetimes)


def _difference(engine: float, reference: float) -> dict[str, Any]:
    return {
        "engine_hex": float(engine).hex(),
        "reference_hex": float(reference).hex(),
        "relative_difference": abs(float(engine) - float(reference)) / abs(float(reference)),
    }


def _subsample_references(
    rows: int, workdir: Path, families: tuple[str, ...]
) -> list[dict[str, Any]]:
    """The scale fit of a subsample of the rung against the in-memory fit of the same rows."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    size = min(rows, SUBSAMPLE_ROWS)
    values = _block(rows, 0)[:size]
    directory = workdir / "subsample"
    directory.mkdir()
    table = pa.table({COLUMN: values}, schema=pa.schema([(COLUMN, pa.float64())]))
    pq.write_table(
        table, directory / "part-00000.parquet", row_group_size=-(-size // 3), use_dictionary=False
    )
    digest = hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()
    records = []
    try:
        for family in families:
            engine = _fit(directory, family, "serial", 1)
            if not isinstance(engine.fit, FitSuccess):
                raise RuntimeError(f"{family} subsample fit did not succeed")
            reference = _in_memory(family, values)
            if not isinstance(reference, FitSuccess):
                raise RuntimeError(f"{family} in-memory reference fit did not succeed")
            records.append(
                {
                    "family": family,
                    "subsample_rows": size,
                    "subsample_sha256": digest,
                    "parameters": {
                        name: _difference(engine.fit.parameters[name], reference.parameters[name])
                        for name in PARAMETER_NAMES[family]
                    },
                    "log_likelihood": _difference(
                        engine.fit.log_likelihood, reference.log_likelihood
                    ),
                }
            )
    finally:
        shutil.rmtree(directory)
    return records


def mpmath_sample() -> list[float]:
    """The fixed small sample: exact binary64 values in ``[0.5, 2.5)`` from integer streams."""

    values = []
    for index in range(MPMATH_SAMPLE_ROWS):
        first = (index * 2654435761 + 12345) % (1 << 32)
        second = (index * 1597334677 + 54321) % (1 << 32)
        values.append((float((1 << 33) + 3 * first + 5 * second)) / float(1 << 34))
    return values


def mpmath_sample_sha256() -> str:
    import struct

    return hashlib.sha256(
        b"".join(struct.pack("<d", value) for value in mpmath_sample())
    ).hexdigest()


def mpmath_references() -> dict[str, dict[str, str]]:
    """The maximum-likelihood estimates of the fixed sample at 50 digits, as decimal strings.

    Closed forms for the exponential, normal and lognormal families, the root of
    ``log k - digamma k = log mean - mean log`` for the gamma shape and the roots of the profile
    equations for the Weibull shape and the right-Gumbel scale. Nothing here uses the engine.
    """

    import mpmath

    with mpmath.workdps(MPMATH_DIGITS):
        return _mpmath_estimates(mpmath)


def _mpmath_estimates(mpmath: Any) -> dict[str, dict[str, str]]:
    values = [mpmath.mpf(item) for item in mpmath_sample()]
    count = len(values)
    total = mpmath.fsum(values)
    logs = [mpmath.log(item) for item in values]
    mean = total / count
    log_mean = mpmath.fsum(logs) / count

    def sd(items: list[Any], center: Any) -> Any:
        return mpmath.sqrt(mpmath.fsum((item - center) ** 2 for item in items) / count)

    gap = mpmath.log(mean) - log_mean
    gamma_shape = mpmath.findroot(
        lambda k: mpmath.log(k) - mpmath.digamma(k) - gap,
        1.0 / (2.0 * float(gap)),
        tol=mpmath.mpf(10) ** -40,
    )
    # Weibull: root of n/k - n sum(y e^{ky}) / sum(e^{ky}) over centred logarithms
    centre = log_mean
    centred = [item - centre for item in logs]

    def weibull_parts(k: Any) -> tuple[Any, Any]:
        weights = [mpmath.exp(k * item) for item in centred]
        return mpmath.fsum(weights), mpmath.fsum(
            w * y for w, y in zip(weights, centred, strict=True)
        )

    def weibull_score(k: Any) -> Any:
        s0, s1 = weibull_parts(k)
        return count / k - count * s1 / s0

    weibull_shape = mpmath.findroot(weibull_score, mpmath.mpf(3), tol=mpmath.mpf(10) ** -40)
    weibull_scale = mpmath.exp(
        centre + mpmath.log(weibull_parts(weibull_shape)[0] / count) / weibull_shape
    )
    # Gumbel: root of sigma - mean + sum(x e^{-x/sigma}) / sum(e^{-x/sigma})
    low = min(values)

    def gumbel_parts(sigma: Any) -> tuple[Any, Any]:
        weights = [mpmath.exp(-(item - low) / sigma) for item in values]
        return mpmath.fsum(weights), mpmath.fsum(
            w * x for w, x in zip(weights, values, strict=True)
        )

    def gumbel_score(sigma: Any) -> Any:
        s0, s1 = gumbel_parts(sigma)
        return sigma - mean + s1 / s0

    gumbel_scale = mpmath.findroot(gumbel_score, mpmath.mpf(1) / 3, tol=mpmath.mpf(10) ** -40)
    gumbel_location = low - gumbel_scale * mpmath.log(gumbel_parts(gumbel_scale)[0] / count)
    estimates = {
        "exponential": {"rate": count / total},
        "normal": {"mu": mean, "sigma": sd(values, mean)},
        "gamma": {"shape": gamma_shape, "scale": mean / gamma_shape},
        "lognormal": {"mu_log": log_mean, "sigma_log": sd(logs, log_mean)},
        "weibull_min": {"shape": weibull_shape, "scale": weibull_scale},
        "gumbel_right": {"location": gumbel_location, "scale": gumbel_scale},
    }
    return {
        family: {
            name: mpmath.nstr(value, MPMATH_REFERENCE_DIGITS, strip_zeros=False)
            for name, value in parameters.items()
        }
        for family, parameters in estimates.items()
    }


def _mpmath_block(workdir: Path) -> dict[str, Any]:
    """The scale fit of the fixed sample against the 50-digit references."""

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = workdir / "mpmath-sample"
    directory.mkdir()
    values = np.array(mpmath_sample(), dtype=np.float64)
    pq.write_table(
        pa.table({COLUMN: values}, schema=pa.schema([(COLUMN, pa.float64())])),
        directory / "part-00000.parquet",
        row_group_size=MPMATH_SAMPLE_ROWS // 4,
        use_dictionary=False,
    )
    references = mpmath_references()
    records = []
    try:
        for family in FAMILIES:
            engine = _fit(directory, family, "serial", 1)
            if not isinstance(engine.fit, FitSuccess):
                raise RuntimeError(f"{family} fit of the fixed sample did not succeed")
            parameters = {}
            for name in PARAMETER_NAMES[family]:
                reference = Decimal(references[family][name])
                difference = abs(Decimal(float(engine.fit.parameters[name])) - reference) / abs(
                    reference
                )
                parameters[name] = {
                    "engine_hex": float(engine.fit.parameters[name]).hex(),
                    "reference": references[family][name],
                    "relative_difference": float(difference),
                }
            records.append({"family": family, "parameters": parameters})
    finally:
        shutil.rmtree(directory)
    return {
        "sample": {
            "rows": MPMATH_SAMPLE_ROWS,
            "formula_version": "1",
            "sha256": mpmath_sample_sha256(),
        },
        "digits": MPMATH_DIGITS,
        "families": records,
    }


# ------------------------------------------------------------------------- the inference


def _gof_block(
    directory: Path, rows: int, workers: int, exponential: ScaleFitResult
) -> dict[str, Any]:
    """The certified goodness-of-fit enclosure of the (well specified) exponential fit."""

    source = ParquetSource(directory, COLUMN)
    report, elapsed = _timed(
        lambda: large_n_gof(
            source,
            exponential,
            bin_exponent=GOF_BIN_EXPONENT,
            threshold=GOF_THRESHOLD,
            executor=ProcessExecutor(workers),
        )
    )
    execution = report.scale.execution
    passes = report.scale.passes
    if execution is None or passes is None or report.decision is None:
        raise RuntimeError("goodness-of-fit report is incomplete")
    if any(item.downgrade is not None for item in execution.passes):
        raise RuntimeError("a pass was read serially although more was requested")
    parameters = exponential.fit.parameters  # type: ignore[union-attr]
    return {
        "rows": rows,
        "family": "exponential",
        "executor": {
            "kind": execution.executor,
            "requested_workers": execution.requested_workers,
            "effective_workers": max(item.workers for item in execution.passes),
        },
        "fit_parameters_hex": {
            name: float(parameters[name]).hex() for name in PARAMETER_NAMES["exponential"]
        },
        "bin_exponent": GOF_BIN_EXPONENT,
        "bins": report.bins,
        "cdf_error": report.cdf_error,
        "ks_lower": report.ks_lower,
        "ks_upper": report.ks_upper,
        "tv_lower": report.tv_lower,
        "tv_upper": report.tv_upper,
        "strip_rows": report.strip_rows,
        "counts_sha256": report.counts_digest,
        "threshold": report.threshold,
        "decision": report.decision.value,
        "passes": {
            "data_passes": len(execution.passes),
            "minimum": passes.minimum_passes,
            "maximum": passes.maximum_passes,
            "budget": passes.pass_budget,
            "roles": [item.role for item in passes.records],
        },
        "elapsed_seconds": elapsed,
    }


def _profile_block(
    directory: Path, rows: int, workers: int, weibull: ScaleFitResult
) -> dict[str, Any]:
    """The profile-likelihood interval of the Weibull parameters."""

    source = ParquetSource(directory, COLUMN)
    report, elapsed = _timed(
        lambda: profile_intervals(
            source, weibull, level=PROFILE_LEVEL, executor=ProcessExecutor(workers)
        )
    )
    execution = report.scale.execution
    passes = report.scale.passes
    if execution is None or passes is None:
        raise RuntimeError("profile report is incomplete")
    if any(item.downgrade is not None for item in execution.passes):
        raise RuntimeError("a pass was read serially although more was requested")
    parameters = weibull.fit.parameters  # type: ignore[union-attr]
    return {
        "rows": rows,
        "family": "weibull_min",
        "level": report.level,
        "executor": {
            "kind": execution.executor,
            "requested_workers": execution.requested_workers,
            "effective_workers": max(item.workers for item in execution.passes),
        },
        "intervals": {
            name: {
                "estimate_hex": float(parameters[name]).hex(),
                "lower": report.parameters[name].lower,
                "upper": report.parameters[name].upper,
                "lower_unbounded": report.parameters[name].lower_unbounded,
                "upper_unbounded": report.parameters[name].upper_unbounded,
            }
            for name in PARAMETER_NAMES["weibull_min"]
        },
        "resolution": {
            "surrogate_degree": report.resolution.surrogate_degree,
            "surrogate_gap": report.resolution.surrogate_gap,
            "confirmation_residual": report.resolution.confirmation_residual,
            "tolerance": report.resolution.tolerance,
        },
        "passes": {
            "data_passes": len(execution.passes),
            "minimum": passes.minimum_passes,
            "maximum": passes.maximum_passes,
            "budget": passes.pass_budget,
            "roles": [item.role for item in passes.records],
        },
        "elapsed_seconds": elapsed,
    }


# ------------------------------------------------------------------------------- the run


def _rung(
    rows: int,
    root: Path,
    workers: int,
    *,
    gof: bool,
    profile: bool,
) -> dict[str, Any]:
    """Measure one rung; at most one layout is on disk at a time."""

    geometry = layouts(rows)
    families = families_at(rows)
    _require_disk(root, rows)
    directory = root / f"rung-{rows}-a"
    dataset_a = write_layout(directory, rows, geometry["a"])
    _log(f"rung {rows}: layout A written, {dataset_a['partitions']} partitions")
    cells: list[dict[str, Any]] = []
    resumed: dict[str, dict[str, Any]] = {}
    inference: dict[str, Any] = {}
    fits: dict[str, ScaleFitResult] = {}
    try:
        for family in families:
            for kind in EXECUTORS:
                cell, result = _cell(directory, rows, family, kind, workers)
                cells.append(cell)
                _log(f"rung {rows}: {family} {kind} cell, {cell['elapsed_seconds']:.1f}s timed")
                if kind == "processes":
                    fits[family] = result
            resumed[family] = _resumed(directory, rows, family, workers, dataset_a["partitions"])
            _log(f"rung {rows}: {family} interrupted and resumed")
        references = _subsample_references(rows, root, families)
        _log(f"rung {rows}: subsample references")
        if gof:
            inference["gof"] = _gof_block(directory, rows, workers, fits["exponential"])
            _log(f"rung {rows}: goodness of fit, {inference['gof']['elapsed_seconds']:.1f}s")
        if profile:
            inference["profile"] = _profile_block(directory, rows, workers, fits["weibull_min"])
            _log(f"rung {rows}: profile interval, {inference['profile']['elapsed_seconds']:.1f}s")
    finally:
        shutil.rmtree(directory)
    directory_b = root / f"rung-{rows}-b"
    _require_disk(root, rows)
    dataset_b = write_layout(directory_b, rows, geometry["b"])
    _log(f"rung {rows}: layout B written, {dataset_b['partitions']} partitions")
    layout_b: dict[str, dict[str, Any]] = {}
    try:
        for family in families:
            gc.collect()
            result = _fit(directory_b, family, "processes", workers)
            facts = _result_facts(result, rows, family)
            layout_b[family] = {
                "partitions": facts["partitions"],
                "requested_workers": facts["executor"]["requested_workers"],
                "digests": facts["digests"],
            }
            _log(f"rung {rows}: {family} on layout B")
    finally:
        shutil.rmtree(directory_b)
    _log(f"rung {rows}: done")
    return {
        "rows": rows,
        "datasets": {"a": dataset_a, "b": dataset_b},
        "cells": cells,
        "invariance": [
            {"family": family, "layout_b": layout_b[family], "resumed": resumed[family]}
            for family in families
        ],
        "references": references,
        "inference": inference,
    }


def ladder(args_rows: str | None, *, top_rung: bool) -> list[int]:
    """The rungs of a run: the standard ladder (and the top rung), or the declared ones."""

    if args_rows is None:
        return [*STANDARD_LADDER, *([TOP_RUNG] if top_rung else [])]
    rungs = [int(item) for item in args_rows.split(",")]
    if top_rung:
        raise ValueError("--top-rung and --rows cannot be combined")
    if (
        not 1 <= len(rungs) <= 3
        or any(item < 1_000 for item in rungs)
        or rungs != sorted(rungs)
        or len(set(rungs)) != len(rungs)
    ):
        raise ValueError("rows must be one to three distinct increasing integers of at least 1000")
    return rungs


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--memory-cell":
        return _memory_child(json.loads(sys.argv[2]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", help="comma-separated rungs (smoke runs only)")
    parser.add_argument(
        "--top-rung",
        action="store_true",
        help=f"also measure the {TOP_RUNG:,}-row rung (needs about 8 GB of disk)",
    )
    parser.add_argument(
        "--process-workers",
        type=int,
        help="workers of the process executor (default: the core count of this machine)",
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
        rungs = ladder(args.rows, top_rung=args.top_rung)
    except (RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    cpu_count = os.cpu_count() or 1
    workers = cpu_count if args.process_workers is None else args.process_workers
    if workers <= 0:
        raise SystemExit("process workers must be positive")
    started = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    import pyarrow

    _log(f"ladder {rungs}, {workers} process workers")
    gof_rows = rungs[-1]
    profile_rows = max([item for item in rungs if item <= PROFILE_MAX_ROWS] or [rungs[0]])
    temporary_parent = None if args.temporary_root is None else str(args.temporary_root)
    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="veridist-scale-engine-", dir=temporary_parent) as work:
        directory = Path(work)
        for rows in rungs:
            results.append(
                _rung(rows, directory, workers, gof=rows == gof_rows, profile=rows == profile_rows)
            )
        mpmath_block = _mpmath_block(directory)
        _log("fixed sample against the 50-digit references")
    inference: dict[str, Any] = {}
    for rung in results:
        inference.update(rung.pop("inference"))
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
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
            "measurement_workers": MEASUREMENT_WORKERS,
            "timing": {"clock": ELAPSED_CLOCK, "preflight": ELAPSED_PREFLIGHT},
            "methodology": dict(METHODOLOGY),
            "environment": {
                "numpy_version": numpy_version(),
                "pyarrow_version": pyarrow.__version__,
                "cpu_count": cpu_count,
                "process_workers": workers,
                "numeric_environment_digest": numeric_environment_digest(),
            },
            "ladder": {"rows": rungs, "top_rung": TOP_RUNG in rungs},
            "limits": {
                "max_rows_per_batch": DEFAULT_MAX_ROWS,
                "pass_budget": PASS_BUDGET_DEFAULT,
                "inference_pass_budget": PROFILE_PASS_BUDGET_DEFAULT,
            },
        },
        "generator": dict(GENERATOR),
        "rungs": results,
        "mpmath_reference": mpmath_block,
        "inference": inference,
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
