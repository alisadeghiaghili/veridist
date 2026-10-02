"""Generate retained one-pass exact-state log-likelihood scale evidence.

Each cell is measured twice in the measuring process: an untraced pass supplies
``elapsed_seconds`` (``tracemalloc`` would distort the timing), then a separate traced
pass supplies the memory facts. Cells run strictly one after another; the artifact
records ``measurement_workers`` (always 1) and the checker rejects anything else.

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
import time
import tracemalloc
from fractions import Fraction
from pathlib import Path

from veridist import DataSourceMetadata, IterableDataSource, Replayability
from veridist.families.registry import FamilyId
from veridist.statistics.log_likelihood import LogLikelihoodSuccess, reduce_log_likelihood_chunks

try:  # direct script execution puts tools/ on sys.path
    from process_memory import peak_rss_bytes
except ImportError:  # imported as ``tools.<module>``
    from tools.process_memory import peak_rss_bytes

# Elapsed time comes from an untraced pass, memory from a separate traced pass, and RSS is
# the process high-water mark (``peak_rss_bytes``), so ``rss_delta_bytes`` is only the growth
# of that mark and is zero when an earlier pass in the process already reached it.
METHODOLOGY = {"passes": "elapsed-untraced-then-memory-traced-v1", "rss": "process-peak-v1"}
MEASUREMENT_WORKERS = 1
ROWS = (10_000, 100_000, 1_000_000)
BUDGETS = (1_024, 8_192, 65_536)
FAMILY_CASES = (
    (FamilyId.NORMAL, 0.0, {"mu": 0.0, "sigma": 1.0}, "-0x1.d67f1c864beb4p-1"),
    (FamilyId.GAMMA, 1.0, {"shape": 2.0, "scale": 1.0}, "-0x1.0000000000000p+0"),
    (FamilyId.WEIBULL_MIN, 1.0, {"shape": 2.0, "scale": 1.0}, "-0x1.3a37a020b8c22p-2"),
    (FamilyId.LOGNORMAL, 1.0, {"mu_log": 0.0, "sigma_log": 1.0}, "-0x1.d67f1c864beb4p-1"),
    (FamilyId.GUMBEL_RIGHT, 0.0, {"location": 0.0, "scale": 1.0}, "-0x1.0000000000000p+0"),
)


def _head(root: Path) -> str:
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True):
        raise RuntimeError("refusing evidence from a dirty checkout")
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def _chunks(rows: int, size: int, observation: float):
    """Return generated chunks whose underlying traversal is observable."""

    return _GeneratedChunks(rows, size, observation)


class _GeneratedChunks:
    """Generate one fixed supported-family observation while tracking traversal."""

    def __init__(self, rows: int, size: int, observation: float) -> None:
        self.rows = rows
        self.size = size
        self.observation = observation
        self.iterator_acquisitions = 0
        self.observation_yields = 0

    def __iter__(self):
        self.iterator_acquisitions += 1
        if self.iterator_acquisitions != 1:
            raise RuntimeError("generated source was iterated more than once")
        for start in range(0, self.rows, self.size):
            yield self._observations(min(self.size, self.rows - start))

    def _observations(self, count: int):
        for _ in range(count):
            self.observation_yields += 1
            yield self.observation


def _oracle_units(rows: int, contribution_hex: str) -> int:
    contribution = float.fromhex(contribution_hex)
    numerator, denominator = contribution.as_integer_ratio()
    return rows * numerator * ((1 << 1074) // denominator)


def _cell(
    rows: int,
    chunk_size: int,
    *,
    family: FamilyId = FamilyId.NORMAL,
    observation: float = 0.0,
    parameters: dict[str, float] | None = None,
    contribution_hex: str = "-0x1.d67f1c864beb4p-1",
) -> dict[str, object]:
    if parameters is None:
        parameters = {"mu": 0.0, "sigma": 1.0}
    def reduce_once() -> tuple[object, _GeneratedChunks]:
        chunks = _chunks(rows, chunk_size, observation)
        source = IterableDataSource(
            chunks,
            DataSourceMetadata(
                source_id=f"scale-{family.value}-{rows}-{chunk_size}",
                schema_version="1",
                provenance_schema_version="1",
                replayability=Replayability.SINGLE_PASS,
                redaction_reason="generated",
            ),
        )
        return reduce_log_likelihood_chunks(family, source, **parameters), chunks

    started = time.perf_counter()
    result, chunks = reduce_once()
    elapsed = time.perf_counter() - started
    rss_before = peak_rss_bytes()
    tracemalloc.start()
    try:
        traced_result, traced_chunks = reduce_once()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    rss_after = peak_rss_bytes()
    if (
        not isinstance(traced_result, LogLikelihoodSuccess)
        or traced_result != result
        or traced_chunks.iterator_acquisitions != 1
        or traced_chunks.observation_yields != rows
    ):
        raise RuntimeError("memory pass did not reproduce the timed reduction")
    if (
        not isinstance(result, LogLikelihoodSuccess)
        or result.family is not family
        or result.observation_count != rows
        or chunks.iterator_acquisitions != 1
        or chunks.observation_yields != rows
    ):
        raise RuntimeError("generated stream did not reduce successfully")
    units = _oracle_units(rows, contribution_hex)
    expected = float(Fraction(units, 1 << 1074))
    if result.total_log_likelihood.hex() != expected.hex():
        raise RuntimeError("returned total does not match independent exact oracle")
    return {
        "family": family.value,
        "rows": rows,
        "chunk_size": chunk_size,
        "one_pass": {
            "iterator_acquisitions": chunks.iterator_acquisitions,
            "observation_yields": chunks.observation_yields,
        },
        "oracle": {
            "oracle_total_units": units,
            "oracle_total_units_bit_length": abs(units).bit_length(),
            "bound_bits": 2162,
        },
        "actual": {
            "observation_count": rows,
            "total_log_likelihood": result.total_log_likelihood,
            "total_log_likelihood_hex": result.total_log_likelihood.hex(),
        },
        "elapsed_seconds": elapsed,
        "throughput_rows_per_second": rows / elapsed if elapsed else float(rows),
        "memory": {
            "tracemalloc_peak_bytes": peak,
            "rss_peak_bytes": rss_after,
            "rss_delta_bytes": max(0, rss_after - rss_before),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sha = _head(root)
    value: dict[str, object] = {
        "schema_version": "5",
        "run": {
            "git_sha": sha,
            "candidate_git_sha": sha,
            "git_dirty": False,
            "generator": "fixed-supported-family-v1",
            "source_contract": "public-iterable-data-source-v1",
            "python": {
                "implementation": platform.python_implementation(),
                "version": platform.python_version(),
            },
            "platform": platform.platform(),
            "measurement_workers": MEASUREMENT_WORKERS,
            "methodology": dict(METHODOLOGY),
        },
        "cells": [
            _cell(
                rows,
                budget,
                family=family,
                observation=observation,
                parameters=parameters,
                contribution_hex=contribution,
            )
            for family, observation, parameters, contribution in FAMILY_CASES
            for rows in ROWS
            for budget in BUDGETS
        ],
    }
    value["artifact_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if _head(root) != sha:
        raise SystemExit("refusing evidence after checkout changed")
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
