"""Collect one platform's real checkpointed-CSV v1 execution evidence.

This command intentionally produces a *raw platform fragment*, never a
checked-in release assertion.  A workflow must collect fragments on all three
platforms and assemble them only after the matrix has actually run.

Each fragment holds six scenarios per row count: an uninterrupted run, a
cooperative cancel followed by a resume, a cancel that is left in place, a
resume against a source that changed in between (which must be refused with the
checkpoint untouched), a replay of already committed offset chunks (which must
change nothing), and a child process that is terminated after its first commit
and then resumed by the parent.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform as host_platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId
from veridist.engine.checkpoint import CheckpointRecord, SQLiteCheckpointStore
from veridist.engine.errors import EngineContractError
from veridist.execution import (
    CheckpointedCsvFitResult,
    fit_exponential_checkpointed_chunks,
    fit_exponential_checkpointed_csv,
)
from veridist.families.exponential import ExponentialFitSuccess

try:  # direct script execution puts tools/ on sys.path
    from process_memory import peak_rss_bytes
except ImportError:  # imported as ``tools.<module>``
    from tools.process_memory import peak_rss_bytes

SCHEMA_VERSION = 2
ROWS = (10_000, 100_000, 1_000_000)
SCENARIOS = (
    "complete",
    "retry_resume",
    "cancel",
    "source_mutated",
    "chunk_replay",
    "process_killed",
)
_SHA_LENGTH = 40
_SCHEMA = CsvLifetimeSchema("time", "event_observed")
_LIMITS = CsvLifetimeLimits(65_536, 65_536)
# The child commits many small batches so that a termination request lands while it is
# still working, however fast the host's storage is.
_KILL_LIMITS = CsvLifetimeLimits(2_048, 2_048)
_KILL_ATTEMPTS = 3
_FIRST_COMMIT_TIMEOUT_SECONDS = 60.0
_PROCESS_TIMEOUT_SECONDS = 30.0
_POLL_INTERVAL_SECONDS = 0.002
_REPLAY_CHUNKS = 10
_CHILD_FLAG = "--child-checkpointed-fit"
_MUTATION_ROW = b"0.001,1\n"


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


def _source_id(rows: int) -> PublicSourceId:
    digest = hashlib.md5(str(rows).encode("ascii"), usedforsecurity=False).hexdigest()
    return PublicSourceId(f"src_{digest}")


def _write_fixture(path: Path, rows: int) -> tuple[str, int]:
    """Write a deterministic, synthetic lifetime source and return its revision."""

    with path.open("w", encoding="utf-8", newline="") as target:
        target.write("time,event_observed\n")
        for index in range(rows):
            target.write(f"{(index % 997 + 1) / 1000:.3f},{0 if index % 3 == 0 else 1}\n")
    return _file_digest(path), path.stat().st_size


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _fit_digest_of(fit: object) -> str:
    if not isinstance(fit, ExponentialFitSuccess):
        raise RuntimeError("checkpointed evidence run did not complete")
    value = {
        "event_count": fit.event_count,
        "observation_count": fit.observation_count,
        "rate": fit.rate,
        "total_time": fit.total_time,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _fit_digest(result: CheckpointedCsvFitResult) -> str:
    if result.code != "COMPLETE":
        raise RuntimeError("checkpointed evidence run did not complete")
    return _fit_digest_of(result.fit)


def _fit_csv(
    source: Path,
    store: SQLiteCheckpointStore,
    revision: str,
    rows: int,
    *,
    cancel_at: int | None = None,
    limits: CsvLifetimeLimits = _LIMITS,
) -> CheckpointedCsvFitResult:
    """Run the checkpointed CSV fit, optionally cancelling once the cursor reaches ``cancel_at``."""

    return fit_exponential_checkpointed_csv(
        path=source,
        schema=_SCHEMA,
        source_id=_source_id(rows),
        limits=limits,
        store=store,
        source_revision=revision,
        cancel=None if cancel_at is None else (lambda cursor: cursor >= cancel_at),
    )


def _delete_and_verify(paths: tuple[Path, ...]) -> bool:
    """Prove cancellation returned with no open source or checkpoint handles."""

    for path in paths:
        if path.exists():
            path.unlink()
    return all(not path.exists() for path in paths)


def _offset_chunks(rows: int) -> Iterator[tuple[int, bytes]]:
    """Yield the fixture as offset-form canonical JSON chunks for the chunk reducer."""

    size = max(1, -(-rows // _REPLAY_CHUNKS))
    for start in range(0, rows, size):
        stop = min(rows, start + size)
        batch = [
            [float(f"{(index % 997 + 1) / 1000:.3f}"), index % 3 != 0]
            for index in range(start, stop)
        ]
        yield start, json.dumps(batch, separators=(",", ":")).encode("utf-8")


def _terminate_child_after_first_commit(
    command: list[str],
    store_path: Path,
    *,
    first_commit_timeout: float = _FIRST_COMMIT_TIMEOUT_SECONDS,
    process_timeout: float = _PROCESS_TIMEOUT_SECONDS,
) -> tuple[int, int]:
    """Start ``command``, call ``Popen.terminate`` once a commit is visible, and reap it.

    Returns the cursor first observed in the store and the child's exit code. Every wait is
    bounded: the poll loop ends at ``first_commit_timeout`` and the final reap at
    ``process_timeout``, and a child that is still alive afterwards is killed.
    """

    probe = SQLiteCheckpointStore(store_path, timeout=0.05)
    process = subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    try:
        deadline = time.monotonic() + first_commit_timeout
        observed = 0
        while observed == 0:
            if process.poll() is not None:
                raise RuntimeError("child process exited before its first commit was observed")
            if time.monotonic() >= deadline:
                raise RuntimeError("child process did not commit within the allowed time")
            try:
                observed = probe.read().cursor
            except EngineContractError:
                observed = 0  # the child holds the write lock; look again
            if observed == 0:
                time.sleep(_POLL_INTERVAL_SECONDS)
        process.terminate()
        try:
            exit_code = process.wait(timeout=process_timeout)
        except subprocess.TimeoutExpired:
            raise RuntimeError("child process did not stop after terminate") from None
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=process_timeout)
    return observed, exit_code


def _kill_child_mid_run(
    root: Path, source: Path, revision: str, rows: int
) -> tuple[SQLiteCheckpointStore, int, int, int]:
    """Terminate a child that is fitting ``source`` and return its interrupted store.

    A child that finishes before the termination lands is not a kill, so it is retried on a
    fresh store, at most ``_KILL_ATTEMPTS`` times; running out of attempts is an error, never
    a recorded success.
    """

    for attempt in range(1, _KILL_ATTEMPTS + 1):
        store = _initial_store(root / f"killed-{attempt}.sqlite3", revision, _source_id(rows))
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            _CHILD_FLAG,
            "--csv",
            str(source),
            "--store",
            str(store.path),
            "--rows",
            str(rows),
            "--revision",
            revision,
        ]
        _, exit_code = _terminate_child_after_first_commit(command, store.path)
        cursor = SQLiteCheckpointStore(store.path).read().cursor
        if exit_code != 0 and 0 < cursor < rows:
            return store, cursor, exit_code, attempt
    raise RuntimeError("child process completed before it could be terminated")


def _child_main(argv: list[str]) -> int:
    """Entry point of the child process: run the checkpointed fit with small commit batches."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args(argv)
    result = _fit_csv(
        args.csv,
        SQLiteCheckpointStore(args.store),
        args.revision,
        args.rows,
        limits=_KILL_LIMITS,
    )
    return 0 if result.code == "COMPLETE" else 3


def _run_scenario(
    rows: int, scenario: str, baseline_digest: str | None = None
) -> dict[str, object]:
    """Run one real observation of a scenario; see the module docstring for the list."""

    if scenario not in SCENARIOS:
        raise ValueError("unknown execution evidence scenario")
    with tempfile.TemporaryDirectory(prefix="veridist-v1-execution-") as directory:
        root = Path(directory)
        source = root / "lifetimes.csv"
        revision, source_bytes = _write_fixture(source, rows)
        before_rss = peak_rss_bytes()
        resources_released = False
        attempt_count = 0
        interrupted_cursor = 0
        final_cursor = 0
        cancellation_observed, retry_initial_code = False, None
        result_digest: str | None = None
        extra: dict[str, object] = {}
        interrupt_at = max(1, rows // 2)
        needs_baseline = scenario in {"retry_resume", "chunk_replay", "process_killed"}
        if needs_baseline and baseline_digest is None:
            baseline_store = _initial_store(root / "baseline.sqlite3", revision, _source_id(rows))
            baseline_digest = _fit_digest(_fit_csv(source, baseline_store, revision, rows))
        if scenario == "complete":
            store = _initial_store(root / "complete.sqlite3", revision, _source_id(rows))
            attempt_count += 1
            completed = _fit_csv(source, store, revision, rows)
            result_code = completed.code
            result_digest = _fit_digest(completed)
            final_cursor = store.read().cursor
        elif scenario == "retry_resume":
            store = _initial_store(root / "retry.sqlite3", revision, _source_id(rows))
            attempt_count += 1
            interrupted = _fit_csv(source, store, revision, rows, cancel_at=interrupt_at)
            interrupted_cursor = store.read().cursor
            if interrupted.code != "CANCELLED" or interrupted_cursor != interrupt_at:
                raise RuntimeError("retry/resume interruption did not retain a nonzero checkpoint")
            attempt_count += 1
            resumed = _fit_csv(source, SQLiteCheckpointStore(store.path), revision, rows)
            result_code = resumed.code
            result_digest = _fit_digest(resumed)
            final_cursor = store.read().cursor
            cancellation_observed, retry_initial_code = True, interrupted.code
        elif scenario == "cancel":
            store = _initial_store(root / "cancel.sqlite3", revision, _source_id(rows))
            attempt_count += 1
            cancelled = _fit_csv(source, store, revision, rows, cancel_at=interrupt_at)
            result_code = cancelled.code
            interrupted_cursor = store.read().cursor
            if cancelled.code != "CANCELLED" or interrupted_cursor != interrupt_at:
                raise RuntimeError("cancellation did not retain the expected nonzero checkpoint")
            final_cursor = interrupted_cursor
            cancellation_observed = True
            resources_released = _delete_and_verify(
                (
                    source,
                    store.path,
                    store.path.with_name(store.path.name + "-wal"),
                    store.path.with_name(store.path.name + "-shm"),
                )
            )
        elif scenario == "source_mutated":
            store = _initial_store(root / "mutated.sqlite3", revision, _source_id(rows))
            attempt_count += 1
            interrupted = _fit_csv(source, store, revision, rows, cancel_at=interrupt_at)
            interrupted_cursor = store.read().cursor
            if interrupted.code != "CANCELLED" or interrupted_cursor != interrupt_at:
                raise RuntimeError("mutation scenario did not retain a nonzero checkpoint")
            before = store.read()
            with source.open("ab") as target:
                target.write(_MUTATION_ROW)
            mutated_revision = _file_digest(source)
            if mutated_revision == revision:
                raise RuntimeError("source mutation did not change the source digest")
            attempt_count += 1
            stale = _fit_csv(source, store, revision, rows)
            rebound = _fit_csv(source, store, mutated_revision, rows)
            after = store.read()
            if {stale.code, rebound.code} != {"SOURCE_REVISION_MISMATCH"}:
                raise RuntimeError("a changed source was not refused with SOURCE_REVISION_MISMATCH")
            if after != before:
                raise RuntimeError("a refused resume changed the checkpoint")
            result_code = stale.code
            final_cursor = after.cursor
            cancellation_observed, retry_initial_code = True, interrupted.code
            extra = {
                "mutated_source_sha256": mutated_revision,
                "rebound_revision_code": rebound.code,
                "checkpoint_unchanged": True,
                "generation_before": before.generation,
                "generation_after": after.generation,
            }
        elif scenario == "chunk_replay":
            store = _initial_store(root / "replay.sqlite3", revision, _source_id(rows))
            attempt_count += 1
            first = fit_exponential_checkpointed_chunks(
                store=store, source_revision=revision, chunks=list(_offset_chunks(rows))
            )
            committed = store.read()
            attempt_count += 1
            replayed = fit_exponential_checkpointed_chunks(
                store=store, source_revision=revision, chunks=list(_offset_chunks(rows))
            )
            after = store.read()
            if after != committed or replayed != first:
                raise RuntimeError("replaying committed chunks changed the checkpoint or result")
            result_code = "COMPLETE"
            result_digest = _fit_digest_of(replayed)
            final_cursor = after.cursor
            extra = {
                "checkpoint_unchanged": True,
                "generation_before": committed.generation,
                "generation_after": after.generation,
                "replayed_chunk_count": len(list(_offset_chunks(rows))),
            }
        else:
            store, interrupted_cursor, exit_code, kill_attempts = _kill_child_mid_run(
                root, source, revision, rows
            )
            attempt_count += 2
            resumed = _fit_csv(source, SQLiteCheckpointStore(store.path), revision, rows)
            result_code = resumed.code
            result_digest = _fit_digest(resumed)
            final_cursor = store.read().cursor
            retry_initial_code = "PROCESS_TERMINATED"
            extra = {"child_exit_code": exit_code, "kill_attempts": kill_attempts}
        peak_rss = peak_rss_bytes()
        if peak_rss < before_rss:
            raise RuntimeError("RSS measurement moved backwards")
        canonical_equal = True
        if needs_baseline:
            canonical_equal = result_digest == baseline_digest
            if not canonical_equal:
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
            "result_code": result_code,
            "retry_initial_code": retry_initial_code,
            "source_bytes": source_bytes,
            "source_sha256": revision,
            "result_sha256": result_digest,
            **extra,
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
        "schema_version": SCHEMA_VERSION,
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
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"wrote raw {args.platform} execution evidence: {args.output.name}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == _CHILD_FLAG:
        raise SystemExit(_child_main(sys.argv[2:]))
    raise SystemExit(main())
