"""Stores that fail on cue, sources that log their reads and comparisons shared by the
checkpoint tests of the scale fit.

Nothing here starts a thread or a process: a store that fails is a parent-side seam, so an
interruption lands at the same commit whatever executor reads the partitions.
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from tests.unit.scale_multi_pass_oracle import gumbel_sample, weibull_sample
from veridist.scale._fit import ScaleFitResult
from veridist.scale._partition_store import SQLitePartitionStore
from veridist.scale._sources import DEFAULT_MAX_ROWS, DowngradeReason, ExecutorKind, Partition

ONE_PASS = ("exponential", "normal", "gamma", "lognormal")
MULTI_PASS = ("weibull_min", "gumbel_right")


class Interruption(Exception):
    """The failure a test injects at a chosen commit."""


def values_for(family: str, size: int = 720, seed: int = 5) -> np.ndarray:
    """Seeded data that every family accepts."""

    rng = np.random.default_rng(seed)
    if family == "exponential":
        return rng.exponential(3.0, size)
    if family == "normal":
        return rng.normal(10.0, 3.0, size)
    if family == "gamma":
        return rng.gamma(3.0, 2.0, size)
    if family == "lognormal":
        return rng.lognormal(1.0, 0.5, size)
    if family == "weibull_min":
        return weibull_sample(1.7, 40.0, size, seed)
    return gumbel_sample(5.0, 2.0, size, seed)


def bits(result: ScaleFitResult) -> tuple[bytes, ...]:
    """Every number of a fit and of its uncertainty, as the bytes of the doubles."""

    success: Any = result.fit
    numbers = [*success.parameters.values(), success.log_likelihood]
    uncertainty = success.uncertainty()
    for row in uncertainty.covariance:
        numbers.extend(row)
    return tuple(struct.pack("<d", float(value)) for value in numbers)


def same_as(result: ScaleFitResult, base: ScaleFitResult) -> bool:
    """Whether ``result`` is ``base`` in everything but how it was executed and checkpointed."""

    return (
        result.fit == base.fit
        and bits(result) == bits(base)
        and replace(result.scale, checkpoint=None, execution=None, spool=None)
        == replace(base.scale, checkpoint=None, execution=None, spool=None)
    )


class Interrupting(SQLitePartitionStore):
    """A store that raises on the ``after + 1``-th partition commit, before writing it.

    Exactly ``after`` states of this store object are committed when it fails, whatever
    order the partitions arrive in. ``after_passes`` does the same for the ledger.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        after: int | None = None,
        after_passes: int | None = None,
        error: BaseException | None = None,
    ) -> None:
        super().__init__(path)
        self.after = after
        self.after_passes = after_passes
        self.error = Interruption() if error is None else error
        self.partition_commits = 0
        self.pass_commits = 0

    def commit_partition(self, *arguments: Any) -> bool:
        if self.after is not None and self.partition_commits >= self.after:
            raise self.error
        self.partition_commits += 1
        return super().commit_partition(*arguments)

    def commit_pass(self, *arguments: Any, **options: Any) -> bool:
        if self.after_passes is not None and self.pass_commits >= self.after_passes:
            raise self.error
        self.pass_commits += 1
        return super().commit_pass(*arguments, **options)


class ReadLog:
    """A source that wraps another, logs which partitions were read and passes the rest on."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.reads: list[int] = []

    @property
    def column(self) -> Any:
        return getattr(self.inner, "column", None)

    def partitions(self) -> tuple[Partition, ...]:
        found: tuple[Partition, ...] = self.inner.partitions()
        return found

    def parallel_obstacle(self, kind: ExecutorKind, partitions: Any) -> DowngradeReason | None:
        obstacle: DowngradeReason | None = self.inner.parallel_obstacle(kind, partitions)
        return obstacle

    def batches(self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS) -> Iterator[Any]:
        self.reads.append(partition.ordinal)
        return self.inner.batches(partition, max_rows=max_rows)


class Tracker:
    """Counts the SQLite connections a block opens and closes."""

    def __init__(self) -> None:
        self.opened = 0
        self.closed = 0
        self.statements: list[str] = []

    @contextmanager
    def watching(self) -> Iterator[Tracker]:
        real = sqlite3.connect

        def connect(*arguments: Any, **options: Any) -> Any:
            connection = real(*arguments, **options)
            self.opened += 1
            return _Closing(connection, self)

        with mock.patch("sqlite3.connect", connect):
            yield self


class _Closing:
    """A connection that tells its tracker when it is closed; everything else is the real one."""

    def __init__(self, connection: sqlite3.Connection, tracker: Tracker) -> None:
        self._connection = connection
        self._tracker = tracker

    def close(self) -> None:
        self._tracker.closed += 1
        self._connection.close()

    def execute(self, statement: str, *arguments: Any) -> Any:
        self._tracker.statements.append(" ".join(statement.split()))
        return self._connection.execute(statement, *arguments)

    def __getattr__(self, name: str) -> Callable[..., Any]:
        attribute: Callable[..., Any] = getattr(self._connection, name)
        return attribute


def rows(path: Path, query: str, *arguments: Any) -> list[tuple[Any, ...]]:
    """Rows of a query against a store file, read through a connection that is closed."""

    connection = sqlite3.connect(path)
    try:
        return connection.execute(query, arguments).fetchall()
    finally:
        connection.close()


def execute(path: Path, statement: str, *arguments: Any) -> None:
    """Run one statement against a store file (a test tampering with it) and commit it."""

    connection = sqlite3.connect(path)
    try:
        with connection:
            connection.execute(statement, arguments)
    finally:
        connection.close()


def make_single_row_database(path: Path) -> None:
    """A database of the single-row checkpoint store (format 1), which is not a job."""

    from veridist.engine.checkpoint import CheckpointRecord, SQLiteCheckpointStore

    record = CheckpointRecord.create(
        format_version=1,
        source_id="src_" + "0" * 32,
        source_schema="schema",
        source_revision="revision",
        reducer_id="reducer",
        accumulator_schema="accumulator",
        plan_digest="plan",
        cursor=0,
        committed_ranges=(),
        generation=0,
        operation_token=None,
        operation_digest=None,
        state=b"",
    )
    SQLiteCheckpointStore.create(path, record)
