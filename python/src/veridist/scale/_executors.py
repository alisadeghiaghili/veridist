"""Executors: reading the partitions of a pass serially, on threads or on processes.

``fit(..., executor=...)`` takes ``None`` (the default), ``"serial"``, ``SerialExecutor()``,
``ThreadExecutor(workers=n)`` or ``ProcessExecutor(workers=n)``. Every data pass of the fit
(the single pass of a one-pass family, every planned pass of a multi-pass one) is split into
one task per partition (:mod:`veridist.scale._tasks`). Workers fold their partition into an
exact state and return its canonical bytes; the parent merges the states as the fixed
balanced tree over the partition ordinals (``merge_tree``). Merging is exact, so the
estimate, the state bytes and the state digest are identical for every executor and every
number of workers, and the arrival order of the results cannot matter.

``None`` asks for nothing: the partitions are read one after the other as before the option
existed, a failed read is not repeated and no execution record is made. The other executors
record what they did in ``ScaleProvenance.execution``, and a partition whose read fails with
a transient input or output error is read again, a bounded number of times
(:mod:`veridist.scale._tasks`).

Downgrade. A pass is read serially, never silently, when the request cannot be honoured:
the source says why (``parallel_obstacle`` of :mod:`veridist.scale._sources`: Arrow streams,
Polars plans, DuckDB relations, data held in memory for worker processes), it has fewer than
two partitions, or it is the first pass of a spooled fit, which writes the spool one
partition at a time in canonical order. The record of that pass holds the reason, a
:class:`~veridist.scale._sources.DowngradeReason`. Later passes of a spooled fit read the
spool's files, in parallel when the executor allows, each read checked against the digests
as before.

Workers. The number of workers is the request, capped by ``MAX_WORKERS``, by the number of
cores (``os.cpu_count()``) and by the number of partitions. The pool is created when the first
pass that runs in parallel starts, is used by every later pass of the fit and is shut down
when the fit ends, however it ends: nothing is left running. A process pool starts its workers
with the ``spawn`` method on every platform: ``fork`` is unsafe in a process that has started
threads (Arrow, Polars and DuckDB all do), and a forkserver would leave a helper process behind
that outlives the fit. A spawned worker imports its modules once, when the pool starts
(:func:`~veridist.scale._tasks.initialize_worker`). A script that starts a process pool on a
platform that spawns must be guarded by ``if __name__ == "__main__":``.

Failures. A task that fails raises in the parent as the error it raised in the worker, the
typed ``ScaleSourceError`` or ``ScaleStateError`` of the serial path included. When several
partitions fail, the error of the one first in canonical order is raised, as it would be
serially: partitions after it are cancelled, those before it are waited for. A
``KeyboardInterrupt`` (or any other exception that is not an ``Exception``) cancels the
tasks that have not started and ends the fit; the pool is shut down after the tasks in
progress return.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures import Executor as PoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from types import TracebackType
from typing import Any, ClassVar, Final

from veridist.engine.provenance import ScaleExecutionPassProvenance, ScaleExecutionProvenance
from veridist.scale._sources import (
    BatchReader,
    DowngradeReason,
    ExecutorKind,
    Partition,
    Source,
    is_retryable,
    parallel_obstacle,
)
from veridist.scale._tasks import (
    MAX_PARTITION_RETRIES,
    Job,
    PartitionTask,
    TaskOutcome,
    initialize_worker,
    merge_tree,
    run_partition_task,
)

MAX_WORKERS: Final = 64
"""The most workers a pool has, whatever is requested."""

WAIT_SLICE_SECONDS: Final = 0.5
"""How long the parent waits for results before it looks again; also how soon an interrupt
is noticed on a platform where a wait without a timeout cannot be interrupted."""

START_METHOD: Final = "spawn"
"""How worker processes are started, on every platform."""

THREAD_NAME_PREFIX: Final = "veridist-scale"

NUMPY_MODULE: Final = "numpy"

PoolFactory = Callable[[int, tuple[str, ...]], PoolExecutor]
"""Makes a pool of the given number of workers; the modules are what a process worker imports.

Only tests give an executor a factory of their own."""


def make_thread_pool(workers: int, modules: tuple[str, ...]) -> PoolExecutor:
    """A pool of ``workers`` threads (they share this process's modules)."""

    return ThreadPoolExecutor(max_workers=workers, thread_name_prefix=THREAD_NAME_PREFIX)


def make_process_pool(workers: int, modules: tuple[str, ...]) -> PoolExecutor:
    """A pool of ``workers`` spawned processes that import ``modules`` as they start."""

    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context(START_METHOD),
        initializer=initialize_worker,
        initargs=(modules,),
    )


def available_cpus() -> int:
    """The number of cores of the host (at least one)."""

    return os.cpu_count() or 1


def _check_workers(workers: object) -> None:
    if type(workers) is not int:
        raise TypeError("workers must be a built-in integer")
    if workers < 1:
        raise ValueError("workers must be at least 1")


def _check_factory(factory: object) -> None:
    if factory is not None and not callable(factory):
        raise TypeError("pool_factory must be callable or None")


@dataclass(frozen=True, slots=True)
class SerialExecutor:
    """Read the partitions one after the other, and record it."""

    kind: ClassVar[ExecutorKind] = ExecutorKind.SERIAL
    workers: ClassVar[int] = 1


@dataclass(frozen=True, slots=True)
class ThreadExecutor:
    """Read the partitions on ``workers`` threads of this process.

    numpy releases the interpreter lock in parts of its work, so the gain is modest.
    """

    workers: int
    pool_factory: PoolFactory | None = field(default=None, kw_only=True, repr=False, compare=False)
    kind: ClassVar[ExecutorKind] = ExecutorKind.THREADS

    def __post_init__(self) -> None:
        _check_workers(self.workers)
        _check_factory(self.pool_factory)


@dataclass(frozen=True, slots=True)
class ProcessExecutor:
    """Read the partitions on ``workers`` worker processes, started with ``spawn``."""

    workers: int
    pool_factory: PoolFactory | None = field(default=None, kw_only=True, repr=False, compare=False)
    kind: ClassVar[ExecutorKind] = ExecutorKind.PROCESSES

    def __post_init__(self) -> None:
        _check_workers(self.workers)
        _check_factory(self.pool_factory)


Executor = SerialExecutor | ThreadExecutor | ProcessExecutor
"""What ``fit`` accepts besides ``None`` and ``"serial"``."""

SERIAL_NAME: Final = "serial"


def resolve_executor(executor: object) -> Executor | None:
    """Validate the ``executor`` argument of ``fit``.

    ``None`` stays ``None``, ``"serial"`` is a :class:`SerialExecutor`, an executor is
    itself. ``TypeError`` for anything else that is not a string, ``ValueError`` for another
    string.
    """

    if executor is None:
        return None
    if isinstance(executor, SerialExecutor | ThreadExecutor | ProcessExecutor):
        return executor
    if type(executor) is not str:
        raise TypeError("executor must be None, 'serial' or an executor")
    if executor != SERIAL_NAME:
        raise ValueError("executor must be 'serial' or an executor object")
    return SerialExecutor()


def _gather(futures: dict[Future[TaskOutcome], int], outcomes: list[TaskOutcome]) -> None:
    """Collect the outcomes of ``futures`` (each mapped to its partition ordinal) into ``outcomes``.

    The error of the lowest ordinal that failed is raised once every task before it has
    returned; tasks after it are cancelled and not waited for. An exception that is not an
    ``Exception`` is raised at once.
    """

    pending = set(futures)
    failures: dict[int, Exception] = {}
    while pending:
        done, pending = wait(pending, timeout=WAIT_SLICE_SECONDS, return_when=FIRST_COMPLETED)
        for future in done:
            error = future.exception()
            if error is None:
                outcomes.append(future.result())
            elif isinstance(error, Exception):
                failures[futures[future]] = error
            else:
                raise error
        if failures:
            first = min(failures)
            for future in pending:
                if futures[future] > first:
                    future.cancel()
            pending = {future for future in pending if futures[future] < first}
    if failures:
        raise failures[min(failures)]


class PassEngine:
    """Runs the passes of one fit on its executor and records how each was executed.

    Use it as a context manager: leaving it shuts the pool down, whether the fit returned,
    failed or was interrupted. ``run`` reads one pass and returns the merged state.
    """

    def __init__(self, executor: Executor | None, max_rows: int) -> None:
        self._executor = executor
        self._max_rows = max_rows
        self._pool: PoolExecutor | None = None
        self._records: list[ScaleExecutionPassProvenance] = []

    def __enter__(self) -> PassEngine:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)

    def provenance(self) -> ScaleExecutionProvenance | None:
        """The record of the passes run so far, or ``None`` if no executor was given."""

        if self._executor is None:
            return None
        return ScaleExecutionProvenance(
            self._executor.kind.value, self._executor.workers, tuple(self._records)
        )

    def _mode(
        self, source: Source, partitions: Sequence[Partition]
    ) -> tuple[ExecutorKind, int, DowngradeReason | None]:
        """How the pass is read: the kind, the number of workers and the downgrade reason."""

        executor = self._executor
        if executor is None or executor.kind is ExecutorKind.SERIAL:
            return ExecutorKind.SERIAL, 1, None
        reason = parallel_obstacle(source, executor.kind, partitions)
        if reason is None and len(partitions) < 2:
            reason = DowngradeReason.SINGLE_PARTITION
        if reason is not None:
            return ExecutorKind.SERIAL, 1, reason
        workers = min(executor.workers, MAX_WORKERS, available_cpus(), len(partitions))
        return executor.kind, workers, None

    def _pool_for(
        self, executor: ThreadExecutor | ProcessExecutor, workers: int, shipped: BatchReader
    ) -> PoolExecutor:
        """The fit's pool, made on first use with the modules ``shipped`` needs."""

        if self._pool is None:
            modules = (NUMPY_MODULE, *getattr(shipped, "WORKER_MODULES", ()))
            default = (
                make_process_pool if executor.kind is ExecutorKind.PROCESSES else make_thread_pool
            )
            self._pool = (executor.pool_factory or default)(workers, modules)
        return self._pool

    def _in_parallel(
        self,
        executor: ThreadExecutor | ProcessExecutor,
        workers: int,
        tasks: list[PartitionTask],
        outcomes: list[TaskOutcome],
    ) -> None:
        pool = self._pool_for(executor, workers, tasks[0].source)
        futures: dict[Future[TaskOutcome], int] = {}
        try:
            for task in tasks:
                futures[pool.submit(run_partition_task, task)] = task.partition.ordinal
            _gather(futures, outcomes)
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    def run(self, source: Source, partitions: Sequence[Partition], job: Job) -> Any:
        """Read ``partitions`` of ``source`` as one pass and return the merged ``job`` value.

        Errors are those of the tasks (see the module documentation); the pass is recorded
        whether it returned or failed.
        """

        kind, workers, reason = self._mode(source, partitions)
        retryable = self._executor is not None and is_retryable(source)
        retries = MAX_PARTITION_RETRIES if retryable else 0
        reader: BatchReader = source
        if kind is ExecutorKind.PROCESSES:
            worker_source = getattr(source, "worker_source", None)
            reader = source if worker_source is None else worker_source()
        tasks = [
            PartitionTask(reader, partition, self._max_rows, job, retries)
            for partition in partitions
        ]
        outcomes: list[TaskOutcome] = []
        try:
            if kind is ExecutorKind.SERIAL:
                outcomes.extend(run_partition_task(task) for task in tasks)
            else:
                assert isinstance(self._executor, ThreadExecutor | ProcessExecutor)
                self._in_parallel(self._executor, workers, tasks, outcomes)
        finally:
            self._records.append(
                ScaleExecutionPassProvenance(
                    kind.value,
                    workers,
                    len(tasks),
                    sum(outcome.retries for outcome in outcomes),
                    None if reason is None else reason.value,
                )
            )
        ordered = sorted(outcomes, key=lambda outcome: outcome.ordinal)
        return merge_tree(
            [job.restore(outcome.data) for outcome in ordered],
            lambda left, right: left.merge(right),
            job.identity(),
        )


__all__ = [
    "MAX_WORKERS",
    "START_METHOD",
    "WAIT_SLICE_SECONDS",
    "Executor",
    "PassEngine",
    "ProcessExecutor",
    "SerialExecutor",
    "ThreadExecutor",
    "available_cpus",
    "resolve_executor",
]
