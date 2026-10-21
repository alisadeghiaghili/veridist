"""Sources, pools and leak checks shared by the executor tests.

``ThreadExecutor`` and ``ProcessExecutor`` tests start threads and, in a few modules,
processes; this module's name is what tells ``tests/process_isolation.py`` that a test module
needs a fresh interpreter when a forked mutation child reaches it.
"""

from __future__ import annotations

import multiprocessing
import threading
import time
import unittest
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from tests.unit.scale_arrow_fixtures import write_parquet
from veridist.scale import _executors
from veridist.scale._arrow import ParquetSource
from veridist.scale._executors import PassEngine, ThreadExecutor
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    DowngradeReason,
    ExecutorKind,
    FingerprintLevel,
    Partition,
)

GATE_SECONDS = 10.0
"""The longest a scripted partition waits for another; a test that needs more has failed."""

LEAK_CHECKS = 20
LEAK_SLEEP = 0.05


class PlainSource:
    """A multi-partition in-memory source that says nothing about parallel reads."""

    def __init__(self, parts: list[np.ndarray]) -> None:
        self.parts = parts
        self.reads: list[int] = []
        self.hooks: dict[int, Callable[[int], None]] = {}
        self._lock = threading.Lock()

    def partitions(self) -> tuple[Partition, ...]:
        return tuple(
            Partition(f"p{i}", i, len(part), None, FingerprintLevel.NONE)
            for i, part in enumerate(self.parts)
        )

    def batches(self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS) -> Iterator[Any]:
        with self._lock:
            self.reads.append(partition.ordinal)
            number = self.reads.count(partition.ordinal)
        hook = self.hooks.get(partition.ordinal)
        if hook is not None:
            hook(number)
        part = self.parts[partition.ordinal]
        for start in range(0, len(part), max_rows):
            yield part[start : start + max_rows]


class PartSource(PlainSource):
    """A multi-partition source that declares what its parallel reads need."""

    def __init__(self, parts: list[np.ndarray], obstacle: DowngradeReason | None = None) -> None:
        super().__init__(parts)
        self.obstacle = obstacle
        self.asked: list[tuple[ExecutorKind, int]] = []

    def parallel_obstacle(self, kind: ExecutorKind, partitions: Any) -> DowngradeReason | None:
        self.asked.append((kind, len(partitions)))
        return self.obstacle


class Gate:
    """Lets one partition wait, bounded, until a condition is set by another."""

    def __init__(self) -> None:
        self.event = threading.Event()

    def wait(self, _number: int = 1) -> None:
        if not self.event.wait(GATE_SECONDS):
            raise AssertionError("a gate was not opened in time")

    def open(self, _number: int = 1) -> None:
        self.event.set()


class FakePool:
    """A pool that runs each task in ``submit`` and records how it is used."""

    def __init__(self, workers: int = 1, modules: tuple[str, ...] = ()) -> None:
        self.workers = workers
        self.modules = modules
        self.tasks: list[Any] = []
        self.futures: list[Future[Any]] = []
        self.shutdowns: list[tuple[bool, bool]] = []
        self.events: list[str] = []

    def submit(self, function: Callable[..., Any], *arguments: Any) -> Future[Any]:
        self.tasks.append(arguments[0] if arguments else None)
        future: Future[Any] = Future()
        try:
            future.set_result(function(*arguments))
        except BaseException as error:
            future.set_exception(error)
        self.futures.append(future)
        return future

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.events.append("shutdown")
        self.shutdowns.append((wait, cancel_futures))


class ScriptedPool(FakePool):
    """A pool whose tasks run only when the test says so, in the order it says.

    ``submit`` queues the task and returns a pending future. ``wait`` below stands in for
    ``concurrent.futures.wait``: each call runs the next scripted step (a list of task
    positions) and then reports what is done. A step may name a task that was cancelled; it
    is skipped. When the script is used up the remaining tasks run in order, so a test
    that ends early cannot hang.
    """

    def __init__(self, workers: int = 1, modules: tuple[str, ...] = (), script: Any = ()) -> None:
        super().__init__(workers, modules)
        self.queue: list[tuple[Future[Any], Callable[..., Any], tuple[Any, ...]]] = []
        self.script: list[list[int]] = [list(step) for step in script]
        self.ran: list[int] = []

    def submit(self, function: Callable[..., Any], *arguments: Any) -> Future[Any]:
        self.tasks.append(arguments[0] if arguments else None)
        future: Future[Any] = Future()
        self.queue.append((future, function, arguments))
        self.futures.append(future)
        return future

    def run(self, position: int) -> None:
        future, function, arguments = self.queue[position]
        if not future.set_running_or_notify_cancel():
            return
        self.ran.append(position)
        try:
            future.set_result(function(*arguments))
        except BaseException as error:
            future.set_exception(error)

    def wait(self, pending: Any, timeout: Any = None, return_when: Any = None) -> Any:
        from concurrent.futures import wait

        step = (
            self.script.pop(0)
            if self.script
            else [position for position, item in enumerate(self.queue) if not item[0].done()]
        )
        for position in step:
            self.run(position)
        return wait(pending, timeout=0, return_when=return_when)


class PoolRecorder:
    """A pool factory that makes fake pools and remembers every call."""

    def __init__(self, make: Callable[[int, tuple[str, ...]], FakePool] = FakePool) -> None:
        self.make = make
        self.calls: list[tuple[int, tuple[str, ...]]] = []
        self.pools: list[Any] = []

    def __call__(self, workers: int, modules: tuple[str, ...]) -> Any:
        self.calls.append((workers, modules))
        pool = self.make(workers, modules)
        self.pools.append(pool)
        return pool


@contextmanager
def cores(count: int) -> Iterator[None]:
    """Run with ``count`` cores as far as the executors can tell."""

    with mock.patch("veridist.scale._executors.available_cpus", return_value=count):
        yield


class Scripted:
    """An engine on a scripted pool, with ``wait`` routed to it; a context manager."""

    def __init__(self, count: int, script: list[list[int]]) -> None:
        self.pools: list[ScriptedPool] = []

        def make(workers: int, modules: tuple[str, ...]) -> ScriptedPool:
            pool = ScriptedPool(workers, modules, script)
            self.pools.append(pool)
            return pool

        self.engine = PassEngine(ThreadExecutor(count, pool_factory=PoolRecorder(make)), 10)
        self.stack = ExitStack()

    def __enter__(self) -> PassEngine:
        self.stack.enter_context(cores(64))
        self.stack.enter_context(
            mock.patch.object(_executors, "wait", lambda *a, **k: self.pools[0].wait(*a, **k))
        )
        self.stack.enter_context(self.engine)
        return self.engine

    def __exit__(self, *exc: Any) -> Any:
        return self.stack.__exit__(*exc)

    @property
    def pool(self) -> ScriptedPool:
        return self.pools[0]


def parquet_dataset(
    root: Path, values: np.ndarray, files: int = 3, row_groups: int = 2
) -> ParquetSource:
    """``values`` as ``files`` Parquet files of ``row_groups`` row groups each, as one source."""

    pieces = np.array_split(values, files)
    for index, piece in enumerate(pieces):
        write_parquet(
            root / f"part-{index}.parquet",
            {"x": piece},
            row_group_size=max(1, -(-len(piece) // row_groups)),
        )
    return ParquetSource(root, "x")


def refuse_process_pool(workers: int, modules: tuple[str, ...]) -> Any:
    """Stands in for the process pool maker in tests that must not start worker processes."""

    raise AssertionError("a process pool was made in a test that runs no worker processes")


class NoLeaksTestCase(unittest.TestCase):
    """A test case that fails if a thread or a child process outlives the test.

    It also refuses to make a real process pool, unless the class sets ``real_processes``.
    A pool of the wrong kind (processes where threads were asked for) would otherwise leave a
    test waiting for spawned workers that cannot take its tasks; the refusal fails that test
    at once instead.
    """

    real_processes = False

    def setUp(self) -> None:
        if not self.real_processes:
            refusal = mock.patch("veridist.scale._executors.make_process_pool", refuse_process_pool)
            refusal.start()
            self.addCleanup(refusal.stop)
        self._children = {process.pid for process in multiprocessing.active_children()}
        self._threads = {thread.ident for thread in threading.enumerate()}
        self.addCleanup(self._assert_nothing_left_running)

    def _leaked(self) -> tuple[list[int | None], list[str]]:
        children = [
            process.pid
            for process in multiprocessing.active_children()
            if process.pid not in self._children
        ]
        threads = [
            thread.name
            for thread in threading.enumerate()
            if thread.ident not in self._threads and thread.is_alive()
        ]
        return children, threads

    def _assert_nothing_left_running(self) -> None:
        for _ in range(LEAK_CHECKS):
            children, threads = self._leaked()
            if not children and not threads:
                return
            time.sleep(LEAK_SLEEP)
        self.fail(f"left running: child processes {children}, threads {threads}")


class ChangedRows:
    """A Parquet source whose partitions claim one more row than the files hold.

    It is a module-level class so that worker processes can unpickle it.
    """

    WORKER_MODULES = ParquetSource.WORKER_MODULES

    def __init__(self, inner: ParquetSource) -> None:
        self.inner = inner

    def partitions(self) -> tuple[Partition, ...]:
        return tuple(
            Partition(
                item.id,
                item.ordinal,
                (item.rows or 0) + 1,
                item.fingerprint,
                item.fingerprint_level,
            )
            for item in self.inner.partitions()
        )

    def parallel_obstacle(self, kind: ExecutorKind, partitions: Any) -> DowngradeReason | None:
        return self.inner.parallel_obstacle(kind, partitions)

    def batches(self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS) -> Iterator[Any]:
        return self.inner.batches(partition, max_rows=max_rows)


def worker_facts() -> dict[str, Any]:
    """What a worker process looks like; run by a pool task, so it must be importable."""

    import os
    import sys

    facts: dict[str, Any] = {
        "pid": os.getpid(),
        "numpy": "numpy" in sys.modules,
        "pyarrow": "pyarrow" in sys.modules,
        "parquet": "pyarrow.parquet" in sys.modules,
        "scipy": "scipy" in sys.modules,
    }
    if "pyarrow" in sys.modules:
        facts["arrow_cpus"] = sys.modules["pyarrow"].cpu_count()
        facts["arrow_io"] = sys.modules["pyarrow"].io_thread_count()
    return facts
