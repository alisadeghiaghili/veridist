"""How a fit's pool is built: which maker, with which arguments, and how often.

``PassEngine._pool_for`` chooses the maker of the pool (a thread pool for threads, a process
pool for processes, or the factory an executor was given), hands it the number of workers
and the modules a worker imports, and keeps the one pool for every later pass. These tests
read that choice from the arguments a recording stand-in receives, without starting a
thread or a process, so a pool built the wrong way fails in microseconds.

The tests are deliberately free of real pools, of the executor classes and of the shared
executor fixtures: a mutation child that was forked by the mutation runner reaches them
before any module that needs a fresh interpreter (``tests/process_isolation.py``), so a
wrong pool is caught before anything slow can start. The tests stand in for an executor
with a plain object that has the three attributes ``_pool_for`` reads.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from veridist.scale import _executors
from veridist.scale._executors import PassEngine
from veridist.scale._sources import ExecutorKind


class Pool:
    """A pool that has nothing to run; it records how it was shut down."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.shutdowns: list[tuple[bool, bool]] = []

    def submit(self, *arguments: Any) -> Any:
        raise AssertionError("no task is submitted in these tests")

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shutdowns.append((wait, cancel_futures))


class Maker:
    """A pool maker that records every call and makes a new labelled pool each time."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.calls: list[tuple[Any, ...]] = []
        self.pools: list[Pool] = []

    def __call__(self, *arguments: Any) -> Pool:
        self.calls.append(arguments)
        pool = Pool(f"{self.label}-{len(self.calls)}")
        self.pools.append(pool)
        return pool


class Shipped:
    """A reader that names the modules its worker processes import."""

    WORKER_MODULES = ("first.module", "second.module")


class Silent:
    """A reader that names no module."""


def executor(kind: ExecutorKind, workers: int, factory: Any = None) -> Any:
    return SimpleNamespace(kind=kind, workers=workers, pool_factory=factory)


def engine_for(given: Any) -> PassEngine:
    return PassEngine(given, 10)


class PoolForTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processes = Maker("processes")
        self.threads = Maker("threads")
        patches = (
            mock.patch.object(_executors, "make_process_pool", self.processes),
            mock.patch.object(_executors, "make_thread_pool", self.threads),
        )
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_the_process_maker_is_the_default_of_processes_and_only_of_processes(self) -> None:
        given = executor(ExecutorKind.PROCESSES, 3)
        pool = engine_for(given)._pool_for(given, 3, Silent())
        self.assertIs(pool, self.processes.pools[0])
        self.assertEqual(self.processes.calls, [(3, ("numpy",))])
        self.assertEqual(self.threads.calls, [])

    def test_the_thread_maker_is_the_default_of_threads_and_only_of_threads(self) -> None:
        given = executor(ExecutorKind.THREADS, 5)
        pool = engine_for(given)._pool_for(given, 5, Silent())
        self.assertIs(pool, self.threads.pools[0])
        self.assertEqual(self.threads.calls, [(5, ("numpy",))])
        self.assertEqual(self.processes.calls, [])

    def test_the_workers_are_passed_as_they_are_and_by_position(self) -> None:
        for kind, maker in (
            (ExecutorKind.PROCESSES, self.processes),
            (ExecutorKind.THREADS, self.threads),
        ):
            for workers in (1, 2, 7):
                with self.subTest(kind=kind, workers=workers):
                    given = executor(kind, 64)
                    engine_for(given)._pool_for(given, workers, Silent())
                    count, modules = maker.calls[-1]
                    self.assertIs(type(count), int)
                    self.assertEqual(count, workers)
                    self.assertIsInstance(modules, tuple)

    def test_the_modules_are_numpy_then_those_the_reader_names(self) -> None:
        given = executor(ExecutorKind.PROCESSES, 2)
        engine_for(given)._pool_for(given, 2, Shipped())
        self.assertEqual(self.processes.calls, [(2, ("numpy", "first.module", "second.module"))])

    def test_a_reader_that_names_no_modules_ships_only_numpy(self) -> None:
        for reader in (Silent(), object(), None):
            with self.subTest(reader=reader):
                given = executor(ExecutorKind.PROCESSES, 2)
                engine_for(given)._pool_for(given, 2, reader)
                self.assertEqual(self.processes.calls[-1], (2, ("numpy",)))

    def test_the_modules_are_the_same_for_threads(self) -> None:
        given = executor(ExecutorKind.THREADS, 2)
        engine_for(given)._pool_for(given, 2, Shipped())
        self.assertEqual(self.threads.calls, [(2, ("numpy", "first.module", "second.module"))])

    def test_the_modules_are_a_tuple_of_strings(self) -> None:
        given = executor(ExecutorKind.PROCESSES, 2)
        engine_for(given)._pool_for(given, 2, Shipped())
        modules = self.processes.calls[0][1]
        self.assertIs(type(modules), tuple)
        self.assertTrue(all(type(name) is str for name in modules))

    def test_a_factory_of_the_executor_is_used_instead_of_either_maker(self) -> None:
        for kind in (ExecutorKind.PROCESSES, ExecutorKind.THREADS):
            with self.subTest(kind=kind):
                own = Maker("own")
                given = executor(kind, 4, own)
                pool = engine_for(given)._pool_for(given, 4, Shipped())
                self.assertIs(pool, own.pools[0])
                self.assertEqual(own.calls, [(4, ("numpy", "first.module", "second.module"))])
        self.assertEqual(self.processes.calls, [])
        self.assertEqual(self.threads.calls, [])

    def test_the_pool_is_made_once_and_kept_for_every_later_pass(self) -> None:
        for kind, maker in (
            (ExecutorKind.PROCESSES, self.processes),
            (ExecutorKind.THREADS, self.threads),
        ):
            with self.subTest(kind=kind):
                before = len(maker.calls)
                given = executor(kind, 2)
                engine = engine_for(given)
                first = engine._pool_for(given, 2, Silent())
                second = engine._pool_for(given, 2, Shipped())
                third = engine._pool_for(given, 1, Silent())
                self.assertIs(first, maker.pools[-1])
                self.assertIs(second, first)
                self.assertIs(third, first)
                self.assertEqual(len(maker.calls), before + 1)

    def test_a_factory_is_asked_once_for_the_whole_fit(self) -> None:
        own = Maker("own")
        given = executor(ExecutorKind.THREADS, 2, own)
        engine = engine_for(given)
        pools = [engine._pool_for(given, 2, Silent()) for _ in range(3)]
        self.assertEqual(len(own.calls), 1)
        self.assertEqual(pools, [own.pools[0]] * 3)

    def test_each_engine_has_its_own_pool(self) -> None:
        given = executor(ExecutorKind.THREADS, 2)
        first = engine_for(given)._pool_for(given, 2, Silent())
        second = engine_for(given)._pool_for(given, 2, Silent())
        self.assertIsNot(first, second)
        self.assertEqual(len(self.threads.calls), 2)

    def test_a_pool_that_is_in_use_is_returned_without_asking_for_another(self) -> None:
        given = executor(ExecutorKind.PROCESSES, 2)
        engine = engine_for(given)
        taken = Pool("taken")
        engine._pool = taken  # type: ignore[assignment]
        self.assertIs(engine._pool_for(given, 2, Shipped()), taken)
        self.assertEqual(self.processes.calls, [])
        self.assertEqual(self.threads.calls, [])

    def test_the_pool_is_shut_down_once_the_engine_is_left_and_made_anew_after(self) -> None:
        given = executor(ExecutorKind.PROCESSES, 2)
        with engine_for(given) as engine:
            pool = engine._pool_for(given, 2, Silent())
            self.assertEqual(pool.shutdowns, [])  # type: ignore[attr-defined]
        self.assertEqual(pool.shutdowns, [(True, True)])  # type: ignore[attr-defined]
        self.assertIsNone(engine._pool)
        again = engine._pool_for(given, 2, Silent())
        self.assertIsNot(again, pool)
        self.assertEqual(len(self.processes.calls), 2)

    def test_the_start_method_and_the_limit_are_the_documented_ones(self) -> None:
        self.assertEqual(_executors.START_METHOD, "spawn")
        self.assertEqual(_executors.MAX_WORKERS, 64)


if __name__ == "__main__":
    unittest.main()
