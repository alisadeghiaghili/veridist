"""Session-wide containment for blocking ``BoundedChunkBuffer`` calls.

``BoundedChunkBuffer.put`` and ``get`` may wait on a condition variable, and
production callers such as the CSV fit path invoke them without a timeout from
a single thread.  A defect in the wait arithmetic, the loop condition or the
wake-ups therefore turns an otherwise failing test into one that never returns.
Under mutation testing that surfaces as a runner timeout rather than a killed
mutant, and under a deliberately broken change it simply hangs the suite.

The guard below makes that impossible for every test without touching the
production module: each ``put``/``get`` call is registered while it runs, and a
daemon watchdog cancels the buffer of any call that outlives its allowance (the
call's own ``timeout`` plus a fixed slack).  Cancelling wakes every waiter, so
the blocked call returns promptly and is reported as a test failure.  If even
cancellation cannot release the call the process is terminated, which a test
runner records as a failed run instead of an endless one.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from time import monotonic
from typing import Any

import pytest

import veridist
from tests import process_isolation
from veridist.engine.delivery import BoundedChunkBuffer

# The optional dataframe libraries start one worker per core.  The tests need no
# parallelism, and the idle workers cost CPU time (a lazy Polars scan over tiny
# batches uses more than twice as much CPU time as wall time on a 16-core host)
# and, in a process that mutation testing forks, are lost.  So every pool is
# pinned to one thread.  Polars reads these variables when it is first imported,
# which happens only when a test module is collected, after this file is loaded;
# a value set by the caller wins.
for _variable in ("POLARS_MAX_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

# A process forked by the mutation runner must not use those libraries after the
# fork; see ``tests/process_isolation.py``.
process_isolation.install()
process_isolation.check_expected_package(veridist.__file__, os.environ)

# A legitimate call finishes within its own timeout; the slack only has to
# absorb scheduler jitter on a heavily loaded host.
CALL_SLACK_SECONDS = 3.0
# Time allowed for ``cancel`` to release a blocked call before the process is
# terminated.
CANCEL_GRACE_SECONDS = 3.0
POLL_INTERVAL_SECONDS = 0.1


@dataclass
class _ActiveCall:
    buffer: BoundedChunkBuffer
    operation: str
    deadline: float
    cancelled_at: float | None = None


_active: dict[int, _ActiveCall] = {}
_active_lock = threading.Lock()
_tripped: set[int] = set()
_next_call_id = 0


def _register(buffer: BoundedChunkBuffer, operation: str, timeout: object) -> int:
    global _next_call_id
    allowance = CALL_SLACK_SECONDS
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0:
        allowance += float(timeout)
    with _active_lock:
        _next_call_id += 1
        call_id = _next_call_id
        _active[call_id] = _ActiveCall(buffer, operation, monotonic() + allowance)
    return call_id


def _finish(call_id: int) -> None:
    with _active_lock:
        _active.pop(call_id, None)
        _tripped.discard(call_id)


def _guarded(operation: str, original: Callable[..., Any]) -> Callable[..., Any]:
    def call(self: BoundedChunkBuffer, *args: Any, **kwargs: Any) -> Any:
        call_id = _register(self, operation, kwargs.get("timeout"))
        try:
            return original(self, *args, **kwargs)
        except BaseException as error:
            with _active_lock:
                blocked = call_id in _tripped
            if blocked:
                raise AssertionError(
                    f"BoundedChunkBuffer.{operation} stayed blocked beyond the test guard "
                    "allowance; the buffer was cancelled to release it"
                ) from error
            raise
        finally:
            _finish(call_id)

    call.__name__ = getattr(original, "__name__", operation)
    call.__doc__ = getattr(original, "__doc__", None)
    return call


def _watch(stop: threading.Event) -> None:
    while not stop.wait(POLL_INTERVAL_SECONDS):
        now = monotonic()
        overdue: list[tuple[int, _ActiveCall]] = []
        with _active_lock:
            for call_id, call in _active.items():
                if now >= call.deadline:
                    overdue.append((call_id, call))
        for call_id, call in overdue:
            if call.cancelled_at is None:
                with _active_lock:
                    _tripped.add(call_id)
                    call.cancelled_at = now
                try:
                    call.buffer.cancel()
                except BaseException:  # release callbacks of an abandoned buffer
                    pass
            elif now - call.cancelled_at >= CANCEL_GRACE_SECONDS:
                sys.stderr.write(
                    f"BoundedChunkBuffer.{call.operation} could not be released by "
                    "cancellation; terminating the test process\n"
                )
                faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
                sys.stderr.flush()
                os._exit(70)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    process_isolation.defer_library_tests(items)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> object | None:
    """Give a forked process a fresh interpreter before its first Polars or DuckDB test."""

    if process_isolation.must_restart(item.path, os.environ):
        session = item.session
        remaining = [entry.nodeid for entry in session.items[session.items.index(item) :]]
        arguments = process_isolation.fresh_process_arguments(
            item.config.invocation_params.args,
            item.config.known_args_namespace.file_or_dir,
            remaining,
        )
        environment = process_isolation.fresh_process_environment(os.environ, veridist.__file__)
        try:
            process_isolation.restart(arguments, environment)
        except OSError:
            pass  # carry on in this process, as before
    return None


@pytest.fixture(scope="session", autouse=True)
def single_threaded_arrow() -> None:
    """Pin the Arrow compute and I/O thread pools to one thread, if pyarrow is installed."""

    try:
        import pyarrow
    except ImportError:
        return
    pyarrow.set_cpu_count(1)
    pyarrow.set_io_thread_count(1)


@pytest.fixture(scope="session", autouse=True)
def bounded_buffer_wait_guard() -> Iterator[None]:
    originals = {name: getattr(BoundedChunkBuffer, name) for name in ("put", "get")}
    for name, original in originals.items():
        setattr(BoundedChunkBuffer, name, _guarded(name, original))
    stop = threading.Event()
    watchdog = threading.Thread(
        target=_watch, args=(stop,), name="bounded-buffer-test-guard", daemon=True
    )
    watchdog.start()
    try:
        yield
    finally:
        stop.set()
        watchdog.join(timeout=1.0)
        for name, original in originals.items():
            setattr(BoundedChunkBuffer, name, original)
