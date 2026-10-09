"""The session guard in ``tests/conftest.py`` releases a stuck buffer call."""

from __future__ import annotations

import threading
import time
import unittest
from collections.abc import Callable
from unittest import mock

from tests import conftest
from veridist.engine.delivery import BoundedChunkBuffer, BufferedChunk, ChunkEnvelope


def chunk(identifier: str, sequence: int) -> BufferedChunk:
    return BufferedChunk(
        envelope=ChunkEnvelope("guard-test", identifier, sequence, sequence, sequence + 1, 4),
        payload=None,
    )


def run_in_daemon(operation: Callable[[], object]) -> tuple[threading.Thread, list[BaseException]]:
    failures: list[BaseException] = []

    def invoke() -> None:
        try:
            operation()
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=invoke, name="guarded-buffer-call", daemon=True)
    worker.start()
    return worker, failures


class BufferWaitGuardTests(unittest.TestCase):
    def test_a_put_without_a_timeout_on_a_full_buffer_is_released_and_reported(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        buffer.put(chunk("first", 0), timeout=0.1)
        started = time.perf_counter()
        with mock.patch.object(conftest, "CALL_SLACK_SECONDS", 0.1):
            worker, failures = run_in_daemon(lambda: buffer.put(chunk("second", 1)))
            try:
                worker.join(10.0)
                self.assertFalse(worker.is_alive(), "the guard did not release the blocked put")
            finally:
                buffer.cancel()
        self.assertLess(time.perf_counter() - started, 8.0)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], AssertionError)
        self.assertIn("BoundedChunkBuffer.put", str(failures[0]))
        self.assertTrue(buffer.cancelled)

    def test_a_get_without_a_timeout_on_an_empty_buffer_is_released_and_reported(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        started = time.perf_counter()
        with mock.patch.object(conftest, "CALL_SLACK_SECONDS", 0.1):
            worker, failures = run_in_daemon(buffer.get)
            try:
                worker.join(10.0)
                self.assertFalse(worker.is_alive(), "the guard did not release the blocked get")
            finally:
                buffer.cancel()
        self.assertLess(time.perf_counter() - started, 8.0)
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], AssertionError)
        self.assertIn("BoundedChunkBuffer.get", str(failures[0]))

    def test_calls_that_finish_in_time_are_untouched_and_leave_no_registration(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        try:
            buffer.put(chunk("first", 0), timeout=0.1)
            received = buffer.get(timeout=0.1)
            self.assertEqual(received.envelope.chunk_id, "first")
            received.release()
            self.assertFalse(buffer.cancelled)
            with conftest._active_lock:
                leftovers = [call for call in conftest._active.values() if call.buffer is buffer]
            self.assertEqual(leftovers, [])
        finally:
            buffer.cancel()

    def test_a_failure_of_a_call_is_not_masked_when_the_guard_did_not_trip(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        released = chunk("released", 0)
        released.release()
        with self.assertRaisesRegex(RuntimeError, "cannot buffer an already released chunk"):
            buffer.put(released, timeout=0.1)
        buffer.cancel()
