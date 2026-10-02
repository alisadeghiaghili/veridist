"""Public stream-source and active-lease delivery contracts."""

from __future__ import annotations

import threading
import time
import unittest
from collections.abc import Callable, Iterable, Iterator
from typing import cast

from veridist.domain.lifetimes import ExactLifetime
from veridist.engine.data_source import DataSourceMetadata, Replayability
from veridist.engine.delivery import BoundedChunkBuffer, BufferedChunk, ChunkEnvelope
from veridist.engine.errors import FailureCode
from veridist.engine.streaming import IterableDataSource, StreamSourceError
from veridist.families import ExponentialFitSuccess, fit_exponential_chunks
from veridist.families.registry import FamilyId
from veridist.statistics.log_likelihood import LogLikelihoodSuccess, reduce_log_likelihood_chunks


def metadata(replayability: Replayability = Replayability.SINGLE_PASS) -> DataSourceMetadata:
    return DataSourceMetadata(
        source_id="stream-test",
        schema_version="1",
        provenance_schema_version="1",
        replayability=replayability,
        redaction_reason="test",
    )


def chunk(identifier: str, sequence: int, callback=None) -> BufferedChunk:
    return BufferedChunk(
        envelope=ChunkEnvelope("stream-test", identifier, sequence, sequence, sequence + 1, 4),
        payload=(sequence,),
        release_callback=callback,
    )


class StreamSourceContractTests(unittest.TestCase):
    def test_source_constructor_rejects_non_metadata_instance(self) -> None:
        with self.assertRaisesRegex(TypeError, "metadata must be DataSourceMetadata"):
            IterableDataSource(((0.0,),), cast(DataSourceMetadata, object()))

    def test_single_pass_source_rejects_iterator_factory(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "single-pass sources require an iterable, not an iterator factory",
        ):
            IterableDataSource(lambda: iter(((0.0,),)), metadata())

    def test_single_pass_source_rejects_non_iterable_chunks(self) -> None:
        with self.assertRaisesRegex(TypeError, "chunks must be an iterable"):
            IterableDataSource(cast(Iterable[tuple[float, ...]], object()), metadata())

    def test_replayable_source_rejects_factory_returning_non_iterator(self) -> None:
        def invalid_factory() -> Iterator[tuple[float, ...]]:
            return cast(Iterator[tuple[float, ...]], ((0.0,),))

        source = IterableDataSource(
            cast(Callable[[], Iterator[tuple[float, ...]]], invalid_factory),
            metadata(Replayability.REPLAYABLE),
        )
        with self.assertRaisesRegex(TypeError, "stream factory must return an iterator"):
            source.iter_chunks()

    def test_generic_source_is_consumed_by_current_streaming_reducers(self) -> None:
        likelihood_source = IterableDataSource(((0.0, 1.0),), metadata())
        likelihood = reduce_log_likelihood_chunks(
            FamilyId.NORMAL, likelihood_source, mu=0.0, sigma=1.0
        )
        self.assertIsInstance(likelihood, LogLikelihoodSuccess)
        self.assertEqual(likelihood.observation_count, 2)

        exponential_source = IterableDataSource(((ExactLifetime(2.0),),), metadata())
        exponential = fit_exponential_chunks(exponential_source)
        self.assertIsInstance(exponential, ExponentialFitSuccess)
        self.assertEqual(exponential.rate, 0.5)

    def test_single_pass_source_fails_with_typed_failure_on_second_acquisition(self) -> None:
        source = IterableDataSource(((0.0,),), metadata())
        self.assertEqual(tuple(source.iter_chunks()), ((0.0,),))
        with self.assertRaises(StreamSourceError) as caught:
            tuple(source.iter_chunks())
        self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_EXCEEDED)

    def test_concurrent_acquisition_admits_exactly_one_thread(self) -> None:
        threads_per_round = 8
        for _ in range(20):
            source = IterableDataSource(((0.0,),), metadata())
            barrier = threading.Barrier(threads_per_round, timeout=10.0)
            results: list[object] = []
            results_lock = threading.Lock()

            def acquire(
                source: IterableDataSource[tuple[float, ...]] = source,
                barrier: threading.Barrier = barrier,
                results: list[object] = results,
                results_lock: threading.Lock = results_lock,
            ) -> None:
                barrier.wait()
                try:
                    outcome: object = source.iter_chunks()
                except StreamSourceError as error:
                    outcome = error
                with results_lock:
                    results.append(outcome)

            workers = [threading.Thread(target=acquire) for _ in range(threads_per_round)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=10.0)
                self.assertFalse(worker.is_alive())

            failures = [item for item in results if isinstance(item, StreamSourceError)]
            winners = [item for item in results if not isinstance(item, StreamSourceError)]
            self.assertEqual(len(winners), 1)
            self.assertEqual(len(failures), threads_per_round - 1)
            for failure in failures:
                self.assertIs(failure.code, FailureCode.PASS_BUDGET_EXCEEDED)

    def test_single_pass_acquisition_is_serialized_by_the_source_lock(self) -> None:
        class SpyLock:
            def __init__(self) -> None:
                self.entered = 0
                self.held = False

            def __enter__(self) -> None:
                self.entered += 1
                self.held = True

            def __exit__(self, *exc_info: object) -> None:
                self.held = False

        source = IterableDataSource(((0.0,),), metadata())
        spy = SpyLock()
        source._lock = cast(threading.Lock, spy)
        source.iter_chunks()
        self.assertEqual(spy.entered, 1)
        self.assertFalse(spy.held)
        with self.assertRaises(StreamSourceError):
            source.iter_chunks()
        self.assertEqual(spy.entered, 2)
        self.assertFalse(spy.held)

    def test_replayable_source_requires_a_factory_and_can_be_acquired_twice(self) -> None:
        with self.assertRaises(ValueError):
            IterableDataSource(((0.0,),), metadata(Replayability.REPLAYABLE))
        source = IterableDataSource(lambda: iter(((0.0,),)), metadata(Replayability.REPLAYABLE))
        self.assertEqual(tuple(source.iter_chunks()), ((0.0,),))
        self.assertEqual(tuple(source.iter_chunks()), ((0.0,),))

    def test_checkpoint_replayable_source_is_rejected_until_checkpoint_semantics_exist(
        self,
    ) -> None:
        with self.assertRaises(StreamSourceError) as caught:
            IterableDataSource(
                lambda: iter(((0.0,),)),
                metadata(Replayability.CHECKPOINT_REPLAYABLE),
            )
        self.assertIs(caught.exception.code, FailureCode.CHECKPOINT_REQUIRED)
        self.assertEqual(
            caught.exception.context,
            {
                "replayability": "checkpoint_replayable",
                "operation": "iterable_data_source",
            },
        )

    def test_public_generic_source_workflow_uses_top_level_exports(self) -> None:
        import veridist

        source_metadata = veridist.DataSourceMetadata(
            source_id="public-stream",
            schema_version="1",
            provenance_schema_version="1",
            replayability=veridist.Replayability.SINGLE_PASS,
            redaction_reason="test",
        )
        source = veridist.IterableDataSource(((0.0, 1.0),), source_metadata)
        result = veridist.reduce_log_likelihood_chunks(
            veridist.FamilyId.NORMAL,
            source,
            mu=0.0,
            sigma=1.0,
        )
        self.assertIsInstance(result, LogLikelihoodSuccess)
        self.assertEqual(result.observation_count, 2)


class ActiveLeaseBufferTests(unittest.TestCase):
    def test_released_chunk_is_rejected_without_mutating_buffer_then_cancel_remains_safe(
        self,
    ) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        released = chunk("released", 0)
        released.release()

        with self.assertRaisesRegex(RuntimeError, "cannot buffer an already released chunk"):
            buffer.put(released)

        self.assertEqual(buffer.queued_chunks, 0)
        self.assertEqual(buffer.inflight_bytes, 0)
        self.assertEqual(buffer.peak_inflight_bytes, 0)
        buffer.cancel()
        self.assertTrue(buffer.cancelled)
        self.assertEqual(buffer.queued_chunks, 0)
        self.assertEqual(buffer.inflight_bytes, 0)

    def test_active_lease_remains_charged_until_release(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        buffer.put(chunk("first", 0))
        received = buffer.get(timeout=0.1)
        self.assertEqual(buffer.inflight_bytes, 4)
        self.assertEqual(buffer.queued_chunks, 0)
        received.release()
        self.assertEqual(buffer.inflight_bytes, 0)

    def test_producer_unblocks_only_after_active_lease_release(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        buffer.put(chunk("first", 0))
        received = buffer.get(timeout=0.1)
        completed = threading.Event()
        producer = threading.Thread(
            target=lambda: (buffer.put(chunk("second", 1)), completed.set())
        )
        producer.start()
        deadline = time.monotonic() + 1.0
        while buffer.waiting_producers == 0 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.assertEqual(buffer.waiting_producers, 1)
        self.assertFalse(completed.is_set())
        received.release()
        producer.join(1.0)
        self.assertTrue(completed.is_set())
        buffer.get(timeout=0.1).release()
        self.assertEqual(buffer.inflight_bytes, 0)

    def test_cancel_releases_every_queued_item_then_reraises_first_callback_error(self) -> None:
        calls: list[str] = []

        def bad() -> None:
            calls.append("bad")
            raise RuntimeError("release failed")

        def good() -> None:
            calls.append("good")

        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=8)
        first = chunk("first", 0, bad)
        second = chunk("second", 1, good)
        buffer.put(first)
        buffer.put(second)
        with self.assertRaisesRegex(RuntimeError, "release failed"):
            buffer.cancel()
        self.assertEqual(calls, ["bad", "good"])
        self.assertTrue(first.released)
        self.assertTrue(second.released)
        self.assertTrue(buffer.cancelled)
        self.assertEqual(buffer.inflight_bytes, 0)
