"""DS-04 through DS-06 contracts for delivery metadata and bounded buffering."""

from __future__ import annotations

import threading
import time
import unittest
from subprocess import TimeoutExpired, run
from sys import executable, getsizeof
from textwrap import dedent

from tests.contract.buffer_watchdog import bounded_buffer_call
from veridist.engine.data_source import Replayability
from veridist.engine.delivery import (
    AdapterCapabilities,
    AdapterKind,
    BoundedChunkBuffer,
    BufferedChunk,
    ChunkEnvelope,
    DeliveryContractError,
    DeliveryValidator,
    OrderingGuarantee,
)


def chunk(
    chunk_id: str,
    start: int,
    stop: int,
    *,
    sequence_number: int = 0,
    byte_size: int = 1,
) -> ChunkEnvelope:
    return ChunkEnvelope(
        source_id="dataset:delivery-001",
        chunk_id=chunk_id,
        sequence_number=sequence_number,
        row_start=start,
        row_stop=stop,
        byte_size=byte_size,
    )


def wait_for_waiting_producers(
    buffer: BoundedChunkBuffer,
    expected: int,
    timeout: float = 1.0,
) -> None:
    """Observe blocked producers without scheduling with ``sleep``."""

    deadline = time.monotonic() + timeout
    tick = threading.Event()
    while time.monotonic() < deadline:
        if buffer.waiting_producers == expected:
            return
        tick.wait(0.001)
    raise AssertionError(f"expected {expected} blocked producers")


def buffered(envelope: ChunkEnvelope) -> BufferedChunk:
    return BufferedChunk(envelope=envelope, payload=object())


class AdapterAndIdentityContractTests(unittest.TestCase):
    """DS-04: declarations and source-offset row identities are stable."""

    def test_ds04_seven_adapter_kinds_declare_capabilities_without_imports(self) -> None:
        replayability = {
            AdapterKind.CSV: Replayability.REPLAYABLE,
            AdapterKind.PARQUET: Replayability.REPLAYABLE,
            AdapterKind.ARROW: Replayability.REPLAYABLE,
            AdapterKind.PANDAS: Replayability.REPLAYABLE,
            AdapterKind.POLARS: Replayability.REPLAYABLE,
            AdapterKind.DASK: Replayability.REPLAYABLE,
            AdapterKind.DATABASE: Replayability.CHECKPOINT_REPLAYABLE,
        }

        declarations = tuple(
            AdapterCapabilities(
                kind=kind,
                replayability=mode,
                ordering=OrderingGuarantee.STABLE_ROW_OFFSETS,
                stable_offsets=True,
            )
            for kind, mode in replayability.items()
        )

        self.assertEqual({item.kind for item in declarations}, set(AdapterKind))
        self.assertTrue(all(item.stable_offsets for item in declarations))
        self.assertFalse(hasattr(declarations[0], "__dict__"))
        with self.assertRaises((AttributeError, TypeError)):
            declarations[0].stable_offsets = False  # type: ignore[misc]

    def test_ds04_row_identity_survives_boundary_changes(self) -> None:
        first_partition = (chunk("a", 0, 2), chunk("b", 2, 5))
        second_partition = (chunk("x", 0, 1), chunk("y", 1, 4), chunk("z", 4, 5))

        first_identities = [
            envelope.row_identity(index)
            for envelope in first_partition
            for index in range(envelope.row_count)
        ]
        second_identities = [
            envelope.row_identity(index)
            for envelope in second_partition
            for index in range(envelope.row_count)
        ]

        self.assertEqual(first_identities, second_identities)
        self.assertEqual(first_identities[3], ("dataset:delivery-001", 3))

    def test_ds04_envelope_rejects_invalid_identity_offsets_and_sizes(self) -> None:
        invalid = (
            {
                "source_id": "",
                "chunk_id": "x",
                "sequence_number": 0,
                "row_start": 0,
                "row_stop": 1,
                "byte_size": 1,
            },
            {
                "source_id": "dataset:delivery-001",
                "chunk_id": "",
                "sequence_number": 0,
                "row_start": 0,
                "row_stop": 1,
                "byte_size": 1,
            },
            {
                "source_id": "dataset:delivery-001",
                "chunk_id": "x",
                "sequence_number": 0,
                "row_start": -1,
                "row_stop": 1,
                "byte_size": 1,
            },
            {
                "source_id": "dataset:delivery-001",
                "chunk_id": "x",
                "sequence_number": 0,
                "row_start": 2,
                "row_stop": 1,
                "byte_size": 1,
            },
            {
                "source_id": "dataset:delivery-001",
                "chunk_id": "x",
                "sequence_number": 0,
                "row_start": 0,
                "row_stop": 1,
                "byte_size": -1,
            },
            {
                "source_id": "dataset:delivery-001",
                "chunk_id": "x",
                "sequence_number": -1,
                "row_start": 0,
                "row_stop": 1,
                "byte_size": 1,
            },
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                ChunkEnvelope(**arguments)

        envelope = chunk("valid", 3, 5)
        with self.assertRaises(IndexError):
            envelope.row_identity(-1)
        with self.assertRaises(IndexError):
            envelope.row_identity(2)

    def test_ds04_envelope_rejects_non_string_ids_with_typeerror_not_attributeerror(
        self,
    ) -> None:
        for field_name in ("source_id", "chunk_id"):
            arguments: dict[str, object] = {
                "source_id": "dataset:delivery-001",
                "chunk_id": "x",
                "sequence_number": 0,
                "row_start": 0,
                "row_stop": 1,
                "byte_size": 1,
            }
            arguments[field_name] = 1
            with self.subTest(field_name=field_name), self.assertRaises(TypeError):
                ChunkEnvelope(**arguments)  # type: ignore[arg-type]


class DeliveryValidationContractTests(unittest.TestCase):
    """DS-05: invalid delivery modes are distinct and never mutate state."""

    def test_ds05_contiguous_ragged_and_empty_chunks_are_valid(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")

        for envelope in (
            chunk("empty-start", 0, 0, sequence_number=0, byte_size=0),
            chunk("first", 0, 1, sequence_number=1),
            chunk("empty-middle", 1, 1, sequence_number=2, byte_size=0),
            chunk("ragged", 1, 4, sequence_number=3),
        ):
            validator.accept(envelope)

        self.assertEqual(validator.next_offset, 4)
        self.assertEqual(validator.next_sequence, 4)
        self.assertEqual(validator.accepted_rows, 4)
        self.assertEqual(validator.accepted_chunks, 4)

    def test_ds05_validator_rejects_a_non_string_source_id_with_typeerror(self) -> None:
        with self.assertRaises(TypeError):
            DeliveryValidator(1)  # type: ignore[arg-type]

    def test_ds05_validator_and_finish_reject_invalid_bounds(self) -> None:
        with self.assertRaises(ValueError):
            DeliveryValidator("")
        with self.assertRaises(ValueError):
            DeliveryValidator("dataset:delivery-001", initial_offset=-1)
        with self.assertRaises(ValueError):
            DeliveryValidator("dataset:delivery-001", initial_sequence=-1)

        validator = DeliveryValidator("dataset:delivery-001", initial_offset=2)
        with self.assertRaises(ValueError):
            validator.finish(expected_row_stop=-1)
        with self.assertRaises(ValueError):
            validator.finish(expected_row_stop=2, expected_chunk_count=-1)
        with self.assertRaises(DeliveryContractError) as caught:
            validator.finish(expected_row_stop=1)
        self.assertEqual(caught.exception.code, "OUT_OF_ORDER_CHUNK")

        resumed = DeliveryValidator("dataset:delivery-001", initial_sequence=2)
        with self.assertRaises(DeliveryContractError) as excess_chunks:
            resumed.finish(expected_row_stop=0, expected_chunk_count=1)
        self.assertEqual(excess_chunks.exception.code, "OUT_OF_ORDER_CHUNK")

    def test_ds05_duplicate_chunk_has_stable_code_and_no_double_count(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        accepted = chunk("same", 0, 2, sequence_number=0)
        validator.accept(accepted)

        with self.assertRaises(DeliveryContractError) as caught:
            validator.accept(accepted)

        self.assertEqual(caught.exception.code, "DUPLICATE_CHUNK")
        self.assertEqual(validator.next_offset, 2)
        self.assertEqual(validator.accepted_rows, 2)
        self.assertEqual(validator.accepted_chunks, 1)

    def test_ds05_missing_and_out_of_order_codes_are_distinct_and_recoverable(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        validator.accept(chunk("first", 0, 2, sequence_number=0))

        with self.assertRaises(DeliveryContractError) as missing:
            validator.accept(chunk("gap", 4, 5, sequence_number=1))
        self.assertEqual(missing.exception.code, "MISSING_CHUNK")
        self.assertEqual(missing.exception.context["expected_offset"], 2)

        with self.assertRaises(DeliveryContractError) as out_of_order:
            validator.accept(chunk("late", 1, 2, sequence_number=1))
        self.assertEqual(out_of_order.exception.code, "OUT_OF_ORDER_CHUNK")
        self.assertEqual(out_of_order.exception.context["expected_offset"], 2)

        with self.assertRaises(DeliveryContractError) as skipped_sequence:
            validator.accept(chunk("skipped", 2, 4, sequence_number=2))
        self.assertEqual(skipped_sequence.exception.code, "OUT_OF_ORDER_CHUNK")

        validator.accept(chunk("recovery", 2, 4, sequence_number=1))
        self.assertEqual(validator.next_offset, 4)
        self.assertEqual(validator.next_sequence, 2)
        self.assertEqual(validator.accepted_rows, 4)
        self.assertEqual(validator.accepted_chunks, 2)

    def test_ds05_finish_detects_tail_loss_with_typed_missing_code(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        validator.accept(chunk("first", 0, 2, sequence_number=0))

        with self.assertRaises(DeliveryContractError) as missing:
            validator.finish(expected_row_stop=4)

        self.assertEqual(missing.exception.code, "MISSING_CHUNK")
        self.assertEqual(missing.exception.context["expected_offset"], 2)
        self.assertEqual(missing.exception.context["expected_row_stop"], 4)
        validator.accept(chunk("tail", 2, 4, sequence_number=1))
        validator.finish(expected_row_stop=4, expected_chunk_count=2)

    def test_ds05_finish_detects_missing_empty_tail_by_chunk_count(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        validator.accept(chunk("data", 0, 2, sequence_number=0))

        with self.assertRaises(DeliveryContractError) as missing:
            validator.finish(expected_row_stop=2, expected_chunk_count=2)

        self.assertEqual(missing.exception.code, "MISSING_CHUNK")
        self.assertEqual(missing.exception.context["expected_sequence"], 1)
        self.assertEqual(missing.exception.context["expected_chunk_count"], 2)

    def test_ds05_validator_state_is_constant_size_in_chunk_count(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        initial_state_bytes = getsizeof(validator) + sum(
            getsizeof(getattr(validator, slot)) for slot in validator.__slots__
        )

        for sequence_number in range(2_048):
            validator.accept(
                chunk(
                    f"empty-{sequence_number}",
                    0,
                    0,
                    sequence_number=sequence_number,
                    byte_size=0,
                )
            )

        self.assertEqual(validator.next_sequence, 2_048)
        self.assertEqual(validator.accepted_chunks, 2_048)
        final_state_bytes = getsizeof(validator) + sum(
            getsizeof(getattr(validator, slot)) for slot in validator.__slots__
        )
        self.assertEqual(final_state_bytes, initial_state_bytes)

    def test_ds05_source_mismatch_is_typed_and_preflight_only(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        other_source = ChunkEnvelope("dataset:other", "other", 0, 0, 1, 1)

        with self.assertRaises(DeliveryContractError) as caught:
            validator.accept(other_source)

        self.assertEqual(caught.exception.code, "SOURCE_MISMATCH")
        self.assertEqual(validator.next_offset, 0)
        self.assertEqual(validator.accepted_chunks, 0)

    def test_ds05_rejected_accepts_preserve_every_counter_and_context(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        validator.accept(chunk("accepted", 0, 2, sequence_number=0))
        before = (
            validator.next_offset,
            validator.next_sequence,
            validator.accepted_rows,
            validator.accepted_chunks,
        )
        cases = (
            (
                "duplicate",
                chunk("duplicate", 2, 3, sequence_number=0),
                "DUPLICATE_CHUNK",
                {
                    "expected_offset": 2,
                    "expected_sequence": 1,
                    "sequence_number": 0,
                    "row_start": 2,
                    "row_stop": 3,
                },
            ),
            (
                "sequence-gap",
                chunk("sequence-gap", 2, 3, sequence_number=2),
                "OUT_OF_ORDER_CHUNK",
                {
                    "expected_offset": 2,
                    "expected_sequence": 1,
                    "sequence_number": 2,
                    "row_start": 2,
                    "row_stop": 3,
                },
            ),
            (
                "row-gap",
                chunk("row-gap", 3, 4, sequence_number=1),
                "MISSING_CHUNK",
                {
                    "expected_offset": 2,
                    "expected_sequence": 1,
                    "sequence_number": 1,
                    "row_start": 3,
                    "row_stop": 4,
                },
            ),
            (
                "overlap",
                chunk("overlap", 1, 3, sequence_number=1),
                "OUT_OF_ORDER_CHUNK",
                {
                    "expected_offset": 2,
                    "expected_sequence": 1,
                    "sequence_number": 1,
                    "row_start": 1,
                    "row_stop": 3,
                },
            ),
            (
                "source",
                ChunkEnvelope("other-source", "foreign", 1, 2, 3, 1),
                "SOURCE_MISMATCH",
                {
                    "stage": "delivery_validation",
                    "expected_offset": 2,
                    "expected_sequence": 1,
                    "sequence_number": 1,
                    "row_start": 2,
                    "row_stop": 3,
                },
            ),
        )
        for name, envelope, code, context in cases:
            with self.subTest(name=name):
                with self.assertRaises(DeliveryContractError) as caught:
                    validator.accept(envelope)
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.context, context)
                self.assertEqual(
                    (
                        validator.next_offset,
                        validator.next_sequence,
                        validator.accepted_rows,
                        validator.accepted_chunks,
                    ),
                    before,
                )
        validator.accept(chunk("recovery", 2, 3, sequence_number=1))
        self.assertEqual(
            (
                validator.next_offset,
                validator.next_sequence,
                validator.accepted_rows,
                validator.accepted_chunks,
            ),
            (3, 2, 3, 2),
        )

    def test_ds05_finish_mismatch_matrix_has_exact_terminal_contexts(self) -> None:
        validator = DeliveryValidator("dataset:delivery-001")
        validator.accept(chunk("first", 0, 2, sequence_number=0))
        cases = (
            (4, None, "MISSING_CHUNK", {"expected_offset": 2, "expected_row_stop": 4}),
            (1, None, "OUT_OF_ORDER_CHUNK", {"expected_offset": 1, "row_stop": 2}),
            (
                2,
                2,
                "MISSING_CHUNK",
                {"expected_sequence": 1, "expected_chunk_count": 2},
            ),
            (
                2,
                0,
                "OUT_OF_ORDER_CHUNK",
                {"expected_sequence": 0, "sequence_number": 1},
            ),
        )
        for row_stop, chunks, code, context in cases:
            with self.subTest(row_stop=row_stop, chunks=chunks):
                with self.assertRaises(DeliveryContractError) as caught:
                    validator.finish(
                        expected_row_stop=row_stop, expected_chunk_count=chunks
                    )
                self.assertEqual(caught.exception.code, code)
                self.assertEqual(caught.exception.context, context)
        with self.assertRaises(ValueError):
            validator.finish(expected_row_stop=-1)
        with self.assertRaises(ValueError):
            validator.finish(expected_row_stop=2, expected_chunk_count=-1)


class BoundedBufferContractTests(unittest.TestCase):
    """DS-06: byte budgets, real backpressure and cancellation are observable."""

    def test_ds06_constructor_and_chunk_limits_are_hard_failures(self) -> None:
        for arguments in (
            {"chunk_bytes": 0, "max_inflight_bytes": 1},
            {"chunk_bytes": 2, "max_inflight_bytes": 1},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                BoundedChunkBuffer(**arguments)

        buffer = BoundedChunkBuffer(chunk_bytes=3, max_inflight_bytes=6)
        with self.assertRaises(DeliveryContractError) as unaccounted:
            bounded_buffer_call(
                buffer,
                lambda: buffer.put(buffered(chunk("unaccounted", 0, 0, byte_size=0)), timeout=0.1),
            )
        self.assertEqual(unaccounted.exception.code, "INVALID_RETAINED_BYTES")

        with self.assertRaises(DeliveryContractError) as caught:
            bounded_buffer_call(
                buffer,
                lambda: buffer.put(buffered(chunk("oversized", 0, 1, byte_size=4)), timeout=0.1),
            )
        self.assertEqual(caught.exception.code, "CHUNK_TOO_LARGE")
        self.assertEqual(buffer.inflight_bytes, 0)
        self.assertEqual(buffer.queued_chunks, 0)

    def test_ds06_release_underflow_is_rejected_before_mutating_inflight_bytes(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        bounded_buffer_call(
            buffer,
            lambda: buffer.put(buffered(chunk("only", 0, 1, byte_size=4))),
        )
        self.assertEqual(buffer.inflight_bytes, 4)

        with self.assertRaises(RuntimeError):
            buffer._release(5)

        # The check runs before the subtraction, so a rejected release leaves
        # the tally exactly as it was instead of first going negative.
        self.assertEqual(buffer.inflight_bytes, 4)

    def test_ds06_fresh_buffer_has_an_exact_empty_runtime_snapshot(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=8)

        self.assertEqual(buffer.inflight_bytes, 0)
        self.assertEqual(buffer.queued_chunks, 0)
        self.assertEqual(buffer.waiting_producers, 0)
        self.assertEqual(buffer.peak_inflight_bytes, 0)
        self.assertFalse(buffer.cancelled)
        observation = buffer.observation
        self.assertEqual(observation.chunk_bytes, 4)
        self.assertEqual(observation.max_inflight_bytes, 8)
        self.assertEqual(observation.peak_inflight_bytes, 0)
        self.assertEqual(observation.largest_retained_chunk_bytes, 0)
        self.assertEqual(observation.backpressure_event_count, 0)

    def test_ds06_buffer_accepts_chunks_that_exactly_fill_its_byte_budget(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=8)
        first = buffered(chunk("first", 0, 1, byte_size=4))
        second = buffered(chunk("second", 1, 2, byte_size=4))

        buffer.put(first, timeout=0.01)
        buffer.put(second, timeout=0.01)

        self.assertEqual(buffer.inflight_bytes, 8)
        self.assertEqual(buffer.queued_chunks, 2)

    def test_ds06_nonempty_queue_returns_oldest_chunk_before_timeout(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        first = buffered(chunk("first", 0, 1, byte_size=4))
        buffer.put(first, timeout=0.01)

        self.assertIs(buffer.get(timeout=0.01), first)

    def test_ds06_producer_blocks_until_get_releases_budget(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        first = buffered(chunk("first", 0, 1, byte_size=4))
        second = buffered(chunk("second", 1, 2, byte_size=4))
        bounded_buffer_call(buffer, lambda: buffer.put(first, timeout=0.1))
        outcome: list[str] = []

        def produce() -> None:
            buffer.put(second, timeout=1.0)
            outcome.append("accepted")

        producer = threading.Thread(target=produce, name="blocked-producer", daemon=True)
        try:
            producer.start()
            wait_for_waiting_producers(buffer, 1)
            self.assertTrue(producer.is_alive())
            self.assertEqual(buffer.inflight_bytes, 4)
            self.assertEqual(buffer.peak_inflight_bytes, 4)
            self.assertLessEqual(buffer.peak_inflight_bytes, buffer.max_inflight_bytes)

            self.assertEqual(bounded_buffer_call(buffer, lambda: buffer.get(timeout=0.2)), first)
            first.release()
            producer.join(timeout=0.5)

            self.assertFalse(producer.is_alive())
            self.assertEqual(outcome, ["accepted"])
            self.assertEqual(buffer.inflight_bytes, 4)
            self.assertEqual(bounded_buffer_call(buffer, lambda: buffer.get(timeout=0.2)), second)
            second.release()
            self.assertEqual(buffer.inflight_bytes, 0)
        finally:
            buffer.cancel()
            producer.join(timeout=0.05)

    def test_ds06_cancel_releases_queued_resources_exactly_once(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        released: list[str] = []
        first = BufferedChunk(
            envelope=chunk("first", 0, 1, byte_size=4),
            payload=object(),
            release_callback=lambda: released.append("first"),
        )
        bounded_buffer_call(buffer, lambda: buffer.put(first, timeout=0.1))

        buffer.cancel()
        buffer.cancel()
        first.release()

        self.assertEqual(released, ["first"])
        self.assertTrue(first.released)
        self.assertEqual(buffer.inflight_bytes, 0)
        self.assertEqual(buffer.queued_chunks, 0)

    def test_ds06_full_buffer_put_timeout_is_bounded(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        first = buffered(chunk("first", 0, 1, byte_size=4))
        bounded_buffer_call(buffer, lambda: buffer.put(first, timeout=0.1))

        with self.assertRaises(DeliveryContractError) as caught:
            bounded_buffer_call(
                buffer,
                lambda: buffer.put(buffered(chunk("timeout", 1, 2, byte_size=4)), timeout=0.01),
            )
        self.assertEqual(caught.exception.code, "BUFFER_TIMEOUT")
        self.assertEqual(caught.exception.context, {"operation": "put"})

        self.assertEqual(buffer.inflight_bytes, 4)
        self.assertEqual(buffer.queued_chunks, 1)
        self.assertEqual(bounded_buffer_call(buffer, lambda: buffer.get(timeout=0.1)), first)

    def test_ds06_cancel_wakes_waiter_and_stops_future_put_get(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        bounded_buffer_call(
            buffer,
            lambda: buffer.put(buffered(chunk("first", 0, 1, byte_size=4)), timeout=0.1),
        )
        outcome: list[str] = []

        def produce() -> None:
            try:
                buffer.put(buffered(chunk("blocked", 1, 2, byte_size=4)), timeout=1.0)
            except DeliveryContractError as error:
                outcome.append(error.code)

        producer = threading.Thread(target=produce, name="cancelled-producer", daemon=True)
        try:
            producer.start()
            wait_for_waiting_producers(buffer, 1)
            buffer.cancel()
            producer.join(timeout=0.5)

            self.assertFalse(producer.is_alive())
            self.assertEqual(outcome, ["CANCELLED"])
            self.assertTrue(buffer.cancelled)
            self.assertEqual(buffer.inflight_bytes, 0)
            self.assertEqual(buffer.queued_chunks, 0)
            with self.assertRaises(DeliveryContractError) as put_cancelled:
                bounded_buffer_call(
                    buffer, lambda: buffer.put(buffered(chunk("future", 2, 3)), timeout=0.1)
                )
            with self.assertRaises(DeliveryContractError) as get_cancelled:
                bounded_buffer_call(buffer, lambda: buffer.get(timeout=0.1))
            self.assertEqual(put_cancelled.exception.code, "CANCELLED")
            self.assertEqual(get_cancelled.exception.code, "CANCELLED")
        finally:
            buffer.cancel()
            producer.join(timeout=0.05)

    def test_ds06_cancel_prevents_new_reads_and_releases_rejected_read(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        bounded_buffer_call(
            buffer,
            lambda: buffer.put(buffered(chunk("first", 0, 1, byte_size=4)), timeout=0.1),
        )
        read_count = 0
        released: list[str] = []
        outcome: list[str] = []

        def read_next() -> BufferedChunk:
            nonlocal read_count
            read_count += 1
            return BufferedChunk(
                envelope=chunk("read", 1, 2, byte_size=4),
                payload=object(),
                release_callback=lambda: released.append("read"),
            )

        def produce() -> None:
            try:
                buffer.read_and_put(read_next, timeout=1.0)
            except DeliveryContractError as error:
                outcome.append(error.code)

        producer = threading.Thread(target=produce, name="instrumented-reader", daemon=True)
        try:
            producer.start()
            wait_for_waiting_producers(buffer, 1)
            self.assertEqual(read_count, 1)
            buffer.cancel()
            producer.join(timeout=0.5)

            self.assertFalse(producer.is_alive())
            self.assertEqual(outcome, ["CANCELLED"])
            self.assertEqual(released, ["read"])
            with self.assertRaises(DeliveryContractError) as caught:
                buffer.read_and_put(read_next, timeout=0.1)
            self.assertEqual(caught.exception.code, "CANCELLED")
            self.assertEqual(read_count, 1)
        finally:
            buffer.cancel()
            producer.join(timeout=0.05)

    def test_ds06_cancel_does_not_wait_for_external_read_callback(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        read_started = threading.Event()
        allow_read_return = threading.Event()
        cancel_returned = threading.Event()
        released: list[str] = []
        outcome: list[str] = []

        def blocking_read() -> BufferedChunk:
            read_started.set()
            if not allow_read_return.wait(1.0):
                raise AssertionError("test did not release the read callback")
            return BufferedChunk(
                envelope=chunk("read-after-cancel", 0, 1, byte_size=4),
                payload=object(),
                release_callback=lambda: released.append("read-after-cancel"),
            )

        def produce() -> None:
            try:
                buffer.read_and_put(blocking_read, timeout=1.0)
            except DeliveryContractError as error:
                outcome.append(error.code)

        def cancel() -> None:
            buffer.cancel()
            cancel_returned.set()

        producer = threading.Thread(target=produce, name="blocked-external-reader", daemon=True)
        canceller = threading.Thread(target=cancel, name="nonblocking-canceller", daemon=True)
        try:
            producer.start()
            self.assertTrue(read_started.wait(0.5))
            canceller.start()
            returned_while_read_blocked = cancel_returned.wait(0.2)
            allow_read_return.set()
            canceller.join(timeout=0.5)
            producer.join(timeout=0.5)

            self.assertTrue(returned_while_read_blocked)
            self.assertFalse(canceller.is_alive())
            self.assertFalse(producer.is_alive())
            self.assertEqual(outcome, ["CANCELLED"])
            self.assertEqual(released, ["read-after-cancel"])
        finally:
            allow_read_return.set()
            buffer.cancel()
            canceller.join(timeout=0.05)
            producer.join(timeout=0.05)

    def test_ds06_empty_queue_timeout_is_bounded(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        with self.assertRaises(DeliveryContractError) as caught:
            bounded_buffer_call(buffer, lambda: buffer.get(timeout=0.01))
        self.assertEqual(caught.exception.code, "BUFFER_TIMEOUT")
        self.assertEqual(caught.exception.context, {"operation": "get"})

    def test_ds06_capacity_release_admits_one_waiter_then_cancel_cleans_up(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        first = buffered(chunk("first", 0, 1, byte_size=4))
        bounded_buffer_call(buffer, lambda: buffer.put(first, timeout=0.1))
        start = (threading.Event(), threading.Event())
        attempted = (threading.Event(), threading.Event())
        admitted = (threading.Event(), threading.Event())
        finished = (threading.Event(), threading.Event())
        history: list[str] = []
        worker_errors: list[BaseException] = []
        history_lock = threading.Lock()

        def produce(index: int) -> None:
            try:
                if not start[index].wait(1.0):
                    raise AssertionError("test did not permit producer")
                attempted[index].set()
                buffer.put(
                    buffered(chunk(f"waiting-{index}", index + 1, index + 2, byte_size=4))
                )
            except DeliveryContractError as error:
                with history_lock:
                    history.append(f"{index}:{error.code}")
            except BaseException as error:
                worker_errors.append(error)
            else:
                with history_lock:
                    history.append(f"{index}:accepted")
                admitted[index].set()
            finally:
                finished[index].set()

        producers = tuple(
            threading.Thread(target=produce, args=(index,), daemon=True) for index in range(2)
        )
        try:
            for producer in producers:
                producer.start()
            start[0].set()
            self.assertTrue(attempted[0].wait(1.0))
            wait_for_waiting_producers(buffer, 1)
            start[1].set()
            self.assertTrue(attempted[1].wait(1.0))
            wait_for_waiting_producers(buffer, 2)
            self.assertEqual(bounded_buffer_call(buffer, lambda: buffer.get(timeout=0.1)), first)
            first.release()
            self.assertTrue(admitted[0].wait(0.5) or admitted[1].wait(0.5))
            buffer.cancel()
            self.assertTrue(finished[0].wait(1.0))
            self.assertTrue(finished[1].wait(1.0))
            for producer in producers:
                producer.join(timeout=1.0)
                self.assertFalse(producer.is_alive())
            self.assertEqual(worker_errors, [])
            self.assertEqual(len(history), 2)
            self.assertEqual(
                {entry.split(":", maxsplit=1)[1] for entry in history},
                {"accepted", "CANCELLED"},
            )
            self.assertEqual(buffer.inflight_bytes, 0)
            self.assertEqual(buffer.queued_chunks, 0)
            self.assertEqual(buffer.observation.backpressure_event_count, 2)
        finally:
            buffer.cancel()
            for producer in producers:
                producer.join(timeout=1.0)
                if producer.is_alive():
                    self.fail(f"producer {producer.name} did not stop after cancellation")

    def test_ds06_cancel_wakes_two_blocked_producers_without_resource_leak(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
        bounded_buffer_call(
            buffer,
            lambda: buffer.put(buffered(chunk("first", 0, 1, byte_size=4)), timeout=0.1),
        )
        ready = threading.Barrier(3)
        outcomes: list[str] = []
        worker_errors: list[BaseException] = []

        def produce(index: int) -> None:
            try:
                ready.wait(timeout=1.0)
                buffer.put(
                    buffered(chunk(f"blocked-{index}", index + 1, index + 2, byte_size=4))
                )
            except DeliveryContractError as error:
                outcomes.append(error.code)
            except BaseException as error:
                worker_errors.append(error)

        producers = tuple(
            threading.Thread(target=produce, args=(index,), daemon=True) for index in range(2)
        )
        try:
            for producer in producers:
                producer.start()
            ready.wait(timeout=1.0)
            wait_for_waiting_producers(buffer, 2)
            buffer.cancel()
            for producer in producers:
                producer.join(timeout=1.0)
                self.assertFalse(producer.is_alive())
            self.assertEqual(worker_errors, [])
            self.assertEqual(sorted(outcomes), ["CANCELLED", "CANCELLED"])
            self.assertTrue(buffer.cancelled)
            self.assertEqual(buffer.inflight_bytes, 0)
            self.assertEqual(buffer.queued_chunks, 0)
            self.assertEqual(buffer.observation.backpressure_event_count, 2)
        finally:
            buffer.cancel()
            for producer in producers:
                producer.join(timeout=1.0)
                if producer.is_alive():
                    self.fail(f"producer {producer.name} did not stop after cancellation")

    def test_ds06_watchdog_finishes_blocking_operations(self) -> None:
        """A child process makes deadlocks a deterministic test failure."""

        program = dedent(
            """
            import sys
            import threading
            from veridist.engine.delivery import (
                BoundedChunkBuffer,
                BufferedChunk,
                ChunkEnvelope,
                DeliveryContractError,
            )

            def item(name, sequence):
                return BufferedChunk(
                    envelope=ChunkEnvelope(
                        'source', name, sequence, sequence, sequence + 1, 4
                    ),
                    payload=None,
                )

            scenario = sys.argv[1]
            buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)
            wait_entered = threading.Event()
            done = threading.Event()
            result = []
            worker = None
            original_wait = buffer._condition.wait

            def require(condition, message):
                if not condition:
                    raise AssertionError(message)

            def observed_wait(timeout=None):
                wait_entered.set()
                return original_wait(timeout)

            buffer._condition.wait = observed_wait

            def blocked(operation):
                try:
                    if operation == 'put':
                        buffer.put(
                            item('second', 1),
                            timeout=0.01 if scenario == 'put-timeout' else None,
                        )
                    else:
                        result.append(
                            buffer.get(timeout=0.01 if scenario == 'get-timeout' else None)
                            .envelope.chunk_id
                        )
                except DeliveryContractError as error:
                    result.append(error.code)
                finally:
                    done.set()

            try:
                if scenario == 'init':
                    require(buffer.inflight_bytes == 0, 'initial bytes were retained')
                    require(buffer.queued_chunks == 0, 'initial queue was nonempty')
                elif scenario.startswith('get-'):
                    worker = threading.Thread(target=blocked, args=('get',), daemon=True)
                    worker.start()
                    require(wait_entered.wait(0.1), 'get did not enter Condition.wait')
                    if scenario == 'get-release':
                        buffer.put(item('first', 0), timeout=0.1)
                        expected = ['first']
                    elif scenario == 'get-cancel':
                        buffer.cancel()
                        expected = ['CANCELLED']
                    else:
                        expected = ['BUFFER_TIMEOUT']
                    require(done.wait(0.2), 'get worker did not finish')
                    worker.join(0.05)
                    require(not worker.is_alive(), 'get worker remained alive')
                    require(result == expected, f'unexpected get result: {result!r}')
                else:
                    buffer.put(item('first', 0), timeout=0.1)
                    worker = threading.Thread(target=blocked, args=('put',), daemon=True)
                    worker.start()
                    require(wait_entered.wait(0.1), 'put did not enter Condition.wait')
                    if scenario == 'put-release':
                        first = buffer.get(timeout=0.1)
                        require(
                            first.envelope.chunk_id == 'first',
                            'initial chunk was not returned',
                        )
                        first.release()
                        expected = []
                    elif scenario == 'put-cancel':
                        buffer.cancel()
                        expected = ['CANCELLED']
                    else:
                        expected = ['BUFFER_TIMEOUT']
                    require(done.wait(0.2), 'put worker did not finish')
                    worker.join(0.05)
                    require(not worker.is_alive(), 'put worker remained alive')
                    require(result == expected, f'unexpected put result: {result!r}')
                    if scenario == 'put-release':
                        require(
                            buffer.get(timeout=0.1).envelope.chunk_id == 'second',
                            'released put was not queued',
                        )
            finally:
                buffer.cancel()
                if worker is not None:
                    worker.join(0.05)
                    require(not worker.is_alive(), 'worker survived cancellation cleanup')
                del buffer._condition.wait
            print('watchdog-ok')
            """
        )
        # The subprocess budget covers interpreter start-up and the import as
        # well as the scenario itself, whose own waits are at most 0.2 s.
        # Start-up alone can approach a second on a loaded Windows host, so
        # measure it once and allow a fixed margin for the scenario on top:
        # a hang is still caught within seconds, independent of host speed.
        started = time.perf_counter()
        warmup = run(
            [executable, "-c", "import threading, veridist.engine.delivery"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(warmup.returncode, 0, warmup.stderr)
        scenario_budget = 2 * (time.perf_counter() - started) + 1.0
        for scenario in (
            "init",
            "get-release",
            "get-cancel",
            "get-timeout",
            "put-release",
            "put-cancel",
            "put-timeout",
        ):
            with self.subTest(scenario=scenario):
                try:
                    completed = run(
                        [executable, "-c", program, scenario],
                        capture_output=True,
                        text=True,
                        timeout=scenario_budget,
                        check=False,
                    )
                except TimeoutExpired as error:
                    self.fail(f"delivery watchdog timed out for {scenario}: {error}")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(completed.stdout.strip(), "watchdog-ok")


if __name__ == "__main__":
    unittest.main()
