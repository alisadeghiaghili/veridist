"""Stdlib-only delivery contracts for chunk identity and bounded buffering."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from threading import Condition, RLock

from veridist.engine.data_source import Replayability
from veridist.engine.errors import EngineContractError, FailureCode


class AdapterKind(StrEnum):
    """Adapter categories whose declarations require no third-party import."""

    CSV = "csv"
    PARQUET = "parquet"
    ARROW = "arrow"
    PANDAS = "pandas"
    POLARS = "polars"
    DASK = "dask"
    DATABASE = "database"


class OrderingGuarantee(StrEnum):
    """Ordering semantics declared by an adapter capability record."""

    STABLE_ROW_OFFSETS = "stable_row_offsets"


@dataclass(frozen=True, slots=True)
class AdapterCapabilities:
    """Dependency-free declaration, not evidence that an adapter exists."""

    kind: AdapterKind
    replayability: Replayability
    ordering: OrderingGuarantee
    stable_offsets: bool


@dataclass(frozen=True, slots=True)
class ChunkEnvelope:
    """Stable sequence/offset identity and retained-byte accounting for one chunk.

    ``byte_size`` is the total memory retained while the chunk is buffered, not
    merely the serialized or payload byte length. Validation-only empty chunks
    may report zero; a :class:`BoundedChunkBuffer` rejects such leases because
    they cannot participate safely in a byte-only bound.
    """

    source_id: str
    chunk_id: str
    sequence_number: int
    row_start: int
    row_stop: int
    byte_size: int

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str):
            raise TypeError("source_id must be a string")
        if not isinstance(self.chunk_id, str):
            raise TypeError("chunk_id must be a string")
        if not self.source_id.strip():
            raise ValueError("source_id must be non-empty")
        if not self.chunk_id.strip():
            raise ValueError("chunk_id must be non-empty")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")
        if self.row_start < 0:
            raise ValueError("row_start must be non-negative")
        if self.row_stop < self.row_start:
            raise ValueError("row_stop must not precede row_start")
        if self.byte_size < 0:
            raise ValueError("byte_size must be non-negative")

    @property
    def row_count(self) -> int:
        return self.row_stop - self.row_start

    def row_identity(self, local_index: int) -> tuple[str, int]:
        """Return partition-independent identity for a row in this envelope."""

        if local_index < 0 or local_index >= self.row_count:
            raise IndexError("local row index is outside the chunk")
        return self.source_id, self.row_start + local_index


class DeliveryContractError(EngineContractError):
    """Stable, localization-independent delivery contract failure."""


class DeliveryValidator:
    """Validate contiguous non-overlapping delivery without silent repair."""

    __slots__ = (
        "_accepted_chunks",
        "_accepted_rows",
        "_next_offset",
        "_next_sequence",
        "_source_id",
    )

    def __init__(
        self,
        source_id: str,
        *,
        initial_offset: int = 0,
        initial_sequence: int = 0,
    ) -> None:
        if not isinstance(source_id, str):
            raise TypeError("source_id must be a string")
        if not source_id.strip():
            raise ValueError("source_id must be non-empty")
        if initial_offset < 0:
            raise ValueError("initial_offset must be non-negative")
        if initial_sequence < 0:
            raise ValueError("initial_sequence must be non-negative")
        self._source_id = source_id
        self._next_offset = initial_offset
        self._next_sequence = initial_sequence
        self._accepted_rows = 0
        self._accepted_chunks = 0

    @property
    def next_offset(self) -> int:
        return self._next_offset

    @property
    def next_sequence(self) -> int:
        return self._next_sequence

    @property
    def accepted_rows(self) -> int:
        return self._accepted_rows

    @property
    def accepted_chunks(self) -> int:
        return self._accepted_chunks

    def accept(self, envelope: ChunkEnvelope) -> None:
        """Accept exactly the next source range or fail without state mutation."""

        context = {
            "expected_offset": self._next_offset,
            "expected_sequence": self._next_sequence,
            "sequence_number": envelope.sequence_number,
            "row_start": envelope.row_start,
            "row_stop": envelope.row_stop,
        }
        if envelope.source_id != self._source_id:
            raise DeliveryContractError(
                FailureCode.SOURCE_MISMATCH,
                {"stage": "delivery_validation", **context},
            )
        if envelope.sequence_number < self._next_sequence:
            raise DeliveryContractError(FailureCode.DUPLICATE_CHUNK, context)
        if envelope.sequence_number > self._next_sequence:
            raise DeliveryContractError(FailureCode.OUT_OF_ORDER_CHUNK, context)
        if envelope.row_start > self._next_offset:
            raise DeliveryContractError(FailureCode.MISSING_CHUNK, context)
        if envelope.row_start < self._next_offset:
            raise DeliveryContractError(FailureCode.OUT_OF_ORDER_CHUNK, context)

        self._next_sequence += 1
        self._next_offset = envelope.row_stop
        self._accepted_rows += envelope.row_count
        self._accepted_chunks += 1

    def finish(
        self,
        *,
        expected_row_stop: int,
        expected_chunk_count: int | None = None,
    ) -> None:
        """Detect a missing terminal range that no subsequent chunk can reveal."""

        if expected_row_stop < 0:
            raise ValueError("expected_row_stop must be non-negative")
        if expected_chunk_count is not None and expected_chunk_count < 0:
            raise ValueError("expected_chunk_count must be non-negative")
        if expected_chunk_count is not None and self._next_sequence < expected_chunk_count:
            raise DeliveryContractError(
                FailureCode.MISSING_CHUNK,
                {
                    "expected_sequence": self._next_sequence,
                    "expected_chunk_count": expected_chunk_count,
                },
            )
        if expected_chunk_count is not None and self._next_sequence > expected_chunk_count:
            raise DeliveryContractError(
                FailureCode.OUT_OF_ORDER_CHUNK,
                {
                    "expected_sequence": expected_chunk_count,
                    "sequence_number": self._next_sequence,
                },
            )
        if self._next_offset < expected_row_stop:
            raise DeliveryContractError(
                FailureCode.MISSING_CHUNK,
                {
                    "expected_offset": self._next_offset,
                    "expected_row_stop": expected_row_stop,
                },
            )
        if self._next_offset > expected_row_stop:
            raise DeliveryContractError(
                FailureCode.OUT_OF_ORDER_CHUNK,
                {
                    "expected_offset": expected_row_stop,
                    "row_stop": self._next_offset,
                },
            )


class BufferedChunk:
    """A payload lease whose release callback is safe to call more than once."""

    __slots__ = ("_release_callback", "_release_lock", "_released", "envelope", "payload")

    def __init__(
        self,
        *,
        envelope: ChunkEnvelope,
        payload: object,
        release_callback: Callable[[], None] | None = None,
    ) -> None:
        self.envelope = envelope
        self.payload = payload
        self._release_callback = release_callback
        self._release_lock = RLock()
        self._released = False

    @property
    def released(self) -> bool:
        with self._release_lock:
            return self._released

    def release(self) -> None:
        callback: Callable[[], None] | None
        with self._release_lock:
            if self._released:
                return
            self._released = True
            callback = self._release_callback
        if callback is not None:
            callback()

    def _compose_release_callback(self, callback: Callable[[], None]) -> None:
        """Append buffer cleanup before any caller cleanup callback.

        This private operation runs only while the buffer owns the chunk. The
        accounting callback is first so a caller callback failure cannot strand
        capacity or block another producer.
        """

        with self._release_lock:
            if self._released:
                raise RuntimeError("cannot buffer an already released chunk")
            previous = self._release_callback

            def composed() -> None:
                callback()
                if previous is not None:
                    previous()

            self._release_callback = composed


@dataclass(frozen=True, slots=True)
class BufferObservation:
    """Historical byte-bound and backpressure facts owned by a chunk buffer."""

    chunk_bytes: int
    max_inflight_bytes: int
    peak_inflight_bytes: int
    largest_retained_chunk_bytes: int
    backpressure_event_count: int

    def __post_init__(self) -> None:
        values = (
            self.chunk_bytes,
            self.max_inflight_bytes,
            self.peak_inflight_bytes,
            self.largest_retained_chunk_bytes,
            self.backpressure_event_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("buffer observation counters must be integers")
        if self.chunk_bytes <= 0 or self.max_inflight_bytes <= 0:
            raise ValueError("buffer byte budgets must be positive")
        if self.max_inflight_bytes < self.chunk_bytes:
            raise ValueError("max_inflight_bytes must cover one allowed chunk")
        if min(
            self.peak_inflight_bytes,
            self.largest_retained_chunk_bytes,
            self.backpressure_event_count,
        ) < 0:
            raise ValueError("buffer observations must be non-negative")
        if self.peak_inflight_bytes > self.max_inflight_bytes:
            raise ValueError("peak inflight bytes exceeded the declared bound")
        if self.largest_retained_chunk_bytes > self.chunk_bytes:
            raise ValueError("largest retained chunk exceeded the declared bound")


class BoundedChunkBuffer:
    """Condition-backed FIFO with hard byte bounds and explicit cancellation."""

    def __init__(self, *, chunk_bytes: int, max_inflight_bytes: int) -> None:
        if (
            isinstance(chunk_bytes, bool)
            or not isinstance(chunk_bytes, int)
            or isinstance(max_inflight_bytes, bool)
            or not isinstance(max_inflight_bytes, int)
        ):
            raise TypeError("buffer byte budgets must be integers")
        if chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be positive")
        if max_inflight_bytes < chunk_bytes:
            raise ValueError("max_inflight_bytes must be at least chunk_bytes")
        self.chunk_bytes = chunk_bytes
        self.max_inflight_bytes = max_inflight_bytes
        self._condition = Condition()
        self._queue: deque[BufferedChunk] = deque()
        self._inflight_bytes = 0
        self._peak_inflight_bytes = 0
        self._largest_retained_chunk_bytes = 0
        self._backpressure_event_count = 0
        self._waiting_producers = 0
        self._cancelled = False

    @property
    def inflight_bytes(self) -> int:
        with self._condition:
            return self._inflight_bytes

    @property
    def peak_inflight_bytes(self) -> int:
        with self._condition:
            return self._peak_inflight_bytes

    @property
    def observation(self) -> BufferObservation:
        """Return an immutable snapshot whose peaks survive queue draining."""

        with self._condition:
            return BufferObservation(
                chunk_bytes=self.chunk_bytes,
                max_inflight_bytes=self.max_inflight_bytes,
                peak_inflight_bytes=self._peak_inflight_bytes,
                largest_retained_chunk_bytes=self._largest_retained_chunk_bytes,
                backpressure_event_count=self._backpressure_event_count,
            )

    @property
    def queued_chunks(self) -> int:
        with self._condition:
            return len(self._queue)

    @property
    def waiting_producers(self) -> int:
        with self._condition:
            return self._waiting_producers

    @property
    def cancelled(self) -> bool:
        with self._condition:
            return self._cancelled

    def _raise_if_cancelled(self) -> None:
        if self._cancelled:
            raise DeliveryContractError(FailureCode.CANCELLED, {})

    def put(self, item: BufferedChunk, *, timeout: float | None = None) -> None:
        """Queue an item, blocking while its bytes would exceed the hard bound.

        ``timeout=None`` waits without a deadline. In a single-threaded
        caller that never releases an earlier lease before calling ``put``
        again, nothing will ever free the capacity this call is waiting for,
        so it blocks forever; pass a finite ``timeout`` whenever that
        ordering is not guaranteed.
        """

        byte_size = item.envelope.byte_size
        if byte_size <= 0:
            raise DeliveryContractError(
                FailureCode.INVALID_RETAINED_BYTES,
                {"byte_size": byte_size},
            )
        if byte_size > self.chunk_bytes:
            raise DeliveryContractError(
                FailureCode.CHUNK_TOO_LARGE,
                {"byte_size": byte_size, "chunk_bytes": self.chunk_bytes},
            )
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            self._raise_if_cancelled()
            counted_backpressure = False
            while self._inflight_bytes + byte_size > self.max_inflight_bytes:
                if not counted_backpressure:
                    self._backpressure_event_count += 1
                    counted_backpressure = True
                self._waiting_producers += 1
                try:
                    remaining = None if deadline is None else deadline - time.monotonic()
                    if remaining is not None and remaining <= 0:
                        raise DeliveryContractError(
                            FailureCode.BUFFER_TIMEOUT,
                            {"operation": "put"},
                        )
                    self._condition.wait(remaining)
                finally:
                    self._waiting_producers -= 1
                self._raise_if_cancelled()
            with item._release_lock:
                if item._released:
                    raise RuntimeError("cannot buffer an already released chunk")
                item._compose_release_callback(lambda: self._release(item.envelope.byte_size))
                self._queue.append(item)
                self._inflight_bytes += byte_size
                self._peak_inflight_bytes = max(self._peak_inflight_bytes, self._inflight_bytes)
                self._largest_retained_chunk_bytes = max(
                    self._largest_retained_chunk_bytes,
                    byte_size,
                )
            self._condition.notify_all()

    def _release(self, byte_size: int) -> None:
        """Release a transferred queue lease exactly once through BufferedChunk."""

        with self._condition:
            if byte_size > self._inflight_bytes:
                raise RuntimeError("buffer inflight byte accounting underflow")
            self._inflight_bytes -= byte_size
            self._condition.notify_all()

    def read_and_put(
        self,
        read_next: Callable[[], BufferedChunk],
        *,
        timeout: float | None = None,
    ) -> None:
        """Read only while active, releasing a read item rejected by the buffer."""

        with self._condition:
            self._raise_if_cancelled()
        item = read_next()
        try:
            self.put(item, timeout=timeout)
        except BaseException:
            item.release()
            raise

    def get(self, *, timeout: float | None = None) -> BufferedChunk:
        """Return the oldest item while retaining its charge until release."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while not self._queue:
                self._raise_if_cancelled()
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise DeliveryContractError(
                        FailureCode.BUFFER_TIMEOUT,
                        {"operation": "get"},
                    )
                self._condition.wait(remaining)
            item = self._queue.popleft()
            self._condition.notify_all()
            return item

    def cancel(self) -> None:
        """Wake waiters and release every resource still owned by the queue."""

        with self._condition:
            if self._cancelled:
                return
            self._cancelled = True
            queued = tuple(self._queue)
            self._queue.clear()
            self._condition.notify_all()
        first_error: BaseException | None = None
        for item in queued:
            try:
                item.release()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


__all__ = [
    "AdapterCapabilities",
    "AdapterKind",
    "BoundedChunkBuffer",
    "BufferObservation",
    "BufferedChunk",
    "ChunkEnvelope",
    "DeliveryContractError",
    "DeliveryValidator",
    "OrderingGuarantee",
]
