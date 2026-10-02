"""Small public stream-source contracts shared by streaming operations."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from threading import Lock
from typing import Generic, Protocol, TypeVar, cast

from veridist.engine.data_source import DataSourceMetadata, Replayability
from veridist.engine.errors import EngineContractError, FailureCode

T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)


class StreamSourceError(EngineContractError):
    """Typed failure raised when a stream cannot honor its declared passes."""


class StreamSource(Protocol[T_co]):
    """A metadata-declared sequential stream of chunks."""

    @property
    def metadata(self) -> DataSourceMetadata:
        """Return immutable source identity and replayability facts."""

    def iter_chunks(self) -> Iterator[T_co]:
        """Acquire one sequential chunk iterator."""


class IterableDataSource(Generic[T]):
    """Adapt caller-owned chunks with explicit one-pass/replayability semantics.

    A single-pass source receives one iterable and may be acquired once. A
    replayable declaration requires a zero-argument iterator factory, so this
    library never infers replayability from a container.

    Acquisition is thread-safe: when several threads race to acquire a
    single-pass source, exactly one receives the iterator and the others get
    ``PASS_BUDGET_EXCEEDED``. The returned iterator itself is not synchronized.
    """

    __slots__ = ("_acquired", "_factory", "_iterable", "_lock", "metadata")

    def __init__(
        self,
        chunks: Iterable[T] | Callable[[], Iterator[T]],
        metadata: DataSourceMetadata,
    ) -> None:
        if type(metadata) is not DataSourceMetadata:
            raise TypeError("metadata must be DataSourceMetadata")
        if metadata.replayability is Replayability.CHECKPOINT_REPLAYABLE:
            raise StreamSourceError(
                FailureCode.CHECKPOINT_REQUIRED,
                {
                    "replayability": metadata.replayability.value,
                    "operation": "iterable_data_source",
                },
            )
        if metadata.replayability is Replayability.SINGLE_PASS:
            if callable(chunks):
                raise TypeError("single-pass sources require an iterable, not an iterator factory")
            if not isinstance(chunks, Iterable):
                raise TypeError("chunks must be an iterable")
            self._iterable: Iterable[T] | None = chunks
            self._factory: Callable[[], Iterator[T]] | None = None
        else:
            if not callable(chunks):
                raise ValueError("replayable sources require an explicit iterator factory")
            self._iterable = None
            self._factory = chunks
        self.metadata = metadata
        self._acquired = 0
        self._lock = Lock()

    def iter_chunks(self) -> Iterator[T]:
        """Acquire chunks according to the immutable replayability declaration."""

        if self.metadata.replayability is Replayability.SINGLE_PASS:
            with self._lock:
                if self._acquired:
                    raise StreamSourceError(
                        FailureCode.PASS_BUDGET_EXCEEDED,
                        {"max_passes": 1, "attempted_pass": self._acquired + 1},
                    )
                self._acquired += 1
            assert self._iterable is not None
            return iter(self._iterable)
        assert self._factory is not None
        iterator = self._factory()
        if not isinstance(iterator, Iterator):
            raise TypeError("stream factory must return an iterator")
        with self._lock:
            self._acquired += 1
        return iterator


def iter_stream(source: StreamSource[T] | Iterable[T]) -> Iterator[T]:
    """Acquire one stream while preserving legacy iterable compatibility."""

    open_chunks = getattr(source, "iter_chunks", None)
    if callable(open_chunks):
        return cast(Iterator[T], open_chunks())
    if not isinstance(source, Iterable):
        raise TypeError("source must be a StreamSource or iterable")
    return iter(source)


__all__ = ["IterableDataSource", "StreamSource", "StreamSourceError", "iter_stream"]
