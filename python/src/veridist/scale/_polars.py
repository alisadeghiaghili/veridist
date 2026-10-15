"""The Polars source. It needs the optional ``polars`` extra.

``polars`` and ``pyarrow`` are imported on first use, never at import time; without
either, the first call that needs it raises ``ScaleSourceError`` with
``OPTIONAL_DEPENDENCY_MISSING`` and a hint naming the ``polars`` extra, which
installs both.

:class:`PolarsSource` reads one column of a ``polars.LazyFrame`` or a
``polars.DataFrame``. It is one partition. Its fingerprint level is ``NONE``: a
lazy plan can read anything and a frame is a value in memory, neither is a stored
object whose identity can be recognised again without reading all of it, so the
partition is *not resumable* and any later resume must refuse it.

The projection is pushed down: only the named column is selected, so a scan reads
only that column. A ``DataFrame`` is read from memory, zero-copy. A ``LazyFrame``
is run by Polars' streaming engine one bounded chunk at a time
(:func:`collect_batches`, the one place that uses Polars' streaming batch API,
which Polars marks unstable) and never materialised whole. Either way every chunk
is handed to the Arrow value policy of :func:`~veridist.scale._arrow.arrow_float64`
and capped at ``max_rows`` with zero-copy slices, so the accepted types, the
refusal of nulls, NaN and infinities, and the error codes are those of the Arrow
source.

A source can be read again: each call of ``batches`` re-reads the frame or runs the
plan again. The caller must keep the data unchanged (and the plan deterministic)
between reads; nothing checks this, which is why the partition is not resumable.
A plan that stops early (a refused batch ends the iteration) is stopped by Polars
when the iterator is released. A failure that Polars reports (a missing file, a
malformed input, a failing expression) is ``SOURCE_UNREADABLE``; an exception
raised by the caller's own Python function inside the plan is not a source
failure and propagates unchanged.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Final

from veridist.scale._arrow import check_column_name, column_batches, load_optional
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    FingerprintLevel,
    Partition,
    check_max_rows,
)

EXTRA_NAME: Final = "polars"


def polars_module() -> Any:
    """Load ``polars`` on first use; ``OPTIONAL_DEPENDENCY_MISSING`` if it is absent."""

    return load_optional("polars", EXTRA_NAME)


def _pyarrow_module() -> Any:
    """Load ``pyarrow``, which Polars needs to hand over Arrow data; the ``polars`` extra has it."""

    return load_optional("pyarrow", EXTRA_NAME)


def collect_batches(lazy: Any, max_rows: int) -> Iterator[Any]:
    """Run a lazy frame with Polars' streaming engine and yield its chunks as frames.

    This is the only call of ``LazyFrame.collect_batches``, which Polars documents
    as unstable. Chunks hold about ``max_rows`` rows, and the caller slices any
    that are larger. An error of the plan is raised by the iterator (from Polars
    1.37.0 on; earlier versions end the stream silently, which is why that is the
    minimum version of the extra).
    """

    chunks: Iterator[Any] = lazy.collect_batches(chunk_size=max_rows)
    return chunks


class PolarsSource:
    """A single-partition source over one column of a Polars frame; see the module text."""

    PARTITION_ID: Final = "polars"

    __slots__ = ("_frame", "_index", "_lazy", "_rows")

    def __init__(self, frame: object, column: str) -> None:
        check_column_name(column)
        pl = polars_module()
        _pyarrow_module()
        if isinstance(frame, pl.LazyFrame):
            self._lazy, self._rows = True, None
        elif isinstance(frame, pl.DataFrame):
            self._lazy, self._rows = False, frame.height
        else:
            raise TypeError("frame must be a polars LazyFrame or DataFrame")
        try:
            names = frame.collect_schema().names()
        except (pl.exceptions.PolarsError, OSError) as error:
            raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE) from error
        if column not in names:
            raise ScaleSourceError(ScaleSourceErrorCode.COLUMN_NOT_FOUND)
        self._frame: Any = frame
        self._index = names.index(column)

    def partitions(self) -> tuple[Partition, ...]:
        """The one partition: the row count of a ``DataFrame`` (``None`` for a plan),
        no fingerprint, hence not resumable."""

        return (Partition(self.PARTITION_ID, 0, self._rows, None, FingerprintLevel.NONE),)

    def batches(
        self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
    ) -> Iterator[Any]:
        """Validated float64 batches of at most ``max_rows`` rows."""

        bound = check_max_rows(max_rows)
        if not isinstance(partition, Partition) or partition.id != self.PARTITION_ID:
            raise ValueError("partition does not belong to this source")
        # Load the optional modules here, so that a missing extra is raised by this
        # call and not by the first batch.
        pl, pa = polars_module(), _pyarrow_module()
        return self._read(pl, pa, bound)

    def _read(self, pl: Any, pa: Any, max_rows: int) -> Iterator[Any]:
        # ``nth`` selects by position, so the name is never read as a pattern.
        projected = self._frame.select(pl.nth(self._index))
        try:
            chunks = collect_batches(projected, max_rows) if self._lazy else (projected,)
            for chunk in chunks:
                yield from column_batches(chunk.to_arrow().to_batches(), 0, max_rows)
        except (pl.exceptions.PolarsError, pa.ArrowException, OSError) as error:
            raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE) from error
