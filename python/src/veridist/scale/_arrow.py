"""Arrow and Parquet sources. They need the optional ``arrow`` extra.

``pyarrow`` is imported on first use, never at import time; without it the first
call that needs it raises ``ScaleSourceError`` with ``OPTIONAL_DEPENDENCY_MISSING``
and a hint naming the extra, so the failure never surfaces in the middle of a
pass as a bare ``ImportError``.

* :class:`ArrowSource` reads a ``pyarrow`` ``Table``, ``RecordBatch`` or
  ``RecordBatchReader``, or any object that exports the Arrow C stream interface
  (``__arrow_c_stream__``). It is one partition. Its fingerprint level is
  ``NONE``: a stream cannot be fingerprinted without being read, so the partition
  is *not resumable* and any later resume must refuse it. A reader or a stream
  object can be read once; a ``Table`` or a ``RecordBatch`` can be read again.
* :class:`ParquetSource` reads one column of one or more Parquet files, one
  partition per (file, row group), pushing the projection down to the reader.

Parquet fingerprints. The default level is ``METADATA``: a digest of a canonical
record of the partition's file identity (path relative to the declared root,
else absolute, in POSIX form), the file size and modification time in
nanoseconds, the position of the row group, its row count and byte size, and for
each column chunk of the projected column its type, codec, encodings, page
offsets, sizes and statistics, together with the projected column's Arrow type.
It detects a replaced or rewritten file. It does **not** detect an in-place byte
edit that leaves the size, the modification time and the metadata unchanged. With
``content_hash=True`` the level is ``CONTENT`` and the record also holds the
SHA-256 of the column chunk's stored (compressed) bytes, read in bounded blocks;
that costs one extra read of the projected column per call of ``partitions()``.
Moving or copying a file changes its modification time and, without a declared
root, its path, and therefore its fingerprint.

Partition ids are ``"<posix path>#<row group>"`` and so are stable across runs and
platforms when a root is declared (or implied by a single directory). The canonical
order is the sorted path, then the row-group index.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Sequence
from hashlib import sha256
from importlib import import_module
from pathlib import Path
from typing import Any, Final

from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    FingerprintLevel,
    Partition,
    canonical_digest,
    check_max_rows,
    float64_values,
)

EXTRA_NAME: Final = "arrow"
PARQUET_SUFFIX: Final = ".parquet"
CONTENT_BLOCK_BYTES: Final = 1 << 20
"""Bytes read at a time when hashing a column chunk."""

_MISSING_EXTRA_HINT: Final = (
    f'install the optional "{EXTRA_NAME}" extra with: pip install "veridist[{EXTRA_NAME}]"'
)


def _load(name: str) -> Any:
    try:
        return import_module(name)
    except ImportError as error:
        raise ScaleSourceError(
            ScaleSourceErrorCode.OPTIONAL_DEPENDENCY_MISSING, _MISSING_EXTRA_HINT
        ) from error


def pyarrow_module() -> Any:
    """Load ``pyarrow`` on first use; ``OPTIONAL_DEPENDENCY_MISSING`` if it is absent."""

    return _load("pyarrow")


def pyarrow_parquet_module() -> Any:
    """Load ``pyarrow.parquet`` on first use; ``OPTIONAL_DEPENDENCY_MISSING`` if absent."""

    return _load("pyarrow.parquet")


def arrow_float64(array: Any) -> Any:
    """Apply the value policy to one Arrow array and return finite float64 values.

    The type gate and the null check come first and read no value. A float64
    array without nulls becomes a zero-copy numpy view (read-only); the numpy
    rules of :func:`~veridist.scale._sources.float64_values` do the rest.
    ``ScaleSourceError``: ``UNSUPPORTED_DTYPE`` (any type that is not a float or
    an integer, and floats other than float32 and float64), ``NULL_VALUE``,
    ``LOSSY_CAST``, ``NAN_VALUE``, ``NON_FINITE_VALUE``.
    """

    types = pyarrow_module().types
    if not (types.is_floating(array.type) or types.is_integer(array.type)):
        raise ScaleSourceError(ScaleSourceErrorCode.UNSUPPORTED_DTYPE)
    if array.null_count:
        raise ScaleSourceError(ScaleSourceErrorCode.NULL_VALUE)
    return float64_values(array.to_numpy(zero_copy_only=True))


def _column_index(schema: Any, column: str) -> int:
    """The index of the one field called ``column``; none or several is ``COLUMN_NOT_FOUND``."""

    indices = schema.get_all_field_indices(column)
    if len(indices) != 1:
        raise ScaleSourceError(ScaleSourceErrorCode.COLUMN_NOT_FOUND)
    return int(indices[0])


def _column_batches(batches: Any, index: int, max_rows: int) -> Iterator[Any]:
    """Zero-copy slices of at most ``max_rows`` rows of one column, validated."""

    pa = pyarrow_module()
    try:
        for batch in batches:
            for start in range(0, batch.num_rows, max_rows):
                yield arrow_float64(batch.slice(start, max_rows).column(index))
    except pa.ArrowException as error:
        raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE) from error


def _check_column_name(column: object) -> str:
    if type(column) is not str or not column:
        raise TypeError("column must be a non-empty string")
    return column


class ArrowSource:
    """A single-partition source over Arrow data; see the module documentation."""

    PARTITION_ID: Final = "arrow"

    __slots__ = ("_batches", "_consumed", "_index", "_reader", "_rows")

    def __init__(self, data: object, column: str) -> None:
        _check_column_name(column)
        pa = pyarrow_module()
        self._batches: tuple[Any, ...] | None = None
        self._reader: Any = None
        self._rows: int | None = None
        self._consumed = False
        if isinstance(data, pa.Table):
            schema = data.schema
            self._batches, self._rows = tuple(data.to_batches()), data.num_rows
        elif isinstance(data, pa.RecordBatch):
            schema = data.schema
            self._batches, self._rows = (data,), data.num_rows
        elif isinstance(data, pa.RecordBatchReader):
            schema, self._reader = data.schema, data
        elif hasattr(data, "__arrow_c_stream__"):
            try:
                self._reader = pa.RecordBatchReader.from_stream(data)
            except pa.ArrowException as error:
                raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE) from error
            schema = self._reader.schema
        else:
            raise TypeError(
                "data must be a pyarrow Table, RecordBatch or RecordBatchReader, "
                "or expose __arrow_c_stream__"
            )
        self._index = _column_index(schema, column)

    def partitions(self) -> tuple[Partition, ...]:
        """The one partition: no fingerprint, hence not resumable."""

        return (Partition(self.PARTITION_ID, 0, self._rows, None, FingerprintLevel.NONE),)

    def batches(
        self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
    ) -> Iterator[Any]:
        """Validated float64 batches of at most ``max_rows`` rows.

        A source built from a reader or a stream object can be read once; a second
        call is ``SOURCE_CONSUMED`` (an exhausted stream would otherwise look like
        an empty one).
        """

        bound = check_max_rows(max_rows)
        if not isinstance(partition, Partition) or partition.id != self.PARTITION_ID:
            raise ValueError("partition does not belong to this source")
        if self._reader is None:
            return _column_batches(self._batches, self._index, bound)
        if self._consumed:
            raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_CONSUMED)
        self._consumed = True
        return _column_batches(self._reader, self._index, bound)


def _stat_text(value: object) -> str:
    return value.hex() if isinstance(value, float) else repr(value)


def _chunk_record(chunk: Any) -> tuple[object, ...]:
    """The canonical record of one column chunk's metadata."""

    stats = chunk.statistics
    if stats is None:
        statistics: tuple[object, ...] | None = None
    elif stats.has_min_max:
        statistics = (
            _stat_text(stats.min),
            _stat_text(stats.max),
            stats.null_count,
            stats.num_values,
        )
    else:
        statistics = (None, None, stats.null_count, stats.num_values)
    return (
        chunk.path_in_schema,
        str(chunk.physical_type),
        str(chunk.compression),
        tuple(sorted(str(encoding) for encoding in chunk.encodings)),
        int(chunk.has_dictionary_page),
        chunk.dictionary_page_offset if chunk.has_dictionary_page else None,
        chunk.data_page_offset,
        chunk.total_compressed_size,
        chunk.total_uncompressed_size,
        chunk.num_values,
        statistics,
    )


def _projected_chunks(row_group: Any, column: str) -> tuple[Any, ...]:
    """The column chunks of ``column`` (one, or the leaves of a nested column)."""

    chunks = (row_group.column(index) for index in range(row_group.num_columns))
    prefix = column + "."
    return tuple(
        chunk
        for chunk in chunks
        if chunk.path_in_schema == column or chunk.path_in_schema.startswith(prefix)
    )


def _content_digest(path: Path, chunks: Sequence[Any]) -> str:
    """The SHA-256 of the stored bytes of the given column chunks, in bounded blocks."""

    digest = sha256()
    with path.open("rb") as stream:
        for chunk in chunks:
            if chunk.has_dictionary_page:
                stream.seek(chunk.dictionary_page_offset)
            else:
                stream.seek(chunk.data_page_offset)
            size = chunk.total_compressed_size
            for position in range(0, size, CONTENT_BLOCK_BYTES):
                digest.update(stream.read(min(CONTENT_BLOCK_BYTES, size - position)))
    return digest.hexdigest()


def row_group_record(
    identity: str,
    size: int,
    mtime_ns: int,
    row_group_index: int,
    row_group_count: int,
    row_group: Any,
    column: str,
    column_type: str,
    content: str | None,
) -> tuple[object, ...]:
    """The canonical record that a Parquet partition fingerprint is the digest of."""

    return (
        "parquet",
        identity,
        size,
        mtime_ns,
        row_group_index,
        row_group_count,
        row_group.num_rows,
        row_group.total_byte_size,
        column,
        column_type,
        tuple(_chunk_record(chunk) for chunk in _projected_chunks(row_group, column)),
        content,
    )


class ParquetSource:
    """One column of one or more Parquet files, one partition per (file, row group).

    ``paths`` is a file, a directory (searched recursively for ``*.parquet``) or a
    sequence of files and directories; files are processed in sorted order of
    their POSIX identity. ``root`` declares the directory that identities are
    relative to; a single directory argument is its own root. Without a root,
    identities are absolute paths. ``content_hash=True`` selects the ``CONTENT``
    fingerprint level (see the module documentation).
    """

    __slots__ = ("_column", "_content_hash", "_files")

    def __init__(
        self,
        paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
        column: str,
        *,
        root: str | os.PathLike[str] | None = None,
        content_hash: bool = False,
    ) -> None:
        self._column = _check_column_name(column)
        if type(content_hash) is not bool:
            raise TypeError("content_hash must be a bool")
        self._content_hash = content_hash
        entries = [paths] if isinstance(paths, str | os.PathLike) else list(paths)
        if not entries:
            raise ValueError("paths must not be empty")
        base = None if root is None else Path(os.path.abspath(root))
        if base is None and len(entries) == 1 and Path(entries[0]).is_dir():
            base = Path(os.path.abspath(entries[0]))
        found: dict[str, Path] = {}
        for entry in entries:
            for path in self._expand(Path(os.path.abspath(entry))):
                identity = self._identity(path, base)
                if identity in found:
                    raise ValueError("the same file is listed more than once")
                found[identity] = path
        if not found:
            raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_NOT_FOUND)
        self._files: dict[str, Path] = dict(sorted(found.items()))

    @staticmethod
    def _expand(path: Path) -> list[Path]:
        if path.is_dir():
            return [item for item in path.rglob(f"*{PARQUET_SUFFIX}") if item.is_file()]
        if path.is_file():
            return [path]
        raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_NOT_FOUND)

    @staticmethod
    def _identity(path: Path, base: Path | None) -> str:
        if base is None:
            return path.as_posix()
        try:
            return path.relative_to(base).as_posix()
        except ValueError as error:
            raise ValueError("every file must be under the declared root") from error

    def partitions(self) -> tuple[Partition, ...]:
        """One partition per (file, row group), in canonical order, from metadata only
        (plus one read of the projected column's bytes with ``content_hash``)."""

        pq = pyarrow_parquet_module()
        pa = pyarrow_module()
        found: list[Partition] = []
        try:
            for identity, path in self._files.items():
                status = path.stat()
                with pq.ParquetFile(path) as parquet:
                    schema = parquet.schema_arrow
                    index = _column_index(schema, self._column)
                    column_type = str(schema.field(index).type)
                    metadata = parquet.metadata
                    for group in range(metadata.num_row_groups):
                        row_group = metadata.row_group(group)
                        content = None
                        if self._content_hash:
                            chunks = _projected_chunks(row_group, self._column)
                            content = _content_digest(path, chunks)
                        record = row_group_record(
                            identity,
                            status.st_size,
                            status.st_mtime_ns,
                            group,
                            metadata.num_row_groups,
                            row_group,
                            self._column,
                            column_type,
                            content,
                        )
                        found.append(
                            Partition(
                                f"{identity}#{group}",
                                len(found),
                                row_group.num_rows,
                                canonical_digest(*record),
                                FingerprintLevel.CONTENT
                                if self._content_hash
                                else FingerprintLevel.METADATA,
                            )
                        )
        except (OSError, pa.ArrowException) as error:
            raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE) from error
        return tuple(found)

    def batches(
        self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
    ) -> Iterator[Any]:
        """Validated float64 batches of one row group, at most ``max_rows`` rows each.

        The row group's presence and row count are checked against the descriptor;
        a mismatch is ``PARTITION_CHANGED``.
        """

        bound = check_max_rows(max_rows)
        if not isinstance(partition, Partition):
            raise ValueError("partition does not belong to this source")
        identity, separator, number = partition.id.rpartition("#")
        known = separator and identity in self._files
        if not (known and number.isascii() and number.isdigit()):
            raise ValueError("partition does not belong to this source")
        # Load the optional modules here, so that a missing extra is raised by this
        # call and not by the first batch.
        pq, pa = pyarrow_parquet_module(), pyarrow_module()
        return self._read(pq, pa, self._files[identity], int(number), partition.rows, bound)

    def _read(
        self, pq: Any, pa: Any, path: Path, group: int, rows: int | None, max_rows: int
    ) -> Iterator[Any]:
        try:
            with pq.ParquetFile(path) as parquet:
                metadata = parquet.metadata
                if group >= metadata.num_row_groups or (
                    rows is not None and metadata.row_group(group).num_rows != rows
                ):
                    raise ScaleSourceError(ScaleSourceErrorCode.PARTITION_CHANGED)
                _column_index(parquet.schema_arrow, self._column)
                reader = parquet.iter_batches(
                    batch_size=max_rows, row_groups=[group], columns=[self._column]
                )
                yield from _column_batches(reader, 0, max_rows)
        except (OSError, pa.ArrowException) as error:
            raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE) from error
