"""Helpers that write small Parquet files and wrap Arrow data, for the source tests.

This module imports ``pyarrow`` at import time: every test that uses it needs the
``arrow`` extra, which the test lanes install.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(
    path: Path, columns: dict[str, Any], row_group_size: int | None = None, **options: Any
) -> Path:
    """Write ``columns`` (name to array-like) to ``path`` and return the path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, row_group_size=row_group_size, **options)
    return path


def set_mtime(path: Path, mtime_ns: int) -> None:
    os.utime(path, ns=(mtime_ns, mtime_ns))


def chunk_bytes(path: Path, row_group: int, column: int = 0) -> tuple[int, int]:
    """``(start, size)`` of a column chunk's stored bytes, from the file's own metadata."""

    with pq.ParquetFile(path) as parquet:
        chunk = parquet.metadata.row_group(row_group).column(column)
        if chunk.has_dictionary_page:
            start = chunk.dictionary_page_offset
        else:
            start = chunk.data_page_offset
        return int(start), int(chunk.total_compressed_size)


class StreamExporter:
    """An object that is neither a pyarrow type nor a reader, only ``__arrow_c_stream__``."""

    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.calls = 0

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        self.calls += 1
        return self.table.__arrow_c_stream__(requested_schema)


def failing_reader(schema: pa.Schema, good: list[Any]) -> pa.RecordBatchReader:
    """A reader that yields the ``good`` batches and then fails with an Arrow error."""

    def generate() -> Iterator[Any]:
        yield from good
        raise pa.ArrowInvalid("injected failure")

    return pa.RecordBatchReader.from_batches(schema, generate())


def float_column(values: list[float] | np.ndarray) -> pa.Array:
    return pa.array(np.asarray(values, dtype=np.float64))
