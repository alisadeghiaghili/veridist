"""Helpers for the Polars and DuckDB source tests.

This module imports ``polars``, ``duckdb`` and ``pyarrow`` at import time: every
test that uses it needs the ``polars`` and ``duckdb`` extras, which the test lanes
install.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl
import pyarrow as pa

from veridist.scale._duckdb import DuckDBSource
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._polars import PolarsSource
from veridist.scale._sources import DEFAULT_MAX_ROWS


def code_of(call: Callable[[], object]) -> ScaleSourceErrorCode:
    """The code of the ``ScaleSourceError`` that ``call`` raises."""

    try:
        call()
    except ScaleSourceError as error:
        return error.code
    raise AssertionError("the call was not refused")


def all_batches(source: Any, max_rows: int = DEFAULT_MAX_ROWS) -> list[np.ndarray]:
    """Every batch of the one partition of ``source``."""

    (partition,) = source.partitions()
    return list(source.batches(partition, max_rows=max_rows))


def values_of(source: Any, max_rows: int = DEFAULT_MAX_ROWS) -> list[float]:
    """Every value of the one partition of ``source``, in the order it is read."""

    return [value for batch in all_batches(source, max_rows) for value in batch.tolist()]


def sql_path(path: Path) -> str:
    """A path as a DuckDB SQL string literal."""

    return "'" + path.as_posix().replace("'", "''") + "'"


def polars_frame(table: pa.Table) -> Any:
    return pl.from_arrow(table)


def polars_lazy(table: pa.Table) -> Any:
    return pl.from_arrow(table).lazy()


def connect() -> Any:
    """An in-memory DuckDB connection that runs on one thread.

    A default connection starts one worker per core, which the tests do not need; one
    thread keeps the number of native threads of the test process small.
    """

    return duckdb.connect(config={"threads": 1})


def duckdb_relation(table: pa.Table) -> Any:
    return connect().from_arrow(table)


FRAME_SOURCES: dict[str, Callable[[pa.Table, str], Any]] = {
    "polars frame": lambda table, column: PolarsSource(polars_frame(table), column),
    "polars lazy frame": lambda table, column: PolarsSource(polars_lazy(table), column),
    "duckdb relation": lambda table, column: DuckDBSource(duckdb_relation(table), column),
}
"""Every adapter built over the same Arrow table: name to a source factory."""
