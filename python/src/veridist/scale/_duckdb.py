"""The DuckDB source. It needs the optional ``duckdb`` extra.

``duckdb`` and ``pyarrow`` are imported on first use, never at import time; without
either, the first call that needs it raises ``ScaleSourceError`` with
``OPTIONAL_DEPENDENCY_MISSING`` and a hint naming the ``duckdb`` extra, which
installs both.

:class:`DuckDBSource` reads one column of a DuckDB relation, or of the result of a
query given as a ``(connection, sql)`` pair (the pair is turned into a relation
with ``connection.sql(sql)``, so the statement must be a query that returns rows;
anything else is a programmer error). It is one partition. Its fingerprint level is
``NONE``: a query can read anything, and a result is not a stored object whose
identity can be recognised again without running it, so the partition is *not
resumable* and any later resume must refuse it.

The projection is pushed down: only the named column is selected, so a scan reads
only that column. The relation is run with ``to_arrow_reader`` and streamed in
batches of at most ``max_rows`` rows, never materialised whole; every batch goes
through the Arrow value policy of :func:`~veridist.scale._arrow.arrow_float64`
(and is capped with zero-copy slices), so the accepted types, the refusal of nulls,
NaN and infinities, and the error codes are those of the Arrow source. The column
types are DuckDB's own: a ``DECIMAL``, ``HUGEINT``, ``VARCHAR`` or ``BOOLEAN``
column is refused, ``DOUBLE``, ``FLOAT`` and the integer types are accepted.

A source can be read again: each call of ``batches`` runs the relation again. The
caller must keep the data unchanged between reads; nothing checks this, which is
why the partition is not resumable. The rows of a result without an ``ORDER BY``
come in no guaranteed order. That does not matter here: a state depends on the
multiset of values only, so the folded state is the same whatever the order.
One connection runs one result at a time, so a source must be read to its end, or
dropped, before the same connection is used for another query.
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

EXTRA_NAME: Final = "duckdb"


def duckdb_module() -> Any:
    """Load ``duckdb`` on first use; ``OPTIONAL_DEPENDENCY_MISSING`` if it is absent."""

    return load_optional("duckdb", EXTRA_NAME)


def _pyarrow_module() -> Any:
    """Load ``pyarrow``, which DuckDB needs to hand over Arrow data; the ``duckdb`` extra has it."""

    return load_optional("pyarrow", EXTRA_NAME)


def to_arrow_reader(relation: Any, max_rows: int) -> Any:
    """Run a relation and return its Arrow ``RecordBatchReader`` of at most ``max_rows`` rows.

    This is the only call of DuckDB's result-streaming API, which has had two names
    (``fetch_arrow_reader`` before DuckDB 1.5.0); the extra's minimum version is the
    first with ``to_arrow_reader``. A failing query is raised by this call or by
    the reader, as a ``duckdb.Error``.
    """

    return relation.to_arrow_reader(max_rows)


def _quoted(name: str) -> str:
    """``name`` as a quoted SQL identifier, so any column name is taken literally."""

    return '"' + name.replace('"', '""') + '"'


def _as_relation(duckdb: Any, source: object) -> Any:
    """The relation a source argument stands for: itself, or the query of a pair."""

    if isinstance(source, duckdb.DuckDBPyRelation):
        return source
    if not (
        type(source) is tuple
        and len(source) == 2
        and isinstance(source[0], duckdb.DuckDBPyConnection)
        and type(source[1]) is str
    ):
        raise TypeError(
            "source must be a duckdb relation or a (connection, sql) pair "
            "of a duckdb connection and a query string"
        )
    relation = source[0].sql(source[1])
    if relation is None:
        raise ValueError("sql must be a query that returns rows")
    return relation


class DuckDBSource:
    """A single-partition source over one column of a DuckDB relation or query."""

    PARTITION_ID: Final = "duckdb"

    __slots__ = ("_relation",)

    def __init__(self, source: object, column: str) -> None:
        check_column_name(column)
        duckdb = duckdb_module()
        _pyarrow_module()
        try:
            relation = _as_relation(duckdb, source)
            if relation.columns.count(column) != 1:
                raise ScaleSourceError(ScaleSourceErrorCode.COLUMN_NOT_FOUND)
            self._relation: Any = relation.project(_quoted(column))
        except duckdb.Error as error:
            raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE) from error

    def partitions(self) -> tuple[Partition, ...]:
        """The one partition: unknown row count, no fingerprint, hence not resumable."""

        return (Partition(self.PARTITION_ID, 0, None, None, FingerprintLevel.NONE),)

    def batches(
        self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
    ) -> Iterator[Any]:
        """Validated float64 batches of at most ``max_rows`` rows."""

        bound = check_max_rows(max_rows)
        if not isinstance(partition, Partition) or partition.id != self.PARTITION_ID:
            raise ValueError("partition does not belong to this source")
        # Load the optional modules here, so that a missing extra is raised by this
        # call and not by the first batch.
        duckdb, pa = duckdb_module(), _pyarrow_module()
        return self._read(duckdb, pa, bound)

    def _read(self, duckdb: Any, pa: Any, max_rows: int) -> Iterator[Any]:
        try:
            yield from column_batches(to_arrow_reader(self._relation, max_rows), 0, max_rows)
        except (duckdb.Error, pa.ArrowException) as error:
            raise ScaleSourceError(ScaleSourceErrorCode.SOURCE_UNREADABLE) from error
