"""An explicit, opt-in local spool of the projected float64 column for multi-pass fits.

A multi-pass family reads its source several times. That fails for a source that can be
read once (an :class:`~veridist.scale._arrow.ArrowSource` over a reader or a stream
object) and is wasteful when a read is expensive (a Polars plan or a DuckDB query that is
executed again for every pass, remote or slow storage). A *spool* keeps one local copy of
the values the first pass has validated, so that the source is read exactly once and every
later pass reads the copy.

It is never implicit. A fit uses a spool only when the caller passes a
:class:`SpoolPolicy`; without one, a multi-pass family refuses a source that can be read
only once before it reads anything (``SPOOL_REQUIRED``). A one-pass family reads its
source once and ignores the policy: it does not create a directory or a file.

Layout. The spool is a new private directory, ``veridist-spool-*``, made inside the
directory the policy names (or the system temporary directory, which the policy must
allow explicitly), with permissions for its owner only where the platform supports them.
It holds one Arrow IPC file per partition, named by the partition's position and never by
its identifier, so a partition identifier cannot influence a path. Each file holds the
column ``x``, float64 and without nulls, in record batches of the batches the first pass
read (at most ``max_rows`` rows). The files are created exclusively and mode ``0o600``.
The first pass writes each partition's batches as it reads them, so the source is
read once, in its canonical order. The spool needs ``pyarrow`` (the ``arrow`` extra),
imported on first use.

Reading. Later passes read a file back one record batch at a time and hand it over in
pieces of at most ``max_rows`` rows. The pieces are zero-copy views of a memory-mapped file
where the platform allows it (not on Windows, which cannot delete a file that is still
mapped while a failure unwinds), and of a batch read into memory otherwise. Either way every
read hashes the values it returns.

Integrity. The spool records, for every partition, the row count, the number of record
batches, the size of its largest batch and the SHA-256 of its values as little-endian
binary64 in the order written. Every read verifies the schema, the batch count, that no
batch is larger than the largest written, the row count and the digest.
Any difference, a missing or truncated file included, is ``SPOOL_CHANGED``. The check
completes at the end of the partition, before the partition's last batch is handed over
(the reader raises instead of handing it over), so a pass that read a changed partition
fails and no estimate is returned from it. A digest is an integrity check against accidental change
and a changed file, not authentication.

Budget. ``max_bytes`` bounds the space the spool may use. Every file is charged
``FILE_FRAMING_BYTES`` and every record batch ``8 * rows + BATCH_FRAMING_BYTES``, each of
which is an upper bound on what the Arrow IPC format adds. The charge is checked before the
file is created or the batch is written, and a charge beyond ``max_bytes`` is
``SPOOL_LIMIT_EXCEEDED``, so the files never exceed the budget.

Cleanup. The directory is deleted when the :class:`Spool` context exits, whether the fit
returned, failed or was interrupted (``KeyboardInterrupt`` included): the open handles are
closed first, then the whole directory is removed. A failure to delete is
``SPOOL_UNAVAILABLE``; it is raised only when nothing else is in flight, so that it never
replaces the error or the interrupt that is being handled. No message or attribute holds a
path or a value.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Final

from veridist.engine.provenance import ScaleSpoolPartitionProvenance, ScaleSpoolProvenance
from veridist.scale._arrow import pyarrow_module
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._kernels import numpy_module
from veridist.scale._sources import DEFAULT_MAX_ROWS, Partition, Source, check_max_rows

DIRECTORY_PREFIX: Final = "veridist-spool-"
"""The name every spool directory starts with."""

FILE_SUFFIX: Final = ".arrow"
FILE_MODE: Final = 0o600
COLUMN_NAME: Final = "x"
VALUE_BYTES: Final = 8
"""Bytes of one float64 value."""

BATCH_FRAMING_BYTES: Final = 256
"""Charged for each record batch beyond its values: an upper bound on its IPC framing."""

FILE_FRAMING_BYTES: Final = 1024
"""Charged for each file: an upper bound on its schema, its footer and its markers."""

MEMORY_MAP: Final = sys.platform != "win32"
"""Whether spool files are memory-mapped when read (see the module documentation)."""

HINT_REQUIRED: Final = (
    "the source can be read only once but this family reads it several times; "
    "pass spool=SpoolPolicy(...) to keep a local copy"
)
HINT_LIMIT: Final = "the spool would exceed max_bytes; raise max_bytes of the SpoolPolicy"
HINT_CHANGED: Final = (
    "the spool no longer holds what was written to it; it was changed or damaged"
)
HINT_UNAVAILABLE: Final = (
    "the spool could not be used; check that its directory exists, is writable and has room"
)
HINT_CLEANUP: Final = (
    "the spool could not be fully deleted; remove the veridist-spool-* directories "
    "left in the spool directory"
)

_CREATE_FLAGS: Final = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)


def batch_cost(rows: int) -> int:
    """The bytes a record batch of ``rows`` values is charged against ``max_bytes``."""

    return rows * VALUE_BYTES + BATCH_FRAMING_BYTES


@dataclass(frozen=True, slots=True, kw_only=True)
class SpoolPolicy:
    """The caller's explicit request for a local spool, and its limits.

    ``max_bytes`` is the budget (a built-in integer of at least one). ``directory`` is an
    existing directory the spool's private directory is made in; it is not created. With
    no ``directory``, ``allow_system_temp=True`` lets the spool use the system temporary
    directory instead, and without that flag a missing directory is refused: a spool never
    chooses where to write on its own.
    """

    max_bytes: int
    directory: str | os.PathLike[str] | None = None
    allow_system_temp: bool = False

    def __post_init__(self) -> None:
        if type(self.max_bytes) is not int:
            raise TypeError("max_bytes must be a built-in integer")
        if self.max_bytes < 1:
            raise ValueError("max_bytes must be at least 1")
        if type(self.allow_system_temp) is not bool:
            raise TypeError("allow_system_temp must be a bool")
        if self.directory is None:
            if not self.allow_system_temp:
                raise ValueError("a spool needs a directory, or allow_system_temp=True")
        else:
            if self.allow_system_temp:
                raise ValueError("directory and allow_system_temp cannot both be given")
            if not isinstance(self.directory, str | os.PathLike) or not isinstance(
                os.fspath(self.directory), str
            ):
                raise TypeError("directory must be a str or a path-like object of str")

    def base_directory(self) -> str:
        """The absolute directory the spool's private directory is made in."""

        if self.directory is None:
            return tempfile.gettempdir()
        return os.path.abspath(os.fspath(self.directory))


@dataclass(frozen=True, slots=True)
class _Entry:
    """A partition the spool holds completely."""

    ordinal: int
    path: str
    rows: int
    batches: int
    largest: int
    digest: str


def _refused(code: ScaleSourceErrorCode, hint: str) -> ScaleSourceError:
    return ScaleSourceError(code, hint)


def _hashed(digest: Any, values: Any) -> None:
    """Feed ``values`` to ``digest`` as little-endian binary64, without a copy where possible."""

    digest.update(numpy_module().ascontiguousarray(values, dtype="<f8"))


def _create_file(path: str) -> Any:
    """A new binary file at ``path``, for its owner only; it must not exist."""

    return os.fdopen(os.open(path, _CREATE_FLAGS, FILE_MODE), "wb")


class SpooledSource:
    """The source a spooled fit reads: the first read of a partition writes it to the spool.

    ``partitions()`` is the underlying source's. The first read of a partition reads the
    underlying source and writes what it yields to the spool; every later read of that
    partition replays the spool, verified. Nothing else reads the underlying source.
    """

    __slots__ = ("_source", "_spool")

    def __init__(self, spool: Spool, source: Source) -> None:
        self._spool = spool
        self._source = source

    def partitions(self) -> tuple[Partition, ...]:
        return self._source.partitions()

    def batches(
        self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
    ) -> Iterator[Any]:
        return self._spool.batches(self._source, partition, check_max_rows(max_rows))


class Spool:
    """A private spool directory; use it as a context manager, which deletes it on exit.

    Creating it needs ``pyarrow`` (``OPTIONAL_DEPENDENCY_MISSING`` otherwise) and a usable
    directory (``SPOOL_UNAVAILABLE`` otherwise); nothing is created in the first case.
    """

    def __init__(self, policy: SpoolPolicy) -> None:
        if not isinstance(policy, SpoolPolicy):
            raise TypeError("policy must be a SpoolPolicy")
        pa = pyarrow_module()
        self._pa = pa
        self._policy = policy
        self._schema = pa.schema([pa.field(COLUMN_NAME, pa.float64(), nullable=False)])
        self._begun: set[str] = set()
        self._sealed: dict[str, _Entry] = {}
        self._handles: list[Any] = []
        self._charged = 0
        self._closed = False
        try:
            self._directory = tempfile.mkdtemp(
                prefix=DIRECTORY_PREFIX, dir=policy.base_directory()
            )
        except OSError:
            raise _refused(ScaleSourceErrorCode.SPOOL_UNAVAILABLE, HINT_UNAVAILABLE) from None

    def __enter__(self) -> Spool:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        failure = self._delete()
        if failure is not None and exc_type is None:
            raise failure

    def source(self, source: Source) -> SpooledSource:
        """The source that reads ``source`` once and then the spool."""

        return SpooledSource(self, source)

    def provenance(self) -> ScaleSpoolProvenance:
        """The row count and the digest of every partition the spool holds, in read order."""

        partitions = tuple(
            ScaleSpoolPartitionProvenance(entry.ordinal, entry.rows, entry.digest)
            for entry in self._sealed.values()
        )
        return ScaleSpoolProvenance(sum(item.rows for item in partitions), partitions)

    def batches(self, source: Source, partition: Partition, max_rows: int) -> Iterator[Any]:
        """Read ``partition``: through ``source`` into the spool the first time, from it after."""

        if not isinstance(partition, Partition):
            raise ValueError("partition does not belong to this source")
        entry = self._sealed.get(partition.id)
        if entry is not None:
            return self._replay(entry, max_rows)
        if partition.id in self._begun:
            raise _refused(ScaleSourceErrorCode.SPOOL_CHANGED, HINT_CHANGED)
        # Reading the underlying source may refuse at once (a foreign partition, a missing
        # extra); that is raised by this call, and then nothing has been begun.
        inner = source.batches(partition, max_rows=max_rows)
        index = len(self._begun)
        self._begun.add(partition.id)
        return self._record(inner, partition, index)

    def _charge(self, amount: int) -> None:
        """Account for ``amount`` bytes before they are written; refuse beyond the budget."""

        if amount > self._policy.max_bytes - self._charged:
            raise _refused(ScaleSourceErrorCode.SPOOL_LIMIT_EXCEEDED, HINT_LIMIT)
        self._charged += amount

    def _record(self, inner: Iterator[Any], partition: Partition, index: int) -> Iterator[Any]:
        pa = self._pa
        path = os.path.join(self._directory, f"{index:08d}{FILE_SUFFIX}")
        self._charge(FILE_FRAMING_BYTES)
        try:
            stream = _create_file(path)
            self._handles.append(stream)
            writer = pa.ipc.new_file(stream, self._schema)
        except (OSError, pa.ArrowException):
            raise _refused(ScaleSourceErrorCode.SPOOL_UNAVAILABLE, HINT_UNAVAILABLE) from None
        digest = hashlib.sha256()
        rows = batches = largest = 0
        for batch in inner:
            count = int(batch.shape[0])
            if count == 0:
                yield batch
                continue
            self._charge(batch_cost(count))
            try:
                writer.write_batch(pa.record_batch([pa.array(batch)], schema=self._schema))
            except (OSError, pa.ArrowException):
                raise _refused(
                    ScaleSourceErrorCode.SPOOL_UNAVAILABLE, HINT_UNAVAILABLE
                ) from None
            _hashed(digest, batch)
            rows += count
            batches += 1
            largest = max(largest, count)
            yield batch
        try:
            writer.close()
            self._release(stream)
        except (OSError, pa.ArrowException):
            raise _refused(ScaleSourceErrorCode.SPOOL_UNAVAILABLE, HINT_UNAVAILABLE) from None
        self._sealed[partition.id] = _Entry(
            partition.ordinal, path, rows, batches, largest, digest.hexdigest()
        )

    def _open_reader(self, path: str) -> tuple[Any, Any]:
        pa = self._pa
        handle = pa.memory_map(path, "r") if MEMORY_MAP else pa.OSFile(path, "rb")
        self._handles.append(handle)
        try:
            return handle, pa.ipc.open_file(handle)
        except BaseException:
            self._release(handle)
            raise

    def _replay(self, entry: _Entry, max_rows: int) -> Iterator[Any]:
        pa = self._pa
        handle = None
        try:
            handle, reader = self._open_reader(entry.path)
            if reader.num_record_batches != entry.batches or not reader.schema.equals(
                self._schema
            ):
                raise _refused(ScaleSourceErrorCode.SPOOL_CHANGED, HINT_CHANGED)
            digest = hashlib.sha256()
            rows = 0
            held = None
            for index in range(entry.batches):
                record = reader.get_batch(index)
                if record.num_rows > entry.largest:
                    raise _refused(ScaleSourceErrorCode.SPOOL_CHANGED, HINT_CHANGED)
                values = record.column(0).to_numpy(zero_copy_only=True)
                _hashed(digest, values)
                rows += record.num_rows
                for start in range(0, record.num_rows, max_rows):
                    if held is not None:
                        yield held
                    held = values[start : start + max_rows]
            if rows != entry.rows or digest.hexdigest() != entry.digest:
                raise _refused(ScaleSourceErrorCode.SPOOL_CHANGED, HINT_CHANGED)
            if held is not None:
                yield held
        except (OSError, pa.ArrowException):
            raise _refused(ScaleSourceErrorCode.SPOOL_CHANGED, HINT_CHANGED) from None
        finally:
            if handle is not None:
                self._release(handle)

    def _release(self, handle: Any) -> None:
        """Close one handle and forget it (it may already be gone if the spool was deleted)."""

        if handle in self._handles:
            self._handles.remove(handle)
        handle.close()

    def _delete(self) -> ScaleSourceError | None:
        """Close every handle and remove the directory; the failure, if any, is returned."""

        if self._closed:
            return None
        self._closed = True
        failed = False
        for handle in self._handles:
            try:
                handle.close()
            except OSError:
                failed = True
        self._handles.clear()
        try:
            shutil.rmtree(self._directory)
        except FileNotFoundError:
            pass
        except OSError:
            failed = True
        if failed:
            return _refused(ScaleSourceErrorCode.SPOOL_UNAVAILABLE, HINT_CLEANUP)
        return None
