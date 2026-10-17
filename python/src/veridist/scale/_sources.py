"""The source contract: partitions, bounded float64 batches and a typed value policy.

A *source* turns stored data into the one input the exact states accept: one-
dimensional, C-contiguous float64 numpy arrays, a bounded number of rows at a
time. It is described by two calls:

* ``partitions()`` returns the source's partitions in their canonical order. A
  :class:`Partition` is an immutable descriptor that is cheap to obtain: a stable
  identifier, the ordinal that fixes its place in that order, the row count when
  the source states one, and a fingerprint with a declared level.
* ``batches(partition, max_rows=...)`` yields the validated batches of one
  partition, each at most ``max_rows`` rows. Batches are zero-copy views of the
  stored data wherever the stored type is already float64, so they may be
  read-only and must not be modified.

Value policy. Every refusal is a :class:`~veridist.scale._errors.ScaleSourceError`
with a closed code, raised before any value reaches a state, so a refused batch
can never half-update one. A column must exist; ``float64`` is accepted as it is;
``float32`` is widened exactly; an integer type is accepted only when every value
of the batch has magnitude at most ``2**53`` (types of at most 32 bits always
are); every other type is refused; a null is refused, never dropped; NaN and an
infinity are refused. No message or attribute carries a data value.

Fingerprints. A fingerprint is ``"sha256:"`` and 64 hexadecimal digits of a
canonical, self-delimiting record, never of a concatenation that could be
ambiguous. Its level says what it covers: ``CONTENT`` hashes the data bytes,
``METADATA`` hashes what the storage reports about the data without reading it
(a weaker guarantee: it detects a replaced or rewritten file, not an in-place
edit that preserves every reported property), and ``NONE`` means the source
cannot be fingerprinted without consuming it. A partition without a fingerprint
is *not resumable*: any later code that resumes from or reuses a stored partition
state must refuse it. A fingerprint is an integrity check against accidental
change, not authentication.

Re-reading. A source is read again by a fit that needs several passes. A source that
can be read only once says so with a ``single_use`` attribute that is ``True`` (see
:func:`is_single_use`); a multi-pass fit refuses it before reading unless a local spool
is requested (:mod:`veridist.scale._spool`).

Parallel reading. A fit given an executor (:mod:`veridist.scale._executors`) reads the
partitions of a pass on several threads or processes. A source takes part only if it says
so: ``parallel_obstacle(kind, partitions)`` returns ``None`` when the partitions can be read
by workers of that :class:`ExecutorKind`, and otherwise the :class:`DowngradeReason` why
they cannot, in which case the pass is read serially and the reason is recorded. A source
without the method is not read in parallel (``UNDECLARED_SOURCE``). Workers of an executor
that uses processes receive ``worker_source()`` if the source has it, else the source
itself, and it must be picklable and must not carry data (a source that holds its data in
memory names ``IN_MEMORY_DATA``). A source that cannot be read again after a failure part
way through a partition says so with ``retryable`` set to ``False``; a failed partition of
every other source may be read again from the start (see :mod:`veridist.scale._tasks`).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Protocol, TypeVar, runtime_checkable

from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._kernels import numpy_module
from veridist.scale._state import MergeableState

DEFAULT_MAX_ROWS: Final = 262144
"""Rows per batch unless the caller asks for fewer (``2**18``)."""

MAX_EXACT_INTEGER: Final = 1 << 53
"""The largest magnitude up to which every integer is exactly a float64."""

VALUE_POLICY_VERSION: Final = 1
"""The version of the value policy; it changes whenever a value is accepted, widened or
refused differently, so that a stored state is never reused across two policies."""

FINGERPRINT_ALGORITHM: Final = "sha256"
HASH_BLOCK_ROWS: Final = 1 << 16
"""Rows hashed at a time when fingerprinting an array, so no copy of it is made."""

_FINGERPRINT_PATTERN: Final = re.compile(r"sha256:[0-9a-f]{64}")
_LENGTH_BYTES: Final = 8


class FingerprintLevel(StrEnum):
    """What a partition fingerprint covers."""

    NONE = "none"
    METADATA = "metadata"
    CONTENT = "content"


@dataclass(frozen=True, slots=True)
class Partition:
    """An immutable description of one independently readable part of a source.

    ``id`` is stable across runs and platforms for the same stored data and is
    unique within its source. ``ordinal`` is the position in the source's
    canonical order. ``rows`` is the exact row count when the source knows it
    without reading. ``fingerprint`` is ``None`` exactly when ``fingerprint_level``
    is ``NONE``; such a partition is not resumable.
    """

    id: str
    ordinal: int
    rows: int | None
    fingerprint: str | None
    fingerprint_level: FingerprintLevel

    def __post_init__(self) -> None:
        if type(self.id) is not str or not self.id:
            raise TypeError("id must be a non-empty string")
        if type(self.ordinal) is not int:
            raise TypeError("ordinal must be a built-in integer")
        if self.ordinal < 0:
            raise ValueError("ordinal must not be negative")
        if self.rows is not None:
            if type(self.rows) is not int:
                raise TypeError("rows must be a built-in integer or None")
            if self.rows < 0:
                raise ValueError("rows must not be negative")
        if not isinstance(self.fingerprint_level, FingerprintLevel):
            raise TypeError("fingerprint_level must be a FingerprintLevel")
        if self.fingerprint_level is FingerprintLevel.NONE:
            if self.fingerprint is not None:
                raise ValueError("a partition without a fingerprint level has no fingerprint")
        elif type(self.fingerprint) is not str or not _FINGERPRINT_PATTERN.fullmatch(
            self.fingerprint
        ):
            raise ValueError("fingerprint must be 'sha256:' and 64 lowercase hex digits")

    @property
    def resumable(self) -> bool:
        """Whether the partition can be recognised again: it has a fingerprint."""

        return self.fingerprint is not None


class ExecutorKind(StrEnum):
    """How the partitions of a pass are read: in turn, on threads or on processes."""

    SERIAL = "serial"
    THREADS = "threads"
    PROCESSES = "processes"


class DowngradeReason(StrEnum):
    """Why a pass that was to run in parallel was read serially; closed, locale neutral."""

    SINGLE_PARTITION = "single_partition"
    """Fewer than two partitions: there is nothing to split."""

    SINGLE_USE_SOURCE = "single_use_source"
    """The source can be read once only, so it cannot be read by several workers."""

    ENGINE_MANAGED = "engine_managed"
    """A dataframe or query engine owns the read and runs its own threads in this process."""

    IN_MEMORY_DATA = "in_memory_data"
    """The data is in this process's memory; handing it to worker processes would copy it."""

    UNDECLARED_SOURCE = "undeclared_source"
    """The source does not say that it can be read by several workers."""

    SPOOL_FIRST_PASS = "spool_first_pass"
    """The first pass writes the spool and does so in canonical order, one partition at a time."""


class BatchReader(Protocol):
    """What reading one partition needs: its validated batches."""

    def batches(
        self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
    ) -> Iterator[Any]: ...


@runtime_checkable
class Source(BatchReader, Protocol):
    """What every source provides; see the module documentation."""

    def partitions(self) -> tuple[Partition, ...]: ...


def is_single_use(source: object) -> bool:
    """Whether ``source`` declares that it can be read only once.

    A source says so with a ``single_use`` attribute that is ``True``. Every other
    source, one without the attribute included, is taken to be readable again.
    """

    return getattr(source, "single_use", False) is True


def parallel_obstacle(
    source: object, kind: ExecutorKind, partitions: Sequence[Partition]
) -> DowngradeReason | None:
    """Why ``source`` cannot be read by workers of ``kind``, or ``None`` if it can.

    A source answers with its ``parallel_obstacle(kind, partitions)`` method; one without
    the method is ``UNDECLARED_SOURCE``.
    """

    method = getattr(source, "parallel_obstacle", None)
    if method is None:
        return DowngradeReason.UNDECLARED_SOURCE
    obstacle: DowngradeReason | None = method(kind, partitions)
    return obstacle


def is_retryable(source: object) -> bool:
    """Whether a partition read that failed part way may be read again from the start.

    A source that says ``retryable`` is ``False`` is not; a single-use source is not either
    (a stream that has been consumed cannot be replayed). Every other source is.
    """

    return getattr(source, "retryable", True) is True and not is_single_use(source)


def check_max_rows(max_rows: object) -> int:
    """Validate a batch bound: a built-in integer of at least one."""

    if type(max_rows) is not int:
        raise TypeError("max_rows must be a built-in integer")
    if max_rows < 1:
        raise ValueError("max_rows must be at least 1")
    return max_rows


def _absorb(digest: Any, item: object) -> None:
    """Feed one item of a canonical record: a type tag, a length, then the payload."""

    if item is None:
        digest.update(b"N")
        return
    if type(item) is int:
        payload, tag = str(item).encode("ascii"), b"I"
    elif type(item) is str:
        payload, tag = item.encode("utf-8"), b"S"
    elif type(item) is bytes:
        payload, tag = item, b"B"
    elif type(item) is tuple:
        digest.update(b"T" + len(item).to_bytes(_LENGTH_BYTES, "big"))
        for member in item:
            _absorb(digest, member)
        return
    else:
        raise TypeError("a fingerprint record holds None, int, str, bytes and tuples only")
    digest.update(tag + len(payload).to_bytes(_LENGTH_BYTES, "big") + payload)


def canonical_digest(*items: object) -> str:
    """``"sha256:"`` and the hex digest of a canonical, unambiguous record of ``items``.

    Each item is ``None``, an ``int`` (not a ``bool``), a ``str``, ``bytes`` or a
    tuple of these. A type tag and a length precede every payload, so two
    different records never encode to the same bytes.
    """

    digest = hashlib.new(FINGERPRINT_ALGORITHM)
    digest.update(b"veridist.scale.fingerprint.v1")
    digest.update(len(items).to_bytes(_LENGTH_BYTES, "big"))
    for item in items:
        _absorb(digest, item)
    return f"{FINGERPRINT_ALGORITHM}:{digest.hexdigest()}"


def _exact_integers(values: Any) -> Any:
    """Widen an integer array to float64 if, and only if, that is exact."""

    np = numpy_module()
    if values.dtype.itemsize > 4 and values.size:
        low, high = int(values.min()), int(values.max())
        if low < -MAX_EXACT_INTEGER or high > MAX_EXACT_INTEGER:
            raise ScaleSourceError(ScaleSourceErrorCode.LOSSY_CAST)
    return values.astype(np.float64)


def float64_values(values: Any) -> Any:
    """Apply the value policy to one numpy array and return finite float64 values.

    ``values`` must be a one-dimensional ``numpy.ndarray`` (the caller has checked
    that). A float64 array is returned as it is when it is already C-contiguous
    and native (no copy), a float32 array is widened exactly, an integer array is
    widened if every value is exactly representable. ``ScaleSourceError``:
    ``UNSUPPORTED_DTYPE`` for any other dtype, ``LOSSY_CAST`` for an integer
    beyond ``2**53``, ``NAN_VALUE`` for a NaN, ``NON_FINITE_VALUE`` for an
    infinity.
    """

    np = numpy_module()
    kind, size = values.dtype.kind, values.dtype.itemsize
    if kind == "f" and size == 8:
        result = np.ascontiguousarray(values, dtype=np.float64)
    elif kind == "f" and size == 4:
        result = values.astype(np.float64)
    elif kind in "iu":
        result = _exact_integers(values)
    else:
        raise ScaleSourceError(ScaleSourceErrorCode.UNSUPPORTED_DTYPE)
    if not np.isfinite(result).all():
        if np.isnan(result).any():
            raise ScaleSourceError(ScaleSourceErrorCode.NAN_VALUE)
        raise ScaleSourceError(ScaleSourceErrorCode.NON_FINITE_VALUE)
    return result


class NumpySource:
    """A source over a one-dimensional numpy array: one partition, or contiguous row ranges.

    The array is read where it is, never copied: batches of a float64 array are
    views, and an array of another supported dtype is widened one batch at a time.
    The caller must not modify the array while it is being read.

    Without ``partition_rows`` there is one partition, ``"array"``, whose fingerprint is at
    level ``CONTENT``: the SHA-256 of the array's dtype, shape and bytes, computed in
    bounded blocks each time ``partitions()`` is called. With ``partition_rows`` (a built-in
    integer of at least one) the array is cut into consecutive ranges of that many rows (the
    last may be shorter), partition ``"array#<n>"`` being the ``n``-th, each fingerprinted
    like the whole array together with its first row. A fit depends on the multiset of the
    values only, so the partitioning changes no state and no result; it only lets an
    executor read the ranges at the same time. The data is in this process's memory, so
    only threads can read the ranges in parallel.
    """

    PARTITION_ID: Final = "array"

    __slots__ = ("_partition_rows", "_values")

    def __init__(self, values: object, *, partition_rows: int | None = None) -> None:
        np = numpy_module()
        if not isinstance(values, np.ndarray) or isinstance(values, np.ma.MaskedArray):
            raise TypeError("values must be a numpy.ndarray")
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        kind, size = values.dtype.kind, values.dtype.itemsize
        if not (kind in "iu" or (kind == "f" and size in (4, 8))):
            raise ScaleSourceError(ScaleSourceErrorCode.UNSUPPORTED_DTYPE)
        if partition_rows is not None:
            if type(partition_rows) is not int:
                raise TypeError("partition_rows must be a built-in integer or None")
            if partition_rows < 1:
                raise ValueError("partition_rows must be at least 1")
        self._values: Any = values
        self._partition_rows = partition_rows

    @staticmethod
    def _content_digest(values: Any) -> str:
        np = numpy_module()
        digest = hashlib.new(FINGERPRINT_ALGORITHM)
        for start in range(0, int(values.shape[0]), HASH_BLOCK_ROWS):
            digest.update(np.ascontiguousarray(values[start : start + HASH_BLOCK_ROWS]))
        return digest.hexdigest()

    def _range_count(self) -> int:
        """How many row ranges the array is cut into (an empty array has one, empty)."""

        step = self._partition_rows
        assert step is not None
        return max(1, -(-int(self._values.shape[0]) // step))

    def partitions(self) -> tuple[Partition, ...]:
        """The partitions, each with a content fingerprint: one, or one per row range."""

        values = self._values
        step = self._partition_rows
        if step is None:
            total = int(values.shape[0])
            fingerprint = canonical_digest(
                "numpy", values.dtype.str, (total,), self._content_digest(values)
            )
            return (
                Partition(self.PARTITION_ID, 0, total, fingerprint, FingerprintLevel.CONTENT),
            )
        found: list[Partition] = []
        for index in range(self._range_count()):
            part = values[index * step : (index + 1) * step]
            rows = int(part.shape[0])
            fingerprint = canonical_digest(
                "numpy", values.dtype.str, (rows,), index * step, self._content_digest(part)
            )
            found.append(
                Partition(
                    f"{self.PARTITION_ID}#{index}",
                    index,
                    rows,
                    fingerprint,
                    FingerprintLevel.CONTENT,
                )
            )
        return tuple(found)

    def parallel_obstacle(
        self, kind: ExecutorKind, partitions: Sequence[Partition]
    ) -> DowngradeReason | None:
        """Threads read the ranges in place; processes would need a copy of the data."""

        if kind is ExecutorKind.PROCESSES:
            return DowngradeReason.IN_MEMORY_DATA
        return None

    def batches(
        self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
    ) -> Iterator[Any]:
        """Validated float64 batches of at most ``max_rows`` rows, in array order."""

        bound = check_max_rows(max_rows)
        if not isinstance(partition, Partition):
            raise ValueError("partition does not belong to this source")
        return self._read(self._range(partition), bound)

    def _range(self, partition: Partition) -> tuple[int, int]:
        """The first row and the end of the rows of ``partition``."""

        total = int(self._values.shape[0])
        step = self._partition_rows
        if step is None:
            if partition.id != self.PARTITION_ID:
                raise ValueError("partition does not belong to this source")
            return 0, total
        name, separator, number = partition.id.rpartition("#")
        known = name == self.PARTITION_ID and separator and number.isascii() and number.isdigit()
        if not known or int(number) >= self._range_count():
            raise ValueError("partition does not belong to this source")
        start = int(number) * step
        return start, min(start + step, total)

    def _read(self, rows: tuple[int, int], max_rows: int) -> Iterator[Any]:
        values = self._values
        first, end = rows
        for start in range(first, end, max_rows):
            yield float64_values(values[start : min(start + max_rows, end)])


StateT = TypeVar("StateT", bound=MergeableState[Any])


def fold_partition(
    source: BatchReader,
    partition: Partition,
    state_type: type[StateT],
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
) -> StateT:
    """The state of one partition: ``state_type.empty()`` updated with every batch of it."""

    state = state_type.empty()
    for batch in source.batches(partition, max_rows=max_rows):
        state = state.update(batch)
    return state


def fold(
    source: Source, state_type: type[StateT], *, max_rows: int = DEFAULT_MAX_ROWS
) -> StateT:
    """The state of a whole source: its partition states merged in canonical order.

    The result depends on the multiset of values only, so it is identical for
    every ``max_rows``, every partition layout and every source of the same
    values. A refused batch raises and nothing is returned, so no partial state
    ever escapes.
    """

    check_max_rows(max_rows)
    state = state_type.empty()
    for partition in source.partitions():
        state = state.merge(fold_partition(source, partition, state_type, max_rows=max_rows))
    return state
