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
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
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


@runtime_checkable
class Source(Protocol):
    """What every source provides; see the module documentation."""

    def partitions(self) -> tuple[Partition, ...]: ...

    def batches(
        self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
    ) -> Iterator[Any]: ...


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
    """A single-partition source over a one-dimensional numpy array.

    The array is read where it is, never copied: batches of a float64 array are
    views, and an array of another supported dtype is widened one batch at a time.
    The caller must not modify the array while it is being read. The partition
    fingerprint is at level ``CONTENT``: the SHA-256 of the array's dtype, shape
    and bytes, computed in bounded blocks each time ``partitions()`` is called.
    """

    PARTITION_ID: Final = "array"

    __slots__ = ("_values",)

    def __init__(self, values: object) -> None:
        np = numpy_module()
        if not isinstance(values, np.ndarray) or isinstance(values, np.ma.MaskedArray):
            raise TypeError("values must be a numpy.ndarray")
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        kind, size = values.dtype.kind, values.dtype.itemsize
        if not (kind in "iu" or (kind == "f" and size in (4, 8))):
            raise ScaleSourceError(ScaleSourceErrorCode.UNSUPPORTED_DTYPE)
        self._values: Any = values

    def _content_digest(self) -> str:
        np = numpy_module()
        values = self._values
        digest = hashlib.new(FINGERPRINT_ALGORITHM)
        for start in range(0, int(values.shape[0]), HASH_BLOCK_ROWS):
            digest.update(np.ascontiguousarray(values[start : start + HASH_BLOCK_ROWS]))
        return digest.hexdigest()

    def partitions(self) -> tuple[Partition, ...]:
        """The one partition, with a content fingerprint."""

        values = self._values
        fingerprint = canonical_digest(
            "numpy", values.dtype.str, (int(values.shape[0]),), self._content_digest()
        )
        return (
            Partition(
                self.PARTITION_ID,
                0,
                int(values.shape[0]),
                fingerprint,
                FingerprintLevel.CONTENT,
            ),
        )

    def batches(
        self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
    ) -> Iterator[Any]:
        """Validated float64 batches of at most ``max_rows`` rows, in array order."""

        bound = check_max_rows(max_rows)
        if not isinstance(partition, Partition) or partition.id != self.PARTITION_ID:
            raise ValueError("partition does not belong to this source")
        return self._read(bound)

    def _read(self, max_rows: int) -> Iterator[Any]:
        values = self._values
        for start in range(0, int(values.shape[0]), max_rows):
            yield float64_values(values[start : start + max_rows])


StateT = TypeVar("StateT", bound=MergeableState[Any])


def fold_partition(
    source: Source,
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
