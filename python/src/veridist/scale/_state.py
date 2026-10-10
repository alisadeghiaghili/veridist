"""The mergeable-state contract and its exact-sum base implementation.

A *state* is an immutable value that summarises a multiset of observations.
Updating with a batch or merging with a compatible state returns a new state;
the result depends only on the multiset of observations, never on their order,
their batching or the merge tree. Equal states have equal canonical bytes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, Protocol, Self, TypeVar, runtime_checkable

from veridist.scale._accumulator import MAX_ELEMENT_UNITS, ExactSum
from veridist.scale._codec import decode_frame, encode_frame
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._kernels import float64_batch, require_finite
from veridist.statistics.log_likelihood import MAX_OBSERVATION_COUNT

ResultT_co = TypeVar("ResultT_co", covariant=True)


@runtime_checkable
class MergeableState(Protocol[ResultT_co]):
    """What every state type provides.

    * ``empty()`` is the identity: ``merge`` with it returns an equal state.
    * ``update(batch)`` adds one batch; ``merge(other)`` combines two states.
      Both are pure, and ``merge`` is associative and commutative. Updating with
      a concatenation of batches equals merging the updates of any split of them.
    * ``merge`` refuses a state of another type or schema version with
      ``INCOMPATIBLE_STATE``; the count is capped at ``2**64 - 1`` and exceeding
      it is ``OBSERVATION_LIMIT_EXCEEDED``, never a wraparound.
    * ``finalize()`` returns the sufficient statistics derived from the state.
    * ``to_bytes()`` is canonical and versioned; ``from_bytes`` restores a state
      of the same type and refuses unknown versions without migrating them.
    """

    STATE_TAG: ClassVar[str]
    SCHEMA_VERSION: ClassVar[int]

    @property
    def count(self) -> int: ...

    @classmethod
    def empty(cls) -> Self: ...

    def update(self, batch: object) -> Self: ...

    def merge(self, other: Self) -> Self: ...

    def finalize(self) -> ResultT_co: ...

    def to_bytes(self) -> bytes: ...

    @classmethod
    def from_bytes(cls, data: object) -> Self: ...


def batch_units(term: Any) -> int:
    """The exact sum of one derived term array in canonical units of ``2**-1074``."""

    accumulator = ExactSum()
    accumulator.update(term)
    return accumulator.total_units()


@dataclass(frozen=True, slots=True)
class ExactSumsState(ABC, Generic[ResultT_co]):
    """An observation count and one exact sum per term, in units of ``2**-1074``.

    A subclass names its tag, schema version and number of sums, and supplies
    :meth:`_terms` (the per-element float64 term arrays of a validated batch) and
    :meth:`finalize`. Every sum is the exact sum of binary64 terms, so a state
    depends on the multiset of its terms only.
    """

    STATE_TAG: ClassVar[str]
    SCHEMA_VERSION: ClassVar[int]
    SUM_COUNT: ClassVar[int]

    count: int
    sums: tuple[int, ...] = field(repr=False)

    def __post_init__(self) -> None:
        if type(self.count) is not int:
            raise TypeError("count must be a built-in integer")
        if not 0 <= self.count <= MAX_OBSERVATION_COUNT:
            raise ValueError("count is outside the unsigned-64 observation bound")
        if type(self.sums) is not tuple or len(self.sums) != self.SUM_COUNT:
            raise ValueError("sums must be a tuple with one entry per sum")
        bound = self.count * MAX_ELEMENT_UNITS
        for value in self.sums:
            if type(value) is not int:
                raise TypeError("each sum must be a built-in integer")
            if abs(value) > bound:
                raise ValueError("a sum exceeds the bound implied by the count")

    @classmethod
    def empty(cls) -> Self:
        """The identity state: no observation and every sum zero."""

        return cls(0, (0,) * cls.SUM_COUNT)

    @staticmethod
    @abstractmethod
    def _terms(values: Any) -> tuple[Any, ...]:
        """The float64 term arrays of a finite batch, one per sum, after domain checks."""

    @abstractmethod
    def finalize(self) -> ResultT_co:
        """The sufficient statistics of the state."""

    def update(self, batch: object) -> Self:
        """A new state with one more batch of float64 observations.

        ``TypeError`` or ``ValueError`` for a wrong container, dtype or shape;
        ``ScaleStateError`` for a non-finite value, a value outside the family's
        support, a non-finite derived term or a count beyond ``2**64 - 1``. A
        refused batch leaves nothing changed (states are immutable).
        """

        values = float64_batch(batch)
        size = int(values.shape[0])
        if self.count > MAX_OBSERVATION_COUNT - size:
            raise ScaleStateError(ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        require_finite(values)
        terms = self._terms(values)
        sums = tuple(
            total + batch_units(term) for total, term in zip(self.sums, terms, strict=True)
        )
        return type(self)(self.count + size, sums)

    def merge(self, other: Self) -> Self:
        """A new state combining two compatible states exactly."""

        if not isinstance(other, ExactSumsState):
            raise TypeError("other must be a mergeable state")
        if other.STATE_TAG != self.STATE_TAG or other.SCHEMA_VERSION != self.SCHEMA_VERSION:
            raise ScaleStateError(ScaleStateErrorCode.INCOMPATIBLE_STATE)
        if self.count > MAX_OBSERVATION_COUNT - other.count:
            raise ScaleStateError(ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        sums = tuple(left + right for left, right in zip(self.sums, other.sums, strict=True))
        return type(self)(self.count + other.count, sums)

    def to_bytes(self) -> bytes:
        """Canonical bytes: tag, schema version, count and each exact sum."""

        return encode_frame(self.STATE_TAG, self.SCHEMA_VERSION, self.count, self.sums)

    @classmethod
    def from_bytes(cls, data: object) -> Self:
        """Restore a state of this type from :meth:`to_bytes` output.

        ``TypeError`` for a non-bytes argument; otherwise ``ScaleStateError`` with
        ``STATE_TAG_MISMATCH`` (another state type), ``STATE_VERSION_UNSUPPORTED``
        or ``STATE_BYTES_INVALID`` (truncated, altered, non-canonical or out of
        range).
        """

        count, sums = decode_frame(
            data, tag=cls.STATE_TAG, version=cls.SCHEMA_VERSION, integer_count=cls.SUM_COUNT
        )
        try:
            return cls(count, sums)
        except (TypeError, ValueError) as error:
            raise ScaleStateError(ScaleStateErrorCode.STATE_BYTES_INVALID) from error
