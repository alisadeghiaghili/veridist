"""The first-pass states of the multi-pass families: count, one exact sum and the extremes.

A multi-pass fit reads its source once to learn what the later passes are centred on, and
that first read is a mergeable state like any other (:mod:`veridist.scale._state`). It
holds the number of observations, the exact sum of one per-element term and the smallest
and largest observation, and it is merged exactly, so it depends on the multiset of the
observations only: not on their order, their batching or their partitioning. Its canonical
bytes and digest are the state record of the fit.

* :class:`WeibullPilotState` sums ``numpy.log(x)`` (the same logarithm the later passes
  use). Observations that are not strictly positive are not an error of the state: they
  contribute nothing to the sum, and the minimum tells the fit that the support is
  invalid, so the source is read once and the row count is exact.
* :class:`GumbelPilotState` sums ``x``.

The extremes are those of the observations themselves, with ``-0.0`` stored as ``0.0``,
so that they have one canonical encoding. The sum is in units of ``2**-1074``.
"""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import copysign, isfinite
from typing import Any, ClassVar, Final, Self

from veridist.scale._accumulator import MAX_ELEMENT_UNITS, units_to_fraction
from veridist.scale._codec import decode_frame, encode_frame
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._kernels import float64_batch, logarithms, numpy_module, require_finite
from veridist.scale._one_pass import ExactStatistic
from veridist.scale._state import batch_units
from veridist.statistics.log_likelihood import MAX_OBSERVATION_COUNT

_INTEGER_COUNT: Final = 3
_SIGNED_64: Final = 1 << 63


@dataclass(frozen=True, slots=True)
class PilotStatistics:
    """Count, exact sum of the term and, for a non-empty state, the extremes.

    The exact sum is authoritative; its rounded value saturates to an infinity when the sum
    is beyond the binary64 range (the sum of values near the largest finite one).
    """

    count: int
    total: ExactStatistic
    minimum: float | None
    maximum: float | None


def _bits(value: float | None) -> int:
    """The signed 64-bit pattern of a float (zero for the absent extreme of an empty state)."""

    if value is None:
        return 0
    return int(struct.unpack("<q", struct.pack("<d", value))[0])


def _from_bits(bits: int) -> float:
    return float(struct.unpack("<d", struct.pack("<q", bits))[0])


@dataclass(frozen=True, slots=True)
class PilotState(ABC):
    """The count, the exact sum of one term and the extremes of a multiset of observations."""

    STATE_TAG: ClassVar[str]
    SCHEMA_VERSION: ClassVar[int] = 1

    count: int
    total_units: int
    minimum: float | None
    maximum: float | None

    def __post_init__(self) -> None:
        if type(self.count) is not int or type(self.total_units) is not int:
            raise TypeError("count and total must be built-in integers")
        if not 0 <= self.count <= MAX_OBSERVATION_COUNT:
            raise ValueError("count is outside the unsigned-64 observation bound")
        if abs(self.total_units) > self.count * MAX_ELEMENT_UNITS:
            raise ValueError("the sum exceeds the bound implied by the count")
        if self.count == 0:
            if self.total_units != 0 or self.minimum is not None or self.maximum is not None:
                raise ValueError("an empty state has no sum and no extremes")
            return
        if type(self.minimum) is not float or type(self.maximum) is not float:
            raise TypeError("the extremes of a non-empty state must be floats")
        if not (isfinite(self.minimum) and isfinite(self.maximum)):
            raise ValueError("the extremes must be finite")
        if self.minimum > self.maximum:
            raise ValueError("the minimum must not exceed the maximum")
        for extreme in (self.minimum, self.maximum):
            if extreme == 0.0 and copysign(1.0, extreme) < 0.0:
                raise ValueError("a zero extreme is stored as positive zero")

    @staticmethod
    @abstractmethod
    def _term(values: Any) -> Any:
        """The per-element term whose exact sum the state keeps, for a finite batch."""

    @classmethod
    def empty(cls) -> Self:
        """The identity state."""

        return cls(0, 0, None, None)

    def update(self, batch: object) -> Self:
        """A new state with one more batch of float64 observations.

        ``TypeError`` or ``ValueError`` for a wrong container, dtype or shape;
        ``ScaleStateError`` for a non-finite value or a count beyond ``2**64 - 1``.
        """

        values = float64_batch(batch)
        size = int(values.shape[0])
        if self.count > MAX_OBSERVATION_COUNT - size:
            raise ScaleStateError(ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        if size == 0:
            return self
        require_finite(values)
        return type(self)._combined(
            self,
            type(self)(
                size,
                batch_units(self._term(values)),
                float(values.min()) + 0.0,
                float(values.max()) + 0.0,
            ),
        )

    def merge(self, other: Self) -> Self:
        """A new state combining two compatible states exactly."""

        if not isinstance(other, PilotState):
            raise TypeError("other must be a mergeable state")
        if other.STATE_TAG != self.STATE_TAG or other.SCHEMA_VERSION != self.SCHEMA_VERSION:
            raise ScaleStateError(ScaleStateErrorCode.INCOMPATIBLE_STATE)
        if self.count > MAX_OBSERVATION_COUNT - other.count:
            raise ScaleStateError(ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        return type(self)._combined(self, other)

    @classmethod
    def _combined(cls, left: PilotState, right: PilotState) -> Self:
        if left.count == 0:
            return cls(right.count, right.total_units, right.minimum, right.maximum)
        if right.count == 0:
            return cls(left.count, left.total_units, left.minimum, left.maximum)
        assert left.minimum is not None and right.minimum is not None
        assert left.maximum is not None and right.maximum is not None
        return cls(
            left.count + right.count,
            left.total_units + right.total_units,
            min(left.minimum, right.minimum),
            max(left.maximum, right.maximum),
        )

    def finalize(self) -> PilotStatistics:
        """The statistics of the state."""

        return PilotStatistics(
            self.count,
            ExactStatistic.saturating(units_to_fraction(self.total_units)),
            self.minimum,
            self.maximum,
        )

    def to_bytes(self) -> bytes:
        """Canonical bytes: tag, schema version, count, the exact sum and the two extremes."""

        return encode_frame(
            self.STATE_TAG,
            self.SCHEMA_VERSION,
            self.count,
            (self.total_units, _bits(self.minimum), _bits(self.maximum)),
        )

    @classmethod
    def from_bytes(cls, data: object) -> Self:
        """Restore a state of this type from :meth:`to_bytes` output.

        ``TypeError`` for a non-bytes argument; otherwise ``ScaleStateError`` with
        ``STATE_TAG_MISMATCH``, ``STATE_VERSION_UNSUPPORTED`` or ``STATE_BYTES_INVALID``.
        """

        count, integers = decode_frame(
            data, tag=cls.STATE_TAG, version=cls.SCHEMA_VERSION, integer_count=_INTEGER_COUNT
        )
        total, low, high = integers
        try:
            if count == 0:
                if low or high:
                    raise ValueError("an empty state has no extremes")
                return cls(count, total, None, None)
            if not (-_SIGNED_64 <= low < _SIGNED_64 and -_SIGNED_64 <= high < _SIGNED_64):
                raise ValueError("an extreme is not a 64-bit pattern")
            return cls(count, total, _from_bits(low), _from_bits(high))
        except (TypeError, ValueError) as error:
            raise ScaleStateError(ScaleStateErrorCode.STATE_BYTES_INVALID) from error


@dataclass(frozen=True, slots=True)
class WeibullPilotState(PilotState):
    """Count, ``sum log x`` and the extremes of ``x``; a value that is not positive adds 0."""

    STATE_TAG: ClassVar[str] = "scale.weibull_pilot"

    @staticmethod
    def _term(values: Any) -> Any:
        np = numpy_module()
        if (values > 0.0).all():
            return logarithms(values)
        return np.zeros(values.shape[0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class GumbelPilotState(PilotState):
    """Count, ``sum x`` and the extremes of ``x``."""

    STATE_TAG: ClassVar[str] = "scale.gumbel_pilot"

    @staticmethod
    def _term(values: Any) -> Any:
        return values


__all__ = ["GumbelPilotState", "PilotState", "PilotStatistics", "WeibullPilotState"]
