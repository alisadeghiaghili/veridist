"""Exact, order-independent summation of binary64 values in units of 2**-1074.

:class:`ExactSum` holds the exact sum of every finite float64 it has received,
whatever the order, the batching or the way partial sums were merged. The
canonical value is the integer ``total_units()``: the sum in units of
``2**-1074`` (the smallest subnormal), the same unit and the same integer that
:class:`veridist.statistics.log_likelihood.LogLikelihoodState` keeps in
``total_units`` for the same terms.

Layout. A finite float64 is ``+-M * 2**(E - 1075)`` with an integer mantissa
``M < 2**53`` and a biased exponent ``E`` (subnormals and zero use ``E = 1``).
The exponent is the bin: 2048 bins, of which 1 to 2046 are used. ``M`` is split
into ``high = M >> 27`` (signed) and ``low = M & (2**27 - 1)``, so
``M == high * 2**27 + low``. For one block of at most ``2**26`` values
``numpy.bincount`` with ``high`` and with ``low`` as weights sums integers whose
every partial sum stays below ``2**53``, so it is exact in float64. Block sums
are added into two int64 lanes per bin. Every operation on the lanes is integer
addition, so the result cannot depend on order or grouping.

Overflow bounds, enforced by counting (never by inspecting the lanes):

* ``unnormalised`` counts value-equivalents whose ``low`` part has not been
  carried into ``high``. A lane ``low`` entry is at most
  ``unnormalised * (2**27 - 1)``. Before a block or a merge would push the count
  above ``carry_interval`` (at most ``2**36 - 1``, default ``2**34``) the lanes
  are carry-normalised: ``high += low >> 27`` and ``low &= 2**27 - 1``, which
  resets the count to one residual value-equivalent (two after a merge, which
  normalises both operands' lanes).
* ``lane_load`` counts value-equivalents added to the lanes since they were last
  emptied. A ``high`` entry is at most ``lane_load * (2**26 + 1)`` (each value
  contributes at most ``2**26`` and its ``low`` part carries less than one
  unit). With ``lane_capacity`` at most ``2**36`` (default ``2**36``) it stays
  below ``2**63``. When a block or a merge would exceed the capacity the lanes
  are folded into a Python integer and emptied, so the total is unbounded; the
  only cap is the unsigned-64 observation count of the log-likelihood state.

An update is atomic: a rejected batch leaves the accumulator unchanged. The
class is a plain mutable accumulator and is not thread-safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Final, Self

from veridist.scale import _codec
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._kernels import float64_batch, numpy_module
from veridist.statistics.log_likelihood import MAX_OBSERVATION_COUNT

BIN_COUNT: Final = 2048
"""Lane bins per sign-split part; the biased exponent field has 2047 finite values."""

LOW_BITS: Final = 27
"""Width of the low part of a 53-bit mantissa; the high part has at most 26 bits."""

UNITS_EXPONENT: Final = 1074
"""The canonical unit is ``2**-UNITS_EXPONENT``."""

MAX_ELEMENT_UNITS: Final = ((1 << 53) - 1) << 2045
"""The largest absolute value of one finite float64 in canonical units."""

DEFAULT_BLOCK_SIZE: Final = 1 << 15
MAX_BLOCK_SIZE: Final = 1 << 26
DEFAULT_CARRY_INTERVAL: Final = 1 << 34
MAX_CARRY_INTERVAL: Final = (1 << 36) - 1
DEFAULT_LANE_CAPACITY: Final = 1 << 36
MAX_LANE_CAPACITY: Final = 1 << 36

_LOW_MASK: Final = (1 << LOW_BITS) - 1
_FRACTION_MASK: Final = (1 << 52) - 1
_EXPONENT_MASK: Final = 0x7FF
_NON_FINITE_EXPONENT: Final = 0x7FF
_UNITS_DENOMINATOR: Final = 1 << UNITS_EXPONENT

STATE_TAG: Final = "scale.exact_sum"
SCHEMA_VERSION: Final = 1


@dataclass(frozen=True, slots=True)
class AccumulatorLimits:
    """The block, carry and lane bounds of one accumulator; defaults are the shipped ones.

    Tests (and nothing else) lower them to reach the carry and fold paths with
    small inputs. Two accumulators merge only when their limits are equal.
    """

    block_size: int = DEFAULT_BLOCK_SIZE
    carry_interval: int = DEFAULT_CARRY_INTERVAL
    lane_capacity: int = DEFAULT_LANE_CAPACITY

    def __post_init__(self) -> None:
        for value in (self.block_size, self.carry_interval, self.lane_capacity):
            if type(value) is not int:
                raise TypeError("accumulator limits must be built-in integers")
        if not 1 <= self.block_size <= MAX_BLOCK_SIZE:
            raise ValueError("block_size must be between 1 and 2**26")
        if not self.block_size < self.carry_interval <= MAX_CARRY_INTERVAL:
            raise ValueError("carry_interval must exceed block_size and be below 2**36")
        if not self.block_size <= self.lane_capacity <= MAX_LANE_CAPACITY:
            raise ValueError("lane_capacity must be between block_size and 2**36")


DEFAULT_LIMITS: Final = AccumulatorLimits()


def units_to_float(units: int) -> float:
    """Round ``units * 2**-1074`` to the nearest binary64 (ties to even), once.

    Python's integer true division is correctly rounded. A result that is not
    finite after rounding is ``TOTAL_NOT_REPRESENTABLE``.
    """

    try:
        return units / _UNITS_DENOMINATOR
    except OverflowError as error:
        raise ScaleStateError(ScaleStateErrorCode.TOTAL_NOT_REPRESENTABLE) from error


def units_to_fraction(units: int) -> Fraction:
    """The exact value of ``units * 2**-1074``."""

    return Fraction(units, _UNITS_DENOMINATOR)


class ExactSum:
    """The exact sum of finite float64 values, mergeable and canonically serializable."""

    __slots__ = ("_count", "_high", "_lane_load", "_limits", "_low", "_spill", "_unnormalised")

    def __init__(self, limits: AccumulatorLimits = DEFAULT_LIMITS) -> None:
        if type(limits) is not AccumulatorLimits:
            raise TypeError("limits must be AccumulatorLimits")
        np = numpy_module()
        self._limits = limits
        self._high: Any = np.zeros(BIN_COUNT, dtype=np.int64)
        self._low: Any = np.zeros(BIN_COUNT, dtype=np.int64)
        self._spill = 0
        self._count = 0
        self._lane_load = 0
        self._unnormalised = 0

    @property
    def count(self) -> int:
        """The number of values accumulated, at most ``2**64 - 1``."""

        return self._count

    @property
    def limits(self) -> AccumulatorLimits:
        return self._limits

    def copy(self) -> Self:
        """An independent accumulator holding the same state."""

        clone = type(self)(self._limits)
        clone._high = self._high.copy()
        clone._low = self._low.copy()
        clone._spill = self._spill
        clone._count = self._count
        clone._lane_load = self._lane_load
        clone._unnormalised = self._unnormalised
        return clone

    # -- accumulation -----------------------------------------------------

    def update(self, values: object) -> None:
        """Add every value of a one-dimensional float64 array exactly.

        ``TypeError`` or ``ValueError`` for a wrong container, dtype or shape;
        ``ScaleStateError`` with ``NON_FINITE_VALUE`` for NaN or an infinity and
        with ``OBSERVATION_LIMIT_EXCEEDED`` when the count would pass
        ``2**64 - 1``. A refused batch leaves the accumulator unchanged.
        """

        array = float64_batch(values)
        size = int(array.shape[0])
        if self._count > MAX_OBSERVATION_COUNT - size:
            raise ScaleStateError(ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        scratch = type(self)(self._limits)
        block = self._limits.block_size
        for start in range(0, size, block):
            scratch._add_block(array[start : start + block])
        self._absorb(scratch)

    def _add_block(self, block: Any) -> None:
        """Add one contiguous block of at most ``block_size`` values to the lanes."""

        np = numpy_module()
        size = int(block.shape[0])
        if self._lane_load + size > self._limits.lane_capacity:
            self._fold()
        if self._unnormalised + size > self._limits.carry_interval:
            self._carry()
        bits = block.view(np.int64)
        exponent = (bits >> 52) & _EXPONENT_MASK
        if int(exponent.max()) == _NON_FINITE_EXPONENT:
            raise ScaleStateError(ScaleStateErrorCode.NON_FINITE_VALUE)
        mantissa = bits & _FRACTION_MASK
        mantissa |= (exponent != 0).astype(np.int64) << 52
        np.maximum(exponent, 1, out=exponent)
        mantissa = np.where(bits < 0, -mantissa, mantissa)
        index = exponent.astype(np.intp, copy=False)
        high = (mantissa >> LOW_BITS).astype(np.float64)
        low = (mantissa & _LOW_MASK).astype(np.float64)
        self._high += np.bincount(index, weights=high, minlength=BIN_COUNT).astype(np.int64)
        self._low += np.bincount(index, weights=low, minlength=BIN_COUNT).astype(np.int64)
        self._count += size
        self._lane_load += size
        self._unnormalised += size

    def _carry(self) -> None:
        """Carry-normalise the lanes; the value is unchanged."""

        self._high += self._low >> LOW_BITS
        self._low &= _LOW_MASK
        self._unnormalised = 1

    def _lane_units(self) -> int:
        """The exact value held in the lanes, in canonical units."""

        np = numpy_module()
        bins = np.flatnonzero(self._high | self._low)
        total = 0
        for position, high, low in zip(
            bins.tolist(), self._high[bins].tolist(), self._low[bins].tolist(), strict=True
        ):
            total += ((high << LOW_BITS) + low) << (position - 1)
        return total

    def _fold(self) -> None:
        """Move the lane content into the Python-integer spill and empty the lanes."""

        self._spill += self._lane_units()
        self._high[:] = 0
        self._low[:] = 0
        self._lane_load = 0
        self._unnormalised = 0

    # -- merge ------------------------------------------------------------

    def merge(self, other: ExactSum) -> Self:
        """A new accumulator holding the exact sum of both; neither operand changes.

        Associative and commutative; the empty accumulator is the identity.
        ``TypeError`` unless ``other`` is an ``ExactSum``; ``ValueError`` when the
        limits differ; ``OBSERVATION_LIMIT_EXCEEDED`` when the combined count would
        pass ``2**64 - 1``.
        """

        if not isinstance(other, ExactSum):
            raise TypeError("other must be an ExactSum")
        if other._limits != self._limits:
            raise ValueError("accumulators with different limits cannot be merged")
        if self._count > MAX_OBSERVATION_COUNT - other._count:
            raise ScaleStateError(ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
        result = self.copy()
        result._absorb(other)
        return result

    def _absorb(self, other: ExactSum) -> None:
        """Add ``other`` into ``self`` with the lane bounds kept by counting."""

        self._count += other._count
        self._spill += other._spill
        if other._lane_load == 0:
            return
        if self._lane_load + other._lane_load > self._limits.lane_capacity:
            self._fold()
        if self._unnormalised + other._unnormalised <= self._limits.carry_interval:
            self._high += other._high
            self._low += other._low
            self._unnormalised += other._unnormalised
        else:
            self._carry()
            self._high += other._high + (other._low >> LOW_BITS)
            self._low += other._low & _LOW_MASK
            self._unnormalised += 1
        self._lane_load += other._lane_load

    # -- results ----------------------------------------------------------

    def total_units(self) -> int:
        """The exact sum in units of ``2**-1074`` as an unbounded Python integer."""

        return self._spill + self._lane_units()

    def exact_total(self) -> Fraction:
        """The exact sum as a ``Fraction``."""

        return units_to_fraction(self.total_units())

    def total(self) -> float:
        """The exact sum rounded once to binary64 (nearest, ties to even).

        An empty accumulator or one whose terms cancel exactly gives ``0.0``
        (never ``-0.0``). A total that rounds beyond the largest finite value is
        ``ScaleStateError`` with ``TOTAL_NOT_REPRESENTABLE``; the accumulator stays
        valid and keeps its exact total.
        """

        return units_to_float(self.total_units())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExactSum):
            return NotImplemented
        return self._count == other._count and self.total_units() == other.total_units()

    # -- canonical serialization -------------------------------------------

    def to_bytes(self) -> bytes:
        """Canonical bytes: equal states give equal bytes whatever their history."""

        return _codec.encode_frame(STATE_TAG, SCHEMA_VERSION, self._count, (self.total_units(),))

    @classmethod
    def from_bytes(cls, data: object, limits: AccumulatorLimits = DEFAULT_LIMITS) -> Self:
        """Restore an accumulator from :meth:`to_bytes` output, refusing anything else.

        ``TypeError`` for a non-bytes argument, otherwise ``ScaleStateError`` with
        ``STATE_BYTES_INVALID``, ``STATE_TAG_MISMATCH`` or ``STATE_VERSION_UNSUPPORTED``.
        """

        count, integers = _codec.decode_frame(
            data, tag=STATE_TAG, version=SCHEMA_VERSION, integer_count=1
        )
        units = integers[0]
        if abs(units) > count * MAX_ELEMENT_UNITS:
            raise ScaleStateError(ScaleStateErrorCode.STATE_BYTES_INVALID)
        restored = cls(limits)
        restored._count = count
        restored._spill = units
        return restored
