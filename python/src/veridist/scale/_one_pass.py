"""Exact sufficient-statistic states of the one-pass families on exact data.

Each state holds the observation count and exact integer sums of per-element terms
(:mod:`veridist.scale._state`). ``finalize()`` returns the sufficient statistics as
exact ``Fraction`` values with their correctly rounded binary64 values; nothing is
fitted here.

Term formulas, one per sum, shared by every code path:

* exponential: ``t`` (times must be finite and non-negative);
* normal: ``x`` and ``x * x``;
* gamma: ``x`` and ``numpy.log(x)`` (``x`` finite and strictly positive);
* lognormal: ``log x`` and ``log x * log x`` with ``log x = numpy.log(x)``
  (``x`` finite and strictly positive).

Sums of values and of logarithms are exact sums of binary64 terms in units of
``2**-1074``. Sums of squares are exact too: the true square ``x * x`` is summed, not
the rounded product, in units of ``2**-2148`` (:mod:`veridist.scale._squares`), for
every finite ``x`` including magnitudes whose square overflows or underflows binary64.
So the centred sum of squares ``sum x*x - (sum x)**2 / n`` of a state is exact; it is
zero exactly when all values (all logarithms, for the lognormal) are equal. The
logarithm is the platform ``numpy.log``: it is exact in the sums but not correctly
rounded per element, so a gamma or lognormal state is bit-identical only within one
numeric environment.

Schema versions. The normal and lognormal states are schema 2; schema 1 summed the
rounded product ``fl(x * x)`` in units of ``2**-1074``. Bytes of the old schema are
refused with ``STATE_VERSION_UNSUPPORTED`` and never read as the new one.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import inf
from typing import Any, ClassVar

from veridist.scale._accumulator import MAX_ELEMENT_UNITS, units_to_fraction
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._kernels import (
    logarithms,
    require_non_negative,
    require_positive,
)
from veridist.scale._squares import MAX_SQUARE_UNITS, SQUARE_UNITS_EXPONENT, squares_units
from veridist.scale._state import ExactSumsState, batch_units


@dataclass(frozen=True, slots=True)
class ExactStatistic:
    """An exact rational statistic and its once-rounded binary64 value."""

    exact: Fraction
    rounded: float

    @classmethod
    def of(cls, exact: Fraction) -> ExactStatistic:
        """Round once (nearest, ties to even); a value beyond the finite range is refused."""

        try:
            return cls(exact, float(exact))
        except OverflowError as error:
            raise ScaleStateError(ScaleStateErrorCode.TOTAL_NOT_REPRESENTABLE) from error

    @classmethod
    def saturating(cls, exact: Fraction) -> ExactStatistic:
        """Round once like :meth:`of`, but a value beyond the finite range rounds to an infinity.

        Used for the sums of squares, which exceed the binary64 range for values above
        about ``1e154`` although the exact value stays authoritative and usable.
        """

        try:
            return cls(exact, float(exact))
        except OverflowError:
            return cls(exact, inf if exact > 0 else -inf)


@dataclass(frozen=True, slots=True)
class ExponentialStatistics:
    """Observation count and total time ``sum t``."""

    count: int
    total_time: ExactStatistic


@dataclass(frozen=True, slots=True)
class NormalStatistics:
    """Count, ``sum x``, ``sum x*x``, and (for a non-empty state) the mean and the
    centered sum of squares ``sum x*x - (sum x)**2 / n``, exact (the squares are not rounded)."""

    count: int
    sum_x: ExactStatistic
    sum_x_squared: ExactStatistic
    mean: ExactStatistic | None
    centered_sum_of_squares: ExactStatistic | None


@dataclass(frozen=True, slots=True)
class GammaStatistics:
    """Count, ``sum x``, ``sum log x`` and (for a non-empty state) their means."""

    count: int
    sum_x: ExactStatistic
    sum_log_x: ExactStatistic
    mean_x: ExactStatistic | None
    mean_log_x: ExactStatistic | None


@dataclass(frozen=True, slots=True)
class LognormalStatistics:
    """Count, ``sum log x``, ``sum (log x)**2``, and (for a non-empty state) the mean
    of ``log x`` and the exact centered sum of squares of ``log x``."""

    count: int
    sum_log_x: ExactStatistic
    sum_log_x_squared: ExactStatistic
    mean_log_x: ExactStatistic | None
    centered_sum_of_squares_log_x: ExactStatistic | None


def _mean(total: Fraction, count: int) -> ExactStatistic | None:
    return ExactStatistic.of(total / count) if count else None


def _centered(total: Fraction, total_of_squares: Fraction, count: int) -> ExactStatistic | None:
    if not count:
        return None
    return ExactStatistic.saturating(total_of_squares - total * total / count)


def _squares_to_fraction(units: int) -> Fraction:
    return Fraction(units, 1 << SQUARE_UNITS_EXPONENT)


@dataclass(frozen=True, slots=True)
class ExponentialState(ExactSumsState[ExponentialStatistics]):
    """Exact state for the exponential family: ``n`` and ``sum t``."""

    STATE_TAG: ClassVar[str] = "scale.exponential"
    SCHEMA_VERSION: ClassVar[int] = 1
    SUM_COUNT: ClassVar[int] = 1

    @staticmethod
    def _batch_sums(values: Any) -> tuple[int, ...]:
        require_non_negative(values)
        return (batch_units(values),)

    def finalize(self) -> ExponentialStatistics:
        return ExponentialStatistics(self.count, ExactStatistic.of(units_to_fraction(self.sums[0])))


@dataclass(frozen=True, slots=True)
class NormalState(ExactSumsState[NormalStatistics]):
    """Exact state for the normal family: ``n``, ``sum x`` and ``sum x*x``."""

    STATE_TAG: ClassVar[str] = "scale.normal"
    SCHEMA_VERSION: ClassVar[int] = 2
    SUM_COUNT: ClassVar[int] = 2

    @classmethod
    def element_bounds(cls) -> tuple[int, ...]:
        return (MAX_ELEMENT_UNITS, MAX_SQUARE_UNITS)

    @staticmethod
    def _batch_sums(values: Any) -> tuple[int, ...]:
        return (batch_units(values), squares_units(values))

    def finalize(self) -> NormalStatistics:
        total = units_to_fraction(self.sums[0])
        total_of_squares = _squares_to_fraction(self.sums[1])
        return NormalStatistics(
            self.count,
            ExactStatistic.of(total),
            ExactStatistic.saturating(total_of_squares),
            _mean(total, self.count),
            _centered(total, total_of_squares, self.count),
        )


@dataclass(frozen=True, slots=True)
class GammaState(ExactSumsState[GammaStatistics]):
    """Exact state for the gamma family: ``n``, ``sum x`` and ``sum log x``."""

    STATE_TAG: ClassVar[str] = "scale.gamma"
    SCHEMA_VERSION: ClassVar[int] = 1
    SUM_COUNT: ClassVar[int] = 2

    @staticmethod
    def _batch_sums(values: Any) -> tuple[int, ...]:
        require_positive(values)
        return (batch_units(values), batch_units(logarithms(values)))

    def finalize(self) -> GammaStatistics:
        total = units_to_fraction(self.sums[0])
        total_of_logs = units_to_fraction(self.sums[1])
        return GammaStatistics(
            self.count,
            ExactStatistic.of(total),
            ExactStatistic.of(total_of_logs),
            _mean(total, self.count),
            _mean(total_of_logs, self.count),
        )


@dataclass(frozen=True, slots=True)
class LognormalState(ExactSumsState[LognormalStatistics]):
    """Exact state for the lognormal family: ``n``, ``sum log x`` and ``sum (log x)**2``."""

    STATE_TAG: ClassVar[str] = "scale.lognormal"
    SCHEMA_VERSION: ClassVar[int] = 2
    SUM_COUNT: ClassVar[int] = 2

    @classmethod
    def element_bounds(cls) -> tuple[int, ...]:
        return (MAX_ELEMENT_UNITS, MAX_SQUARE_UNITS)

    @staticmethod
    def _batch_sums(values: Any) -> tuple[int, ...]:
        require_positive(values)
        logs = logarithms(values)
        return (batch_units(logs), squares_units(logs))

    def finalize(self) -> LognormalStatistics:
        total = units_to_fraction(self.sums[0])
        total_of_squares = _squares_to_fraction(self.sums[1])
        return LognormalStatistics(
            self.count,
            ExactStatistic.of(total),
            ExactStatistic.saturating(total_of_squares),
            _mean(total, self.count),
            _centered(total, total_of_squares, self.count),
        )
