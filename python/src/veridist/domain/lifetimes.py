"""Typed, locale-neutral lifetime observations for supported fit verticals.

Four kinds of observation of a non-negative duration are described, each a frozen value:

* :class:`ExactLifetime`: the event was observed at ``time``;
* :class:`RightCensoredLifetime`: the unit was still event-free at ``time``;
* :class:`LeftCensoredLifetime`: the event had already happened by ``time`` (a detection
  limit, or a unit found failed at its first inspection);
* :class:`IntervalCensoredLifetime`: the event happened after ``lower`` and by ``upper``
  (a unit found failed between two inspections).  Equal bounds are refused: an exact
  observation contributes a density and an interval a probability, and the two are not on
  one scale, so the exact type must be used.

Each of them may also carry a truncation window through the keyword-only attributes
``truncation_lower`` and ``truncation_upper`` (``None`` for an open side); see
:mod:`veridist.domain._truncation` for the rules.  The censoring and the truncation are
assumed independent of the lifetime (coarsening at random).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from math import isfinite

from veridist.domain._numeric import is_real
from veridist.domain._truncation import (
    check_exact,
    check_interval,
    check_left,
    check_right,
    closed_observation,
    lifetime_window,
)


def _finite_non_negative_time(
    value: float | Decimal | int, name: str = "time", *, positive: bool = False
) -> float:
    """Return a finite non-negative (or positive) time as ``float``.

    Python or numpy reals, or ``Decimal``.  A positive ``value`` that would round to zero is
    refused, so that "positive" survives the conversion.
    """

    if not (is_real(value) or isinstance(value, Decimal)):
        raise TypeError(f"{name} must be a finite numeric value")
    sign = "positive" if positive else "non-negative"
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite and {sign}") from error
    if not isfinite(numeric) or numeric < 0.0 or (positive and numeric == 0.0 and value <= 0):
        raise ValueError(f"{name} must be finite and {sign}")
    if value > 0 and numeric == 0.0:
        raise ValueError(f"positive {name} must remain representable")
    return numeric


def _set_window(
    observation: (
        ExactLifetime | RightCensoredLifetime | LeftCensoredLifetime | IntervalCensoredLifetime
    ),
) -> tuple[float | None, float | None]:
    """Store the normal form of the window of ``observation`` and return it."""

    low, high = lifetime_window(observation.truncation_lower, observation.truncation_upper)
    object.__setattr__(observation, "truncation_lower", low)
    object.__setattr__(observation, "truncation_upper", high)
    return low, high


@closed_observation
@dataclass(frozen=True, slots=True, repr=False)
class ExactLifetime:
    """An observed event time for a supported lifetime model."""

    time: float | Decimal | int
    truncation_lower: float | Decimal | int | None = field(default=None, kw_only=True)
    truncation_upper: float | Decimal | int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _finite_non_negative_time(self.time))
        low, high = _set_window(self)
        check_exact(float(self.time), low, high)


@closed_observation
@dataclass(frozen=True, slots=True, repr=False)
class RightCensoredLifetime:
    """A right-censoring time under the declared independent-censoring assumption."""

    time: float | Decimal | int
    truncation_lower: float | Decimal | int | None = field(default=None, kw_only=True)
    truncation_upper: float | Decimal | int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _finite_non_negative_time(self.time))
        low, high = _set_window(self)
        check_right(float(self.time), low, high)


@closed_observation
@dataclass(frozen=True, slots=True, repr=False)
class LeftCensoredLifetime:
    """A left-censoring time: the event happened at or before ``time``, which is positive.

    An event at or before time 0 has probability zero under every positive-support family, so
    ``time`` must be strictly positive.
    """

    time: float | Decimal | int
    truncation_lower: float | Decimal | int | None = field(default=None, kw_only=True)
    truncation_upper: float | Decimal | int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _finite_non_negative_time(self.time, positive=True))
        low, high = _set_window(self)
        check_left(float(self.time), low, high)


@closed_observation
@dataclass(frozen=True, slots=True, repr=False)
class IntervalCensoredLifetime:
    """An interval-censored lifetime: the event happened in ``(lower, upper]``.

    Needs ``0 <= lower < upper``, both finite.  Equal bounds are refused: use
    :class:`ExactLifetime` for an event observed at a time.  An open bound is not an interval
    but a :class:`LeftCensoredLifetime` or :class:`RightCensoredLifetime`.
    """

    lower: float | Decimal | int
    upper: float | Decimal | int
    truncation_lower: float | Decimal | int | None = field(default=None, kw_only=True)
    truncation_upper: float | Decimal | int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        lower = _finite_non_negative_time(self.lower, "lower")
        upper = _finite_non_negative_time(self.upper, "upper", positive=True)
        if lower == upper:
            raise ValueError(
                "lower and upper must differ: an interval with equal bounds is an exact "
                "observation, use ExactLifetime"
            )
        if not lower < upper:
            raise ValueError("upper must be greater than lower")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        low, high = _set_window(self)
        check_interval(lower, upper, low, high)


LifetimeObservation = ExactLifetime | RightCensoredLifetime
"""The observation kinds the fits take (exact and right-censored, without a window)."""

AnyLifetimeObservation = (
    ExactLifetime | RightCensoredLifetime | LeftCensoredLifetime | IntervalCensoredLifetime
)
"""Every lifetime observation kind: what the log-likelihood entry points take."""


__all__ = [
    "AnyLifetimeObservation",
    "ExactLifetime",
    "IntervalCensoredLifetime",
    "LeftCensoredLifetime",
    "LifetimeObservation",
    "RightCensoredLifetime",
]
