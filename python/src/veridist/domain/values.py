"""Typed real-valued observations for families on the whole real line.

The lifetime types in :mod:`veridist.domain.lifetimes` describe a non-negative
duration and belong to the positive-support families (exponential, Weibull,
lognormal, gamma).  The types here describe a finite real measurement and
belong to the families whose support is the whole real line (normal, right
Gumbel).  The two groups are deliberately not interchangeable: every fit
accepts exactly the types valid for its family and raises ``TypeError``
for the other.

There are four kinds, mirroring the lifetime ones: :class:`ExactValue`,
:class:`RightCensoredValue`, :class:`LeftCensoredValue` (the value is at most ``value``,
for example a detection limit) and :class:`IntervalCensoredValue` (the value is in
``(lower, upper]``; equal bounds are refused, use :class:`ExactValue`).  Each may carry a
truncation window through the keyword-only attributes ``truncation_lower`` and
``truncation_upper`` (``None`` for an open side); see :mod:`veridist.domain._truncation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from math import isfinite
from numbers import Real

from veridist.domain._truncation import (
    check_exact,
    check_interval,
    check_left,
    check_right,
    closed_observation,
    value_window,
)


def _finite_real(value: object, name: str = "value") -> float:
    if isinstance(value, bool) or not isinstance(value, Real | Decimal):
        raise TypeError(f"{name} must be a finite real number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be finite") from error
    if not isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _set_window(
    observation: ExactValue | RightCensoredValue | LeftCensoredValue | IntervalCensoredValue,
) -> tuple[float | None, float | None]:
    """Store the normal form of the window of ``observation`` and return it."""

    low, high = value_window(observation.truncation_lower, observation.truncation_upper)
    object.__setattr__(observation, "truncation_lower", low)
    object.__setattr__(observation, "truncation_upper", high)
    return low, high


@closed_observation
@dataclass(frozen=True, slots=True, repr=False)
class ExactValue:
    """An exactly observed finite real value (the real-line analogue of an event time)."""

    value: float
    truncation_lower: float | Decimal | int | None = field(default=None, kw_only=True)
    truncation_upper: float | Decimal | int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite_real(self.value))
        low, high = _set_window(self)
        check_exact(self.value, low, high)


@closed_observation
@dataclass(frozen=True, slots=True, repr=False)
class RightCensoredValue:
    """A right-censored finite real value: the true value is known to exceed ``value``.

    The censoring is assumed independent of the measured quantity, as for
    :class:`~veridist.domain.lifetimes.RightCensoredLifetime`.
    """

    value: float
    truncation_lower: float | Decimal | int | None = field(default=None, kw_only=True)
    truncation_upper: float | Decimal | int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite_real(self.value))
        low, high = _set_window(self)
        check_right(self.value, low, high)


@closed_observation
@dataclass(frozen=True, slots=True, repr=False)
class LeftCensoredValue:
    """A left-censored finite real value: the true value is known to be at most ``value``."""

    value: float
    truncation_lower: float | Decimal | int | None = field(default=None, kw_only=True)
    truncation_upper: float | Decimal | int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite_real(self.value))
        low, high = _set_window(self)
        check_left(self.value, low, high)


@closed_observation
@dataclass(frozen=True, slots=True, repr=False)
class IntervalCensoredValue:
    """An interval-censored finite real value: the true value is in ``(lower, upper]``.

    Needs ``lower < upper``, both finite.  Equal bounds are refused: use :class:`ExactValue`
    for a value observed exactly.  An open bound is not an interval but a
    :class:`LeftCensoredValue` or :class:`RightCensoredValue`.
    """

    lower: float
    upper: float
    truncation_lower: float | Decimal | int | None = field(default=None, kw_only=True)
    truncation_upper: float | Decimal | int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        lower = _finite_real(self.lower, "lower")
        upper = _finite_real(self.upper, "upper")
        if lower == upper:
            raise ValueError(
                "lower and upper must differ: an interval with equal bounds is an exact "
                "observation, use ExactValue"
            )
        if not lower < upper:
            raise ValueError("upper must be greater than lower")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        low, high = _set_window(self)
        check_interval(lower, upper, low, high)


RealObservation = ExactValue | RightCensoredValue
"""The observation kinds the fits take (exact and right-censored, without a window)."""

AnyRealObservation = ExactValue | RightCensoredValue | LeftCensoredValue | IntervalCensoredValue
"""Every real-value observation kind: what the log-likelihood entry points take."""


__all__ = [
    "AnyRealObservation",
    "ExactValue",
    "IntervalCensoredValue",
    "LeftCensoredValue",
    "RealObservation",
    "RightCensoredValue",
]
