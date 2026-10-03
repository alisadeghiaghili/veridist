"""Typed real-valued observations for families on the whole real line.

The lifetime types in :mod:`veridist.domain.lifetimes` describe a non-negative
duration and belong to the positive-support families (exponential, Weibull,
lognormal, gamma).  The types here describe a finite real measurement and
belong to the families whose support is the whole real line (normal, right
Gumbel).  The two pairs are deliberately not interchangeable: every fit
accepts exactly the pair that is valid for its family and raises ``TypeError``
for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import isfinite
from numbers import Real


def _finite_real(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real | Decimal):
        raise TypeError("value must be a finite real number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise ValueError("value must be finite") from error
    if not isfinite(numeric):
        raise ValueError("value must be finite")
    return numeric


@dataclass(frozen=True, slots=True)
class ExactValue:
    """An exactly observed finite real value (the real-line analogue of an event time)."""

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite_real(self.value))


@dataclass(frozen=True, slots=True)
class RightCensoredValue:
    """A right-censored finite real value: the true value is known to exceed ``value``.

    The censoring is assumed independent of the measured quantity, as for
    :class:`~veridist.domain.lifetimes.RightCensoredLifetime`.
    """

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite_real(self.value))


RealObservation = ExactValue | RightCensoredValue


__all__ = ["ExactValue", "RealObservation", "RightCensoredValue"]
