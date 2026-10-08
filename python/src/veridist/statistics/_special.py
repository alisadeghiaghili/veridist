"""Digamma and trigamma, needed for the gamma family's observed information.

Everything here is pure Python (``math`` only), deterministic, and bounded: each
recurrence runs a fixed number of steps.
"""

from __future__ import annotations

from math import fsum, log
from typing import Final

#: The recurrence ``psi(x) = psi(x + 1) - 1 / x`` is applied until ``x`` reaches this
#: value, from where the asymptotic series is accurate to well below binary64
#: rounding (its first omitted term is about ``1e-24`` relative at ``x = 20``).
_ASYMPTOTIC_FROM: Final = 20.0

# Bernoulli-number coefficients of the asymptotic expansions, B_{2k} / (2k).
_DIGAMMA_COEFFICIENTS: Final = (
    1.0 / 12.0,
    -1.0 / 120.0,
    1.0 / 252.0,
    -1.0 / 240.0,
    1.0 / 132.0,
    -691.0 / 32760.0,
    1.0 / 12.0,
)
# Bernoulli numbers B_{2k} for the trigamma expansion: 1/x + 1/(2x^2) + sum B_2k / x^(2k+1).
_TRIGAMMA_COEFFICIENTS: Final = (
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
)


def _shifted(x: float) -> tuple[float, tuple[float, ...]]:
    """Return ``(x + m, (x, x + 1, ..., x + m - 1))`` with ``x + m >= _ASYMPTOTIC_FROM``."""

    steps = max(0, int(_ASYMPTOTIC_FROM - x) + 1)
    return x + steps, tuple(x + index for index in range(steps))


def digamma(x: float) -> float:
    """Return the digamma function ``psi(x) = d/dx log Gamma(x)`` for ``x > 0``.

    Evaluated by the recurrence up to ``x >= 20`` followed by the asymptotic
    series; the relative error is at the level of binary64 rounding.
    """

    if not x > 0.0:
        raise ValueError("digamma is defined here only for x > 0")
    top, below = _shifted(x)
    inverse_square = 1.0 / (top * top)
    series = 0.0
    power = inverse_square
    for coefficient in _DIGAMMA_COEFFICIENTS:
        series += coefficient * power
        power *= inverse_square
    tail = log(top) - 0.5 / top - series
    return fsum((tail, *(-1.0 / point for point in below)))


def trigamma(x: float) -> float:
    """Return the trigamma function ``psi'(x) = d^2/dx^2 log Gamma(x)`` for ``x > 0``.

    Evaluated by the recurrence ``psi'(x) = psi'(x + 1) + 1 / x^2`` up to
    ``x >= 20`` followed by the asymptotic series
    ``1/x + 1/(2 x^2) + sum B_2k / x^(2k+1)``; the relative error is at the level
    of binary64 rounding.
    """

    if not x > 0.0:
        raise ValueError("trigamma is defined here only for x > 0")
    top, below = _shifted(x)
    inverse = 1.0 / top
    inverse_square = inverse * inverse
    series = 0.0
    power = inverse * inverse_square
    for coefficient in _TRIGAMMA_COEFFICIENTS:
        series += coefficient * power
        power *= inverse_square
    tail = inverse + 0.5 * inverse_square + series
    return fsum((tail, *(1.0 / (point * point) for point in below)))


__all__ = ["digamma", "trigamma"]
