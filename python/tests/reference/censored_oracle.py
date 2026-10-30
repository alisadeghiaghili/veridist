"""Independent 50-digit mpmath oracle for the log-likelihood contribution of one observation.

Used only by tests.  Every binary64 input is the exact ``mpmath`` number of the same value, the
working precision is 50 decimal digits and the contribution is formed entirely at that
precision before it is rounded to binary64:

* exact ``x``: ``log f(x)``;
* censored: ``log(F(beta) - F(alpha))`` for the censoring set ``(alpha, beta]`` read inside the
  window, with ``F(-inf) = 0`` and ``F(+inf) = 1`` (and ``F(x) = 0`` at or below zero for a
  lifetime);
* minus ``log(F(b) - F(a))`` when the observation has a truncation window ``(a, b]``.

Each cumulative value comes from its own closed form (``expm1`` for the hazard families, the
regularized incomplete gamma, ``erfc``), and the probability of an interval is the difference
of the better-conditioned pair of tail values, so nothing is lost at 50 digits except where a
probability is smaller than ``1e-50`` times its tail values (``None`` is returned there).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mpmath

from tests.reference import log_probability_oracle as cdf_oracle

DIGITS = 50
INF = math.inf


def _density(family: str, parameters: tuple[float, ...]) -> Callable[[mpmath.mpf], mpmath.mpf]:
    p = [mpmath.mpf(value) for value in parameters]
    if family == "exponential":
        return lambda x: mpmath.log(p[0]) - p[0] * x
    if family == "weibull_min":
        return lambda x: (
            mpmath.log(p[0] / p[1]) + (p[0] - 1) * mpmath.log(x / p[1]) - (x / p[1]) ** p[0]
        )
    if family == "gumbel_right":
        return lambda x: -mpmath.log(p[1]) - (x - p[0]) / p[1] - mpmath.exp(-(x - p[0]) / p[1])
    if family == "normal":
        return lambda x: (
            -((x - p[0]) ** 2) / (2 * p[1] ** 2) - mpmath.log(p[1]) - mpmath.log(2 * mpmath.pi) / 2
        )
    if family == "lognormal":
        return lambda x: (
            -((mpmath.log(x) - p[0]) ** 2) / (2 * p[1] ** 2)
            - mpmath.log(x)
            - mpmath.log(p[1])
            - mpmath.log(2 * mpmath.pi) / 2
        )
    return lambda x: (
        (p[0] - 1) * mpmath.log(x / p[1]) - x / p[1] - mpmath.loggamma(p[0]) - mpmath.log(p[1])
    )


def log_probability(
    family: str, parameters: tuple[float, ...], low: float, high: float
) -> mpmath.mpf | None:
    """``log(F(high) - F(low))`` at working precision (``None`` if it cancels to nothing)."""

    positive = family not in ("normal", "gumbel_right")
    cumulative = cdf_oracle._cumulative(family, parameters)

    def values(x: float) -> tuple[mpmath.mpf, mpmath.mpf]:
        if x == -INF or (positive and x <= 0.0):
            return mpmath.mpf(0), mpmath.mpf(1)
        if x == INF:
            return mpmath.mpf(1), mpmath.mpf(0)
        return cumulative(mpmath.mpf(x))

    at_low, survival_low = values(low)
    at_high, survival_high = values(high)
    if at_high <= mpmath.mpf(1) / 2:
        probability = at_high - at_low
    else:
        probability = survival_low - survival_high
    if probability <= 0:
        return None
    return mpmath.log(probability)


def contribution(
    family: str,
    parameters: tuple[float, ...],
    kind: str,
    point: float,
    upper: float | None,
    window: tuple[float, float] | None,
) -> float | None:
    """The contribution of one observation, rounded to binary64 (``None`` if it cancels)."""

    with mpmath.workdps(DIGITS):
        if kind == "exact":
            numerator = _density(family, parameters)(mpmath.mpf(point))
        else:
            low, high = {
                "right": (point, INF),
                "left": (-INF, point),
                "interval": (point, upper),
            }[kind]
            if window is not None:
                low, high = max(low, window[0]), min(high, window[1])
            numerator = log_probability(family, parameters, low, high)
            if numerator is None:
                return None
        if window is not None:
            denominator = log_probability(family, parameters, window[0], window[1])
            if denominator is None:
                return None
            numerator = numerator - denominator
        return float(numerator)


def terms(
    family: str,
    parameters: tuple[float, ...],
    kind: str,
    point: float,
    upper: float | None,
    window: tuple[float, float] | None,
) -> tuple[float, float] | None:
    """``(log numerator, log denominator)`` rounded to binary64, the denominator ``0`` untruncated.

    Only for the magnitudes of the two terms (the accuracy bound scales with them).
    """

    with mpmath.workdps(DIGITS):
        if kind == "exact":
            numerator = _density(family, parameters)(mpmath.mpf(point))
        else:
            low, high = {
                "right": (point, INF),
                "left": (-INF, point),
                "interval": (point, upper),
            }[kind]
            if window is not None:
                low, high = max(low, window[0]), min(high, window[1])
            found = log_probability(family, parameters, low, high)
            if found is None:
                return None
            numerator = found
        denominator = mpmath.mpf(0)
        if window is not None:
            found = log_probability(family, parameters, window[0], window[1])
            if found is None:
                return None
            denominator = found
        return float(numerator), float(denominator)
