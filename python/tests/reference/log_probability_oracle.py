"""Independent 50-digit mpmath oracle for ``log F``, ``log S`` and window log-probabilities.

Used only by tests.  Every binary64 input is the exact ``mpmath`` number of the same value
(``mpmath.mpf(float)`` is exact), the working precision is 50 decimal digits, and each tail
probability is formed with full relative accuracy from its own closed form, so a window is the
difference of two such values taken in the better-conditioned tail.  The results are rounded
to binary64; a result below the smallest binary64 logarithm range is ``-inf``.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import mpmath

DIGITS = 50

_Cumulative = Callable[[mpmath.mpf], tuple[mpmath.mpf, mpmath.mpf]]

#: Canonical parameter names, in the order the oracle takes them.
PARAMETER_NAMES = {
    "exponential": ("rate",),
    "weibull_min": ("shape", "scale"),
    "gumbel_right": ("location", "scale"),
    "normal": ("mu", "sigma"),
    "lognormal": ("mu_log", "sigma_log"),
    "gamma": ("shape", "scale"),
}


def _cumulative(family: str, parameters: tuple[float, ...]) -> _Cumulative:
    """Return ``x -> (F(x), S(x))`` at the current working precision."""

    p = [mpmath.mpf(value) for value in parameters]
    if family == "exponential":

        def exponential(x: mpmath.mpf) -> tuple[mpmath.mpf, mpmath.mpf]:
            t = p[0] * x
            return -mpmath.expm1(-t), mpmath.exp(-t)

        return exponential
    if family == "weibull_min":

        def weibull(x: mpmath.mpf) -> tuple[mpmath.mpf, mpmath.mpf]:
            t = (x / p[1]) ** p[0]
            return -mpmath.expm1(-t), mpmath.exp(-t)

        return weibull
    if family == "gumbel_right":

        def gumbel(x: mpmath.mpf) -> tuple[mpmath.mpf, mpmath.mpf]:
            u = mpmath.exp(-(x - p[0]) / p[1])
            return mpmath.exp(-u), -mpmath.expm1(-u)

        return gumbel
    if family in ("normal", "lognormal"):

        def normal(x: mpmath.mpf) -> tuple[mpmath.mpf, mpmath.mpf]:
            z = ((mpmath.log(x) if family == "lognormal" else x) - p[0]) / p[1]
            root = mpmath.sqrt(2)
            return mpmath.erfc(-z / root) / 2, mpmath.erfc(z / root) / 2

        return normal

    def gamma(x: mpmath.mpf) -> tuple[mpmath.mpf, mpmath.mpf]:
        y = x / p[1]
        return (
            mpmath.gammainc(p[0], 0, y, regularized=True),
            mpmath.gammainc(p[0], y, mpmath.inf, regularized=True),
        )

    return gamma


def _log(value: mpmath.mpf) -> float:
    return float(mpmath.log(value)) if value > 0 else -math.inf


def log_cdf(family: str, x: float, *parameters: float) -> float:
    """``log F(x)`` rounded to binary64."""

    with mpmath.workdps(DIGITS):
        return _log(_cumulative(family, parameters)(mpmath.mpf(x))[0])


def log_sf(family: str, x: float, *parameters: float) -> float:
    """``log S(x)`` rounded to binary64."""

    with mpmath.workdps(DIGITS):
        return _log(_cumulative(family, parameters)(mpmath.mpf(x))[1])


def log_window(family: str, lower: float, upper: float, *parameters: float) -> float:
    """``log(F(upper) - F(lower))`` rounded to binary64, from the better-conditioned tail."""

    with mpmath.workdps(DIGITS):
        cumulative = _cumulative(family, parameters)
        at_lower, survival_lower = cumulative(mpmath.mpf(lower))
        at_upper, survival_upper = cumulative(mpmath.mpf(upper))
        if at_upper <= mpmath.mpf(1) / 2:
            return _log(at_upper - at_lower)
        return _log(survival_lower - survival_upper)


def log_gamma_prefactor(shape: float, y: float) -> float:
    """``-y + shape * log(y) - lgamma(shape)`` rounded to binary64."""

    with mpmath.workdps(DIGITS):
        a, point = mpmath.mpf(shape), mpmath.mpf(y)
        return float(a * mpmath.log(point) - point - mpmath.loggamma(a))


def scaled_error(computed: float, expected: float) -> float:
    """``|computed - expected| / max(1, |expected|)``; infinite when only one is infinite."""

    if math.isinf(expected) or math.isinf(computed):
        return 0.0 if computed == expected else math.inf
    return abs(computed - expected) / max(1.0, abs(expected))
