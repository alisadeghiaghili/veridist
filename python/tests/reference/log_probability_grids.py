"""Seeded adversarial grids for the log-probability kernels.

Each family has parameter sets from benign to extreme.  Points are drawn on the family's
standardised coordinate ``z`` (the coordinate its density is smooth in) from deep in the lower
tail through the median to deep in the upper tail, then mapped to ``x``; windows are drawn in
four regimes (lower tail, upper tail, across the median, narrow anywhere) with widths from
``1e-12`` to ``40`` standardised units (``1e-3`` for the narrow regime).
"""

from __future__ import annotations

import math
from typing import Any

import mpmath
import numpy as np

from tests.reference import log_probability_oracle as oracle

CASES: dict[str, tuple[tuple[float, ...], ...]] = {
    "exponential": ((1.0,), (1e-3,), (1e3,), (3.7e-9,), (1e-300,), (1e300,)),
    "weibull_min": (
        (0.05, 1.0),
        (0.3, 10.0),
        (1.7, 1000.0),
        (5.0, 2.0),
        (20.0, 3e-3),
        (50.0, 1.0),
        (0.3, 1e-300),
    ),
    "gumbel_right": ((0.0, 1.0), (50.0, 8.0), (-1e3, 250.0), (3.0, 1e-3), (1e6, 0.5)),
    "normal": ((0.0, 1.0), (100.0, 15.0), (-3.5, 1e-3), (1e6, 250.0)),
    "lognormal": ((0.0, 1.0), (7.0, 0.8), (-2.0, 3.0), (3.0, 0.05)),
    "gamma": (
        (0.003, 1.0),
        (0.0099, 3.0),
        (0.01, 1.0),
        (0.05, 2.0),
        (0.5, 1.0),
        (1.0, 1.0),
        (2.5, 10.0),
        (7.9, 1.0),
        (8.0, 1.0),
        (10.0, 0.1),
        (100.0, 1.0),
        (1000.0, 3.0),
        (3900.0, 1.0),
        (30.0, 1e-8),
        (100.0, 1e300),
        (10.0, 1e-300),
    ),
}
REGIMES = ("lower", "upper", "central", "narrow")
_LOG_LOG_TWO = math.log(math.log(2.0))
_MEDIANS: dict[tuple[str, tuple[float, ...]], float] = {}


def median_z(family: str, parameters: tuple[float, ...]) -> float:
    """The median in the family's standardised coordinate."""

    if family in ("normal", "lognormal"):
        return 0.0
    if family == "gumbel_right":
        return -_LOG_LOG_TWO
    if family in ("weibull_min", "exponential"):
        return _LOG_LOG_TWO
    key = (family, parameters)
    if key not in _MEDIANS:
        shape = parameters[0]
        with mpmath.workdps(30):

            def residual(log_y: Any) -> Any:
                value = mpmath.gammainc(shape, 0, mpmath.exp(log_y), regularized=True)
                return mpmath.log(value) - mpmath.log(0.5)

            _MEDIANS[key] = float(mpmath.findroot(residual, math.log(max(shape, 1e-3))))
    return _MEDIANS[key]


def z_to_x(family: str, parameters: tuple[float, ...], z: Any) -> Any:
    z = np.asarray(z, dtype=float)
    with np.errstate(all="ignore"):
        if family in ("normal", "gumbel_right"):
            return parameters[0] + parameters[1] * z
        if family == "lognormal":
            return np.exp(parameters[0] + parameters[1] * z)
        if family == "weibull_min":
            return parameters[1] * np.exp(z / parameters[0])
        if family == "exponential":
            return np.exp(z) / parameters[0]
        return parameters[1] * np.exp(z)


def z_ranges(
    family: str, parameters: tuple[float, ...]
) -> tuple[tuple[float, float], tuple[float, float]]:
    """``((lower tail z range), (upper tail z range))``, deep tails included."""

    median = median_z(family, parameters)
    if family in ("normal", "lognormal"):
        return (-40.0, median), (median, 40.0)
    if family == "gumbel_right":
        return (-6.55, median), (median, 700.0)
    if family in ("weibull_min", "exponential"):
        return (-700.0, median), (median, 6.55)
    shape = parameters[0]
    top = math.log(shape + 60.0 * math.sqrt(shape) + 100.0)
    return (min(math.log(1e-30), median - 40.0), median), (median, top)


def _valid(family: str, x: Any) -> Any:
    positive = family not in ("normal", "gumbel_right")
    return np.isfinite(x) & (x > 0 if positive else True) & (np.abs(x) < 1.7e308)


def single_points(
    family: str, parameters: tuple[float, ...], rng: np.random.Generator, count: int
) -> Any:
    (low, _), (_, high) = z_ranges(family, parameters)
    median = median_z(family, parameters)
    third = count // 3
    z = np.concatenate(
        [
            np.linspace(low, high, third),
            rng.uniform(low, high, third),
            median + rng.normal(0.0, 0.6, count - 2 * third),
        ]
    )
    x = z_to_x(family, parameters, z)
    return x[_valid(family, x)]


def window_points(
    family: str,
    parameters: tuple[float, ...],
    regime: str,
    rng: np.random.Generator,
    count: int,
) -> tuple[Any, Any]:
    """Windows ``x1 < x2`` in ``regime``, widths log-uniform in ``[1e-12, 40]`` (narrow: 1e-3)."""

    (low, _), (_, high) = z_ranges(family, parameters)
    median = median_z(family, parameters)
    widest = 1e-3 if regime == "narrow" else 40.0
    width = 10.0 ** rng.uniform(-12.0, math.log10(widest), count)
    if regime == "lower":
        z2 = rng.uniform(low, median, count)
        z1 = z2 - width
    elif regime == "upper":
        z1 = rng.uniform(median, high, count)
        z2 = z1 + width
    elif regime == "central":
        share = rng.uniform(0.0, 1.0, count)
        z1 = median - share * width
        z2 = median + (1.0 - share) * width
    else:
        centre = rng.uniform(low, high, count)
        z1, z2 = centre - width / 2.0, centre + width / 2.0
    if family == "gamma":  # widths are scaled by the local dispersion of ``log y``
        shape = parameters[0]
        centre = 0.5 * (z1 + z2)
        half = 0.5 * (z2 - z1) / max(math.sqrt(shape), 1.0)
        z1, z2 = centre - half, centre + half
    x1, x2 = z_to_x(family, parameters, z1), z_to_x(family, parameters, z2)
    keep = _valid(family, x1) & _valid(family, x2) & (x1 < x2)
    return x1[keep], x2[keep]


def reference_window(
    family: str, parameters: tuple[float, ...], lower: float, upper: float
) -> float | None:
    """The oracle window value, or ``None`` where it cancels to nothing at 50 digits."""

    try:
        return oracle.log_window(family, lower, upper, *parameters)
    except (ValueError, ZeroDivisionError, ArithmeticError):
        return None
