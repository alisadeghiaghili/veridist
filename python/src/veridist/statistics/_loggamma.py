"""Vectorised log-domain regularised incomplete gamma ratios.

``log_regularized_gamma(shape, y, log_y)`` returns ``(log P(shape, y), log Q(shape, y))`` for
one scalar shape and an array of arguments.  The upper ratio ``Q`` is never formed as a
probability and then logged: the prefactor ``exp(-y) y**shape / Gamma(shape)`` is carried as a
logarithm, so a ratio far below the smallest binary64 keeps a finite logarithm.

* ``y < shape + 1``: the power series ``sum y**n / (shape (shape + 1) ... (shape + n))``.  Its
  prefactor is the saddle-point (deviance) form for ``shape >= 8``, where the direct form
  cancels, and ``shape * log(y) - y - lgamma(shape)`` below.  Below shape ``0.01`` the upper
  ratio comes from the small-shape expansion, where ``1 - P`` would cancel.
* ``y >= shape + 1``: the modified Lentz continued fraction for ``Q``.

Every element is iterated on its own and a converged element leaves the working set (lane
compaction), so the result for an element never depends on its neighbours or on its position.
Every loop is bounded by ``GAMMA_MAX_ITERATIONS``; an element that has not converged by then
raises a typed :class:`LogProbabilityError` naming the shape region, never a NaN, and a shape
above ``GAMMA_MAX_SHAPE``, where the series needs more than the bound, is refused before any
work.
"""

from __future__ import annotations

from importlib import import_module
from math import lgamma, log, pi
from typing import Any, Final

from veridist.statistics._logerrors import LogProbabilityError, LogProbabilityErrorCode
from veridist.statistics._logmath import log1mexp
from veridist.statistics.distributions import (
    _SMALL_SHAPE,
    _SMALL_SHAPE_TERMS,
    _lgamma_one_plus_small,
)
from veridist.statistics.log_density import _stirling_error

np: Any = import_module("numpy")

#: Iteration bound of the power series and of the continued fraction (fixed by measurement:
#: the series needs about ``8.4 * sqrt(shape)`` iterations at its worst argument).
GAMMA_MAX_ITERATIONS: Final = 512
#: Shapes above this are refused.  The series needs 509 of its 512 iterations at its worst
#: argument (just below ``shape + 1``) for shape 3900 and 516 for shape 4000.
GAMMA_MAX_SHAPE: Final = 3900.0
#: Shapes from here on use the saddle-point form of the prefactor.
SADDLE_POINT_SHAPE: Final = 8.0
#: Shapes below this use the small-shape expansion for the upper ratio.
SMALL_SHAPE: Final = _SMALL_SHAPE
#: ``|ratio - 1|`` up to which ``log1p(delta) - delta`` is summed as a series.
_NEAR_ONE: Final = 0.03
#: Terms of that series (``delta**n / n`` for ``n = 2 .. _NEAR_ONE_LAST``).
_NEAR_ONE_LAST: Final = 15
#: Relative convergence tolerance and smallest magnitude of the Lentz recurrence.
_EPSILON: Final = 2e-16
_FPMIN: Final = 1e-300
#: From this argument on the continued fraction is ``1 / y`` to binary64 precision (its first
#: correction is ``(1 - shape) / y``), and its reciprocals would be subnormal near ``1e308``.
_HUGE_ARGUMENT: Final = 1e100
_REGION_BOUNDS: Final = (SMALL_SHAPE, SADDLE_POINT_SHAPE, 1000.0, GAMMA_MAX_SHAPE)


def shape_region(shape: float) -> str:
    """Name the range of shapes that ``shape`` lies in (the region a refusal reports)."""

    low, saddle, large, limit = _REGION_BOUNDS
    if shape < low:
        return f"shape below {low:g}"
    if shape < saddle:
        return f"shape {low:g} to {saddle:g}"
    if shape < large:
        return f"shape {saddle:g} to {large:g}"
    if shape <= limit:
        return f"shape {large:g} to {limit:g}"
    return f"shape above {limit:g}"


def _nonconvergence(shape: float) -> LogProbabilityError:
    return LogProbabilityError(LogProbabilityErrorCode.GAMMA_NONCONVERGENCE, shape_region(shape))


def _log1pmx(delta: Any, log_ratio: Any) -> Any:
    """``log1p(delta) - delta`` for ``delta = ratio - 1``, where ``log_ratio = log(ratio)``.

    Near zero a power series (no cancellation); for ``delta <= -0.5`` the form
    ``log(ratio) - ratio + 1`` from the logarithm itself, because ``delta`` rounded to
    binary64 holds no information about a ratio near zero; ``log1p`` between.
    """

    result = np.empty_like(delta)
    near = np.abs(delta) <= _NEAR_ONE
    if near.any():
        base = delta[near]
        power = base * base
        total = np.zeros_like(base)
        for index in range(2, _NEAR_ONE_LAST + 1):
            term = power / index
            total += -term if index % 2 == 0 else term
            power = power * base
        result[near] = total
    far = ~near
    if far.any():
        outer = delta[far]
        with np.errstate(all="ignore"):
            result[far] = np.where(outer > -0.5, np.log1p(outer) - outer, log_ratio[far] - outer)
    return result


def log_prefactor(shape: float, y: Any, log_y: Any) -> Any:
    """``-y + shape * log(y) - lgamma(shape)``; the saddle-point form for ``shape >= 8``.

    ``log_y`` is the logarithm of the mathematical ``y``, which may have underflowed.
    """

    if shape < SADDLE_POINT_SHAPE:
        return shape * log_y - y - lgamma(shape)
    log_shape = log(shape)
    ratio = y / shape
    deviance = _log1pmx(ratio - 1.0, log_y - log_shape)
    return shape * deviance + (0.5 * log(shape / (2.0 * pi)) - _stirling_error(shape))


def _series_sum(shape: float, y: Any, max_iterations: int) -> tuple[Any, int]:
    """``sum_{n>=0} y**n / (shape (shape + 1) ... (shape + n))`` per element, and the count.

    Raises the typed non-convergence when an element has not converged in ``max_iterations``.
    """

    count = y.shape[0]
    result = np.empty(count)
    if count == 0:
        return result, 0
    index = np.arange(count)
    active = y
    term = np.full(count, 1.0 / shape)
    total = term.copy()
    for iteration in range(1, max_iterations + 1):
        term = term * (active / (shape + iteration))
        total = total + term
        done = term <= total * _EPSILON
        if done.any():
            result[index[done]] = total[done]
            keep = ~done
            index, active, term, total = index[keep], active[keep], term[keep], total[keep]
            if index.size == 0:
                return result, iteration
    raise _nonconvergence(shape)


def _fraction_value(shape: float, y: Any, max_iterations: int) -> tuple[Any, int]:
    """``h`` with ``Q = exp(prefactor) * h`` by the modified Lentz fraction, per element."""

    count = y.shape[0]
    result = np.empty(count)
    if count == 0:
        return result, 0
    index = np.arange(count)
    active = y
    b = active + 1.0 - shape
    c = np.full(count, 1.0 / _FPMIN)
    d = 1.0 / b
    h = d.copy()
    for iteration in range(1, max_iterations + 1):
        coefficient = -iteration * (iteration - shape)
        b = b + 2.0
        d = coefficient * d + b
        d = np.where(np.abs(d) < _FPMIN, np.copysign(_FPMIN, d), d)
        c = b + coefficient / c
        c = np.where(np.abs(c) < _FPMIN, np.copysign(_FPMIN, c), c)
        d = 1.0 / d
        delta = d * c
        h = h * delta
        done = np.abs(delta - 1.0) <= _EPSILON
        if done.any():
            result[index[done]] = h[done]
            keep = ~done
            index, b, c, d, h = index[keep], b[keep], c[keep], d[keep], h[keep]
            if index.size == 0:
                return result, iteration
    raise _nonconvergence(shape)


def _sum_three(first: Any, second: Any, third: Any) -> Any:
    """The three-term sum with its rounding errors carried (error-free transformations)."""

    partial = first + second
    virtual = partial - first
    error_one = (first - (partial - virtual)) + (second - virtual)
    total = partial + third
    virtual = total - partial
    error_two = (partial - (total - virtual)) + (third - virtual)
    return total + (error_one + error_two)


def _log_upper_small_shape(shape: float, y: Any, log_y: Any) -> Any:
    """``log Q(shape, y)`` for ``shape < 0.01`` and ``y < shape + 1``.

    With ``g = y**shape / Gamma(shape + 1)`` and
    ``T = shape * sum((-y)**n / (n! (shape + n)))`` the lower ratio is ``g (1 + T)`` and
    ``Q = -(u + T + u T)`` for ``u = g - 1``, the second form from ``expm1`` so that a ``Q``
    far below machine epsilon keeps its relative accuracy.
    """

    exponent = shape * log_y - _lgamma_one_plus_small(shape)
    term = np.ones_like(y)
    total = np.zeros_like(y)
    for order in range(1, _SMALL_SHAPE_TERMS + 1):
        term = term * (-y / order)
        total = total + term / (shape + order)
    correction = shape * total
    lower = np.exp(exponent) * (1.0 + correction)
    unit = np.expm1(exponent)
    upper = -_sum_three(unit, correction, unit * correction)
    return np.where(lower <= 0.5, np.log1p(-lower), np.log(upper))


def log_regularized_gamma(shape: float, y: Any, log_y: Any) -> tuple[Any, Any]:
    """``(log P(shape, y), log Q(shape, y))`` for ``0 < shape <= GAMMA_MAX_SHAPE``.

    ``y`` and ``log_y`` are one-dimensional float64 arrays with ``log_y`` the logarithm of the
    mathematical ``y`` (``y`` itself may have underflowed to zero or be infinite while
    ``log_y`` is finite).  ``log_y = -inf`` is the point zero and ``log_y = +inf`` the point at
    infinity.  The refusals are typed: a shape above ``GAMMA_MAX_SHAPE`` and an element that
    does not converge within ``GAMMA_MAX_ITERATIONS`` iterations.
    """

    if shape > GAMMA_MAX_SHAPE:
        raise LogProbabilityError(LogProbabilityErrorCode.GAMMA_SHAPE_REFUSED, shape_region(shape))
    count = y.shape[0]
    lower = np.empty(count)
    upper = np.empty(count)
    with np.errstate(all="ignore"):
        bottom = log_y == -np.inf
        top = (log_y == np.inf) | ~np.isfinite(y)
        lower[bottom], upper[bottom] = -np.inf, 0.0
        lower[top], upper[top] = 0.0, -np.inf
        inner = ~(bottom | top)
        series = np.flatnonzero(inner & (y < shape + 1.0))
        fraction = np.flatnonzero(inner & ~(y < shape + 1.0))
        if series.size:
            argument, log_argument = y[series], log_y[series]
            total, _ = _series_sum(shape, argument, GAMMA_MAX_ITERATIONS)
            log_lower = np.log(total) + log_prefactor(shape, argument, log_argument)
            lower[series] = log_lower
            if shape < SMALL_SHAPE:
                upper[series] = _log_upper_small_shape(shape, argument, log_argument)
            else:
                upper[series] = log1mexp(log_lower)
        if fraction.size:
            argument, log_argument = y[fraction], log_y[fraction]
            log_value = -log_argument
            regular = np.flatnonzero(argument < _HUGE_ARGUMENT)
            value, _ = _fraction_value(shape, argument[regular], GAMMA_MAX_ITERATIONS)
            log_value[regular] = np.log(value)
            log_upper = log_prefactor(shape, argument, log_argument) + log_value
            upper[fraction] = log_upper
            lower[fraction] = log1mexp(log_upper)
    return lower, upper


__all__ = [
    "GAMMA_MAX_ITERATIONS",
    "GAMMA_MAX_SHAPE",
    "SADDLE_POINT_SHAPE",
    "SMALL_SHAPE",
    "log_prefactor",
    "log_regularized_gamma",
    "shape_region",
]
