"""Vectorised log-probability kernels for the six families.

``log_cdf``, ``log_sf`` and ``log_window`` take the family, numpy arrays of points and the
canonical parameters as keywords (scalars), and return float64 arrays of the broadcast shape.
Nothing here forms a probability and then takes its logarithm:

* ``log_cdf(x) = log F(x)`` and ``log_sf(x) = log S(x)`` are accurate in both tails;
* ``log_window(a, b) = log(F(b) - F(a))`` for ``a < b`` (either end may be infinite) uses the
  stable form of the side the window lies on: ``log S(a)`` for an open upper end, ``log F(b)``
  for an open lower end, ``log F(b) + log1mexp(log F(a) - log F(b))`` in the lower tail,
  ``log S(a) + log1mexp(log S(b) - log S(a))`` in the upper tail and
  ``log1p(-(F(a) + S(b)))`` across the median.  Where the probability complement of the form
  (the share of the larger tail value the window carries, or the probability itself across the
  median) is below ``0.02`` the window is integrated instead by an 8-node Gauss-Legendre
  quadrature of the density in the log domain: in ``t = log x`` for the positive-support
  families and in ``x`` for the location families.

Every element is computed by one formula, independent of the other elements: its result does
not depend on its position in the batch, the batch length or the memory layout of the input.
The exponential, Weibull-minimum and right-Gumbel kernels are numpy-native; the normal and
lognormal ``log Phi`` maps the scalar ``math.erfc`` and ``math.log`` over the array, so it
equals the in-memory scalar kernels bit for bit; the gamma kernel is
:mod:`veridist.statistics._loggamma`.

A result of ``-inf`` means a probability of exactly zero (outside the support) or a logarithm
below the smallest binary64.  Programmer errors in the parameters raise ``TypeError`` or
``ValueError`` as the registry validation does; a request the kernels cannot serve raises a
typed :class:`~veridist.statistics._logerrors.LogProbabilityError`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from importlib import import_module
from math import erfc, exp, lgamma, log, log1p, pi, sqrt
from typing import Any, Final

from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.statistics._logerrors import LogProbabilityError, LogProbabilityErrorCode
from veridist.statistics._loggamma import (
    GAMMA_MAX_SHAPE,
    log_prefactor,
    log_regularized_gamma,
    shape_region,
)
from veridist.statistics._logmath import log1mexp, log_ratio

np: Any = import_module("numpy")

#: A window is integrated instead of subtracted where the complement of its stable form is
#: below this share (fixed by measurement against 50-digit references).
NARROW_SWITCH: Final = 0.02
#: Positive nodes and weights of the 8-point Gauss-Legendre rule on ``[-1, 1]`` (the rule is
#: symmetric); they are the roots of the Legendre polynomial of degree 8 and their weights.
QUADRATURE_NODES: Final = (
    0.1834346424956498,
    0.525532409916329,
    0.7966664774136267,
    0.9602898564975363,
)
QUADRATURE_WEIGHTS: Final = (
    0.362683783378362,
    0.31370664587788727,
    0.22238103445337448,
    0.10122853629037626,
)
#: Below this ``log t`` of a cumulative hazard ``t`` the cdf is ``t`` to binary64 precision.
_TINY_LOG_HAZARD: Final = -40.0
_TINY_HAZARD: Final = exp(_TINY_LOG_HAZARD)
#: From this standardised position on the right-Gumbel survival is ``exp(-z)`` to precision.
_GUMBEL_LINEAR_FROM: Final = 40.0
#: Products ``rate * x`` from here on are normal binary64 numbers (the exponential's ``log``).
_NORMAL_PRODUCT: Final = 1e-290
#: Survival values below this use the asymptotic Mills-ratio expansion (as the scalar kernel).
_ERFC_FLOOR: Final = 1e-300
_SQRT_TWO: Final = sqrt(2.0)
_HALF_LOG_2PI: Final = 0.5 * log(2.0 * pi)
_LOG_TWO: Final = log(2.0)


def _log_upper_tail_asymptotic(z: float) -> float:
    """``log(0.5 * erfc(z / sqrt 2))`` from the Mills-ratio series, for survival below 1e-300.

    The same expression as the scalar kernel of the in-memory likelihoods (``z * z`` rather
    than a power, which would raise for huge ``z``).
    """

    z_squared = z * z
    a = 1.0 / z_squared
    correction = -a + 3.0 * a * a - 15.0 * a * a * a
    return -0.5 * z_squared - log(z) - _HALF_LOG_2PI + log1p(correction)


def log_phi(z: Any) -> Any:
    """``log Phi(z)`` for a float64 array of non-NaN ``z``, equal to the scalar kernel bit for bit.

    ``log Phi(z) = log(0.5 * erfc(-z / sqrt 2))``: ``math.erfc`` and ``math.log`` are mapped over
    the array (the same numbers as the in-memory scalar kernel), the rest is numpy.  Where the
    survival falls below ``1e-300`` the asymptotic expansion of the scalar kernel is used.
    """

    count = z.shape[0]
    reflected = -z
    survival = 0.5 * np.fromiter(
        map(erfc, (reflected / _SQRT_TWO).tolist()), dtype=np.float64, count=count
    )
    result = np.empty(count)
    direct = np.flatnonzero(survival >= _ERFC_FLOOR)
    result[direct] = np.fromiter(
        map(log, survival[direct].tolist()), dtype=np.float64, count=direct.size
    )
    tail = np.flatnonzero(~(survival >= _ERFC_FLOOR))
    result[tail] = np.fromiter(
        map(_log_upper_tail_asymptotic, reflected[tail].tolist()),
        dtype=np.float64,
        count=tail.size,
    )
    return result


class _Kernel(ABC):
    """One family at fixed parameters: log cdf, log survival and the quadrature coordinates."""

    #: Positive-support families are integrated in ``log x``, location families in ``x``.
    lifetime: bool

    @abstractmethod
    def log_cdf(self, x: Any) -> Any:
        """``log F(x)`` for a float64 array of non-NaN points."""

    @abstractmethod
    def log_sf(self, x: Any) -> Any:
        """``log S(x)`` for a float64 array of non-NaN points."""

    @abstractmethod
    def median(self) -> float:
        """A point near the median, used only to pick the side a window is computed from."""

    @abstractmethod
    def position(self, x: Any) -> Any:
        """The standardised coordinate the density is integrated in, at the lower end ``x``."""

    @abstractmethod
    def width(self, lower: Any, upper: Any) -> Any:
        """The width of ``[lower, upper]`` in the integration coordinate."""

    @abstractmethod
    def log_density(self, position: Any) -> Any:
        """The log density of the integration coordinate at standardised ``position``."""


class _HazardKernel(_Kernel):
    """Exponential and Weibull minimum: ``S = exp(-t)`` with ``t = exp(shape * log(x / scale))``."""

    lifetime = True

    def __init__(self, shape: float) -> None:
        self.shape = shape
        self.log_shape = log(shape)

    @abstractmethod
    def _log_standardised(self, x: Any) -> Any:
        """``log(x / scale)`` for ``x >= 0`` (``-inf`` at zero), the family's standard position."""

    def log_sf(self, x: Any) -> Any:
        with np.errstate(all="ignore"):
            log_hazard = self.shape * self._log_standardised(np.maximum(x, 0.0))
            return 0.0 - np.exp(log_hazard)

    def log_cdf(self, x: Any) -> Any:
        with np.errstate(all="ignore"):
            log_hazard = self.shape * self._log_standardised(np.maximum(x, 0.0))
            return np.where(
                log_hazard < _TINY_LOG_HAZARD,
                log_hazard,
                log1mexp(0.0 - np.exp(log_hazard)),
            )

    def position(self, x: Any) -> Any:
        return self._log_standardised(x)

    def width(self, lower: Any, upper: Any) -> Any:
        with np.errstate(all="ignore"):
            return np.log1p((upper - lower) / lower)

    def log_density(self, position: Any) -> Any:
        with np.errstate(all="ignore"):
            scaled = self.shape * position
            return self.log_shape + scaled - np.exp(scaled)


class _Exponential(_HazardKernel):
    """``S = exp(-rate x)``: the cumulative hazard is the product itself, no logarithm needed."""

    def __init__(self, rate: float) -> None:
        super().__init__(1.0)
        self.rate = rate
        self.log_rate = log(rate)

    def median(self) -> float:
        return _LOG_TWO / self.rate

    def _log_standardised(self, x: Any) -> Any:
        with np.errstate(all="ignore"):
            product = self.rate * x
            return np.where(product >= _NORMAL_PRODUCT, np.log(product), self.log_rate + np.log(x))

    def log_sf(self, x: Any) -> Any:
        with np.errstate(all="ignore"):
            return 0.0 - self.rate * np.maximum(x, 0.0)

    def log_cdf(self, x: Any) -> Any:
        positive = np.maximum(x, 0.0)
        with np.errstate(all="ignore"):
            hazard = self.rate * positive
            value = log1mexp(0.0 - hazard)
            tiny = np.flatnonzero(hazard < _TINY_HAZARD)
            value[tiny] = self._log_standardised(positive[tiny])
            return value


class _WeibullMin(_HazardKernel):
    def __init__(self, shape: float, scale: float) -> None:
        super().__init__(shape)
        self.scale = scale

    def median(self) -> float:
        return float(self.scale * _LOG_TWO ** (1.0 / self.shape))

    def _log_standardised(self, x: Any) -> Any:
        return log_ratio(x, self.scale)


class _GumbelRight(_Kernel):
    lifetime = False

    def __init__(self, location: float, scale: float) -> None:
        self.location = location
        self.scale = scale

    def median(self) -> float:
        return self.location - self.scale * log(_LOG_TWO)

    def _z(self, x: Any) -> Any:
        return (x - self.location) / self.scale

    def log_cdf(self, x: Any) -> Any:
        with np.errstate(all="ignore"):
            return 0.0 - np.exp(-self._z(x))

    def log_sf(self, x: Any) -> Any:
        with np.errstate(all="ignore"):
            z = self._z(x)
            return np.where(z >= _GUMBEL_LINEAR_FROM, -z, log1mexp(0.0 - np.exp(-z)))

    def position(self, x: Any) -> Any:
        return self._z(x)

    def width(self, lower: Any, upper: Any) -> Any:
        return (upper - lower) / self.scale

    def log_density(self, position: Any) -> Any:
        with np.errstate(all="ignore"):
            return -position - np.exp(-position)


class _NormalKernel(_Kernel):
    """Normal and lognormal: ``log Phi`` of the standardised position."""

    @abstractmethod
    def _z(self, x: Any) -> Any:
        """The standardised position of ``x``."""

    def log_cdf(self, x: Any) -> Any:
        return log_phi(self._z(x))

    def log_sf(self, x: Any) -> Any:
        return log_phi(-self._z(x))

    def position(self, x: Any) -> Any:
        return self._z(x)

    def log_density(self, position: Any) -> Any:
        return -0.5 * position * position - _HALF_LOG_2PI


class _Normal(_NormalKernel):
    lifetime = False

    def __init__(self, mu: float, sigma: float) -> None:
        self.mu = mu
        self.sigma = sigma

    def median(self) -> float:
        return self.mu

    def _z(self, x: Any) -> Any:
        with np.errstate(all="ignore"):
            return (x - self.mu) / self.sigma

    def width(self, lower: Any, upper: Any) -> Any:
        return (upper - lower) / self.sigma


class _Lognormal(_NormalKernel):
    lifetime = True

    def __init__(self, mu_log: float, sigma_log: float) -> None:
        self.mu_log = mu_log
        self.sigma_log = sigma_log

    def median(self) -> float:
        return exp(self.mu_log) if self.mu_log < 709.0 else float("inf")

    def _z(self, x: Any) -> Any:
        with np.errstate(all="ignore"):
            return (np.log(np.maximum(x, 0.0)) - self.mu_log) / self.sigma_log

    def width(self, lower: Any, upper: Any) -> Any:
        with np.errstate(all="ignore"):
            return np.log1p((upper - lower) / lower) / self.sigma_log


class _Gamma(_Kernel):
    lifetime = True

    def __init__(self, shape: float, scale: float) -> None:
        if shape > GAMMA_MAX_SHAPE:
            raise LogProbabilityError(
                LogProbabilityErrorCode.GAMMA_SHAPE_REFUSED, shape_region(shape)
            )
        self.shape = shape
        self.scale = scale

    def median(self) -> float:
        shape = self.shape
        if shape >= 1.0:
            return self.scale * shape * (1.0 - 1.0 / (9.0 * shape)) ** 3
        return self.scale * exp((lgamma(shape + 1.0) - _LOG_TWO) / shape)

    def _both(self, x: Any) -> tuple[Any, Any]:
        positive = np.maximum(x, 0.0)
        with np.errstate(all="ignore"):
            return log_regularized_gamma(
                self.shape, positive / self.scale, log_ratio(positive, self.scale)
            )

    def log_cdf(self, x: Any) -> Any:
        return self._both(x)[0]

    def log_sf(self, x: Any) -> Any:
        return self._both(x)[1]

    def position(self, x: Any) -> Any:
        return log_ratio(x, self.scale)

    def width(self, lower: Any, upper: Any) -> Any:
        with np.errstate(all="ignore"):
            return np.log1p((upper - lower) / lower)

    def log_density(self, position: Any) -> Any:
        with np.errstate(all="ignore"):
            y = np.exp(position)
            return np.where(np.isfinite(y), log_prefactor(self.shape, y, position), -np.inf)


_BUILDERS: Final[Mapping[FamilyId, Callable[[Mapping[str, float]], _Kernel]]] = {
    FamilyId.EXPONENTIAL: lambda p: _Exponential(p["rate"]),
    FamilyId.WEIBULL_MIN: lambda p: _WeibullMin(p["shape"], p["scale"]),
    FamilyId.GUMBEL_RIGHT: lambda p: _GumbelRight(p["location"], p["scale"]),
    FamilyId.NORMAL: lambda p: _Normal(p["mu"], p["sigma"]),
    FamilyId.LOGNORMAL: lambda p: _Lognormal(p["mu_log"], p["sigma_log"]),
    FamilyId.GAMMA: lambda p: _Gamma(p["shape"], p["scale"]),
}


def _kernel(family: object, parameters: Mapping[str, object]) -> _Kernel:
    specification = FAMILY_REGISTRY.lookup(family)
    return _BUILDERS[specification.id](specification.validate_parameters(**parameters))


def _real_array(values: object, name: str) -> Any:
    """A float64 copy of a real array-like, with NaN refused; infinities are allowed."""

    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real number or an array of real numbers")
    array = array.astype(np.float64)
    if np.isnan(array).any():
        raise LogProbabilityError(LogProbabilityErrorCode.NONFINITE_INPUT)
    return array


def log_cdf(family: object, x: object, /, **parameters: object) -> Any:
    """``log F(x)`` of ``family`` at the parameters, accurate in both tails.

    ``-inf`` outside the support below it (and at ``F = 0``), ``0`` at ``+inf``.
    """

    kernel = _kernel(family, parameters)
    points = _real_array(x, "x")
    return kernel.log_cdf(np.ravel(points)).reshape(points.shape)


def log_sf(family: object, x: object, /, **parameters: object) -> Any:
    """``log S(x)`` of ``family`` at the parameters, accurate in both tails."""

    kernel = _kernel(family, parameters)
    points = _real_array(x, "x")
    return kernel.log_sf(np.ravel(points)).reshape(points.shape)


def _finite_window(kernel: _Kernel, lower: Any, upper: Any, switch: float = NARROW_SWITCH) -> Any:
    """``log(F(upper) - F(lower))`` for finite ``lower < upper`` (positive for lifetimes).

    A window whose complement is below ``switch`` is integrated instead of subtracted.
    """

    count = lower.shape[0]
    value = np.empty(count)
    complement = np.empty(count)
    median = kernel.median()
    low_side = upper <= median
    high_side = ~low_side & (lower >= median)
    across = ~(low_side | high_side)
    with np.errstate(all="ignore"):
        side = np.flatnonzero(low_side)
        if side.size:
            at_upper = kernel.log_cdf(upper[side])
            difference = kernel.log_cdf(lower[side]) - at_upper
            value[side] = np.where(at_upper == -np.inf, -np.inf, at_upper + log1mexp(difference))
            complement[side] = -np.expm1(difference)
        side = np.flatnonzero(high_side)
        if side.size:
            at_lower = kernel.log_sf(lower[side])
            difference = kernel.log_sf(upper[side]) - at_lower
            value[side] = np.where(at_lower == -np.inf, -np.inf, at_lower + log1mexp(difference))
            complement[side] = -np.expm1(difference)
        side = np.flatnonzero(across)
        if side.size:
            tails = -(np.exp(kernel.log_cdf(lower[side])) + np.exp(kernel.log_sf(upper[side])))
            value[side] = np.log1p(tails)
            complement[side] = 1.0 + tails
        narrow = np.flatnonzero(complement < switch)
        if narrow.size:
            value[narrow] = _quadrature(kernel, lower[narrow], upper[narrow])
    return value


def _quadrature(kernel: _Kernel, lower: Any, upper: Any) -> Any:
    """``log`` of the integral of the density over ``[lower, upper]`` by the 8-node rule.

    ``log(width) + log g(middle) + log(sum of weights * exp(log g(node) - log g(middle)))`` in
    the integration coordinate of the family, so nothing is subtracted.
    """

    start = kernel.position(lower)
    width = kernel.width(lower, upper)
    with np.errstate(all="ignore"):
        middle = kernel.log_density(start + 0.5 * width)
        total = np.zeros_like(width)
        for node, weight in zip(QUADRATURE_NODES, QUADRATURE_WEIGHTS, strict=True):
            near = kernel.log_density(start + width * (0.5 - 0.5 * node))
            far = kernel.log_density(start + width * (0.5 + 0.5 * node))
            total += (0.5 * weight) * (np.exp(near - middle) + np.exp(far - middle))
        return np.where(np.isfinite(middle), np.log(width) + middle + np.log(total), -np.inf)


def log_window(family: object, lower: object, upper: object, /, **parameters: object) -> Any:
    """``log(F(upper) - F(lower))`` for ``lower < upper``, either end possibly infinite.

    A refused window (any ``lower >= upper``, or a NaN end) raises
    :class:`~veridist.statistics._logerrors.LogProbabilityError` before anything is computed.
    For the positive-support families a ``lower`` at or below zero is the open lower end and a
    window entirely at or below zero has probability zero (``-inf``).
    """

    kernel = _kernel(family, parameters)
    first, second = np.broadcast_arrays(_real_array(lower, "lower"), _real_array(upper, "upper"))
    shape = first.shape
    result = kernel_log_window(kernel, np.ravel(first).copy(), np.ravel(second).copy())
    return result.reshape(shape)


def build_kernel(family: object, **parameters: object) -> _Kernel:
    """The kernel of ``family`` at validated ``parameters``, to evaluate many windows at once."""

    return _kernel(family, parameters)


def kernel_log_window(kernel: _Kernel, low: Any, high: Any) -> Any:
    """``log(F(high) - F(low))`` of a built kernel for flat float64 arrays of non-NaN ends.

    The body of :func:`log_window` for callers that build the kernel once; every element is
    computed on its own, so the result equals the one-element result for any batch.
    """

    if not (low < high).all():
        raise LogProbabilityError(LogProbabilityErrorCode.INVALID_WINDOW)
    result = np.empty(low.shape[0])
    open_low = (low == -np.inf) | (kernel.lifetime & (low <= 0.0))
    open_high = high == np.inf
    empty = kernel.lifetime & (high <= 0.0)
    result[empty] = -np.inf
    both = open_low & open_high
    result[both] = 0.0
    only_low = open_low & ~open_high & ~empty
    result[only_low] = kernel.log_cdf(high[only_low])
    only_high = open_high & ~open_low
    result[only_high] = kernel.log_sf(low[only_high])
    finite = ~(open_low | open_high | empty)
    result[finite] = _finite_window(kernel, low[finite], high[finite])
    return result


__all__ = [
    "NARROW_SWITCH",
    "QUADRATURE_NODES",
    "QUADRATURE_WEIGHTS",
    "build_kernel",
    "kernel_log_window",
    "log1mexp",
    "log_cdf",
    "log_phi",
    "log_sf",
    "log_window",
]
