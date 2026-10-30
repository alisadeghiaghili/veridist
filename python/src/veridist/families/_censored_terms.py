"""Per-row log-likelihood terms, with gradients and Hessians, of censored and truncated rows.

For a row with censoring set ``C = (A, B]`` and truncation window ``W = (a, b]`` (both read
inside each other, see :mod:`veridist.families._censored_rows`) the term is
``log P(A < X <= B) - log P(a < X <= b)`` for a censored row and ``log f(x) - log P(a < X <= b)``
for an exact one, the second part absent without a window.  The probabilities come from
:func:`~veridist.statistics._logprob.kernel_log_window`, which has one stable form per side of
the distribution and an 8-node quadrature for narrow windows, so no probability is formed and
then logged.

Coordinates, in the (standardised) units of the optimizer:

* location-scale families: ``theta = (m, u) = (mu / sigma, 1 / sigma)`` with
  ``z = u y - m``, whose base density ``g`` is the standard normal or the standard smallest
  extreme value; the derivatives of ``log P`` follow from ``+-g(z)`` and ``+-g'(z)`` at the two
  ends of the set;
* exponential: ``theta = (rate,)``, ``z = rate x``, base density ``exp(-z)``;
* gamma: ``theta = (alpha, beta) = (log shape, log scale)``; the derivatives in ``beta`` are
  analytic and those in ``alpha`` central differences (a five-point stencil in ``log shape``)
  of ``log P`` and of its ``beta`` derivative, the one place no closed form exists.

Each ``*_terms`` function returns one array per term (the log-likelihood, the gradient and the
upper triangle of the Hessian), row by row; :func:`exact_sums` adds them exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from importlib import import_module
from math import exp, fsum, isfinite, lgamma, log, pi
from typing import Any, Final

from veridist.families._censored_rows import CensoredRows
from veridist.families.registry import FamilyId
from veridist.statistics._logerrors import LogProbabilityError
from veridist.statistics._logprob import build_kernel, kernel_log_window
from veridist.statistics._special import digamma, trigamma

np: Any = import_module("numpy")

_HALF_LOG_2PI: Final = 0.5 * log(2.0 * pi)

#: The step, in ``log shape``, of the shape differences of the optimizer (five-point stencil,
#: truncation error of order ``step**4``, rounding error of order ``1e-16 / step**2``).
GAMMA_OPTIMIZER_STEP: Final = 2e-3
#: The step of the shape differences of the observed information (as the earlier releases).
GAMMA_INFORMATION_STEP: Final = 0.03
_FIVE_POINT: Final = (-2, -1, 0, 1, 2)


class _NotEvaluable(Exception):
    """The terms cannot be evaluated at this parameter vector (no finite, ordered window)."""


def _window(kernel: Any, low: Any, high: Any) -> Any:
    try:
        return kernel_log_window(kernel, np.ascontiguousarray(low), np.ascontiguousarray(high))
    except LogProbabilityError as error:
        raise _NotEvaluable from error


class _Normal:
    kernel: Any = None

    @staticmethod
    def logg(z: Any) -> Any:
        return -0.5 * z * z - _HALF_LOG_2PI

    @staticmethod
    def dlogg(z: Any) -> Any:
        return -z

    @staticmethod
    def d2logg(z: Any) -> Any:
        return -np.ones_like(z)

    @staticmethod
    def log_window(low: Any, high: Any) -> Any:
        if _Normal.kernel is None:
            _Normal.kernel = build_kernel(FamilyId.NORMAL, mu=0.0, sigma=1.0)
        return _window(_Normal.kernel, low, high)


class _SmallestExtreme:
    """``G(z) = 1 - exp(-exp(z))``: the right Gumbel reflected, so its windows are reflected."""

    kernel: Any = None

    @staticmethod
    def logg(z: Any) -> Any:
        with np.errstate(all="ignore"):
            return z - np.exp(z)

    @staticmethod
    def dlogg(z: Any) -> Any:
        with np.errstate(all="ignore"):
            return 1.0 - np.exp(z)

    @staticmethod
    def d2logg(z: Any) -> Any:
        with np.errstate(all="ignore"):
            return -np.exp(z)

    @staticmethod
    def log_window(low: Any, high: Any) -> Any:
        if _SmallestExtreme.kernel is None:
            _SmallestExtreme.kernel = build_kernel(FamilyId.GUMBEL_RIGHT, location=0.0, scale=1.0)
        return _window(_SmallestExtreme.kernel, -high, -low)


class _StandardExponential:
    """``G(z) = 1 - exp(-z)``: only its window probabilities are needed."""

    kernel: Any = None

    @staticmethod
    def log_window(low: Any, high: Any) -> Any:
        if _StandardExponential.kernel is None:
            _StandardExponential.kernel = build_kernel(FamilyId.EXPONENTIAL, rate=1.0)
        return _window(_StandardExponential.kernel, low, high)


BASES: Final[dict[FamilyId, Any]] = {
    FamilyId.NORMAL: _Normal,
    FamilyId.LOGNORMAL: _Normal,
    FamilyId.WEIBULL_MIN: _SmallestExtreme,
    FamilyId.GUMBEL_RIGHT: _SmallestExtreme,
    FamilyId.EXPONENTIAL: _StandardExponential,
}


def exact_sums(terms: Iterable[Any], extras: tuple[float, ...] | None = None) -> list[float]:
    """The exact (correctly rounded) sum of each array of ``terms``.

    ``extras`` holds one more addend per term (the summarised block of the exponential).  A
    sum that is not a finite number is ``nan``.
    """

    sums: list[float] = []
    for position, term in enumerate(terms):
        values = term.tolist()
        if extras is not None:
            values.append(extras[position])
        try:
            total = fsum(values)
        except (OverflowError, ValueError):
            total = float("nan")
        sums.append(total if isfinite(total) else float("nan"))
    return sums


# ------------------------------------------------------------------ location-scale families


def _interval_terms(base: Any, low: Any, high: Any, m: float, u: float) -> tuple[Any, ...]:
    """``log P`` of the sets ``(low, high]`` (``y`` space, infinite ends allowed) and its
    gradient and Hessian in ``theta = (m, u)``: ``(lp, g_m, g_u, h_mm, h_mu, h_uu)``."""

    with np.errstate(all="ignore"):
        z_low = np.where(np.isinf(low), low, u * low - m)
        z_high = np.where(np.isinf(high), high, u * high - m)
        lp = base.log_window(z_low, z_high)
        finite_low, finite_high = np.isfinite(z_low), np.isfinite(z_high)
        safe_low = np.where(finite_low, z_low, 0.0)
        safe_high = np.where(finite_high, z_high, 0.0)
        ratio_low = np.where(finite_low, np.exp(base.logg(safe_low) - lp), 0.0)
        ratio_high = np.where(finite_high, np.exp(base.logg(safe_high) - lp), 0.0)
        # A density that underflows contributes nothing, even where its log-derivative is
        # infinite (``0 * inf`` would be ``nan``).
        slope_low = np.where(ratio_low > 0.0, ratio_low * base.dlogg(safe_low), 0.0)
        slope_high = np.where(ratio_high > 0.0, ratio_high * base.dlogg(safe_high), 0.0)
        y_low = np.where(finite_low, low, 0.0)
        y_high = np.where(finite_high, high, 0.0)
        score_m = ratio_low - ratio_high
        score_u = ratio_high * y_high - ratio_low * y_low
        p_mm = slope_high - slope_low
        p_mu = -(slope_high * y_high - slope_low * y_low)
        p_uu = slope_high * y_high * y_high - slope_low * y_low * y_low
        return (
            lp,
            score_m,
            score_u,
            p_mm - score_m * score_m,
            p_mu - score_m * score_u,
            p_uu - score_u * score_u,
        )


def location_scale_terms(rows: CensoredRows, theta: tuple[float, ...]) -> tuple[Any, ...]:
    """Per-row ``(ell, g_m, g_u, h_mm, h_mu, h_uu)`` at ``theta = (m, u)``."""

    base = BASES[rows.family]
    m, u = theta
    count = rows.explicit_count
    columns = [np.zeros(count) for _ in range(6)]
    ell, g_m, g_u, h_mm, h_mu, h_uu = columns
    exact = np.flatnonzero(rows.exact)
    if exact.size:
        y = rows.yv[exact]
        with np.errstate(all="ignore"):
            z = u * y - m
            first, second = base.dlogg(z), base.d2logg(z)
            ell[exact] = log(u) + base.logg(z)
            g_m[exact] = -first
            g_u[exact] = first * y + 1.0 / u
            h_mm[exact] = second
            h_mu[exact] = -second * y
            h_uu[exact] = second * y * y - 1.0 / (u * u)
    censored = np.flatnonzero(~rows.exact)
    if censored.size:
        low = np.maximum(rows.lo[censored], rows.wl[censored])
        high = np.minimum(rows.hi[censored], rows.wh[censored])
        for column, term in zip(columns, _interval_terms(base, low, high, m, u), strict=True):
            column[censored] = term
    truncated = np.flatnonzero(np.isfinite(rows.wl) | np.isfinite(rows.wh))
    if truncated.size:
        window = _interval_terms(base, rows.wl[truncated], rows.wh[truncated], m, u)
        for column, term in zip(columns, window, strict=True):
            column[truncated] -= term
    return tuple(columns)


# ------------------------------------------------------------------------------ exponential


def _exponential_part(rate: float, low: Any, high: Any) -> tuple[Any, Any, Any]:
    """``log P(low < X <= high)`` of the exponential at ``rate`` and its two derivatives."""

    base = _StandardExponential
    with np.errstate(all="ignore"):
        z_low = rate * low
        z_high = np.where(np.isinf(high), np.inf, rate * high)
        lp = base.log_window(z_low, z_high)
        finite_low, finite_high = np.isfinite(z_low), np.isfinite(z_high)
        ratio_low = np.where(finite_low, np.exp(-np.where(finite_low, z_low, 0.0) - lp), 0.0)
        ratio_high = np.where(finite_high, np.exp(-np.where(finite_high, z_high, 0.0) - lp), 0.0)
        y_low = np.where(finite_low, low, 0.0)
        y_high = np.where(finite_high, high, 0.0)
        score = ratio_high * y_high - ratio_low * y_low
        second = -(ratio_high * y_high * y_high - ratio_low * y_low * y_low)
        return lp, score, second - score * score


def exponential_terms(rows: CensoredRows, theta: tuple[float, ...]) -> tuple[Any, ...]:
    """Per-row ``(ell, g, h)`` at ``theta = (rate,)`` of the rows held in the arrays."""

    (rate,) = theta
    count = rows.explicit_count
    ell, gradient, hessian = np.zeros(count), np.zeros(count), np.zeros(count)
    exact = np.flatnonzero(rows.exact)
    if exact.size:
        y = rows.yv[exact]
        ell[exact] = log(rate) - rate * y
        gradient[exact] = 1.0 / rate - y
        hessian[exact] = -1.0 / (rate * rate)
    censored = np.flatnonzero(~rows.exact)
    if censored.size:
        low = np.maximum(np.maximum(rows.lo[censored], rows.wl[censored]), 0.0)
        high = np.minimum(rows.hi[censored], rows.wh[censored])
        ell[censored], gradient[censored], hessian[censored] = _exponential_part(rate, low, high)
    truncated = np.flatnonzero(np.isfinite(rows.wl) | np.isfinite(rows.wh))
    if truncated.size:
        value, score, second = _exponential_part(
            rate, np.maximum(rows.wl[truncated], 0.0), rows.wh[truncated]
        )
        ell[truncated] -= value
        gradient[truncated] -= score
        hessian[truncated] -= second
    return ell, gradient, hessian


def exponential_block(rows: CensoredRows, theta: tuple[float, ...]) -> tuple[float, ...]:
    """The ``(ell, g, h)`` of the summarised block of exact and right-censored rows."""

    (rate,) = theta
    events, total = float(rows.plain_events), rows.plain_total
    return (
        events * log(rate) - rate * total if events else -rate * total,
        events / rate - total,
        -events / (rate * rate),
    )


# ---------------------------------------------------------------------------------- gamma


def _gamma_part(shape: float, log_scale: float, low: Any, high: Any) -> tuple[Any, Any, Any]:
    """``log P(low < X <= high)`` of the gamma and its ``log scale`` derivatives.

    Returns ``(lp, s, p_bb)``: ``d log P / d beta`` and ``P_bb / P``.
    """

    scale = exp(log_scale)
    kernel = build_kernel(FamilyId.GAMMA, shape=shape, scale=scale)
    with np.errstate(all="ignore"):
        lp = _window(kernel, low, high)
        y_low = low / scale
        y_high = np.where(np.isinf(high), np.inf, high / scale)
        finite_low = (y_low > 0.0) & np.isfinite(y_low)
        finite_high = np.isfinite(y_high) & (y_high > 0.0)
        safe_low = np.where(finite_low, y_low, 1.0)
        safe_high = np.where(finite_high, y_high, 1.0)
        log_gamma = lgamma(shape)

        def weighted_density(y: Any) -> Any:
            return np.log(y) + (shape - 1.0) * np.log(y) - y - log_gamma

        ratio_low = np.where(finite_low, np.exp(weighted_density(safe_low) - lp), 0.0)
        ratio_high = np.where(finite_high, np.exp(weighted_density(safe_high) - lp), 0.0)
        slope_low = np.where(finite_low, ratio_low * (shape - safe_low), 0.0)
        slope_high = np.where(finite_high, ratio_high * (shape - safe_high), 0.0)
        return lp, -(ratio_high - ratio_low), slope_high - slope_low


def _gamma_interval(
    alpha: float, beta: float, low: Any, high: Any, step: float, value_only: bool
) -> tuple[Any, ...]:
    """``(lp, g_a, g_b, h_aa, h_ab, h_bb)`` of ``log P(low < X <= high)`` in ``(alpha, beta)``."""

    shape = exp(alpha)
    lp0, score_b, p_bb = _gamma_part(shape, beta, low, high)
    zeros = np.zeros_like(lp0)
    if value_only:
        return lp0, zeros, zeros, zeros, zeros, zeros
    lps: dict[int, Any] = {0: lp0}
    scores: dict[int, Any] = {0: score_b}
    for offset in _FIVE_POINT:
        if offset == 0:
            continue
        lps[offset], scores[offset], _ = _gamma_part(exp(alpha + offset * step), beta, low, high)
    first = (8.0 * (lps[1] - lps[-1]) - (lps[2] - lps[-2])) / (12.0 * step)
    second = (-lps[2] + 16.0 * lps[1] - 30.0 * lps[0] + 16.0 * lps[-1] - lps[-2]) / (
        12.0 * step * step
    )
    mixed = (8.0 * (scores[1] - scores[-1]) - (scores[2] - scores[-2])) / (12.0 * step)
    return lp0, first, score_b, second, mixed, p_bb - score_b * score_b


def gamma_terms(
    rows: CensoredRows,
    theta: tuple[float, ...],
    *,
    step: float = GAMMA_OPTIMIZER_STEP,
    value_only: bool = False,
) -> tuple[Any, ...]:
    """Per-row ``(ell, g_a, g_b, h_aa, h_ab, h_bb)`` at ``theta = (log shape, log scale)``."""

    alpha, beta = theta
    shape = exp(alpha)
    count = rows.explicit_count
    columns = [np.zeros(count) for _ in range(6)]
    ell, g_a, g_b, h_aa, h_ab, h_bb = columns
    exact = np.flatnonzero(rows.exact)
    if exact.size:
        with np.errstate(all="ignore"):
            log_y = np.log(rows.yv[exact]) - beta
            y = np.exp(log_y)
            psi, trigamma_value = digamma(shape), trigamma(shape)
            ell[exact] = (shape - 1.0) * log_y - y - lgamma(shape) - beta
            g_a[exact] = shape * (log_y - psi)
            g_b[exact] = y - shape
            h_aa[exact] = shape * (log_y - psi) - shape * shape * trigamma_value
            h_ab[exact] = -shape
            h_bb[exact] = -y
    censored = np.flatnonzero(~rows.exact)
    if censored.size:
        low = np.maximum(np.maximum(rows.lo[censored], rows.wl[censored]), 0.0)
        high = np.minimum(rows.hi[censored], rows.wh[censored])
        parts = _gamma_interval(alpha, beta, low, high, step, value_only)
        for column, term in zip(columns, parts, strict=True):
            column[censored] = term
    truncated = np.flatnonzero(np.isfinite(rows.wl) | np.isfinite(rows.wh))
    if truncated.size:
        window = _gamma_interval(
            alpha, beta, np.maximum(rows.wl[truncated], 0.0), rows.wh[truncated], step, value_only
        )
        for column, term in zip(columns, window, strict=True):
            column[truncated] -= term
    return tuple(columns)


def terms_function(family: FamilyId) -> Callable[..., tuple[Any, ...]]:
    """The per-row terms of ``family``."""

    if family is FamilyId.EXPONENTIAL:
        return exponential_terms
    if family is FamilyId.GAMMA:
        return gamma_terms
    return location_scale_terms


def theta_sums(rows: CensoredRows, theta: tuple[float, ...]) -> list[float] | None:
    """The exact sums of the terms at ``theta`` (log-likelihood, gradient, upper Hessian), or
    ``None`` where the likelihood cannot be evaluated (a refused kernel request, an empty
    window, a term that is not a finite number)."""

    try:
        with np.errstate(all="ignore"):
            terms = terms_function(rows.family)(rows, theta)
            extras = exponential_block(rows, theta) if rows.family is FamilyId.EXPONENTIAL else None
    except (_NotEvaluable, LogProbabilityError, ArithmeticError, ValueError):
        return None
    sums = exact_sums(terms, extras)
    return None if any(value != value for value in sums) else sums


__all__ = [
    "BASES",
    "GAMMA_INFORMATION_STEP",
    "GAMMA_OPTIMIZER_STEP",
    "_NotEvaluable",
    "exact_sums",
    "exponential_block",
    "exponential_terms",
    "gamma_terms",
    "location_scale_terms",
    "terms_function",
    "theta_sums",
]
