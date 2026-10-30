"""Maximum-likelihood estimates for samples with left- or interval-censored rows or windows.

:func:`estimate` is the one entry the per-family fits call for data that the closed forms and
the bounded searches of the earlier releases do not take (see
:attr:`~veridist.families._fit_facts.ObservationSummary.is_new_data`).  It returns a
:class:`CensoredEstimate` or the name of the failure code that applies.

* **Exponential**, exact and right-censored rows with at most left truncation: the closed
  form ``rate = events / (sum of times - sum of left-truncation points)``, both sums added
  together with ``math.fsum`` so that the quantity has one rounding (memorylessness: each
  ``x - a`` given ``x > a`` is exponential with the same rate).
* **Weibull minimum**, exact and right-censored rows with left truncation only: the scale has a
  closed form for every shape (``scale**k = (sum of x**k - sum of a**k) / events``), so the
  fit is the root of the one-parameter profile score in ``log k``.  The profile score is
  followed over a grid of ``log k``; the maximum of the highest profile value is refined by
  bisection and the number of stationary points the grid showed is recorded.
* **Everything else**: the damped Newton method of :mod:`veridist.families._newton` in the
  coordinates ``rate``, ``(mu / sigma, 1 / sigma)`` (of ``log x`` for the lognormal and the
  Weibull minimum, of ``-x`` for the right Gumbel) and ``(log shape, log scale)``, started
  from the pseudo-sample of :func:`~veridist.families._censored_rows.pilot`.

The reported log-likelihood is the library's own total
(:func:`~veridist.statistics.lifetime_log_likelihood.reduce_lifetime_log_likelihood_chunks`) at
the estimate, so it equals what the log-likelihood entry points give for the same data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from math import exp, fsum, isfinite, log
from typing import Any, Final

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.families._censored_rows import (
    CensoredRows,
    Pilot,
    build_rows,
    pilot,
)
from veridist.families._censored_terms import theta_sums
from veridist.families._fit_facts import FitDiagnostics, ObservationSummary, OptimumKind
from veridist.families._newton import (
    GAMMA_TOLERANCE,
    TOLERANCE,
    concave_in_coordinates,
    maximize,
)
from veridist.families.registry import FamilyId

np: Any = import_module("numpy")

#: The log of the Weibull shape is searched on this grid for stationary points of the profile
#: (the hard limits of the earlier releases, in steps of a quarter).
PROFILE_LOG_SHAPE_LIMIT: Final = 20.0
PROFILE_GRID_STEP: Final = 0.25
#: Bisections of the profile score on the bracket of a stationary point (a quarter divided by
#: 2**48 is below the spacing of the binary64 numbers there).
PROFILE_BISECTIONS: Final = 48

_POSITIVE_FAMILIES: Final = frozenset({FamilyId.WEIBULL_MIN, FamilyId.LOGNORMAL, FamilyId.GAMMA})


@dataclass(frozen=True, slots=True)
class CensoredEstimate:
    """A converged estimate: canonical parameters, the log-likelihood and what is claimed."""

    parameters: tuple[float, ...]
    log_likelihood: float
    optimum: OptimumKind
    diagnostics: FitDiagnostics
    rows: CensoredRows
    total_time: float | None = None


@dataclass(frozen=True, slots=True)
class CensoredRefusal:
    """The name of the failure code that applies, and for the exponential the total time."""

    code: str
    total_time: float | None = None


def has_invalid_support(family: FamilyId, observations: Sequence[Any]) -> bool:
    """Whether a positive-support family is offered an exact or right-censored time at or
    below zero (as in the earlier releases, an exact zero is not a lifetime of these families)."""

    if family not in _POSITIVE_FAMILIES:
        return False
    return any(
        type(observation) in (ExactLifetime, RightCensoredLifetime) and observation.time <= 0.0
        for observation in observations
    )


def _kernel_total(
    family: FamilyId, observations: Sequence[Any], parameters: dict[str, Any]
) -> float | None:
    """The log-likelihood total of the library's reducers, or ``None`` if it is not a number."""

    from veridist.statistics.lifetime_log_likelihood import (
        reduce_lifetime_log_likelihood_chunks,
        reduce_value_log_likelihood_chunks,
    )
    from veridist.statistics.log_likelihood import LogLikelihoodSuccess

    reducer = (
        reduce_value_log_likelihood_chunks
        if family in (FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT)
        else reduce_lifetime_log_likelihood_chunks
    )
    try:
        outcome = reducer(family, [observations], **parameters)
    except (ValueError, TypeError, ArithmeticError):
        return None
    if type(outcome) is not LogLikelihoodSuccess:
        return None
    return outcome.total_log_likelihood


PARAMETER_NAMES: Final = {
    FamilyId.EXPONENTIAL: ("rate",),
    FamilyId.WEIBULL_MIN: ("shape", "scale"),
    FamilyId.LOGNORMAL: ("mu_log", "sigma_log"),
    FamilyId.GAMMA: ("shape", "scale"),
    FamilyId.NORMAL: ("mu", "sigma"),
    FamilyId.GUMBEL_RIGHT: ("location", "scale"),
}


POSITIVE_PARAMETERS: Final = {
    FamilyId.EXPONENTIAL: (0,),
    FamilyId.WEIBULL_MIN: (0, 1),
    FamilyId.LOGNORMAL: (1,),
    FamilyId.GAMMA: (0, 1),
    FamilyId.NORMAL: (1,),
    FamilyId.GUMBEL_RIGHT: (1,),
}


def to_parameters(
    family: FamilyId, theta: tuple[float, ...], centre: float, scale: float
) -> tuple[float, ...]:
    """The canonical parameters of the iterate ``theta`` of the optimizer's coordinates."""

    if family is FamilyId.EXPONENTIAL:
        return (theta[0] / scale,)
    if family is FamilyId.GAMMA:
        return (exp(theta[0]), scale * exp(theta[1]))
    m, u = theta
    mean, spread = centre + scale * (m / u), scale / u
    if family is FamilyId.WEIBULL_MIN:
        return (1.0 / spread, exp(mean))
    if family is FamilyId.GUMBEL_RIGHT:
        return (-mean, spread)
    return (mean, spread)


def _finish(
    family: FamilyId,
    values: Sequence[Any],
    rows: CensoredRows,
    parameters: tuple[float, ...],
    optimum: OptimumKind,
    diagnostics: FitDiagnostics,
    total_time: float | None = None,
) -> CensoredEstimate | CensoredRefusal:
    if not all(isfinite(value) for value in parameters) or any(
        parameters[index] <= 0.0 for index in POSITIVE_PARAMETERS[family]
    ):
        return CensoredRefusal("OPTIMIZER_EXHAUSTED")
    named = dict(zip(PARAMETER_NAMES[family], parameters, strict=True))
    total = _kernel_total(family, values, named)
    if total is None:
        return CensoredRefusal("OPTIMIZER_EXHAUSTED")
    if rows.plain_count:
        rate = parameters[0]
        block = -rate * rows.plain_total
        if rows.plain_events:
            block += rows.plain_events * log(rate)
        total = fsum((total, block))
    if not isfinite(total):
        return CensoredRefusal("OPTIMIZER_EXHAUSTED")
    return CensoredEstimate(parameters, total, optimum, diagnostics, rows, total_time)


# ----------------------------------------------------------------------- closed forms


def _only_left_windows(rows: CensoredRows) -> bool:
    return bool(np.isinf(rows.wh).all() and (rows.wh > 0).all())


def _plain_kinds(rows: CensoredRows) -> bool:
    """Whether every explicit row is exact or right-censored (no left or interval rows)."""

    censored = ~rows.exact
    return bool(np.isinf(rows.hi[censored]).all())


def exponential_closed_form_applies(rows: CensoredRows) -> bool:
    """Exact and right-censored rows with at most left truncation."""

    return _plain_kinds(rows) and _only_left_windows(rows)


def _exponential_closed_form(rows: CensoredRows) -> CensoredEstimate | CensoredRefusal:
    points = np.where(rows.exact, rows.yv, np.maximum(rows.lo, 0.0))
    lower = np.where(np.isfinite(rows.wl), rows.wl, 0.0)
    try:
        total = fsum([*points.tolist(), *(-lower).tolist(), rows.plain_total])
    except OverflowError:
        return CensoredRefusal("NUMERICAL_OVERFLOW")
    events = int(rows.exact.sum()) + rows.plain_events
    if total <= 0.0:
        return CensoredRefusal("UNBOUNDED_LIKELIHOOD", 0.0)
    rate = events / total
    if not (isfinite(rate) and rate > 0.0):
        return CensoredRefusal("NUMERICAL_OVERFLOW")
    return CensoredEstimate(
        (rate,),
        events * log(rate) - rate * total,
        OptimumKind.CLOSED_FORM,
        FitDiagnostics(0, 0, method="closed_form"),
        rows,
        total,
    )


def weibull_profile_applies(rows: CensoredRows) -> bool:
    """Exact and right-censored rows with left truncation only, and some truncated row."""

    return (
        _plain_kinds(rows)
        and _only_left_windows(rows)
        and bool(np.isfinite(rows.wl).any())
        and bool(rows.exact.any())
    )


class _Profile:
    """The profile of the left-truncated Weibull in ``kappa = log shape``.

    With ``w = log x - mean`` (standardised), ``v`` the same for the truncation point,
    ``S0(k) = sum exp(k w) - sum exp(k v)`` and ``S1`` its derivative in ``k``, the scale is
    ``scale**k = S0 / events`` and the derivative of the profile log-likelihood in ``kappa`` (the
    score) is ``events + k L - events k S1 / S0`` with ``L`` the sum of the ``w`` of the exact
    rows.  The sums are formed relative to the largest ``w``, so nothing overflows.
    """

    def __init__(self, rows: CensoredRows) -> None:
        points = np.where(rows.exact, rows.yv, rows.lo)
        self.mean = fsum(points.tolist()) / points.shape[0]
        self.w = points - self.mean
        truncated = np.isfinite(rows.wl)
        self.v = np.where(truncated, rows.wl - self.mean, 0.0)
        self.truncated = truncated
        self.events = int(rows.exact.sum())
        self.sum_exact = fsum(self.w[rows.exact].tolist())
        self.top = float(self.w.max())

    def _sums(self, k: float) -> tuple[float, float, float]:
        with np.errstate(all="ignore"):
            near = np.exp(k * (self.w - self.top))
            drop = k * (self.v - self.w)
            gap = np.where(self.truncated, np.exp(drop), 0.0)
            # ``1 - gap`` through ``expm1``: for a small shape ``gap`` is within ``k`` of one and
            # the plain difference would keep only a handful of correct digits, enough for the
            # sign of the score near the lower end of the grid to depend on the last bit of the
            # platform's ``exp`` (and so on the number of stationary points seen).
            short = np.where(self.truncated, -np.expm1(drop), 1.0)
            s0 = fsum((near * short).tolist())
            s1 = fsum((near * (self.w - np.where(self.truncated, self.v * gap, 0.0))).tolist())
        return s0, s1, self.top

    def score(self, kappa: float) -> float:
        """The derivative of :meth:`value` in ``kappa`` (``nan`` where ``S0`` is not positive)."""

        k = exp(kappa)
        s0, s1, _ = self._sums(k)
        if not s0 > 0.0:
            return float("nan")
        return self.events + k * self.sum_exact - self.events * k * s1 / s0

    def value(self, kappa: float) -> float:
        """The profile log-likelihood up to a constant that does not depend on ``kappa``."""

        k = exp(kappa)
        s0, _, top = self._sums(k)
        log_sum = k * top + log(s0)
        return self.events * (kappa - log_sum + log(self.events)) + (k - 1.0) * self.sum_exact

    def scale(self, kappa: float) -> float:
        """The profiled scale in the original units."""

        k = exp(kappa)
        s0, _, top = self._sums(k)
        return exp(self.mean + top + (log(s0) - log(self.events)) / k)


def _weibull_profile(
    values: Sequence[Any], rows: CensoredRows
) -> CensoredEstimate | CensoredRefusal:
    profile = _Profile(rows)
    count = int(round(2.0 * PROFILE_LOG_SHAPE_LIMIT / PROFILE_GRID_STEP)) + 1
    grid = [-PROFILE_LOG_SHAPE_LIMIT + index * PROFILE_GRID_STEP for index in range(count)]
    scores = [profile.score(kappa) for kappa in grid]
    seen = 0
    maxima: list[float] = []
    for index in range(count - 1):
        first, second = scores[index], scores[index + 1]
        if not (isfinite(first) and isfinite(second)) or (first > 0.0) == (second > 0.0):
            continue
        seen += 1
        if first > 0.0:
            low, high = grid[index], grid[index + 1]
            for _ in range(PROFILE_BISECTIONS):
                middle = 0.5 * (low + high)
                if profile.score(middle) > 0.0:
                    low = middle
                else:
                    high = middle
            maxima.append(0.5 * (low + high))
    if not maxima:
        return CensoredRefusal("BOUNDARY_SOLUTION")
    best = max(maxima, key=profile.value)
    shape = exp(best)
    scale = profile.scale(best)
    if not (isfinite(shape) and isfinite(scale) and scale > 0.0):
        return CensoredRefusal("OPTIMIZER_EXHAUSTED")
    diagnostics = FitDiagnostics(
        1,
        len(maxima),
        distinct_stationary_points=len(maxima),
        stationary_points_seen=seen,
        method="profile",
    )
    return _finish(
        FamilyId.WEIBULL_MIN,
        values,
        rows,
        (shape, scale),
        OptimumKind.STATIONARY_POINT,
        diagnostics,
    )


# --------------------------------------------------------------------------- the estimate


def estimate(
    family: FamilyId,
    values: Sequence[Any],
    summary: ObservationSummary,
    *,
    plain_count: int = 0,
    plain_events: int = 0,
    plain_total: float = 0.0,
) -> CensoredEstimate | CensoredRefusal:
    """Fit ``family`` to ``values`` (any of the eight observation types, with windows).

    ``summary`` describes ``values`` (the summarised block of the exponential not included).
    Returns the estimate, or a :class:`CensoredRefusal` naming the failure code of the family
    (``NO_OBSERVED_EVENTS``, ``INVALID_SUPPORT``, ``DEGENERATE_SAMPLE``,
    ``OPTIMIZER_EXHAUSTED``, ``BOUNDARY_SOLUTION``, and for the exponential
    ``UNBOUNDED_LIKELIHOOD`` and ``NUMERICAL_OVERFLOW``).
    """

    rows = build_rows(
        family, values, plain_count=plain_count, plain_events=plain_events, plain_total=plain_total
    )
    if summary.exact + summary.left_censored + summary.interval_censored + plain_events == 0:
        points = np.where(rows.exact, rows.yv, np.maximum(rows.lo, 0.0))
        try:
            total: float | None = fsum([*points.tolist(), rows.plain_total])
        except OverflowError:
            total = None
        if total is None and family is FamilyId.EXPONENTIAL:
            return CensoredRefusal("NUMERICAL_OVERFLOW")
        return CensoredRefusal("NO_OBSERVED_EVENTS", total)
    if has_invalid_support(family, values):
        return CensoredRefusal("INVALID_SUPPORT")
    if family is FamilyId.EXPONENTIAL and exponential_closed_form_applies(rows):
        return _exponential_closed_form(rows)
    start = pilot(rows)
    if start.failure is not None:
        return CensoredRefusal(start.failure)
    if family is FamilyId.WEIBULL_MIN and weibull_profile_applies(rows):
        return _weibull_profile(values, rows)
    return _newton(family, values, rows, summary, start)


def _newton(
    family: FamilyId,
    values: Sequence[Any],
    rows: CensoredRows,
    summary: ObservationSummary,
    start: Pilot,
) -> CensoredEstimate | CensoredRefusal:
    scaled = rows.scaled(start.centre, start.scale)
    if scaled is None:
        return CensoredRefusal("OPTIMIZER_EXHAUSTED")
    right_or_two_sided = summary.right_truncated + summary.two_sided_truncated > 0
    concave = concave_in_coordinates(family, summary.truncated > 0, right_or_two_sided)
    found = maximize(
        family,
        lambda theta: theta_sums(scaled, theta),
        start.start,
        concave=concave,
        tolerance=GAMMA_TOLERANCE if family is FamilyId.GAMMA else TOLERANCE,
    )
    if found is None or found.run.status != "CONVERGED":
        return CensoredRefusal("OPTIMIZER_EXHAUSTED" if found is None else found.run.status)
    run = found.run
    diagnostics = FitDiagnostics(
        found.passes,
        run.iterations,
        run.shifted,
        run.stencil_failures,
        found.starts,
        found.distinct,
    )
    optimum = OptimumKind.GLOBAL_MAXIMUM if concave else OptimumKind.STATIONARY_POINT
    parameters = to_parameters(family, run.theta, start.centre, start.scale)
    total_time = start.pseudo_total if family is FamilyId.EXPONENTIAL else None
    return _finish(family, values, rows, parameters, optimum, diagnostics, total_time)


__all__ = [
    "PARAMETER_NAMES",
    "CensoredEstimate",
    "CensoredRefusal",
    "estimate",
    "exponential_closed_form_applies",
    "has_invalid_support",
    "to_parameters",
    "weibull_profile_applies",
]
