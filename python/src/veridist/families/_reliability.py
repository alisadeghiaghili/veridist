"""Shared validation and deterministic scalar optimization for reliability MLEs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from math import exp, inf, isfinite
from typing import TypeVar

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime
from veridist.domain.values import ExactValue, RealObservation, RightCensoredValue
from veridist.engine.errors import CapabilityCode, CapabilityError

_Observation = TypeVar("_Observation")


def admitted_observations(
    observations: Iterable[LifetimeObservation],
    *,
    frequency_weights: Iterable[int] | None,
    analytic_weights: object | None,
    censoring: str,
    truncation: object | None,
) -> tuple[LifetimeObservation, ...]:
    """Validate v1 semantics and expand integer frequencies exactly.

    Only ``ExactLifetime`` and ``RightCensoredLifetime`` are admitted; any other
    observation (including the real-valued types) raises ``TypeError``.
    """

    return _admit(
        observations,
        frequency_weights=frequency_weights,
        analytic_weights=analytic_weights,
        censoring=censoring,
        truncation=truncation,
        kinds=(ExactLifetime, RightCensoredLifetime),
        message="observations must be exact or independently right-censored lifetimes",
    )


def admitted_real_observations(
    observations: Iterable[RealObservation],
    *,
    frequency_weights: Iterable[int] | None,
    analytic_weights: object | None,
    censoring: str,
    truncation: object | None,
) -> tuple[RealObservation, ...]:
    """Validate v1 semantics and expand integer frequencies exactly, for real values.

    Only ``ExactValue`` and ``RightCensoredValue`` are admitted; any other
    observation (including the lifetime types) raises ``TypeError``.
    """

    return _admit(
        observations,
        frequency_weights=frequency_weights,
        analytic_weights=analytic_weights,
        censoring=censoring,
        truncation=truncation,
        kinds=(ExactValue, RightCensoredValue),
        message="observations must be exact or independently right-censored real values",
    )


def _admit(
    observations: Iterable[_Observation],
    *,
    frequency_weights: Iterable[int] | None,
    analytic_weights: object | None,
    censoring: str,
    truncation: object | None,
    kinds: tuple[type, type],
    message: str,
) -> tuple[_Observation, ...]:
    if analytic_weights is not None:
        raise CapabilityError(CapabilityCode.ANALYTIC_WEIGHTS_UNSUPPORTED)
    if truncation is not None:
        raise CapabilityError(CapabilityCode.TRUNCATION_UNSUPPORTED)
    unsupported = {"interval": CapabilityCode.INTERVAL_CENSORING_UNSUPPORTED,
                   "left": CapabilityCode.LEFT_CENSORING_UNSUPPORTED}
    if censoring in unsupported:
        raise CapabilityError(unsupported[censoring])
    if censoring != "right":
        raise ValueError("censoring must be 'right', 'left', or 'interval'")
    values = tuple(observations)
    if any(type(value) not in kinds for value in values):
        raise TypeError(message)
    if frequency_weights is None:
        return values
    weights = tuple(frequency_weights)
    if len(weights) != len(values):
        raise ValueError("frequency_weights must match observations")
    expanded: list[_Observation] = []
    for value, weight in zip(values, weights, strict=True):
        if isinstance(weight, bool) or not isinstance(weight, int) or weight < 0:
            raise TypeError("frequency_weights must be non-negative built-in integers")
        expanded.extend((value,) * weight)
    return tuple(expanded)


def bounded_maximize(
    objective: Callable[[float], float], *, lower: float, upper: float, steps: int = 80
) -> tuple[float, float, bool]:
    """Maximize a finite scalar objective by deterministic golden-section search.

    Returns ``(point, value, at_boundary)``. ``at_boundary`` is true when the
    final point lies within ``1e-6 * (upper - lower)`` of either bound, which
    signals that the search range -- not an interior critical point -- decided
    the result.
    """

    if not (isfinite(lower) and isfinite(upper) and lower < upper):
        raise ValueError("optimization bounds must be finite and ordered")
    ratio = (5.0**0.5 - 1.0) / 2.0
    left, right = lower, upper
    first, second = right - ratio * (right - left), left + ratio * (right - left)
    first_value, second_value = objective(first), objective(second)
    for _ in range(steps):
        if first_value < second_value:
            left, first, first_value = first, second, second_value
            second = left + ratio * (right - left)
            second_value = objective(second)
        else:
            right, second, second_value = second, first, first_value
            first = right - ratio * (right - left)
            first_value = objective(first)
    point = (left + right) / 2.0
    value = objective(point)
    tolerance = 1e-6 * (upper - lower)
    at_boundary = min(point - lower, upper - point) <= tolerance
    return point, value, at_boundary


# Each expansion at least doubles the bracket width, so this bound is far above
# what any finite hard limit can require; it only guarantees termination.
_MAX_BRACKET_EXPANSIONS = 64


def expand_bracket(
    objective: Callable[[float], float],
    *,
    lower: float,
    upper: float,
    hard_lower: float,
    hard_upper: float,
    steps: int = 80,
) -> tuple[float, float, bool]:
    """Maximize ``objective``, widening a boundary-bound bracket up to a hard limit.

    Starts the search at ``[lower, upper]``. Whenever the golden-section result
    lands on a bound, the side that hit the bound is widened (doubling that
    side's contribution to the interval width) and the search restarts, up to
    ``[hard_lower, hard_upper]``. Returns ``(point, value, boundary_solution)``;
    ``boundary_solution`` is true only when the hard limit is reached and the
    optimum still lies on a bound -- an interior optimum was never found.
    """

    if not (
        isfinite(hard_lower)
        and isfinite(hard_upper)
        and hard_lower <= lower < upper <= hard_upper
    ):
        raise ValueError("hard limits must contain and bound the starting bracket")
    current_lower, current_upper = lower, upper
    point = value = 0.0
    for _ in range(_MAX_BRACKET_EXPANSIONS):
        point, value, at_boundary = bounded_maximize(
            objective, lower=current_lower, upper=current_upper, steps=steps
        )
        if not at_boundary:
            return point, value, False
        width = current_upper - current_lower
        tolerance = 1e-6 * width
        hit_lower = (point - current_lower) <= tolerance
        hit_upper = (current_upper - point) <= tolerance
        next_lower, next_upper = current_lower, current_upper
        expanded = False
        if hit_lower and current_lower > hard_lower:
            next_lower = max(hard_lower, current_lower - width)
            expanded = True
        if hit_upper and current_upper < hard_upper:
            next_upper = min(hard_upper, current_upper + width)
            expanded = True
        if not expanded:
            return point, value, True
        current_lower, current_upper = next_lower, next_upper
    return point, value, True


def positive_support(values: tuple[LifetimeObservation, ...]) -> bool:
    """Return whether all fixed-location reliability times are strictly positive."""

    return all(isfinite(value.time) and value.time > 0.0 for value in values)


def exp_or_inf(exponent: float) -> float:
    """Return ``exp(exponent)``, or ``inf`` instead of ``OverflowError`` when it overflows.

    A log-likelihood that would need ``inf`` here is ``-inf`` in the limit, which
    a maximizer handles; an exception would abort the whole search instead.
    """

    return inf if exponent > 709.0 else exp(exponent)


def is_degenerate_sample(exact: Sequence[float], censored: Sequence[float]) -> bool:
    """Return whether the likelihood is unbounded because every event is identical.

    With one common exact value and every censoring point at or below it, a
    scale parameter can shrink to zero while the likelihood grows without
    bound, so no finite maximum exists.
    """

    common = exact[0]
    return max(exact) == min(exact) and all(value <= common for value in censored)


def maximize_nested(
    at_outer: Callable[[float], Callable[[float], float]],
    *,
    outer_bounds: tuple[float, float],
    outer_hard_limits: tuple[float, float],
    inner_bounds: tuple[float, float],
    inner_hard_limits: tuple[float, float],
    steps: int = 60,
) -> tuple[float, float, float, bool]:
    """Maximize a two-parameter log-likelihood by profiling out an inner variable.

    ``at_outer(outer)`` returns the log-likelihood as a function of the inner
    variable for that outer value, so quantities that depend only on the outer
    variable are computed once per outer value rather than once per inner one.
    The inner variable is profiled out by a golden-section search, then the
    outer variable is maximized over the profile; both searches widen a
    boundary-bound bracket up to its hard limit with :func:`expand_bracket`.

    Returns ``(outer, inner, value, boundary)``.  ``boundary`` is true when
    either variable still sits on a hard limit at the end, which means that no
    interior optimum was found.
    """

    def profile(outer: float) -> float:
        _, value, _ = expand_bracket(
            at_outer(outer),
            lower=inner_bounds[0],
            upper=inner_bounds[1],
            hard_lower=inner_hard_limits[0],
            hard_upper=inner_hard_limits[1],
            steps=steps,
        )
        return value

    outer, _, outer_boundary = expand_bracket(
        profile,
        lower=outer_bounds[0],
        upper=outer_bounds[1],
        hard_lower=outer_hard_limits[0],
        hard_upper=outer_hard_limits[1],
        steps=steps,
    )
    inner, value, inner_boundary = expand_bracket(
        at_outer(outer),
        lower=inner_bounds[0],
        upper=inner_bounds[1],
        hard_lower=inner_hard_limits[0],
        hard_upper=inner_hard_limits[1],
        steps=steps,
    )
    return outer, inner, value, outer_boundary or inner_boundary
