"""Shared validation and deterministic scalar optimization for reliability MLEs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from math import isfinite

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime
from veridist.engine.errors import CapabilityCode, CapabilityError


def admitted_observations(
    observations: Iterable[LifetimeObservation],
    *,
    frequency_weights: Iterable[int] | None,
    analytic_weights: object | None,
    censoring: str,
    truncation: object | None,
) -> tuple[LifetimeObservation, ...]:
    """Validate v1 semantics and expand integer frequencies exactly."""

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
    if any(type(value) not in {ExactLifetime, RightCensoredLifetime} for value in values):
        raise TypeError("observations must be exact or independently right-censored lifetimes")
    if frequency_weights is None:
        return values
    weights = tuple(frequency_weights)
    if len(weights) != len(values):
        raise ValueError("frequency_weights must match observations")
    expanded: list[LifetimeObservation] = []
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
    while True:
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


def positive_support(values: tuple[LifetimeObservation, ...]) -> bool:
    """Return whether all fixed-location reliability times are strictly positive."""

    return all(isfinite(value.time) and value.time > 0.0 for value in values)
