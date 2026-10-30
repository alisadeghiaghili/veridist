"""Helpers shared by the censored log-likelihood tests: building observations from cases."""

from __future__ import annotations

import math
from typing import Any

from veridist.domain.lifetimes import (
    ExactLifetime,
    IntervalCensoredLifetime,
    LeftCensoredLifetime,
    RightCensoredLifetime,
)
from veridist.domain.values import (
    ExactValue,
    IntervalCensoredValue,
    LeftCensoredValue,
    RightCensoredValue,
)
from veridist.families.registry import FamilyId
from veridist.statistics.lifetime_log_likelihood import (
    reduce_lifetime_log_likelihood_chunks,
    reduce_value_log_likelihood_chunks,
)

LIFETIME_FAMILIES = ("exponential", "weibull_min", "lognormal", "gamma")
VALUE_FAMILIES = ("normal", "gumbel_right")
TYPES = {
    True: {
        "exact": ExactLifetime,
        "right": RightCensoredLifetime,
        "left": LeftCensoredLifetime,
        "interval": IntervalCensoredLifetime,
    },
    False: {
        "exact": ExactValue,
        "right": RightCensoredValue,
        "left": LeftCensoredValue,
        "interval": IntervalCensoredValue,
    },
}


def observation(
    family: str,
    kind: str,
    point: float,
    upper: float | None,
    window: tuple[float, float] | None,
) -> Any:
    """The observation of ``kind`` at ``point`` (and ``upper``) with ``window`` (``None``: none)."""

    build = TYPES[family in LIFETIME_FAMILIES][kind]
    keywords: dict[str, float] = {}
    if window is not None:
        keywords = {"truncation_lower": window[0], "truncation_upper": window[1]}
    if kind == "interval":
        return build(point, upper, **keywords)
    return build(point, **keywords)


def contribution(
    family: str, parameters: dict[str, float], observations: Any, **options: Any
) -> Any:
    """The result of the matching streaming reducer for ``observations`` as one chunk."""

    reducer = (
        reduce_lifetime_log_likelihood_chunks
        if family in LIFETIME_FAMILIES
        else reduce_value_log_likelihood_chunks
    )
    return reducer(FamilyId(family), [list(observations)], **options, **parameters)


def term(family: str, parameters: dict[str, float], one: Any) -> float:
    """The contribution of one observation (the total of a one-row reduction)."""

    result = contribution(family, parameters, [one])
    assert result.observation_count == 1, result
    return float(result.total_log_likelihood)


def scaled_error(computed: float, expected: float, *magnitudes: float) -> float:
    """``|computed - expected|`` relative to ``max(1, |expected|, magnitudes...)``."""

    scale = max(1.0, abs(expected), *(abs(m) for m in magnitudes))
    if math.isinf(expected) or math.isinf(computed):
        return 0.0 if computed == expected else math.inf
    return abs(computed - expected) / scale
