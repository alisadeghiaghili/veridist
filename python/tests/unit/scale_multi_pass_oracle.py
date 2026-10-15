"""Independent oracles and shared data for the multi-pass scale fit tests.

Nothing here uses the code under test. The references rebuild each estimate from the
statistical definition: the root of the profile equation at 50 digits with ``mpmath``,
evaluated on the exact logarithms (Weibull) or values (Gumbel) of the binary64 data.
"""

from __future__ import annotations

from collections.abc import Callable
from math import exp, fsum
from typing import Any

import mpmath
import numpy as np

from veridist.scale._planner import PassRequest, ProfileSums

mpmath.mp.dps = 50


def weibull_sample(shape: float, scale: float, size: int, seed: int) -> np.ndarray:
    """Seeded Weibull-minimum draws."""

    return scale * np.random.default_rng(seed).weibull(shape, size)


def gumbel_sample(location: float, scale: float, size: int, seed: int) -> np.ndarray:
    """Seeded right-Gumbel draws."""

    return np.random.default_rng(seed).gumbel(location, scale, size)


def mp_weibull(values: np.ndarray, start: float) -> tuple[Any, Any, Any]:
    """``(shape, scale, log-likelihood)`` of exact Weibull data at 50 digits.

    The shape is the root of ``n/k - n sum(y e^{ky}) / sum(e^{ky}) + sum(y)`` for the exact
    logarithms ``y`` (centred on their mean, which does not move the root) found from
    ``start``; the scale and the log-likelihood are the profile expressions at the root.
    """

    logs = [mpmath.log(mpmath.mpf(float(value))) for value in values]
    count = len(logs)
    total = mpmath.fsum(logs)
    center = total / count
    centred = [item - center for item in logs]

    def parts(k: Any) -> tuple[Any, Any]:
        weights = [mpmath.exp(k * item) for item in centred]
        return mpmath.fsum(weights), mpmath.fsum(
            w * y for w, y in zip(weights, centred, strict=True)
        )

    def score(k: Any) -> Any:
        s0, s1 = parts(k)
        return count / k - count * s1 / s0

    shape = mpmath.findroot(score, mpmath.mpf(start), tol=mpmath.mpf(10) ** -40)
    s0, _ = parts(shape)
    scale = mpmath.exp(center + mpmath.log(s0 / count) / shape)
    likelihood = (
        count * mpmath.log(shape) - count * shape * mpmath.log(scale) + (shape - 1) * total - count
    )
    return shape, scale, likelihood


def mp_gumbel(values: np.ndarray, start: float) -> tuple[Any, Any, Any, Any]:
    """``(location, scale, log-likelihood, minimum)`` of exact right-Gumbel data at 50 digits.

    The scale is the root of ``sigma - mean(x) + sum(x e^{-x/sigma}) / sum(e^{-x/sigma})``
    found from ``start`` and the location is ``-sigma log(mean(e^{-x/sigma}))``. The data
    are divided by their largest magnitude for the search, so any magnitude is solvable.
    """

    raw = [mpmath.mpf(float(value)) for value in values]
    count = len(raw)
    unit = max(abs(p) for p in raw)
    points = [p / unit for p in raw]
    low = min(points)
    mean = mpmath.fsum(points) / count

    def parts(sigma: Any) -> tuple[Any, Any]:
        weights = [mpmath.exp(-(p - low) / sigma) for p in points]
        return mpmath.fsum(weights), mpmath.fsum(
            w * p for w, p in zip(weights, points, strict=True)
        )

    def score(sigma: Any) -> Any:
        s0, s1 = parts(sigma)
        return sigma - mean + s1 / s0

    sigma = mpmath.findroot(score, mpmath.mpf(start) / unit, tol=mpmath.mpf(10) ** -40)
    s0, _ = parts(sigma)
    location = low - sigma * mpmath.log(s0 / count)
    likelihood = (
        -count * mpmath.log(sigma * unit)
        - mpmath.fsum((p - location) / sigma for p in points)
        - mpmath.fsum(mpmath.exp(-(p - location) / sigma) for p in points)
    )
    return location * unit, sigma * unit, likelihood, low * unit


def relative(value: float, reference: Any) -> float:
    """``|value - reference| / |reference|`` as a float."""

    return float(abs(mpmath.mpf(value) - reference) / abs(reference))


def sums_of(centred: np.ndarray, parameter: float) -> ProfileSums:
    """The sums ``S0``, ``S1``, ``S2`` of centred observations at ``parameter``, by ``fsum``."""

    weights = [exp(parameter * float(item)) for item in centred]
    values = [float(item) for item in centred]
    return ProfileSums(
        fsum(weights),
        fsum(w * y for w, y in zip(weights, values, strict=True)),
        fsum(w * y * y for w, y in zip(weights, values, strict=True)),
    )


class Recorder:
    """An evaluator over an array of centred observations that records every request."""

    def __init__(self, centred: np.ndarray) -> None:
        self.centred = centred
        self.requests: list[PassRequest] = []

    def __call__(self, request: PassRequest) -> list[ProfileSums]:
        self.requests.append(request)
        return [sums_of(self.centred, parameter) for parameter in request.parameters]

    def roles(self) -> list[str]:
        return [request.role.value for request in self.requests]


def synthetic(
    score_of: Callable[[float], float], mean_w: float
) -> Callable[[PassRequest], list[ProfileSums]]:
    """An evaluator whose score is ``score_of(k)`` and whose slope is ``-1 / k**2``.

    The sums are ``S0 = 1``, ``S1 = 1/k + mean_w - score_of(k)`` and ``S2 = S1**2``, so the
    weighted variance is zero; this lets a test choose any decreasing score.
    """

    def evaluate(request: PassRequest) -> list[ProfileSums]:
        out = []
        for parameter in request.parameters:
            first = 1.0 / parameter + mean_w - score_of(parameter)
            out.append(ProfileSums(1.0, first, first * first))
        return out

    return evaluate
