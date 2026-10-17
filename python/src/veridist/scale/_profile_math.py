"""The arithmetic of profile-likelihood intervals of the multi-pass families, with no data in it.

Both planned families have a likelihood that depends on the sample only through the function
``S0(k) = sum exp(k * w)`` of the centred observations ``w <= 0`` (:mod:`veridist.scale._planner`).
With ``a`` the logarithm of ``S0 / n`` at the maximum over the location-like parameter, the
log-likelihood is, up to a constant that does not depend on the parameters,

    l(k, a) = n log k + k W - n a - exp(-a) S0(k),        W = sum w,

in the coordinates ``(t, a)`` with ``k = exp(t)``. For a fixed ``t`` it is maximal at
``a = psi(t) = log(S0(k) / n)``, where it equals ``n t + k W - n psi - n``; that is the profile
log-likelihood of ``t`` and its deviance from the maximum ``D(t) = 2 (l_max - l_profile(t))``.
With ``a = psi(t) + y`` the deviance of any point is

    D(t, a) = D(t) + 2 n (y + exp(-y) - 1),

so the region ``D(t, a) <= delta`` (``delta`` the chi-square quantile of the level) has, for each
``t`` with ``D(t) <= delta``, the two boundary points ``y_-(t) <= 0 <= y_+(t)`` that solve
``y + exp(-y) - 1 = (delta - D(t)) / (2 n)``. A profile-likelihood interval of a function of the
parameters is the range of that function over the region, and for a function that is monotone in
``a`` at every ``t`` the range is attained on the boundary curve. Two ranges are needed for the
two families: the shape (``t`` itself: the roots of ``D(t) = delta``) and every other quantity
(the minimum and the maximum of a smooth function of ``t`` along the boundary).

Surrogate. ``psi`` is analytic, and everything above is a closed form in ``psi(t)``. It is read
from the data only at chosen values of ``t``, in passes, and is interpolated between them by the
polynomial through Chebyshev-Lobatto nodes of a window around the estimate
(:class:`Surrogate`), whose degrees double with the nodes nested, so that a refinement reads only
the new nodes. The accuracy of the polynomial is judged by the difference between two degrees,
in units of deviance, and every endpoint that is reported is re-evaluated from exact sums read at
that very point, so that the surrogate decides where to look and never what is reported.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import cos, exp, expm1, fsum, log1p, pi, sqrt
from typing import Final

LOG_DOMAIN_LIMIT: Final = 20.0
"""The search domain of the parameter ``t = log k``: within plus and minus this limit."""

SURROGATE_DEGREES: Final = (8, 16, 32, 64, 128)
"""The degrees of the polynomial surrogate; each doubles the previous one and keeps its nodes."""

SCAN_POINTS: Final = 256
"""Points of the scan that locates an extremum along a boundary curve."""

GOLDEN_ITERATIONS: Final = 60
"""Golden-section iterations that refine a located extremum."""

BISECTION_LIMIT: Final = 200
"""Bisection cap for a root of the profile deviance; resolution is reached long before."""

OFFSET_ITERATIONS: Final = 100
"""Newton iteration cap for a boundary offset."""

SERIES_LIMIT: Final = 0.5
"""Offsets below this use the series of ``y + exp(-y) - 1``."""

SERIES_TERMS: Final = 24
"""Terms of that series (the first omitted one is below ``0.5**25 / 25!``)."""

SURROGATE_TOLERANCE: Final = 1e-6
"""Deviance the two highest degrees of the surrogate may differ by, before the noise allowance."""

CONFIRMATION_TOLERANCE: Final = 1e-5
"""Deviance a confirmed point may differ from the surrogate by, before the noise allowance."""

NOISE_UNIT: Final = 2.0**-40
"""Allowance per observation for rounding in the sums, in units of deviance."""


def noise_allowance(count: int) -> float:
    """The deviance that rounding of exact sums to binary64 can move a profile by."""

    return count * NOISE_UNIT


def lobatto_logs(center: float, half_width: float, degree: int) -> tuple[float, ...]:
    """The ``degree + 1`` Chebyshev-Lobatto nodes of the window, ascending, ends included."""

    return tuple(center - half_width * cos(pi * (index / degree)) for index in range(degree + 1))


def _barycentric_weights(degree: int) -> tuple[float, ...]:
    return tuple(
        (-1.0 if index % 2 else 1.0) * (0.5 if index in (0, degree) else 1.0)
        for index in range(degree + 1)
    )


@dataclass(frozen=True, slots=True)
class Surrogate:
    """The polynomial through ``values`` at the Lobatto nodes ``logs`` of one window."""

    center: float
    half_width: float
    degree: int
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.degree not in (SURROGATE_DEGREES[0] // 2, *SURROGATE_DEGREES):
            raise ValueError("degree must be one of the surrogate degrees or half the lowest")
        if len(self.values) != self.degree + 1:
            raise ValueError("a surrogate has one value per node")
        if not self.half_width > 0.0:
            raise ValueError("the window must have a positive half width")

    @property
    def logs(self) -> tuple[float, ...]:
        return lobatto_logs(self.center, self.half_width, self.degree)

    @property
    def low(self) -> float:
        return self.center - self.half_width

    @property
    def high(self) -> float:
        return self.center + self.half_width

    def value(self, point: float) -> float:
        """The polynomial at ``point`` (barycentric form; a node gives its value exactly)."""

        numerator = denominator = 0.0
        for node, weight, value in zip(
            self.logs, _barycentric_weights(self.degree), self.values, strict=True
        ):
            offset = point - node
            if offset == 0.0:
                return value
            numerator += weight * value / offset
            denominator += weight / offset
        return numerator / denominator

    def coarsened(self) -> Surrogate:
        """The surrogate of half the degree through every other node."""

        return Surrogate(self.center, self.half_width, self.degree // 2, self.values[::2])

    def gap(self) -> float:
        """The largest difference between this polynomial and its coarsened one.

        Measured at the midpoints between the nodes of this degree, where an interpolant is
        least accurate.
        """

        coarse = self.coarsened()
        return max(
            abs(self.value(point) - coarse.value(point))
            for point in (
                self.center - self.half_width * cos(pi * (index + 0.5) / self.degree)
                for index in range(self.degree)
            )
        )


@dataclass(frozen=True, slots=True)
class LikelihoodForm:
    """The constants of the profile deviance: the count, ``W`` and the maximum.

    ``reduced_maximum`` is the log-likelihood at the maximum without the constant that does not
    depend on the parameters, that is ``n t + k W - n psi - n`` at the maximum.
    """

    count: int
    sum_w: float
    reduced_maximum: float

    def deviance(self, point: float, psi: float) -> float:
        """The profile deviance at ``t = point`` given ``psi = log(S0 / n)`` there."""

        count = float(self.count)
        return 2.0 * fsum(
            (
                self.reduced_maximum,
                -count * point,
                -exp(point) * self.sum_w,
                count * psi,
                count,
            )
        )


def surrogate_deviance(form: LikelihoodForm, surrogate: Surrogate, point: float) -> float:
    """The profile deviance at ``point`` from the surrogate of ``psi``."""

    return form.deviance(point, surrogate.value(point))


def root_between(function: Callable[[float], float], outside: float, inside: float) -> float:
    """A root of ``function`` between ``outside`` (where it is at least zero) and ``inside``.

    ``function(inside)`` is negative. The bracket is halved at most ``BISECTION_LIMIT`` times and
    stops when its midpoint is no longer strictly inside it; the midpoint of the last bracket is
    returned.
    """

    for _ in range(BISECTION_LIMIT):
        middle = 0.5 * (outside + inside)
        if not min(outside, inside) < middle < max(outside, inside):
            break
        if function(middle) >= 0.0:
            outside = middle
        else:
            inside = middle
    return 0.5 * (outside + inside)


def boundary_offsets(slack: float) -> tuple[float, float]:
    """The roots ``y_- <= 0 <= y_+`` of ``y + exp(-y) - 1 = slack`` (``slack`` at least zero).

    ``g(s) = f(+-s) - slack`` is increasing and convex in ``s >= 0`` for both signs, so each root
    is reached by Newton steps from a start at or beyond it, which approach it monotonically. A
    slack of zero gives ``(0.0, 0.0)``.
    """

    if slack <= 0.0:
        return 0.0, 0.0
    return -_offset(slack, -1.0), _offset(slack, 1.0)


def _excess_function(offset: float) -> float:
    """``offset + exp(-offset) - 1``, without cancellation near zero.

    Below ``SERIES_LIMIT`` the series ``sum_{k>=2} (-offset)**k / k!`` is summed to
    ``SERIES_TERMS`` terms, whose truncation is far below binary64 resolution there.
    """

    if abs(offset) >= SERIES_LIMIT:
        return offset + expm1(-offset)
    term = 0.5 * offset * offset
    total = term
    for order in range(3, SERIES_TERMS + 1):
        term *= -offset / order
        total += term
    return total


def _offset(slack: float, sign: float) -> float:
    def excess(distance: float) -> float:
        return _excess_function(sign * distance) - slack

    def slope(distance: float) -> float:
        return -expm1(-sign * distance) * sign

    # Both starts are at or beyond the root: s - 1 + exp(-s) >= s - 1, so ``1 + slack`` is one for
    # the positive offset; exp(s) - s - 1 >= s**2 / 2 and, for r = slack,
    # exp(1 + log1p(r)) - (1 + log1p(r)) - 1 >= r, so the smaller of those two is one for the
    # negative offset.
    upper = 1.0 + slack if sign > 0.0 else min(sqrt(2.0 * slack), 1.0 + log1p(slack))
    for _ in range(OFFSET_ITERATIONS):
        step = excess(upper) / slope(upper)
        if not 0.0 < step < upper:
            break
        upper -= step
        if step <= 4.0 * 2.0**-53 * upper:
            break
    return upper


@dataclass(frozen=True, slots=True)
class Boundary:
    """The boundary points of the region for one ``t``: ``a_- <= a_+`` and the slack used."""

    lower: float
    upper: float
    slack: float


def boundary_at(form: LikelihoodForm, delta: float, point: float, psi: float) -> Boundary:
    """The boundary of ``D(t, a) <= delta`` at ``t = point`` given ``psi`` there.

    A point whose profile deviance exceeds ``delta`` (by rounding, at the end of the range) has
    no slack: both boundary values are ``psi``.
    """

    slack = max((delta - form.deviance(point, psi)) / (2.0 * form.count), 0.0)
    below, above = boundary_offsets(slack)
    return Boundary(psi + below, psi + above, slack)


def extremum(
    function: Callable[[float], float], low: float, high: float, *, maximize: bool
) -> tuple[float, float]:
    """The point of ``[low, high]`` where ``function`` is largest (or smallest) and its value.

    The function is scanned at ``SCAN_POINTS + 1`` points that cluster at the ends (equal steps
    of an angle), and the best point's two neighbours bracket a golden-section search of at most
    ``GOLDEN_ITERATIONS`` iterations. The best value seen is returned, so the result is never worse
    than the scan.
    """

    sign = 1.0 if maximize else -1.0
    middle, half = 0.5 * (low + high), 0.5 * (high - low)

    def at_angle(angle: float) -> float:
        return middle - half * cos(angle)

    angles = [pi * index / SCAN_POINTS for index in range(SCAN_POINTS + 1)]
    scores = [sign * function(at_angle(angle)) for angle in angles]
    best_index = max(range(len(scores)), key=scores.__getitem__)
    best_angle, best_score = angles[best_index], scores[best_index]
    left = angles[max(best_index - 1, 0)]
    right = angles[min(best_index + 1, SCAN_POINTS)]
    ratio = 0.5 * (sqrt(5.0) - 1.0)
    first = right - ratio * (right - left)
    second = left + ratio * (right - left)
    score_first = sign * function(at_angle(first))
    score_second = sign * function(at_angle(second))
    for _ in range(GOLDEN_ITERATIONS):
        if score_first > best_score:
            best_angle, best_score = first, score_first
        if score_second > best_score:
            best_angle, best_score = second, score_second
        if score_first >= score_second:
            right, second, score_second = second, first, score_first
            first = right - ratio * (right - left)
            score_first = sign * function(at_angle(first))
        else:
            left, first, score_first = first, second, score_second
            second = left + ratio * (right - left)
            score_second = sign * function(at_angle(second))
    for angle, score in ((first, score_first), (second, score_second)):
        if score > best_score:
            best_angle, best_score = angle, score
    return at_angle(best_angle), sign * best_score


__all__ = [
    "BISECTION_LIMIT",
    "CONFIRMATION_TOLERANCE",
    "GOLDEN_ITERATIONS",
    "LOG_DOMAIN_LIMIT",
    "NOISE_UNIT",
    "OFFSET_ITERATIONS",
    "SCAN_POINTS",
    "SERIES_LIMIT",
    "SERIES_TERMS",
    "SURROGATE_DEGREES",
    "SURROGATE_TOLERANCE",
    "Boundary",
    "LikelihoodForm",
    "Surrogate",
    "boundary_at",
    "boundary_offsets",
    "extremum",
    "lobatto_logs",
    "noise_allowance",
    "root_between",
    "surrogate_deviance",
]
