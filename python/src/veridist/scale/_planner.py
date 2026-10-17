"""The multi-pass planner: pass budget, pass ledger and the safeguarded profile root finder.

A multi-pass fit reads its source several times. Each read is a *pass*: the partitions are
folded in canonical order into exact accumulators (:mod:`veridist.scale._accumulator`) and
merged, so what a pass returns depends on the multiset of the rows and on the candidate
parameter values it was given, and on nothing else. The planner decides which candidates
the next pass receives from the finalized exact sums of the passes before it, so the whole
sequence of passes, and the result, is the same for every batch cap, partition layout, file
split and row order.

The profile equation. Both planned families reduce to one equation in a single positive
parameter ``k`` (the Weibull shape; for the right Gumbel the reciprocal of the scale of the
data standardised as the in-memory fit does). For observations that have been centred to
``w <= 0`` (the largest ``w`` is zero) and the exact sums ``S0 = sum e``, ``S1 = sum e*w``,
``S2 = sum e*w*w`` with ``e = exp(k*w)``, the profile score divided by the count is::

    g(k) = 1/k - S1/S0 + mean(w)        g'(k) = -1/k**2 - (S2/S0 - (S1/S0)**2)

``g`` is strictly decreasing, so it has at most one root and a sign change brackets it. The
exponent ``k*w`` is never positive, so ``e`` lies in ``(0, 1]`` and ``S0 >= 1``: nothing
overflows for any ``k`` in the search domain, however spread the data.

The plan, in passes:

* pass 1 reads the pilot state (count, exact sums, extremes);
* pass 2 evaluates ``GRID_POINTS`` candidates, geometric in ``log k``, over
  ``[-INITIAL_LOG_WINDOW, INITIAL_LOG_WINDOW]``;
* if the score has no sign change there, up to ``MAX_EXPANSION_PASSES`` expansion passes
  each evaluate ``GRID_POINTS`` candidates over the next band outward on the side the root
  must lie (``EXPANSION_LOG_LIMITS``), to the search domain ``+-LOG_DOMAIN_LIMIT``; still
  no sign change is the boundary solution;
* with the bracket, the start is the root of the cubic Hermite interpolant of ``g`` between
  its two ends (no data is read for it), and safeguarded Newton iterations follow, one
  pass each (one candidate), at most ``MAX_NEWTON_PASSES``. A Newton step that already
  meets ``CONVERGENCE_TOLERANCE`` is accepted before the bracket safeguard is applied; the
  last pass is evaluated at the iterate that produced it and carries the report.

Bounds. A plan needs at least ``MINIMUM_PASSES`` (the pilot, the grid and one iteration)
and at most ``MAXIMUM_PASSES``. A budget below the minimum is refused before any read
(``PASS_BUDGET_INSUFFICIENT``); a run that would read more passes than the budget is refused
before the excess pass is read (``PASS_BUDGET_EXCEEDED``). Neither returns an estimate.

Convergence. A Newton step of relative size ``t`` leaves an error of order ``t**2``, so a
step below ``CONVERGENCE_TOLERANCE = 1e-8`` is applied (the reported parameter is the
updated one, with an error near binary64 resolution) and the quantities that depend on the
parameter through ``log(S0)`` are carried to it by their second order expansion, whose
neglected term is of order ``t**3``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from math import exp, fsum, log, sqrt
from typing import Final

from veridist.engine.errors import FailureCode
from veridist.engine.pass_budget import PassBudgetError, PassEnforcer
from veridist.engine.provenance import ScalePassProvenance, ScalePassRecord
from veridist.scale._sources import Partition

PASS_BUDGET_DEFAULT: Final = 8
"""The pass budget of a fit unless the caller gives one."""

GRID_POINTS: Final = 8
"""Candidates of a grid or of an expansion pass."""

INITIAL_LOG_WINDOW: Final = 8.0
"""The first grid covers ``log k`` in ``[-8, 8]``, about ``3e-4`` to ``3e3``."""

EXPANSION_LOG_LIMITS: Final = (12.0, 16.0, 20.0)
"""The outer ``log k`` limit of each expansion pass; the last is the search domain."""

LOG_DOMAIN_LIMIT: Final = 20.0
"""The search domain is ``log k`` within plus and minus this limit, as in the in-memory fits."""

MAX_EXPANSION_PASSES: Final = len(EXPANSION_LOG_LIMITS)
MAX_NEWTON_PASSES: Final = 12
"""Iteration cap of the Newton passes."""

CONVERGENCE_TOLERANCE: Final = 1e-8
"""A Newton step whose size relative to the iterate is at most this converges."""

MAX_HERMITE_BISECTIONS: Final = 100
"""Bisection cap for the root of the Hermite interpolant (resolution is reached long before)."""

MINIMUM_PASSES: Final = 3
MAXIMUM_PASSES: Final = 2 + MAX_EXPANSION_PASSES + MAX_NEWTON_PASSES


def plan_identity() -> tuple[object, ...]:
    """The constants that fix the pass sequence, as a canonical record.

    Two runs that share it and the exact sums of every pass take the same passes, so a
    stored pass is only replayed under the plan that made it. Floats are their exact
    hexadecimal form.
    """

    return (
        GRID_POINTS,
        INITIAL_LOG_WINDOW.hex(),
        tuple(limit.hex() for limit in EXPANSION_LOG_LIMITS),
        LOG_DOMAIN_LIMIT.hex(),
        MAX_NEWTON_PASSES,
        CONVERGENCE_TOLERANCE.hex(),
        MAX_HERMITE_BISECTIONS,
    )


class PassRole(StrEnum):
    """What a pass is for; the value is its name in the provenance record."""

    STATISTICS = "statistics"
    GRID = "grid"
    EXPANSION = "expansion"
    ITERATION = "iteration"
    PROFILE_WINDOW = "profile_window"
    PROFILE_REFINEMENT = "profile_refinement"
    PROFILE_CONFIRMATION = "profile_confirmation"
    GOF = "gof"


@dataclass(frozen=True, slots=True)
class PlanBounds:
    """The fewest and most passes the plan can read, published before the first read."""

    minimum: int
    maximum: int


def check_pass_budget(pass_budget: object) -> int:
    """Validate a pass budget: a built-in integer of at least one."""

    if type(pass_budget) is not int:
        raise TypeError("pass_budget must be a built-in integer")
    if pass_budget < 1:
        raise ValueError("pass_budget must be at least 1")
    return pass_budget


def plan_passes(pass_budget: int) -> PlanBounds:
    """The plan's bounds for ``pass_budget``, refusing a budget below the minimum.

    ``PassBudgetError`` with ``PASS_BUDGET_INSUFFICIENT`` and the budget and both bounds in
    its context. Nothing has been read when this is called.
    """

    bounds = PlanBounds(MINIMUM_PASSES, MAXIMUM_PASSES)
    if pass_budget < bounds.minimum:
        raise PassBudgetError(
            FailureCode.PASS_BUDGET_INSUFFICIENT,
            {
                "pass_budget": pass_budget,
                "minimum_passes": bounds.minimum,
                "maximum_passes": bounds.maximum,
            },
        )
    return bounds


class PassLedger:
    """Counts the passes of one fit against its budget and records each of them.

    :meth:`begin` reserves a pass *before* the source is touched (the pass enforcer of the
    engine refuses the pass beyond the budget) and returns the partitions to read, in
    canonical order.
    """

    __slots__ = ("_bounds", "_budget", "_enforcer", "_partitions", "_records")

    def __init__(
        self, partitions: tuple[Partition, ...], pass_budget: int, bounds: PlanBounds
    ) -> None:
        self._partitions = partitions
        self._budget = pass_budget
        self._bounds = bounds
        self._enforcer = PassEnforcer(max_passes=pass_budget)
        self._records: list[ScalePassRecord] = []

    @property
    def pass_count(self) -> int:
        return self._enforcer.actual_pass_count

    def begin(
        self,
        role: PassRole,
        points: int,
        log_low: float | None = None,
        log_high: float | None = None,
        *,
        bracket_log_width: float | None = None,
    ) -> Iterator[Partition]:
        """Reserve one pass and return its partitions; ``PASS_BUDGET_EXCEEDED`` if none is left.

        ``bracket_log_width`` is the natural logarithm of the ratio of the ends of the
        bracket the fit has, if it has one, and is reported in the refusal.
        """

        try:
            partitions = self._enforcer.begin_pass(self._partitions)
        except PassBudgetError as error:
            context = dict(error.context)
            if bracket_log_width is not None:
                context["bracket_log_width"] = bracket_log_width
            raise PassBudgetError(error.code, context) from None
        self._records.append(ScalePassRecord(role.value, points, log_low, log_high))
        return partitions

    def provenance(self) -> ScalePassProvenance:
        """The record of the passes begun so far."""

        return ScalePassProvenance(
            self._budget,
            self._bounds.minimum,
            self._bounds.maximum,
            len(self._records),
            tuple(self._records),
        )


@dataclass(frozen=True, slots=True)
class ProfileSums:
    """The exact sums ``S0``, ``S1``, ``S2`` of one candidate, each rounded once."""

    s0: float
    s1: float
    s2: float


class ProfileRefusal(StrEnum):
    """Why the profile equation has no root the planner can report."""

    BOUNDARY = "boundary"
    """No sign change over the whole search domain: the root is at or beyond a limit."""

    EXHAUSTED = "exhausted"
    """The Newton passes did not converge within their cap."""


@dataclass(frozen=True, slots=True)
class ProfileSolution:
    """The root of the profile equation and what the final pass reports at it.

    ``parameter`` is the root. ``iterate`` is the candidate the final pass was evaluated at
    and ``sums`` its sums; ``log_ratio`` is ``log(S0 / n)`` carried from the iterate to the
    root by its second order expansion.
    """

    parameter: float
    iterate: float
    log_ratio: float
    sums: ProfileSums


@dataclass(frozen=True, slots=True)
class PassRequest:
    """What the planner asks of one pass: its role and the candidates to evaluate.

    ``log_low`` and ``log_high`` are the extreme logarithms of the candidates of a grid or
    an expansion (``None`` for an iteration) and ``bracket_log_width`` the logarithm of the
    ratio of the ends of the bracket, once there is one.
    """

    role: PassRole
    parameters: tuple[float, ...]
    log_low: float | None = None
    log_high: float | None = None
    bracket_log_width: float | None = None


Evaluator = Callable[[PassRequest], list[ProfileSums]]
"""Reads one pass over the source for the requested candidates and returns their sums in order."""


def score(parameter: float, sums: ProfileSums, mean_w: float) -> tuple[float, float]:
    """``g(k)`` and ``g'(k)`` (the profile score per observation) from the sums."""

    first = sums.s1 / sums.s0
    second = sums.s2 / sums.s0
    variance = max(second - first * first, 0.0)
    return (
        fsum((1.0 / parameter, -first, mean_w)),
        -fsum((1.0 / (parameter * parameter), variance)),
    )


@dataclass(frozen=True, slots=True)
class ProfileNode:
    """A candidate with its score ``g`` and the slope ``g'``."""

    parameter: float
    value: float
    slope: float


def hermite_root(low: ProfileNode, high: ProfileNode) -> float:
    """A root of the cubic Hermite interpolant of ``g`` in ``log k`` on ``[low, high]``.

    ``g(low) > 0 >= g(high)``. The candidates of a grid are geometric, so the interpolant is
    built in the logarithm of the parameter, from the values and the slopes ``k g'(k)`` at the
    ends only (no data is read). The bisection is capped at ``MAX_HERMITE_BISECTIONS``
    halvings and stops when the midpoint is no longer strictly inside the interval.
    """

    start = log(low.parameter)
    width = log(high.parameter) - start
    low_slope = low.parameter * low.slope
    high_slope = high.parameter * high.slope

    def interpolant(point: float) -> float:
        t = (point - start) / width
        t2, t3 = t * t, t * t * t
        return (
            (2.0 * t3 - 3.0 * t2 + 1.0) * low.value
            + (t3 - 2.0 * t2 + t) * width * low_slope
            + (-2.0 * t3 + 3.0 * t2) * high.value
            + (t3 - t2) * width * high_slope
        )

    left, right = start, start + width
    for _ in range(MAX_HERMITE_BISECTIONS):
        middle = 0.5 * (left + right)
        if not left < middle < right:
            break
        if interpolant(middle) > 0.0:
            left = middle
        else:
            right = middle
    return min(max(exp(0.5 * (left + right)), low.parameter), high.parameter)


def _initial_logs() -> tuple[float, ...]:
    count = GRID_POINTS
    return tuple(
        -INITIAL_LOG_WINDOW + 2.0 * INITIAL_LOG_WINDOW * index / (count - 1)
        for index in range(count)
    )


def _band_logs(inner: float, outer: float, *, upward: bool) -> tuple[float, ...]:
    """Ascending ``log k`` of one expansion band: ``GRID_POINTS`` points, outer end included."""

    count = GRID_POINTS
    if upward:
        return tuple(inner + (outer - inner) * (index + 1) / count for index in range(count))
    return tuple(-outer + (outer - inner) * index / count for index in range(count))


def _bracket(nodes: list[ProfileNode]) -> tuple[ProfileNode, ProfileNode] | None:
    """The adjacent pair with ``g > 0`` below and ``g <= 0`` above, if there is one."""

    for lower, upper in zip(nodes, nodes[1:], strict=False):
        if lower.value > 0.0 >= upper.value:
            return lower, upper
    return None


def _evaluate_nodes(
    evaluate: Evaluator,
    role: PassRole,
    logs: tuple[float, ...],
    mean_w: float,
) -> list[ProfileNode]:
    parameters = tuple(exp(value) for value in logs)
    sums = evaluate(PassRequest(role, parameters, logs[0], logs[-1]))
    return [
        ProfileNode(parameter, *score(parameter, item, mean_w))
        for parameter, item in zip(parameters, sums, strict=True)
    ]


def _find_bracket(
    evaluate: Evaluator, mean_w: float
) -> tuple[ProfileNode, ProfileNode] | ProfileRefusal:
    """Grid pass and expansions until the score changes sign, or the domain is exhausted."""

    nodes = _evaluate_nodes(evaluate, PassRole.GRID, _initial_logs(), mean_w)
    bracket = _bracket(nodes)
    for band in range(MAX_EXPANSION_PASSES):
        if bracket is not None:
            return bracket
        inner = INITIAL_LOG_WINDOW if band == 0 else EXPANSION_LOG_LIMITS[band - 1]
        logs = _band_logs(inner, EXPANSION_LOG_LIMITS[band], upward=nodes[0].value > 0.0)
        nodes = sorted(
            [*nodes, *_evaluate_nodes(evaluate, PassRole.EXPANSION, logs, mean_w)],
            key=lambda node: node.parameter,
        )
        bracket = _bracket(nodes)
    return ProfileRefusal.BOUNDARY if bracket is None else bracket


def solve_profile(
    evaluate: Evaluator, *, count: int, mean_w: float
) -> ProfileSolution | ProfileRefusal:
    """The root of the profile equation by the plan of the module documentation.

    ``count`` is the number of observations and ``mean_w`` the mean of the centred
    observations. Every decision is a function of the finalized sums returned by
    ``evaluate``, so the iterates do not depend on how the passes were computed.
    """

    found = _find_bracket(evaluate, mean_w)
    if isinstance(found, ProfileRefusal):
        return found
    below, above = found
    low, high = below.parameter, above.parameter
    point = hermite_root(below, above)
    for _ in range(MAX_NEWTON_PASSES):
        request = PassRequest(PassRole.ITERATION, (point,), bracket_log_width=log(high) - log(low))
        (sums,) = evaluate(request)
        value, slope = score(point, sums, mean_w)
        step = -value / slope
        if abs(step) <= CONVERGENCE_TOLERANCE * point:
            first = sums.s1 / sums.s0
            variance = max(sums.s2 / sums.s0 - first * first, 0.0)
            log_ratio = fsum((log(sums.s0 / count), step * first, 0.5 * step * step * variance))
            return ProfileSolution(point + step, point, log_ratio, sums)
        if value > 0.0:
            low = point
        else:
            high = point
        candidate = point + step
        point = candidate if low < candidate < high else sqrt(low * high)
    return ProfileRefusal.EXHAUSTED


__all__ = [
    "CONVERGENCE_TOLERANCE",
    "EXPANSION_LOG_LIMITS",
    "GRID_POINTS",
    "INITIAL_LOG_WINDOW",
    "LOG_DOMAIN_LIMIT",
    "MAXIMUM_PASSES",
    "MAX_EXPANSION_PASSES",
    "MAX_HERMITE_BISECTIONS",
    "MAX_NEWTON_PASSES",
    "MINIMUM_PASSES",
    "PASS_BUDGET_DEFAULT",
    "Evaluator",
    "PassLedger",
    "PassRequest",
    "PassRole",
    "PlanBounds",
    "ProfileNode",
    "ProfileRefusal",
    "ProfileSolution",
    "ProfileSums",
    "check_pass_budget",
    "hermite_root",
    "plan_identity",
    "plan_passes",
    "score",
    "solve_profile",
]
