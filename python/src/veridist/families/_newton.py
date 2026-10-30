"""A damped Newton maximization of a censored or truncated log-likelihood on exact sums.

One iteration is one pass over the rows.  After the pilot (pass 1) the start grid of nine
candidates is evaluated together (pass 2) and the candidate of the highest log-likelihood is
kept.  Every later pass evaluates a stencil of trial points along the Newton direction
``d = -H^-1 g`` of the accepted point; the trial of the largest step that satisfies the
Armijo condition on the exact log-likelihood is accepted, and the next direction comes from
that trial's own gradient and Hessian, which the same pass already holds.  Every decision is a
function of exact sums, so the iterates depend on the multiset of the rows and on nothing else.

The rule, in the units of the pilot's standardisation:

* **Direction.** ``d = -H^-1 g``.  If ``H`` is not negative definite (its largest eigenvalue is
  above ``-1e-10`` times the spectral scale) it is shifted by the largest eigenvalue plus
  ``0.1`` times the scale plus ``1e-8``, times the identity, and a step taken after a shift is
  never a converged step.
* **Step limits.** The direction is scaled so that its largest coordinate is at most
  :data:`CLAMP` and, for the coordinate that is a rate (the exponential rate, ``u = 1 / sigma``),
  so that the trial keeps it within ``[0.1, 10]`` times the current value.  A trial outside the
  hard limits is not evaluated; a stencil with every trial outside them is
  ``BOUNDARY_SOLUTION``.
* **Acceptance.** Armijo with ``c1 = 1e-4``: the log-likelihood at the trial is at least that at
  the iterate plus ``c1`` times the step times the directional derivative, less the precision
  :data:`ARMIJO_SLACK` of the sums (``1e-12`` times the larger of 1 and the magnitude of the
  log-likelihood: a gain below that is beyond what the kernels resolve).
* **Trials.** ``{1, 1/2}`` of the limited direction; if neither is accepted the next pass
  evaluates ``{1/4, 1/8}``, then ``{1/16, 1/32}``, at most :data:`STENCIL_CAP` stencils in
  all, after which the result is ``OPTIMIZER_EXHAUSTED``.
* **Convergence.** The relative step ``max |d_i| / max(1, |theta_i|)`` is below the tolerance
  with no shift, the iteration cap is :data:`ITERATION_CAP`, and the Hessian at the accepted
  point must be negative definite.  A likelihood that is flat in some direction ends at the cap,
  never as an estimate.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib import import_module
from math import isfinite, log
from typing import Any, Final

from veridist.families.registry import FamilyId

np: Any = import_module("numpy")

#: Largest coordinate of a step, in standardised units (the gamma coordinates are logarithms).
CLAMP: Final = 2.0
#: Armijo constant of the acceptance rule.
ARMIJO: Final = 1e-4
#: The log-likelihood is known only to the precision of the kernels, ``1e-12`` times the larger of
#: 1 and the magnitude of what is summed; the Armijo test is not asked to resolve a gain below
#: that (where the predicted gain of a step is below the noise of a sum, the test would refuse a
#: good step, as it did for narrow intervals near the maximum).
ARMIJO_SLACK: Final = 1e-12
#: Newton iterations at most (the pilot and the start pass come before them).
ITERATION_CAP: Final = 20
#: Stencils of trial steps at most for one iteration, the first included.
STENCIL_CAP: Final = 6
#: The steps of the first stencil; each later stencil is the previous one times ``SHRINK``.
TRIALS: Final = (1.0, 0.5)
SHRINK: Final = 0.5 ** len(TRIALS)
#: The trial may move a rate within ``[LOWER_FACTOR, UPPER_FACTOR]`` times its current value.
LOWER_FACTOR: Final = 0.1
UPPER_FACTOR: Final = 10.0
#: Hard limits of a trial: the log of a scale-like coordinate, the location coordinate ``m``.
LOG_LIMIT: Final = 20.0
LOCATION_LIMIT: Final = 1e6
#: Relative step below which Newton has converged; the gamma, whose shape derivatives are
#: finite differences, converges to the looser one.
TOLERANCE: Final = 1e-8
GAMMA_TOLERANCE: Final = 1e-6
#: The Hessian is not negative definite if its largest eigenvalue exceeds this share of its
#: spectral scale below zero; the shift is ``largest + SHIFT_SCALE * scale + SHIFT_ABSOLUTE``.
NEGATIVE_DEFINITE: Final = 1e-10
SHIFT_SCALE: Final = 0.1
SHIFT_ABSOLUTE: Final = 1e-8
#: Two stationary points differ if some coordinate differs by more than this (relative).
DISTINCT: Final = 1e-5
#: Passes the plan counts before the first stencil: the pilot and the start grid.
SETUP_PASSES: Final = 2

SumsAt = Callable[[tuple[float, ...]], "list[float] | None"]


@dataclass(frozen=True, slots=True)
class Run:
    """One Newton run: its outcome and its deterministic counters."""

    status: str
    theta: tuple[float, ...]
    log_likelihood: float
    passes: int
    iterations: int
    shifted: int
    stencil_failures: int


@dataclass(frozen=True, slots=True)
class Maximum:
    """The result of :func:`maximize`: the run kept and the facts about all runs."""

    run: Run
    passes: int
    starts: int
    distinct: int


def unpack(sums: Sequence[float], dimension: int) -> tuple[float, Any, Any]:
    """``(log-likelihood, gradient, Hessian)`` from the sums of the terms."""

    if dimension == 1:
        return sums[0], np.array([sums[1]]), np.array([[sums[2]]])
    return (
        sums[0],
        np.array([sums[1], sums[2]]),
        np.array([[sums[3], sums[4]], [sums[4], sums[5]]]),
    )


def start_candidates(family: FamilyId, start: tuple[float, ...]) -> list[tuple[float, ...]]:
    """The nine candidates of the start grid; the first is the family's moment start."""

    if family is FamilyId.EXPONENTIAL:
        return [
            (start[0] * factor,) for factor in (1.0, 0.1, 0.25, 0.5, 2.0, 4.0, 10.0, 0.04, 25.0)
        ]
    if family is FamilyId.GAMMA:
        alpha, beta = start
        grid = [start]
        for shift in (0.0, -1.0, 1.0):
            for move in (0.0, -0.7, 0.7):
                if shift == 0.0 and move == 0.0:
                    continue
                grid.append((alpha + shift, beta - shift + move))
        return grid
    m0, u0 = start
    mu, sigma = m0 / u0, 1.0 / u0
    grid = [start]
    for factor in (1.0, 0.3, 3.0):
        for shift in (0.0, -1.0, 1.0):
            if factor == 1.0 and shift == 0.0:
                continue
            spread = sigma * factor
            grid.append(((mu + shift * sigma) / spread, 1.0 / spread))
    return grid


def in_domain(family: FamilyId, theta: tuple[float, ...]) -> bool:
    """Whether ``theta`` is inside the hard limits."""

    if family is FamilyId.EXPONENTIAL:
        return theta[0] > 0.0 and abs(log(theta[0])) < LOG_LIMIT
    if family is FamilyId.GAMMA:
        return abs(theta[0]) < LOG_LIMIT and abs(theta[1]) < LOG_LIMIT
    return theta[1] > 0.0 and abs(log(theta[1])) < LOG_LIMIT and abs(theta[0]) < LOCATION_LIMIT


def positive_index(family: FamilyId) -> int | None:
    """The coordinate that is kept within a factor of its current value, if any."""

    if family is FamilyId.EXPONENTIAL:
        return 0
    if family is FamilyId.GAMMA:
        return None
    return 1


def concave_in_coordinates(family: FamilyId, truncation: bool, right_or_two_sided: bool) -> bool:
    """Whether the log-likelihood is concave in the coordinates, so that a converged result is
    the global maximum.

    The location-scale families (normal, lognormal, Weibull minimum, right Gumbel) are concave
    in ``(mu / sigma, 1 / sigma)`` without truncation for exact, right-, left- and
    interval-censored rows (Burridge 1981; Pratt 1981); the exponential is concave in the rate
    with at most left truncation (the truncation term is linear in it).  Truncation otherwise,
    and the gamma, carry no guarantee.
    """

    if family is FamilyId.GAMMA:
        return False
    if family is FamilyId.EXPONENTIAL:
        return not right_or_two_sided
    return not truncation


def direction(hessian: Any, gradient: Any) -> tuple[Any, bool]:
    """The Newton direction ``-H^-1 g`` and whether the Hessian had to be shifted."""

    eigenvalues = np.linalg.eigvalsh(hessian)
    largest = float(eigenvalues.max())
    scale = float(np.abs(eigenvalues).max()) + 1e-300
    shifted = False
    if largest > -NEGATIVE_DEFINITE * scale:
        shifted = True
        hessian = hessian - (largest + SHIFT_SCALE * scale + SHIFT_ABSOLUTE) * np.eye(
            gradient.shape[0]
        )
    return -np.linalg.solve(hessian, gradient), shifted


def feasible_scale(family: FamilyId, theta: tuple[float, ...], step: Any) -> float:
    """The factor of the full direction that keeps the trial within the step limits."""

    factor = 1.0
    index = positive_index(family)
    if index is not None:
        position, move = theta[index], float(step[index])
        if move < 0.0:
            factor = min(factor, (1.0 - LOWER_FACTOR) * position / -move)
        elif move > (UPPER_FACTOR - 1.0) * position:
            factor = min(factor, (UPPER_FACTOR - 1.0) * position / move)
    largest = float(np.max(np.abs(step)))
    if largest * factor > CLAMP:
        factor = CLAMP / largest
    return factor


def _run(
    family: FamilyId,
    sums_at: SumsAt,
    start: tuple[float, ...],
    start_sums: list[float],
    tolerance: float,
) -> Run:
    """One damped Newton run from ``start`` whose sums are ``start_sums``."""

    dimension = 1 if family is FamilyId.EXPONENTIAL else 2
    theta = start
    value, gradient, hessian = unpack(start_sums, dimension)
    passes = shifted_count = failures = 0
    iteration = 0
    while True:
        step, shifted = direction(hessian, gradient)
        size = float(np.max(np.abs(step) / np.maximum(1.0, np.abs(np.array(theta)))))
        if shifted:
            shifted_count += 1
        if size <= tolerance and not shifted:
            theta = tuple(float(x) for x in np.array(theta) + step)
            return Run("CONVERGED", theta, value, passes, iteration, shifted_count, failures)
        if iteration == ITERATION_CAP:
            return Run(
                "OPTIMIZER_EXHAUSTED", theta, value, passes, iteration, shifted_count, failures
            )
        factor = feasible_scale(family, theta, step)
        limited = step * factor
        slope = float(gradient @ limited)
        if slope <= 0.0:
            limited = gradient / (np.linalg.norm(gradient) + 1e-300) * min(1.0, factor)
            slope = float(gradient @ limited)
        accepted = None
        shrink = 1.0
        allowance = ARMIJO_SLACK * max(1.0, abs(value))
        for _attempt in range(STENCIL_CAP):
            trials = [alpha * shrink for alpha in TRIALS]
            points = [
                tuple(float(x) for x in np.array(theta) + alpha * limited) for alpha in trials
            ]
            inside = [index for index, point in enumerate(points) if in_domain(family, point)]
            if not inside:
                return Run(
                    "BOUNDARY_SOLUTION", theta, value, passes, iteration, shifted_count, failures
                )
            passes += 1
            for index in inside:
                sums = sums_at(points[index])
                if sums is None:
                    continue
                candidate = unpack(sums, dimension)
                if (
                    isfinite(candidate[0])
                    and candidate[0] >= value + ARMIJO * trials[index] * slope - allowance
                ):
                    accepted = (points[index], candidate)
                    break
            if accepted is not None:
                break
            failures += 1
            shrink *= SHRINK
        if accepted is None:
            return Run(
                "OPTIMIZER_EXHAUSTED", theta, value, passes, iteration, shifted_count, failures
            )
        theta = accepted[0]
        value, gradient, hessian = accepted[1]
        iteration += 1


def maximize(
    family: FamilyId,
    sums_at: SumsAt,
    start: tuple[float, ...],
    *,
    concave: bool,
    tolerance: float,
) -> Maximum | None:
    """Maximize from the start grid of ``start``; ``None`` if no start can be evaluated.

    Where the likelihood is not known to be concave (``concave`` false) a second run starts at
    the second-best candidate of the grid and the higher log-likelihood of the converged runs is
    kept (the first on a tie).
    """

    scored: list[tuple[float, int, tuple[float, ...], list[float]]] = []
    for index, candidate in enumerate(start_candidates(family, start)):
        sums = sums_at(candidate)
        if sums is not None and isfinite(sums[0]):
            scored.append((sums[0], index, candidate, sums))
    if not scored:
        return None
    # The highest value first; the grid order breaks ties.
    scored.sort(key=lambda item: (-item[0], item[1]))
    starts = scored[:1] if concave else scored[:2]
    runs = [_run(family, sums_at, point, sums, tolerance) for _, _, point, sums in starts]
    converged = [run for run in runs if run.status == "CONVERGED"]
    chosen = runs[0]
    if converged:
        chosen = converged[0]
        for run in converged[1:]:
            if run.log_likelihood > chosen.log_likelihood:
                chosen = run
    distinct = 1
    if len(converged) == 2 and _apart(converged[0].theta, converged[1].theta):
        distinct = 2
    passes = SETUP_PASSES + sum(run.passes for run in runs)
    return Maximum(chosen, passes, len(runs), distinct)


def _apart(first: tuple[float, ...], second: tuple[float, ...]) -> bool:
    return any(abs(a - b) > DISTINCT * max(1.0, abs(a)) for a, b in zip(first, second, strict=True))


__all__ = [
    "ARMIJO",
    "ARMIJO_SLACK",
    "CLAMP",
    "GAMMA_TOLERANCE",
    "ITERATION_CAP",
    "STENCIL_CAP",
    "TOLERANCE",
    "TRIALS",
    "Maximum",
    "Run",
    "concave_in_coordinates",
    "direction",
    "feasible_scale",
    "in_domain",
    "maximize",
    "start_candidates",
    "unpack",
]
