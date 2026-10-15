"""Maximum-likelihood fits of ``weibull_min`` and ``gumbel_right`` over several passes.

Both families have a one-parameter profile equation that cannot be reduced to a fixed set
of sufficient statistics, so the data is read again for each step of a root search. The
search itself (grid, expansions, Hermite start, safeguarded Newton, budget) is in
:mod:`veridist.scale._planner`; this module supplies what is specific to a family:

* the first pass, a pilot state (:mod:`veridist.scale._pilot`);
* the centred observations ``w <= 0`` of the later passes, a per-element kernel;
* the report: the estimate, the log-likelihood and the information sums, computed from the
  final pass.

Weibull-minimum on exact, strictly positive times. With ``l = log x`` and ``c = log max x``
(the logarithm of the largest observation), ``w = l - c``. The profile score of the shape
``k`` is ``n * g(k)`` with ``g`` of the planner, the scale is
``exp(c) * (S0 / n)**(1 / k)``, and the log-likelihood is
``n log k + k W - L - n log(S0 / n) - n`` with ``W = sum w`` and ``L = sum l``, both exact.

Right Gumbel on exact values. The data is standardised as the in-memory fit does,
``x' = (x - center) / peak`` with ``center`` the mean and ``peak`` the largest deviation of
any observation from it, and ``w = x'_min - x'``. With ``k = 1 / sigma'`` (``sigma'`` the scale
of the standardised data) the profile score of ``k`` is again ``n * g(k)`` (the Gumbel maximum
of ``x`` and the Weibull minimum of ``exp(-x)`` have the same profile equation), the location
profiled out is ``mu' = x'_min - sigma' * log(S0 / n)``, and the search domain is the
in-memory fit's: ``log sigma'`` within plus and minus ``LOG_DOMAIN_LIMIT``.

Failures are those of the in-memory fits: ``EMPTY_SAMPLE`` for no rows, ``INVALID_SUPPORT``
(Weibull) for a value that is not strictly positive, ``DEGENERATE_SAMPLE`` when every value
is equal (a single row included), ``BOUNDARY_SOLUTION`` when the root is not inside the
search domain and ``OPTIMIZER_EXHAUSTED`` when the root cannot be evaluated or the Newton
passes do not converge.

Uncertainty. The final pass carries the sums of the information matrix, so the result has a
Wald covariance, standard errors and intervals (:class:`~veridist.scale._evidence.ProfileEvidence`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import partial
from math import exp, fsum, isfinite, log
from typing import Any, Final

from veridist.engine.provenance import ScalePassProvenance
from veridist.families.gumbel import (
    GumbelFit,
    GumbelFitFailure,
    GumbelFitFailureCode,
    GumbelFitSuccess,
)
from veridist.families.registry import FamilyId
from veridist.families.results import FitFailure, FitSuccess
from veridist.families.weibull import (
    WeibullFit,
    WeibullFitFailure,
    WeibullFitFailureCode,
    WeibullFitSuccess,
)
from veridist.scale._accumulator import ExactSum
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._evidence import (
    ProfileEvidence,
    gumbel_evaluation,
    weibull_evaluation,
)
from veridist.scale._kernels import float64_batch, logarithms, numpy_module
from veridist.scale._pilot import GumbelPilotState, PilotState, WeibullPilotState
from veridist.scale._planner import (
    PassLedger,
    PassRequest,
    PassRole,
    ProfileRefusal,
    ProfileSolution,
    ProfileSums,
    solve_profile,
)
from veridist.scale._sources import Source, fold_partition

Kernel = Callable[[Any], Any]
"""Maps one validated float64 batch to its centred observations ``w`` (all at most zero)."""

SUMS_PER_CANDIDATE: Final = 3


class ProfileAccumulator:
    """The exact sums ``S0``, ``S1``, ``S2`` of every candidate of one pass over some rows.

    One formula serves every element and every candidate: ``e = numpy.exp(k * w)``, then
    ``e``, ``e * w`` and ``e * w * w`` are added exactly. The sums depend on the multiset of
    the rows and the candidates only.
    """

    __slots__ = ("count", "sums")

    def __init__(self, candidates: int) -> None:
        self.count = 0
        self.sums: tuple[tuple[ExactSum, ...], ...] = tuple(
            tuple(ExactSum() for _ in range(SUMS_PER_CANDIDATE)) for _ in range(candidates)
        )

    def update(self, centred: Any, parameters: tuple[float, ...]) -> None:
        """Add one batch of centred observations for every candidate in ``parameters``."""

        np = numpy_module()
        with np.errstate(all="ignore"):
            for parameter, (zeroth, first, second) in zip(parameters, self.sums, strict=True):
                weights = np.exp(parameter * centred)
                moment = weights * centred
                zeroth.update(weights)
                first.update(moment)
                second.update(moment * centred)
        self.count += int(centred.shape[0])

    def merge(self, other: ProfileAccumulator) -> ProfileAccumulator:
        """A new accumulator holding the exact sums of both."""

        merged = ProfileAccumulator(len(self.sums))
        merged.count = self.count + other.count
        merged.sums = tuple(
            tuple(left.merge(right) for left, right in zip(mine, theirs, strict=True))
            for mine, theirs in zip(self.sums, other.sums, strict=True)
        )
        return merged

    def totals(self) -> list[ProfileSums]:
        """The sums of every candidate, each rounded once."""

        return [
            ProfileSums(zeroth.total(), first.total(), second.total())
            for zeroth, first, second in self.sums
        ]


@dataclass(frozen=True, slots=True)
class MultiPassResult:
    """A multi-pass fit: the in-memory result, its first-pass state and the passes made."""

    fit: FitSuccess | FitFailure
    state: PilotState | None
    row_count: int
    passes: ScalePassProvenance


def _pilot_pass(
    ledger: PassLedger, source: Source, state_type: type[PilotState], max_rows: int
) -> PilotState:
    """The first pass: the pilot state of the source, merged in canonical order."""

    state = state_type.empty()
    for partition in ledger.begin(PassRole.STATISTICS, 0):
        state = state.merge(fold_partition(source, partition, state_type, max_rows=max_rows))
    return state


def _profile_pass(
    ledger: PassLedger,
    source: Source,
    max_rows: int,
    kernel: Kernel,
    rows: int,
    request: PassRequest,
) -> list[ProfileSums]:
    """One pass over the source for ``request``: the sums of each candidate.

    ``ScaleSourceError`` with ``PARTITION_CHANGED`` if the pass does not read the rows the
    first pass read.
    """

    total = ProfileAccumulator(len(request.parameters))
    partitions = ledger.begin(
        request.role,
        len(request.parameters),
        request.log_low,
        request.log_high,
        bracket_log_width=request.bracket_log_width,
    )
    for partition in partitions:
        part = ProfileAccumulator(len(request.parameters))
        for batch in source.batches(partition, max_rows=max_rows):
            part.update(kernel(float64_batch(batch)), request.parameters)
        total = total.merge(part)
    if total.count != rows:
        raise ScaleSourceError(ScaleSourceErrorCode.PARTITION_CHANGED)
    return total.totals()


def _evaluator(
    ledger: PassLedger, source: Source, max_rows: int, kernel: Kernel, rows: int
) -> Callable[[PassRequest], list[ProfileSums]]:
    def evaluate(request: PassRequest) -> list[ProfileSums]:
        return _profile_pass(ledger, source, max_rows, kernel, rows, request)

    return evaluate


# ------------------------------------------------------------------------------ weibull


def _weibull_failure(code: WeibullFitFailureCode, rows: int) -> WeibullFitFailure:
    return WeibullFitFailure(code, rows, rows, 0)


def _weibull_report(
    solution: ProfileSolution,
    count: int,
    peak_log: float,
    sum_w: float,
    sum_logs: float,
) -> WeibullFit:
    """The fit from the solved shape: scale, log-likelihood and the information sums."""

    shape = solution.parameter
    try:
        scale = exp(fsum((peak_log, solution.log_ratio / shape)))
    except OverflowError:
        return _weibull_failure(WeibullFitFailureCode.OPTIMIZER_EXHAUSTED, count)
    likelihood = fsum(
        (
            count * log(shape),
            shape * sum_w,
            -sum_logs,
            -count * solution.log_ratio,
            -float(count),
        )
    )
    if not (isfinite(scale) and scale > 0.0 and isfinite(likelihood)):
        return _weibull_failure(WeibullFitFailureCode.OPTIMIZER_EXHAUSTED, count)
    evidence = ProfileEvidence(
        FamilyId.WEIBULL_MIN,
        (shape, scale),
        partial(
            weibull_evaluation, count, shape, scale, solution.log_ratio, solution.sums, likelihood
        ),
    )
    return WeibullFitSuccess(shape, scale, likelihood, count, count, 0, _evidence=evidence)


def fit_weibull_min(
    source: Source,
    ledger: PassLedger,
    max_rows: int,
) -> MultiPassResult:
    """The Weibull-minimum fit of ``source`` on exact data; see the module documentation."""

    np = numpy_module()
    state = _pilot_pass(ledger, source, WeibullPilotState, max_rows)
    statistics = state.finalize()
    count = statistics.count

    def done(fit: WeibullFit, record: PilotState | None = state) -> MultiPassResult:
        return MultiPassResult(fit, record, count, ledger.provenance())

    if count == 0:
        return done(_weibull_failure(WeibullFitFailureCode.EMPTY_SAMPLE, 0))
    assert statistics.minimum is not None and statistics.maximum is not None
    if statistics.minimum <= 0.0:
        return done(_weibull_failure(WeibullFitFailureCode.INVALID_SUPPORT, count), None)
    if statistics.minimum == statistics.maximum:
        return done(_weibull_failure(WeibullFitFailureCode.DEGENERATE_SAMPLE, count))
    peak_log = float(logarithms(np.array([statistics.maximum], dtype=np.float64))[0])
    exact_w = statistics.total.exact - count * Fraction(peak_log)
    sum_w = float(exact_w)
    mean_w = float(exact_w / count)

    def kernel(batch: Any) -> Any:
        return logarithms(batch) - peak_log

    solution = solve_profile(
        _evaluator(ledger, source, max_rows, kernel, count), count=count, mean_w=mean_w
    )
    if solution is ProfileRefusal.BOUNDARY:
        return done(_weibull_failure(WeibullFitFailureCode.BOUNDARY_SOLUTION, count))
    if solution is ProfileRefusal.EXHAUSTED:
        return done(_weibull_failure(WeibullFitFailureCode.OPTIMIZER_EXHAUSTED, count))
    return done(_weibull_report(solution, count, peak_log, sum_w, statistics.total.rounded))


# ------------------------------------------------------------------------------- gumbel


def _gumbel_failure(code: GumbelFitFailureCode, rows: int) -> GumbelFitFailure:
    return GumbelFitFailure(code, rows, rows, 0)


def _gumbel_report(
    solution: ProfileSolution,
    count: int,
    center: float,
    peak: float,
    minimum_standardised: float,
    sum_w: float,
    mean_w: float,
) -> GumbelFit:
    """The fit from the solved reciprocal scale: location, scale, likelihood, information."""

    rate = solution.parameter
    scale_standardised = 1.0 / rate
    location_standardised = fsum((minimum_standardised, -scale_standardised * solution.log_ratio))
    location = fsum((center, peak * location_standardised))
    scale = peak * scale_standardised
    likelihood = fsum(
        (
            count * log(rate),
            rate * sum_w,
            -count * solution.log_ratio,
            -float(count),
            -count * log(peak),
        )
    )
    if not (isfinite(location) and isfinite(scale) and scale > 0.0 and isfinite(likelihood)):
        return _gumbel_failure(GumbelFitFailureCode.OPTIMIZER_EXHAUSTED, count)
    evidence = ProfileEvidence(
        FamilyId.GUMBEL_RIGHT,
        (location, scale),
        partial(
            gumbel_evaluation,
            count,
            scale,
            rate,
            solution.log_ratio,
            mean_w,
            solution.sums,
            likelihood,
        ),
    )
    return GumbelFitSuccess(location, scale, likelihood, count, count, 0, _evidence=evidence)


def fit_gumbel_right(
    source: Source,
    ledger: PassLedger,
    max_rows: int,
) -> MultiPassResult:
    """The right-Gumbel fit of ``source`` on exact data; see the module documentation."""

    state = _pilot_pass(ledger, source, GumbelPilotState, max_rows)
    statistics = state.finalize()
    count = statistics.count

    def done(fit: GumbelFit) -> MultiPassResult:
        return MultiPassResult(fit, state, count, ledger.provenance())

    if count == 0:
        return done(_gumbel_failure(GumbelFitFailureCode.EMPTY_SAMPLE, 0))
    assert statistics.minimum is not None and statistics.maximum is not None
    minimum, maximum = statistics.minimum, statistics.maximum
    if minimum == maximum:
        return done(_gumbel_failure(GumbelFitFailureCode.DEGENERATE_SAMPLE, count))
    center = float(statistics.total.exact / count)
    peak = max(maximum - center, center - minimum)
    if not isfinite(peak):
        return done(_gumbel_failure(GumbelFitFailureCode.OPTIMIZER_EXHAUSTED, count))
    minimum_standardised = (minimum - center) / peak
    exact_standardised = (statistics.total.exact - count * Fraction(center)) / Fraction(peak)
    exact_w = count * Fraction(minimum_standardised) - exact_standardised
    sum_w = float(exact_w)
    mean_w = float(exact_w / count)

    def kernel(batch: Any) -> Any:
        return minimum_standardised - (batch - center) / peak

    solution = solve_profile(
        _evaluator(ledger, source, max_rows, kernel, count), count=count, mean_w=mean_w
    )
    if solution is ProfileRefusal.BOUNDARY:
        return done(_gumbel_failure(GumbelFitFailureCode.BOUNDARY_SOLUTION, count))
    if solution is ProfileRefusal.EXHAUSTED:
        return done(_gumbel_failure(GumbelFitFailureCode.OPTIMIZER_EXHAUSTED, count))
    return done(_gumbel_report(solution, count, center, peak, minimum_standardised, sum_w, mean_w))


__all__ = [
    "MultiPassResult",
    "ProfileAccumulator",
    "fit_gumbel_right",
    "fit_weibull_min",
]
