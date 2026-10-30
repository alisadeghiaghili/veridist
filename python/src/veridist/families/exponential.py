"""Closed-form exponential rate MLE for exact and right-censored lifetimes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import isclose, isfinite, log
from types import MappingProxyType
from typing import Any, Final

from veridist.domain.coarsening import CoarseningAssumption
from veridist.domain.lifetimes import (
    ExactLifetime,
    IntervalCensoredLifetime,
    LeftCensoredLifetime,
    LifetimeObservation,
    RightCensoredLifetime,
)
from veridist.engine.streaming import StreamSource, iter_stream
from veridist.families._evidence import FitEvidence
from veridist.families._fit_facts import (
    FitDiagnostics,
    ObservationSummary,
    OptimumKind,
    summarize_observations,
)
from veridist.families.registry import FamilyId
from veridist.families.uncertainty import (
    FitUncertainty,
    UncertaintyUnavailable,
    exponential_uncertainty,
)
from veridist.statistics.exponential import (
    ExponentialReductionState,
    _ReductionOverflow,
    reduce_exponential_chunks,
)


class ExponentialFitFailureCode(StrEnum):
    """Stable, locale-neutral reasons that no finite point estimate exists."""

    EMPTY_SAMPLE = "EMPTY_SAMPLE"
    NO_OBSERVED_EVENTS = "NO_OBSERVED_EVENTS"
    UNBOUNDED_LIKELIHOOD = "UNBOUNDED_LIKELIHOOD"
    NUMERICAL_OVERFLOW = "NUMERICAL_OVERFLOW"
    OPTIMIZER_EXHAUSTED = "OPTIMIZER_EXHAUSTED"
    BOUNDARY_SOLUTION = "BOUNDARY_SOLUTION"
    # The last two only arise for data with left- or interval-censored rows or truncation
    # windows beyond a left end, which is fitted by the Newton method.


FailureFactValidator = Callable[[int, int, float | None], bool]


def _valid_empty_sample(observation_count: int, event_count: int, total_time: float | None) -> bool:
    return (
        observation_count == 0
        and event_count == 0
        and type(total_time) is float
        and total_time == 0.0
    )


def _valid_no_observed_events(
    observation_count: int, event_count: int, total_time: float | None
) -> bool:
    return (
        observation_count > 0
        and event_count == 0
        and isinstance(total_time, float)
        and total_time >= 0.0
        and isfinite(total_time)
    )


def _valid_unbounded_likelihood(
    observation_count: int, event_count: int, total_time: float | None
) -> bool:
    return (
        observation_count > 0
        and event_count > 0
        and type(total_time) is float
        and total_time == 0.0
    )


def _valid_numerical_overflow(
    observation_count: int, event_count: int, total_time: float | None
) -> bool:
    return observation_count > 0 and total_time is None


def _valid_iterative_failure(
    observation_count: int, event_count: int, total_time: float | None
) -> bool:
    return observation_count > 0 and total_time is None


_FAILURE_FACT_VALIDATORS: Final[Mapping[ExponentialFitFailureCode, FailureFactValidator]] = (
    MappingProxyType(
        {
            ExponentialFitFailureCode.EMPTY_SAMPLE: _valid_empty_sample,
            ExponentialFitFailureCode.NO_OBSERVED_EVENTS: _valid_no_observed_events,
            ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD: _valid_unbounded_likelihood,
            ExponentialFitFailureCode.NUMERICAL_OVERFLOW: _valid_numerical_overflow,
            ExponentialFitFailureCode.OPTIMIZER_EXHAUSTED: _valid_iterative_failure,
            ExponentialFitFailureCode.BOUNDARY_SOLUTION: _valid_iterative_failure,
        }
    )
)


@dataclass(frozen=True, slots=True)
class ExponentialFitProvenance:
    """Closed reduction facts that deliberately contain no source locator or rows."""

    accumulator_schema_version: str = "1"
    state_complexity: str = "O(1)"
    reduction_order: str = "canonical_input_order"
    partition_order_contract: str = "tolerance_only"
    raw_data_retained: bool = False

    def __post_init__(self) -> None:
        # Data of left- or interval-censored rows or truncation windows beyond a left end is
        # fitted by the Newton method and its fit keeps the rows for its uncertainty.
        retained = (self.state_complexity, self.raw_data_retained) == ("O(n)", True)
        kept = (self.state_complexity, self.raw_data_retained) == ("O(1)", False)
        if (
            self.accumulator_schema_version != "1"
            or not (retained or kept)
            or self.reduction_order != "canonical_input_order"
            or self.partition_order_contract != "tolerance_only"
        ):
            raise ValueError("exponential provenance facts are fixed by the vertical contract")


@dataclass(frozen=True, slots=True)
class ExponentialFitSuccess:
    """A finite rate-only exponential MLE with declared capability facts.

    Satisfies :class:`~veridist.families.results.FitSuccess`: ``parameters`` is
    ``{"rate": rate}`` and ``converged`` is always ``True`` (the closed-form MLE
    has no optimizer).
    """

    rate: float
    observation_count: int
    event_count: int
    total_time: float
    mean: float
    log_likelihood: float
    censored_count: int
    provenance: ExponentialFitProvenance = ExponentialFitProvenance()
    family: FamilyId = FamilyId.EXPONENTIAL
    parameterization: str = "rate"
    location: float = 0.0
    inference: str = "not_provided"
    censoring_assumption: str = "independent_right_censoring"
    observation_summary: ObservationSummary | None = field(default=None, compare=False, repr=False)
    coarsening_assumption: CoarseningAssumption | None = field(
        default=None, compare=False, repr=False
    )
    optimum: OptimumKind = field(default=OptimumKind.UNASSESSED, compare=False, repr=False)
    diagnostics: FitDiagnostics | None = field(default=None, compare=False, repr=False)
    _evidence: FitEvidence | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not isfinite(self.rate) or self.rate <= 0.0:
            raise ValueError("rate must be finite and positive")
        for name, value in (
            ("observation_count", self.observation_count),
            ("event_count", self.event_count),
            ("censored_count", self.censored_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.event_count + self.censored_count != self.observation_count:
            raise ValueError("event and censored counts must equal observation count")
        if not isfinite(self.total_time) or self.total_time <= 0.0:
            raise ValueError("total_time must be finite and positive")
        iterative = self.optimum in (OptimumKind.GLOBAL_MAXIMUM, OptimumKind.STATIONARY_POINT)
        if not iterative:
            if not isclose(
                self.rate, self.event_count / self.total_time, rel_tol=1e-15, abs_tol=0.0
            ):
                raise ValueError("rate must equal event count divided by total time")
            expected_likelihood = self.event_count * log(self.rate) - self.rate * self.total_time
            if not isclose(self.log_likelihood, expected_likelihood, rel_tol=1e-15, abs_tol=0.0):
                raise ValueError("log_likelihood does not match declared sufficient statistics")
        if not isclose(self.mean, 1.0 / self.rate, rel_tol=1e-15, abs_tol=0.0):
            raise ValueError("mean must equal inverse rate")
        new_data = self.observation_summary is not None and self.observation_summary.is_new_data
        if (
            type(self.provenance) is not ExponentialFitProvenance
            or self.family != "exponential"
            or self.parameterization != "rate"
            or self.location != 0.0
            or self.inference != "not_provided"
            or self.censoring_assumption
            != (
                "independent_censoring_and_truncation"
                if new_data
                else "independent_right_censoring"
            )
        ):
            raise ValueError("exponential capability facts are fixed by the vertical contract")
        object.__setattr__(self, "family", FamilyId.EXPONENTIAL)

    @property
    def parameters(self) -> Mapping[str, float]:
        """The fitted parameters under their canonical registry names, read-only."""

        return MappingProxyType({"rate": self.rate})

    def uncertainty(self) -> FitUncertainty | UncertaintyUnavailable:
        """Covariance, standard errors and confidence intervals of the fit.

        Computed from the sufficient statistics (event count and total time), so nothing
        beyond them is retained.  Besides the Wald and profile intervals, uncensored data
        also admit ``method="exact"``.
        """

        if self._evidence is not None:
            return self._evidence.uncertainty(self.family, self.parameters)
        return exponential_uncertainty(
            self.rate, self.event_count, self.total_time, self.censored_count
        )

    @property
    def converged(self) -> bool:
        """Always ``True``: the closed-form estimate involves no iteration."""

        return True


@dataclass(frozen=True, slots=True)
class ExponentialFitFailure:
    """A typed statistical non-estimate, distinct from an engine failure.

    Satisfies :class:`~veridist.families.results.FitFailure`.
    """

    code: ExponentialFitFailureCode
    observation_count: int
    event_count: int
    total_time: float | None
    observation_summary: ObservationSummary | None = field(default=None, compare=False, repr=False)
    coarsening_assumption: CoarseningAssumption | None = field(
        default=None, compare=False, repr=False
    )

    @property
    def family(self) -> FamilyId:
        """The family this failure belongs to."""

        return FamilyId.EXPONENTIAL

    @property
    def censored_count(self) -> int:
        """The number of right-censored observations offered."""

        return self.observation_count - self.event_count

    def __post_init__(self) -> None:
        if type(self.code) is not ExponentialFitFailureCode:
            raise TypeError("code must be an ExponentialFitFailureCode")
        if self.total_time is not None and type(self.total_time) is not float:
            raise TypeError("total_time must be a built-in float or None")
        for name, value in (
            ("observation_count", self.observation_count),
            ("event_count", self.event_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.event_count > self.observation_count:
            raise ValueError("event_count cannot exceed observation_count")
        valid = _FAILURE_FACT_VALIDATORS[self.code](
            self.observation_count, self.event_count, self.total_time
        )
        if not valid:
            raise ValueError("failure facts are inconsistent with its code")


ExponentialFit = ExponentialFitSuccess | ExponentialFitFailure


def fit_exponential(observations: Iterable[LifetimeObservation]) -> ExponentialFit:
    """Fit the fixed-location exponential rate by its closed-form likelihood MLE.

    ``observations`` is consumed once. A one-shot iterator that was already
    exhausted (for example a generator that was iterated earlier) looks exactly
    like an empty sample, so the result is an ``EMPTY_SAMPLE`` failure rather
    than an error. Pass a re-iterable collection, or wrap the stream in
    :class:`~veridist.engine.streaming.IterableDataSource`, when the number of
    passes must be enforced and a second acquisition must fail loudly.
    """

    return fit_exponential_chunks((observations,))


def fit_exponential_chunks(
    chunks: StreamSource[Iterable[LifetimeObservation]] | Iterable[Iterable[LifetimeObservation]],
) -> ExponentialFit:
    """Fit from ragged chunks without retaining raw observations or chunk payloads.

    ``chunks`` is consumed once. An already exhausted one-shot iterator yields
    no chunks and therefore an ``EMPTY_SAMPLE`` failure, not an error; the
    library cannot tell it apart from an empty input. To enforce single-pass
    semantics (a second acquisition raises ``PASS_BUDGET_EXCEEDED``), pass the
    chunks through :class:`~veridist.engine.streaming.IterableDataSource`.
    """

    diverted: list[Any] = []
    try:
        state = reduce_exponential_chunks(_plain_chunks(iter_stream(chunks), diverted))
    except _ReductionOverflow as error:
        extra = summarize_observations(diverted)
        return ExponentialFitFailure(
            ExponentialFitFailureCode.NUMERICAL_OVERFLOW,
            error.observation_count + extra.total,
            error.event_count + extra.exact,
            None,
        )
    if diverted:
        return _fit_new_data(state, diverted)
    return fit_exponential_reduction_state(state)


def _is_new_lifetime(observation: object) -> bool:
    """Whether ``observation`` is left- or interval-censored or carries a truncation window."""

    kind = type(observation)
    if kind is LeftCensoredLifetime or kind is IntervalCensoredLifetime:
        return True
    if kind is ExactLifetime or kind is RightCensoredLifetime:
        return (
            observation.truncation_lower is not None  # type: ignore[attr-defined]
            or observation.truncation_upper is not None  # type: ignore[attr-defined]
        )
    return False


def _plain_chunks(chunks: Iterable[Any], diverted: list[Any]) -> Iterator[Any]:
    """The chunks without the rows the closed form does not take, which go to ``diverted``.

    Anything that is not a lifetime observation passes through so that the reducer refuses it
    as before.
    """

    for chunk in chunks:
        yield chunk if not isinstance(chunk, Iterable) else _plain_rows(chunk, diverted)


def _plain_rows(chunk: Iterable[Any], diverted: list[Any]) -> Iterator[Any]:
    for observation in chunk:
        if _is_new_lifetime(observation):
            diverted.append(observation)
        else:
            yield observation


def _summary_of(state: ExponentialReductionState) -> ObservationSummary:
    """The summary of the exact and right-censored rows without a window of ``state``."""

    return ObservationSummary(
        exact=state.event_count,
        right_censored=state.observation_count - state.event_count,
        untruncated=state.observation_count,
    )


def _merged(first: ObservationSummary, second: ObservationSummary) -> ObservationSummary:
    return ObservationSummary(
        first.exact + second.exact,
        first.right_censored + second.right_censored,
        first.left_censored + second.left_censored,
        first.interval_censored + second.interval_censored,
        first.untruncated + second.untruncated,
        first.left_truncated + second.left_truncated,
        first.right_truncated + second.right_truncated,
        first.two_sided_truncated + second.two_sided_truncated,
    )


def _fit_new_data(state: ExponentialReductionState, rows: list[Any]) -> ExponentialFit:
    """Fit data with left- or interval-censored rows or truncation windows.

    ``rows`` are the observations the closed form does not take; the other rows are summarised
    by ``state`` (the exponential likelihood of exact and right-censored rows without a window
    depends on them through their number of events and their total time alone).  With exact and
    right-censored rows and at most left truncation the rate has a closed form; otherwise it is
    the Newton maximizer of the likelihood.  The rows are kept for the uncertainty of the
    latter.
    """

    from veridist.families._censored_fit import CensoredRefusal, estimate

    own = summarize_observations(rows)
    summary = _merged(own, _summary_of(state))
    count, events = summary.total, summary.exact
    assumption = CoarseningAssumption.INDEPENDENT_CENSORING_AND_TRUNCATION

    def failure(code: str, total_time: float | None) -> ExponentialFitFailure:
        return ExponentialFitFailure(
            ExponentialFitFailureCode[code],
            count,
            events,
            total_time,
            observation_summary=summary,
            coarsening_assumption=assumption,
        )

    try:
        plain_total = state.summed_time
    except _ReductionOverflow:
        return failure("NUMERICAL_OVERFLOW", None)
    outcome = estimate(
        FamilyId.EXPONENTIAL,
        rows,
        own,
        plain_count=state.observation_count,
        plain_events=state.event_count,
        plain_total=plain_total,
    )
    if isinstance(outcome, CensoredRefusal):
        return failure(outcome.code, outcome.total_time)
    (rate,) = outcome.parameters
    total_time = outcome.total_time
    assert total_time is not None
    iterative = outcome.optimum is not OptimumKind.CLOSED_FORM
    return ExponentialFitSuccess(
        rate,
        count,
        events,
        total_time,
        1.0 / rate,
        outcome.log_likelihood,
        count - events,
        ExponentialFitProvenance(
            state_complexity="O(n)" if iterative else "O(1)", raw_data_retained=iterative
        ),
        censoring_assumption="independent_censoring_and_truncation",
        observation_summary=summary,
        coarsening_assumption=assumption,
        optimum=outcome.optimum,
        diagnostics=outcome.diagnostics,
        _evidence=FitEvidence(rows=outcome.rows) if iterative else None,
    )


def fit_exponential_reduction_state(state: ExponentialReductionState) -> ExponentialFit:
    """Finalize a fixed-state streaming reduction without retaining samples."""

    if type(state) is not ExponentialReductionState:
        raise TypeError("state must be ExponentialReductionState")
    summary = _summary_of(state)

    def failure(code: ExponentialFitFailureCode, total: float | None) -> ExponentialFitFailure:
        return ExponentialFitFailure(
            code, state.observation_count, state.event_count, total, observation_summary=summary
        )

    try:
        total_time = state.summed_time
    except _ReductionOverflow:
        return failure(ExponentialFitFailureCode.NUMERICAL_OVERFLOW, None)
    count = state.observation_count
    events = state.event_count
    if count == 0:
        return failure(ExponentialFitFailureCode.EMPTY_SAMPLE, 0.0)
    if events == 0:
        return failure(ExponentialFitFailureCode.NO_OBSERVED_EVENTS, total_time)
    if total_time == 0.0:
        return failure(ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD, total_time)
    rate = events / total_time
    if not isfinite(rate) or rate <= 0.0:
        return failure(ExponentialFitFailureCode.NUMERICAL_OVERFLOW, None)
    mean = 1.0 / rate
    log_rate = log(rate)
    log_likelihood = events * log_rate - rate * total_time
    if not isfinite(mean) or not isfinite(log_rate) or not isfinite(log_likelihood):
        return failure(ExponentialFitFailureCode.NUMERICAL_OVERFLOW, None)
    return ExponentialFitSuccess(
        rate,
        count,
        events,
        total_time,
        mean,
        log_likelihood,
        count - events,
        ExponentialFitProvenance(),
        observation_summary=summary,
    )


__all__ = [
    "ExponentialFit",
    "ExponentialFitFailure",
    "ExponentialFitFailureCode",
    "ExponentialFitProvenance",
    "ExponentialFitSuccess",
    "fit_exponential",
    "fit_exponential_chunks",
    "fit_exponential_reduction_state",
]
