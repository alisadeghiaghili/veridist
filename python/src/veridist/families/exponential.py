"""Closed-form exponential rate MLE for exact and right-censored lifetimes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import isclose, isfinite, log
from types import MappingProxyType
from typing import Final

from veridist.domain.lifetimes import LifetimeObservation
from veridist.engine.streaming import StreamSource, iter_stream
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


_FAILURE_FACT_VALIDATORS: Final[Mapping[ExponentialFitFailureCode, FailureFactValidator]] = (
    MappingProxyType(
        {
            ExponentialFitFailureCode.EMPTY_SAMPLE: _valid_empty_sample,
            ExponentialFitFailureCode.NO_OBSERVED_EVENTS: _valid_no_observed_events,
            ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD: _valid_unbounded_likelihood,
            ExponentialFitFailureCode.NUMERICAL_OVERFLOW: _valid_numerical_overflow,
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
        if (
            self.accumulator_schema_version != "1"
            or self.state_complexity != "O(1)"
            or self.reduction_order != "canonical_input_order"
            or self.partition_order_contract != "tolerance_only"
            or self.raw_data_retained is not False
        ):
            raise ValueError("exponential provenance facts are fixed by the vertical contract")


@dataclass(frozen=True, slots=True)
class ExponentialFitSuccess:
    """A finite rate-only exponential MLE with declared capability facts."""

    rate: float
    observation_count: int
    event_count: int
    total_time: float
    mean: float
    log_likelihood: float
    censored_count: int
    provenance: ExponentialFitProvenance = ExponentialFitProvenance()
    family: str = "exponential"
    parameterization: str = "rate"
    location: float = 0.0
    inference: str = "not_provided"
    censoring_assumption: str = "independent_right_censoring"

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
        if not isclose(self.rate, self.event_count / self.total_time, rel_tol=1e-15, abs_tol=0.0):
            raise ValueError("rate must equal event count divided by total time")
        if not isclose(self.mean, 1.0 / self.rate, rel_tol=1e-15, abs_tol=0.0):
            raise ValueError("mean must equal inverse rate")
        expected_likelihood = self.event_count * log(self.rate) - self.rate * self.total_time
        if not isclose(self.log_likelihood, expected_likelihood, rel_tol=1e-15, abs_tol=0.0):
            raise ValueError("log_likelihood does not match declared sufficient statistics")
        if (
            type(self.provenance) is not ExponentialFitProvenance
            or self.family != "exponential"
            or self.parameterization != "rate"
            or self.location != 0.0
            or self.inference != "not_provided"
            or self.censoring_assumption != "independent_right_censoring"
        ):
            raise ValueError("exponential capability facts are fixed by the vertical contract")


@dataclass(frozen=True, slots=True)
class ExponentialFitFailure:
    """A typed statistical non-estimate, distinct from an engine failure."""

    code: ExponentialFitFailureCode
    observation_count: int
    event_count: int
    total_time: float | None

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
    """Fit the fixed-location exponential rate by its closed-form likelihood MLE."""

    return fit_exponential_chunks((observations,))


def fit_exponential_chunks(
    chunks: StreamSource[Iterable[LifetimeObservation]] | Iterable[Iterable[LifetimeObservation]],
) -> ExponentialFit:
    """Fit from ragged chunks without retaining raw observations or chunk payloads."""

    try:
        state = reduce_exponential_chunks(iter_stream(chunks))
    except _ReductionOverflow as error:
        return ExponentialFitFailure(
            ExponentialFitFailureCode.NUMERICAL_OVERFLOW,
            error.observation_count,
            error.event_count,
            None,
        )
    return fit_exponential_reduction_state(state)


def fit_exponential_reduction_state(state: ExponentialReductionState) -> ExponentialFit:
    """Finalize a fixed-state streaming reduction without retaining samples."""

    if type(state) is not ExponentialReductionState:
        raise TypeError("state must be ExponentialReductionState")
    try:
        total_time = state.summed_time
    except _ReductionOverflow:
        return ExponentialFitFailure(
            ExponentialFitFailureCode.NUMERICAL_OVERFLOW,
            state.observation_count,
            state.event_count,
            None,
        )
    count = state.observation_count
    events = state.event_count
    if count == 0:
        return ExponentialFitFailure(ExponentialFitFailureCode.EMPTY_SAMPLE, 0, 0, 0.0)
    if events == 0:
        return ExponentialFitFailure(
            ExponentialFitFailureCode.NO_OBSERVED_EVENTS, count, events, total_time
        )
    if total_time == 0.0:
        return ExponentialFitFailure(
            ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD, count, events, total_time
        )
    rate = events / total_time
    if not isfinite(rate) or rate <= 0.0:
        return ExponentialFitFailure(
            ExponentialFitFailureCode.NUMERICAL_OVERFLOW, count, events, None
        )
    mean = 1.0 / rate
    log_rate = log(rate)
    log_likelihood = events * log_rate - rate * total_time
    if not isfinite(mean) or not isfinite(log_rate) or not isfinite(log_likelihood):
        return ExponentialFitFailure(
            ExponentialFitFailureCode.NUMERICAL_OVERFLOW, count, events, None
        )
    return ExponentialFitSuccess(
        rate,
        count,
        events,
        total_time,
        mean,
        log_likelihood,
        count - events,
        ExponentialFitProvenance(),
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
