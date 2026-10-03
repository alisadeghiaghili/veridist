"""Fixed-location Weibull-min MLE cell for exact/right-censored lifetimes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import exp, fsum, isfinite, log
from types import MappingProxyType

from veridist.domain._numeric import is_real
from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation
from veridist.families._reliability import admitted_observations, expand_bracket, positive_support
from veridist.families.registry import FamilyId

#: Starting and hard-limit half-widths for the log-shape search (natural log units).
_LOG_SHAPE_BOUNDS = (-6.0, 6.0)
_LOG_SHAPE_HARD_LIMITS = (-20.0, 20.0)


class WeibullFitFailureCode(StrEnum):
    """Reasons a finite Weibull point estimate is unavailable."""

    EMPTY_SAMPLE = "EMPTY_SAMPLE"
    NO_OBSERVED_EVENTS = "NO_OBSERVED_EVENTS"
    INVALID_SUPPORT = "INVALID_SUPPORT"
    OPTIMIZER_EXHAUSTED = "OPTIMIZER_EXHAUSTED"
    DEGENERATE_SAMPLE = "DEGENERATE_SAMPLE"
    BOUNDARY_SOLUTION = "BOUNDARY_SOLUTION"


@dataclass(frozen=True, slots=True)
class WeibullFitFailure:
    """A declared reason no Weibull point estimate is reported.

    ``converged`` is always ``False`` and ``restart_failures`` is always ``0``:
    a failure never ran an interior optimization to convergence, and no restart
    strategy is attempted.  Satisfies :class:`~veridist.families.results.FitFailure`.
    """

    code: WeibullFitFailureCode
    observation_count: int
    event_count: int
    censored_count: int
    complete: bool = False
    converged: bool = False
    restart_failures: int = 0

    @property
    def family(self) -> FamilyId:
        """The family this failure belongs to."""

        return FamilyId.WEIBULL_MIN


@dataclass(frozen=True, slots=True)
class WeibullFitSuccess:
    """A Weibull MLE point estimate from an interior, converged optimum.

    ``converged`` is always ``True``: a result that only exists at the edge of
    the search range is reported as :attr:`WeibullFitFailureCode.BOUNDARY_SOLUTION`
    instead of a success. ``restart_failures`` is always ``0`` because the
    deterministic golden-section search never restarts.  Satisfies
    :class:`~veridist.families.results.FitSuccess`.
    """

    shape: float
    scale: float
    log_likelihood: float
    observation_count: int
    event_count: int
    censored_count: int
    converged: bool = True
    restart_failures: int = 0
    complete: bool = True
    family: FamilyId = FamilyId.WEIBULL_MIN
    location: float = 0.0

    def __post_init__(self) -> None:
        if not (isfinite(self.shape) and self.shape > 0.0):
            raise ValueError("shape must be finite and positive")
        if not (isfinite(self.scale) and self.scale > 0.0 and isfinite(self.log_likelihood)):
            raise ValueError("scale and log likelihood must be finite")
        if self.family != FamilyId.WEIBULL_MIN:
            raise ValueError("a Weibull fit result belongs to the weibull_min family")
        object.__setattr__(self, "family", FamilyId.WEIBULL_MIN)

    @property
    def parameters(self) -> Mapping[str, float]:
        """The fitted parameters under their canonical registry names, read-only."""

        return MappingProxyType({"shape": self.shape, "scale": self.scale})


WeibullFit = WeibullFitSuccess | WeibullFitFailure


def fit_weibull(
    observations: Iterable[LifetimeObservation],
    *,
    fixed_shape: float | None = None,
    frequency_weights: Iterable[int] | None = None,
    analytic_weights: object | None = None,
    censoring: str = "right",
    truncation: object | None = None,
) -> WeibullFit:
    """Fit a fixed-location Weibull-min MLE for admitted v1 observations."""

    values = admitted_observations(
        observations,
        frequency_weights=frequency_weights,
        analytic_weights=analytic_weights,
        censoring=censoring,
        truncation=truncation,
    )
    count = len(values)
    events = sum(type(value) is ExactLifetime for value in values)
    def failure(code: WeibullFitFailureCode) -> WeibullFitFailure:
        return WeibullFitFailure(code, count, events, count - events)
    if count == 0:
        return failure(WeibullFitFailureCode.EMPTY_SAMPLE)
    if events == 0:
        return failure(WeibullFitFailureCode.NO_OBSERVED_EVENTS)
    if not positive_support(values):
        return failure(WeibullFitFailureCode.INVALID_SUPPORT)
    if fixed_shape is None:
        exact_times = tuple(value.time for value in values if type(value) is ExactLifetime)
        common_time = exact_times[0]
        if max(exact_times) == min(exact_times) and all(
            value.time <= common_time
            for value in values
            if type(value) is not ExactLifetime
        ):
            return failure(WeibullFitFailureCode.DEGENERATE_SAMPLE)

    # Divide every time by the sample's geometric mean before optimizing, so
    # `exp(shape * log_time)` cannot overflow merely because the input times
    # are reported in a different unit/scale. `mean_log` is the arithmetic
    # mean of the logs, i.e. the log of the geometric mean. The optimum shape
    # does not depend on this rescaling (it only shifts the profiled
    # log-likelihood by a shape-independent constant), and the reported scale
    # and log-likelihood are converted back to the original units below.
    log_times = tuple(log(value.time) for value in values)
    mean_log = fsum(log_times) / count
    scaled_log_times = tuple(value - mean_log for value in log_times)
    scaled_event_logs = tuple(
        scaled
        for scaled, value in zip(scaled_log_times, values, strict=True)
        if type(value) is ExactLifetime
    )

    sum_scaled_event_logs = fsum(scaled_event_logs)

    def at_shape(shape: float) -> tuple[float, float]:
        """Return `(scale, log_likelihood)` in the geometric-mean-scaled unit system."""

        total = fsum(exp(shape * value) for value in scaled_log_times)
        scale = exp(log(total / events) / shape)
        likelihood = (
            events * log(shape)
            - events * shape * log(scale)
            + (shape - 1.0) * sum_scaled_event_logs
            - total / scale**shape
        )
        return scale, likelihood

    try:
        if fixed_shape is None:
            def profile(log_shape: float) -> float:
                return at_shape(exp(log_shape))[1]

            lower, upper = _LOG_SHAPE_BOUNDS
            hard_lower, hard_upper = _LOG_SHAPE_HARD_LIMITS
            log_shape, _, boundary = expand_bracket(
                profile, lower=lower, upper=upper, hard_lower=hard_lower, hard_upper=hard_upper
            )
            if boundary:
                return failure(WeibullFitFailureCode.BOUNDARY_SOLUTION)
            shape = exp(log_shape)
        else:
            if not is_real(fixed_shape):
                raise TypeError("fixed_shape must be a positive real number or None")
            shape = float(fixed_shape)
            if not isfinite(shape) or shape <= 0.0:
                raise ValueError("fixed_shape must be finite and positive")
        scaled_scale, scaled_likelihood = at_shape(shape)
        scale = scaled_scale * exp(mean_log)
        likelihood = scaled_likelihood - events * mean_log
    except (ArithmeticError, OverflowError, ValueError):
        return failure(WeibullFitFailureCode.OPTIMIZER_EXHAUSTED)
    if not (isfinite(scale) and scale > 0.0 and isfinite(likelihood)):
        return failure(WeibullFitFailureCode.OPTIMIZER_EXHAUSTED)
    return WeibullFitSuccess(shape, scale, likelihood, count, events, count - events)


__all__ = [
    "WeibullFit",
    "WeibullFitFailure",
    "WeibullFitFailureCode",
    "WeibullFitSuccess",
    "fit_weibull",
]
