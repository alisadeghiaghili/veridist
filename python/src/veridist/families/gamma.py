"""Fixed-location gamma MLE for exact and right-censored lifetimes.

The gamma family has support ``(0, inf)``, so its observations are
:class:`~veridist.domain.lifetimes.ExactLifetime` and
:class:`~veridist.domain.lifetimes.RightCensoredLifetime`.  The real-valued
types (:class:`~veridist.domain.values.ExactValue` and
:class:`~veridist.domain.values.RightCensoredValue`) belong to the
whole-real-line families and are rejected with ``TypeError``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import exp, fsum, isfinite, lgamma, log, pi
from types import MappingProxyType
from typing import TYPE_CHECKING

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation
from veridist.families._evidence import FitEvidence
from veridist.families._reliability import (
    admitted_observations,
    expand_bracket,
    is_degenerate_sample,
    maximize_nested,
    positive_support,
)
from veridist.families.registry import FamilyId

if TYPE_CHECKING:
    from veridist.families.uncertainty import FitUncertainty, UncertaintyUnavailable
from veridist.statistics.distributions import _log_regularized_gamma_q
from veridist.statistics.log_density import _stirling_error

#: Starting and hard-limit bounds for the log-shape and the log-scale, the latter in
#: units of the mean of the exact times.
_LOG_SHAPE_BOUNDS = (-6.0, 6.0)
_LOG_SHAPE_HARD_LIMITS = (-20.0, 20.0)
_LOG_SCALE_BOUNDS = (-6.0, 6.0)
_LOG_SCALE_HARD_LIMITS = (-20.0, 20.0)
#: Shapes from here on use the Stirling series for ``k*log(k) - k - lgamma(k)``.
_STIRLING_SHAPE = 8.0


class GammaFitFailureCode(StrEnum):
    """Reasons a finite gamma point estimate is unavailable."""

    EMPTY_SAMPLE = "EMPTY_SAMPLE"
    NO_OBSERVED_EVENTS = "NO_OBSERVED_EVENTS"
    INVALID_SUPPORT = "INVALID_SUPPORT"
    OPTIMIZER_EXHAUSTED = "OPTIMIZER_EXHAUSTED"
    DEGENERATE_SAMPLE = "DEGENERATE_SAMPLE"
    BOUNDARY_SOLUTION = "BOUNDARY_SOLUTION"


@dataclass(frozen=True, slots=True)
class GammaFitFailure:
    """A declared reason no gamma point estimate is reported.

    ``converged`` is always ``False`` and ``restart_failures`` is always ``0``.
    Satisfies :class:`~veridist.families.results.FitFailure`.
    """

    code: GammaFitFailureCode
    observation_count: int
    event_count: int
    censored_count: int
    complete: bool = False
    converged: bool = False
    restart_failures: int = 0

    @property
    def family(self) -> FamilyId:
        """The family this failure belongs to."""

        return FamilyId.GAMMA


@dataclass(frozen=True, slots=True)
class GammaFitSuccess:
    """A gamma MLE point estimate from an interior, converged optimum.

    ``converged`` is always ``True``: a result that only exists at the edge of
    the search range is a :attr:`GammaFitFailureCode.BOUNDARY_SOLUTION` failure
    instead.  ``location`` is the fixed location ``0.0``.  Satisfies
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
    location: float = 0.0
    _evidence: FitEvidence = field(default_factory=FitEvidence, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (isfinite(self.shape) and self.shape > 0.0):
            raise ValueError("shape must be finite and positive")
        if not (isfinite(self.scale) and self.scale > 0.0 and isfinite(self.log_likelihood)):
            raise ValueError("scale and log likelihood must be finite")

    @property
    def family(self) -> FamilyId:
        """The family this result belongs to."""

        return FamilyId.GAMMA

    @property
    def parameters(self) -> Mapping[str, float]:
        """The fitted parameters under their canonical registry names, read-only."""

        return MappingProxyType({"shape": self.shape, "scale": self.scale})

    @property
    def uncertainty(self) -> FitUncertainty | UncertaintyUnavailable:
        """Covariance, standard errors and confidence intervals of the fit, computed lazily.

        An :class:`~veridist.families.uncertainty.UncertaintyUnavailable` (never an exception)
        when the observed information cannot be inverted.
        """

        return self._evidence.uncertainty(self.family, self.parameters)


GammaFit = GammaFitSuccess | GammaFitFailure


def _shape_profile(shape: float, mean_log: float) -> float:
    """Return the log-likelihood per exact time at the profiled scale ``1 / shape``.

    The times are standardized to have mean one and ``mean_log`` is the mean of
    their logarithms (so ``mean_log <= 0``).  The value is
    ``shape*log(shape) - shape - lgamma(shape) + (shape - 1)*mean_log``; from
    ``_STIRLING_SHAPE`` on, the first three terms are evaluated through the
    Stirling series so that they do not cancel.
    """

    if shape >= _STIRLING_SHAPE:
        base = 0.5 * log(shape / (2.0 * pi)) - _stirling_error(shape)
    else:
        base = shape * log(shape) - shape - lgamma(shape)
    return base + (shape - 1.0) * mean_log


def _log_likelihood(
    values: tuple[LifetimeObservation, ...], shape: float, scale: float
) -> float:
    """Log-likelihood of the original observations at ``(shape, scale)``."""

    log_scale = log(scale)
    constant = -lgamma(shape) - shape * log_scale
    terms = []
    for value in values:
        time = float(value.time)
        if type(value) is ExactLifetime:
            terms.append((shape - 1.0) * log(time) - time / scale + constant)
        else:
            terms.append(_log_regularized_gamma_q(shape, time / scale, log(time) - log_scale))
    return fsum(terms)


def fit_gamma(
    observations: Iterable[LifetimeObservation],
    *,
    frequency_weights: Iterable[int] | None = None,
    analytic_weights: object | None = None,
    censoring: str = "right",
    truncation: object | None = None,
) -> GammaFit:
    """Fit a fixed-location gamma distribution by maximum likelihood.

    ``observations`` are :class:`~veridist.domain.lifetimes.ExactLifetime` and
    :class:`~veridist.domain.lifetimes.RightCensoredLifetime` items with
    strictly positive times; the real-valued observation types raise
    ``TypeError``.

    The estimate maximizes the likelihood over ``log(shape)``.  Without
    censoring the scale for a given shape is the closed form ``mean / shape`` and
    the profile has a stable closed form; with right censoring the scale is
    profiled out by an inner search and each censored time contributes the
    log-survival ``log Q(shape, t / scale)`` from the log-domain incomplete gamma
    (never ``-inf`` for finite inputs).  The searches run on the times divided by
    the mean of the exact times, so the estimate is scale-equivariant, and they
    widen their bracket up to a hard limit.  If an optimum stays on the limit
    (which is what happens when the likelihood has no maximum, for example under
    very heavy censoring) the result is ``BOUNDARY_SOLUTION``, never a converged
    estimate.

    ``DEGENERATE_SAMPLE`` is returned when every exact time is identical (this
    includes a single observation) and no censoring time exceeds it, since the
    likelihood is then unbounded.
    """

    values = admitted_observations(
        observations,
        frequency_weights=frequency_weights,
        analytic_weights=analytic_weights,
        censoring=censoring,
        truncation=truncation,
    )
    count = len(values)
    events = sum(type(value) is ExactLifetime for value in values)

    def failure(code: GammaFitFailureCode) -> GammaFitFailure:
        return GammaFitFailure(code, count, events, count - events)

    if count == 0:
        return failure(GammaFitFailureCode.EMPTY_SAMPLE)
    if events == 0:
        return failure(GammaFitFailureCode.NO_OBSERVED_EVENTS)
    if not positive_support(values):
        return failure(GammaFitFailureCode.INVALID_SUPPORT)
    exact = tuple(float(value.time) for value in values if type(value) is ExactLifetime)
    censored = tuple(float(value.time) for value in values if type(value) is not ExactLifetime)
    if is_degenerate_sample(exact, censored):
        return failure(GammaFitFailureCode.DEGENERATE_SAMPLE)
    try:
        mean = fsum(exact) / events
        standardized = tuple(time / mean for time in exact)
        standardized_censored = tuple(time / mean for time in censored)
        if not all(isfinite(time) for time in (*standardized, *standardized_censored)):
            raise OverflowError("standardized times are not representable")
        sum_exact = fsum(standardized)
        sum_exact_logs = fsum(log(time) for time in standardized)
        mean_log = sum_exact_logs / events
        if events == count:
            log_shape, _, boundary = expand_bracket(
                lambda log_shape: _shape_profile(exp(log_shape), mean_log),
                lower=_LOG_SHAPE_BOUNDS[0],
                upper=_LOG_SHAPE_BOUNDS[1],
                hard_lower=_LOG_SHAPE_HARD_LIMITS[0],
                hard_upper=_LOG_SHAPE_HARD_LIMITS[1],
            )
            shape = exp(log_shape)
            scale_standardized = sum_exact / events / shape
        else:
            censored_groups = tuple(
                (log(time), occurrences)
                for time, occurrences in Counter(standardized_censored).items()
            )

            def at_log_shape(log_shape: float) -> Callable[[float], float]:
                shape_ = exp(log_shape)
                constant = (shape_ - 1.0) * sum_exact_logs - events * lgamma(shape_)

                def at_log_scale(log_scale: float) -> float:
                    exact_part = (
                        constant - sum_exact * exp(-log_scale) - events * shape_ * log_scale
                    )
                    terms = []
                    for log_time, occurrences in censored_groups:
                        log_ratio = log_time - log_scale
                        terms.append(
                            occurrences
                            * _log_regularized_gamma_q(shape_, exp(log_ratio), log_ratio)
                        )
                    return exact_part + fsum(terms)

                return at_log_scale

            log_shape, log_scale_standardized, _, boundary = maximize_nested(
                at_log_shape,
                outer_bounds=_LOG_SHAPE_BOUNDS,
                outer_hard_limits=_LOG_SHAPE_HARD_LIMITS,
                inner_bounds=_LOG_SCALE_BOUNDS,
                inner_hard_limits=_LOG_SCALE_HARD_LIMITS,
            )
            shape = exp(log_shape)
            scale_standardized = exp(log_scale_standardized)
        if boundary:
            return failure(GammaFitFailureCode.BOUNDARY_SOLUTION)
        scale = scale_standardized * mean
        likelihood = _log_likelihood(values, shape, scale)
        # The result class rejects a non-finite or non-positive estimate.
        return GammaFitSuccess(
            shape,
            scale,
            likelihood,
            count,
            events,
            count - events,
            _evidence=FitEvidence(exact, censored),
        )
    except (ArithmeticError, OverflowError, ValueError):
        return failure(GammaFitFailureCode.OPTIMIZER_EXHAUSTED)


__all__ = [
    "GammaFit",
    "GammaFitFailure",
    "GammaFitFailureCode",
    "GammaFitSuccess",
    "fit_gamma",
]
