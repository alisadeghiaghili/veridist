"""Right-Gumbel (largest-extreme-value) MLE for exact and right-censored real values.

The Gumbel family lives on the whole real line, so its observations are
:class:`~veridist.domain.values.ExactValue` and
:class:`~veridist.domain.values.RightCensoredValue`.  Lifetime observations
(:class:`~veridist.domain.lifetimes.ExactLifetime` and
:class:`~veridist.domain.lifetimes.RightCensoredLifetime`) belong to the
positive-support families and are rejected with ``TypeError``.

The density is ``exp(-z - exp(-z)) / scale`` with ``z = (x - location) / scale``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import exp, expm1, fsum, isfinite, log, log1p
from types import MappingProxyType
from typing import TYPE_CHECKING

from veridist.domain.values import ExactValue, RealObservation
from veridist.families._evidence import FitEvidence
from veridist.families._reliability import (
    admitted_real_observations,
    exp_or_inf,
    expand_bracket,
    is_degenerate_sample,
    maximize_nested,
)
from veridist.families.registry import FamilyId

if TYPE_CHECKING:
    from veridist.families.uncertainty import FitUncertainty, UncertaintyUnavailable

#: Starting and hard-limit bounds for the log-scale and the location, in units of the
#: largest deviation of any observation from the mean of the exact values.
_LOG_SCALE_BOUNDS = (-6.0, 6.0)
_LOG_SCALE_HARD_LIMITS = (-20.0, 20.0)
_LOCATION_BOUNDS = (-8.0, 8.0)
_LOCATION_HARD_LIMITS = (-64.0, 64.0)
#: Beyond this ``exp`` overflows in binary64 (``log(max_float)`` is about 709.78).
_EXP_LIMIT = 700.0


class GumbelFitFailureCode(StrEnum):
    """Reasons a finite right-Gumbel point estimate is unavailable."""

    EMPTY_SAMPLE = "EMPTY_SAMPLE"
    NO_OBSERVED_EVENTS = "NO_OBSERVED_EVENTS"
    OPTIMIZER_EXHAUSTED = "OPTIMIZER_EXHAUSTED"
    DEGENERATE_SAMPLE = "DEGENERATE_SAMPLE"
    BOUNDARY_SOLUTION = "BOUNDARY_SOLUTION"


@dataclass(frozen=True, slots=True)
class GumbelFitFailure:
    """A declared reason no right-Gumbel point estimate is reported.

    ``converged`` is always ``False`` and ``restart_failures`` is always ``0``.
    Satisfies :class:`~veridist.families.results.FitFailure`.
    """

    code: GumbelFitFailureCode
    observation_count: int
    event_count: int
    censored_count: int
    complete: bool = False
    converged: bool = False
    restart_failures: int = 0

    @property
    def family(self) -> FamilyId:
        """The family this failure belongs to."""

        return FamilyId.GUMBEL_RIGHT


@dataclass(frozen=True, slots=True)
class GumbelFitSuccess:
    """A right-Gumbel MLE point estimate from an interior, converged optimum.

    ``converged`` is always ``True``: a result that only exists at the edge of
    the search range is a :attr:`GumbelFitFailureCode.BOUNDARY_SOLUTION`
    failure instead.  Satisfies :class:`~veridist.families.results.FitSuccess`.
    """

    location: float
    scale: float
    log_likelihood: float
    observation_count: int
    event_count: int
    censored_count: int
    converged: bool = True
    restart_failures: int = 0
    complete: bool = True
    _evidence: FitEvidence = field(default_factory=FitEvidence, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (isfinite(self.location) and isfinite(self.scale) and self.scale > 0.0):
            raise ValueError("gumbel location and scale must be finite, with positive scale")
        if not isfinite(self.log_likelihood):
            raise ValueError("log likelihood must be finite")

    @property
    def family(self) -> FamilyId:
        """The family this result belongs to."""

        return FamilyId.GUMBEL_RIGHT

    @property
    def parameters(self) -> Mapping[str, float]:
        """The fitted parameters under their canonical registry names, read-only."""

        return MappingProxyType({"location": self.location, "scale": self.scale})

    def uncertainty(self) -> FitUncertainty | UncertaintyUnavailable:
        """Covariance, standard errors and confidence intervals of the fit, computed lazily.

        An :class:`~veridist.families.uncertainty.UncertaintyUnavailable` (never an exception)
        when the observed information cannot be inverted.
        """

        return self._evidence.uncertainty(self.family, self.parameters)


GumbelFit = GumbelFitSuccess | GumbelFitFailure


def _log_survival(z: float) -> float:
    """Return ``log(1 - exp(-exp(-z)))`` for a finite standardized ``z``, never ``-inf``."""

    if z < -_EXP_LIMIT:
        return 0.0
    tail = exp(-z)
    if tail > 0.7:
        return log1p(-exp(-tail))
    if tail < 1e-300:
        return -z
    return log(-expm1(-tail))


def _log_sum_exp_negated(values: tuple[float, ...], scale: float) -> float:
    """Return ``log(sum(exp(-value / scale)))`` without overflow."""

    exponents = tuple(-value / scale for value in values)
    peak = max(exponents)
    return peak + log(fsum(exp(exponent - peak) for exponent in exponents))


def _log_likelihood(values: tuple[RealObservation, ...], location: float, scale: float) -> float:
    """Log-likelihood of the original observations at ``(location, scale)``."""

    terms = []
    for value in values:
        z = (value.value - location) / scale
        if type(value) is ExactValue:
            terms.append(-log(scale) - z - exp(-z))
        else:
            terms.append(_log_survival(z))
    return fsum(terms)


def fit_gumbel_right(
    observations: Iterable[RealObservation],
    *,
    frequency_weights: Iterable[int] | None = None,
    analytic_weights: object | None = None,
    censoring: str = "right",
    truncation: object | None = None,
) -> GumbelFit:
    """Fit a right-Gumbel distribution by maximum likelihood to exact/right-censored values.

    ``observations`` are :class:`~veridist.domain.values.ExactValue` and
    :class:`~veridist.domain.values.RightCensoredValue` items; lifetime
    observations raise ``TypeError``.

    Without censoring the location has a closed form for each scale, and the
    estimate maximizes the resulting one-dimensional profile of ``log(scale)``.
    With right censoring the location is profiled out of a search over
    ``log(scale)``.  Both searches widen their bracket up to a hard limit and
    report ``BOUNDARY_SOLUTION`` instead of a converged result if the optimum
    stays on the limit.  They run on the data standardized by the mean of the
    exact values and the largest deviation of any observation from it, so the
    estimate is equivariant under ``x -> a * x + b`` with ``a > 0`` (a negative
    ``a`` would turn the right Gumbel into a left one); the search covers scales
    from about ``2e-9`` to ``5e8`` times that deviation and locations within
    ``64`` of them.

    ``DEGENERATE_SAMPLE`` is returned when every exact value is identical (this
    includes a single observation) and no censoring point exceeds it, since the
    likelihood is then unbounded.
    """

    values = admitted_real_observations(
        observations,
        frequency_weights=frequency_weights,
        analytic_weights=analytic_weights,
        censoring=censoring,
        truncation=truncation,
    )
    count = len(values)
    events = sum(type(value) is ExactValue for value in values)

    def failure(code: GumbelFitFailureCode) -> GumbelFitFailure:
        return GumbelFitFailure(code, count, events, count - events)

    if count == 0:
        return failure(GumbelFitFailureCode.EMPTY_SAMPLE)
    if events == 0:
        return failure(GumbelFitFailureCode.NO_OBSERVED_EVENTS)
    exact = tuple(value.value for value in values if type(value) is ExactValue)
    censored = tuple(value.value for value in values if type(value) is not ExactValue)
    if is_degenerate_sample(exact, censored):
        return failure(GumbelFitFailureCode.DEGENERATE_SAMPLE)
    try:
        center = fsum(exact) / events
        peak = max(abs(value.value - center) for value in values)
        if not isfinite(peak):
            raise OverflowError("deviations are not representable")
        standardized = tuple((value - center) / peak for value in exact)
        sum_exact = fsum(standardized)
        if events == count:

            def profile(log_scale: float) -> float:
                scale_ = exp(log_scale)
                location_ = -scale_ * (_log_sum_exp_negated(standardized, scale_) - log(events))
                return -events * log_scale - (sum_exact - events * location_) / scale_ - events

            log_scale, _, boundary = expand_bracket(
                profile,
                lower=_LOG_SCALE_BOUNDS[0],
                upper=_LOG_SCALE_BOUNDS[1],
                hard_lower=_LOG_SCALE_HARD_LIMITS[0],
                hard_upper=_LOG_SCALE_HARD_LIMITS[1],
            )
            scale_standardized = exp(log_scale)
            location_standardized = -scale_standardized * (
                _log_sum_exp_negated(standardized, scale_standardized) - log(events)
            )
        else:
            censored_groups = tuple(Counter((value - center) / peak for value in censored).items())

            def at_log_scale(log_scale: float) -> Callable[[float], float]:
                scale_ = exp(log_scale)
                log_total = _log_sum_exp_negated(standardized, scale_)

                def at_location(location_: float) -> float:
                    exact_part = (
                        -events * log_scale
                        - (sum_exact - events * location_) / scale_
                        - exp_or_inf(location_ / scale_ + log_total)
                    )
                    censored_part = fsum(
                        occurrences * _log_survival((time - location_) / scale_)
                        for time, occurrences in censored_groups
                    )
                    return exact_part + censored_part

                return at_location

            log_scale, location_standardized, _, boundary = maximize_nested(
                at_log_scale,
                outer_bounds=_LOG_SCALE_BOUNDS,
                outer_hard_limits=_LOG_SCALE_HARD_LIMITS,
                inner_bounds=_LOCATION_BOUNDS,
                inner_hard_limits=_LOCATION_HARD_LIMITS,
            )
            scale_standardized = exp(log_scale)
        if boundary:
            return failure(GumbelFitFailureCode.BOUNDARY_SOLUTION)
        location = center + peak * location_standardized
        scale = peak * scale_standardized
        likelihood = _log_likelihood(values, location, scale)
        # The result class rejects a non-finite or non-positive estimate.
        return GumbelFitSuccess(
            location,
            scale,
            likelihood,
            count,
            events,
            count - events,
            _evidence=FitEvidence(exact, censored),
        )
    except (ArithmeticError, OverflowError, ValueError):
        return failure(GumbelFitFailureCode.OPTIMIZER_EXHAUSTED)


__all__ = [
    "GumbelFit",
    "GumbelFitFailure",
    "GumbelFitFailureCode",
    "GumbelFitSuccess",
    "fit_gumbel_right",
]
