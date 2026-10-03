"""Normal MLE for exact and right-censored real values.

The normal family lives on the whole real line, so its observations are
:class:`~veridist.domain.values.ExactValue` and
:class:`~veridist.domain.values.RightCensoredValue`.  Lifetime observations
(:class:`~veridist.domain.lifetimes.ExactLifetime` and
:class:`~veridist.domain.lifetimes.RightCensoredLifetime`) belong to the
positive-support families and are rejected with ``TypeError``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from math import exp, fsum, isfinite, log, pi, sqrt
from types import MappingProxyType

from veridist.domain.values import ExactValue, RealObservation
from veridist.families._reliability import (
    admitted_real_observations,
    is_degenerate_sample,
    maximize_nested,
)
from veridist.families.lognormal import _log_normal_sf
from veridist.families.registry import FamilyId

_HALF_LOG_2PI = 0.5 * log(2.0 * pi)

#: Starting and hard-limit bounds for the log-sigma and the mean, in units of the
#: largest deviation of any observation from the mean of the exact values.
_LOG_SIGMA_BOUNDS = (-6.0, 6.0)
_LOG_SIGMA_HARD_LIMITS = (-20.0, 20.0)
_MU_BOUNDS = (-8.0, 8.0)
_MU_HARD_LIMITS = (-64.0, 64.0)


class NormalFitFailureCode(StrEnum):
    """Reasons a finite normal point estimate is unavailable."""

    EMPTY_SAMPLE = "EMPTY_SAMPLE"
    NO_OBSERVED_EVENTS = "NO_OBSERVED_EVENTS"
    OPTIMIZER_EXHAUSTED = "OPTIMIZER_EXHAUSTED"
    DEGENERATE_SAMPLE = "DEGENERATE_SAMPLE"
    BOUNDARY_SOLUTION = "BOUNDARY_SOLUTION"


@dataclass(frozen=True, slots=True)
class NormalFitFailure:
    """A declared reason no normal point estimate is reported.

    ``converged`` is always ``False`` and ``restart_failures`` is always ``0``.
    Satisfies :class:`~veridist.families.results.FitFailure`.
    """

    code: NormalFitFailureCode
    observation_count: int
    event_count: int
    censored_count: int
    complete: bool = False
    converged: bool = False
    restart_failures: int = 0

    @property
    def family(self) -> FamilyId:
        """The family this failure belongs to."""

        return FamilyId.NORMAL


@dataclass(frozen=True, slots=True)
class NormalFitSuccess:
    """A normal MLE point estimate.

    ``sigma`` is the **maximum-likelihood** standard deviation: with no
    censoring it is the root of the mean squared deviation (divisor ``n``), not
    the unbiased sample estimate (divisor ``n - 1``).  ``converged`` is always
    ``True``: a result that only exists at the edge of the search range is a
    :attr:`NormalFitFailureCode.BOUNDARY_SOLUTION` failure instead.
    Satisfies :class:`~veridist.families.results.FitSuccess`.
    """

    mu: float
    sigma: float
    log_likelihood: float
    observation_count: int
    event_count: int
    censored_count: int
    converged: bool = True
    restart_failures: int = 0
    complete: bool = True

    def __post_init__(self) -> None:
        if not (isfinite(self.mu) and isfinite(self.sigma) and self.sigma > 0.0):
            raise ValueError("normal location and scale must be finite, with positive scale")
        if not isfinite(self.log_likelihood):
            raise ValueError("log likelihood must be finite")

    @property
    def family(self) -> FamilyId:
        """The family this result belongs to."""

        return FamilyId.NORMAL

    @property
    def parameters(self) -> Mapping[str, float]:
        """The fitted parameters under their canonical registry names, read-only."""

        return MappingProxyType({"mu": self.mu, "sigma": self.sigma})


NormalFit = NormalFitSuccess | NormalFitFailure


def _log_likelihood(values: tuple[RealObservation, ...], mu: float, sigma: float) -> float:
    """Log-likelihood of the original observations at ``(mu, sigma)``."""

    terms = []
    for value in values:
        z = (value.value - mu) / sigma
        if type(value) is ExactValue:
            terms.append(-log(sigma) - _HALF_LOG_2PI - 0.5 * z * z)
        else:
            terms.append(_log_normal_sf(z))
    return fsum(terms)


def fit_normal(
    observations: Iterable[RealObservation],
    *,
    frequency_weights: Iterable[int] | None = None,
    analytic_weights: object | None = None,
    censoring: str = "right",
    truncation: object | None = None,
) -> NormalFit:
    """Fit a normal distribution by maximum likelihood to exact/right-censored values.

    ``observations`` are :class:`~veridist.domain.values.ExactValue` and
    :class:`~veridist.domain.values.RightCensoredValue` items; lifetime
    observations raise ``TypeError``.

    Without censoring the estimate is closed form: ``mu`` is the sample mean and
    ``sigma`` the maximum-likelihood standard deviation, which divides the sum
    of squared deviations by ``n`` (it is *not* the unbiased ``n - 1``
    estimate).  With right censoring the estimate maximizes the censored
    log-likelihood by a profile search (the mean is profiled out of a search
    over ``log(sigma)``) that widens its bracket up to a hard limit and reports
    ``BOUNDARY_SOLUTION`` instead of a converged result if the optimum stays on
    the limit.  The search runs on the data standardized by the mean of the exact
    values and the largest deviation of any observation from it, so the estimate
    is equivariant under ``x -> a * x + b`` with ``a > 0``; it covers scales from
    about ``2e-9`` to ``5e8`` times that deviation and means within ``64`` of them,
    and anything beyond is reported as ``BOUNDARY_SOLUTION``.

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

    def failure(code: NormalFitFailureCode) -> NormalFitFailure:
        return NormalFitFailure(code, count, events, count - events)

    if count == 0:
        return failure(NormalFitFailureCode.EMPTY_SAMPLE)
    if events == 0:
        return failure(NormalFitFailureCode.NO_OBSERVED_EVENTS)
    exact = tuple(value.value for value in values if type(value) is ExactValue)
    censored = tuple(value.value for value in values if type(value) is not ExactValue)
    if is_degenerate_sample(exact, censored):
        return failure(NormalFitFailureCode.DEGENERATE_SAMPLE)
    try:
        center = fsum(exact) / events
        peak = max(abs(value.value - center) for value in values)
        if not isfinite(peak):
            raise OverflowError("deviations are not representable")
        if events == count:
            mu = center
            sigma = peak * sqrt(fsum(((value - center) / peak) ** 2 for value in exact) / count)
        else:
            standardized = tuple((value - center) / peak for value in exact)
            sum_exact = fsum(standardized)
            sum_exact_squares = fsum(value * value for value in standardized)
            censored_groups = tuple(Counter((value - center) / peak for value in censored).items())

            def at_log_sigma(log_sigma: float) -> Callable[[float], float]:
                sigma_ = exp(log_sigma)
                base = -events * (log_sigma + _HALF_LOG_2PI)

                def at_mu(mu_: float) -> float:
                    squares = sum_exact_squares - 2.0 * mu_ * sum_exact + events * mu_ * mu_
                    exact_part = base - squares / (2.0 * sigma_ * sigma_)
                    censored_part = fsum(
                        occurrences * _log_normal_sf((time - mu_) / sigma_)
                        for time, occurrences in censored_groups
                    )
                    return exact_part + censored_part

                return at_mu

            log_sigma, mu_standardized, _, boundary = maximize_nested(
                at_log_sigma,
                outer_bounds=_LOG_SIGMA_BOUNDS,
                outer_hard_limits=_LOG_SIGMA_HARD_LIMITS,
                inner_bounds=_MU_BOUNDS,
                inner_hard_limits=_MU_HARD_LIMITS,
            )
            if boundary:
                return failure(NormalFitFailureCode.BOUNDARY_SOLUTION)
            mu = center + peak * mu_standardized
            sigma = peak * exp(log_sigma)
        likelihood = _log_likelihood(values, mu, sigma)
        # The result class rejects a non-finite or non-positive estimate.
        return NormalFitSuccess(mu, sigma, likelihood, count, events, count - events)
    except (ArithmeticError, OverflowError, ValueError):
        return failure(NormalFitFailureCode.OPTIMIZER_EXHAUSTED)


__all__ = [
    "NormalFit",
    "NormalFitFailure",
    "NormalFitFailureCode",
    "NormalFitSuccess",
    "fit_normal",
]
