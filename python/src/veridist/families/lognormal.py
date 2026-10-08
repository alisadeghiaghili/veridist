"""Fixed-location Lognormal MLE cell for exact/right-censored lifetimes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import erfc, exp, fsum, isfinite, log, log1p, pi, sqrt
from types import MappingProxyType
from typing import TYPE_CHECKING

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation
from veridist.families._evidence import FitEvidence
from veridist.families._reliability import (
    admitted_observations,
    bounded_maximize,
    expand_bracket,
    positive_support,
)
from veridist.families.registry import FamilyId

if TYPE_CHECKING:
    from veridist.families.uncertainty import FitUncertainty, UncertaintyUnavailable

_HALF_LOG_2PI = 0.5 * log(2.0 * pi)
_SQRT_2 = sqrt(2.0)

#: Starting and hard-limit half-widths for the log-sigma search (natural log units).
_LOG_SIGMA_BOUNDS = (-6.0, 6.0)
_LOG_SIGMA_HARD_LIMITS = (-20.0, 20.0)
#: mu's starting and hard-limit half-widths are these multiples of
#: ``max(1.0, sd(exact_logs))``.
_MU_START_MULTIPLE = 8.0
_MU_HARD_MULTIPLE = 64.0


class LognormalFitFailureCode(StrEnum):
    """Reasons a finite Lognormal point estimate is unavailable."""

    EMPTY_SAMPLE = "EMPTY_SAMPLE"
    NO_OBSERVED_EVENTS = "NO_OBSERVED_EVENTS"
    INVALID_SUPPORT = "INVALID_SUPPORT"
    OPTIMIZER_EXHAUSTED = "OPTIMIZER_EXHAUSTED"
    BOUNDARY_SOLUTION = "BOUNDARY_SOLUTION"


@dataclass(frozen=True, slots=True)
class LognormalFitFailure:
    """A declared reason no Lognormal point estimate is reported.

    ``converged`` is always ``False`` and ``restart_failures`` is always ``0``:
    a failure never ran an interior optimization to convergence, and no restart
    strategy is attempted.  Satisfies :class:`~veridist.families.results.FitFailure`.
    """

    code: LognormalFitFailureCode
    observation_count: int
    event_count: int
    censored_count: int
    complete: bool = False
    converged: bool = False
    restart_failures: int = 0

    @property
    def family(self) -> FamilyId:
        """The family this failure belongs to."""

        return FamilyId.LOGNORMAL


@dataclass(frozen=True, slots=True)
class LognormalFitSuccess:
    """A Lognormal MLE point estimate from an interior, converged optimum.

    ``converged`` is always ``True``: a result that only exists at the edge of
    the search range is reported as :attr:`LognormalFitFailureCode.BOUNDARY_SOLUTION`
    instead of a success. ``restart_failures`` is always ``0`` because the
    deterministic golden-section search never restarts.  Satisfies
    :class:`~veridist.families.results.FitSuccess`.
    """

    mu_log: float
    sigma_log: float
    log_likelihood: float
    observation_count: int
    event_count: int
    censored_count: int
    converged: bool = True
    restart_failures: int = 0
    complete: bool = True
    family: FamilyId = FamilyId.LOGNORMAL
    location: float = 0.0
    _evidence: FitEvidence = field(default_factory=FitEvidence, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not (isfinite(self.mu_log) and isfinite(self.sigma_log) and self.sigma_log > 0.0):
            raise ValueError("lognormal location and scale must be finite, with positive scale")
        if not isfinite(self.log_likelihood):
            raise ValueError("log likelihood must be finite")
        if self.family != FamilyId.LOGNORMAL:
            raise ValueError("a lognormal fit result belongs to the lognormal family")
        object.__setattr__(self, "family", FamilyId.LOGNORMAL)

    @property
    def parameters(self) -> Mapping[str, float]:
        """The fitted parameters under their canonical registry names, read-only."""

        return MappingProxyType({"mu_log": self.mu_log, "sigma_log": self.sigma_log})

    @property
    def uncertainty(self) -> FitUncertainty | UncertaintyUnavailable:
        """Covariance, standard errors and confidence intervals of the fit, computed lazily.

        An :class:`~veridist.families.uncertainty.UncertaintyUnavailable` (never an exception)
        when the observed information cannot be inverted.
        """

        return self._evidence.uncertainty(self.family, self.parameters)


LognormalFit = LognormalFitSuccess | LognormalFitFailure


def _log_normal_sf(z: float) -> float:
    """Stable log of the standard-normal upper-tail survival probability.

    Computes ``log(0.5 * erfc(z / sqrt(2)))`` directly while that stays
    representable (``>= 1e-300``). For larger ``z`` -- where ``erfc`` would
    underflow to exactly ``0.0`` and the direct formula would raise -- this
    falls back to the asymptotic Mills-ratio expansion of the upper tail:
    ``-z**2/2 - log(z) - 0.5*log(2*pi) + log1p(-1/z**2 + 3/z**4 - 15/z**6)``.
    Never raises for finite ``z``.
    """

    survival = 0.5 * erfc(z / _SQRT_2)
    if survival >= 1e-300:
        return log(survival)
    # Use `*` rather than `**` to form the powers of `z`: for huge `z` (e.g.
    # 1e100+) `z ** 4` raises OverflowError in CPython instead of saturating,
    # while `z * z` saturates to `inf`, and the correction terms it feeds into
    # (`1/z**2` etc.) are negligible deep in this regime regardless.
    z_squared = z * z
    a = 1.0 / z_squared
    correction = -a + 3.0 * a * a - 15.0 * a * a * a
    return -0.5 * z_squared - log(z) - _HALF_LOG_2PI + log1p(correction)


def _log_sf(time: float, mu: float, sigma: float) -> float:
    return _log_normal_sf((log(time) - mu) / sigma)


def fit_lognormal(
    observations: Iterable[LifetimeObservation],
    *,
    frequency_weights: Iterable[int] | None = None,
    analytic_weights: object | None = None,
    censoring: str = "right",
    truncation: object | None = None,
) -> LognormalFit:
    """Fit a Lognormal MLE for admitted v1 exact/right-censored observations."""

    values = admitted_observations(
        observations,
        frequency_weights=frequency_weights,
        analytic_weights=analytic_weights,
        censoring=censoring,
        truncation=truncation,
    )
    count = len(values)
    events = sum(type(value) is ExactLifetime for value in values)
    def failure(code: LognormalFitFailureCode) -> LognormalFitFailure:
        return LognormalFitFailure(code, count, events, count - events)
    if count == 0:
        return failure(LognormalFitFailureCode.EMPTY_SAMPLE)
    if events == 0:
        return failure(LognormalFitFailureCode.NO_OBSERVED_EVENTS)
    if not positive_support(values):
        return failure(LognormalFitFailureCode.INVALID_SUPPORT)
    exact_logs = tuple(log(value.time) for value in values if type(value) is ExactLifetime)
    try:
        if events == count:
            mu = fsum(exact_logs) / count
            sigma = sqrt(fsum((value - mu) ** 2 for value in exact_logs) / count)
            if sigma <= 0.0:
                raise ValueError("degenerate lognormal sample")
            likelihood = fsum(
                -log(value.time) - log(sigma) - 0.5 * log(2.0 * pi)
                - (log(value.time) - mu) ** 2 / (2.0 * sigma**2)
                for value in values
            )
        else:
            center = fsum(exact_logs) / events
            # Hoist the exact-observation sums out of the inner likelihood loop:
            # the exact-part log-likelihood is a quadratic in mu, so it only
            # needs these sums, computed once, regardless of n. Deviations are
            # taken from the exact-log center, so tightly clustered samples do
            # not lose precision to the cancellation of sum(x**2) - n*mu**2.
            sum_exact_logs = fsum(exact_logs)
            deviations = tuple(value - center for value in exact_logs)
            sum_deviation = fsum(deviations)
            sum_deviation_sq = fsum(value * value for value in deviations)
            # Group censored observations by identical time (a single censoring
            # time is common) so the censored-part sum is O(distinct times)
            # instead of O(n) per likelihood evaluation.
            censored_groups = tuple(
                Counter(
                    value.time for value in values if type(value) is not ExactLifetime
                ).items()
            )

            def at_log_sigma(mu: float, log_sigma: float) -> float:
                sigma = exp(log_sigma)
                shift = mu - center
                sum_sq_deviation = (
                    sum_deviation_sq - 2.0 * shift * sum_deviation + events * shift * shift
                )
                exact = (
                    -sum_exact_logs
                    - events * log(sigma)
                    - events * _HALF_LOG_2PI
                    - sum_sq_deviation / (2.0 * sigma**2)
                )
                censored = fsum(
                    occurrences * _log_sf(float(time), mu, sigma)
                    for time, occurrences in censored_groups
                )
                return exact + censored

            def profile(mu: float) -> float:
                _, likelihood, _ = bounded_maximize(
                    lambda log_sigma: at_log_sigma(mu, log_sigma), lower=-6.0, upper=6.0
                )
                return likelihood

            spread = sqrt(sum_deviation_sq / events)
            mu_half_width = _MU_START_MULTIPLE * max(1.0, spread)
            mu_hard_half_width = _MU_HARD_MULTIPLE * max(1.0, spread)
            mu, _, mu_boundary = expand_bracket(
                profile,
                lower=center - mu_half_width,
                upper=center + mu_half_width,
                hard_lower=center - mu_hard_half_width,
                hard_upper=center + mu_hard_half_width,
            )
            if mu_boundary:
                return failure(LognormalFitFailureCode.BOUNDARY_SOLUTION)

            lower, upper = _LOG_SIGMA_BOUNDS
            hard_lower, hard_upper = _LOG_SIGMA_HARD_LIMITS
            log_sigma, likelihood, sigma_boundary = expand_bracket(
                lambda log_sigma: at_log_sigma(mu, log_sigma),
                lower=lower,
                upper=upper,
                hard_lower=hard_lower,
                hard_upper=hard_upper,
            )
            if sigma_boundary:
                return failure(LognormalFitFailureCode.BOUNDARY_SOLUTION)
            sigma = exp(log_sigma)
    except (ArithmeticError, OverflowError, ValueError):
        return failure(LognormalFitFailureCode.OPTIMIZER_EXHAUSTED)
    if not (isfinite(mu) and isfinite(sigma) and sigma > 0.0 and isfinite(likelihood)):
        return failure(LognormalFitFailureCode.OPTIMIZER_EXHAUSTED)
    evidence = FitEvidence(
        tuple(float(value.time) for value in values if type(value) is ExactLifetime),
        tuple(float(value.time) for value in values if type(value) is not ExactLifetime),
    )
    return LognormalFitSuccess(
        mu, sigma, likelihood, count, events, count - events, _evidence=evidence
    )


__all__ = [
    "LognormalFit",
    "LognormalFitFailure",
    "LognormalFitFailureCode",
    "LognormalFitSuccess",
    "fit_lognormal",
]
