"""Small, explicit inference cells with caller-owned Monte Carlo randomness."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from math import expm1, isfinite, log, sqrt


class GofStatistic(StrEnum):
    """Statistics admitted to the uncensored exponential refit simulation cell."""

    KS = "KS"
    AD = "AD"
    CVM = "CVM"


class SelectionCode(StrEnum):
    SELECTED = "SELECTED"
    NONE_ADEQUATE = "NONE_ADEQUATE"


@dataclass(frozen=True, slots=True)
class InformationCriteria:
    aic: float
    bic: float
    free_parameters: int


@dataclass(frozen=True, slots=True)
class RefitMonteCarloGof:
    """Refit Monte Carlo goodness-of-fit result for one or more EDF statistics.

    `monte_carlo_standard_error` and `interval` describe exactly one
    statistic's p-value -- `primary_statistic` names which one. When more
    than one statistic was requested, the other p-values are still present
    in `p_values`, but they have no standard error or interval reported here.
    """

    requested_replicates: int
    successful_replicates: int
    failed_replicates: int
    monte_carlo_standard_error: float
    interval: tuple[float, float]
    p_values: Mapping[GofStatistic, float]
    primary_statistic: GofStatistic
    method: str = "refit_monte_carlo"
    rng_policy: str = "caller_owned_generator"


@dataclass(frozen=True, slots=True)
class ModelSelection:
    code: SelectionCode
    selected_family: str | None


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    rejection_rate: float
    standard_error: float
    scope: str = "declared_grid_only"


@dataclass(frozen=True, slots=True)
class BootstrapInterval:
    lower: float
    upper: float
    requested_replicates: int
    successful_replicates: int
    failed_replicates: int
    method: str
    rng_policy: str = "caller_owned_generator"


def information_criteria(
    *, log_likelihood: float, sample_size: int, free_parameters: int
) -> InformationCriteria:
    """Compute AIC/BIC from an explicit declared free-parameter count."""

    if type(log_likelihood) is not float or not isfinite(log_likelihood):
        raise ValueError("log_likelihood must be a finite built-in float")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 1:
        raise ValueError("sample_size must be a positive built-in integer")
    if (
        isinstance(free_parameters, bool)
        or not isinstance(free_parameters, int)
        or free_parameters < 0
    ):
        raise ValueError("free_parameters must be a non-negative built-in integer")
    return InformationCriteria(
        -2.0 * log_likelihood + 2.0 * free_parameters,
        -2.0 * log_likelihood + free_parameters * log(sample_size),
        free_parameters,
    )


def _empirical_statistics(values: tuple[float, ...]) -> Mapping[GofStatistic, float]:
    """Calculate three EDF statistics after the exponential MLE is refit."""

    if not values or any(not isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("the admitted cell requires finite positive uncensored observations")
    ordered = tuple(sorted(values))
    rate = len(ordered) / sum(ordered)
    probabilities = tuple(-expm1(-rate * value) for value in ordered)
    size = len(ordered)
    ks = max(
        max((index + 1) / size - probability, probability - index / size)
        for index, probability in enumerate(probabilities)
    )
    clipped = tuple(min(1.0 - 1e-15, max(1e-15, value)) for value in probabilities)
    ad = (
        -size
        - sum(
            (2 * index + 1) * (log(probability) + log(1.0 - clipped[-index - 1]))
            for index, probability in enumerate(clipped)
        )
        / size
    )
    cvm = 1.0 / (12.0 * size) + sum(
        (probability - (2 * index + 1) / (2.0 * size)) ** 2
        for index, probability in enumerate(clipped)
    )
    return {GofStatistic.KS: ks, GofStatistic.AD: ad, GofStatistic.CVM: cvm}


def refit_monte_carlo_gof(
    *,
    observations: Iterable[float],
    family: str,
    statistics: frozenset[GofStatistic],
    replicates: int,
    rng: object,
) -> RefitMonteCarloGof:
    """Calibrate admitted exponential EDF statistics by refitting every replicate."""

    np = import_module("numpy")

    if family != "exponential":
        raise ValueError("only the uncensored exponential refit cell is admitted")
    if (
        type(statistics) is not frozenset
        or not statistics
        or any(type(statistic) is not GofStatistic for statistic in statistics)
    ):
        raise TypeError("statistics must be a non-empty frozenset of GofStatistic values")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive built-in integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    values = tuple(observations)
    observed = _empirical_statistics(values)
    requested = tuple(sorted(statistics, key=str))
    exceedances = {statistic: 0 for statistic in requested}
    successful = 0
    failed = 0
    sample_size = len(values)
    fitted_rate = sample_size / sum(values)
    for _ in range(replicates):
        try:
            generated = tuple(
                float(value) for value in rng.exponential(1.0 / fitted_rate, sample_size)
            )
            trial = _empirical_statistics(generated)
        except (ArithmeticError, ValueError):
            failed += 1
            continue
        successful += 1
        for statistic in requested:
            if trial[statistic] >= observed[statistic]:
                exceedances[statistic] += 1
    if successful == 0:
        raise RuntimeError("all Monte Carlo refits failed")
    p_values = {
        statistic: (exceedances[statistic] + 1.0) / (successful + 1.0) for statistic in requested
    }
    primary_statistic = requested[0]
    primary = p_values[primary_statistic]
    standard_error = sqrt(primary * (1.0 - primary) / successful)
    return RefitMonteCarloGof(
        replicates,
        successful,
        failed,
        standard_error,
        (max(0.0, primary - 1.96 * standard_error), min(1.0, primary + 1.96 * standard_error)),
        p_values,
        primary_statistic,
    )


def compare_models(
    *, candidates: Iterable[Mapping[str, object]], adequacy_threshold: float
) -> ModelSelection:
    """Choose the lowest-AIC adequate candidate, or return the typed empty outcome.

    Passing the adequacy gate (``p_value >= adequacy_threshold``) means a
    candidate was not rejected by the admitted goodness-of-fit test; it is
    not evidence that the candidate is an adequate model in any absolute
    sense, and a small or unrepresentative sample can fail to reject a poor
    model. Each candidate's ``aic`` must be computed from the same data as
    every other candidate's; AIC values fit on different samples, subsets,
    or censoring are not comparable and this function has no way to detect
    that they were not.
    """

    if type(adequacy_threshold) is not float or not 0.0 <= adequacy_threshold <= 1.0:
        raise ValueError("adequacy_threshold must be a built-in probability")
    admitted: list[tuple[float, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("candidates must be mappings")
        family, aic, p_value = (
            candidate.get("family"),
            candidate.get("aic"),
            candidate.get("p_value"),
        )
        if type(family) is not str or type(aic) is not float or type(p_value) is not float:
            raise TypeError("candidate family, aic, and p_value must be built-in values")
        if not isfinite(aic) or not isfinite(p_value):
            raise ValueError("candidate evidence must be finite")
        if p_value >= adequacy_threshold:
            admitted.append((aic, family))
    if not admitted:
        return ModelSelection(SelectionCode.NONE_ADEQUATE, None)
    return ModelSelection(SelectionCode.SELECTED, min(admitted)[1])


def summarize_calibration(
    *, rejections: int, replicates: int, nominal_alpha: float
) -> CalibrationSummary:
    """Report a binomial calibration rate and its declared binomial uncertainty."""

    if isinstance(rejections, bool) or not isinstance(rejections, int) or rejections < 0:
        raise ValueError("rejections must be a non-negative built-in integer")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive built-in integer")
    if rejections > replicates:
        raise ValueError("rejections cannot exceed replicates")
    if type(nominal_alpha) is not float or not 0.0 < nominal_alpha < 1.0:
        raise ValueError("nominal_alpha must be a built-in interior probability")
    return CalibrationSummary(
        rejections / replicates,
        sqrt(nominal_alpha * (1.0 - nominal_alpha) / replicates),
    )


__all__ = [
    "BootstrapInterval",
    "CalibrationSummary",
    "GofStatistic",
    "InformationCriteria",
    "ModelSelection",
    "RefitMonteCarloGof",
    "SelectionCode",
    "compare_models",
    "information_criteria",
    "refit_monte_carlo_gof",
    "summarize_calibration",
]
