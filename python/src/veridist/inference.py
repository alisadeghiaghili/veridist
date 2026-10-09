"""Small, explicit inference cells with caller-owned Monte Carlo randomness."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module
from math import expm1, fsum, isfinite, log, sqrt
from typing import Any, Final, cast

from veridist.domain._numeric import is_float, is_integer, is_real
from veridist.domain.lifetimes import ExactLifetime
from veridist.domain.values import ExactValue
from veridist.engine.errors import VeridistError
from veridist.families.dispatch import fit as _fit_family
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.families.results import FitSuccess
from veridist.statistics.distributions import cdf, sample


class GofStatistic(StrEnum):
    """Statistics admitted to the uncensored refit simulation cell."""

    KS = "KS"
    AD = "AD"
    CVM = "CVM"


class SelectionCode(StrEnum):
    SELECTED = "SELECTED"
    NONE_ADEQUATE = "NONE_ADEQUATE"


class GofFitError(VeridistError, RuntimeError):
    """The observed sample could not be fitted, so no p-value can be computed.

    ``family`` is the :class:`~veridist.families.registry.FamilyId` that was
    being fitted and ``code`` is the reason: the ``code`` value of the fit
    failure (for example ``"DEGENERATE_SAMPLE"``), ``"NOT_CONVERGED"`` when the
    fit reported no interior optimum, or ``"NOT_REPRESENTABLE"`` when the fitted
    CDF could not be evaluated at the observations.
    """

    def __init__(self, family: FamilyId, code: str) -> None:
        super().__init__(
            f"the {family.value} fit of the observed sample failed ({code}); "
            "no p-value is computed from a failed fit"
        )
        self.family = family
        self.code = code


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
class FamilyCandidate:
    """One family's row in a :class:`FamilyAssessment`.

    A family whose observed fit succeeded carries its log-likelihood, the
    number of free parameters, AIC/BIC, the refit Monte Carlo result and the
    p-value of the assessed statistic; ``failure_code`` is ``None`` and
    ``adequate`` says whether that p-value reached the adequacy threshold.  A
    family that could not be assessed carries only ``failure_code`` (a fit
    failure code, ``"INVALID_SUPPORT"``, ``"SAMPLE_TOO_SMALL"``,
    ``"NOT_CONVERGED"`` or ``"NOT_REPRESENTABLE"``), every other value is
    ``None`` and ``adequate`` is ``False``: such a family is not eligible for
    selection.
    """

    family: FamilyId
    failure_code: str | None
    log_likelihood: float | None
    free_parameters: int | None
    aic: float | None
    bic: float | None
    gof: RefitMonteCarloGof | None
    p_value: float | None
    adequate: bool


@dataclass(frozen=True, slots=True)
class FamilyAssessment:
    """Per-family evidence and the adequacy-gated selection that follows from it.

    ``candidates`` has one row per assessed family, in the order the families
    were assessed (the caller's order, or the registry order by default).
    ``selection`` is the lowest-AIC family among the candidates whose p-value
    reached ``adequacy_threshold``, or ``NONE_ADEQUATE``.
    """

    candidates: tuple[FamilyCandidate, ...]
    selection: ModelSelection
    statistic: GofStatistic
    adequacy_threshold: float
    sample_size: int
    requested_replicates: int
    method: str = "refit_monte_carlo"
    rng_policy: str = "caller_owned_generator"


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


#: Families whose observations are lifetimes: strictly positive values.
_LIFETIME_FAMILIES: Final = frozenset(
    {FamilyId.EXPONENTIAL, FamilyId.GAMMA, FamilyId.WEIBULL_MIN, FamilyId.LOGNORMAL}
)
#: The fewest observations the refit cell accepts.  A two-parameter fit to two
#: points leaves no degree of freedom for the test to check, so those families
#: need three; the one-parameter exponential cell keeps its original minimum.
_MINIMUM_OBSERVATIONS: Final = {
    FamilyId.NORMAL: 3,
    FamilyId.GAMMA: 3,
    FamilyId.WEIBULL_MIN: 3,
    FamilyId.LOGNORMAL: 3,
    FamilyId.GUMBEL_RIGHT: 3,
    FamilyId.EXPONENTIAL: 1,
}
#: Probabilities are clipped to this margin before the logarithms of AD and CvM.
_PROBABILITY_MARGIN: Final = 1e-15


def information_criteria(
    *, log_likelihood: float, sample_size: int, free_parameters: int
) -> InformationCriteria:
    """Compute AIC/BIC from an explicit declared free-parameter count.

    ``log_likelihood`` is a finite real scalar and the two counts are integers;
    Python and numpy scalars are accepted alike, ``bool`` never is.
    """

    if not is_float(log_likelihood) or not isfinite(log_likelihood):
        raise ValueError("log_likelihood must be a finite real float")
    if not is_integer(sample_size) or sample_size < 1:
        raise ValueError("sample_size must be a positive integer")
    if not is_integer(free_parameters) or free_parameters < 0:
        raise ValueError("free_parameters must be a non-negative integer")
    value = float(log_likelihood)
    count = int(free_parameters)
    return InformationCriteria(
        -2.0 * value + 2.0 * count,
        -2.0 * value + count * log(int(sample_size)),
        count,
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


def _edf_statistics(probabilities: Sequence[float]) -> Mapping[GofStatistic, float]:
    """KS, AD and CvM from the fitted-CDF values of the ordered observations.

    The formulas are those of :func:`_empirical_statistics`; the sums use
    ``fsum``.  The exponential cell keeps its own arithmetic so that its
    results do not change.
    """

    size = len(probabilities)
    ks = max(
        max((index + 1) / size - probability, probability - index / size)
        for index, probability in enumerate(probabilities)
    )
    clipped = tuple(
        min(1.0 - _PROBABILITY_MARGIN, max(_PROBABILITY_MARGIN, value)) for value in probabilities
    )
    ad = (
        -size
        - fsum(
            (2 * index + 1) * (log(probability) + log(1.0 - clipped[-index - 1]))
            for index, probability in enumerate(clipped)
        )
        / size
    )
    cvm = 1.0 / (12.0 * size) + fsum(
        (probability - (2 * index + 1) / (2.0 * size)) ** 2
        for index, probability in enumerate(clipped)
    )
    return {GofStatistic.KS: ks, GofStatistic.AD: ad, GofStatistic.CVM: cvm}


def _model_statistics(
    family: FamilyId, ordered: Sequence[float], parameters: Mapping[str, float]
) -> Mapping[GofStatistic, float]:
    """EDF statistics of the ordered observations against the fitted model's CDF."""

    np = import_module("numpy")

    probabilities = cdf(family, np.asarray(ordered, dtype=np.float64), **parameters).tolist()
    if not all(isfinite(probability) for probability in probabilities):
        raise ArithmeticError("the fitted CDF is not finite at every observation")
    return _edf_statistics(probabilities)


def _typed_observations(family: FamilyId, values: Iterable[float]) -> tuple[Any, ...]:
    """Wrap plain values in the exact-observation type ``family`` is fitted from."""

    if family in _LIFETIME_FAMILIES:
        return tuple(ExactLifetime(value) for value in values)
    return tuple(ExactValue(value) for value in values)


def _converged_fit(family: FamilyId, values: Iterable[float]) -> tuple[FitSuccess | None, str]:
    """Fit ``family`` to exact observations: the fit, or ``None`` and the reason it failed."""

    result = _fit_family(family, _typed_observations(family, values))
    if not isinstance(result, FitSuccess):
        return None, str(result.code.value)
    if not result.converged:
        return None, "NOT_CONVERGED"
    return result, ""


def _observed_fit(family: FamilyId, values: Sequence[float]) -> FitSuccess:
    """Fit the observed sample, or raise :class:`GofFitError` naming the failure code."""

    result, code = _converged_fit(family, values)
    if result is None:
        raise GofFitError(family, code)
    return result


def _observed_statistics(
    family: FamilyId, ordered: Sequence[float], fitted: FitSuccess
) -> Mapping[GofStatistic, float]:
    try:
        return _model_statistics(family, ordered, fitted.parameters)
    except (ArithmeticError, ValueError) as error:
        raise GofFitError(family, "NOT_REPRESENTABLE") from error


def _replicate_statistics(
    family: FamilyId,
    size: int,
    parameters: Mapping[str, float],
    rng: Any,
) -> Mapping[GofStatistic, float] | None:
    """Draw one sample from the fitted model, refit it and return its EDF statistics.

    ``None`` when the refit fails; non-finite draws and unrepresentable
    probabilities raise ``ValueError`` or ``ArithmeticError`` for the caller to
    count.
    """

    drawn = sorted(float(value) for value in sample(family, size, rng=rng, **parameters))
    refit, _ = _converged_fit(family, drawn)
    if refit is None:
        return None
    return _model_statistics(family, drawn, refit.parameters)


def _summarize(
    requested: tuple[GofStatistic, ...],
    exceedances: Mapping[GofStatistic, int],
    replicates: int,
    successful: int,
    failed: int,
) -> RefitMonteCarloGof:
    """Turn exceedance counts into p-values, a Monte Carlo standard error and an interval."""

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


def _exponential_refit_gof(
    values: tuple[float, ...],
    requested: tuple[GofStatistic, ...],
    replicates: int,
    rng: Any,
) -> RefitMonteCarloGof:
    """The original uncensored exponential cell, with its original arithmetic."""

    observed = _empirical_statistics(values)
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
    return _summarize(requested, exceedances, replicates, successful, failed)


def _model_refit_gof(
    family: FamilyId,
    values: Sequence[float],
    fitted: FitSuccess,
    requested: tuple[GofStatistic, ...],
    replicates: int,
    rng: Any,
) -> RefitMonteCarloGof:
    """Refit Monte Carlo for a family fitted to the observed ``values``.

    Each replicate draws ``len(values)`` observations from the fitted model with
    ``rng``, refits the family and compares the refit model's statistics with
    the observed ones.  Refits that fail are counted, not retried.
    """

    observed = _observed_statistics(family, sorted(values), fitted)
    parameters = dict(fitted.parameters)
    exceedances = {statistic: 0 for statistic in requested}
    successful = 0
    failed = 0
    for _ in range(replicates):
        try:
            trial = _replicate_statistics(family, len(values), parameters, rng)
        except (ArithmeticError, ValueError):
            trial = None
        if trial is None:
            failed += 1
            continue
        successful += 1
        for statistic in requested:
            if trial[statistic] >= observed[statistic]:
                exceedances[statistic] += 1
    return _summarize(requested, exceedances, replicates, successful, failed)


def _checked_gof_arguments(
    family: object,
    statistics: object,
    replicates: object,
    rng: object,
) -> tuple[FamilyId, tuple[GofStatistic, ...], int]:
    """Validate the arguments every refit cell shares before any observation is read."""

    np = import_module("numpy")

    spec = FAMILY_REGISTRY.lookup(family)
    if (
        type(statistics) is not frozenset
        or not statistics
        or any(type(statistic) is not GofStatistic for statistic in statistics)
    ):
        raise TypeError("statistics must be a non-empty frozenset of GofStatistic values")
    if not is_integer(replicates) or cast(int, replicates) < 1:
        raise ValueError("replicates must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    return spec.id, tuple(sorted(statistics, key=str)), int(cast(int, replicates))


def _plain_values(observations: Iterable[float]) -> tuple[float, ...]:
    """Collect the observations, which must be plain real numbers."""

    values = tuple(observations)
    if not all(is_real(value) for value in values):
        raise TypeError(
            "observations must be plain real numbers: censored or typed observation objects "
            "are not supported by refit goodness-of-fit"
        )
    return values


def _model_values(family: FamilyId, values: tuple[float, ...]) -> tuple[float, ...]:
    """Validate the observations of a non-exponential family and return them as floats."""

    if not values:
        raise ValueError("observations must not be empty")
    try:
        floats = tuple(float(value) for value in values)
    except OverflowError as error:
        raise ValueError("observations must be finite") from error
    if not all(isfinite(value) for value in floats):
        raise ValueError("observations must be finite")
    if family in _LIFETIME_FAMILIES and not all(value > 0.0 for value in floats):
        raise ValueError(f"the {family.value} cell requires strictly positive observations")
    minimum = _MINIMUM_OBSERVATIONS[family]
    if len(floats) < minimum:
        raise ValueError(f"the {family.value} cell requires at least {minimum} observations")
    return floats


def refit_monte_carlo_gof(
    *,
    observations: Iterable[float],
    family: FamilyId | str,
    statistics: frozenset[GofStatistic],
    replicates: int,
    rng: object,
) -> RefitMonteCarloGof:
    """Calibrate EDF statistics of one family by refitting every Monte Carlo replicate.

    ``family`` is any of the six registered families (a
    :class:`~veridist.families.registry.FamilyId` or its string value) and
    ``observations`` are finite, exactly observed real numbers: strictly
    positive for the lifetime families (``exponential``, ``gamma``,
    ``weibull_min``, ``lognormal``), any real number for ``normal`` and
    ``gumbel_right``.  Censored observations are not supported and raise
    ``TypeError``.  At least three observations are needed for the two-parameter
    families and one for ``exponential``.

    The observed sample is fitted with the library's own maximum-likelihood fit;
    if that fit fails, :class:`GofFitError` names the failure code and no
    p-value is computed.  Each of the ``replicates`` replicates then draws as
    many observations as the sample has from the fitted model with ``rng``,
    refits the family, and evaluates the KS, AD and CvM statistics of the sample
    against the refit model's CDF.  The p-value of a statistic is
    ``(exceedances + 1) / (successful + 1)``.  A replicate whose refit fails is
    counted in ``failed_replicates``; it is neither retried nor hidden, and if
    every replicate fails a ``RuntimeError`` is raised.

    The cost is ``replicates`` times the cost of one fit to a sample of this size.
    """

    family_id, requested, count = _checked_gof_arguments(family, statistics, replicates, rng)
    values = _plain_values(observations)
    if family_id is FamilyId.EXPONENTIAL:
        return _exponential_refit_gof(values, requested, count, rng)
    floats = _model_values(family_id, values)
    return _model_refit_gof(
        family_id, floats, _observed_fit(family_id, floats), requested, count, rng
    )


def _checked_threshold(adequacy_threshold: object) -> float:
    if not is_float(adequacy_threshold) or not 0.0 <= cast(float, adequacy_threshold) <= 1.0:
        raise ValueError("adequacy_threshold must be a real probability")
    return float(cast(float, adequacy_threshold))


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

    threshold = _checked_threshold(adequacy_threshold)
    admitted: list[tuple[float, str]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise TypeError("candidates must be mappings")
        family, aic, p_value = (
            candidate.get("family"),
            candidate.get("aic"),
            candidate.get("p_value"),
        )
        if type(family) is not str or not is_float(aic) or not is_float(p_value):
            raise TypeError("candidate family must be a string and aic and p_value real floats")
        numeric_aic, numeric_p_value = float(cast(float, aic)), float(cast(float, p_value))
        if not isfinite(numeric_aic) or not isfinite(numeric_p_value):
            raise ValueError("candidate evidence must be finite")
        if numeric_p_value >= threshold:
            admitted.append((numeric_aic, family))
    if not admitted:
        return ModelSelection(SelectionCode.NONE_ADEQUATE, None)
    return ModelSelection(SelectionCode.SELECTED, min(admitted)[1])


def _admits(family: FamilyId, values: Sequence[float]) -> bool:
    """Whether every value lies inside the support the refit cell admits for ``family``."""

    return family not in _LIFETIME_FAMILIES or all(value > 0.0 for value in values)


def _assessed_families(
    families: Iterable[FamilyId | str] | None, values: Sequence[float]
) -> tuple[FamilyId, ...]:
    if families is None:
        return tuple(spec.id for spec in FAMILY_REGISTRY.list() if _admits(spec.id, values))
    if isinstance(families, str):
        raise TypeError("families must be an iterable of families, not a single string")
    chosen = tuple(FAMILY_REGISTRY.lookup(family).id for family in families)
    if not chosen:
        raise ValueError("families must not be empty")
    if len(set(chosen)) != len(chosen):
        raise ValueError("families must not repeat a family")
    return chosen


def _failed_candidate(family: FamilyId, code: str) -> FamilyCandidate:
    return FamilyCandidate(family, code, None, None, None, None, None, None, False)


def assess_families(
    *,
    observations: Iterable[float],
    families: Iterable[FamilyId | str] | None = None,
    statistic: GofStatistic = GofStatistic.AD,
    replicates: int,
    rng: object,
    adequacy_threshold: float = 0.05,
) -> FamilyAssessment:
    """Fit several families to one sample, test each by refit Monte Carlo and select one.

    Every family is fitted to the same finite, exactly observed ``observations``
    (see :func:`refit_monte_carlo_gof` for what they may be).  ``families`` are
    :class:`~veridist.families.registry.FamilyId` values or their strings, in the
    order they are assessed; by default every registered family whose support
    contains the sample (strictly positive values for the lifetime families), in
    registry order.  For each family the candidate row records the log-likelihood,
    the number of free parameters, AIC and BIC (from :func:`information_criteria`)
    and the refit Monte Carlo result for ``statistic``; the p-value of that
    statistic decides adequacy.  ``selection`` is the lowest-AIC family among
    those whose p-value is at least ``adequacy_threshold`` (see
    :func:`compare_models`), or ``NONE_ADEQUATE``.

    A family that cannot be assessed gets a row with its ``failure_code`` and is
    not eligible: the observed fit failed, the sample is outside the family's
    support (``"INVALID_SUPPORT"``) or smaller than the family's minimum size
    (``"SAMPLE_TOO_SMALL"``).

    One generator drives the whole assessment: families are processed in order and
    each eligible family's Monte Carlo draws its replicates from ``rng`` in turn, so
    the result is reproducible for a given seed, sample, family order and
    ``replicates``.  A family that cannot be assessed draws nothing.  The cost is
    the sum of one Monte Carlo run per eligible family.
    """

    np = import_module("numpy")

    if type(statistic) is not GofStatistic:
        raise TypeError("statistic must be a GofStatistic")
    if not is_integer(replicates) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    threshold = _checked_threshold(adequacy_threshold)
    count = int(replicates)
    plain = _plain_values(observations)
    if not plain:
        raise ValueError("observations must not be empty")
    try:
        values = tuple(float(value) for value in plain)
    except OverflowError as error:
        raise ValueError("observations must be finite") from error
    if not all(isfinite(value) for value in values):
        raise ValueError("observations must be finite")
    assessed = _assessed_families(families, values)

    candidates: list[FamilyCandidate] = []
    for family in assessed:
        if not _admits(family, values):
            candidates.append(_failed_candidate(family, "INVALID_SUPPORT"))
            continue
        if len(values) < _MINIMUM_OBSERVATIONS[family]:
            candidates.append(_failed_candidate(family, "SAMPLE_TOO_SMALL"))
            continue
        try:
            fitted = _observed_fit(family, values)
            gof = (
                _exponential_refit_gof(values, (statistic,), count, rng)
                if family is FamilyId.EXPONENTIAL
                else _model_refit_gof(family, values, fitted, (statistic,), count, rng)
            )
        except GofFitError as error:
            candidates.append(_failed_candidate(family, error.code))
            continue
        criteria = information_criteria(
            log_likelihood=fitted.log_likelihood,
            sample_size=len(values),
            free_parameters=len(fitted.parameters),
        )
        p_value = gof.p_values[statistic]
        candidates.append(
            FamilyCandidate(
                family,
                None,
                fitted.log_likelihood,
                criteria.free_parameters,
                criteria.aic,
                criteria.bic,
                gof,
                p_value,
                p_value >= threshold,
            )
        )
    selection = compare_models(
        candidates=tuple(
            {"family": row.family.value, "aic": row.aic, "p_value": row.p_value}
            for row in candidates
            if row.aic is not None
        ),
        adequacy_threshold=threshold,
    )
    return FamilyAssessment(tuple(candidates), selection, statistic, threshold, len(values), count)


def summarize_calibration(
    *, rejections: int, replicates: int, nominal_alpha: float
) -> CalibrationSummary:
    """Report a binomial calibration rate and its declared binomial uncertainty."""

    if not is_integer(rejections) or rejections < 0:
        raise ValueError("rejections must be a non-negative integer")
    if not is_integer(replicates) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    if rejections > replicates:
        raise ValueError("rejections cannot exceed replicates")
    if not is_float(nominal_alpha) or not 0.0 < nominal_alpha < 1.0:
        raise ValueError("nominal_alpha must be a real interior probability")
    alpha = float(nominal_alpha)
    return CalibrationSummary(
        int(rejections) / int(replicates),
        sqrt(alpha * (1.0 - alpha) / int(replicates)),
    )


__all__ = [
    "BootstrapInterval",
    "CalibrationSummary",
    "FamilyAssessment",
    "FamilyCandidate",
    "GofFitError",
    "GofStatistic",
    "InformationCriteria",
    "ModelSelection",
    "RefitMonteCarloGof",
    "SelectionCode",
    "assess_families",
    "compare_models",
    "information_criteria",
    "refit_monte_carlo_gof",
    "summarize_calibration",
]
