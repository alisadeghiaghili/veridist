"""Fit evidence backed by an exact state instead of the observed values.

The likelihood of the normal, lognormal and gamma families on exact data depends on
the sample only through its sufficient statistics, so the observed information, the
profile likelihoods and every derived quantity of a fit can be computed from the
statistics of a state. :class:`StateEvidence` keeps those statistics, never a data
value, and answers ``uncertainty()`` through the same models and the same code as the
in-memory evidence (:class:`veridist.families._evidence.FitEvidence`), so the report
has the same shape and the same meaning on both paths.

The models are summarised about the rounded mean (normal and lognormal) so that the
sums the models use are those of the in-memory evidence, evaluated from the exact
sums instead of from the values: ``sum (y - c)`` and ``sum (y - c)**2`` are exact
rationals of the state rounded once. For values of extreme magnitude (squares beyond
the binary64 range) the sums cannot be represented and the uncertainty is the typed
unavailable value, as it is for the in-memory evidence.

The multi-pass families (Weibull-minimum and right Gumbel) are different: their likelihood
is not a function of fixed sufficient statistics, so the evidence of a fit is the
observed information matrix at the maximum, built from the sums of the final pass
(:class:`ProfileEvidence`). It answers the covariance, the standard errors, the Wald
intervals and the Wald intervals of the derived quantities. A profile-likelihood interval
needs the likelihood at other parameter values, that is, further passes over the data, which
the evidence cannot make because it holds no source: ``uncertainty()`` refuses it with
``NotImplementedError`` and never replaces it by a Wald interval, and
:func:`veridist.scale._profile_intervals.profile_intervals` computes it from the source.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from math import fsum

from veridist.families._evidence import FitEvidence
from veridist.families._models import (
    Evaluation,
    GammaModel,
    GaussianModel,
    GumbelModel,
    Model,
    WeibullModel,
)
from veridist.families.registry import FamilyId
from veridist.families.uncertainty import (
    DerivedEstimate,
    FitUncertainty,
    ParameterInterval,
    UncertaintyUnavailable,
    UncertaintyUnavailableReason,
    _from_model,
)
from veridist.scale._one_pass import (
    ExactStatistic,
    GammaStatistics,
    LognormalStatistics,
    NormalStatistics,
)
from veridist.scale._planner import ProfileSums
from veridist.scale._profile_basis import ProfileBasis

ScaleStatistics = NormalStatistics | GammaStatistics | LognormalStatistics


def squares_about(
    count: int, total: Fraction, total_of_squares: Fraction, center: float
) -> Fraction:
    """The exact ``sum (y - center)**2`` from the exact ``sum y`` and ``sum y*y``."""

    anchor = Fraction(center)
    return total_of_squares - 2 * anchor * total + count * anchor * anchor


def _gaussian_model(
    family: FamilyId,
    names: tuple[str, str],
    count: int,
    total: ExactStatistic,
    total_of_squares: ExactStatistic,
    mean: ExactStatistic,
    constant: float,
) -> GaussianModel:
    """The uncensored location-scale model of ``count`` values summarised by exact sums."""

    center = mean.rounded
    anchor = Fraction(center)
    deviations = total.exact - count * anchor
    squares = squares_about(count, total.exact, total_of_squares.exact, center)
    return GaussianModel.from_statistics(
        family, names, count, center, float(deviations), float(squares), constant
    )


def model_of(statistics: ScaleStatistics) -> Model:
    """The likelihood model of a non-empty state's statistics.

    ``OverflowError`` if a derived sum is beyond the float range, which the callers
    report as a typed failure or as unavailable uncertainty.
    """

    if isinstance(statistics, NormalStatistics):
        assert statistics.mean is not None
        return _gaussian_model(
            FamilyId.NORMAL,
            ("mu", "sigma"),
            statistics.count,
            statistics.sum_x,
            statistics.sum_x_squared,
            statistics.mean,
            0.0,
        )
    if isinstance(statistics, LognormalStatistics):
        assert statistics.mean_log_x is not None
        return _gaussian_model(
            FamilyId.LOGNORMAL,
            ("mu_log", "sigma_log"),
            statistics.count,
            statistics.sum_log_x,
            statistics.sum_log_x_squared,
            statistics.mean_log_x,
            -statistics.sum_log_x.rounded,
        )
    return GammaModel.from_statistics(
        statistics.count, statistics.sum_x.rounded, statistics.sum_log_x.rounded
    )


class StateEvidence(FitEvidence):
    """The sufficient statistics of a fit, in place of its observed values.

    ``exact`` and ``censored`` stay empty: nothing here holds an observation. The
    object is created by the scale fit and is not part of any public interface.
    """

    __slots__ = ("statistics",)

    def __init__(self, statistics: ScaleStatistics) -> None:
        super().__init__()
        self.statistics = statistics

    def uncertainty(
        self, family: FamilyId, parameters: Mapping[str, float]
    ) -> FitUncertainty | UncertaintyUnavailable:
        """Return the uncertainty at ``parameters``, computed once per parameter vector."""

        key = (family, *parameters.values())
        cached = self._cache.get(key)
        if cached is None:
            cached = self._compute(family, parameters)
            self._cache[key] = cached
        return cached

    def _compute(
        self, family: FamilyId, parameters: Mapping[str, float]
    ) -> FitUncertainty | UncertaintyUnavailable:
        try:
            model = model_of(self.statistics)
        except (ArithmeticError, ValueError):
            return UncertaintyUnavailable(
                UncertaintyUnavailableReason.NOT_COMPUTABLE,
                "the sufficient statistics could not be prepared for the likelihood",
            )
        theta = tuple(float(parameters[name]) for name in model.names)
        return _from_model(family, model, theta)


# ----------------------------------------------------------- multi-pass information


def _moments(sums: ProfileSums) -> tuple[float, float]:
    """The weighted mean ``S1 / S0`` and the weighted variance of the centred observations."""

    first = sums.s1 / sums.s0
    return first, max(sums.s2 / sums.s0 - first * first, 0.0)


def weibull_evaluation(
    count: int,
    shape: float,
    scale: float,
    log_ratio: float,
    sums: ProfileSums,
    likelihood: float,
) -> Evaluation:
    """The log-likelihood, gradient and Hessian of a Weibull fit at its maximum, from sums.

    With ``w = log x - c`` centred as the fit centres it and the sums ``S1``, ``S2`` of the
    final pass, ``d = w - u`` with ``u = log(S0 / n) / k`` is ``log x - log(scale)`` up to
    the centring constant, and ``sum exp(k d) = n`` at the fitted scale. The Hessian is the
    one of :class:`~veridist.families._models.WeibullModel` with that sum and with
    ``sum exp(k d) d = n (S1/S0 - u)`` and ``sum exp(k d) d**2 = n (var + (S1/S0 - u)**2)``.
    The gradient is zero at the maximum and is stored as such.
    """

    events = float(count)
    first, variance = _moments(sums)
    offset = log_ratio / shape
    total_first = events * (first - offset)
    total_second = events * fsum((variance, (first - offset) ** 2))
    mixed = shape * total_first / scale
    return Evaluation(
        likelihood,
        (0.0, 0.0),
        (
            (-events / (shape * shape) - total_second, mixed),
            (mixed, -events * shape * shape / (scale * scale)),
        ),
    )


def gumbel_evaluation(
    count: int,
    scale: float,
    rate: float,
    log_ratio: float,
    mean_w: float,
    sums: ProfileSums,
    likelihood: float,
) -> Evaluation:
    """The log-likelihood, gradient and Hessian of a right-Gumbel fit at its maximum.

    ``rate`` is the reciprocal of the scale of the standardised data and ``w`` the centred
    observations of the fit, so that ``z = log(S0 / n) - rate * w`` is the standardised
    deviation of an observation. With ``t = exp(-z)`` the sums ``sum t = n``,
    ``sum z t = n (log(S0/n) - rate S1/S0)`` and ``sum z**2 t`` and ``sum z`` follow from
    ``S1``, ``S2`` and ``mean(w)``; they enter the Hessian through the same expressions as
    :class:`~veridist.families._models.GumbelModel`. The gradient is zero at the maximum
    and is stored as such.
    """

    events = float(count)
    first, variance = _moments(sums)
    tail_first = log_ratio - rate * first
    tail_second = fsum((rate * rate * variance, (rate * first - log_ratio) ** 2))
    cross = rate * (mean_w - first)
    squared = scale * scale
    mixed = -events * tail_first / squared
    return Evaluation(
        likelihood,
        (0.0, 0.0),
        (
            (-events / squared, mixed),
            (mixed, events * fsum((1.0, 2.0 * cross, -tail_second)) / squared),
        ),
    )


class _FittedPoint:
    """Mixin: a model that has the information at one parameter vector and nowhere else."""

    _theta: tuple[float, ...]
    _evaluation: Evaluation

    def evaluate(self, theta: tuple[float, ...], *, precise: bool = True) -> Evaluation:
        if theta != self._theta:
            raise ValueError("the information is available only at the fitted parameters")
        return self._evaluation


class WeibullSumsModel(_FittedPoint, WeibullModel):
    """The Weibull model of a multi-pass fit: the information at the fitted parameters."""

    def __init__(self, theta: tuple[float, ...], evaluation: Evaluation) -> None:
        self._theta = theta
        self._evaluation = evaluation


class GumbelSumsModel(_FittedPoint, GumbelModel):
    """The right-Gumbel model of a multi-pass fit: the information at the fitted parameters."""

    def __init__(self, theta: tuple[float, ...], evaluation: Evaluation) -> None:
        self._theta = theta
        self._evaluation = evaluation


@dataclass(frozen=True, slots=True)
class WaldUncertainty(FitUncertainty):
    """The uncertainty of a fit that kept the observed information and no data.

    Everything :class:`~veridist.families.uncertainty.FitUncertainty` reports from the
    covariance and the model's closed forms is available; a profile-likelihood interval of a
    parameter or of a derived quantity would need further passes over the data, which this
    object cannot make, and raises ``NotImplementedError``.
    """

    def interval_details(
        self, level: float = 0.95, method: str = "wald"
    ) -> Mapping[str, ParameterInterval]:
        _refuse_profile(method)
        return FitUncertainty.interval_details(self, level, method)

    def _derived(
        self, kind: str, argument: float, level: float, method: str
    ) -> DerivedEstimate:
        _refuse_profile(method)
        return FitUncertainty._derived(self, kind, argument, level, method)


def _refuse_profile(method: object) -> None:
    if method == "profile":
        raise NotImplementedError(
            "profile likelihood intervals of a multi-pass scale fit need further passes over "
            "the source: compute them with profile_intervals(source, result) of "
            'veridist.scale._profile_intervals, or use method="wald"'
        )


class ProfileEvidence(FitEvidence):
    """The observed information of a multi-pass fit, in place of its observed values.

    ``exact`` and ``censored`` stay empty: nothing here holds an observation. The object is
    created by the scale fit and is not part of any public interface.
    """

    __slots__ = ("_build", "_family", "_theta", "basis")

    def __init__(
        self,
        family: FamilyId,
        theta: tuple[float, ...],
        build: Callable[[], Evaluation],
        basis: ProfileBasis | None = None,
    ) -> None:
        super().__init__()
        self._theta = theta
        self._build = build
        self._family = family
        self.basis = basis

    def uncertainty(
        self, family: FamilyId, parameters: Mapping[str, float]
    ) -> FitUncertainty | UncertaintyUnavailable:
        """Return the Wald uncertainty at ``parameters``, computed once per parameter vector."""

        key = (family, *parameters.values())
        cached = self._cache.get(key)
        if cached is None:
            cached = self._compute(family, parameters)
            self._cache[key] = cached
        return cached

    def _compute(
        self, family: FamilyId, parameters: Mapping[str, float]
    ) -> FitUncertainty | UncertaintyUnavailable:
        try:
            evaluation = self._build()
        except (ArithmeticError, ValueError):
            return UncertaintyUnavailable(
                UncertaintyUnavailableReason.NOT_COMPUTABLE,
                "the information sums could not be prepared for the likelihood",
            )
        model: Model = (
            WeibullSumsModel(self._theta, evaluation)
            if self._family is FamilyId.WEIBULL_MIN
            else GumbelSumsModel(self._theta, evaluation)
        )
        theta = tuple(float(parameters[name]) for name in model.names)
        result = _from_model(family, model, theta)
        if isinstance(result, UncertaintyUnavailable):
            return result
        return WaldUncertainty(
            result.family,
            result.parameter_names,
            result.estimates,
            result.covariance,
            result._model,
        )


__all__ = [
    "GumbelSumsModel",
    "ProfileEvidence",
    "ScaleStatistics",
    "StateEvidence",
    "WaldUncertainty",
    "WeibullSumsModel",
    "gumbel_evaluation",
    "model_of",
    "squares_about",
    "weibull_evaluation",
]
