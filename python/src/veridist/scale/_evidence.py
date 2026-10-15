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
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction

from veridist.families._evidence import FitEvidence
from veridist.families._models import GammaModel, GaussianModel, Model
from veridist.families.registry import FamilyId
from veridist.families.uncertainty import (
    FitUncertainty,
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


__all__ = ["ScaleStatistics", "StateEvidence", "model_of", "squares_about"]
