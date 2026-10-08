"""Distribution family definitions."""

from veridist.families.dispatch import fit
from veridist.families.exponential import (
    ExponentialFit,
    ExponentialFitFailure,
    ExponentialFitFailureCode,
    ExponentialFitProvenance,
    ExponentialFitSuccess,
    fit_exponential,
    fit_exponential_chunks,
)
from veridist.families.gamma import (
    GammaFit,
    GammaFitFailure,
    GammaFitFailureCode,
    GammaFitSuccess,
    fit_gamma,
)
from veridist.families.gumbel import (
    GumbelFit,
    GumbelFitFailure,
    GumbelFitFailureCode,
    GumbelFitSuccess,
    fit_gumbel_right,
)
from veridist.families.lognormal import (
    LognormalFit,
    LognormalFitFailure,
    LognormalFitFailureCode,
    LognormalFitSuccess,
    fit_lognormal,
)
from veridist.families.normal import (
    NormalFit,
    NormalFitFailure,
    NormalFitFailureCode,
    NormalFitSuccess,
    fit_normal,
)
from veridist.families.results import FitFailure, FitSuccess
from veridist.families.uncertainty import (
    DerivedEstimate,
    FitUncertainty,
    ParameterInterval,
    UncertaintyUnavailable,
    UncertaintyUnavailableReason,
)
from veridist.families.weibull import (
    WeibullFit,
    WeibullFitFailure,
    WeibullFitFailureCode,
    WeibullFitSuccess,
    fit_weibull,
)

__all__ = [
    "DerivedEstimate",
    "ExponentialFit",
    "ExponentialFitFailure",
    "ExponentialFitFailureCode",
    "ExponentialFitProvenance",
    "ExponentialFitSuccess",
    "FitFailure",
    "FitSuccess",
    "FitUncertainty",
    "GammaFit",
    "GammaFitFailure",
    "GammaFitFailureCode",
    "GammaFitSuccess",
    "GumbelFit",
    "GumbelFitFailure",
    "GumbelFitFailureCode",
    "GumbelFitSuccess",
    "LognormalFit",
    "LognormalFitFailure",
    "LognormalFitFailureCode",
    "LognormalFitSuccess",
    "NormalFit",
    "NormalFitFailure",
    "NormalFitFailureCode",
    "NormalFitSuccess",
    "ParameterInterval",
    "UncertaintyUnavailable",
    "UncertaintyUnavailableReason",
    "WeibullFit",
    "WeibullFitFailure",
    "WeibullFitFailureCode",
    "WeibullFitSuccess",
    "fit",
    "fit_exponential",
    "fit_exponential_chunks",
    "fit_gamma",
    "fit_gumbel_right",
    "fit_lognormal",
    "fit_normal",
    "fit_weibull",
]
