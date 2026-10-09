"""Veridist's public API: fits, distribution operations, observations and CSV entry points.

The top level is the documented user API. Engine, provenance and delivery
internals stay in their submodules (``veridist.engine`` and friends). The
package imports only the standard library; ``numpy`` is loaded on first use of
an array operation.
"""

__version__ = "2.1.0"

from veridist.adapters.csv_lifetimes import CsvLifetimeLimits, CsvLifetimeSchema
from veridist.domain.arrays import lifetimes_from_arrays, values_from_arrays
from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime
from veridist.domain.values import ExactValue, RealObservation, RightCensoredValue
from veridist.engine.data_source import DataSourceMetadata, Replayability
from veridist.engine.errors import CapabilityError, EngineContractError, VeridistError
from veridist.engine.provenance import PublicSourceId
from veridist.engine.streaming import IterableDataSource, StreamSource, StreamSourceError
from veridist.execution import (
    ExponentialSourceFitResult,
    create_checkpointed_csv_store,
    fit_exponential_checkpointed_chunks,
    fit_exponential_checkpointed_csv,
    fit_exponential_csv,
)
from veridist.families.dispatch import fit
from veridist.families.exponential import fit_exponential
from veridist.families.gamma import fit_gamma
from veridist.families.gumbel import fit_gumbel_right
from veridist.families.lognormal import fit_lognormal
from veridist.families.normal import fit_normal
from veridist.families.registry import FamilyId
from veridist.families.results import FitFailure, FitSuccess
from veridist.families.weibull import fit_weibull
from veridist.statistics.distributions import cdf, logpdf, ppf, sample, sf
from veridist.statistics.lifetime_log_likelihood import (
    reduce_lifetime_log_likelihood_chunks,
    reduce_value_log_likelihood_chunks,
)
from veridist.statistics.log_likelihood import reduce_log_likelihood_chunks

__all__ = [
    "__version__",
    "CapabilityError",
    "CsvLifetimeLimits",
    "CsvLifetimeSchema",
    "DataSourceMetadata",
    "EngineContractError",
    "ExactLifetime",
    "ExactValue",
    "ExponentialSourceFitResult",
    "FamilyId",
    "FitFailure",
    "FitSuccess",
    "IterableDataSource",
    "LifetimeObservation",
    "PublicSourceId",
    "RealObservation",
    "Replayability",
    "RightCensoredLifetime",
    "RightCensoredValue",
    "StreamSource",
    "StreamSourceError",
    "VeridistError",
    "cdf",
    "create_checkpointed_csv_store",
    "fit",
    "fit_exponential",
    "fit_exponential_checkpointed_chunks",
    "fit_exponential_checkpointed_csv",
    "fit_exponential_csv",
    "fit_gamma",
    "fit_gumbel_right",
    "fit_lognormal",
    "fit_normal",
    "fit_weibull",
    "lifetimes_from_arrays",
    "logpdf",
    "ppf",
    "reduce_lifetime_log_likelihood_chunks",
    "reduce_log_likelihood_chunks",
    "reduce_value_log_likelihood_chunks",
    "sample",
    "sf",
    "values_from_arrays",
]
