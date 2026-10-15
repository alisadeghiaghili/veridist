"""End-to-end fitting of the one-pass families from a source of exact observations.

``fit(source, family)`` folds the partitions of ``source`` into the family's exact
state (:mod:`veridist.scale._one_pass`), computes the maximum-likelihood estimate from
the exact statistics of that state (:mod:`veridist.scale._estimates`) and returns the
result of the in-memory fit of the same family together with a scale provenance
record. The estimate does not depend on the batch size, the partition layout, the
file split or the order of the rows: the state, its canonical bytes and its digest are
identical for every such choice within one numeric environment, and so are the result
bits.

Supported families are ``exponential``, ``normal``, ``gamma`` and ``lognormal`` on
exact (uncensored) data; any other family is refused with ``ValueError`` before any
read. Argument errors (``TypeError`` or ``ValueError``) and every refusal of the value
policy (:class:`~veridist.scale._errors.ScaleSourceError`) or of a state
(:class:`~veridist.scale._errors.ScaleStateError`) are raised, never turned into a
fit failure, with one exception that mirrors the in-memory fit: for ``gamma`` and
``lognormal`` a value that is not strictly positive is the failure
``INVALID_SUPPORT``, reported with the number of rows the source holds.

Uncertainty. A successful normal, gamma or lognormal result carries
:class:`~veridist.scale._evidence.StateEvidence`, so ``uncertainty()`` is computed
from the sufficient statistics and needs no data; the exponential result computes it
from its own count and total time, as the in-memory result does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final

from veridist.engine.provenance import (
    SCALE_PROVENANCE_SCHEMA_VERSION,
    ScaleEnvironmentProvenance,
    ScalePartitionProvenance,
    ScaleProvenance,
    ScaleStateProvenance,
)
from veridist.families.gamma import GammaFitFailure, GammaFitFailureCode
from veridist.families.lognormal import LognormalFitFailure, LognormalFitFailureCode
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.families.results import FitFailure, FitSuccess
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._environment import numeric_environment_digest, numpy_version
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._estimates import (
    fit_exponential_state,
    fit_gamma_state,
    fit_lognormal_state,
    fit_normal_state,
)
from veridist.scale._one_pass import ExponentialState, GammaState, LognormalState, NormalState
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    NumpySource,
    Partition,
    Source,
    check_max_rows,
    fold_partition,
)
from veridist.scale._state import MergeableState

SOURCE_KINDS: Final[dict[type, str]] = {
    NumpySource: "numpy",
    ArrowSource: "arrow",
    ParquetSource: "parquet",
}
"""The provenance name of each bundled source; any other source is ``"custom"``."""

CUSTOM_SOURCE_KIND: Final = "custom"


@dataclass(frozen=True, slots=True)
class ScaleFitResult:
    """A fit from a scale source: the in-memory result type and the scale provenance.

    ``fit`` is the very result class the in-memory fit of the family returns (a
    success or a failure). ``scale`` records where the numbers came from: the source
    kind, the partitions with their fingerprints, the row count, the batch cap, the
    digest of the canonical state and the numeric environment digest.
    """

    fit: FitSuccess | FitFailure
    scale: ScaleProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.fit, FitSuccess | FitFailure):
            raise TypeError("fit must be a family fit success or failure")
        if type(self.scale) is not ScaleProvenance:
            raise TypeError("scale must be a ScaleProvenance")


@dataclass(frozen=True, slots=True)
class _Plan:
    """What one supported family needs: its state, its estimator and its support failure."""

    state_type: type[Any]
    estimate: Callable[[Any], FitSuccess | FitFailure]
    invalid_support: Callable[[int], FitFailure] | None


def _gamma_invalid_support(rows: int) -> FitFailure:
    return GammaFitFailure(GammaFitFailureCode.INVALID_SUPPORT, rows, rows, 0)


def _lognormal_invalid_support(rows: int) -> FitFailure:
    return LognormalFitFailure(LognormalFitFailureCode.INVALID_SUPPORT, rows, rows, 0)


_PLANS: Final[dict[FamilyId, _Plan]] = {
    FamilyId.EXPONENTIAL: _Plan(ExponentialState, fit_exponential_state, None),
    FamilyId.NORMAL: _Plan(NormalState, fit_normal_state, None),
    FamilyId.GAMMA: _Plan(GammaState, fit_gamma_state, _gamma_invalid_support),
    FamilyId.LOGNORMAL: _Plan(LognormalState, fit_lognormal_state, _lognormal_invalid_support),
}


def _fold_partitions(
    source: Source,
    partitions: tuple[Partition, ...],
    state_type: type[MergeableState[Any]],
    max_rows: int,
) -> MergeableState[Any]:
    """The state of the given partitions, merged in the order given."""

    state = state_type.empty()
    for partition in partitions:
        state = state.merge(fold_partition(source, partition, state_type, max_rows=max_rows))
    return state


def _count_rows(source: Source, partitions: tuple[Partition, ...], max_rows: int) -> int:
    """The number of rows the partitions hold, by reading them under the value policy."""

    return sum(
        int(batch.shape[0])
        for partition in partitions
        for batch in source.batches(partition, max_rows=max_rows)
    )


def _partition_record(partition: Partition) -> ScalePartitionProvenance:
    return ScalePartitionProvenance(
        partition.id,
        partition.ordinal,
        partition.rows,
        partition.fingerprint,
        partition.fingerprint_level.value,
    )


def _provenance(
    source: Source,
    partitions: tuple[Partition, ...],
    row_count: int,
    state: MergeableState[Any] | None,
    max_rows: int,
) -> ScaleProvenance:
    state_record = None
    if state is not None:
        state_record = ScaleStateProvenance(
            state.STATE_TAG, state.SCHEMA_VERSION, sha256(state.to_bytes()).hexdigest()
        )
    return ScaleProvenance(
        SCALE_PROVENANCE_SCHEMA_VERSION,
        SOURCE_KINDS.get(type(source), CUSTOM_SOURCE_KIND),
        tuple(_partition_record(partition) for partition in partitions),
        len(partitions),
        row_count,
        max_rows,
        state_record,
        ScaleEnvironmentProvenance(numpy_version(), numeric_environment_digest()),
    )


def fit(
    source: Source, family: FamilyId | str, *, max_rows: int = DEFAULT_MAX_ROWS
) -> ScaleFitResult:
    """Fit ``family`` to the exact observations of ``source`` in one pass over the data.

    ``family`` is a :class:`~veridist.families.registry.FamilyId` or its string value
    (``TypeError`` for anything else, ``ValueError`` for an unknown name and for a
    family this fit does not support). ``max_rows`` bounds the rows of one batch.
    ``source`` is read once, partition by partition, in the source's canonical order.
    See the module documentation for the result and the refusals.
    """

    spec = FAMILY_REGISTRY.lookup(family)
    plan = _PLANS.get(spec.id)
    if plan is None:
        raise ValueError("family is not supported by the one-pass scale fit")
    bound = check_max_rows(max_rows)
    if not isinstance(source, Source):
        raise TypeError("source must provide partitions() and batches()")
    partitions = source.partitions()
    try:
        state = _fold_partitions(source, partitions, plan.state_type, bound)
    except ScaleStateError as error:
        if error.code is not ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT or (
            plan.invalid_support is None
        ):
            raise
        rows = _count_rows(source, partitions, bound)
        return ScaleFitResult(
            plan.invalid_support(rows), _provenance(source, partitions, rows, None, bound)
        )
    return ScaleFitResult(
        plan.estimate(state), _provenance(source, partitions, state.count, state, bound)
    )


__all__ = ["CUSTOM_SOURCE_KIND", "SOURCE_KINDS", "ScaleFitResult", "fit"]
