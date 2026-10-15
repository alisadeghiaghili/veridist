"""End-to-end fitting from a source of exact observations.

``fit(source, family)`` returns the result of the in-memory fit of the family together
with a scale provenance record. The estimate does not depend on the batch size, the
partition layout, the file split or the order of the rows: the state, its canonical
bytes and its digest are identical for every such choice within one numeric
environment, and so are the result bits.

One-pass families: ``exponential``, ``normal``, ``gamma`` and ``lognormal`` on exact
(uncensored) data. The partitions of ``source`` are folded into the family's exact state
(:mod:`veridist.scale._one_pass`) and the maximum-likelihood estimate is computed from the
exact statistics of that state (:mod:`veridist.scale._estimates`).

Multi-pass families: ``weibull_min`` and ``gumbel_right`` on exact data. The source is read
several times, as planned by :mod:`veridist.scale._planner` and carried out by
:mod:`veridist.scale._multi_pass`, within an explicit ``pass_budget`` (default
``PASS_BUDGET_DEFAULT``). A budget below the least the plan needs is refused before any
read with ``PASS_BUDGET_INSUFFICIENT``; a run that would need a pass beyond the budget is
refused before that pass is read with ``PASS_BUDGET_EXCEEDED`` (both are
:class:`~veridist.engine.pass_budget.PassBudgetError`), and no estimate is returned in
either case. The pass count and the record of the passes are in the provenance
(``scale.passes``). The first pass is the state of the provenance record.

Spool. A multi-pass family reads its source several times, which a source that can be read
only once (an Arrow reader or stream) does not allow and which is wasteful when a read is
expensive. ``spool=SpoolPolicy(...)`` (:mod:`veridist.scale._spool`) is the explicit,
opt-in remedy: the first pass writes the values it validated to a private local directory,
the source is not read again, every later pass reads that copy and verifies it against its
digests, and the directory is deleted when the fit ends, however it ends. Without a spool,
a multi-pass family refuses a source that can be read only once before it reads anything
(``SPOOL_REQUIRED``); it does not read the first pass and fail in the second. The result is
bit for bit the one the same fit without a spool gives (and, for a source that can be
read once, the one a re-readable copy of the same data gives). The provenance records the
spool, its row count and the digest of each partition, and no path (``scale.spool``). A
one-pass family reads its source once and ignores ``spool``: no directory or file is
created for it.

Argument errors (``TypeError`` or ``ValueError``, raised before any read) and every
refusal of the value policy (:class:`~veridist.scale._errors.ScaleSourceError`) or of a
state (:class:`~veridist.scale._errors.ScaleStateError`) are raised, never turned into a
fit failure, with one exception that mirrors the in-memory fit: for ``gamma``,
``lognormal`` and ``weibull_min`` a value that is not strictly positive is the failure
``INVALID_SUPPORT``, reported with the number of rows the source holds.

Uncertainty. A successful normal, gamma or lognormal result carries
:class:`~veridist.scale._evidence.StateEvidence`, so ``uncertainty()`` is computed
from the sufficient statistics and needs no data; the exponential result computes it
from its own count and total time, as the in-memory result does. A Weibull or Gumbel
result carries the observed information of its final pass
(:class:`~veridist.scale._evidence.ProfileEvidence`): the covariance, standard errors and
Wald intervals; a profile-likelihood interval is refused (it needs further passes).
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
    ScalePassProvenance,
    ScaleProvenance,
    ScaleSpoolProvenance,
    ScaleStateProvenance,
)
from veridist.families.gamma import GammaFitFailure, GammaFitFailureCode
from veridist.families.lognormal import LognormalFitFailure, LognormalFitFailureCode
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.families.results import FitFailure, FitSuccess
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._environment import numeric_environment_digest, numpy_version
from veridist.scale._errors import (
    ScaleSourceError,
    ScaleSourceErrorCode,
    ScaleStateError,
    ScaleStateErrorCode,
)
from veridist.scale._estimates import (
    fit_exponential_state,
    fit_gamma_state,
    fit_lognormal_state,
    fit_normal_state,
)
from veridist.scale._multi_pass import MultiPassResult, fit_gumbel_right, fit_weibull_min
from veridist.scale._one_pass import ExponentialState, GammaState, LognormalState, NormalState
from veridist.scale._pilot import PilotState
from veridist.scale._planner import (
    PASS_BUDGET_DEFAULT,
    PassLedger,
    check_pass_budget,
    plan_passes,
)
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    NumpySource,
    Partition,
    Source,
    check_max_rows,
    fold_partition,
    is_single_use,
)
from veridist.scale._spool import HINT_REQUIRED, Spool, SpoolPolicy
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


_MULTI_PASS: Final[dict[FamilyId, Callable[[Source, PassLedger, int], MultiPassResult]]] = {
    FamilyId.WEIBULL_MIN: fit_weibull_min,
    FamilyId.GUMBEL_RIGHT: fit_gumbel_right,
}
"""The multi-pass families: each reads the source as often as its plan needs."""


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
    state: MergeableState[Any] | PilotState | None,
    max_rows: int,
    passes: ScalePassProvenance | None = None,
    spool: ScaleSpoolProvenance | None = None,
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
        passes,
        spool,
    )


def _fit_multi_pass(
    source: Source,
    family: FamilyId,
    max_rows: int,
    pass_budget: int,
    spool: SpoolPolicy | None,
) -> ScaleFitResult:
    """A multi-pass fit; the pass budget has been validated and the plan accepted it.

    ``ScaleSourceError`` with ``SPOOL_REQUIRED``, before anything is read, for a source
    that can be read only once when no spool is requested.
    """

    bounds = plan_passes(pass_budget)
    partitions = source.partitions()
    ledger = PassLedger(partitions, pass_budget, bounds)
    run = _MULTI_PASS[family]
    if spool is None:
        if is_single_use(source):
            raise ScaleSourceError(ScaleSourceErrorCode.SPOOL_REQUIRED, HINT_REQUIRED)
        outcome = run(source, ledger, max_rows)
        record = None
    else:
        with Spool(spool) as local:
            outcome = run(local.source(source), ledger, max_rows)
            record = local.provenance()
    return ScaleFitResult(
        outcome.fit,
        _provenance(
            source,
            partitions,
            outcome.row_count,
            outcome.state,
            max_rows,
            outcome.passes,
            record,
        ),
    )


def fit(
    source: Source,
    family: FamilyId | str,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    pass_budget: int = PASS_BUDGET_DEFAULT,
    spool: SpoolPolicy | None = None,
) -> ScaleFitResult:
    """Fit ``family`` to the exact observations of ``source``.

    ``family`` is a :class:`~veridist.families.registry.FamilyId` or its string value
    (``TypeError`` for anything else, ``ValueError`` for an unknown name).
    ``max_rows`` bounds the rows of one batch and
    ``pass_budget`` the number of times the source may be read. A one-pass family reads
    ``source`` once, partition by partition, in the source's canonical order; a
    multi-pass family reads it as many times as its plan needs, at most ``pass_budget``.
    ``spool`` is ``None`` or a :class:`~veridist.scale._spool.SpoolPolicy`: the explicit
    request for a local spool, which a multi-pass family needs for a source that can be
    read only once and a one-pass family ignores. See the module documentation for the
    result and the refusals.
    """

    spec = FAMILY_REGISTRY.lookup(family)
    bound = check_max_rows(max_rows)
    budget = check_pass_budget(pass_budget)
    if not isinstance(source, Source):
        raise TypeError("source must provide partitions() and batches()")
    if spool is not None and not isinstance(spool, SpoolPolicy):
        raise TypeError("spool must be a SpoolPolicy or None")
    plan = _PLANS.get(spec.id)
    if plan is None:
        return _fit_multi_pass(source, spec.id, bound, budget, spool)
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
