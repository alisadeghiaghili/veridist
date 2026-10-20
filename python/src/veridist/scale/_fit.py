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

Checkpoint. ``checkpoint=`` (a path, or a ``SQLitePartitionStore`` of
:mod:`veridist.scale._partition_store`) keeps the exact state of every partition, and the
merged result of every completed pass, in a local store, so that an interrupted fit continues
with ``resume=True`` and reads only the partitions that have no committed state; the result,
its state digest and the provenance but the checkpoint record are bit for bit those of an
uninterrupted fit. ``incremental=True`` (with ``resume=True``, one-pass families only) fits a
source that has gained partitions by reading only the new ones. ``recompute_changed=True``
reads again a partition whose fingerprint changed and replaces its state,
``allow_removed=True`` (incremental) excludes a partition that is gone. Without them a changed,
missing or new partition is a typed refusal before any read. A source whose partitions have no
fingerprint (Arrow streams, Polars plans, DuckDB relations) cannot be checkpointed. The record
is ``scale.checkpoint``; see :mod:`veridist.scale._checkpoint`.

Uncertainty. A successful normal, gamma or lognormal result carries
:class:`~veridist.scale._evidence.StateEvidence`, so ``uncertainty()`` is computed
from the sufficient statistics and needs no data; the exponential result computes it
from its own count and total time, as the in-memory result does. A Weibull or Gumbel
result carries the observed information of its final pass
(:class:`~veridist.scale._evidence.ProfileEvidence`): the covariance, standard errors and
Wald intervals; a profile-likelihood interval is refused (it needs further passes).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Final

from veridist.engine.provenance import (
    SCALE_PROVENANCE_SCHEMA_VERSION,
    ScaleCheckpointProvenance,
    ScaleEnvironmentProvenance,
    ScaleExecutionProvenance,
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
from veridist.scale._checkpoint import (
    CheckpointRequest,
    CheckpointRun,
    job_identity,
    refuse_incremental,
    require_resumable,
    resolve_checkpoint,
)
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
from veridist.scale._executors import Executor, PassEngine, resolve_executor
from veridist.scale._multi_pass import (
    PROFILE_SCHEMA_VERSION,
    PROFILE_TAG,
    MultiPassResult,
    fit_gumbel_right,
    fit_weibull_min,
)
from veridist.scale._one_pass import ExponentialState, GammaState, LognormalState, NormalState
from veridist.scale._partition_store import SQLitePartitionStore
from veridist.scale._pilot import GumbelPilotState, PilotState, WeibullPilotState
from veridist.scale._planner import (
    PASS_BUDGET_DEFAULT,
    PassLedger,
    check_pass_budget,
    plan_identity,
    plan_passes,
)
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    NumpySource,
    Partition,
    Source,
    check_max_rows,
    is_single_use,
)
from veridist.scale._spool import HINT_REQUIRED, Spool, SpoolPolicy
from veridist.scale._state import MergeableState
from veridist.scale._tasks import FoldJob

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


_MULTI_PASS: Final[dict[FamilyId, Callable[[Source, PassLedger, PassEngine], MultiPassResult]]] = {
    FamilyId.WEIBULL_MIN: fit_weibull_min,
    FamilyId.GUMBEL_RIGHT: fit_gumbel_right,
}
"""The multi-pass families: each reads the source as often as its plan needs."""

_MULTI_PASS_PILOTS: Final[dict[FamilyId, type[PilotState]]] = {
    FamilyId.WEIBULL_MIN: WeibullPilotState,
    FamilyId.GUMBEL_RIGHT: GumbelPilotState,
}
"""The first-pass state of each multi-pass family."""


def _state_schema(*schemas: tuple[str, int]) -> str:
    """The tags and schema versions of the states a family keeps, as one component."""

    return ",".join(f"{tag}:{version}" for tag, version in schemas)


def _open_checkpoint(
    request: CheckpointRequest,
    family: FamilyId,
    source: Source,
    max_rows: int,
    partitions: tuple[Partition, ...],
    state_schema: str,
    *,
    multi_pass: bool,
) -> CheckpointRun:
    """Start or resume the job of this fit in the store (the partitions are resumable)."""

    identity = job_identity(
        family=family.value,
        column=getattr(source, "column", None),
        source_kind=SOURCE_KINDS.get(type(source), CUSTOM_SOURCE_KIND),
        max_rows=max_rows,
        partitions=partitions,
        state_schema=state_schema,
        plan=plan_identity() if multi_pass else None,
    )
    return CheckpointRun.open(request, identity, partitions, multi_pass=multi_pass)


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


def build_provenance(
    source: Source,
    partitions: tuple[Partition, ...],
    row_count: int,
    state: ScaleStateProvenance | None,
    max_rows: int,
    passes: ScalePassProvenance | None = None,
    spool: ScaleSpoolProvenance | None = None,
    execution: ScaleExecutionProvenance | None = None,
    checkpoint: ScaleCheckpointProvenance | None = None,
) -> ScaleProvenance:
    """The scale provenance record of a computation over ``source``.

    ``state`` is the record of the canonical state the numbers were derived from, if any. The
    fit and the inference computed from a fit build their records here.
    """

    return ScaleProvenance(
        SCALE_PROVENANCE_SCHEMA_VERSION,
        SOURCE_KINDS.get(type(source), CUSTOM_SOURCE_KIND),
        tuple(_partition_record(partition) for partition in partitions),
        len(partitions),
        row_count,
        max_rows,
        state,
        ScaleEnvironmentProvenance(numpy_version(), numeric_environment_digest()),
        passes,
        spool,
        execution,
        checkpoint,
    )


def _provenance(
    source: Source,
    partitions: tuple[Partition, ...],
    row_count: int,
    state: MergeableState[Any] | PilotState | None,
    max_rows: int,
    passes: ScalePassProvenance | None = None,
    spool: ScaleSpoolProvenance | None = None,
    execution: ScaleExecutionProvenance | None = None,
    checkpoint: ScaleCheckpointProvenance | None = None,
) -> ScaleProvenance:
    state_record = None
    if state is not None:
        state_record = ScaleStateProvenance(
            state.STATE_TAG, state.SCHEMA_VERSION, sha256(state.to_bytes()).hexdigest()
        )
    return build_provenance(
        source, partitions, row_count, state_record, max_rows, passes, spool, execution, checkpoint
    )


def _fit_multi_pass(
    source: Source,
    family: FamilyId,
    max_rows: int,
    pass_budget: int,
    spool: SpoolPolicy | None,
    executor: Executor | None,
    request: CheckpointRequest | None,
) -> ScaleFitResult:
    """A multi-pass fit; the pass budget has been validated and the plan accepted it.

    ``ScaleSourceError`` with ``SPOOL_REQUIRED``, before anything is read, for a source
    that can be read only once when no spool is requested. With a checkpoint request, a
    source that cannot be fingerprinted is refused first, and the store is not touched
    before the other refusals have been made.
    """

    bounds = plan_passes(pass_budget)
    partitions = source.partitions()
    checkpoint = None
    if request is not None:
        require_resumable(partitions)
    ledger = PassLedger(partitions, pass_budget, bounds)
    run = _MULTI_PASS[family]
    if spool is None and is_single_use(source):
        raise ScaleSourceError(ScaleSourceErrorCode.SPOOL_REQUIRED, HINT_REQUIRED)
    if request is not None:
        schema = _state_schema(
            (_MULTI_PASS_PILOTS[family].STATE_TAG, _MULTI_PASS_PILOTS[family].SCHEMA_VERSION),
            (PROFILE_TAG, PROFILE_SCHEMA_VERSION),
        )
        checkpoint = _open_checkpoint(
            request, family, source, max_rows, partitions, schema, multi_pass=True
        )
    engine = PassEngine(executor, max_rows, checkpoint)
    if spool is None:
        with engine:
            outcome = run(source, ledger, engine)
        record = None
    else:
        # The engine is left first: no worker holds a spool file when the directory goes.
        with Spool(spool) as local, engine:
            outcome = run(local.source(source), ledger, engine)
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
            engine.provenance(),
            engine.checkpoint_provenance(),
        ),
    )


def fit(
    source: Source,
    family: FamilyId | str,
    *,
    max_rows: int = DEFAULT_MAX_ROWS,
    pass_budget: int = PASS_BUDGET_DEFAULT,
    spool: SpoolPolicy | None = None,
    executor: Executor | str | None = None,
    checkpoint: SQLitePartitionStore | str | os.PathLike[str] | None = None,
    resume: bool = False,
    incremental: bool = False,
    recompute_changed: bool = False,
    allow_removed: bool = False,
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
    read once and a one-pass family ignores. ``executor`` is ``None`` (the partitions are
    read in turn), ``"serial"``, a :class:`~veridist.scale._executors.ThreadExecutor` or a
    :class:`~veridist.scale._executors.ProcessExecutor`: the partitions of every pass are then
    read by that executor, with no change to the result, the state or its digest; see
    :mod:`veridist.scale._executors`. ``checkpoint`` is ``None``, a path or a
    :class:`~veridist.scale._partition_store.SQLitePartitionStore`; ``resume``,
    ``incremental``, ``recompute_changed`` and ``allow_removed`` say how a store that holds
    a job is used (``TypeError`` for a flag that is not a ``bool``, ``ValueError`` for a flag
    without a checkpoint, ``incremental`` without ``resume`` and ``allow_removed`` without
    ``incremental``); see :mod:`veridist.scale._checkpoint`. See the module documentation for
    the result and the refusals.
    """

    spec = FAMILY_REGISTRY.lookup(family)
    bound = check_max_rows(max_rows)
    budget = check_pass_budget(pass_budget)
    if not isinstance(source, Source):
        raise TypeError("source must provide partitions() and batches()")
    if spool is not None and not isinstance(spool, SpoolPolicy):
        raise TypeError("spool must be a SpoolPolicy or None")
    resolved = resolve_executor(executor)
    request = resolve_checkpoint(checkpoint, resume, incremental, recompute_changed, allow_removed)
    plan = _PLANS.get(spec.id)
    if request is not None:
        refuse_incremental(request, multi_pass=plan is None)
    if plan is None:
        return _fit_multi_pass(source, spec.id, bound, budget, spool, resolved, request)
    partitions = source.partitions()
    opened = None
    if request is not None:
        require_resumable(partitions)
        schema = _state_schema((plan.state_type.STATE_TAG, plan.state_type.SCHEMA_VERSION))
        opened = _open_checkpoint(
            request, spec.id, source, bound, partitions, schema, multi_pass=False
        )
    engine = PassEngine(resolved, bound, opened)
    try:
        with engine:
            state = engine.run(source, partitions, FoldJob(plan.state_type))
    except ScaleStateError as error:
        if error.code is not ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT or (
            plan.invalid_support is None
        ):
            raise
        rows = _count_rows(source, partitions, bound)
        return ScaleFitResult(
            plan.invalid_support(rows),
            _provenance(
                source,
                partitions,
                rows,
                None,
                bound,
                execution=engine.provenance(),
                checkpoint=engine.checkpoint_provenance(),
            ),
        )
    return ScaleFitResult(
        plan.estimate(state),
        _provenance(
            source,
            partitions,
            state.count,
            state,
            bound,
            execution=engine.provenance(),
            checkpoint=engine.checkpoint_provenance(),
        ),
    )


__all__ = ["CUSTOM_SOURCE_KIND", "SOURCE_KINDS", "ScaleFitResult", "build_provenance", "fit"]
