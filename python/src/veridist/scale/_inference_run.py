"""What inference from a scale fit shares with the fit: options, passes, spool and checkpoint.

A profile-likelihood interval or a goodness-of-fit report is computed from extra passes over the
source of a fit (:mod:`veridist.scale._profile_intervals`, :mod:`veridist.scale._gof`). Those passes
are ordinary scale passes: they are counted against an explicit budget and refused before they are
read when it is exhausted, they are read by the same executors, kept in the same partition
checkpoint (with a job of their own: one store holds one job) and recorded in the same provenance.

Pass budget. Inference has a budget of its own, ``inference_pass_budget``, separate from the
``pass_budget`` of the fit: the fit has been paid for and returned, and a budget that covered both
would make the same fit succeed or fail by what is asked of it afterwards. A budget below the
fewest passes the plan needs is refused before any read with ``PASS_BUDGET_INSUFFICIENT``; a run
that would need a pass beyond the budget is refused before that pass is read with
``PASS_BUDGET_EXCEEDED``. Neither falls back to anything: no interval or report is returned.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from os import PathLike
from typing import Final, TypeVar

from veridist.engine.errors import FailureCode
from veridist.engine.pass_budget import PassBudgetError
from veridist.engine.provenance import ScaleProvenance, ScaleStateProvenance
from veridist.scale._checkpoint import (
    CheckpointRequest,
    CheckpointRun,
    job_identity,
    require_resumable,
    resolve_checkpoint,
)
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._executors import Executor, PassEngine, resolve_executor
from veridist.scale._fit import CUSTOM_SOURCE_KIND, SOURCE_KINDS, build_provenance
from veridist.scale._partition_store import SQLitePartitionStore
from veridist.scale._planner import PassLedger, PlanBounds, check_pass_budget
from veridist.scale._sources import Source, check_max_rows, is_single_use
from veridist.scale._spool import HINT_REQUIRED, ScaleSpoolPolicy, Spool

ResultT = TypeVar("ResultT")

INFERENCE_JOB_PREFIX: Final = "inference"
"""The family component of an inference job names what it computes after this prefix."""


@dataclass(frozen=True, slots=True)
class InferenceOptions:
    """The validated arguments every inference entry point takes besides its own."""

    max_rows: int
    budget: int
    spool: ScaleSpoolPolicy | None
    executor: Executor | None
    request: CheckpointRequest | None


def check_options(
    source: object,
    *,
    max_rows: object,
    budget: object,
    spool: object,
    executor: object,
    checkpoint: object,
    resume: object,
    recompute_changed: object,
) -> InferenceOptions:
    """Validate the arguments before anything is read or created.

    ``TypeError`` or ``ValueError`` as the arguments of ``fit`` are refused. ``incremental`` and
    ``allow_removed`` do not exist here: a source that gained or lost a partition is another
    data set, and every sum of an inference pass would change.
    """

    bound = check_max_rows(max_rows)
    passes = check_pass_budget(budget)
    if not isinstance(source, Source):
        raise TypeError("source must provide partitions() and batches()")
    if spool is not None and not isinstance(spool, ScaleSpoolPolicy):
        raise TypeError("spool must be a ScaleSpoolPolicy or None")
    resolved = resolve_executor(executor)
    request = resolve_checkpoint(checkpoint, resume, False, recompute_changed, False)
    return InferenceOptions(bound, passes, spool, resolved, request)


def inference_bounds(budget: int, minimum: int, maximum: int) -> PlanBounds:
    """The fewest and most passes of an inference plan, refusing a budget below the fewest.

    ``PassBudgetError`` with ``PASS_BUDGET_INSUFFICIENT`` and the budget and both bounds in its
    context; nothing has been read when this is called.
    """

    if budget < minimum:
        raise PassBudgetError(
            FailureCode.PASS_BUDGET_INSUFFICIENT,
            {"pass_budget": budget, "minimum_passes": minimum, "maximum_passes": maximum},
        )
    return PlanBounds(minimum, maximum)


def run_inference(
    source: Source,
    options: InferenceOptions,
    bounds: PlanBounds,
    *,
    kind: str,
    state: ScaleStateProvenance | None,
    rows: int,
    schema: str,
    plan: tuple[object, ...],
    rereads: bool,
    work: Callable[[PassLedger, Source, PassEngine], ResultT],
) -> tuple[ResultT, ScaleProvenance]:
    """Run ``work`` over ``source`` with a ledger, an engine, an optional spool and checkpoint.

    ``work(ledger, source, engine)`` takes the passes it needs through the ledger and the
    engine and returns its result; ``state`` is the record of the state of the fit the numbers
    are derived from and ``rows`` the number of rows each pass must read. ``kind`` names the
    computation in the identity of a checkpoint job, ``schema`` the byte formats of its
    passes and ``plan`` the constants and arguments that fix its passes. ``rereads`` says
    that the plan may read the source more than once: a source that can be read only once is then
    refused before anything is read (``SPOOL_REQUIRED``) unless a spool is requested, exactly as
    a multi-pass fit refuses it.
    """

    partitions = source.partitions()
    if options.request is not None:
        require_resumable(partitions)
    ledger = PassLedger(partitions, options.budget, bounds)
    if rereads and options.spool is None and is_single_use(source):
        raise ScaleSourceError(ScaleSourceErrorCode.SPOOL_REQUIRED, HINT_REQUIRED)
    checkpoint = None
    if options.request is not None:
        identity = job_identity(
            family=f"{INFERENCE_JOB_PREFIX}:{kind}",
            column=getattr(source, "column", None),
            source_kind=SOURCE_KINDS.get(type(source), CUSTOM_SOURCE_KIND),
            max_rows=options.max_rows,
            partitions=partitions,
            state_schema=schema,
            plan=plan,
        )
        checkpoint = CheckpointRun.open(options.request, identity, partitions, multi_pass=True)
    engine = PassEngine(options.executor, options.max_rows, checkpoint)
    if options.spool is None:
        with engine:
            outcome = work(ledger, source, engine)
        record = None
    else:
        # The engine is left first: no worker holds a spool file when the directory goes.
        with Spool(options.spool) as local, engine:
            outcome = work(ledger, local.source(source), engine)
            record = local.provenance()
    return outcome, build_provenance(
        source,
        partitions,
        rows,
        state,
        options.max_rows,
        ledger.provenance(),
        record,
        engine.provenance(),
        engine.checkpoint_provenance(),
    )


StoreArgument = SQLitePartitionStore | str | PathLike[str] | None

__all__ = [
    "INFERENCE_JOB_PREFIX",
    "InferenceOptions",
    "StoreArgument",
    "check_options",
    "inference_bounds",
    "run_inference",
]
