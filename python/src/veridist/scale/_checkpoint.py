"""Partition-level checkpoint, resume and incremental fitting for the scale fit.

``fit(..., checkpoint=store_or_path)`` keeps the exact state of every partition it reads in a
local :class:`~veridist.scale._partition_store.SQLitePartitionStore` and the merged result of
every completed pass in its pass ledger. This module is the parent-side logic between the
store and the executors; workers never see it, they return state bytes and the parent commits
them one partition at a time, as the results arrive and in any order (ADR-0015).

Job identity. A store belongs to one job, named by components that are compared exactly:
``family``, ``column`` (as a digest), ``source_kind``, ``max_rows``, ``value_policy``,
``fingerprint_level``, ``state_schema``, ``plan`` (the constants that fix the passes of a
multi-pass family) and ``environment`` (the digest of the numeric environment). A resume
whose job differs is refused with ``JOB_MISMATCH`` and the names of the components that
differ, and nothing is read or written. The executor, the number of workers, the spool, the
pass budget and the order in which partitions are read are not part of the identity: the
result does not depend on them, which is what a resume relies on.

Starting and resuming. With ``resume=False`` a store that already holds a job is refused
(``JOB_EXISTS``); an existing one is never overwritten. With ``resume=True`` a store that
holds no job starts one, and a store that holds a job continues it. Only a source whose
partitions all have a fingerprint can be checkpointed (``SOURCE_REVISION_UNAVAILABLE`` for
an Arrow stream, a Polars plan or a DuckDB relation: they cannot be recognised again).

What a resume does. The partitions of the source are compared with the manifest the job was
started with. A partition whose fingerprint differs is the typed ``PARTITION_CHANGED``
unless ``recompute_changed=True`` asks for it to be read again and its state replaced; a
partition that is gone is ``PARTITION_MISSING``; a partition that is new is
``PARTITION_ADDED``. Every refusal comes before any read and changes nothing. In a pass,
every partition that has a committed state under the pass's exact parameters is not read;
the others are, and each is committed the moment it returns, so an interrupted run loses at
most the partitions in flight. The merged result is the exact merge of the same states an
uninterrupted run would have made, so it is bit for bit that result whatever the order in
which they were merged.

The pass ledger. When every partition of a pass is stored, the parent merges them, and
commits the pass (its index, the exact record of its parameters and the merged state) in one
transaction. A resumed fit runs the planner from the start: a pass in the ledger is replayed
from it, without reading and without merging, and its parameters are checked against the
ones the planner derives (``LEDGER_MISMATCH`` if they differ); the first pass not in the
ledger is the one that continues. A replayed pass still counts as a pass of the budget, so a
resumed fit takes, and is bounded by, exactly the passes an uninterrupted one does. A multi-pass
fit keeps the partition states of its latest
pass only; the one-pass families keep theirs, which is the index that incremental fitting
reuses.

Incremental fitting (``incremental=True``, with ``resume=True``): for a one-pass family, a
source that holds every committed partition and more is fitted by reading only the new
partitions and merging their states with the stored ones; a partition that is gone is
``PARTITION_MISSING`` unless ``allow_removed=True``, which excludes it and records it. A
multi-pass family refuses it up front (``INCREMENTAL_UNSUPPORTED``): a new partition changes
the sums of every pass. For a multi-pass family a changed partition, accepted with
``recompute_changed=True``, makes every stored pass stale, so the stored passes are
discarded and the fit starts again from the first pass.

Privacy. The record in provenance holds counts, the digest of the job and the index of every
pass; no path, partition identifier, fingerprint or state.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

from veridist.engine.provenance import ScaleCheckpointPassProvenance, ScaleCheckpointProvenance
from veridist.scale._environment import numeric_environment_digest
from veridist.scale._errors import (
    ScaleCheckpointError,
    ScaleCheckpointErrorCode,
    ScaleSourceError,
    ScaleSourceErrorCode,
)
from veridist.scale._partition_store import (
    PARTITION_STORE_FORMAT_VERSION,
    STORE_KIND,
    ManifestEntry,
    SQLitePartitionStore,
    StoredState,
)
from veridist.scale._sources import VALUE_POLICY_VERSION, Partition, canonical_digest
from veridist.scale._tasks import Job, TaskOutcome

COMPONENT_NAMES: Final = (
    "family",
    "column",
    "source_kind",
    "max_rows",
    "value_policy",
    "fingerprint_level",
    "state_schema",
    "plan",
    "environment",
)
"""The components of the identity of a job, in the order a mismatch is reported."""

NO_COMPONENT: Final = "-"
"""The value of a component that does not apply (no column, no plan)."""

HINT_CHANGED: Final = (
    "a partition no longer matches the state that was committed for it; "
    "pass recompute_changed=True to read it again and replace that state"
)


@dataclass(frozen=True, slots=True)
class CheckpointRequest:
    """What ``fit`` was asked to checkpoint: the store and how to treat what it holds."""

    store: SQLitePartitionStore
    resume: bool
    incremental: bool
    recompute_changed: bool
    allow_removed: bool


def _flag(name: str, value: object) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool")
    return value


def resolve_checkpoint(
    checkpoint: object,
    resume: object,
    incremental: object,
    recompute_changed: object,
    allow_removed: object,
) -> CheckpointRequest | None:
    """Validate the checkpoint arguments of ``fit``; ``None`` if no checkpoint was asked for.

    ``checkpoint`` is a :class:`SQLitePartitionStore` or a path (``str`` or path-like of
    ``str``). ``TypeError`` for another type or a flag that is not a ``bool``; ``ValueError``
    for a flag without a checkpoint, for ``incremental`` without ``resume`` and for
    ``allow_removed`` without ``incremental``. Nothing is created or opened here.
    """

    flags = (
        _flag("resume", resume),
        _flag("incremental", incremental),
        _flag("recompute_changed", recompute_changed),
        _flag("allow_removed", allow_removed),
    )
    if checkpoint is None:
        if any(flags):
            raise ValueError(
                "resume, incremental, recompute_changed and allow_removed need a checkpoint"
            )
        return None
    if isinstance(checkpoint, SQLitePartitionStore):
        store = checkpoint
    elif isinstance(checkpoint, str | os.PathLike) and isinstance(os.fspath(checkpoint), str):
        store = SQLitePartitionStore(checkpoint)
    else:
        raise TypeError("checkpoint must be a SQLitePartitionStore or a path")
    if incremental and not resume:
        raise ValueError("incremental needs resume=True")
    if allow_removed and not incremental:
        raise ValueError("allow_removed needs incremental=True")
    return CheckpointRequest(store, *flags)


def require_resumable(partitions: Sequence[Partition]) -> None:
    """Refuse a source with a partition that has no fingerprint: ``SOURCE_REVISION_UNAVAILABLE``."""

    if not all(partition.resumable for partition in partitions):
        raise ScaleCheckpointError(ScaleCheckpointErrorCode.SOURCE_REVISION_UNAVAILABLE)


def refuse_incremental(request: CheckpointRequest, *, multi_pass: bool) -> None:
    """``INCREMENTAL_UNSUPPORTED`` for an incremental request on a multi-pass family."""

    if request.incremental and multi_pass:
        raise ScaleCheckpointError(ScaleCheckpointErrorCode.INCREMENTAL_UNSUPPORTED)


@dataclass(frozen=True, slots=True)
class JobIdentity:
    """The named components of a job and their digest."""

    components: tuple[tuple[str, str], ...]

    @property
    def digest(self) -> str:
        return sha256(
            canonical_digest(*(item for pair in self.components for item in pair)).encode("ascii")
        ).hexdigest()


def job_identity(
    *,
    family: str,
    column: str | None,
    source_kind: str,
    max_rows: int,
    partitions: Sequence[Partition],
    state_schema: str,
    plan: tuple[object, ...] | None,
) -> JobIdentity:
    """The identity of the job a fit is: see the module documentation for its components."""

    levels = ",".join(sorted({partition.fingerprint_level.value for partition in partitions}))
    values = {
        "family": family,
        "column": NO_COMPONENT if column is None else canonical_digest("column", column),
        "source_kind": source_kind,
        "max_rows": str(max_rows),
        "value_policy": str(VALUE_POLICY_VERSION),
        "fingerprint_level": levels or NO_COMPONENT,
        "state_schema": state_schema,
        "plan": NO_COMPONENT if plan is None else canonical_digest(*plan),
        "environment": numeric_environment_digest(),
    }
    return JobIdentity(tuple((name, values[name]) for name in COMPONENT_NAMES))


def _entry(partition: Partition) -> ManifestEntry:
    assert partition.fingerprint is not None
    return ManifestEntry(partition.id, partition.fingerprint, partition.fingerprint_level.value)


class PassCheckpoint:
    """One pass of a checkpointed fit: what is stored, what is left to read, what was done.

    ``replayed`` is the merged state of the pass if the ledger holds it (nothing is read);
    otherwise ``pending`` are the partitions to read, in canonical order, and the committed
    states of every other partition are handed out one at a time by :meth:`take_reused`. The
    parent calls :meth:`commit` for each state a task returns and :meth:`finish` with the
    merged bytes.
    """

    def __init__(
        self,
        run: CheckpointRun,
        index: int,
        key: str,
        descriptor: str,
        partitions: Sequence[Partition],
        replayed: bytes | None,
        reused: dict[int, bytes],
    ) -> None:
        self._run = run
        self._index = index
        self._key = key
        self._descriptor = descriptor
        self._partitions = {partition.ordinal: partition for partition in partitions}
        self._total = len(partitions)
        self.replayed = replayed
        self.reused = reused
        self._reused_count = len(reused)
        self.pending: tuple[Partition, ...] = ()
        if replayed is None:
            self.pending = tuple(
                partition for partition in partitions if partition.ordinal not in reused
            )
        self._read = 0

    def take_reused(self) -> Iterator[bytes]:
        """The committed states of the partitions that were not read, in canonical order.

        ``reused`` holds them by ordinal until then. Each is handed over once and is no
        longer held by this record once it has been yielded, so the caller can merge it and
        drop it.
        """

        for ordinal in sorted(self.reused):
            yield self.reused.pop(ordinal)

    def commit(self, outcome: TaskOutcome) -> None:
        """Commit the state one partition task returned; the parent is the only writer."""

        partition = self._partitions[outcome.ordinal]
        self._run.store.commit_partition(
            self._index,
            self._key,
            partition.id,
            *_fingerprint(partition),
            outcome.data,
        )
        self._read += 1

    def finish(self, merged: bytes) -> None:
        """Commit the completed pass to the ledger (and keep only the latest pass's states)."""

        if self.replayed is None:
            self._run.store.commit_pass(
                self._index, self._key, self._descriptor, merged, prune=self._run.multi_pass
            )

    def record(self) -> ScaleCheckpointPassProvenance:
        """What this pass did with the store."""

        if self.replayed is not None:
            return ScaleCheckpointPassProvenance(self._index, self._total, True, self._total, 0)
        return ScaleCheckpointPassProvenance(
            self._index, self._total, False, self._reused_count, self._read
        )


def _fingerprint(partition: Partition) -> tuple[str, str]:
    entry = _entry(partition)
    return entry.fingerprint, entry.level


class CheckpointRun:
    """The checkpoint of one fit: its job, its ledger and the records of its passes."""

    def __init__(
        self,
        request: CheckpointRequest,
        identity: JobIdentity,
        partitions: tuple[Partition, ...],
        *,
        multi_pass: bool,
        resumed: bool,
        added: int,
        removed: int,
        recomputed: int,
        restarted: bool,
    ) -> None:
        self.store = request.store
        self.multi_pass = multi_pass
        self._request = request
        self._identity = identity
        self._partitions = partitions
        self._resumed = resumed
        self._added = added
        self._removed = removed
        self._recomputed = recomputed
        self._restarted = restarted
        self._ledger = self.store.read_ledger() if resumed else {}
        self._records: list[PassCheckpoint] = []

    @classmethod
    def open(
        cls,
        request: CheckpointRequest,
        identity: JobIdentity,
        partitions: tuple[Partition, ...],
        *,
        multi_pass: bool,
    ) -> CheckpointRun:
        """Start the job, or resume the one the store holds; refusals precede every read.

        The partitions must all be resumable (:func:`require_resumable`).
        """

        store = request.store
        current = tuple(_entry(partition) for partition in partitions)
        stored = store.read_job()
        if stored is None:
            store.create_job(identity.digest, identity.components, current)
            return cls(
                request,
                identity,
                partitions,
                multi_pass=multi_pass,
                resumed=False,
                added=0,
                removed=0,
                recomputed=0,
                restarted=False,
            )
        if not request.resume:
            raise ScaleCheckpointError(ScaleCheckpointErrorCode.JOB_EXISTS)
        kept = dict(stored.components)
        differing = tuple(name for name, value in identity.components if kept.get(name) != value)
        if differing:
            raise ScaleCheckpointError(ScaleCheckpointErrorCode.JOB_MISMATCH, differing)
        before = {entry.partition_id: entry for entry in stored.manifest}
        now = {entry.partition_id: entry for entry in current}
        changed = tuple(
            entry
            for entry in current
            if entry.partition_id in before and entry != before[entry.partition_id]
        )
        removed = tuple(name for name in before if name not in now)
        added = tuple(entry for entry in current if entry.partition_id not in before)
        if changed and not request.recompute_changed:
            raise ScaleSourceError(ScaleSourceErrorCode.PARTITION_CHANGED, HINT_CHANGED)
        if removed and not request.allow_removed:
            raise ScaleCheckpointError(ScaleCheckpointErrorCode.PARTITION_MISSING)
        if added and not request.incremental:
            raise ScaleCheckpointError(ScaleCheckpointErrorCode.PARTITION_ADDED)
        if changed or removed or added:
            store.amend_job(add=added, remove=removed, replace=changed, reset=multi_pass)
        return cls(
            request,
            identity,
            partitions,
            multi_pass=multi_pass,
            resumed=True,
            added=len(added),
            removed=len(removed),
            recomputed=len(changed),
            restarted=multi_pass and bool(changed),
        )

    def begin_pass(self, index: int, job: Job, partitions: Sequence[Partition]) -> PassCheckpoint:
        """Look at what the store holds for pass ``index`` of ``job`` over ``partitions``.

        ``LEDGER_MISMATCH`` if the ledger has this pass with other parameters.
        """

        descriptor = json.dumps(job.checkpoint_key(), separators=(",", ":"))
        key = sha256(descriptor.encode("utf-8")).hexdigest()
        entry = self._ledger.get(index)
        if entry is not None and entry.pass_key != key:
            raise ScaleCheckpointError(ScaleCheckpointErrorCode.LEDGER_MISMATCH)
        replayed = None if entry is None else entry.data
        reused: dict[int, bytes] = {}
        if replayed is None:
            held = self.store.read_pass(index, key) if self._resumed else {}
            for partition in partitions:
                state = held.get(partition.id)
                if state is not None:
                    reused[partition.ordinal] = _usable(partition, state)
        step = PassCheckpoint(self, index, key, descriptor, partitions, replayed, reused)
        self._records.append(step)
        return step

    def provenance(self) -> ScaleCheckpointProvenance:
        """The record of the checkpoint: the job, what was reused, read and replayed."""

        return ScaleCheckpointProvenance(
            STORE_KIND,
            PARTITION_STORE_FORMAT_VERSION,
            self._identity.digest,
            self._resumed,
            self._request.incremental,
            self._added,
            self._removed,
            self._recomputed,
            self._restarted,
            tuple(step.record() for step in self._records),
        )


def _usable(partition: Partition, state: StoredState) -> bytes:
    """The committed state of ``partition`` if it was computed from what it is now."""

    if (state.fingerprint, state.level) != _fingerprint(partition):
        raise ScaleSourceError(ScaleSourceErrorCode.PARTITION_CHANGED, HINT_CHANGED)
    return state.data


__all__ = [
    "COMPONENT_NAMES",
    "HINT_CHANGED",
    "CheckpointRequest",
    "CheckpointRun",
    "JobIdentity",
    "PassCheckpoint",
    "job_identity",
    "refuse_incremental",
    "require_resumable",
    "resolve_checkpoint",
]
