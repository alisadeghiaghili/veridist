"""Partition tasks: the unit of work an executor runs and its retry rule.

A pass over a source is a set of *partition tasks*, one per partition. A task is a pure
function of its description: a :class:`PartitionTask` holds the source (or, for worker
processes, the picklable source to rebuild the reader from), the partition descriptor, the
batch cap and a *job*, and running it reads that one partition from the start and returns
the canonical bytes of the exact state or accumulator the job folds it into. Bytes, a few
kilobytes, are all that leaves a worker; data never does.

Jobs. A job knows what is folded (:class:`FoldJob` folds a mergeable state; the multi-pass
fit has one for the profile sums), how to read the bytes back, and the identity of the merge.
Merging is exact, so the merged value does not depend on the order in which the bytes
arrive or on the shape of the merge: the executors merge each value into one running value as
it arrives and drop it (:mod:`veridist.scale._executors`).

Retry. A task that fails with a transient input or output error is run again from the start
of its partition, at most ``PartitionTask.retries`` times (``MAX_PARTITION_RETRIES`` for an
executor). The error is transient if it is an ``OSError`` other than ``FileNotFoundError``
and ``PermissionError``, either raised as it is or the cause of the source's
``SOURCE_UNREADABLE``. Every other error is final and is raised as it is, so a refusal of the
value policy or of a state is the same typed error whatever the executor. The outcome
reports how many times the partition was read again.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Final, Protocol

from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._sources import BatchReader, Partition, fold_partition
from veridist.scale._state import MergeableState

MAX_PARTITION_RETRIES: Final = 2
"""How many times a partition is read again after a transient error (so at most 3 reads)."""


def initialize_worker(modules: tuple[str, ...]) -> None:
    """Import ``modules`` in a new worker process before its first task.

    The import cost (numpy and the reader's library) is paid when the pool starts, in every
    worker at once, and not inside the first task. A worker is one of several, so Arrow's own
    thread pools are pinned to one thread each, as the cores are shared out by the executor.
    """

    for module in [import_module(name) for name in modules]:
        if module.__name__ == "pyarrow":
            module.set_cpu_count(1)
            module.set_io_thread_count(1)


class Job(Protocol):
    """What a pass folds each partition into; picklable, so that workers can receive it."""

    def run(self, source: BatchReader, partition: Partition, max_rows: int) -> bytes:
        """Read ``partition`` and return the canonical bytes of what it folds into."""
        ...

    def restore(self, data: bytes) -> Any:
        """The mergeable value that ``data`` encodes (``merge`` combines two of them)."""
        ...

    def identity(self) -> Any:
        """The value of no rows: merging it into another changes nothing."""
        ...

    def checkpoint_key(self) -> tuple[object, ...]:
        """The exact identity of the pass: what is folded and every parameter it is folded at.

        A canonical record of integers, strings and tuples of them (a float is its
        hexadecimal form). A stored partition state is reused only under an equal key.
        """
        ...


@dataclass(frozen=True, slots=True)
class FoldJob:
    """Fold a partition into the mergeable state of ``state_type``."""

    state_type: type[MergeableState[Any]]

    def run(self, source: BatchReader, partition: Partition, max_rows: int) -> bytes:
        return fold_partition(source, partition, self.state_type, max_rows=max_rows).to_bytes()

    def restore(self, data: bytes) -> MergeableState[Any]:
        return self.state_type.from_bytes(data)

    def identity(self) -> MergeableState[Any]:
        return self.state_type.empty()

    def checkpoint_key(self) -> tuple[object, ...]:
        return ("fold", self.state_type.STATE_TAG, self.state_type.SCHEMA_VERSION)


@dataclass(frozen=True, slots=True)
class PartitionTask:
    """One partition of one pass: everything a worker needs, and nothing else."""

    source: BatchReader
    partition: Partition
    max_rows: int
    job: Job
    retries: int = 0


@dataclass(frozen=True, slots=True)
class TaskOutcome:
    """What a task returns: the partition's ordinal, its state bytes and the re-reads it took."""

    ordinal: int
    data: bytes
    retries: int


def is_transient(error: BaseException | None) -> bool:
    """Whether ``error`` is a transient input or output failure worth reading again.

    An ``OSError`` that is not a ``FileNotFoundError`` or a ``PermissionError`` is; so is the
    ``SOURCE_UNREADABLE`` of a source whose cause is one. Nothing else is.
    """

    if isinstance(error, ScaleSourceError):
        if error.code is not ScaleSourceErrorCode.SOURCE_UNREADABLE:
            return False
        error = error.__cause__
    return isinstance(error, OSError) and not isinstance(error, FileNotFoundError | PermissionError)


def run_partition_task(task: PartitionTask) -> TaskOutcome:
    """Run one task, reading its partition again after a transient error, up to its cap.

    The last read is not guarded: its error, whatever it is, is raised.
    """

    for retried in range(task.retries):
        try:
            data = task.job.run(task.source, task.partition, task.max_rows)
        except Exception as error:
            if not is_transient(error):
                raise
            continue
        return TaskOutcome(task.partition.ordinal, data, retried)
    return TaskOutcome(
        task.partition.ordinal,
        task.job.run(task.source, task.partition, task.max_rows),
        task.retries,
    )


__all__ = [
    "MAX_PARTITION_RETRIES",
    "FoldJob",
    "Job",
    "PartitionTask",
    "TaskOutcome",
    "initialize_worker",
    "is_transient",
    "run_partition_task",
]
