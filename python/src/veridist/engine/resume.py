"""Strict checkpoint compatibility validation and immutable resume results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from veridist.engine.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    CheckpointRecord,
    CheckpointStore,
)
from veridist.engine.errors import EngineContractError, FailureCode, safe_exception_type
from veridist.engine.retry import PureReducer

Accumulator = TypeVar("Accumulator")


def _require_text(label: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")


@dataclass(frozen=True, slots=True)
class ResumeExpectation:
    """Private compatibility facts supplied by source and execution preflight."""

    format_version: int
    source_id: str
    source_schema: str
    source_revision: str | None
    reducer_id: str
    accumulator_schema: str
    plan_digest: str
    cursor: int

    def __post_init__(self) -> None:
        if isinstance(self.format_version, bool) or self.format_version < 1:
            raise ValueError("format_version must be positive")
        if isinstance(self.cursor, bool) or self.cursor < 0:
            raise ValueError("cursor must be non-negative")
        for label, value in (
            ("source_id", self.source_id),
            ("source_schema", self.source_schema),
            ("reducer_id", self.reducer_id),
            ("accumulator_schema", self.accumulator_schema),
            ("plan_digest", self.plan_digest),
        ):
            _require_text(label, value)


@dataclass(frozen=True, slots=True)
class PublicResumeMetadata:
    """Public checkpoint facts; the private source revision is intentionally absent."""

    format_version: int
    source_id: str
    source_schema: str
    reducer_id: str
    accumulator_schema: str
    plan_digest: str
    cursor: int
    committed_ranges: tuple[tuple[int, int], ...]
    generation: int


@dataclass(frozen=True, slots=True)
class ResumedCheckpoint(Generic[Accumulator]):
    """Validated decoded state in a frozen envelope with privacy-safe metadata."""

    accumulator: Accumulator
    cursor: int
    committed_ranges: tuple[tuple[int, int], ...]
    generation: int
    public_metadata: PublicResumeMetadata


def source_id_mismatch(record: CheckpointRecord, source_id: str) -> bool:
    """Return whether the checkpoint's public source id differs from the expected one.

    Shared by `_validate_compatibility` and
    `execution.fit_exponential_checkpointed_csv` so the comparison is defined
    exactly once.
    """

    return record.source_id != source_id


def source_schema_mismatch(record: CheckpointRecord, source_schema: str) -> bool:
    """Return whether the checkpoint's source schema differs from the expected one.

    Shared by `_validate_compatibility` and
    `execution.fit_exponential_checkpointed_csv` so the comparison is defined
    exactly once.
    """

    return record.source_schema != source_schema


def _validate_compatibility(
    record: CheckpointRecord,
    expected: ResumeExpectation,
    reducer: PureReducer[Accumulator],
) -> None:
    """Validate checksum, format, source, revision, accumulator, plan and range."""

    if not record.has_valid_checksum():
        raise EngineContractError(FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)
    if (
        record.format_version != CHECKPOINT_FORMAT_VERSION
        or expected.format_version != CHECKPOINT_FORMAT_VERSION
    ):
        raise EngineContractError(FailureCode.CHECKPOINT_FORMAT_UNSUPPORTED)
    if source_id_mismatch(record, expected.source_id):
        raise EngineContractError(FailureCode.SOURCE_ID_MISMATCH)
    if source_schema_mismatch(record, expected.source_schema):
        raise EngineContractError(FailureCode.SOURCE_SCHEMA_MISMATCH)
    if record.source_revision != expected.source_revision:
        raise EngineContractError(FailureCode.SOURCE_REVISION_MISMATCH)
    if record.reducer_id != expected.reducer_id or reducer.reducer_id != expected.reducer_id:
        raise EngineContractError(FailureCode.REDUCER_MISMATCH)
    if (
        record.accumulator_schema != expected.accumulator_schema
        or reducer.accumulator_schema != expected.accumulator_schema
    ):
        raise EngineContractError(FailureCode.ACCUMULATOR_SCHEMA_MISMATCH)
    if record.plan_digest != expected.plan_digest:
        raise EngineContractError(FailureCode.PLAN_MISMATCH)
    canonical_ranges = () if record.cursor == 0 else ((0, record.cursor),)
    if record.cursor != expected.cursor or record.committed_ranges != canonical_ranges:
        raise EngineContractError(FailureCode.RANGE_MISMATCH)


def resume_checkpoint(
    *,
    store: CheckpointStore,
    expected: ResumeExpectation,
    reducer: PureReducer[Accumulator],
) -> ResumedCheckpoint[Accumulator]:
    """Load, validate in fixed order, then decode without writing checkpoint state."""

    if not isinstance(reducer, PureReducer):
        raise EngineContractError(FailureCode.RETRY_NOT_ADMISSIBLE)
    if expected.source_revision is None or not expected.source_revision.strip():
        raise EngineContractError(FailureCode.SOURCE_REVISION_UNAVAILABLE)
    record = store.read()
    _validate_compatibility(record, expected, reducer)
    try:
        accumulator = reducer.decode_state(record.state)
    except Exception as exc:
        raise EngineContractError(
            FailureCode.REDUCER_FAILURE,
            {"failure_type": safe_exception_type(exc)},
        ) from None
    public = PublicResumeMetadata(
        format_version=record.format_version,
        source_id=record.source_id,
        source_schema=record.source_schema,
        reducer_id=record.reducer_id,
        accumulator_schema=record.accumulator_schema,
        plan_digest=record.plan_digest,
        cursor=record.cursor,
        committed_ranges=record.committed_ranges,
        generation=record.generation,
    )
    return ResumedCheckpoint(
        accumulator=accumulator,
        cursor=record.cursor,
        committed_ranges=record.committed_ranges,
        generation=record.generation,
        public_metadata=public,
    )
