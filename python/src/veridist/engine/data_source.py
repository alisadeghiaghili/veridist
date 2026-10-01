"""Preflight contracts for data-source identity, replayability, and spooling.

This module deliberately plans work from source metadata only.  It neither
iterates a source nor creates a spool; execution adapters remain out of scope.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol

from veridist.engine.errors import EngineContractError, FailureCode

SUPPORTED_PROVENANCE_SCHEMA_VERSIONS = frozenset({"1"})


class Replayability(StrEnum):
    """Declared ability of a source to provide data for subsequent passes."""

    SINGLE_PASS = "single_pass"
    REPLAYABLE = "replayable"
    CHECKPOINT_REPLAYABLE = "checkpoint_replayable"


class DataSourceCapabilityError(EngineContractError):
    """A stable, localization-independent planner capability failure."""


@dataclass(frozen=True, slots=True)
class DataSourceMetadata:
    """Immutable identity and provenance declaration required before a read."""

    source_id: str
    schema_version: str
    provenance_schema_version: str
    replayability: Replayability
    source_hash: str | None = None
    redaction_reason: str | None = None
    checkpoint_schema_version: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("source_id", self.source_id),
            ("schema_version", self.schema_version),
            ("provenance_schema_version", self.provenance_schema_version),
        ):
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")
        if self.provenance_schema_version not in SUPPORTED_PROVENANCE_SCHEMA_VERSIONS:
            raise ValueError("unsupported provenance_schema_version")
        if self.source_hash is not None and not self.source_hash.strip():
            raise ValueError("source_hash must be non-empty when supplied")
        if self.redaction_reason is not None and not self.redaction_reason.strip():
            raise ValueError("redaction_reason must be non-empty when supplied")
        if (self.source_hash is None) == (self.redaction_reason is None):
            raise ValueError("provide exactly one of source_hash or redaction_reason")
        if (
            self.checkpoint_schema_version is not None
            and not self.checkpoint_schema_version.strip()
        ):
            raise ValueError("checkpoint_schema_version must be non-empty when supplied")


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """The source identity and schema declared by a checkpoint candidate."""

    source_id: str
    schema_version: str

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.schema_version.strip():
            raise ValueError("checkpoint source_id and schema_version must be non-empty")


@dataclass(frozen=True, slots=True)
class SpoolPolicy:
    """An explicit, declarative spool policy; it performs no I/O itself."""

    enabled: bool
    disk_budget_bytes: int | None = None
    retention: str | None = None
    cleanup_required: bool | None = None

    def __post_init__(self) -> None:
        if not self.enabled:
            if any(
                value is not None
                for value in (self.disk_budget_bytes, self.retention, self.cleanup_required)
            ):
                raise ValueError("disabled spool policy cannot declare spool requirements")
            return
        if self.disk_budget_bytes is None or self.disk_budget_bytes <= 0:
            raise ValueError("enabled spool policy requires a positive disk_budget_bytes")
        if self.retention is None or not self.retention.strip():
            raise ValueError("enabled spool policy requires a retention policy")
        if self.cleanup_required is None:
            raise ValueError("enabled spool policy requires a cleanup policy")

    @classmethod
    def disabled(cls) -> SpoolPolicy:
        """Return the explicit no-spool policy."""

        return cls(enabled=False)


class DataSourceLike(Protocol):
    """Minimum source surface consumed by the preflight planner."""

    metadata: DataSourceMetadata


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Approved preflight plan, without an execution or materialization action."""

    required_passes: int
    spool_requirements: SpoolPolicy | None
    provenance: Mapping[str, object]
    replayability: Replayability

    @property
    def spool_enabled(self) -> bool:
        return self.spool_requirements is not None


def plan_passes(
    source: DataSourceLike,
    *,
    required_passes: int,
    spool: SpoolPolicy | None = None,
    checkpoint: CheckpointMetadata | None = None,
) -> ExecutionPlan:
    """Validate a requested pass count from metadata before a source is read."""

    if required_passes < 1:
        raise ValueError("required_passes must be at least one")

    metadata = source.metadata
    _validate_checkpoint(metadata, checkpoint, required_passes)
    spool_requirements = _spool_requirements(metadata, required_passes, spool)
    provenance: dict[str, object] = {
        "source_id": metadata.source_id,
        "schema_version": metadata.schema_version,
        "provenance_schema_version": metadata.provenance_schema_version,
        "replayability": metadata.replayability.value,
        "required_passes": required_passes,
    }
    if metadata.source_hash is not None:
        provenance["source_hash"] = metadata.source_hash
    else:
        provenance["redaction_reason"] = metadata.redaction_reason
    if spool_requirements is not None:
        provenance["spool"] = MappingProxyType(
            {
                "disk_budget_bytes": spool_requirements.disk_budget_bytes,
                "retention": spool_requirements.retention,
                "cleanup_required": spool_requirements.cleanup_required,
            }
        )
    return ExecutionPlan(
        required_passes=required_passes,
        spool_requirements=spool_requirements,
        provenance=MappingProxyType(provenance),
        replayability=metadata.replayability,
    )


def _validate_checkpoint(
    metadata: DataSourceMetadata,
    checkpoint: CheckpointMetadata | None,
    required_passes: int,
) -> None:
    if metadata.replayability is not Replayability.CHECKPOINT_REPLAYABLE or required_passes == 1:
        return
    if checkpoint is None:
        raise DataSourceCapabilityError(
            FailureCode.CHECKPOINT_REQUIRED,
            {"required_passes": required_passes, "replayability": metadata.replayability.value},
        )
    if checkpoint.source_id != metadata.source_id:
        raise DataSourceCapabilityError(
            FailureCode.CHECKPOINT_SOURCE_ID_MISMATCH,
            {"stage": "checkpoint_preflight"},
        )
    if checkpoint.schema_version != metadata.checkpoint_schema_version:
        raise DataSourceCapabilityError(
            FailureCode.CHECKPOINT_SCHEMA_MISMATCH,
            {"stage": "checkpoint_preflight"},
        )


def _spool_requirements(
    metadata: DataSourceMetadata,
    required_passes: int,
    spool: SpoolPolicy | None,
) -> SpoolPolicy | None:
    if metadata.replayability is not Replayability.SINGLE_PASS or required_passes == 1:
        return None
    if spool is None or not spool.enabled:
        raise DataSourceCapabilityError(
            FailureCode.SPOOL_REQUIRED,
            {
                "required_passes": required_passes,
                "replayability": metadata.replayability.value,
                "spool_enabled": False,
            },
        )
    return spool


__all__ = [
    "SUPPORTED_PROVENANCE_SCHEMA_VERSIONS",
    "CheckpointMetadata",
    "DataSourceCapabilityError",
    "DataSourceMetadata",
    "ExecutionPlan",
    "Replayability",
    "SpoolPolicy",
    "plan_passes",
]
