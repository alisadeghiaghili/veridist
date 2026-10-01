"""Closed, redacted execution provenance and deterministic JSON encoding."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias, cast

from veridist import __version__
from veridist.engine.data_source import ExecutionPlan, Replayability
from veridist.engine.delivery import AdapterKind, BoundedChunkBuffer, BufferObservation
from veridist.engine.errors import EngineContractError
from veridist.engine.outcome import (
    CompleteOutcome,
    ExecutionOutcome,
    FailedOutcome,
    FailureRecord,
    FailureStage,
    KnownCoverage,
    PartialOutcome,
    RowRange,
    UnknownMissingRanges,
)
from veridist.engine.pass_budget import PassEnforcer, PassObservation
from veridist.engine.resume import PublicResumeMetadata

PROVENANCE_SCHEMA_VERSION = "1"
_PUBLIC_SOURCE_ID = re.compile(r"src_[0-9a-f]{32}")
_RUN_ID = re.compile(r"run_[0-9a-f]{32}")
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_SEED = 2**128 - 1


def _require_token(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _SAFE_TOKEN.fullmatch(value) is None:
        raise ValueError(f"{name} must be an allowlisted token")


def _require_sha256(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 value")


def _require_non_negative_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _require_positive_integer(name: str, value: int) -> None:
    _require_non_negative_integer(name, value)
    if value == 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class PublicSourceId:
    """A caller-supplied opaque pseudonym, never derived from a path or URI."""

    value: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("public source ID must be a string")
        if _PUBLIC_SOURCE_ID.fullmatch(self.value) is None:
            raise ValueError("public source ID must be an opaque src_ pseudonym")


class SourceHashAlgorithm(StrEnum):
    SHA256 = "sha256"


@dataclass(frozen=True, slots=True)
class SourceHash:
    algorithm: SourceHashAlgorithm
    value: str

    def __post_init__(self) -> None:
        if type(self.algorithm) is not SourceHashAlgorithm:
            raise TypeError("source hash algorithm must be typed")
        _require_sha256("source hash", self.value)


class SourceRedactionReason(StrEnum):
    POLICY = "policy"
    USER_REQUEST = "user_request"
    HASH_UNAVAILABLE = "hash_unavailable"


@dataclass(frozen=True, slots=True)
class SourceRedaction:
    reason: SourceRedactionReason

    def __post_init__(self) -> None:
        if type(self.reason) is not SourceRedactionReason:
            raise TypeError("source redaction reason must be typed")


class SourceMutationStatus(StrEnum):
    """Whether a source's identity was found unchanged across one read.

    For the bundled CSV adapter, ``VERIFIED_UNCHANGED`` means the file's
    OS-reported identity -- device, inode, size, and modification time, see
    :meth:`~veridist.adapters.csv_lifetimes._FilesystemCsvSource.identity` --
    matched immediately before and after the read. It is a best-effort
    mutation guard, not a content hash or checksum: an in-place rewrite that
    happens to preserve all four of those values is not detected, and this
    status is never upgraded by re-reading or hashing the bytes themselves.
    """

    NOT_CHECKED = "not_checked"
    VERIFIED_UNCHANGED = "verified_unchanged"
    MISMATCH_DETECTED = "mismatch_detected"
    UNAVAILABLE = "unavailable"


SourceDisclosure: TypeAlias = SourceHash | SourceRedaction


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    public_source_id: PublicSourceId
    schema_version: str
    disclosure: SourceDisclosure
    mutation_status: SourceMutationStatus

    def __post_init__(self) -> None:
        if type(self.public_source_id) is not PublicSourceId:
            raise TypeError("public_source_id must be typed")
        _require_token("source schema version", self.schema_version)
        if type(self.disclosure) not in {SourceHash, SourceRedaction}:
            raise TypeError("source disclosure must be hash or redaction")
        if type(self.mutation_status) is not SourceMutationStatus:
            raise TypeError("source mutation status must be typed")


@dataclass(frozen=True, slots=True)
class AdapterProvenance:
    kind: AdapterKind
    version: str

    def __post_init__(self) -> None:
        if type(self.kind) is not AdapterKind:
            raise TypeError("adapter kind must be typed")
        _require_token("adapter version", self.version)


class SpoolRetention(StrEnum):
    DELETE_ON_CLOSE = "delete_on_close"
    DELETE_ON_SUCCESS = "delete_on_success"
    USER_MANAGED = "user_managed"


class SpoolCleanupStatus(StrEnum):
    COMPLETED = "completed"
    PENDING = "pending"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SpoolNotUsed:
    pass


@dataclass(frozen=True, slots=True)
class SpoolObservation:
    disk_budget_bytes: int
    retention: SpoolRetention
    cleanup_status: SpoolCleanupStatus

    def __post_init__(self) -> None:
        _require_positive_integer("disk_budget_bytes", self.disk_budget_bytes)
        if type(self.retention) is not SpoolRetention:
            raise TypeError("spool retention must be typed")
        if type(self.cleanup_status) is not SpoolCleanupStatus:
            raise TypeError("spool cleanup status must be typed")


SpoolProvenance: TypeAlias = SpoolNotUsed | SpoolObservation


@dataclass(frozen=True, slots=True)
class ExecutionObservation:
    adapter: AdapterProvenance
    engine_version: str
    replayability: Replayability
    required_passes: int
    passes: PassObservation
    buffer: BufferObservation
    spool: SpoolProvenance

    def __post_init__(self) -> None:
        if type(self.adapter) is not AdapterProvenance:
            raise TypeError("adapter provenance must be typed")
        _require_token("engine version", self.engine_version)
        if type(self.replayability) is not Replayability:
            raise TypeError("replayability must be typed")
        _require_positive_integer("required_passes", self.required_passes)
        if type(self.passes) is not PassObservation:
            raise TypeError("pass observation must be typed")
        if self.required_passes > self.passes.max_passes:
            raise ValueError("required passes exceed the enforced pass budget")
        if type(self.buffer) is not BufferObservation:
            raise TypeError("buffer observation must be typed")
        if type(self.spool) not in {SpoolNotUsed, SpoolObservation}:
            raise TypeError("spool provenance must be typed")


@dataclass(frozen=True, slots=True)
class EstimatorProvenance:
    family_id: str
    estimator_id: str
    estimator_version: str
    settings_sha256: str

    def __post_init__(self) -> None:
        _require_token("family ID", self.family_id)
        _require_token("estimator ID", self.estimator_id)
        _require_token("estimator version", self.estimator_version)
        _require_sha256("estimator settings", self.settings_sha256)


class RngPolicy(StrEnum):
    NO_RANDOMNESS = "no_randomness"
    EXPLICIT_SEED = "explicit_seed"
    EXTERNAL_GENERATOR = "external_generator"


@dataclass(frozen=True, slots=True)
class RngProvenance:
    policy: RngPolicy
    algorithm_id: str
    seed: int | None

    def __post_init__(self) -> None:
        if type(self.policy) is not RngPolicy:
            raise TypeError("RNG policy must be typed")
        _require_token("RNG algorithm ID", self.algorithm_id)
        if self.policy is RngPolicy.NO_RANDOMNESS:
            if self.algorithm_id != "none" or self.seed is not None:
                raise ValueError("no-randomness policy requires algorithm none and no seed")
            return
        if self.policy is RngPolicy.EXPLICIT_SEED:
            if self.algorithm_id == "none" or self.seed is None:
                raise ValueError("explicit-seed policy requires an algorithm and seed")
            _require_non_negative_integer("seed", self.seed)
            if self.seed > _MAX_SEED:
                raise ValueError("seed exceeds the public 128-bit contract")
            return
        if self.algorithm_id == "none" or self.seed is not None:
            raise ValueError("external-generator policy requires an algorithm and no seed")


@dataclass(frozen=True, slots=True)
class ExactComputation:
    method_id: str

    def __post_init__(self) -> None:
        _require_token("exact method ID", self.method_id)


@dataclass(frozen=True, slots=True)
class ApproximateComputation:
    method_id: str
    error_contract_id: str

    def __post_init__(self) -> None:
        _require_token("approximation method ID", self.method_id)
        _require_token("approximation error contract ID", self.error_contract_id)


ApproximationProvenance: TypeAlias = ExactComputation | ApproximateComputation


class CheckpointStoreKind(StrEnum):
    IN_MEMORY_TEST_DOUBLE = "in_memory_test_double"
    SQLITE_LOCAL = "sqlite_local"


@dataclass(frozen=True, slots=True)
class CheckpointNotUsed:
    pass


@dataclass(frozen=True, slots=True)
class CheckpointUsed:
    resumed: bool
    checkpoint_schema_version: str
    accumulator_schema_version: str
    initial_generation: int
    final_generation: int
    retry_count: int
    commit_count: int
    store_kind: CheckpointStoreKind
    store_version: str

    def __post_init__(self) -> None:
        if type(self.resumed) is not bool:
            raise TypeError("resumed must be a bool")
        _require_token("checkpoint schema version", self.checkpoint_schema_version)
        _require_token("accumulator schema version", self.accumulator_schema_version)
        _require_non_negative_integer("initial_generation", self.initial_generation)
        _require_non_negative_integer("final_generation", self.final_generation)
        _require_non_negative_integer("retry_count", self.retry_count)
        _require_non_negative_integer("commit_count", self.commit_count)
        if self.final_generation - self.initial_generation != self.commit_count:
            raise ValueError("checkpoint generation delta must equal commit_count")
        if type(self.store_kind) is not CheckpointStoreKind:
            raise TypeError("checkpoint store kind must be typed")
        _require_token("checkpoint store version", self.store_version)


CheckpointProvenance: TypeAlias = CheckpointNotUsed | CheckpointUsed


@dataclass(frozen=True, slots=True)
class ExecutionProvenance:
    """Run metadata excluding outcome-owned coverage, status and failure."""

    schema_version: str
    run_id: str
    source: SourceProvenance
    execution: ExecutionObservation
    estimator: EstimatorProvenance
    rng: RngProvenance
    approximation: ApproximationProvenance
    checkpoint: CheckpointProvenance

    def __post_init__(self) -> None:
        if self.schema_version != PROVENANCE_SCHEMA_VERSION:
            raise ValueError("unsupported provenance schema version")
        if not isinstance(self.run_id, str):
            raise TypeError("run ID must be a string")
        if _RUN_ID.fullmatch(self.run_id) is None:
            raise ValueError("run ID must be an opaque run_ pseudonym")
        if type(self.source) is not SourceProvenance:
            raise TypeError("source provenance must be typed")
        if type(self.execution) is not ExecutionObservation:
            raise TypeError("execution observation must be typed")
        if type(self.estimator) is not EstimatorProvenance:
            raise TypeError("estimator provenance must be typed")
        if type(self.rng) is not RngProvenance:
            raise TypeError("RNG provenance must be typed")
        if type(self.approximation) not in {ExactComputation, ApproximateComputation}:
            raise TypeError("approximation provenance must be typed")
        if type(self.checkpoint) not in {CheckpointNotUsed, CheckpointUsed}:
            raise TypeError("checkpoint provenance must be typed")


@dataclass(frozen=True, slots=True)
class ExecutionReport:
    """One outcome plus metadata; the outcome is the sole coverage truth."""

    outcome: ExecutionOutcome
    provenance: ExecutionProvenance

    def __post_init__(self) -> None:
        if type(self.outcome) not in {CompleteOutcome, PartialOutcome, FailedOutcome}:
            raise TypeError("outcome must be a closed execution outcome")
        if type(self.provenance) is not ExecutionProvenance:
            raise TypeError("provenance must be closed execution metadata")


def snapshot_execution_observation(
    *,
    plan: ExecutionPlan,
    pass_enforcer: PassEnforcer,
    buffer: BoundedChunkBuffer,
    adapter: AdapterProvenance,
    spool: SpoolProvenance,
) -> ExecutionObservation:
    """Snapshot only allowlisted typed and numeric facts from internal engine objects."""

    if type(plan) is not ExecutionPlan:
        raise TypeError("plan must be an ExecutionPlan")
    if type(pass_enforcer) is not PassEnforcer:
        raise TypeError("pass_enforcer must be a PassEnforcer")
    if type(buffer) is not BoundedChunkBuffer:
        raise TypeError("buffer must be a BoundedChunkBuffer")
    return ExecutionObservation(
        adapter=adapter,
        engine_version=__version__,
        replayability=plan.replayability,
        required_passes=plan.required_passes,
        passes=pass_enforcer.observation,
        buffer=buffer.observation,
        spool=spool,
    )


def checkpoint_observation_from_resume(
    resume: PublicResumeMetadata,
    *,
    accumulator_schema_version: str,
    final_generation: int,
    retry_count: int,
    commit_count: int,
    store_version: str,
    store_kind: CheckpointStoreKind = CheckpointStoreKind.IN_MEMORY_TEST_DOUBLE,
) -> CheckpointUsed:
    """Project only safe format/generation facts from internal resume metadata."""

    if type(resume) is not PublicResumeMetadata:
        raise TypeError("resume must be PublicResumeMetadata")
    return CheckpointUsed(
        resumed=True,
        checkpoint_schema_version=str(resume.format_version),
        accumulator_schema_version=accumulator_schema_version,
        initial_generation=resume.generation,
        final_generation=final_generation,
        retry_count=retry_count,
        commit_count=commit_count,
        store_kind=store_kind,
        store_version=store_version,
    )


def failure_record_from_error(
    error: EngineContractError,
    stage: FailureStage,
) -> FailureRecord:
    """Project only the typed code from an error; its context is never traversed."""

    if not isinstance(error, EngineContractError):
        raise TypeError("error must be an EngineContractError")
    if type(stage) is not FailureStage:
        raise TypeError("stage must be a FailureStage")
    return FailureRecord(error.code, stage)


def _range_value(value: RowRange) -> list[int]:
    return [value.start, value.stop]


def _coverage_value(value: KnownCoverage | UnknownMissingRanges) -> dict[str, object]:
    common: dict[str, object] = {
        "accepted_chunk_count": value.accepted_chunk_count,
        "empty_chunk_count": value.empty_chunk_count,
        "processed_ranges": [_range_value(item) for item in value.processed_ranges],
        "processed_row_count": value.processed_row_count,
    }
    if type(value) is UnknownMissingRanges:
        return {
            **common,
            "kind": "unknown_missing_ranges",
            "reason": value.reason.value,
        }
    known = cast(KnownCoverage, value)
    return {
        **common,
        "extent": [known.extent.start, known.extent.stop],
        "kind": "known",
        "missing_ranges": [_range_value(item) for item in known.missing_ranges],
        "missing_row_count": known.missing_row_count,
    }


def _source_value(value: SourceProvenance) -> dict[str, object]:
    disclosure: dict[str, object]
    if type(value.disclosure) is SourceHash:
        disclosure = {
            "algorithm": value.disclosure.algorithm.value,
            "kind": "hash",
            "value": value.disclosure.value,
        }
    else:
        redaction = cast(SourceRedaction, value.disclosure)
        disclosure = {"kind": "redacted", "reason": redaction.reason.value}
    return {
        "disclosure": disclosure,
        "mutation_status": value.mutation_status.value,
        "public_source_id": value.public_source_id.value,
        "schema_version": value.schema_version,
    }


def _spool_value(value: SpoolProvenance) -> dict[str, object]:
    if type(value) is SpoolNotUsed:
        return {"kind": "not_used"}
    used = cast(SpoolObservation, value)
    return {
        "cleanup_status": used.cleanup_status.value,
        "disk_budget_bytes": used.disk_budget_bytes,
        "kind": "used",
        "retention": used.retention.value,
    }


def _execution_value(value: ExecutionObservation) -> dict[str, object]:
    return {
        "adapter": {"kind": value.adapter.kind.value, "version": value.adapter.version},
        "buffer": {
            "backpressure_event_count": value.buffer.backpressure_event_count,
            "chunk_bytes": value.buffer.chunk_bytes,
            "largest_retained_chunk_bytes": value.buffer.largest_retained_chunk_bytes,
            "max_inflight_bytes": value.buffer.max_inflight_bytes,
            "peak_inflight_bytes": value.buffer.peak_inflight_bytes,
        },
        "engine_version": value.engine_version,
        "passes": {
            "actual_pass_count": value.passes.actual_pass_count,
            "max_passes": value.passes.max_passes,
        },
        "replayability": value.replayability.value,
        "required_passes": value.required_passes,
        "spool": _spool_value(value.spool),
    }


def _rng_value(value: RngProvenance) -> dict[str, object]:
    result: dict[str, object] = {
        "algorithm_id": value.algorithm_id,
        "policy": value.policy.value,
    }
    if value.seed is not None:
        result["seed"] = value.seed
    return result


def _approximation_value(value: ApproximationProvenance) -> dict[str, object]:
    if type(value) is ExactComputation:
        return {"kind": "exact", "method_id": value.method_id}
    approximate = cast(ApproximateComputation, value)
    return {
        "error_contract_id": approximate.error_contract_id,
        "kind": "approximate",
        "method_id": approximate.method_id,
    }


def _checkpoint_value(value: CheckpointProvenance) -> dict[str, object]:
    if type(value) is CheckpointNotUsed:
        return {"kind": "not_used"}
    used = cast(CheckpointUsed, value)
    return {
        "accumulator_schema_version": used.accumulator_schema_version,
        "checkpoint_schema_version": used.checkpoint_schema_version,
        "commit_count": used.commit_count,
        "final_generation": used.final_generation,
        "initial_generation": used.initial_generation,
        "kind": "used",
        "resumed": used.resumed,
        "retry_count": used.retry_count,
        "store_kind": used.store_kind.value,
        "store_version": used.store_version,
    }


def to_canonical_json_bytes(report: ExecutionReport) -> bytes:
    """Encode the closed report schema as deterministic compact UTF-8 JSON."""

    if type(report) is not ExecutionReport:
        raise TypeError("report must be an ExecutionReport")
    outcome = report.outcome
    metadata = report.provenance
    value: dict[str, object] = {
        "approximation": _approximation_value(metadata.approximation),
        "checkpoint": _checkpoint_value(metadata.checkpoint),
        "complete": outcome.complete,
        "coverage": _coverage_value(outcome.coverage),
        "estimator": {
            "estimator_id": metadata.estimator.estimator_id,
            "family_id": metadata.estimator.family_id,
            "settings_sha256": metadata.estimator.settings_sha256,
            "version": metadata.estimator.estimator_version,
        },
        "execution": _execution_value(metadata.execution),
        "rng": _rng_value(metadata.rng),
        "run_id": metadata.run_id,
        "schema_version": metadata.schema_version,
        "source": _source_value(metadata.source),
        "status": outcome.status.value,
    }
    if type(outcome) in {PartialOutcome, FailedOutcome}:
        incomplete = cast(PartialOutcome | FailedOutcome, outcome)
        value["failure"] = {
            "code": incomplete.failure.code.value,
            "stage": incomplete.failure.stage.value,
        }
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "AdapterProvenance",
    "ApproximateComputation",
    "ApproximationProvenance",
    "CheckpointNotUsed",
    "CheckpointProvenance",
    "CheckpointStoreKind",
    "CheckpointUsed",
    "EstimatorProvenance",
    "ExactComputation",
    "ExecutionObservation",
    "ExecutionProvenance",
    "ExecutionReport",
    "PublicSourceId",
    "RngPolicy",
    "RngProvenance",
    "SourceDisclosure",
    "SourceHash",
    "SourceHashAlgorithm",
    "SourceMutationStatus",
    "SourceProvenance",
    "SourceRedaction",
    "SourceRedactionReason",
    "SpoolCleanupStatus",
    "SpoolNotUsed",
    "SpoolObservation",
    "SpoolProvenance",
    "SpoolRetention",
    "checkpoint_observation_from_resume",
    "failure_record_from_error",
    "snapshot_execution_observation",
    "to_canonical_json_bytes",
]
