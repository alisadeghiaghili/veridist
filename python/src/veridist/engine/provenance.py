"""Closed, redacted execution provenance and deterministic JSON encoding."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from math import isfinite
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
SCALE_PROVENANCE_SCHEMA_VERSION = "2"
_PUBLIC_SOURCE_ID = re.compile(r"src_[0-9a-f]{32}")
_RUN_ID = re.compile(r"run_[0-9a-f]{32}")
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}")
_FINGERPRINT_LEVELS = frozenset({"none", "metadata", "content"})
_SCALE_PASS_ROLES = frozenset(
    {
        "statistics",
        "grid",
        "expansion",
        "iteration",
        "profile_window",
        "profile_refinement",
        "profile_confirmation",
        "gof",
    }
)
_SCALE_RANGED_ROLES = frozenset(
    {"grid", "expansion", "profile_window", "profile_refinement", "profile_confirmation"}
)
_SCALE_EXECUTORS = frozenset({"serial", "threads", "processes"})
_SCALE_DOWNGRADES = frozenset(
    {
        "single_partition",
        "single_use_source",
        "engine_managed",
        "in_memory_data",
        "undeclared_source",
        "spool_first_pass",
    }
)
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


@dataclass(frozen=True, slots=True)
class ScalePartitionProvenance:
    """One partition of a scale source: its identity, size and fingerprint.

    ``fingerprint`` is ``None`` exactly when ``fingerprint_level`` is ``"none"``;
    such a partition is not resumable. The identifier is whatever the source calls
    the partition and, for a file source, names the file, so it is a locator.
    """

    partition_id: str
    ordinal: int
    rows: int | None
    fingerprint: str | None
    fingerprint_level: str

    def __post_init__(self) -> None:
        if not isinstance(self.partition_id, str) or not self.partition_id:
            raise TypeError("partition ID must be a non-empty string")
        _require_non_negative_integer("ordinal", self.ordinal)
        if self.rows is not None:
            _require_non_negative_integer("rows", self.rows)
        if self.fingerprint_level not in _FINGERPRINT_LEVELS:
            raise ValueError("fingerprint level must be none, metadata or content")
        if self.fingerprint_level == "none":
            if self.fingerprint is not None:
                raise ValueError("a partition without a fingerprint level has no fingerprint")
        elif (
            not isinstance(self.fingerprint, str)
            or _FINGERPRINT.fullmatch(self.fingerprint) is None
        ):
            raise ValueError("fingerprint must be sha256: and 64 lowercase hex digits")


@dataclass(frozen=True, slots=True)
class ScaleStateProvenance:
    """The canonical state a scale fit was computed from: its type and its SHA-256."""

    state_tag: str
    state_schema_version: int
    digest_sha256: str

    def __post_init__(self) -> None:
        _require_token("state tag", self.state_tag)
        _require_positive_integer("state schema version", self.state_schema_version)
        _require_sha256("state digest", self.digest_sha256)


@dataclass(frozen=True, slots=True)
class ScaleEnvironmentProvenance:
    """The numpy version and the digest of its log and exp results on a reference sample."""

    numpy_version: str
    digest_sha256: str

    def __post_init__(self) -> None:
        _require_token("numpy version", self.numpy_version)
        _require_sha256("environment digest", self.digest_sha256)


@dataclass(frozen=True, slots=True)
class ScalePassRecord:
    """One read of the source by a multi-pass fit or by inference from a fit.

    ``role`` is ``"statistics"`` (the first pass: the count, the exact sums and the
    extremes), ``"grid"`` (the initial grid of candidate parameter values),
    ``"expansion"`` (a further grid outward) or ``"iteration"`` (one Newton step, whose
    last one also carries the report). The passes of inference from a fit are
    ``"profile_window"`` (the nodes of the window of a profile likelihood),
    ``"profile_refinement"`` (further nodes of the same window),
    ``"profile_confirmation"`` (exact sums at the points the intervals are reported at) and
    ``"gof"`` (the counts of the goodness-of-fit pass, one parameter vector).
    ``points`` is the number of candidate parameter values the pass evaluated (zero for
    ``"statistics"``). ``log_low`` and ``log_high`` are the smallest and largest natural
    logarithm of a candidate of a pass over a range of candidates, and ``None`` for the
    other roles.
    """

    role: str
    points: int
    log_low: float | None
    log_high: float | None

    def __post_init__(self) -> None:
        if self.role not in _SCALE_PASS_ROLES:
            raise ValueError("pass role is not one of the scale pass roles")
        _require_non_negative_integer("points", self.points)
        for name, value in (("log_low", self.log_low), ("log_high", self.log_high)):
            if value is not None and (type(value) is not float or not isfinite(value)):
                raise TypeError(f"{name} must be a finite float or None")
        if (self.log_low is None) != (self.log_high is None):
            raise ValueError("log_low and log_high are given together or not at all")
        if self.log_low is not None and self.log_high is not None and self.log_low > self.log_high:
            raise ValueError("log_low must not exceed log_high")
        if (self.role in _SCALE_RANGED_ROLES) != (self.log_low is not None):
            raise ValueError("only a pass over a range of candidates has a logarithmic range")
        if (self.role == "statistics") != (self.points == 0):
            raise ValueError("only the statistics pass has no candidate points")


@dataclass(frozen=True, slots=True)
class ScalePassProvenance:
    """The passes a multi-pass fit made over its source.

    ``pass_budget`` is the caller's budget, ``minimum_passes`` and ``maximum_passes`` the
    bounds the plan published before the first read, ``pass_count`` the number of reads
    the fit made and ``records`` one record per read in order. The record holds no data
    value.
    """

    pass_budget: int
    minimum_passes: int
    maximum_passes: int
    pass_count: int
    records: tuple[ScalePassRecord, ...]

    def __post_init__(self) -> None:
        _require_positive_integer("pass_budget", self.pass_budget)
        _require_positive_integer("minimum_passes", self.minimum_passes)
        _require_positive_integer("maximum_passes", self.maximum_passes)
        _require_non_negative_integer("pass_count", self.pass_count)
        if self.minimum_passes > self.maximum_passes:
            raise ValueError("minimum_passes must not exceed maximum_passes")
        if self.pass_budget < self.minimum_passes:
            raise ValueError("pass_budget must not be below minimum_passes")
        if type(self.records) is not tuple or any(
            type(item) is not ScalePassRecord for item in self.records
        ):
            raise TypeError("records must be a tuple of pass records")
        if self.pass_count != len(self.records):
            raise ValueError("pass_count must equal the number of records")
        if self.pass_count > min(self.pass_budget, self.maximum_passes):
            raise ValueError("pass_count must not exceed the budget or the plan maximum")


@dataclass(frozen=True, slots=True)
class ScaleSpoolPartitionProvenance:
    """What a local spool holds of one partition: its row count and content digest.

    ``ordinal`` is the partition's place in the source's canonical order (the same
    ordinal as its :class:`ScalePartitionProvenance`), ``rows`` the number of values
    written and ``digest_sha256`` the SHA-256 of those values as little-endian binary64,
    in the order they were written. The record holds no path and no data value.
    """

    ordinal: int
    rows: int
    digest_sha256: str

    def __post_init__(self) -> None:
        _require_non_negative_integer("ordinal", self.ordinal)
        _require_non_negative_integer("rows", self.rows)
        _require_sha256("spool digest", self.digest_sha256)


@dataclass(frozen=True, slots=True)
class ScaleSpoolProvenance:
    """A fit used a local spool: the rows it holds and the digest of each partition.

    ``row_count`` is the total of the partition rows and ``partitions`` has one record
    per partition in canonical order. Every read of the spool was checked against these
    digests.
    """

    row_count: int
    partitions: tuple[ScaleSpoolPartitionProvenance, ...]

    def __post_init__(self) -> None:
        _require_non_negative_integer("row_count", self.row_count)
        if type(self.partitions) is not tuple or any(
            type(item) is not ScaleSpoolPartitionProvenance for item in self.partitions
        ):
            raise TypeError("partitions must be a tuple of spool partition provenance")
        if self.row_count != sum(item.rows for item in self.partitions):
            raise ValueError("row_count must equal the total of the partition rows")


@dataclass(frozen=True, slots=True)
class ScaleExecutionPassProvenance:
    """How one data pass of a scale fit was executed.

    ``executor`` is what read the partitions (``"serial"``, ``"threads"`` or
    ``"processes"``) and ``workers`` how many threads or processes did (1 when serial).
    ``tasks`` is the number of partitions and ``retries`` how many times a partition was read
    again after a transient error. ``downgrade`` is the reason the pass was read serially
    though the fit asked for threads or processes, and ``None`` when it was not.
    """

    executor: str
    workers: int
    tasks: int
    retries: int
    downgrade: str | None

    def __post_init__(self) -> None:
        if self.executor not in _SCALE_EXECUTORS:
            raise ValueError("executor must be serial, threads or processes")
        _require_positive_integer("workers", self.workers)
        _require_non_negative_integer("tasks", self.tasks)
        _require_non_negative_integer("retries", self.retries)
        if self.downgrade is not None and self.downgrade not in _SCALE_DOWNGRADES:
            raise ValueError("downgrade must be a known reason or None")
        if self.executor == "serial" and self.workers != 1:
            raise ValueError("a serial pass has one worker")
        if self.downgrade is not None and self.executor != "serial":
            raise ValueError("only a serial pass was downgraded")


@dataclass(frozen=True, slots=True)
class ScaleExecutionProvenance:
    """The executor a scale fit was given and what each of its data passes did with it.

    ``executor`` and ``requested_workers`` are the request (``"serial"`` and 1 for the serial
    executor). ``passes`` has one record per data pass in order; for a multi-pass fit they
    are the passes of ``ScaleProvenance.passes`` and for a one-pass fit there is one.
    ``effective_workers`` is the most workers any pass used, ``retries`` the re-reads of all
    passes and ``downgrade`` the reason of the first downgraded pass, if any. A request is
    capped by the cores of the host and by the number of partitions, so the effective count
    can be lower without a downgrade. The record holds no data value and no path.
    """

    executor: str
    requested_workers: int
    passes: tuple[ScaleExecutionPassProvenance, ...]

    def __post_init__(self) -> None:
        if self.executor not in _SCALE_EXECUTORS:
            raise ValueError("executor must be serial, threads or processes")
        _require_positive_integer("requested_workers", self.requested_workers)
        if self.executor == "serial" and self.requested_workers != 1:
            raise ValueError("the serial executor has one worker")
        if type(self.passes) is not tuple or any(
            type(item) is not ScaleExecutionPassProvenance for item in self.passes
        ):
            raise TypeError("passes must be a tuple of execution pass provenance")
        for item in self.passes:
            if item.workers > self.requested_workers:
                raise ValueError("a pass cannot use more workers than were requested")
            if item.downgrade is None and item.executor != self.executor:
                raise ValueError("a pass that was not downgraded used the requested executor")
            if self.executor == "serial" and item.downgrade is not None:
                raise ValueError("the serial executor is not downgraded")

    @property
    def effective_workers(self) -> int:
        return max((item.workers for item in self.passes), default=1)

    @property
    def retries(self) -> int:
        return sum(item.retries for item in self.passes)

    @property
    def downgrade(self) -> str | None:
        return next((item.downgrade for item in self.passes if item.downgrade is not None), None)


@dataclass(frozen=True, slots=True)
class ScaleCheckpointPassProvenance:
    """What one data pass of a checkpointed fit did with the partition-state store.

    ``partitions`` is the number of partitions of the pass and ``pass_index`` its place in
    the fit. ``replayed`` says the whole pass was taken from the pass ledger: nothing was
    read, and ``reused`` is then every partition. Otherwise ``reused`` partitions had a
    committed state that was used without reading them and ``read`` partitions were read
    and committed in this run (fewer than the rest when the run ended with a failure).
    """

    pass_index: int
    partitions: int
    replayed: bool
    reused: int
    read: int

    def __post_init__(self) -> None:
        _require_non_negative_integer("pass_index", self.pass_index)
        _require_non_negative_integer("partitions", self.partitions)
        _require_non_negative_integer("reused", self.reused)
        _require_non_negative_integer("read", self.read)
        if type(self.replayed) is not bool:
            raise TypeError("replayed must be a bool")
        if self.reused + self.read > self.partitions:
            raise ValueError("reused and read partitions cannot exceed the partitions")
        if self.replayed and (self.read != 0 or self.reused != self.partitions):
            raise ValueError("a replayed pass reuses every partition and reads none")


@dataclass(frozen=True, slots=True)
class ScaleCheckpointProvenance:
    """A scale fit used a partition-state store: the job, and what it reused and read.

    ``store_kind`` and ``store_format_version`` name the store and ``job_digest_sha256`` is
    the digest of the identity of the job. ``resumed`` says the store already held the job.
    ``incremental`` repeats the request; ``added_partitions`` and ``removed_partitions``
    count the partitions that joined or left the source since the job was started (accepted
    by an incremental fit), ``recomputed_partitions`` the partitions that had changed and
    were read again, and ``restarted`` says the stored passes of a multi-pass fit were
    discarded because of that. ``passes`` has one record per data pass in order. The record
    holds no path, partition identifier, fingerprint or state.
    """

    store_kind: str
    store_format_version: int
    job_digest_sha256: str
    resumed: bool
    incremental: bool
    added_partitions: int
    removed_partitions: int
    recomputed_partitions: int
    restarted: bool
    passes: tuple[ScaleCheckpointPassProvenance, ...]

    def __post_init__(self) -> None:
        _require_token("store kind", self.store_kind)
        _require_positive_integer("store_format_version", self.store_format_version)
        _require_sha256("job digest", self.job_digest_sha256)
        for name in ("resumed", "incremental", "restarted"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        _require_non_negative_integer("added_partitions", self.added_partitions)
        _require_non_negative_integer("removed_partitions", self.removed_partitions)
        _require_non_negative_integer("recomputed_partitions", self.recomputed_partitions)
        if type(self.passes) is not tuple or any(
            type(item) is not ScaleCheckpointPassProvenance for item in self.passes
        ):
            raise TypeError("passes must be a tuple of checkpoint pass provenance")
        if tuple(item.pass_index for item in self.passes) != tuple(range(len(self.passes))):
            raise ValueError("pass records are numbered from zero in order")
        if not self.resumed and (
            self.added_partitions
            or self.removed_partitions
            or self.recomputed_partitions
            or self.restarted
            or any(item.replayed or item.reused for item in self.passes)
        ):
            raise ValueError("a job that was started has nothing to reuse or amend")
        if (self.added_partitions or self.removed_partitions) and not self.incremental:
            raise ValueError("partitions are added or removed by an incremental fit only")
        if self.restarted and not self.recomputed_partitions:
            raise ValueError("a restart follows from a recomputed partition")

    @property
    def reused_partitions(self) -> int:
        """The partitions whose committed state was used, over all passes."""

        return sum(item.reused for item in self.passes)

    @property
    def read_partitions(self) -> int:
        """The partitions read and committed in this run, over all passes."""

        return sum(item.read for item in self.passes)

    @property
    def replayed_passes(self) -> int:
        """The passes taken from the pass ledger."""

        return sum(item.replayed for item in self.passes)


@dataclass(frozen=True, slots=True)
class ScaleProvenance:
    """Closed facts about a fit computed from a scale source (provenance schema 2).

    ``partitions`` is the source's partition list in canonical order and
    ``partition_count`` its length. ``row_count`` is the number of rows the fit
    read, ``max_rows`` the cap on the rows of one batch, ``state`` the state the
    estimate was derived from (``None`` only when no state exists because the data
    were refused) and ``environment`` the numeric environment the logarithms were
    evaluated in. ``passes`` is the pass record of a multi-pass fit and ``None`` for a
    fit that reads its source once. ``spool`` is the record of the local spool a fit
    used and ``None`` for a fit that used none; its partitions are those of
    ``partitions``. ``execution`` is the record of the executor a fit was given and ``None``
    for a fit that was given none. The record holds no data value and no path.
    """

    schema_version: str
    source_kind: str
    partitions: tuple[ScalePartitionProvenance, ...]
    partition_count: int
    row_count: int
    max_rows: int
    state: ScaleStateProvenance | None
    environment: ScaleEnvironmentProvenance
    passes: ScalePassProvenance | None = None
    spool: ScaleSpoolProvenance | None = None
    execution: ScaleExecutionProvenance | None = None
    checkpoint: ScaleCheckpointProvenance | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCALE_PROVENANCE_SCHEMA_VERSION:
            raise ValueError("unsupported scale provenance schema version")
        _require_token("source kind", self.source_kind)
        if type(self.partitions) is not tuple or any(
            type(item) is not ScalePartitionProvenance for item in self.partitions
        ):
            raise TypeError("partitions must be a tuple of partition provenance")
        _require_non_negative_integer("partition_count", self.partition_count)
        if self.partition_count != len(self.partitions):
            raise ValueError("partition_count must equal the number of partitions")
        if len({item.partition_id for item in self.partitions}) != len(self.partitions):
            raise ValueError("partition IDs must be unique")
        _require_non_negative_integer("row_count", self.row_count)
        _require_positive_integer("max_rows", self.max_rows)
        if self.state is not None and type(self.state) is not ScaleStateProvenance:
            raise TypeError("state provenance must be typed or None")
        if type(self.environment) is not ScaleEnvironmentProvenance:
            raise TypeError("environment provenance must be typed")
        if self.passes is not None and type(self.passes) is not ScalePassProvenance:
            raise TypeError("pass provenance must be typed or None")
        if self.spool is not None:
            if type(self.spool) is not ScaleSpoolProvenance:
                raise TypeError("spool provenance must be typed or None")
            spooled = tuple(item.ordinal for item in self.spool.partitions)
            if self.checkpoint is None:
                if spooled != tuple(item.ordinal for item in self.partitions):
                    raise ValueError("a spool holds exactly the partitions of the source")
                if self.spool.row_count != self.row_count:
                    raise ValueError("the spool must hold exactly the rows the fit read")
            elif len(set(spooled)) != len(spooled) or not set(spooled) <= {
                item.ordinal for item in self.partitions
            }:
                raise ValueError("a spool holds partitions of the source, each once")
        if self.execution is not None:
            if type(self.execution) is not ScaleExecutionProvenance:
                raise TypeError("execution provenance must be typed or None")
            expected = 1 if self.passes is None else self.passes.pass_count
            if len(self.execution.passes) != expected:
                raise ValueError("an execution record has one entry per data pass")
        if self.checkpoint is not None:
            if type(self.checkpoint) is not ScaleCheckpointProvenance:
                raise TypeError("checkpoint provenance must be typed or None")
            expected = 1 if self.passes is None else self.passes.pass_count
            if len(self.checkpoint.passes) != expected:
                raise ValueError("a checkpoint record has one entry per data pass")
            if any(item.partitions > self.partition_count for item in self.checkpoint.passes):
                raise ValueError("a pass cannot cover more partitions than the source has")


def scale_provenance_to_canonical_json_bytes(record: ScaleProvenance) -> bytes:
    """Encode a schema 2 scale record as deterministic compact UTF-8 JSON."""

    if type(record) is not ScaleProvenance:
        raise TypeError("record must be a ScaleProvenance")
    state: dict[str, object] | None = None
    if record.state is not None:
        state = {
            "digest_sha256": record.state.digest_sha256,
            "state_schema_version": record.state.state_schema_version,
            "state_tag": record.state.state_tag,
        }
    value: dict[str, object] = {
        "environment": {
            "digest_sha256": record.environment.digest_sha256,
            "numpy_version": record.environment.numpy_version,
        },
        "max_rows": record.max_rows,
        "partition_count": record.partition_count,
        "partitions": [
            {
                "fingerprint": item.fingerprint,
                "fingerprint_level": item.fingerprint_level,
                "ordinal": item.ordinal,
                "partition_id": item.partition_id,
                "rows": item.rows,
            }
            for item in record.partitions
        ],
        "row_count": record.row_count,
        "schema_version": record.schema_version,
        "source_kind": record.source_kind,
        "state": state,
    }
    if record.passes is not None:
        value["passes"] = {
            "maximum_passes": record.passes.maximum_passes,
            "minimum_passes": record.passes.minimum_passes,
            "pass_budget": record.passes.pass_budget,
            "pass_count": record.passes.pass_count,
            "records": [
                {
                    "log_high": item.log_high,
                    "log_low": item.log_low,
                    "points": item.points,
                    "role": item.role,
                }
                for item in record.passes.records
            ],
        }
    if record.spool is not None:
        value["spool"] = {
            "partitions": [
                {
                    "digest_sha256": item.digest_sha256,
                    "ordinal": item.ordinal,
                    "rows": item.rows,
                }
                for item in record.spool.partitions
            ],
            "row_count": record.spool.row_count,
        }
    if record.checkpoint is not None:
        value["checkpoint"] = {
            "added_partitions": record.checkpoint.added_partitions,
            "incremental": record.checkpoint.incremental,
            "job_digest_sha256": record.checkpoint.job_digest_sha256,
            "passes": [
                {
                    "partitions": item.partitions,
                    "pass_index": item.pass_index,
                    "read": item.read,
                    "replayed": item.replayed,
                    "reused": item.reused,
                }
                for item in record.checkpoint.passes
            ],
            "recomputed_partitions": record.checkpoint.recomputed_partitions,
            "removed_partitions": record.checkpoint.removed_partitions,
            "restarted": record.checkpoint.restarted,
            "resumed": record.checkpoint.resumed,
            "store_format_version": record.checkpoint.store_format_version,
            "store_kind": record.checkpoint.store_kind,
        }
    if record.execution is not None:
        value["execution"] = {
            "downgrade": record.execution.downgrade,
            "effective_workers": record.execution.effective_workers,
            "executor": record.execution.executor,
            "passes": [
                {
                    "downgrade": item.downgrade,
                    "executor": item.executor,
                    "retries": item.retries,
                    "tasks": item.tasks,
                    "workers": item.workers,
                }
                for item in record.execution.passes
            ],
            "requested_workers": record.execution.requested_workers,
            "retries": record.execution.retries,
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
    "SCALE_PROVENANCE_SCHEMA_VERSION",
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
    "ScaleCheckpointPassProvenance",
    "ScaleCheckpointProvenance",
    "ScaleEnvironmentProvenance",
    "ScaleExecutionPassProvenance",
    "ScaleExecutionProvenance",
    "ScalePartitionProvenance",
    "ScalePassProvenance",
    "ScalePassRecord",
    "ScaleProvenance",
    "ScaleSpoolPartitionProvenance",
    "ScaleSpoolProvenance",
    "ScaleStateProvenance",
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
    "scale_provenance_to_canonical_json_bytes",
    "snapshot_execution_observation",
    "to_canonical_json_bytes",
]
