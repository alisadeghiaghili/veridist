"""One-pass source-to-family execution orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import warnings
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from veridist.adapters.csv_lifetimes import (
    CsvLifetimeAdapter,
    CsvLifetimeAdapterError,
    CsvLifetimeLimits,
    CsvLifetimeSchema,
)
from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation
from veridist.engine.checkpoint import CheckpointRecord, CheckpointStore, SQLiteCheckpointStore
from veridist.engine.data_source import DataSourceLike, ExecutionPlan, SpoolPolicy, plan_passes
from veridist.engine.delivery import (
    AdapterKind,
    BoundedChunkBuffer,
    BufferedChunk,
    DeliveryValidator,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.outcome import (
    CompleteOutcome,
    FailureStage,
    KnownCoverage,
    KnownExtent,
    RowRange,
    UnknownMissingRanges,
    classify_execution_outcome,
)
from veridist.engine.pass_budget import PassEnforcer
from veridist.engine.provenance import (
    AdapterProvenance,
    CheckpointNotUsed,
    EstimatorProvenance,
    ExactComputation,
    ExecutionProvenance,
    ExecutionReport,
    PublicSourceId,
    RngPolicy,
    RngProvenance,
    SourceMutationStatus,
    SourceProvenance,
    SourceRedaction,
    SourceRedactionReason,
    SpoolNotUsed,
    failure_record_from_error,
    snapshot_execution_observation,
)
from veridist.engine.resume import source_id_mismatch, source_schema_mismatch
from veridist.engine.retry import _operation_digest, apply_pure_update
from veridist.engine.streaming import iter_stream
from veridist.families.exponential import (
    ExponentialFit,
    fit_exponential_chunks,
    fit_exponential_reduction_state,
)
from veridist.statistics.exponential import ExponentialCheckpointReducer, ExponentialReductionState

#: Source schema recorded for every checkpointed CSV reduction. Also the
#: literal value `tools/collect_v1_execution_evidence.py` uses for its own
#: hand-built store, so the two stay contractually aligned.
CHECKPOINTED_CSV_SOURCE_SCHEMA = "csv-lifetime-v1"

#: Fixed, documented plan digest for stores built by
#: `create_checkpointed_csv_store`. It identifies "the one checkpointed CSV
#: exponential reduction plan" this module implements; it is not a hash of
#: caller-supplied data and never needs to vary between stores.
CHECKPOINTED_CSV_PLAN_DIGEST = "checkpointed-csv-exponential-v1"

_FILE_HASH_BLOCK_BYTES = 1024 * 1024

_LEGACY_CHECKPOINTED_CHUNK_WARNING = (
    "passing bare bytes to fit_exponential_checkpointed_chunks is deprecated; "
    "pass (row_start, payload) tuples so a replayed chunk can be detected"
)


def _hash_file_sha256(path: Path) -> str:
    """Stream-hash a file with SHA-256 in bounded blocks; never load it whole."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(_FILE_HASH_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ExponentialSourceFitResult:
    """Closed result shape for a completed fit or a failed execution."""

    fit: ExponentialFit | None
    execution: ExecutionReport

    def __post_init__(self) -> None:
        if type(self.execution) is not ExecutionReport:
            raise TypeError("execution must be ExecutionReport")
        if (self.fit is not None) is not isinstance(self.execution.outcome, CompleteOutcome):
            raise ValueError("fit presence must match complete execution")


@dataclass(frozen=True, slots=True)
class CheckpointedCsvFitResult:
    """Result of a resumable CSV reduction without exposing checkpoint state."""

    code: str
    fit: ExponentialFit | None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("code must be non-empty")
        if self.code == "COMPLETE" and self.fit is None:
            raise ValueError("a complete reduction must contain a fit result")
        if self.code != "COMPLETE" and self.fit is not None:
            raise ValueError("an incomplete reduction cannot contain a fit result")


def fit_exponential_source(adapter: object) -> ExponentialSourceFitResult:
    """Fit the exponential vertical through exactly one sequential source pass."""

    if type(adapter) is not CsvLifetimeAdapter:
        raise TypeError("adapter must be CsvLifetimeAdapter")
    # CsvLifetimeAdapter exposes an immutable metadata property; the planning
    # protocol's legacy writable attribute is semantically narrower.
    plan = plan_passes(
        cast(DataSourceLike, adapter), required_passes=1, spool=SpoolPolicy.disabled()
    )
    buffer = BoundedChunkBuffer(
        chunk_bytes=adapter.limits.chunk_bytes,
        max_inflight_bytes=adapter.limits.max_inflight_bytes,
    )
    validator = DeliveryValidator(adapter.source_id.value)
    stage = FailureStage.PREFLIGHT
    error: EngineContractError | None = None
    fit: ExponentialFit | None = None
    coverage: KnownCoverage | UnknownMissingRanges
    expected_chunk_count = 0
    iterator: object | None = None

    def delivered_payloads() -> Generator[tuple[LifetimeObservation, ...], None, None]:
        """Deliver one validated payload at a time and release it before advancing."""

        nonlocal stage, expected_chunk_count, iterator
        acquired = iter_stream(adapter)
        iterator = acquired
        try:
            for chunk in acquired:
                stage = FailureStage.DELIVERY
                validator.accept(chunk.envelope)
                expected_chunk_count += 1
                lease = BufferedChunk(envelope=chunk.envelope, payload=chunk.observations)
                buffer.put(lease)
                received = buffer.get()
                try:
                    payload = received.payload
                    if type(payload) is not tuple:
                        raise TypeError("CSV chunk payload must be a tuple")
                    yield cast(tuple[LifetimeObservation, ...], payload)
                finally:
                    received.release()
            stage = FailureStage.FINALIZATION
            # The adapter tracks how many records it actually parsed through
            # EOF on its own, independent of what made it through delivery, so
            # this expectation is not simply re-derived from the same
            # envelopes `validator` already accepted; a chunk silently lost
            # between parsing and delivery now makes `finish` fail. A source
            # that does not offer this fact (for example a test double that
            # replaces `iter_chunks` entirely) falls back to the validator's
            # own tally, which cannot be independently wrong.
            terminal_record_count = adapter.terminal_record_count
            validator.finish(
                expected_row_stop=(
                    terminal_record_count
                    if terminal_record_count is not None
                    else validator.next_offset
                ),
                expected_chunk_count=expected_chunk_count,
            )
        finally:
            close = getattr(acquired, "close", None)
            if callable(close):
                close()
            # This is idempotent; it drains/reclaims a lease if a consumer,
            # validator, or reducer fails between put and get.
            buffer.cancel()

    try:
        fit = fit_exponential_chunks(delivered_payloads())
    except EngineContractError as captured:
        error = captured
        if type(captured) is CsvLifetimeAdapterError:
            stage = FailureStage(captured.phase.value)
    finally:
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if callable(close):
                close()
        buffer.cancel()

    if error is None:
        extent = KnownExtent(0, validator.next_offset)
        coverage = KnownCoverage(
            extent,
            (RowRange(0, validator.next_offset),) if validator.next_offset else (),
            validator.accepted_chunks,
            0,
        )
        outcome = classify_execution_outcome(coverage, None)
        assert fit is not None
        mutation = SourceMutationStatus.VERIFIED_UNCHANGED
    else:
        coverage = UnknownMissingRanges(
            (RowRange(0, validator.next_offset),) if validator.next_offset else (),
            validator.accepted_chunks,
            0,
        )
        outcome = classify_execution_outcome(coverage, failure_record_from_error(error, stage))
        fit = None
        mutation = (
            error.mutation_status
            if type(error) is CsvLifetimeAdapterError
            else SourceMutationStatus.NOT_CHECKED
        )
    execution = ExecutionReport(
        outcome,
        _provenance(adapter, plan, adapter.passes, buffer, mutation),
    )
    return ExponentialSourceFitResult(fit, execution)


def fit_exponential_csv(
    path: Path,
    *,
    schema: CsvLifetimeSchema,
    source_id: PublicSourceId,
    limits: CsvLifetimeLimits,
) -> ExponentialSourceFitResult:
    """Construct the strict CSV adapter and execute its single allowed pass."""

    return fit_exponential_source(CsvLifetimeAdapter(path, schema, source_id, limits))


def fit_exponential_checkpointed_chunks(
    *,
    store: CheckpointStore,
    source_revision: str,
    chunks: object,
) -> ExponentialFit:
    """Reduce canonical JSON lifetime chunks through a durable checkpoint store.

    The caller owns acquisition and must provide a store initialized for the
    same source/reducer contract. Only one chunk is decoded at a time; the
    checkpoint contains sufficient statistics, never raw input rows.

    Each chunk is either the offset form `(row_start, payload)` or, for
    compatibility, bare `bytes`. Prefer the offset form: because its
    `row_start` is caller-supplied rather than read from the live cursor, a
    replay of an already-committed chunk is skipped as a safe no-op instead
    of being double-counted, and a gap or partial overlap against the
    checkpoint's cursor is rejected as `RANGE_MISMATCH`. The bare-bytes form
    always derives `row_start` from the current cursor, so it cannot
    recognize a replay in general; it emits `DeprecationWarning` once per
    call and only ever recognizes one specific case -- the first chunk of
    this call re-sending the single most recently committed chunk (the
    common shape of a retry after a crash) -- by comparing against the
    checkpoint's own recorded operation digest.
    """

    if not isinstance(chunks, Iterable):
        raise TypeError("chunks must be an iterable")
    reducer = ExponentialCheckpointReducer()
    legacy_warned = False
    for index, item in enumerate(chunks):
        legacy: bool
        if type(item) is tuple:
            if (
                len(item) != 2
                or isinstance(item[0], bool)
                or not isinstance(item[0], int)
                or not isinstance(item[1], bytes)
            ):
                raise TypeError(
                    "offset-form checkpointed chunks must be (row_start, payload)"
                )
            row_start, payload = cast(tuple[int, bytes], item)
            legacy = False
        elif isinstance(item, bytes):
            payload = item
            row_start = None
            legacy = True
        else:
            raise TypeError("checkpointed chunks must be bytes or (row_start, payload)")

        rows = json.loads(payload.decode("utf-8"))
        if not isinstance(rows, list):
            raise ValueError("checkpointed chunk must be a JSON array")

        base = store.read()
        cursor = base.cursor
        payload_sha256 = hashlib.sha256(payload).hexdigest()

        if legacy:
            if not legacy_warned:
                warnings.warn(
                    _LEGACY_CHECKPOINTED_CHUNK_WARNING, DeprecationWarning, stacklevel=2
                )
                legacy_warned = True
            if index == 0 and len(rows) <= cursor:
                candidate_start = cursor - len(rows)
                candidate_stop = cursor
                candidate_token = f"chunk-{candidate_start}-{candidate_stop}"
                candidate_digest = _operation_digest(
                    base,
                    payload_sha256=payload_sha256,
                    row_start=candidate_start,
                    row_stop=candidate_stop,
                    operation_token=candidate_token,
                )
                if candidate_digest == base.operation_digest:
                    continue
            row_start = cursor
            row_stop = row_start + len(rows)
        else:
            assert row_start is not None
            row_stop = row_start + len(rows)
            if row_stop <= cursor:
                continue
            if row_start != cursor:
                raise EngineContractError(
                    FailureCode.RANGE_MISMATCH,
                    {
                        "checkpoint_cursor": cursor,
                        "row_start": row_start,
                        "row_stop": row_stop,
                    },
                )

        apply_pure_update(
            store=store,
            source_revision=source_revision,
            payload=payload,
            payload_sha256=payload_sha256,
            row_start=row_start,
            row_stop=row_stop,
            operation_token=f"chunk-{row_start}-{row_stop}",
            reducer=reducer,
        )
    return fit_exponential_reduction_state(reducer.decode_state(store.read().state))


def fit_exponential_checkpointed_csv(
    *,
    path: Path,
    schema: CsvLifetimeSchema,
    source_id: PublicSourceId,
    limits: CsvLifetimeLimits,
    store: CheckpointStore,
    source_revision: str,
    cancel: Callable[[int], bool] | None,
) -> CheckpointedCsvFitResult:
    """Resume a strict CSV lifetime reduction from its committed row cursor.

    Each adapter chunk, or its prefix before cancellation, is one pure
    reducer/CAS transition. The source is read once for this attempt, but rows
    in the committed prefix are only decoded to reach the cursor; they are
    never submitted to the reducer again. Cancellation is observed before
    each row is added to a batch, and any preceding batch prefix is committed
    before the cancelled result is returned.

    `source_revision` is a contract, not a free-form label: it must equal
    the file's current SHA-256 digest (lowercase hex), stream-hashed in 1 MiB
    blocks before any reducer call. A caller that reuses an old revision
    string against a changed file, or a correct-looking revision that still
    does not match the checkpoint's own recorded revision, gets
    `SOURCE_REVISION_MISMATCH` rather than a silently mixed result. The
    checkpoint's `source_id` and `source_schema` must likewise match
    `source_id` and `CHECKPOINTED_CSV_SOURCE_SCHEMA`, or the call returns
    `SOURCE_ID_MISMATCH` / `SOURCE_SCHEMA_MISMATCH` before touching the file.
    `create_checkpointed_csv_store` builds a store that satisfies all of
    this automatically.

    Known limitation (TOCTOU): the file can still change between this hash
    and the adapter's later parse of it. The adapter's own stat-identity
    check covers that narrower parse-time window; this function does not
    attempt to close it further.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    if type(schema) is not CsvLifetimeSchema:
        raise TypeError("schema must be CsvLifetimeSchema")
    if type(source_id) is not PublicSourceId:
        raise TypeError("source_id must be PublicSourceId")
    if type(limits) is not CsvLifetimeLimits:
        raise TypeError("limits must be CsvLifetimeLimits")
    if cancel is not None and not callable(cancel):
        raise TypeError("cancel must be callable or None")

    reducer = ExponentialCheckpointReducer()
    try:
        # This check is deliberately before source acquisition and every
        # reducer call.  `apply_pure_update` repeats it at the CAS boundary.
        checkpoint = store.read()
        if not checkpoint.has_valid_checksum():
            return CheckpointedCsvFitResult("CHECKPOINT_CHECKSUM_MISMATCH", None)
        if source_id_mismatch(checkpoint, source_id.value):
            return CheckpointedCsvFitResult("SOURCE_ID_MISMATCH", None)
        if source_schema_mismatch(checkpoint, CHECKPOINTED_CSV_SOURCE_SCHEMA):
            return CheckpointedCsvFitResult("SOURCE_SCHEMA_MISMATCH", None)
        if checkpoint.reducer_id != reducer.reducer_id:
            return CheckpointedCsvFitResult("REDUCER_MISMATCH", None)
        if checkpoint.accumulator_schema != reducer.accumulator_schema:
            return CheckpointedCsvFitResult("ACCUMULATOR_SCHEMA_MISMATCH", None)
        try:
            file_revision = _hash_file_sha256(path)
        except OSError:
            return CheckpointedCsvFitResult("SOURCE_OPEN_FAILED", None)
        if source_revision != file_revision:
            return CheckpointedCsvFitResult("SOURCE_REVISION_MISMATCH", None)
        if checkpoint.source_revision != source_revision:
            return CheckpointedCsvFitResult("SOURCE_REVISION_MISMATCH", None)

        # The strict adapter's public logical-payload accounting has a fixed
        # object-graph overhead. Retain and checkpoint one bounded adapter
        # chunk at a time without materializing the source.
        adapter_limit = max(2048, limits.chunk_bytes)
        adapter = CsvLifetimeAdapter(
            path,
            schema,
            source_id,
            CsvLifetimeLimits(adapter_limit, max(adapter_limit, limits.max_inflight_bytes)),
        )
        for chunk in adapter.iter_chunks():
            checkpoint = store.read()
            if checkpoint.source_revision != source_revision:
                return CheckpointedCsvFitResult("SOURCE_REVISION_MISMATCH", None)
            row_start = chunk.envelope.row_start
            row_stop = chunk.envelope.row_stop
            if checkpoint.cursor >= row_stop:
                continue
            if checkpoint.cursor < row_start:
                return CheckpointedCsvFitResult("RANGE_MISMATCH", None)
            batch_start = checkpoint.cursor
            batch: list[list[float | bool]] = []
            observations = chunk.observations[batch_start - row_start :]
            for observation in observations:
                cursor = batch_start + len(batch)
                if cancel is not None and cancel(cursor):
                    if batch:
                        payload = json.dumps(batch, separators=(",", ":")).encode("utf-8")
                        apply_pure_update(
                            store=store,
                            source_revision=source_revision,
                            payload=payload,
                            payload_sha256=hashlib.sha256(payload).hexdigest(),
                            row_start=batch_start,
                            row_stop=cursor,
                            operation_token=f"rows-{batch_start}-{cursor}",
                            reducer=reducer,
                        )
                    return CheckpointedCsvFitResult("CANCELLED", None)
                batch.append([float(observation.time), type(observation) is ExactLifetime])
            if batch:
                payload = json.dumps(batch, separators=(",", ":")).encode("utf-8")
                apply_pure_update(
                    store=store,
                    source_revision=source_revision,
                    payload=payload,
                    payload_sha256=hashlib.sha256(payload).hexdigest(),
                    row_start=batch_start,
                    row_stop=batch_start + len(batch),
                    operation_token=f"rows-{batch_start}-{batch_start + len(batch)}",
                    reducer=reducer,
                )
        final = store.read()
        if final.source_revision != source_revision:
            return CheckpointedCsvFitResult("SOURCE_REVISION_MISMATCH", None)
        return CheckpointedCsvFitResult(
            "COMPLETE", fit_exponential_reduction_state(reducer.decode_state(final.state))
        )
    except EngineContractError as error:
        return CheckpointedCsvFitResult(error.code.value, None)


def create_checkpointed_csv_store(
    store_path: str | os.PathLike[str],
    *,
    csv_path: Path,
    source_id: PublicSourceId,
) -> SQLiteCheckpointStore:
    """Create a durable store contracted to one CSV file, source id and reducer.

    The revision is the file's current SHA-256 digest, stream-hashed once
    here; `fit_exponential_checkpointed_csv` re-hashes the file on every
    call and rejects the resume with `SOURCE_REVISION_MISMATCH` if it no
    longer matches. Callers no longer hand-write the checkpoint record
    themselves.
    """

    if not isinstance(csv_path, Path):
        raise TypeError("csv_path must be a pathlib.Path")
    if type(source_id) is not PublicSourceId:
        raise TypeError("source_id must be PublicSourceId")
    revision = _hash_file_sha256(csv_path)
    reducer = ExponentialCheckpointReducer()
    record = CheckpointRecord.create(
        format_version=1,
        source_id=source_id.value,
        source_schema=CHECKPOINTED_CSV_SOURCE_SCHEMA,
        source_revision=revision,
        reducer_id=reducer.reducer_id,
        accumulator_schema=reducer.accumulator_schema,
        plan_digest=CHECKPOINTED_CSV_PLAN_DIGEST,
        cursor=0,
        committed_ranges=(),
        generation=0,
        operation_token=None,
        operation_digest=None,
        state=reducer.encode_state(ExponentialReductionState.empty()),
    )
    return SQLiteCheckpointStore.create(store_path, record)


def _provenance(
    adapter: CsvLifetimeAdapter,
    plan: ExecutionPlan,
    passes: PassEnforcer,
    buffer: BoundedChunkBuffer,
    mutation: SourceMutationStatus,
) -> ExecutionProvenance:
    return ExecutionProvenance(
        schema_version="1",
        run_id=f"run_{secrets.token_hex(16)}",
        source=SourceProvenance(
            adapter.source_id,
            "1",
            SourceRedaction(SourceRedactionReason.HASH_UNAVAILABLE),
            mutation,
        ),
        execution=snapshot_execution_observation(
            plan=plan,
            pass_enforcer=passes,
            buffer=buffer,
            adapter=AdapterProvenance(AdapterKind.CSV, "1"),
            spool=SpoolNotUsed(),
        ),
        estimator=EstimatorProvenance(
            "exponential", "censored_mle", "1", _exponential_settings_sha256()
        ),
        rng=RngProvenance(RngPolicy.NO_RANDOMNESS, "none", None),
        approximation=ExactComputation("closed_form_mle"),
        checkpoint=CheckpointNotUsed(),
    )


def _exponential_settings_sha256() -> str:
    """Hash the complete canonical settings contract rather than a placeholder."""

    settings = {
        "censoring_assumption": "independent_right_censoring",
        "family": "exponential",
        "location": 0.0,
        "parameterization": "rate",
        "reduction": "neumaier_canonical_input_order",
    }
    encoded = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CHECKPOINTED_CSV_PLAN_DIGEST",
    "CHECKPOINTED_CSV_SOURCE_SCHEMA",
    "CheckpointedCsvFitResult",
    "ExponentialSourceFitResult",
    "create_checkpointed_csv_store",
    "fit_exponential_checkpointed_csv",
    "fit_exponential_checkpointed_chunks",
    "fit_exponential_csv",
    "fit_exponential_source",
]
