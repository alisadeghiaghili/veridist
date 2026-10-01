"""Strict sequential CSV lifetime ingestion with no dialect or type inference."""

from __future__ import annotations

import csv
import hashlib
import os
import re
import sys
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field, fields, is_dataclass, replace
from decimal import Decimal
from enum import StrEnum
from io import BufferedIOBase, TextIOWrapper
from math import isfinite
from pathlib import Path
from typing import BinaryIO, Protocol, cast

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime
from veridist.engine.data_source import DataSourceMetadata, Replayability
from veridist.engine.delivery import ChunkEnvelope
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.pass_budget import PassEnforcer
from veridist.engine.provenance import PublicSourceId, SourceMutationStatus

CSV_SCHEMA_VERSION = "1"
_TIME = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
_CHUNK_ID = re.compile(r"chk_[0-9a-f]{32}")
_CSV_FAILURE_CODES = frozenset(
    {
        FailureCode.SOURCE_OPEN_FAILED,
        FailureCode.SOURCE_DECODE_FAILED,
        FailureCode.SOURCE_SCHEMA_INVALID,
        FailureCode.SOURCE_ROW_INVALID,
        FailureCode.CHUNK_TOO_LARGE,
        FailureCode.SOURCE_REVISION_MISMATCH,
        FailureCode.SOURCE_REVISION_UNAVAILABLE,
    }
)
_REASONS = frozenset(
    {
        "open_failed",
        "invalid_utf8",
        "header_missing",
        "header_duplicate",
        "header_columns_mismatch",
        "blank_record",
        "malformed_record",
        "invalid_time",
        "invalid_event_token",
        "record_too_large",
        "source_mutated",
        "identity_unavailable",
    }
)


class CsvAdapterFailurePhase(StrEnum):
    """Lifecycle phase assigned by the adapter, never inferred from yielded rows."""

    PREFLIGHT = "preflight"
    DELIVERY = "delivery"
    FINALIZATION = "finalization"


class CsvBinarySource(Protocol):
    """A replayable, testable binary source seam."""

    def open_binary(self, path: Path) -> BinaryIO:
        """Open a fresh binary stream for one iterator acquisition."""

    def identity(self, path: Path) -> object:
        """Return an opaque best-effort identity snapshot."""


class _FilesystemCsvSource:
    """Private local-file source; its path/identity never enters public output."""

    def open_binary(self, path: Path) -> BinaryIO:
        return path.open("rb")

    def identity(self, path: Path) -> object:
        """Return a cheap OS-reported identity snapshot, not a content hash.

        The tuple is ``(st_dev, st_ino, st_size, st_mtime_ns)``. Two snapshots
        comparing equal is reported upstream as
        :attr:`~veridist.engine.provenance.SourceMutationStatus.VERIFIED_UNCHANGED`,
        but it only means those four OS-reported values did not change; it is
        not a guarantee that the file's bytes are identical, since a rewrite
        that preserves device, inode, size, and modification time is not
        detected.
        """

        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


@dataclass(frozen=True, slots=True)
class CsvLifetimeSchema:
    """Exact names of the two CSV columns accepted by this adapter."""

    time_column: str
    event_observed_column: str

    def __post_init__(self) -> None:
        for value in (self.time_column, self.event_observed_column):
            if not isinstance(value, str) or not value:
                raise ValueError("CSV column names must be non-empty strings")
        if self.time_column == self.event_observed_column:
            raise ValueError("CSV column names must be distinct")


@dataclass(frozen=True, slots=True)
class CsvLifetimeLimits:
    """Logical retained-payload limits for sequential CSV delivery."""

    chunk_bytes: int
    max_inflight_bytes: int

    def __post_init__(self) -> None:
        for value in (self.chunk_bytes, self.max_inflight_bytes):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("CSV byte limits must be built-in integers")
            if value <= 0:
                raise ValueError("CSV byte limits must be positive")
        if self.max_inflight_bytes < self.chunk_bytes:
            raise ValueError("max_inflight_bytes must cover one chunk")

    @staticmethod
    def default() -> CsvLifetimeLimits:
        """Return the default byte limits: 65,536 for chunk and inflight bytes.

        The unit is CPython retained object-graph bytes, as measured by
        :func:`retained_object_graph_bytes` -- the memory a chunk and its
        parsed observations retain while buffered -- not serialized file
        bytes, not process RSS, and not any other platform byte count.
        """

        return CsvLifetimeLimits(65_536, 65_536)


class CsvLifetimeAdapterError(EngineContractError):
    """Typed redacted adapter failure from the closed engine code taxonomy."""

    def __init__(
        self,
        code: FailureCode,
        *,
        reason: str,
        phase: CsvAdapterFailurePhase = CsvAdapterFailurePhase.PREFLIGHT,
        mutation_status: SourceMutationStatus = SourceMutationStatus.NOT_CHECKED,
        record_offset: int | None = None,
    ) -> None:
        if code not in _CSV_FAILURE_CODES:
            raise ValueError("unsupported CSV adapter failure code")
        if reason not in _REASONS:
            raise ValueError("unsupported CSV adapter failure reason")
        if type(phase) is not CsvAdapterFailurePhase:
            raise TypeError("phase must be CsvAdapterFailurePhase")
        if type(mutation_status) is not SourceMutationStatus:
            raise TypeError("mutation_status must be SourceMutationStatus")
        self.phase = phase
        self.mutation_status = mutation_status
        # Phase is typed metadata on the exception, not public diagnostic
        # context.  Keeping context restricted prevents a lifecycle refactor
        # from expanding the redacted failure payload.
        context: dict[str, object] = {"reason": reason}
        if record_offset is not None:
            if isinstance(record_offset, bool) or not isinstance(record_offset, int):
                raise TypeError("record_offset must be an integer")
            if record_offset < 0:
                raise ValueError("record_offset must be non-negative")
            context["record_offset"] = record_offset
        super().__init__(code, context)

    def with_mutation_status(self, status: SourceMutationStatus) -> CsvLifetimeAdapterError:
        """Return a fresh error carrying a later identity-verification fact."""

        offset = self.context.get("record_offset")
        return CsvLifetimeAdapterError(
            self.code,
            reason=cast(str, self.context["reason"]),
            phase=self.phase,
            mutation_status=status,
            record_offset=cast(int | None, offset),
        )


@dataclass(frozen=True, slots=True)
class CsvLifetimeChunk:
    """One bounded parsed chunk with no raw CSV cells or source path."""

    envelope: ChunkEnvelope
    observations: tuple[LifetimeObservation, ...]
    retained_payload_bytes: int

    def __post_init__(self) -> None:
        if type(self.envelope) is not ChunkEnvelope:
            raise TypeError("envelope must be ChunkEnvelope")
        if type(self.observations) is not tuple:
            raise TypeError("observations must be a tuple")
        if any(
            type(item) not in {ExactLifetime, RightCensoredLifetime} for item in self.observations
        ):
            raise TypeError("observations must be lifetime observations")
        if isinstance(self.retained_payload_bytes, bool) or not isinstance(
            self.retained_payload_bytes, int
        ):
            raise TypeError("retained_payload_bytes must be an integer")
        if self.retained_payload_bytes <= 0:
            raise ValueError("retained_payload_bytes must be positive")
        if self.envelope.row_count != len(self.observations):
            raise ValueError("envelope range must match observations")
        if self.envelope.byte_size != self.retained_payload_bytes:
            raise ValueError("envelope byte size must equal retained payload bytes")


def retained_object_graph_bytes(value: object) -> int:
    """Return closed owned-object accounting for a chunk's retained graph."""

    seen: set[int] = set()

    def visit(item: object) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        if isinstance(item, type) or isinstance(item, type(sys)):
            return 0
        total = sys.getsizeof(item)
        if type(item) is tuple:
            return total + sum(visit(child) for child in item)
        if is_dataclass(item) and not isinstance(item, type):
            return total + sum(visit(getattr(item, field.name)) for field in fields(item))
        return total

    return visit(value)


@dataclass(frozen=True, slots=True)
class CsvLifetimeAdapter:
    """Family-neutral strict CSV source with replayable sequential chunks."""

    path: Path | str | os.PathLike[str]
    schema: CsvLifetimeSchema
    source_id: PublicSourceId
    limits: CsvLifetimeLimits
    opener: CsvBinarySource | None = None
    _estimated_fixed_bytes: int = field(init=False, repr=False, compare=False)
    _passes: PassEnforcer = field(init=False, repr=False, compare=False)
    _terminal_record_count: int | None = field(init=False, repr=False, compare=False)
    _exact_lifetime_bytes: int = field(init=False, repr=False, compare=False)
    _censored_lifetime_bytes: int = field(init=False, repr=False, compare=False)
    _exact_lifetime_needs_single_check: bool = field(init=False, repr=False, compare=False)
    _censored_lifetime_needs_single_check: bool = field(init=False, repr=False, compare=False)
    _tuple_slot_bytes: int = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            if isinstance(self.path, (str, os.PathLike)):
                object.__setattr__(self, "path", Path(self.path))
            else:
                raise TypeError("path must be a pathlib.Path, str, or os.PathLike[str]")
        if type(self.schema) is not CsvLifetimeSchema:
            raise TypeError("schema must be CsvLifetimeSchema")
        if type(self.source_id) is not PublicSourceId:
            raise TypeError("source_id must be PublicSourceId")
        if type(self.limits) is not CsvLifetimeLimits:
            raise TypeError("limits must be CsvLifetimeLimits")
        if self.opener is not None:
            if not callable(getattr(self.opener, "open_binary", None)):
                raise TypeError("opener must define open_binary")
            if not callable(getattr(self.opener, "identity", None)):
                raise TypeError("opener must define identity")
        object.__setattr__(self, "_passes", PassEnforcer(max_passes=1))
        object.__setattr__(self, "_terminal_record_count", None)
        empty_envelope = ChunkEnvelope(
            source_id=self.source_id.value,
            chunk_id=_chunk_id(self.source_id, 0, 0),
            sequence_number=0,
            row_start=0,
            row_stop=0,
            byte_size=1,
        )
        fixed = retained_object_graph_bytes(CsvLifetimeChunk(empty_envelope, (), 1))
        # `retained_object_graph_bytes` deduplicates by object identity, and
        # CPython's small-int cache means a chunk's envelope fields (offset,
        # sequence number, byte size) can share objects with each other --
        # or not -- depending on their actual values, in a way the
        # all-zero probe above does not reproduce. An exhaustive sweep over
        # realistic offsets, sequence numbers, and byte sizes found that
        # swing to be at most one machine int's worth (28 bytes on a 64-bit
        # build); the padding below is nearly three times that, so it stays
        # a genuine over-estimate with a wide margin. It is still never
        # treated as exact: it is always re-verified against a real
        # measurement before a chunk boundary decision is final (see
        # `_read_chunks`).
        fixed += 80
        object.__setattr__(self, "_estimated_fixed_bytes", fixed)

        # `ExactLifetime`/`RightCensoredLifetime` each hold exactly one
        # built-in `float` field, and CPython's float object size does not
        # depend on the float's value, so the retained graph size of either
        # type is a per-type constant. Measuring it once here -- instead of
        # on every parsed row -- is the whole point of this cache; the
        # sampled values below are not just a correctness check, they prove
        # the invariant the cache depends on actually holds before anything
        # relies on it. Unlike small ints, built-in floats are not
        # value-cached by CPython, so this measurement is not subject to
        # the identity-sharing swing described above.
        exact_bytes = retained_object_graph_bytes(ExactLifetime(1.0))
        censored_bytes = retained_object_graph_bytes(RightCensoredLifetime(1.0))
        for sample in (0.0, 1e-300, 1.0, 123456.789, 1e300):
            if retained_object_graph_bytes(ExactLifetime(sample)) != exact_bytes:
                raise AssertionError(
                    "ExactLifetime's retained object-graph size must not depend on "
                    "its float value"
                )
            if retained_object_graph_bytes(RightCensoredLifetime(sample)) != censored_bytes:
                raise AssertionError(
                    "RightCensoredLifetime's retained object-graph size must not "
                    "depend on its float value"
                )
        object.__setattr__(self, "_exact_lifetime_bytes", exact_bytes)
        object.__setattr__(self, "_censored_lifetime_bytes", censored_bytes)

        # Conservative (over-estimating, never under-estimating) single-record
        # chunk size, built the same way `_estimated_chunk_bytes` is: it is
        # safe to skip the exact per-row check below whenever this constant
        # already fits comfortably inside the declared limit.
        tuple_slot_bytes = sys.getsizeof((None,)) - sys.getsizeof(())
        object.__setattr__(self, "_tuple_slot_bytes", tuple_slot_bytes)
        object.__setattr__(
            self,
            "_exact_lifetime_needs_single_check",
            fixed + tuple_slot_bytes + exact_bytes > self.limits.chunk_bytes,
        )
        object.__setattr__(
            self,
            "_censored_lifetime_needs_single_check",
            fixed + tuple_slot_bytes + censored_bytes > self.limits.chunk_bytes,
        )

    @property
    def passes(self) -> PassEnforcer:
        """Expose this adapter's own single-pass enforcer for provenance snapshots.

        This is the enforcer actually consulted by :meth:`iter_chunks`, so its
        ``observation`` reflects how many times this adapter's source was
        really acquired -- unlike a pass enforcer created fresh around a
        caller-side iterator, which can only ever report one pass regardless
        of what the adapter itself did.
        """

        return self._passes

    @property
    def terminal_record_count(self) -> int | None:
        """Return the number of records parsed through EOF, or ``None`` before then.

        This is recorded independently of the chunks actually delivered to a
        caller, so a delivery-layer validator can compare the two and detect a
        chunk silently dropped between parsing and delivery, instead of only
        ever re-deriving its expectation from the same chunks it already
        accepted.
        """

        return self._terminal_record_count

    @property
    def metadata(self) -> DataSourceMetadata:
        """Expose source planning facts without a file path or private revision."""

        return DataSourceMetadata(
            source_id=self.source_id.value,
            schema_version=CSV_SCHEMA_VERSION,
            provenance_schema_version="1",
            replayability=Replayability.SINGLE_PASS,
            redaction_reason="hash_unavailable",
        )

    def iter_chunks(self) -> Iterator[CsvLifetimeChunk]:
        """Parse one fresh source pass into bounded, ordered lifetime chunks."""

        # Reserve before opening: a second acquisition is rejected without a
        # second source open or a hidden retry.
        self._passes.begin_pass((None,))
        source = self.opener if self.opener is not None else _FilesystemCsvSource()
        resolved_path = cast(Path, self.path)
        open_failure: CsvLifetimeAdapterError | None = None
        try:
            binary = source.open_binary(resolved_path)
        except OSError:
            open_failure = CsvLifetimeAdapterError(
                FailureCode.SOURCE_OPEN_FAILED,
                reason="open_failed",
                phase=CsvAdapterFailurePhase.PREFLIGHT,
            )
            binary = None
        if open_failure is not None:
            raise open_failure
        assert binary is not None
        text: TextIOWrapper | None = None
        primary_error: BaseException | None = None
        translated: CsvLifetimeAdapterError | None = None
        try:
            before_identity, identity_failure = _identity_or_failure(
                source, resolved_path, binary=binary
            )
            if identity_failure is not None:
                raise identity_failure
            if not isinstance(binary, BufferedIOBase) and not hasattr(binary, "read"):
                raise TypeError("source must return a binary readable stream")
            text = TextIOWrapper(binary, encoding="utf-8-sig", newline="")
            reader = cast(Iterator[list[str]], csv.reader(text, strict=True))
            header = self._read_header(reader)
            self._validate_header(header)
            semantic_failure = yield from self._read_chunks(reader)
            after_identity, identity_failure = _identity_or_failure(
                source, resolved_path, phase=CsvAdapterFailurePhase.FINALIZATION
            )
            if identity_failure is not None:
                raise identity_failure
            if after_identity != before_identity:
                raise CsvLifetimeAdapterError(
                    FailureCode.SOURCE_REVISION_MISMATCH,
                    reason="source_mutated",
                    phase=CsvAdapterFailurePhase.FINALIZATION,
                    mutation_status=SourceMutationStatus.MISMATCH_DETECTED,
                )
            if semantic_failure is not None:
                raise semantic_failure.with_mutation_status(SourceMutationStatus.VERIFIED_UNCHANGED)
        except UnicodeDecodeError as error:
            primary_error = error
            translated = CsvLifetimeAdapterError(
                FailureCode.SOURCE_DECODE_FAILED,
                reason="invalid_utf8",
                phase=CsvAdapterFailurePhase.DELIVERY,
            )
        except csv.Error as error:
            primary_error = error
            translated = CsvLifetimeAdapterError(
                FailureCode.SOURCE_ROW_INVALID,
                reason="malformed_record",
                phase=CsvAdapterFailurePhase.DELIVERY,
            )
        except BaseException as error:
            primary_error = error
            raise
        finally:
            try:
                if text is not None:
                    text.close()
                else:
                    binary.close()
            except Exception:
                if primary_error is None:
                    raise
        if translated is not None:
            raise translated

    def _read_header(self, reader: Iterator[list[str]]) -> list[str]:
        failure: CsvLifetimeAdapterError | None = None
        try:
            return next(reader)
        except StopIteration:
            failure = CsvLifetimeAdapterError(
                FailureCode.SOURCE_SCHEMA_INVALID,
                reason="header_missing",
                phase=CsvAdapterFailurePhase.PREFLIGHT,
            )
        assert failure is not None
        raise failure

    def _validate_header(self, header: list[str]) -> None:
        if len(header) != len(set(header)):
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_SCHEMA_INVALID,
                reason="header_duplicate",
                phase=CsvAdapterFailurePhase.PREFLIGHT,
            )
        if header != [self.schema.time_column, self.schema.event_observed_column]:
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_SCHEMA_INVALID,
                reason="header_columns_mismatch",
                phase=CsvAdapterFailurePhase.PREFLIGHT,
            )

    def _read_chunks(
        self, reader: Iterator[list[str]]
    ) -> Generator[CsvLifetimeChunk, None, CsvLifetimeAdapterError | None]:
        records: list[LifetimeObservation] = []
        record_bytes = 0
        start = 0
        record_offset = 0
        sequence = 0
        failure: CsvLifetimeAdapterError | None = None
        # A blank record is only valid when it is the last thing in the file
        # (for example a trailing "\r\n\r\n" an editor or spreadsheet added).
        # Its offset is held here, undecided, until either a later row proves
        # it was not actually trailing (and it fails at this offset) or the
        # reader reaches EOF (and it is silently dropped).
        pending_blank_offset: int | None = None
        for row in reader:
            if not row:
                if pending_blank_offset is None:
                    pending_blank_offset = record_offset
                record_offset += 1
                continue
            if pending_blank_offset is not None:
                if failure is None:
                    failure = CsvLifetimeAdapterError(
                        FailureCode.SOURCE_ROW_INVALID,
                        reason="blank_record",
                        phase=CsvAdapterFailurePhase.DELIVERY,
                        record_offset=pending_blank_offset,
                    )
                pending_blank_offset = None
            if failure is not None:
                record_offset += 1
                continue
            try:
                observation = self._parse_row(row, record_offset)
            except CsvLifetimeAdapterError as error:
                failure = error
                record_offset += 1
                continue
            if type(observation) is ExactLifetime:
                observation_bytes = self._exact_lifetime_bytes
                needs_single_check = self._exact_lifetime_needs_single_check
            else:
                observation_bytes = self._censored_lifetime_bytes
                needs_single_check = self._censored_lifetime_needs_single_check
            if needs_single_check:
                # `ExactLifetime`/`RightCensoredLifetime` with a float payload
                # have a value-independent retained graph size (asserted in
                # `__post_init__`), so this exact rebuild only ever runs when
                # the declared limit is already tight enough that it might
                # matter -- never once per row on an ordinary file.
                single = self._chunk((observation,), record_offset, sequence)
                if single.retained_payload_bytes > self.limits.chunk_bytes:
                    raise CsvLifetimeAdapterError(
                        FailureCode.CHUNK_TOO_LARGE,
                        reason="record_too_large",
                        phase=CsvAdapterFailurePhase.DELIVERY,
                        record_offset=record_offset,
                    )
            candidate_bytes = self._estimated_chunk_bytes(
                records, record_bytes + observation_bytes, start, sequence
            )
            if records and candidate_bytes > self.limits.chunk_bytes:
                # The conservative tally can deliberately over-estimate.  An
                # exact check at this impending boundary retains exact-fit
                # semantics while still avoiding a graph walk per record.
                candidate = self._chunk(tuple([*records, observation]), start, sequence)
                if candidate.retained_payload_bytes <= self.limits.chunk_bytes:
                    records.append(observation)
                    record_bytes += observation_bytes
                else:
                    yield self._chunk(tuple(records), start, sequence)
                    start = record_offset
                    sequence += 1
                    records = [observation]
                    record_bytes = observation_bytes
            else:
                records.append(observation)
                record_bytes += observation_bytes
            record_offset += 1
        # Recorded once the source reader itself has reached EOF, independent
        # of whatever a caller does with the chunks yielded below -- so a
        # chunk silently lost downstream can be detected by comparison.
        object.__setattr__(self, "_terminal_record_count", record_offset)
        if failure is not None:
            return failure
        if records:
            yield self._chunk(tuple(records), start, sequence)
        return None

    def _estimated_chunk_bytes(
        self,
        records: list[LifetimeObservation],
        observation_bytes: int,
        start: int,
        sequence: int,
    ) -> int:
        """O(1) conservative bound used while accumulating a chunk.

        The exact owned graph is measured once for each emitted chunk.  This
        deliberately avoids reconstructing and walking every prefix of a long
        chunk (the former quadratic hot path).
        """
        count = len(records) + 1
        # Account from declared graph components.  The tuple capacity follows
        # CPython's published object-size seam (empty tuple plus one element),
        # cached once in `__post_init__`, so no O(k) prefix tuple is ever
        # constructed merely to estimate it.  The final emitted chunk is
        # still measured exactly below.
        # Object ownership is exact at emission, while this pre-emission tally
        # intentionally avoids an O(k) graph walk.  Different chunk-id values
        # can have small interpreter-specific graph overhead, so reserve a
        # fixed conservative guard.  Without it an otherwise valid chunk can
        # cross the public byte cap only after emission.
        return (
            self._estimated_fixed_bytes
            + self._tuple_slot_bytes * count
            + observation_bytes
        )

    def _parse_row(self, row: list[str], record_offset: int) -> LifetimeObservation:
        # Blank records are intercepted by `_read_chunks` before this is
        # called, because a blank record's validity depends on whether it is
        # the last row in the file -- something only the caller's loop, not
        # this single-row parser, can know.
        if len(row) != 2:
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_ROW_INVALID,
                reason="malformed_record",
                phase=CsvAdapterFailurePhase.DELIVERY,
                record_offset=record_offset,
            )
        time = self._parse_time(row[0], record_offset)
        if row[1] == "1":
            return ExactLifetime(time)
        if row[1] == "0":
            return RightCensoredLifetime(time)
        raise CsvLifetimeAdapterError(
            FailureCode.SOURCE_ROW_INVALID,
            reason="invalid_event_token",
            phase=CsvAdapterFailurePhase.DELIVERY,
            record_offset=record_offset,
        )

    @staticmethod
    def _parse_time(value: str, record_offset: int) -> Decimal:
        if _TIME.fullmatch(value) is None:
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_ROW_INVALID,
                reason="invalid_time",
                phase=CsvAdapterFailurePhase.DELIVERY,
                record_offset=record_offset,
            )
        decimal = Decimal(value)
        converted = float(decimal)
        if not isfinite(converted) or (decimal > 0 and converted == 0.0):
            raise CsvLifetimeAdapterError(
                FailureCode.SOURCE_ROW_INVALID,
                reason="invalid_time",
                phase=CsvAdapterFailurePhase.DELIVERY,
                record_offset=record_offset,
            )
        return decimal

    def _chunk(
        self,
        observations: tuple[LifetimeObservation, ...],
        start: int,
        sequence: int,
    ) -> CsvLifetimeChunk:
        stop = start + len(observations)
        envelope = ChunkEnvelope(
            source_id=self.source_id.value,
            chunk_id=_chunk_id(self.source_id, start, stop),
            sequence_number=sequence,
            row_start=start,
            row_stop=stop,
            byte_size=1,
        )
        chunk = CsvLifetimeChunk(envelope, observations, 1)
        measured = retained_object_graph_bytes(chunk)
        chunk = replace(
            chunk,
            envelope=replace(envelope, byte_size=measured),
            retained_payload_bytes=measured,
        )
        final = retained_object_graph_bytes(chunk)
        if final != measured:
            chunk = replace(
                chunk,
                envelope=replace(chunk.envelope, byte_size=final),
                retained_payload_bytes=final,
            )
        return chunk


def _chunk_id(source_id: PublicSourceId, start: int, stop: int) -> str:
    digest = hashlib.sha256(f"{source_id.value}:{start}:{stop}".encode("ascii")).hexdigest()
    value = f"chk_{digest[:32]}"
    assert _CHUNK_ID.fullmatch(value) is not None
    return value


def _identity_or_failure(
    source: CsvBinarySource,
    path: Path,
    *,
    phase: CsvAdapterFailurePhase = CsvAdapterFailurePhase.PREFLIGHT,
    binary: BinaryIO | None = None,
) -> tuple[object | None, CsvLifetimeAdapterError | None]:
    try:
        # For the built-in filesystem source the initial identity comes from the
        # opened handle, not a second path lookup that could race the open.
        if type(source) is _FilesystemCsvSource and binary is not None:
            stat = os.fstat(binary.fileno())
            value: object = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
        else:
            value = source.identity(path)
    except OSError:
        return None, CsvLifetimeAdapterError(
            FailureCode.SOURCE_REVISION_UNAVAILABLE,
            reason="identity_unavailable",
            phase=phase,
            mutation_status=SourceMutationStatus.UNAVAILABLE,
        )
    if value is None:
        return None, CsvLifetimeAdapterError(
            FailureCode.SOURCE_REVISION_UNAVAILABLE,
            reason="identity_unavailable",
            phase=phase,
            mutation_status=SourceMutationStatus.UNAVAILABLE,
        )
    return value, None


__all__ = [
    "CSV_SCHEMA_VERSION",
    "CsvAdapterFailurePhase",
    "CsvBinarySource",
    "CsvLifetimeAdapter",
    "CsvLifetimeAdapterError",
    "CsvLifetimeChunk",
    "CsvLifetimeLimits",
    "CsvLifetimeSchema",
    "retained_object_graph_bytes",
]
