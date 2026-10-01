"""Immutable checkpoint records and a non-durable in-memory CAS test double."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Protocol

from veridist.engine.errors import EngineContractError, FailureCode

CHECKPOINT_FORMAT_VERSION = 1


def _require_text(label: str, value: str) -> None:
    if not value.strip():
        raise ValueError(f"{label} must be non-empty")


def _canonical_payload(record: CheckpointRecord) -> bytes:
    value = {
        "accumulator_schema": record.accumulator_schema,
        "committed_ranges": [list(item) for item in record.committed_ranges],
        "cursor": record.cursor,
        "format_version": record.format_version,
        "generation": record.generation,
        "operation_digest": record.operation_digest,
        "operation_token": record.operation_token,
        "plan_digest": record.plan_digest,
        "reducer_id": record.reducer_id,
        "source_id": record.source_id,
        "source_revision": record.source_revision,
        "source_schema": record.source_schema,
        "state": base64.b64encode(record.state).decode("ascii"),
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _checksum(record: CheckpointRecord) -> str:
    return hashlib.sha256(_canonical_payload(record)).hexdigest()


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    """Canonical private resume state protected by a SHA-256 integrity checksum."""

    format_version: int
    source_id: str
    source_schema: str
    source_revision: str
    reducer_id: str
    accumulator_schema: str
    plan_digest: str
    cursor: int
    committed_ranges: tuple[tuple[int, int], ...]
    generation: int
    operation_token: str | None
    operation_digest: str | None
    state: bytes
    checksum: str

    @classmethod
    def create(
        cls,
        *,
        format_version: int,
        source_id: str,
        source_schema: str,
        source_revision: str,
        reducer_id: str,
        accumulator_schema: str,
        plan_digest: str,
        cursor: int,
        committed_ranges: tuple[tuple[int, int], ...],
        generation: int,
        operation_token: str | None,
        operation_digest: str | None,
        state: bytes | bytearray | memoryview,
    ) -> CheckpointRecord:
        """Build a validated record and derive its canonical checksum."""

        for label, value in (
            ("source_id", source_id),
            ("source_schema", source_schema),
            ("source_revision", source_revision),
            ("reducer_id", reducer_id),
            ("accumulator_schema", accumulator_schema),
            ("plan_digest", plan_digest),
        ):
            _require_text(label, value)
        if isinstance(format_version, bool) or format_version < 1:
            raise ValueError("format_version must be positive")
        if (
            isinstance(cursor, bool)
            or isinstance(generation, bool)
            or cursor < 0
            or generation < 0
        ):
            raise ValueError("cursor and generation must be non-negative")
        if (operation_token is None) != (operation_digest is None):
            raise ValueError("operation token and digest must both be present or absent")
        if operation_token is not None:
            _require_text("operation_token", operation_token)
            assert operation_digest is not None
            _require_text("operation_digest", operation_digest)
        if not isinstance(state, bytes | bytearray | memoryview):
            raise TypeError("state must be bytes-like")
        for start, stop in committed_ranges:
            if (
                isinstance(start, bool)
                or isinstance(stop, bool)
                or start < 0
                or stop <= start
            ):
                raise ValueError("committed ranges must be non-empty non-negative intervals")
        canonical_ranges = () if cursor == 0 else ((0, cursor),)
        if committed_ranges != canonical_ranges:
            raise ValueError("committed ranges must be the canonical contiguous cursor prefix")
        provisional = cls(
            format_version=format_version,
            source_id=source_id,
            source_schema=source_schema,
            source_revision=source_revision,
            reducer_id=reducer_id,
            accumulator_schema=accumulator_schema,
            plan_digest=plan_digest,
            cursor=cursor,
            committed_ranges=committed_ranges,
            generation=generation,
            operation_token=operation_token,
            operation_digest=operation_digest,
            state=bytes(state),
            checksum="",
        )
        return replace(provisional, checksum=_checksum(provisional))

    def has_valid_checksum(self) -> bool:
        """Return whether the stored checksum matches the canonical record."""

        return self.checksum == _checksum(self)

    def next_generation(
        self,
        *,
        cursor: int,
        committed_ranges: tuple[tuple[int, int], ...],
        operation_token: str,
        operation_digest: str,
        state: bytes,
    ) -> CheckpointRecord:
        """Return the immutable candidate for the next atomic CAS transition."""

        return self.create(
            format_version=self.format_version,
            source_id=self.source_id,
            source_schema=self.source_schema,
            source_revision=self.source_revision,
            reducer_id=self.reducer_id,
            accumulator_schema=self.accumulator_schema,
            plan_digest=self.plan_digest,
            cursor=cursor,
            committed_ranges=committed_ranges,
            generation=self.generation + 1,
            operation_token=operation_token,
            operation_digest=operation_digest,
            state=state,
        )


class CheckpointCommitUncertain(RuntimeError):
    """The store cannot say whether a requested CAS transition committed."""


class CheckpointStore(Protocol):
    """Minimal atomic checkpoint protocol; implementations define durability."""

    def read(self) -> CheckpointRecord: ...

    def compare_and_swap(
        self,
        expected_generation: int,
        candidate: CheckpointRecord,
    ) -> CheckpointRecord: ...


class InMemoryCheckpointStore:
    """Deterministic test double with no durability or cross-process guarantees."""

    __slots__ = ("_lock", "_record", "_read_count", "_write_count")

    def __init__(self, initial: CheckpointRecord) -> None:
        self._record = initial
        self._lock = Lock()
        self._read_count = 0
        self._write_count = 0

    @property
    def read_count(self) -> int:
        return self._read_count

    @property
    def write_count(self) -> int:
        return self._write_count

    def read(self) -> CheckpointRecord:
        with self._lock:
            self._read_count += 1
            return self._record

    def compare_and_swap(
        self,
        expected_generation: int,
        candidate: CheckpointRecord,
    ) -> CheckpointRecord:
        with self._lock:
            if self._record.generation != expected_generation:
                raise EngineContractError(
                    FailureCode.CHECKPOINT_CONFLICT,
                    {
                        "actual_generation": self._record.generation,
                        "expected_generation": expected_generation,
                    },
                )
            if candidate.generation != expected_generation + 1:
                raise EngineContractError(
                    FailureCode.CHECKPOINT_CONFLICT,
                    {
                        "candidate_generation": candidate.generation,
                        "expected_generation": expected_generation + 1,
                    },
                )
            if not candidate.has_valid_checksum():
                raise EngineContractError(FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)
            self._record = candidate
            self._write_count += 1
            return self._record


class SQLiteCheckpointStore:
    """A local SQLite checkpoint store with transactional generation CAS.

    It is scoped to local filesystems and a single host. It is not an
    encryption, authentication, network-filesystem, or distributed-store API.
    """

    __slots__ = ("_path", "_timeout")

    def __init__(self, path: str | os.PathLike[str], *, timeout: float = 5.0) -> None:
        if isinstance(timeout, bool) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self._path = Path(path)
        self._timeout = timeout

    @property
    def path(self) -> Path:
        """Return the local database location without placing it in diagnostics."""

        return self._path

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(
                self._path,
                timeout=self._timeout,
                isolation_level=None,
            )
            connection.execute("PRAGMA synchronous = FULL")
            return connection
        except (OSError, sqlite3.Error):
            if connection is not None:
                connection.close()
            raise EngineContractError(FailureCode.CHECKPOINT_STORAGE_FAILED) from None

    @contextmanager
    def _opened(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE checkpoint (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                format_version INTEGER NOT NULL,
                generation INTEGER NOT NULL,
                payload TEXT NOT NULL,
                checksum TEXT NOT NULL
            )
            """
        )

    @staticmethod
    def _commit(connection: sqlite3.Connection) -> None:
        """Commit one validated transaction; isolated for fault-injection evidence."""

        connection.execute("COMMIT")

    def _reconcile_uncertain_commit(
        self,
        expected_generation: int,
        candidate: CheckpointRecord,
    ) -> CheckpointRecord:
        """Resolve a lost commit acknowledgement without replaying the transition."""

        try:
            observed = self.read()
        except EngineContractError:
            raise CheckpointCommitUncertain(
                "checkpoint commit acknowledgement is uncertain"
            ) from None
        if (
            observed.generation == candidate.generation
            and observed.operation_token == candidate.operation_token
            and observed.operation_digest == candidate.operation_digest
            and observed.checksum == candidate.checksum
        ):
            return observed
        if observed.generation != expected_generation:
            raise EngineContractError(FailureCode.CHECKPOINT_CONFLICT)
        raise CheckpointCommitUncertain("checkpoint commit acknowledgement is uncertain")

    @staticmethod
    def _encode(record: CheckpointRecord) -> tuple[str, str]:
        if not record.has_valid_checksum():
            raise EngineContractError(FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)
        return _canonical_payload(record).decode("utf-8"), record.checksum

    @staticmethod
    def _decode(
        format_version: object,
        generation: object,
        payload: object,
        checksum: object,
    ) -> CheckpointRecord:
        if type(format_version) is not int or format_version != CHECKPOINT_FORMAT_VERSION:
            raise EngineContractError(FailureCode.CHECKPOINT_FORMAT_UNSUPPORTED)
        if type(generation) is not int or type(payload) is not str or type(checksum) is not str:
            raise EngineContractError(FailureCode.CHECKPOINT_DECODE_FAILED)
        try:
            value = json.loads(payload)
            if not isinstance(value, dict):
                raise ValueError("checkpoint payload is not an object")
            state = base64.b64decode(value["state"], validate=True)
            record = CheckpointRecord.create(
                format_version=value["format_version"],
                source_id=value["source_id"],
                source_schema=value["source_schema"],
                source_revision=value["source_revision"],
                reducer_id=value["reducer_id"],
                accumulator_schema=value["accumulator_schema"],
                plan_digest=value["plan_digest"],
                cursor=value["cursor"],
                committed_ranges=tuple(tuple(item) for item in value["committed_ranges"]),
                generation=value["generation"],
                operation_token=value["operation_token"],
                operation_digest=value["operation_digest"],
                state=state,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            raise EngineContractError(FailureCode.CHECKPOINT_DECODE_FAILED) from None
        if record.format_version != format_version or record.generation != generation:
            raise EngineContractError(FailureCode.CHECKPOINT_DECODE_FAILED)
        if record.checksum != checksum:
            raise EngineContractError(FailureCode.CHECKPOINT_CHECKSUM_MISMATCH)
        return record

    @classmethod
    def create(
        cls,
        path: str | os.PathLike[str],
        initial: CheckpointRecord,
        *,
        timeout: float = 5.0,
    ) -> SQLiteCheckpointStore:
        """Create a new store with the schema and initial row in one transaction.

        `sqlite3.connect` brings the file into existence as a side effect of
        opening it, before any statement runs. Since `path` was confirmed
        absent above, this call is always the one that created that file, so
        any failure past this point deletes it again: a failed create leaves
        either no file, or a file holding a fully valid record. It never
        leaves a file with the table created but no row, or no table at all.
        """

        store = cls(path, timeout=timeout)
        payload, checksum = store._encode(initial)
        if store._path.exists():
            raise EngineContractError(FailureCode.CHECKPOINT_ALREADY_EXISTS)
        created_file = False
        try:
            store._path.parent.mkdir(parents=True, exist_ok=True)
            with store._opened() as connection:
                created_file = True
                connection.execute("BEGIN IMMEDIATE")
                try:
                    store._create_schema(connection)
                    connection.execute(
                        "INSERT INTO checkpoint VALUES (1, ?, ?, ?, ?)",
                        (initial.format_version, initial.generation, payload, checksum),
                    )
                    store._commit(connection)
                except sqlite3.Error:
                    try:
                        connection.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
        except sqlite3.IntegrityError:
            if created_file:
                store._path.unlink(missing_ok=True)
            raise EngineContractError(FailureCode.CHECKPOINT_ALREADY_EXISTS) from None
        except sqlite3.OperationalError as error:
            if created_file:
                store._path.unlink(missing_ok=True)
            if "already exists" in str(error).casefold():
                raise EngineContractError(FailureCode.CHECKPOINT_ALREADY_EXISTS) from None
            raise EngineContractError(FailureCode.CHECKPOINT_STORAGE_FAILED) from None
        except OSError:
            if created_file:
                store._path.unlink(missing_ok=True)
            raise EngineContractError(FailureCode.CHECKPOINT_STORAGE_FAILED) from None
        return store

    def read(self) -> CheckpointRecord:
        if not self._path.exists():
            raise EngineContractError(FailureCode.CHECKPOINT_NOT_FOUND)
        try:
            with self._opened() as connection:
                row = connection.execute(
                    "SELECT format_version, generation, payload, checksum "
                    "FROM checkpoint WHERE singleton = 1"
                ).fetchone()
        except sqlite3.Error:
            raise EngineContractError(FailureCode.CHECKPOINT_STORAGE_FAILED) from None
        if row is None:
            raise EngineContractError(FailureCode.CHECKPOINT_NOT_FOUND)
        return self._decode(*row)

    def compare_and_swap(
        self,
        expected_generation: int,
        candidate: CheckpointRecord,
    ) -> CheckpointRecord:
        if isinstance(expected_generation, bool) or expected_generation < 0:
            raise ValueError("expected_generation must be non-negative")
        if candidate.generation != expected_generation + 1:
            raise EngineContractError(FailureCode.CHECKPOINT_CONFLICT)
        payload, checksum = self._encode(candidate)
        acknowledgement_lost = False
        try:
            with self._opened() as connection:
                connection.execute("BEGIN IMMEDIATE")
                updated = connection.execute(
                    """
                    UPDATE checkpoint
                    SET format_version = ?, generation = ?, payload = ?, checksum = ?
                    WHERE singleton = 1 AND generation = ?
                    """,
                    (
                        candidate.format_version,
                        candidate.generation,
                        payload,
                        checksum,
                        expected_generation,
                    ),
                ).rowcount
                if updated != 1:
                    connection.execute("ROLLBACK")
                    raise EngineContractError(FailureCode.CHECKPOINT_CONFLICT)
                try:
                    self._commit(connection)
                except sqlite3.Error:
                    try:
                        connection.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    acknowledgement_lost = True
        except EngineContractError:
            raise
        except sqlite3.Error:
            raise EngineContractError(FailureCode.CHECKPOINT_STORAGE_FAILED) from None
        if acknowledgement_lost:
            return self._reconcile_uncertain_commit(expected_generation, candidate)
        return candidate

