"""The local SQLite partition-state store: committed partition states and the pass ledger.

A scale fit that is given a checkpoint keeps what it has computed in this store, so that a
run that was interrupted continues from it and a source that has gained partitions is not
read again from the start. The store is one local database file for one *job*. It holds:

* the **job** (one row): the digest and the named components of the identity of the job,
  fixed when the store is created and compared exactly on every later use;
* the **manifest**: the partitions the job covers, each with its identifier, fingerprint and
  fingerprint level (they are what a later run is checked against);
* the **partition states**: for every pass (its index and the digest of the exact kernel
  parameters it was evaluated at) and every partition, the serialized state, its SHA-256 and
  the fingerprint it was computed from;
* the **pass ledger**: for every completed pass, its index, its parameters (the record the
  digest is of) and the canonical bytes of the merged state, with their SHA-256.

Transactions follow ADR-0015 and ADR-0023. Every operation opens its own short-lived
connection and closes it; a write sets ``synchronous = FULL`` and begins with
``BEGIN IMMEDIATE``. A partition state is one transaction: it is committed whole, with its
digest, or it is absent, and a transaction that was never committed (the process died
first) leaves no trace. Committing a key again with the same bytes is recognised as already
applied; the same key with other bytes is the typed ``PARTITION_STATE_CONFLICT`` (it means
non-determinism or a changed environment) and is never merged. If the acknowledgement of a
commit is lost, the store reads the key back: it returns only on an exact match, reports a
conflict for different bytes and otherwise ``STORE_COMMIT_UNCERTAIN``. Only the parent
process writes; a worker never opens the store.

Every row carries a checksum over all its fields and every state and merged state its
SHA-256, verified on every read. A mismatch, a malformed row or an unknown format is a typed
:class:`~veridist.scale._errors.ScaleCheckpointError`, never a state that is used. This is
the format version 2 of the store; a version-1 single-row checkpoint database (ADR-0023) is
refused as an unsupported format and is not migrated, and the single-row store does not
read this one. Neither paths nor SQLite's messages nor state bytes appear in an error. This
is an integrity check against accidental change, not authentication, and the store is
scoped to local filesystems.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, TypeVar

from veridist.scale._errors import ScaleCheckpointError, ScaleCheckpointErrorCode
from veridist.scale._sources import canonical_digest

PARTITION_STORE_FORMAT_VERSION: Final = 2
"""The format of this database; the single-row checkpoint database is format 1."""

STORE_KIND: Final = "sqlite_partition"
"""The name of this store in provenance."""

DEFAULT_TIMEOUT_SECONDS: Final = 5.0

_TABLES: Final = ("job", "manifest", "partition_state", "pass_ledger")
_SINGLE_ROW_TABLE: Final = "checkpoint"

_SCHEMA: Final = (
    """
    CREATE TABLE job (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        format_version INTEGER NOT NULL,
        job_digest TEXT NOT NULL,
        components TEXT NOT NULL,
        checksum TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE manifest (
        partition_id TEXT PRIMARY KEY,
        fingerprint TEXT NOT NULL,
        fingerprint_level TEXT NOT NULL,
        checksum TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE partition_state (
        pass_index INTEGER NOT NULL,
        pass_key TEXT NOT NULL,
        partition_id TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        fingerprint_level TEXT NOT NULL,
        state BLOB NOT NULL,
        state_sha256 TEXT NOT NULL,
        checksum TEXT NOT NULL,
        PRIMARY KEY (pass_index, pass_key, partition_id)
    )
    """,
    """
    CREATE TABLE pass_ledger (
        pass_index INTEGER PRIMARY KEY,
        pass_key TEXT NOT NULL,
        descriptor TEXT NOT NULL,
        merged BLOB NOT NULL,
        merged_sha256 TEXT NOT NULL,
        checksum TEXT NOT NULL
    )
    """,
)

_COMPONENT_SEPARATOR: Final = "\x1f"
_PAIR_SEPARATOR: Final = "\x1e"

T = TypeVar("T")


def _storage_failed(error: sqlite3.Error | None = None) -> ScaleCheckpointError:
    """A storage failure; only SQLite's symbolic error name is ever attached."""

    name = getattr(error, "sqlite_errorname", None)
    detail = name if isinstance(name, str) and name else None
    return ScaleCheckpointError(ScaleCheckpointErrorCode.STORE_STORAGE_FAILED, detail=detail)


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _encode_components(components: Sequence[tuple[str, str]]) -> str:
    return _PAIR_SEPARATOR.join(
        f"{name}{_COMPONENT_SEPARATOR}{value}" for name, value in components
    )


def _decode_components(text: str) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for item in text.split(_PAIR_SEPARATOR) if text else ():
        name, separator, value = item.partition(_COMPONENT_SEPARATOR)
        if not separator:
            raise ScaleCheckpointError(ScaleCheckpointErrorCode.STORE_DECODE_FAILED)
        pairs.append((name, value))
    return tuple(pairs)


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One partition of the job: its identifier, fingerprint and fingerprint level."""

    partition_id: str
    fingerprint: str
    level: str


@dataclass(frozen=True, slots=True)
class StoredJob:
    """The job a store holds: its digest, named components and partition manifest."""

    digest: str
    components: tuple[tuple[str, str], ...]
    manifest: tuple[ManifestEntry, ...]


@dataclass(frozen=True, slots=True)
class StoredState:
    """A committed partition state and the fingerprint it was computed from."""

    fingerprint: str
    level: str
    data: bytes


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """A completed pass: the digest of its parameters, their record and the merged state."""

    pass_key: str
    descriptor: str
    data: bytes


def _row_checksum(kind: str, *fields: object) -> str:
    return canonical_digest("veridist.scale.partition_store", kind, *fields)


def _text(value: object) -> str:
    if type(value) is not str:
        raise ScaleCheckpointError(ScaleCheckpointErrorCode.STORE_DECODE_FAILED)
    return value


def _blob(value: object) -> bytes:
    if type(value) is not bytes:
        raise ScaleCheckpointError(ScaleCheckpointErrorCode.STORE_DECODE_FAILED)
    return value


def _verified(expected: str, actual: str) -> None:
    if expected != actual:
        raise ScaleCheckpointError(ScaleCheckpointErrorCode.STORE_DIGEST_MISMATCH)


def _manifest_entry(row: tuple[object, ...]) -> ManifestEntry:
    entry = ManifestEntry(_text(row[0]), _text(row[1]), _text(row[2]))
    _verified(
        _text(row[3]),
        _row_checksum("manifest", entry.partition_id, entry.fingerprint, entry.level),
    )
    return entry


class SQLitePartitionStore:
    """A local SQLite store of the partition states and the pass ledger of one job.

    Scoped to local filesystems and a single host, like the checkpoint store of ADR-0023;
    it is not an encryption, authentication, network-filesystem or distributed-store API.
    The parent process of a fit is its only writer.
    """

    __slots__ = ("_path", "_timeout")

    def __init__(
        self, path: str | os.PathLike[str], *, timeout: float = DEFAULT_TIMEOUT_SECONDS
    ) -> None:
        if isinstance(timeout, bool) or not isinstance(timeout, int | float) or timeout <= 0:
            raise ValueError("timeout must be positive")
        self._path = Path(path)
        self._timeout = float(timeout)

    @property
    def path(self) -> Path:
        """The local database location; it is kept out of every error and record."""

        return self._path

    @contextmanager
    def _opened(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        """One short-lived connection, always closed; durability is requested for writes."""

        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self._path, timeout=self._timeout, isolation_level=None)
            if write:
                connection.execute("PRAGMA synchronous = FULL")
            yield connection
        except sqlite3.Error as error:
            raise _storage_failed(error) from None
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _commit(connection: sqlite3.Connection) -> None:
        """Commit one validated transaction; isolated for fault-injection evidence."""

        connection.execute("COMMIT")

    @staticmethod
    def _tables(connection: sqlite3.Connection) -> frozenset[str]:
        rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        return frozenset(str(row[0]) for row in rows)

    @classmethod
    def _require_format(cls, connection: sqlite3.Connection) -> bool:
        """Whether the database holds this format; ``False`` for one that holds nothing.

        ``STORE_FORMAT_UNSUPPORTED`` for the single-row checkpoint database, for any other
        content and for another version.
        """

        names = cls._tables(connection)
        if not names:
            return False
        if not set(_TABLES) <= names:
            raise ScaleCheckpointError(ScaleCheckpointErrorCode.STORE_FORMAT_UNSUPPORTED)
        row = connection.execute("SELECT format_version FROM job WHERE singleton = 1").fetchone()
        if row is not None and row[0] != PARTITION_STORE_FORMAT_VERSION:
            raise ScaleCheckpointError(ScaleCheckpointErrorCode.STORE_FORMAT_UNSUPPORTED)
        return True

    def _transact(
        self,
        work: Callable[[sqlite3.Connection], T],
        confirm: Callable[[], bool],
        *,
        existing: bool = True,
    ) -> T:
        """Run ``work`` as one ``BEGIN IMMEDIATE`` transaction and commit it.

        ``work`` raises to abandon the transaction, which is rolled back. When only the
        acknowledgement of the commit is lost, ``confirm`` (a read through a new
        connection) says whether the transaction was applied: if it was, the result of
        ``work`` stands; if it was not, or cannot be read, the outcome is
        ``STORE_COMMIT_UNCERTAIN``. ``existing`` requires the database to hold the format
        already (the job is created by the call that passes ``False``); a store that does
        not exist is ``STORE_STORAGE_FAILED``, and is not created.
        """

        if existing and not self._path.exists():
            raise _storage_failed()
        acknowledgement_lost = False
        with self._opened(write=True) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                if existing and not self._require_format(connection):
                    raise ScaleCheckpointError(ScaleCheckpointErrorCode.STORE_FORMAT_UNSUPPORTED)
                result = work(connection)
            except BaseException:
                self._rollback(connection)
                raise
            try:
                self._commit(connection)
            except sqlite3.Error:
                self._rollback(connection)
                acknowledgement_lost = True
        if acknowledgement_lost and not self._applied(confirm):
            raise ScaleCheckpointError(ScaleCheckpointErrorCode.STORE_COMMIT_UNCERTAIN)
        return result

    @staticmethod
    def _rollback(connection: sqlite3.Connection) -> None:
        try:
            connection.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    @staticmethod
    def _applied(confirm: Callable[[], bool]) -> bool:
        try:
            return confirm()
        except ScaleCheckpointError:
            return False

    # ------------------------------------------------------------------------- the job

    def read_job(self) -> StoredJob | None:
        """The job the store holds, or ``None`` if the file does not exist or holds nothing.

        ``STORE_FORMAT_UNSUPPORTED``, ``STORE_DECODE_FAILED``, ``STORE_DIGEST_MISMATCH``
        or ``STORE_STORAGE_FAILED`` for anything that is not a sound job.
        """

        if not self._path.exists():
            return None
        with self._opened() as connection:
            if not self._require_format(connection):
                return None
            job = connection.execute(
                "SELECT format_version, job_digest, components, checksum "
                "FROM job WHERE singleton = 1"
            ).fetchone()
            manifest = connection.execute(
                "SELECT partition_id, fingerprint, fingerprint_level, checksum "
                "FROM manifest ORDER BY partition_id"
            ).fetchall()
        if job is None:
            return None
        digest, text = _text(job[1]), _text(job[2])
        _verified(_text(job[3]), _row_checksum("job", PARTITION_STORE_FORMAT_VERSION, digest, text))
        return StoredJob(
            digest,
            _decode_components(text),
            tuple(_manifest_entry(row) for row in manifest),
        )

    def create_job(
        self,
        digest: str,
        components: Sequence[tuple[str, str]],
        manifest: Sequence[ManifestEntry],
    ) -> None:
        """Create the database (if needed) and write the job with its manifest, atomically.

        ``JOB_EXISTS`` when it already holds a job and ``STORE_FORMAT_UNSUPPORTED`` when it
        holds anything else; a failed creation leaves no file that this call made.
        """

        text = _encode_components(components)
        entries = tuple(manifest)
        created = not self._path.exists()

        def work(connection: sqlite3.Connection) -> None:
            if self._tables(connection):
                held = self._require_format(connection)
                raise ScaleCheckpointError(
                    ScaleCheckpointErrorCode.JOB_EXISTS
                    if held
                    else ScaleCheckpointErrorCode.STORE_FORMAT_UNSUPPORTED
                )
            for statement in _SCHEMA:
                connection.execute(statement)
            connection.execute(
                "INSERT INTO job VALUES (1, ?, ?, ?, ?)",
                (
                    PARTITION_STORE_FORMAT_VERSION,
                    digest,
                    text,
                    _row_checksum("job", PARTITION_STORE_FORMAT_VERSION, digest, text),
                ),
            )
            self._insert_manifest(connection, entries)

        def confirm() -> bool:
            stored = self.read_job()
            return (
                stored is not None
                and stored.digest == digest
                and stored.manifest == tuple(sorted(entries, key=lambda entry: entry.partition_id))
            )

        try:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                raise _storage_failed() from None
            self._transact(work, confirm, existing=False)
        except BaseException:
            if created:
                self._path.unlink(missing_ok=True)
            raise

    @staticmethod
    def _insert_manifest(connection: sqlite3.Connection, entries: Sequence[ManifestEntry]) -> None:
        connection.executemany(
            "INSERT INTO manifest VALUES (?, ?, ?, ?)",
            [
                (
                    entry.partition_id,
                    entry.fingerprint,
                    entry.level,
                    _row_checksum("manifest", entry.partition_id, entry.fingerprint, entry.level),
                )
                for entry in entries
            ],
        )

    def amend_job(
        self,
        *,
        add: Sequence[ManifestEntry] = (),
        remove: Sequence[str] = (),
        replace: Sequence[ManifestEntry] = (),
        reset: bool = False,
    ) -> None:
        """Change the manifest in one transaction, and drop what the change makes stale.

        ``add`` partitions join the manifest, ``remove`` identifiers leave it and
        ``replace`` entries take a new fingerprint. The states of removed and replaced
        partitions are deleted, and ``reset`` deletes every state; the pass ledger is
        always emptied, because the merged states of its passes covered the old manifest.
        """

        added, removed, replaced = tuple(add), tuple(remove), tuple(replace)

        def work(connection: sqlite3.Connection) -> None:
            if reset:
                connection.execute("DELETE FROM partition_state")
            connection.execute("DELETE FROM pass_ledger")
            for identifier in (*removed, *(entry.partition_id for entry in replaced)):
                connection.execute(
                    "DELETE FROM partition_state WHERE partition_id = ?", (identifier,)
                )
            connection.executemany(
                "DELETE FROM manifest WHERE partition_id = ?", [(item,) for item in removed]
            )
            for entry in replaced:
                connection.execute(
                    "DELETE FROM manifest WHERE partition_id = ?", (entry.partition_id,)
                )
            self._insert_manifest(connection, (*added, *replaced))

        def confirm() -> bool:
            stored = self.read_job()
            return stored is not None and all(
                entry in stored.manifest for entry in (*added, *replaced)
            )

        self._transact(work, confirm)

    # --------------------------------------------------------------- partition states

    def read_pass(self, pass_index: int, pass_key: str) -> dict[str, StoredState]:
        """The committed states of one pass at one parameter digest, by partition identifier.

        Every row is verified (its checksum and the SHA-256 of its state) before any is
        returned: ``STORE_DIGEST_MISMATCH`` or ``STORE_DECODE_FAILED`` otherwise.
        """

        if not self._path.exists():
            return {}
        with self._opened() as connection:
            if not self._require_format(connection):
                return {}
            rows = connection.execute(
                "SELECT partition_id, fingerprint, fingerprint_level, state, state_sha256, "
                "checksum, pass_index, pass_key FROM partition_state "
                "WHERE pass_index = ? AND pass_key = ? ORDER BY partition_id",
                (pass_index, pass_key),
            ).fetchall()
        found: dict[str, StoredState] = {}
        for row in rows:
            identifier, fingerprint, level = _text(row[0]), _text(row[1]), _text(row[2])
            data, data_digest = _blob(row[3]), _text(row[4])
            _verified(data_digest, _digest(data))
            _verified(
                _text(row[5]),
                _row_checksum(
                    "state", pass_index, pass_key, identifier, fingerprint, level, data_digest
                ),
            )
            found[identifier] = StoredState(fingerprint, level, data)
        return found

    def commit_partition(
        self,
        pass_index: int,
        pass_key: str,
        partition_id: str,
        fingerprint: str,
        level: str,
        data: bytes,
    ) -> bool:
        """Commit one partition state: ``True`` if written, ``False`` if already applied.

        The same key with the same bytes is already applied; the same key with other bytes
        is ``PARTITION_STATE_CONFLICT`` and nothing is changed.
        """

        state_digest = _digest(data)

        def held(connection: sqlite3.Connection) -> str | None:
            row = connection.execute(
                "SELECT state_sha256 FROM partition_state "
                "WHERE pass_index = ? AND pass_key = ? AND partition_id = ?",
                (pass_index, pass_key, partition_id),
            ).fetchone()
            return None if row is None else str(row[0])

        def work(connection: sqlite3.Connection) -> bool:
            present = held(connection)
            if present is not None:
                if present != state_digest:
                    raise ScaleCheckpointError(ScaleCheckpointErrorCode.PARTITION_STATE_CONFLICT)
                return False
            connection.execute(
                "INSERT INTO partition_state VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    pass_index,
                    pass_key,
                    partition_id,
                    fingerprint,
                    level,
                    data,
                    state_digest,
                    _row_checksum(
                        "state",
                        pass_index,
                        pass_key,
                        partition_id,
                        fingerprint,
                        level,
                        state_digest,
                    ),
                ),
            )
            return True

        def confirm() -> bool:
            found = self.read_pass(pass_index, pass_key).get(partition_id)
            return found is not None and _digest(found.data) == state_digest

        return self._transact(work, confirm)

    # ------------------------------------------------------------------- the ledger

    def read_ledger(self) -> dict[int, LedgerEntry]:
        """The completed passes by index, each verified: its row and the merged state's digest."""

        if not self._path.exists():
            return {}
        with self._opened() as connection:
            if not self._require_format(connection):
                return {}
            rows = connection.execute(
                "SELECT pass_index, pass_key, descriptor, merged, merged_sha256, checksum "
                "FROM pass_ledger ORDER BY pass_index"
            ).fetchall()
        found: dict[int, LedgerEntry] = {}
        for row in rows:
            index, key, descriptor = int(row[0]), _text(row[1]), _text(row[2])
            data, data_digest = _blob(row[3]), _text(row[4])
            _verified(data_digest, _digest(data))
            _verified(key, _digest(descriptor.encode("utf-8")))
            _verified(
                _text(row[5]),
                _row_checksum("ledger", index, key, descriptor, data_digest),
            )
            found[index] = LedgerEntry(key, descriptor, data)
        return found

    def commit_pass(
        self,
        pass_index: int,
        pass_key: str,
        descriptor: str,
        data: bytes,
        *,
        prune: bool,
    ) -> bool:
        """Commit a completed pass: ``True`` if written, ``False`` if already applied.

        With ``prune`` the partition states of every earlier pass are deleted in the same
        transaction (a multi-pass fit keeps the states of its latest pass only). A row at
        this index with other content is ``PARTITION_STATE_CONFLICT`` and nothing changes.
        """

        merged_digest = _digest(data)

        def work(connection: sqlite3.Connection) -> bool:
            row = connection.execute(
                "SELECT pass_key, merged_sha256 FROM pass_ledger WHERE pass_index = ?",
                (pass_index,),
            ).fetchone()
            if row is not None:
                if (row[0], row[1]) != (pass_key, merged_digest):
                    raise ScaleCheckpointError(ScaleCheckpointErrorCode.PARTITION_STATE_CONFLICT)
                return False
            connection.execute(
                "INSERT INTO pass_ledger VALUES (?, ?, ?, ?, ?, ?)",
                (
                    pass_index,
                    pass_key,
                    descriptor,
                    data,
                    merged_digest,
                    _row_checksum("ledger", pass_index, pass_key, descriptor, merged_digest),
                ),
            )
            if prune:
                connection.execute(
                    "DELETE FROM partition_state WHERE pass_index < ?", (pass_index,)
                )
            return True

        def confirm() -> bool:
            found = self.read_ledger().get(pass_index)
            return found is not None and _digest(found.data) == merged_digest

        return self._transact(work, confirm)


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "PARTITION_STORE_FORMAT_VERSION",
    "STORE_KIND",
    "LedgerEntry",
    "ManifestEntry",
    "SQLitePartitionStore",
    "StoredJob",
    "StoredState",
]
