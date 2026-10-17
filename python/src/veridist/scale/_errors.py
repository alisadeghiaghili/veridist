"""Typed failures of the internal mergeable-state namespace.

Programmer errors (a wrong container, dtype or shape) are the built-in
``TypeError`` and ``ValueError``, as elsewhere in the package. Failures that
depend on the data or on a stored state are a :class:`ScaleStateError` with a
closed code. Failures of a data source (a missing column, an unsupported type, a
null, a missing optional dependency, a spool that is required, full, changed or
unusable) are a :class:`ScaleSourceError` with its own closed code. Neither the message
nor the attributes ever carry a data value or a path. Both classes pickle, so a failure
raised in a worker process reaches the parent as the same typed error.
"""

from __future__ import annotations

from enum import StrEnum

from veridist.engine.errors import VeridistError


class ScaleStateErrorCode(StrEnum):
    """Closed, locale-neutral failure codes of the mergeable-state namespace."""

    NON_FINITE_VALUE = "NON_FINITE_VALUE"
    VALUE_OUT_OF_SUPPORT = "VALUE_OUT_OF_SUPPORT"
    OBSERVATION_LIMIT_EXCEEDED = "OBSERVATION_LIMIT_EXCEEDED"
    TOTAL_NOT_REPRESENTABLE = "TOTAL_NOT_REPRESENTABLE"
    INCOMPATIBLE_STATE = "INCOMPATIBLE_STATE"
    STATE_BYTES_INVALID = "STATE_BYTES_INVALID"
    STATE_TAG_MISMATCH = "STATE_TAG_MISMATCH"
    STATE_VERSION_UNSUPPORTED = "STATE_VERSION_UNSUPPORTED"


class ScaleStateError(VeridistError, ValueError):
    """A data-dependent or stored-state failure identified only by its code."""

    def __init__(self, code: ScaleStateErrorCode | str) -> None:
        resolved = ScaleStateErrorCode(code)
        super().__init__(resolved.value)
        self.code = resolved

    def __reduce__(self) -> tuple[type[ScaleStateError], tuple[ScaleStateErrorCode]]:
        return (type(self), (self.code,))


class ScaleSourceErrorCode(StrEnum):
    """Closed, locale-neutral failure codes of the data-source layer."""

    OPTIONAL_DEPENDENCY_MISSING = "OPTIONAL_DEPENDENCY_MISSING"
    COLUMN_NOT_FOUND = "COLUMN_NOT_FOUND"
    UNSUPPORTED_DTYPE = "UNSUPPORTED_DTYPE"
    LOSSY_CAST = "LOSSY_CAST"
    NULL_VALUE = "NULL_VALUE"
    NAN_VALUE = "NAN_VALUE"
    NON_FINITE_VALUE = "NON_FINITE_VALUE"
    SOURCE_NOT_FOUND = "SOURCE_NOT_FOUND"
    SOURCE_UNREADABLE = "SOURCE_UNREADABLE"
    SOURCE_CONSUMED = "SOURCE_CONSUMED"
    PARTITION_CHANGED = "PARTITION_CHANGED"
    SPOOL_REQUIRED = "SPOOL_REQUIRED"
    SPOOL_LIMIT_EXCEEDED = "SPOOL_LIMIT_EXCEEDED"
    SPOOL_CHANGED = "SPOOL_CHANGED"
    SPOOL_UNAVAILABLE = "SPOOL_UNAVAILABLE"


class ScaleSourceError(VeridistError, ValueError):
    """A data-source failure identified by its code and, at most, a fixed hint.

    The hint is a constant sentence that tells the caller what to do (for a
    missing optional dependency, which extra to install; for a source that can be
    read once, that a spool must be requested). It never contains a path, a column
    name or a data value.
    """

    def __init__(self, code: ScaleSourceErrorCode | str, hint: str | None = None) -> None:
        resolved = ScaleSourceErrorCode(code)
        super().__init__(resolved.value if hint is None else f"{resolved.value}: {hint}")
        self.code = resolved
        self.hint = hint

    def __reduce__(
        self,
    ) -> tuple[type[ScaleSourceError], tuple[ScaleSourceErrorCode, str | None]]:
        return (type(self), (self.code, self.hint))


class ScaleCheckpointErrorCode(StrEnum):
    """Closed, locale-neutral failure codes of partition-level checkpoints.

    A failure of a partition whose fingerprint changed is the ``PARTITION_CHANGED`` of
    :class:`ScaleSourceErrorCode`; the codes here are those of the store, of the identity
    of a job and of the partitions it covers.
    """

    SOURCE_REVISION_UNAVAILABLE = "SOURCE_REVISION_UNAVAILABLE"
    INCREMENTAL_UNSUPPORTED = "INCREMENTAL_UNSUPPORTED"
    JOB_EXISTS = "JOB_EXISTS"
    JOB_MISMATCH = "JOB_MISMATCH"
    PARTITION_MISSING = "PARTITION_MISSING"
    PARTITION_ADDED = "PARTITION_ADDED"
    PARTITION_STATE_CONFLICT = "PARTITION_STATE_CONFLICT"
    LEDGER_MISMATCH = "LEDGER_MISMATCH"
    STORE_FORMAT_UNSUPPORTED = "STORE_FORMAT_UNSUPPORTED"
    STORE_DECODE_FAILED = "STORE_DECODE_FAILED"
    STORE_DIGEST_MISMATCH = "STORE_DIGEST_MISMATCH"
    STORE_STORAGE_FAILED = "STORE_STORAGE_FAILED"
    STORE_COMMIT_UNCERTAIN = "STORE_COMMIT_UNCERTAIN"


class ScaleCheckpointError(VeridistError, ValueError):
    """A checkpoint failure identified by its code and, at most, names of components.

    ``components`` names the parts of the identity of a job that differ from the stored
    one (``JOB_MISMATCH``) and is empty otherwise; ``detail`` is the symbolic name of a
    storage error (for example ``SQLITE_BUSY``) when the store has one. Neither the
    message nor an attribute ever carries a path, a partition identifier, a fingerprint,
    a state or a value, and the error pickles.
    """

    def __init__(
        self,
        code: ScaleCheckpointErrorCode | str,
        components: tuple[str, ...] = (),
        detail: str | None = None,
    ) -> None:
        resolved = ScaleCheckpointErrorCode(code)
        text = resolved.value
        if components:
            text += f" ({', '.join(components)})"
        if detail is not None:
            text += f" [{detail}]"
        super().__init__(text)
        self.code = resolved
        self.components = components
        self.detail = detail

    def __reduce__(
        self,
    ) -> tuple[
        type[ScaleCheckpointError],
        tuple[ScaleCheckpointErrorCode, tuple[str, ...], str | None],
    ]:
        return (type(self), (self.code, self.components, self.detail))
