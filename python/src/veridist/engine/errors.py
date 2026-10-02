"""Typed, localization-independent failures for engine contracts."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import Enum, StrEnum
from math import isfinite
from types import MappingProxyType
from typing import cast


class FailureCode(StrEnum):
    """Stable machine-readable failure codes exposed by the execution engine."""

    ACCUMULATOR_SCHEMA_MISMATCH = "ACCUMULATOR_SCHEMA_MISMATCH"
    BUFFER_TIMEOUT = "BUFFER_TIMEOUT"
    CANCELLED = "CANCELLED"
    CHECKPOINT_CHECKSUM_MISMATCH = "CHECKPOINT_CHECKSUM_MISMATCH"
    CHECKPOINT_CONFLICT = "CHECKPOINT_CONFLICT"
    CHECKPOINT_DECODE_FAILED = "CHECKPOINT_DECODE_FAILED"
    CHECKPOINT_FORMAT_UNSUPPORTED = "CHECKPOINT_FORMAT_UNSUPPORTED"
    CHECKPOINT_ALREADY_EXISTS = "CHECKPOINT_ALREADY_EXISTS"
    CHECKPOINT_NOT_FOUND = "CHECKPOINT_NOT_FOUND"
    CHECKPOINT_REQUIRED = "CHECKPOINT_REQUIRED"
    CHECKPOINT_SCHEMA_MISMATCH = "CHECKPOINT_SCHEMA_MISMATCH"
    CHECKPOINT_SOURCE_ID_MISMATCH = "CHECKPOINT_SOURCE_ID_MISMATCH"
    CHECKPOINT_STORAGE_FAILED = "CHECKPOINT_STORAGE_FAILED"
    CHUNK_TOO_LARGE = "CHUNK_TOO_LARGE"
    DUPLICATE_CHUNK = "DUPLICATE_CHUNK"
    INVALID_RETAINED_BYTES = "INVALID_RETAINED_BYTES"
    MISSING_CHUNK = "MISSING_CHUNK"
    MISSING_RANGE_UNKNOWN = "MISSING_RANGE_UNKNOWN"
    OPERATION_DIGEST_CONFLICT = "OPERATION_DIGEST_CONFLICT"
    OUT_OF_ORDER_CHUNK = "OUT_OF_ORDER_CHUNK"
    PASS_BUDGET_EXCEEDED = "PASS_BUDGET_EXCEEDED"
    PAYLOAD_CHECKSUM_MISMATCH = "PAYLOAD_CHECKSUM_MISMATCH"
    PLAN_MISMATCH = "PLAN_MISMATCH"
    RANGE_MISMATCH = "RANGE_MISMATCH"
    REDUCER_FAILURE = "REDUCER_FAILURE"
    REDUCER_MISMATCH = "REDUCER_MISMATCH"
    RETRY_EXHAUSTED = "RETRY_EXHAUSTED"
    RETRY_NOT_ADMISSIBLE = "RETRY_NOT_ADMISSIBLE"
    SINK_FAILURE = "SINK_FAILURE"
    SOURCE_DECODE_FAILED = "SOURCE_DECODE_FAILED"
    SOURCE_ID_MISMATCH = "SOURCE_ID_MISMATCH"
    SOURCE_MISMATCH = "SOURCE_MISMATCH"
    SOURCE_OPEN_FAILED = "SOURCE_OPEN_FAILED"
    SOURCE_ROW_INVALID = "SOURCE_ROW_INVALID"
    SOURCE_REVISION_MISMATCH = "SOURCE_REVISION_MISMATCH"
    SOURCE_REVISION_UNAVAILABLE = "SOURCE_REVISION_UNAVAILABLE"
    SOURCE_SCHEMA_INVALID = "SOURCE_SCHEMA_INVALID"
    SOURCE_SCHEMA_MISMATCH = "SOURCE_SCHEMA_MISMATCH"
    SPOOL_REQUIRED = "SPOOL_REQUIRED"


_SENSITIVE_CONTEXT_KEY_PARTS = frozenset(
    {
        "checksum",
        "credential",
        "digest",
        "dsn",
        "exception",
        "message",
        "password",
        "path",
        "payload",
        "query",
        "revision",
        "token",
        "traceback",
        "uri",
    }
)


# Strings are shown in exception text only when they look like a code-defined
# token (a stage, operation or library error name). Anything else - a path, a
# URI, free text - is replaced, because values are not otherwise screened.
_DISPLAYABLE_TEXT = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}")
_REDACTED_TEXT = "<redacted>"


def _display_context_value(value: object) -> str:
    """Render one frozen context value for exception text without leaking data."""

    if value is None or type(value) in {bool, int, float}:
        return str(value)
    if type(value) is str:
        return value if _DISPLAYABLE_TEXT.fullmatch(value) else _REDACTED_TEXT
    if type(value) is tuple:
        return "(" + ", ".join(_display_context_value(item) for item in value) + ")"
    if isinstance(value, Mapping):
        return "{" + _display_pairs(value) + "}"
    return _REDACTED_TEXT


def _display_pairs(context: Mapping[str, object]) -> str:
    return ", ".join(
        f"{key}={_display_context_value(context[key])}" for key in sorted(context)
    )


def _freeze_context_value(value: object) -> object:
    """Recursively freeze a context value, rejecting unsafe shapes and keys.

    This is a **key-name allowlist**, not data redaction: a mapping key is
    rejected only when one of its ``_``-separated parts matches
    :data:`_SENSITIVE_CONTEXT_KEY_PARTS` exactly (so ``file_path`` is
    rejected but ``filepath``, having one undivided part, is not). Values are
    never inspected, so a caller that stores a sensitive value under an
    unlisted key (or an unsplit key) is not protected by this function; the
    callers in this package are expected to use only narrow, reviewed key
    names (see ``tests/contract/test_context_key_redaction.py``).
    """

    if isinstance(value, Enum):
        return _freeze_context_value(value.value)
    if type(value) is float:
        if not isfinite(value):
            raise TypeError("failure context floats must be finite")
        return value
    if type(value) in {str, int, bool, type(None)}:
        return value
    if type(value) is tuple:
        return tuple(_freeze_context_value(item) for item in value)
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("failure context mapping keys must be strings")
            key_parts = frozenset(key.casefold().split("_"))
            if key_parts & _SENSITIVE_CONTEXT_KEY_PARTS:
                raise TypeError("failure context contains a sensitive key")
            frozen[key] = _freeze_context_value(item)
        return MappingProxyType(frozen)
    raise TypeError("failure context contains an unsafe value type")


class VeridistError(Exception):
    """Common base of every exception class that veridist defines.

    Catch this to handle any typed veridist failure without enumerating the
    individual classes. Built-in exceptions raised for programmer errors
    (``TypeError``, ``ValueError``) are deliberately not wrapped.
    """


class EngineContractError(VeridistError):
    """A typed engine failure with immutable diagnostic context."""

    def __init__(self, code: FailureCode, context: Mapping[str, object] | None = None) -> None:
        if not isinstance(code, FailureCode):
            raise TypeError("code must be a FailureCode")
        if context is not None and not isinstance(context, Mapping):
            raise TypeError("failure context must be a mapping")
        super().__init__(code.value)
        self.code = code
        self.context = cast(Mapping[str, object], _freeze_context_value(context or {}))

    def __str__(self) -> str:
        if not self.context:
            return self.code.value
        return f"{self.code.value} ({_display_pairs(self.context)})"

    def __repr__(self) -> str:
        if not self.context:
            return f"{type(self).__name__}(code={self.code.value})"
        pairs = _display_pairs(self.context)
        return f"{type(self).__name__}(code={self.code.value}, context={{{pairs}}})"


class CapabilityCode(StrEnum):
    """Stable capability cells deliberately not admitted by the current API."""

    ANALYTIC_WEIGHTS_UNSUPPORTED = "ANALYTIC_WEIGHTS_UNSUPPORTED"
    INTERVAL_CENSORING_UNSUPPORTED = "INTERVAL_CENSORING_UNSUPPORTED"
    LEFT_CENSORING_UNSUPPORTED = "LEFT_CENSORING_UNSUPPORTED"
    TRUNCATION_UNSUPPORTED = "TRUNCATION_UNSUPPORTED"


class CapabilityError(VeridistError):
    """Typed request for a declared but unsupported statistical capability."""

    def __init__(self, code: CapabilityCode) -> None:
        if type(code) is not CapabilityCode:
            raise TypeError("code must be a CapabilityCode")
        super().__init__(code.value)
        self.code = code


def safe_exception_type(exc: Exception) -> str:
    """Return an allowlist-safe type label without exposing exception text."""

    return type(exc).__name__ if type(exc).__module__ == "builtins" else "Exception"
