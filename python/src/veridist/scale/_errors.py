"""Typed failures of the internal mergeable-state namespace.

Programmer errors (a wrong container, dtype or shape) are the built-in
``TypeError`` and ``ValueError``, as elsewhere in the package. Failures that
depend on the data or on a stored state are a :class:`ScaleStateError` with a
closed code. Neither the message nor the attributes ever carry a data value.
"""

from __future__ import annotations

from enum import StrEnum

from veridist.engine.errors import VeridistError


class ScaleStateErrorCode(StrEnum):
    """Closed, locale-neutral failure codes of the mergeable-state namespace."""

    NON_FINITE_VALUE = "NON_FINITE_VALUE"
    VALUE_OUT_OF_SUPPORT = "VALUE_OUT_OF_SUPPORT"
    DERIVED_VALUE_NOT_FINITE = "DERIVED_VALUE_NOT_FINITE"
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
