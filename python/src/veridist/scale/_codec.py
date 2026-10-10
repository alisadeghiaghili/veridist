"""Canonical, versioned byte framing of exact integer states.

A frame is::

    magic (4) | tag length (1) | tag (ASCII) | schema version (2, big endian)
    | observation count (8, big endian) | integer count (1)
    | per integer: sign (1: 0 or 1) | magnitude length (4) | magnitude (big endian)
    | SHA-256 of every preceding byte (32)

A magnitude has no leading zero byte, zero is the empty magnitude with sign 0,
and a frame has no trailing bytes, so one state has exactly one encoding.
Decoding checks the magic, the tag and the schema version first (an unknown
version is refused, never migrated), then the checksum, then the structure.
"""

from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from hmac import compare_digest
from typing import Final

from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode

MAGIC: Final = b"VDSS"
CHECKSUM_BYTES: Final = 32
_COUNT_BYTES: Final = 8
_VERSION_BYTES: Final = 2
_LENGTH_BYTES: Final = 4


def _invalid() -> ScaleStateError:
    return ScaleStateError(ScaleStateErrorCode.STATE_BYTES_INVALID)


def encode_frame(tag: str, version: int, count: int, integers: Sequence[int]) -> bytes:
    """Encode trusted state facts; callers have already bounded every value."""

    tag_bytes = tag.encode("ascii")
    body = bytearray(MAGIC)
    body.append(len(tag_bytes))
    body += tag_bytes
    body += version.to_bytes(_VERSION_BYTES, "big")
    body += count.to_bytes(_COUNT_BYTES, "big")
    body.append(len(integers))
    for value in integers:
        size = abs(value)
        magnitude = size.to_bytes((size.bit_length() + 7) // 8, "big")
        body.append(1 if value < 0 else 0)
        body += len(magnitude).to_bytes(_LENGTH_BYTES, "big")
        body += magnitude
    return bytes(body) + sha256(body).digest()


class _Cursor:
    """A bounded reader over the checksummed part of a frame."""

    def __init__(self, data: bytes, position: int, end: int) -> None:
        self._data = data
        self._position = position
        self._end = end

    def take(self, size: int) -> bytes:
        stop = self._position + size
        if stop > self._end:
            raise _invalid()
        chunk = self._data[self._position : stop]
        self._position = stop
        return chunk

    def number(self, size: int) -> int:
        return int.from_bytes(self.take(size), "big")

    def finished(self) -> bool:
        return self._position == self._end


def decode_frame(
    data: object, *, tag: str, version: int, integer_count: int
) -> tuple[int, tuple[int, ...]]:
    """Strictly decode one frame into ``(observation count, integers)``.

    ``TypeError`` for a non-bytes argument; a :class:`ScaleStateError` with
    ``STATE_BYTES_INVALID``, ``STATE_TAG_MISMATCH`` or ``STATE_VERSION_UNSUPPORTED``
    for every other refusal. No message carries decoded content.
    """

    if not isinstance(data, bytes | bytearray):
        raise TypeError("state data must be bytes")
    raw = bytes(data)
    tag_bytes = tag.encode("ascii")
    tag_field = bytes([len(tag_bytes)]) + tag_bytes
    tag_end = len(MAGIC) + len(tag_field)
    header = tag_end + _VERSION_BYTES
    if raw[: len(MAGIC)] != MAGIC or len(raw) < header + CHECKSUM_BYTES:
        raise _invalid()
    if raw[len(MAGIC) : tag_end] != tag_field:
        raise ScaleStateError(ScaleStateErrorCode.STATE_TAG_MISMATCH)
    if raw[tag_end:header] != version.to_bytes(_VERSION_BYTES, "big"):
        raise ScaleStateError(ScaleStateErrorCode.STATE_VERSION_UNSUPPORTED)
    end = len(raw) - CHECKSUM_BYTES
    if not compare_digest(sha256(raw[:end]).digest(), raw[end:]):
        raise _invalid()
    cursor = _Cursor(raw, header, end)
    count = cursor.number(_COUNT_BYTES)
    if cursor.number(1) != integer_count:
        raise _invalid()
    integers: list[int] = []
    for _ in range(integer_count):
        sign = cursor.number(1)
        magnitude = cursor.take(cursor.number(_LENGTH_BYTES))
        if sign > 1 or magnitude[:1] == b"\x00" or (sign == 1 and not magnitude):
            raise _invalid()
        value = int.from_bytes(magnitude, "big")
        integers.append(-value if sign else value)
    if not cursor.finished():
        raise _invalid()
    return count, tuple(integers)
