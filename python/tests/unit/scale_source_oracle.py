"""An independent encoding of the documented fingerprint record, and a trace state.

``reference_digest`` follows the written rule (a type tag, an 8-byte big-endian
length and the payload, with a fixed header and item count) without importing the
production encoder, so the production bytes are pinned by an outside description.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from veridist.scale._sources import DEFAULT_MAX_ROWS, FingerprintLevel, Partition


def _encode(item: object) -> bytes:
    if item is None:
        return b"N"
    if isinstance(item, int):
        payload = str(item).encode("ascii")
        return b"I" + len(payload).to_bytes(8, "big") + payload
    if isinstance(item, str):
        payload = item.encode("utf-8")
        return b"S" + len(payload).to_bytes(8, "big") + payload
    if isinstance(item, bytes):
        return b"B" + len(item).to_bytes(8, "big") + item
    assert isinstance(item, tuple)
    return b"T" + len(item).to_bytes(8, "big") + b"".join(_encode(member) for member in item)


def reference_digest(*items: object) -> str:
    digest = hashlib.sha256()
    digest.update(b"veridist.scale.fingerprint.v1")
    digest.update(len(items).to_bytes(8, "big"))
    for item in items:
        digest.update(_encode(item))
    return "sha256:" + digest.hexdigest()


def numpy_fingerprint(values: np.ndarray) -> str:
    """The documented fingerprint of an array source, from first principles."""

    content = hashlib.sha256(values.tobytes()).hexdigest()
    return reference_digest("numpy", values.dtype.str, (int(values.shape[0]),), content)


def unresumable(identity: str, ordinal: int = 0, rows: int | None = None) -> Partition:
    return Partition(identity, ordinal, rows, None, FingerprintLevel.NONE)


@dataclass(frozen=True)
class TraceState:
    """A state that records how it was built, to pin the order of updates and merges."""

    events: tuple[tuple[str, int], ...] = ()

    @classmethod
    def empty(cls) -> TraceState:
        return cls()

    def update(self, batch: object) -> TraceState:
        assert isinstance(batch, np.ndarray)
        return TraceState(self.events + (("update", int(batch.shape[0])),))

    def merge(self, other: TraceState) -> TraceState:
        return TraceState(self.events + (("merge", len(other.events)),) + other.events)


class ListSource:
    """A source over in-memory batches that records the ``max_rows`` it was asked for."""

    def __init__(self, layout: list[list[np.ndarray]]) -> None:
        self.layout = layout
        self.max_rows_seen: list[int] = []
        self.partition_calls = 0

    def partitions(self) -> tuple[Partition, ...]:
        self.partition_calls += 1
        return tuple(unresumable(f"p{index}", index) for index in range(len(self.layout)))

    def batches(self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS) -> Any:
        self.max_rows_seen.append(max_rows)
        return iter(self.layout[partition.ordinal])
