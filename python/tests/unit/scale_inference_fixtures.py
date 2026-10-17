"""Comparisons shared by the tests of inference from a scale fit."""

from __future__ import annotations

import struct
from dataclasses import replace
from typing import Any

from veridist.families.uncertainty import DerivedEstimate
from veridist.scale._gof import LargeNGof
from veridist.scale._profile_intervals import ProfileIntervals, ProfileUnavailable


def pack(value: float) -> bytes:
    return struct.pack("<d", value)


def profile_bits(out: ProfileIntervals) -> tuple[Any, ...]:
    """Every number of a profile result as binary64 patterns, and the pass record."""

    numbers: list[bytes] = [pack(out.level), pack(out.deviance)]
    for interval in out.parameters.values():
        numbers.extend((pack(interval.lower), pack(interval.upper)))
    for name, answer in out.derived.items():
        if isinstance(answer, DerivedEstimate):
            numbers.extend(pack(v) for v in (answer.estimate, answer.lower, answer.upper))
        else:
            assert isinstance(answer, ProfileUnavailable)
            numbers.append(name.encode() + answer.reason.value.encode())
    resolution = out.resolution
    numbers.extend(
        pack(v)
        for v in (resolution.surrogate_gap, resolution.confirmation_residual, resolution.tolerance)
    )
    numbers.append(bytes([resolution.surrogate_degree]))
    assert out.scale.passes is not None
    return (tuple(numbers), out.scale.passes.records, out.scale.state)


def gof_bits(out: LargeNGof) -> tuple[Any, ...]:
    return (
        out.rows,
        out.bins,
        pack(out.ks_lower),
        pack(out.ks_upper),
        pack(out.tv_lower),
        pack(out.tv_upper),
        out.strip_rows,
        out.counts_digest,
        out.decision,
    )


def without_execution(out: Any) -> Any:
    """The scale provenance of ``out`` without the records of execution, checkpoint and spool."""

    return replace(out.scale, execution=None, checkpoint=None, spool=None)
