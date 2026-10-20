"""The exact sum ``S0 = sum exp(k * w)`` of every candidate of one inference pass.

A profile-likelihood pass needs, for each parameter value ``k`` it is given, the sum of
``exp(k * w)`` over the centred observations ``w <= 0`` of the fit, and nothing else
(:mod:`veridist.scale._profile_math`). One formula serves every element and every candidate,
``numpy.exp(k * w)``, exactly as the passes of the fit evaluate it
(:class:`~veridist.scale._multi_pass.ProfileAccumulator`), so that the sum at the parameter a fit
ended on is bit for bit the one the fit recorded. The sums are exact
(:mod:`veridist.scale._accumulator`), so they depend on the multiset of the rows and the
candidates only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from veridist.scale._accumulator import ExactSum
from veridist.scale._codec import decode_frame, encode_frame
from veridist.scale._kernels import float64_batch, numpy_module
from veridist.scale._multi_pass import Kernel
from veridist.scale._sources import BatchReader, Partition

POWER_SUM_TAG: Final = "scale.power_sum_accumulator"
POWER_SUM_SCHEMA_VERSION: Final = 1
MAX_POWER_SUM_CANDIDATES: Final = 255
"""A frame holds at most 255 integers, one per candidate."""


class PowerSumAccumulator:
    """The exact sum of ``exp(k * w)`` for each candidate ``k``, over some rows."""

    __slots__ = ("count", "sums")

    def __init__(self, candidates: int) -> None:
        if candidates > MAX_POWER_SUM_CANDIDATES:
            raise ValueError("too many candidates")
        self.count = 0
        self.sums = tuple(ExactSum() for _ in range(candidates))

    def update(self, centred: Any, parameters: tuple[float, ...]) -> None:
        """Add one batch of centred observations for every candidate in ``parameters``."""

        np = numpy_module()
        with np.errstate(all="ignore"):
            for parameter, total in zip(parameters, self.sums, strict=True):
                total.update(np.exp(parameter * centred))
        self.count += int(centred.shape[0])

    def merge(self, other: PowerSumAccumulator) -> PowerSumAccumulator:
        """A new accumulator holding the exact sums of both."""

        merged = PowerSumAccumulator(len(self.sums))
        merged.count = self.count + other.count
        merged.sums = tuple(
            left.merge(right) for left, right in zip(self.sums, other.sums, strict=True)
        )
        return merged

    def to_bytes(self) -> bytes:
        """Canonical bytes: tag, schema version, the row count and each exact sum in units."""

        return encode_frame(
            POWER_SUM_TAG,
            POWER_SUM_SCHEMA_VERSION,
            self.count,
            tuple(total.total_units() for total in self.sums),
        )

    @classmethod
    def from_bytes(cls, data: object, candidates: int) -> PowerSumAccumulator:
        """Restore the accumulator of ``candidates`` candidates from :meth:`to_bytes` output.

        ``ScaleStateError`` with ``STATE_BYTES_INVALID``, ``STATE_TAG_MISMATCH`` or
        ``STATE_VERSION_UNSUPPORTED``, as for every state.
        """

        count, units = decode_frame(
            data,
            tag=POWER_SUM_TAG,
            version=POWER_SUM_SCHEMA_VERSION,
            integer_count=candidates,
        )
        restored = cls(candidates)
        restored.count = count
        restored.sums = tuple(ExactSum.from_units(count, value) for value in units)
        return restored

    def totals(self) -> list[float]:
        """The sum of every candidate, each rounded once."""

        return [total.total() for total in self.sums]


@dataclass(frozen=True, slots=True)
class PowerSumJob:
    """Fold a partition into the power sums of every candidate of one pass."""

    kernel: Kernel
    parameters: tuple[float, ...]

    def run(self, source: BatchReader, partition: Partition, max_rows: int) -> bytes:
        part = PowerSumAccumulator(len(self.parameters))
        for batch in source.batches(partition, max_rows=max_rows):
            part.update(self.kernel(float64_batch(batch)), self.parameters)
        return part.to_bytes()

    def restore(self, data: bytes) -> PowerSumAccumulator:
        return PowerSumAccumulator.from_bytes(data, len(self.parameters))

    def identity(self) -> PowerSumAccumulator:
        return PowerSumAccumulator(len(self.parameters))

    def checkpoint_key(self) -> tuple[object, ...]:
        return (
            "power_sums",
            POWER_SUM_TAG,
            POWER_SUM_SCHEMA_VERSION,
            *self.kernel.key(),
            tuple(parameter.hex() for parameter in self.parameters),
        )


__all__ = [
    "MAX_POWER_SUM_CANDIDATES",
    "POWER_SUM_SCHEMA_VERSION",
    "POWER_SUM_TAG",
    "PowerSumAccumulator",
    "PowerSumJob",
]
