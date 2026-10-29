"""Fixed-state reductions for the exponential lifetime vertical."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from veridist.domain._combinations import fit_unavailable
from veridist.domain.lifetimes import (
    ExactLifetime,
    IntervalCensoredLifetime,
    LeftCensoredLifetime,
    LifetimeObservation,
    RightCensoredLifetime,
)
from veridist.engine.retry import PureReducer


class _ReductionOverflow(Exception):
    """Internal overflow signal carrying only derived aggregate counts."""

    def __init__(self, observation_count: int, event_count: int) -> None:
        self.observation_count = observation_count
        self.event_count = event_count


def _plain_lifetime(observation: object) -> type:
    """The type of an exact or right-censored lifetime without a truncation window.

    The other lifetime kinds, and any kind with a window, are refused with the typed
    ``CapabilityError`` of the fits (the exponential closed form does not take them yet); any
    other object is a ``TypeError``.
    """

    kind = type(observation)
    if kind is ExactLifetime or kind is RightCensoredLifetime:
        if observation.truncation_lower is None and observation.truncation_upper is None:  # type: ignore[attr-defined]
            return kind
        raise fit_unavailable()
    if kind is LeftCensoredLifetime or kind is IntervalCensoredLifetime:
        raise fit_unavailable()
    raise TypeError("observation must be an exact or right-censored lifetime")


def _require_count(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class ExponentialReductionState:
    """O(1) Neumaier-compensated sufficient statistics, with no raw observations."""

    observation_count: int
    event_count: int
    total_time: float
    compensation: float

    def __post_init__(self) -> None:
        _require_count(self.observation_count, "observation_count")
        _require_count(self.event_count, "event_count")
        if self.event_count > self.observation_count:
            raise ValueError("event_count cannot exceed observation_count")
        if not isfinite(self.total_time) or not isfinite(self.compensation):
            raise ValueError("reduction state must remain finite")
        if self.total_time < 0.0:
            raise ValueError("reduction state cannot contain negative lifetime totals")
        if self.compensation < -max(1e-12, self.total_time * 1e-12):
            raise ValueError("reduction compensation is outside the numerical error envelope")
        if self.total_time + self.compensation < 0.0:
            raise ValueError("compensated lifetime total cannot be negative")
        if self.observation_count == 0 and (self.total_time != 0.0 or self.compensation != 0.0):
            raise ValueError("empty reduction state must have zero totals")

    @classmethod
    def empty(cls) -> ExponentialReductionState:
        """Return the neutral fixed-state reduction value."""

        return cls(0, 0, 0.0, 0.0)

    @property
    def summed_time(self) -> float:
        """Return the compensated total, failing if the final addition overflows."""

        value = self.total_time + self.compensation
        if not isfinite(value):
            raise _ReductionOverflow(self.observation_count, self.event_count)
        return value

    def _add_value(
        self, value: float, observation_increment: int, event_increment: int
    ) -> ExponentialReductionState:
        updated_total = self.total_time + value
        if not isfinite(updated_total):
            raise _ReductionOverflow(
                self.observation_count + observation_increment,
                self.event_count + event_increment,
            )
        if abs(self.total_time) >= abs(value):
            updated_compensation = self.compensation + (self.total_time - updated_total + value)
        else:
            updated_compensation = self.compensation + (value - updated_total + self.total_time)
        if not isfinite(updated_compensation):
            raise _ReductionOverflow(
                self.observation_count + observation_increment,
                self.event_count + event_increment,
            )
        return ExponentialReductionState(
            self.observation_count + observation_increment,
            self.event_count + event_increment,
            updated_total,
            updated_compensation,
        )

    def add(self, observation: LifetimeObservation) -> ExponentialReductionState:
        """Return a new state after one validated exact or right-censored observation."""

        kind = _plain_lifetime(observation)
        return self._add_value(float(observation.time), 1, 1 if kind is ExactLifetime else 0)

    def merge(self, other: ExponentialReductionState) -> ExponentialReductionState:
        """Merge two compatible states; partition order has a tolerance-only contract."""

        if type(other) is not ExponentialReductionState:
            raise TypeError("other must be an ExponentialReductionState")
        combined = self._add_value(other.total_time, other.observation_count, other.event_count)
        return combined._add_value(other.compensation, 0, 0)


class ExponentialCheckpointReducer(PureReducer[ExponentialReductionState]):
    """Pure reducer adapter for durable checkpointed exponential chunks.

    State uses hexadecimal float spellings so checkpoint round trips preserve
    the exact binary64 sufficient statistics without retaining raw rows.
    Payloads are canonical JSON arrays of ``[time, event_observed]`` pairs.
    """

    reducer_id = "exponential-reduction-v1"
    accumulator_schema = "exponential-reduction-v1"

    def decode_state(self, state: bytes) -> ExponentialReductionState:
        value = json.loads(state.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError("reduction state must be an object")
        return ExponentialReductionState(
            int(value["observation_count"]),
            int(value["event_count"]),
            float.fromhex(value["total_time"]),
            float.fromhex(value["compensation"]),
        )

    def reduce(
        self,
        accumulator: ExponentialReductionState,
        payload: bytes,
    ) -> ExponentialReductionState:
        rows = json.loads(payload.decode("utf-8"))
        if not isinstance(rows, list):
            raise ValueError("reduction payload must be an array")
        state = accumulator
        for row in rows:
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError("reduction row must contain time and event flag")
            time, event_observed = row
            observation: LifetimeObservation = (
                ExactLifetime(time) if event_observed else RightCensoredLifetime(time)
            )
            state = state.add(observation)
        return state

    def encode_state(self, accumulator: ExponentialReductionState) -> bytes:
        if type(accumulator) is not ExponentialReductionState:
            raise TypeError("accumulator must be ExponentialReductionState")
        return json.dumps(
            {
                "compensation": accumulator.compensation.hex(),
                "event_count": accumulator.event_count,
                "observation_count": accumulator.observation_count,
                "total_time": accumulator.total_time.hex(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def reduce_exponential_chunks(
    chunks: Iterable[Iterable[LifetimeObservation]],
) -> ExponentialReductionState:
    """Reduce ragged chunks in their caller-declared canonical input order."""

    if not isinstance(chunks, Iterable):
        raise TypeError("chunks must be an iterable of observation iterables")
    state = ExponentialReductionState.empty()
    overflow_counts: tuple[int, int] | None = None
    for chunk in chunks:
        if not isinstance(chunk, Iterable):
            raise TypeError("each chunk must be an iterable of lifetime observations")
        for observation in chunk:
            if overflow_counts is None:
                try:
                    state = state.add(observation)
                    continue
                except _ReductionOverflow as error:
                    overflow_counts = (error.observation_count, error.event_count)
                    continue
            events = 1 if _plain_lifetime(observation) is ExactLifetime else 0
            assert overflow_counts is not None
            overflow_counts = (overflow_counts[0] + 1, overflow_counts[1] + events)
    if overflow_counts is not None:
        raise _ReductionOverflow(*overflow_counts)
    return state


__all__ = ["ExponentialCheckpointReducer", "ExponentialReductionState", "reduce_exponential_chunks"]
