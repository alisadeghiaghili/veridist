"""Exact-state streaming reduction of finite scalar binary64 log densities."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from hashlib import sha256
from math import isfinite
from typing import Final, TypeAlias

from veridist.engine.streaming import StreamSource, iter_stream
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.statistics.log_density import (
    LogDensityErrorCode,
    LogDensityFailure,
    LogDensitySuccess,
    _evaluate_validated_log_density,
)

MAX_OBSERVATION_COUNT: Final = (1 << 64) - 1
"""The explicit unsigned-64 observation cap for one reducer state."""

_UNITS_DENOMINATOR: Final = 1 << 1074
_MAX_CONTRIBUTION_UNITS: Final = ((1 << 53) - 1) << 2045
MAX_TOTAL_UNITS: Final = MAX_OBSERVATION_COUNT * _MAX_CONTRIBUTION_UNITS
"""Maximum absolute exact unit total under the declared count cap (2162 bits)."""


class LogLikelihoodErrorCode(StrEnum):
    """Closed, locale-neutral streaming log-likelihood failure codes."""

    SCALAR_EVALUATION_FAILURE = "scalar_evaluation_failure"
    OBSERVATION_LIMIT_EXCEEDED = "observation_limit_exceeded"
    FINAL_TOTAL_NOT_REPRESENTABLE = "final_total_not_representable"


@dataclass(frozen=True, slots=True, init=False)
class LogLikelihoodState:
    """A compatible exact sum in subnormal binary64 units, with no observations."""

    family: FamilyId
    parameter_fingerprint: str
    observation_count: int
    total_units: int
    _canonical_identity: tuple[str, ...] = field(repr=False, compare=True)

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("LogLikelihoodState construction is closed; use empty or restore")

    @classmethod
    def _create(
        cls, family: FamilyId, fingerprint: str, count: int, units: int, identity: tuple[str, ...]
    ) -> LogLikelihoodState:
        state = object.__new__(cls)
        object.__setattr__(state, "family", family)
        object.__setattr__(state, "parameter_fingerprint", fingerprint)
        object.__setattr__(state, "observation_count", count)
        object.__setattr__(state, "total_units", units)
        object.__setattr__(state, "_canonical_identity", identity)
        state.__post_init__()
        return state

    def __post_init__(self) -> None:
        if type(self.family) is not FamilyId:
            raise TypeError("family must be a FamilyId")
        _validate_fingerprint(self.parameter_fingerprint)
        if type(self.observation_count) is not int:
            raise TypeError("observation_count must be a built-in integer")
        if not 0 <= self.observation_count <= MAX_OBSERVATION_COUNT:
            raise ValueError("observation_count is outside the unsigned-64 reducer bound")
        if type(self.total_units) is not int:
            raise TypeError("total_units must be a built-in integer")
        maximum_for_count = self.observation_count * _MAX_CONTRIBUTION_UNITS
        if abs(self.total_units) > maximum_for_count:
            raise ValueError("total_units exceeds the declared count-specific state bound")

    @classmethod
    def empty(cls, family: FamilyId, /, **parameters: object) -> LogLikelihoodState:
        """Create the neutral state after one canonical parameter validation."""

        validated = _validate_family_and_parameters(family, parameters)
        identity, fingerprint = _identity_and_fingerprint(family, validated)
        return cls._create(family, fingerprint, 0, 0, identity)

    @classmethod
    def restore(
        cls, family: FamilyId, /, observation_count: int, total_units: int, **parameters: object
    ) -> LogLikelihoodState:
        validated = _validate_family_and_parameters(family, parameters)
        identity, fingerprint = _identity_and_fingerprint(family, validated)
        return cls._create(family, fingerprint, observation_count, total_units, identity)

    def add_log_density(self, log_density: float) -> LogLikelihoodState:
        """Add one successful finite binary64 scalar evaluator output exactly."""

        if type(log_density) is not float or not isfinite(log_density):
            raise ValueError("log_density must be a finite built-in float")
        if self.observation_count == MAX_OBSERVATION_COUNT:
            raise _ObservationLimitExceeded
        numerator, denominator = log_density.as_integer_ratio()
        units = numerator * (_UNITS_DENOMINATOR // denominator)
        return LogLikelihoodState._create(
            self.family,
            self.parameter_fingerprint,
            self.observation_count + 1,
            self.total_units + units,
            self._canonical_identity,
        )

    def merge(self, other: LogLikelihoodState) -> LogLikelihoodState:
        """Merge compatible exact states; integer addition is order-independent."""

        if type(other) is not LogLikelihoodState:
            raise TypeError("other must be a LogLikelihoodState")
        if self.family is not other.family:
            raise ValueError("cannot merge states for different families")
        if self._canonical_identity != other._canonical_identity:
            raise ValueError("cannot merge states with different canonical parameters")
        if self.observation_count > MAX_OBSERVATION_COUNT - other.observation_count:
            raise _ObservationLimitExceeded
        return LogLikelihoodState._create(
            self.family,
            self.parameter_fingerprint,
            self.observation_count + other.observation_count,
            self.total_units + other.total_units,
            self._canonical_identity,
        )

    def finalize(self) -> float:
        """Correctly round the exact rational unit total once to binary64."""

        try:
            result = float(Fraction(self.total_units, _UNITS_DENOMINATOR))
        except OverflowError as error:
            raise _FinalTotalNotRepresentable from error
        if not isfinite(result):
            raise _FinalTotalNotRepresentable
        return result


@dataclass(frozen=True, slots=True)
class LogLikelihoodSuccess:
    """A finite final total with only compatibility and count facts retained."""

    family: FamilyId
    parameter_fingerprint: str
    observation_count: int
    total_log_likelihood: float

    def __post_init__(self) -> None:
        if type(self.family) is not FamilyId:
            raise TypeError("family must be a FamilyId")
        _validate_fingerprint(self.parameter_fingerprint)
        if (
            type(self.observation_count) is not int
            or not 0 <= self.observation_count <= MAX_OBSERVATION_COUNT
        ):
            raise ValueError("observation_count is outside the unsigned-64 reducer bound")
        if type(self.total_log_likelihood) is not float or not isfinite(self.total_log_likelihood):
            raise ValueError("total_log_likelihood must be a finite built-in float")

    def to_json(self) -> str:
        """Serialize stable, locale-neutral success facts without raw observations."""

        return json.dumps(
            {
                "family": self.family.value,
                "observation_count": self.observation_count,
                "outcome": "success",
                "total_log_likelihood": self.total_log_likelihood,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class LogLikelihoodFailure:
    """A closed failure with no partial total or raw observation value."""

    family: FamilyId
    code: LogLikelihoodErrorCode
    processed_count: int
    scalar_error_code: LogDensityErrorCode | None

    def __post_init__(self) -> None:
        if type(self.family) is not FamilyId:
            raise TypeError("family must be a FamilyId")
        if type(self.code) is not LogLikelihoodErrorCode:
            raise TypeError("code must be a LogLikelihoodErrorCode")
        if (
            type(self.processed_count) is not int
            or not 0 <= self.processed_count <= MAX_OBSERVATION_COUNT
        ):
            raise ValueError("processed_count is outside the unsigned-64 reducer bound")
        if self.code is LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE:
            if type(self.scalar_error_code) is not LogDensityErrorCode:
                raise ValueError("scalar_error_code is required for scalar evaluation failures")
        elif self.scalar_error_code is not None:
            raise ValueError("scalar_error_code is only valid for scalar evaluation failures")

    def to_json(self) -> str:
        """Serialize only closed failure facts; count is incomplete on early failure."""

        return json.dumps(
            {
                "code": self.code.value,
                "family": self.family.value,
                "outcome": "failure",
                "processed_count": self.processed_count,
                "scalar_error_code": (
                    None if self.scalar_error_code is None else self.scalar_error_code.value
                ),
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


LogLikelihoodResult: TypeAlias = LogLikelihoodSuccess | LogLikelihoodFailure


class _ObservationLimitExceeded(Exception):
    """Internal count-boundary signal without observation data."""


class _FinalTotalNotRepresentable(Exception):
    """Internal finalization-boundary signal without the exact total."""


@dataclass(slots=True)
class _ExactAccumulator:
    """Trusted local hot-path accumulator; public state remains immutable."""

    observation_count: int = 0
    total_units: int = 0

    def add(self, log_density: float) -> None:
        if self.observation_count == MAX_OBSERVATION_COUNT:
            raise _ObservationLimitExceeded
        numerator, denominator = log_density.as_integer_ratio()
        self.observation_count += 1
        self.total_units += numerator * (_UNITS_DENOMINATOR // denominator)


def reduce_log_likelihood_chunks(
    family: FamilyId,
    chunks: StreamSource[Iterable[object]] | Iterable[Iterable[object]],
    /,
    **parameters: object,
) -> LogLikelihoodResult:
    """Reduce ragged chunks in one pass, without materializing source or chunks.

    Parameters are validated before iteration.  `processed_count` on a failure
    counts only successful observations preceding the terminal failure and is
    deliberately not a complete-input count.

    ``chunks`` is consumed once. An already exhausted one-shot iterator (for
    example a generator iterated earlier) is indistinguishable from an empty
    input, so it silently reduces to a success with ``observation_count == 0``
    and a total of ``0.0`` rather than raising. To enforce single-pass
    semantics (a second acquisition raises ``PASS_BUDGET_EXCEEDED``), pass the
    chunks through :class:`~veridist.engine.streaming.IterableDataSource`.
    """

    validated = _validate_family_and_parameters(family, parameters)
    identity, fingerprint = _identity_and_fingerprint(family, validated)
    accumulator = _ExactAccumulator()
    for chunk in iter_stream(chunks):
        if not isinstance(chunk, Iterable):
            raise TypeError("each chunk must be an iterable of observations")
        for observation in chunk:
            try:
                evaluated = _evaluate_validated_log_density(family, validated, observation)
                if type(evaluated) is LogDensityFailure:
                    return LogLikelihoodFailure(
                        family,
                        LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE,
                        accumulator.observation_count,
                        evaluated.code,
                    )
                assert isinstance(evaluated, LogDensitySuccess)
                accumulator.add(evaluated.log_density)
            except _ObservationLimitExceeded:
                return LogLikelihoodFailure(
                    family,
                    LogLikelihoodErrorCode.OBSERVATION_LIMIT_EXCEEDED,
                    accumulator.observation_count,
                    None,
                )
    state = LogLikelihoodState._create(
        family, fingerprint, accumulator.observation_count, accumulator.total_units, identity
    )
    try:
        total = state.finalize()
    except _FinalTotalNotRepresentable:
        return LogLikelihoodFailure(
            family,
            LogLikelihoodErrorCode.FINAL_TOTAL_NOT_REPRESENTABLE,
            state.observation_count,
            None,
        )
    return LogLikelihoodSuccess(family, fingerprint, state.observation_count, total)


def _validate_family_and_parameters(
    family: FamilyId, parameters: Mapping[str, object]
) -> Mapping[str, float]:
    if type(family) is not FamilyId:
        raise TypeError("family must be a FamilyId")
    return FAMILY_REGISTRY.families[family].validate_parameters(**parameters)


def _identity_and_fingerprint(
    family: FamilyId, validated: Mapping[str, float]
) -> tuple[tuple[str, ...], str]:
    """Hash family plus registry-ordered, normalized canonical parameter bytes."""

    specification = FAMILY_REGISTRY.families[family]
    encoded: list[str] = []
    for parameter in specification.parameters:
        value = validated[parameter.name]
        normalized = 0.0 if value == 0.0 else value
        encoded.append(normalized.hex())
    identity = tuple(encoded)
    payload = "veridist.log_likelihood.v1\0" + family.value + "\0" + "\0".join(identity)
    return identity, sha256(payload.encode("ascii")).hexdigest()


def _validate_fingerprint(fingerprint: object) -> None:
    if type(fingerprint) is not str or len(fingerprint) != 64:
        raise TypeError("parameter_fingerprint must be a SHA-256 hex string")
    if any(character not in "0123456789abcdef" for character in fingerprint):
        raise ValueError("parameter_fingerprint must be lowercase hexadecimal")


__all__ = [
    "MAX_OBSERVATION_COUNT",
    "MAX_TOTAL_UNITS",
    "LogLikelihoodErrorCode",
    "LogLikelihoodFailure",
    "LogLikelihoodResult",
    "LogLikelihoodState",
    "LogLikelihoodSuccess",
    "reduce_log_likelihood_chunks",
]
