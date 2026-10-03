"""Finite scalar log-density evaluation for the closed family registry."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import StrEnum
from math import exp, expm1, fsum, isfinite, lgamma, log, log1p, pi, ulp
from types import MappingProxyType
from typing import Final, TypeAlias, cast

from veridist.families.registry import FAMILY_REGISTRY, FamilyId, FamilySpec, Operation


class LogDensityErrorCode(StrEnum):
    """Closed, locale-neutral scalar evaluation failure codes."""

    NONFINITE_OBSERVATION = "nonfinite_observation"
    SUPPORT_VIOLATION = "support_violation"
    NONFINITE_LOG_DENSITY = "nonfinite_log_density"
    NUMERICAL_OVERFLOW = "numerical_overflow"


@dataclass(frozen=True, slots=True)
class LogDensitySuccess:
    """A finite log-density with no retained observation or parameter values."""

    family: FamilyId
    log_density: float

    def __post_init__(self) -> None:
        if type(self.family) is not FamilyId:
            raise TypeError("family must be a FamilyId")
        if type(self.log_density) is not float or not isfinite(self.log_density):
            raise ValueError("log_density must be a finite built-in float")

    def to_json(self) -> str:
        """Serialize the safe success surface deterministically."""

        return json.dumps(
            {"family": self.family.value, "log_density": self.log_density, "outcome": "success"},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class LogDensityFailure:
    """A typed failure that deliberately contains no raw input data."""

    family: FamilyId
    code: LogDensityErrorCode

    def __post_init__(self) -> None:
        if type(self.family) is not FamilyId:
            raise TypeError("family must be a FamilyId")
        if type(self.code) is not LogDensityErrorCode:
            raise TypeError("code must be a LogDensityErrorCode")

    def to_json(self) -> str:
        """Serialize only the stable failure identity deterministically."""

        return json.dumps(
            {"code": self.code.value, "family": self.family.value, "outcome": "failure"},
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )


LogDensityResult: TypeAlias = LogDensitySuccess | LogDensityFailure
_Evaluator: TypeAlias = Callable[[float, Mapping[str, float]], float]
_LOG_SQRT_TWO_PI: Final = 0.5 * log(2.0 * pi)


class _NumericalOverflow(Exception):
    """Internal signal for a non-representable required intermediate."""


def evaluate_log_density(
    family: FamilyId, observation: object, /, **parameters: object
) -> LogDensityResult:
    """Evaluate one exact-observation log-density under closed metadata contracts.

    Programmer misuse of family identity or canonical parameter names raises
    ``TypeError``/``ValueError``.  Data-domain and binary64 numerical failures
    are returned as typed, locale-neutral values.
    """

    if type(family) is not FamilyId:
        raise TypeError("family must be a FamilyId")
    specification = FAMILY_REGISTRY.families[family]
    validated = specification.validate_parameters(**parameters)
    return _evaluate_validated_log_density(family, validated, observation)


def _evaluate_validated_log_density(
    family: FamilyId, parameters: Mapping[str, float], observation: object
) -> LogDensityResult:
    """Evaluate with registry-validated canonical parameters owned by a caller.

    This internal seam deliberately accepts only the mapping returned by the
    immutable registry validation boundary.  It is used by streaming reduction
    to validate parameters once per operation, rather than once per scalar.
    """

    specification = FAMILY_REGISTRY.families[family]
    numeric_observation = _validate_observation(observation)
    if numeric_observation is None:
        return LogDensityFailure(family, LogDensityErrorCode.NONFINITE_OBSERVATION)
    if not specification.contains(numeric_observation):
        return LogDensityFailure(family, LogDensityErrorCode.SUPPORT_VIOLATION)
    try:
        candidate = _DISPATCH[family](numeric_observation, parameters)
    except (OverflowError, _NumericalOverflow):
        return LogDensityFailure(family, LogDensityErrorCode.NUMERICAL_OVERFLOW)
    except ValueError:
        return LogDensityFailure(family, LogDensityErrorCode.NONFINITE_LOG_DENSITY)
    if not isfinite(candidate):
        return LogDensityFailure(family, LogDensityErrorCode.NONFINITE_LOG_DENSITY)
    return LogDensitySuccess(family, candidate)


def _validate_observation(observation: object) -> float | None:
    if type(observation) not in (int, float):
        return None
    try:
        numeric = float(cast(int | float, observation))
    except OverflowError:
        return None
    return numeric if isfinite(numeric) else None


def _normal(observation: float, parameters: Mapping[str, float]) -> float:
    z = _scaled_difference(observation, parameters["mu"], parameters["sigma"])
    quadratic = _finite_intermediate(z * z)
    return -_LOG_SQRT_TWO_PI - log(parameters["sigma"]) - 0.5 * quadratic


def _gamma(observation: float, parameters: Mapping[str, float]) -> float:
    shape = parameters["shape"]
    scale = parameters["scale"]
    delta = _ratio_minus_one(observation, shape, scale)
    if shape >= 8.0:
        if delta == -1.0:
            # The exact positive ratio can round below binary64 resolution here.
            # Its log-density remains finite, but the deviance representation
            # would call log1p(-1); use the stable far-left direct form instead.
            log_observation = _finite_intermediate(log(observation))
            quotient = _finite_intermediate(observation / scale)
            log_gamma = _finite_intermediate(lgamma(shape))
            scale_term = _finite_intermediate(shape * log(scale))
            return (shape - 1.0) * log_observation - quotient - log_gamma - scale_term
        deviance = _finite_intermediate(shape * _log1pmx(delta))
        return fsum(
            (
                -log(scale),
                -0.5 * (log(2.0 * pi) + log(shape)),
                -_stirling_error(shape),
                deviance,
                -log1p(delta),
            )
        )
    log_observation = _finite_intermediate(log(observation))
    quotient = _finite_intermediate(observation / scale)
    log_gamma = _finite_intermediate(lgamma(shape))
    scale_term = _finite_intermediate(shape * log(scale))
    return (shape - 1.0) * log_observation - quotient - log_gamma - scale_term


def _weibull_min(observation: float, parameters: Mapping[str, float]) -> float:
    shape = parameters["shape"]
    scale = parameters["scale"]
    log_ratio = _log_ratio(observation, scale)
    exponent = _finite_intermediate(shape * log_ratio)
    if exponent > _LOG_MAX_FLOAT:
        raise _NumericalOverflow
    if abs(exponent) <= 0.5:
        return log(shape) - log(scale) - 1.0 - log_ratio - (expm1(exponent) - exponent)
    powered_ratio = exp(exponent)
    return log(shape) - log(scale) + (shape - 1.0) * log_ratio - powered_ratio


def _lognormal(observation: float, parameters: Mapping[str, float]) -> float:
    log_observation = log(observation)
    centered = _finite_intermediate(log_observation - parameters["mu_log"])
    if _lognormal_needs_decimal(log_observation, parameters["mu_log"], parameters["sigma_log"]):
        return _lognormal_decimal(observation, parameters["mu_log"], parameters["sigma_log"])
    z = _finite_intermediate(centered / parameters["sigma_log"])
    quadratic = _finite_intermediate(z * z)
    return -log_observation - log(parameters["sigma_log"]) - _LOG_SQRT_TWO_PI - 0.5 * quadratic


def _gumbel_right(observation: float, parameters: Mapping[str, float]) -> float:
    scale = parameters["scale"]
    z = _scaled_difference(observation, parameters["location"], scale)
    if z < -_LOG_MAX_FLOAT:
        raise _NumericalOverflow
    if abs(z) <= 0.5:
        return -log(scale) - 1.0 - (z + expm1(-z))
    return -log(scale) - z - exp(-z)


def _exponential(observation: float, parameters: Mapping[str, float]) -> float:
    """Return ``log(rate) - rate * observation``; at ``0`` this is ``log(rate)``.

    The registry support gate (``observation >= 0`` for the exponential) is
    applied by the caller.
    """

    rate = parameters["rate"]
    return log(rate) - _finite_intermediate(rate * observation)


def _finite_intermediate(value: float) -> float:
    if not isfinite(value):
        raise _NumericalOverflow
    return value


def _lognormal_needs_decimal(log_observation: float, mean: float, sigma: float) -> bool:
    """Detect when binary64 log subtraction can dominate a tiny sigma contract."""

    uncertainty = 4.0 * max(ulp(log_observation), ulp(mean))
    return abs(log_observation - mean) <= 32.0 * uncertainty or uncertainty > sigma * 1.0e-12


def _lognormal_decimal(observation: float, mean: float, sigma: float) -> float:
    """Recompute a cancellation-sensitive lognormal value in bounded local precision."""

    with localcontext() as context:
        context.prec = _decimal_precision(observation, mean, sigma)
        decimal_observation = Decimal.from_float(observation)
        decimal_mean = Decimal.from_float(mean)
        decimal_sigma = Decimal.from_float(sigma)
        log_observation = decimal_observation.ln()
        centered = log_observation - decimal_mean
        z = centered / decimal_sigma
        candidate = (
            -log_observation
            - decimal_sigma.ln()
            - Decimal.from_float(_LOG_SQRT_TWO_PI)
            - Decimal("0.5") * z * z
        )
        return _finite_intermediate(float(candidate))


def _decimal_precision(observation: float, mean: float, sigma: float) -> int:
    """Choose a bounded local Decimal precision from the exact binary64 scale."""

    values = (observation, mean, sigma)
    magnitude = max(abs(log(abs(value), 10.0)) for value in values if value != 0.0)
    return min(450, max(100, 80 + int(magnitude)))


_LOG_MAX_FLOAT: Final = log(float.fromhex("0x1.fffffffffffffp+1023"))


def _ratio_minus_one(numerator: float, *denominators: float) -> float:
    """Return an exact-binary ratio minus one without rounded products."""

    numerator_value, numerator_scale = numerator.as_integer_ratio()
    denominator_value = numerator_scale
    for denominator in denominators:
        value, scale = denominator.as_integer_ratio()
        numerator_value *= scale
        denominator_value *= value
    return _integer_ratio_to_float(numerator_value - denominator_value, denominator_value)


def _scaled_difference(left: float, right: float, scale: float) -> float:
    """Return an exact-binary ``(left - right) / scale`` or signal overflow."""

    left_value, left_scale = left.as_integer_ratio()
    right_value, right_scale = right.as_integer_ratio()
    scale_value, scale_scale = scale.as_integer_ratio()
    difference = left_value * right_scale - right_value * left_scale
    denominator = left_scale * right_scale * scale_value
    numerator = difference * scale_scale
    return _finite_intermediate(_integer_ratio_to_float(numerator, denominator))


def _integer_ratio_to_float(numerator: int, denominator: int) -> float:
    """Convert one exact rational to binary64, preserving a finite small ratio."""

    try:
        return numerator / denominator
    except OverflowError as error:
        raise _NumericalOverflow from error


def _log_ratio(numerator: float, denominator: float) -> float:
    """Compute log(numerator / denominator), retaining adjacent-float differences."""

    delta = _ratio_minus_one(numerator, denominator)
    if isfinite(delta) and abs(delta) <= 0.5:
        return log1p(delta)
    return _finite_intermediate(log(numerator) - log(denominator))


def _log1pmx(delta: float) -> float:
    """Compute ``log1p(delta) - delta`` without near-zero cancellation."""

    if abs(delta) > 0.03:
        return log1p(delta) - delta
    power = delta * delta
    total = 0.0
    for index in range(2, 36):
        term = power / index
        total += -term if index % 2 == 0 else term
        power *= delta
    return total


def _stirling_error(shape: float) -> float:
    """Return the positive-real log-Gamma Stirling correction for shape >= 8."""

    inverse = 1.0 / shape
    squared = inverse * inverse
    polynomial = (
        1.0 / 12.0
        + squared
        * (
            -1.0 / 360.0
            + squared
            * (
                1.0 / 1260.0
                + squared
                * (
                    -1.0 / 1680.0
                    + squared
                    * (1.0 / 1188.0 + squared * (-691.0 / 360360.0 + squared / 156.0))
                )
            )
        )
    )
    return inverse * polynomial


_DISPATCH: Final[Mapping[FamilyId, _Evaluator]] = MappingProxyType(
    {
        FamilyId.NORMAL: _normal,
        FamilyId.GAMMA: _gamma,
        FamilyId.WEIBULL_MIN: _weibull_min,
        FamilyId.LOGNORMAL: _lognormal,
        FamilyId.GUMBEL_RIGHT: _gumbel_right,
        FamilyId.EXPONENTIAL: _exponential,
    }
)


def _verify_registry_dispatch(
    registry: Mapping[FamilyId, FamilySpec], dispatch: Mapping[FamilyId, _Evaluator]
) -> None:
    """Fail deterministically if advertised scalar capability loses parity."""

    if set(registry) != set(dispatch):
        raise RuntimeError("log-density dispatch must exactly match the family registry")
    if any(not spec.supports(Operation.LOGPDF) for spec in registry.values()):
        raise RuntimeError("log-density dispatch cannot exceed advertised registry capability")


_verify_registry_dispatch(FAMILY_REGISTRY.families, _DISPATCH)


__all__ = [
    "LogDensityErrorCode",
    "LogDensityFailure",
    "LogDensityResult",
    "LogDensitySuccess",
    "evaluate_log_density",
]
