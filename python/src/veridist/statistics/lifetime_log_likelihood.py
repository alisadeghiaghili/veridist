"""Exact-state streaming log-likelihood of exact and right-censored observations.

An exact observation contributes its log-density and a right-censored
observation contributes its log-survival.  Two reducers share one
implementation and differ only in the observation types and families they
admit:

* :func:`reduce_lifetime_log_likelihood_chunks` takes
  :class:`~veridist.domain.lifetimes.ExactLifetime` and
  :class:`~veridist.domain.lifetimes.RightCensoredLifetime` for the
  fixed-location lifetime families ``EXPONENTIAL``, ``WEIBULL_MIN``,
  ``LOGNORMAL`` and ``GAMMA``.
* :func:`reduce_value_log_likelihood_chunks` takes
  :class:`~veridist.domain.values.ExactValue` and
  :class:`~veridist.domain.values.RightCensoredValue` for the real-line
  families ``NORMAL`` and ``GUMBEL_RIGHT``.

Numerical guarantees
--------------------

* **Accumulation is exact and carries over unchanged.** Every term is a finite
  binary64 number, and the terms are summed in the same exact integer unit
  accumulator that :func:`~veridist.statistics.log_likelihood.reduce_log_likelihood_chunks`
  uses (multiples of ``2**-1074``). The total is therefore independent of
  chunking and observation order, and is rounded to binary64 exactly once.
  No floating-point summation order is involved, so ``math.fsum`` is not
  needed for the reduction itself.
* **Exact observations** use the existing scalar log-density evaluators and
  inherit their documented accuracy.
* **Censored observations do not inherit that accuracy statement.** The
  log-survival terms are computed by the evaluators below, each accurate to a
  stated bound but not covered by the log-density oracle envelope:

  - Weibull: ``-(t/scale)**shape``, evaluated as ``-exp(shape * log(t/scale))``
    with the ratio's logarithm taken from the exact-binary ratio, so the
    error is about ``|term| * |shape * log(t/scale)| * eps``. A term below
    the smallest binary64 ``-max_float`` is not representable and is reported
    as ``numerical_overflow``.
  - Lognormal: the stable ``_log_normal_sf`` of the lognormal fit, evaluated
    at ``z = (log(t) - mu_log) / sigma_log`` (recomputed in bounded decimal
    precision when binary64 cancellation in ``log(t) - mu_log`` would
    dominate). Its tail expansion is accurate to about ``1e-12`` relative.
  - Exponential: ``-rate * t``, which is ``0`` at ``t = 0`` (the exponential
    support is closed at zero); a term below ``-max_float`` is reported as
    ``numerical_overflow``.
  - Normal: the stable ``_log_normal_sf`` at the exact-binary
    ``z = (x - mu) / sigma``, accurate to about ``1e-12`` relative in its tail
    expansion.
  - Gumbel (right): ``log(1 - exp(-exp(-z)))``, evaluated as
    ``log(-expm1(-e))`` or ``log1p(-exp(-e))`` for ``e = exp(-z)`` and as
    ``-z`` once ``e`` is below ``1e-300``; ``z`` is the exact-binary scaled
    difference, and a ``z`` below ``-700`` gives exactly ``0``.
  - Gamma: ``log Q(shape, t/scale)`` from a log-domain incomplete-gamma
    evaluation (see ``veridist.statistics.distributions``). ``Q`` is never
    formed in the far tail, so a survival probability below the smallest
    binary64 is still a finite term, and no ``-inf`` can be produced. Accuracy
    is about ``eps * max(|t/scale|, shape * |log(t/scale)|, |lgamma(shape)|)``;
    for very large ``shape`` (roughly ``1e5`` and up, where the expansions
    need more iterations than their budget) the term is reported as
    ``nonfinite_log_density`` instead of being guessed.
* A term that is not a finite binary64 number is never accumulated: it is
  reported as a typed :class:`LogLikelihoodFailure` instead.

The result types are those of the density-only reducer. The parameter
fingerprint is salted with a different domain label, so a lifetime
log-likelihood is never mistaken for a density-only one with the same
parameters.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal, localcontext
from hashlib import sha256
from math import exp, expm1, inf, isfinite, log, log1p
from typing import Final

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue, RightCensoredValue
from veridist.engine.streaming import StreamSource, iter_stream
from veridist.families.lognormal import _log_normal_sf
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.statistics.distributions import _log_regularized_gamma_q
from veridist.statistics.log_density import (
    _LOG_MAX_FLOAT,
    LogDensityErrorCode,
    LogDensityFailure,
    LogDensityResult,
    LogDensitySuccess,
    _decimal_precision,
    _evaluate_validated_log_density,
    _finite_intermediate,
    _log_ratio,
    _lognormal_needs_decimal,
    _NumericalOverflow,
    _scaled_difference,
)
from veridist.statistics.log_likelihood import (
    LogLikelihoodErrorCode,
    LogLikelihoodFailure,
    LogLikelihoodResult,
    LogLikelihoodState,
    LogLikelihoodSuccess,
    _ExactAccumulator,
    _FinalTotalNotRepresentable,
    _ObservationLimitExceeded,
    _validate_family_and_parameters,
)

#: The fixed-location lifetime families the lifetime reducer admits.
SUPPORTED_LIFETIME_FAMILIES: Final = frozenset(
    {FamilyId.EXPONENTIAL, FamilyId.WEIBULL_MIN, FamilyId.LOGNORMAL, FamilyId.GAMMA}
)
#: The whole-real-line families the value reducer admits.
SUPPORTED_VALUE_FAMILIES: Final = frozenset({FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT})
_FINGERPRINT_DOMAIN: Final = "veridist.lifetime_log_likelihood.v1"
_VALUE_FINGERPRINT_DOMAIN: Final = "veridist.value_log_likelihood.v1"
#: Below ``-700`` the Gumbel survival ``1 - exp(-exp(-z))`` is exactly 1 in binary64.
_GUMBEL_OVERFLOW_Z: Final = 700.0


def _exponential_log_sf(time: float, parameters: Mapping[str, float]) -> float:
    return -_finite_intermediate(parameters["rate"] * time)


def _normal_log_sf(value: float, parameters: Mapping[str, float]) -> float:
    return _log_normal_sf(_scaled_difference(value, parameters["mu"], parameters["sigma"]))


def _gumbel_right_log_sf(value: float, parameters: Mapping[str, float]) -> float:
    z = _scaled_difference(value, parameters["location"], parameters["scale"])
    if z < -_GUMBEL_OVERFLOW_Z:
        # exp(-z) overflows, so exp(-exp(-z)) is 0 and the survival is exactly 1.
        return 0.0
    tail = exp(-z)
    if tail > 0.7:
        return log1p(-exp(-tail))
    if tail < 1e-300:
        return -z
    return log(-expm1(-tail))


def _weibull_log_sf(time: float, parameters: Mapping[str, float]) -> float:
    exponent = _finite_intermediate(parameters["shape"] * _log_ratio(time, parameters["scale"]))
    if exponent > _LOG_MAX_FLOAT:
        raise _NumericalOverflow
    return -exp(exponent)


def _lognormal_log_sf(time: float, parameters: Mapping[str, float]) -> float:
    mean = parameters["mu_log"]
    sigma = parameters["sigma_log"]
    log_time = log(time)
    if _lognormal_needs_decimal(log_time, mean, sigma):
        z = _lognormal_z_decimal(time, mean, sigma)
    else:
        z = _finite_intermediate((log_time - mean) / sigma)
    return _log_normal_sf(z)


def _lognormal_z_decimal(time: float, mean: float, sigma: float) -> float:
    """Recompute the standardized log-time in bounded decimal precision."""

    with localcontext() as context:
        context.prec = _decimal_precision(time, mean, sigma)
        centered = Decimal.from_float(time).ln() - Decimal.from_float(mean)
        return _finite_intermediate(float(centered / Decimal.from_float(sigma)))


def _gamma_log_sf(time: float, parameters: Mapping[str, float]) -> float:
    scale = parameters["scale"]
    log_ratio = _log_ratio(time, scale)
    # `_log_ratio` has already signalled `_NumericalOverflow` for a ratio that
    # does not fit binary64 (log Q is then about -ratio, which no binary64
    # can hold), so this quotient is finite; it may still be zero or subnormal.
    return _log_regularized_gamma_q(parameters["shape"], time / scale, log_ratio)


_LOG_SURVIVAL: Final[Mapping[FamilyId, Callable[[float, Mapping[str, float]], float]]] = {
    FamilyId.EXPONENTIAL: _exponential_log_sf,
    FamilyId.WEIBULL_MIN: _weibull_log_sf,
    FamilyId.LOGNORMAL: _lognormal_log_sf,
    FamilyId.GAMMA: _gamma_log_sf,
    FamilyId.NORMAL: _normal_log_sf,
    FamilyId.GUMBEL_RIGHT: _gumbel_right_log_sf,
}


def _evaluate_log_survival(
    family: FamilyId, parameters: Mapping[str, float], time: float
) -> LogDensityResult:
    """Evaluate one right-censored term, returning typed scalar failures."""

    if not FAMILY_REGISTRY.families[family].contains(time):
        return LogDensityFailure(family, LogDensityErrorCode.SUPPORT_VIOLATION)
    try:
        candidate = _LOG_SURVIVAL[family](time, parameters)
    except (OverflowError, _NumericalOverflow):
        return LogDensityFailure(family, LogDensityErrorCode.NUMERICAL_OVERFLOW)
    except (ValueError, ArithmeticError):
        # `ValueError`: math domain error (for example a log of zero).
        # `ArithmeticError`: an incomplete-gamma expansion did not converge.
        return LogDensityFailure(family, LogDensityErrorCode.NONFINITE_LOG_DENSITY)
    if not isfinite(candidate):
        # -inf: the true term is finite in exact arithmetic but lies below
        # -max_float. Anything else (NaN, +inf) is simply not a log-survival.
        code = (
            LogDensityErrorCode.NUMERICAL_OVERFLOW
            if candidate == -inf
            else LogDensityErrorCode.NONFINITE_LOG_DENSITY
        )
        return LogDensityFailure(family, code)
    return LogDensitySuccess(family, candidate)


def _fingerprint(domain: str, family: FamilyId, validated: Mapping[str, float]) -> str:
    specification = FAMILY_REGISTRY.families[family]
    encoded = [validated[parameter.name].hex() for parameter in specification.parameters]
    payload = domain + "\0" + family.value + "\0" + "\0".join(encoded)
    return sha256(payload.encode("ascii")).hexdigest()


def _reduce(
    family: FamilyId,
    chunks: StreamSource[Iterable[object]] | Iterable[Iterable[object]],
    validated: Mapping[str, float],
    fingerprint_domain: str,
    exact_type: type,
    censored_type: type,
    read: Callable[[object], float],
) -> LogLikelihoodResult:
    fingerprint = _fingerprint(fingerprint_domain, family, validated)
    identity = tuple(
        validated[parameter.name].hex() for parameter in FAMILY_REGISTRY.families[family].parameters
    )
    accumulator = _ExactAccumulator()
    for chunk in iter_stream(chunks):
        if not isinstance(chunk, Iterable):
            raise TypeError("each chunk must be an iterable of observations")
        for observation in chunk:
            evaluated: LogDensityResult
            if type(observation) is exact_type:
                evaluated = _evaluate_validated_log_density(family, validated, read(observation))
            elif type(observation) is censored_type:
                evaluated = _evaluate_log_survival(family, validated, read(observation))
            else:
                evaluated = LogDensityFailure(family, LogDensityErrorCode.NONFINITE_OBSERVATION)
            if type(evaluated) is LogDensityFailure:
                return LogLikelihoodFailure(
                    family,
                    LogLikelihoodErrorCode.SCALAR_EVALUATION_FAILURE,
                    accumulator.observation_count,
                    evaluated.code,
                )
            assert isinstance(evaluated, LogDensitySuccess)
            try:
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


def _lifetime_time(observation: object) -> float:
    assert isinstance(observation, ExactLifetime | RightCensoredLifetime)
    return float(observation.time)


def _real_value(observation: object) -> float:
    assert isinstance(observation, ExactValue | RightCensoredValue)
    return observation.value


def reduce_lifetime_log_likelihood_chunks(
    family: FamilyId,
    chunks: StreamSource[Iterable[object]] | Iterable[Iterable[object]],
    /,
    **parameters: object,
) -> LogLikelihoodResult:
    """Reduce exact and right-censored lifetimes into one log-likelihood.

    ``family`` must be ``EXPONENTIAL``, ``WEIBULL_MIN``, ``LOGNORMAL`` or
    ``GAMMA`` (any other family raises ``ValueError``; the real-line families
    use :func:`reduce_value_log_likelihood_chunks`), and ``parameters`` are its
    canonical parameters, validated before iteration. Each observation must be
    an ``ExactLifetime`` (log-density term) or a ``RightCensoredLifetime``
    (log-survival term); anything else, and every non-positive time, is a
    typed failure rather than an exception. ``processed_count`` on a failure
    counts only the successful observations before the terminal one and is
    not a complete-input count; ``observation_count`` on success is the number
    of exact plus censored observations.

    ``chunks`` is consumed once; an already exhausted one-shot iterator is
    indistinguishable from an empty input and reduces to a success with
    ``observation_count == 0`` and a total of ``0.0``. Use
    :class:`~veridist.engine.streaming.IterableDataSource` to enforce a
    single pass. See the module docstring for the numerical guarantees.
    """

    validated = _validate_family_and_parameters(family, parameters)
    if family not in SUPPORTED_LIFETIME_FAMILIES:
        raise ValueError("family must be one of exponential, weibull_min, lognormal or gamma")
    return _reduce(
        family,
        chunks,
        validated,
        _FINGERPRINT_DOMAIN,
        ExactLifetime,
        RightCensoredLifetime,
        _lifetime_time,
    )


def reduce_value_log_likelihood_chunks(
    family: FamilyId,
    chunks: StreamSource[Iterable[object]] | Iterable[Iterable[object]],
    /,
    **parameters: object,
) -> LogLikelihoodResult:
    """Reduce exact and right-censored real values into one log-likelihood.

    The counterpart of :func:`reduce_lifetime_log_likelihood_chunks` for the
    whole-real-line families: ``family`` must be ``NORMAL`` or
    ``GUMBEL_RIGHT`` (any other family raises ``ValueError``). Each
    observation must be an ``ExactValue`` (log-density term) or a
    ``RightCensoredValue`` (log-survival term); a lifetime observation or any
    other object is a typed failure, and there is no support restriction on
    the finite values. Everything else, including the single-pass behaviour of
    ``chunks`` and the failure accounting, is as documented for the lifetime
    reducer.
    """

    validated = _validate_family_and_parameters(family, parameters)
    if family not in SUPPORTED_VALUE_FAMILIES:
        raise ValueError("family must be one of normal or gumbel_right")
    return _reduce(
        family,
        chunks,
        validated,
        _VALUE_FINGERPRINT_DOMAIN,
        ExactValue,
        RightCensoredValue,
        _real_value,
    )


__all__ = [
    "SUPPORTED_LIFETIME_FAMILIES",
    "SUPPORTED_VALUE_FAMILIES",
    "reduce_lifetime_log_likelihood_chunks",
    "reduce_value_log_likelihood_chunks",
]
