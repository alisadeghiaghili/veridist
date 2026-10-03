"""Scalar log-density, CDF, survival, quantile and sampling for the registry families.

Every operation has one calling convention: the family (a
:class:`~veridist.families.registry.FamilyId` or its string value) and the
point are positional, and the canonical parameters are keywords::

    logpdf("weibull_min", 2.0, shape=1.5, scale=3.0)
    cdf(FamilyId.GAMMA, 2.0, shape=2.0, scale=1.0)
    ppf("normal", 0.975, mu=0.0, sigma=1.0)
    sample("exponential", 100, rng=generator, rate=0.5)

``logpdf``, ``cdf``, ``sf`` and ``ppf`` also take numpy arrays (or any array-like) for
the point and for every parameter, broadcast against each other.  With only
scalar operands (Python or numpy real scalars, or 0-d arrays) the result is a
Python ``float``; otherwise it is a ``float64`` ndarray of the broadcast shape.
The exponential, Weibull and right-Gumbel families are evaluated by numpy-native
kernels; the normal, lognormal and gamma families wrap the verified scalar kernels
element by element, so they match the scalar result exactly but are not fast.

The earlier forms that passed the parameters as a mapping
(``cdf(family, x, {"rate": 1.0})`` and
``sample(family, size, parameters, rng)``) still work and return the identical
result, but emit a :class:`DeprecationWarning`; they will be removed in 3.0.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from importlib import import_module
from math import copysign, erfc, exp, expm1, fsum, inf, isfinite, lgamma, log, log1p, nan, sqrt
from types import MappingProxyType
from typing import Any, Final, TypeAlias, overload

from veridist.domain._numeric import is_integer
from veridist.families.registry import (
    FAMILY_REGISTRY,
    FamilyId,
    FamilySpec,
    Operation,
)
from veridist.statistics import _vectorized as _vec
from veridist.statistics.log_density import (
    LogDensityErrorCode,
    LogDensitySuccess,
    _evaluate_validated_log_density,
)

_SQRT_TWO = sqrt(2.0)
_SQRT_TWO_PI = sqrt(2.0 * 3.141592653589793)

_Parameters: TypeAlias = Mapping[str, float]
_Kernel: TypeAlias = Callable[[float, _Parameters], float]
_Sampler: TypeAlias = Callable[[Any, int, _Parameters], Any]
_MISSING: Final = object()


_PROBABILITY_MESSAGE: Final = "probability must be a finite real strictly between zero and one"


def _probability(value: object) -> Any:
    """Return a scalar probability as ``float`` or an array validated element-wise.

    Every element must lie strictly inside ``(0, 1)``.  Any other kind of value is a
    ``ValueError`` too, as it always was for a probability.
    """

    try:
        probability = _vec.coerce(value, "probability")
    except TypeError:
        raise ValueError(_PROBABILITY_MESSAGE) from None
    if isinstance(probability, float):
        if not 0.0 < probability < 1.0:  # NaN and the infinities fail this too
            raise ValueError(_PROBABILITY_MESSAGE)
        return probability
    _vec.require_probability(probability, "q")
    return probability


def _finite_point(value: object) -> Any:
    """Return a finite scalar point as ``float`` or an array validated element-wise."""

    point = _vec.coerce(value, "value")
    if isinstance(point, float):
        if not isfinite(point):
            raise ValueError("value must be finite")
        return point
    _vec.require_finite(point, "x")
    return point


def _call_parameters(
    operation: str, args: tuple[object, ...], keywords: Mapping[str, object]
) -> tuple[Mapping[str, object], bool]:
    """Return the parameter mapping and whether the deprecated mapping form was used.

    The deprecated form passes the parameters as one mapping, positionally or as
    ``parameters=``.  It is detected by shape alone (no canonical parameter is
    named ``parameters``), warns, and is otherwise handled exactly like the
    keyword form.
    """

    if args:
        if len(args) != 1 or keywords:
            raise TypeError(f"{operation}() takes the parameters as keyword arguments")
        legacy: object = args[0]
    elif set(keywords) == {"parameters"} and isinstance(keywords["parameters"], Mapping):
        legacy = keywords["parameters"]
    else:
        return keywords, False
    warnings.warn(
        f"{operation}(): passing the parameters as a mapping is deprecated and will be "
        "removed in 3.0; pass them as keyword arguments instead",
        DeprecationWarning,
        stacklevel=3,
    )
    if not isinstance(legacy, Mapping):
        raise TypeError("parameters must be a mapping")
    return legacy, True


def _validated_parameters(
    spec: FamilySpec, parameters: Mapping[str, object], legacy: bool
) -> Mapping[str, Any]:
    try:
        return _vec.validated_parameters(spec, parameters)
    except TypeError:
        # The deprecated form reported a non-numeric exponential ``rate`` as a
        # ValueError; keep that for the callers that still use it.
        if legacy and spec.id is FamilyId.EXPONENTIAL and set(parameters) == {"rate"}:
            raise ValueError("rate must be finite and positive") from None
        raise


_GAMMA_FPMIN = 1e-300
_GAMMA_EPS = 2e-16
_GAMMA_MAXIT = 512


def _upper_gamma_continued_fraction(shape: float, value: float) -> float:
    """Return ``h`` with ``Q(shape, value) = exp(-value + shape*log(value) - lgamma(shape)) * h``.

    Modified Lentz evaluation of the upper incomplete gamma continued
    fraction, valid for ``value >= shape + 1``. Raises ``ArithmeticError`` if
    it does not converge within the iteration budget.
    """

    b = value + 1.0 - shape
    c = 1.0 / _GAMMA_FPMIN
    d = 1.0 / b
    h = d
    for index in range(1, _GAMMA_MAXIT + 1):
        coefficient = -index * (index - shape)
        b += 2.0
        d = coefficient * d + b
        if abs(d) < _GAMMA_FPMIN:
            d = copysign(_GAMMA_FPMIN, d)
        c = b + coefficient / c
        if abs(c) < _GAMMA_FPMIN:
            c = copysign(_GAMMA_FPMIN, c)
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) <= _GAMMA_EPS:
            break
    else:
        raise ArithmeticError("gamma continued fraction failed to converge")
    return h


def _regularized_gamma_pq(shape: float, value: float) -> tuple[float, float]:
    """Return the regularized lower and upper incomplete gamma ratios ``(P, Q)``.

    Uses the convergent power series for ``value < shape + 1`` and the
    modified Lentz continued fraction otherwise. The continued fraction
    evaluates ``Q`` directly and derives ``P = 1 - Q`` from it, which keeps
    the right tail accurate where ``P`` alone would underflow to zero.  Its
    intermediate terms ``c``/``d`` keep their natural sign when clamped away
    from zero (clamping only the magnitude, as an earlier version did,
    silently flips the sign of later terms and makes the result wrong).
    """

    if value <= 0.0:
        return 0.0, 1.0
    if value < shape + 1.0:
        term = 1.0 / shape
        total = term
        current = shape
        for _ in range(_GAMMA_MAXIT):
            current += 1.0
            term *= value / current
            total += term
            if abs(term) <= abs(total) * _GAMMA_EPS:
                break
        else:
            raise ArithmeticError("gamma series expansion failed to converge")
        lower = min(1.0, total * exp(-value + shape * log(value) - lgamma(shape)))
        return lower, 1.0 - lower
    h = _upper_gamma_continued_fraction(shape, value)
    upper = max(0.0, min(1.0, exp(-value + shape * log(value) - lgamma(shape)) * h))
    return 1.0 - upper, upper


_EULER_GAMMA = 0.5772156649015329
#: ``zeta(2) .. zeta(9)``, rounded from a 50-digit evaluation.
_ZETA_2_TO_9 = (
    1.6449340668482264,
    1.2020569031595942,
    1.0823232337111381,
    1.03692775514337,
    1.0173430619844492,
    1.008349277381923,
    1.0040773561979444,
    1.0020083928260821,
)
#: Shapes below this use the small-shape upper-tail identity, where ``1 - P``
#: would cancel catastrophically.
_SMALL_SHAPE = 0.01
_SMALL_SHAPE_TERMS = 40


def _lgamma_one_plus_small(shape: float) -> float:
    """Return ``lgamma(1 + shape)`` for ``0 < shape < 0.01`` without forming ``1 + shape``.

    ``1.0 + shape`` rounds away the low bits of a tiny shape, so ``lgamma``
    of it loses all relative accuracy; the Taylor series
    ``-gamma*s + sum((-1)**k * zeta(k) * s**k / k)`` does not.
    """

    total = -_EULER_GAMMA * shape
    power = shape
    for order, zeta in enumerate(_ZETA_2_TO_9, start=2):
        power *= shape
        term = zeta * power / order
        total += term if order % 2 == 0 else -term
    return total


def _log_upper_gamma_small_shape(shape: float, value: float, log_value: float) -> float:
    """Return ``log Q(shape, value)`` for ``0 < shape < 0.01`` and ``value < shape + 1``.

    With ``g = value**shape / Gamma(shape + 1)`` and
    ``T = shape * sum((-value)**n / (n! * (shape + n)))``, the lower ratio is
    ``P = g * (1 + T)`` and ``Q = -(u + T + u*T)`` for ``u = g - 1``. The
    second form is evaluated with ``expm1`` so that a ``Q`` far below
    machine epsilon (which ``1 - P`` rounds to zero) keeps full relative
    accuracy.
    """

    exponent = shape * log_value - _lgamma_one_plus_small(shape)
    term = 1.0
    total = 0.0
    for order in range(1, _SMALL_SHAPE_TERMS + 1):
        term *= -value / order
        total += term / (shape + order)
    correction = shape * total
    lower = exp(exponent) * (1.0 + correction)
    if lower <= 0.5:
        return log1p(-lower)
    u = expm1(exponent)
    return log(-fsum((u, correction, u * correction)))


def _log_regularized_gamma_q(shape: float, value: float, log_value: float) -> float:
    """Return ``log Q(shape, value)`` for positive finite ``shape`` and ``value >= 0``.

    ``log_value`` must be the finite natural logarithm of the mathematical
    ``value``; ``value`` itself may have underflowed to ``0.0`` or a
    subnormal, which is why the logarithm is supplied separately.

    ``Q`` itself is never formed in the upper tail: the continued fraction's
    prefactor ``exp(-value + shape*log(value) - lgamma(shape))`` is kept as a
    sum of logarithms, so a ``Q`` far below the smallest binary64 (where
    ``log(Q)`` of a rounded ``Q`` would be ``-inf``) is still returned as a
    finite number, and no asymptotic fallback is needed. Every outcome is
    either a finite float or an exception: ``ArithmeticError`` when an
    expansion does not converge in its iteration budget (very large
    ``shape``), ``OverflowError`` or ``ValueError`` when a required
    intermediate or the result is not representable.
    """

    if value >= shape + 1.0:
        h = _upper_gamma_continued_fraction(shape, value)
        return fsum((-value, shape * log_value, -lgamma(shape), log(h)))
    if shape < _SMALL_SHAPE:
        return _log_upper_gamma_small_shape(shape, value, log_value)
    if value < _GAMMA_FPMIN:
        lower = exp(shape * log_value - lgamma(shape + 1.0))
    else:
        lower = _regularized_gamma_pq(shape, value)[0]
    return log1p(-lower)




def _exponential_cdf(point: float, parameter: _Parameters) -> float:
    return 0.0 if point < 0.0 else -expm1(-parameter["rate"] * point)


def _normal_cdf(point: float, parameter: _Parameters) -> float:
    return 0.5 * erfc(-(point - parameter["mu"]) / (parameter["sigma"] * _SQRT_TWO))


def _gamma_cdf(point: float, parameter: _Parameters) -> float:
    lower, _ = _regularized_gamma_pq(parameter["shape"], point / parameter["scale"])
    return lower


def _exp_or_inf(value: float) -> float:
    """Return ``exp(value)``, saturating to ``inf`` instead of raising on overflow."""

    try:
        return exp(value)
    except OverflowError:
        return inf


def _pow_or_inf(base: float, exponent: float) -> float:
    """Return ``base ** exponent`` for ``base >= 0``, saturating to ``inf`` on overflow."""

    try:
        return float(base**exponent)
    except OverflowError:
        return inf


def _weibull_min_cdf(point: float, parameter: _Parameters) -> float:
    if point <= 0.0:
        return 0.0
    return -expm1(-_pow_or_inf(point / parameter["scale"], parameter["shape"]))


def _lognormal_cdf(point: float, parameter: _Parameters) -> float:
    if point <= 0.0:
        return 0.0
    return 0.5 * erfc(-(log(point) - parameter["mu_log"]) / (parameter["sigma_log"] * _SQRT_TWO))


def _gumbel_right_cdf(point: float, parameter: _Parameters) -> float:
    return exp(-_exp_or_inf(-(point - parameter["location"]) / parameter["scale"]))


def _exponential_sf(point: float, parameter: _Parameters) -> float:
    return 1.0 if point < 0.0 else exp(-parameter["rate"] * point)


def _normal_sf(point: float, parameter: _Parameters) -> float:
    return 0.5 * erfc((point - parameter["mu"]) / (parameter["sigma"] * _SQRT_TWO))


def _gamma_sf(point: float, parameter: _Parameters) -> float:
    _, upper = _regularized_gamma_pq(parameter["shape"], point / parameter["scale"])
    return upper


def _weibull_min_sf(point: float, parameter: _Parameters) -> float:
    if point <= 0.0:
        return 1.0
    return exp(-_pow_or_inf(point / parameter["scale"], parameter["shape"]))


def _lognormal_sf(point: float, parameter: _Parameters) -> float:
    if point <= 0.0:
        return 1.0
    return 0.5 * erfc((log(point) - parameter["mu_log"]) / (parameter["sigma_log"] * _SQRT_TWO))


def _gumbel_right_sf(point: float, parameter: _Parameters) -> float:
    return -expm1(-_exp_or_inf(-(point - parameter["location"]) / parameter["scale"]))


def _normal_ppf(probability: float) -> float:
    """Acklam's rational inverse-normal approximation, refined by Newton steps."""

    lower = 0.02425
    upper = 1.0 - lower
    a = (
        -39.69683028665376,
        220.9460984245205,
        -275.9285104469687,
        138.357751867269,
        -30.66479806614716,
        2.506628277459239,
    )
    b = (
        -54.47609879822406,
        161.5858368580409,
        -155.6989798598866,
        66.80131188771972,
        -13.28068155288572,
    )
    c = (
        -0.007784894002430293,
        -0.3223964580411365,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416)
    if probability < lower:
        q = sqrt(-2.0 * log(probability))
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        result = numerator / denominator
    elif probability > upper:
        q = sqrt(-2.0 * log1p(-probability))
        numerator = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        denominator = ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        result = -numerator / denominator
    else:
        q = probability - 0.5
        r = q * q
        numerator = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        denominator = (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        result = numerator * q / denominator
    for _ in range(2):
        error = 0.5 * erfc(-result / _SQRT_TWO) - probability
        density = exp(-result * result / 2.0) / _SQRT_TWO_PI
        if density == 0.0:  # pragma: no branch - subnormal-tail guard
            break
        result -= error / density
    return result


_TINY_POSITIVE = 5e-324  # smallest positive (subnormal) built-in float


def _inverse_by_bisection(
    cdf_kernel: _Kernel, sf_kernel: _Kernel, probability: float, parameters: _Parameters
) -> float:
    """Locate a quantile by bisection for families without a closed-form inverse.

    The sole current caller is the gamma family, whose support is strictly
    positive and whose CDF can climb from zero to its plateau across many
    decades of ``x`` when the shape parameter is small. A fixed count of
    equal-width (additive) halvings cannot resolve a root that may sit dozens
    of orders of magnitude below one, so the bracket is refined multiplicatively
    (bisecting the logarithm of ``x``, a relative-tolerance stop) instead.
    """

    if cdf_kernel(_TINY_POSITIVE, parameters) >= probability:
        # No representable positive value is small enough to be closer to the
        # true (unrepresentable) root than zero is; zero is the correctly
        # rounded answer.
        return 0.0
    upper = 1.0
    for _ in range(2048):
        if cdf_kernel(upper, parameters) >= probability:
            break
        upper *= 2.0
    else:  # pragma: no cover - finite distribution support guarantees a bracket
        raise ArithmeticError("unable to bracket distribution quantile")
    lower = _TINY_POSITIVE
    # A fixed count of log-domain halvings, rather than an early relative-
    # tolerance exit, reaches binary64 precision deterministically: the
    # widest possible log-ratio between ``lower`` and ``upper`` here is a few
    # thousand, and each step halves it, so 200 steps leave a residual
    # log-width far below the ulp of any representable quantile.
    if probability > 0.5:
        # Bisecting on the survival function keeps upper-tail precision: the
        # complementary probability stays representable in binary64 long
        # after ``1 - cdf(midpoint)`` would have already underflowed to zero.
        complement = 1.0 - probability
        for _ in range(200):
            midpoint = exp(0.5 * (log(lower) + log(upper)))
            if sf_kernel(midpoint, parameters) > complement:
                lower = midpoint
            else:
                upper = midpoint
    else:
        for _ in range(200):
            midpoint = exp(0.5 * (log(lower) + log(upper)))
            if cdf_kernel(midpoint, parameters) < probability:
                lower = midpoint
            else:
                upper = midpoint
    return (lower + upper) / 2.0


def _exponential_ppf(probability: float, parameter: _Parameters) -> float:
    return -log1p(-probability) / parameter["rate"]


def _normal_quantile(probability: float, parameter: _Parameters) -> float:
    return parameter["mu"] + parameter["sigma"] * _normal_ppf(probability)


def _gamma_ppf(probability: float, parameter: _Parameters) -> float:
    return _inverse_by_bisection(_gamma_cdf, _gamma_sf, probability, parameter)


def _weibull_min_ppf(probability: float, parameter: _Parameters) -> float:
    return parameter["scale"] * _pow_or_inf(-log1p(-probability), 1.0 / parameter["shape"])


def _lognormal_ppf(probability: float, parameter: _Parameters) -> float:
    return _exp_or_inf(parameter["mu_log"] + parameter["sigma_log"] * _normal_ppf(probability))


def _gumbel_right_ppf(probability: float, parameter: _Parameters) -> float:
    return parameter["location"] - parameter["scale"] * log(-log(probability))


def _exponential_sample(rng: Any, size: int, parameter: _Parameters) -> Any:
    return rng.exponential(1.0 / parameter["rate"], size=size)


def _normal_sample(rng: Any, size: int, parameter: _Parameters) -> Any:
    return rng.normal(parameter["mu"], parameter["sigma"], size=size)


def _gamma_sample(rng: Any, size: int, parameter: _Parameters) -> Any:
    return rng.gamma(parameter["shape"], parameter["scale"], size=size)


def _weibull_min_sample(rng: Any, size: int, parameter: _Parameters) -> Any:
    return parameter["scale"] * rng.weibull(parameter["shape"], size=size)


def _lognormal_sample(rng: Any, size: int, parameter: _Parameters) -> Any:
    return rng.lognormal(parameter["mu_log"], parameter["sigma_log"], size=size)


def _gumbel_right_sample(rng: Any, size: int, parameter: _Parameters) -> Any:
    return rng.gumbel(parameter["location"], parameter["scale"], size=size)


_CDF: Final[Mapping[FamilyId, _Kernel]] = MappingProxyType(
    {
        FamilyId.NORMAL: _normal_cdf,
        FamilyId.GAMMA: _gamma_cdf,
        FamilyId.WEIBULL_MIN: _weibull_min_cdf,
        FamilyId.LOGNORMAL: _lognormal_cdf,
        FamilyId.GUMBEL_RIGHT: _gumbel_right_cdf,
        FamilyId.EXPONENTIAL: _exponential_cdf,
    }
)
_SF: Final[Mapping[FamilyId, _Kernel]] = MappingProxyType(
    {
        FamilyId.NORMAL: _normal_sf,
        FamilyId.GAMMA: _gamma_sf,
        FamilyId.WEIBULL_MIN: _weibull_min_sf,
        FamilyId.LOGNORMAL: _lognormal_sf,
        FamilyId.GUMBEL_RIGHT: _gumbel_right_sf,
        FamilyId.EXPONENTIAL: _exponential_sf,
    }
)
_PPF: Final[Mapping[FamilyId, _Kernel]] = MappingProxyType(
    {
        FamilyId.NORMAL: _normal_quantile,
        FamilyId.GAMMA: _gamma_ppf,
        FamilyId.WEIBULL_MIN: _weibull_min_ppf,
        FamilyId.LOGNORMAL: _lognormal_ppf,
        FamilyId.GUMBEL_RIGHT: _gumbel_right_ppf,
        FamilyId.EXPONENTIAL: _exponential_ppf,
    }
)
_SAMPLE: Final[Mapping[FamilyId, _Sampler]] = MappingProxyType(
    {
        FamilyId.NORMAL: _normal_sample,
        FamilyId.GAMMA: _gamma_sample,
        FamilyId.WEIBULL_MIN: _weibull_min_sample,
        FamilyId.LOGNORMAL: _lognormal_sample,
        FamilyId.GUMBEL_RIGHT: _gumbel_right_sample,
        FamilyId.EXPONENTIAL: _exponential_sample,
    }
)
_TABLES: Final[Mapping[Operation, Mapping[FamilyId, Any]]] = MappingProxyType(
    {
        Operation.CDF: _CDF,
        Operation.SF: _SF,
        Operation.PPF: _PPF,
        Operation.SAMPLE: _SAMPLE,
    }
)


def _verify_operation_tables(
    registry: Mapping[FamilyId, FamilySpec], tables: Mapping[Operation, Mapping[FamilyId, Any]]
) -> None:
    """Fail at import if an advertised operation lacks a kernel, or the reverse."""

    for operation, table in tables.items():
        advertised = {family for family, spec in registry.items() if spec.supports(operation)}
        if set(table) != advertised:
            raise RuntimeError(f"{operation.value} kernels must exactly match the family registry")


_verify_operation_tables(FAMILY_REGISTRY.families, _TABLES)


def _logpdf_kernel(family: FamilyId) -> _vec.ScalarKernel:
    """Return the scalar log-density as a kernel.

    The kernel gives the finite value, ``-inf`` outside the support and NaN for
    a value that binary64 cannot represent.
    """

    def kernel(point: float, parameters: _Parameters) -> float:
        evaluated = _evaluate_validated_log_density(family, parameters, point)
        if isinstance(evaluated, LogDensitySuccess):
            return evaluated.log_density
        if evaluated.code is LogDensityErrorCode.SUPPORT_VIOLATION:
            return -inf
        return nan

    return kernel


def _array_table(
    native: Mapping[FamilyId, _vec.ArrayKernel], scalar: Mapping[FamilyId, _vec.ScalarKernel]
) -> Mapping[FamilyId, _vec.ArrayKernel]:
    """Use the numpy-native kernel where there is one, else wrap the scalar kernel."""

    table: dict[FamilyId, _vec.ArrayKernel] = {}
    for family, kernel in scalar.items():
        names = tuple(parameter.name for parameter in FAMILY_REGISTRY.families[family].parameters)
        table[family] = native[family] if family in native else _vec.wrap_scalar(kernel, names)
    return MappingProxyType(table)


_ARRAY: Final[Mapping[Operation, Mapping[FamilyId, _vec.ArrayKernel]]] = MappingProxyType(
    {
        Operation.LOGPDF: _array_table(
            _vec.NATIVE_LOGPDF, {family: _logpdf_kernel(family) for family in FamilyId}
        ),
        Operation.CDF: _array_table(_vec.NATIVE_CDF, _CDF),
        Operation.SF: _array_table(_vec.NATIVE_SF, _SF),
        Operation.PPF: _array_table(_vec.NATIVE_PPF, _PPF),
    }
)
_verify_operation_tables(FAMILY_REGISTRY.families, _ARRAY)


def _evaluate_array(
    operation: Operation,
    spec: FamilySpec,
    point: Any,
    parameters: Mapping[str, Any],
) -> tuple[Any, Any]:
    """Broadcast the operands and evaluate ``operation``: ``(broadcast point, float64 values)``."""

    broadcast_point, broadcast_parameters = _vec.broadcast(point, parameters)
    kernel = _ARRAY[operation][spec.id]
    if spec.id in _vec.NATIVE_FAMILIES:
        return broadcast_point, _vec.evaluate_native(kernel, broadcast_point, broadcast_parameters)
    return broadcast_point, kernel(broadcast_point, broadcast_parameters)


def logpdf(family: FamilyId | str, x: object, /, **parameters: object) -> Any:
    """Return the log-density of ``family`` at the finite point ``x``.

    ``x`` and every parameter may be a real scalar (Python or numpy; a ``bool`` is
    rejected) or an array-like, broadcast against each other.  Scalar operands
    give a Python ``float``; otherwise the result is a ``float64`` array of the
    broadcast shape.

    The result is ``-inf`` outside the family's support (element-wise for
    arrays): ``x <= 0`` for ``gamma``, ``weibull_min`` and ``lognormal``,
    ``x < 0`` for ``exponential`` (whose log-density at zero is ``log(rate)``),
    see :attr:`~veridist.families.registry.FamilySpec.support`.  A non-finite
    ``x`` (any element) raises ``ValueError``, and a non-real or ``bool`` ``x``
    raises ``TypeError``.  Invalid parameters raise ``ValueError``; for an array
    parameter the message names the parameter and its first invalid flat index.
    A value that binary64 cannot represent (an overflowing intermediate) raises
    ``ArithmeticError``.  Use
    :func:`veridist.statistics.log_density.evaluate_log_density` for the typed,
    non-raising scalar result.
    """

    point = _finite_point(x)
    spec = FAMILY_REGISTRY.lookup(family)
    validated = _vec.validated_parameters(spec, parameters)
    if not _vec.is_scalar_call(point, validated):
        broadcast_point, raw = _evaluate_array(Operation.LOGPDF, spec, point, validated)
        return _vec.finish_logpdf(spec, broadcast_point, raw)
    evaluated = _evaluate_validated_log_density(spec.id, validated, point)
    if isinstance(evaluated, LogDensitySuccess):
        return evaluated.log_density
    if evaluated.code is LogDensityErrorCode.SUPPORT_VIOLATION:
        return -inf
    raise ArithmeticError(
        f"the {spec.id.value} log-density is not representable ({evaluated.code.value})"
    )


def _evaluate(
    operation: Operation,
    tables: Mapping[FamilyId, _Kernel],
    family: FamilyId | str,
    point: Any,
    mapping: Mapping[str, object],
    legacy: bool,
) -> Any:
    spec = FAMILY_REGISTRY.lookup(family)
    validated = _validated_parameters(spec, mapping, legacy)
    if _vec.is_scalar_call(point, validated):
        return tables[spec.id](point, validated)
    return _evaluate_array(operation, spec, point, validated)[1]


@overload
def cdf(family: FamilyId | str, x: object, /, **parameters: object) -> Any: ...
@overload
def cdf(family: str, value: object, parameters: Mapping[str, object], /) -> Any: ...
def cdf(family: FamilyId | str, x: object, /, *args: object, **parameters: object) -> Any:
    """Return the cumulative probability ``P(X <= x)``.

    ``cdf(family, x, **parameters)``; the mapping form
    ``cdf(family, x, {...})`` is deprecated.  ``x`` and the parameters follow the
    scalar and array rules of :func:`logpdf`.
    """

    point = _finite_point(x)
    mapping, legacy = _call_parameters("cdf", args, parameters)
    return _evaluate(Operation.CDF, _CDF, family, point, mapping, legacy)


@overload
def sf(family: FamilyId | str, x: object, /, **parameters: object) -> Any: ...
@overload
def sf(family: str, value: object, parameters: Mapping[str, object], /) -> Any: ...
def sf(family: FamilyId | str, x: object, /, *args: object, **parameters: object) -> Any:
    """Return the stable survival probability ``P(X > x)``.

    ``sf(family, x, **parameters)``; the mapping form ``sf(family, x, {...})`` is
    deprecated.  ``x`` and the parameters follow the scalar and array rules of
    :func:`logpdf`.
    """

    point = _finite_point(x)
    mapping, legacy = _call_parameters("sf", args, parameters)
    return _evaluate(Operation.SF, _SF, family, point, mapping, legacy)


@overload
def ppf(family: FamilyId | str, q: object, /, **parameters: object) -> Any: ...
@overload
def ppf(family: str, probability: object, parameters: Mapping[str, object], /) -> Any: ...
def ppf(family: FamilyId | str, q: object, /, *args: object, **parameters: object) -> Any:
    """Return the quantile for a strictly interior probability ``q``.

    ``ppf(family, q, **parameters)``; the mapping form ``ppf(family, q, {...})``
    is deprecated.  ``q`` may be an array: every element must lie strictly
    inside ``(0, 1)``, otherwise ``ValueError`` names the first invalid flat
    index.  The parameters follow the scalar and array rules of :func:`logpdf`.
    """

    probability = _probability(q)
    mapping, legacy = _call_parameters("ppf", args, parameters)
    return _evaluate(Operation.PPF, _PPF, family, probability, mapping, legacy)


@overload
def sample(family: FamilyId | str, size: int, /, *, rng: object, **parameters: object) -> Any: ...
@overload
def sample(
    family: str, size: int, parameters: Mapping[str, object], rng: object, /
) -> Any: ...
def sample(
    family: FamilyId | str,
    size: int,
    /,
    *args: object,
    rng: object = _MISSING,
    **parameters: object,
) -> Any:
    """Draw ``size`` values using only the caller-owned ``numpy.random.Generator`` ``rng``.

    ``sample(family, size, rng=generator, **parameters)``; the form
    ``sample(family, size, parameters, rng)`` is deprecated.  ``size`` may be a
    Python or numpy integer and the parameters may be Python or numpy scalars.
    """

    np = import_module("numpy")

    if not is_integer(size) or int(size) < 0:
        raise ValueError("size must be a non-negative integer")
    if len(args) == 2:
        if rng is not _MISSING:
            raise TypeError("sample() got the generator twice")
        rng = args[1]
        args = args[:1]
    mapping, legacy = _call_parameters("sample", args, parameters)
    if rng is _MISSING:
        raise TypeError("sample() requires the keyword argument rng")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    spec = FAMILY_REGISTRY.lookup(family)
    scalars = _validated_parameters(spec, mapping, legacy)
    if not all(isinstance(value, float) for value in scalars.values()):
        raise TypeError("sample() takes scalar parameters")
    return _SAMPLE[spec.id](rng, int(size), scalars)


__all__ = ["cdf", "logpdf", "ppf", "sample", "sf"]
