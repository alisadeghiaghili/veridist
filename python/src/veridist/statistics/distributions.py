"""Scalar CDF, survival, quantile, and sampling operations for v1 families."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from math import copysign, erfc, exp, expm1, fsum, isfinite, lgamma, log, log1p, sqrt
from typing import cast

from veridist.families.registry import FAMILY_REGISTRY

_SQRT_TWO = sqrt(2.0)
_SQRT_TWO_PI = sqrt(2.0 * 3.141592653589793)


def _parameters(family: str, parameters: Mapping[str, object]) -> Mapping[str, float]:
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    if family == "exponential":
        if set(parameters) != {"rate"}:
            raise TypeError("parameter keys must equal the canonical parameter tuple")
        rate = parameters["rate"]
        if type(rate) not in {int, float}:
            raise ValueError("rate must be finite and positive")
        numeric_rate = float(cast(int | float, rate))
        if not isfinite(numeric_rate) or numeric_rate <= 0.0:
            raise ValueError("rate must be finite and positive")
        return {"rate": numeric_rate}
    return FAMILY_REGISTRY.resolve(family).validate_parameters(**dict(parameters))


def _probability(value: float) -> None:
    if type(value) is not float or not isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError(
            "probability must be a finite built-in float strictly between zero and one"
        )


def _finite_scalar(value: object) -> float:
    if type(value) not in {int, float}:
        raise TypeError("value must be a built-in real number")
    numeric = float(cast(int | float, value))
    if not isfinite(numeric):
        raise ValueError("value must be finite")
    return numeric


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


def cdf(family: str, value: object, parameters: Mapping[str, object]) -> float:
    """Return one finite scalar cumulative probability for a registered family."""

    point = _finite_scalar(value)
    parameter = _parameters(family, parameters)
    if family == "exponential":
        return 0.0 if point < 0.0 else -expm1(-parameter["rate"] * point)
    if family == "normal":
        return 0.5 * erfc(-(point - parameter["mu"]) / (parameter["sigma"] * _SQRT_TWO))
    if family == "gamma":
        lower, _ = _regularized_gamma_pq(parameter["shape"], point / parameter["scale"])
        return lower
    if family == "weibull_min":
        if point <= 0.0:
            return 0.0
        return -expm1(-((point / parameter["scale"]) ** parameter["shape"]))
    if family == "lognormal":
        if point <= 0.0:
            return 0.0
        return 0.5 * erfc(
            -(log(point) - parameter["mu_log"]) / (parameter["sigma_log"] * _SQRT_TWO)
        )
    if family == "gumbel_right":
        return exp(-exp(-(point - parameter["location"]) / parameter["scale"]))
    raise AssertionError("registry resolution must reject unknown families")  # pragma: no cover


def sf(family: str, value: object, parameters: Mapping[str, object]) -> float:
    """Return the stable scalar survival probability for a registered family."""

    point = _finite_scalar(value)
    parameter = _parameters(family, parameters)
    if family == "exponential":
        return 1.0 if point < 0.0 else exp(-parameter["rate"] * point)
    if family == "normal":
        return 0.5 * erfc((point - parameter["mu"]) / (parameter["sigma"] * _SQRT_TWO))
    if family == "gamma":
        _, upper = _regularized_gamma_pq(parameter["shape"], point / parameter["scale"])
        return upper
    if family == "weibull_min":
        return 1.0 if point <= 0.0 else exp(-((point / parameter["scale"]) ** parameter["shape"]))
    if family == "lognormal":
        return (
            1.0
            if point <= 0.0
            else 0.5
            * erfc((log(point) - parameter["mu_log"]) / (parameter["sigma_log"] * _SQRT_TWO))
        )
    if family == "gumbel_right":
        return -expm1(-exp(-(point - parameter["location"]) / parameter["scale"]))
    raise AssertionError("registry resolution must reject unknown families")  # pragma: no cover


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
    family: str, probability: float, parameters: Mapping[str, object]
) -> float:
    """Locate a quantile by bisection for families without a closed-form inverse.

    The sole current caller is the gamma family, whose support is strictly
    positive and whose CDF can climb from zero to its plateau across many
    decades of ``x`` when the shape parameter is small. A fixed count of
    equal-width (additive) halvings cannot resolve a root that may sit dozens
    of orders of magnitude below one, so the bracket is refined multiplicatively
    (bisecting the logarithm of ``x``, a relative-tolerance stop) instead.
    """

    if cdf(family, _TINY_POSITIVE, parameters) >= probability:
        # No representable positive value is small enough to be closer to the
        # true (unrepresentable) root than zero is; zero is the correctly
        # rounded answer.
        return 0.0
    upper = 1.0
    for _ in range(2048):
        if cdf(family, upper, parameters) >= probability:
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
            if sf(family, midpoint, parameters) > complement:
                lower = midpoint
            else:
                upper = midpoint
    else:
        for _ in range(200):
            midpoint = exp(0.5 * (log(lower) + log(upper)))
            if cdf(family, midpoint, parameters) < probability:
                lower = midpoint
            else:
                upper = midpoint
    return (lower + upper) / 2.0


def ppf(family: str, probability: float, parameters: Mapping[str, object]) -> float:
    """Return a scalar quantile for a strictly interior probability."""

    _probability(probability)
    parameter = _parameters(family, parameters)
    if family == "exponential":
        return -log1p(-probability) / parameter["rate"]
    if family == "normal":
        return parameter["mu"] + parameter["sigma"] * _normal_ppf(probability)
    if family == "weibull_min":
        return float(parameter["scale"] * (-log1p(-probability)) ** (1.0 / parameter["shape"]))
    if family == "lognormal":
        return exp(parameter["mu_log"] + parameter["sigma_log"] * _normal_ppf(probability))
    if family == "gumbel_right":
        return parameter["location"] - parameter["scale"] * log(-log(probability))
    return _inverse_by_bisection(family, probability, parameter)


def sample(family: str, size: int, parameters: Mapping[str, object], rng: object) -> object:
    """Sample with only the caller-owned NumPy generator as a randomness source."""

    np = import_module("numpy")

    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative built-in integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    parameter = _parameters(family, parameters)
    if family == "exponential":
        return rng.exponential(1.0 / parameter["rate"], size=size)
    if family == "normal":
        return rng.normal(parameter["mu"], parameter["sigma"], size=size)
    if family == "gamma":
        return rng.gamma(parameter["shape"], parameter["scale"], size=size)
    if family == "weibull_min":
        return parameter["scale"] * rng.weibull(parameter["shape"], size=size)
    if family == "lognormal":
        return rng.lognormal(parameter["mu_log"], parameter["sigma_log"], size=size)
    if family == "gumbel_right":
        return rng.gumbel(parameter["location"], parameter["scale"], size=size)
    raise AssertionError("registry resolution must reject unknown families")  # pragma: no cover


__all__ = ["cdf", "ppf", "sample", "sf"]
