"""Array argument handling and numpy-native kernels for the scalar distribution operations.

numpy is imported only when an array argument is actually seen, so the scalar
paths (and ``import veridist``) stay free of it.

Three families (exponential, Weibull, right Gumbel) have kernels written
directly in numpy that mirror the formulas of the verified scalar kernels
(``expm1``/``log1p`` wherever the scalar code uses them).  The other three
(normal, lognormal, gamma) are evaluated by wrapping the scalar kernels with
``numpy.frompyfunc``: numpy has no ``erfc`` or incomplete gamma function and
scipy is not a runtime dependency, so those results equal the scalar path
exactly and are correct, but not fast.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, Final, TypeAlias

from veridist.domain._numeric import is_real
from veridist.families.registry import FamilyId, FamilySpec, ParameterRole, Support

#: A scalar kernel evaluated element by element: ``(point, {name: value}) -> value``.
ScalarKernel: TypeAlias = Callable[[float, Mapping[str, float]], float]
#: An array kernel: broadcast float64 arrays in, one float64 array out.
ArrayKernel: TypeAlias = Callable[[Any, Mapping[str, Any]], Any]


def _numpy() -> Any:
    return import_module("numpy")


def coerce(value: object, name: str) -> Any:
    """Return a Python ``float`` for a real scalar or a 0-d array, else a float64 ndarray.

    Only the numeric kind is checked here (integers and floats); booleans,
    complex numbers, strings and objects raise ``TypeError``.  Finiteness and
    range are the caller's concern.
    """

    if is_real(value):
        try:
            return float(value)  # type: ignore[arg-type]
        except OverflowError:
            return float("inf")
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number or an array of real numbers")
    np = _numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must be a real number or an array of real numbers")
    if array.ndim == 0:
        return float(array)
    return array.astype(np.float64, copy=False)


def _first_true(mask: Any) -> int:
    return int(mask.reshape(-1).argmax())


def require_finite(array: Any, name: str) -> None:
    """Raise ``ValueError`` naming the first non-finite element of ``array``."""

    np = _numpy()
    bad = ~np.isfinite(array)
    if bad.any():
        raise ValueError(
            f"{name} must be finite (first invalid element at flat index {_first_true(bad)})"
        )


def require_probability(array: Any, name: str) -> None:
    """Raise ``ValueError`` naming the first element outside the open interval (0, 1)."""

    bad = ~((array > 0.0) & (array < 1.0))
    if bad.any():
        raise ValueError(
            f"{name} must be strictly between zero and one "
            f"(first invalid element at flat index {_first_true(bad)})"
        )


def validated_parameters(spec: FamilySpec, parameters: Mapping[str, object]) -> dict[str, Any]:
    """Validate the exact canonical parameter set; values are floats or float64 arrays.

    A scalar parameter is validated exactly as in the scalar operations.  An
    array parameter is validated element-wise, and the error names the
    parameter and the first invalid flat index of that array.
    """

    names = tuple(parameter.name for parameter in spec.parameters)
    if set(parameters) != set(names):
        raise TypeError("parameter keys must equal the canonical parameter tuple")
    validated: dict[str, Any] = {}
    for parameter in spec.parameters:
        value = coerce(parameters[parameter.name], parameter.name)
        if isinstance(value, float):
            validated[parameter.name] = parameter.validate(value)
            continue
        np = _numpy()
        bad = ~np.isfinite(value)
        if bad.any():
            raise ValueError(
                f"{parameter.name} must be finite "
                f"(first invalid element at flat index {_first_true(bad)})"
            )
        if parameter.role is ParameterRole.POSITIVE:
            bad = ~(value > 0.0)
            if bad.any():
                raise ValueError(
                    f"{parameter.name} must be positive "
                    f"(first invalid element at flat index {_first_true(bad)})"
                )
        validated[parameter.name] = value
    return validated


def is_scalar_call(point: object, parameters: Mapping[str, object]) -> bool:
    """Return whether every operand is a Python float (so the scalar path applies)."""

    return isinstance(point, float) and all(isinstance(v, float) for v in parameters.values())


def broadcast(point: Any, parameters: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Broadcast ``point`` and every parameter to one common shape (read-only views)."""

    np = _numpy()
    names = tuple(parameters)
    try:
        arrays = np.broadcast_arrays(point, *(parameters[name] for name in names))
    except ValueError as error:
        raise ValueError("the point and the parameters cannot be broadcast together") from error
    return arrays[0], dict(zip(names, arrays[1:], strict=True))


def inside_support(support: Support, point: Any) -> Any:
    """Return the boolean mask of points inside ``support`` (mirrors ``FamilySpec.contains``)."""

    np = _numpy()
    if support is Support.REAL_LINE:
        return np.ones(point.shape, dtype=bool)
    if support is Support.NON_NEGATIVE:
        return point >= 0.0
    return point > 0.0


def wrap_scalar(kernel: ScalarKernel, names: tuple[str, ...]) -> ArrayKernel:
    """Evaluate a scalar kernel over broadcast arrays; the results equal the scalar path exactly."""

    def apply(point: Any, parameters: Mapping[str, Any]) -> Any:
        np = _numpy()

        def element(value: float, *values: float) -> float:
            return kernel(value, dict(zip(names, values, strict=True)))

        ufunc = np.frompyfunc(element, 1 + len(names), 1)
        # The scalar kernels report an unrepresentable value themselves; the hardware
        # overflow flag their float arithmetic leaves behind is not a warning here.
        with np.errstate(all="ignore"):
            return ufunc(point, *(parameters[name] for name in names)).astype(np.float64)

    return apply


def finish_logpdf(spec: FamilySpec, point: Any, raw: Any) -> Any:
    """Apply the support convention to raw log-density values and reject unrepresentable ones.

    Points outside the support give ``-inf``.  A point inside the support whose
    value is not finite (an overflowing intermediate) raises ``ArithmeticError``
    naming the first such flat index, like the scalar operation.
    """

    np = _numpy()
    inside = inside_support(spec.support, point)
    bad = inside & ~np.isfinite(raw)
    if bad.any():
        raise ArithmeticError(
            f"the {spec.id.value} log-density is not representable "
            f"(first such element at flat index {_first_true(bad)})"
        )
    return np.where(inside, raw, -np.inf)


# --- numpy-native kernels -------------------------------------------------
# Each mirrors the formula of the scalar kernel of the same name in
# ``distributions`` / ``log_density``; they are evaluated under
# ``numpy.errstate(all="ignore")`` by ``evaluate_native``.


def _exponential_logpdf(x: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    return np.log(p["rate"]) - p["rate"] * x


def _weibull_min_logpdf(x: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    shape, scale = p["shape"], p["scale"]
    # ``(x - scale) / scale`` is exact in the numerator for ``scale/2 <= x <= 2*scale``
    # (Sterbenz), which covers the ``|delta| <= 0.5`` branch that uses it.
    delta = (x - scale) / scale
    log_ratio = np.where(np.abs(delta) <= 0.5, np.log1p(delta), np.log(x) - np.log(scale))
    exponent = shape * log_ratio
    near = np.log(shape) - np.log(scale) - 1.0 - log_ratio - (np.expm1(exponent) - exponent)
    far = np.log(shape) - np.log(scale) + (shape - 1.0) * log_ratio - np.exp(exponent)
    return np.where(np.abs(exponent) <= 0.5, near, far)


_SPLITTER: Final = 134217729.0  # 2**27 + 1, Veltkamp's splitter for binary64


def _two_product(a: Any, b: Any) -> tuple[Any, Any]:
    """Return ``(p, e)`` with ``p = fl(a * b)`` and ``a * b = p + e`` exactly (Dekker)."""

    p = a * b
    c = _SPLITTER * a
    a_high = c - (c - a)
    a_low = a - a_high
    c = _SPLITTER * b
    b_high = c - (c - b)
    b_low = b - b_high
    return p, ((a_high * b_high - p) + a_high * b_low + a_low * b_high) + a_low * b_low


def _scaled_difference(x: Any, location: Any, scale: Any) -> Any:
    """Return ``(x - location) / scale`` rounded once, like the scalar kernel's exact quotient.

    ``x - location`` and the quotient's remainder are carried through error-free
    transforms, so the result is the correctly rounded quotient of the exact
    difference rather than the twice-rounded ``(x - location) / scale``; the
    right tail of the Gumbel log-density is sensitive to that last bit.  Where an
    intermediate overflows the correction is dropped and the plain quotient stays.
    """

    np = _numpy()
    difference = x - location
    virtual = difference - x
    error = (x - (difference - virtual)) + (-location - virtual)
    quotient = difference / scale
    product, product_error = _two_product(quotient, scale)
    correction = (((difference - product) - product_error) + error) / scale
    return quotient + np.where(np.isfinite(correction), correction, 0.0)


def _gumbel_right_logpdf(x: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    scale = p["scale"]
    z = _scaled_difference(x, p["location"], scale)
    near = -np.log(scale) - 1.0 - (z + np.expm1(-z))
    far = -np.log(scale) - z - np.exp(-z)
    return np.where(np.abs(z) <= 0.5, near, far)


def _exponential_cdf(x: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    return np.where(x < 0.0, 0.0, -np.expm1(-p["rate"] * x))


def _exponential_sf(x: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    return np.where(x < 0.0, 1.0, np.exp(-p["rate"] * x))


def _weibull_min_cdf(x: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    return np.where(x <= 0.0, 0.0, -np.expm1(-((x / p["scale"]) ** p["shape"])))


def _weibull_min_sf(x: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    return np.where(x <= 0.0, 1.0, np.exp(-((x / p["scale"]) ** p["shape"])))


def _gumbel_right_cdf(x: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    return np.exp(-np.exp(-(x - p["location"]) / p["scale"]))


def _gumbel_right_sf(x: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    return -np.expm1(-np.exp(-(x - p["location"]) / p["scale"]))


def _exponential_ppf(q: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    return -np.log1p(-q) / p["rate"]


def _weibull_min_ppf(q: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    return p["scale"] * (-np.log1p(-q)) ** (1.0 / p["shape"])


def _gumbel_right_ppf(q: Any, p: Mapping[str, Any]) -> Any:
    np = _numpy()
    return p["location"] - p["scale"] * np.log(-np.log(q))


NATIVE_LOGPDF: Final[Mapping[FamilyId, ArrayKernel]] = {
    FamilyId.EXPONENTIAL: _exponential_logpdf,
    FamilyId.WEIBULL_MIN: _weibull_min_logpdf,
    FamilyId.GUMBEL_RIGHT: _gumbel_right_logpdf,
}
NATIVE_CDF: Final[Mapping[FamilyId, ArrayKernel]] = {
    FamilyId.EXPONENTIAL: _exponential_cdf,
    FamilyId.WEIBULL_MIN: _weibull_min_cdf,
    FamilyId.GUMBEL_RIGHT: _gumbel_right_cdf,
}
NATIVE_SF: Final[Mapping[FamilyId, ArrayKernel]] = {
    FamilyId.EXPONENTIAL: _exponential_sf,
    FamilyId.WEIBULL_MIN: _weibull_min_sf,
    FamilyId.GUMBEL_RIGHT: _gumbel_right_sf,
}
NATIVE_PPF: Final[Mapping[FamilyId, ArrayKernel]] = {
    FamilyId.EXPONENTIAL: _exponential_ppf,
    FamilyId.WEIBULL_MIN: _weibull_min_ppf,
    FamilyId.GUMBEL_RIGHT: _gumbel_right_ppf,
}
#: The families evaluated by numpy-native kernels; the others wrap the scalar kernels.
NATIVE_FAMILIES: Final = frozenset(NATIVE_LOGPDF)


def evaluate_native(kernel: ArrayKernel, point: Any, parameters: Mapping[str, Any]) -> Any:
    """Run a numpy-native kernel with floating-point warnings silenced.

    Overflow and underflow are the legitimate limits of these formulas (a CDF
    saturating at 1, a density underflowing to 0), so they are not warnings;
    values that are not representable are detected from the result instead.
    """

    np = _numpy()
    with np.errstate(all="ignore"):
        return np.asarray(kernel(point, parameters), dtype=np.float64)


__all__ = [
    "NATIVE_CDF",
    "NATIVE_FAMILIES",
    "NATIVE_LOGPDF",
    "NATIVE_PPF",
    "NATIVE_SF",
    "ArrayKernel",
    "ScalarKernel",
    "broadcast",
    "coerce",
    "evaluate_native",
    "finish_logpdf",
    "inside_support",
    "is_scalar_call",
    "require_finite",
    "require_probability",
    "validated_parameters",
    "wrap_scalar",
]
