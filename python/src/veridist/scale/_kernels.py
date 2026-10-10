"""Batch validation and per-element term kernels shared by the exact states.

Every transcendental or product term is computed with one formula for the
whole batch: ``numpy.log`` for logarithms and a plain ``x * x`` product for
squares, never ``numpy.power`` (which special-cases some exponents). A kernel
returns a new contiguous float64 array and its result for an element does not
depend on the element's position in the batch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode


def numpy_module() -> Any:
    """Load numpy on first use so that importing the package stays stdlib-only."""

    return import_module("numpy")


def float64_batch(values: object) -> Any:
    """Return ``values`` as a C-contiguous one-dimensional float64 array, unchanged.

    ``TypeError`` unless ``values`` is a plain ``numpy.ndarray`` (a masked array
    is refused because its mask would be silently ignored) of dtype ``float64``
    in native byte order; ``ValueError`` unless it is one-dimensional. No value is
    inspected and nothing is converted: a wider or narrower dtype is never cast.
    """

    np = numpy_module()
    if not isinstance(values, np.ndarray) or isinstance(values, np.ma.MaskedArray):
        raise TypeError("values must be a numpy.ndarray")
    if values.dtype != np.float64:
        raise TypeError("values must have dtype float64")
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    return np.ascontiguousarray(values)


def require_finite(values: Any) -> None:
    """Refuse NaN and both infinities with ``NON_FINITE_VALUE``."""

    if not numpy_module().isfinite(values).all():
        raise ScaleStateError(ScaleStateErrorCode.NON_FINITE_VALUE)


def require_non_negative(values: Any) -> None:
    """Refuse a value below zero (``-0.0`` is zero) with ``VALUE_OUT_OF_SUPPORT``."""

    if not (values >= 0.0).all():
        raise ScaleStateError(ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)


def require_positive(values: Any) -> None:
    """Refuse zero and negative values with ``VALUE_OUT_OF_SUPPORT``."""

    if not (values > 0.0).all():
        raise ScaleStateError(ScaleStateErrorCode.VALUE_OUT_OF_SUPPORT)


def squares(values: Any) -> Any:
    """Per-element binary64 product ``x * x``; an overflow is ``DERIVED_VALUE_NOT_FINITE``.

    The product is rounded to binary64 for each element (it is not an exact
    square); only the later summation is exact.
    """

    np = numpy_module()
    with np.errstate(over="ignore"):
        result = np.multiply(values, values)
    if not np.isfinite(result).all():
        raise ScaleStateError(ScaleStateErrorCode.DERIVED_VALUE_NOT_FINITE)
    return result


def logarithms(values: Any) -> Any:
    """Per-element ``numpy.log`` of finite strictly positive values (always finite)."""

    return numpy_module().log(values)
