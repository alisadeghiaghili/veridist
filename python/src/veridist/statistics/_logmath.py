"""Small log-domain helpers shared by the log-probability kernels (numpy arrays in and out)."""

from __future__ import annotations

from importlib import import_module
from math import frexp, log
from typing import Any, Final

np: Any = import_module("numpy")

_LOG_TWO: Final = log(2.0)
#: ``|(x - scale) / scale|`` up to which ``log1p`` of that relative difference is used.
_NEAR_SCALE: Final = 0.5


def log1mexp(value: Any) -> Any:
    """``log(1 - exp(value))`` for ``value <= 0``: ``0`` at ``-inf`` and ``-inf`` at ``0``.

    ``log(-expm1(value))`` for ``value > -log 2`` and ``log1p(-exp(value))`` below, so neither
    form loses digits.  A positive ``value`` gives NaN (it has no real logarithm).
    """

    with np.errstate(all="ignore"):
        return np.where(value > -_LOG_TWO, np.log(-np.expm1(value)), np.log1p(-np.exp(value)))


def log_ratio(x: Any, scale: float) -> Any:
    """``log(x / scale)`` for ``x >= 0`` (``-inf`` at zero, ``+inf`` at infinity), array in and out.

    Near the scale it is ``log1p`` of the exactly subtracted relative difference (the
    subtraction is exact for ``scale / 2 <= x <= 2 scale``).  Elsewhere the binary exponents are
    split off first, so no ``log(x) - log(scale)`` cancellation of two large logarithms is left
    and a subnormal ``x`` keeps its exact mantissa; ``x / scale`` is never formed there.
    """

    with np.errstate(all="ignore"):
        delta = (x - scale) / scale
        x_mantissa, x_exponent = np.frexp(x)
        scale_mantissa, scale_exponent = frexp(scale)
        far = np.log(x_mantissa / scale_mantissa) + (x_exponent - scale_exponent) * _LOG_TWO
        return np.where(np.abs(delta) <= _NEAR_SCALE, np.log1p(delta), far)


__all__ = ["log1mexp", "log_ratio"]
