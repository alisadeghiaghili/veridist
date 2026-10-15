"""The exact sum of squares of a batch of binary64 values, as an integer.

``squares_units(values)`` returns ``sum(x * x)`` over the batch exactly, in units of
``2**-2148`` (the square of the unit ``2**-1074`` of :mod:`veridist.scale._accumulator`).
Every binary64 value is an integer multiple of ``2**-1074``, so its square is an
integer multiple of ``2**-2148`` and the sum is an exact integer in that unit; nothing
is rounded, nothing underflows and nothing overflows, whatever the magnitudes.

Exactness argument. The product ``p = fl(x * x)`` differs from ``x * x`` by a rounding
error ``e`` that is itself a binary64 number, and Dekker's algorithm computes it with
plain binary64 arithmetic: Veltkamp's split ``t = (2**27 + 1) * x``, ``h = t - (t - x)``,
``l = x - h`` writes ``x = h + l`` with ``h`` and ``l`` of at most 26 significant bits,
so that ``h*h``, ``2*h*l`` and ``l*l`` are exact, and ``e = ((h*h - p) + 2*h*l) + l*l``
is evaluated without a rounding error (each partial result is representable, which is
the content of Dekker's theorem). Hence ``x * x == p + e`` exactly, and ``p`` and ``e``
are summed exactly by :class:`~veridist.scale._accumulator.ExactSum`. The proof needs
three things, which the magnitude classes below enforce:

* no overflow of ``(2**27 + 1) * x`` or of ``x * x``: ``|x| < 2**500``;
* no underflow that would round a term: every partial result is an integer multiple
  of ``2**(2 * (E - 52))`` for ``E`` the binary exponent of ``x``, which is a multiple
  of ``2**-1074`` once ``|x| >= 2**-480`` (then ``2 * (E - 52) >= -1064``);
* the result is a multiple of the accumulator unit ``2**-1074``, which the previous
  point gives.

A value outside ``[2**-480, 2**500)`` is multiplied by an exact power of two first:
``|x| >= 2**500`` by ``2**-600`` (to ``[2**-100, 2**424)``) and ``0 < |x| < 2**-480``
by ``2**600`` (to ``[2**-474, 2**120)``; a subnormal times ``2**600`` is exact). Both
land in the safe window, so the pair ``p, e`` of the scaled value is exact, and the
sum of the scaled squares is moved back by an exact integer shift: left by ``2274``
bits for the large class (``x**2 = x'**2 * 2**1200``) and right by ``126`` bits for the
small class (``x**2 = x'**2 * 2**-1200``). The right shift loses nothing: for
``x = m * 2**-1074`` the scaled square is ``m**2 * 2**-948 = (m**2 * 2**126) * 2**-1074``,
so the units of the small class are a multiple of ``2**126``. Zero belongs to the
safe window (its pair is zero).

Every step is integer or exact binary64 arithmetic, so the result depends only on the
multiset of values, never on their order or batching.
"""

from __future__ import annotations

from typing import Any, Final

from veridist.scale._accumulator import ExactSum
from veridist.scale._kernels import numpy_module

SQUARE_UNITS_EXPONENT: Final = 2148
"""The unit of a sum of squares is ``2**-SQUARE_UNITS_EXPONENT``."""

MAX_SQUARE_UNITS: Final = (((1 << 53) - 1) ** 2) << 4090
"""The largest absolute value of one square of a finite binary64, in units."""

_SPLIT: Final = 134217729.0
"""Veltkamp's constant ``2**27 + 1``."""

_LOW_EDGE: Final = 2.0**-480
_HIGH_EDGE: Final = 2.0**500
_SHRINK: Final = 2.0**-600
_GROW: Final = 2.0**600
_WINDOW_SHIFT: Final = 1074
_LARGE_SHIFT: Final = 2274
_SMALL_SHIFT: Final = 126


def _window_units(values: Any) -> int:
    """``sum(x * x)`` in units of ``2**-1074``, for ``|x|`` in ``[2**-480, 2**500)`` or zero."""

    np = numpy_module()
    # In-place form of: h = (C*x) - ((C*x) - x); l = x - h; p = x*x;
    # e = ((h*h - p) + 2*h*l) + l*l, with the same operations in the same order.
    high = np.multiply(values, _SPLIT)
    low = np.subtract(high, values)
    np.subtract(high, low, out=high)
    np.subtract(values, high, out=low)
    product = np.multiply(values, values)
    error = np.multiply(high, high)
    np.subtract(error, product, out=error)
    np.multiply(high, low, out=high)
    np.multiply(high, 2.0, out=high)
    np.add(error, high, out=error)
    np.multiply(low, low, out=low)
    np.add(error, low, out=error)
    accumulator = ExactSum()
    accumulator.update(product)
    accumulator.update(error)
    return accumulator.total_units()


def squares_units(values: Any) -> int:
    """The exact ``sum(x * x)`` of a finite float64 array, in units of ``2**-2148``.

    ``values`` is a one-dimensional float64 ``numpy.ndarray`` of finite numbers (the
    caller has checked that). An empty array gives zero.
    """

    np = numpy_module()
    magnitude = np.abs(values)
    small = (magnitude < _LOW_EDGE) & (magnitude > 0.0)
    large = magnitude >= _HIGH_EDGE
    any_small, any_large = bool(small.any()), bool(large.any())
    if not (any_small or any_large):
        return _window_units(values) << _WINDOW_SHIFT
    total = _window_units(values[~(small | large)]) << _WINDOW_SHIFT
    if any_large:
        total += _window_units(values[large] * _SHRINK) << _LARGE_SHIFT
    if any_small:
        total += _window_units(values[small] * _GROW) >> _SMALL_SHIFT
    return total


__all__ = ["MAX_SQUARE_UNITS", "SQUARE_UNITS_EXPONENT", "squares_units"]
