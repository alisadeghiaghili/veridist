"""Shared acceptance rules for real-number arguments of the public entry points.

Python and numpy scalars are accepted alike: ``numbers.Real`` covers ``int``,
``float`` and every ``numpy.integer`` and ``numpy.floating`` type, and
``numbers.Integral`` covers ``int`` and every ``numpy.integer`` type.  ``bool`` and
``numpy.bool_`` are never numbers here, because a truth value passed where a
number is expected is almost always a mistake.  The checks use only the
standard library, so a program that never touches an array never imports numpy.
"""

from __future__ import annotations

from numbers import Integral, Real


def is_real(value: object) -> bool:
    """Return whether ``value`` is a real scalar: ``int``, ``float`` or a numpy real, not a bool."""

    return isinstance(value, Real) and not isinstance(value, bool)


def is_integer(value: object) -> bool:
    """Return whether ``value`` is an integer scalar (``int`` or ``numpy.integer``), not a bool."""

    return isinstance(value, Integral) and not isinstance(value, bool)


def is_float(value: object) -> bool:
    """Return whether ``value`` is a non-integer real scalar (``float`` or ``numpy.floating``).

    Used where an argument must be a real number and an integer is a mistake, as
    it was when only built-in floats were accepted.
    """

    return is_real(value) and not is_integer(value)


__all__ = ["is_float", "is_integer", "is_real"]
