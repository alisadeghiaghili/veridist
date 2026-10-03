"""Build tuples of observations from parallel arrays.

``lifetimes_from_arrays(time, event)`` and ``values_from_arrays(value, event)``
turn two equal-length one-dimensional array-likes into the observation tuples the
fits take, validating the whole input vectorially first so that the error names the
first bad row instead of failing somewhere inside a Python loop.  numpy is imported
only when one of them is called.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime
from veridist.domain.values import ExactValue, RealObservation, RightCensoredValue


def _numpy() -> Any:
    return import_module("numpy")


def _first(mask: Any) -> int:
    return int(mask.argmax())


def _numeric_column(data: object, name: str) -> Any:
    """Return ``data`` as a one-dimensional float64 array.

    Only integer and floating dtypes of at most 64 bits are accepted: a ``bool``
    column is not a numeric one, and a wider float cannot be checked to keep a
    positive value representable.  ``TypeError`` for the wrong kind of data,
    ``ValueError`` for a wrong shape.
    """

    np = _numpy()
    array = np.asarray(data)
    if array.dtype.kind not in "iuf" or array.dtype.itemsize > 8:
        raise TypeError(f"{name} must be an array of real numbers of at most 64 bits")
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array.astype(np.float64)


def _event_column(data: object, length: int) -> Any:
    """Return ``data`` as a boolean array: booleans, or integers that are all 0 or 1."""

    np = _numpy()
    array = np.asarray(data)
    if array.size == 0 and array.dtype.kind == "f":
        array = array.astype(bool)  # ``[]`` defaults to float64 but has no value to misread
    if array.dtype.kind == "b":
        flags = array
    elif array.dtype.kind in "iu":
        bad = (array != 0) & (array != 1)
        if bad.any():
            raise ValueError(f"event must be boolean or 0/1 (first bad row {_first(bad)})")
        flags = array.astype(bool)
    else:
        raise TypeError("event must be a boolean array or an integer array of 0 and 1")
    if flags.ndim != 1:
        raise ValueError("event must be one-dimensional")
    if len(flags) != length:
        raise ValueError("the columns must have the same length")
    return flags


def _checked(column: Any, *, name: str, non_negative: bool) -> Any:
    """Return ``column`` as a list after checking every row; the error names the first bad one."""

    np = _numpy()
    bad = ~np.isfinite(column)
    if non_negative:
        bad |= column < 0.0
    if bad.any():
        what = "finite and non-negative" if non_negative else "finite"
        raise ValueError(f"{name} must be {what} (first bad row {_first(bad)})")
    return column.tolist()


def lifetimes_from_arrays(time: object, event: object, /) -> tuple[LifetimeObservation, ...]:
    """Build the lifetime observations of two parallel columns.

    Row ``i`` is an :class:`~veridist.domain.lifetimes.ExactLifetime` when
    ``event[i]`` is true (the event was observed at ``time[i]``) and a
    :class:`~veridist.domain.lifetimes.RightCensoredLifetime` when it is false
    (the unit was still event-free at ``time[i]``).

    ``time`` is a one-dimensional array-like of finite, non-negative real numbers
    (integers or floats of at most 64 bits) and ``event`` a boolean array or an
    integer array whose values are all 0 or 1; anything else is rejected
    (``TypeError`` for the wrong kind of data, ``ValueError`` for a bad value,
    naming the first bad row).  The columns must have equal length, and may both
    be empty.  The result equals building the tuple row by row with the lifetime
    constructors.
    """

    column = _numeric_column(time, "time")
    flags = _event_column(event, len(column))
    times = _checked(column, name="time", non_negative=True)
    return tuple(
        ExactLifetime(value) if observed else RightCensoredLifetime(value)
        for value, observed in zip(times, flags.tolist(), strict=True)
    )


def values_from_arrays(value: object, event: object, /) -> tuple[RealObservation, ...]:
    """Build the real-valued observations of two parallel columns.

    The real-line analogue of :func:`lifetimes_from_arrays`: row ``i`` is an
    :class:`~veridist.domain.values.ExactValue` when ``event[i]`` is true and a
    :class:`~veridist.domain.values.RightCensoredValue` when it is false.  Every
    value must be finite; it may be negative.
    """

    column = _numeric_column(value, "value")
    flags = _event_column(event, len(column))
    values = _checked(column, name="value", non_negative=False)
    return tuple(
        ExactValue(item) if observed else RightCensoredValue(item)
        for item, observed in zip(values, flags.tolist(), strict=True)
    )


__all__ = ["lifetimes_from_arrays", "values_from_arrays"]
