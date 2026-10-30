"""Build tuples of observations from parallel arrays.

``lifetimes_from_arrays(time, event)`` and ``values_from_arrays(value, event)``
turn two equal-length one-dimensional array-likes into the observation tuples the
fits take, validating the whole input vectorially first so that the error names the
first bad row instead of failing somewhere inside a Python loop.
``lifetimes_from_columns`` and ``values_from_columns`` are the multi-column form: an
explicit status column says what each row is (exact, right-, left- or interval-censored),
and optional columns give the upper bound of an interval and the truncation window of each
row.  numpy is imported only when one of them is called.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from veridist.domain.lifetimes import (
    AnyLifetimeObservation,
    ExactLifetime,
    IntervalCensoredLifetime,
    LeftCensoredLifetime,
    LifetimeObservation,
    RightCensoredLifetime,
)
from veridist.domain.values import (
    AnyRealObservation,
    ExactValue,
    IntervalCensoredValue,
    LeftCensoredValue,
    RealObservation,
    RightCensoredValue,
)

#: The status codes of the column constructors (those of ``survival::Surv``).
STATUS_RIGHT, STATUS_EXACT, STATUS_LEFT, STATUS_INTERVAL = 0, 1, 2, 3


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


def _status_column(data: object, length: int) -> Any:
    """Return ``data`` as an int8 array of status codes: booleans, or integers 0, 1, 2 or 3."""

    np = _numpy()
    array = np.asarray(data)
    if array.size == 0 and array.dtype.kind == "f":
        array = array.astype(np.int8)  # ``[]`` defaults to float64 but has no value to misread
    if array.dtype.kind == "b":
        codes = array.astype(np.int8)
    elif array.dtype.kind in "iu":
        bad = (array < STATUS_RIGHT) | (array > STATUS_INTERVAL)
        if bad.any():
            raise ValueError(
                f"status must be boolean or 0, 1, 2 or 3 (first bad row {_first(bad)})"
            )
        codes = array.astype(np.int8)
    else:
        raise TypeError("status must be a boolean array or an integer array of 0, 1, 2 and 3")
    if codes.ndim != 1:
        raise ValueError("status must be one-dimensional")
    if len(codes) != length:
        raise ValueError("the columns must have the same length")
    return codes


def _optional_column(data: object, name: str, length: int) -> Any:
    """Return an optional numeric column as float64 (``None`` stays ``None``), length checked."""

    if data is None:
        return None
    column = _numeric_column(data, name)
    if len(column) != length:
        raise ValueError("the columns must have the same length")
    return column


def _upper_column(upper: Any, codes: Any) -> Any:
    """Check ``upper``: finite on interval rows, NaN on every other row."""

    np = _numpy()
    interval = codes == STATUS_INTERVAL
    if upper is None:
        if interval.any():
            raise ValueError(
                f"upper is required for interval-censored rows (first bad row {_first(interval)})"
            )
        return upper
    bad = interval & ~np.isfinite(upper)
    if bad.any():
        raise ValueError(
            f"upper must be finite on interval-censored rows (first bad row {_first(bad)})"
        )
    bad = ~interval & ~np.isnan(upper)
    if bad.any():
        raise ValueError(
            "upper must be NaN on rows that are not interval-censored "
            f"(first bad row {_first(bad)})"
        )
    return upper


def _window_column(column: Any, name: str, *, lower_end: bool, lifetime: bool) -> Any:
    """Check a truncation column: finite or the open infinity of its side, never NaN."""

    np = _numpy()
    if column is None:
        return None
    open_end = -np.inf if lower_end else np.inf
    bad = np.isnan(column) | (np.isinf(column) & (column != open_end))
    if bad.any():
        infinity = "negative infinity" if lower_end else "positive infinity"
        raise ValueError(f"{name} must be finite or {infinity} (first bad row {_first(bad)})")
    if lifetime:
        bad = np.isfinite(column) & ((column < 0.0) if lower_end else (column <= 0.0))
        if bad.any():
            sign = "non-negative" if lower_end else "positive"
            raise ValueError(f"{name} must be {sign} (first bad row {_first(bad)})")
    return column


def _from_columns(
    value: object,
    status: object,
    upper: object,
    truncation_lower: object,
    truncation_upper: object,
    *,
    lifetime: bool,
) -> tuple[Any, ...]:
    np = _numpy()
    column = _numeric_column(value, "value")
    codes = _status_column(status, len(column))
    ends = _upper_column(_optional_column(upper, "upper", len(column)), codes)
    lowest = _window_column(
        _optional_column(truncation_lower, "truncation_lower", len(column)),
        "truncation_lower",
        lower_end=True,
        lifetime=lifetime,
    )
    highest = _window_column(
        _optional_column(truncation_upper, "truncation_upper", len(column)),
        "truncation_upper",
        lower_end=False,
        lifetime=lifetime,
    )
    bad = ~np.isfinite(column)
    if lifetime:
        bad |= column < 0.0
    if bad.any():
        what = "finite and non-negative" if lifetime else "finite"
        raise ValueError(f"value must be {what} (first bad row {_first(bad)})")
    simple = (
        (RightCensoredLifetime, ExactLifetime, LeftCensoredLifetime)
        if lifetime
        else (RightCensoredValue, ExactValue, LeftCensoredValue)
    )
    interval = IntervalCensoredLifetime if lifetime else IntervalCensoredValue
    points = column.tolist()
    flags = codes.tolist()
    bounds = None if ends is None else ends.tolist()
    first = None if lowest is None else lowest.tolist()
    last = None if highest is None else highest.tolist()
    rows: list[Any] = []
    for index, point in enumerate(points):
        window = {
            "truncation_lower": None if first is None else first[index],
            "truncation_upper": None if last is None else last[index],
        }
        try:
            if flags[index] == STATUS_INTERVAL:
                assert bounds is not None
                rows.append(interval(point, bounds[index], **window))
            else:
                rows.append(simple[flags[index]](point, **window))
        except ValueError as error:
            raise ValueError(f"{error} (first bad row {index})") from None
    return tuple(rows)


def lifetimes_from_columns(
    value: object,
    status: object,
    /,
    *,
    upper: object = None,
    truncation_lower: object = None,
    truncation_upper: object = None,
) -> tuple[AnyLifetimeObservation, ...]:
    """Build the lifetime observations of parallel columns, one status per row.

    ``status`` is a boolean array or an integer array with the codes of ``survival::Surv``:
    ``1`` an :class:`~veridist.domain.lifetimes.ExactLifetime` (as ``event=1`` of
    :func:`lifetimes_from_arrays`), ``0`` a
    :class:`~veridist.domain.lifetimes.RightCensoredLifetime`, ``2`` a
    :class:`~veridist.domain.lifetimes.LeftCensoredLifetime` and ``3`` an
    :class:`~veridist.domain.lifetimes.IntervalCensoredLifetime`; any other value is refused.
    Row ``i`` uses ``value[i]`` as the observation, the censoring point or the lower bound
    according to its status.

    ``upper`` is the upper bound of the interval rows: it must be finite and greater than
    ``value`` on every status-3 row and NaN on every other row (so that the status column, and
    never a missing value, says what a row means); it may be omitted when no row has status 3.
    ``truncation_lower`` and ``truncation_upper`` hold one number per row, finite or the open
    infinity (negative infinity for the lower end and positive infinity for the upper end),
    which means "not truncated" on that side; for a lifetime a lower end of ``0`` means the
    same.  Omitted, no row is truncated.  NaN and nulls are refused in every column.

    Every error names the first bad row.  The result equals building the tuple row by row with
    the observation constructors (``TypeError`` for the wrong kind of data, ``ValueError`` for
    a bad value, a wrong shape, columns of different length, or a row that its truncation window
    excludes); the columns may be empty.
    """

    rows: tuple[AnyLifetimeObservation, ...] = _from_columns(
        value, status, upper, truncation_lower, truncation_upper, lifetime=True
    )
    return rows


def values_from_columns(
    value: object,
    status: object,
    /,
    *,
    upper: object = None,
    truncation_lower: object = None,
    truncation_upper: object = None,
) -> tuple[AnyRealObservation, ...]:
    """Build the real-valued observations of parallel columns, one status per row.

    The real-line analogue of :func:`lifetimes_from_columns`, with the types of
    :mod:`veridist.domain.values`; every value must be finite and may be negative.
    """

    rows: tuple[AnyRealObservation, ...] = _from_columns(
        value, status, upper, truncation_lower, truncation_upper, lifetime=False
    )
    return rows


__all__ = [
    "lifetimes_from_arrays",
    "lifetimes_from_columns",
    "values_from_arrays",
    "values_from_columns",
]
