"""Truncation windows of observations: their normal form, their consistency and shared semantics.

A truncation window ``(a, b]`` says that an observation was drawn from the distribution
conditioned on the value lying in it.  Every observation type carries the window as two
optional keyword-only attributes, ``truncation_lower`` (``a``) and ``truncation_upper``
(``b``); ``None`` means "not truncated on that side".  A window is stored in one normal form so
that equal windows compare equal: negative infinity as a lower end and positive infinity as an
upper end are ``None``, and for a lifetime a lower end of ``0`` is ``None`` as well (a lifetime
cannot be below ``0``).

The consistency rules refuse, and never clip, an observation that its own window excludes: an
exact value needs ``a <= x <= b``, an interval ``a <= lower < upper <= b``, a right-censoring
point ``a <= c < b`` (it then stands for ``(c, b]``) and a left-censoring point ``a < u <= b``
(it stands for ``(a, u]``).  Messages name the attribute and never carry a data value.

Observations without a window keep the equality, hash, ``repr`` and pickle of the types before
truncation existed: :func:`closed_observation` gives them their own state, ``repr`` and hash.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from decimal import Decimal
from math import inf, isnan
from typing import Any, TypeVar

from veridist.domain._numeric import is_real

_TRUNCATION_FIELDS = ("truncation_lower", "truncation_upper")
_Class = TypeVar("_Class", bound=type)


def _end(value: object, name: str, *, open_end: float) -> float | None:
    """One end of a window as a ``float``, or ``None`` for ``None`` and for the open infinity."""

    if value is None:
        return None
    if not (is_real(value) or isinstance(value, Decimal)):
        raise TypeError(f"{name} must be a finite real number or None")
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except OverflowError as error:
        raise ValueError(f"{name} must be finite or None") from error
    if numeric == open_end:
        return None
    if isnan(numeric) or numeric in (inf, -inf):
        raise ValueError(f"{name} must be finite or None")
    if value != 0 and numeric == 0.0:
        raise ValueError(f"nonzero {name} must remain representable")
    return numeric


def value_window(lower: object, upper: object) -> tuple[float | None, float | None]:
    """The normal form of a window for a real-line observation, or the refusal."""

    low = _end(lower, "truncation_lower", open_end=-inf)
    high = _end(upper, "truncation_upper", open_end=inf)
    _ordered(low, high)
    return low, high


def lifetime_window(lower: object, upper: object) -> tuple[float | None, float | None]:
    """The normal form of a window for a lifetime: ``0 <= a < b`` with ``b > 0``."""

    low = _end(lower, "truncation_lower", open_end=-inf)
    high = _end(upper, "truncation_upper", open_end=inf)
    if low is not None and low < 0.0:
        raise ValueError("truncation_lower must be non-negative")
    if high is not None and high <= 0.0:
        raise ValueError("truncation_upper must be positive")
    _ordered(low, high)
    return (None if low == 0.0 else low), high


def _ordered(low: float | None, high: float | None) -> None:
    if low is not None and high is not None and not low < high:
        raise ValueError("truncation_lower must be less than truncation_upper")


def check_exact(point: float, low: float | None, high: float | None) -> None:
    """An exact observation must lie in the closed window."""

    if low is not None and point < low:
        raise ValueError("truncation_lower must not be after the exact observation")
    if high is not None and point > high:
        raise ValueError("truncation_upper must not be before the exact observation")


def check_right(point: float, low: float | None, high: float | None) -> None:
    """A right-censoring point needs ``a <= c < b``."""

    if low is not None and point < low:
        raise ValueError("truncation_lower must not be after the censoring point")
    if high is not None and point >= high:
        raise ValueError("truncation_upper must be after the censoring point")


def check_left(point: float, low: float | None, high: float | None) -> None:
    """A left-censoring point needs ``a < u <= b``."""

    if low is not None and point <= low:
        raise ValueError("truncation_lower must be before the censoring point")
    if high is not None and point > high:
        raise ValueError("truncation_upper must not be before the censoring point")


def check_interval(lower: float, upper: float, low: float | None, high: float | None) -> None:
    """An interval needs ``a <= lower < upper <= b``."""

    if low is not None and lower < low:
        raise ValueError("truncation_lower must not be after the lower bound")
    if high is not None and upper > high:
        raise ValueError("truncation_upper must not be before the upper bound")


def _state(self: Any) -> list[object]:
    values = [getattr(self, member.name) for member in fields(self)]
    if values[-1] is None and values[-2] is None:
        del values[-2:]
    return values


def _set_state(self: Any, state: object) -> None:
    members = fields(self)
    given = list(state)  # type: ignore[call-overload]
    padded = given + [None] * (len(members) - len(given))
    for member, value in zip(members, padded, strict=True):
        object.__setattr__(self, member.name, value)


def _repr(self: Any) -> str:
    shown = [
        f"{member.name}={getattr(self, member.name)!r}"
        for member in fields(self)
        if member.name not in _TRUNCATION_FIELDS or getattr(self, member.name) is not None
    ]
    return f"{type(self).__qualname__}({', '.join(shown)})"


def _hash(self: Any) -> int:
    return hash(tuple(_state(self)))


def closed_observation(cls: _Class) -> _Class:
    """Give a frozen slotted observation class the state, ``repr`` and hash of its untruncated form.

    The truncation attributes are the last two fields.  Without a window the pickled state,
    the ``repr`` and the hash are exactly those of the class without them (an old pickle loads
    into the new class); with a window they list the window as well.
    """

    setters: dict[str, Callable[..., Any]] = {
        "__getstate__": _state,
        "__setstate__": _set_state,
        "__repr__": _repr,
        "__hash__": _hash,
    }
    for name, function in setters.items():
        setattr(cls, name, function)
    return cls


__all__ = [
    "check_exact",
    "check_interval",
    "check_left",
    "check_right",
    "closed_observation",
    "lifetime_window",
    "value_window",
]
