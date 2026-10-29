"""The combination matrix: which kind of observation, truncation and weight each operation takes.

Every cell of the cross product of a censoring kind (exact, right, left, interval), a
truncation type (none, left, right, two-sided) and a weight kind (none, integer frequency,
analytic) is either supported by an operation or refused with a typed
:class:`~veridist.engine.errors.CapabilityCode`; nothing falls through untyped.  The matrix is
one table, :func:`cell_status`, and the entry points that gate their input consult it.

* ``LOG_LIKELIHOOD``: all sixteen censoring and truncation cells, with no weights or with
  integer frequency weights.
* ``FIT``: exact and right-censored observations without a window, with no weights or
  frequency weights (as before censoring and truncation were modelled); every other cell is
  refused until fitting for that data kind is available.
* Analytic weights are refused by both until they are available.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import product
from typing import Any, Final, cast

from veridist.domain.lifetimes import (
    ExactLifetime,
    IntervalCensoredLifetime,
    LeftCensoredLifetime,
    RightCensoredLifetime,
)
from veridist.domain.values import (
    ExactValue,
    IntervalCensoredValue,
    LeftCensoredValue,
    RightCensoredValue,
)
from veridist.engine.errors import CapabilityCode, CapabilityError


class CensoringKind(StrEnum):
    """What is known about an observation: its value, or a set the value lies in."""

    EXACT = "exact"
    RIGHT = "right"
    LEFT = "left"
    INTERVAL = "interval"


class TruncationKind(StrEnum):
    """Which sides of its window an observation is truncated on."""

    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    TWO_SIDED = "two_sided"


class WeightKind(StrEnum):
    """The declared kind of the weights of the rows."""

    NONE = "none"
    FREQUENCY = "frequency"
    ANALYTIC = "analytic"


class CellOperation(StrEnum):
    """The operations the matrix describes."""

    LOG_LIKELIHOOD = "log_likelihood"
    FIT = "fit"


FIT_UNAVAILABLE_MESSAGE: Final = "not supported until fitting for this data kind is available"
ANALYTIC_WEIGHTS_MESSAGE: Final = (
    "analytic weights are not supported yet; they arrive in a later release"
)
_DECLARE = (
    "declare it per observation with the typed observation classes "
    "(LeftCensoredLifetime, IntervalCensoredLifetime, LeftCensoredValue, "
    "IntervalCensoredValue, or the truncation_lower and truncation_upper attributes) "
    "or with lifetimes_from_columns and values_from_columns"
)
KEYWORD_MESSAGES: Final = {
    "censoring": f"the censoring keyword takes only 'right'; {_DECLARE}",
    "truncation": f"the truncation keyword is not supported; {_DECLARE}",
}


@dataclass(frozen=True, slots=True)
class CellStatus:
    """Whether a cell is supported and, when it is not, the typed reason."""

    supported: bool
    code: CapabilityCode | None = None
    message: str | None = None


_SUPPORTED: Final = CellStatus(True)


def cell_status(
    operation: CellOperation,
    censoring: CensoringKind,
    truncation: TruncationKind,
    weights: WeightKind,
) -> CellStatus:
    """The status of one cell of the matrix for ``operation``."""

    if (
        type(operation) is not CellOperation
        or type(censoring) is not CensoringKind
        or type(truncation) is not TruncationKind
        or type(weights) is not WeightKind
    ):
        raise TypeError("a cell is an operation, a censoring kind, a truncation and a weight kind")
    if weights is WeightKind.ANALYTIC:
        return CellStatus(
            False, CapabilityCode.ANALYTIC_WEIGHTS_UNSUPPORTED, ANALYTIC_WEIGHTS_MESSAGE
        )
    if operation is CellOperation.FIT and (
        censoring in (CensoringKind.LEFT, CensoringKind.INTERVAL)
        or truncation is not TruncationKind.NONE
    ):
        return CellStatus(False, CapabilityCode.DATA_KIND_FIT_UNAVAILABLE, FIT_UNAVAILABLE_MESSAGE)
    return _SUPPORTED


def all_cells() -> tuple[tuple[CensoringKind, TruncationKind, WeightKind], ...]:
    """The full cross product of the matrix, in a fixed order."""

    return tuple(product(CensoringKind, TruncationKind, WeightKind))


CENSORING_KINDS: Final = {
    ExactLifetime: CensoringKind.EXACT,
    RightCensoredLifetime: CensoringKind.RIGHT,
    LeftCensoredLifetime: CensoringKind.LEFT,
    IntervalCensoredLifetime: CensoringKind.INTERVAL,
    ExactValue: CensoringKind.EXACT,
    RightCensoredValue: CensoringKind.RIGHT,
    LeftCensoredValue: CensoringKind.LEFT,
    IntervalCensoredValue: CensoringKind.INTERVAL,
}


def observation_cell(observation: object) -> tuple[CensoringKind, TruncationKind]:
    """The censoring kind and the truncation type of one observation.

    ``TypeError`` for an object that is not one of the eight observation types.
    """

    kind = CENSORING_KINDS.get(type(observation))
    if kind is None:
        raise TypeError("observation must be a lifetime or real-value observation")
    window = cast(Any, observation)
    low = window.truncation_lower is not None
    high = window.truncation_upper is not None
    if low and high:
        return kind, TruncationKind.TWO_SIDED
    if low:
        return kind, TruncationKind.LEFT
    if high:
        return kind, TruncationKind.RIGHT
    return kind, TruncationKind.NONE


def fit_unavailable() -> CapabilityError:
    """The typed refusal of an observation kind that fitting does not take yet."""

    return CapabilityError(CapabilityCode.DATA_KIND_FIT_UNAVAILABLE, FIT_UNAVAILABLE_MESSAGE)


def require_fit_supported(observations: Iterable[object]) -> None:
    """Refuse, with the typed code of the matrix, the first observation a fit cannot take.

    ``CapabilityError`` for a left- or interval-censored observation or one with a truncation
    window; ``TypeError`` for an object that is not an observation at all.
    """

    seen: set[tuple[CensoringKind, TruncationKind]] = set()
    for observation in observations:
        cell = observation_cell(observation)
        if cell in seen:
            continue
        status = cell_status(CellOperation.FIT, cell[0], cell[1], WeightKind.NONE)
        if not status.supported:
            raise fit_unavailable()
        seen.add(cell)


__all__ = [
    "ANALYTIC_WEIGHTS_MESSAGE",
    "CENSORING_KINDS",
    "FIT_UNAVAILABLE_MESSAGE",
    "KEYWORD_MESSAGES",
    "CellOperation",
    "CellStatus",
    "CensoringKind",
    "TruncationKind",
    "WeightKind",
    "all_cells",
    "cell_status",
    "fit_unavailable",
    "observation_cell",
    "require_fit_supported",
]
