"""The rows of a sample of exact, censored and truncated observations, and the pilot.

A fit of left- or interval-censored data or of truncated data works on arrays: for every row
the exact value ``yv`` (``nan`` for a censored row), the censoring set ``(lo, hi]`` (infinite
ends for an open side) and the truncation window ``(wl, wh]`` (infinite where a side is not
truncated).  The values are held in the space of the optimizer, ``y``:

* ``y = x`` for the normal, the gamma and the exponential;
* ``y = log x`` for the lognormal and the Weibull minimum (a time of zero is ``-inf``);
* ``y = -x`` for the right Gumbel, whose sets are reflected (``(lo, hi]`` becomes
  ``[-hi, -lo)``, immaterial for a continuous distribution), so that it is the smallest
  extreme value distribution in ``y``.

The pilot looks at the pseudo-sample (exact rows as they are, right- and left-censored rows at
their point, intervals at their midpoint) and returns the centre and scale that standardise
the rows, and the start of the optimizer.  Every moment of the pilot is an exact sum
(``math.fsum``), never ``numpy.mean`` or ``numpy.std``, so the standardisation does not depend
on the order of the rows.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from math import fsum, inf, isfinite, log, pi, sqrt
from typing import Any, Final

from veridist.domain._combinations import CENSORING_KINDS, CensoringKind
from veridist.families.registry import FamilyId

np: Any = import_module("numpy")

_EULER_GAMMA: Final = 0.5772156649015329
#: The standard deviation of the standardised smallest-extreme-value distribution's scale:
#: ``sigma = sqrt(6) / pi`` times the standard deviation gives the moment estimate.
_SEV_SCALE_PER_SD: Final = sqrt(6.0) / pi
#: The minimum of the shape start of the gamma (the mean log-deficit is at least this).
_MINIMUM_LOG_DEFICIT: Final = 1e-6

LOCATION_SCALE_FAMILIES: Final = frozenset(
    {FamilyId.NORMAL, FamilyId.LOGNORMAL, FamilyId.WEIBULL_MIN, FamilyId.GUMBEL_RIGHT}
)
LOG_SPACE_FAMILIES: Final = frozenset({FamilyId.LOGNORMAL, FamilyId.WEIBULL_MIN})


@dataclass(frozen=True, slots=True)
class CensoredRows:
    """Rows in ``y`` space; see the module docstring.

    ``plain_count``, ``plain_events`` and ``plain_total`` describe, for the exponential only,
    a block of exact and right-censored rows without a window that was summarised before the
    other rows were seen (a stream): their likelihood depends on the data through the number
    of events and the total time alone.
    """

    family: FamilyId
    exact: Any
    yv: Any
    lo: Any
    hi: Any
    wl: Any
    wh: Any
    plain_count: int = 0
    plain_events: int = 0
    plain_total: float = 0.0

    @property
    def count(self) -> int:
        """The number of rows, the summarised block included."""

        return int(self.exact.shape[0]) + self.plain_count

    @property
    def explicit_count(self) -> int:
        """The number of rows held in the arrays."""

        return int(self.exact.shape[0])

    def scaled(self, centre: float, scale: float) -> CensoredRows | None:
        """The rows standardised: ``(y - centre) / scale``, or ``y / scale`` with ``centre`` 0.

        ``None`` if the standardisation is not faithful: an exact value that is no longer a
        finite number, an end that became ``nan``, or an end whose infinity changed (a finite
        end that overflowed, which would turn a bound into "no bound").
        """

        with np.errstate(all="ignore"):
            if self.family in (FamilyId.GAMMA, FamilyId.EXPONENTIAL):

                def move(values: Any) -> Any:
                    return values / scale

            else:

                def move(values: Any) -> Any:
                    return (values - centre) / scale

            moved = [move(end) for end in (self.lo, self.hi, self.wl, self.wh)]
            value = move(self.yv)
            total = self.plain_total / scale if self.family is FamilyId.EXPONENTIAL else 0.0
            faithful = (
                bool(np.isfinite(value[self.exact]).all())
                and isfinite(total)
                and all(
                    not bool(np.isnan(after).any())
                    and bool((np.isinf(after) == np.isinf(before)).all())
                    for after, before in zip(
                        moved, (self.lo, self.hi, self.wl, self.wh), strict=True
                    )
                )
            )
        if not faithful:
            return None
        return CensoredRows(
            self.family,
            self.exact,
            value,
            moved[0],
            moved[1],
            moved[2],
            moved[3],
            self.plain_count,
            self.plain_events,
            total,
        )


def _y(family: FamilyId, values: Sequence[float]) -> Any:
    """``log`` for the families fitted on ``log x`` (``-inf`` stays ``-inf``, zero gives it)."""

    array = np.asarray(values, dtype=np.float64)
    if family in LOG_SPACE_FAMILIES:
        bottom = np.isneginf(array)
        with np.errstate(divide="ignore"):
            return np.where(bottom, -inf, np.log(np.where(bottom, 1.0, array)))
    return array


def build_rows(
    family: FamilyId,
    observations: Sequence[Any],
    *,
    plain_count: int = 0,
    plain_events: int = 0,
    plain_total: float = 0.0,
) -> CensoredRows:
    """The rows of ``observations`` (any of the eight observation types) in ``y`` space."""

    count = len(observations)
    exact = np.zeros(count, dtype=bool)
    value = np.full(count, np.nan)
    low = np.full(count, -inf)
    high = np.full(count, inf)
    window_low = np.full(count, -inf)
    window_high = np.full(count, inf)
    for index, observation in enumerate(observations):
        kind = CENSORING_KINDS[type(observation)]
        if kind is CensoringKind.EXACT:
            exact[index] = True
            value[index] = _point(observation)
        elif kind is CensoringKind.RIGHT:
            low[index] = _point(observation)
        elif kind is CensoringKind.LEFT:
            high[index] = _point(observation)
        else:
            low[index] = float(observation.lower)
            high[index] = float(observation.upper)
        if observation.truncation_lower is not None:
            window_low[index] = float(observation.truncation_lower)
        if observation.truncation_upper is not None:
            window_high[index] = float(observation.truncation_upper)
    if _is_lifetime(family):
        # The bottom of the support is an open lower end, whatever way it is written.
        window_low = np.where(window_low <= 0.0, -inf, window_low)
        low = np.where(low <= 0.0, -inf, low)
    if family is FamilyId.GUMBEL_RIGHT:
        value = -value
        low, high = -high, -low
        window_low, window_high = -window_high, -window_low
        return CensoredRows(
            family,
            exact,
            value,
            low,
            high,
            window_low,
            window_high,
            plain_count,
            plain_events,
            plain_total,
        )
    return CensoredRows(
        family,
        exact,
        _y(family, value),
        _y(family, low),
        _y(family, high),
        _y(family, window_low),
        _y(family, window_high),
        plain_count,
        plain_events,
        plain_total,
    )


def _point(observation: Any) -> float:
    """The time of a lifetime observation or the value of a real-value one."""

    return float(observation.time if hasattr(observation, "time") else observation.value)


def _is_lifetime(family: FamilyId) -> bool:
    return family not in (FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT)


def pseudo_values(rows: CensoredRows) -> Any:
    """The pseudo-sample in ``y`` space: exact rows as they are, censoring points at the point,
    intervals at their midpoint."""

    out = np.where(rows.exact, rows.yv, np.nan)
    censored = ~rows.exact
    low, high = rows.lo[censored], rows.hi[censored]
    both = np.isfinite(low) & np.isfinite(high)
    with np.errstate(all="ignore"):
        middle = np.where(both, 0.5 * (low + high), np.where(np.isfinite(low), low, high))
    out[censored] = middle
    return out


@dataclass(frozen=True, slots=True)
class Pilot:
    """The standardisation, the start and the facts the failure codes need.

    ``failure`` is ``None`` or the name of the failure code the pilot decided
    (``DEGENERATE_SAMPLE``, ``OPTIMIZER_EXHAUSTED``).  ``pseudo_total`` is the exact sum of the
    pseudo-sample, the summarised block included.
    """

    centre: float
    scale: float
    start: tuple[float, ...]
    pseudo_total: float
    failure: str | None = None


def pilot(rows: CensoredRows) -> Pilot:
    """Standardise the pseudo-sample of ``rows`` and choose the start, by exact sums.

    A sum that overflows binary64 is the failure ``OPTIMIZER_EXHAUSTED``, as in the earlier
    releases (their ``OverflowError`` is that failure).
    """

    try:
        return _pilot(rows)
    except OverflowError:
        return Pilot(0.0, 0.0, (), float("nan"), "OPTIMIZER_EXHAUSTED")


def _pilot(rows: CensoredRows) -> Pilot:
    family = rows.family
    values = pseudo_values(rows)
    values = values[np.isfinite(values)]
    count = int(values.shape[0]) + rows.plain_count
    total = fsum([*values.tolist(), rows.plain_total])
    if family is FamilyId.EXPONENTIAL:
        scale = total / count
        if not (isfinite(scale) and scale > 0.0):
            return Pilot(0.0, scale, (), total, "OPTIMIZER_EXHAUSTED")
        return Pilot(0.0, scale, (1.0,), total)
    if float(values.min()) == float(values.max()):
        return Pilot(0.0, 0.0, (), total, "DEGENERATE_SAMPLE")
    if family is FamilyId.GAMMA:
        scale = total / count
        deficit = -fsum(np.log(values / scale).tolist()) / count
        deficit = max(deficit, _MINIMUM_LOG_DEFICIT)
        shape = (3.0 - deficit + sqrt((deficit - 3.0) ** 2 + 24.0 * deficit)) / (12.0 * deficit)
        return Pilot(0.0, scale, (log(shape), -log(shape)), total)
    centre = total / count
    scale = sqrt(fsum(((values - centre) ** 2).tolist()) / count)
    if not scale > 0.0:
        return Pilot(centre, scale, (), total, "DEGENERATE_SAMPLE")
    if family in (FamilyId.WEIBULL_MIN, FamilyId.GUMBEL_RIGHT):
        sigma = _SEV_SCALE_PER_SD
        return Pilot(centre, scale, (_EULER_GAMMA, 1.0 / sigma), total)
    return Pilot(centre, scale, (0.0, 1.0), total)


__all__ = [
    "LOCATION_SCALE_FAMILIES",
    "LOG_SPACE_FAMILIES",
    "CensoredRows",
    "Pilot",
    "build_rows",
    "pilot",
    "pseudo_values",
]
