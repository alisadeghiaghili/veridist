"""What a fit records about its data and about the maximum it reports.

Three small, frozen, locale-neutral values travel with every in-memory fit result:

* :class:`ObservationSummary`: how many rows of each censoring kind and each truncation type
  the fit was given;
* :class:`OptimumKind`: what is claimed about the reported point (a closed form, a global
  maximum guaranteed by concavity, or a certified stationary point without such a guarantee);
* :class:`FitDiagnostics`: the deterministic facts of an iterative fit (passes, starts, the
  stationary points seen).

None of them holds an observation value.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from veridist.domain._combinations import CensoringKind, TruncationKind, observation_cell


class OptimumKind(StrEnum):
    """What is claimed about the point a fit reports."""

    UNASSESSED = "unassessed"
    """The bounded deterministic search of the earlier releases, which claims no more than its
    interior optimum (data the earlier releases accepted)."""

    CLOSED_FORM = "closed_form"
    """The maximizer has a closed form (the global maximum of the likelihood)."""

    GLOBAL_MAXIMUM = "global_maximum"
    """A converged Newton result of a likelihood that is concave in the coordinates of the
    optimizer (location-scale families without truncation; the exponential with at most left
    truncation), so the stationary point found is the global maximum."""

    STATIONARY_POINT = "stationary_point"
    """A certified stationary point: the gradient is below its tolerance and the Hessian is
    negative definite, but the likelihood is not known to be concave, so no global maximum
    is claimed."""


@dataclass(frozen=True, slots=True)
class ObservationSummary:
    """Counts of the rows a fit was given, by censoring kind and by truncation type.

    The four censoring counts add up to ``total`` and so do the four truncation counts
    (``untruncated`` and the three truncated ones); a row is counted once in each family of
    counts.  ``total`` counts replicated rows of integer frequency weights as separate rows.
    """

    exact: int = 0
    right_censored: int = 0
    left_censored: int = 0
    interval_censored: int = 0
    untruncated: int = 0
    left_truncated: int = 0
    right_truncated: int = 0
    two_sided_truncated: int = 0

    def __post_init__(self) -> None:
        for name in (
            "exact",
            "right_censored",
            "left_censored",
            "interval_censored",
            "untruncated",
            "left_truncated",
            "right_truncated",
            "two_sided_truncated",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if (
            self.exact + self.right_censored + self.left_censored + self.interval_censored
            != self.untruncated
            + self.left_truncated
            + self.right_truncated
            + self.two_sided_truncated
        ):
            raise ValueError("the censoring counts and the truncation counts must add up alike")

    @property
    def total(self) -> int:
        """The number of rows."""

        return self.exact + self.right_censored + self.left_censored + self.interval_censored

    @property
    def censored(self) -> int:
        """The rows that are not exact."""

        return self.total - self.exact

    @property
    def truncated(self) -> int:
        """The rows that carry a truncation window."""

        return self.left_truncated + self.right_truncated + self.two_sided_truncated

    @property
    def is_new_data(self) -> bool:
        """Whether a row is left- or interval-censored or carries a truncation window.

        Fits of such data use the Newton method; exact and right-censored rows without a
        window are fitted as in the earlier releases.
        """

        return bool(self.left_censored or self.interval_censored or self.truncated)


_CENSORING_FIELD = {
    CensoringKind.EXACT: 0,
    CensoringKind.RIGHT: 1,
    CensoringKind.LEFT: 2,
    CensoringKind.INTERVAL: 3,
}
_TRUNCATION_FIELD = {
    TruncationKind.NONE: 0,
    TruncationKind.LEFT: 1,
    TruncationKind.RIGHT: 2,
    TruncationKind.TWO_SIDED: 3,
}


def summarize_observations(observations: Iterable[object]) -> ObservationSummary:
    """Count ``observations`` by censoring kind and truncation type.

    ``TypeError`` for an object that is not one of the eight observation types.
    """

    kinds = [0, 0, 0, 0]
    windows = [0, 0, 0, 0]
    for observation in observations:
        kind, truncation = observation_cell(observation)
        kinds[_CENSORING_FIELD[kind]] += 1
        windows[_TRUNCATION_FIELD[truncation]] += 1
    return ObservationSummary(*kinds, *windows)


@dataclass(frozen=True, slots=True)
class FitDiagnostics:
    """The deterministic facts of an iterative fit; ``None`` fields did not apply.

    ``passes`` counts the evaluations of the likelihood over all rows that decide an iteration
    (the pilot, the start grid and one per stencil of trial steps) and ``iterations`` the
    accepted Newton steps.  ``shifted_directions`` counts the directions whose Hessian had to
    be shifted to be negative definite.  ``starts`` is the number of starts run (two where the
    likelihood is not known to be concave: the best of the start grid and one further start,
    the higher log-likelihood kept) and ``distinct_stationary_points`` how many different
    points they ended at.  ``stationary_points_seen`` is the number of stationary points the
    profile of a left-truncated Weibull fit showed on its grid.
    """

    passes: int
    iterations: int
    shifted_directions: int = 0
    stencil_failures: int = 0
    starts: int = 1
    distinct_stationary_points: int = 1
    stationary_points_seen: int | None = None
    method: str = "newton"

    def __post_init__(self) -> None:
        for name in (
            "passes",
            "iterations",
            "shifted_directions",
            "stencil_failures",
            "starts",
            "distinct_stationary_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


__all__ = [
    "FitDiagnostics",
    "ObservationSummary",
    "OptimumKind",
    "summarize_observations",
]
