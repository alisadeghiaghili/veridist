"""The interface every fit result shares, whatever the family.

A fit returns either a success or a failure.  The two :class:`typing.Protocol`
types below describe what every success and every failure class exposes, so
code that handles a fit generically (reporting, model comparison, a loop over
families) does not need to know which family produced it.  Both are
``runtime_checkable``: ``isinstance(result, FitSuccess)`` distinguishes a
success from a failure.

The family-specific attributes (``rate``, ``shape``, ``scale``, ``mu_log``,
``sigma_log``, ``mu``, ``sigma``, ``location``) remain on the concrete classes.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from veridist.families.registry import FamilyId

if TYPE_CHECKING:
    from veridist.families.uncertainty import FitUncertainty, UncertaintyUnavailable


@runtime_checkable
class FitSuccess(Protocol):
    """A fit that produced a point estimate.

    ``parameters`` is a read-only mapping from the canonical registry parameter
    names of ``family`` to the fitted values, in registry order.  ``converged``
    is truthful: it is ``True`` only for an interior optimum (a result found at
    the edge of the search range is reported as a failure instead).
    ``event_count`` is the number of exact observations and ``censored_count``
    the number of right-censored ones, so
    ``observation_count == event_count + censored_count``.

    ``uncertainty`` is the covariance, standard errors, confidence intervals and derived
    quantities of the fit (see :mod:`veridist.families.uncertainty`), computed on first access;
    it is an :class:`~veridist.families.uncertainty.UncertaintyUnavailable` value, never an
    exception, when the observed information cannot be inverted.
    """

    @property
    def family(self) -> FamilyId: ...

    @property
    def parameters(self) -> Mapping[str, float]: ...

    @property
    def log_likelihood(self) -> float: ...

    @property
    def observation_count(self) -> int: ...

    @property
    def event_count(self) -> int: ...

    @property
    def censored_count(self) -> int: ...

    @property
    def converged(self) -> bool: ...

    @property
    def uncertainty(self) -> FitUncertainty | UncertaintyUnavailable: ...


@runtime_checkable
class FitFailure(Protocol):
    """A fit that declined to report a point estimate, with the reason as ``code``.

    ``code`` is a member of a family-specific :class:`~enum.StrEnum` whose
    values are stable, locale-neutral strings such as ``"EMPTY_SAMPLE"``,
    ``"DEGENERATE_SAMPLE"`` or ``"BOUNDARY_SOLUTION"``.  The counts describe
    the data that was offered.
    """

    @property
    def family(self) -> FamilyId: ...

    @property
    def code(self) -> StrEnum: ...

    @property
    def observation_count(self) -> int: ...

    @property
    def event_count(self) -> int: ...

    @property
    def censored_count(self) -> int: ...


__all__ = ["FitFailure", "FitSuccess"]
