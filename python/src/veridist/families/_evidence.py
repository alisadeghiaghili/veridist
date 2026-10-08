"""The sample a fit keeps so that its uncertainty can be computed on demand.

A fit success is an immutable value, but its uncertainty (observed information,
profile likelihood) needs the observed values again.  :class:`FitEvidence` holds
them, as plain tuples of floats, together with a small result cache.  The cache is
the only mutable state: the success classes are frozen and compare, hash and
print without it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from veridist.families.registry import FamilyId

if TYPE_CHECKING:
    from veridist.families.uncertainty import FitUncertainty, UncertaintyUnavailable


class FitEvidence:
    """Observed values behind one fit, plus a cache of the uncertainty derived from them.

    ``exact`` holds the exactly observed values and ``censored`` the right-censoring
    points, as floats (event times for the lifetime families).  ``fixed`` names the
    parameters the caller fixed instead of estimating; the observed information of
    a fit with a fixed parameter does not describe the fixed parameter, so no
    uncertainty is reported for it.

    The object keeps the values, not the observation objects, and is created by the
    fit functions; it is not part of the public interface.
    """

    __slots__ = ("_cache", "censored", "exact", "fixed")

    def __init__(
        self,
        exact: tuple[float, ...] = (),
        censored: tuple[float, ...] = (),
        fixed: tuple[str, ...] = (),
    ) -> None:
        self.exact = exact
        self.censored = censored
        self.fixed = fixed
        self._cache: dict[tuple[Any, ...], Any] = {}

    def uncertainty(
        self, family: FamilyId, parameters: Mapping[str, float]
    ) -> FitUncertainty | UncertaintyUnavailable:
        """Return the uncertainty at ``parameters``, computed once per parameter vector."""

        key = (family, *parameters.values())
        cached = self._cache.get(key)
        if cached is None:
            from veridist.families.uncertainty import compute_uncertainty

            cached = compute_uncertainty(family, parameters, self)
            self._cache[key] = cached
        return cached


__all__ = ["FitEvidence"]
