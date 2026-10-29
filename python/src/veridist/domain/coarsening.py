"""The assumption behind every censored or truncated likelihood.

Censoring and truncation are per-observation coarsening of the exact value, and each
likelihood contribution of a censored or truncated observation assumes that the coarsening is
non-informative: the censoring and truncation mechanism is independent of the lifetime (or
value) given what the model conditions on, that is, coarsening at random.  The data cannot
verify this and the library does not claim that it holds.  Results that rest on such
observations record the assumption as a closed value of this enumeration, so that the
assumption travels with the number.
"""

from __future__ import annotations

from enum import StrEnum


class CoarseningAssumption(StrEnum):
    """The closed set of coarsening assumptions a result can record."""

    INDEPENDENT_CENSORING_AND_TRUNCATION = "independent_censoring_and_truncation"
    """Non-informative, independent censoring and truncation (coarsening at random)."""


__all__ = ["CoarseningAssumption"]
