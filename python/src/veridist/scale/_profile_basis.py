"""What a multi-pass fit keeps so that its profile likelihood can be computed from further passes.

The fit has read the data once for its pilot state and several times for its root search; of
all that, a profile-likelihood interval needs a few numbers and the kernel that centres an
observation (:mod:`veridist.scale._multi_pass`). They are kept with the fit's evidence, and no
observation is among them. A pass over the same source at another parameter value adds the exact
sum ``S0 = sum exp(k * w)`` of that value, and with it the likelihood there is a closed form
(:mod:`veridist.scale._profile_math`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from veridist.families.registry import FamilyId


@dataclass(frozen=True, slots=True)
class ProfileBasis:
    """The facts of one multi-pass fit that an inference pass needs.

    ``count`` is the number of observations, ``sum_w`` the exact sum of the centred
    observations rounded once, ``reduced_maximum`` the log-likelihood at the maximum without the
    constant that does not depend on the parameters, ``kernel`` the picklable map from a batch to
    its centred observations, ``rate`` the fitted ``k`` (the Weibull shape, the reciprocal of the
    scale of the standardised Gumbel data), ``log_ratio`` the fitted ``log(S0 / n)`` and
    ``iterate`` and ``iterate_s0`` the parameter the final pass was evaluated at and its exact
    ``S0``. ``anchor``, ``spread`` and ``offset`` map ``(k, a)`` to the parameters of the family:
    the Weibull scale is ``exp(anchor + a / k)``; the Gumbel location is
    ``anchor + spread * (offset - a / k)`` and its scale ``spread / k``.
    """

    family: FamilyId
    count: int
    sum_w: float
    reduced_maximum: float
    kernel: Any
    rate: float
    log_ratio: float
    iterate: float
    iterate_s0: float
    anchor: float
    spread: float
    offset: float


__all__ = ["ProfileBasis"]
