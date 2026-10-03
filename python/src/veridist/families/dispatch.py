"""One entry point, :func:`fit`, for every family's maximum-likelihood fit."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any, Final

from veridist.families.exponential import fit_exponential
from veridist.families.gamma import fit_gamma
from veridist.families.gumbel import fit_gumbel_right
from veridist.families.lognormal import fit_lognormal
from veridist.families.normal import fit_normal
from veridist.families.registry import FAMILY_REGISTRY, FamilyId, FamilySpec, Operation
from veridist.families.results import FitFailure, FitSuccess
from veridist.families.weibull import fit_weibull

_Fitter = Callable[..., FitSuccess | FitFailure]

_FITTERS: Final[Mapping[FamilyId, _Fitter]] = MappingProxyType(
    {
        FamilyId.NORMAL: fit_normal,
        FamilyId.GAMMA: fit_gamma,
        FamilyId.WEIBULL_MIN: fit_weibull,
        FamilyId.LOGNORMAL: fit_lognormal,
        FamilyId.GUMBEL_RIGHT: fit_gumbel_right,
        FamilyId.EXPONENTIAL: fit_exponential,
    }
)

_CENSORING_OPTIONS: Final = frozenset(
    {"frequency_weights", "analytic_weights", "censoring", "truncation"}
)
#: The keyword options each family's fit accepts, besides the observations.
_OPTIONS: Final[Mapping[FamilyId, frozenset[str]]] = MappingProxyType(
    {
        FamilyId.NORMAL: _CENSORING_OPTIONS,
        FamilyId.GAMMA: _CENSORING_OPTIONS,
        FamilyId.WEIBULL_MIN: _CENSORING_OPTIONS | {"fixed_shape"},
        FamilyId.LOGNORMAL: _CENSORING_OPTIONS,
        FamilyId.GUMBEL_RIGHT: _CENSORING_OPTIONS,
        FamilyId.EXPONENTIAL: frozenset(),
    }
)


def _verify_fitters(
    registry: Mapping[FamilyId, FamilySpec],
    fitters: Mapping[FamilyId, _Fitter],
    options: Mapping[FamilyId, frozenset[str]],
) -> None:
    """Fail at import if the registry and the fit tables disagree."""

    advertised = {family for family, spec in registry.items() if spec.supports(Operation.FIT)}
    if set(fitters) != advertised or set(options) != advertised:
        raise RuntimeError("fit dispatch must exactly match the family registry")


_verify_fitters(FAMILY_REGISTRY.families, _FITTERS, _OPTIONS)


def fit(
    family: FamilyId | str, observations: Iterable[Any], /, **options: Any
) -> FitSuccess | FitFailure:
    """Fit ``family`` by maximum likelihood to ``observations``.

    ``family`` is a :class:`~veridist.families.registry.FamilyId` or its string
    value; anything else raises ``TypeError`` and an unknown name raises
    ``ValueError``.  ``observations`` must be the type pair valid for the family:

    * ``EXPONENTIAL``, ``WEIBULL_MIN``, ``LOGNORMAL`` and ``GAMMA`` take
      :class:`~veridist.domain.lifetimes.ExactLifetime` and
      :class:`~veridist.domain.lifetimes.RightCensoredLifetime`;
    * ``NORMAL`` and ``GUMBEL_RIGHT`` take
      :class:`~veridist.domain.values.ExactValue` and
      :class:`~veridist.domain.values.RightCensoredValue`.

    The other pair raises ``TypeError``.  ``options`` are forwarded to the
    family's own fit function (``fit_weibull`` additionally accepts
    ``fixed_shape``; the exponential fit takes no options); an option the family
    does not accept raises ``TypeError`` naming it.

    The result is a :class:`~veridist.families.results.FitSuccess` or a
    :class:`~veridist.families.results.FitFailure`.
    """

    spec = FAMILY_REGISTRY.lookup(family)
    unknown = sorted(set(options) - _OPTIONS[spec.id])
    if unknown:
        names = ", ".join(repr(name) for name in unknown)
        raise TypeError(f"fit({spec.id.value!r}) got unexpected option(s): {names}")
    return _FITTERS[spec.id](observations, **options)


__all__ = ["fit"]
