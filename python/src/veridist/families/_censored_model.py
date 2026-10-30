"""The likelihood model of a sample with left- or interval-censored rows or truncation windows.

:class:`CensoredModel` is the :class:`~veridist.families._models.Model` the uncertainty
machinery (observed information, Wald and profile intervals, derived quantities) uses for such
samples.  It evaluates the log-likelihood, its gradient and its Hessian at any canonical
parameter vector from the per-row terms of :mod:`veridist.families._censored_terms`:

* the location-scale families (normal, lognormal, Weibull minimum, right Gumbel) have analytic
  gradients and Hessians in ``(mu / sigma, 1 / sigma)``, which the chain rule maps to the
  canonical parameters;
* the exponential has analytic derivatives in the rate;
* the gamma has analytic derivatives in the log scale and a five-point central difference (in
  ``log shape``) for the shape, as in the earlier releases.

Sums are exact (``math.fsum``).  The value of a lognormal or Weibull model is the log-likelihood
of ``log x`` (no Jacobian), a constant that profile intervals do not see.
"""

from __future__ import annotations

from importlib import import_module
from math import exp, log
from typing import Any

from veridist.families._censored_rows import CensoredRows
from veridist.families._censored_terms import (
    GAMMA_INFORMATION_STEP,
    GAMMA_OPTIMIZER_STEP,
    _NotEvaluable,
    exact_sums,
    exponential_block,
    gamma_terms,
    terms_function,
)
from veridist.families._models import Evaluation, Model, Reparameterization
from veridist.families.registry import FamilyId
from veridist.statistics._logerrors import LogProbabilityError

np: Any = import_module("numpy")

_NAMES = {
    FamilyId.EXPONENTIAL: ("rate",),
    FamilyId.WEIBULL_MIN: ("shape", "scale"),
    FamilyId.LOGNORMAL: ("mu_log", "sigma_log"),
    FamilyId.GAMMA: ("shape", "scale"),
    FamilyId.NORMAL: ("mu", "sigma"),
    FamilyId.GUMBEL_RIGHT: ("location", "scale"),
}
_POSITIVE = {
    FamilyId.EXPONENTIAL: (True,),
    FamilyId.WEIBULL_MIN: (True, True),
    FamilyId.LOGNORMAL: (False, True),
    FamilyId.GAMMA: (True, True),
    FamilyId.NORMAL: (False, True),
    FamilyId.GUMBEL_RIGHT: (False, True),
}


def _coordinates(
    family: FamilyId, theta: tuple[float, ...]
) -> tuple[tuple[float, float], Any, tuple[Any, Any]]:
    """The optimizer coordinates ``(m, u)`` of the canonical ``theta`` of a location-scale
    family, the Jacobian ``d(m, u) / d theta`` and the Hessians of ``m`` and of ``u``."""

    first, second = theta
    zero = np.zeros((2, 2))
    if family is FamilyId.WEIBULL_MIN:
        # u = shape, m = shape * log(scale)
        jacobian = np.array([[log(second), first / second], [1.0, 0.0]])
        curvature_m = np.array([[0.0, 1.0 / second], [1.0 / second, -first / (second * second)]])
        return (first * log(second), first), jacobian, (curvature_m, zero)
    if family is FamilyId.GUMBEL_RIGHT:
        # y = -x: u = 1 / scale, m = -location / scale
        s2 = second * second
        jacobian = np.array([[-1.0 / second, first / s2], [0.0, -1.0 / s2]])
        curvature_m = np.array([[0.0, 1.0 / s2], [1.0 / s2, -2.0 * first / (s2 * second)]])
        curvature_u = np.array([[0.0, 0.0], [0.0, 2.0 / (s2 * second)]])
        return (-first / second, 1.0 / second), jacobian, (curvature_m, curvature_u)
    # normal, lognormal: u = 1 / sigma, m = mu / sigma
    s2 = second * second
    jacobian = np.array([[1.0 / second, -first / s2], [0.0, -1.0 / s2]])
    curvature_m = np.array([[0.0, -1.0 / s2], [-1.0 / s2, 2.0 * first / (s2 * second)]])
    curvature_u = np.array([[0.0, 0.0], [0.0, 2.0 / (s2 * second)]])
    return (first / second, 1.0 / second), jacobian, (curvature_m, curvature_u)


class CensoredModel(Model):
    """Log-likelihood, gradient and Hessian of a censored and truncated sample, by family."""

    def __init__(self, family: FamilyId, rows: CensoredRows) -> None:
        self.family = family
        self.names = _NAMES[family]
        self.positive = _POSITIVE[family]
        self.positive_support = family in (
            FamilyId.EXPONENTIAL,
            FamilyId.WEIBULL_MIN,
            FamilyId.LOGNORMAL,
            FamilyId.GAMMA,
        )
        self.rows = rows

    def _sums(
        self, coordinates: tuple[float, ...], *, step: float, value_only: bool
    ) -> list[float]:
        try:
            with np.errstate(all="ignore"):
                if self.family is FamilyId.GAMMA:
                    terms = gamma_terms(self.rows, coordinates, step=step, value_only=value_only)
                else:
                    terms = terms_function(self.family)(self.rows, coordinates)
                extras = (
                    exponential_block(self.rows, coordinates)
                    if self.family is FamilyId.EXPONENTIAL
                    else None
                )
        except (_NotEvaluable, LogProbabilityError) as error:
            raise ArithmeticError("the likelihood cannot be evaluated here") from error
        return exact_sums(terms, extras)

    def evaluate(self, theta: tuple[float, ...], *, precise: bool = True) -> Evaluation:
        family = self.family
        if family is FamilyId.EXPONENTIAL:
            sums = self._sums(theta, step=0.0, value_only=False)
            return Evaluation(sums[0], (sums[1],), ((sums[2],),))
        if family is FamilyId.GAMMA:
            shape, scale = theta
            step = GAMMA_INFORMATION_STEP if precise else GAMMA_OPTIMIZER_STEP
            sums = self._sums((log(shape), log(scale)), step=step, value_only=False)
            g_a, g_b = sums[1], sums[2]
            mixed = sums[4] / (shape * scale)
            return Evaluation(
                sums[0],
                (g_a / shape, g_b / scale),
                (
                    ((sums[3] - g_a) / (shape * shape), mixed),
                    (mixed, (sums[5] - g_b) / (scale * scale)),
                ),
            )
        (m, u), jacobian, curvature = _coordinates(family, theta)
        sums = self._sums((m, u), step=0.0, value_only=False)
        gradient = np.array([sums[1], sums[2]])
        hessian = np.array([[sums[3], sums[4]], [sums[4], sums[5]]])
        with np.errstate(all="ignore"):
            canonical_gradient = jacobian.T @ gradient
            canonical_hessian = (
                jacobian.T @ hessian @ jacobian
                + gradient[0] * curvature[0]
                + gradient[1] * curvature[1]
            )
            canonical_hessian = 0.5 * (canonical_hessian + canonical_hessian.T)
        return Evaluation(
            sums[0],
            (float(canonical_gradient[0]), float(canonical_gradient[1])),
            (
                (float(canonical_hessian[0, 0]), float(canonical_hessian[0, 1])),
                (float(canonical_hessian[1, 0]), float(canonical_hessian[1, 1])),
            ),
        )

    def log_likelihood(self, theta: tuple[float, ...]) -> float:
        family = self.family
        if family is FamilyId.GAMMA:
            return self._sums((log(theta[0]), log(theta[1])), step=0.0, value_only=True)[0]
        if family is FamilyId.EXPONENTIAL:
            return self._sums(theta, step=0.0, value_only=False)[0]
        (m, u), _, _ = _coordinates(family, theta)
        return self._sums((m, u), step=0.0, value_only=False)[0]

    def mean(self, theta: tuple[float, ...]) -> float:
        from math import gamma

        family = self.family
        if family is FamilyId.EXPONENTIAL:
            return 1.0 / theta[0]
        if family is FamilyId.WEIBULL_MIN:
            return theta[1] * gamma(1.0 + 1.0 / theta[0])
        if family is FamilyId.LOGNORMAL:
            return exp(theta[0] + 0.5 * theta[1] * theta[1])
        if family is FamilyId.GUMBEL_RIGHT:
            return theta[0] + 0.5772156649015329 * theta[1]
        if family is FamilyId.GAMMA:
            return theta[0] * theta[1]
        return theta[0]

    def reparameterization(self, quantity: str, argument: float) -> Reparameterization | None:
        from math import gamma, log1p

        if self.family is FamilyId.EXPONENTIAL:
            if quantity == "mean":
                return Reparameterization(None, lambda value, _free: (1.0 / value,))
            if quantity == "quantile":
                level = -log1p(-argument)
                return Reparameterization(None, lambda value, _free: (level / value,))
            return Reparameterization(None, lambda value, _free: (-log(value) / argument,))
        if self.family is FamilyId.WEIBULL_MIN:
            if quantity == "mean":
                return Reparameterization(
                    0, lambda value, shape: (shape, value / gamma(1.0 + 1.0 / shape))
                )
            if quantity == "quantile":
                level = -log1p(-argument)
                return Reparameterization(
                    0, lambda value, shape: (shape, value / level ** (1.0 / shape))
                )
            return Reparameterization(
                0, lambda value, shape: (shape, argument / (-log(value)) ** (1.0 / shape))
            )
        return None


__all__ = ["CensoredModel"]
