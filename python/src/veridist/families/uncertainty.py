"""Parameter and derived-quantity uncertainty for every fit, from one shared implementation.

A fit success exposes ``result.uncertainty``: a :class:`FitUncertainty` or, when the
observed information cannot be inverted, an :class:`UncertaintyUnavailable` that says
why.  The computation is lazy (nothing is paid until ``uncertainty`` is read) and cached
on the result.

**Observed information.**  The covariance is the inverse of the negative Hessian of the
censored log-likelihood at the fitted parameters, in the canonical parameterization
(``result.parameters``).  The Hessians are analytic (``_models``); the gamma family
differentiates ``log Q`` numerically in the shape only.

**Intervals.**  ``method="wald"`` is a large-sample interval; for a positive parameter it
is built on the log scale and mapped back, so it never contains a non-positive value.
``method="profile"`` inverts the likelihood-ratio statistic: the endpoints solve
``2 (l_max - l_profile(theta)) = chi2_1(level)`` on each side.  A side that does not cross
within the hard search range (``_HARD_LOG_OFFSET`` on the log scale, ``_HARD_IDENTITY_FACTOR``
Wald half-widths otherwise) is reported as unbounded: ``0`` (or ``-inf`` for a parameter on
the whole real line) below, ``inf`` above, and the matching ``*_unbounded`` flag is set.
``method="exact"`` is the chi-square interval for the exponential rate on uncensored data.

**Assumptions.**  Independent right censoring, maximum-likelihood regularity (an interior
maximum with a positive-definite information matrix) and, for the Wald and delta-method
intervals, a sample large enough for the likelihood to be roughly quadratic.  Profile
intervals do not need the quadratic approximation but can be unbounded when the sample
carries too little information.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from functools import lru_cache
from math import exp, fsum, inf, isfinite, log, log1p, sqrt
from types import MappingProxyType
from typing import Final, cast

from veridist.domain._numeric import is_real
from veridist.families._evidence import FitEvidence
from veridist.families._models import (
    Evaluation,
    ExponentialModel,
    Matrix,
    Model,
    Reparameterization,
    build_model,
)
from veridist.families._reliability import exp_or_inf, expand_bracket
from veridist.families.registry import FamilyId
from veridist.statistics.distributions import ppf

_METHODS: Final = ("wald", "profile", "exact")

#: Relative size of ``1 - correlation**2`` below which the information is called singular.
_SINGULAR_TOLERANCE: Final = 1e-12
#: Profile search: how far (log scale) from the estimate a crossing is looked for, and the same
#: in units of the starting step for parameters on the whole real line.
_HARD_LOG_OFFSET: Final = 30.0
_HARD_IDENTITY_FACTOR: Final = 1e4
_MAX_EXPANSIONS: Final = 64
_MAX_ROOT_ITERATIONS: Final = 100
#: After this many root iterations only bisection is used.
_BISECTION_AFTER: Final = 30
_ROOT_TOLERANCE: Final = 1e-10
_NEWTON_STEPS: Final = 40
_BACKTRACKS: Final = 30
#: Newton iteration stops when the log-likelihood it could still gain, ``g**2 / (2 |h|)``, is
#: below this (absolute, in log-likelihood units).
_NEWTON_DECREMENT: Final = 1e-12
#: Rounding allowance when accepting a Newton step that does not increase the value.
_ACCEPT_SLACK: Final = 1e-13
#: A Newton step on the nuisance parameter is capped at this many standard errors.
_NEWTON_CAP_ERRORS: Final = 10.0
#: Finite-difference step for delta-method gradients, in standard errors (and at most
#: ``_GRADIENT_RELATIVE_LIMIT`` of a positive parameter).
_GRADIENT_STEP: Final = 0.05
_GRADIENT_RELATIVE_LIMIT: Final = 0.1


class UncertaintyUnavailableReason(StrEnum):
    """Stable, locale-neutral reasons that no uncertainty is reported."""

    NO_DATA = "NO_DATA"
    FIXED_PARAMETER = "FIXED_PARAMETER"
    SINGULAR_INFORMATION = "SINGULAR_INFORMATION"
    NOT_POSITIVE_DEFINITE = "NOT_POSITIVE_DEFINITE"
    NOT_COMPUTABLE = "NOT_COMPUTABLE"


@dataclass(frozen=True, slots=True)
class UncertaintyUnavailable:
    """The typed absence of uncertainty: ``reason`` is the stable code, ``detail`` a sentence."""

    reason: UncertaintyUnavailableReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class ParameterInterval:
    """A confidence interval for one parameter.

    ``lower_unbounded`` / ``upper_unbounded`` are set when a profile-likelihood side did not
    cross within the search range; the corresponding bound is then ``0`` (or ``-inf``) or
    ``inf``.
    """

    lower: float
    upper: float
    level: float
    method: str
    lower_unbounded: bool = False
    upper_unbounded: bool = False


@dataclass(frozen=True, slots=True)
class DerivedEstimate:
    """A derived quantity (mean, quantile or survival) with a confidence interval."""

    estimate: float
    lower: float
    upper: float
    method: str
    level: float
    lower_unbounded: bool = False
    upper_unbounded: bool = False


# ---------------------------------------------------------------------- small numerics


def _real(value: object, name: str) -> float:
    """Return ``value`` as a float if it is a real scalar (Python or numpy, not a bool)."""

    if not is_real(value):
        raise TypeError(f"{name} must be a real number")
    return float(cast(float, value))


def _validated_level(level: object) -> float:
    value = _real(level, "level")
    if not 0.0 < value < 1.0:
        raise ValueError("level must be strictly between 0 and 1")
    return value


def _validated_method(method: object) -> str:
    if not isinstance(method, str):
        raise TypeError("method must be a string")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {', '.join(repr(name) for name in _METHODS)}")
    return method


def _normal_quantile(level: float) -> float:
    """Return the two-sided standard-normal critical value ``z`` for ``level``."""

    return -float(ppf("normal", 0.5 * (1.0 - level), mu=0.0, sigma=1.0))


@lru_cache(maxsize=64)
def _chi_square_1(level: float) -> float:
    """Return the ``level`` quantile of chi-square with one degree of freedom.

    Chi-square with ``k`` degrees of freedom is gamma with shape ``k / 2`` and scale ``2``.
    """

    return float(ppf("gamma", level, shape=0.5, scale=2.0))


def _expit(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + exp(-value))
    shifted = exp(value)
    return shifted / (1.0 + shifted)


def _logit(probability: float) -> float:
    return log(probability) - log1p(-probability)


def _invert_information(information: Matrix) -> Matrix | UncertaintyUnavailable:
    """Invert a symmetric information matrix of order one or two, or say why that is impossible."""

    if not all(isfinite(entry) for row in information for entry in row):
        return UncertaintyUnavailable(
            UncertaintyUnavailableReason.NOT_COMPUTABLE,
            "the observed information is not finite",
        )
    first = information[0][0]
    if len(information) == 1:
        if first > 0.0:
            return ((1.0 / first,),)
        return UncertaintyUnavailable(
            UncertaintyUnavailableReason.NOT_POSITIVE_DEFINITE,
            "the observed information is not positive",
        )
    second = information[1][1]
    mixed = 0.5 * (information[0][1] + information[1][0])
    if not (first > 0.0 and second > 0.0):
        return UncertaintyUnavailable(
            UncertaintyUnavailableReason.NOT_POSITIVE_DEFINITE,
            "the observed information has a non-positive diagonal entry",
        )
    # Work with the correlation form so the test does not depend on the parameters' scales.
    correlation = mixed / sqrt(first * second)
    margin = 1.0 - correlation * correlation
    if margin <= _SINGULAR_TOLERANCE:
        reason = (
            UncertaintyUnavailableReason.SINGULAR_INFORMATION
            if margin >= -_SINGULAR_TOLERANCE
            else UncertaintyUnavailableReason.NOT_POSITIVE_DEFINITE
        )
        return UncertaintyUnavailable(reason, "the observed information is not invertible")
    return (
        (1.0 / (first * margin), -correlation / (sqrt(first * second) * margin)),
        (-correlation / (sqrt(first * second) * margin), 1.0 / (second * margin)),
    )


def _quadratic_form(gradient: tuple[float, ...], covariance: Matrix) -> float:
    return fsum(
        gradient[row] * covariance[row][column] * gradient[column]
        for row in range(len(gradient))
        for column in range(len(gradient))
    )


def _gradient(
    function: Callable[[tuple[float, ...]], float],
    theta: tuple[float, ...],
    steps: tuple[float, ...],
) -> tuple[float, ...]:
    """Central-difference gradient with Richardson extrapolation (error of order ``step**4``)."""

    def at(index: int, offset: float) -> float:
        shifted = list(theta)
        shifted[index] += offset
        return function(tuple(shifted))

    return tuple(
        (
            8.0 * (at(index, step) - at(index, -step))
            - (at(index, 2.0 * step) - at(index, -2.0 * step))
        )
        / (12.0 * step)
        for index, step in enumerate(steps)
    )


# ------------------------------------------------------------------- profile likelihood


class _Curve:
    """A one-dimensional profile of the log-likelihood on a search coordinate ``xi``.

    ``at(xi)`` returns the profile log-likelihood and, when available, its derivative in
    ``xi`` (``None`` otherwise); ``value(xi)`` maps the coordinate to the quantity itself.
    ``center`` is the coordinate of the estimate, ``step`` the Wald half-width in ``xi`` and
    ``hard`` the largest offset from ``center`` that is searched.
    """

    center: float
    step: float
    hard: float

    def reset(self) -> None:
        """Forget the warm start of the nuisance search."""

    def at(self, xi: float) -> tuple[float, float | None]:
        raise NotImplementedError

    def value(self, xi: float) -> float:
        raise NotImplementedError


def _safe_evaluate(model: Model, theta: tuple[float, ...]) -> Evaluation | None:
    try:
        evaluation = model.evaluate(theta, precise=False)
    except (ArithmeticError, ValueError):
        return None
    return evaluation if isfinite(evaluation.value) else None


def _maximize_nuisance(
    model: Model,
    theta_at: Callable[[float], tuple[float, ...]],
    free: int,
    start: float,
    cap: float,
) -> tuple[float, Evaluation | None]:
    """Maximize the log-likelihood over parameter ``free`` of a two-parameter model.

    ``theta_at(phi)`` builds the parameter vector from the search coordinate ``phi`` of the
    free parameter (its logarithm if it is positive, itself otherwise).  Safeguarded Newton
    iteration with step halving, at most ``_NEWTON_STEPS`` steps of at most ``cap``; if the
    curvature is not negative or no step improves the value, the bounded golden-section search
    of the fits takes over.  Returns the coordinate and the evaluation there, or ``None`` for
    the evaluation when the likelihood cannot be evaluated.
    """

    log_scale = model.positive[free]
    phi = start
    current = _safe_evaluate(model, theta_at(phi))
    if current is not None:
        for _ in range(_NEWTON_STEPS):
            weight = exp(phi) if log_scale else 1.0
            gradient = current.gradient[free] * weight
            curvature = current.hessian[free][free] * weight * weight + (
                gradient if log_scale else 0.0
            )
            if not curvature < 0.0:
                break
            if gradient * gradient <= -2.0 * curvature * _NEWTON_DECREMENT:
                return phi, current
            step = max(-cap, min(cap, -gradient / curvature))
            accepted = None
            for _ in range(_BACKTRACKS):
                candidate = _safe_evaluate(model, theta_at(phi + step))
                floor = current.value - _ACCEPT_SLACK * (1.0 + abs(current.value))
                if candidate is not None and candidate.value >= floor:
                    accepted = candidate
                    break
                step *= 0.5
            if accepted is None:
                break
            phi += step
            current = accepted

    def objective(point: float) -> float:
        evaluation = _safe_evaluate(model, theta_at(point))
        return -inf if evaluation is None else evaluation.value

    point, _, _ = expand_bracket(
        objective,
        lower=start - cap,
        upper=start + cap,
        hard_lower=start - 100.0 * cap,
        hard_upper=start + 100.0 * cap,
    )
    return point, _safe_evaluate(model, theta_at(point))


class _ParameterCurve(_Curve):
    """The profile of one parameter; the other parameter is maximized by Newton iteration."""

    def __init__(
        self, model: Model, theta: tuple[float, ...], covariance: Matrix, index: int, z: float
    ) -> None:
        self.model = model
        self.theta = theta
        self.index = index
        self.positive = model.positive[index]
        estimate = theta[index]
        error = sqrt(covariance[index][index])
        self.center = log(estimate) if self.positive else estimate
        self.step = z * (error / estimate if self.positive else error)
        self.hard = _HARD_LOG_OFFSET if self.positive else _HARD_IDENTITY_FACTOR * self.step
        self.free = None if len(theta) == 1 else 1 - index
        self.free_positive = False
        self.cap = 0.0
        self.start = 0.0
        self.slope = 0.0
        if self.free is not None:
            free_error = sqrt(covariance[self.free][self.free])
            free_estimate = theta[self.free]
            self.free_positive = model.positive[self.free]
            self.cap = _NEWTON_CAP_ERRORS * (
                free_error / free_estimate if self.free_positive else free_error
            )
            self.start = log(free_estimate) if self.free_positive else free_estimate
            # Regression slope of the free coordinate on the interest coordinate: the starting
            # point of the nuisance search is extrapolated linearly from the last solution.
            weight = estimate if self.positive else 1.0
            free_weight = free_estimate if self.free_positive else 1.0
            self.slope = (
                covariance[index][self.free] * weight / (free_weight * covariance[index][index])
            )
        self.warm = self.start
        self.warm_xi = self.center

    def reset(self) -> None:
        self.warm = self.start
        self.warm_xi = self.center

    def value(self, xi: float) -> float:
        return exp(xi) if self.positive else xi

    def at(self, xi: float) -> tuple[float, float | None]:
        value = self.value(xi)
        if self.free is None:
            evaluation = _safe_evaluate(self.model, (value,))
        else:
            free = self.free

            def theta_at(phi: float) -> tuple[float, ...]:
                built = [0.0, 0.0]
                built[self.index] = value
                built[free] = exp(phi) if self.free_positive else phi
                return tuple(built)

            guess = self.warm + self.slope * (xi - self.warm_xi)
            phi, evaluation = _maximize_nuisance(self.model, theta_at, free, guess, self.cap)
            if evaluation is not None:
                self.warm = phi
                self.warm_xi = xi
        if evaluation is None:
            return -inf, None
        weight = value if self.positive else 1.0
        return evaluation.value, evaluation.gradient[self.index] * weight


class _QuantityCurve(_Curve):
    """The profile of a derived quantity, reparameterized so the quantity is a coordinate.

    The free parameter (if any) is maximized by the bounded golden-section search of the
    fits, so this curve has no derivative.
    """

    def __init__(
        self,
        model: Model,
        theta: tuple[float, ...],
        reparameterization: Reparameterization,
        scale: str,
        estimate: float,
        step: float,
    ) -> None:
        self.model = model
        self.scale = scale
        self.reparameterization = reparameterization
        self.center = _logit(estimate) if scale == "logit" else log(estimate)
        self.step = step
        self.hard = _HARD_LOG_OFFSET
        free = reparameterization.free_index
        self.start = 0.0 if free is None else log(theta[free])

    def value(self, xi: float) -> float:
        return _expit(xi) if self.scale == "logit" else exp(xi)

    def _log_likelihood(self, quantity: float, free: float) -> float:
        try:
            theta = self.reparameterization.build(quantity, free)
            value = self.model.log_likelihood(theta)
        except (ArithmeticError, ValueError):
            return -inf
        return value if isfinite(value) else -inf

    def at(self, xi: float) -> tuple[float, float | None]:
        quantity = self.value(xi)
        if self.reparameterization.free_index is None:
            return self._log_likelihood(quantity, 0.0), None
        _, best, _ = expand_bracket(
            lambda phi: self._log_likelihood(quantity, exp(phi)),
            lower=self.start - 2.0,
            upper=self.start + 2.0,
            hard_lower=self.start - 30.0,
            hard_upper=self.start + 30.0,
            steps=60,
        )
        return best, None


def _solve_side(curve: _Curve, direction: float, target: float, maximum: float) -> float | None:
    """Find where ``2 (maximum - profile)`` reaches ``target`` on one side of the estimate.

    Returns the coordinate of the crossing, or ``None`` if the deviance stays below
    ``target`` within ``curve.hard`` of the estimate.  The crossing is first bracketed by
    doubling the offset (at most ``_MAX_EXPANSIONS`` times), then found by safeguarded Newton
    iteration (false position when the curve has no derivative) that falls back to bisection;
    at most ``_MAX_ROOT_ITERATIONS`` iterations.
    """

    curve.reset()

    def evaluate(xi: float) -> tuple[float, float | None]:
        profile, slope = curve.at(xi)
        return (
            2.0 * (maximum - profile) - target,
            None if slope is None else -2.0 * slope,
        )

    inner, f_inner = curve.center, -target
    offset = curve.step
    bracket: tuple[float, float, float | None] | None = None
    for _ in range(_MAX_EXPANSIONS):
        point = curve.center + direction * offset
        f_point, slope_point = evaluate(point)
        if f_point >= 0.0:
            bracket = (point, f_point, slope_point)
            break
        inner, f_inner = point, f_point
        if offset >= curve.hard:
            return None
        offset = min(2.0 * offset, curve.hard)
    if bracket is None:
        return None
    outer, f_outer = bracket[0], bracket[1]
    x, fx, slope = bracket
    for iteration in range(_MAX_ROOT_ITERATIONS):
        if abs(fx) <= _ROOT_TOLERANCE * target:
            return x
        if fx < 0.0:
            inner, f_inner = x, fx
        else:
            outer, f_outer = x, fx
        low, high = min(inner, outer), max(inner, outer)
        if slope is not None and slope != 0.0:
            candidate = x - fx / slope
        else:
            candidate = inner + (outer - inner) * (-f_inner) / (f_outer - f_inner)
        if not low < candidate < high or iteration >= _BISECTION_AFTER:
            candidate = 0.5 * (inner + outer)
        x = candidate
        fx, slope = evaluate(x)
        if high - low <= 1e-13 * (1.0 + abs(x)):
            return x
    return x


def _profile_interval(
    curve: _Curve, level: float, maximum: float, *, floor: float
) -> tuple[float, float, bool, bool]:
    """Return ``(lower, upper, lower_unbounded, upper_unbounded)`` of a profile interval.

    ``floor`` is the value reported for a lower side that never crosses.
    """

    target = _chi_square_1(level)
    lower_xi = _solve_side(curve, -1.0, target, maximum)
    upper_xi = _solve_side(curve, 1.0, target, maximum)
    lower = floor if lower_xi is None else curve.value(lower_xi)
    upper = inf if upper_xi is None else curve.value(upper_xi)
    return lower, upper, lower_xi is None, upper_xi is None


# ------------------------------------------------------------------ the public object


@dataclass(frozen=True, slots=True)
class FitUncertainty:
    """Covariance, standard errors, confidence intervals and derived quantities of one fit.

    ``covariance`` is a tuple of tuples in the order of ``parameter_names`` (the canonical
    parameter names of ``family``), the inverse of the observed information at ``estimates``.
    """

    family: FamilyId
    parameter_names: tuple[str, ...]
    estimates: tuple[float, ...]
    covariance: Matrix
    _model: Model = field(repr=False, compare=False)

    @property
    def standard_errors(self) -> Mapping[str, float]:
        """Standard error of each parameter, read-only, in parameter order."""

        return MappingProxyType(
            {
                name: sqrt(self.covariance[index][index])
                for index, name in enumerate(self.parameter_names)
            }
        )

    # ------------------------------------------------------------------- parameters

    def confidence_intervals(
        self, level: float = 0.95, method: str = "wald"
    ) -> Mapping[str, tuple[float, float]]:
        """Return a two-sided ``level`` confidence interval for every parameter.

        ``method`` is ``"wald"`` (log scale for positive parameters), ``"profile"``
        (likelihood ratio; a side that does not cross is ``0``/``-inf`` or ``inf``, see
        :meth:`interval_details` for the flags) or ``"exact"`` (the chi-square interval for
        the exponential rate on uncensored data; anything else raises ``ValueError``).
        """

        return MappingProxyType(
            {
                name: (interval.lower, interval.upper)
                for name, interval in self.interval_details(level, method).items()
            }
        )

    def interval_details(
        self, level: float = 0.95, method: str = "wald"
    ) -> Mapping[str, ParameterInterval]:
        """Like :meth:`confidence_intervals`, with the method, level and unbounded-side flags."""

        level = _validated_level(level)
        method = _validated_method(method)
        if method == "exact":
            lower, upper = self._exact_rate_interval(level)
            return MappingProxyType(
                {self.parameter_names[0]: ParameterInterval(lower, upper, level, "exact")}
            )
        z = _normal_quantile(level)
        details: dict[str, ParameterInterval] = {}
        for index, name in enumerate(self.parameter_names):
            if method == "wald":
                details[name] = self._wald_interval(index, level, z)
            else:
                details[name] = self._profile_parameter_interval(index, level, z)
        return MappingProxyType(details)

    def _wald_interval(self, index: int, level: float, z: float) -> ParameterInterval:
        estimate = self.estimates[index]
        error = sqrt(self.covariance[index][index])
        if self._model.positive[index]:
            factor = exp_or_inf(z * error / estimate)
            return ParameterInterval(estimate / factor, estimate * factor, level, "wald")
        return ParameterInterval(estimate - z * error, estimate + z * error, level, "wald")

    def _profile_parameter_interval(self, index: int, level: float, z: float) -> ParameterInterval:
        model = self._model
        curve = _ParameterCurve(model, self.estimates, self.covariance, index, z)
        maximum = model.log_likelihood(self.estimates)
        floor = 0.0 if model.positive[index] else -inf
        lower, upper, lower_open, upper_open = _profile_interval(curve, level, maximum, floor=floor)
        return ParameterInterval(lower, upper, level, "profile", lower_open, upper_open)

    def _exact_rate_interval(self, level: float) -> tuple[float, float]:
        model = self._model
        if not isinstance(model, ExponentialModel):
            raise ValueError('method="exact" is available only for the exponential family')
        if not model.exact_interval_available:
            raise ValueError(
                'method="exact" needs uncensored data: with right censoring the total time on '
                "test is not a sum of event times, so the chi-square pivot does not apply; use "
                'method="profile" or method="wald"'
            )
        # 2 * rate * T follows chi-square with 2r degrees of freedom, i.e. gamma(shape=r, scale=2).
        events = float(model.events)
        total = 2.0 * model.total_time
        tail = 0.5 * (1.0 - level)
        return (
            float(ppf("gamma", tail, shape=events, scale=2.0)) / total,
            float(ppf("gamma", 1.0 - tail, shape=events, scale=2.0)) / total,
        )

    # --------------------------------------------------------------------- derived

    def mean(self, level: float = 0.95, method: str = "wald") -> DerivedEstimate:
        """Return the distribution mean (the MTTF for a lifetime family) with an interval."""

        return self._derived("mean", 0.0, level, method)

    def quantile(
        self, probability: float, level: float = 0.95, method: str = "wald"
    ) -> DerivedEstimate:
        """Return the ``probability`` quantile with an interval.

        For a lifetime family this is the B-life: ``quantile(0.1)`` is B10, the time by which
        10% of the population has failed.
        """

        value = _real(probability, "probability")
        if not 0.0 < value < 1.0:
            raise ValueError("probability must be strictly between 0 and 1")
        return self._derived("quantile", value, level, method)

    def survival(self, time: float, level: float = 0.95, method: str = "wald") -> DerivedEstimate:
        """Return the survival probability at ``time`` with an interval on the logit scale."""

        value = _real(time, "time")
        if not isfinite(value):
            raise ValueError("time must be finite")
        return self._derived("survival", value, level, method)

    def _quantity(self, kind: str, argument: float) -> Callable[[tuple[float, ...]], float]:
        model = self._model
        if kind == "mean":
            return model.mean
        if kind == "quantile":
            return lambda theta: model.quantile(theta, argument)
        return lambda theta: model.survival(theta, argument)

    def _derived(self, kind: str, argument: float, level: float, method: str) -> DerivedEstimate:
        level = _validated_level(level)
        method = _validated_method(method)
        model = self._model
        quantity = self._quantity(kind, argument)
        estimate = quantity(self.estimates)
        if kind == "survival":
            scale = "logit"
            if not 0.0 < estimate < 1.0:
                return DerivedEstimate(estimate, estimate, estimate, method, level)
        else:
            scale = "log" if model.positive_support else "identity"
        if method == "exact":
            return self._exact_derived(kind, argument, level, estimate)
        z = _normal_quantile(level)
        error = self._delta_error(quantity)
        if method == "wald":
            lower, upper = _wald_bounds(scale, estimate, error, z)
            return DerivedEstimate(estimate, lower, upper, "wald", level)
        reparameterization = model.reparameterization(kind, argument)
        if reparameterization is None or scale == "identity":
            raise NotImplementedError(
                f"profile intervals for the {kind} of the {self.family.value} family are not "
                'implemented; use method="wald"'
            )
        step = z * (error / (estimate * (1.0 - estimate)) if scale == "logit" else error / estimate)
        curve = _QuantityCurve(model, self.estimates, reparameterization, scale, estimate, step)
        lower, upper, lower_open, upper_open = _profile_interval(
            curve, level, model.log_likelihood(self.estimates), floor=0.0
        )
        return DerivedEstimate(estimate, lower, upper, "profile", level, lower_open, upper_open)

    def _delta_error(self, quantity: Callable[[tuple[float, ...]], float]) -> float:
        steps = tuple(
            _GRADIENT_STEP * sqrt(self.covariance[index][index])
            if not positive
            else min(
                _GRADIENT_STEP * sqrt(self.covariance[index][index]),
                _GRADIENT_RELATIVE_LIMIT * self.estimates[index],
            )
            for index, positive in enumerate(self._model.positive)
        )
        gradient = _gradient(quantity, self.estimates, steps)
        return sqrt(_quadratic_form(gradient, self.covariance))

    def _exact_derived(
        self, kind: str, argument: float, level: float, estimate: float
    ) -> DerivedEstimate:
        rate_lower, rate_upper = self._exact_rate_interval(level)
        # The exponential mean, quantiles and survival are all decreasing in the rate.
        quantity = self._quantity(kind, argument)
        lower = quantity((rate_upper,))
        upper = quantity((rate_lower,))
        return DerivedEstimate(estimate, lower, upper, "exact", level)


def _wald_bounds(scale: str, estimate: float, error: float, z: float) -> tuple[float, float]:
    """Delta-method bounds on the ``scale`` ("log", "logit" or "identity") mapped back."""

    if scale == "log":
        factor = exp_or_inf(z * error / estimate)
        return estimate / factor, estimate * factor
    if scale == "logit":
        half = z * error / (estimate * (1.0 - estimate))
        center = _logit(estimate)
        return _expit(center - half), _expit(center + half)
    return estimate - z * error, estimate + z * error


# --------------------------------------------------------------------------- building


def _from_model(
    family: FamilyId, model: Model, theta: tuple[float, ...]
) -> FitUncertainty | UncertaintyUnavailable:
    try:
        evaluation = model.evaluate(theta)
    except (ArithmeticError, ValueError):
        return UncertaintyUnavailable(
            UncertaintyUnavailableReason.NOT_COMPUTABLE,
            "the log-likelihood could not be evaluated at the fitted parameters",
        )
    covariance = _invert_information(
        tuple(tuple(-entry for entry in row) for row in evaluation.hessian)
    )
    if isinstance(covariance, UncertaintyUnavailable):
        return covariance
    return FitUncertainty(family, model.names, theta, covariance, model)


def compute_uncertainty(
    family: FamilyId, parameters: Mapping[str, float], evidence: FitEvidence
) -> FitUncertainty | UncertaintyUnavailable:
    """Return the uncertainty of a fit at ``parameters`` from the sample in ``evidence``."""

    if evidence.fixed:
        names = ", ".join(evidence.fixed)
        return UncertaintyUnavailable(
            UncertaintyUnavailableReason.FIXED_PARAMETER,
            f"the {names} was fixed by the caller, so the observed information of the fit does "
            "not describe it",
        )
    if not evidence.exact:
        return UncertaintyUnavailable(
            UncertaintyUnavailableReason.NO_DATA,
            "the result carries no observations (it was not produced by a fit function)",
        )
    try:
        model = build_model(family, evidence.exact, evidence.censored)
    except (ArithmeticError, ValueError):
        return UncertaintyUnavailable(
            UncertaintyUnavailableReason.NOT_COMPUTABLE,
            "the observations could not be prepared for the likelihood",
        )
    theta = tuple(float(parameters[name]) for name in model.names)
    return _from_model(family, model, theta)


def exponential_uncertainty(
    rate: float, events: int, total_time: float, censored: int
) -> FitUncertainty | UncertaintyUnavailable:
    """Return the uncertainty of an exponential fit from its sufficient statistics."""

    return _from_model(
        FamilyId.EXPONENTIAL, ExponentialModel(events, total_time, censored), (rate,)
    )


__all__ = [
    "DerivedEstimate",
    "FitUncertainty",
    "ParameterInterval",
    "UncertaintyUnavailable",
    "UncertaintyUnavailableReason",
    "compute_uncertainty",
    "exponential_uncertainty",
]
