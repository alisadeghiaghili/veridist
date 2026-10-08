"""Censored log-likelihoods with analytic derivatives, one model per family.

Each model evaluates the log-likelihood of one fitted sample, together with its
gradient and Hessian in the canonical parameters, at any parameter vector.  The
uncertainty machinery (observed information, profile likelihood) is written once
against this small interface.

An exactly observed value contributes its log-density and a right-censoring point
contributes its log-survival.

* Exponential and Weibull-minimum: closed-form derivatives.
* Normal, lognormal and right Gumbel are location-scale families.  With
  ``z = (y - mu) / sigma``, the derivative of each observation's term with respect to
  ``z`` determines the whole Hessian (:func:`location_scale_evaluation`), so the three
  share one derivation.
* Gamma: the exact terms are analytic (digamma and trigamma).  The derivative of
  ``log Q(shape, x)`` in the shape has no closed form, so the censored terms use analytic
  derivatives in the scale and a central difference with Richardson extrapolation on
  ``log(shape)`` for the shape derivatives (step ``_GAMMA_SHAPE_STEP``).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable
from math import exp, expm1, fsum, gamma, lgamma, log, log1p, pi
from typing import Final, NamedTuple

from veridist.families.gumbel import _log_survival as _gumbel_log_survival
from veridist.families.lognormal import _log_normal_sf
from veridist.families.registry import FamilyId
from veridist.statistics._special import digamma, trigamma
from veridist.statistics.distributions import _log_regularized_gamma_q, ppf, sf

_HALF_LOG_2PI: Final = 0.5 * log(2.0 * pi)
_EULER_GAMMA: Final = 0.5772156649015329
#: ``exp`` overflows beyond this argument.
_EXP_LIMIT: Final = 700.0
#: Step, in ``log(shape)``, of the gamma shape differences.  The truncation error of the
#: Richardson-extrapolated stencils is of order ``step**4`` and the rounding error of
#: order ``1e-16 / step**2``; both are far below the 1e-6 agreement the tests demand.
_GAMMA_SHAPE_STEP: Final = 0.03
#: The same for ``evaluate(precise=False)``: a plain three-point central difference, whose
#: truncation error is of order ``step**2`` and rounding error of order ``1e-16 / step**2``.
_GAMMA_FAST_STEP: Final = 0.002
_PRECISE_OFFSETS: Final = (-2, -1, 0, 1, 2)
_FAST_OFFSETS: Final = (-1, 0, 1)
#: The normal hazard switches to its continued fraction from here.
_HAZARD_CONTINUED_FRACTION_FROM: Final = 12.0
_HAZARD_CONTINUED_FRACTION_TERMS: Final = 60

Matrix = tuple[tuple[float, ...], ...]


class Evaluation(NamedTuple):
    """Log-likelihood, gradient and Hessian at one parameter vector."""

    value: float
    gradient: tuple[float, ...]
    hessian: Matrix


class Reparameterization(NamedTuple):
    """Express the parameters through a derived quantity and one free parameter.

    ``build(quantity, free)`` returns the full parameter vector for which the derived
    quantity equals ``quantity`` while the parameter at ``free_index`` equals ``free``.
    ``free_index`` is ``None`` for a one-parameter family, where the quantity alone fixes
    the parameter and ``free`` is ignored.
    """

    free_index: int | None
    build: Callable[[float, float], tuple[float, ...]]


class Model:
    """The interface of every likelihood model.

    ``names`` are the canonical parameter names in registry order, ``positive`` marks the
    parameters that live on ``(0, inf)``, and ``positive_support`` tells whether the mean
    and the quantiles of the distribution are positive.  ``exact_interval_available`` is
    true only for the exponential model on uncensored data.
    """

    family: FamilyId
    names: tuple[str, ...]
    positive: tuple[bool, ...]
    positive_support: bool
    exact_interval_available: bool = False

    def evaluate(self, theta: tuple[float, ...], *, precise: bool = True) -> Evaluation:
        """Return the log-likelihood, gradient and Hessian at ``theta``.

        ``precise`` matters only to a model whose derivatives are numerical (gamma): ``False``
        trades about a factor of ten in derivative accuracy for speed, which the iterative
        profile searches can afford because they only need the derivatives to converge.
        """

        raise NotImplementedError

    def log_likelihood(self, theta: tuple[float, ...]) -> float:
        """Return the log-likelihood at ``theta`` (a model may override with a cheaper form)."""

        return self.evaluate(theta).value

    def mean(self, theta: tuple[float, ...]) -> float:
        """Return the distribution mean at ``theta``."""

        raise NotImplementedError

    def quantile(self, theta: tuple[float, ...], probability: float) -> float:
        """Return the quantile at ``probability``, through the public ``ppf``."""

        return float(ppf(self.family, probability, **dict(zip(self.names, theta, strict=True))))

    def survival(self, theta: tuple[float, ...], time: float) -> float:
        """Return the survival probability at ``time``, through the public ``sf``."""

        return float(sf(self.family, time, **dict(zip(self.names, theta, strict=True))))

    def reparameterization(self, quantity: str, argument: float) -> Reparameterization | None:
        """Return the reparameterization for a profile of ``quantity``, or ``None``.

        ``quantity`` is ``"mean"``, ``"quantile"`` or ``"survival"`` and ``argument`` the
        probability or the time (unused for the mean).  ``None`` means that no profile is
        implemented for that quantity.
        """

        return None


def _grouped(values: Iterable[float]) -> tuple[tuple[float, int], ...]:
    """Collapse repeated values into ``(value, multiplicity)`` pairs, in first-seen order."""

    return tuple(Counter(values).items())


# --------------------------------------------------------------------------- exponential


class ExponentialModel(Model):
    """Rate-only exponential: ``events * log(rate) - rate * total_time``."""

    family = FamilyId.EXPONENTIAL
    names = ("rate",)
    positive = (True,)
    positive_support = True

    def __init__(self, events: int, total_time: float, censored: int) -> None:
        self.events = events
        self.total_time = total_time
        self.exact_interval_available = censored == 0

    def evaluate(self, theta: tuple[float, ...], *, precise: bool = True) -> Evaluation:
        (rate,) = theta
        events = float(self.events)
        return Evaluation(
            events * log(rate) - rate * self.total_time,
            (events / rate - self.total_time,),
            ((-events / (rate * rate),),),
        )

    def mean(self, theta: tuple[float, ...]) -> float:
        return 1.0 / theta[0]

    def reparameterization(self, quantity: str, argument: float) -> Reparameterization | None:
        if quantity == "mean":
            return Reparameterization(None, lambda value, _free: (1.0 / value,))
        if quantity == "quantile":
            level = -log1p(-argument)
            return Reparameterization(None, lambda value, _free: (level / value,))
        return Reparameterization(None, lambda value, _free: (-log(value) / argument,))


# ------------------------------------------------------------------------------ weibull


class WeibullModel(Model):
    """Weibull-minimum with ``shape`` and ``scale``."""

    family = FamilyId.WEIBULL_MIN
    names = ("shape", "scale")
    positive = (True, True)
    positive_support = True

    def __init__(self, exact: tuple[float, ...], censored: tuple[float, ...]) -> None:
        self.events = len(exact)
        self.exact_log_sum = fsum(log(time) for time in exact)
        self.groups = tuple(
            (log(time), multiplicity) for time, multiplicity in _grouped((*exact, *censored))
        )

    def evaluate(self, theta: tuple[float, ...], *, precise: bool = True) -> Evaluation:
        shape, scale = theta
        events = float(self.events)
        log_scale = log(scale)
        powers: list[float] = []
        first: list[float] = []
        second: list[float] = []
        for log_time, multiplicity in self.groups:
            deviation = log_time - log_scale
            weighted = multiplicity * exp(shape * deviation)
            powers.append(weighted)
            first.append(weighted * deviation)
            second.append(weighted * deviation * deviation)
        total = fsum(powers)
        total_first = fsum(first)
        total_second = fsum(second)
        value = (
            events * log(shape)
            - events * shape * log_scale
            + (shape - 1.0) * self.exact_log_sum
            - total
        )
        excess = total - events
        gradient = (
            events / shape + (self.exact_log_sum - events * log_scale) - total_first,
            shape / scale * excess,
        )
        mixed = (excess + shape * total_first) / scale
        hessian = (
            (-events / (shape * shape) - total_second, mixed),
            (mixed, -shape / (scale * scale) * (excess + shape * total)),
        )
        return Evaluation(value, gradient, hessian)

    def mean(self, theta: tuple[float, ...]) -> float:
        shape, scale = theta
        return scale * gamma(1.0 + 1.0 / shape)

    def reparameterization(self, quantity: str, argument: float) -> Reparameterization | None:
        def with_scale(scale_of: Callable[[float, float], float]) -> Reparameterization:
            return Reparameterization(0, lambda value, shape: (shape, scale_of(value, shape)))

        if quantity == "mean":
            return with_scale(lambda value, shape: value / gamma(1.0 + 1.0 / shape))
        if quantity == "quantile":
            level = -log1p(-argument)
            return with_scale(lambda value, shape: value / level ** (1.0 / shape))
        return with_scale(lambda value, shape: argument / (-log(value)) ** (1.0 / shape))


# ------------------------------------------------------------------- location-scale core


class _Sums:
    """Running sums of ``g``, ``z g``, ``g'``, ``z g'`` and ``z^2 g'``, each added with ``fsum``."""

    __slots__ = ("g", "gp", "zg", "zgp", "zzgp")

    def __init__(self) -> None:
        self.g: list[float] = []
        self.zg: list[float] = []
        self.gp: list[float] = []
        self.zgp: list[float] = []
        self.zzgp: list[float] = []

    def add(self, weight: float, z: float, first: float, second: float) -> None:
        """Add ``weight`` identical observations with derivatives ``first`` and ``second``."""

        self.g.append(weight * first)
        self.zg.append(weight * z * first)
        self.gp.append(weight * second)
        self.zgp.append(weight * z * second)
        self.zzgp.append(weight * z * z * second)


def location_scale_evaluation(
    value: float, exact_count: int, sigma: float, sums: _Sums
) -> Evaluation:
    """Assemble the gradient and Hessian of a location-scale log-likelihood.

    ``sums`` accumulates, over all observations, ``g`` and ``g'``: the first and second
    derivative with respect to ``z = (y - mu) / sigma`` of the observation's log-likelihood
    term (the ``-log(sigma)`` of an exact density is handled here through ``exact_count``).
    Then

    * ``dl/dmu = -S(g) / sigma`` and ``dl/dsigma = -(d + S(z g)) / sigma``;
    * ``d2l/dmu2 = S(g') / sigma^2``, ``d2l/dmu dsigma = (S(g) + S(z g')) / sigma^2`` and
      ``d2l/dsigma2 = (d + 2 S(z g) + S(z^2 g')) / sigma^2``,

    with ``S`` the sum over observations and ``d`` the number of exact observations.
    """

    sum_g, sum_zg, sum_gp = fsum(sums.g), fsum(sums.zg), fsum(sums.gp)
    sum_zgp, sum_zzgp = fsum(sums.zgp), fsum(sums.zzgp)
    variance = sigma * sigma
    mixed = (sum_g + sum_zgp) / variance
    return Evaluation(
        value,
        (-sum_g / sigma, -(exact_count + sum_zg) / sigma),
        (
            (sum_gp / variance, mixed),
            (mixed, (exact_count + 2.0 * sum_zg + sum_zzgp) / variance),
        ),
    )


def normal_hazard(z: float) -> tuple[float, float]:
    """Return ``(r, r - z)`` for the standard-normal hazard ``r(z) = phi(z) / Q(z)``.

    ``Q`` is the upper tail.  ``r - z`` is returned separately because the derivative of
    the hazard is ``r' = r (r - z)`` and the difference cancels badly in the upper tail;
    from ``_HAZARD_CONTINUED_FRACTION_FROM`` on it is taken directly from the continued
    fraction ``r(z) = z + 1 / (z + 2 / (z + 3 / (z + ...)))``.
    """

    if z >= _HAZARD_CONTINUED_FRACTION_FROM:
        tail = z
        for order in range(_HAZARD_CONTINUED_FRACTION_TERMS, 1, -1):
            tail = z + order / tail
        excess = 1.0 / tail
        return z + excess, excess
    ratio = exp(-0.5 * z * z - _HALF_LOG_2PI - _log_normal_sf(z))
    return ratio, ratio - z


class GaussianModel(Model):
    """Normal (``mu``, ``sigma``) or lognormal (``mu_log``, ``sigma_log``) on ``y``.

    For the lognormal model ``y`` is the logarithm of the time and ``constant`` carries the
    Jacobian ``-sum(log t)`` of the exact observations.  The exact observations enter only
    through their count and the sums of deviations from their mean, so one evaluation costs
    a single pass over the censoring points.
    """

    positive = (False, True)

    def __init__(
        self,
        family: FamilyId,
        names: tuple[str, str],
        exact: tuple[float, ...],
        censored: tuple[float, ...],
        constant: float,
    ) -> None:
        self.family = family
        self.names = names
        self.positive_support = family is FamilyId.LOGNORMAL
        self.constant = constant
        self.events = len(exact)
        self.center = fsum(exact) / len(exact)
        deviations = tuple(value - self.center for value in exact)
        self.sum_deviations = fsum(deviations)
        self.sum_squares = fsum(value * value for value in deviations)
        self.censored = _grouped(censored)

    def evaluate(self, theta: tuple[float, ...], *, precise: bool = True) -> Evaluation:
        mu, sigma = theta
        events = self.events
        shift = mu - self.center
        sum_z = (self.sum_deviations - events * shift) / sigma
        sum_z2 = (self.sum_squares - 2.0 * shift * self.sum_deviations + events * shift * shift) / (
            sigma * sigma
        )
        terms = [self.constant - events * (log(sigma) + _HALF_LOG_2PI), -0.5 * sum_z2]
        sums = _Sums()
        # Exact observations: g = -z and g' = -1, summed in closed form.
        sums.g.append(-sum_z)
        sums.zg.append(-sum_z2)
        sums.gp.append(-float(events))
        sums.zgp.append(-sum_z)
        sums.zzgp.append(-sum_z2)
        for point, multiplicity in self.censored:
            z = (point - mu) / sigma
            hazard, excess = normal_hazard(z)
            terms.append(multiplicity * _log_normal_sf(z))
            sums.add(multiplicity, z, -hazard, -hazard * excess)
        return location_scale_evaluation(fsum(terms), events, sigma, sums)

    def mean(self, theta: tuple[float, ...]) -> float:
        mu, sigma = theta
        return exp(mu + 0.5 * sigma * sigma) if self.positive_support else mu


# ----------------------------------------------------------------------------- gumbel


def _gumbel_censored(z: float) -> tuple[float, float, float]:
    """Return ``(log S, g, g')`` of a right-censored Gumbel observation at ``z``.

    With ``w = exp(-z)`` the survival is ``1 - exp(-w)``; its log has first derivative
    ``-r`` and second derivative ``-r'`` in ``z`` where ``r = w / expm1(w)`` and
    ``r' = r (r exp(w) - 1)``.  Far left (``w`` huge) the survival is one and the
    derivatives vanish; far right (``w`` underflows) ``r`` tends to one and ``r'`` to zero.
    """

    if z < -_EXP_LIMIT:
        return 0.0, 0.0, 0.0
    tail = exp(-z)
    if tail > _EXP_LIMIT:
        return _gumbel_log_survival(z), 0.0, 0.0
    if tail == 0.0:
        return _gumbel_log_survival(z), -1.0, 0.0
    growth = expm1(tail)
    hazard = tail / growth
    return _gumbel_log_survival(z), -hazard, -hazard * (hazard * (growth + 1.0) - 1.0)


class GumbelModel(Model):
    """Right Gumbel (``location``, ``scale``): density ``exp(-z - exp(-z)) / scale``."""

    family = FamilyId.GUMBEL_RIGHT
    names = ("location", "scale")
    positive = (False, True)
    positive_support = False

    def __init__(self, exact: tuple[float, ...], censored: tuple[float, ...]) -> None:
        self.events = len(exact)
        self.exact = _grouped(exact)
        self.censored = _grouped(censored)

    def evaluate(self, theta: tuple[float, ...], *, precise: bool = True) -> Evaluation:
        location, scale = theta
        terms = [-self.events * log(scale)]
        sums = _Sums()
        for point, multiplicity in self.exact:
            z = (point - location) / scale
            tail = exp(-z)
            terms.append(multiplicity * (-z - tail))
            sums.add(multiplicity, z, tail - 1.0, -tail)
        for point, multiplicity in self.censored:
            z = (point - location) / scale
            log_survival, first, second = _gumbel_censored(z)
            terms.append(multiplicity * log_survival)
            sums.add(multiplicity, z, first, second)
        return location_scale_evaluation(fsum(terms), self.events, scale, sums)

    def mean(self, theta: tuple[float, ...]) -> float:
        location, scale = theta
        return location + _EULER_GAMMA * scale


# ------------------------------------------------------------------------------- gamma


class GammaModel(Model):
    """Gamma (``shape``, ``scale``) with a censored part differentiated numerically in the shape."""

    family = FamilyId.GAMMA
    names = ("shape", "scale")
    positive = (True, True)
    positive_support = True

    def __init__(self, exact: tuple[float, ...], censored: tuple[float, ...]) -> None:
        self.events = len(exact)
        self.sum_exact = fsum(exact)
        self.sum_logs = fsum(log(time) for time in exact)
        self.censored = tuple(
            (time, log(time), multiplicity) for time, multiplicity in _grouped(censored)
        )

    def _exact_value(self, shape: float, scale: float) -> float:
        events = float(self.events)
        return (
            -events * lgamma(shape)
            - events * shape * log(scale)
            + (shape - 1.0) * self.sum_logs
            - self.sum_exact / scale
        )

    def log_likelihood(self, theta: tuple[float, ...]) -> float:
        shape, scale = theta
        log_scale = log(scale)
        terms = [self._exact_value(shape, scale)]
        for time, log_time, multiplicity in self.censored:
            terms.append(
                multiplicity * _log_regularized_gamma_q(shape, time / scale, log_time - log_scale)
            )
        return fsum(terms)

    def evaluate(self, theta: tuple[float, ...], *, precise: bool = True) -> Evaluation:
        shape, scale = theta
        events = float(self.events)
        log_scale = log(scale)
        offsets = _PRECISE_OFFSETS if precise else _FAST_OFFSETS
        h = _GAMMA_SHAPE_STEP if precise else _GAMMA_FAST_STEP
        value = [self._exact_value(shape, scale)]
        d_a: list[float] = []  # derivative in log(shape)
        d_aa: list[float] = []
        d_b: list[float] = []  # derivative in log(scale)
        d_ab: list[float] = []
        d_bb: list[float] = []
        for time, log_time, multiplicity in self.censored:
            y = time / scale
            log_y = log_time - log_scale
            # log Q and y times the hazard (the derivative of log Q in log(scale)) at the
            # shapes exp(log(shape) + j h), for j in ``offsets``.
            log_q: list[float] = []
            slope: list[float] = []
            for step in offsets:
                shape_j = shape * exp(step * h)
                value_j = _log_regularized_gamma_q(shape_j, y, log_y)
                hazard = exp((shape_j - 1.0) * log_y - y - lgamma(shape_j) - value_j)
                log_q.append(value_j)
                slope.append(hazard * y)
            center = len(offsets) // 2
            first, second = _shape_derivatives(log_q, h)
            value.append(multiplicity * log_q[center])
            d_a.append(multiplicity * first)
            d_aa.append(multiplicity * second)
            d_b.append(multiplicity * slope[center])
            d_ab.append(multiplicity * _shape_derivatives(slope, h)[0])
            d_bb.append(-multiplicity * slope[center] * (shape - y + slope[center]))
        f_a, f_aa, f_b, f_ab, f_bb = (fsum(d_a), fsum(d_aa), fsum(d_b), fsum(d_ab), fsum(d_bb))
        gradient = (
            -events * digamma(shape) - events * log_scale + self.sum_logs + f_a / shape,
            -events * shape / scale + self.sum_exact / (scale * scale) + f_b / scale,
        )
        mixed = -events / scale + f_ab / (shape * scale)
        hessian = (
            (-events * trigamma(shape) + (f_aa - f_a) / (shape * shape), mixed),
            (
                mixed,
                events * shape / (scale * scale)
                - 2.0 * self.sum_exact / scale**3
                + (f_bb - f_b) / (scale * scale),
            ),
        )
        return Evaluation(fsum(value), gradient, hessian)

    def mean(self, theta: tuple[float, ...]) -> float:
        shape, scale = theta
        return shape * scale


def _shape_derivatives(values: list[float], step: float) -> tuple[float, float]:
    """Return the first and second derivative at the middle of equally spaced ``values``.

    Five values (offsets ``-2..2``) give the Richardson-extrapolated stencils
    ``(8 (f1 - f-1) - (f2 - f-2)) / 12h`` and ``(-f2 + 16 f1 - 30 f0 + 16 f-1 - f-2) / 12h^2``;
    three values give the central differences ``(f1 - f-1) / 2h`` and
    ``(f1 - 2 f0 + f-1) / h^2``.
    """

    if len(values) == 5:
        low2, low1, middle, high1, high2 = values
        return (
            (8.0 * (high1 - low1) - (high2 - low2)) / (12.0 * step),
            (-high2 + 16.0 * high1 - 30.0 * middle + 16.0 * low1 - low2) / (12.0 * step * step),
        )
    low1, middle, high1 = values
    return (high1 - low1) / (2.0 * step), (high1 - 2.0 * middle + low1) / (step * step)


def build_model(family: FamilyId, exact: tuple[float, ...], censored: tuple[float, ...]) -> Model:
    """Return the likelihood model of ``family`` for the sample ``exact`` / ``censored``.

    For the lifetime families the values are the positive event and censoring times; for
    the real-line families they are the values themselves.  The exponential model is built
    from its sufficient statistics instead (:class:`ExponentialModel`).
    """

    if family is FamilyId.WEIBULL_MIN:
        return WeibullModel(exact, censored)
    if family is FamilyId.GAMMA:
        return GammaModel(exact, censored)
    if family is FamilyId.GUMBEL_RIGHT:
        return GumbelModel(exact, censored)
    if family is FamilyId.NORMAL:
        return GaussianModel(family, ("mu", "sigma"), exact, censored, 0.0)
    logs = tuple(log(time) for time in exact)
    return GaussianModel(
        FamilyId.LOGNORMAL,
        ("mu_log", "sigma_log"),
        logs,
        tuple(log(time) for time in censored),
        -fsum(logs),
    )


__all__ = [
    "Evaluation",
    "ExponentialModel",
    "GammaModel",
    "GaussianModel",
    "GumbelModel",
    "Model",
    "Reparameterization",
    "WeibullModel",
    "build_model",
    "location_scale_evaluation",
    "normal_hazard",
]
