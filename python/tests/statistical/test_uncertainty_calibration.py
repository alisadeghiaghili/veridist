"""Coverage calibration of the 95% Wald and profile-likelihood intervals, family by family.

For each of the six families, two simulated settings are repeated many times with a seeded
``numpy.random.default_rng``:

* ``n = 30``, no censoring;
* ``n = 60`` with independent right censoring of about 30% (censoring times are exponential
  with a family-specific scale ``s`` chosen so that ``P(C < T) = 0.30`` for the true
  distribution: ``s`` solves ``1 - E[exp(-T / s)] = 0.30``, computed once with
  ``scipy.integrate.quad`` and ``scipy.optimize.brentq`` and frozen below).

For every dataset the fit is formed, the 95% Wald and profile intervals of every parameter are
computed and the empirical coverage of the true value is recorded.

Acceptance bands are exact binomial: with ``R`` replicates the number of covering intervals is
``Binomial(R, p)`` when the interval has true coverage ``p``, and the 99.9% band is the central
interval holding 99.9% of that distribution (``0.05%`` in each tail).  The band is derived from
``R`` alone; nothing is tuned.

* Profile intervals must fall inside the band around the nominal 0.95.
* Wald intervals are only first-order accurate and are known to undercover at these sample
  sizes, so they must fall inside the band that runs from the lower end of the band around 0.90
  (a documented allowance of five coverage points) to the upper end of the band around 0.95.

Replicates are traded off against cost (see ``REPLICATES``): the exponential and uncensored
fits are cheap, the censored gamma profile is not.  Every case has at least 400.  The measured
coverages are printed (run with ``pytest -s``).

Where a fit is cheap (every uncensored fit, the exponential and the censored Weibull) the public
``fit`` is used.  The four censored fits that rely on a nested golden-section search (normal,
lognormal, right Gumbel and gamma; 60 to 500 ms per fit) are replaced by a damped Newton search on
the same log-likelihood, started from moment estimates and run to a Newton decrement below 1e-12;
``test_newton_estimates_agree_with_fit`` checks on the first datasets of each of these cases that
it lands on the same estimate as ``fit``.
"""

from __future__ import annotations

import math
import statistics
import time
import unittest
from collections.abc import Callable
from typing import Any

import numpy as np

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.domain.values import ExactValue, RightCensoredValue
from veridist.families import (
    FitSuccess,
    FitUncertainty,
    GammaFitSuccess,
    GumbelFitSuccess,
    LognormalFitSuccess,
    NormalFitSuccess,
    fit,
)
from veridist.families._evidence import FitEvidence
from veridist.families._models import Model, build_model
from veridist.families.registry import FamilyId

NORMAL = statistics.NormalDist()
LEVEL = 0.95
BAND_MASS = 0.999
#: Allowed Wald undercoverage at these sample sizes (the band is taken around this coverage).
WALD_FLOOR = 0.90
CENSORED_FRACTION = 0.30

TRUE_PARAMETERS: dict[FamilyId, tuple[float, ...]] = {
    FamilyId.EXPONENTIAL: (0.1,),
    FamilyId.WEIBULL_MIN: (1.5, 10.0),
    FamilyId.LOGNORMAL: (2.0, 0.6),
    FamilyId.GAMMA: (2.5, 4.0),
    FamilyId.NORMAL: (50.0, 8.0),
    FamilyId.GUMBEL_RIGHT: (50.0, 6.0),
}
#: Scale of the exponential censoring time so that about 30% of the observations are censored.
CENSORING_SCALE: dict[FamilyId, float] = {
    FamilyId.EXPONENTIAL: 23.3333,
    FamilyId.WEIBULL_MIN: 23.2404,
    FamilyId.LOGNORMAL: 23.0492,
    FamilyId.GAMMA: 26.0843,
    FamilyId.NORMAL: 139.5407,
    FamilyId.GUMBEL_RIGHT: 149.3484,
}
#: (replicates uncensored, replicates censored); the cost per replicate drives the split.
REPLICATES: dict[FamilyId, tuple[int, int]] = {
    FamilyId.EXPONENTIAL: (2000, 2000),
    FamilyId.WEIBULL_MIN: (500, 500),
    FamilyId.LOGNORMAL: (800, 400),
    FamilyId.GAMMA: (800, 400),
    FamilyId.NORMAL: (800, 400),
    FamilyId.GUMBEL_RIGHT: (400, 400),
}
SAMPLE_SIZES = (30, 60)
SEEDS = {
    FamilyId.EXPONENTIAL: 20261001,
    FamilyId.WEIBULL_MIN: 20261002,
    FamilyId.LOGNORMAL: 20261003,
    FamilyId.GAMMA: 20261004,
    FamilyId.NORMAL: 20261005,
    FamilyId.GUMBEL_RIGHT: 20261006,
}
LIFETIME_FAMILIES = (
    FamilyId.EXPONENTIAL,
    FamilyId.WEIBULL_MIN,
    FamilyId.LOGNORMAL,
    FamilyId.GAMMA,
)
#: Censored fits whose own optimizer is too slow to repeat hundreds of times.
NEWTON_FAMILIES = (
    FamilyId.LOGNORMAL,
    FamilyId.GAMMA,
    FamilyId.NORMAL,
    FamilyId.GUMBEL_RIGHT,
)
SUCCESS_TYPES: dict[FamilyId, Callable[..., FitSuccess]] = {
    FamilyId.LOGNORMAL: LognormalFitSuccess,
    FamilyId.GAMMA: GammaFitSuccess,
    FamilyId.NORMAL: NormalFitSuccess,
    FamilyId.GUMBEL_RIGHT: GumbelFitSuccess,
}


def binomial_band(
    replicates: int, probability: float, mass: float = BAND_MASS
) -> tuple[float, float]:
    """Return the central ``mass`` band of ``Binomial(replicates, probability) / replicates``.

    The lower end is the smallest ``k`` with ``P(X <= k) >= (1 - mass) / 2`` and the upper end
    the largest ``k`` with ``P(X >= k) >= (1 - mass) / 2``, both as proportions of ``replicates``.
    """

    tail = 0.5 * (1.0 - mass)
    pmf = [
        math.exp(
            math.lgamma(replicates + 1)
            - math.lgamma(k + 1)
            - math.lgamma(replicates - k + 1)
            + k * math.log(probability)
            + (replicates - k) * math.log1p(-probability)
        )
        for k in range(replicates + 1)
    ]
    lower = next(k for k in range(replicates + 1) if math.fsum(pmf[: k + 1]) >= tail)
    upper = next(k for k in range(replicates, -1, -1) if math.fsum(pmf[k:]) >= tail)
    return lower / replicates, upper / replicates


def draw(family: FamilyId, uniforms: Any) -> float:
    """Map independent uniforms on ``(0, 1)`` to one draw from the true distribution.

    Only the raw uniform stream of ``numpy.random.Generator.random`` is used (its bits are
    frozen for a given seed), and every transform is closed-form or standard library, so the
    simulated data do not depend on how a numpy release implements its distribution methods.
    """

    truth = TRUE_PARAMETERS[family]
    u = float(uniforms[0])
    if family is FamilyId.EXPONENTIAL:
        return -math.log1p(-u) / truth[0]
    if family is FamilyId.WEIBULL_MIN:
        return truth[1] * (-math.log1p(-u)) ** (1.0 / truth[0])
    if family is FamilyId.LOGNORMAL:
        return math.exp(truth[0] + truth[1] * NORMAL.inv_cdf(u))
    if family is FamilyId.GAMMA:
        # Gamma(2.5, scale 4) = 4 * (E1 + E2) + 2 * Z^2: shapes 1 + 1 + 1/2 add at a common
        # scale, and Gamma(1/2, scale 4) is 2 Z^2 for a standard normal Z.
        first, second, third = (float(value) for value in uniforms[:3])
        return 4.0 * (-math.log1p(-first) - math.log1p(-second)) + 2.0 * NORMAL.inv_cdf(third) ** 2
    if family is FamilyId.NORMAL:
        return truth[0] + truth[1] * NORMAL.inv_cdf(u)
    return truth[0] - truth[1] * math.log(-math.log(u))


def simulate(
    family: FamilyId, size: int, rng: np.random.Generator, *, censored: bool
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Draw one dataset: exact values and right-censoring points."""

    uniforms = rng.random((size, 3))
    times = [draw(family, row) for row in uniforms]
    if not censored:
        return tuple(times), ()
    limits = [-CENSORING_SCALE[family] * math.log1p(-float(u)) for u in rng.random(size)]
    exact = tuple(t for t, c in zip(times, limits, strict=True) if t <= c)
    return exact, tuple(c for t, c in zip(times, limits, strict=True) if t > c)


def observations(
    family: FamilyId, exact: tuple[float, ...], censored: tuple[float, ...]
) -> list[Any]:
    if family in LIFETIME_FAMILIES:
        return [ExactLifetime(x) for x in exact] + [RightCensoredLifetime(x) for x in censored]
    return [ExactValue(x) for x in exact] + [RightCensoredValue(x) for x in censored]


def start_values(
    family: FamilyId, exact: tuple[float, ...], censored: tuple[float, ...]
) -> tuple[float, ...]:
    everything = exact + censored
    mean_all = math.fsum(everything) / len(everything)
    mean_exact = math.fsum(exact) / len(exact)
    if family is FamilyId.GAMMA:
        return (2.0, mean_all / 2.0)
    if family is FamilyId.LOGNORMAL:
        logs = [math.log(x) for x in exact]
        center = math.fsum(logs) / len(logs)
        return (center, max(math.sqrt(math.fsum((v - center) ** 2 for v in logs) / len(logs)), 0.1))
    spread = math.sqrt(math.fsum((x - mean_exact) ** 2 for x in exact) / len(exact))
    return (mean_exact, max(spread, 0.1))


def newton_estimate(model: Model, start: tuple[float, ...]) -> tuple[float, ...] | None:
    """Damped Newton on the log-likelihood, in log coordinates for positive parameters.

    Returns ``None`` if the iteration meets a non-concave point or does not converge within 60
    steps; the caller then falls back on ``fit``.
    """

    theta = list(start)
    for _ in range(60):
        evaluation = model.evaluate(tuple(theta), precise=False)
        weight = [
            value if positive else 1.0
            for value, positive in zip(theta, model.positive, strict=True)
        ]
        gradient = [evaluation.gradient[i] * weight[i] for i in range(2)]
        hessian = [
            [
                evaluation.hessian[i][j] * weight[i] * weight[j]
                + (gradient[i] if i == j and model.positive[i] else 0.0)
                for j in range(2)
            ]
            for i in range(2)
        ]
        determinant = hessian[0][0] * hessian[1][1] - hessian[0][1] ** 2
        if not (hessian[0][0] < 0.0 and determinant > 0.0):
            return None
        step = [
            -(hessian[1][1] * gradient[0] - hessian[0][1] * gradient[1]) / determinant,
            -(hessian[0][0] * gradient[1] - hessian[0][1] * gradient[0]) / determinant,
        ]
        decrement = 0.5 * (gradient[0] * step[0] + gradient[1] * step[1])
        if decrement < 1e-12:
            return tuple(theta)
        scale = 1.0
        for _ in range(30):
            candidate = tuple(
                value * math.exp(scale * move) if positive else value + scale * move
                for value, move, positive in zip(theta, step, model.positive, strict=True)
            )
            try:
                improved = model.log_likelihood(candidate) >= evaluation.value - 1e-12
            except (ArithmeticError, ValueError):
                improved = False
            if improved:
                theta = list(candidate)
                break
            scale *= 0.5
        else:
            return None
    return None


def success_at(
    family: FamilyId,
    theta: tuple[float, ...],
    exact: tuple[float, ...],
    censored: tuple[float, ...],
) -> FitSuccess:
    """Build the success a fit would return at ``theta`` for this sample."""

    model = build_model(family, exact, censored)
    return SUCCESS_TYPES[family](
        *theta,
        model.log_likelihood(theta),
        len(exact) + len(censored),
        len(exact),
        len(censored),
        _evidence=FitEvidence(exact, censored),
    )


def estimate(
    family: FamilyId, exact: tuple[float, ...], censored: tuple[float, ...], *, use_newton: bool
) -> tuple[FitSuccess | None, bool]:
    """Return the fit success for the sample and whether ``fit`` had to be used as a fallback."""

    if use_newton:
        theta = newton_estimate(
            build_model(family, exact, censored), start_values(family, exact, censored)
        )
        if theta is not None:
            return success_at(family, theta, exact, censored), False
    result = fit(family, observations(family, exact, censored))
    return (result if isinstance(result, FitSuccess) else None), use_newton


class CoverageCalibration(unittest.TestCase):
    def run_family(self, family: FamilyId) -> None:
        names = {
            FamilyId.EXPONENTIAL: ("rate",),
            FamilyId.WEIBULL_MIN: ("shape", "scale"),
            FamilyId.LOGNORMAL: ("mu_log", "sigma_log"),
            FamilyId.GAMMA: ("shape", "scale"),
            FamilyId.NORMAL: ("mu", "sigma"),
            FamilyId.GUMBEL_RIGHT: ("location", "scale"),
        }[family]
        rng = np.random.default_rng(SEEDS[family])
        for censored, size, replicates in zip(
            (False, True), SAMPLE_SIZES, REPLICATES[family], strict=True
        ):
            started = time.perf_counter()
            hits = {method: [0] * len(names) for method in ("wald", "profile")}
            censoring_fractions: list[float] = []
            skipped = flagged = fallbacks = 0
            for _ in range(replicates):
                exact, limits = simulate(family, size, rng, censored=censored)
                censoring_fractions.append(len(limits) / size)
                result, fell_back = estimate(
                    family, exact, limits, use_newton=censored and family in NEWTON_FAMILIES
                )
                fallbacks += fell_back
                uncertainty = None if result is None else result.uncertainty()
                if not isinstance(uncertainty, FitUncertainty):
                    skipped += 1
                    continue
                for method in ("wald", "profile"):
                    details = uncertainty.interval_details(LEVEL, method)
                    for index, name in enumerate(names):
                        interval = details[name]
                        hits[method][index] += (
                            interval.lower <= TRUE_PARAMETERS[family][index] <= interval.upper
                        )
                        flagged += interval.lower_unbounded or interval.upper_unbounded
            elapsed = time.perf_counter() - started
            setting = f"n={size}, {'30% censored' if censored else 'uncensored'}"
            used = replicates - skipped
            print(
                f"\n[calibration] {family.value} ({setting}): {replicates} replicates, "
                f"{skipped} without uncertainty, {flagged} unbounded sides, "
                f"{fallbacks} fit fallbacks, censored fraction "
                f"{math.fsum(censoring_fractions) / replicates:.3f}, {elapsed:.1f} s"
            )
            profile_band = binomial_band(used, LEVEL)
            wald_band = (binomial_band(used, WALD_FLOOR)[0], profile_band[1])
            for method, band in (("profile", profile_band), ("wald", wald_band)):
                for index, name in enumerate(names):
                    coverage = hits[method][index] / used
                    print(
                        f"[calibration]   {method:7s} {name:10s} coverage {coverage:.4f}  "
                        f"band [{band[0]:.4f}, {band[1]:.4f}]"
                    )
                    with self.subTest(
                        family=family.value, setting=setting, method=method, parameter=name
                    ):
                        self.assertGreaterEqual(coverage, band[0])
                        self.assertLessEqual(coverage, band[1])
            with self.subTest(family=family.value, setting=setting, check="sample"):
                self.assertLessEqual(skipped, 0.01 * replicates)
                self.assertLessEqual(flagged, 0.01 * replicates)
                self.assertEqual(fallbacks, 0)
                if censored:
                    self.assertAlmostEqual(
                        math.fsum(censoring_fractions) / replicates, CENSORED_FRACTION, delta=0.015
                    )

    def test_exponential(self) -> None:
        self.run_family(FamilyId.EXPONENTIAL)

    def test_weibull(self) -> None:
        self.run_family(FamilyId.WEIBULL_MIN)

    def test_lognormal(self) -> None:
        self.run_family(FamilyId.LOGNORMAL)

    def test_gamma(self) -> None:
        self.run_family(FamilyId.GAMMA)

    def test_normal(self) -> None:
        self.run_family(FamilyId.NORMAL)

    def test_gumbel_right(self) -> None:
        self.run_family(FamilyId.GUMBEL_RIGHT)

    def test_newton_estimates_agree_with_fit(self) -> None:
        for family in NEWTON_FAMILIES:
            rng = np.random.default_rng(SEEDS[family])
            for _ in range(2):
                exact, limits = simulate(family, SAMPLE_SIZES[1], rng, censored=True)
                theta = newton_estimate(
                    build_model(family, exact, limits), start_values(family, exact, limits)
                )
                result = fit(family, observations(family, exact, limits))
                assert isinstance(result, FitSuccess)
                self.assertIsNotNone(theta)
                for value, reference in zip(theta, result.parameters.values(), strict=True):  # type: ignore[arg-type]
                    with self.subTest(family=family.value):
                        self.assertLessEqual(abs(value - reference), 1e-5 * abs(reference))

    def test_binomial_band_matches_known_values(self) -> None:
        # scipy.stats.binom: ppf(0.0005, n, p) is the lower count and isf(0.0005, n, p) the
        # upper count of the central 99.9% band (the same smallest/largest counts as above).
        for replicates, probability, lower, upper in (
            (400, 0.95, 364, 393),
            (400, 0.90, 339, 378),
            (800, 0.95, 738, 779),
            (2000, 0.95, 1867, 1931),
        ):
            with self.subTest(replicates=replicates, probability=probability):
                band = binomial_band(replicates, probability)
                self.assertEqual(band, (lower / replicates, upper / replicates))

    def test_the_wald_floor_is_looser_than_the_profile_band(self) -> None:
        self.assertLess(binomial_band(400, WALD_FLOOR)[0], binomial_band(400, LEVEL)[0])


if __name__ == "__main__":
    unittest.main()
