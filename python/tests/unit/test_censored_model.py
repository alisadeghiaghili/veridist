"""The likelihood model of censored and truncated samples, in the canonical parameters.

The chain rule from the optimizer's coordinates to ``(mu, sigma)``, ``(shape, scale)``,
``(location, scale)`` or the rate is checked three ways: against central differences of the
model's own log-likelihood, against the analytic models of the earlier releases on samples both
accept (gradient and Hessian, whose values may differ by a constant that does not depend on the
parameters), and through the quantities the uncertainty machinery derives from it.
"""

from __future__ import annotations

import math
import unittest

from tests.unit.test_censored_terms import SAMPLES
from veridist import (
    ExactLifetime,
    ExactValue,
    LeftCensoredLifetime,
    RightCensoredLifetime,
    RightCensoredValue,
)
from veridist.families._censored_model import CensoredModel
from veridist.families._censored_rows import build_rows
from veridist.families._models import ExponentialModel, build_model
from veridist.families.registry import FamilyId

CANONICAL = {
    FamilyId.EXPONENTIAL: (0.4,),
    FamilyId.GAMMA: (1.7, 1.3),
    FamilyId.WEIBULL_MIN: (1.6, 2.3),
    FamilyId.LOGNORMAL: (0.4, 0.9),
    FamilyId.NORMAL: (2.5, 1.4),
    FamilyId.GUMBEL_RIGHT: (2.2, 1.1),
}


class DerivativeTests(unittest.TestCase):
    def test_the_canonical_gradient_and_hessian_are_differences_of_the_likelihood(self) -> None:
        for family, theta in CANONICAL.items():
            with self.subTest(family=family):
                sample, _, _ = SAMPLES[family]
                model = CensoredModel(family, build_rows(family, sample()))
                evaluation = model.evaluate(theta)
                self.assertEqual(evaluation.value, model.log_likelihood(theta))
                step = 1e-5
                for i in range(len(theta)):
                    up, down = list(theta), list(theta)
                    up[i] += step
                    down[i] -= step
                    e_up, e_down = model.evaluate(tuple(up)), model.evaluate(tuple(down))
                    gradient = (e_up.value - e_down.value) / (2 * step)
                    tolerance = 2e-4 if family is FamilyId.GAMMA else 5e-6
                    self.assertAlmostEqual(
                        gradient, evaluation.gradient[i], delta=tolerance * (1 + abs(gradient))
                    )
                    for j in range(len(theta)):
                        difference = (e_up.gradient[j] - e_down.gradient[j]) / (2 * step)
                        self.assertAlmostEqual(
                            difference,
                            evaluation.hessian[j][i],
                            delta=tolerance * 5 * (1 + abs(difference)),
                        )
                if len(theta) == 2:
                    self.assertEqual(evaluation.hessian[0][1], evaluation.hessian[1][0])

    def test_plain_data_gives_the_derivatives_of_the_earlier_models(self) -> None:
        samples = {
            FamilyId.WEIBULL_MIN: (
                [
                    ExactLifetime(2.0),
                    ExactLifetime(3.5),
                    ExactLifetime(1.2),
                    RightCensoredLifetime(4.0),
                    RightCensoredLifetime(2.5),
                ]
            ),
            FamilyId.LOGNORMAL: (
                [
                    ExactLifetime(2.0),
                    ExactLifetime(3.5),
                    ExactLifetime(1.2),
                    RightCensoredLifetime(4.0),
                    RightCensoredLifetime(2.5),
                ]
            ),
            FamilyId.GAMMA: (
                [
                    ExactLifetime(2.0),
                    ExactLifetime(3.5),
                    ExactLifetime(1.2),
                    RightCensoredLifetime(4.0),
                    RightCensoredLifetime(2.5),
                ]
            ),
            FamilyId.NORMAL: (
                [
                    ExactValue(2.0),
                    ExactValue(3.5),
                    ExactValue(1.2),
                    RightCensoredValue(4.0),
                    RightCensoredValue(2.5),
                ]
            ),
            FamilyId.GUMBEL_RIGHT: (
                [
                    ExactValue(2.0),
                    ExactValue(3.5),
                    ExactValue(1.2),
                    RightCensoredValue(4.0),
                    RightCensoredValue(2.5),
                ]
            ),
        }
        for family, sample in samples.items():
            with self.subTest(family=family):
                exact = tuple(
                    float(o.time if hasattr(o, "time") else o.value)
                    for o in sample
                    if type(o).__name__.startswith("Exact")
                )
                censored = tuple(
                    float(o.time if hasattr(o, "time") else o.value)
                    for o in sample
                    if type(o).__name__.startswith("Right")
                )
                legacy = build_model(family, exact, censored)
                model = CensoredModel(family, build_rows(family, sample))
                theta = CANONICAL[family]
                ours, theirs = model.evaluate(theta), legacy.evaluate(theta)
                for a, b in zip(ours.gradient, theirs.gradient, strict=True):
                    self.assertAlmostEqual(
                        a,
                        b,
                        delta=1e-4 * (1 + abs(b))
                        if family is FamilyId.GAMMA
                        else 1e-10 * (1 + abs(b)),
                    )
                for row_a, row_b in zip(ours.hessian, theirs.hessian, strict=True):
                    for a, b in zip(row_a, row_b, strict=True):
                        self.assertAlmostEqual(
                            a, b, delta=(1e-3 if family is FamilyId.GAMMA else 1e-9) * (1 + abs(b))
                        )

    def test_the_exponential_with_its_block_equals_the_closed_form_model(self) -> None:
        rows = build_rows(
            FamilyId.EXPONENTIAL,
            [ExactLifetime(2.0, truncation_lower=1.0)],
            plain_count=3,
            plain_events=2,
            plain_total=7.0,
        )
        model = CensoredModel(FamilyId.EXPONENTIAL, rows)
        legacy = ExponentialModel(3, 8.0, 1)
        # events: 2 plain + 1 exact; total time: 7 + (2 - 1)
        for rate in (0.3, 0.7):
            ours, theirs = model.evaluate((rate,)), legacy.evaluate((rate,))
            self.assertAlmostEqual(ours.value, theirs.value, places=12)
            self.assertAlmostEqual(ours.gradient[0], theirs.gradient[0], places=12)
            self.assertAlmostEqual(ours.hessian[0][0], theirs.hessian[0][0], places=12)


class QuantityTests(unittest.TestCase):
    def test_the_names_and_flags_of_each_family(self) -> None:
        expected = {
            FamilyId.EXPONENTIAL: (("rate",), (True,), True),
            FamilyId.WEIBULL_MIN: (("shape", "scale"), (True, True), True),
            FamilyId.LOGNORMAL: (("mu_log", "sigma_log"), (False, True), True),
            FamilyId.GAMMA: (("shape", "scale"), (True, True), True),
            FamilyId.NORMAL: (("mu", "sigma"), (False, True), False),
            FamilyId.GUMBEL_RIGHT: (("location", "scale"), (False, True), False),
        }
        for family, (names, positive, support) in expected.items():
            model = CensoredModel(family, build_rows(family, SAMPLES[family][0]()))
            self.assertEqual(
                (model.names, model.positive, model.positive_support), (names, positive, support)
            )
            self.assertFalse(model.exact_interval_available)

    def test_the_means(self) -> None:
        rows = {f: build_rows(f, SAMPLES[f][0]()) for f in CANONICAL}
        mean = {f: CensoredModel(f, rows[f]).mean(CANONICAL[f]) for f in CANONICAL}
        self.assertAlmostEqual(mean[FamilyId.EXPONENTIAL], 2.5)
        self.assertAlmostEqual(mean[FamilyId.WEIBULL_MIN], 2.3 * math.gamma(1 + 1 / 1.6))
        self.assertAlmostEqual(mean[FamilyId.LOGNORMAL], math.exp(0.4 + 0.5 * 0.81))
        self.assertAlmostEqual(mean[FamilyId.GUMBEL_RIGHT], 2.2 + 0.5772156649015329 * 1.1)
        self.assertAlmostEqual(mean[FamilyId.GAMMA], 1.7 * 1.3)
        self.assertEqual(mean[FamilyId.NORMAL], 2.5)

    def test_the_quantile_and_the_survival_are_those_of_the_family(self) -> None:
        model = CensoredModel(
            FamilyId.EXPONENTIAL,
            build_rows(FamilyId.EXPONENTIAL, SAMPLES[FamilyId.EXPONENTIAL][0]()),
        )
        self.assertAlmostEqual(model.quantile((0.5,), 0.5), math.log(2.0) / 0.5)
        self.assertAlmostEqual(model.survival((0.5,), 2.0), math.exp(-1.0))

    def test_the_profile_reparameterizations_of_the_exponential_and_the_weibull(self) -> None:
        exponential = CensoredModel(
            FamilyId.EXPONENTIAL,
            build_rows(FamilyId.EXPONENTIAL, SAMPLES[FamilyId.EXPONENTIAL][0]()),
        )
        for quantity, argument, value, expected in (
            ("mean", 0.0, 4.0, 0.25),
            ("quantile", 0.5, 2.0, math.log(2.0) / 2.0),
            ("survival", 3.0, 0.25, -math.log(0.25) / 3.0),
        ):
            reparameterization = exponential.reparameterization(quantity, argument)
            assert reparameterization is not None
            self.assertIsNone(reparameterization.free_index)
            self.assertAlmostEqual(reparameterization.build(value, 0.0)[0], expected)
        weibull = CensoredModel(
            FamilyId.WEIBULL_MIN,
            build_rows(FamilyId.WEIBULL_MIN, SAMPLES[FamilyId.WEIBULL_MIN][0]()),
        )
        shape = 1.6
        for quantity, argument, value, scale in (
            ("mean", 0.0, 3.0, 3.0 / math.gamma(1 + 1 / shape)),
            ("quantile", 0.5, 3.0, 3.0 / math.log(2.0) ** (1 / shape)),
            ("survival", 3.0, 0.25, 3.0 / (-math.log(0.25)) ** (1 / shape)),
        ):
            reparameterization = weibull.reparameterization(quantity, argument)
            assert reparameterization is not None
            self.assertEqual(reparameterization.free_index, 0)
            built = reparameterization.build(value, shape)
            self.assertEqual(built[0], shape)
            self.assertAlmostEqual(built[1], scale)
        for family in (FamilyId.NORMAL, FamilyId.GAMMA, FamilyId.LOGNORMAL, FamilyId.GUMBEL_RIGHT):
            model = CensoredModel(family, build_rows(family, SAMPLES[family][0]()))
            self.assertIsNone(model.reparameterization("mean", 0.0))

    def test_a_likelihood_that_cannot_be_evaluated_is_an_arithmetic_error(self) -> None:
        rows = build_rows(FamilyId.GAMMA, [ExactLifetime(1.0), LeftCensoredLifetime(2.0)])
        model = CensoredModel(FamilyId.GAMMA, rows)
        with self.assertRaises(ArithmeticError):
            model.evaluate((1e5, 1.0))
        with self.assertRaises(ArithmeticError):
            model.log_likelihood((1e5, 1.0))


if __name__ == "__main__":
    unittest.main()
