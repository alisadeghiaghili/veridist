"""Probes of the uncertainty internals: matrix inversion, root finding, Newton search, models."""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable

from veridist.families import _models, uncertainty
from veridist.families._evidence import FitEvidence
from veridist.families._models import (
    Evaluation,
    ExponentialModel,
    GammaModel,
    GaussianModel,
    GumbelModel,
    Model,
    WeibullModel,
    build_model,
    location_scale_evaluation,
)
from veridist.families.gumbel import _log_survival
from veridist.families.registry import FamilyId
from veridist.statistics.distributions import ppf, sf

Reason = uncertainty.UncertaintyUnavailableReason


class InvertInformationProbes(unittest.TestCase):
    def test_one_parameter_information(self) -> None:
        self.assertEqual(uncertainty._invert_information(((4.0,),)), ((0.25,),))
        for bad in (0.0, -2.0):
            result = uncertainty._invert_information(((bad,),))
            self.assertIsInstance(result, uncertainty.UncertaintyUnavailable)
            self.assertIs(result.reason, Reason.NOT_POSITIVE_DEFINITE)  # type: ignore[union-attr]

    def test_two_parameter_inverse(self) -> None:
        inverse = uncertainty._invert_information(((2.0, 0.5), (0.5, 1.0)))
        determinant = 2.0 * 1.0 - 0.5 * 0.5
        expected = (
            (1.0 / determinant, -0.5 / determinant),
            (-0.5 / determinant, 2.0 / determinant),
        )
        assert not isinstance(inverse, uncertainty.UncertaintyUnavailable)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(inverse[i][j], expected[i][j], delta=1e-15)
        self.assertEqual(inverse[0][1], inverse[1][0])

    def test_the_inverse_does_not_depend_on_the_scale_of_the_parameters(self) -> None:
        inverse = uncertainty._invert_information(((2e12, 0.5e6), (0.5e6, 1.0)))
        assert not isinstance(inverse, uncertainty.UncertaintyUnavailable)
        determinant = 2e12 - 0.25e12
        self.assertAlmostEqual(inverse[0][0] * determinant / 1.0, 1.0, delta=1e-12)
        self.assertAlmostEqual(inverse[1][1] * determinant / 2e12, 1.0, delta=1e-12)
        self.assertAlmostEqual(inverse[0][1] * determinant / -0.5e6, 1.0, delta=1e-12)

    def test_asymmetric_input_is_symmetrized(self) -> None:
        inverse = uncertainty._invert_information(((2.0, 0.4), (0.6, 1.0)))
        assert not isinstance(inverse, uncertainty.UncertaintyUnavailable)
        self.assertEqual(inverse[0][1], inverse[1][0])
        self.assertEqual(inverse, uncertainty._invert_information(((2.0, 0.5), (0.5, 1.0))))

    def test_unusable_information_is_classified(self) -> None:
        cases = (
            (((1.0, 1.0), (1.0, 1.0)), Reason.SINGULAR_INFORMATION),
            (((1.0, 2.0), (2.0, 1.0)), Reason.NOT_POSITIVE_DEFINITE),
            (((-1.0, 0.0), (0.0, 1.0)), Reason.NOT_POSITIVE_DEFINITE),
            (((1.0, 0.0), (0.0, 0.0)), Reason.NOT_POSITIVE_DEFINITE),
            (((1.0, math.nan), (math.nan, 1.0)), Reason.NOT_COMPUTABLE),
            (((math.inf, 0.0), (0.0, 1.0)), Reason.NOT_COMPUTABLE),
        )
        for information, reason in cases:
            with self.subTest(information=information):
                result = uncertainty._invert_information(information)
                self.assertIsInstance(result, uncertainty.UncertaintyUnavailable)
                self.assertIs(result.reason, reason)  # type: ignore[union-attr]
                self.assertTrue(result.detail)  # type: ignore[union-attr]

    def test_the_singularity_threshold_sits_between_1e_9_and_1e_13(self) -> None:
        def with_margin(margin: float) -> object:
            rho = math.sqrt(1.0 - margin)
            return uncertainty._invert_information(((1.0, rho), (rho, 1.0)))

        self.assertNotIsInstance(with_margin(1e-9), uncertainty.UncertaintyUnavailable)
        borderline = with_margin(1e-13)
        self.assertIsInstance(borderline, uncertainty.UncertaintyUnavailable)
        self.assertIs(borderline.reason, Reason.SINGULAR_INFORMATION)  # type: ignore[attr-defined]


class SmallNumericsProbes(unittest.TestCase):
    def test_chi_square_one_quantile(self) -> None:
        # scipy.stats.chi2.ppf(0.95, 1) and ppf(0.99, 1).
        self.assertAlmostEqual(uncertainty._chi_square_1(0.95), 3.841458820694124, delta=1e-12)
        self.assertAlmostEqual(uncertainty._chi_square_1(0.99), 6.634896601021214, delta=1e-12)

    def test_normal_critical_value(self) -> None:
        self.assertAlmostEqual(uncertainty._normal_quantile(0.95), 1.959963984540054, delta=1e-13)
        self.assertAlmostEqual(uncertainty._normal_quantile(0.5), 0.6744897501960817, delta=1e-13)

    def test_expit_and_logit(self) -> None:
        self.assertEqual(uncertainty._expit(0.0), 0.5)
        self.assertEqual(uncertainty._expit(800.0), 1.0)
        self.assertEqual(uncertainty._expit(-800.0), 0.0)
        self.assertAlmostEqual(uncertainty._expit(2.0), 1.0 / (1.0 + math.exp(-2.0)), delta=1e-16)
        self.assertAlmostEqual(uncertainty._expit(-2.0), 1.0 / (1.0 + math.exp(2.0)), delta=1e-16)
        for value in (-30.0, -2.5, 0.0, 3.2, 10.0):
            self.assertAlmostEqual(uncertainty._logit(uncertainty._expit(value)), value, delta=1e-9)

    def test_richardson_gradient_is_exact_to_fourth_order(self) -> None:
        def function(theta: tuple[float, ...]) -> float:
            return theta[0] ** 2 * math.exp(theta[1])

        gradient = uncertainty._gradient(function, (1.5, 0.4), (0.01, 0.01))
        self.assertAlmostEqual(gradient[0], 2.0 * 1.5 * math.exp(0.4), delta=1e-8)
        self.assertAlmostEqual(gradient[1], 1.5**2 * math.exp(0.4), delta=1e-8)

    def test_quadratic_form(self) -> None:
        self.assertEqual(
            uncertainty._quadratic_form((1.0, 2.0), ((4.0, 1.0), (1.0, 3.0))), 4.0 + 2 * 2.0 + 12.0
        )

    def test_wald_bounds_on_each_scale(self) -> None:
        z = 2.0
        lower, upper = uncertainty._wald_bounds("identity", 5.0, 0.5, z)
        self.assertEqual((lower, upper), (4.0, 6.0))
        lower, upper = uncertainty._wald_bounds("log", 5.0, 0.5, z)
        self.assertAlmostEqual(lower, 5.0 * math.exp(-0.2), delta=1e-15)
        self.assertAlmostEqual(upper, 5.0 * math.exp(0.2), delta=1e-15)
        lower, upper = uncertainty._wald_bounds("logit", 0.25, 0.05, z)
        half = 2.0 * 0.05 / (0.25 * 0.75)
        self.assertAlmostEqual(lower, 1.0 / (1.0 + 3.0 * math.exp(half)), delta=1e-14)
        self.assertAlmostEqual(upper, 1.0 / (1.0 + 3.0 * math.exp(-half)), delta=1e-14)
        lower, upper = uncertainty._wald_bounds("log", 5.0, 1e9, z)
        self.assertEqual((lower, upper), (0.0, math.inf))

    def test_validators(self) -> None:
        self.assertEqual(uncertainty._validated_level(0.9), 0.9)
        self.assertEqual(uncertainty._validated_method("profile"), "profile")
        self.assertEqual(uncertainty._real(3, "x"), 3.0)
        with self.assertRaises(TypeError) as caught:
            uncertainty._real("3", "time")
        self.assertIn("time", str(caught.exception))


class QuadraticCurve(uncertainty._Curve):
    """Profile ``-(xi - c)^2 / (2 s^2)``: the deviance reaches ``t`` at ``c +- s sqrt(t)``."""

    def __init__(
        self,
        scale: float,
        *,
        step: float,
        hard: float,
        center: float = 0.0,
        slope: Callable[[float, float], float | None] | None = None,
    ) -> None:
        self.scale = scale
        self.center = center
        self.step = step
        self.hard = hard
        self._slope = slope
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def value(self, xi: float) -> float:
        return xi

    def at(self, xi: float) -> tuple[float, float | None]:
        profile = -0.5 * ((xi - self.center) / self.scale) ** 2
        exact = -(xi - self.center) / self.scale**2
        return profile, exact if self._slope is None else self._slope(xi, exact)


class SolveSideProbes(unittest.TestCase):
    TARGET = 3.841458820694124

    def root(self, curve: uncertainty._Curve, direction: float) -> float | None:
        return uncertainty._solve_side(curve, direction, self.TARGET, 0.0)

    def test_newton_path_finds_both_roots(self) -> None:
        curve = QuadraticCurve(2.0, step=1.0, hard=100.0, center=1.0)
        lower, upper = self.root(curve, -1.0), self.root(curve, 1.0)
        half = 2.0 * math.sqrt(self.TARGET)
        self.assertAlmostEqual(lower, 1.0 - half, delta=1e-9)  # type: ignore[arg-type]
        self.assertAlmostEqual(upper, 1.0 + half, delta=1e-9)  # type: ignore[arg-type]
        self.assertEqual(curve.resets, 2)

    def test_false_position_without_a_derivative(self) -> None:
        curve = QuadraticCurve(2.0, step=1.0, hard=100.0, slope=lambda xi, exact: None)
        half = 2.0 * math.sqrt(self.TARGET)
        self.assertAlmostEqual(self.root(curve, 1.0), half, delta=1e-8)  # type: ignore[arg-type]
        self.assertAlmostEqual(self.root(curve, -1.0), -half, delta=1e-8)  # type: ignore[arg-type]

    def test_the_search_expands_from_a_step_that_is_too_small(self) -> None:
        curve = QuadraticCurve(2.0, step=0.01, hard=100.0)
        self.assertAlmostEqual(
            self.root(curve, 1.0),
            2.0 * math.sqrt(self.TARGET),
            delta=1e-9,  # type: ignore[arg-type]
        )

    def test_a_curve_that_never_crosses_is_unbounded(self) -> None:
        flat = QuadraticCurve(1e9, step=1.0, hard=64.0)
        self.assertIsNone(self.root(flat, 1.0))
        self.assertIsNone(self.root(flat, -1.0))

    def test_a_crossing_just_inside_the_hard_limit_is_found(self) -> None:
        curve = QuadraticCurve(1.0, step=1.0, hard=2.5)
        # sqrt(3.84) = 1.96 < 2.5: found; with hard = 1.9 it is not.
        self.assertIsNotNone(self.root(curve, 1.0))
        self.assertIsNone(self.root(QuadraticCurve(1.0, step=1.0, hard=1.9), 1.0))

    def test_a_misleading_derivative_falls_back_to_bisection(self) -> None:
        for wrong in (1e-9, 0.0, -5.0):
            curve = QuadraticCurve(2.0, step=1.0, hard=100.0, slope=lambda xi, exact, w=wrong: w)
            with self.subTest(slope=wrong):
                self.assertAlmostEqual(
                    self.root(curve, 1.0),
                    2.0 * math.sqrt(self.TARGET),
                    delta=1e-9,  # type: ignore[arg-type]
                )

    def jump_search(self, step: float) -> tuple[float | None, int]:
        """Search a profile that steps at 0.3, so the deviance never equals the target."""

        calls = 0
        target = self.TARGET

        class Jump(uncertainty._Curve):
            center = 0.0
            hard = 1e20

            def value(self, xi: float) -> float:
                return xi

            def at(self, xi: float) -> tuple[float, float | None]:
                nonlocal calls
                calls += 1
                return (0.0 if xi < 0.3 else -target), None

        jump = Jump()
        jump.step = step
        return uncertainty._solve_side(jump, 1.0, target, 0.0), calls

    def test_a_search_that_cannot_hit_the_target_stops_when_the_bracket_is_tight(self) -> None:
        result, calls = self.jump_search(1.0)
        self.assertAlmostEqual(result, 0.3, delta=1e-11)  # type: ignore[arg-type]
        self.assertLess(calls, uncertainty._MAX_ROOT_ITERATIONS)

    def test_the_iteration_cap_bounds_a_search_that_cannot_converge(self) -> None:
        # A bracket of width 1e19 needs more than 100 bisections to reach 1e-13.
        result, calls = self.jump_search(1e19)
        self.assertAlmostEqual(result, 0.3, delta=1e-9)  # type: ignore[arg-type]
        # One evaluation brackets the root, then exactly the capped number of iterations.
        self.assertEqual(calls, 1 + uncertainty._MAX_ROOT_ITERATIONS)

    def test_a_tiny_step_that_cannot_reach_the_hard_limit_in_time_is_unbounded(self) -> None:
        curve = QuadraticCurve(1e30, step=1e-9, hard=1e300)
        self.assertIsNone(self.root(curve, 1.0))

    def test_the_abstract_curve_has_no_implementation(self) -> None:
        curve = uncertainty._Curve()
        with self.assertRaises(NotImplementedError):
            curve.at(0.0)
        with self.assertRaises(NotImplementedError):
            curve.value(0.0)
        self.assertIsNone(curve.reset())


class FakeModel(Model):
    """A two-parameter model whose likelihood in ``b`` is chosen per test (``a`` is inert)."""

    family = FamilyId.NORMAL
    names = ("a", "b")
    positive = (False, False)
    positive_support = False

    def __init__(
        self,
        function: Callable[[float], tuple[float, float, float]],
        *,
        positive: bool = False,
        broken_curvature: bool = False,
        fail_at: Callable[[float], bool] = lambda value: False,
    ) -> None:
        self.function = function
        self.positive = (False, positive)
        self.broken_curvature = broken_curvature
        self.fail_at = fail_at
        self.calls = 0

    def evaluate(self, theta: tuple[float, ...], *, precise: bool = True) -> Evaluation:
        self.calls += 1
        if self.fail_at(theta[1]):
            raise ArithmeticError("boom")
        value, slope, curvature = self.function(theta[1])
        if self.broken_curvature:
            curvature = 1.0
        return Evaluation(value, (0.0, slope), ((0.0, 0.0), (0.0, curvature)))


def _concave(optimum: float) -> Callable[[float], tuple[float, float, float]]:
    return lambda b: (-0.5 * (b - optimum) ** 2, -(b - optimum), -1.0)


class MaximizeNuisanceProbes(unittest.TestCase):
    def search(
        self, model: FakeModel, start: float, cap: float = 10.0
    ) -> tuple[float, Evaluation | None]:
        positive = model.positive[1]
        return uncertainty._maximize_nuisance(
            model,
            lambda phi: (0.0, math.exp(phi) if positive else phi),
            1,
            start,
            cap,
        )

    def test_newton_converges_on_a_quadratic_in_two_evaluations(self) -> None:
        model = FakeModel(_concave(3.0))
        phi, evaluation = self.search(model, 0.0)
        self.assertAlmostEqual(phi, 3.0, delta=1e-12)
        self.assertEqual(
            model.calls, 2
        )  # the start and the Newton step, which lands on the optimum
        assert evaluation is not None
        self.assertAlmostEqual(evaluation.value, 0.0, delta=1e-20)

    def test_a_start_at_the_optimum_needs_one_evaluation(self) -> None:
        model = FakeModel(_concave(3.0))
        phi, _ = self.search(model, 3.0)
        self.assertEqual((phi, model.calls), (3.0, 1))

    def test_a_positive_nuisance_is_searched_on_the_log_scale(self) -> None:
        # l(b) = -(log b - 1)^2 / 2: dl/db = -(log b - 1) / b, d2l/db2 = (log b - 2) / b^2.
        def function(b: float) -> tuple[float, float, float]:
            log_b = math.log(b)
            return -0.5 * (log_b - 1.0) ** 2, -(log_b - 1.0) / b, (log_b - 2.0) / b**2

        model = FakeModel(function, positive=True)
        phi, evaluation = self.search(model, 0.0)
        self.assertAlmostEqual(phi, 1.0, delta=1e-6)
        assert evaluation is not None
        self.assertAlmostEqual(evaluation.value, 0.0, delta=1e-10)

    def test_the_step_is_capped(self) -> None:
        model = FakeModel(_concave(100.0))
        phi, _ = self.search(model, 0.0, cap=10.0)
        self.assertAlmostEqual(phi, 100.0, delta=1e-9)
        self.assertEqual(model.calls, 11)  # the start and ten capped steps

    def test_an_overshooting_step_is_halved(self) -> None:
        def function(b: float) -> tuple[float, float, float]:
            root = math.sqrt(1.0 + b * b)
            return -root, -b / root, -1.0 / root**3

        model = FakeModel(function)
        phi, _ = self.search(model, 2.0)
        self.assertAlmostEqual(phi, 0.0, delta=1e-6)

    def test_a_convex_curvature_hands_over_to_golden_section(self) -> None:
        model = FakeModel(_concave(3.0), broken_curvature=True)
        phi, evaluation = self.search(model, 0.0)
        self.assertAlmostEqual(phi, 3.0, delta=1e-5)
        assert evaluation is not None
        self.assertAlmostEqual(evaluation.value, 0.0, delta=1e-9)

    def test_a_gradient_that_points_downhill_exhausts_the_backtracking(self) -> None:
        def function(b: float) -> tuple[float, float, float]:
            return -0.5 * (b - 3.0) ** 2, (b - 3.0), -1.0

        model = FakeModel(function)
        phi, _ = self.search(model, 0.0)
        self.assertAlmostEqual(phi, 3.0, delta=1e-5)

    def test_a_failing_start_still_finds_the_maximum(self) -> None:
        model = FakeModel(_concave(3.0), fail_at=lambda b: b == 0.0)
        phi, evaluation = self.search(model, 0.0)
        self.assertAlmostEqual(phi, 3.0, delta=1e-5)
        self.assertIsNotNone(evaluation)

    def test_an_unevaluable_likelihood_returns_no_evaluation(self) -> None:
        model = FakeModel(_concave(3.0), fail_at=lambda b: True)
        phi, evaluation = self.search(model, 0.0)
        self.assertIsNone(evaluation)
        self.assertTrue(math.isfinite(phi))

    def test_a_search_that_makes_slow_progress_is_handed_to_golden_section(self) -> None:
        model = FakeModel(_concave(1000.0))
        phi, evaluation = self.search(model, 0.0, cap=0.5)
        # 40 capped Newton steps reach 20; the golden-section window then spans +-50 around 0.
        self.assertGreater(model.calls, uncertainty._NEWTON_STEPS)
        self.assertTrue(math.isfinite(phi))
        self.assertIsNotNone(evaluation)

    def test_a_parameter_curve_whose_likelihood_cannot_be_evaluated_has_no_profile(self) -> None:
        model = FakeModel(_concave(3.0), fail_at=lambda b: True)
        covariance = ((1.0, 0.0), (0.0, 1.0))
        curve = uncertainty._ParameterCurve(model, (1.0, 2.0), covariance, 0, 1.96)
        self.assertEqual(curve.at(0.5), (-math.inf, None))
        self.assertEqual(curve.warm, curve.start)

    def test_a_quantity_curve_treats_an_impossible_parameter_vector_as_zero_likelihood(
        self,
    ) -> None:
        model = WeibullModel((1.0, 2.0, 4.0), ())
        reparameterization = model.reparameterization("quantile", 0.1)
        assert reparameterization is not None
        curve = uncertainty._QuantityCurve(model, (1.5, 3.0), reparameterization, "log", 2.0, 0.3)
        self.assertEqual(curve._log_likelihood(-1.0, 1.5), -math.inf)
        self.assertEqual(curve._log_likelihood(2.0, 1e9), -math.inf)
        self.assertTrue(math.isfinite(curve._log_likelihood(2.0, 1.5)))

    def test_a_non_finite_value_counts_as_unevaluable(self) -> None:
        model = FakeModel(lambda b: (math.inf, 0.0, -1.0))
        self.assertIsNone(uncertainty._safe_evaluate(model, (0.0, 0.0)))
        self.assertIsNone(
            uncertainty._safe_evaluate(FakeModel(_concave(0.0), fail_at=lambda b: True), (0.0, 0.0))
        )
        self.assertIsNotNone(uncertainty._safe_evaluate(FakeModel(_concave(0.0)), (0.0, 0.0)))


class EvidenceProbes(unittest.TestCase):
    def test_uncertainty_is_cached_per_parameter_vector(self) -> None:
        evidence = FitEvidence((1.0, 2.0, 4.0, 3.0), (5.0,))
        first = evidence.uncertainty(FamilyId.WEIBULL_MIN, {"shape": 1.5, "scale": 3.0})
        again = evidence.uncertainty(FamilyId.WEIBULL_MIN, {"shape": 1.5, "scale": 3.0})
        other = evidence.uncertainty(FamilyId.WEIBULL_MIN, {"shape": 1.6, "scale": 3.0})
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual(len(evidence._cache), 2)

    def test_the_family_is_part_of_the_cache_key(self) -> None:
        evidence = FitEvidence((1.0, 2.0, 4.0, 3.0), ())
        weibull = evidence.uncertainty(FamilyId.WEIBULL_MIN, {"shape": 1.5, "scale": 3.0})
        gamma = evidence.uncertainty(FamilyId.GAMMA, {"shape": 1.5, "scale": 3.0})
        self.assertNotEqual(weibull, gamma)

    def test_no_observations_means_no_data(self) -> None:
        result = FitEvidence().uncertainty(FamilyId.NORMAL, {"mu": 0.0, "sigma": 1.0})
        self.assertIs(result.reason, Reason.NO_DATA)  # type: ignore[union-attr]

    def test_unpreparable_observations_are_not_computable(self) -> None:
        result = FitEvidence((0.0, 1.0), ()).uncertainty(
            FamilyId.LOGNORMAL, {"mu_log": 0.0, "sigma_log": 1.0}
        )
        self.assertIs(result.reason, Reason.NOT_COMPUTABLE)  # type: ignore[union-attr]


class ModelProbes(unittest.TestCase):
    def test_the_base_model_has_no_implementation(self) -> None:
        base = Model()
        with self.assertRaises(NotImplementedError):
            base.evaluate((1.0,))
        with self.assertRaises(NotImplementedError):
            base.mean((1.0,))
        self.assertIsNone(base.reparameterization("mean", 0.0))
        self.assertFalse(base.exact_interval_available)

    def test_default_log_likelihood_is_the_evaluated_value(self) -> None:
        model = FakeModel(_concave(3.0))
        self.assertEqual(Model.log_likelihood(model, (0.0, 1.0)), -2.0)

    def test_grouping_collapses_repeats_in_first_seen_order(self) -> None:
        self.assertEqual(
            _models._grouped([3.0, 1.0, 3.0, 3.0, 1.0, 2.0]), ((3.0, 3), (1.0, 2), (2.0, 1))
        )

    def test_exponential_model(self) -> None:
        model = ExponentialModel(4, 10.0, 0)
        evaluation = model.evaluate((0.5,))
        self.assertAlmostEqual(evaluation.value, 4 * math.log(0.5) - 5.0, delta=1e-15)
        self.assertEqual(evaluation.gradient, (8.0 - 10.0,))
        self.assertEqual(evaluation.hessian, ((-16.0,),))
        self.assertTrue(model.exact_interval_available)
        self.assertFalse(ExponentialModel(4, 10.0, 1).exact_interval_available)
        self.assertEqual(model.mean((0.5,)), 2.0)

    def test_reparameterizations_round_trip_through_the_distribution(self) -> None:
        cases = (
            (ExponentialModel(4, 10.0, 0), (0.4,), None),
            (WeibullModel((1.0, 2.0, 4.0), (3.0,)), (1.7, 3.5), 0),
        )
        for model, theta, free_index in cases:
            for kind, argument, value in (
                ("mean", 0.0, 2.2),
                ("quantile", 0.1, 1.3),
                ("survival", 2.0, 0.6),
            ):
                with self.subTest(family=model.family, kind=kind):
                    reparameterization = model.reparameterization(kind, argument)
                    assert reparameterization is not None
                    self.assertEqual(reparameterization.free_index, free_index)
                    built = reparameterization.build(value, 1.9)
                    if free_index is not None:
                        self.assertEqual(built[free_index], 1.9)
                    recovered = {
                        "mean": lambda: model.mean(built),
                        "quantile": lambda: model.quantile(built, argument),
                        "survival": lambda: model.survival(built, argument),
                    }[kind]()
                    self.assertAlmostEqual(recovered, value, delta=1e-12)

    def test_models_without_a_reparameterization(self) -> None:
        for model in (
            GammaModel((1.0, 2.0, 3.0), ()),
            GaussianModel(FamilyId.NORMAL, ("mu", "sigma"), (1.0, 2.0, 3.0), (), 0.0),
            GumbelModel((1.0, 2.0, 3.0), ()),
        ):
            self.assertIsNone(model.reparameterization("mean", 0.0))

    def test_quantile_and_survival_use_the_public_operations(self) -> None:
        model = WeibullModel((1.0, 2.0, 4.0), ())
        self.assertEqual(
            model.quantile((1.5, 3.0), 0.2), ppf("weibull_min", 0.2, shape=1.5, scale=3.0)
        )
        self.assertEqual(
            model.survival((1.5, 3.0), 2.0), sf("weibull_min", 2.0, shape=1.5, scale=3.0)
        )

    def test_gaussian_means(self) -> None:
        normal = GaussianModel(FamilyId.NORMAL, ("mu", "sigma"), (1.0, 2.0, 3.0), (), 0.0)
        lognormal = GaussianModel(FamilyId.LOGNORMAL, ("mu_log", "sigma_log"), (1.0, 2.0), (), 0.0)
        self.assertEqual(normal.mean((2.0, 0.5)), 2.0)
        self.assertEqual(lognormal.mean((2.0, 0.5)), math.exp(2.0 + 0.125))
        self.assertTrue(lognormal.positive_support)
        self.assertFalse(normal.positive_support)

    def test_gaussian_exact_part_matches_the_generic_location_scale_assembly(self) -> None:
        data = (1.0, 2.5, 4.0, 3.0)
        mu, sigma = 2.2, 1.3
        model = GaussianModel(FamilyId.NORMAL, ("mu", "sigma"), data, (), 0.0)
        sums = _models._Sums()
        for y in data:
            z = (y - mu) / sigma
            sums.add(1.0, z, -z, -1.0)
        value = math.fsum(
            -math.log(sigma) - 0.5 * math.log(2 * math.pi) - 0.5 * ((y - mu) / sigma) ** 2
            for y in data
        )
        generic = location_scale_evaluation(value, len(data), sigma, sums)
        evaluated = model.evaluate((mu, sigma))
        self.assertAlmostEqual(evaluated.value, generic.value, delta=1e-13)
        for i in range(2):
            self.assertAlmostEqual(evaluated.gradient[i], generic.gradient[i], delta=1e-12)
            for j in range(2):
                self.assertAlmostEqual(evaluated.hessian[i][j], generic.hessian[i][j], delta=1e-12)

    def test_gumbel_censored_derivatives(self) -> None:
        def log_survival(z: float) -> float:
            return _log_survival(z)

        for z in (-5.0, -1.0, 0.0, 0.7, 3.0, 12.0):
            value, first, second = _models._gumbel_censored(z)
            h = 1e-3
            slope = (
                8 * (log_survival(z + h) - log_survival(z - h))
                - (log_survival(z + 2 * h) - log_survival(z - 2 * h))
            ) / (12 * h)
            curvature = (
                -log_survival(z + 2 * h)
                + 16 * log_survival(z + h)
                - 30 * log_survival(z)
                + 16 * log_survival(z - h)
                - log_survival(z - 2 * h)
            ) / (12 * h * h)
            with self.subTest(z=z):
                self.assertEqual(value, log_survival(z))
                self.assertAlmostEqual(first, slope, delta=1e-8)
                self.assertAlmostEqual(second, curvature, delta=1e-6)

    def test_gumbel_censored_limits(self) -> None:
        self.assertEqual(_models._gumbel_censored(-800.0), (0.0, 0.0, 0.0))
        value, first, second = _models._gumbel_censored(-20.0)
        self.assertEqual((first, second), (0.0, 0.0))
        self.assertEqual(value, _log_survival(-20.0))
        value, first, second = _models._gumbel_censored(800.0)
        self.assertEqual((first, second), (-1.0, 0.0))
        self.assertEqual(value, _log_survival(800.0))

    def test_shape_derivative_stencils(self) -> None:
        five = [math.exp(0.03 * k) for k in (-2, -1, 0, 1, 2)]
        three = [math.exp(0.002 * k) for k in (-1, 0, 1)]
        first, second = _models._shape_derivatives(five, 0.03)
        self.assertAlmostEqual(first, 1.0, delta=1e-7)
        self.assertAlmostEqual(second, 1.0, delta=1e-7)
        first, second = _models._shape_derivatives(three, 0.002)
        self.assertAlmostEqual(first, 1.0, delta=1e-5)
        self.assertAlmostEqual(second, 1.0, delta=1e-5)

    def test_build_model_selects_the_family_model(self) -> None:
        exact, censored = (1.0, 2.0, 4.0), (3.0,)
        expected = {
            FamilyId.WEIBULL_MIN: WeibullModel,
            FamilyId.GAMMA: GammaModel,
            FamilyId.GUMBEL_RIGHT: GumbelModel,
            FamilyId.NORMAL: GaussianModel,
            FamilyId.LOGNORMAL: GaussianModel,
        }
        for family, kind in expected.items():
            model = build_model(family, exact, censored)
            self.assertIsInstance(model, kind)
            self.assertEqual(model.family, family)

    def test_lognormal_model_works_on_logs_with_the_jacobian(self) -> None:
        exact = (1.5, 2.0, 4.0)
        model = build_model(FamilyId.LOGNORMAL, exact, ())
        self.assertEqual(model.names, ("mu_log", "sigma_log"))
        mu, sigma = 0.6, 0.5
        expected = math.fsum(
            -math.log(t)
            - math.log(sigma)
            - 0.5 * math.log(2 * math.pi)
            - (math.log(t) - mu) ** 2 / (2 * sigma**2)
            for t in exact
        )
        self.assertAlmostEqual(model.log_likelihood((mu, sigma)), expected, delta=1e-13)

    def test_gamma_shape_step_modes_agree_to_their_accuracy(self) -> None:
        model = GammaModel((2.0, 3.0, 5.5, 4.0), (6.0, 6.0, 7.5))
        precise = model.evaluate((2.2, 2.1), precise=True)
        fast = model.evaluate((2.2, 2.1), precise=False)
        self.assertEqual(precise.value, fast.value)
        self.assertEqual(precise.value, model.log_likelihood((2.2, 2.1)))
        for i in range(2):
            self.assertAlmostEqual(precise.gradient[i], fast.gradient[i], delta=1e-4)
            for j in range(2):
                self.assertAlmostEqual(precise.hessian[i][j], fast.hessian[i][j], delta=1e-4)

    def test_weibull_mean(self) -> None:
        model = WeibullModel((1.0, 2.0), ())
        self.assertAlmostEqual(model.mean((2.0, 3.0)), 3.0 * math.gamma(1.5), delta=1e-15)


if __name__ == "__main__":
    unittest.main()
