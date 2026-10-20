"""Exact probes of the safeguarded Newton search for the nuisance parameter of a profile.

Each likelihood here is a hand-made function of the free parameter with a known gradient and
curvature, chosen so that the arithmetic of one step is exact (steps of 3.0, 1.0, 0.25) and so that
every constant of the search (the gradient and the curvature on the log scale, the step cap, the
acceptance floor, the halving, the stopping test, the hard limits of the fallback) changes a
coordinate, an evaluation count or a returned point that the test pins. They need no data and
run in milliseconds; the likelihoods of ``test_uncertainty_probes`` all have a curvature of
exactly ``-1``, which hides the difference between a quotient and a product with it.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from unittest import mock

from tests.unit.test_uncertainty_probes import FakeModel
from veridist.families import uncertainty

Triple = tuple[float, float, float]


def search(
    model: FakeModel, start: float, cap: float = 10.0
) -> tuple[float, uncertainty.Evaluation | None]:
    positive = model.positive[1]
    return uncertainty._maximize_nuisance(
        model,
        lambda phi: (0.0, math.exp(phi) if positive else phi),
        1,
        start,
        cap,
    )


def quadratic(stiffness: float, optimum: float) -> Callable[[float], Triple]:
    """``-stiffness (b - optimum)^2 / 2``: its Newton step is ``optimum - b`` for any stiffness."""

    return lambda b: (-0.5 * stiffness * (b - optimum) ** 2, -stiffness * (b - optimum), -stiffness)


def table(
    rows: dict[float, Triple | None], other: Triple = (-1000.0, 0.0, -1.0)
) -> Callable[[float], Triple]:
    """The likelihood at the listed points (``None`` cannot be evaluated), ``other`` elsewhere."""

    def function(b: float) -> Triple:
        if b in rows:
            row = rows[b]
            if row is None:
                raise ArithmeticError("boom")
            return row
        return other

    return function


class NewtonStepTests(unittest.TestCase):
    def test_a_stiff_quadratic_is_solved_by_one_exact_step(self) -> None:
        model = FakeModel(quadratic(4.0, 3.0))
        phi, evaluation = search(model, 0.0)
        self.assertEqual(phi, 3.0)
        self.assertEqual(model.calls, 2)
        assert evaluation is not None
        self.assertEqual(evaluation.value, 0.0)

    def test_a_soft_quadratic_is_solved_by_one_exact_step(self) -> None:
        model = FakeModel(quadratic(0.25, 3.0))
        phi, _ = search(model, 0.0)
        self.assertEqual(phi, 3.0)
        self.assertEqual(model.calls, 2)

    def test_the_step_is_capped_whatever_the_stiffness(self) -> None:
        model = FakeModel(quadratic(4.0, 100.0))
        phi, _ = search(model, 0.0, cap=10.0)
        self.assertAlmostEqual(phi, 100.0, delta=1e-9)
        self.assertEqual(model.calls, 11)

    def test_the_step_on_the_log_scale_uses_the_gradient_and_curvature_of_the_log(self) -> None:
        # l(b) = -2 (log b - 1.5)^2: in phi = log b the gradient is b l' and the curvature
        # b^2 l'' + b l', which is -4 everywhere, so the step from phi = 0.5 is exactly 1.
        def function(b: float) -> Triple:
            offset = math.log(b) - 1.5
            return -2.0 * offset**2, -4.0 * offset / b, 4.0 * (offset - 1.0) / b**2

        model = FakeModel(function, positive=True)
        phi, evaluation = search(model, 0.5)
        self.assertAlmostEqual(phi, 1.5, delta=1e-12)
        self.assertEqual(model.calls, 2)
        assert evaluation is not None
        self.assertAlmostEqual(evaluation.value, 0.0, delta=1e-20)


class StoppingTests(unittest.TestCase):
    def setUp(self) -> None:
        # a decrement of a quarter makes ``g^2 <= -2 c D`` exact: with c = -2 it is ``g^2 <= 1``
        patch = mock.patch.object(uncertainty, "_NEWTON_DECREMENT", 0.25)
        patch.start()
        self.addCleanup(patch.stop)

    def test_a_gradient_exactly_at_the_decrement_stops_at_the_start(self) -> None:
        model = FakeModel(quadratic(2.0, 0.5))
        phi, evaluation = search(model, 0.0)
        self.assertEqual((phi, model.calls), (0.0, 1))
        assert evaluation is not None
        self.assertEqual(evaluation.gradient[1], 1.0)

    def test_a_gradient_above_the_decrement_takes_the_step(self) -> None:
        model = FakeModel(quadratic(2.0, 0.55))
        phi, _ = search(model, 0.0)
        self.assertAlmostEqual(phi, 0.55, delta=1e-12)
        self.assertEqual(model.calls, 2)

    def test_a_gradient_below_the_decrement_stops_at_the_start(self) -> None:
        model = FakeModel(quadratic(2.0, 0.25))
        phi, _ = search(model, 0.0)
        self.assertEqual((phi, model.calls), (0.0, 1))


class AcceptanceTests(unittest.TestCase):
    """The first step is from ``b = 0`` (value -1, gradient 1, curvature -1) to ``b = 1``."""

    START: Triple = (-1.0, 1.0, -1.0)
    LANDED: Triple = (0.0, 0.0, -1.0)

    def setUp(self) -> None:
        # the floor of a step is ``value - slack (1 + |value|)``: with a slack of a half it is -2
        patch = mock.patch.object(uncertainty, "_ACCEPT_SLACK", 0.5)
        patch.start()
        self.addCleanup(patch.stop)

    def reached(self, rows: dict[float, Triple | None]) -> float:
        model = FakeModel(table({0.0: self.START, **rows}))
        phi, _ = search(model, 0.0)
        return phi

    def test_a_step_above_the_floor_is_accepted(self) -> None:
        self.assertEqual(self.reached({1.0: (-1.5, 0.0, -1.0)}), 1.0)

    def test_a_step_exactly_at_the_floor_is_accepted(self) -> None:
        self.assertEqual(self.reached({1.0: (-2.0, 0.0, -1.0)}), 1.0)

    def test_a_step_below_the_floor_is_not_accepted(self) -> None:
        self.assertNotEqual(self.reached({1.0: (-2.2, 0.0, -1.0)}), 1.0)

    def test_a_step_that_cannot_be_evaluated_is_halved(self) -> None:
        self.assertEqual(self.reached({1.0: None, 0.5: (-1.2, 0.0, -1.0)}), 0.5)

    def test_the_step_is_halved_until_it_is_accepted(self) -> None:
        rows = {1.0: (-1000.0, 0.0, -1.0), 0.5: (-1000.0, 0.0, -1.0), 0.25: (-1.2, 0.0, -1.0)}
        self.assertEqual(self.reached(rows), 0.25)

    def test_a_step_that_is_never_accepted_hands_over_to_the_golden_section(self) -> None:
        model = FakeModel(quadratic(1.0, 3.0), broken_curvature=True)
        phi, _ = search(model, 0.0)
        self.assertAlmostEqual(phi, 3.0, delta=1e-5)
        # the start, then the golden-section search: two points, eighty steps and the point
        # it returns, and the evaluation of that point once more
        self.assertEqual(model.calls, 1 + (2 + 80 + 1) + 1)


class FallbackTests(unittest.TestCase):
    def test_a_curvature_that_is_not_negative_does_not_try_a_newton_step(self) -> None:
        for curvature in (0.0, 0.5):
            with self.subTest(curvature=curvature):
                model = FakeModel(lambda b, c=curvature: (-0.5 * (b - 3.0) ** 2, -(b - 3.0), c))
                phi, evaluation = search(model, 0.0)
                self.assertAlmostEqual(phi, 3.0, delta=1e-5)
                self.assertIsNotNone(evaluation)
                self.assertEqual(model.calls, 1 + (2 + 80 + 1) + 1)

    def test_a_flat_likelihood_does_not_divide_by_a_zero_curvature(self) -> None:
        model = FakeModel(lambda b: (b, 1.0, 0.0))
        phi, evaluation = search(model, 0.0, cap=1.0)
        self.assertAlmostEqual(phi, 100.0, delta=1e-3)
        self.assertIsNotNone(evaluation)

    def test_the_golden_section_avoids_a_region_that_cannot_be_evaluated(self) -> None:
        model = FakeModel(quadratic(1.0, 3.0), broken_curvature=True, fail_at=lambda b: b > 4.0)
        phi, evaluation = search(model, 0.0)
        self.assertAlmostEqual(phi, 3.0, delta=1e-5)
        self.assertIsNotNone(evaluation)

    def test_the_golden_section_stops_at_a_hundred_caps_from_the_start(self) -> None:
        for optimum, expected in ((1000.0, 50.0), (-1000.0, -50.0)):
            with self.subTest(optimum=optimum):
                model = FakeModel(quadratic(1.0, optimum), broken_curvature=True)
                phi, evaluation = search(model, 0.0, cap=0.5)
                self.assertAlmostEqual(phi, expected, delta=1e-3)
                self.assertIsNotNone(evaluation)

    def test_the_hard_limits_follow_the_start(self) -> None:
        model = FakeModel(quadratic(1.0, 1000.0), broken_curvature=True)
        phi, _ = search(model, 20.0, cap=0.5)
        self.assertAlmostEqual(phi, 70.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
