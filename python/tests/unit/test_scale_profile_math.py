"""The arithmetic of profile-likelihood intervals: nodes, polynomial, deviance, extremes."""

from __future__ import annotations

import dataclasses
import math
import unittest
from typing import Any
from unittest import mock

import mpmath

from veridist.scale import _profile_math as pm
from veridist.scale._profile_math import (
    Boundary,
    LikelihoodForm,
    Surrogate,
    boundary_at,
    boundary_offsets,
    extremum,
    lobatto_logs,
    noise_allowance,
    root_between,
    surrogate_deviance,
)

mpmath.mp.dps = 40


class ConstantsTests(unittest.TestCase):
    def test_every_loop_is_bounded_by_a_pinned_constant(self) -> None:
        self.assertEqual(pm.SURROGATE_DEGREES, (8, 16, 32, 64, 128))
        self.assertEqual(pm.SCAN_POINTS, 256)
        self.assertEqual(pm.GOLDEN_ITERATIONS, 60)
        self.assertEqual(pm.BISECTION_LIMIT, 200)
        self.assertEqual(pm.OFFSET_ITERATIONS, 100)
        self.assertEqual(pm.LOG_DOMAIN_LIMIT, 20.0)
        self.assertEqual(pm.SURROGATE_TOLERANCE, 1e-6)
        self.assertEqual(pm.CONFIRMATION_TOLERANCE, 1e-5)
        self.assertEqual(pm.NOISE_UNIT, 2.0**-40)

    def test_the_noise_allowance_is_proportional_to_the_count(self) -> None:
        self.assertEqual(noise_allowance(0), 0.0)
        self.assertEqual(noise_allowance(1 << 40), 1.0)
        self.assertEqual(noise_allowance(3_000_000), 3_000_000 * 2.0**-40)


class ImmutabilityTests(unittest.TestCase):
    def test_the_values_are_immutable_and_slotted(self) -> None:
        for value in (
            Surrogate(0.0, 1.0, 8, (0.0,) * 9),
            LikelihoodForm(1, 0.0, 0.0),
            Boundary(0.0, 0.0, 0.0),
        ):
            with self.subTest(type=type(value).__name__):
                first = next(iter(type(value).__dataclass_fields__))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(value, first, 1.0)
                self.assertFalse(hasattr(value, "__dict__"))


class NodeTests(unittest.TestCase):
    def test_the_nodes_are_the_lobatto_nodes_of_the_window(self) -> None:
        logs = lobatto_logs(2.0, 0.5, 8)
        self.assertEqual(len(logs), 9)
        self.assertEqual(logs[0], 1.5)
        self.assertEqual(logs[-1], 2.5)
        for index, node in enumerate(logs):
            self.assertEqual(node, 2.0 - 0.5 * math.cos(math.pi * index / 8), index)
        self.assertTrue(all(a < b for a, b in zip(logs, logs[1:], strict=False)))
        for index in range(9):
            self.assertAlmostEqual(logs[index] + logs[8 - index], 4.0, places=14)

    def test_the_nodes_of_a_degree_are_every_other_node_of_the_next(self) -> None:
        for center, half in ((0.0, 1.0), (0.3, 0.017), (-4.2, 6.5)):
            for degree in (4, 8, 16, 32, 64):
                with self.subTest(center=center, half=half, degree=degree):
                    self.assertEqual(
                        lobatto_logs(center, half, 2 * degree)[::2],
                        lobatto_logs(center, half, degree),
                    )


class SurrogateTests(unittest.TestCase):
    def build(self, function: Any, degree: int = 8, center: float = 0.2, half: float = 0.7) -> Any:
        values = tuple(function(node) for node in lobatto_logs(center, half, degree))
        return Surrogate(center, half, degree, values)

    def test_it_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "degree"):
            Surrogate(0.0, 1.0, 7, (0.0,) * 8)
        with self.assertRaisesRegex(ValueError, "one value per node"):
            Surrogate(0.0, 1.0, 8, (0.0,) * 8)
        with self.assertRaisesRegex(ValueError, "positive half width"):
            Surrogate(0.0, 0.0, 8, (0.0,) * 9)
        with self.assertRaisesRegex(ValueError, "positive half width"):
            Surrogate(0.0, -1.0, 8, (0.0,) * 9)
        self.assertEqual(Surrogate(0.0, 1.0, 4, (0.0,) * 5).degree, 4)
        self.assertEqual(Surrogate(0.0, 1.0, 128, (0.0,) * 129).degree, 128)

    def test_the_window_is_reported(self) -> None:
        surrogate = self.build(math.exp)
        self.assertEqual(surrogate.low, 0.2 - 0.7)
        self.assertEqual(surrogate.high, 0.2 + 0.7)
        self.assertEqual(surrogate.logs, lobatto_logs(0.2, 0.7, 8))

    def test_a_node_gives_its_value_exactly(self) -> None:
        surrogate = self.build(lambda t: math.sin(3 * t))
        for node, value in zip(surrogate.logs, surrogate.values, strict=True):
            self.assertEqual(surrogate.value(node), value)

    def test_a_polynomial_of_the_degree_is_reproduced(self) -> None:
        def poly(t: float) -> float:
            return 1.0 + 2.0 * t - 3.0 * t**2 + 0.5 * t**5 - t**8

        surrogate = self.build(poly)
        for point in (-0.45, -0.1, 0.0, 0.33, 0.5, 0.88):
            self.assertAlmostEqual(surrogate.value(point), poly(point), places=11)

    def test_an_analytic_function_converges_with_the_degree(self) -> None:
        errors = []
        for degree in (8, 16, 32):
            surrogate = self.build(math.exp, degree)
            errors.append(
                max(abs(surrogate.value(t) - math.exp(t)) for t in (-0.3, 0.1, 0.45, 0.8))
            )
        self.assertLess(errors[0], 1e-6)
        self.assertLess(errors[1], 1e-14)
        self.assertLess(errors[2], 1e-15)

    def test_the_coarsened_surrogate_uses_every_other_node(self) -> None:
        surrogate = self.build(math.exp, 16)
        coarse = surrogate.coarsened()
        self.assertEqual(coarse.degree, 8)
        self.assertEqual(coarse.values, surrogate.values[::2])
        self.assertEqual((coarse.center, coarse.half_width), (0.2, 0.7))
        self.assertEqual(coarse.logs, lobatto_logs(0.2, 0.7, 8))

    def test_the_gap_is_the_largest_difference_at_the_midpoints(self) -> None:
        surrogate = self.build(math.exp, 16)
        coarse = surrogate.coarsened()
        midpoints = [0.2 - 0.7 * math.cos(math.pi * (i + 0.5) / 16) for i in range(16)]
        expected = max(abs(surrogate.value(t) - coarse.value(t)) for t in midpoints)
        self.assertEqual(surrogate.gap(), expected)
        self.assertGreater(expected, 1e-12)
        self.assertLess(expected, 1e-4)

    def test_the_gap_of_a_low_degree_polynomial_vanishes(self) -> None:
        surrogate = self.build(lambda t: 1.0 - 2.0 * t + 0.25 * t**3, 8)
        self.assertLess(surrogate.gap(), 1e-13)


class DevianceTests(unittest.TestCase):
    def test_the_deviance_is_twice_the_gap_to_the_maximum(self) -> None:
        form = LikelihoodForm(10, -7.5, -3.25)
        point, psi = 0.5, 0.1
        reduced = 10 * mpmath.mpf(point) + mpmath.exp(point) * mpmath.mpf(-7.5)
        reduced = reduced - 10 * mpmath.mpf(psi) - 10
        expected = 2 * (mpmath.mpf(-3.25) - reduced)
        self.assertAlmostEqual(form.deviance(point, psi), float(expected), places=12)

    def test_the_terms_are_summed_exactly(self) -> None:
        form = LikelihoodForm(1, 0.0, 1e16)
        # 1e16 - 1 * 1.0 + 1 * 1.0 ... the terms cancel exactly only if summed without rounding
        self.assertEqual(form.deviance(0.0, 0.0), 2.0 * math.fsum((1e16, -0.0, -0.0, 0.0, 1.0)))
        self.assertEqual(
            LikelihoodForm(3, 1.0, -1.0).deviance(0.0, 0.5),
            2.0 * (-1.0 - 1.0 + 1.5 + 3.0 - 0.0 - 0.0 - 0.0 + 0.0),
        )

    def test_the_surrogate_deviance_uses_the_polynomial_psi(self) -> None:
        form = LikelihoodForm(100, -40.0, -12.0)
        surrogate = Surrogate(0.0, 1.0, 8, tuple(0.1 * t for t in lobatto_logs(0.0, 1.0, 8)))
        self.assertAlmostEqual(
            surrogate_deviance(form, surrogate, 0.3), form.deviance(0.3, 0.03), places=12
        )


class RootTests(unittest.TestCase):
    def test_a_root_is_found_from_either_side(self) -> None:
        self.assertAlmostEqual(
            root_between(lambda t: t * t - 2.0, 3.0, 1.0), math.sqrt(2.0), places=15
        )
        self.assertAlmostEqual(
            root_between(lambda t: t * t - 2.0, -3.0, -1.0), -math.sqrt(2.0), places=15
        )
        self.assertAlmostEqual(root_between(lambda t: 1.0 - t, 0.0, 5.0), 1.0, places=15)

    def test_a_bracket_of_adjacent_floats_ends_the_search(self) -> None:
        low = 1.0
        high = math.nextafter(1.0, 2.0)
        self.assertIn(root_between(lambda t: t - 1.0, high, low), (low, high))

    def test_the_bisection_is_bounded(self) -> None:
        calls = []

        def counted(point: float) -> float:
            calls.append(point)
            return point

        root_between(counted, 1.0, -1.0 + 1e-300)
        self.assertLessEqual(len(calls), pm.BISECTION_LIMIT)


class SeriesTests(unittest.TestCase):
    def test_the_function_is_accurate_on_both_sides_of_the_series_limit(self) -> None:
        for offset in (1e-9, 1e-4, -0.2, 0.3, 0.4999999, 0.5, -0.5, 0.75, -2.0, 9.0):
            with self.subTest(offset=offset):
                exact = mpmath.mpf(offset) + mpmath.exp(-mpmath.mpf(offset)) - 1
                got = pm._excess_function(offset)
                self.assertLess(abs(mpmath.mpf(got) - exact) / abs(exact), 1e-14)

    def test_the_series_constants_are_pinned(self) -> None:
        self.assertEqual(pm.SERIES_LIMIT, 0.5)
        self.assertEqual(pm.SERIES_TERMS, 24)


class OffsetTests(unittest.TestCase):
    def residual(self, offset: float) -> Any:
        y = mpmath.mpf(offset)
        return y + mpmath.exp(-y) - 1

    def test_no_slack_has_no_offset(self) -> None:
        self.assertEqual(boundary_offsets(0.0), (0.0, 0.0))
        self.assertEqual(boundary_offsets(-1e-3), (0.0, 0.0))

    def test_the_offsets_solve_the_equation(self) -> None:
        for slack in (1e-14, 1e-12, 2e-7, 1e-3, 0.0192, 0.5, 1.0, 7.0, 50.0, 1e4, 1e8):
            with self.subTest(slack=slack):
                below, above = boundary_offsets(slack)
                self.assertLess(below, 0.0)
                self.assertGreater(above, 0.0)
                for offset in (below, above):
                    self.assertLess(abs(self.residual(offset) - mpmath.mpf(slack)) / slack, 1e-9)

    def test_the_offsets_are_close_to_the_square_root_for_small_slack(self) -> None:
        below, above = boundary_offsets(2e-8)
        self.assertAlmostEqual(above, 2e-4, delta=2e-8)
        self.assertAlmostEqual(below, -2e-4, delta=2e-8)
        self.assertGreater(above, -below)

    def test_the_newton_steps_are_bounded(self) -> None:
        with mock.patch.object(pm, "OFFSET_ITERATIONS", 0):
            self.assertEqual(boundary_offsets(0.5), (-1.0, 1.5))
        below, above = boundary_offsets(0.5)
        self.assertLess(-1.0, below)
        self.assertLess(above, 1.5)

    def test_a_slack_beyond_any_deviance_still_has_offsets(self) -> None:
        for slack in (1e12, 1e30):
            with self.subTest(slack=slack):
                below, above = boundary_offsets(slack)
                self.assertAlmostEqual(above / slack, 1.0, places=10)
                residual = mpmath.exp(-mpmath.mpf(below)) + mpmath.mpf(below) - 1
                self.assertAlmostEqual(float(residual / slack), 1.0, places=9)
        below, above = boundary_offsets(1e300)
        self.assertAlmostEqual(above / 1e300, 1.0, places=12)
        self.assertGreater(-below, 600.0)
        self.assertLess(-below, 710.0)

    def test_the_offsets_grow_with_the_slack(self) -> None:
        previous = (0.0, 0.0)
        for slack in (1e-9, 1e-6, 1e-3, 1.0, 100.0):
            current = boundary_offsets(slack)
            self.assertLess(current[0], previous[0])
            self.assertGreater(current[1], previous[1])
            previous = current


class BoundaryTests(unittest.TestCase):
    def test_the_boundary_is_psi_plus_the_offsets_of_the_slack(self) -> None:
        point, psi, delta = 0.2, 0.4, 3.84
        reduced = 1000 * point + math.exp(point) * -300.0 - 1000 * psi - 1000.0
        form = LikelihoodForm(1000, -300.0, reduced + 1.0)
        deviance = form.deviance(point, psi)
        self.assertAlmostEqual(deviance, 2.0, places=10)
        bound = boundary_at(form, delta, point, psi)
        slack = (delta - deviance) / 2000.0
        below, above = boundary_offsets(slack)
        self.assertGreater(slack, 0.0)
        self.assertEqual(bound, Boundary(psi + below, psi + above, slack))

    def test_a_point_beyond_the_level_has_no_width(self) -> None:
        form = LikelihoodForm(10, -3.0, -5.0)
        bound = boundary_at(form, 1e-9, 0.0, 0.3)
        self.assertEqual(bound, Boundary(0.3, 0.3, 0.0))


class ExtremumTests(unittest.TestCase):
    def test_an_interior_maximum_and_minimum_are_found(self) -> None:
        top = extremum(lambda t: -((t - 0.3) ** 2), 0.0, 1.0, maximize=True)
        self.assertAlmostEqual(top[0], 0.3, places=7)
        self.assertAlmostEqual(top[1], 0.0, places=12)
        bottom = extremum(lambda t: (t - 0.6) ** 2 + 2.0, 0.0, 1.0, maximize=False)
        self.assertAlmostEqual(bottom[0], 0.6, places=7)
        self.assertAlmostEqual(bottom[1], 2.0, places=12)

    def test_a_monotone_function_is_extreme_at_the_ends(self) -> None:
        up = extremum(lambda t: t, -2.0, 3.0, maximize=True)
        down = extremum(lambda t: t, -2.0, 3.0, maximize=False)
        self.assertAlmostEqual(up[0], 3.0, places=12)
        self.assertAlmostEqual(down[0], -2.0, places=12)
        self.assertAlmostEqual(up[1], 3.0, places=12)
        self.assertAlmostEqual(down[1], -2.0, places=12)

    def test_an_extremum_between_scan_points_is_refined(self) -> None:
        target = 0.5 + 0.4 * 1e-3
        top = extremum(lambda t: -((t - target) ** 2), 0.0, 1.0, maximize=True)
        self.assertAlmostEqual(top[0], target, places=8)

    def test_a_tie_goes_to_the_first_scan_point(self) -> None:
        point, value = extremum(lambda t: 1.0, 2.0, 5.0, maximize=True)
        self.assertEqual(point, 2.0)
        self.assertEqual(value, 1.0)

    def test_the_work_is_bounded_by_the_scan_and_the_golden_iterations(self) -> None:
        calls: list[float] = []

        def counted(t: float) -> float:
            calls.append(t)
            return math.sin(5.0 * t)

        extremum(counted, 0.0, 1.0, maximize=True)
        self.assertEqual(len(calls), pm.SCAN_POINTS + 1 + 2 + pm.GOLDEN_ITERATIONS)
        self.assertTrue(all(0.0 <= point <= 1.0 for point in calls))

    def test_the_best_point_seen_is_returned_never_a_worse_one(self) -> None:
        # a function with two peaks: the scan picks the higher, the golden search stays near it
        def two(t: float) -> float:
            return math.exp(-200 * (t - 0.2) ** 2) + 0.5 * math.exp(-200 * (t - 0.8) ** 2)

        point, value = extremum(two, 0.0, 1.0, maximize=True)
        self.assertAlmostEqual(point, 0.2, places=6)
        self.assertGreaterEqual(value, 1.0 - 1e-12)


if __name__ == "__main__":
    unittest.main()
