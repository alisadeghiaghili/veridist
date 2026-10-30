"""The damped Newton rule of the censored fits, pinned constant by constant.

Every constant of the step rule is asserted, and each branch of the rule (the direction and its
shift, the step limits, the acceptance test, the fallback stencils, the cap, the boundary, the
choice of start) is exercised on small synthetic objectives whose exact answers are known, so
that a change to any of them changes a test.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from veridist.families import _newton as newton
from veridist.families.registry import FamilyId

EXPONENTIAL = FamilyId.EXPONENTIAL
GAMMA = FamilyId.GAMMA
NORMAL = FamilyId.NORMAL


def exponential_sums(events: float, total: float):
    """Sums of ``events * log(rate) - rate * total`` (maximum at ``events / total``)."""

    def at(theta: tuple[float, ...]) -> list[float]:
        (rate,) = theta
        return [events * math.log(rate) - rate * total, events / rate - total, -events / rate**2]

    return at


def quartic(centre: tuple[float, float]):
    """Sums of ``-(a - ca)**4 - (b - cb)**4``: Newton converges only linearly."""

    def at(theta: tuple[float, ...]) -> list[float]:
        da, db = theta[0] - centre[0], theta[1] - centre[1]
        return [-(da**4 + db**4), -4 * da**3, -4 * db**3, -12 * da**2, 0.0, -12 * db**2]

    return at


class ConstantsTests(unittest.TestCase):
    def test_the_step_rule_constants(self) -> None:
        self.assertEqual(newton.CLAMP, 2.0)
        self.assertEqual(newton.ARMIJO, 1e-4)
        self.assertEqual(newton.ARMIJO_SLACK, 1e-12)
        self.assertEqual(newton.ITERATION_CAP, 20)
        self.assertEqual(newton.STENCIL_CAP, 6)
        self.assertEqual(newton.TRIALS, (1.0, 0.5))
        self.assertEqual(newton.SHRINK, 0.25)
        self.assertEqual((newton.LOWER_FACTOR, newton.UPPER_FACTOR), (0.1, 10.0))
        self.assertEqual((newton.LOG_LIMIT, newton.LOCATION_LIMIT), (20.0, 1e6))
        self.assertEqual((newton.TOLERANCE, newton.GAMMA_TOLERANCE), (1e-8, 1e-6))
        self.assertEqual(newton.NEGATIVE_DEFINITE, 1e-10)
        self.assertEqual((newton.SHIFT_SCALE, newton.SHIFT_ABSOLUTE), (0.1, 1e-8))
        self.assertEqual(newton.DISTINCT, 1e-5)
        self.assertEqual(newton.SETUP_PASSES, 2)

    def test_the_plan_is_at_most_twenty_two_passes(self) -> None:
        self.assertEqual(newton.SETUP_PASSES + newton.ITERATION_CAP, 22)


class StartGridTests(unittest.TestCase):
    def test_the_exponential_grid_scales_the_start_rate(self) -> None:
        grid = newton.start_candidates(EXPONENTIAL, (2.0,))
        self.assertEqual(
            grid, [(2.0 * f,) for f in (1.0, 0.1, 0.25, 0.5, 2.0, 4.0, 10.0, 0.04, 25.0)]
        )

    def test_the_location_scale_grid_shifts_mu_and_scales_sigma(self) -> None:
        grid = newton.start_candidates(NORMAL, (0.0, 1.0))
        self.assertEqual(len(grid), 9)
        self.assertEqual(grid[0], (0.0, 1.0))
        expected = [(0.0, 1.0)]
        for factor in (1.0, 0.3, 3.0):
            for shift in (0.0, -1.0, 1.0):
                if (factor, shift) != (1.0, 0.0):
                    expected.append((shift / factor, 1.0 / factor))
        self.assertEqual(grid, expected)

    def test_the_location_scale_grid_follows_a_general_start(self) -> None:
        m0, u0 = 0.6, 2.0
        grid = newton.start_candidates(FamilyId.WEIBULL_MIN, (m0, u0))
        mu, sigma = m0 / u0, 1.0 / u0
        for (m, u), (factor, shift) in zip(
            grid[1:],
            [(f, s) for f in (1.0, 0.3, 3.0) for s in (0.0, -1.0, 1.0) if (f, s) != (1.0, 0.0)],
            strict=True,
        ):
            self.assertAlmostEqual(1.0 / u, sigma * factor, places=14)
            self.assertAlmostEqual(m / u, mu + shift * sigma, places=14)

    def test_the_gamma_grid_moves_the_shape_keeping_the_mean_and_the_scale(self) -> None:
        grid = newton.start_candidates(GAMMA, (1.0, -1.0))
        self.assertEqual(len(grid), 9)
        self.assertEqual(
            grid,
            [
                (1.0, -1.0),
                (1.0, -1.7),
                (1.0, -0.30000000000000004),
                (0.0, 0.0),
                (0.0, -0.7),
                (0.0, 0.7),
                (2.0, -2.0),
                (2.0, -2.7),
                (2.0, -1.3),
            ],
        )
        for alpha, beta in grid[3:6]:
            self.assertEqual(alpha, 0.0)
        for alpha, beta in grid[6:]:
            self.assertEqual(alpha, 2.0)


class DomainAndLimitsTests(unittest.TestCase):
    def test_the_hard_limits(self) -> None:
        self.assertTrue(newton.in_domain(EXPONENTIAL, (1.0,)))
        self.assertTrue(newton.in_domain(EXPONENTIAL, (math.exp(19.9),)))
        self.assertFalse(newton.in_domain(EXPONENTIAL, (math.exp(20.1),)))
        self.assertFalse(newton.in_domain(EXPONENTIAL, (math.exp(-20.1),)))
        self.assertFalse(newton.in_domain(EXPONENTIAL, (0.0,)))
        self.assertTrue(newton.in_domain(GAMMA, (19.9, -19.9)))
        self.assertFalse(newton.in_domain(GAMMA, (20.0, 0.0)))
        self.assertFalse(newton.in_domain(GAMMA, (0.0, -20.0)))
        self.assertTrue(newton.in_domain(NORMAL, (999999.0, 1.0)))
        self.assertFalse(newton.in_domain(NORMAL, (1e6, 1.0)))
        self.assertFalse(newton.in_domain(NORMAL, (0.0, 0.0)))
        self.assertFalse(newton.in_domain(NORMAL, (0.0, -1.0)))
        self.assertFalse(newton.in_domain(NORMAL, (0.0, math.exp(20.5))))

    def test_the_coordinate_that_is_kept_within_a_factor(self) -> None:
        self.assertEqual(newton.positive_index(EXPONENTIAL), 0)
        self.assertIsNone(newton.positive_index(GAMMA))
        for family in (NORMAL, FamilyId.LOGNORMAL, FamilyId.WEIBULL_MIN, FamilyId.GUMBEL_RIGHT):
            self.assertEqual(newton.positive_index(family), 1)

    def test_the_largest_coordinate_of_a_step_is_two(self) -> None:
        step = np.array([10.0, -4.0])
        self.assertEqual(newton.feasible_scale(GAMMA, (0.0, 0.0), step), 0.2)
        self.assertEqual(newton.feasible_scale(GAMMA, (0.0, 0.0), np.array([1.0, -2.0])), 1.0)
        self.assertEqual(newton.feasible_scale(GAMMA, (0.0, 0.0), np.array([1.0, -4.0])), 0.5)

    def test_a_rate_stays_within_a_tenth_and_ten_times_its_value(self) -> None:
        down = newton.feasible_scale(NORMAL, (0.0, 2.0), np.array([0.0, -4.0]))
        self.assertAlmostEqual(down, 0.9 * 2.0 / 4.0, places=15)
        up = newton.feasible_scale(NORMAL, (0.0, 0.05), np.array([0.0, 1.0]))
        self.assertAlmostEqual(up, 9.0 * 0.05, places=15)
        self.assertEqual(newton.feasible_scale(NORMAL, (0.0, 2.0), np.array([0.0, 1.0])), 1.0)
        # The exponential's only coordinate is the rate.
        self.assertAlmostEqual(
            newton.feasible_scale(EXPONENTIAL, (1.0,), np.array([-5.0])), 0.18, places=15
        )
        # Both limits: the clamp applies after the factor.
        self.assertEqual(newton.feasible_scale(NORMAL, (0.0, 100.0), np.array([8.0, 1.0])), 0.25)

    def test_a_step_that_stays_inside_is_not_scaled(self) -> None:
        self.assertEqual(newton.feasible_scale(NORMAL, (0.0, 1.0), np.array([-0.5, 0.5])), 1.0)
        self.assertEqual(newton.feasible_scale(NORMAL, (0.0, 0.1), np.array([0.0, 0.8])), 1.0)
        self.assertAlmostEqual(
            newton.feasible_scale(NORMAL, (0.0, 0.1), np.array([0.0, 1.0])), 0.9, places=15
        )


class ConcavityTests(unittest.TestCase):
    def test_which_cells_have_a_global_maximum(self) -> None:
        concave = newton.concave_in_coordinates
        for family in (NORMAL, FamilyId.LOGNORMAL, FamilyId.WEIBULL_MIN, FamilyId.GUMBEL_RIGHT):
            self.assertTrue(concave(family, False, False))
            self.assertFalse(concave(family, True, False))
            self.assertFalse(concave(family, True, True))
        self.assertTrue(concave(EXPONENTIAL, False, False))
        self.assertTrue(concave(EXPONENTIAL, True, False))
        self.assertFalse(concave(EXPONENTIAL, True, True))
        self.assertFalse(concave(GAMMA, False, False))
        self.assertFalse(concave(GAMMA, True, False))


class DirectionTests(unittest.TestCase):
    def test_a_negative_definite_hessian_gives_the_newton_direction(self) -> None:
        hessian = np.array([[-2.0, 0.5], [0.5, -1.0]])
        gradient = np.array([1.0, -2.0])
        step, shifted = newton.direction(hessian, gradient)
        self.assertFalse(shifted)
        np.testing.assert_allclose(hessian @ step, -gradient, rtol=0, atol=1e-15)

    def test_a_hessian_that_is_not_negative_definite_is_shifted(self) -> None:
        hessian = np.array([[1.0, 0.0], [0.0, -3.0]])
        gradient = np.array([1.0, 1.0])
        step, shifted = newton.direction(hessian, gradient)
        self.assertTrue(shifted)
        # largest eigenvalue 1, scale 3: shift = 1 + 0.1 * 3 + 1e-8, so H - shift I has the
        # eigenvalues 1 - shift = -(0.3 + 1e-8) and -3 - shift.
        shift = 1.0 + 0.1 * 3.0 + 1e-8
        np.testing.assert_allclose(step, [1.0 / (shift - 1.0), 1.0 / (3.0 + shift)], rtol=1e-14)

    def test_the_negative_definite_test_is_relative_to_the_spectral_scale(self) -> None:
        near = np.array([[-1e-11, 0.0], [0.0, -1.0]])
        self.assertTrue(newton.direction(near, np.array([1.0, 1.0]))[1])
        firm = np.array([[-2e-10, 0.0], [0.0, -1.0]])
        self.assertFalse(newton.direction(firm, np.array([1.0, 1.0]))[1])
        self.assertTrue(newton.direction(np.zeros((1, 1)), np.array([1.0]))[1])

    def test_the_sums_are_unpacked_in_order(self) -> None:
        value, gradient, hessian = newton.unpack([1.0, 2.0, 3.0], 1)
        self.assertEqual((value, gradient.tolist(), hessian.tolist()), (1.0, [2.0], [[3.0]]))
        value, gradient, hessian = newton.unpack([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2)
        self.assertEqual(value, 1.0)
        self.assertEqual(gradient.tolist(), [2.0, 3.0])
        self.assertEqual(hessian.tolist(), [[4.0, 5.0], [5.0, 6.0]])


class RunTests(unittest.TestCase):
    def test_a_start_at_the_maximum_converges_without_a_step(self) -> None:
        found = newton.maximize(
            EXPONENTIAL, exponential_sums(5.0, 10.0), (1.0,), concave=True, tolerance=1e-8
        )
        assert found is not None
        self.assertEqual(found.run.status, "CONVERGED")
        self.assertEqual(found.run.theta, (0.5,))
        self.assertEqual(found.run.iterations, 0)
        self.assertEqual(found.passes, 2)
        self.assertEqual((found.starts, found.distinct), (1, 1))

    def test_newton_converges_quadratically_to_the_maximum(self) -> None:
        found = newton.maximize(
            EXPONENTIAL, exponential_sums(5.0, 10.0), (1.3,), concave=True, tolerance=1e-8
        )
        assert found is not None
        self.assertEqual(found.run.status, "CONVERGED")
        self.assertAlmostEqual(found.run.theta[0], 0.5, places=14)
        self.assertGreaterEqual(found.run.iterations, 1)
        self.assertLessEqual(found.run.iterations, 6)
        self.assertEqual(found.passes, 2 + found.run.passes)
        self.assertEqual(found.run.shifted, 0)

    def test_the_tolerance_decides_when_a_run_stops(self) -> None:
        sums = quartic((1.0, 1.0))
        loose = newton.maximize(GAMMA, sums, (0.0, 0.0), concave=True, tolerance=1e-1)
        tight = newton.maximize(GAMMA, sums, (0.0, 0.0), concave=True, tolerance=1e-2)
        assert loose is not None and tight is not None
        self.assertEqual(loose.run.status, "CONVERGED")
        self.assertEqual(tight.run.status, "CONVERGED")
        self.assertLess(loose.run.iterations, tight.run.iterations)

    def test_the_iteration_cap_ends_a_slow_run_without_an_estimate(self) -> None:
        found = newton.maximize(
            GAMMA, quartic((1.0, 1.0)), (0.0, 0.0), concave=True, tolerance=1e-8
        )
        assert found is not None
        self.assertEqual(found.run.status, "OPTIMIZER_EXHAUSTED")
        self.assertEqual(found.run.iterations, 20)
        self.assertEqual(found.passes, 2 + 20)

    def test_a_flat_likelihood_ends_at_the_cap_with_shifted_directions(self) -> None:
        def flat(theta: tuple[float, ...]) -> list[float]:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        found = newton.maximize(GAMMA, flat, (0.0, 0.0), concave=True, tolerance=1e-8)
        assert found is not None
        self.assertEqual(found.run.status, "OPTIMIZER_EXHAUSTED")
        self.assertEqual(found.run.iterations, 20)
        self.assertEqual(found.run.shifted, 21)

    def test_a_trial_that_never_satisfies_armijo_exhausts_the_stencils(self) -> None:
        def misleading(theta: tuple[float, ...]) -> list[float]:
            if theta == (1.0,):
                return [0.0, 1.0, -1.0]
            return [-1.0, 1.0, -1.0]

        found = newton.maximize(EXPONENTIAL, misleading, (1.0,), concave=True, tolerance=1e-8)
        assert found is not None
        self.assertEqual(found.run.status, "OPTIMIZER_EXHAUSTED")
        self.assertEqual(found.run.stencil_failures, 6)
        self.assertEqual(found.run.passes, 6)
        self.assertEqual(found.passes, 2 + 6)

    def test_a_gain_below_the_precision_of_the_sums_does_not_refuse_a_step(self) -> None:
        # Near the maximum the predicted gain (1e-4 * 1e-12) is below the noise of a sum of
        # terms of size one; a trial whose value is lower by less than the precision of the
        # sums (1e-12 times the larger of one and the value) is accepted, one lower by more is not.
        def at(drop: float):
            def sums(theta: tuple[float, ...]) -> list[float]:
                if theta == (1.0,):
                    return [-1.0, 1e-6, -1.0]
                if theta[0] > 1.0 and theta[0] < 1.01:
                    return [-1.0 - drop, 0.0, -1.0]
                return [-2.0, 0.0, -1.0]

            return sums

        accepted = newton.maximize(EXPONENTIAL, at(5e-13), (1.0,), concave=True, tolerance=1e-8)
        assert accepted is not None
        self.assertEqual(accepted.run.status, "CONVERGED")
        self.assertEqual(accepted.run.stencil_failures, 0)
        self.assertAlmostEqual(accepted.run.theta[0], 1.0 + 1e-6, places=12)
        refused = newton.maximize(EXPONENTIAL, at(5e-12), (1.0,), concave=True, tolerance=1e-8)
        assert refused is not None
        self.assertEqual(refused.run.status, "OPTIMIZER_EXHAUSTED")
        self.assertEqual(refused.run.stencil_failures, 6)

    def test_the_fallback_stencil_halves_the_trials_twice(self) -> None:
        # The curvature is claimed to be -0.25 where it is -2: from the grid's best point 2.0 the
        # direction is 4, limited to 2; the trials 4 and 3 fail Armijo, the next stencil's
        # first trial (a quarter of the limited direction) reaches the maximum 2.5.
        visited: list[float] = []

        def objective(theta: tuple[float, ...]) -> list[float]:
            (rate,) = theta
            visited.append(rate)
            return [-((rate - 2.5) ** 2), -2.0 * (rate - 2.5), -0.25]

        found = newton.maximize(EXPONENTIAL, objective, (1.0,), concave=True, tolerance=1e-8)
        assert found is not None
        self.assertEqual(found.run.status, "CONVERGED")
        self.assertEqual(found.run.theta, (2.5,))
        self.assertEqual(found.run.stencil_failures, 1)
        self.assertEqual(found.run.passes, 2)
        self.assertEqual(found.run.iterations, 1)
        self.assertEqual(found.passes, 4)
        self.assertEqual(visited[:9], [1.0, 0.1, 0.25, 0.5, 2.0, 4.0, 10.0, 0.04, 25.0])
        self.assertEqual(visited[9:], [4.0, 3.0, 2.5])

    def test_a_trial_outside_the_limits_is_not_evaluated_and_all_outside_is_a_boundary(
        self,
    ) -> None:
        calls: list[tuple[float, ...]] = []

        def rising(theta: tuple[float, ...]) -> list[float]:
            calls.append(theta)
            return [theta[0], 1.0, -1e-12, 0.0, 0.0, -1e-12]

        found = newton.maximize(GAMMA, rising, (0.0, 0.0), concave=True, tolerance=1e-8)
        assert found is not None
        self.assertEqual(found.run.status, "BOUNDARY_SOLUTION")
        self.assertTrue(all(abs(point[0]) < 20.0 and abs(point[1]) < 20.0 for point in calls))

    def test_the_best_of_the_start_grid_is_the_start_and_ties_keep_the_first(self) -> None:
        def level(theta: tuple[float, ...]) -> list[float]:
            return [0.0, 0.0, -1.0]

        found = newton.maximize(EXPONENTIAL, level, (3.0,), concave=True, tolerance=1e-8)
        assert found is not None
        self.assertEqual(found.run.status, "CONVERGED")
        self.assertEqual(found.run.theta, (3.0,))

    def test_no_evaluable_start_is_none(self) -> None:
        self.assertIsNone(
            newton.maximize(EXPONENTIAL, lambda theta: None, (1.0,), concave=True, tolerance=1e-8)
        )
        self.assertIsNone(
            newton.maximize(
                EXPONENTIAL,
                lambda theta: [math.nan, 0.0, -1.0],
                (1.0,),
                concave=True,
                tolerance=1e-8,
            )
        )

    def test_a_trial_that_cannot_be_evaluated_is_skipped(self) -> None:
        # A bowl at 0.3; the full Newton step from the grid's best point 0.25 lands on 0.3 where
        # the likelihood "cannot be evaluated", so the half step is taken.
        refused: list[float] = []

        def bowl(theta: tuple[float, ...]) -> list[float] | None:
            (rate,) = theta
            if abs(rate - 0.3) < 1e-12 and not refused:
                refused.append(rate)
                return None
            return [-((rate - 0.3) ** 2), -2.0 * (rate - 0.3), -2.0]

        found = newton.maximize(EXPONENTIAL, bowl, (1.0,), concave=True, tolerance=1e-8)
        assert found is not None
        self.assertEqual(found.run.status, "CONVERGED")
        self.assertEqual(len(refused), 1)
        self.assertAlmostEqual(found.run.theta[0], 0.3, places=7)


class SecondStartTests(unittest.TestCase):
    def two_peaks(self):
        """Log-likelihood in ``(alpha, beta)`` with a low maximum at ``(-3, 0)`` and a high one at
        ``(3, 0)``, each a quadratic bowl on its own side of ``alpha = 0``."""

        def at(theta: tuple[float, ...]) -> list[float]:
            a, b = theta
            if a < 0.0:
                return [-((a + 3.0) ** 2) - b**2, -2 * (a + 3.0), -2 * b, -2.0, 0.0, -2.0]
            return [1.0 - (a - 3.0) ** 2 - b**2, -2 * (a - 3.0), -2 * b, -2.0, 0.0, -2.0]

        return at

    def test_without_a_concavity_guarantee_a_second_start_is_tried(self) -> None:
        # The grid's best point is on the low bowl, its second-best on the high one.
        found = newton.maximize(GAMMA, self.two_peaks(), (-0.2, 0.0), concave=False, tolerance=1e-8)
        assert found is not None
        self.assertEqual(found.starts, 2)
        self.assertEqual(found.run.status, "CONVERGED")
        self.assertAlmostEqual(found.run.theta[0], 3.0, places=10)
        self.assertEqual(found.distinct, 2)

    def test_with_a_concavity_guarantee_one_start_is_run(self) -> None:
        found = newton.maximize(GAMMA, self.two_peaks(), (-0.2, 0.0), concave=True, tolerance=1e-8)
        assert found is not None
        self.assertEqual(found.starts, 1)
        self.assertAlmostEqual(found.run.theta[0], -3.0, places=10)
        self.assertEqual(found.distinct, 1)

    def test_two_starts_that_meet_are_one_stationary_point(self) -> None:
        found = newton.maximize(
            EXPONENTIAL, exponential_sums(5.0, 10.0), (1.0,), concave=False, tolerance=1e-8
        )
        assert found is not None
        self.assertEqual(found.starts, 2)
        self.assertEqual(found.distinct, 1)
        self.assertAlmostEqual(found.run.theta[0], 0.5, places=14)

    def test_a_failed_first_run_does_not_hide_a_converged_second_one(self) -> None:
        # The highest grid value is a plateau at rate 25 (no ascent is found from it); the
        # second-best point sits on a bowl at 0.3, where the second run converges.
        def at(theta: tuple[float, ...]) -> list[float]:
            (rate,) = theta
            if rate > 20.0:
                return [10.0, 1.0, 0.0]
            return [-((rate - 0.3) ** 2), -2 * (rate - 0.3), -2.0]

        found = newton.maximize(EXPONENTIAL, at, (1.0,), concave=False, tolerance=1e-8)
        assert found is not None
        self.assertEqual(found.starts, 2)
        self.assertEqual(found.run.status, "CONVERGED")
        self.assertAlmostEqual(found.run.theta[0], 0.3, places=10)
        self.assertEqual(found.distinct, 1)

    def test_when_no_run_converges_the_first_run_is_reported(self) -> None:
        found = newton.maximize(
            GAMMA, quartic((1.0, 1.0)), (0.0, 0.0), concave=False, tolerance=1e-8
        )
        assert found is not None
        self.assertEqual(found.run.status, "OPTIMIZER_EXHAUSTED")
        self.assertEqual(found.starts, 2)
        self.assertEqual(found.distinct, 1)


if __name__ == "__main__":
    unittest.main()
