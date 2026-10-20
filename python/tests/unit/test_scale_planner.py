"""The multi-pass planner: budget, ledger, score, Hermite start and the root search."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import numpy as np

from tests.unit.scale_multi_pass_oracle import (
    Recorder,
    mp_weibull,
    relative,
    sums_of,
    synthetic,
    weibull_sample,
)
from veridist.engine.errors import FailureCode
from veridist.engine.pass_budget import PassBudgetError
from veridist.families import gumbel, weibull
from veridist.scale import _planner as planner
from veridist.scale._estimates import LOG_SHAPE_LIMIT
from veridist.scale._planner import (
    PassLedger,
    PassRequest,
    PassRole,
    PlanBounds,
    ProfileNode,
    ProfileRefusal,
    ProfileSolution,
    ProfileSums,
    check_pass_budget,
    hermite_root,
    plan_passes,
    score,
    solve_profile,
)
from veridist.scale._sources import FingerprintLevel, Partition


def centred_weibull(shape: float, size: int, seed: int) -> np.ndarray:
    logs = np.log(weibull_sample(shape, 1000.0, size, seed))
    return logs - logs.max()


def mean_of(centred: np.ndarray) -> float:
    return math.fsum(centred.tolist()) / len(centred)


class ConstantTests(unittest.TestCase):
    def test_every_cap_and_window_is_pinned(self) -> None:
        self.assertEqual(planner.PASS_BUDGET_DEFAULT, 8)
        self.assertEqual(planner.GRID_POINTS, 8)
        self.assertEqual(planner.INITIAL_LOG_WINDOW, 8.0)
        self.assertEqual(planner.EXPANSION_LOG_LIMITS, (12.0, 16.0, 20.0))
        self.assertEqual(planner.MAX_EXPANSION_PASSES, 3)
        self.assertEqual(planner.MAX_NEWTON_PASSES, 12)
        self.assertEqual(planner.CONVERGENCE_TOLERANCE, 1e-8)
        self.assertEqual(planner.MAX_HERMITE_BISECTIONS, 100)
        self.assertEqual(planner.MINIMUM_PASSES, 3)
        self.assertEqual(planner.MAXIMUM_PASSES, 17)

    def test_the_domain_is_the_in_memory_domain(self) -> None:
        self.assertEqual(planner.LOG_DOMAIN_LIMIT, 20.0)
        self.assertEqual(planner.LOG_DOMAIN_LIMIT, LOG_SHAPE_LIMIT)
        self.assertEqual(weibull._LOG_SHAPE_HARD_LIMITS, (-20.0, 20.0))
        self.assertEqual(gumbel._LOG_SCALE_HARD_LIMITS, (-20.0, 20.0))
        self.assertEqual(planner.EXPANSION_LOG_LIMITS[-1], planner.LOG_DOMAIN_LIMIT)

    def test_the_roles_are_the_provenance_names(self) -> None:
        self.assertEqual(
            [role.value for role in PassRole],
            [
                "statistics",
                "grid",
                "expansion",
                "iteration",
                "profile_window",
                "profile_refinement",
                "profile_confirmation",
                "gof",
            ],
        )

    def test_the_initial_grid_is_eight_points_geometric_over_the_window(self) -> None:
        logs = planner._initial_logs()
        self.assertEqual(len(logs), 8)
        self.assertEqual(logs[0], -8.0)
        self.assertEqual(logs[-1], 8.0)
        for index, value in enumerate(logs):
            self.assertAlmostEqual(value, -8.0 + 16.0 * index / 7.0, places=14)

    def test_an_expansion_band_has_eight_points_and_includes_its_outer_end_only(self) -> None:
        up = planner._band_logs(8.0, 12.0, upward=True)
        self.assertEqual(up, tuple(8.0 + 0.5 * (index + 1) for index in range(8)))
        down = planner._band_logs(8.0, 12.0, upward=False)
        self.assertEqual(down, tuple(-12.0 + 0.5 * index for index in range(8)))
        self.assertEqual(up[-1], 12.0)
        self.assertEqual(down[0], -12.0)
        self.assertLess(down[-1], -8.0)
        self.assertGreater(up[0], 8.0)


class BudgetTests(unittest.TestCase):
    def test_a_budget_is_a_built_in_integer_of_at_least_one(self) -> None:
        self.assertEqual(check_pass_budget(1), 1)
        self.assertEqual(check_pass_budget(8), 8)
        for bad in (0, -3):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                check_pass_budget(bad)
        for bad in (True, 2.0, "3", None):
            with self.subTest(bad=bad), self.assertRaises(TypeError):
                check_pass_budget(bad)

    def test_the_plan_publishes_its_bounds(self) -> None:
        self.assertEqual(plan_passes(8), PlanBounds(3, 17))
        self.assertEqual(plan_passes(3), PlanBounds(3, 17))
        self.assertEqual(plan_passes(100), PlanBounds(3, 17))

    def test_a_budget_below_the_minimum_is_refused_with_the_bounds(self) -> None:
        for budget in (1, 2):
            with self.subTest(budget=budget), self.assertRaises(PassBudgetError) as caught:
                plan_passes(budget)
            self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_INSUFFICIENT)
            self.assertEqual(
                dict(caught.exception.context),
                {"pass_budget": budget, "minimum_passes": 3, "maximum_passes": 17},
            )


def partitions(count: int = 2) -> tuple[Partition, ...]:
    return tuple(Partition(f"p{i}", i, 5, None, FingerprintLevel.NONE) for i in range(count))


class LedgerTests(unittest.TestCase):
    def test_begin_returns_the_partitions_in_order_and_records_the_pass(self) -> None:
        parts = partitions(3)
        ledger = PassLedger(parts, 8, PlanBounds(3, 17))
        self.assertEqual(ledger.pass_count, 0)
        self.assertEqual(tuple(ledger.begin(PassRole.STATISTICS, 0)), parts)
        self.assertEqual(ledger.pass_count, 1)
        self.assertEqual(tuple(ledger.begin(PassRole.GRID, 8, -8.0, 8.0)), parts)
        self.assertEqual(tuple(ledger.begin(PassRole.ITERATION, 1)), parts)
        record = ledger.provenance()
        self.assertEqual(
            (record.pass_budget, record.minimum_passes, record.maximum_passes, record.pass_count),
            (8, 3, 17, 3),
        )
        self.assertEqual(
            [(item.role, item.points, item.log_low, item.log_high) for item in record.records],
            [
                ("statistics", 0, None, None),
                ("grid", 8, -8.0, 8.0),
                ("iteration", 1, None, None),
            ],
        )

    def test_a_pass_beyond_the_budget_is_refused_and_not_recorded(self) -> None:
        ledger = PassLedger(partitions(), 3, PlanBounds(3, 17))
        ledger.begin(PassRole.STATISTICS, 0)
        ledger.begin(PassRole.GRID, 8, -8.0, 8.0)
        ledger.begin(PassRole.ITERATION, 1)
        with self.assertRaises(PassBudgetError) as caught:
            ledger.begin(PassRole.ITERATION, 1, bracket_log_width=0.25)
        self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_EXCEEDED)
        self.assertEqual(
            dict(caught.exception.context),
            {
                "max_passes": 3,
                "actual_pass_count": 3,
                "attempted_pass": 4,
                "bracket_log_width": 0.25,
            },
        )
        self.assertEqual(ledger.pass_count, 3)
        self.assertEqual(ledger.provenance().pass_count, 3)

    def test_the_refusal_without_a_bracket_has_no_width(self) -> None:
        ledger = PassLedger(partitions(), 3, PlanBounds(3, 17))
        for _ in range(3):
            ledger.begin(PassRole.STATISTICS if _ == 0 else PassRole.ITERATION, 0 if _ == 0 else 1)
        with self.assertRaises(PassBudgetError) as caught:
            ledger.begin(PassRole.GRID, 8, -8.0, 8.0)
        self.assertNotIn("bracket_log_width", caught.exception.context)


class ScoreTests(unittest.TestCase):
    def test_the_score_and_its_slope_from_the_sums(self) -> None:
        sums = ProfileSums(4.0, -2.0, 1.5)
        value, slope = score(2.0, sums, -0.25)
        # 1/k - S1/S0 + mean = 0.5 + 0.5 - 0.25; slope = -1/k^2 - (S2/S0 - (S1/S0)^2)
        self.assertEqual(value, 0.75)
        self.assertEqual(slope, -(0.25 + (0.375 - 0.25)))

    def test_a_negative_rounded_variance_is_clipped_to_zero(self) -> None:
        sums = ProfileSums(1.0, 1.0, 0.999999999999)
        _, slope = score(2.0, sums, 0.0)
        self.assertEqual(slope, -0.25)


class HermiteTests(unittest.TestCase):
    @staticmethod
    def node(parameter: float, function: object, derivative: object) -> ProfileNode:
        assert callable(function) and callable(derivative)
        return ProfileNode(
            parameter, function(math.log(parameter)), derivative(math.log(parameter)) / parameter
        )

    def test_the_root_of_a_cubic_in_the_logarithm_is_recovered(self) -> None:
        def cubic(t: float) -> float:
            return -(t - 0.7) * (1.0 + 0.05 * (t - 0.7) ** 2)

        def derivative(t: float) -> float:
            return -(1.0 + 0.15 * (t - 0.7) ** 2)

        low = self.node(math.exp(-1.0), cubic, derivative)
        high = self.node(math.exp(2.5), cubic, derivative)
        root = hermite_root(low, high)
        self.assertAlmostEqual(math.log(root), 0.7, places=12)

    def test_the_root_is_inside_the_bracket_and_a_poor_slope_still_gives_one(self) -> None:
        low = ProfileNode(1.0, 2.0, -1.0)
        high = ProfileNode(10.0, -3.0, -0.001)
        root = hermite_root(low, high)
        self.assertTrue(1.0 <= root <= 10.0)

    def test_a_root_at_the_upper_end_is_returned_within_the_bracket(self) -> None:
        low = ProfileNode(1.0, 1.0, -1.0)
        high = ProfileNode(2.0, 0.0, -1.0)
        root = hermite_root(low, high)
        self.assertLessEqual(root, 2.0)
        self.assertAlmostEqual(root, 2.0, places=12)

    def test_the_bisection_cap_limits_the_work(self) -> None:
        def line(t: float) -> float:
            return 0.3 - t

        low = self.node(math.exp(-1.0), line, lambda t: -1.0)
        high = self.node(math.exp(2.0), line, lambda t: -1.0)
        with mock.patch.object(planner, "MAX_HERMITE_BISECTIONS", 1):
            coarse = hermite_root(low, high)
        fine = hermite_root(low, high)
        self.assertAlmostEqual(math.log(fine), 0.3, places=12)
        self.assertGreater(abs(math.log(coarse) - 0.3), 0.1)


class BracketTests(unittest.TestCase):
    @staticmethod
    def nodes(*values: float) -> list[ProfileNode]:
        return [ProfileNode(float(index + 1), value, -1.0) for index, value in enumerate(values)]

    def test_the_bracket_is_the_adjacent_pair_from_positive_to_not_positive(self) -> None:
        nodes = self.nodes(3.0, 2.0, 1.0, -1.0, -2.0)
        low, high = planner._bracket(nodes) or (None, None)
        self.assertEqual((low, high), (nodes[2], nodes[3]))

    def test_a_score_of_exactly_zero_closes_the_bracket_from_above(self) -> None:
        nodes = self.nodes(2.0, 0.0, -1.0)
        self.assertEqual(planner._bracket(nodes), (nodes[0], nodes[1]))

    def test_a_score_of_exactly_zero_does_not_open_a_bracket_from_below(self) -> None:
        self.assertIsNone(planner._bracket(self.nodes(0.0, -1.0)))

    def test_no_sign_change_is_no_bracket(self) -> None:
        for values in ((1.0, 2.0), (-1.0, -2.0), (-1.0, 1.0), (1.0,), ()):
            with self.subTest(values=values):
                self.assertIsNone(planner._bracket(self.nodes(*values)))


class SolveTests(unittest.TestCase):
    def test_a_typical_sample_takes_a_grid_and_a_few_iterations(self) -> None:
        centred = centred_weibull(1.7, 400, 3)
        recorder = Recorder(centred)
        solution = solve_profile(recorder, count=400, mean_w=mean_of(centred))
        assert isinstance(solution, ProfileSolution)
        self.assertEqual(recorder.roles()[0], "grid")
        self.assertTrue(all(role == "iteration" for role in recorder.roles()[1:]))
        self.assertTrue(2 <= len(recorder.roles()) - 1 <= 6)
        grid = recorder.requests[0]
        self.assertEqual(len(grid.parameters), 8)
        self.assertEqual((grid.log_low, grid.log_high), (-8.0, 8.0))
        self.assertEqual(grid.parameters, tuple(math.exp(v) for v in planner._initial_logs()))
        for request in recorder.requests[1:]:
            self.assertEqual(len(request.parameters), 1)
            self.assertIsNone(request.log_low)
            self.assertIsNone(request.log_high)
            assert request.bracket_log_width is not None
            self.assertGreater(request.bracket_log_width, 0.0)
        self.assertEqual(solution.iterate, recorder.requests[-1].parameters[0])
        self.assertLessEqual(abs(solution.parameter / solution.iterate - 1.0), 1e-8)

    def test_an_iteration_reports_the_log_width_of_the_bracket_it_works_in(self) -> None:
        centred = centred_weibull(1.7, 100, 4)
        recorder = Recorder(centred)
        solve_profile(recorder, count=100, mean_w=mean_of(centred))
        grid = recorder.requests[0].parameters
        values = [score(k, sums_of(centred, k), mean_of(centred))[0] for k in grid]
        index = max(i for i, value in enumerate(values) if value > 0.0)
        first = recorder.requests[1]
        self.assertEqual(first.bracket_log_width, math.log(grid[index + 1]) - math.log(grid[index]))
        for earlier, later in zip(recorder.requests[1:], recorder.requests[2:], strict=False):
            assert earlier.bracket_log_width is not None and later.bracket_log_width is not None
            self.assertLess(later.bracket_log_width, earlier.bracket_log_width)

    def test_the_log_ratio_is_the_second_order_expansion_about_the_iterate(self) -> None:
        point = 1.0e6
        step = 9.0e-3
        first, variance, count = -1.5, 0.25, 7
        value = step * (1.0 / (point * point) + variance)
        mean_w = value - 1.0 / point + first
        sums = ProfileSums(2.0, 2.0 * first, 2.0 * (first * first + variance))

        def evaluate(request: PassRequest) -> list[ProfileSums]:
            if request.role is PassRole.ITERATION:
                return [sums]
            return synthetic(lambda k: (1.0 - k) / (k * k), mean_w)(request)

        with mock.patch.object(planner, "hermite_root", return_value=point):
            solution = solve_profile(evaluate, count=count, mean_w=mean_w)
        assert isinstance(solution, ProfileSolution)
        actual_value, actual_slope = score(point, sums, mean_w)
        actual_step = -actual_value / actual_slope
        self.assertAlmostEqual(actual_step, step, delta=1e-12)
        expected = math.fsum(
            (
                math.log(2.0 / count),
                actual_step * first,
                0.5 * actual_step * actual_step * variance,
            )
        )
        self.assertEqual(solution.log_ratio, expected)
        self.assertEqual(solution.parameter, point + actual_step)
        self.assertEqual(solution.iterate, point)
        self.assertEqual(solution.sums, sums)

    def test_the_root_agrees_with_a_high_precision_root(self) -> None:
        for shape, size, seed in ((0.3, 60, 1), (1.7, 200, 2), (20.0, 120, 3)):
            logs = np.log(weibull_sample(shape, 1000.0, size, seed))
            centred = logs - logs.max()
            solution = solve_profile(Recorder(centred), count=size, mean_w=mean_of(centred))
            assert isinstance(solution, ProfileSolution)
            reference, *_ = mp_weibull(np.exp(logs), shape)
            with self.subTest(shape=shape):
                self.assertLess(relative(solution.parameter, reference), 1e-12)

    def test_the_log_ratio_is_that_of_the_root(self) -> None:
        centred = centred_weibull(1.7, 300, 5)
        solution = solve_profile(Recorder(centred), count=300, mean_w=mean_of(centred))
        assert isinstance(solution, ProfileSolution)
        exact = math.log(math.fsum(math.exp(solution.parameter * float(v)) for v in centred) / 300)
        self.assertAlmostEqual(solution.log_ratio, exact, delta=1e-13)

    def test_the_result_depends_on_the_sums_only(self) -> None:
        centred = centred_weibull(1.7, 100, 8)
        first = solve_profile(Recorder(centred), count=100, mean_w=mean_of(centred))
        second = solve_profile(Recorder(centred[::-1].copy()), count=100, mean_w=mean_of(centred))
        self.assertEqual(first, second)

    def test_a_newton_step_that_has_converged_is_accepted_before_the_safeguard(self) -> None:
        root = math.exp(-8.0 + 16.0 * 4 / 7) * (1.0 - 1e-10)
        evaluate = synthetic(lambda k: 2.0 * (root - k) / (k * k), 0.0)
        start = root * (1.0 - 1e-9)
        with mock.patch.object(planner, "hermite_root", return_value=start):
            solution = solve_profile(evaluate, count=10, mean_w=0.0)
        assert isinstance(solution, ProfileSolution)
        self.assertEqual(solution.iterate, start)
        self.assertAlmostEqual(solution.parameter / root, 1.0 + 1e-9, places=12)

    def test_a_step_outside_the_bracket_is_replaced_by_the_geometric_midpoint(self) -> None:
        points: list[float] = []

        def wild(k: float) -> float:
            points.append(k)
            return 1e6 if len(points) % 2 else -1e6

        def evaluate(request: PassRequest) -> list[ProfileSums]:
            if request.role is PassRole.ITERATION:
                return synthetic(wild, 0.0)(request)
            return synthetic(lambda k: (1.0 - k) / (k * k), 0.0)(request)

        self.assertIs(solve_profile(evaluate, count=10, mean_w=0.0), ProfileRefusal.EXHAUSTED)
        below = math.exp(-8.0 + 16.0 * 3 / 7)
        above = math.exp(-8.0 + 16.0 * 4 / 7)
        self.assertTrue(below < points[0] < above)
        # the first iterate scores above zero: it is the new lower end, and the step overshoots
        self.assertEqual(points[1], math.sqrt(points[0] * above))
        # the second scores below zero: it is the new upper end, and the step undershoots
        self.assertEqual(points[2], math.sqrt(points[0] * points[1]))

    def test_the_newton_cap_is_twelve_passes_then_exhausted(self) -> None:
        calls = {"count": 0}

        def alternating(request: PassRequest) -> list[ProfileSums]:
            if request.role is PassRole.ITERATION:
                calls["count"] += 1
                sign = 1.0 if calls["count"] % 2 else -1.0
                return synthetic(lambda k: sign * 1e3, 0.0)(request)
            return synthetic(lambda k: 1.0 - k, 0.0)(request)

        self.assertIs(solve_profile(alternating, count=10, mean_w=0.0), ProfileRefusal.EXHAUSTED)
        self.assertEqual(calls["count"], 12)


class ExpansionTests(unittest.TestCase):
    @staticmethod
    def run_with_root(log_root: float) -> tuple[Recorder, ProfileSolution | ProfileRefusal]:
        """A recorder over two points whose profile root is near ``exp(log_root)``."""

        root = math.exp(log_root)
        recorder = Recorder(np.array([0.0, -1.0]))
        evaluate = synthetic(lambda k: (root - k) / (k * k), 0.0)

        def wrapper(request: PassRequest) -> list[ProfileSums]:
            recorder.requests.append(request)
            return evaluate(request)

        result = solve_profile(wrapper, count=2, mean_w=0.0)
        return recorder, result

    def test_a_root_inside_the_window_needs_no_expansion(self) -> None:
        recorder, result = self.run_with_root(5.0)
        self.assertIsInstance(result, ProfileSolution)
        self.assertEqual(recorder.roles().count("expansion"), 0)

    def test_a_root_above_the_window_is_found_by_one_to_three_expansions(self) -> None:
        for log_root, expected in ((10.0, 1), (14.0, 2), (18.0, 3)):
            recorder, result = self.run_with_root(log_root)
            with self.subTest(log_root=log_root):
                self.assertIsInstance(result, ProfileSolution)
                self.assertEqual(recorder.roles().count("expansion"), expected)
                self.assertEqual(
                    recorder.roles()[: 1 + expected], ["grid"] + ["expansion"] * expected
                )
                bands = [
                    request for request in recorder.requests if request.role is PassRole.EXPANSION
                ]
                self.assertEqual(
                    [(r.log_low, r.log_high) for r in bands],
                    [(8.5 + 4.0 * index, 12.0 + 4.0 * index) for index in range(expected)],
                )
                assert isinstance(result, ProfileSolution)
                self.assertAlmostEqual(math.log(result.parameter), log_root, places=6)

    def test_a_root_below_the_window_is_found_by_expansions_downward(self) -> None:
        for log_root, expected in ((-10.0, 1), (-14.0, 2), (-18.0, 3)):
            recorder, result = self.run_with_root(log_root)
            with self.subTest(log_root=log_root):
                self.assertIsInstance(result, ProfileSolution)
                self.assertEqual(recorder.roles().count("expansion"), expected)
                bands = [
                    request for request in recorder.requests if request.role is PassRole.EXPANSION
                ]
                self.assertEqual(
                    [(r.log_low, r.log_high) for r in bands],
                    [(-12.0 - 4.0 * index, -8.5 - 4.0 * index) for index in range(expected)],
                )

    def test_no_sign_change_in_the_whole_domain_is_a_boundary_after_three_expansions(self) -> None:
        for log_root in (22.0, -22.0):
            recorder, result = self.run_with_root(log_root)
            with self.subTest(log_root=log_root):
                self.assertIs(result, ProfileRefusal.BOUNDARY)
                self.assertEqual(recorder.roles(), ["grid", "expansion", "expansion", "expansion"])

    def test_an_expansion_grid_has_eight_points_each(self) -> None:
        recorder, _ = self.run_with_root(18.0)
        for request in recorder.requests:
            if request.role is not PassRole.ITERATION:
                self.assertEqual(len(request.parameters), 8)

    def test_the_expansion_pass_beyond_the_budget_is_not_read(self) -> None:
        ledger = PassLedger(partitions(), 4, PlanBounds(3, 17))
        recorder = Recorder(np.array([0.0, -1.0]))
        root = math.exp(18.0)
        evaluate = synthetic(lambda k: (root - k) / (k * k), 0.0)

        def read(request: PassRequest) -> list[ProfileSums]:
            list(
                ledger.begin(
                    request.role, len(request.parameters), request.log_low, request.log_high
                )
            )
            recorder.requests.append(request)
            return evaluate(request)

        with self.assertRaises(PassBudgetError) as caught:
            solve_profile(read, count=2, mean_w=0.0)
        self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_EXCEEDED)
        self.assertEqual(len(recorder.requests), 4)


if __name__ == "__main__":
    unittest.main()
