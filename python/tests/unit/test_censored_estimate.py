"""The estimates of censored and truncated samples: closed forms, the profile, the Newton route.

Each route is checked by value (the closed form of the exponential, the stationary point of the
left-truncated Weibull profile against the two-parameter Newton run on the same data) and by
its facts (what is claimed about the point, how many stationary points the profile saw), and
every failure code is produced by the smallest sample that has it.
"""

from __future__ import annotations

import math
import unittest
from unittest import mock

from veridist import (
    ExactLifetime,
    ExactValue,
    IntervalCensoredLifetime,
    IntervalCensoredValue,
    LeftCensoredLifetime,
    LeftCensoredValue,
    RightCensoredLifetime,
    RightCensoredValue,
)
from veridist.families import _censored_fit as censored
from veridist.families._censored_fit import CensoredEstimate, CensoredRefusal, estimate
from veridist.families._censored_rows import build_rows, pilot
from veridist.families._fit_facts import OptimumKind, summarize_observations
from veridist.families.registry import FamilyId


def run(family: FamilyId, sample: list) -> CensoredEstimate | CensoredRefusal:
    return estimate(family, sample, summarize_observations(sample))


class ExponentialClosedFormTests(unittest.TestCase):
    def test_the_rate_is_events_over_total_time_minus_entry_times(self) -> None:
        sample = [
            ExactLifetime(3.0, truncation_lower=1.0),
            ExactLifetime(5.0, truncation_lower=2.0),
            RightCensoredLifetime(6.0, truncation_lower=1.0),
            RightCensoredLifetime(4.0),
        ]
        found = run(FamilyId.EXPONENTIAL, sample)
        assert isinstance(found, CensoredEstimate)
        self.assertEqual(found.total_time, 3 + 5 + 6 + 4 - 1 - 2 - 1)
        self.assertEqual(found.parameters, (2 / 14,))
        self.assertEqual(found.log_likelihood, 2 * math.log(2 / 14) - (2 / 14) * 14)
        self.assertIs(found.optimum, OptimumKind.CLOSED_FORM)
        self.assertEqual(found.diagnostics.method, "closed_form")
        self.assertEqual((found.diagnostics.passes, found.diagnostics.iterations), (0, 0))

    def test_the_two_sums_have_one_rounding(self) -> None:
        # 1e16 + 1 - 1e16 is 1 with one rounding, 0 or 2 with two.
        sample = [
            ExactLifetime(1e16 + 2.0, truncation_lower=1e16),
            ExactLifetime(1.0),
        ]
        found = run(FamilyId.EXPONENTIAL, sample)
        assert isinstance(found, CensoredEstimate)
        self.assertEqual(found.total_time, 3.0)

    def test_the_summarised_block_enters_the_total(self) -> None:
        sample = [ExactLifetime(2.0, truncation_lower=1.0)]
        found = estimate(
            FamilyId.EXPONENTIAL,
            sample,
            summarize_observations(sample),
            plain_count=3,
            plain_events=2,
            plain_total=9.0,
        )
        assert isinstance(found, CensoredEstimate)
        self.assertEqual(found.total_time, 10.0)
        self.assertEqual(found.parameters, (3 / 10,))

    def test_a_window_below_zero_or_at_zero_is_no_entry_time(self) -> None:
        sample = [ExactLifetime(4.0, truncation_upper=9.0)]
        # right truncation: not the closed form
        self.assertFalse(
            censored.exponential_closed_form_applies(build_rows(FamilyId.EXPONENTIAL, sample))
        )
        left = [ExactLifetime(4.0, truncation_lower=1.0), ExactLifetime(3.0)]
        self.assertTrue(
            censored.exponential_closed_form_applies(build_rows(FamilyId.EXPONENTIAL, left))
        )
        found = run(FamilyId.EXPONENTIAL, left)
        assert isinstance(found, CensoredEstimate)
        self.assertEqual(found.total_time, 6.0)

    def test_the_closed_form_does_not_take_left_or_interval_rows(self) -> None:
        for extra in (LeftCensoredLifetime(2.0), IntervalCensoredLifetime(1.0, 2.0)):
            rows = build_rows(
                FamilyId.EXPONENTIAL, [ExactLifetime(4.0, truncation_lower=1.0), extra]
            )
            self.assertFalse(censored.exponential_closed_form_applies(rows))

    def test_exact_times_at_their_entry_times_are_an_unbounded_likelihood(self) -> None:
        sample = [
            ExactLifetime(2.0, truncation_lower=2.0),
            ExactLifetime(3.0, truncation_lower=3.0),
        ]
        self.assertEqual(
            run(FamilyId.EXPONENTIAL, sample), CensoredRefusal("UNBOUNDED_LIKELIHOOD", 0.0)
        )

    def test_a_total_that_overflows_is_a_numerical_overflow(self) -> None:
        sample = [ExactLifetime(1.7e308, truncation_lower=1.0), ExactLifetime(1.7e308)]
        self.assertEqual(run(FamilyId.EXPONENTIAL, sample), CensoredRefusal("NUMERICAL_OVERFLOW"))
        tiny = [ExactLifetime(1e-320, truncation_lower=1e-321)] * 2 + [ExactLifetime(5e-324)]
        self.assertEqual(run(FamilyId.EXPONENTIAL, tiny), CensoredRefusal("NUMERICAL_OVERFLOW"))


class FailureCodeTests(unittest.TestCase):
    def test_every_row_right_censored_is_no_observed_events(self) -> None:
        sample = [RightCensoredLifetime(2.0, truncation_lower=1.0), RightCensoredLifetime(3.0)]
        for family in (FamilyId.WEIBULL_MIN, FamilyId.LOGNORMAL, FamilyId.GAMMA):
            self.assertEqual(run(family, sample).code, "NO_OBSERVED_EVENTS")
        found = run(FamilyId.EXPONENTIAL, sample)
        self.assertEqual(found, CensoredRefusal("NO_OBSERVED_EVENTS", 5.0))
        values = [RightCensoredValue(2.0, truncation_lower=1.0), RightCensoredValue(3.0)]
        for family in (FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT):
            self.assertEqual(run(family, values).code, "NO_OBSERVED_EVENTS")

    def test_a_left_or_interval_row_bounds_the_value_from_above(self) -> None:
        for extra in (LeftCensoredLifetime(5.0), IntervalCensoredLifetime(3.0, 5.0)):
            sample = [RightCensoredLifetime(2.0), RightCensoredLifetime(1.0), extra]
            for family in (FamilyId.EXPONENTIAL, FamilyId.LOGNORMAL):
                found = run(family, sample)
                self.assertNotEqual(getattr(found, "code", None), "NO_OBSERVED_EVENTS")

    def test_an_overflowing_total_of_right_censored_times_keeps_the_code(self) -> None:
        sample = [
            RightCensoredLifetime(1.7e308, truncation_lower=1.0),
            RightCensoredLifetime(1.7e308),
        ]
        self.assertEqual(run(FamilyId.WEIBULL_MIN, sample).code, "NO_OBSERVED_EVENTS")
        self.assertEqual(run(FamilyId.EXPONENTIAL, sample), CensoredRefusal("NUMERICAL_OVERFLOW"))

    def test_a_time_at_or_below_zero_is_an_invalid_support(self) -> None:
        for family in (FamilyId.WEIBULL_MIN, FamilyId.LOGNORMAL, FamilyId.GAMMA):
            for bad in (ExactLifetime(0.0), RightCensoredLifetime(0.0)):
                sample = [ExactLifetime(2.0), LeftCensoredLifetime(3.0), bad]
                with self.subTest(family=family, bad=bad):
                    self.assertEqual(run(family, sample), CensoredRefusal("INVALID_SUPPORT"))
            # an interval from zero and a window from zero are fine
            ok = [
                ExactLifetime(2.0),
                IntervalCensoredLifetime(0.0, 3.0),
                ExactLifetime(4.0, truncation_lower=0.0),
            ]
            self.assertFalse(censored.has_invalid_support(family, ok))
        self.assertFalse(censored.has_invalid_support(FamilyId.EXPONENTIAL, [ExactLifetime(0.0)]))
        self.assertFalse(censored.has_invalid_support(FamilyId.NORMAL, [ExactValue(-1.0)]))

    def test_a_constant_pseudo_sample_is_degenerate_before_any_route(self) -> None:
        sample = [ExactLifetime(5.0, truncation_lower=1.0)] * 3
        for family in (FamilyId.WEIBULL_MIN, FamilyId.LOGNORMAL, FamilyId.GAMMA):
            self.assertEqual(run(family, sample), CensoredRefusal("DEGENERATE_SAMPLE"))
        for family in (FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT):
            self.assertEqual(
                run(family, [ExactValue(5.0), LeftCensoredValue(5.0)]),
                CensoredRefusal("DEGENERATE_SAMPLE"),
            )

    def test_a_flat_likelihood_is_never_an_estimate(self) -> None:
        sample = [LeftCensoredLifetime(5.0), LeftCensoredLifetime(6.0), LeftCensoredLifetime(7.0)]
        for family in (
            FamilyId.EXPONENTIAL,
            FamilyId.WEIBULL_MIN,
            FamilyId.LOGNORMAL,
            FamilyId.GAMMA,
        ):
            with self.subTest(family=family):
                self.assertEqual(run(family, sample), CensoredRefusal("OPTIMIZER_EXHAUSTED"))

    def test_an_overflowing_pilot_is_an_exhausted_optimizer(self) -> None:
        sample = [ExactValue(1.7e308), ExactValue(1.7e308), LeftCensoredValue(1.7e308)]
        self.assertEqual(run(FamilyId.NORMAL, sample), CensoredRefusal("OPTIMIZER_EXHAUSTED"))

    def test_a_standardisation_that_overflows_is_an_exhausted_optimizer(self) -> None:
        sample = [
            ExactValue(1.0),
            ExactValue(1.0 + 1e-10),
            LeftCensoredValue(1.0 + 3e-10, truncation_lower=-1e300),
        ]
        self.assertEqual(run(FamilyId.NORMAL, sample), CensoredRefusal("OPTIMIZER_EXHAUSTED"))

    def test_the_reducer_failing_at_the_estimate_is_an_exhausted_optimizer(self) -> None:
        sample = [ExactLifetime(2.0), ExactLifetime(3.0), LeftCensoredLifetime(4.0)]
        with mock.patch.object(censored, "_kernel_total", return_value=None):
            self.assertEqual(run(FamilyId.GAMMA, sample), CensoredRefusal("OPTIMIZER_EXHAUSTED"))

    def test_an_estimate_that_is_not_a_finite_positive_parameter_is_an_exhausted_optimizer(
        self,
    ) -> None:
        sample = [ExactLifetime(2.0)]
        rows = build_rows(FamilyId.WEIBULL_MIN, sample)
        facts = censored.FitDiagnostics(1, 1)
        for parameters in ((math.inf, 2.0), (1.0, 0.0), (-1.0, 2.0), (math.nan, 1.0)):
            found = censored._finish(
                FamilyId.WEIBULL_MIN, sample, rows, parameters, OptimumKind.GLOBAL_MAXIMUM, facts
            )
            self.assertEqual(found, CensoredRefusal("OPTIMIZER_EXHAUSTED"))
        normal = censored._finish(
            FamilyId.NORMAL,
            [ExactValue(2.0)],
            build_rows(FamilyId.NORMAL, [ExactValue(2.0)]),
            (-3.0, 2.0),
            OptimumKind.GLOBAL_MAXIMUM,
            facts,
        )
        self.assertIsInstance(normal, CensoredEstimate)
        self.assertEqual(
            censored._finish(
                FamilyId.NORMAL,
                [ExactValue(2.0)],
                build_rows(FamilyId.NORMAL, [ExactValue(2.0)]),
                (1.0, -2.0),
                OptimumKind.GLOBAL_MAXIMUM,
                facts,
            ),
            CensoredRefusal("OPTIMIZER_EXHAUSTED"),
        )


class ParameterMapTests(unittest.TestCase):
    def test_the_canonical_parameters_of_each_iterate(self) -> None:
        to = censored.to_parameters
        self.assertEqual(to(FamilyId.EXPONENTIAL, (2.0,), 0.0, 4.0), (0.5,))
        shape, scale = to(FamilyId.GAMMA, (math.log(3.0), math.log(2.0)), 0.0, 10.0)
        self.assertAlmostEqual(shape, 3.0, places=14)
        self.assertAlmostEqual(scale, 20.0, places=13)
        # (m, u) = (mu / sigma, 1 / sigma) in units centre + scale * z
        self.assertEqual(to(FamilyId.NORMAL, (1.0, 0.5), 3.0, 2.0), (3.0 + 2.0 * 2.0, 2.0 * 2.0))
        self.assertEqual(to(FamilyId.LOGNORMAL, (1.0, 0.5), 3.0, 2.0), (7.0, 4.0))
        shape, scale = to(FamilyId.WEIBULL_MIN, (1.0, 0.5), 3.0, 2.0)
        self.assertEqual(shape, 1.0 / 4.0)
        self.assertEqual(scale, math.exp(7.0))
        self.assertEqual(to(FamilyId.GUMBEL_RIGHT, (1.0, 0.5), 3.0, 2.0), (-7.0, 4.0))

    def test_the_names_and_the_positive_parameters(self) -> None:
        self.assertEqual(censored.PARAMETER_NAMES[FamilyId.GUMBEL_RIGHT], ("location", "scale"))
        self.assertEqual(censored.POSITIVE_PARAMETERS[FamilyId.WEIBULL_MIN], (0, 1))
        self.assertEqual(censored.POSITIVE_PARAMETERS[FamilyId.NORMAL], (1,))


class WeibullProfileTests(unittest.TestCase):
    sample = [
        ExactLifetime(2.2, truncation_lower=1.0),
        ExactLifetime(3.1, truncation_lower=1.5),
        ExactLifetime(4.7, truncation_lower=1.0),
        ExactLifetime(1.9),
        ExactLifetime(5.9, truncation_lower=2.0),
        RightCensoredLifetime(6.5, truncation_lower=1.0),
        RightCensoredLifetime(3.5),
    ]

    def test_the_constants_of_the_profile(self) -> None:
        self.assertEqual(censored.PROFILE_LOG_SHAPE_LIMIT, 20.0)
        self.assertEqual(censored.PROFILE_GRID_STEP, 0.25)
        self.assertEqual(censored.PROFILE_BISECTIONS, 48)

    def test_the_cells_the_profile_takes(self) -> None:
        applies = censored.weibull_profile_applies
        self.assertTrue(applies(build_rows(FamilyId.WEIBULL_MIN, self.sample)))
        no_window = [ExactLifetime(2.0), RightCensoredLifetime(3.0)]
        self.assertFalse(applies(build_rows(FamilyId.WEIBULL_MIN, no_window)))
        right = [ExactLifetime(2.0, truncation_upper=5.0), ExactLifetime(3.0, truncation_lower=1.0)]
        self.assertFalse(applies(build_rows(FamilyId.WEIBULL_MIN, right)))
        interval = [*self.sample, IntervalCensoredLifetime(1.0, 2.0)]
        self.assertFalse(applies(build_rows(FamilyId.WEIBULL_MIN, interval)))
        only_right_censored = [RightCensoredLifetime(3.0, truncation_lower=1.0)]
        self.assertFalse(applies(build_rows(FamilyId.WEIBULL_MIN, only_right_censored)))

    def test_the_profile_root_is_the_two_parameter_newton_maximum(self) -> None:
        found = run(FamilyId.WEIBULL_MIN, self.sample)
        assert isinstance(found, CensoredEstimate)
        self.assertEqual(found.diagnostics.method, "profile")
        self.assertEqual(found.diagnostics.stationary_points_seen, 1)
        self.assertEqual(found.diagnostics.distinct_stationary_points, 1)
        self.assertIs(found.optimum, OptimumKind.STATIONARY_POINT)
        rows = build_rows(FamilyId.WEIBULL_MIN, self.sample)
        newton = censored._newton(
            FamilyId.WEIBULL_MIN,
            self.sample,
            rows,
            summarize_observations(self.sample),
            pilot(rows),
        )
        assert isinstance(newton, CensoredEstimate)
        for a, b in zip(found.parameters, newton.parameters, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-8 * abs(b))
        self.assertAlmostEqual(found.log_likelihood, newton.log_likelihood, places=9)

    def test_the_score_is_the_derivative_of_the_profile_value(self) -> None:
        profile = censored._Profile(build_rows(FamilyId.WEIBULL_MIN, self.sample))
        for kappa in (-1.0, 0.3, 1.2):
            step = 1e-6
            difference = (profile.value(kappa + step) - profile.value(kappa - step)) / (2 * step)
            self.assertAlmostEqual(difference, profile.score(kappa), delta=1e-5)

    def test_the_profiled_scale_solves_the_scale_equation(self) -> None:
        profile = censored._Profile(build_rows(FamilyId.WEIBULL_MIN, self.sample))
        for kappa in (-0.5, 0.7):
            k = math.exp(kappa)
            scale = profile.scale(kappa)
            events = 5
            total = math.fsum(
                (o.time if hasattr(o, "time") else 0.0) ** k - (o.truncation_lower or 0.0) ** k
                for o in self.sample
            )
            self.assertAlmostEqual(scale**k, total / events, delta=1e-12 * total)

    def test_the_stationary_points_seen_are_counted_and_the_highest_is_kept(self) -> None:
        sample = [ExactLifetime(2.0, truncation_lower=1.0), ExactLifetime(3.0)]

        class TwoMaxima:
            def __init__(self, rows: object) -> None:
                pass

            @staticmethod
            def value(kappa: float) -> float:
                return -((kappa * kappa - 4.0) ** 2) + 0.1 * kappa

            @staticmethod
            def score(kappa: float) -> float:
                return -4.0 * kappa * (kappa * kappa - 4.0) + 0.1

            @staticmethod
            def scale(kappa: float) -> float:
                return 2.0

        rows = build_rows(FamilyId.WEIBULL_MIN, sample)
        with mock.patch.object(censored, "_Profile", TwoMaxima):
            found = censored._weibull_profile(sample, rows)
        assert isinstance(found, CensoredEstimate)
        self.assertEqual(found.diagnostics.stationary_points_seen, 3)
        self.assertEqual(found.diagnostics.distinct_stationary_points, 2)
        self.assertAlmostEqual(math.log(found.parameters[0]), 2.0, delta=0.01)
        self.assertEqual(found.parameters[1], 2.0)

    def test_a_profile_without_a_maximum_is_a_boundary_solution(self) -> None:
        sample = [
            ExactLifetime(3.0, truncation_lower=3.0),
            ExactLifetime(3.5, truncation_lower=3.5),
        ]
        self.assertEqual(run(FamilyId.WEIBULL_MIN, sample), CensoredRefusal("BOUNDARY_SOLUTION"))

    def test_an_estimate_that_is_not_finite_is_an_exhausted_optimizer(self) -> None:
        sample = [ExactLifetime(2.0, truncation_lower=1.0), ExactLifetime(3.0)]

        class Overflowing:
            def __init__(self, rows: object) -> None:
                pass

            @staticmethod
            def value(kappa: float) -> float:
                return -((kappa - 1.0) ** 2)

            @staticmethod
            def score(kappa: float) -> float:
                return -2.0 * (kappa - 1.0)

            @staticmethod
            def scale(kappa: float) -> float:
                return math.inf

        rows = build_rows(FamilyId.WEIBULL_MIN, sample)
        with mock.patch.object(censored, "_Profile", Overflowing):
            self.assertEqual(
                censored._weibull_profile(sample, rows), CensoredRefusal("OPTIMIZER_EXHAUSTED")
            )


class NewtonRouteTests(unittest.TestCase):
    def test_what_is_claimed_for_each_cell(self) -> None:
        interval = [ExactValue(1.0), ExactValue(2.5), IntervalCensoredValue(1.5, 3.0)]
        found = run(FamilyId.NORMAL, interval)
        assert isinstance(found, CensoredEstimate)
        self.assertIs(found.optimum, OptimumKind.GLOBAL_MAXIMUM)
        self.assertEqual(found.diagnostics.starts, 1)
        truncated = [ExactValue(1.0, truncation_lower=0.5), ExactValue(2.5), LeftCensoredValue(3.0)]
        found = run(FamilyId.NORMAL, truncated)
        assert isinstance(found, CensoredEstimate)
        self.assertIs(found.optimum, OptimumKind.STATIONARY_POINT)
        self.assertEqual(found.diagnostics.starts, 2)
        left = [
            ExactLifetime(1.0, truncation_lower=0.5),
            ExactLifetime(2.5),
            LeftCensoredLifetime(3.0),
        ]
        found = run(FamilyId.EXPONENTIAL, left)
        assert isinstance(found, CensoredEstimate)
        self.assertIs(found.optimum, OptimumKind.GLOBAL_MAXIMUM)
        right = [
            ExactLifetime(1.0, truncation_upper=5.5),
            ExactLifetime(2.5),
            LeftCensoredLifetime(3.0),
        ]
        found = run(FamilyId.EXPONENTIAL, right)
        assert isinstance(found, CensoredEstimate)
        self.assertIs(found.optimum, OptimumKind.STATIONARY_POINT)
        gamma = run(
            FamilyId.GAMMA, [ExactLifetime(1.0), ExactLifetime(2.5), LeftCensoredLifetime(3.0)]
        )
        assert isinstance(gamma, CensoredEstimate)
        self.assertIs(gamma.optimum, OptimumKind.STATIONARY_POINT)

    def test_the_total_time_of_an_iterative_exponential_is_the_pseudo_sample_total(self) -> None:
        sample = [ExactLifetime(2.0), LeftCensoredLifetime(3.0), IntervalCensoredLifetime(1.0, 5.0)]
        found = run(FamilyId.EXPONENTIAL, sample)
        assert isinstance(found, CensoredEstimate)
        self.assertEqual(found.total_time, 2.0 + 3.0 + 3.0)

    def test_the_reported_likelihood_is_the_librarys(self) -> None:
        from veridist import reduce_lifetime_log_likelihood_chunks

        sample = [ExactLifetime(2.0), LeftCensoredLifetime(3.0), IntervalCensoredLifetime(1.0, 5.0)]
        found = run(FamilyId.EXPONENTIAL, sample)
        assert isinstance(found, CensoredEstimate)
        total = reduce_lifetime_log_likelihood_chunks(
            FamilyId.EXPONENTIAL, [sample], rate=found.parameters[0]
        ).total_log_likelihood
        self.assertEqual(found.log_likelihood, total)

    def test_the_summarised_block_is_added_to_the_likelihood(self) -> None:
        sample = [LeftCensoredLifetime(3.0), IntervalCensoredLifetime(1.0, 5.0)]
        found = estimate(
            FamilyId.EXPONENTIAL,
            sample,
            summarize_observations(sample),
            plain_count=2,
            plain_events=2,
            plain_total=3.0,
        )
        assert isinstance(found, CensoredEstimate)
        whole = [ExactLifetime(1.0), ExactLifetime(2.0), *sample]
        reference = run(FamilyId.EXPONENTIAL, whole)
        assert isinstance(reference, CensoredEstimate)
        self.assertAlmostEqual(found.parameters[0], reference.parameters[0], delta=1e-12)
        self.assertAlmostEqual(found.log_likelihood, reference.log_likelihood, places=12)
        no_events = estimate(
            FamilyId.EXPONENTIAL,
            sample,
            summarize_observations(sample),
            plain_count=2,
            plain_events=0,
            plain_total=3.0,
        )
        assert isinstance(no_events, CensoredEstimate)
        censored_whole = [RightCensoredLifetime(1.0), RightCensoredLifetime(2.0), *sample]
        reference = run(FamilyId.EXPONENTIAL, censored_whole)
        assert isinstance(reference, CensoredEstimate)
        self.assertAlmostEqual(no_events.parameters[0], reference.parameters[0], delta=1e-12)
        self.assertAlmostEqual(no_events.log_likelihood, reference.log_likelihood, places=12)

    def test_a_likelihood_that_is_not_a_number_is_an_exhausted_optimizer(self) -> None:
        sample = [ExactLifetime(2.0), LeftCensoredLifetime(3.0), IntervalCensoredLifetime(1.0, 5.0)]
        rows = build_rows(
            FamilyId.EXPONENTIAL, sample, plain_count=1, plain_events=0, plain_total=1.0
        )
        with mock.patch.object(censored, "_kernel_total", return_value=math.inf):
            self.assertEqual(
                censored._finish(
                    FamilyId.EXPONENTIAL,
                    sample,
                    rows,
                    (0.5,),
                    OptimumKind.GLOBAL_MAXIMUM,
                    censored.FitDiagnostics(1, 1),
                ),
                CensoredRefusal("OPTIMIZER_EXHAUSTED"),
            )


class KernelTotalTests(unittest.TestCase):
    def test_a_total_that_cannot_be_formed_is_none(self) -> None:
        sample = [ExactLifetime(2.0)]
        good = censored._kernel_total(FamilyId.WEIBULL_MIN, sample, {"shape": 1.5, "scale": 2.0})
        self.assertIsNotNone(good)
        # parameters the family refuses
        self.assertIsNone(
            censored._kernel_total(FamilyId.WEIBULL_MIN, sample, {"shape": -1.0, "scale": 2.0})
        )
        # a term the kernels cannot evaluate (a gamma shape beyond their range)
        left = [LeftCensoredLifetime(2.0)]
        self.assertIsNone(
            censored._kernel_total(FamilyId.GAMMA, left, {"shape": 1e5, "scale": 1.0})
        )
        values = [ExactValue(2.0), LeftCensoredValue(3.0)]
        self.assertIsNotNone(
            censored._kernel_total(FamilyId.NORMAL, values, {"mu": 2.0, "sigma": 1.0})
        )


if __name__ == "__main__":
    unittest.main()
