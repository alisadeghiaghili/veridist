"""The rows of censored and truncated samples and the pilot: spaces, pseudo-sample, start.

The expected arrays are written out by hand for one row of each kind, in the ``y`` space of each
family (``x`` for the normal, the gamma and the exponential, ``log x`` for the lognormal and the
Weibull minimum, ``-x`` with reflected sets for the right Gumbel), and the pilot is checked
against exact sums computed here with ``fractions``.
"""

from __future__ import annotations

import math
import unittest
from fractions import Fraction

import numpy as np

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
from veridist.families import _censored_rows as rows_module
from veridist.families._censored_rows import build_rows, pilot, pseudo_values
from veridist.families.registry import FamilyId

INF = math.inf
EULER = 0.5772156649015329


def lifetimes() -> list:
    return [
        ExactLifetime(2.0),
        RightCensoredLifetime(3.0),
        LeftCensoredLifetime(4.0),
        IntervalCensoredLifetime(1.0, 5.0, truncation_lower=0.5, truncation_upper=9.0),
        IntervalCensoredLifetime(0.0, 2.0),
    ]


def values() -> list:
    return [
        ExactValue(2.0),
        RightCensoredValue(3.0),
        LeftCensoredValue(4.0),
        IntervalCensoredValue(1.0, 5.0, truncation_lower=0.5, truncation_upper=9.0),
        ExactValue(-1.0, truncation_upper=0.0),
    ]


def same(first: np.ndarray, expected: list[float]) -> None:
    np.testing.assert_array_equal(first, np.array(expected))


class SpaceTests(unittest.TestCase):
    def test_the_gamma_and_the_exponential_keep_x_and_open_the_bottom_of_the_support(self) -> None:
        for family in (FamilyId.GAMMA, FamilyId.EXPONENTIAL):
            with self.subTest(family=family):
                rows = build_rows(family, lifetimes())
                self.assertEqual(rows.exact.tolist(), [True, False, False, False, False])
                np.testing.assert_array_equal(
                    np.nan_to_num(rows.yv, nan=-7.0), [2.0, -7.0, -7.0, -7.0, -7.0]
                )
                same(rows.lo, [-INF, 3.0, -INF, 1.0, -INF])
                same(rows.hi, [INF, INF, 4.0, 5.0, 2.0])
                same(rows.wl, [-INF, -INF, -INF, 0.5, -INF])
                same(rows.wh, [INF, INF, INF, 9.0, INF])
                self.assertEqual((rows.count, rows.explicit_count), (5, 5))

    def test_the_lognormal_and_the_weibull_work_on_the_logarithm(self) -> None:
        for family in (FamilyId.LOGNORMAL, FamilyId.WEIBULL_MIN):
            with self.subTest(family=family):
                rows = build_rows(family, lifetimes())
                self.assertEqual(rows.yv[0], math.log(2.0))
                self.assertTrue(np.isnan(rows.yv[1:]).all())
                same(rows.lo, [-INF, math.log(3.0), -INF, 0.0, -INF])
                same(rows.hi, [INF, INF, math.log(4.0), math.log(5.0), math.log(2.0)])
                same(rows.wl, [-INF, -INF, -INF, math.log(0.5), -INF])
                same(rows.wh, [INF, INF, INF, math.log(9.0), INF])

    def test_the_normal_keeps_the_values(self) -> None:
        rows = build_rows(FamilyId.NORMAL, values())
        self.assertEqual(rows.exact.tolist(), [True, False, False, False, True])
        np.testing.assert_array_equal(np.nan_to_num(rows.yv, nan=-7.0), [2.0, -7, -7, -7, -1.0])
        same(rows.lo, [-INF, 3.0, -INF, 1.0, -INF])
        same(rows.hi, [INF, INF, 4.0, 5.0, INF])
        same(rows.wl, [-INF, -INF, -INF, 0.5, -INF])
        same(rows.wh, [INF, INF, INF, 9.0, 0.0])

    def test_the_right_gumbel_is_reflected(self) -> None:
        rows = build_rows(FamilyId.GUMBEL_RIGHT, values())
        np.testing.assert_array_equal(np.nan_to_num(rows.yv, nan=-7.0), [-2.0, -7, -7, -7, 1.0])
        # right-censored (3, inf) -> (-inf, -3]; left-censored (-inf, 4] -> [-4, inf);
        # interval (1, 5] -> [-5, -1); windows (0.5, 9] -> [-9, -0.5), (-inf, 0] -> [0, inf).
        same(rows.lo, [-INF, -INF, -4.0, -5.0, -INF])
        same(rows.hi, [INF, -3.0, INF, -1.0, INF])
        same(rows.wl, [-INF, -INF, -INF, -9.0, 0.0])
        same(rows.wh, [INF, INF, INF, -0.5, INF])

    def test_a_lower_window_end_at_the_bottom_of_the_support_is_no_window(self) -> None:
        rows = build_rows(FamilyId.WEIBULL_MIN, [ExactLifetime(2.0, truncation_upper=5.0)])
        self.assertEqual(rows.wl.tolist(), [-INF])
        self.assertEqual(rows.wh.tolist(), [math.log(5.0)])

    def test_the_summarised_block_is_counted(self) -> None:
        rows = build_rows(
            FamilyId.EXPONENTIAL,
            lifetimes(),
            plain_count=7,
            plain_events=3,
            plain_total=12.5,
        )
        self.assertEqual((rows.count, rows.explicit_count), (12, 5))
        self.assertEqual((rows.plain_count, rows.plain_events, rows.plain_total), (7, 3, 12.5))


class PseudoSampleTests(unittest.TestCase):
    def test_exact_at_the_value_censored_at_the_point_intervals_at_the_midpoint(self) -> None:
        rows = build_rows(FamilyId.NORMAL, values())
        same(pseudo_values(rows), [2.0, 3.0, 4.0, 3.0, -1.0])

    def test_an_interval_from_the_bottom_of_the_support_sits_at_its_upper_bound(self) -> None:
        rows = build_rows(FamilyId.WEIBULL_MIN, lifetimes())
        expected = [
            math.log(2.0),
            math.log(3.0),
            math.log(4.0),
            0.5 * (0.0 + math.log(5.0)),
            math.log(2.0),
        ]
        same(pseudo_values(rows), expected)

    def test_the_gumbel_pseudo_sample_is_the_negated_one(self) -> None:
        rows = build_rows(FamilyId.GUMBEL_RIGHT, values())
        same(pseudo_values(rows), [-2.0, -3.0, -4.0, -3.0, 1.0])


def exact_mean(numbers: list[float]) -> float:
    return float(sum(Fraction(x) for x in numbers) / len(numbers))


class PilotTests(unittest.TestCase):
    def test_the_exponential_is_standardised_by_the_mean_and_starts_at_one(self) -> None:
        rows = build_rows(FamilyId.EXPONENTIAL, lifetimes())
        found = pilot(rows)
        self.assertIsNone(found.failure)
        # pseudo-sample 2, 3, 4, 3 (midpoint of (1, 5]), 1 (the upper end of (-inf, 2]... the
        # lower end is the bottom, so the point is the upper bound 2.0).
        pseudo = [2.0, 3.0, 4.0, 3.0, 2.0]
        self.assertEqual(found.scale, exact_mean(pseudo))
        self.assertEqual(found.centre, 0.0)
        self.assertEqual(found.start, (1.0,))
        self.assertEqual(found.pseudo_total, 14.0)

    def test_the_summarised_block_enters_the_mean(self) -> None:
        rows = build_rows(
            FamilyId.EXPONENTIAL,
            [ExactLifetime(2.0, truncation_lower=1.0)],
            plain_count=3,
            plain_events=3,
            plain_total=10.0,
        )
        found = pilot(rows)
        self.assertEqual(found.scale, 3.0)
        self.assertEqual(found.pseudo_total, 12.0)

    def test_the_location_scale_start_is_the_moment_start_of_the_pseudo_sample(self) -> None:
        sample = values()
        found = pilot(build_rows(FamilyId.NORMAL, sample))
        pseudo = [2.0, 3.0, 4.0, 3.0, -1.0]
        centre = float(sum(Fraction(x) for x in pseudo) / 5)
        scale = math.sqrt(float(sum((Fraction(x) - Fraction(centre)) ** 2 for x in pseudo) / 5))
        self.assertAlmostEqual(found.centre, centre, places=15)
        self.assertAlmostEqual(found.scale, scale, places=14)
        self.assertEqual(found.start, (0.0, 1.0))

    def test_the_extreme_value_families_start_at_the_euler_moment_estimate(self) -> None:
        sigma = math.sqrt(6.0) / math.pi
        for family, sample in (
            (FamilyId.GUMBEL_RIGHT, values()),
            (FamilyId.WEIBULL_MIN, lifetimes()),
        ):
            with self.subTest(family=family):
                found = pilot(build_rows(family, sample))
                self.assertEqual(found.start, (EULER, 1.0 / sigma))

    def test_the_gamma_start_is_minkas_shape_in_log_coordinates(self) -> None:
        rows = build_rows(FamilyId.GAMMA, lifetimes())
        found = pilot(rows)
        pseudo = np.array([2.0, 3.0, 4.0, 3.0, 2.0])
        mean = pseudo.mean()
        s = -math.fsum(np.log(pseudo / mean).tolist()) / 5
        shape = (3 - s + math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
        self.assertAlmostEqual(found.scale, mean, places=14)
        self.assertAlmostEqual(found.start[0], math.log(shape), places=13)
        self.assertEqual(found.start[1], -found.start[0])

    def test_a_constant_pseudo_sample_is_degenerate_for_every_family_but_the_exponential(
        self,
    ) -> None:
        for family, sample in (
            (FamilyId.NORMAL, [ExactValue(5.0), LeftCensoredValue(5.0)]),
            (FamilyId.GUMBEL_RIGHT, [ExactValue(5.0), ExactValue(5.0)]),
            (FamilyId.WEIBULL_MIN, [ExactLifetime(5.0), LeftCensoredLifetime(5.0)]),
            (FamilyId.LOGNORMAL, [ExactLifetime(5.0), IntervalCensoredLifetime(0.0, 5.0)]),
            (FamilyId.GAMMA, [ExactLifetime(5.0), LeftCensoredLifetime(5.0)]),
        ):
            with self.subTest(family=family):
                found = pilot(build_rows(family, sample))
                self.assertEqual(found.failure, "DEGENERATE_SAMPLE")
        exponential = pilot(
            build_rows(FamilyId.EXPONENTIAL, [ExactLifetime(5.0), LeftCensoredLifetime(5.0)])
        )
        self.assertIsNone(exponential.failure)

    def test_the_test_is_the_equality_of_the_extremes_not_a_variance_test(self) -> None:
        # Seven equal values whose mean is not exactly the value: the variance is positive by
        # rounding, yet the extremes are equal.
        sample = [ExactValue(0.1)] * 7
        found = pilot(build_rows(FamilyId.NORMAL, sample))
        self.assertEqual(found.failure, "DEGENERATE_SAMPLE")
        near = [ExactValue(0.1), ExactValue(math.nextafter(0.1, 1.0))]
        self.assertIsNone(pilot(build_rows(FamilyId.NORMAL, near)).failure)

    def test_a_pilot_that_overflows_is_an_exhausted_optimizer(self) -> None:
        huge = [ExactLifetime(1e308), ExactLifetime(1e308), LeftCensoredLifetime(1e308)]
        self.assertEqual(
            pilot(build_rows(FamilyId.EXPONENTIAL, huge)).failure, "OPTIMIZER_EXHAUSTED"
        )
        self.assertEqual(pilot(build_rows(FamilyId.GAMMA, huge)).failure, "OPTIMIZER_EXHAUSTED")


class ScalingTests(unittest.TestCase):
    def test_the_gamma_and_the_exponential_are_divided_by_the_scale(self) -> None:
        for family in (FamilyId.GAMMA, FamilyId.EXPONENTIAL):
            rows = build_rows(
                family,
                [ExactLifetime(4.0), IntervalCensoredLifetime(2.0, 6.0, truncation_lower=1.0)],
            ).scaled(0.0, 2.0)
            self.assertEqual(rows.yv[0], 2.0)
            same(rows.lo, [-INF, 1.0])
            same(rows.hi, [INF, 3.0])
            same(rows.wl, [-INF, 0.5])

    def test_the_summarised_total_is_scaled_only_for_the_exponential(self) -> None:
        rows = build_rows(
            FamilyId.EXPONENTIAL,
            [ExactLifetime(4.0, truncation_lower=1.0)],
            plain_count=2,
            plain_events=1,
            plain_total=8.0,
        )
        self.assertEqual(rows.scaled(0.0, 2.0).plain_total, 4.0)

    def test_the_location_scale_families_are_centred_and_scaled(self) -> None:
        rows = build_rows(FamilyId.NORMAL, [ExactValue(4.0), RightCensoredValue(8.0)]).scaled(
            2.0, 4.0
        )
        self.assertEqual(rows.yv[0], 0.5)
        self.assertEqual(rows.lo[1], 1.5)
        self.assertEqual(rows.hi[1], INF)

    def test_a_standardisation_that_overflows_is_refused(self) -> None:
        rows = build_rows(FamilyId.NORMAL, [ExactValue(4.0), RightCensoredValue(8.0)])
        self.assertIsNotNone(rows.scaled(0.0, 2.0))
        self.assertIsNone(rows.scaled(0.0, 1e-320))
        window = build_rows(FamilyId.NORMAL, [ExactValue(4.0, truncation_lower=1.0)])
        self.assertIsNotNone(window.scaled(0.0, 2.0))
        # A finite end that overflows would turn into "no bound": refused, not silently opened.
        far = build_rows(FamilyId.NORMAL, [ExactValue(4.0, truncation_lower=-1e300)])
        self.assertIsNone(far.scaled(0.0, 1e-20))
        self.assertIsNotNone(far.scaled(0.0, 1.0))
        self.assertIsNone(window.scaled(0.0, 0.0))
        block = build_rows(
            FamilyId.EXPONENTIAL,
            [ExactLifetime(4.0, truncation_lower=1.0)],
            plain_count=1,
            plain_events=1,
            plain_total=1e300,
        )
        self.assertIsNone(block.scaled(0.0, 1e-300))
        self.assertIsNotNone(block.scaled(0.0, 1.0))


class ConstantsTests(unittest.TestCase):
    def test_the_family_sets(self) -> None:
        self.assertEqual(
            rows_module.LOCATION_SCALE_FAMILIES,
            {FamilyId.NORMAL, FamilyId.LOGNORMAL, FamilyId.WEIBULL_MIN, FamilyId.GUMBEL_RIGHT},
        )
        self.assertEqual(rows_module.LOG_SPACE_FAMILIES, {FamilyId.LOGNORMAL, FamilyId.WEIBULL_MIN})


class DegenerateScaleTests(unittest.TestCase):
    def test_a_pseudo_sample_whose_variance_underflows_is_degenerate(self) -> None:
        sample = [ExactValue(1e-200), ExactValue(2e-200)]
        found = pilot(build_rows(FamilyId.NORMAL, sample))
        self.assertEqual(found.failure, "DEGENERATE_SAMPLE")

    def test_an_exponential_pseudo_sample_of_zeros_has_no_scale(self) -> None:
        sample = [ExactLifetime(0.0, truncation_upper=5.0)] * 2
        found = pilot(build_rows(FamilyId.EXPONENTIAL, sample))
        self.assertEqual(found.failure, "OPTIMIZER_EXHAUSTED")


if __name__ == "__main__":
    unittest.main()
