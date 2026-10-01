"""Capability-boundary contracts for v1 censoring and weight semantics."""

from __future__ import annotations

import unittest

from veridist.domain.lifetimes import ExactLifetime


class V1CensoringWeightTests(unittest.TestCase):
    def test_frequency_weights_equal_explicit_replication(self) -> None:
        from veridist.families.weibull import fit_weibull

        weighted = fit_weibull(
            (ExactLifetime(1.0), ExactLifetime(4.0)), frequency_weights=(2, 1)
        )
        replicated = fit_weibull(
            (ExactLifetime(1.0), ExactLifetime(1.0), ExactLifetime(4.0))
        )
        self.assertAlmostEqual(weighted.shape, replicated.shape, places=12)
        self.assertAlmostEqual(weighted.scale, replicated.scale, places=12)
        self.assertAlmostEqual(weighted.log_likelihood, replicated.log_likelihood, places=12)

    def test_frequency_weights_reject_non_integral_negative_and_misaligned_values(self) -> None:
        from veridist.families.weibull import fit_weibull

        observations = (ExactLifetime(1.0), ExactLifetime(2.0))
        for weights, expected in (
            ((1,), ValueError),
            ((1, -1), TypeError),
            ((1, 0.5), TypeError),
            ((True, 1), TypeError),
        ):
            with self.subTest(weights=weights):
                with self.assertRaises(expected):
                    fit_weibull(observations, frequency_weights=weights)

    def test_invalid_observation_and_optimizer_bounds_are_rejected(self) -> None:
        from veridist.families._reliability import bounded_maximize
        from veridist.families.weibull import fit_weibull

        with self.assertRaises(TypeError):
            fit_weibull((object(),))
        with self.assertRaises(ValueError):
            bounded_maximize(lambda value: value, lower=1.0, upper=1.0)

    def test_bounded_maximize_flags_an_interior_point_as_not_at_the_boundary(self) -> None:
        from veridist.families._reliability import bounded_maximize

        point, value, at_boundary = bounded_maximize(
            lambda x: -((x - 2.0) ** 2), lower=-10.0, upper=10.0
        )
        self.assertAlmostEqual(point, 2.0, places=9)
        self.assertAlmostEqual(value, 0.0, places=12)
        self.assertFalse(at_boundary)

    def test_bounded_maximize_flags_a_monotonic_objective_as_at_the_boundary(self) -> None:
        from veridist.families._reliability import bounded_maximize

        point, _, at_boundary = bounded_maximize(lambda x: x, lower=-1.0, upper=1.0)
        self.assertAlmostEqual(point, 1.0, places=6)
        self.assertTrue(at_boundary)

    def test_expand_bracket_finds_an_interior_optimum_without_widening(self) -> None:
        from veridist.families._reliability import expand_bracket

        point, value, boundary = expand_bracket(
            lambda x: -((x - 2.0) ** 2),
            lower=-1.0,
            upper=1.0,
            hard_lower=-1000.0,
            hard_upper=1000.0,
        )
        self.assertAlmostEqual(point, 2.0, places=6)
        self.assertAlmostEqual(value, 0.0, places=10)
        self.assertFalse(boundary)

    def test_expand_bracket_widens_toward_an_interior_optimum_beyond_the_start(self) -> None:
        from veridist.families._reliability import expand_bracket

        # The optimum (50.0) lies well outside the starting bracket but
        # inside the hard limit: expansion must still find it, not just
        # report a boundary failure at the starting bracket's edge.
        point, _, boundary = expand_bracket(
            lambda x: -((x - 50.0) ** 2),
            lower=-1.0,
            upper=1.0,
            hard_lower=-1000.0,
            hard_upper=1000.0,
        )
        self.assertAlmostEqual(point, 50.0, places=3)
        self.assertFalse(boundary)

    def test_expand_bracket_widens_the_lower_side_toward_an_interior_optimum(self) -> None:
        from veridist.families._reliability import expand_bracket

        # The optimum (-50.0) lies below the starting bracket but above the
        # hard lower limit: the lower side must be the one that widens.
        point, _, boundary = expand_bracket(
            lambda x: -((x + 50.0) ** 2),
            lower=-1.0,
            upper=1.0,
            hard_lower=-1000.0,
            hard_upper=1000.0,
        )
        self.assertAlmostEqual(point, -50.0, places=3)
        self.assertFalse(boundary)

    def test_expand_bracket_reports_a_boundary_solution_at_the_hard_limit(self) -> None:
        from veridist.families._reliability import expand_bracket

        # A monotonically increasing objective has no interior maximum: even
        # after widening up to the hard limit, the result stays pinned to
        # the upper bound.
        point, _, boundary = expand_bracket(
            lambda x: x, lower=-1.0, upper=1.0, hard_lower=-10.0, hard_upper=10.0
        )
        self.assertAlmostEqual(point, 10.0, places=6)
        self.assertTrue(boundary)

    def test_expand_bracket_stops_after_its_expansion_budget(self) -> None:
        from unittest import mock

        from veridist.families import _reliability

        # With one expansion allowed, a monotonic objective is still on a bound
        # after the single widening, far inside the hard limit; the search must
        # stop and report a boundary solution instead of widening forever.
        with mock.patch.object(_reliability, "_MAX_BRACKET_EXPANSIONS", 1):
            point, _, boundary = _reliability.expand_bracket(
                lambda x: x, lower=-1.0, upper=1.0, hard_lower=-1e6, hard_upper=1e6
            )
        self.assertTrue(boundary)
        self.assertAlmostEqual(point, 1.0, delta=1e-6)

    def test_expand_bracket_rejects_hard_limits_that_do_not_contain_the_start(self) -> None:
        from veridist.families._reliability import expand_bracket

        with self.assertRaises(ValueError):
            expand_bracket(lambda x: x, lower=-1.0, upper=1.0, hard_lower=0.0, hard_upper=1.0)

    def test_analytic_weights_and_undeclared_censoring_fail_loudly(self) -> None:
        from veridist.engine.errors import CapabilityError
        from veridist.families.weibull import fit_weibull

        with self.assertRaisesRegex(CapabilityError, "ANALYTIC_WEIGHTS_UNSUPPORTED"):
            fit_weibull((ExactLifetime(1.0),), analytic_weights=(0.5,))
        with self.assertRaisesRegex(CapabilityError, "INTERVAL_CENSORING_UNSUPPORTED"):
            fit_weibull(((1.0, 2.0),), censoring="interval")
        with self.assertRaisesRegex(CapabilityError, "LEFT_CENSORING_UNSUPPORTED"):
            fit_weibull(((1.0, 2.0),), censoring="left")
        with self.assertRaisesRegex(CapabilityError, "TRUNCATION_UNSUPPORTED"):
            fit_weibull((ExactLifetime(1.0),), truncation=(0.5, None))
        with self.assertRaises(ValueError):
            fit_weibull((ExactLifetime(1.0),), censoring="unknown")
        with self.assertRaises(TypeError):
            CapabilityError("ANALYTIC_WEIGHTS_UNSUPPORTED")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
