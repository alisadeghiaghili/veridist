"""The per-row terms of censored and truncated rows: gradients, Hessians and the likelihood.

Three independent checks for every family, on a sample with all four kinds and every window:
the gradient is the central difference of the log-likelihood, the Hessian that of the gradient,
and the differences of the log-likelihood between two parameter vectors equal those of the
library's own log-likelihood entry points (which share no code with the derivative terms), so
that the terms are the likelihood and not just something with consistent derivatives.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from tests.reference import censored_support as support
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
from veridist.families._censored_fit import PARAMETER_NAMES, to_parameters
from veridist.families._censored_rows import build_rows
from veridist.families._censored_terms import (
    GAMMA_INFORMATION_STEP,
    GAMMA_OPTIMIZER_STEP,
    exact_sums,
    exponential_block,
    exponential_terms,
    gamma_terms,
    location_scale_terms,
    terms_function,
    theta_sums,
)
from veridist.families.registry import FamilyId

WINDOWS = (None, (1.0, math.inf), (-math.inf, 9.0), (1.0, 9.0))


def lifetime_sample() -> list:
    out = []
    for index, window in enumerate(WINDOWS * 2):
        keywords = (
            {} if window is None else {"truncation_lower": window[0], "truncation_upper": window[1]}
        )
        keywords = {k: v for k, v in keywords.items() if not math.isinf(v)}
        shift = 0.15 * index
        out += [
            ExactLifetime(2.0 + shift, **keywords),
            RightCensoredLifetime(1.5 + shift, **keywords),
            LeftCensoredLifetime(4.0 + shift, **keywords),
            IntervalCensoredLifetime(2.5 + shift, 4.5 + shift, **keywords),
        ]
    return out


def value_sample() -> list:
    out = []
    for index, window in enumerate(WINDOWS * 2):
        keywords = (
            {}
            if window is None
            else {"truncation_lower": window[0] - 3.0, "truncation_upper": window[1]}
        )
        keywords = {k: v for k, v in keywords.items() if not math.isinf(v)}
        shift = 0.15 * index
        out += [
            ExactValue(2.0 + shift, **keywords),
            RightCensoredValue(1.5 + shift, **keywords),
            LeftCensoredValue(4.0 + shift, **keywords),
            IntervalCensoredValue(2.5 + shift, 4.5 + shift, **keywords),
        ]
    return out


SAMPLES = {
    FamilyId.EXPONENTIAL: (lifetime_sample, (0.4,), (0.5,)),
    FamilyId.GAMMA: (lifetime_sample, (0.5, 0.3), (0.7, 0.1)),
    FamilyId.WEIBULL_MIN: (lifetime_sample, (0.3, 0.8), (0.5, 1.1)),
    FamilyId.LOGNORMAL: (lifetime_sample, (0.3, 0.8), (0.5, 1.1)),
    FamilyId.NORMAL: (value_sample, (0.3, 0.4), (0.8, 0.5)),
    FamilyId.GUMBEL_RIGHT: (value_sample, (0.3, 0.4), (0.8, 0.5)),
}
#: Relative tolerance of the derivative checks by family (the gamma shape derivatives are
#: finite differences themselves).
DERIVATIVE_TOLERANCE = {FamilyId.GAMMA: 2e-5}


def total_terms(family: FamilyId, rows, theta, **keywords) -> list[float]:
    function = terms_function(family)
    return exact_sums(function(rows, theta, **keywords))


class DerivativeTests(unittest.TestCase):
    def test_the_gradient_and_the_hessian_are_the_differences_of_the_likelihood(self) -> None:
        position = {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 2}
        for family, (sample, theta, _) in SAMPLES.items():
            with self.subTest(family=family):
                rows = build_rows(family, sample())
                dimension = len(theta)
                tolerance = DERIVATIVE_TOLERANCE.get(family, 2e-7)
                step = 1e-5
                base = total_terms(family, rows, theta)
                for i in range(dimension):
                    up, down = list(theta), list(theta)
                    up[i] += step
                    down[i] -= step
                    s_up = total_terms(family, rows, tuple(up))
                    s_down = total_terms(family, rows, tuple(down))
                    gradient = (s_up[0] - s_down[0]) / (2 * step)
                    self.assertLess(
                        abs(gradient - base[1 + i]), 10 * tolerance * max(1.0, abs(base[1 + i]))
                    )
                    for j in range(dimension):
                        stored = base[1 + dimension + (0 if dimension == 1 else position[(i, j)])]
                        difference = (s_up[1 + j] - s_down[1 + j]) / (2 * step)
                        self.assertLess(
                            abs(difference - stored), 100 * tolerance * max(1.0, abs(stored))
                        )

    def test_the_value_only_form_of_the_gamma_equals_the_full_one(self) -> None:
        rows = build_rows(FamilyId.GAMMA, lifetime_sample())
        full = gamma_terms(rows, (0.5, 0.3))
        value = gamma_terms(rows, (0.5, 0.3), value_only=True)
        np.testing.assert_array_equal(full[0], value[0])
        untouched = ~rows.exact & np.isinf(rows.wl) & np.isinf(rows.wh)
        self.assertTrue(untouched.any())
        for column in value[1:]:
            self.assertFalse(column[untouched].any())
        for column in full[1:]:
            self.assertTrue(column[untouched].all())

    def test_the_gamma_steps(self) -> None:
        self.assertEqual(GAMMA_OPTIMIZER_STEP, 2e-3)
        self.assertEqual(GAMMA_INFORMATION_STEP, 0.03)
        rows = build_rows(FamilyId.GAMMA, lifetime_sample())
        coarse = gamma_terms(rows, (0.5, 0.3), step=GAMMA_INFORMATION_STEP)
        fine = gamma_terms(rows, (0.5, 0.3), step=GAMMA_OPTIMIZER_STEP)
        for a, b in zip(coarse[1:], fine[1:], strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6)
        self.assertFalse(np.array_equal(coarse[3], fine[3]))


class LikelihoodTests(unittest.TestCase):
    def test_the_terms_differ_between_parameters_as_the_library_likelihood_does(self) -> None:
        for family, (sample, first, second) in SAMPLES.items():
            with self.subTest(family=family):
                observations = sample()
                rows = build_rows(family, observations)
                ours = []
                theirs = []
                for theta in (first, second):
                    coordinates = theta
                    ours.append(total_terms(family, rows, coordinates)[0])
                    parameters = dict(
                        zip(
                            PARAMETER_NAMES[family],
                            to_parameters(family, coordinates, 0.0, 1.0),
                            strict=True,
                        )
                    )
                    result = support.contribution(family.value, parameters, observations)
                    theirs.append(result.total_log_likelihood)
                self.assertAlmostEqual(ours[0] - ours[1], theirs[0] - theirs[1], places=9)


class BlockTests(unittest.TestCase):
    def test_the_summarised_block_equals_its_rows(self) -> None:
        plain = [ExactLifetime(1.0), ExactLifetime(2.5), RightCensoredLifetime(3.0)]
        extra = [LeftCensoredLifetime(2.0), IntervalCensoredLifetime(1.0, 2.0)]
        whole = build_rows(FamilyId.EXPONENTIAL, [*plain, *extra])
        block = build_rows(
            FamilyId.EXPONENTIAL, extra, plain_count=3, plain_events=2, plain_total=6.5
        )
        for theta in ((0.4,), (1.7,)):
            expected = exact_sums(exponential_terms(whole, theta))
            sums = exact_sums(exponential_terms(block, theta), exponential_block(block, theta))
            np.testing.assert_allclose(sums, expected, rtol=1e-13)
            self.assertEqual(theta_sums(block, theta), sums)

    def test_a_block_without_events_has_no_log_rate_term(self) -> None:
        rows = build_rows(
            FamilyId.EXPONENTIAL,
            [ExactLifetime(1.0)],
            plain_count=2,
            plain_events=0,
            plain_total=4.0,
        )
        self.assertEqual(exponential_block(rows, (2.0,)), (-8.0, -4.0, 0.0))
        rows = build_rows(
            FamilyId.EXPONENTIAL,
            [ExactLifetime(1.0)],
            plain_count=2,
            plain_events=2,
            plain_total=4.0,
        )
        self.assertEqual(
            exponential_block(rows, (2.0,)), (2 * math.log(2.0) - 8.0, 1.0 - 4.0, -0.5)
        )


class SumTests(unittest.TestCase):
    def test_the_sums_are_exact(self) -> None:
        values = np.array([1e16, 1.0, -1e16, 1.0])
        self.assertEqual(exact_sums([values]), [2.0])
        self.assertEqual(exact_sums([values], (0.5,)), [2.5])

    def test_a_sum_that_is_not_finite_is_nan(self) -> None:
        for bad in (np.array([math.inf, 1.0]), np.array([math.nan]), np.array([1.7e308, 1.7e308])):
            (total,) = exact_sums([bad])
            self.assertTrue(math.isnan(total))

    def test_unevaluable_parameters_have_no_sums(self) -> None:
        rows = build_rows(FamilyId.GAMMA, [ExactLifetime(1.0), LeftCensoredLifetime(2.0)])
        self.assertIsNotNone(theta_sums(rows, (0.0, 0.0)))
        # A shape beyond the kernel's range is refused, not guessed.
        self.assertIsNone(theta_sums(rows, (math.log(1e5), 0.0)))
        # A candidate that makes an interval empty after rounding.
        narrow = build_rows(FamilyId.NORMAL, [IntervalCensoredValue(1.0, math.nextafter(1.0, 2.0))])
        self.assertIsNone(theta_sums(narrow, (0.0, 1e-300)))

    def test_the_term_function_of_each_family(self) -> None:
        self.assertIs(terms_function(FamilyId.EXPONENTIAL), exponential_terms)
        self.assertIs(terms_function(FamilyId.GAMMA), gamma_terms)
        for family in (
            FamilyId.NORMAL,
            FamilyId.LOGNORMAL,
            FamilyId.WEIBULL_MIN,
            FamilyId.GUMBEL_RIGHT,
        ):
            self.assertIs(terms_function(family), location_scale_terms)


if __name__ == "__main__":
    unittest.main()
