"""Censored and truncated log-likelihood contributions against 50-digit mpmath (CENS-LL-01).

Every family, every censoring kind (exact, right, left, interval) and every truncation type
(none, left, right, two-sided) is evaluated on seeded adversarial observations (deep tails,
extreme parameters, narrow intervals inside the quadrature region, windows from ``1e-6`` to ``6``
standardised units wide, and censoring points that sit on a window end) and compared with
the contribution formed entirely at 50 digits.

The bound is the one of the design record.  A contribution is a difference of two stable
log-probabilities, so its absolute error is at most the sum of theirs: the scaled error
``|computed - reference| / max(1, |reference|, |numerator|, |denominator|)`` is at most ``1e-12``.
Where a closed ratio removes the cancellation (the exponential and the Weibull minimum with a
lower truncation point only) the bound is the strict one, ``max(1, |reference|)``, and a
negative control shows the plain difference of the two kernel terms is far outside it.
"""

from __future__ import annotations

import math
import unittest

from tests.reference import censored_grids as grids
from tests.reference import censored_oracle as oracle
from tests.reference import censored_support as support
from veridist.statistics import _logprob as logprob

#: The accuracy bound of the contributions (scaled error).
BOUND = 1.0e-12
#: The cases of every (kind, truncation) cell compared per family.
PER_CELL = 24


class ContributionsAgainstMpmathTests(unittest.TestCase):
    def _check(self, family: str, cases: list[grids.Case], *, strict: bool) -> float:
        worst = 0.0
        for case in cases:
            reference = oracle.contribution(
                family, case.parameters, case.kind, case.point, case.upper, case.window
            )
            parts = oracle.terms(
                family, case.parameters, case.kind, case.point, case.upper, case.window
            )
            self.assertIsNotNone(reference, case)
            self.assertIsNotNone(parts, case)
            assert reference is not None and parts is not None
            one = support.observation(family, case.kind, case.point, case.upper, case.window)
            computed = support.term(family, case.keywords(), one)
            magnitudes = () if strict else parts
            error = support.scaled_error(computed, reference, *magnitudes)
            worst = max(worst, error)
            self.assertLessEqual(error, BOUND, (case, computed, reference))
        return worst

    def test_every_family_kind_and_truncation_cell_is_within_the_bound(self) -> None:
        for family in (*support.LIFETIME_FAMILIES, *support.VALUE_FAMILIES):
            with self.subTest(family=family):
                cases = grids.by_cell(family, PER_CELL)
                cells = {(case.kind, case.truncation) for case in cases}
                self.assertEqual(len(cells), len(grids.KINDS) * len(grids.TRUNCATIONS))
                self._check(family, cases, strict=False)

    def test_the_closed_ratio_rows_are_within_the_strict_bound(self) -> None:
        for family in ("exponential", "weibull_min"):
            with self.subTest(family=family):
                cases = grids.stable_sample(grids.lower_only(family), 60)
                self.assertGreaterEqual(len(cases), 40)
                self._check(family, cases, strict=True)

    def test_a_censoring_point_at_a_window_end_contributes_exactly_zero(self) -> None:
        for family in (*support.LIFETIME_FAMILIES, *support.VALUE_FAMILIES):
            edge = [
                case
                for case in grids.cases(family)
                if case.window is not None
                and (
                    (case.kind == "right" and case.window[0] == case.point)
                    or (case.kind == "left" and case.window[1] == case.point)
                )
            ]
            self.assertGreater(len(edge), 10)
            for case in edge:
                with self.subTest(family=family, kind=case.kind, point=case.point):
                    one = support.observation(family, case.kind, case.point, None, case.window)
                    self.assertEqual(support.term(family, case.keywords(), one), 0.0)


class CancellationTests(unittest.TestCase):
    """Rows whose numerator and denominator share an end and are huge while the term is small."""

    def test_exponential_left_truncation_far_in_the_tail_is_exact(self) -> None:
        # log S(c) is about -1e5 and log S(a) about -1e5 with the term itself near -0.03: the
        # difference of the two kernel terms has an absolute error near 1e-11, the closed ratio
        # -rate (c - a) is within a few units of the last place.
        rate, a, c = 0.1, 1e6, 1e6 + 0.3
        window = (a, math.inf)
        for kind, point in (("right", c), ("exact", c)):
            with self.subTest(kind=kind):
                reference = oracle.contribution("exponential", (rate,), kind, point, None, window)
                assert reference is not None
                one = support.observation("exponential", kind, point, None, window)
                computed = support.term("exponential", {"rate": rate}, one)
                self.assertLessEqual(support.scaled_error(computed, reference), 1e-15)
        right = oracle.contribution("exponential", (rate,), "right", c, None, window)
        assert right is not None
        naive = logprob.log_sf("exponential", c, rate=rate) - logprob.log_sf(
            "exponential", a, rate=rate
        )
        self.assertGreater(abs(float(naive) - right), 1e-12)

    def test_weibull_left_truncation_far_in_the_tail_is_within_the_strict_bound(self) -> None:
        parameters = (2.0, 1.0)
        for kind, point in (("right", 1000.0 + 1e-3), ("exact", 1000.0 + 2e-3)):
            with self.subTest(kind=kind):
                window = (1000.0, math.inf)
                reference = oracle.contribution(
                    "weibull_min", parameters, kind, point, None, window
                )
                assert reference is not None
                one = support.observation("weibull_min", kind, point, None, window)
                computed = support.term("weibull_min", {"shape": 2.0, "scale": 1.0}, one)
                self.assertLessEqual(support.scaled_error(computed, reference), BOUND)


if __name__ == "__main__":
    unittest.main()
