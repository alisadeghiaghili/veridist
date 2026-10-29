"""Seeded properties of the censored and truncated log-likelihood contributions (CENS-LL-03).

* a narrow interval is the exact observation scaled by its width: ``log P(l, u) - log(u - l)``
  tends to ``log f`` at the middle of the interval;
* a left-censoring point far in the upper tail contributes ``0``; an interval ``(0, u]`` of a
  lifetime is the left-censored observation at ``u`` and an interval reaching far into the upper
  tail is the right-censored observation at its lower bound;
* a window at the ends of the support (or at infinity) is no truncation, and a window that
  grows to the whole support tends to it;
* a two-sided window with an open end equals the one-sided window;
* the contribution does not depend on the position of a row among others, nor on its batch;
* ``x -> a x + b`` (location-scale families) and ``x -> a x`` (lifetime families) with the
  parameters, bounds and windows mapped alike leave a censored term unchanged and move an exact
  term by ``-log a``.
"""

from __future__ import annotations

import math
import random
import unittest

from tests.reference import censored_grids as grids
from tests.reference import censored_support as support
from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.statistics._log_contributions import ContributionEvaluator
from veridist.statistics.lifetime_log_likelihood import _evaluate_log_survival

FAMILIES = (*support.LIFETIME_FAMILIES, *support.VALUE_FAMILIES)


def scaled(computed: float, expected: float) -> float:
    return abs(computed - expected) / max(1.0, abs(expected))


def parameter_sets(family: str) -> list[tuple[float, ...]]:
    return list(grids.PARAMETERS[family][:2])


def keywords(family: str, parameters: tuple[float, ...]) -> dict[str, float]:
    return grids.Case(family, parameters, "exact", 1.0, None, None, "none").keywords()


class NarrowIntervalTests(unittest.TestCase):
    def test_an_interval_tends_to_the_exact_observation_scaled_by_its_width(self) -> None:
        for family in FAMILIES:
            for parameters in parameter_sets(family):
                for case in grids.stable_sample(
                    [
                        c
                        for c in grids.cases(family)
                        if c.kind == "exact" and c.truncation == "none"
                    ],
                    5,
                ):
                    x = case.point
                    for relative in (1e-8, 1e-10):
                        width = (
                            relative * max(abs(x), 1e-3)
                            if family in support.VALUE_FAMILIES
                            else relative * x
                        )
                        lower, upper = x - width / 2.0, x + width / 2.0
                        if not lower < x < upper or case.parameters != parameters:
                            continue
                        with self.subTest(family=family, x=x, relative=relative):
                            middle = lower + (upper - lower) / 2.0
                            interval = support.observation(family, "interval", lower, upper, None)
                            exact = support.observation(family, "exact", middle, None, None)
                            term = support.term(family, case.keywords(), interval)
                            density = support.term(family, case.keywords(), exact)
                            difference = term - math.log(upper - lower)
                            self.assertLessEqual(scaled(difference, density), 1e-6)


class LimitTests(unittest.TestCase):
    def test_a_left_censoring_point_in_the_far_upper_tail_contributes_zero(self) -> None:
        far = {
            "exponential": ((1.0,), 1e30),
            "weibull_min": ((1.5, 6.0), 1e30),
            "lognormal": ((0.0, 1.0), 1e300),
            "gamma": ((2.5, 10.0), 1e6),
            "normal": ((0.0, 1.0), 1e6),
            "gumbel_right": ((0.0, 1.0), 1e6),
        }
        for family, (parameters, point) in far.items():
            with self.subTest(family=family):
                one = support.observation(family, "left", point, None, None)
                self.assertEqual(support.term(family, keywords(family, parameters), one), 0.0)

    def test_a_lifetime_interval_from_zero_is_the_left_censored_observation(self) -> None:
        for family in support.LIFETIME_FAMILIES:
            for case in grids.stable_sample(
                [
                    c
                    for c in grids.cases(family)
                    if c.kind == "left" and c.truncation in ("none", "right")
                ],
                24,
            ):
                with self.subTest(family=family, window=case.window, point=case.point):
                    window = case.window
                    interval = support.observation(family, "interval", 0.0, case.point, window)
                    left = support.observation(family, "left", case.point, None, window)
                    self.assertEqual(
                        support.term(family, case.keywords(), interval),
                        support.term(family, case.keywords(), left),
                    )

    def test_an_interval_reaching_the_top_of_the_support_is_the_right_censored_one(self) -> None:
        tops = {
            "exponential": 1e20,
            "weibull_min": 1e300,
            "lognormal": 1e150,
            "gamma": 1e9,
            "normal": 1e12,
            "gumbel_right": 1e12,
        }
        for family in FAMILIES:
            for case in grids.stable_sample(
                [c for c in grids.cases(family) if c.kind == "right" and c.truncation == "none"], 12
            ):
                if case.point >= tops[family] / 10.0:
                    continue
                with self.subTest(family=family, point=case.point):
                    interval = support.observation(
                        family, "interval", case.point, tops[family], None
                    )
                    right = support.observation(family, "right", case.point, None, None)
                    self.assertLessEqual(
                        scaled(
                            support.term(family, case.keywords(), interval),
                            support.term(family, case.keywords(), right),
                        ),
                        1e-12,
                    )


class WindowTests(unittest.TestCase):
    def test_a_window_at_the_ends_of_the_support_is_no_truncation(self) -> None:
        for family in FAMILIES:
            for case in grids.stable_sample(
                [c for c in grids.cases(family) if c.truncation == "none"], 20
            ):
                with self.subTest(family=family, kind=case.kind, point=case.point):
                    plain = support.observation(family, case.kind, case.point, case.upper, None)
                    bottom = 0.0 if family in support.LIFETIME_FAMILIES else -math.inf
                    for window in ((bottom, math.inf), (-math.inf, math.inf)):
                        windowed = support.observation(
                            family, case.kind, case.point, case.upper, window
                        )
                        self.assertEqual(windowed, plain)
                        self.assertEqual(
                            support.term(family, case.keywords(), windowed),
                            support.term(family, case.keywords(), plain),
                        )

    def test_a_window_that_grows_to_the_whole_support_tends_to_no_truncation(self) -> None:
        wide = {
            "exponential": (1e-300, 1e300),
            "weibull_min": (1e-300, 1e300),
            "lognormal": (1e-300, 1e300),
            "gamma": (1e-300, 1e300),
            "normal": (-1e9, 1e9),
            "gumbel_right": (-1e9, 1e9),
        }
        for family in FAMILIES:
            for case in grids.stable_sample(
                [c for c in grids.cases(family) if c.truncation == "none"], 24
            ):
                low, high = wide[family]
                if not low < case.point < high or (family == "gamma" and case.parameters[0] < 0.1):
                    continue  # a shape below 0.1 keeps mass near zero that no window reaches
                with self.subTest(family=family, kind=case.kind, point=case.point):
                    plain = support.observation(family, case.kind, case.point, case.upper, None)
                    windowed = support.observation(
                        family, case.kind, case.point, case.upper, (low, high)
                    )
                    self.assertLessEqual(
                        scaled(
                            support.term(family, case.keywords(), windowed),
                            support.term(family, case.keywords(), plain),
                        ),
                        1e-12,
                    )

    def test_a_two_sided_window_with_a_far_open_end_is_the_one_sided_window(self) -> None:
        far = {
            "exponential": 1e300,
            "weibull_min": 1e300,
            "lognormal": 1e300,
            "gamma": 1e300,
            "normal": 1e9,
            "gumbel_right": 1e9,
        }
        for family in FAMILIES:
            for case in grids.stable_sample(
                [c for c in grids.cases(family) if c.truncation == "left"], 24
            ):
                assert case.window is not None
                if not case.point < far[family] / 10.0:
                    continue
                with self.subTest(family=family, kind=case.kind, point=case.point):
                    one = support.observation(
                        family, case.kind, case.point, case.upper, case.window
                    )
                    two = support.observation(
                        family, case.kind, case.point, case.upper, (case.window[0], far[family])
                    )
                    self.assertLessEqual(
                        scaled(
                            support.term(family, case.keywords(), two),
                            support.term(family, case.keywords(), one),
                        ),
                        1e-12,
                    )


class PositionTests(unittest.TestCase):
    def test_a_term_equals_its_one_row_term_whatever_its_neighbours_and_position(self) -> None:
        generator = random.Random(11)
        for family in FAMILIES:
            cases = grids.by_cell(family, 3)
            rows = [support.observation(family, c.kind, c.point, c.upper, c.window) for c in cases]
            parameters = cases[0].keywords()
            validated = FAMILY_REGISTRY.families[FamilyId(family)].validate_parameters(**parameters)
            evaluator = ContributionEvaluator(FamilyId(family), validated, _evaluate_log_survival)
            singles = [evaluator.evaluate([row])[0] for row in rows]
            # one parameter set per family: only rows of that set share the evaluator
            same = [i for i, c in enumerate(cases) if c.parameters == cases[0].parameters]
            for _ in range(4):
                order = same[:]
                generator.shuffle(order)
                batch = evaluator.evaluate([rows[i] for i in order])
                for position, index in enumerate(order):
                    with self.subTest(family=family, index=index):
                        self.assertEqual(batch[position], singles[index])

    def test_the_total_is_invariant_to_row_order_for_every_family(self) -> None:
        generator = random.Random(5)
        for family in FAMILIES:
            cases = [
                c for c in grids.by_cell(family, 2) if c.parameters == grids.PARAMETERS[family][0]
            ]
            rows = [support.observation(family, c.kind, c.point, c.upper, c.window) for c in cases]
            parameters = keywords(family, grids.PARAMETERS[family][0])
            reference = support.contribution(family, parameters, rows)
            for _ in range(3):
                shuffled = rows[:]
                generator.shuffle(shuffled)
                again = support.contribution(family, parameters, shuffled)
                self.assertEqual(
                    getattr(again, "total_log_likelihood", None),
                    getattr(reference, "total_log_likelihood", None),
                    family,
                )


def mapped_parameters(
    family: str, parameters: tuple[float, ...], a: float, b: float
) -> tuple[float, ...]:
    if family in ("normal", "gumbel_right"):
        return (a * parameters[0] + b, a * parameters[1])
    if family == "exponential":
        return (parameters[0] / a,)
    if family == "lognormal":
        return (parameters[0] + math.log(a), parameters[1])
    return (parameters[0], parameters[1] * a)


class EquivarianceTests(unittest.TestCase):
    def test_affine_maps_leave_censored_terms_and_shift_exact_terms_by_minus_log_a(self) -> None:
        for family in FAMILIES:
            location = family in support.VALUE_FAMILIES
            a, b = 3.7, (-12.5 if location else 0.0)
            for case in grids.stable_sample(
                [c for c in grids.cases(family) if c.parameters == grids.PARAMETERS[family][0]], 40
            ):

                def move(x: float | None) -> float | None:
                    return None if x is None else (x if math.isinf(x) else a * x + b)

                ends = [case.point, *(x for x in (case.upper,) if x is not None)]
                ends += [x for x in (case.window or ()) if math.isfinite(x)]
                if any(
                    abs(p - q) < 1e-3 * (max(abs(p), abs(q)) + abs(b))
                    for i, p in enumerate(ends)
                    for q in ends[i + 1 :]
                ):
                    continue  # ends that close would make the rounding of the map the test
                window = None if case.window is None else tuple(move(x) for x in case.window)
                if window is not None and not (window[0] is not None and window[1] is not None):
                    continue
                parameters = mapped_parameters(family, case.parameters, a, b)
                point, upper = move(case.point), move(case.upper)
                assert point is not None
                try:
                    original = support.term(
                        family,
                        case.keywords(),
                        support.observation(family, case.kind, case.point, case.upper, case.window),
                    )
                    moved = support.term(
                        family,
                        keywords(family, parameters),
                        support.observation(family, case.kind, point, upper, window),  # type: ignore[arg-type]
                    )
                except AssertionError:
                    continue  # a term that is not representable has no counterpart to compare
                expected = original - (math.log(a) if case.kind == "exact" else 0.0)
                with self.subTest(family=family, kind=case.kind, point=case.point):
                    self.assertLessEqual(scaled(moved, expected), 1e-10)


if __name__ == "__main__":
    unittest.main()
