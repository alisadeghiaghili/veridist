"""Seeded properties of the fits of censored and truncated data (CENS-FIT, CENS-LL-03).

* a narrow interval around each value fits to the exact fit, and the difference shrinks with the
  width;
* a window that grows to the whole support fits to the untruncated fit, and a window end at the
  bottom of the support (or at infinity) is no window at all, bit for bit;
* the estimate does not depend on the order of the rows, bit for bit;
* integer frequency weights equal replicated rows, bit for bit, whatever the weights;
* ``x -> a x`` (lifetime families) and ``x -> a x + b`` (location-scale families), with the
  bounds and the windows mapped alike, map the estimate alike;
* a left-censored row far in the upper tail, and a right-censored row far in the lower tail,
  carry no information; an interval ``(0, u]`` of a lifetime is the left-censored row at ``u``;
* the closed form of the exponential equals the Newton maximum on the data both take.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from tests.reference import censored_support as support
from veridist import fit, ppf, sample
from veridist.domain._combinations import observation_cell
from veridist.families import _censored_fit as censored
from veridist.families._censored_rows import build_rows, pilot
from veridist.families._fit_facts import summarize_observations
from veridist.families.registry import FamilyId

FAMILIES = ("exponential", "weibull_min", "lognormal", "normal", "gumbel_right")
TRUTH = {
    "exponential": {"rate": 0.5},
    "weibull_min": {"shape": 1.7, "scale": 4.0},
    "lognormal": {"mu_log": 1.2, "sigma_log": 0.5},
    "normal": {"mu": 10.0, "sigma": 2.0},
    "gumbel_right": {"location": 10.0, "scale": 2.0},
}


def draw(family: str, size: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(v) for v in sample(family, size, rng=rng, **TRUTH[family])]


def build(family: str, kind: str, point: float, upper: float | None = None, **window: float) -> Any:
    return support.observation(
        family, kind, point, upper, (window["low"], window["high"]) if window else None
    )


def relative(first: Any, second: Any) -> float:
    return max(
        abs(a - b) / max(abs(b), 1e-300)
        for a, b in zip(first.parameters.values(), second.parameters.values(), strict=True)
    )


class IntervalTests(unittest.TestCase):
    def test_narrow_intervals_fit_to_the_exact_fit(self) -> None:
        for family in FAMILIES:
            values = draw(family, 20, 1)
            exact = fit(family, [build(family, "exact", v) for v in values])
            differences = []
            for width in (1e-2, 1e-4, 1e-6):
                rows = [
                    build(family, "interval", v - width * abs(v), v + width * abs(v))
                    for v in values
                ]
                differences.append(relative(fit(family, rows), exact))
            with self.subTest(family=family):
                self.assertLess(differences[2], 1e-5)
                self.assertLess(differences[1], differences[0])
                self.assertLess(differences[2], differences[1])


class WindowTests(unittest.TestCase):
    def test_a_window_that_grows_to_the_support_fits_to_the_untruncated_fit(self) -> None:
        for family in FAMILIES:
            values = draw(family, 20, 2)
            plain = fit(family, [build(family, "exact", v) for v in values])
            lifetime = family in support.LIFETIME_FAMILIES
            wide = (1e-9, 1e9) if lifetime else (-1e9, 1e9)
            rows = [build(family, "exact", v, low=wide[0], high=wide[1]) for v in values]
            truncated = fit(family, rows)
            with self.subTest(family=family):
                self.assertTrue(truncated.observation_summary.is_new_data)
                self.assertLess(relative(truncated, plain), 1e-6)

    def test_a_window_end_at_the_bottom_of_the_support_or_at_infinity_is_no_window(self) -> None:
        for family in FAMILIES:
            values = draw(family, 12, 3)
            plain = fit(family, [build(family, "exact", v) for v in values])
            lifetime = family in support.LIFETIME_FAMILIES
            low = 0.0 if lifetime else -math.inf
            rows = [build(family, "exact", v, low=low, high=math.inf) for v in values]
            with self.subTest(family=family):
                self.assertEqual(fit(family, rows), plain)
                self.assertFalse(fit(family, rows).observation_summary.is_new_data)


class InvarianceTests(unittest.TestCase):
    def mixed(self, family: str, seed: int) -> list[Any]:
        values = draw(family, 24, seed)
        rows = []
        for index, v in enumerate(values):
            kind = ("exact", "right", "left", "interval")[index % 4]
            low, high = (0.3 * min(values), 2.0 * max(values) + 1.0)
            window = {"low": low, "high": high} if index % 3 == 0 else {}
            if kind == "interval":
                rows.append(build(family, kind, v * 0.9, v * 1.1, **window))
            elif kind == "right":
                rows.append(build(family, kind, v * 0.8, **window))
            elif kind == "left":
                rows.append(build(family, kind, v * 1.2, **window))
            else:
                rows.append(build(family, kind, v, **window))
        return rows

    def test_the_estimate_does_not_depend_on_the_order_of_the_rows(self) -> None:
        for family in FAMILIES:
            for seed in (4,):
                rows = self.mixed(family, seed)
                reference = fit(family, rows)
                order = np.random.default_rng(seed).permutation(len(rows))
                shuffled = fit(family, [rows[i] for i in order])
                with self.subTest(family=family, seed=seed):
                    self.assertEqual(shuffled, reference)
                    self.assertEqual(shuffled.log_likelihood, reference.log_likelihood)
                    self.assertEqual(shuffled.diagnostics, reference.diagnostics)

    def test_integer_frequency_weights_equal_replicated_rows(self) -> None:
        for family in FAMILIES[1:]:
            rows = self.mixed(family, 6)
            weights = [int(w) for w in np.random.default_rng(6).integers(0, 4, len(rows))]
            weights[0] = 1
            replicated = [
                row for row, weight in zip(rows, weights, strict=True) for _ in range(weight)
            ]
            with self.subTest(family=family):
                weighted = fit(family, rows, frequency_weights=weights)
                direct = fit(family, replicated)
                self.assertEqual(weighted, direct)
                self.assertEqual(weighted.log_likelihood, direct.log_likelihood)

    def test_the_estimate_is_equivariant_under_a_change_of_unit(self) -> None:
        factor, shift = 3.0, 7.0
        for family in FAMILIES:
            rows = self.mixed(family, 7)
            lifetime = family in support.LIFETIME_FAMILIES
            moved = [map_row(family, row, factor, 0.0 if lifetime else shift) for row in rows]
            base, other = fit(family, rows), fit(family, moved)
            expected = map_parameters(family, base.parameters, factor, 0.0 if lifetime else shift)
            with self.subTest(family=family):
                for name, value in expected.items():
                    self.assertAlmostEqual(
                        other.parameters[name], value, delta=1e-6 * abs(value) + 1e-9
                    )


def map_row(family: str, row: Any, a: float, b: float) -> Any:
    """The observation ``row`` after ``x -> a x + b`` (bounds and windows included)."""

    censoring, _ = observation_cell(row)
    low = row.truncation_lower
    high = row.truncation_upper
    window = None
    if low is not None or high is not None:
        window = (
            -math.inf if low is None else a * low + b,
            math.inf if high is None else a * high + b,
        )
    if censoring.value == "interval":
        return support.observation(family, "interval", a * row.lower + b, a * row.upper + b, window)
    point = row.time if hasattr(row, "time") else row.value
    return support.observation(family, censoring.value, a * point + b, None, window)


def map_parameters(family: str, parameters: Any, a: float, b: float) -> dict[str, float]:
    p = dict(parameters)
    if family == "exponential":
        return {"rate": p["rate"] / a}
    if family == "weibull_min":
        return {"shape": p["shape"], "scale": p["scale"] * a}
    if family == "lognormal":
        return {"mu_log": p["mu_log"] + math.log(a), "sigma_log": p["sigma_log"]}
    if family == "normal":
        return {"mu": a * p["mu"] + b, "sigma": a * p["sigma"]}
    return {"location": a * p["location"] + b, "scale": a * p["scale"]}


class UninformativeRowTests(unittest.TestCase):
    def far(self, family: str, probability: float, parameters: Any) -> float:
        """The point beyond which (or below which) the fitted distribution has this mass."""

        return float(ppf(family, probability, **dict(parameters)))

    def test_a_left_censored_row_far_in_the_upper_tail_carries_no_information(self) -> None:
        for family in ("weibull_min", "normal", "exponential"):
            rows = InvarianceTests().mixed(family, 8)
            base = fit(family, rows)
            far = [build(family, "left", self.far(family, 1.0 - 1e-14, base.parameters))] * 3
            with self.subTest(family=family):
                self.assertLess(relative(fit(family, [*rows, *far]), base), 1e-8)

    def test_a_right_censored_row_far_in_the_lower_tail_carries_no_information(self) -> None:
        for family in ("normal", "gumbel_right"):
            rows = InvarianceTests().mixed(family, 9)
            base = fit(family, rows)
            far = [build(family, "right", self.far(family, 1e-14, base.parameters))] * 3
            with self.subTest(family=family):
                self.assertLess(relative(fit(family, [*rows, *far]), base), 1e-8)

    def test_an_interval_from_zero_is_the_left_censored_row(self) -> None:
        for family in ("exponential", "weibull_min", "lognormal"):
            values = draw(family, 16, 10)
            exact = [build(family, "exact", v) for v in values[:12]]
            upper = [round(v, 3) + 0.1 for v in values[12:]]
            left = fit(family, [*exact, *[build(family, "left", u) for u in upper]])
            interval = fit(family, [*exact, *[build(family, "interval", 0.0, u) for u in upper]])
            with self.subTest(family=family):
                self.assertEqual(
                    tuple(left.parameters.values()), tuple(interval.parameters.values())
                )
                self.assertEqual(left.log_likelihood, interval.log_likelihood)
                self.assertEqual(interval.observation_summary.interval_censored, 4)
                self.assertEqual(left.observation_summary.left_censored, 4)


class ClosedFormTests(unittest.TestCase):
    def test_the_exponential_closed_form_is_the_newton_maximum(self) -> None:
        values = draw("exponential", 30, 11)
        rows = []
        for index, v in enumerate(values):
            entry = 0.2 * (index % 4)
            rows.append(
                build(
                    "exponential",
                    "exact" if index % 3 else "right",
                    v + entry + 0.1,
                    low=entry,
                    high=math.inf,
                )
            )
        closed = fit("exponential", rows)
        self.assertEqual(closed.optimum.value, "closed_form")
        parsed = build_rows(FamilyId.EXPONENTIAL, rows)
        newton = censored._newton(
            FamilyId.EXPONENTIAL, rows, parsed, summarize_observations(rows), pilot(parsed)
        )
        assert isinstance(newton, censored.CensoredEstimate)
        self.assertAlmostEqual(newton.parameters[0], closed.rate, delta=1e-12 * closed.rate)
        self.assertAlmostEqual(newton.log_likelihood, closed.log_likelihood, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
