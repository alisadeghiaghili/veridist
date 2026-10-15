"""The scale fit against the in-memory fit, family by family, for estimates and uncertainty.

The two paths compute the same estimator by different arithmetic, so the agreement that
can be claimed differs by family (every bound below was measured and then given a
margin of at least ten):

* exponential: identical. Both paths divide the event count by the total time, and the
  total is the same binary64 (the exact sum rounded once, which the compensated sum of
  the in-memory path reproduces on these data). Every reported quantity is equal.
* normal and lognormal: a few units in the last place for the estimates (the scale path
  rounds the exact mean and the exact variance once, the in-memory path rounds
  floating-point sums), up to about ``1e-14`` for quantities that pass through the
  numerical derivatives and root finders of the uncertainty machinery. The scale
  estimates are exact whatever the size of the spread relative to the values
  (``ExtremeConditioningTests``); the in-memory variance carries a relative error of
  about ``(ulp(mean) / sd)**2`` from its rounded centre, which is where the two paths
  part for data such as mean ``1e12`` with sd ``1e-3``.
* gamma: the scale path solves the profile equation to binary64, the in-memory path
  locates the shape by a golden-section search whose value comparisons resolve the
  log-shape only to about ``sqrt(eps)``, so the estimates and everything derived from
  them agree to about ``5e-8`` (bound ``1e-6``); the log-likelihood, flat at its
  maximum, and the profile-likelihood endpoints agree to about ``1e-14``.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_fit_oracle import (
    FAMILIES,
    dataset,
    gamma_estimates,
    gaussian_estimates,
    observations,
)
from veridist.families import fit as in_memory_fit
from veridist.families.registry import FamilyId
from veridist.families.uncertainty import FitUncertainty, UncertaintyUnavailable
from veridist.scale._arrow import ParquetSource
from veridist.scale._fit import fit
from veridist.scale._sources import NumpySource

LEVELS = (0.9, 0.95)


def close(left: float, right: float, tolerance: float) -> bool:
    if left == right:
        return True
    if math.isinf(left) or math.isinf(right):
        return False
    return abs(left - right) <= tolerance * max(abs(left), abs(right))


def attempt(call: Callable[[], Any]) -> Any:
    try:
        return call()
    except (NotImplementedError, ValueError, ArithmeticError) as error:
        return type(error)


class Comparison(unittest.TestCase):
    def assertClose(self, left: float, right: float, tolerance: float, label: str) -> None:
        self.assertTrue(close(left, right, tolerance), f"{label}: {left!r} vs {right!r}")

    def compare_estimates(
        self, family: FamilyId, values: np.ndarray, estimate: float, likelihood: float, scale: Any
    ) -> tuple[Any, Any]:
        memory = in_memory_fit(family, observations(family, values))
        result = fit(scale, family).fit
        self.assertIs(type(result), type(memory))
        self.assertEqual(
            (result.observation_count, result.event_count, result.censored_count),
            (memory.observation_count, memory.event_count, memory.censored_count),
        )
        self.assertEqual(result.converged, memory.converged)
        for name, value in memory.parameters.items():
            self.assertClose(result.parameters[name], value, estimate, f"parameter {name}")
        self.assertClose(result.log_likelihood, memory.log_likelihood, likelihood, "likelihood")
        return memory, result

    def compare_uncertainty(
        self,
        memory: Any,
        result: Any,
        values: np.ndarray,
        *,
        estimate: float,
        profile: float,
        off_diagonal: float | None = None,
        derived: float | None = None,
    ) -> None:
        left, right = memory.uncertainty(), result.uncertainty()
        assert isinstance(left, FitUncertainty) and isinstance(right, FitUncertainty)
        self.assertEqual(right.family, left.family)
        self.assertEqual(right.parameter_names, left.parameter_names)
        for one, other in zip(right.estimates, left.estimates, strict=True):
            self.assertClose(one, other, estimate, "estimate")
        for row in range(len(left.covariance)):
            for column in range(len(left.covariance)):
                # Off-diagonal entries vanish at the maximum, so compare them to the scale of
                # the diagonal rather than to themselves.
                size = math.sqrt(left.covariance[row][row] * left.covariance[column][column])
                allowed = off_diagonal if row != column and off_diagonal is not None else estimate
                self.assertLessEqual(
                    abs(right.covariance[row][column] - left.covariance[row][column]),
                    allowed * size,
                    f"covariance[{row}][{column}]",
                )
        for name in left.parameter_names:
            self.assertClose(
                right.standard_errors[name], left.standard_errors[name], estimate, f"se {name}"
            )
        for level, method, tolerance in (
            (0.9, "wald", estimate),
            (0.95, "wald", estimate),
            (0.95, "profile", profile),
        ):
            expected = left.interval_details(level, method)
            actual = right.interval_details(level, method)
            for name in expected:
                one, other = actual[name], expected[name]
                label = f"{method} {level} {name}"
                self.assertClose(one.lower, other.lower, tolerance, label)
                self.assertClose(one.upper, other.upper, tolerance, label)
                self.assertEqual(
                    (one.level, one.method, one.lower_unbounded, one.upper_unbounded),
                    (other.level, other.method, other.lower_unbounded, other.upper_unbounded),
                )
        median = float(np.median(values))
        level_tol = estimate if derived is None else derived
        quantities: tuple[tuple[str, Callable[[FitUncertainty, str], Any]], ...] = (
            ("mean", lambda u, m: u.mean(0.9, m)),
            ("quantile", lambda u, m: u.quantile(0.1, 0.9, m)),
            ("survival", lambda u, m: u.survival(median, 0.9, m)),
        )
        for label, call in quantities:
            for method, tolerance in (("wald", estimate), ("profile", profile)):
                one = attempt(lambda c=call, m=method: c(right, m))
                other = attempt(lambda c=call, m=method: c(left, m))
                if isinstance(other, type):
                    self.assertIs(one, other, f"{label} {method}")
                    continue
                self.assertClose(one.estimate, other.estimate, level_tol, f"{label} estimate")
                self.assertClose(one.lower, other.lower, max(tolerance, level_tol), label)
                self.assertClose(one.upper, other.upper, max(tolerance, level_tol), label)
                self.assertEqual(
                    (one.method, one.level, one.lower_unbounded, one.upper_unbounded),
                    (other.method, other.level, other.lower_unbounded, other.upper_unbounded),
                )


class ExponentialDifferentialTests(Comparison):
    def test_everything_is_identical(self) -> None:
        for size in (30, 400):
            for seed in (1, 2, 3):
                values = dataset(FamilyId.EXPONENTIAL, size, seed)
                with self.subTest(size=size, seed=seed):
                    memory, result = self.compare_estimates(
                        FamilyId.EXPONENTIAL, values, 0.0, 0.0, NumpySource(values)
                    )
                    self.assertEqual(result, memory)
                    self.compare_uncertainty(memory, result, values, estimate=0.0, profile=0.0)
                    for level in LEVELS:
                        self.assertEqual(
                            result.uncertainty().confidence_intervals(level, "exact"),
                            memory.uncertainty().confidence_intervals(level, "exact"),
                        )

    def test_the_total_time_is_the_exact_sum_rounded_once(self) -> None:
        values = np.concatenate([np.full(1, 1e16), np.full(7, 1.0)])
        result = fit(NumpySource(values), "exponential").fit
        self.assertEqual(result.total_time, 1e16 + 8.0)  # type: ignore[union-attr]


class NormalDifferentialTests(Comparison):
    def test_agreement_is_a_few_ulps_for_the_estimates(self) -> None:
        for size in (30, 400):
            for seed in (1, 2, 3):
                values = dataset(FamilyId.NORMAL, size, seed)
                with self.subTest(size=size, seed=seed):
                    memory, result = self.compare_estimates(
                        FamilyId.NORMAL, values, 1e-15, 1e-14, NumpySource(values)
                    )
                    self.compare_uncertainty(
                        memory, result, values, estimate=1e-14, profile=1e-13
                    )

    def test_the_scale_sigma_is_exact_whatever_the_conditioning(self) -> None:
        rng = np.random.default_rng(11)
        for mean, spread in ((5.0, 2.0), (-50.0, 0.5), (300.0, 1.0), (1e6, 1e-2)):
            values = rng.normal(mean, spread, 300)
            result = fit(NumpySource(values), "normal").fit
            with self.subTest(mean=mean, spread=spread):
                self.assertEqual(
                    (result.parameters["mu"], result.parameters["sigma"]),  # type: ignore[union-attr]
                    gaussian_estimates(values),
                )


class LognormalDifferentialTests(Comparison):
    def test_agreement_is_a_few_ulps_for_the_estimates(self) -> None:
        for size in (30, 400):
            for seed in (1, 2, 3):
                values = dataset(FamilyId.LOGNORMAL, size, seed)
                with self.subTest(size=size, seed=seed):
                    memory, result = self.compare_estimates(
                        FamilyId.LOGNORMAL, values, 1e-14, 1e-14, NumpySource(values)
                    )
                    self.compare_uncertainty(
                        memory, result, values, estimate=1e-14, profile=1e-13
                    )


class ExtremeConditioningTests(Comparison):
    """Spread far below the size of the values, and magnitudes whose squares leave binary64.

    The sum of squares of a state is exact, so the scale estimate equals the exact one
    (``gaussian_estimates``) for every such sample. The in-memory fit centres at a rounded
    mean, so its variance carries a relative error of about ``(ulp(mean) / sd)**2``; the
    comparison with it is bounded by that, which is the in-memory path's limit, not the
    scale path's.
    """

    NORMAL_CASES = (
        ("mean 1e9, sd 1", 1e9, 1.0),
        ("mean 1e12, sd 1e-3", 1e12, 1e-3),
        ("mean 1e15, sd 3", 1e15, 3.0),
        ("mean 1e-200, sd 1e-201", 1e-200, 1e-201),
        ("mean 1e200, sd 1e199", 1e200, 1e199),
        ("mean -3e-200, sd 1e-200", -3e-200, 1e-200),
    )

    def test_the_normal_estimates_are_the_exact_ones(self) -> None:
        rng = np.random.default_rng(4)
        for name, mean, spread in self.NORMAL_CASES:
            values = rng.normal(mean, spread, 400)
            result = fit(NumpySource(values), "normal").fit
            with self.subTest(case=name):
                self.assertEqual(
                    (result.parameters["mu"], result.parameters["sigma"]),  # type: ignore[union-attr]
                    gaussian_estimates(values),
                )

    def test_the_in_memory_fit_agrees_to_its_own_resolution(self) -> None:
        rng = np.random.default_rng(4)
        for name, mean, spread in self.NORMAL_CASES:
            values = rng.normal(mean, spread, 400)
            memory = in_memory_fit(FamilyId.NORMAL, observations(FamilyId.NORMAL, values))
            result = fit(NumpySource(values), "normal").fit
            center = float(np.mean(values))
            bound = 1e-14 + (math.ulp(center) / spread) ** 2
            with self.subTest(case=name):
                self.assertIs(type(result), type(memory))
                self.assertLessEqual(
                    abs(result.parameters["mu"] / memory.parameters["mu"] - 1.0), 2.5e-16 + bound  # type: ignore[union-attr]
                )
                self.assertLessEqual(
                    abs(result.parameters["sigma"] / memory.parameters["sigma"] - 1.0), bound  # type: ignore[union-attr]
                )
                self.assertTrue(math.isfinite(result.log_likelihood))  # type: ignore[union-attr]

    def test_the_uncertainty_of_a_large_offset_matches_where_the_in_memory_path_resolves_it(
        self,
    ) -> None:
        values = np.random.default_rng(4).normal(1e9, 1.0, 400)
        memory, result = self.compare_estimates(
            FamilyId.NORMAL, values, 1e-14, 1e-14, NumpySource(values)
        )
        self.compare_uncertainty(
            memory, result, values, estimate=1e-13, profile=1e-12, off_diagonal=1e-6, derived=1e-6
        )

    def test_extreme_magnitudes_have_no_uncertainty_on_either_path(self) -> None:
        rng = np.random.default_rng(4)
        for mean, spread in ((1e200, 1e199), (1e-200, 1e-201)):
            values = rng.normal(mean, spread, 400)
            result = fit(NumpySource(values), "normal").fit
            memory = in_memory_fit(FamilyId.NORMAL, observations(FamilyId.NORMAL, values))
            with self.subTest(mean=mean):
                self.assertIsInstance(result.uncertainty(), UncertaintyUnavailable)  # type: ignore[union-attr]
                self.assertIsInstance(memory.uncertainty(), UncertaintyUnavailable)  # type: ignore[union-attr]

    def test_tight_logarithms_are_estimated_exactly_and_agree_with_the_in_memory_fit(self) -> None:
        rng = np.random.default_rng(4)
        for name, center, spread in (
            ("logs near 700, sd 1e-3", 700.0, 1e-3),
            ("logs near -700, sd 1e-3", -700.0, 1e-3),
            ("logs near 0, sd 1e-9", 0.0, 1e-9),
            ("logs near 30, sd 0.5", 30.0, 0.5),
        ):
            values = np.exp(rng.normal(center, spread, 400))
            memory, result = self.compare_estimates(
                FamilyId.LOGNORMAL, values, 1e-14, 1e-14, NumpySource(values)
            )
            with self.subTest(case=name):
                self.assertEqual(
                    (result.parameters["mu_log"], result.parameters["sigma_log"]),
                    gaussian_estimates(np.log(values)),
                )
                self.compare_uncertainty(memory, result, values, estimate=1e-13, profile=1e-12)


class GammaDifferentialTests(Comparison):
    def test_agreement_is_bounded_by_the_in_memory_search_resolution(self) -> None:
        for size in (30, 400):
            for seed in (1, 2, 3):
                values = dataset(FamilyId.GAMMA, size, seed)
                with self.subTest(size=size, seed=seed):
                    memory, result = self.compare_estimates(
                        FamilyId.GAMMA, values, 1e-6, 1e-13, NumpySource(values)
                    )
                    self.compare_uncertainty(
                        memory, result, values, estimate=1e-6, profile=1e-11
                    )

    def test_the_scale_shape_is_the_root_to_binary64(self) -> None:
        for seed in (1, 2, 3):
            values = dataset(FamilyId.GAMMA, 500, seed)
            shape, scale = gamma_estimates(values)
            result = fit(NumpySource(values), "gamma").fit
            with self.subTest(seed=seed):
                self.assertLess(abs(result.parameters["shape"] / shape - 1.0), 1e-13)
                self.assertLess(abs(result.parameters["scale"] / scale - 1.0), 1e-13)


class ParquetDifferentialTests(Comparison):
    def test_a_parquet_layout_gives_the_array_result_and_the_in_memory_estimates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for family in FAMILIES:
                values = dataset(family, 300, 7)
                folder = root / family.value
                write_parquet(folder / "a.parquet", {"x": values[:170]}, row_group_size=64)
                write_parquet(folder / "b.parquet", {"x": values[170:]}, row_group_size=50)
                with self.subTest(family=family):
                    expected = fit(NumpySource(values), family).fit
                    actual = fit(ParquetSource(folder, "x"), family, max_rows=33).fit
                    self.assertEqual(actual, expected)
                    self.assertEqual(
                        actual.uncertainty().covariance,  # type: ignore[union-attr]
                        expected.uncertainty().covariance,  # type: ignore[union-attr]
                    )
                    memory = in_memory_fit(family, observations(family, values))
                    tolerance = 1e-6 if family is FamilyId.GAMMA else 1e-13
                    for name, value in memory.parameters.items():
                        self.assertClose(actual.parameters[name], value, tolerance, name)


if __name__ == "__main__":
    unittest.main()
