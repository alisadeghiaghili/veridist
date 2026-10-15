"""Millions of rows per family through Parquet layouts give one fit, bit for bit.

Every layout stores the same multiset of values (one of them shuffled) and every fit
must produce the result bits and the state digest of the in-memory array fit. The
estimates are checked against independent arithmetic (``math.fsum`` and ``mpmath``) and
the uncertainty, computed from the sufficient statistics alone, against the closed forms
of the observed information. The data are seeded; nothing is random between runs.
"""

from __future__ import annotations

import math
import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any

import mpmath
import numpy as np

from tests.unit.scale_arrow_fixtures import write_parquet
from veridist.families.registry import FamilyId
from veridist.scale._arrow import ParquetSource
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._sources import NumpySource

SIZE = 3_000_000
SEED = 20261014

TRUTH = {
    FamilyId.EXPONENTIAL: {"rate": 1.0 / 3.0},
    FamilyId.NORMAL: {"mu": 10.0, "sigma": 3.0},
    FamilyId.GAMMA: {"shape": 3.0, "scale": 2.0},
    FamilyId.LOGNORMAL: {"mu_log": 1.0, "sigma_log": 0.5},
}


def sample(family: FamilyId) -> np.ndarray:
    rng = np.random.default_rng(SEED + sum(map(ord, family.value)))
    if family is FamilyId.EXPONENTIAL:
        return rng.exponential(3.0, SIZE)
    if family is FamilyId.NORMAL:
        return rng.normal(10.0, 3.0, SIZE)
    if family is FamilyId.GAMMA:
        return rng.gamma(3.0, 2.0, SIZE)
    return rng.lognormal(1.0, 0.5, SIZE)


def write_files(root: Path, values: np.ndarray, sizes: list[int], group: int) -> ParquetSource:
    assert sum(sizes) == len(values)
    start = 0
    for index, size in enumerate(sizes):
        write_parquet(
            root / f"part{index % 4}" / f"f{index:03d}.parquet",
            {"x": values[start : start + size]},
            row_group_size=group,
        )
        start += size
    return ParquetSource(root, "x")


def result_bits(result: ScaleFitResult) -> tuple[bytes, ...]:
    success: Any = result.fit
    numbers = [*success.parameters.values(), success.log_likelihood]
    uncertainty = success.uncertainty()
    for row in uncertainty.covariance:
        numbers.extend(row)
    for method in ("wald", "profile"):
        for lower, upper in uncertainty.confidence_intervals(0.95, method).values():
            numbers.extend((lower, upper))
    return tuple(struct.pack("<d", float(value)) for value in numbers)


def digest(result: ScaleFitResult) -> str:
    assert result.scale.state is not None
    return result.scale.state.digest_sha256


def fsum_gaussian(values: np.ndarray) -> tuple[float, float]:
    mean = math.fsum(values.tolist()) / len(values)
    deviations = values - mean
    return mean, math.sqrt(math.fsum((deviations * deviations).tolist()) / len(values))


class LayoutSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        base = Path(cls._directory.name)
        cls.values: dict[FamilyId, np.ndarray] = {}
        cls.base: dict[FamilyId, ScaleFitResult] = {}
        cls.layouts: dict[FamilyId, dict[str, ScaleFitResult]] = {}
        order = np.random.default_rng(5).permutation(SIZE)
        for family in TRUTH:
            values = sample(family)
            cls.values[family] = values
            cls.base[family] = fit(NumpySource(values), family)
            root = base / family.value
            single = write_files(root / "a", values, [SIZE], 1_000_000)
            uneven = write_files(
                root / "b", values, [1_200_000, 900_000, 900_000], 250_000
            )
            shuffled = write_files(root / "c", values[order], [500_000] * 6, 131_072)
            cls.layouts[family] = {
                "one file, groups of 1M": fit(single, family),
                "three uneven files, groups of 250k": fit(uneven, family, max_rows=100_000),
                "six shuffled files, groups of 131072": fit(shuffled, family, max_rows=77_777),
            }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_every_layout_gives_the_array_result_and_state_digest(self) -> None:
        for family, results in self.layouts.items():
            base = self.base[family]
            for name, result in results.items():
                with self.subTest(family=family.value, layout=name):
                    self.assertEqual(result.fit, base.fit)
                    self.assertEqual(result_bits(result), result_bits(base))
                    self.assertEqual(digest(result), digest(base))
                    self.assertEqual(result.scale.row_count, SIZE)

    def test_the_partition_lists_describe_each_layout(self) -> None:
        counts = {"one file, groups of 1M": 3, "three uneven files, groups of 250k": 13}
        for family, results in self.layouts.items():
            for name, expected in counts.items():
                self.assertEqual(results[name].scale.partition_count, expected, family.value)
            shuffled = results["six shuffled files, groups of 131072"].scale
            self.assertEqual(shuffled.partition_count, 6 * 4)
            self.assertEqual(sum(item.rows or 0 for item in shuffled.partitions), SIZE)

    def test_the_estimates_agree_with_independent_arithmetic(self) -> None:
        exponential = self.base[FamilyId.EXPONENTIAL].fit
        values = self.values[FamilyId.EXPONENTIAL]
        self.assertEqual(
            exponential.parameters["rate"], SIZE / math.fsum(values.tolist())  # type: ignore[union-attr]
        )
        normal = self.base[FamilyId.NORMAL].fit
        mean, sd = fsum_gaussian(self.values[FamilyId.NORMAL])
        self.assertLessEqual(abs(normal.parameters["mu"] - mean), 2 * math.ulp(mean))  # type: ignore[union-attr]
        self.assertLess(abs(normal.parameters["sigma"] / sd - 1.0), 1e-13)  # type: ignore[union-attr]
        lognormal = self.base[FamilyId.LOGNORMAL].fit
        mean, sd = fsum_gaussian(np.log(self.values[FamilyId.LOGNORMAL]))
        self.assertLessEqual(abs(lognormal.parameters["mu_log"] - mean), 2 * math.ulp(mean))  # type: ignore[union-attr]
        self.assertLess(abs(lognormal.parameters["sigma_log"] / sd - 1.0), 1e-13)  # type: ignore[union-attr]

    def test_the_gamma_shape_solves_the_profile_equation(self) -> None:
        values = self.values[FamilyId.GAMMA]
        gamma = self.base[FamilyId.GAMMA].fit
        mean = math.fsum(values.tolist()) / SIZE
        mean_log = math.fsum(np.log(values).tolist()) / SIZE
        gap = mpmath.log(mean) - mean_log
        shape = float(gamma.parameters["shape"])  # type: ignore[union-attr]
        residual = abs(mpmath.log(shape) - mpmath.digamma(shape) - gap)
        self.assertLess(float(residual / gap), 1e-9)
        self.assertLess(abs(gamma.parameters["scale"] * shape / mean - 1.0), 1e-13)  # type: ignore[union-attr]

    def test_every_estimate_is_within_five_standard_errors_of_the_truth(self) -> None:
        for family, base in self.base.items():
            uncertainty: Any = base.fit.uncertainty()  # type: ignore[union-attr]
            for name, truth in TRUTH[family].items():
                error = uncertainty.standard_errors[name]
                with self.subTest(family=family.value, parameter=name):
                    self.assertLess(abs(base.fit.parameters[name] - truth), 5.0 * error)  # type: ignore[union-attr]

    def test_the_standard_errors_follow_the_closed_forms(self) -> None:
        n = float(SIZE)
        exponential: Any = self.base[FamilyId.EXPONENTIAL].fit
        rate = exponential.parameters["rate"]
        self.assertLess(
            abs(exponential.uncertainty().standard_errors["rate"] / (rate / math.sqrt(n)) - 1.0),
            1e-12,
        )
        for family, mean_name, sd_name in (
            (FamilyId.NORMAL, "mu", "sigma"),
            (FamilyId.LOGNORMAL, "mu_log", "sigma_log"),
        ):
            result: Any = self.base[family].fit
            sigma = result.parameters[sd_name]
            errors = result.uncertainty().standard_errors
            with self.subTest(family=family.value):
                self.assertLess(abs(errors[mean_name] / (sigma / math.sqrt(n)) - 1.0), 1e-12)
                self.assertLess(abs(errors[sd_name] / (sigma / math.sqrt(2.0 * n)) - 1.0), 1e-12)
        gamma: Any = self.base[FamilyId.GAMMA].fit
        shape, scale = gamma.parameters["shape"], gamma.parameters["scale"]
        trigamma = float(mpmath.psi(1, shape))
        determinant = shape * trigamma - 1.0
        errors = gamma.uncertainty().standard_errors
        self.assertLess(
            abs(errors["shape"] / math.sqrt(shape / (n * determinant)) - 1.0), 1e-9
        )
        self.assertLess(
            abs(errors["scale"] / math.sqrt(scale * scale * trigamma / (n * determinant)) - 1.0),
            1e-9,
        )

    def test_profile_and_wald_intervals_agree_at_this_size(self) -> None:
        for family, base in self.base.items():
            uncertainty: Any = base.fit.uncertainty()  # type: ignore[union-attr]
            wald = uncertainty.confidence_intervals(0.95, "wald")
            profile = uncertainty.confidence_intervals(0.95, "profile")
            for name in wald:
                error = uncertainty.standard_errors[name]
                with self.subTest(family=family.value, parameter=name):
                    for side in (0, 1):
                        self.assertLess(abs(profile[name][side] - wald[name][side]), 1e-3 * error)

    def test_derived_quantities_have_ordered_intervals(self) -> None:
        for family, base in self.base.items():
            uncertainty: Any = base.fit.uncertainty()  # type: ignore[union-attr]
            for label, derived in (
                ("mean", uncertainty.mean(0.95)),
                ("q10", uncertainty.quantile(0.1)),
            ):
                with self.subTest(family=family.value, quantity=label):
                    self.assertLess(derived.lower, derived.estimate)
                    self.assertLess(derived.estimate, derived.upper)


class LargeOffsetSweepTests(unittest.TestCase):
    """Millions of values with a spread of one part in a billion of their size."""

    def test_the_spread_is_estimated_and_layout_does_not_matter(self) -> None:
        values = np.random.default_rng(SEED).normal(1e9, 1.0, SIZE)
        base = fit(NumpySource(values), "normal")
        with tempfile.TemporaryDirectory() as directory:
            source = write_files(
                Path(directory), values[::-1].copy(), [1_800_000, 1_200_000], 400_000
            )
            other = fit(source, "normal", max_rows=123_457)
        self.assertEqual(other.fit, base.fit)
        self.assertEqual(digest(other), digest(base))
        mean = math.fsum(values.tolist()) / SIZE
        deviations = values - mean
        sd = math.sqrt(math.fsum((deviations * deviations).tolist()) / SIZE)
        success: Any = base.fit
        self.assertLessEqual(abs(success.parameters["mu"] - mean), 2 * math.ulp(mean))
        self.assertLess(abs(success.parameters["sigma"] / sd - 1.0), 1e-12)
        self.assertLess(abs(success.parameters["sigma"] - 1.0), 5e-3)
        errors = success.uncertainty().standard_errors
        expected = success.parameters["sigma"] / math.sqrt(SIZE)
        self.assertLess(abs(errors["mu"] / expected - 1.0), 1e-9)


if __name__ == "__main__":
    unittest.main()
