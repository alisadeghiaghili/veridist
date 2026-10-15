"""Millions of rows through Parquet layouts give one multi-pass fit, bit for bit.

Every layout stores the same multiset of values (one of them shuffled) and every fit of
``weibull_min`` and ``gumbel_right`` must produce the result bits, the state digest and the
pass record of the in-memory array fit. The estimates are checked against the truth of the
simulation, against the in-memory fits on a subsample and, at 4e4 rows, against the
50-digit profile equation (one Newton step at the fitted parameter measures its distance to
the exact root). The data are seeded; nothing is random between runs.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any

import mpmath
import numpy as np

from tests.unit.scale_arrow_fixtures import write_parquet
from veridist.domain.lifetimes import ExactLifetime
from veridist.domain.values import ExactValue
from veridist.families.gumbel import fit_gumbel_right
from veridist.families.weibull import fit_weibull
from veridist.scale._arrow import ParquetSource
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._sources import NumpySource

SIZE = 1_200_000
SEED = 20261015
SUBSAMPLE = 20_000
EXACT_SIZE = 40_000

TRUTH = {
    "weibull_min": {"shape": 1.7, "scale": 1000.0},
    "gumbel_right": {"location": 50.0, "scale": 4.0},
}


def sample(family: str, size: int = SIZE) -> np.ndarray:
    rng = np.random.default_rng(SEED + sum(map(ord, family)))
    if family == "weibull_min":
        return 1000.0 * rng.weibull(1.7, size)
    return rng.gumbel(50.0, 4.0, size)


def write_files(root: Path, values: np.ndarray, sizes: list[int], group: int) -> ParquetSource:
    assert sum(sizes) == len(values)
    start = 0
    for index, size in enumerate(sizes):
        write_parquet(
            root / f"part{index % 3}" / f"f{index:03d}.parquet",
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
    for lower, upper in uncertainty.confidence_intervals(0.95, "wald").values():
        numbers.extend((lower, upper))
    return tuple(struct.pack("<d", float(value)) for value in numbers)


def newton_gap_weibull(values: np.ndarray, shape: float) -> float:
    """``|Newton step| / shape`` of the exact profile score at ``shape``, at 30 digits."""

    with mpmath.workdps(30):
        logs = [mpmath.log(mpmath.mpf(float(value))) for value in values]
        count = len(logs)
        center = mpmath.fsum(logs) / count
        centred = [item - center for item in logs]
        k = mpmath.mpf(shape)
        weights = [mpmath.exp(k * item) for item in centred]
        s0 = mpmath.fsum(weights)
        s1 = mpmath.fsum(w * y for w, y in zip(weights, centred, strict=True))
        s2 = mpmath.fsum(w * y * y for w, y in zip(weights, centred, strict=True))
        score = 1 / k - s1 / s0
        slope = -1 / (k * k) - (s2 / s0 - (s1 / s0) ** 2)
        return float(abs(score / slope) / k)


def newton_gap_gumbel(values: np.ndarray, scale: float) -> float:
    """``|Newton step| / scale`` of the exact profile equation at ``scale``, at 30 digits."""

    with mpmath.workdps(30):
        points = [mpmath.mpf(float(value)) for value in values]
        count = len(points)
        low = min(points)
        mean = mpmath.fsum(points) / count
        sigma = mpmath.mpf(scale)
        weights = [mpmath.exp(-(p - low) / sigma) for p in points]
        s0 = mpmath.fsum(weights)
        s1 = mpmath.fsum(w * (p - low) for w, p in zip(weights, points, strict=True))
        s2 = mpmath.fsum(w * (p - low) ** 2 for w, p in zip(weights, points, strict=True))
        score = sigma - (mean - low) + s1 / s0
        slope = 1 + (s2 / s0 - (s1 / s0) ** 2) / (sigma * sigma)
        return float(abs(score / slope) / sigma)


class LayoutSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        base = Path(cls._directory.name)
        cls.values: dict[str, np.ndarray] = {}
        cls.base: dict[str, ScaleFitResult] = {}
        cls.layouts: dict[str, dict[str, ScaleFitResult]] = {}
        order = np.random.default_rng(5).permutation(SIZE)
        for family in TRUTH:
            values = sample(family)
            cls.values[family] = values
            cls.base[family] = fit(NumpySource(values), family)
            root = base / family
            uneven = write_files(root / "a", values, [600_000, 350_000, 250_000], 250_000)
            shuffled = write_files(root / "b", values[order], [300_000] * 4, 131_072)
            cls.layouts[family] = {
                "three uneven files, groups of 250k": fit(uneven, family, max_rows=100_000),
                "four shuffled files, groups of 131072": fit(shuffled, family, max_rows=77_777),
            }

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()

    def test_every_layout_gives_the_array_result_the_digest_and_the_passes(self) -> None:
        for family, results in self.layouts.items():
            base = self.base[family]
            for name, result in results.items():
                with self.subTest(family=family, layout=name):
                    self.assertEqual(result.fit, base.fit)
                    self.assertEqual(result_bits(result), result_bits(base))
                    assert result.scale.state is not None and base.scale.state is not None
                    self.assertEqual(
                        result.scale.state.digest_sha256, base.scale.state.digest_sha256
                    )
                    self.assertEqual(result.scale.passes, base.scale.passes)
                    self.assertEqual(result.scale.row_count, SIZE)

    def test_the_passes_are_the_typical_five_or_six(self) -> None:
        for family, base in self.base.items():
            assert base.scale.passes is not None
            with self.subTest(family=family):
                self.assertTrue(5 <= base.scale.passes.pass_count <= 6)

    def test_the_estimates_recover_the_truth_of_the_simulation(self) -> None:
        for family, base in self.base.items():
            success: Any = base.fit
            for name, truth in TRUTH[family].items():
                with self.subTest(family=family, parameter=name):
                    self.assertLess(abs(success.parameters[name] - truth) / truth, 0.01)

    def test_the_wald_intervals_cover_the_truth(self) -> None:
        for family, base in self.base.items():
            success: Any = base.fit
            intervals = success.uncertainty().confidence_intervals(0.99, "wald")
            for name, truth in TRUTH[family].items():
                lower, upper = intervals[name]
                with self.subTest(family=family, parameter=name):
                    self.assertLess(lower, truth)
                    self.assertLess(truth, upper)


class SubsampleAgreementTests(unittest.TestCase):
    def test_the_in_memory_fits_agree_on_a_subsample(self) -> None:
        weibull_values = sample("weibull_min")[:SUBSAMPLE]
        ours: Any = fit(NumpySource(weibull_values), "weibull_min").fit
        theirs: Any = fit_weibull([ExactLifetime(float(v)) for v in weibull_values])
        self.assertLess(abs(ours.shape - theirs.shape) / theirs.shape, 1e-6)
        self.assertLess(abs(ours.scale - theirs.scale) / theirs.scale, 1e-6)
        self.assertLess(
            abs(ours.log_likelihood - theirs.log_likelihood) / abs(theirs.log_likelihood), 1e-12
        )
        gumbel_values = sample("gumbel_right")[:SUBSAMPLE]
        ours_g: Any = fit(NumpySource(gumbel_values), "gumbel_right").fit
        theirs_g: Any = fit_gumbel_right([ExactValue(float(v)) for v in gumbel_values])
        self.assertLess(abs(ours_g.scale - theirs_g.scale) / theirs_g.scale, 1e-6)
        self.assertLess(abs(ours_g.location - theirs_g.location) / theirs_g.scale, 1e-6)
        self.assertLess(
            abs(ours_g.log_likelihood - theirs_g.log_likelihood) / abs(theirs_g.log_likelihood),
            1e-12,
        )

    def test_the_fit_is_the_exact_root_at_forty_thousand_rows(self) -> None:
        weibull_values = sample("weibull_min", EXACT_SIZE)
        shape = fit(NumpySource(weibull_values), "weibull_min").fit.parameters["shape"]  # type: ignore[union-attr]
        self.assertLess(newton_gap_weibull(weibull_values, shape), 1e-12)
        gumbel_values = sample("gumbel_right", EXACT_SIZE)
        scale = fit(NumpySource(gumbel_values), "gumbel_right").fit.parameters["scale"]  # type: ignore[union-attr]
        self.assertLess(newton_gap_gumbel(gumbel_values, scale), 1e-12)


if __name__ == "__main__":
    unittest.main()
