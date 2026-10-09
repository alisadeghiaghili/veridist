"""Seeded calibration and power evidence for the multi-family refit Monte Carlo test.

**Size.**  For each non-exponential family and each cell of a small declared
grid, datasets are simulated from the family itself, the refit Monte Carlo test
of the Anderson-Darling statistic is run on each, and the share of datasets
rejected at ``alpha = 0.10`` is compared with ``alpha`` within
``TOLERANCE_SES`` binomial standard errors (``summarize_calibration``).

With ``REPLICATES = 19`` the p-value ``(exceedances + 1) / 20`` is at most
``0.10`` exactly when at most one replicate statistic reaches the observed one,
so a statistic whose null distribution does not depend on the parameters (the
location-scale families normal and Gumbel, and through the logarithm the
lognormal and Weibull) is rejected with probability exactly ``0.10``.  The gamma
shape is not a location-scale parameter, so its size is only approximately the
nominal one; its cells check that approximation, not exactness.

The grid, the seeds and the sizes are fixed in this file.  The evidence is for
that grid only: it says nothing about other sample sizes or parameter values,
and a rejection-rate check of this size cannot detect a small distortion (the
tolerance is several binomial standard errors).

**Power.**  Clearly misspecified samples are rejected.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from veridist import sample
from veridist.inference import (
    GofStatistic,
    assess_families,
    refit_monte_carlo_gof,
    summarize_calibration,
)

ALPHA = 0.10
REPLICATES = 19
TOLERANCE_SES = 3.5
AD = frozenset({GofStatistic.AD})

# family, parameters, sample size, number of simulated datasets, seed
SIZE_GRID: tuple[tuple[str, dict[str, float], int, int, int], ...] = (
    ("normal", {"mu": 10.0, "sigma": 2.0}, 30, 150, 20260101),
    ("normal", {"mu": 10.0, "sigma": 2.0}, 100, 80, 20260102),
    ("gamma", {"shape": 0.8, "scale": 2.0}, 30, 150, 20260103),
    ("gamma", {"shape": 5.0, "scale": 1.0}, 100, 80, 20260104),
    ("weibull_min", {"shape": 0.8, "scale": 3.0}, 30, 150, 20260105),
    ("weibull_min", {"shape": 3.0, "scale": 3.0}, 100, 80, 20260106),
    ("lognormal", {"mu_log": 1.0, "sigma_log": 0.5}, 30, 150, 20260107),
    ("lognormal", {"mu_log": 1.0, "sigma_log": 0.5}, 100, 80, 20260108),
    ("gumbel_right", {"location": 5.0, "scale": 2.0}, 30, 150, 20260109),
    ("gumbel_right", {"location": 5.0, "scale": 2.0}, 100, 80, 20260110),
)

# source family, source parameters, family tested, sample size, seed
POWER_GRID: tuple[tuple[str, dict[str, float], str, int, int], ...] = (
    ("lognormal", {"mu_log": 0.0, "sigma_log": 1.0}, "normal", 50, 20260201),
    ("exponential", {"rate": 1.0}, "gumbel_right", 100, 20260202),
    ("weibull_min", {"shape": 3.0, "scale": 2.0}, "exponential", 60, 20260203),
)
POWER_DATASETS = 8
POWER_REPLICATES = 39


def rejection_count(
    family: str, parameters: dict[str, float], size: int, datasets: int, seed: int
) -> tuple[int, int]:
    """Simulate ``datasets`` samples from the family and count the rejections at ``ALPHA``."""

    rng = np.random.default_rng(seed)
    rejections = 0
    failed = 0
    for _ in range(datasets):
        observations = sample(family, size, rng=rng, **parameters)
        result = refit_monte_carlo_gof(
            observations=observations,
            family=family,
            statistics=AD,
            replicates=REPLICATES,
            rng=rng,
        )
        failed += result.failed_replicates
        if result.p_values[GofStatistic.AD] <= ALPHA + 1e-12:
            rejections += 1
    return rejections, failed


class SizeCalibrationTests(unittest.TestCase):
    def test_rejection_rate_matches_the_nominal_level_on_the_declared_grid(self) -> None:
        for family, parameters, size, datasets, seed in SIZE_GRID:
            with self.subTest(family=family, parameters=parameters, n=size):
                rejections, failed = rejection_count(family, parameters, size, datasets, seed)
                summary = summarize_calibration(
                    rejections=rejections, replicates=datasets, nominal_alpha=ALPHA
                )
                self.assertEqual(summary.scope, "declared_grid_only")
                self.assertEqual(failed, 0, "refits of well-behaved samples must not fail")
                deviation = abs(summary.rejection_rate - ALPHA)
                self.assertLessEqual(
                    deviation,
                    TOLERANCE_SES * summary.standard_error,
                    f"{family} n={size}: rejected {rejections} of {datasets} at alpha={ALPHA}",
                )


class PowerTests(unittest.TestCase):
    def test_clearly_misspecified_samples_are_rejected(self) -> None:
        for source, parameters, tested, size, seed in POWER_GRID:
            with self.subTest(source=source, tested=tested, n=size):
                rng = np.random.default_rng(seed)
                p_values: list[float] = []
                for _ in range(POWER_DATASETS):
                    observations = sample(source, size, rng=rng, **parameters)
                    result = refit_monte_carlo_gof(
                        observations=observations,
                        family=tested,
                        statistics=AD,
                        replicates=POWER_REPLICATES,
                        rng=rng,
                    )
                    p_values.append(result.p_values[GofStatistic.AD])
                self.assertTrue(
                    all(value <= ALPHA for value in p_values),
                    f"{source} data tested as {tested} was not always rejected: {p_values}",
                )

    def test_the_assessment_does_not_select_a_clearly_wrong_family(self) -> None:
        rng = np.random.default_rng(20260301)
        observations = sample("lognormal", 60, rng=rng, mu_log=0.0, sigma_log=1.0)
        result = assess_families(
            observations=observations,
            replicates=POWER_REPLICATES,
            rng=rng,
            adequacy_threshold=0.10,
        )
        rows: dict[str, Any] = {row.family.value: row for row in result.candidates}
        self.assertFalse(rows["normal"].adequate)
        self.assertFalse(rows["exponential"].adequate)
        self.assertIsNotNone(result.selection.selected_family)
        self.assertNotIn(result.selection.selected_family, ("normal", "exponential"))


if __name__ == "__main__":
    unittest.main()
