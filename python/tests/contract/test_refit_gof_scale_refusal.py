"""Refit Monte Carlo goodness of fit refuses a sample too large to simulate, never subsamples it.

Every replicate draws and refits as many observations as the sample has, so the cost grows with
the number of rows times the number of replicates. Above a declared number of observations (and
for a scale source, which cannot be simulated at all) the run is refused with a typed reason before
anything is fitted or drawn; the limit is the caller's to raise.
"""

from __future__ import annotations

import itertools
import pickle
import unittest
from typing import Any
from unittest import mock

import numpy as np

from veridist import inference
from veridist.engine.errors import VeridistError
from veridist.inference import (
    REFIT_MAX_OBSERVATIONS,
    GofStatistic,
    RefitGofRefused,
    RefitRefusedReason,
    assess_families,
    refit_monte_carlo_gof,
)
from veridist.scale._sources import NumpySource

KS = frozenset({GofStatistic.KS})


def values(size: int) -> list[float]:
    return [float(item) for item in np.random.default_rng(3).exponential(2.0, size)]


class RefusalTests(unittest.TestCase):
    def test_the_declared_limit_is_pinned(self) -> None:
        self.assertEqual(REFIT_MAX_OBSERVATIONS, 100_000)

    def test_a_sample_above_the_limit_is_refused_with_its_reason(self) -> None:
        rng = np.random.default_rng(1)
        before = rng.bit_generator.state
        with self.assertRaises(RefitGofRefused) as caught:
            refit_monte_carlo_gof(
                observations=values(11),
                family="exponential",
                statistics=KS,
                replicates=5,
                rng=rng,
                max_observations=10,
            )
        error = caught.exception
        self.assertEqual(error.code, "SCALE_REFIT_REFUSED")
        self.assertIs(error.reason, RefitRefusedReason.ROW_LIMIT_EXCEEDED)
        self.assertEqual(error.limit, 10)
        self.assertIn("at most 10 observations", str(error))
        self.assertIn("large-n goodness of fit", str(error))
        self.assertIsInstance(error, VeridistError)
        self.assertIsInstance(error, ValueError)
        self.assertEqual(rng.bit_generator.state, before)

    def test_a_sample_at_the_limit_is_accepted(self) -> None:
        result = refit_monte_carlo_gof(
            observations=values(10),
            family="exponential",
            statistics=KS,
            replicates=5,
            rng=np.random.default_rng(1),
            max_observations=10,
        )
        self.assertEqual(result.requested_replicates, 5)

    def test_every_family_refuses_above_the_limit(self) -> None:
        for family in (
            "exponential",
            "normal",
            "gamma",
            "weibull_min",
            "lognormal",
            "gumbel_right",
        ):
            with self.subTest(family=family), self.assertRaises(RefitGofRefused):
                refit_monte_carlo_gof(
                    observations=values(40),
                    family=family,
                    statistics=KS,
                    replicates=2,
                    rng=np.random.default_rng(1),
                    max_observations=39,
                )

    def test_the_rest_of_a_large_stream_is_never_read(self) -> None:
        consumed = itertools.count()

        def stream() -> Any:
            for index in consumed:
                yield 1.0 + index

        with self.assertRaises(RefitGofRefused):
            refit_monte_carlo_gof(
                observations=stream(),
                family="exponential",
                statistics=KS,
                replicates=2,
                rng=np.random.default_rng(1),
                max_observations=25,
            )
        self.assertEqual(next(consumed), 26)

    def test_a_sized_sample_is_refused_without_being_copied(self) -> None:
        big = np.ones(1000)
        with mock.patch.object(inference, "islice", side_effect=AssertionError("read")):
            with self.assertRaises(RefitGofRefused):
                refit_monte_carlo_gof(
                    observations=big,
                    family="exponential",
                    statistics=KS,
                    replicates=2,
                    rng=np.random.default_rng(1),
                    max_observations=999,
                )

    def test_a_scale_source_is_refused_whatever_its_size(self) -> None:
        source = NumpySource(np.ones(5))
        with self.assertRaises(RefitGofRefused) as caught:
            refit_monte_carlo_gof(
                observations=source,  # type: ignore[arg-type]
                family="exponential",
                statistics=KS,
                replicates=2,
                rng=np.random.default_rng(1),
            )
        self.assertIs(caught.exception.reason, RefitRefusedReason.SCALE_SOURCE)
        self.assertEqual(caught.exception.limit, REFIT_MAX_OBSERVATIONS)
        with self.assertRaises(RefitGofRefused) as caught:
            assess_families(
                observations=source,  # type: ignore[arg-type]
                replicates=2,
                rng=np.random.default_rng(1),
            )
        self.assertIs(caught.exception.reason, RefitRefusedReason.SCALE_SOURCE)

    def test_the_assessment_is_refused_before_any_family_is_fitted(self) -> None:
        with mock.patch.object(inference, "_observed_fit", side_effect=AssertionError("fitted")):
            with self.assertRaises(RefitGofRefused) as caught:
                assess_families(
                    observations=values(30),
                    replicates=2,
                    rng=np.random.default_rng(1),
                    max_observations=29,
                )
        self.assertIs(caught.exception.reason, RefitRefusedReason.ROW_LIMIT_EXCEEDED)
        result = assess_families(
            observations=values(30),
            families=["exponential"],
            replicates=2,
            rng=np.random.default_rng(1),
            max_observations=30,
        )
        self.assertEqual(result.sample_size, 30)

    def test_the_limit_is_validated(self) -> None:
        for bad in (0, -1, 2.5, "3", None, True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                refit_monte_carlo_gof(
                    observations=values(5),
                    family="exponential",
                    statistics=KS,
                    replicates=2,
                    rng=np.random.default_rng(1),
                    max_observations=bad,  # type: ignore[arg-type]
                )
            with self.subTest(bad=bad, call="assess"), self.assertRaises(ValueError):
                assess_families(
                    observations=values(5),
                    replicates=2,
                    rng=np.random.default_rng(1),
                    max_observations=bad,  # type: ignore[arg-type]
                )

    def test_the_error_pickles_and_is_exported(self) -> None:
        error = RefitGofRefused(RefitRefusedReason.SCALE_SOURCE, 7)
        copy = pickle.loads(pickle.dumps(error))
        self.assertIs(copy.reason, RefitRefusedReason.SCALE_SOURCE)
        self.assertEqual((copy.limit, copy.code), (7, "SCALE_REFIT_REFUSED"))
        for name in ("RefitGofRefused", "RefitRefusedReason", "REFIT_MAX_OBSERVATIONS"):
            self.assertIn(name, inference.__all__)
        self.assertEqual(
            [item.value for item in RefitRefusedReason], ["ROW_LIMIT_EXCEEDED", "SCALE_SOURCE"]
        )


if __name__ == "__main__":
    unittest.main()
