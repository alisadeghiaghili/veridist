"""Golden values for the exponential refit Monte Carlo cell.

The exponential path of ``refit_monte_carlo_gof`` predates the multi-family
generalisation and must keep producing exactly the same numbers for a given
sample, seed and request.  The values below were recorded from the
exponential-only implementation of 2.0.0; every field is compared exactly.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from veridist.inference import GofStatistic, RefitMonteCarloGof, refit_monte_carlo_gof

ALL_STATISTICS = frozenset({GofStatistic.KS, GofStatistic.AD, GofStatistic.CVM})

# (observations, statistics, replicates, seed, expected)
GOLDEN: tuple[tuple[Any, ...], ...] = (
    (
        (0.2, 0.4, 0.8, 1.6),
        ALL_STATISTICS,
        40,
        7,
        {
            "p_values": {
                "AD": 0.9512195121951219,
                "CVM": 0.975609756097561,
                "KS": 0.8536585365853658,
            },
            "successful": 40,
            "failed": 0,
            "standard_error": 0.03405912205797304,
            "interval": (0.8844636329614948, 1.0),
            "primary": "AD",
        },
    ),
    (
        (0.5, 1.0),
        frozenset({GofStatistic.KS}),
        25,
        3,
        {
            "p_values": {"KS": 0.38461538461538464},
            "successful": 25,
            "failed": 0,
            "standard_error": 0.09730085108210398,
            "interval": (0.19390571649446084, 0.5753250527363084),
            "primary": "KS",
        },
    ),
    (
        (0.31, 0.62, 0.9, 1.4, 2.2, 3.5, 0.15, 0.77),
        frozenset({GofStatistic.CVM, GofStatistic.KS}),
        60,
        2024,
        {
            "p_values": {"CVM": 0.9836065573770492, "KS": 0.9672131147540983},
            "successful": 60,
            "failed": 0,
            "standard_error": 0.016393442622950827,
            "interval": (0.9514754098360656, 1.0),
            "primary": "CVM",
        },
    ),
    (
        (1.0, 1.0, 1.0, 1.0, 1.0),
        frozenset({GofStatistic.AD}),
        30,
        11,
        {
            "p_values": {"AD": 0.03225806451612903},
            "successful": 30,
            "failed": 0,
            "standard_error": 0.03225806451612903,
            "interval": (0.0, 0.09548387096774193),
            "primary": "AD",
        },
    ),
    (
        (0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6),
        ALL_STATISTICS,
        100,
        99,
        {
            "p_values": {
                "AD": 0.009900990099009901,
                "CVM": 0.009900990099009901,
                "KS": 0.009900990099009901,
            },
            "successful": 100,
            "failed": 0,
            "standard_error": 0.009900990099009901,
            "interval": (0.0, 0.029306930693069305),
            "primary": "AD",
        },
    ),
)


class ExponentialRefitGoldenTests(unittest.TestCase):
    def assert_golden(
        self, result: RefitMonteCarloGof, replicates: int, expected: dict[str, Any]
    ) -> None:
        self.assertEqual(result.requested_replicates, replicates)
        self.assertEqual(result.successful_replicates, expected["successful"])
        self.assertEqual(result.failed_replicates, expected["failed"])
        self.assertEqual(result.monte_carlo_standard_error, expected["standard_error"])
        self.assertEqual(result.interval, expected["interval"])
        self.assertEqual(
            {statistic.value: value for statistic, value in result.p_values.items()},
            expected["p_values"],
        )
        self.assertEqual(result.primary_statistic.value, expected["primary"])
        self.assertEqual(result.method, "refit_monte_carlo")
        self.assertEqual(result.rng_policy, "caller_owned_generator")

    def test_exponential_results_are_pinned(self) -> None:
        for observations, statistics, replicates, seed, expected in GOLDEN:
            with self.subTest(observations=observations, seed=seed):
                result = refit_monte_carlo_gof(
                    observations=observations,
                    family="exponential",
                    statistics=statistics,
                    replicates=replicates,
                    rng=np.random.default_rng(seed),
                )
                self.assert_golden(result, replicates, expected)

    def test_exponential_accepts_the_family_id_with_identical_results(self) -> None:
        from veridist.families.registry import FamilyId

        observations, statistics, replicates, seed, expected = GOLDEN[0]
        result = refit_monte_carlo_gof(
            observations=observations,
            family=FamilyId.EXPONENTIAL,
            statistics=statistics,
            replicates=replicates,
            rng=np.random.default_rng(seed),
        )
        self.assert_golden(result, replicates, expected)

    def test_exponential_results_do_not_depend_on_the_observation_container(self) -> None:
        observations, statistics, replicates, seed, expected = GOLDEN[2]
        for container in (list, tuple, np.array, iter):
            with self.subTest(container=container.__name__):
                result = refit_monte_carlo_gof(
                    observations=container(observations),
                    family="exponential",
                    statistics=statistics,
                    replicates=replicates,
                    rng=np.random.default_rng(seed),
                )
                self.assert_golden(result, replicates, expected)


if __name__ == "__main__":
    unittest.main()
