"""RED contracts for refit GoF, bootstrap and model selection."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch


class V1InferenceSelectionTests(unittest.TestCase):
    def test_information_criteria_use_declared_free_parameters(self) -> None:
        from veridist.inference import information_criteria

        result = information_criteria(log_likelihood=-10.0, sample_size=100, free_parameters=2)
        self.assertAlmostEqual(result.aic, 24.0)
        self.assertAlmostEqual(result.bic, 20.0 + 2.0 * math.log(100.0))

    def test_refit_monte_carlo_records_failures_and_uncertainty(self) -> None:
        import numpy as np

        from veridist.inference import GofStatistic, refit_monte_carlo_gof

        result = refit_monte_carlo_gof(
            observations=(0.2, 0.4, 0.8, 1.6),
            family="exponential",
            statistics=frozenset({GofStatistic.KS, GofStatistic.AD, GofStatistic.CVM}),
            replicates=40,
            rng=np.random.default_rng(7),
        )
        self.assertEqual(result.requested_replicates, 40)
        self.assertEqual(result.successful_replicates + result.failed_replicates, 40)
        self.assertGreater(result.successful_replicates, 0)
        self.assertGreater(result.monte_carlo_standard_error, 0.0)
        self.assertEqual(result.method, "refit_monte_carlo")
        self.assertEqual(result.rng_policy, "caller_owned_generator")
        # "AD" sorts before "CVM" and "KS", so it is the primary statistic
        # whose p-value the standard error and interval above describe.
        self.assertEqual(result.primary_statistic, GofStatistic.AD)

    def test_model_selection_returns_none_adequate(self) -> None:
        from veridist.inference import SelectionCode, compare_models

        result = compare_models(
            candidates=(
                {"family": "normal", "aic": 10.0, "bic": 12.0, "p_value": 0.001},
                {"family": "lognormal", "aic": 11.0, "bic": 13.0, "p_value": 0.01},
            ),
            adequacy_threshold=0.05,
        )
        self.assertEqual(result.code, SelectionCode.NONE_ADEQUATE)
        self.assertIsNone(result.selected_family)

    def test_information_criteria_reject_invalid_inputs(self) -> None:
        from veridist.inference import information_criteria

        invalid = (
            {"log_likelihood": float("nan"), "sample_size": 2, "free_parameters": 1},
            {"log_likelihood": -1.0, "sample_size": True, "free_parameters": 1},
            {"log_likelihood": -1.0, "sample_size": 0, "free_parameters": 1},
            {"log_likelihood": -1.0, "sample_size": 2, "free_parameters": -1},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                information_criteria(**arguments)

    def test_refit_gof_rejects_invalid_contracts_and_all_failed_refits(self) -> None:
        import numpy as np

        import veridist.inference as inference
        from veridist.inference import GofStatistic, refit_monte_carlo_gof

        generator = np.random.default_rng(4)
        common = {
            "observations": (0.5, 1.0),
            "family": "exponential",
            "statistics": frozenset({GofStatistic.KS}),
            "replicates": 1,
            "rng": generator,
        }
        invalid = (
            {**common, "family": "cauchy"},
            {**common, "statistics": frozenset()},
            {**common, "statistics": frozenset({"KS"})},
            {**common, "replicates": 0},
            {**common, "rng": object()},
            {**common, "observations": ()},
            {**common, "observations": (float("nan"),)},
            {**common, "observations": (0.0,)},
        )
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises((TypeError, ValueError)):
                refit_monte_carlo_gof(**arguments)

        # An unknown family is a ValueError naming the family problem.
        with self.assertRaisesRegex(ValueError, "unknown evaluated family"):
            refit_monte_carlo_gof(**{**common, "family": "cauchy"})
        # A two-parameter family needs at least three observations; two are
        # enough for the exponential cell, which has one parameter.
        for family in ("normal", "gamma", "weibull_min", "lognormal", "gumbel_right"):
            with self.subTest(family=family), self.assertRaisesRegex(
                ValueError, "at least 3 observations"
            ):
                refit_monte_carlo_gof(**{**common, "family": family})
        self.assertEqual(refit_monte_carlo_gof(**common).successful_replicates, 1)

        observed = inference._empirical_statistics((0.5, 1.0))
        with patch.object(
            inference,
            "_empirical_statistics",
            side_effect=(observed, ArithmeticError("synthetic refit failure")),
        ):
            with self.assertRaises(RuntimeError):
                refit_monte_carlo_gof(**common)

    def test_model_selection_validates_evidence_and_selects_lowest_aic(self) -> None:
        from veridist.inference import SelectionCode, compare_models

        result = compare_models(
            candidates=(
                {"family": "normal", "aic": 12.0, "p_value": 0.2},
                {"family": "gamma", "aic": 8.0, "p_value": 0.1},
            ),
            adequacy_threshold=0.05,
        )
        self.assertEqual(result.code, SelectionCode.SELECTED)
        self.assertEqual(result.selected_family, "gamma")

        invalid = (
            (({"family": "normal", "aic": 1.0, "p_value": 0.2},), float("nan")),
            ((object(),), 0.05),
            (({"family": 1, "aic": 1.0, "p_value": 0.2},), 0.05),
            (({"family": "normal", "aic": float("inf"), "p_value": 0.2},), 0.05),
        )
        for candidates, threshold in invalid:
            with self.subTest(candidates=candidates), self.assertRaises((TypeError, ValueError)):
                compare_models(candidates=candidates, adequacy_threshold=threshold)


if __name__ == "__main__":
    unittest.main()
