"""Public entry points accept numpy real scalars and still reject ``bool`` and ``numpy.bool_``."""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from veridist.domain import ExactLifetime, ExactValue, RightCensoredLifetime, RightCensoredValue
from veridist.families.registry import FamilyId
from veridist.families.weibull import WeibullFitFailure, fit_weibull
from veridist.inference import (
    GofStatistic,
    SelectionCode,
    compare_models,
    information_criteria,
    refit_monte_carlo_gof,
    summarize_calibration,
)
from veridist.statistics.distributions import cdf, logpdf, ppf, sf
from veridist.statistics.log_density import (
    LogDensityErrorCode,
    LogDensityFailure,
    LogDensitySuccess,
    evaluate_log_density,
)

REAL_SCALARS: list[Any] = [
    np.float64(0.75),
    np.float32(0.75),
    np.float16(0.75),
    np.longdouble(0.75),
]
INTEGER_SCALARS: list[Any] = [np.int64(3), np.int32(3), np.uint16(3), np.uint64(3), np.int8(3)]
BOOLEANS: list[Any] = [True, False, np.True_, np.False_]


class ScalarOperationAcceptanceTests(unittest.TestCase):
    def test_numpy_scalars_give_the_same_float_as_python_numbers(self) -> None:
        for scalars in (REAL_SCALARS, INTEGER_SCALARS):
            for value in scalars:
                number = float(value)
                for operation in (logpdf, cdf, sf):
                    with self.subTest(operation=operation.__name__, value=repr(value)):
                        got = operation("gamma", value, shape=np.float32(2.0), scale=np.int64(2))
                        self.assertIs(type(got), float)
                        self.assertEqual(got, operation("gamma", number, shape=2.0, scale=2.0))
        for value in REAL_SCALARS:
            with self.subTest(operation="ppf", value=repr(value)):
                got = ppf("normal", value, mu=np.float32(0.0), sigma=np.int8(1))
                self.assertIs(type(got), float)
                self.assertEqual(got, ppf("normal", float(value), mu=0.0, sigma=1.0))

    def test_numpy_parameters_are_accepted_for_every_family(self) -> None:
        samples = {
            FamilyId.NORMAL: {"mu": np.float32(1.0), "sigma": np.float64(2.0)},
            FamilyId.GAMMA: {"shape": np.int64(2), "scale": np.float32(0.5)},
            FamilyId.WEIBULL_MIN: {"shape": np.float32(1.5), "scale": np.int32(2)},
            FamilyId.LOGNORMAL: {"mu_log": np.float32(0.5), "sigma_log": np.float16(0.5)},
            FamilyId.GUMBEL_RIGHT: {"location": np.int64(1), "scale": np.float32(2.0)},
            FamilyId.EXPONENTIAL: {"rate": np.float32(0.5)},
        }
        for family, parameters in samples.items():
            plain = {name: float(value) for name, value in parameters.items()}
            with self.subTest(family=family.value):
                self.assertEqual(
                    logpdf(family, np.float64(1.5), **parameters), logpdf(family, 1.5, **plain)
                )

    def test_booleans_are_rejected_wherever_a_number_is_expected(self) -> None:
        for bad in BOOLEANS:
            with self.subTest(bad=repr(bad)):
                with self.assertRaises(TypeError):
                    logpdf("normal", bad, mu=0.0, sigma=1.0)
                with self.assertRaises(TypeError):
                    cdf("normal", 0.0, mu=bad, sigma=1.0)
                with self.assertRaises(ValueError):
                    ppf("normal", bad, mu=0.0, sigma=1.0)

    def test_log_density_evaluation_accepts_numpy_observations_and_parameters(self) -> None:
        for value in REAL_SCALARS + INTEGER_SCALARS:
            with self.subTest(value=repr(value)):
                result = evaluate_log_density(
                    FamilyId.WEIBULL_MIN, value, shape=np.float32(1.5), scale=np.int64(2)
                )
                self.assertIsInstance(result, LogDensitySuccess)
                expected = evaluate_log_density(
                    FamilyId.WEIBULL_MIN, float(value), shape=1.5, scale=2.0
                )
                self.assertEqual(result, expected)

    def test_log_density_evaluation_still_flags_booleans_and_non_finite_numpy_values(self) -> None:
        for bad in (*BOOLEANS, np.float64(np.nan), np.float32(np.inf), "1.0", None):
            with self.subTest(bad=repr(bad)):
                result = evaluate_log_density(FamilyId.EXPONENTIAL, bad, rate=1.0)
                self.assertIsInstance(result, LogDensityFailure)
                self.assertIs(result.code, LogDensityErrorCode.NONFINITE_OBSERVATION)
        for bad in (np.True_, True):
            with self.assertRaises(TypeError):
                evaluate_log_density(FamilyId.EXPONENTIAL, 1.0, rate=bad)


class InferenceAcceptanceTests(unittest.TestCase):
    def test_information_criteria_accepts_numpy_scalars(self) -> None:
        expected = information_criteria(log_likelihood=-12.5, sample_size=40, free_parameters=2)
        for log_likelihood in (np.float64(-12.5), np.float32(-12.5), np.longdouble(-12.5)):
            for count in INTEGER_SCALARS[:3]:
                with self.subTest(log_likelihood=repr(log_likelihood), count=repr(count)):
                    got = information_criteria(
                        log_likelihood=log_likelihood,
                        sample_size=np.int64(40),
                        free_parameters=np.int8(2),
                    )
                    self.assertEqual(got, expected)
                    self.assertIs(type(got.aic), float)
                    self.assertIs(type(got.bic), float)
                    self.assertIs(type(got.free_parameters), int)
                    information_criteria(
                        log_likelihood=-1.0, sample_size=count, free_parameters=count
                    )

    def test_information_criteria_still_rejects_booleans_integers_and_non_finite_values(
        self,
    ) -> None:
        for bad in (*BOOLEANS, np.int64(-1), np.float64(np.nan), np.float32(-np.inf), 3, "x"):
            with self.subTest(log_likelihood=repr(bad)), self.assertRaises(ValueError):
                information_criteria(log_likelihood=bad, sample_size=5, free_parameters=1)
        for bad in (*BOOLEANS, np.float64(3.0), 0, np.int64(0), -1, "3"):
            with self.subTest(sample_size=repr(bad)), self.assertRaises(ValueError):
                information_criteria(log_likelihood=-1.0, sample_size=bad, free_parameters=1)
        for bad in (*BOOLEANS, np.float64(1.0), -1, np.int64(-1)):
            with self.subTest(free_parameters=repr(bad)), self.assertRaises(ValueError):
                information_criteria(log_likelihood=-1.0, sample_size=5, free_parameters=bad)

    def test_compare_models_accepts_numpy_floats(self) -> None:
        result = compare_models(
            candidates=(
                {"family": "normal", "aic": np.float64(12.0), "p_value": np.float32(0.2)},
                {"family": "gamma", "aic": np.float32(8.0), "p_value": np.float64(0.1)},
                {"family": "weibull", "aic": 1.0, "p_value": np.float64(0.01)},
            ),
            adequacy_threshold=np.float32(0.05),
        )
        self.assertEqual((result.code, result.selected_family), (SelectionCode.SELECTED, "gamma"))
        none = compare_models(
            candidates=({"family": "normal", "aic": np.float64(1.0), "p_value": np.float64(0.01)},),
            adequacy_threshold=np.float64(0.05),
        )
        self.assertEqual(none.code, SelectionCode.NONE_ADEQUATE)

    def test_compare_models_still_rejects_booleans_and_non_finite_numpy_floats(self) -> None:
        good = {"family": "normal", "aic": 1.0, "p_value": 0.5}
        for key in ("aic", "p_value"):
            for bad in (*BOOLEANS, "1.0", None):
                with self.subTest(key=key, bad=repr(bad)), self.assertRaises(TypeError):
                    compare_models(candidates=({**good, key: bad},), adequacy_threshold=0.05)
            for bad in (np.float64(np.nan), np.float32(np.inf)):
                with self.subTest(key=key, bad=repr(bad)), self.assertRaises(ValueError):
                    compare_models(candidates=({**good, key: bad},), adequacy_threshold=0.05)
        for bad in (*BOOLEANS, np.float64(1.5), np.float32(-0.1), np.float64(np.nan), 1):
            with self.subTest(threshold=repr(bad)), self.assertRaises(ValueError):
                compare_models(candidates=(good,), adequacy_threshold=bad)

    def test_summarize_calibration_accepts_numpy_scalars(self) -> None:
        expected = summarize_calibration(rejections=7, replicates=100, nominal_alpha=0.05)
        got = summarize_calibration(
            rejections=np.int64(7), replicates=np.uint16(100), nominal_alpha=np.float32(0.05)
        )
        self.assertEqual(got.rejection_rate, expected.rejection_rate)
        self.assertAlmostEqual(got.standard_error, expected.standard_error, places=7)
        self.assertIs(type(got.rejection_rate), float)
        self.assertEqual(
            summarize_calibration(
                rejections=np.int8(3), replicates=np.int64(10), nominal_alpha=np.float64(0.5)
            ),
            summarize_calibration(rejections=3, replicates=10, nominal_alpha=0.5),
        )

    def test_summarize_calibration_still_rejects_booleans_and_floats_as_counts(self) -> None:
        for bad in (*BOOLEANS, np.float64(1.0), 1.0, -1, np.int64(-1)):
            with self.subTest(rejections=repr(bad)), self.assertRaises(ValueError):
                summarize_calibration(rejections=bad, replicates=10, nominal_alpha=0.05)
        for bad in (*BOOLEANS, np.float64(10.0), 0, np.int64(0)):
            with self.subTest(replicates=repr(bad)), self.assertRaises(ValueError):
                summarize_calibration(rejections=0, replicates=bad, nominal_alpha=0.05)
        for bad in (*BOOLEANS, 0.5 + 0j, 1, np.int64(0), np.float64(0.0), np.float32(1.0), np.nan):
            with self.subTest(alpha=repr(bad)), self.assertRaises(ValueError):
                summarize_calibration(rejections=1, replicates=10, nominal_alpha=bad)
        with self.assertRaises(ValueError):
            summarize_calibration(
                rejections=np.int64(11), replicates=np.int64(10), nominal_alpha=0.5
            )

    def test_refit_monte_carlo_gof_accepts_numpy_replicates_and_observations(self) -> None:
        common: dict[str, Any] = {
            "family": "exponential",
            "statistics": frozenset({GofStatistic.KS}),
        }
        plain = refit_monte_carlo_gof(
            observations=(0.2, 0.4, 0.8, 1.6),
            replicates=20,
            rng=np.random.default_rng(5),
            **common,
        )
        numpy_valued = refit_monte_carlo_gof(
            observations=np.array([0.2, 0.4, 0.8, 1.6]),
            replicates=np.int64(20),
            rng=np.random.default_rng(5),
            **common,
        )
        self.assertEqual(numpy_valued.requested_replicates, 20)
        self.assertIs(type(numpy_valued.requested_replicates), int)
        self.assertEqual(numpy_valued.p_values, plain.p_values)
        for bad in (*BOOLEANS, np.float64(20.0), np.int64(0)):
            with self.subTest(replicates=repr(bad)), self.assertRaises(ValueError):
                refit_monte_carlo_gof(
                    observations=(0.5, 1.0), replicates=bad, rng=np.random.default_rng(1), **common
                )


class FitOptionAcceptanceTests(unittest.TestCase):
    observations = (
        ExactLifetime(1.0),
        ExactLifetime(2.5),
        ExactLifetime(4.0),
        RightCensoredLifetime(3.0),
    )

    def test_fixed_shape_accepts_numpy_real_scalars(self) -> None:
        reference = fit_weibull(self.observations, fixed_shape=2.0)
        for value in (np.float64(2.0), np.float32(2.0), np.int64(2), np.uint8(2)):
            with self.subTest(fixed_shape=repr(value)):
                result = fit_weibull(self.observations, fixed_shape=value)
                self.assertEqual(result.shape, 2.0)  # type: ignore[union-attr]
                self.assertEqual(result, reference)

    def test_fixed_shape_still_rejects_booleans_and_non_positive_values(self) -> None:
        for bad in BOOLEANS:
            with self.subTest(fixed_shape=repr(bad)), self.assertRaises(TypeError):
                fit_weibull(self.observations, fixed_shape=bad)
        for bad in (np.float64(0.0), np.int64(-1), np.float32(np.nan), np.float64(np.inf)):
            with self.subTest(fixed_shape=repr(bad)):
                # As for a Python float, an unusable value is a typed failure, not a raise.
                result = fit_weibull(self.observations, fixed_shape=bad)
                self.assertIsInstance(result, WeibullFitFailure)

    def test_frequency_weights_accept_numpy_integers(self) -> None:
        reference = fit_weibull(self.observations, frequency_weights=(1, 2, 1, 3))
        for dtype in (np.int64, np.int8, np.uint16):
            with self.subTest(dtype=dtype.__name__):
                weights = np.array([1, 2, 1, 3], dtype=dtype)
                self.assertEqual(
                    fit_weibull(self.observations, frequency_weights=weights), reference
                )
        self.assertEqual(
            fit_weibull(self.observations, frequency_weights=[np.int64(1), 2, np.uint8(1), 3]),
            reference,
        )

    def test_frequency_weights_still_reject_booleans_floats_and_negatives(self) -> None:
        for bad in (*BOOLEANS, 1.0, np.float64(1.0), -1, np.int64(-1)):
            with self.subTest(weight=repr(bad)), self.assertRaises(TypeError):
                fit_weibull(self.observations, frequency_weights=(1, 1, 1, bad))


class DomainValueAcceptanceTests(unittest.TestCase):
    def test_observation_types_accept_numpy_real_scalars(self) -> None:
        for value in (np.float64(2.5), np.float32(2.5), np.float16(2.5), np.longdouble(2.5)):
            with self.subTest(value=repr(value)):
                self.assertEqual(ExactLifetime(value).time, 2.5)
                self.assertEqual(RightCensoredLifetime(value).time, 2.5)
                self.assertEqual(ExactValue(value).value, 2.5)
                self.assertEqual(RightCensoredValue(value).value, 2.5)
                self.assertIs(type(ExactLifetime(value).time), float)
        for value in INTEGER_SCALARS:
            with self.subTest(value=repr(value)):
                self.assertEqual(ExactLifetime(value).time, 3.0)
                self.assertEqual(RightCensoredValue(value).value, 3.0)
        self.assertEqual(ExactValue(np.float64(-2.5)).value, -2.5)

    def test_observation_types_reject_numpy_booleans_and_non_finite_numpy_values(self) -> None:
        for kind in (ExactLifetime, RightCensoredLifetime, ExactValue, RightCensoredValue):
            for bad in BOOLEANS:
                with self.subTest(kind=kind.__name__, bad=repr(bad)), self.assertRaises(TypeError):
                    kind(bad)
            for bad in (np.float64(np.nan), np.float32(np.inf)):
                with self.subTest(kind=kind.__name__, bad=repr(bad)), self.assertRaises(ValueError):
                    kind(bad)
        for kind in (ExactLifetime, RightCensoredLifetime):
            for bad in (np.float64(-1.0), np.int64(-1)):
                with self.subTest(kind=kind.__name__, bad=repr(bad)), self.assertRaises(ValueError):
                    kind(bad)


if __name__ == "__main__":
    unittest.main()
