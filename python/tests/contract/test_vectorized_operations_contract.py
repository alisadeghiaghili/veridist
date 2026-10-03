"""Contract of ``logpdf``/``cdf``/``sf``/``ppf`` and ``sample`` for scalar and array operands."""

from __future__ import annotations

import unittest
import warnings
from typing import Any

import numpy as np

from veridist.families.registry import FamilyId
from veridist.statistics.distributions import cdf, logpdf, ppf, sample, sf

NORMAL = {"mu": 0.5, "sigma": 2.0}
WEIBULL = {"shape": 1.5, "scale": 3.0}
GUMBEL = {"location": 1.0, "scale": 2.0}
EXPONENTIAL = {"rate": 0.5}


class ScalarAndArrayResultTypeTests(unittest.TestCase):
    def test_scalar_like_operands_return_a_python_float(self) -> None:
        scalars: list[Any] = [
            2.0,
            2,
            np.float64(2.0),
            np.float32(2.0),
            np.int64(2),
            np.uint8(2),
            np.array(2.0),
            np.array(2),
        ]
        for operation in (logpdf, cdf, sf):
            reference = operation("weibull_min", 2.0, **WEIBULL)
            for value in scalars:
                with self.subTest(operation=operation.__name__, value=repr(value)):
                    result = operation("weibull_min", value, **WEIBULL)
                    self.assertIs(type(result), float)
                    self.assertEqual(result, reference)
        for value in (np.float64(0.25), np.float32(0.25), np.array(0.25)):
            self.assertIs(type(ppf("exponential", value, **EXPONENTIAL)), float)
        self.assertIs(type(logpdf("normal", 1.0, mu=np.float32(0.0), sigma=np.array(1.0))), float)

    def test_any_array_operand_gives_a_float64_array_of_the_broadcast_shape(self) -> None:
        column = np.array([[0.5], [1.5], [2.5]])
        row = np.array([1.0, 2.0, 3.0, 4.0])
        result = cdf("weibull_min", column, shape=1.5, scale=row)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.dtype, np.float64)
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(sf("gumbel_right", 0.5, location=row, scale=2.0).shape, (4,))
        self.assertEqual(ppf("exponential", [0.1, 0.2], rate=1.0).shape, (2,))
        self.assertEqual(logpdf("normal", 1.0, mu=[[0.0], [1.0]], sigma=1.0).shape, (2, 1))

    def test_array_likes_integer_and_narrow_float_arrays_are_accepted(self) -> None:
        reference = cdf("exponential", np.array([1.0, 2.0, 3.0]), rate=0.5)
        for value in ([1, 2, 3], (1, 2, 3), np.arange(1, 4), np.arange(1, 4, dtype=np.uint8)):
            with self.subTest(value=repr(value)):
                np.testing.assert_array_equal(cdf("exponential", value, rate=0.5), reference)
        narrow = np.array([1.0, 2.0], dtype=np.float32)
        self.assertEqual(cdf("exponential", narrow, rate=0.5).dtype, np.float64)

    def test_empty_operands_give_empty_float64_arrays(self) -> None:
        for family, parameters in (
            ("exponential", EXPONENTIAL),
            ("normal", NORMAL),
            ("gamma", {"shape": 2.0, "scale": 1.0}),
        ):
            for operation in (logpdf, cdf, sf):
                result = operation(family, np.empty((0,)), **parameters)
                self.assertEqual((result.shape, result.dtype), ((0,), np.float64))
            self.assertEqual(ppf(family, np.empty((3, 0)), **parameters).shape, (3, 0))
        self.assertEqual(cdf("exponential", 1.0, rate=np.empty((0, 2))).shape, (0, 2))

    def test_family_may_be_named_by_string_alias_or_enum(self) -> None:
        x = np.array([0.0, 1.0])
        expected = cdf(FamilyId.NORMAL, x, **NORMAL)
        np.testing.assert_array_equal(cdf("normal", x, **NORMAL), expected)
        np.testing.assert_array_equal(cdf("gaussian", x, **NORMAL), expected)

    def test_results_are_fresh_writable_arrays_and_inputs_are_not_modified(self) -> None:
        x = np.array([0.5, 1.5, 2.5])
        x.setflags(write=False)
        rate = np.array([1.0, 2.0, 3.0])
        before = (x.copy(), rate.copy())
        for operation in (logpdf, cdf, sf):
            result = operation("exponential", x, rate=rate)
            self.assertTrue(result.flags.writeable)
            result[0] = -123.0
        np.testing.assert_array_equal(x, before[0])
        np.testing.assert_array_equal(rate, before[1])


class ArrayInputRejectionTests(unittest.TestCase):
    def test_boolean_and_non_numeric_operands_are_type_errors(self) -> None:
        bad_values: list[Any] = [
            True,
            np.True_,
            np.array([True, False]),
            "1.0",
            np.array(["a", "b"]),
            None,
            [None],
            1 + 2j,
            np.array([1 + 0j]),
            {1.0},
            [object()],
        ]
        for operation in (logpdf, cdf, sf):
            for bad in bad_values:
                with self.subTest(operation=operation.__name__, point=repr(bad)):
                    with self.assertRaises(TypeError):
                        operation("normal", bad, **NORMAL)
                with self.subTest(operation=operation.__name__, parameter=repr(bad)):
                    with self.assertRaises(TypeError):
                        operation("normal", 0.0, mu=bad, sigma=1.0)

    def test_probabilities_that_are_not_real_numbers_are_value_errors(self) -> None:
        for bad in (True, np.True_, np.array([True]), "0.5", None, [0.5, "x"], 0.5 + 0j):
            with self.subTest(q=repr(bad)), self.assertRaises(ValueError):
                ppf("normal", bad, **NORMAL)

    def test_non_finite_points_name_the_first_bad_flat_index(self) -> None:
        for operation in (logpdf, cdf, sf):
            for bad in (np.nan, np.inf, -np.inf):
                with self.subTest(operation=operation.__name__, bad=bad):
                    with self.assertRaisesRegex(ValueError, r"^x must be finite .*flat index 3\)$"):
                        operation("exponential", [[0.0, 1.0], [2.0, bad]], rate=1.0)
        with self.assertRaisesRegex(ValueError, "flat index 0"):
            cdf("normal", [np.nan, np.nan], **NORMAL)

    def test_invalid_parameter_elements_name_the_parameter_and_first_flat_index(self) -> None:
        index = r".*index "
        cases: list[tuple[str, dict[str, Any], str]] = [
            (
                "normal",
                {"mu": 0.0, "sigma": [1.0, 2.0, -1.0, 0.0]},
                r"^sigma must be positive" + index + "2",
            ),
            ("normal", {"mu": [0.0, np.nan], "sigma": 1.0}, r"^mu must be finite" + index + "1"),
            (
                "normal",
                {"mu": [[0.0, 1.0], [np.inf, 2.0]], "sigma": 1.0},
                r"^mu must be finite" + index + "2",
            ),
            ("normal", {"mu": 0.0, "sigma": [1.0, np.inf]}, r"^sigma must be finite" + index + "1"),
            (
                "weibull_min",
                {"shape": [1.0, 0.0], "scale": 1.0},
                r"^shape must be positive" + index + "1",
            ),
            ("gamma", {"shape": 1.0, "scale": np.nan}, r"^scale must be finite"),
            ("exponential", {"rate": [0.0]}, r"^rate must be positive" + index + "0"),
            (
                "gumbel_right",
                {"location": [0.0, 0.0, np.nan], "scale": 1.0},
                r"^location must be finite" + index + "2",
            ),
        ]
        for family, parameters, pattern in cases:
            for operation in (logpdf, cdf, sf):
                with self.subTest(family=family, operation=operation.__name__):
                    with self.assertRaisesRegex(ValueError, pattern):
                        operation(family, [0.5, 1.0, 1.5, 2.0], **parameters)
            with self.subTest(family=family, operation="ppf"):
                with self.assertRaisesRegex(ValueError, pattern):
                    ppf(family, [0.1, 0.5, 0.9, 0.95], **parameters)

    def test_scalar_parameter_errors_are_unchanged_when_the_point_is_an_array(self) -> None:
        with self.assertRaisesRegex(ValueError, "^sigma must be positive$"):
            cdf("normal", [0.0, 1.0], mu=0.0, sigma=0.0)
        with self.assertRaisesRegex(ValueError, "^mu must be finite$"):
            cdf("normal", [0.0, 1.0], mu=np.nan, sigma=1.0)
        with self.assertRaisesRegex(ValueError, "^mu must be finite$"):
            cdf("normal", [0.0, 1.0], mu=10**400, sigma=1.0)

    def test_probabilities_outside_the_open_unit_interval_name_the_first_flat_index(self) -> None:
        for bad in (0.0, 1.0, -0.5, 1.5, np.nan, np.inf):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, r"^q must be strictly .*flat index 2\)$"):
                    ppf("weibull_min", [[0.2, 0.4], [bad, 0.6]], **WEIBULL)
        with self.assertRaisesRegex(ValueError, "flat index 0"):
            ppf("normal", np.array([1.0, 0.5]), **NORMAL)

    def test_operands_that_cannot_be_broadcast_together_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cdf("normal", np.zeros(3), mu=np.zeros(4), sigma=1.0)
        with self.assertRaises(ValueError):
            ppf("exponential", np.full((2, 3), 0.5), rate=np.ones((3, 2)))

    def test_parameter_names_are_still_checked_for_array_calls(self) -> None:
        with self.assertRaises(TypeError):
            cdf("exponential", [1.0, 2.0], scale=1.0)
        with self.assertRaises(TypeError):
            cdf("exponential", [1.0, 2.0], rate=[1.0, 2.0], extra=1.0)
        with self.assertRaises(TypeError):
            logpdf("normal", [1.0, 2.0], mu=[0.0, 1.0])

    def test_ragged_array_likes_are_rejected(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            cdf("exponential", [[1.0], [1.0, 2.0]], rate=1.0)


class UnrepresentableValueTests(unittest.TestCase):
    def test_logpdf_overflow_inside_the_support_raises_with_the_first_flat_index(self) -> None:
        cases = (
            ("gumbel_right", np.array([0.0, -1e5, -2e5]), GUMBEL),  # native kernel
            ("exponential", np.array([1.0, 1e308]), {"rate": 1e10}),  # native kernel
            ("normal", np.array([0.0, 1e200]), NORMAL),  # wrapped scalar kernel
        )
        for family, x, parameters in cases:
            with self.subTest(family=family):
                with self.assertRaisesRegex(ArithmeticError, r"not representable.*flat index 1"):
                    logpdf(family, x, **parameters)
                with self.assertRaises(ArithmeticError):
                    logpdf(family, float(x[1]), **parameters)

    def test_points_outside_the_support_are_minus_infinity_not_an_error(self) -> None:
        result = logpdf("exponential", np.array([-1e308, 1.0]), rate=1e10)
        self.assertEqual(result[0], -np.inf)
        self.assertTrue(np.isfinite(result[1]))

    def test_cdf_and_sf_saturate_where_the_scalar_intermediate_overflows(self) -> None:
        # exp(1000) overflows binary64; the limits are 0 and 1 exactly.
        x = np.array([-1000.0, 0.0])
        np.testing.assert_array_equal(cdf("gumbel_right", x, location=0.0, scale=1.0)[0], 0.0)
        np.testing.assert_array_equal(sf("gumbel_right", x, location=0.0, scale=1.0)[0], 1.0)


class WrappedFamilyArrayTests(unittest.TestCase):
    def test_wrapped_families_reproduce_the_scalar_path_exactly(self) -> None:
        x = np.array([0.3, 1.1, 4.0])
        for family, parameters in (
            ("normal", NORMAL),
            ("lognormal", {"mu_log": 0.1, "sigma_log": 0.7}),
            ("gamma", {"shape": 2.5, "scale": 1.5}),
        ):
            for operation in (logpdf, cdf, sf):
                got = operation(family, x, **parameters)
                want = [operation(family, float(v), **parameters) for v in x]
                self.assertEqual(got.tolist(), want)
            q = np.array([0.1, 0.5, 0.97])
            self.assertEqual(
                ppf(family, q, **parameters).tolist(),
                [ppf(family, float(v), **parameters) for v in q],
            )

    def test_wrapped_logpdf_gives_minus_infinity_outside_the_support(self) -> None:
        result = logpdf("gamma", np.array([-1.0, 0.0, 1.0]), shape=2.0, scale=1.0)
        self.assertEqual(result[:2].tolist(), [-np.inf, -np.inf])
        self.assertTrue(np.isfinite(result[2]))


class DeprecatedMappingFormWithArraysTests(unittest.TestCase):
    def test_the_mapping_form_accepts_arrays_and_still_warns(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            legacy = cdf("normal", x, {"mu": 0.0, "sigma": 1.0})
        self.assertTrue(any(issubclass(item.category, DeprecationWarning) for item in caught))
        np.testing.assert_array_equal(legacy, cdf("normal", x, mu=0.0, sigma=1.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            np.testing.assert_array_equal(
                ppf("exponential", [0.2, 0.8], {"rate": [1.0, 2.0]}),
                ppf("exponential", [0.2, 0.8], rate=[1.0, 2.0]),
            )


class OverflowSaturationTests(unittest.TestCase):
    """Intermediate overflow yields the limiting value on both evaluation paths."""

    CASES = (
        (cdf, FamilyId.GUMBEL_RIGHT, -800.0, {"location": 0.0, "scale": 1.0}, 0.0),
        (sf, FamilyId.GUMBEL_RIGHT, -800.0, {"location": 0.0, "scale": 1.0}, 1.0),
        (cdf, FamilyId.WEIBULL_MIN, 1e300, {"shape": 5.0, "scale": 1.0}, 1.0),
        (sf, FamilyId.WEIBULL_MIN, 1e300, {"shape": 5.0, "scale": 1.0}, 0.0),
        (ppf, FamilyId.WEIBULL_MIN, 0.999, {"shape": 0.001, "scale": 1.0}, float("inf")),
        (ppf, FamilyId.LOGNORMAL, 0.999, {"mu_log": 700.0, "sigma_log": 10.0}, float("inf")),
    )

    def test_scalar_path_returns_the_limit_instead_of_raising(self) -> None:
        for operation, family, point, parameters, expected in self.CASES:
            with self.subTest(operation=operation.__name__, family=family):
                self.assertEqual(operation(family, point, **parameters), expected)

    def test_array_path_agrees_with_the_scalar_path(self) -> None:
        for operation, family, point, parameters, expected in self.CASES:
            with self.subTest(operation=operation.__name__, family=family):
                result = operation(family, np.array([point]), **parameters)
                self.assertEqual(result.tolist(), [expected])


class SampleNumpyScalarTests(unittest.TestCase):
    def test_sample_accepts_numpy_integer_size_and_numpy_scalar_parameters(self) -> None:
        reference = sample("weibull_min", 5, rng=np.random.default_rng(3), **WEIBULL)
        for size in (np.int64(5), np.uint8(5), np.int32(5)):
            with self.subTest(size=repr(size)):
                draws = sample(
                    "weibull_min",
                    size,
                    rng=np.random.default_rng(3),
                    shape=np.float32(1.5),
                    scale=np.float64(3.0),
                )
                self.assertEqual(draws.shape, (5,))
                np.testing.assert_array_equal(draws, reference)
        zero_dimensional = sample(
            "normal", 3, rng=np.random.default_rng(1), mu=np.array(0.0), sigma=np.int64(2)
        )
        self.assertEqual(zero_dimensional.shape, (3,))

    def test_sample_rejects_boolean_size_and_array_parameters(self) -> None:
        generator = np.random.default_rng(1)
        for bad_size in (True, np.True_, 2.0, np.float64(2.0), np.int64(-1), "3"):
            with self.subTest(size=repr(bad_size)), self.assertRaises(ValueError):
                sample("exponential", bad_size, rng=generator, rate=1.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            sample("exponential", 3, rng=generator, rate=np.True_)
        with self.assertRaises(TypeError):
            sample("exponential", 3, rng=generator, rate=[1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
