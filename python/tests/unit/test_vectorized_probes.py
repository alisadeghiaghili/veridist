"""Boundary probes for the array machinery and the numeric acceptance helpers."""

from __future__ import annotations

import unittest
from fractions import Fraction
from math import inf, isinf

import numpy as np

from veridist.domain._numeric import is_float, is_integer, is_real
from veridist.families.registry import FAMILY_REGISTRY, FamilyId, Operation, Support
from veridist.statistics import _vectorized as vec
from veridist.statistics import distributions


class NumericAcceptanceProbeTests(unittest.TestCase):
    def test_real_integer_and_float_classification(self) -> None:
        real = [1, 1.5, np.float32(1), np.float64(1), np.int8(1), np.uint64(1), np.float16(1)]
        for value in real:
            with self.subTest(value=repr(value)):
                self.assertTrue(is_real(value))
        self.assertEqual(
            [is_integer(v) for v in real], [True, False, False, False, True, True, False]
        )
        self.assertEqual([is_float(v) for v in real], [False, True, True, True, False, False, True])
        for value in (True, False, np.True_, np.False_, "1", None, 1j, [1], np.array(1.0)):
            with self.subTest(value=repr(value)):
                self.assertFalse(is_real(value))
                self.assertFalse(is_integer(value))
                self.assertFalse(is_float(value))


class CoercionProbeTests(unittest.TestCase):
    def test_scalars_become_floats_and_arrays_become_float64(self) -> None:
        for value in (1, np.int8(1), np.float32(1.0), np.array(1), np.array(1.0)):
            result = vec.coerce(value, "x")
            self.assertIs(type(result), float)
            self.assertEqual(result, 1.0)
        for value in ([1, 2], np.array([1, 2], dtype=np.uint8), np.array([1.0, 2.0], np.float32)):
            result = vec.coerce(value, "x")
            self.assertEqual((result.dtype, result.shape), (np.float64, (2,)))

    def test_an_integer_too_large_for_binary64_becomes_infinity_for_the_finite_check(self) -> None:
        self.assertTrue(isinf(vec.coerce(10**400, "x")))
        self.assertTrue(isinf(vec.coerce(-(10**400), "x")))

    def test_non_numeric_kinds_are_type_errors(self) -> None:
        for bad in (True, np.True_, "x", np.array(["x"]), np.array([True]), 1j, None, {1}):
            with self.subTest(bad=repr(bad)), self.assertRaises(TypeError):
                vec.coerce(bad, "x")

    def test_scalar_call_detection(self) -> None:
        self.assertTrue(vec.is_scalar_call(1.0, {"a": 2.0}))
        self.assertTrue(vec.is_scalar_call(1.0, {}))
        self.assertFalse(vec.is_scalar_call(np.zeros(2), {"a": 2.0}))
        self.assertFalse(vec.is_scalar_call(1.0, {"a": np.zeros(2)}))
        self.assertFalse(vec.is_scalar_call(1.0, {"a": 2.0, "b": np.zeros(1)}))


class SupportMaskProbeTests(unittest.TestCase):
    def test_masks_follow_the_family_spec_contains_semantics(self) -> None:
        points = np.array([-1.0, -0.0, 0.0, 1e-320, 1.0])
        for family, spec in FAMILY_REGISTRY.families.items():
            with self.subTest(family=family.value):
                mask = vec.inside_support(spec.support, points)
                self.assertEqual(mask.tolist(), [spec.contains(float(x)) for x in points])
        self.assertEqual(vec.inside_support(Support.REAL_LINE, points).tolist(), [True] * 5)
        self.assertEqual(
            vec.inside_support(Support.NON_NEGATIVE, points).tolist(),
            [False, True, True, True, True],
        )
        self.assertEqual(
            vec.inside_support(Support.POSITIVE, points).tolist(), [False, False, False, True, True]
        )


class KernelTableProbeTests(unittest.TestCase):
    def test_exactly_three_families_have_native_kernels_for_every_operation(self) -> None:
        native = {FamilyId.EXPONENTIAL, FamilyId.WEIBULL_MIN, FamilyId.GUMBEL_RIGHT}
        self.assertEqual(set(vec.NATIVE_FAMILIES), native)
        for table in (vec.NATIVE_LOGPDF, vec.NATIVE_CDF, vec.NATIVE_SF, vec.NATIVE_PPF):
            self.assertEqual(set(table), native)

    def test_array_tables_cover_every_family_and_operation(self) -> None:
        tables = distributions._ARRAY
        self.assertEqual(
            set(tables), {Operation.LOGPDF, Operation.CDF, Operation.SF, Operation.PPF}
        )
        for operation, table in tables.items():
            with self.subTest(operation=operation.value):
                self.assertEqual(set(table), set(FamilyId))
                self.assertTrue(all(callable(kernel) for kernel in table.values()))
        distributions._verify_operation_tables(FAMILY_REGISTRY.families, tables)

    def test_native_and_wrapped_kernels_agree_on_the_calling_convention(self) -> None:
        points = np.array([0.5, 1.5, 2.5])
        for family in FamilyId:
            names = [parameter.name for parameter in FAMILY_REGISTRY.families[family].parameters]
            parameters = {name: np.full(3, 1.5) for name in names}
            for operation, table in distributions._ARRAY.items():
                with self.subTest(family=family.value, operation=operation.value):
                    probabilities = np.array([0.2, 0.5, 0.8])
                    argument = probabilities if operation is Operation.PPF else points
                    if family in vec.NATIVE_FAMILIES:
                        result = vec.evaluate_native(table[family], argument, parameters)
                    else:
                        result = table[family](argument, parameters)
                    self.assertEqual((result.dtype, result.shape), (np.float64, (3,)))

    def test_a_missing_array_kernel_is_detected(self) -> None:
        incomplete = {
            operation: {
                family: kernel for family, kernel in table.items() if family is not FamilyId.GAMMA
            }
            for operation, table in distributions._ARRAY.items()
        }
        with self.assertRaises(RuntimeError):
            distributions._verify_operation_tables(FAMILY_REGISTRY.families, incomplete)


class ErrorFreeTransformProbeTests(unittest.TestCase):
    def test_two_product_is_exact(self) -> None:
        rng = np.random.default_rng(11)
        a = rng.uniform(-1e3, 1e3, 2000) * np.exp(rng.uniform(-20, 20, 2000))
        b = rng.uniform(-1e3, 1e3, 2000) * np.exp(rng.uniform(-20, 20, 2000))
        product, error = vec._two_product(a, b)
        for x, y, p, e in zip(
            a.tolist(), b.tolist(), product.tolist(), error.tolist(), strict=True
        ):
            self.assertEqual(Fraction(x) * Fraction(y), Fraction(p) + Fraction(e))

    def test_scaled_difference_is_the_correctly_rounded_exact_quotient(self) -> None:
        rng = np.random.default_rng(12)
        size = 4000
        location = rng.uniform(-60.0, 60.0, size)
        scale = np.exp(rng.uniform(-8.0, 8.0, size))
        x = location + scale * rng.uniform(-30.0, 30.0, size)
        got = vec._scaled_difference(x, location, scale)
        wanted = np.array(
            [
                float((Fraction(a) - Fraction(b)) / Fraction(c))
                for a, b, c in zip(x.tolist(), location.tolist(), scale.tolist(), strict=True)
            ]
        )
        # Rounded once, not twice: at most one last-place unit away, and almost always equal.
        distance = np.abs(got - wanted) / np.spacing(np.abs(wanted))
        self.assertLessEqual(float(np.max(distance)), 1.0)
        self.assertLessEqual(int(np.count_nonzero(distance)), size // 500)
        plain = (x - location) / scale
        self.assertGreater(
            int(np.count_nonzero(plain != wanted)), 10 * int(np.count_nonzero(got != wanted))
        )

    def test_scaled_difference_falls_back_to_the_plain_quotient_where_a_split_overflows(
        self,
    ) -> None:
        x = np.array([1e308, 1e300, 0.0])
        location = np.array([-1e308, 0.0, 0.0])
        scale = np.array([1.0, 1e-5, 1.0])
        with np.errstate(all="ignore"):
            result = vec._scaled_difference(x, location, scale)
        self.assertEqual(result[0], inf)
        self.assertEqual(result[1], np.float64(1e300) / np.float64(1e-5))
        self.assertEqual(result[2], 0.0)


if __name__ == "__main__":
    unittest.main()
