"""Conformance of the unified scalar operations, for all six registry families."""

from __future__ import annotations

import math
import unittest
import warnings

import numpy as np

from veridist.families.registry import FAMILY_REGISTRY, FamilyId, FamilySpec, Operation, Support
from veridist.statistics import distributions
from veridist.statistics.distributions import cdf, logpdf, ppf, sample, sf
from veridist.statistics.log_density import (
    LogDensityErrorCode,
    LogDensityFailure,
    LogDensitySuccess,
    _exponential,
    evaluate_log_density,
)

PARAMETERS: dict[FamilyId, dict[str, float]] = {
    FamilyId.NORMAL: {"mu": 1.0, "sigma": 2.0},
    FamilyId.GAMMA: {"shape": 2.0, "scale": 3.0},
    FamilyId.WEIBULL_MIN: {"shape": 1.5, "scale": 4.0},
    FamilyId.LOGNORMAL: {"mu_log": 0.5, "sigma_log": 0.75},
    FamilyId.GUMBEL_RIGHT: {"location": 1.0, "scale": 2.0},
    FamilyId.EXPONENTIAL: {"rate": 2.0},
}
OPEN_SUPPORT = (FamilyId.GAMMA, FamilyId.WEIBULL_MIN, FamilyId.LOGNORMAL)
FIXED_LOCATION = tuple(
    family for family in FamilyId if FAMILY_REGISTRY.families[family].fixed_location == 0.0
)


class ReferencePointTests(unittest.TestCase):
    def test_reference_values_for_every_family(self) -> None:
        self.assertEqual(cdf(FamilyId.NORMAL, 0.0, mu=0.0, sigma=1.0), 0.5)
        self.assertEqual(sf("normal", 0.0, mu=0.0, sigma=1.0), 0.5)
        self.assertAlmostEqual(
            ppf("normal", 0.975, mu=0.0, sigma=1.0), 1.959963984540054, places=14
        )
        self.assertAlmostEqual(cdf("gamma", 6.0, shape=2.0, scale=3.0), 0.5939941503, places=9)
        self.assertAlmostEqual(
            cdf("weibull_min", 4.0, shape=1.5, scale=4.0), 0.6321205588, places=9
        )
        self.assertEqual(cdf("lognormal", math.exp(0.5), mu_log=0.5, sigma_log=0.75), 0.5)
        self.assertAlmostEqual(
            ppf("gumbel_right", 0.5, location=0.0, scale=2.0),
            -2.0 * math.log(math.log(2.0)),
            places=14,
        )
        self.assertAlmostEqual(cdf("exponential", 1.0, rate=2.0), 1.0 - math.exp(-2.0), places=15)
        self.assertAlmostEqual(sf("exponential", 1.0, rate=2.0), math.exp(-2.0), places=15)
        self.assertAlmostEqual(ppf("exponential", 0.5, rate=2.0), math.log(2.0) / 2.0, places=15)
        self.assertAlmostEqual(logpdf("exponential", 1.0, rate=2.0), math.log(2.0) - 2.0, places=15)

    def test_exponential_tail_uses_the_stable_survival_path(self) -> None:
        tail = sf(FamilyId.EXPONENTIAL, 700.0, rate=1.0)
        self.assertGreater(tail, 0.0)
        self.assertAlmostEqual(math.log(tail), -700.0, places=12)

    def test_ppf_inverts_cdf_for_every_family(self) -> None:
        for family, parameters in PARAMETERS.items():
            for probability in (0.05, 0.25, 0.5, 0.9):
                with self.subTest(family=family, q=probability):
                    quantile = ppf(family, probability, **parameters)
                    self.assertAlmostEqual(
                        cdf(family, quantile, **parameters), probability, places=10
                    )

    def test_every_operation_accepts_the_enum_the_string_and_the_alias(self) -> None:
        aliases = {
            FamilyId.NORMAL: "gaussian",
            FamilyId.WEIBULL_MIN: "weibull",
            FamilyId.GUMBEL_RIGHT: "gumbel",
        }
        for family, parameters in PARAMETERS.items():
            point = 0.8
            for spelling in (family, family.value, aliases.get(family, family.value)):
                with self.subTest(family=family, spelling=spelling):
                    self.assertEqual(
                        cdf(spelling, point, **parameters), cdf(family, point, **parameters)
                    )
                    self.assertEqual(
                        sf(spelling, point, **parameters), sf(family, point, **parameters)
                    )
                    self.assertEqual(
                        ppf(spelling, 0.4, **parameters), ppf(family, 0.4, **parameters)
                    )
                    self.assertEqual(
                        logpdf(spelling, point, **parameters), logpdf(family, point, **parameters)
                    )


class SupportConventionTests(unittest.TestCase):
    def test_the_registry_documents_the_support_of_every_family(self) -> None:
        for family in FamilyId:
            spec = FAMILY_REGISTRY.families[family]
            if family is FamilyId.EXPONENTIAL:
                expected = Support.NON_NEGATIVE
            elif family in FIXED_LOCATION:
                expected = Support.POSITIVE
            else:
                expected = Support.REAL_LINE
            with self.subTest(family=family):
                self.assertIs(spec.support, expected)
        self.assertIn("exponential", FamilySpec.__doc__ or "")
        self.assertIn("[0, inf)", FamilySpec.__doc__ or "")

    def test_exponential_is_a_fixed_location_family_with_closed_support(self) -> None:
        spec = FAMILY_REGISTRY.resolve("exponential")
        self.assertIs(spec.id, FamilyId.EXPONENTIAL)
        self.assertEqual(spec.fixed_location, 0.0)
        self.assertEqual([p.name for p in spec.parameters], ["rate"])
        self.assertEqual(spec.free_parameter_count, 1)
        self.assertEqual(spec.available_operations, frozenset(Operation))

    def test_logpdf_is_minus_infinity_outside_the_support_of_the_positive_families(
        self,
    ) -> None:
        for family in OPEN_SUPPORT:
            for x in (-3.0, -1e-300, 0.0):
                with self.subTest(family=family, x=x):
                    self.assertEqual(logpdf(family, x, **PARAMETERS[family]), -math.inf)
                    typed = evaluate_log_density(family, x, **PARAMETERS[family])
                    assert isinstance(typed, LogDensityFailure)
                    self.assertIs(typed.code, LogDensityErrorCode.SUPPORT_VIOLATION)

    def test_exponential_support_is_closed_at_zero(self) -> None:
        for rate in (0.25, 1.0, 2.0, 1e-9, 1e9):
            with self.subTest(rate=rate):
                self.assertEqual(logpdf(FamilyId.EXPONENTIAL, 0.0, rate=rate), math.log(rate))
                typed = evaluate_log_density(FamilyId.EXPONENTIAL, 0.0, rate=rate)
                assert isinstance(typed, LogDensitySuccess)
                self.assertEqual(typed.log_density, math.log(rate))
        spec = FAMILY_REGISTRY.families[FamilyId.EXPONENTIAL]
        self.assertTrue(spec.contains(0.0))
        self.assertTrue(spec.contains(5e-324))
        self.assertFalse(spec.contains(-5e-324))

    def test_exponential_negative_values_are_still_a_support_violation(self) -> None:
        for x in (-5e-324, -1e-300, -1.0, -1e300):
            with self.subTest(x=x):
                self.assertEqual(logpdf("exponential", x, rate=2.0), -math.inf)
                typed = evaluate_log_density(FamilyId.EXPONENTIAL, x, rate=2.0)
                assert isinstance(typed, LogDensityFailure)
                self.assertIs(typed.code, LogDensityErrorCode.SUPPORT_VIOLATION)

    def test_support_membership_is_declared_per_family(self) -> None:
        for family in OPEN_SUPPORT:
            spec = FAMILY_REGISTRY.families[family]
            self.assertFalse(spec.contains(0.0))
            self.assertTrue(spec.contains(1e-300))
            self.assertFalse(spec.contains(-1.0))
        for family in (FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT):
            spec = FAMILY_REGISTRY.families[family]
            self.assertTrue(spec.contains(-1e300) and spec.contains(0.0))

    def test_declared_support_is_validated(self) -> None:
        from veridist.families.registry import ParameterRole, ParameterSpec

        def build(**changes: object) -> FamilySpec:
            values: dict[str, object] = {
                "id": FamilyId.GAMMA,
                "aliases": (),
                "parameters": (ParameterSpec("scale", ParameterRole.POSITIVE),),
                "fixed_location": 0.0,
                "planned_operations": frozenset({Operation.LOGPDF}),
                "available_operations": frozenset(),
            }
            values.update(changes)
            return FamilySpec(**values)  # type: ignore[arg-type]

        self.assertIs(build().support, Support.POSITIVE)
        self.assertIs(build(declared_support=Support.NON_NEGATIVE).support, Support.NON_NEGATIVE)
        self.assertIs(build(declared_support=None, fixed_location=None).support, Support.REAL_LINE)
        for bad in ("positive", 1):
            with self.assertRaises(TypeError):
                build(declared_support=bad)
        with self.assertRaises(ValueError):
            build(declared_support=Support.REAL_LINE)
        with self.assertRaises(ValueError):
            build(declared_support=Support.POSITIVE, fixed_location=None)

    def test_logpdf_has_no_support_limit_on_the_real_line_families(self) -> None:
        for family in (FamilyId.NORMAL, FamilyId.GUMBEL_RIGHT):
            for x in (-40.0, 0.0, 12.0):
                with self.subTest(family=family, x=x):
                    self.assertTrue(math.isfinite(logpdf(family, x, **PARAMETERS[family])))

    def test_the_exponential_density_formula_is_log_rate_at_the_origin(self) -> None:
        # The kernel is the plain formula, and zero is inside the exponential support.
        self.assertEqual(_exponential(0.0, {"rate": 3.5}), math.log(3.5))
        self.assertEqual(_exponential(2.0, {"rate": 3.5}), math.log(3.5) - 7.0)

    def test_cumulative_operations_are_defined_on_the_whole_line(self) -> None:
        for family in FIXED_LOCATION:
            for x in (-3.0, 0.0):
                with self.subTest(family=family, x=x):
                    self.assertEqual(cdf(family, x, **PARAMETERS[family]), 0.0)
                    self.assertEqual(sf(family, x, **PARAMETERS[family]), 1.0)

    def test_logpdf_agrees_with_the_typed_evaluator_wherever_that_succeeds(self) -> None:
        for family, parameters in PARAMETERS.items():
            for x in (0.3, 1.0, 2.5, 11.0):
                with self.subTest(family=family, x=x):
                    typed = evaluate_log_density(family, x, **parameters)
                    assert isinstance(typed, LogDensitySuccess)
                    self.assertEqual(logpdf(family, x, **parameters), typed.log_density)

    def test_logpdf_raises_arithmetic_error_when_the_value_is_not_representable(self) -> None:
        with self.assertRaises(ArithmeticError):
            logpdf("weibull_min", 1e300, shape=10.0, scale=1.0)
        with self.assertRaises(ArithmeticError):
            logpdf(FamilyId.EXPONENTIAL, 1e300, rate=1e300)
        typed = evaluate_log_density(FamilyId.EXPONENTIAL, 1e300, rate=1e300)
        assert isinstance(typed, LogDensityFailure)
        self.assertIs(typed.code, LogDensityErrorCode.NUMERICAL_OVERFLOW)


class SamplingTests(unittest.TestCase):
    def test_sampling_matches_the_numpy_oracle_with_canonical_parameters(self) -> None:
        oracles = {
            FamilyId.NORMAL: lambda g: g.normal(1.0, 2.0, size=4),
            FamilyId.GAMMA: lambda g: g.gamma(2.0, 3.0, size=4),
            FamilyId.WEIBULL_MIN: lambda g: 4.0 * g.weibull(1.5, size=4),
            FamilyId.LOGNORMAL: lambda g: g.lognormal(0.5, 0.75, size=4),
            FamilyId.GUMBEL_RIGHT: lambda g: g.gumbel(1.0, 2.0, size=4),
            FamilyId.EXPONENTIAL: lambda g: g.exponential(0.5, size=4),
        }
        for family, oracle in oracles.items():
            with self.subTest(family=family):
                got = sample(family, 4, rng=np.random.default_rng(91), **PARAMETERS[family])
                self.assertEqual(got.tolist(), oracle(np.random.default_rng(91)).tolist())

    def test_sampling_uses_only_the_caller_generator(self) -> None:
        before = np.random.get_state()
        sample("exponential", 8, rng=np.random.default_rng(42), rate=1.5)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        self.assertTrue((before[1] == after[1]).all())

    def test_sampling_size_zero_gives_an_empty_array(self) -> None:
        self.assertEqual(
            sample("normal", 0, rng=np.random.default_rng(1), mu=0.0, sigma=1.0).shape, (0,)
        )


class MisuseTests(unittest.TestCase):
    def test_family_must_be_a_family_id_or_a_string(self) -> None:
        for bad in (None, 1, 2.5, b"normal", ("normal",), object()):
            with self.subTest(bad=bad):
                for call in (
                    lambda: cdf(bad, 0.0, mu=0.0, sigma=1.0),  # type: ignore[arg-type]
                    lambda: sf(bad, 0.0, mu=0.0, sigma=1.0),  # type: ignore[arg-type]
                    lambda: ppf(bad, 0.5, mu=0.0, sigma=1.0),  # type: ignore[arg-type]
                    lambda: logpdf(bad, 0.0, mu=0.0, sigma=1.0),  # type: ignore[arg-type]
                    lambda: sample(bad, 1, rng=np.random.default_rng(1), mu=0.0, sigma=1.0),  # type: ignore[arg-type]
                ):
                    with self.assertRaises(TypeError):
                        call()

    def test_unknown_family_names_are_value_errors(self) -> None:
        for name in ("", "Normal", "student_t", "weibull-min", "exp"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                cdf(name, 0.0, rate=1.0)

    def test_points_must_be_finite_built_in_reals(self) -> None:
        for call in (cdf, sf, logpdf):
            for bad in (True, "1", None, 1 + 0j, {1.0}):
                with self.subTest(call=call.__name__, bad=bad), self.assertRaises(TypeError):
                    call("exponential", bad, rate=1.0)
            for bad in (math.nan, math.inf, -math.inf, 10**400):
                with self.subTest(call=call.__name__, bad=bad), self.assertRaises(ValueError):
                    call("exponential", bad, rate=1.0)

    def test_probability_must_be_an_interior_float(self) -> None:
        for bad in (0.0, 1.0, -0.1, 1.5, math.nan, math.inf, 1, "0.5", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ppf("exponential", bad, rate=1.0)  # type: ignore[arg-type]

    def test_parameter_names_and_values_are_validated_by_the_registry(self) -> None:
        for call in (cdf, sf, logpdf):
            with self.assertRaises(TypeError):
                call("exponential", 1.0, scale=1.0)
            with self.assertRaises(TypeError):
                call("exponential", 1.0, rate=1.0, extra=2.0)
            with self.assertRaises(TypeError):
                call("exponential", 1.0)
            for bad in (0.0, -1.0, math.nan, math.inf):
                with self.assertRaises(ValueError):
                    call("exponential", 1.0, rate=bad)
            for bad in (True, "1", None):
                with self.assertRaises(TypeError):
                    call("exponential", 1.0, rate=bad)

    def test_sample_arguments_are_validated(self) -> None:
        generator = np.random.default_rng(1)
        for bad_size in (-1, True, 1.5, "3", None):
            with self.subTest(size=bad_size), self.assertRaises(ValueError):
                sample("exponential", bad_size, rng=generator, rate=1.0)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            sample("exponential", 1, rate=1.0)  # type: ignore[call-overload]
        for bad_rng in (None, object(), np.random.RandomState(1), 7):
            with self.subTest(rng=bad_rng), self.assertRaises(TypeError):
                sample("exponential", 1, rng=bad_rng, rate=1.0)

    def test_malformed_deprecated_calls_are_rejected(self) -> None:
        generator = np.random.default_rng(1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with self.assertRaises(TypeError):
                cdf("exponential", 1.0, [("rate", 1.0)])  # type: ignore[call-overload]
            with self.assertRaises(TypeError):
                cdf("exponential", 1.0, {"rate": 1.0}, {"rate": 1.0})  # type: ignore[call-overload]
            with self.assertRaises(TypeError):
                cdf("exponential", 1.0, {"rate": 1.0}, rate=1.0)  # type: ignore[call-overload]
            with self.assertRaises(TypeError):
                sample("exponential", 1, {"rate": 1.0}, generator, generator)  # type: ignore[call-overload]
            with self.assertRaises(TypeError):
                sample("exponential", 1, {"rate": 1.0}, generator, rng=generator)  # type: ignore[call-overload]
            with self.assertRaises(TypeError):
                sample("exponential", 1, {"rate": 1.0})  # type: ignore[call-overload]
            with self.assertRaises(TypeError):
                sample("exponential", 1, {"rate": 1.0}, rng=7)
            with self.assertRaises(TypeError):
                sample("exponential", 1, {"rate": 1.0}, rate=1.0, rng=generator)  # type: ignore[call-overload]

    def test_deprecated_exponential_rate_errors_keep_their_historical_type(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            for bad in ("one", True, None, {1.0}):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    cdf("exponential", 0.0, {"rate": bad})  # type: ignore[dict-item]
            with self.assertRaises(TypeError):
                cdf("exponential", 0.0, {"scale": 1.0})
            # The keyword form reports the same mistake as a TypeError.
            with self.assertRaises(TypeError):
                cdf("exponential", 0.0, rate="one")
            # Other families never had the special case.
            with self.assertRaises(TypeError):
                cdf("normal", 0.0, {"mu": "zero", "sigma": 1.0})  # type: ignore[dict-item]


class OperationTableTests(unittest.TestCase):
    def test_every_advertised_operation_has_a_kernel_and_the_reverse(self) -> None:
        registry = FAMILY_REGISTRY.families
        distributions._verify_operation_tables(registry, distributions._TABLES)
        for operation, table in distributions._TABLES.items():
            self.assertEqual(set(table), set(FamilyId), operation)
            incomplete = {k: v for k, v in table.items() if k is not FamilyId.GAMMA}
            with self.assertRaises(RuntimeError):
                distributions._verify_operation_tables(registry, {operation: incomplete})
            with self.assertRaises(RuntimeError):
                distributions._verify_operation_tables(
                    {k: v for k, v in registry.items() if k is not FamilyId.GAMMA},
                    {operation: table},
                )
        self.assertEqual(
            set(distributions._TABLES), set(Operation) - {Operation.LOGPDF, Operation.FIT}
        )


if __name__ == "__main__":
    unittest.main()
