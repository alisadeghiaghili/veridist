"""RED contracts for the evaluated-family registry boundary."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError
from math import inf, nan
from types import MappingProxyType

from veridist.families.registry import (
    FAMILY_REGISTRY,
    FamilyId,
    FamilyRegistry,
    FamilySpec,
    Operation,
    ParameterRole,
    ParameterSpec,
    list_families,
)


class FamilyRegistryContractTests(unittest.TestCase):
    """FAM-REG and FAM-PAR contracts before numerical evaluation exists."""

    def test_fam_reg_01_canonical_ids_and_listing_order_are_closed(self) -> None:
        self.assertEqual(
            tuple(family.id for family in list_families()),
            (
                FamilyId.NORMAL,
                FamilyId.GAMMA,
                FamilyId.WEIBULL_MIN,
                FamilyId.LOGNORMAL,
                FamilyId.GUMBEL_RIGHT,
                FamilyId.EXPONENTIAL,
            ),
        )
        self.assertEqual(
            tuple(family.id.value for family in list_families()),
            ("normal", "gamma", "weibull_min", "lognormal", "gumbel_right", "exponential"),
        )
        self.assertEqual(tuple(FAMILY_REGISTRY), tuple(family.id for family in list_families()))
        self.assertEqual(list_families(), list_families())

    def test_fam_reg_02_declared_aliases_resolve_without_normalization(self) -> None:
        self.assertIs(FAMILY_REGISTRY.resolve("normal"), FAMILY_REGISTRY.resolve("gaussian"))
        self.assertIs(FAMILY_REGISTRY.resolve("weibull_min"), FAMILY_REGISTRY.resolve("weibull"))
        self.assertIs(FAMILY_REGISTRY.resolve("gumbel_right"), FAMILY_REGISTRY.resolve("gumbel"))
        for unregistered_name in ("Normal", "weibull-min", "gumbel right", "gamma_dist"):
            with self.subTest(unregistered_name=unregistered_name):
                with self.assertRaises(ValueError):
                    FAMILY_REGISTRY.resolve(unregistered_name)
        with self.assertRaises(TypeError):
            FAMILY_REGISTRY.resolve(1)  # type: ignore[arg-type]

    def test_fam_reg_02_alias_collisions_fail_during_registry_construction(self) -> None:
        first = _test_family(FamilyId.NORMAL, aliases=("shared",))
        second = _test_family(FamilyId.GAMMA, aliases=("shared",))
        with self.assertRaisesRegex(ValueError, "alias"):
            FamilyRegistry((first, second))
        with self.assertRaisesRegex(ValueError, "canonical"):
            FamilyRegistry((_test_family(FamilyId.NORMAL), _test_family(FamilyId.NORMAL)))

    def test_fam_reg_03_specs_and_registry_are_immutable(self) -> None:
        normal = FAMILY_REGISTRY.resolve("normal")
        self.assertIsInstance(FAMILY_REGISTRY.families, MappingProxyType)
        with self.assertRaises(TypeError):
            FAMILY_REGISTRY.families[FamilyId.NORMAL] = normal  # type: ignore[index]
        with self.assertRaises(FrozenInstanceError):
            normal.aliases = ()  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            normal.parameters[0].name = "changed"  # type: ignore[misc]
        with self.assertRaises(AttributeError):
            normal.planned_operations.add(Operation.LOGPDF)  # type: ignore[attr-defined]
        for field in ("_families", "_lookup", "_ordered"):
            with self.subTest(field=field):
                with self.assertRaises(AttributeError):
                    setattr(FAMILY_REGISTRY, field, None)
        self.assertIs(FAMILY_REGISTRY.resolve("normal"), normal)

    def test_fam_par_01_parameter_tuples_counts_and_operations_are_exact(self) -> None:
        expected = {
            "normal": (("mu", "sigma"), 2, None),
            "gamma": (("shape", "scale"), 2, 0.0),
            "weibull_min": (("shape", "scale"), 2, 0.0),
            "lognormal": (("mu_log", "sigma_log"), 2, 0.0),
            "gumbel_right": (("location", "scale"), 2, None),
            "exponential": (("rate",), 1, 0.0),
        }
        for identifier, (names, free_count, fixed_location) in expected.items():
            with self.subTest(identifier=identifier):
                family = FAMILY_REGISTRY.resolve(identifier)
                self.assertEqual(tuple(parameter.name for parameter in family.parameters), names)
                self.assertEqual(family.free_parameter_count, free_count)
                self.assertEqual(family.fixed_location, fixed_location)
                self.assertEqual(family.planned_operations, frozenset(Operation))
                self.assertEqual(family.available_operations, frozenset(Operation))
                self.assertTrue(family.plans(Operation.LOGPDF))
                self.assertTrue(family.supports(Operation.LOGPDF))

    def test_fam_par_02_finite_locations_are_valid_and_positive_parameters_are_strict(self) -> None:
        valid = {
            "normal": {"mu": -3.5, "sigma": 1},
            "gamma": {"shape": 2, "scale": 0.5},
            "weibull_min": {"shape": 3.0, "scale": 2},
            "lognormal": {"mu_log": -2, "sigma_log": 0.25},
            "gumbel_right": {"location": 7, "scale": 1.5},
        }
        for identifier, parameters in valid.items():
            with self.subTest(identifier=identifier):
                self.assertEqual(
                    FAMILY_REGISTRY.resolve(identifier).validate_parameters(**parameters),
                    MappingProxyType({name: float(value) for name, value in parameters.items()}),
                )
        for identifier in ("normal", "gamma", "weibull_min", "lognormal", "gumbel_right"):
            family = FAMILY_REGISTRY.resolve(identifier)
            valid_parameters = {parameter.name: 1.0 for parameter in family.parameters}
            for parameter in family.parameters:
                if parameter.role is ParameterRole.POSITIVE:
                    for value in (True, nan, inf, -inf, 0.0, -1.0):
                        with self.subTest(
                            identifier=identifier, parameter=parameter.name, value=value
                        ):
                            invalid = {**valid_parameters, parameter.name: value}
                            with self.assertRaises((TypeError, ValueError)):
                                family.validate_parameters(**invalid)
            for parameter in family.parameters:
                if parameter.role is ParameterRole.FINITE:
                    for value in (nan, inf, -inf):
                        with self.subTest(
                            identifier=identifier, parameter=parameter.name, value=value
                        ):
                            invalid = {**valid_parameters, parameter.name: value}
                            with self.assertRaises(ValueError):
                                family.validate_parameters(**invalid)
        with self.assertRaises(TypeError):
            FAMILY_REGISTRY.resolve("normal").validate_parameters(mu=True, sigma=1.0)
        with self.assertRaises(TypeError):
            FAMILY_REGISTRY.resolve("normal").validate_parameters(mu=0.0, sigma="1")
        with self.assertRaises(ValueError):
            FAMILY_REGISTRY.resolve("normal").validate_parameters(mu=10**10000, sigma=1.0)

    def test_fam_par_02_parameter_keys_are_closed_not_mapping_length_driven(self) -> None:
        family = FAMILY_REGISTRY.resolve("normal")
        self.assertEqual(family.free_parameter_count, 2)
        with self.assertRaises(TypeError):
            family.validate_parameters(mu=0.0)
        with self.assertRaises(TypeError):
            family.validate_parameters(mu=0.0, sigma=1.0, unused=2.0)

    def test_fam_iso_01_kernel_carries_no_fitted_state_or_localized_text(self) -> None:
        for family in list_families():
            self.assertFalse(hasattr(family, "fit"))
            self.assertFalse(hasattr(family, "display_name"))
            self.assertFalse(hasattr(family, "description"))

    def test_fam_reg_02_alias_tokens_are_closed_and_collision_safe(self) -> None:
        for invalid_alias in ("a-b", "A", "has space", "", "punctuation!"):
            with self.subTest(invalid_alias=invalid_alias):
                with self.assertRaises((TypeError, ValueError)):
                    _test_family(FamilyId.NORMAL, aliases=(invalid_alias,))
        with self.assertRaisesRegex(ValueError, "collision"):
            FamilyRegistry(
                (
                    _test_family(FamilyId.NORMAL, aliases=("a_b",)),
                    _test_family(FamilyId.GAMMA, aliases=("ab",)),
                )
            )
        with self.assertRaisesRegex(ValueError, "collision"):
            FamilyRegistry(
                (
                    _test_family(FamilyId.NORMAL, aliases=("gamma",)),
                    _test_family(FamilyId.GAMMA),
                )
            )


def _test_family(identifier: FamilyId, *, aliases: tuple[str, ...] = ()) -> FamilySpec:
    return FamilySpec(
        id=identifier,
        aliases=aliases,
        parameters=(ParameterSpec("scale", ParameterRole.POSITIVE),),
        fixed_location=None,
        planned_operations=frozenset({Operation.LOGPDF}),
        available_operations=frozenset(),
    )
