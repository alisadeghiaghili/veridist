"""Array evaluation must reproduce the scalar path on randomized broadcast shapes.

Normal, lognormal and gamma wrap the verified scalar kernels, so their array results
must be *identical* to the scalar results.  Exponential, Weibull and right Gumbel use
numpy-native kernels that mirror the scalar formulas; they are compared in units of
the last place (``numpy.spacing``) rather than bit for bit, because numpy's and the C
library's ``exp``/``log``/``pow`` may differ in the last bit between platforms.  The
design target is 2 ulp; the tolerance used here leaves headroom for those differences
and is applied on a domain that keeps the outer function well conditioned (a tail
where ``exp(-exp(-z))`` amplifies a last-bit difference of the inner ``exp`` is not a
property of the implementation).
"""

from __future__ import annotations

import itertools
import unittest
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from veridist.families.registry import FAMILY_REGISTRY, FamilyId, Operation
from veridist.statistics.distributions import cdf, logpdf, ppf, sf

#: Family -> whether its array kernels are numpy-native (compared in ulp) or wrapped (exact).
NATIVE = {FamilyId.EXPONENTIAL, FamilyId.WEIBULL_MIN, FamilyId.GUMBEL_RIGHT}
ULP_BUDGET = 4.0

#: Shapes that numpy can broadcast together, including empty and 0-d ones.
TARGET_SHAPES = ((), (6,), (3, 1), (1, 4), (2, 3, 4), (0,), (3, 0), (1, 1))
CASES_PER_COMBINATION = 12

_OPERATIONS: dict[Operation, Callable[..., Any]] = {
    Operation.LOGPDF: logpdf,
    Operation.CDF: cdf,
    Operation.SF: sf,
    Operation.PPF: ppf,
}


def _operand_shape(target: tuple[int, ...], rng: np.random.Generator) -> tuple[int, ...]:
    """Return a shape that broadcasts to ``target``: drop leading axes, squeeze some others."""

    keep = int(rng.integers(0, len(target) + 1))
    trailing = target[len(target) - keep :]
    return tuple(1 if rng.random() < 0.4 else size for size in trailing)


def _draw(family: FamilyId, name: str, shape: tuple[int, ...], rng: np.random.Generator) -> Any:
    """Draw a well-conditioned parameter array (positive parameters in a moderate range)."""

    if name in {"mu", "mu_log", "location"}:
        return rng.uniform(-3.0, 3.0, shape)
    if name == "shape" and family is FamilyId.WEIBULL_MIN:
        return rng.uniform(0.6, 4.0, shape)
    if name == "shape":
        return rng.uniform(0.4, 9.0, shape)
    return rng.uniform(0.4, 3.0, shape)


def _points(
    family: FamilyId, operation: Operation, shape: tuple[int, ...], rng: np.random.Generator
) -> Any:
    """Draw the evaluation points: probabilities for ``ppf``, support-straddling values else."""

    if operation is Operation.PPF:
        return np.asarray(rng.uniform(1e-6, 1.0 - 1e-6, shape))
    if FAMILY_REGISTRY.families[family].fixed_location == 0.0:
        raw = np.asarray(rng.uniform(-0.5, 6.0, shape))
        if raw.size:  # make sure the boundary itself is exercised
            raw.reshape(-1)[0] = 0.0
        return raw
    return np.asarray(rng.uniform(-8.0, 8.0, shape))


def _scalar_reference(
    operation: Operation,
    family: FamilyId,
    point: Any,
    parameters: Mapping[str, Any],
) -> Any:
    """Evaluate element by element through the scalar path (Python floats only)."""

    arrays = np.broadcast_arrays(point, *parameters.values())
    names = tuple(parameters)
    out = np.empty(arrays[0].shape, dtype=np.float64)
    for index in np.ndindex(*arrays[0].shape):
        values = {name: float(arrays[1 + i][index]) for i, name in enumerate(names)}
        result = _OPERATIONS[operation](family, float(arrays[0][index]), **values)
        assert type(result) is float
        out[index] = result
    return out


def _ulp_distance(got: Any, want: Any) -> Any:
    """Distance in last-place units of the larger magnitude; equal values (also -inf) give 0."""

    scale = np.spacing(np.maximum(np.abs(got), np.abs(want)))
    with np.errstate(invalid="ignore"):
        distance = np.abs(got - want) / np.where(scale == 0.0, np.spacing(0.0), scale)
    return np.where(got == want, 0.0, distance)


class VectorizedEquivalenceTests(unittest.TestCase):
    def test_array_results_match_the_scalar_path_on_random_broadcast_shapes(self) -> None:
        rng = np.random.default_rng(20261004)
        compared = 0
        for family, operation in itertools.product(FamilyId, _OPERATIONS):
            names = [parameter.name for parameter in FAMILY_REGISTRY.families[family].parameters]
            for _ in range(CASES_PER_COMBINATION):
                target = TARGET_SHAPES[int(rng.integers(len(TARGET_SHAPES)))]
                parameters = {
                    name: _draw(family, name, _operand_shape(target, rng), rng) for name in names
                }
                if family is FamilyId.GUMBEL_RIGHT and operation is not Operation.PPF:
                    z = np.asarray(rng.uniform(-1.5, 6.0, target))
                    point = parameters["location"] + parameters["scale"] * z
                else:
                    point = _points(family, operation, target, rng)
                with self.subTest(family=family.value, operation=operation.value, shape=target):
                    got = _OPERATIONS[operation](family, point, **parameters)
                    want = _scalar_reference(operation, family, point, parameters)
                    expected_shape = np.broadcast_shapes(
                        np.shape(point), *(np.shape(v) for v in parameters.values())
                    )
                    if expected_shape == ():
                        self.assertIs(type(got), float)
                        got = np.asarray(got)
                    else:
                        self.assertIsInstance(got, np.ndarray)
                        self.assertEqual(got.dtype, np.float64)
                        self.assertEqual(got.shape, expected_shape)
                    if family in NATIVE:
                        worst = float(np.max(_ulp_distance(got, want), initial=0.0))
                        self.assertLessEqual(worst, ULP_BUDGET)
                    else:
                        np.testing.assert_array_equal(got, want)
                    compared += got.size
        self.assertGreater(compared, 500)

    def test_logpdf_is_minus_infinity_outside_the_support_element_wise(self) -> None:
        points = np.array([-2.0, -1e-300, 0.0, 1e-300, 0.5, 3.0])
        for family in FamilyId:
            spec = FAMILY_REGISTRY.families[family]
            names = [parameter.name for parameter in spec.parameters]
            parameters = {name: np.full((6,), 1.5) for name in names}
            with self.subTest(family=family.value):
                got = logpdf(family, points, **parameters)
                inside = np.array([spec.contains(float(x)) for x in points])
                self.assertTrue(np.all(got[~inside] == -np.inf))
                self.assertTrue(np.all(np.isfinite(got[inside])))
                for value, x in zip(got, points, strict=True):
                    self.assertEqual(value, logpdf(family, float(x), **{n: 1.5 for n in names}))

    def test_exponential_is_closed_at_zero_and_the_open_families_are_not(self) -> None:
        zero = np.zeros(2)
        self.assertEqual(logpdf("exponential", zero, rate=2.0)[0], np.log(2.0))
        for family, parameters in (
            ("gamma", {"shape": 2.0, "scale": 1.0}),
            ("weibull_min", {"shape": 2.0, "scale": 1.0}),
            ("lognormal", {"mu_log": 0.0, "sigma_log": 1.0}),
        ):
            with self.subTest(family=family):
                self.assertTrue(np.all(logpdf(family, zero, **parameters) == -np.inf))

    def test_cdf_and_sf_saturate_outside_the_support_for_fixed_location_families(self) -> None:
        points = np.array([-3.0, -1e-300, 0.0])
        for family, parameters in (
            ("exponential", {"rate": 1.0}),
            ("gamma", {"shape": 2.0, "scale": 1.0}),
            ("weibull_min", {"shape": 2.0, "scale": 1.0}),
            ("lognormal", {"mu_log": 0.0, "sigma_log": 1.0}),
        ):
            with self.subTest(family=family):
                self.assertTrue(np.all(cdf(family, points, **parameters) == 0.0))
                self.assertTrue(np.all(sf(family, points, **parameters) == 1.0))


if __name__ == "__main__":
    unittest.main()
