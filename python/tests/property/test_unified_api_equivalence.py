"""The deprecated mapping call forms must return exactly what the keyword forms do.

``cdf``/``sf``/``ppf`` formerly took ``(family, value, mapping)`` and ``sample``
took ``(family, size, mapping, rng)``.  Those forms still work through a shim
that warns; this module proves, on randomized inputs for every family, that the
shim and the keyword form are bit-for-bit identical, including which exception
type an invalid call raises.
"""

from __future__ import annotations

import random
import unittest
import warnings
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from veridist.families.registry import FAMILY_REGISTRY, FamilyId
from veridist.statistics.distributions import cdf, ppf, sample, sf

CASES_PER_FAMILY_AND_OPERATION = 40


def _parameters(family: FamilyId, rng: random.Random) -> dict[str, float]:
    spec = FAMILY_REGISTRY.families[family]
    values: dict[str, float] = {}
    for parameter in spec.parameters:
        if parameter.role.value == "positive":
            values[parameter.name] = rng.choice(
                (rng.uniform(0.05, 6.0), 10 ** rng.uniform(-3.0, 3.0), float(rng.randint(1, 9)))
            )
        else:
            values[parameter.name] = rng.uniform(-8.0, 8.0)
    return values


def _point(family: FamilyId, rng: random.Random) -> float:
    if FAMILY_REGISTRY.families[family].fixed_location == 0.0:
        return rng.choice((rng.uniform(-1.0, 0.0), 10 ** rng.uniform(-4.0, 3.0), 0.0))
    return rng.uniform(-30.0, 30.0)


def _outcome(call: Callable[[], object]) -> tuple[str, object]:
    """Return ``("value", result)`` or ``("error", exception_type)``."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            return ("value", call())
        except (TypeError, ValueError, ArithmeticError) as error:
            return ("error", type(error))


class UnifiedApiEquivalenceTests(unittest.TestCase):
    def test_deprecated_mapping_forms_are_identical_to_the_keyword_forms(self) -> None:
        rng = random.Random(20261003)
        operations: dict[str, Callable[[object, float, Mapping[str, float], bool], object]] = {
            "cdf": lambda family, x, p, legacy: (
                cdf(family, x, p) if legacy else cdf(family, x, **p)  # type: ignore[arg-type]
            ),
            "sf": lambda family, x, p, legacy: (
                sf(family, x, p) if legacy else sf(family, x, **p)  # type: ignore[arg-type]
            ),
            "ppf": lambda family, x, p, legacy: (
                ppf(family, 1.0 / (1.0 + abs(x)), p)  # type: ignore[arg-type]
                if legacy
                else ppf(family, 1.0 / (1.0 + abs(x)), **p)
            ),
        }
        compared = 0
        differences: list[str] = []
        for family in FamilyId:
            for name, operation in operations.items():
                for _ in range(CASES_PER_FAMILY_AND_OPERATION):
                    parameters = _parameters(family, rng)
                    point = _point(family, rng)
                    # Both spellings of the family, and an occasional invalid value, so
                    # that error types are compared as well as results.
                    spelling: object = family if rng.random() < 0.5 else family.value
                    if rng.random() < 0.1:
                        parameters[next(iter(parameters))] = rng.choice((0.0, -1.0, float("nan")))
                    old = _outcome(lambda: operation(spelling, point, parameters, True))
                    new = _outcome(lambda: operation(spelling, point, parameters, False))
                    compared += 1
                    if old != new:
                        differences.append(f"{name} {family.value} {point!r} {parameters!r}")
        self.assertGreaterEqual(compared, 6 * 3 * CASES_PER_FAMILY_AND_OPERATION)
        self.assertEqual(differences, [])

    def test_sampling_forms_draw_identical_streams(self) -> None:
        rng = random.Random(7)
        compared = 0
        for family in FamilyId:
            for _ in range(CASES_PER_FAMILY_AND_OPERATION):
                parameters = _parameters(family, rng)
                size = rng.randint(0, 17)
                seed = rng.randint(0, 2**31)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    positional = sample(family, size, parameters, np.random.default_rng(seed))
                    keyword_rng = sample(
                        family.value, size, parameters, rng=np.random.default_rng(seed)
                    )
                    mapping_keyword = sample(
                        family, size, parameters=parameters, rng=np.random.default_rng(seed)
                    )
                new = sample(family, size, rng=np.random.default_rng(seed), **parameters)
                compared += 1
                with self.subTest(family=family, size=size):
                    self.assertEqual(new.shape, (size,))
                    self.assertTrue(np.array_equal(positional, new))
                    self.assertTrue(np.array_equal(keyword_rng, new))
                    self.assertTrue(np.array_equal(mapping_keyword, new))
        self.assertEqual(compared, 6 * CASES_PER_FAMILY_AND_OPERATION)

    def test_every_deprecated_spelling_warns_and_no_new_spelling_does(self) -> None:
        parameters = {"rate": 2.0}
        generator = np.random.default_rng(1)
        deprecated = (
            lambda: cdf("exponential", 1.0, parameters),
            lambda: cdf("exponential", 1.0, parameters=parameters),
            lambda: sf("exponential", 1.0, parameters),
            lambda: ppf("exponential", 0.5, parameters),
            lambda: sample("exponential", 2, parameters, generator),
            lambda: sample("exponential", 2, parameters, rng=generator),
            lambda: sample("exponential", 2, parameters=parameters, rng=generator),
        )
        for call in deprecated:
            with self.subTest(call=call), self.assertWarns(DeprecationWarning) as caught:
                call()
            self.assertIn("deprecated", str(caught.warning))
            self.assertIn("3.0", str(caught.warning))
            # The warning must point past the library to its caller. Checking
            # that it is not attributed to the distributions module keeps the
            # test valid when a mutation runner wraps calls in its own frames.
            self.assertNotEqual(Path(caught.filename).name, "distributions.py")
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            cdf("exponential", 1.0, rate=2.0)
            cdf(FamilyId.EXPONENTIAL, 1.0, rate=2.0)
            sf("exponential", 1.0, rate=2.0)
            ppf("exponential", 0.5, rate=2.0)
            sample("exponential", 2, rng=generator, rate=2.0)


if __name__ == "__main__":
    unittest.main()
