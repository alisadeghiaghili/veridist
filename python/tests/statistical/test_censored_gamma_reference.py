"""The gamma cells of the fit reference, and the reproduction of the reference constants.

The gamma likelihood has no closed-form shape derivative, so its fits use five-point central
differences in ``log shape`` and the looser step tolerance of ``1e-6``; each fit also costs
many kernel evaluations (five per row and candidate).  These cells are the gamma part of
``tests/reference/test_censored_fits_reference.py``, kept out of the mutation selection for
their cost, with the sample whose likelihood has no interior maximum, and the check that the
30-digit oracle reproduces the stored constants (a stationary point, with a gradient at the
noise level of the working precision) for one mixed, two-sided cell of every family.
"""

from __future__ import annotations

import unittest

import mpmath

import veridist as vd
from tests.reference import censored_fit_cells as cells
from tests.reference import censored_fit_oracle as oracle
from tests.reference.censored_fit_constants import MAXIMA
from veridist.families._fit_facts import OptimumKind

#: Observed worst relative difference 2.3e-10 (finite-difference derivatives, step 1e-6).
GAMMA_TOLERANCE = 1e-8


class GammaReferenceTests(unittest.TestCase):
    def test_every_gamma_cell_agrees_with_the_thirty_digit_maximum(self) -> None:
        gamma_cells = [cell for cell in cells.cells() if cell.family == "gamma"]
        self.assertEqual(len(gamma_cells), 20)
        for cell in gamma_cells:
            with self.subTest(cell=cell.key):
                parameters, likelihood = MAXIMA[cell.key]
                result = vd.fit("gamma", cells.observations(cell))
                self.assertTrue(hasattr(result, "parameters"), result)
                expected = (
                    OptimumKind.UNASSESSED
                    if cell.pattern in ("exact", "right") and cell.truncation == "none"
                    else OptimumKind.STATIONARY_POINT
                )
                self.assertIs(result.optimum, expected)
                tolerance = 1e-6 if expected is OptimumKind.UNASSESSED else GAMMA_TOLERANCE
                for value, reference in zip(result.parameters.values(), parameters, strict=True):
                    self.assertLessEqual(abs(value - reference), tolerance * abs(reference))
                self.assertLessEqual(
                    abs(result.log_likelihood - likelihood), 1e-9 * (1 + abs(likelihood))
                )
                if expected is OptimumKind.STATIONARY_POINT:
                    self.assertEqual(result.diagnostics.starts, 2)
                    self.assertLessEqual(result.diagnostics.passes, 44)

    def test_a_sample_without_an_interior_maximum_is_never_an_estimate(self) -> None:
        result = vd.fit("gamma", cells.observations(cells.NO_INTERIOR_MAXIMUM))
        self.assertEqual(result.code.value, "OPTIMIZER_EXHAUSTED")
        self.assertEqual(result.coarsening_assumption.value, "independent_censoring_and_truncation")
        self.assertEqual(result.observation_summary.total, 24)


class OracleReproducesConstantsTests(unittest.TestCase):
    def test_a_mixed_two_sided_cell_of_every_family_is_a_stationary_point(self) -> None:
        for cell in cells.cells():
            if cell.pattern != "mixed" or cell.truncation != "two_sided":
                continue
            with self.subTest(cell=cell.key):
                result = vd.fit(cell.family, cells.observations(cell))
                start = tuple(result.parameters.values())
                parameters, value, gradient = oracle.maximise(
                    cell.family, cells.rows(cell), start, digits=30
                )
                stored, stored_value = MAXIMA[cell.key]
                self.assertEqual(tuple(float(p) for p in parameters), stored)
                self.assertEqual(float(value), stored_value)
                with mpmath.workdps(45):
                    self.assertLess(max(abs(g) for g in gradient), mpmath.mpf(10) ** -18)
                self.assertLess(oracle.relative_difference(start, parameters), GAMMA_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
