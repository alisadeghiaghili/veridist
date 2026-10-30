"""Fits of censored and truncated data against a 30-digit maximisation (CENS-FIT-01, CENS-REF).

For every family, censoring pattern (exact, right, left, interval and mixed rows) and truncation
type (none, left, right, two-sided) one seeded subsample of 24 to 36 rows
(``tests/reference/censored_fit_cells.py``) was maximised in ``mpmath`` at 30 digits by Newton
iteration on its own log-likelihood (``censored_fit_oracle.py``, which shares no code with the
library's kernels); the stationary points are the constants of ``censored_fit_constants.py``.
The library's estimate must agree with them to a declared relative tolerance, its reported
log-likelihood must equal the oracle's maximum, and the claim it makes about its point (a closed
form, a global maximum by concavity, or a certified stationary point) must be the one the
record gives for the cell.  The gamma cells, whose shape derivatives are finite differences and
which cost more per fit, are in ``tests/statistical/test_censored_gamma_reference.py``
with the sample that has no interior maximum.

A cell of exact and right-censored rows without a window is run by the bounded search of the
earlier releases, which is accurate to about ``1e-7`` only; its tolerance is declared as such.
"""

from __future__ import annotations

import unittest
from typing import Any

import veridist as vd
from tests.reference import censored_fit_cells as cells
from tests.reference.censored_fit_constants import MAXIMA
from veridist.families._fit_facts import OptimumKind

#: Relative tolerance of the parameters by family (binary64 rounding of a well-conditioned
#: maximum; the loosest observed difference is 1.2e-15).
TOLERANCE = 1e-11
#: The bounded golden-section search of the earlier releases.
LEGACY_TOLERANCE = 1e-6
LIKELIHOOD_TOLERANCE = 1e-9


def expected_optimum(cell: cells.Cell) -> OptimumKind:
    """The claim the record makes for the cell."""

    plain_kinds = cell.pattern in ("exact", "right")
    if plain_kinds and cell.truncation == "none":
        return OptimumKind.UNASSESSED
    if cell.family == "exponential":
        if plain_kinds and cell.truncation == "left":
            return OptimumKind.CLOSED_FORM
        if cell.truncation in ("none", "left"):
            return OptimumKind.GLOBAL_MAXIMUM
        return OptimumKind.STATIONARY_POINT
    if cell.family == "gamma":
        return OptimumKind.STATIONARY_POINT
    if cell.truncation == "none":
        return OptimumKind.GLOBAL_MAXIMUM
    return OptimumKind.STATIONARY_POINT


def uses_profile(cell: cells.Cell) -> bool:
    return (
        cell.family == "weibull_min"
        and cell.pattern in ("exact", "right")
        and (cell.truncation == "left")
    )


class CensoredFitReferenceTests(unittest.TestCase):
    def check(self, cell: cells.Cell) -> Any:
        parameters, likelihood = MAXIMA[cell.key]
        result = vd.fit(cell.family, cells.observations(cell))
        self.assertTrue(hasattr(result, "parameters"), result)
        optimum = expected_optimum(cell)
        self.assertIs(result.optimum, optimum)
        tolerance = LEGACY_TOLERANCE if optimum is OptimumKind.UNASSESSED else TOLERANCE
        found = tuple(result.parameters.values())
        for name, value, reference in zip(result.parameters, found, parameters, strict=True):
            with self.subTest(parameter=name):
                self.assertLessEqual(abs(value - reference), tolerance * abs(reference))
        self.assertLessEqual(
            abs(result.log_likelihood - likelihood), LIKELIHOOD_TOLERANCE * (1 + abs(likelihood))
        )
        self.assertTrue(result.converged)
        rows = cells.rows(cell)
        self.assertEqual(result.observation_count, len(rows))
        self.assertEqual(result.event_count, sum(row[0] == "exact" for row in rows))
        summary = result.observation_summary
        for kind, count in (
            ("exact", summary.exact),
            ("right", summary.right_censored),
            ("left", summary.left_censored),
            ("interval", summary.interval_censored),
        ):
            self.assertEqual(count, sum(row[0] == kind for row in rows), kind)
        self.assertEqual(summary.truncated, sum(row[3] is not None for row in rows))
        return result


def _add(cell: cells.Cell) -> None:
    def test(self: CensoredFitReferenceTests) -> None:
        result = self.check(cell)
        if uses_profile(cell):
            self.assertEqual(result.diagnostics.method, "profile")
            self.assertEqual(result.diagnostics.stationary_points_seen, 1)
        elif expected_optimum(cell) in (OptimumKind.GLOBAL_MAXIMUM, OptimumKind.STATIONARY_POINT):
            self.assertEqual(result.diagnostics.method, "newton")
            self.assertLessEqual(result.diagnostics.passes, 22 * result.diagnostics.starts)
        if result.optimum is OptimumKind.GLOBAL_MAXIMUM:
            self.assertEqual(result.diagnostics.starts, 1)
        elif result.optimum is OptimumKind.STATIONARY_POINT and not uses_profile(cell):
            self.assertEqual(result.diagnostics.starts, 2)

    name = f"test_{cell.family}_{cell.pattern}_{cell.truncation}"
    test.__name__ = name
    setattr(CensoredFitReferenceTests, name, test)


for _cell in cells.cells():
    if _cell.family != "gamma":
        _add(_cell)


class CellTests(unittest.TestCase):
    def test_the_constants_cover_every_cell(self) -> None:
        self.assertEqual(sorted(MAXIMA), sorted(cell.key for cell in cells.cells()))
        self.assertEqual(len(MAXIMA), 120)


if __name__ == "__main__":
    unittest.main()
