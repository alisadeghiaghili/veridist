"""Coverage of Wald and profile intervals for censored and truncated data (CENS-UNC calibration).

A seeded grid, declared as such and not a general guarantee (``scope: declared_grid_only``): five
families, four schemes (an inspection scheme with exact, right-, left- and interval-censored
rows; a left-, a right- and a two-sided-truncated scheme with a window per row), sample size
``SIZE`` and ``WALD_REPLICATES`` replicates each for the Wald intervals, simulated with the
library's own distribution functions (``censored_simulation.py``); the profile intervals, which
cost many likelihood evaluations (seconds per fit in the two-sided truncated cells, where a side
may be unbounded and the search runs to its limit), are covered for the inspection and the
left-truncated schemes with ``PROFILE_REPLICATES`` replicates, and the gamma, whose fits cost many
kernel evaluations, has a small Wald grid of its own.  For every cell, parameter and method the
share of 95 percent intervals that contain the true value is compared with the nominal level within
``STANDARD_ERRORS`` binomial standard errors (3.5, as the refit goodness-of-fit calibration).  A fit
that fails, and an uncertainty that is unavailable, are counted and reported, never dropped from the
table.

The grid runs in under a minute, so its standard errors are wide (0.03 to 0.06): it catches an
interval that is wrong by construction (a wrong Hessian, a wrong profile), not a few percent of
miscoverage, which is what Wald intervals show at these sample sizes.  A cell is calibrated only
on this grid.
"""

from __future__ import annotations

import math
import unittest
from collections import defaultdict

import numpy as np

import veridist as vd
from tests.statistical.censored_simulation import SCHEMES, simulate, truth

FAMILIES = ("exponential", "weibull_min", "lognormal", "normal", "gumbel_right")
LEVEL = 0.95
SIZE = 80
WALD_REPLICATES = 40
PROFILE_REPLICATES = 24
GAMMA_REPLICATES = 10
STANDARD_ERRORS = 3.5
PROFILE_SCHEMES = ("inspection", "left")


class CoverageTests(unittest.TestCase):
    def run_cell(self, family: str, scheme: str, methods: tuple[str, ...], replicates: int) -> dict:
        rng = np.random.default_rng(zlib_seed(family, scheme))
        hits: dict[tuple[str, str], int] = defaultdict(int)
        trials: dict[tuple[str, str], int] = defaultdict(int)
        failures = 0
        unavailable = 0
        for _ in range(replicates):
            result = vd.fit(family, simulate(family, SIZE, scheme, rng))
            if not hasattr(result, "parameters"):
                failures += 1
                continue
            uncertainty = result.uncertainty()
            if not hasattr(uncertainty, "covariance"):
                unavailable += 1
                continue
            for method in methods:
                for name, interval in uncertainty.interval_details(LEVEL, method).items():
                    key = (method, name)
                    trials[key] += 1
                    hits[key] += int(interval.lower <= truth(family)[name] <= interval.upper)
        return {"hits": hits, "trials": trials, "failures": failures, "unavailable": unavailable}

    def check(self, family: str, scheme: str, methods: tuple[str, ...], replicates: int) -> None:
        cell = self.run_cell(family, scheme, methods, replicates)
        self.assertLessEqual(cell["failures"] + cell["unavailable"], 2, cell)
        for key, count in cell["trials"].items():
            rate = cell["hits"][key] / count
            error = math.sqrt(LEVEL * (1.0 - LEVEL) / count)
            print(
                f"coverage {family:13s} {scheme:10s} {key[0]:8s} {key[1]:11s} "
                f"{rate:.3f} +- {error:.3f} (n={count}, failed={cell['failures']}, "
                f"unavailable={cell['unavailable']})"
            )
            self.assertLessEqual(abs(rate - LEVEL), STANDARD_ERRORS * error, (family, scheme, key))

    def test_the_wald_grid(self) -> None:
        for family in FAMILIES:
            for scheme in SCHEMES:
                with self.subTest(family=family, scheme=scheme):
                    self.check(family, scheme, ("wald",), WALD_REPLICATES)

    def test_the_profile_grid(self) -> None:
        for family in FAMILIES:
            for scheme in PROFILE_SCHEMES:
                with self.subTest(family=family, scheme=scheme):
                    self.check(family, scheme, ("profile",), PROFILE_REPLICATES)

    def test_the_gamma_cells(self) -> None:
        with self.subTest(scheme="inspection"):
            self.check("gamma", "inspection", ("wald",), GAMMA_REPLICATES)


def zlib_seed(family: str, scheme: str) -> int:
    import zlib

    return zlib.crc32(f"{family}/{scheme}".encode("ascii"))


if __name__ == "__main__":
    unittest.main()
