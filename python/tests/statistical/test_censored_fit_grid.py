"""A seeded grid of fits: every family, truncation type and coarsening (CENS-FIT-03 in memory).

Six families by four truncation types by five coarsenings (exact, right-, left-, interval-censored
and mixed rows), one simulated sample each (``censored_simulation.py``).  Every fit must end as
an estimate or as a typed failure of a sample whose likelihood has no interior maximum (the
exponential with half of its rows right-censored and two-sided windows is one: its likelihood
increases as the rate tends to zero), and a plan that needs more than the maximum of 22 passes per
start is a failure of the planner; the failures are listed.  The gamma, whose fits cost many
kernel evaluations, is run on a smaller sample and for the mixed and interval coarsenings.  The
pass counts are reported as a histogram.
"""

from __future__ import annotations

import unittest
import zlib
from collections import Counter
from itertools import product

import numpy as np

import veridist as vd
from tests.statistical.censored_simulation import COARSENINGS, TRUNCATIONS, simulate_cell

SIZE = 100
GAMMA_SIZE = 60
MAXIMUM_PASSES = 22
#: A converged fit may take more than the default budget of 10 passes only rarely.
DEFAULT_BUDGET = 10


class FitGridTests(unittest.TestCase):
    def run_grid(self, family: str, size: int, coarsenings: tuple[str, ...]) -> list[int]:
        passes: list[int] = []
        failures = []
        for coarsening, truncation in product(coarsenings, TRUNCATIONS):
            seed = zlib.crc32(f"{family}/{coarsening}/{truncation}".encode("ascii"))
            rows = simulate_cell(family, size, coarsening, truncation, np.random.default_rng(seed))
            result = vd.fit(family, rows)
            if not hasattr(result, "parameters"):
                failures.append((coarsening, truncation, result.code.value))
                continue
            facts = result.diagnostics
            if facts is not None:
                self.assertLessEqual(facts.passes, MAXIMUM_PASSES * facts.starts)
                if facts.method == "newton":
                    passes.append(facts.passes // facts.starts)
        print(family, "typed failures:", failures)
        self.assertLessEqual(len(failures), 1, (family, failures))
        for failure in failures:
            self.assertIn(failure[2], ("BOUNDARY_SOLUTION", "OPTIMIZER_EXHAUSTED"))
        return passes

    def test_the_fast_families(self) -> None:
        histogram: Counter[int] = Counter()
        for family in ("exponential", "weibull_min", "lognormal", "normal", "gumbel_right"):
            histogram.update(self.run_grid(family, SIZE, COARSENINGS))
        total = sum(histogram.values())
        print("passes per start:", dict(sorted(histogram.items())))
        within = sum(count for passes, count in histogram.items() if passes <= DEFAULT_BUDGET)
        self.assertGreater(within / total, 0.9)

    def test_the_gamma(self) -> None:
        passes = self.run_grid("gamma", GAMMA_SIZE, ("interval", "mixed"))
        print("gamma passes per start:", sorted(passes))


if __name__ == "__main__":
    unittest.main()
