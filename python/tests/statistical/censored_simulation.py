"""Seeded simulation of censored and truncated samples for the statistical tests.

A latent value is drawn by inverting the library's own distribution function between the ends of
its truncation window (so it is conditional on the window), and then coarsened:

* units are inspected at fixed times (quantiles of the true distribution); a unit that is not
  observed exactly is known only to lie between two inspections, which makes it left-censored
  before the first, interval-censored between two and right-censored after the last, and the
  interval is clipped to the unit's window;
* ``coarsening`` says which rows are coarsened and how: ``exact`` (none), ``right``, ``left`` or
  ``interval`` (half of the rows, to that kind where the grid allows it, else exact), ``mixed``
  (a quarter of each kind) and ``inspection`` (70 percent of the rows by the grid, whichever kind
  the grid gives, as an inspection scheme does);
* ``truncation`` says which windows the rows carry: ``none``, ``left`` (delayed entry),
  ``right`` (a cut-off) or ``two_sided``, a window per row.

Every row is built with the typed observation constructors, so the consistency rules of the
observation model are exercised by the simulation itself.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tests.reference import censored_support as support
from veridist import ppf

#: True parameters of each family: ordinary reliability-type values.
TRUTH: dict[str, dict[str, float]] = {
    "exponential": {"rate": 0.004},
    "weibull_min": {"shape": 1.7, "scale": 300.0},
    "lognormal": {"mu_log": 5.0, "sigma_log": 0.6},
    "gamma": {"shape": 3.0, "scale": 80.0},
    "normal": {"mu": 100.0, "sigma": 15.0},
    "gumbel_right": {"location": 100.0, "scale": 12.0},
}
#: The schemes of the calibration grid: (coarsening, truncation).
SCHEMES: dict[str, tuple[str, str]] = {
    "inspection": ("inspection", "none"),
    "left": ("interval", "left"),
    "right": ("interval", "right"),
    "two_sided": ("interval", "two_sided"),
}
COARSENINGS = ("exact", "right", "left", "interval", "mixed")
TRUNCATIONS = ("none", "left", "right", "two_sided")
INSPECTION_LEVELS = tuple(np.linspace(0.05, 0.95, 10))
#: The share of rows that are coarsened, by coarsening.
SHARE = {"exact": 0.0, "right": 0.5, "left": 0.5, "interval": 0.5, "mixed": 1.0, "inspection": 0.7}


def _quantile(family: str, probability: Any) -> Any:
    return ppf(family, np.clip(probability, 1e-12, 1.0 - 1e-12), **TRUTH[family])


def simulate(family: str, size: int, scheme: str, rng: np.random.Generator) -> list[Any]:
    """``size`` observations of ``family`` of one of the calibration ``SCHEMES``."""

    coarsening, truncation = SCHEMES[scheme]
    return simulate_cell(family, size, coarsening, truncation, rng)


def simulate_cell(
    family: str, size: int, coarsening: str, truncation: str, rng: np.random.Generator
) -> list[Any]:
    """``size`` observations coarsened as ``coarsening`` says, with the ``truncation`` windows."""

    lifetime = family in support.LIFETIME_FAMILIES
    grid = [float(_quantile(family, level)) for level in INSPECTION_LEVELS]
    low_p = rng.uniform(0.0, 0.4, size) if truncation in ("left", "two_sided") else np.zeros(size)
    high_p = rng.uniform(0.6, 0.99, size) if truncation in ("right", "two_sided") else np.ones(size)
    window_low = np.where(low_p > 0.0, _quantile(family, low_p), -math.inf)
    window_high = np.where(high_p < 1.0, _quantile(family, high_p), math.inf)
    latent = _quantile(family, low_p + (high_p - low_p) * rng.uniform(size=size))
    coarse = rng.uniform(size=size) < SHARE[coarsening]
    picks = rng.integers(0, 4, size)
    rows = []
    for index in range(size):
        x = float(latent[index])
        a, b = float(window_low[index]), float(window_high[index])
        window = None if math.isinf(a) and math.isinf(b) else (a, b)
        exact = support.observation(family, "exact", x, None, window)
        if not coarse[index]:
            rows.append(exact)
            continue
        position = int(np.searchsorted(grid, x, side="left"))
        lower = max(grid[position - 1] if position > 0 else -math.inf, a)
        upper = min(grid[position] if position < len(grid) else math.inf, b)
        if not lower < x <= upper:
            rows.append(exact)
            continue
        if lifetime and lower <= 0.0:
            lower = -math.inf
        want = {
            "right": "right",
            "left": "left",
            "interval": "interval",
            "mixed": ("right", "left", "interval", "interval")[int(picks[index])],
            "inspection": "grid",
        }[coarsening]
        finite_lower, finite_upper = not math.isinf(lower), not math.isinf(upper)
        if want == "grid":
            want = (
                "interval"
                if finite_lower and finite_upper
                else "right"
                if finite_lower
                else "left"
                if finite_upper
                else "exact"
            )
        if want == "right" and finite_lower:
            rows.append(support.observation(family, "right", lower, None, window))
        elif want == "left" and finite_upper:
            rows.append(support.observation(family, "left", upper, None, window))
        elif want == "interval" and finite_lower and finite_upper:
            rows.append(support.observation(family, "interval", lower, upper, window))
        else:
            rows.append(exact)
    return rows


def truth(family: str) -> dict[str, float]:
    return dict(TRUTH[family])
