"""Seeded subsamples of censored and truncated data for the fit reference tests.

One *cell* is a family, a censoring pattern (``exact``, ``right``, ``left``, ``interval`` or
``mixed``) and a truncation type (``none``, ``left``, ``right``, ``two_sided``).  Its data are
drawn by inverting the distribution function between the window ends (so each latent value is
conditional on its window), rounded to four significant digits, and then coarsened according to
the pattern.  Only ``numpy.random.Generator.random`` and the library's own ``ppf`` are used, and
every number is rounded, so the cell is the same on every platform.

The patterns keep the likelihood informative about both parameters: ``right``, ``left`` and
``interval`` cells keep a share of exact rows, as ``interval`` and ``mixed`` do for the
inspection scheme, because a sample with no exact row and a single censoring limit has a flat
likelihood (a case the failure-code tests cover on their own).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tests.reference.censored_support import LIFETIME_FAMILIES, observation
from veridist.statistics.distributions import cdf, ppf

#: True parameters of each family (benign, mid-range).
TRUTH: dict[str, dict[str, float]] = {
    "exponential": {"rate": 0.004},
    "weibull_min": {"shape": 1.7, "scale": 300.0},
    "lognormal": {"mu_log": 5.0, "sigma_log": 0.6},
    "gamma": {"shape": 3.0, "scale": 80.0},
    "normal": {"mu": 100.0, "sigma": 15.0},
    "gumbel_right": {"location": 100.0, "scale": 12.0},
}
PATTERNS = ("exact", "right", "left", "interval", "mixed")
TRUNCATIONS = ("none", "left", "right", "two_sided")
#: Rows per cell: enough for a stable maximum, few enough for a 40-digit reference.
SIZE = 36
GAMMA_SIZE = 24
LOW_LEVELS = (0.05, 0.1, 0.18, 0.25)
HIGH_LEVELS = (0.8, 0.88, 0.94, 0.97)


@dataclass(frozen=True)
class Cell:
    family: str
    pattern: str
    truncation: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.family}/{self.pattern}/{self.truncation}"


#: A cell whose first sample has no interior maximum of the likelihood (a typed failure, not an
#: estimate: see ``NO_INTERIOR_MAXIMUM``) is drawn again with the seed shifted by this many
#: hundred thousands.
RESEEDED: dict[str, int] = {"gamma/left/two_sided": 3}
#: That first sample of the reseeded cell: the gamma likelihood increases as the shape tends to
#: zero, so the fit ends at its iteration cap.
NO_INTERIOR_MAXIMUM = Cell("gamma", "left", "two_sided", 4023)


def cells() -> list[Cell]:
    """Every family by pattern by truncation, with a seed that is a function of the cell."""

    out = []
    for family in TRUTH:
        for pattern in PATTERNS:
            for truncation in TRUNCATIONS:
                seed = (
                    1000 * (list(TRUTH).index(family) + 1)
                    + 10 * PATTERNS.index(pattern)
                    + TRUNCATIONS.index(truncation)
                )
                key = f"{family}/{pattern}/{truncation}"
                out.append(Cell(family, pattern, truncation, seed + 100000 * RESEEDED.get(key, 0)))
    return out


def _round(value: float) -> float:
    return float(f"{value:.4g}")


def _cdf(family: str, x: float) -> float:
    return float(cdf(family, x, **TRUTH[family]))


def _ppf(family: str, probability: float) -> float:
    return float(ppf(family, probability, **TRUTH[family]))


def rows(cell: Cell) -> list[tuple[str, float, float | None, tuple[float, float] | None]]:
    """``(kind, point, upper, window)`` of each row of ``cell`` (``window`` ``None``: none)."""

    family = cell.family
    size = GAMMA_SIZE if family == "gamma" else SIZE
    generator = np.random.default_rng(cell.seed)
    uniforms = generator.random((size, 3))
    lifetime = family in LIFETIME_FAMILIES
    spread = _ppf(family, 0.75) - _ppf(family, 0.25)
    out: list[tuple[str, float, float | None, tuple[float, float] | None]] = []
    for index in range(size):
        a = b = None
        if cell.truncation in ("left", "two_sided"):
            a = _round(_ppf(family, LOW_LEVELS[index % 4]))
        if cell.truncation in ("right", "two_sided"):
            b = _round(_ppf(family, HIGH_LEVELS[(index // 4) % 4]))
        low_p = 0.0 if a is None else _cdf(family, a)
        high_p = 1.0 if b is None else _cdf(family, b)
        x = _round(_ppf(family, low_p + (high_p - low_p) * float(uniforms[index, 0])))
        x = min(max(x, -math.inf if a is None else a), math.inf if b is None else b)
        window = (
            None
            if a is None and b is None
            else (
                -math.inf if a is None else a,
                math.inf if b is None else b,
            )
        )
        floor = (0.0 if lifetime else -math.inf) if a is None else a
        ceiling = math.inf if b is None else b
        kind = _kind(cell.pattern, index)
        if kind == "right":
            point = max(floor, _round(x - 0.6 * spread * float(uniforms[index, 1])))
            row = (kind, point if point > 0.0 or not lifetime else _round(0.5 * x), None, window)
        elif kind == "left":
            row = (
                kind,
                min(ceiling, _round(x + 0.6 * spread * float(uniforms[index, 1]))),
                None,
                window,
            )
        elif kind == "interval":
            half = 0.35 * spread
            lower = max(floor, _round(x - half * (0.4 + float(uniforms[index, 1]))))
            upper = min(ceiling, _round(x + half * (0.4 + float(uniforms[index, 2]))))
            row = (kind, lower, upper, window)
        else:
            row = ("exact", x, None, window)
        try:
            observation(family, row[0], row[1], row[2], row[3])
        except ValueError:
            row = ("exact", x, None, window)
        out.append(row)
    return out


def _kind(pattern: str, index: int) -> str:
    if pattern == "exact":
        return "exact"
    if pattern == "mixed":
        return ("exact", "right", "left", "interval")[index % 4]
    if pattern == "interval":
        return "interval" if index % 3 else "exact"
    return pattern if index % 3 == 0 else "exact"


def observations(cell: Cell) -> list[Any]:
    """The observation objects of ``cell``."""

    return [
        observation(cell.family, kind, point, upper, window)
        for kind, point, upper, window in rows(cell)
    ]
