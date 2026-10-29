"""Seeded adversarial observations for the censored and truncated log-likelihood contributions.

For each family a few parameter sets from benign to extreme (those of the kernel grids); for
each, standardised positions ``z`` from deep in the lower tail through the median to deep in the
upper tail.  At each position every censoring kind (exact, right, left, interval of widths from
``1e-9`` to ``4`` standardised units, the narrow ones inside the quadrature region) is built with
every truncation type (none, left, right, two-sided), the window ends at standardised offsets
from ``1e-6`` to ``6`` from the observation, and with the two edge cases in which a censoring
point equals a window end (the contribution is then exactly zero).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from tests.reference.log_probability_grids import _valid, median_z, z_ranges, z_to_x

#: Parameter sets per family: benign to extreme (gamma shapes up to 1000).
PARAMETERS: dict[str, tuple[tuple[float, ...], ...]] = {
    "exponential": ((1.0,), (3.7e-9,), (1e3,)),
    "weibull_min": ((0.05, 1.0), (1.7, 1000.0), (20.0, 3e-3), (50.0, 1.0)),
    "gumbel_right": ((0.0, 1.0), (50.0, 8.0), (1e6, 0.5)),
    "normal": ((0.0, 1.0), (100.0, 15.0), (1e6, 250.0)),
    "lognormal": ((0.0, 1.0), (7.0, 0.8), (3.0, 0.05)),
    "gamma": ((0.01, 1.0), (0.5, 1.0), (2.5, 10.0), (8.0, 1.0), (100.0, 1.0), (1000.0, 3.0)),
}
KINDS = ("exact", "right", "left", "interval")
TRUNCATIONS = ("none", "left", "right", "two_sided")
#: Standardised offsets of a window end from the observation.
OFFSETS = (1e-6, 0.7, 6.0)
WIDTHS = (1e-9, 0.3, 4.0)


@dataclass(frozen=True)
class Case:
    """One observation: its family and parameters, kind, point, upper bound and window."""

    family: str
    parameters: tuple[float, ...]
    kind: str
    point: float
    upper: float | None
    window: tuple[float, float] | None
    truncation: str

    def keywords(self) -> dict[str, float]:
        names = {
            "exponential": ("rate",),
            "weibull_min": ("shape", "scale"),
            "gumbel_right": ("location", "scale"),
            "normal": ("mu", "sigma"),
            "lognormal": ("mu_log", "sigma_log"),
            "gamma": ("shape", "scale"),
        }[self.family]
        return dict(zip(names, self.parameters, strict=True))


def _x(family: str, parameters: tuple[float, ...], z: float) -> float | None:
    value = float(z_to_x(family, parameters, z))
    return value if bool(_valid(family, np.array([value]))[0]) else None


def _windows(
    family: str,
    parameters: tuple[float, ...],
    low_z: float,
    high_z: float,
    truncation: str,
    offset: float,
) -> tuple[float, float] | None:
    """The window of ``truncation`` around the standardised censoring set ``[low_z, high_z]``."""

    if truncation == "none":
        return (-math.inf, math.inf)
    lower = _x(family, parameters, low_z - offset) if truncation != "right" else -math.inf
    upper = _x(family, parameters, high_z + offset) if truncation != "left" else math.inf
    if lower is None or upper is None:
        return None
    return (lower, upper)


def cases(family: str, seed: int = 20261025, positions: int = 7) -> list[Case]:
    """Every adversarial observation of ``family``, in a fixed order."""

    out: list[Case] = []
    for index, parameters in enumerate(PARAMETERS[family]):
        rng = np.random.default_rng([seed, index, len(family)])
        low, high = (z_ranges(family, parameters)[0][0], z_ranges(family, parameters)[1][1])
        median = median_z(family, parameters)
        zs = np.concatenate(
            [np.linspace(low, high, positions - 2), [median], rng.uniform(low, high, 1)]
        )
        for z in zs.tolist():
            point = _x(family, parameters, z)
            if point is None:
                continue
            out.extend(_at(family, parameters, z, point))
    return out


def _at(family: str, parameters: tuple[float, ...], z: float, point: float) -> list[Case]:
    out: list[Case] = []
    for truncation in TRUNCATIONS:
        for offset in OFFSETS if truncation != "none" else (0.0,):
            for kind, ends in (
                ("exact", (z, z)),
                ("right", (z, z)),
                ("left", (z, z)),
            ):
                window = _windows(family, parameters, ends[0], ends[1], truncation, offset)
                if window is None:
                    continue
                out.append(
                    Case(
                        family,
                        parameters,
                        kind,
                        point,
                        None,
                        None if truncation == "none" else window,
                        truncation,
                    )
                )
            for width in WIDTHS:
                upper = _x(family, parameters, z + width)
                window = _windows(family, parameters, z, z + width, truncation, offset)
                if upper is None or window is None or not point < upper:
                    continue
                out.append(
                    Case(
                        family,
                        parameters,
                        "interval",
                        point,
                        upper,
                        None if truncation == "none" else window,
                        truncation,
                    )
                )
    # A censoring point at a window end: the observation carries all of the window.
    out.append(Case(family, parameters, "right", point, None, (point, math.inf), "left"))
    out.append(Case(family, parameters, "left", point, None, (-math.inf, point), "right"))
    return out


def lower_only(family: str) -> list[Case]:
    """The exact and right-censored cases with a lower window end only (the closed-ratio rows)."""

    return [
        case
        for case in cases(family)
        if case.truncation == "left" and case.kind in ("exact", "right") and case.window
    ]


def by_cell(family: str, per_cell: int) -> list[Case]:
    """``per_cell`` evenly spaced cases of every (kind, truncation) cell of ``family``."""

    every = cases(family)
    chosen: list[Case] = []
    for kind in KINDS:
        for truncation in TRUNCATIONS:
            cell = [c for c in every if c.kind == kind and c.truncation == truncation]
            chosen.extend(stable_sample(cell, per_cell))
    return chosen


def stable_sample(items: list[Any], count: int) -> list[Any]:
    """``count`` evenly spaced items (all of them when fewer)."""

    if len(items) <= count:
        return items
    step = len(items) / count
    return [items[int(i * step)] for i in range(count)]
