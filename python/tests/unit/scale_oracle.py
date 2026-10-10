"""Independent exact oracles and shared fixtures for the mergeable-state tests.

Nothing here uses the accumulator under test: sums are formed with Python
integers (and ``Fraction``) from ``float.as_integer_ratio``, the construction the
scalar log-likelihood reducer uses, so a disagreement is a defect in one of them.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence
from fractions import Fraction

import numpy as np

from veridist.scale._one_pass import ExponentialState, GammaState, LognormalState, NormalState

STATES = (ExponentialState, NormalState, GammaState, LognormalState)

UNITS = 1 << 1074
MAX_FLOAT = float.fromhex("0x1.fffffffffffffp+1023")
MAX_MANTISSA = float(np.nextafter(2.0, 0.0))  # 0x1.fffffffffffffp+0: all 53 mantissa bits set


def exact_units(values: Iterable[float]) -> int:
    """The exact sum of float values in units of 2**-1074, by integer arithmetic."""

    total = 0
    for value in values:
        numerator, denominator = float(value).as_integer_ratio()
        total += numerator * (UNITS // denominator)
    return total


def exact_fraction(values: Iterable[float]) -> Fraction:
    """The exact sum of float values as a ``Fraction``."""

    total = Fraction(0)
    for value in values:
        total += Fraction(float(value))
    return total


def correctly_rounded(units: int) -> float:
    """The nearest binary64 (ties to even) of ``units * 2**-1074`` through ``Fraction``."""

    return float(Fraction(units, UNITS))


def bits(value: float) -> int:
    return int(np.float64(value).view(np.int64))


def as_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def adversarial_sets() -> dict[str, np.ndarray]:
    """Small deterministic data sets that defeat naive and compensated summation."""

    sets: dict[str, np.ndarray] = {}
    for seed in (1, 2, 3):
        rng = np.random.default_rng(seed)
        sets[f"mixed_magnitude_sign_{seed}"] = rng.standard_normal(1500) * np.exp2(
            rng.uniform(-300.0, 300.0, 1500)
        )
    rng = np.random.default_rng(7)
    big = rng.standard_normal(400) * 1e16
    sets["cancellation"] = np.concatenate([big, -big, rng.standard_normal(40)])
    sets["cancellation_to_zero"] = np.concatenate([big, -big])
    rng = np.random.default_rng(8)
    sets["subnormals"] = rng.integers(-(2**20), 2**20, 1500).astype(np.float64) * 5e-324
    sets["ones_plus_tiny"] = np.concatenate([np.ones(500), np.full(500, 1e-17)])
    sets["max_mantissa_repeated"] = np.full(3000, MAX_MANTISSA)
    sets["max_mantissa_alternating"] = np.tile([MAX_MANTISSA, -MAX_MANTISSA * 0.5], 700)
    sets["every_power_of_two"] = np.array(
        [math.ldexp(1.0, exponent) for exponent in range(-1074, 1024)]
        + [-math.ldexp(1.0, exponent) for exponent in range(-1074, 1000)]
    )
    sets["near_maximum_cancelling"] = np.array(
        [MAX_FLOAT, -MAX_FLOAT, MAX_FLOAT * 0.5, MAX_FLOAT * 0.25, -MAX_FLOAT * 0.75, 1.0]
    )
    sets["signed_zeros"] = np.array([0.0, -0.0, 0.0, -0.0, 3.0, -0.0])
    sets["tie_to_even_up"] = np.array([1.0 + 2.0**-52, 2.0**-53])
    sets["tie_to_even_down"] = np.array([1.0, 2.0**-53])
    sets["just_above_tie"] = np.array([1.0, 2.0**-53, 2.0**-106])
    return sets


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def random_chunk_sizes(rng: np.random.Generator, total: int, largest: int) -> list[int]:
    """Random positive chunk sizes that sum to ``total`` (bounded by ``total`` draws)."""

    sizes: list[int] = []
    remaining = total
    for _ in range(total):
        if remaining == 0:
            break
        size = min(int(rng.integers(1, largest + 1)), remaining)
        sizes.append(size)
        remaining -= size
    return sizes


def batch(cls: type, rng: np.random.Generator, size: int) -> np.ndarray:
    """A valid random batch for a state type, spanning many binary exponents."""

    if cls is ExponentialState:
        return np.abs(rng.standard_normal(size)) * np.exp2(rng.uniform(-30, 30, size))
    if cls is NormalState:
        return rng.standard_normal(size) * np.exp2(rng.uniform(-100, 100, size))
    return np.exp2(rng.uniform(-300, 300, size)) * rng.uniform(0.5, 2.0, size)


def expected_sums(cls: type, values: np.ndarray) -> tuple[int, ...]:
    """The exact sums a state must hold, from independent integer arithmetic."""

    if cls is ExponentialState:
        return (exact_units(values.tolist()),)
    if cls is NormalState:
        return (exact_units(values.tolist()), exact_units((values * values).tolist()))
    logs = np.log(values)
    if cls is GammaState:
        return (exact_units(values.tolist()), exact_units(logs.tolist()))
    return (exact_units(logs.tolist()), exact_units((logs * logs).tolist()))


def seal(body: bytes) -> bytes:
    """Append the SHA-256 checksum that closes a state frame."""

    return body + hashlib.sha256(body).digest()


def frame_body(
    tag: bytes = b"scale.exact_sum",
    version: int = 1,
    count: int = 2,
    integers: tuple[tuple[int, bytes], ...] = ((0, b"\x07"),),
    declared: int | None = None,
    extra: bytes = b"",
    magic: bytes = b"VDSS",
) -> bytes:
    """An unsealed frame body built by hand, so tests can break it in a chosen way."""

    body = magic + bytes([len(tag)]) + tag + version.to_bytes(2, "big") + count.to_bytes(8, "big")
    body += bytes([len(integers) if declared is None else declared])
    for sign, magnitude in integers:
        body += bytes([sign]) + len(magnitude).to_bytes(4, "big") + magnitude
    return body + extra
