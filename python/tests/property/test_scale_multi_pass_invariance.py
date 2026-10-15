"""A multi-pass scale fit does not depend on how the data are batched, split, ordered or stored.

For ``weibull_min`` and ``gumbel_right`` the result bits (estimates, likelihood and the Wald
uncertainty), the canonical state digest, the pass count and the pass record are identical
across batch caps, Parquet row group layouts, file splits, row order shuffles and the order
and number of the partitions of a custom source. The data are seeded; nothing is random
between runs.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_fit_oracle import ListSource
from tests.unit.scale_multi_pass_oracle import gumbel_sample, weibull_sample
from veridist.scale._arrow import ParquetSource
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._sources import NumpySource

SIZE = 120
FAMILIES = ("weibull_min", "gumbel_right")


def sample(family: str, seed: int, size: int = SIZE) -> np.ndarray:
    if family == "weibull_min":
        return weibull_sample(1.7, 1000.0, size, seed)
    return gumbel_sample(5.0, 2.0, size, seed)


def bits(result: ScaleFitResult) -> tuple[Any, ...]:
    """Every number a result reports, as exact binary64 patterns."""

    success: Any = result.fit

    def pack(value: float) -> bytes:
        return struct.pack("<d", value)

    numbers = [pack(value) for value in success.parameters.values()]
    numbers.append(pack(success.log_likelihood))
    uncertainty = success.uncertainty()
    for row in uncertainty.covariance:
        numbers.extend(pack(value) for value in row)
    for lower, upper in uncertainty.confidence_intervals(0.95, "wald").values():
        numbers.extend((pack(lower), pack(upper)))
    mean = uncertainty.mean(0.9)
    numbers.extend(pack(value) for value in (mean.estimate, mean.lower, mean.upper))
    return tuple(numbers)


def state_digest(result: ScaleFitResult) -> str:
    assert result.scale.state is not None
    return result.scale.state.digest_sha256


def write_layout(
    root: Path, values: np.ndarray, sizes: list[int], group: int | None
) -> ParquetSource:
    assert sum(sizes) == len(values)
    start = 0
    for index, size in enumerate(sizes):
        write_parquet(
            root / f"d{index % 3}" / f"f{index:02d}.parquet",
            {"x": values[start : start + size]},
            row_group_size=group,
        )
        start += size
    return ParquetSource(root, "x")


class InvarianceTests(unittest.TestCase):
    def assertSame(
        self, result: ScaleFitResult, base: ScaleFitResult, family: str, label: str
    ) -> None:
        with self.subTest(family=family, layout=label):
            self.assertEqual(result.fit, base.fit)
            self.assertEqual(bits(result), bits(base))
            self.assertEqual(state_digest(result), state_digest(base))
            self.assertEqual(result.scale.passes, base.scale.passes)
            self.assertEqual(result.scale.row_count, SIZE)

    def test_batch_caps_and_row_orders_of_the_weibull_fit(self) -> None:
        self.batch_caps_and_row_orders("weibull_min")

    def test_batch_caps_and_row_orders_of_the_gumbel_fit(self) -> None:
        self.batch_caps_and_row_orders("gumbel_right")

    def batch_caps_and_row_orders(self, family: str) -> None:
        values = sample(family, 31)
        base = fit(NumpySource(values), family)
        for max_rows in (7, 119, 120, 121, 100000):
            self.assertSame(
                fit(NumpySource(values), family, max_rows=max_rows),
                base,
                family,
                f"max_rows={max_rows}",
            )
        rng = np.random.default_rng(97)
        shuffled = values[rng.permutation(SIZE)]
        self.assertSame(fit(NumpySource(shuffled), family, max_rows=89), base, family, "shuffle")
        self.assertSame(fit(NumpySource(values[::-1].copy()), family), base, family, "reversed")

    def test_one_row_batches(self) -> None:
        for family in FAMILIES:
            values = sample(family, 34, 14)
            base = fit(NumpySource(values), family)
            result = fit(NumpySource(values), family, max_rows=1)
            self.assertEqual(result.fit, base.fit)
            self.assertEqual(bits(result), bits(base))
            self.assertEqual(state_digest(result), state_digest(base))

    def test_parquet_layouts_and_file_splits_of_the_weibull_fit(self) -> None:
        self.parquet_layouts("weibull_min")

    def test_parquet_layouts_and_file_splits_of_the_gumbel_fit(self) -> None:
        self.parquet_layouts("gumbel_right")

    def parquet_layouts(self, family: str) -> None:
        shuffled_order = np.random.default_rng(5).permutation(SIZE)
        with tempfile.TemporaryDirectory() as directory:
            values = sample(family, 33)
            base = fit(NumpySource(values), family)
            root = Path(directory) / family
            uneven = write_layout(root / "uneven", values, [70, 30, 20], 25)
            self.assertSame(fit(uneven, family, max_rows=13), base, family, "three files")
            shuffled = write_layout(root / "shuffled", values[shuffled_order], [41, 30, 49], 40)
            self.assertSame(fit(shuffled, family), base, family, "shuffled over three files")

    def test_partition_splits_and_orders_of_a_custom_source(self) -> None:
        rng = np.random.default_rng(55)
        for family in FAMILIES:
            values = sample(family, 32)
            base = fit(NumpySource(values), family)
            for trial in range(2):
                count = int(rng.integers(1, 9))
                cuts = np.sort(rng.choice(np.arange(1, SIZE), size=count, replace=False))
                parts = np.split(values[rng.permutation(SIZE)], cuts)
                source = ListSource([parts[i] for i in rng.permutation(len(parts))])
                result = fit(source, family, max_rows=int(rng.integers(1, 120)))
                self.assertSame(result, base, family, f"trial {trial}, {len(parts)} partitions")
                self.assertEqual(result.scale.partition_count, len(parts))

    def test_the_expanded_case_is_invariant_too(self) -> None:
        rng = np.random.default_rng(71)
        values = 1000.0 * (1.0 + 2e-8 * rng.standard_normal(200))
        base = fit(NumpySource(values), "weibull_min")
        assert base.scale.passes is not None
        self.assertGreaterEqual(
            sum(item.role == "expansion" for item in base.scale.passes.records), 1
        )
        parts = np.split(values[rng.permutation(200)], [17, 60, 131])
        result = fit(ListSource(parts), "weibull_min", max_rows=23)
        self.assertEqual(result.fit, base.fit)
        self.assertEqual(state_digest(result), state_digest(base))
        self.assertEqual(result.scale.passes, base.scale.passes)


class DigestSensitivityTests(unittest.TestCase):
    def test_different_data_give_a_different_digest(self) -> None:
        for family in FAMILIES:
            values = sample(family, 1, 100)
            changed = values.copy()
            changed[3] = changed[3] * (1.0 + 1e-9) + 1e-9
            self.assertNotEqual(
                state_digest(fit(NumpySource(values), family)),
                state_digest(fit(NumpySource(changed), family)),
            )

    def test_the_digest_differs_between_families(self) -> None:
        values = weibull_sample(1.7, 1000.0, 100, 1)
        self.assertNotEqual(
            state_digest(fit(NumpySource(values), "weibull_min")),
            state_digest(fit(NumpySource(values), "gumbel_right")),
        )


if __name__ == "__main__":
    unittest.main()
