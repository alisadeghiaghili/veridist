"""A scale fit does not depend on how the data are batched, split, ordered or stored.

For every family the result bits (point estimates, likelihood and the uncertainty it
reports) and the canonical state digest are identical across batch caps, Parquet row
group layouts, file splits, row order shuffles and the order and number of the
partitions of a custom source. The data are seeded; nothing is random between runs.
"""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_fit_oracle import FAMILIES, ListSource, dataset
from veridist.families.registry import FamilyId
from veridist.scale._arrow import ParquetSource
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._sources import NumpySource

SIZE = 500


def bits(result: ScaleFitResult) -> tuple[Any, ...]:
    """Every number a result reports, as exact binary64 patterns."""

    success = result.fit

    def pack(value: float) -> bytes:
        return struct.pack("<d", value)

    numbers = [pack(value) for value in success.parameters.values()]  # type: ignore[union-attr]
    numbers.append(pack(success.log_likelihood))  # type: ignore[union-attr]
    uncertainty = success.uncertainty()  # type: ignore[union-attr]
    for row in uncertainty.covariance:
        numbers.extend(pack(value) for value in row)
    for method in ("wald", "profile"):
        for lower, upper in uncertainty.confidence_intervals(0.95, method).values():
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
    def reference(self, family: FamilyId, seed: int) -> tuple[np.ndarray, ScaleFitResult]:
        values = dataset(family, SIZE, seed)
        return values, fit(NumpySource(values), family)

    def assertSame(
        self, result: ScaleFitResult, base: ScaleFitResult, family: FamilyId, label: str
    ) -> None:
        with self.subTest(family=family.value, layout=label):
            self.assertEqual(result.fit, base.fit)
            self.assertEqual(bits(result), bits(base))
            self.assertEqual(state_digest(result), state_digest(base))
            self.assertEqual(result.scale.row_count, SIZE)

    def test_batch_caps_and_row_orders(self) -> None:
        for family in FAMILIES:
            values, base = self.reference(family, 31)
            for max_rows in (3, 7, 499, 500, 501, 100000):
                result = fit(NumpySource(values), family, max_rows=max_rows)
                self.assertSame(result, base, family, f"max_rows={max_rows}")
            rng = np.random.default_rng(97)
            for index in range(2):
                shuffled = values[rng.permutation(SIZE)]
                result = fit(NumpySource(shuffled), family, max_rows=89)
                self.assertSame(result, base, family, f"shuffle {index}")
            result = fit(NumpySource(values[::-1].copy()), family)
            self.assertSame(result, base, family, "reversed")

    def test_one_row_batches(self) -> None:
        for family in FAMILIES:
            values = dataset(family, 60, 34)
            base = fit(NumpySource(values), family)
            result = fit(NumpySource(values), family, max_rows=1)
            self.assertEqual(result.fit, base.fit)
            self.assertEqual(bits(result), bits(base))
            self.assertEqual(state_digest(result), state_digest(base))

    def test_parquet_layouts_and_file_splits(self) -> None:
        shuffled_order = np.random.default_rng(5).permutation(SIZE)
        with tempfile.TemporaryDirectory() as directory:
            for family in (FamilyId.EXPONENTIAL, FamilyId.GAMMA):
                values, base = self.reference(family, 33)
                root = Path(directory) / family.value
                source = write_layout(root / "uneven", values, [300, 150, 50], 64)
                self.assertSame(fit(source, family, max_rows=33), base, family, "three files")
                source = write_layout(
                    root / "shuffled", values[shuffled_order], [123, 200, 177], 125
                )
                self.assertSame(fit(source, family), base, family, "shuffled over three files")

    def test_partition_splits_and_orders_of_a_custom_source(self) -> None:
        rng = np.random.default_rng(55)
        for family in FAMILIES:
            values, base = self.reference(family, 32)
            for trial in range(4):
                count = int(rng.integers(1, 9))
                cuts = np.sort(rng.choice(np.arange(1, SIZE), size=count, replace=False))
                parts = np.split(values[rng.permutation(SIZE)], cuts)
                source = ListSource([parts[i] for i in rng.permutation(len(parts))])
                result = fit(source, family, max_rows=int(rng.integers(1, 120)))
                self.assertSame(result, base, family, f"trial {trial}, {len(parts)} partitions")
                self.assertEqual(result.scale.partition_count, len(parts))

    def test_extreme_conditioning_and_magnitudes_are_invariant_too(self) -> None:
        rng = np.random.default_rng(71)
        for mean, spread in ((1e9, 1.0), (1e200, 1e199), (1e-200, 1e-201)):
            values = rng.normal(mean, spread, 300)
            base = fit(NumpySource(values), "normal")
            for trial in range(3):
                parts = np.split(
                    values[rng.permutation(300)], np.sort(rng.choice(299, 4, replace=False) + 1)
                )
                result = fit(ListSource(parts), "normal", max_rows=int(rng.integers(1, 90)))
                with self.subTest(mean=mean, trial=trial):
                    self.assertEqual(result.fit, base.fit)
                    self.assertEqual(state_digest(result), state_digest(base))

    def test_every_family_is_covered(self) -> None:
        self.assertEqual(
            set(FAMILIES),
            {FamilyId.EXPONENTIAL, FamilyId.NORMAL, FamilyId.GAMMA, FamilyId.LOGNORMAL},
        )


class DigestSensitivityTests(unittest.TestCase):
    def test_different_data_give_a_different_digest(self) -> None:
        values = dataset(FamilyId.NORMAL, 100, 1)
        changed = values.copy()
        changed[3] = np.nextafter(changed[3], np.inf)
        self.assertNotEqual(
            state_digest(fit(NumpySource(values), "normal")),
            state_digest(fit(NumpySource(changed), "normal")),
        )

    def test_the_digest_differs_between_families(self) -> None:
        values = dataset(FamilyId.GAMMA, 100, 1)
        digests = {state_digest(fit(NumpySource(values), family)) for family in FAMILIES}
        self.assertEqual(len(digests), 4)


if __name__ == "__main__":
    unittest.main()
