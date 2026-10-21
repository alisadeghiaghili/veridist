"""Millions of rows through a single-use stream and a spool give the Parquet fit, bit for bit.

The stream is an Arrow reader, which can be read once; a multi-pass family reads it through a
local spool. The fit must be the one of the same values in a Parquet file (which is read again
for every pass): the same result bits, state digest and pass record. The source is pulled
exactly once, the spool holds the digest of the values, and it is gone afterwards. The data are
seeded; nothing is random between runs.
"""

from __future__ import annotations

import hashlib
import struct
import tempfile
import unittest
from collections.abc import Iterator
from math import ceil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import write_parquet
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._spool import ScaleSpoolPolicy

SEED = 20261016
BATCH = 262144
SCHEMA = pa.schema([pa.field("x", pa.float64())])


def result_bits(result: ScaleFitResult) -> tuple[bytes, ...]:
    success: Any = result.fit
    numbers = [*success.parameters.values(), success.log_likelihood]
    uncertainty = success.uncertainty()
    for row in uncertainty.covariance:
        numbers.extend(row)
    for lower, upper in uncertainty.confidence_intervals(0.95, "wald").values():
        numbers.extend((lower, upper))
    return tuple(struct.pack("<d", float(value)) for value in numbers)


class StreamSpoolSweepTests(unittest.TestCase):
    def check(self, family: str, values: np.ndarray, truth: dict[str, float]) -> None:
        size = len(values)
        with tempfile.TemporaryDirectory() as holder:
            root = Path(holder)
            write_parquet(root / "data" / "a.parquet", {"x": values}, row_group_size=1 << 20)
            spools = root / "spools"
            spools.mkdir()
            pulled: list[int] = []

            def generate() -> Iterator[Any]:
                for start in range(0, size, BATCH):
                    part = values[start : start + BATCH]
                    pulled.append(len(part))
                    yield pa.record_batch([pa.array(part)], names=["x"])

            source = ArrowSource(pa.RecordBatchReader.from_batches(SCHEMA, generate()), "x")
            with self.assertRaises(ScaleSourceError) as refused:
                fit(source, family)
            self.assertIs(refused.exception.code, ScaleSourceErrorCode.SPOOL_REQUIRED)
            self.assertEqual(pulled, [])

            reference = fit(ParquetSource(root / "data", "x"), family)
            streamed = fit(
                source, family, spool=ScaleSpoolPolicy(directory=spools, max_bytes=size * 8 + 4096)
            )

            self.assertEqual(sum(pulled), size)
            self.assertEqual(len(pulled), ceil(size / BATCH))
            self.assertEqual(result_bits(streamed), result_bits(reference))
            self.assertEqual(streamed.scale.state, reference.scale.state)
            self.assertEqual(streamed.scale.passes, reference.scale.passes)
            assert streamed.scale.passes is not None
            self.assertGreaterEqual(streamed.scale.passes.pass_count, 3)
            self.assertEqual(streamed.scale.row_count, size)
            assert streamed.scale.spool is not None
            self.assertEqual(streamed.scale.spool.row_count, size)
            self.assertEqual(
                streamed.scale.spool.partitions[0].digest_sha256,
                hashlib.sha256(values.astype("<f8").tobytes()).hexdigest(),
            )
            self.assertEqual(list(spools.iterdir()), [])
            success: Any = streamed.fit
            for name, expected in truth.items():
                self.assertAlmostEqual(success.parameters[name] / expected, 1.0, delta=0.01)

    def test_three_million_weibull_rows(self) -> None:
        values = 1000.0 * np.random.default_rng(SEED).weibull(1.7, 3_000_000)
        self.check("weibull_min", values, {"shape": 1.7, "scale": 1000.0})

    def test_a_million_gumbel_rows(self) -> None:
        values = np.random.default_rng(SEED + 1).gumbel(50.0, 4.0, 1_000_000)
        self.check("gumbel_right", values, {"location": 50.0, "scale": 4.0})


if __name__ == "__main__":
    unittest.main()
