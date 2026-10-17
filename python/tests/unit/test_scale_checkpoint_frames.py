"""A Polars frame or plan and a DuckDB relation cannot be recognised again, so they are not
checkpointed: the refusal is typed, comes before the store is touched and reads nothing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import polars as pl

from veridist.scale._duckdb import DuckDBSource
from veridist.scale._errors import ScaleCheckpointError, ScaleCheckpointErrorCode
from veridist.scale._fit import fit
from veridist.scale._polars import PolarsSource

ONE_PASS_AND_MULTI_PASS = ("exponential", "weibull_min")


class FrameRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "job.db"
        self.values = np.random.default_rng(4).exponential(3.0, 300) + 0.1
        connection = duckdb.connect()
        self.addCleanup(connection.close)
        self.connection = connection

    def sources(self) -> dict[str, Any]:
        return {
            "polars frame": PolarsSource(pl.DataFrame({"x": self.values}), "x"),
            "polars plan": PolarsSource(pl.LazyFrame({"x": self.values}), "x"),
            "duckdb relation": DuckDBSource(
                (self.connection, "select unnest(range(1, 301))::double as x"), "x"
            ),
        }

    def test_each_is_refused_for_a_checkpoint_in_every_mode(self) -> None:
        for name, source in self.sources().items():
            for family in ONE_PASS_AND_MULTI_PASS:
                for options in ({}, {"resume": True}):
                    with self.subTest(source=name, family=family, options=options):
                        with self.assertRaises(ScaleCheckpointError) as caught:
                            fit(source, family, checkpoint=self.path, **options)
                        self.assertIs(
                            caught.exception.code,
                            ScaleCheckpointErrorCode.SOURCE_REVISION_UNAVAILABLE,
                        )
                        self.assertFalse(self.path.exists())

    def test_the_same_sources_are_fitted_without_a_checkpoint(self) -> None:
        for name, source in self.sources().items():
            with self.subTest(source=name):
                self.assertEqual(fit(source, "exponential").scale.row_count, 300)
