"""Inference from a scale fit does not depend on how the data are batched, split, ordered or read.

For ``weibull_min`` and ``gumbel_right`` the profile-likelihood intervals (with every number of
the result and the record of the passes) and, for every family, the large-n goodness-of-fit
report are bit for bit the same across batch caps, partition layouts, row order shuffles, the
order of the partitions of a custom source, a thread executor and a local spool. The data are
seeded; nothing is random between runs.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from tests.unit.scale_executor_fixtures import NoLeaksTestCase
from tests.unit.scale_fit_oracle import ListSource
from tests.unit.scale_inference_fixtures import gof_bits, profile_bits
from tests.unit.scale_inference_oracle import gumbel_data, weibull_data
from veridist.scale import _gof as gof
from veridist.scale import _profile_intervals as pi
from veridist.scale._executors import SerialExecutor, ThreadExecutor
from veridist.scale._fit import fit
from veridist.scale._sources import NumpySource
from veridist.scale._spool import SpoolPolicy

SIZE = 42
"""Rows of the data: small enough for the run of every variant to take milliseconds. The same
properties at 900 rows, with the three derived quantities, are in
``tests/statistical/test_scale_inference_full_size.py``."""

QUANTITIES: tuple[pi.Quantity, ...] = (pi.survival(30.0),)
SWEEP_QUANTITIES: tuple[pi.Quantity, ...] = ()
"""The search for the extremes of a quantity costs more than the rest of a run, so the sweep of
the layouts asks for none and ``QUANTITIES`` is asked for once, in the test of the fits."""


CORE = frozenset(
    {
        "batches of 1",
        "shuffled rows",
        "rows per partition a tenth",
        "custom partitions shuffled",
        "threads",
    }
)
"""The variants run by the collected tests; the statistical tests run every one."""


def data(family: str, size: int = SIZE) -> np.ndarray:
    if family == "weibull_min":
        return weibull_data(1.7, 40.0, size, 21)
    if family == "gumbel_right":
        return gumbel_data(5.0, 2.0, size, 21)
    if family == "gamma":
        return np.random.default_rng(21).gamma(3.0, 2.0, size)
    return np.random.default_rng(21).normal(10.0, 3.0, size)


def variants(
    values: np.ndarray, *, singletons: bool = True
) -> list[tuple[str, Any, dict[str, Any]]]:
    """Sources and options that must not change a result: ``(label, source, options)``.

    The layouts are proportional to the number of rows: seven custom partitions of about a
    seventh, partitions of a tenth and of a half, batches of 1, 7 and a third. A partition of
    one row (``singletons``) costs a pass over every row as a partition of its own.
    """

    size = len(values)
    order = np.random.default_rng(5).permutation(size)
    piece = -(-size // 7)
    tenth, half = max(1, size // 10), size // 2
    pieces = [values[i : i + piece] for i in range(0, size, piece)]
    assert len(pieces) == 7
    return [
        ("batches of 1", NumpySource(values), {"max_rows": 1}),
        ("batches of 7", NumpySource(values), {"max_rows": 7}),
        ("batches of a third", NumpySource(values), {"max_rows": size // 3}),
        ("shuffled rows", NumpySource(values[order]), {}),
        *(
            [("rows per partition 1", NumpySource(values, partition_rows=1), {"max_rows": 5})]
            if singletons
            else []
        ),
        ("rows per partition a tenth", NumpySource(values, partition_rows=tenth), {}),
        ("rows per partition a half", NumpySource(values, partition_rows=half), {}),
        ("custom partitions reversed", ListSource(pieces[::-1]), {}),
        ("custom partitions shuffled", ListSource([pieces[i] for i in (3, 0, 6, 1, 5, 2, 4)]), {}),
        ("serial executor", NumpySource(values, partition_rows=tenth), {"executor": "serial"}),
        ("threads", NumpySource(values, partition_rows=tenth), {"executor": ThreadExecutor(3)}),
        ("one thread", NumpySource(values, partition_rows=tenth), {"executor": ThreadExecutor(1)}),
    ]


class ProfileInvarianceTests(NoLeaksTestCase):
    SIZE = SIZE
    QUANTITIES = QUANTITIES
    SWEEP_QUANTITIES = SWEEP_QUANTITIES
    SINGLETONS = False
    LABELS: frozenset[str] | None = CORE

    def check_layouts(self, family: str) -> None:
        values = data(family, self.SIZE)
        quantities = self.SWEEP_QUANTITIES if family == "weibull_min" else ()
        result = fit(NumpySource(values), family)
        base = pi.profile_intervals(NumpySource(values), result, quantities=quantities)
        for label, source, options in variants(values, singletons=self.SINGLETONS):
            if self.LABELS is not None and label not in self.LABELS:
                continue
            with self.subTest(family=family, variant=label):
                out = pi.profile_intervals(source, result, quantities=quantities, **options)
                self.assertEqual(profile_bits(out), profile_bits(base))

    def test_the_weibull_intervals_do_not_depend_on_the_layout_or_the_executor(self) -> None:
        self.check_layouts("weibull_min")

    def test_the_gumbel_intervals_do_not_depend_on_the_layout_or_the_executor(self) -> None:
        self.check_layouts("gumbel_right")

    def test_the_result_of_a_fit_made_in_another_layout_gives_the_same_intervals(self) -> None:
        values = data("weibull_min", self.SIZE)
        base = pi.profile_intervals(
            NumpySource(values), fit(NumpySource(values), "weibull_min"), quantities=self.QUANTITIES
        )
        for source in (
            NumpySource(values, partition_rows=max(1, self.SIZE // 14)),
            ListSource([values[: self.SIZE // 3], values[self.SIZE // 3 :]]),
        ):
            result = fit(source, "weibull_min", max_rows=50)
            out = pi.profile_intervals(source, result, quantities=self.QUANTITIES, max_rows=77)
            self.assertEqual(profile_bits(out), profile_bits(base))

    def test_a_spool_does_not_change_the_intervals(self) -> None:
        for family in ("weibull_min", "gumbel_right"):
            values = data(family, self.SIZE)
            result = fit(NumpySource(values), family)
            base = pi.profile_intervals(NumpySource(values), result)
            with tempfile.TemporaryDirectory() as directory:
                policy = SpoolPolicy(max_bytes=1 << 24, directory=Path(directory))
                out = pi.profile_intervals(
                    NumpySource(values, partition_rows=self.SIZE // 3), result, spool=policy
                )
            self.assertEqual(profile_bits(out), profile_bits(base))
            assert out.scale.spool is not None
            self.assertEqual(out.scale.spool.row_count, self.SIZE)

    def test_a_spool_makes_a_source_that_can_be_read_once_enough(self) -> None:
        values = data("weibull_min", self.SIZE)
        result = fit(NumpySource(values), "weibull_min")
        base = pi.profile_intervals(NumpySource(values), result)

        class Once(NumpySource):
            single_use = True
            reads = 0

            def batches(self, partition: Any, *, max_rows: int = 262144) -> Any:
                type(self).reads += 1
                if type(self).reads > 1:
                    raise AssertionError("a source that can be read once was read twice")
                return super().batches(partition, max_rows=max_rows)

        with tempfile.TemporaryDirectory() as directory:
            policy = SpoolPolicy(max_bytes=1 << 24, directory=Path(directory))
            out = pi.profile_intervals(Once(values), result, spool=policy)
        self.assertEqual(profile_bits(out), profile_bits(base))
        self.assertEqual(Once.reads, 1)


class GofInvarianceTests(NoLeaksTestCase):
    SIZE = SIZE

    def test_the_report_does_not_depend_on_the_layout_or_the_executor(self) -> None:
        for family in ("weibull_min", "gumbel_right", "gamma", "normal"):
            values = data(family, self.SIZE)
            result = fit(NumpySource(values), family)
            base = gof.large_n_gof(NumpySource(values), result, bin_exponent=9, threshold=0.05)
            for label, source, options in variants(values):
                with self.subTest(family=family, variant=label):
                    out = gof.large_n_gof(source, result, bin_exponent=9, threshold=0.05, **options)
                    self.assertEqual(gof_bits(out), gof_bits(base))

    def test_a_serial_executor_object_is_the_default(self) -> None:
        values = data("gamma", self.SIZE)
        result = fit(NumpySource(values), "gamma")
        base = gof.large_n_gof(NumpySource(values), result)
        out = gof.large_n_gof(NumpySource(values), result, executor=SerialExecutor())
        self.assertEqual(gof_bits(out), gof_bits(base))


if __name__ == "__main__":
    unittest.main()
