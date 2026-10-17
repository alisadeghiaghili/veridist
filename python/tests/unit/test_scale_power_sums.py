"""The exact power sums of an inference pass: one formula, exact, mergeable, canonical."""

from __future__ import annotations

import dataclasses
import unittest

import numpy as np

from veridist.scale._accumulator import ExactSum
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._multi_pass import ProfileAccumulator, WeibullKernel
from veridist.scale._power_sums import (
    MAX_POWER_SUM_CANDIDATES,
    POWER_SUM_SCHEMA_VERSION,
    POWER_SUM_TAG,
    PowerSumAccumulator,
    PowerSumJob,
)
from veridist.scale._sources import NumpySource


def centred(size: int = 1000, seed: int = 3) -> np.ndarray:
    return -np.abs(np.random.default_rng(seed).normal(0.0, 4.0, size))


class AccumulatorTests(unittest.TestCase):
    def test_the_sums_are_the_exact_sums_of_the_one_formula_terms(self) -> None:
        values = -np.array([0.0, 0.5, 1.25, 3.0, 0.125])
        accumulator = PowerSumAccumulator(2)
        accumulator.update(values, (0.5, 3.0))
        self.assertEqual(accumulator.count, 5)
        for index, parameter in enumerate((0.5, 3.0)):
            expected = ExactSum()
            expected.update(np.exp(parameter * values))
            self.assertEqual(accumulator.totals()[index], expected.total())

    def test_the_sum_at_a_parameter_is_the_one_the_fit_accumulator_makes(self) -> None:
        values = centred()
        parameters = (0.3, 1.7, 20.0)
        ours = PowerSumAccumulator(3)
        ours.update(values, parameters)
        theirs = ProfileAccumulator(3)
        theirs.update(values, parameters)
        self.assertEqual(ours.totals(), [item.s0 for item in theirs.totals()])

    def test_the_totals_do_not_depend_on_batching_or_merge_order(self) -> None:
        values = centred()
        parameters = (0.3, 1.7, 20.0)
        whole = PowerSumAccumulator(3)
        whole.update(values, parameters)
        pieces = []
        for part in (values[:1], values[1:400], values[400:401], values[401:]):
            piece = PowerSumAccumulator(3)
            piece.update(part, parameters)
            pieces.append(piece)
        forward = pieces[0].merge(pieces[1]).merge(pieces[2]).merge(pieces[3])
        backward = pieces[3].merge(pieces[2]).merge(pieces[1]).merge(pieces[0])
        self.assertEqual(forward.count, 1000)
        self.assertEqual(forward.totals(), whole.totals())
        self.assertEqual(backward.totals(), whole.totals())
        self.assertEqual(forward.to_bytes(), whole.to_bytes())

    def test_a_sum_underflowing_to_nothing_is_zero_and_the_largest_term_is_one(self) -> None:
        accumulator = PowerSumAccumulator(2)
        accumulator.update(np.array([0.0, -1.0]), (1e9, 0.0))
        first, second = accumulator.totals()
        self.assertEqual(first, 1.0)
        self.assertEqual(second, 2.0)

    def test_the_bytes_round_trip_and_are_canonical(self) -> None:
        values = centred(300)
        accumulator = PowerSumAccumulator(3)
        accumulator.update(values, (0.5, 2.0, 9.0))
        data = accumulator.to_bytes()
        restored = PowerSumAccumulator.from_bytes(data, 3)
        self.assertEqual(restored.count, 300)
        self.assertEqual(restored.totals(), accumulator.totals())
        self.assertEqual(restored.to_bytes(), data)
        self.assertEqual(restored.merge(PowerSumAccumulator(3)).to_bytes(), data)

    def test_foreign_or_damaged_bytes_are_refused(self) -> None:
        accumulator = PowerSumAccumulator(2)
        accumulator.update(centred(10), (1.0, 2.0))
        data = accumulator.to_bytes()
        with self.assertRaises(ScaleStateError) as caught:
            PowerSumAccumulator.from_bytes(data[:-1] + bytes([data[-1] ^ 1]), 2)
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.STATE_BYTES_INVALID)
        with self.assertRaises(ScaleStateError) as caught:
            PowerSumAccumulator.from_bytes(data, 3)
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.STATE_BYTES_INVALID)
        other = ProfileAccumulator(2)
        with self.assertRaises(ScaleStateError) as caught:
            PowerSumAccumulator.from_bytes(other.to_bytes(), 2)
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.STATE_TAG_MISMATCH)
        with self.assertRaises(TypeError):
            PowerSumAccumulator.from_bytes("text", 2)

    def test_the_number_of_candidates_is_bounded_by_the_frame(self) -> None:
        self.assertEqual(MAX_POWER_SUM_CANDIDATES, 255)
        self.assertEqual(len(PowerSumAccumulator(255).sums), 255)
        with self.assertRaisesRegex(ValueError, "too many"):
            PowerSumAccumulator(256)
        full = PowerSumAccumulator(255)
        full.update(centred(5), tuple(0.01 * (i + 1) for i in range(255)))
        self.assertEqual(
            PowerSumAccumulator.from_bytes(full.to_bytes(), 255).totals(), full.totals()
        )

    def test_the_format_is_pinned(self) -> None:
        self.assertEqual(POWER_SUM_TAG, "scale.power_sum_accumulator")
        self.assertEqual(POWER_SUM_SCHEMA_VERSION, 1)
        data = PowerSumAccumulator(1).to_bytes()
        tag = POWER_SUM_TAG.encode()
        self.assertEqual(data[4 : 5 + len(tag)], bytes([len(tag)]) + tag)
        self.assertEqual(data[5 + len(tag) : 7 + len(tag)], (1).to_bytes(2, "big"))

    def test_candidates_and_parameters_must_match(self) -> None:
        accumulator = PowerSumAccumulator(2)
        with self.assertRaises(ValueError):
            accumulator.update(centred(5), (1.0,))
        with self.assertRaises(ValueError):
            accumulator.update(centred(5), (1.0, 2.0, 3.0))

    def test_the_value_is_immutable(self) -> None:
        job = PowerSumJob(WeibullKernel(0.0), (1.0,))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            job.parameters = ()  # type: ignore[misc]
        self.assertFalse(hasattr(job, "__dict__"))

    def test_merging_accumulators_of_other_widths_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            PowerSumAccumulator(2).merge(PowerSumAccumulator(3))


class JobTests(unittest.TestCase):
    def test_a_partition_is_folded_with_the_kernel_of_the_fit(self) -> None:
        values = np.random.default_rng(5).uniform(1.0, 9.0, 500)
        source = NumpySource(values, partition_rows=200)
        kernel = WeibullKernel(float(np.log(values.max())))
        job = PowerSumJob(kernel, (0.5, 2.5))
        total = job.identity()
        for partition in source.partitions():
            total = total.merge(job.restore(job.run(source, partition, 77)))
        direct = PowerSumAccumulator(2)
        direct.update(kernel(values), (0.5, 2.5))
        self.assertEqual(total.count, 500)
        self.assertEqual(total.totals(), direct.totals())

    def test_the_checkpoint_key_names_the_pass_exactly(self) -> None:
        kernel = WeibullKernel(1.5)
        key = PowerSumJob(kernel, (0.5, 2.0)).checkpoint_key()
        self.assertEqual(
            key,
            (
                "power_sums",
                POWER_SUM_TAG,
                POWER_SUM_SCHEMA_VERSION,
                "weibull",
                (1.5).hex(),
                ((0.5).hex(), (2.0).hex()),
            ),
        )
        self.assertNotEqual(key, PowerSumJob(kernel, (0.5, 2.000000000000001)).checkpoint_key())
        self.assertNotEqual(key, PowerSumJob(WeibullKernel(1.6), (0.5, 2.0)).checkpoint_key())


if __name__ == "__main__":
    unittest.main()
