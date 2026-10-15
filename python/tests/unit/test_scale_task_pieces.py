"""The picklable parts the executors rely on: kernels, profile bytes, jobs, units, frames."""

from __future__ import annotations

import math
import pickle
import unittest
from typing import Any

import duckdb
import numpy as np
import polars as pl

from tests.unit.scale_fit_oracle import ListSource
from veridist.scale import _multi_pass
from veridist.scale._accumulator import MAX_ELEMENT_UNITS, AccumulatorLimits, ExactSum
from veridist.scale._duckdb import DuckDBSource
from veridist.scale._errors import ScaleStateError, ScaleStateErrorCode
from veridist.scale._kernels import logarithms
from veridist.scale._multi_pass import (
    MAX_SERIALIZED_CANDIDATES,
    GumbelKernel,
    ProfileAccumulator,
    ProfileJob,
    WeibullKernel,
)
from veridist.scale._pilot import WeibullPilotState
from veridist.scale._polars import PolarsSource
from veridist.scale._sources import DowngradeReason, ExecutorKind, is_retryable


def accumulate(candidates: tuple[float, ...], *batches: np.ndarray) -> ProfileAccumulator:
    accumulator = ProfileAccumulator(len(candidates))
    for batch in batches:
        accumulator.update(-np.abs(batch), candidates)
    return accumulator


def triples(accumulator: ProfileAccumulator) -> list[tuple[int, int, int]]:
    return [
        tuple(item.total_units() for item in candidate)  # type: ignore[misc]
        for candidate in accumulator.sums
    ]


class FromUnitsTests(unittest.TestCase):
    def test_the_accumulator_holds_the_count_and_the_exact_total(self) -> None:
        for units in (0, 5, -7, 3 * MAX_ELEMENT_UNITS, -3 * MAX_ELEMENT_UNITS):
            restored = ExactSum.from_units(3, units)
            with self.subTest(units=units):
                self.assertEqual(restored.count, 3)
                self.assertEqual(restored.total_units(), units)

    def test_a_total_beyond_what_the_count_allows_is_refused(self) -> None:
        for count, units in (
            (3, 3 * MAX_ELEMENT_UNITS + 1),
            (3, -3 * MAX_ELEMENT_UNITS - 1),
            (0, 1),
        ):
            with self.subTest(count=count, units=units):
                with self.assertRaises(ScaleStateError) as caught:
                    ExactSum.from_units(count, units)
                self.assertEqual(caught.exception.code, ScaleStateErrorCode.STATE_BYTES_INVALID)

    def test_it_merges_and_serializes_like_any_other_accumulator(self) -> None:
        values = np.array([1.5, 2.25, -0.75])
        direct = ExactSum()
        direct.update(values)
        restored = ExactSum.from_units(3, direct.total_units())
        self.assertEqual(restored, direct)
        self.assertEqual(restored.to_bytes(), direct.to_bytes())
        more = ExactSum()
        more.update(np.array([4.0]))
        self.assertEqual(restored.merge(more).total_units(), direct.merge(more).total_units())
        self.assertEqual(ExactSum.from_bytes(direct.to_bytes()), direct)

    def test_the_limits_are_kept(self) -> None:
        limits = AccumulatorLimits(block_size=2, carry_interval=8, lane_capacity=16)
        self.assertEqual(ExactSum.from_units(1, 4, limits).limits, limits)
        self.assertEqual(ExactSum.from_units(1, 4).limits, ExactSum().limits)


class KernelTests(unittest.TestCase):
    def test_the_weibull_kernel_centres_the_logarithms(self) -> None:
        x = np.array([0.5, 1.0, 7.5, 100.0])
        peak = float(np.log(100.0))
        result = WeibullKernel(peak)(x)
        self.assertEqual(result.tolist(), (np.log(x) - peak).tolist())
        self.assertEqual(result.tolist(), (logarithms(x) - peak).tolist())
        self.assertEqual(float(result[-1]), 0.0)

    def test_the_gumbel_kernel_standardises_and_flips(self) -> None:
        x = np.array([-3.0, 0.0, 2.5, 9.0])
        center, peak = 1.5, 7.5
        minimum = (-3.0 - center) / peak
        result = GumbelKernel(minimum, center, peak)(x)
        self.assertEqual(result.tolist(), (minimum - (x - center) / peak).tolist())
        self.assertEqual(float(result[0]), 0.0)
        self.assertTrue((result <= 0.0).all())

    def test_the_kernel_fields_matter(self) -> None:
        x = np.array([1.0, 2.0])
        self.assertNotEqual(WeibullKernel(0.5)(x).tolist(), WeibullKernel(0.6)(x).tolist())
        base = GumbelKernel(-1.0, 0.5, 2.0)(x).tolist()
        self.assertNotEqual(base, GumbelKernel(-1.5, 0.5, 2.0)(x).tolist())
        self.assertNotEqual(base, GumbelKernel(-1.0, 0.75, 2.0)(x).tolist())
        self.assertNotEqual(base, GumbelKernel(-1.0, 0.5, 4.0)(x).tolist())

    def test_kernels_pickle_and_keep_every_bit(self) -> None:
        x = np.random.default_rng(1).uniform(0.1, 9.0, 50)
        for kernel in (WeibullKernel(math.log(9.0)), GumbelKernel(-0.3, 4.2, 5.1)):
            copy = pickle.loads(pickle.dumps(kernel))
            with self.subTest(kernel=type(kernel).__name__):
                self.assertEqual(copy, kernel)
                self.assertEqual(copy(x).tobytes(), kernel(x).tobytes())


class ProfileBytesTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.batches = [rng.uniform(0.0, 5.0, 40), rng.uniform(0.0, 5.0, 7)]
        self.candidates = (0.5, 1.7, 30.0)

    def test_the_bytes_restore_the_same_sums_and_count(self) -> None:
        accumulator = accumulate(self.candidates, *self.batches)
        restored = ProfileAccumulator.from_bytes(accumulator.to_bytes(), 3)
        self.assertEqual(restored.count, 47)
        self.assertEqual(triples(restored), triples(accumulator))
        self.assertEqual(restored.totals(), accumulator.totals())
        self.assertEqual(restored.to_bytes(), accumulator.to_bytes())

    def test_the_bytes_depend_on_the_rows_and_not_on_how_they_were_added(self) -> None:
        whole = accumulate(self.candidates, np.concatenate(self.batches))
        split = accumulate(self.candidates, *self.batches)
        parts = accumulate(self.candidates, self.batches[0]).merge(
            accumulate(self.candidates, self.batches[1])
        )
        self.assertEqual(whole.to_bytes(), split.to_bytes())
        self.assertEqual(whole.to_bytes(), parts.to_bytes())
        other = accumulate(self.candidates, self.batches[0])
        self.assertNotEqual(whole.to_bytes(), other.to_bytes())

    def test_a_restored_accumulator_merges_like_the_original(self) -> None:
        left = accumulate(self.candidates, self.batches[0])
        right = accumulate(self.candidates, self.batches[1])
        merged = ProfileAccumulator.from_bytes(left.to_bytes(), 3).merge(
            ProfileAccumulator.from_bytes(right.to_bytes(), 3)
        )
        self.assertEqual(merged.to_bytes(), left.merge(right).to_bytes())
        self.assertEqual(merged.count, 47)

    def test_an_empty_accumulator_round_trips(self) -> None:
        empty = ProfileAccumulator(2)
        restored = ProfileAccumulator.from_bytes(empty.to_bytes(), 2)
        self.assertEqual(restored.count, 0)
        self.assertEqual(triples(restored), [(0, 0, 0)] * 2)

    def test_the_candidate_count_must_match_the_bytes(self) -> None:
        data = accumulate(self.candidates, *self.batches).to_bytes()
        for wrong in (1, 2, 4):
            with self.subTest(candidates=wrong), self.assertRaises(ScaleStateError) as caught:
                ProfileAccumulator.from_bytes(data, wrong)
            self.assertEqual(caught.exception.code, ScaleStateErrorCode.STATE_BYTES_INVALID)

    def test_altered_or_foreign_bytes_are_refused(self) -> None:
        data = accumulate(self.candidates, *self.batches).to_bytes()
        flipped = bytearray(data)
        flipped[-40] ^= 1
        with self.assertRaises(ScaleStateError) as caught:
            ProfileAccumulator.from_bytes(bytes(flipped), 3)
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.STATE_BYTES_INVALID)
        pilot = WeibullPilotState.empty().to_bytes()
        with self.assertRaises(ScaleStateError) as caught:
            ProfileAccumulator.from_bytes(pilot, 3)
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.STATE_TAG_MISMATCH)
        with self.assertRaises(TypeError):
            ProfileAccumulator.from_bytes("text", 3)

    def test_the_tag_and_version_are_pinned(self) -> None:
        self.assertEqual(_multi_pass.PROFILE_TAG, "scale.profile_accumulator")
        self.assertEqual(_multi_pass.PROFILE_SCHEMA_VERSION, 1)
        data = ProfileAccumulator(1).to_bytes()
        self.assertEqual(data[:4], b"VDSS")
        self.assertEqual(
            data[5 : 5 + len("scale.profile_accumulator")], b"scale.profile_accumulator"
        )

    def test_a_frame_holds_at_most_eighty_five_candidates(self) -> None:
        self.assertEqual(MAX_SERIALIZED_CANDIDATES, 85)
        full = ProfileAccumulator(85)
        self.assertEqual(ProfileAccumulator.from_bytes(full.to_bytes(), 85).count, 0)
        with self.assertRaises(ValueError):
            ProfileAccumulator(86).to_bytes()

    def test_a_total_beyond_the_count_is_refused(self) -> None:
        from veridist.scale._codec import encode_frame

        data = encode_frame("scale.profile_accumulator", 1, 1, (MAX_ELEMENT_UNITS + 1, 0, 0))
        with self.assertRaises(ScaleStateError) as caught:
            ProfileAccumulator.from_bytes(data, 1)
        self.assertEqual(caught.exception.code, ScaleStateErrorCode.STATE_BYTES_INVALID)


class ProfileJobTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(8)
        self.source = ListSource([rng.uniform(1.0, 9.0, 30), rng.uniform(1.0, 9.0, 11)])
        self.kernel = WeibullKernel(math.log(9.0))
        self.parameters = (0.8, 2.5)

    def expected(self, ordinal: int) -> ProfileAccumulator:
        accumulator = ProfileAccumulator(2)
        accumulator.update(self.kernel(self.source.parts[ordinal]), self.parameters)
        return accumulator

    def test_a_partition_folds_into_the_bytes_of_its_sums(self) -> None:
        job = ProfileJob(self.kernel, self.parameters)
        for partition in self.source.partitions():
            with self.subTest(ordinal=partition.ordinal):
                data = job.run(self.source, partition, 1000)
                self.assertEqual(data, self.expected(partition.ordinal).to_bytes())

    def test_the_batches_may_be_any_size(self) -> None:
        job = ProfileJob(self.kernel, self.parameters)
        (first, _) = self.source.partitions()
        whole = job.run(self.source, first, 1000)
        for size in (1, 7, 30):
            self.assertEqual(job.run(self.source, first, size), whole)

    def test_restore_and_identity_follow_the_candidates(self) -> None:
        job = ProfileJob(self.kernel, self.parameters)
        data = job.run(self.source, self.source.partitions()[0], 100)
        restored = job.restore(data)
        self.assertEqual(restored.count, 30)
        self.assertEqual(len(restored.sums), 2)
        self.assertEqual(restored.to_bytes(), data)
        identity = job.identity()
        self.assertEqual((identity.count, len(identity.sums)), (0, 2))
        self.assertEqual(identity.merge(restored).to_bytes(), data)

    def test_the_candidates_decide_the_shape(self) -> None:
        job = ProfileJob(self.kernel, (1.0,))
        data = job.run(self.source, self.source.partitions()[0], 100)
        self.assertEqual(len(job.restore(data).sums), 1)
        with self.assertRaises(ScaleStateError):
            ProfileJob(self.kernel, self.parameters).restore(data)

    def test_a_job_pickles_with_its_kernel(self) -> None:
        for kernel in (self.kernel, GumbelKernel(-0.2, 3.0, 4.0)):
            job = ProfileJob(kernel, self.parameters)
            copy = pickle.loads(pickle.dumps(job))
            with self.subTest(kernel=type(kernel).__name__):
                self.assertEqual(copy, job)
                partition = self.source.partitions()[1]
                self.assertEqual(
                    copy.run(self.source, partition, 50), job.run(self.source, partition, 50)
                )


class FrameDeclarationTests(unittest.TestCase):
    def test_polars_and_duckdb_run_their_own_threads_and_are_never_shared(self) -> None:
        connection = duckdb.connect()
        self.addCleanup(connection.close)
        sources: list[Any] = [
            PolarsSource(pl.DataFrame({"x": [1.0, 2.0]}), "x"),
            PolarsSource(pl.LazyFrame({"x": [1.0, 2.0]}), "x"),
            DuckDBSource((connection, "select 1.0::double as x"), "x"),
        ]
        for source in sources:
            for kind in (ExecutorKind.THREADS, ExecutorKind.PROCESSES):
                with self.subTest(source=type(source).__name__, kind=kind):
                    self.assertIs(
                        source.parallel_obstacle(kind, source.partitions()),
                        DowngradeReason.ENGINE_MANAGED,
                    )
            self.assertTrue(is_retryable(source))
