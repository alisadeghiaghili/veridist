"""Large-n goodness of fit of a scale fit: counts, the certified enclosure, the decision."""

from __future__ import annotations

import dataclasses
import math
import unittest
from dataclasses import replace
from functools import cache
from hashlib import sha256
from typing import Any
from unittest import mock

import numpy as np

from tests.unit.scale_checkpoint_fixtures import ReadLog
from tests.unit.scale_inference_oracle import gumbel_data, weibull_data
from veridist.engine.errors import FailureCode
from veridist.engine.pass_budget import PassBudgetError
from veridist.families.registry import FamilyId
from veridist.scale import _gof as gof
from veridist.scale._errors import (
    ScaleInferenceError,
    ScaleInferenceErrorCode,
    ScaleSourceError,
    ScaleSourceErrorCode,
    ScaleStateError,
    ScaleStateErrorCode,
)
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._multi_pass import ProfileAccumulator
from veridist.scale._sources import NumpySource
from veridist.statistics.distributions import cdf

FAMILIES = ("exponential", "normal", "gamma", "lognormal", "weibull_min", "gumbel_right")


def sample(family: str, size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "exponential":
        return rng.exponential(3.0, size)
    if family == "normal":
        return rng.normal(10.0, 3.0, size)
    if family == "gamma":
        return rng.gamma(3.0, 2.0, size)
    if family == "lognormal":
        return rng.lognormal(1.0, 0.5, size)
    if family == "weibull_min":
        return weibull_data(1.7, 40.0, size, seed)
    return gumbel_data(5.0, 2.0, size, seed)


@cache
def fitted(family: str, size: int, seed: int) -> tuple[np.ndarray, ScaleFitResult]:
    values = sample(family, size, seed)
    return values, fit(NumpySource(values), family)


def exact_ks(values: np.ndarray, result: ScaleFitResult) -> float:
    """The Kolmogorov-Smirnov distance of the sorted values from the fitted distribution."""

    success: Any = result.fit
    probabilities = np.sort(cdf(success.family, np.sort(values), **dict(success.parameters)))
    return ks_of(probabilities)


def ks_of(probabilities: np.ndarray) -> float:
    size = len(probabilities)
    index = np.arange(1, size + 1)
    return float(
        max((index / size - probabilities).max(), (probabilities - (index - 1) / size).max())
    )


def brute_enclosure(probabilities: list[float], exponent: int, error: float) -> gof.Enclosure:
    """The enclosure of the counts of ``probabilities``, counted one at a time."""

    bins = 1 << exponent
    cuts = gof.edge_cuts(exponent, error)
    counts = np.zeros(2 * bins - 1, dtype=np.int64)
    for value in probabilities:
        counts[int(np.searchsorted(cuts, value, side="right"))] += 1
    return gof.enclose(counts, len(probabilities), exponent)


class ConstantsTests(unittest.TestCase):
    def test_the_limits_are_pinned(self) -> None:
        self.assertEqual(gof.DEFAULT_BIN_EXPONENT, 12)
        self.assertEqual((gof.MIN_BIN_EXPONENT, gof.MAX_BIN_EXPONENT), (1, 20))
        self.assertEqual(gof.GOF_PASS_BUDGET_DEFAULT, 1)
        self.assertEqual(gof.GOF_PASSES, 1)
        self.assertEqual(gof.MAX_PRODUCT, 1 << 62)

    def test_the_declared_error_of_the_distribution_function(self) -> None:
        self.assertEqual(gof.cdf_error_bound(FamilyId.EXPONENTIAL, {"rate": 2.0}), 2.0**-48)
        self.assertEqual(gof.cdf_error_bound(FamilyId.NORMAL, {"mu": 0.0, "sigma": 1.0}), 2.0**-48)
        self.assertEqual(
            gof.cdf_error_bound(FamilyId.GUMBEL_RIGHT, {"location": 0.0, "scale": 1.0}), 2.0**-48
        )
        self.assertEqual(
            gof.cdf_error_bound(FamilyId.WEIBULL_MIN, {"shape": 1.5, "scale": 3.0}),
            9.5 * 2.0**-49,
        )
        self.assertEqual(
            gof.cdf_error_bound(FamilyId.LOGNORMAL, {"mu_log": -3.0, "sigma_log": 0.5}),
            2.0**-52 * (16.0 + 4.0 / 0.5),
        )
        self.assertEqual(
            gof.cdf_error_bound(FamilyId.GAMMA, {"shape": 30.0, "scale": 2.0}), 38.0 * 2.0**-46
        )


class CutTests(unittest.TestCase):
    def test_the_cuts_surround_every_interior_edge_outward(self) -> None:
        cuts = gof.edge_cuts(2, 0.01)
        self.assertEqual(len(cuts), 6)
        for index, edge in enumerate((0.25, 0.5, 0.75)):
            below, above = cuts[2 * index], cuts[2 * index + 1]
            self.assertLess(below, edge - 0.01)
            self.assertGreater(above, edge + 0.01)
            self.assertEqual(below, math.nextafter(edge - 0.01, -1.0))
            self.assertEqual(above, math.nextafter(edge + 0.01, 2.0))
        self.assertTrue(all(a < b for a, b in zip(cuts, cuts[1:], strict=False)))

    def test_one_edge_has_one_pair_of_cuts(self) -> None:
        cuts = gof.edge_cuts(1, 0.0)
        self.assertEqual(list(cuts), [math.nextafter(0.5, -1.0), math.nextafter(0.5, 2.0)])

    def test_strips_that_would_touch_are_refused(self) -> None:
        gof.edge_cuts(4, 1.0 / 16 / 4.1)
        with self.assertRaises(ScaleInferenceError) as caught:
            gof.edge_cuts(4, 1.0 / 16 / 4.0)
        self.assertIs(caught.exception.code, ScaleInferenceErrorCode.GOF_RESOLUTION_UNSUPPORTED)


class CountTests(unittest.TestCase):
    def test_each_value_is_counted_in_its_region(self) -> None:
        cuts = gof.edge_cuts(2, 0.01)
        counts = gof.PitCounts(2)
        values = np.array(
            [0.0, 0.1, 0.2399, 0.24, 0.25, 0.2599, 0.26, 0.3, 0.49, 0.5, 0.74, 0.76, 0.9, 1.0]
        )
        counts.update(values, cuts)
        expected = [0] * 7
        for value in values.tolist():
            expected[int(np.searchsorted(cuts, value, side="right"))] += 1
        self.assertEqual(counts.counts.tolist(), expected)
        self.assertEqual(counts.count, 14)
        self.assertEqual(expected[0], 3)
        self.assertEqual(expected[1], 4)
        self.assertEqual(sum(expected), 14)

    def test_a_value_that_is_not_finite_is_refused(self) -> None:
        cuts = gof.edge_cuts(2, 0.01)
        for bad in (float("nan"), float("inf"), float("-inf")):
            counts = gof.PitCounts(2)
            with self.assertRaises(ScaleInferenceError) as caught:
                counts.update(np.array([0.3, bad]), cuts)
            self.assertIs(caught.exception.code, ScaleInferenceErrorCode.GOF_NOT_REPRESENTABLE)
            self.assertEqual(counts.count, 0)

    def test_counts_merge_by_addition_and_only_with_their_own_width(self) -> None:
        cuts = gof.edge_cuts(3, 1e-9)
        values = np.random.default_rng(2).uniform(0.0, 1.0, 500)
        left, right, whole = gof.PitCounts(3), gof.PitCounts(3), gof.PitCounts(3)
        left.update(values[:123], cuts)
        right.update(values[123:], cuts)
        whole.update(values, cuts)
        merged = left.merge(right)
        self.assertEqual(merged.counts.tolist(), whole.counts.tolist())
        self.assertEqual(merged.count, 500)
        self.assertEqual(merged.to_bytes(), whole.to_bytes())
        self.assertEqual(merged.to_bytes(), right.merge(left).to_bytes())
        with self.assertRaises(ScaleStateError) as caught:
            left.merge(gof.PitCounts(2))
        self.assertIs(caught.exception.code, ScaleStateErrorCode.INCOMPATIBLE_STATE)

    def test_the_bytes_round_trip_and_are_canonical(self) -> None:
        counts = gof.PitCounts(3)
        counts.update(np.random.default_rng(4).uniform(0.0, 1.0, 300), gof.edge_cuts(3, 1e-9))
        data = counts.to_bytes()
        restored = gof.PitCounts.from_bytes(data, 3)
        self.assertEqual(restored.counts.tolist(), counts.counts.tolist())
        self.assertEqual(restored.count, 300)
        self.assertEqual(restored.to_bytes(), data)
        self.assertEqual(gof.PitCounts.from_bytes(bytearray(data), 3).count, 300)

    def test_foreign_or_damaged_bytes_are_refused(self) -> None:
        counts = gof.PitCounts(3)
        counts.update(np.array([0.1, 0.5, 0.9]), gof.edge_cuts(3, 1e-9))
        data = counts.to_bytes()

        def code(blob: bytes, exponent: int = 3) -> ScaleStateErrorCode:
            with self.assertRaises(ScaleStateError) as caught:
                gof.PitCounts.from_bytes(blob, exponent)
            return caught.exception.code  # type: ignore[no-any-return]

        def resealed(body: bytearray) -> bytes:
            return bytes(body) + sha256(body).digest()

        invalid = ScaleStateErrorCode.STATE_BYTES_INVALID
        self.assertIs(code(data[:-1] + bytes([data[-1] ^ 1])), invalid)
        self.assertIs(code(data[:10]), invalid)
        self.assertIs(code(b"XXXX" + data[4:]), invalid)
        self.assertIs(code(data, 4), invalid)
        self.assertIs(code(ProfileAccumulator(1).to_bytes()), invalid)
        with self.assertRaises(TypeError):
            gof.PitCounts.from_bytes("text", 3)  # type: ignore[arg-type]
        header = 4 + 1 + len(gof.PIT_TAG)
        body = bytearray(data[:-32])
        body[5] = ord("X")
        self.assertIs(code(resealed(body)), ScaleStateErrorCode.STATE_TAG_MISMATCH)
        body = bytearray(data[:-32])
        body[header + 1] = 2
        self.assertIs(code(resealed(body)), ScaleStateErrorCode.STATE_VERSION_UNSUPPORTED)
        body = bytearray(data[:-32])
        body[header + 3 + 8 - 1] ^= 1
        self.assertIs(code(resealed(body)), invalid)


class JobTests(unittest.TestCase):
    def test_the_names_and_values_of_the_parameters_must_match(self) -> None:
        source = NumpySource(np.array([1.0, 2.0, 3.0]))
        job = gof.PitJob("normal", ("mu", "sigma"), (0.0,), 3, 1e-12)
        with self.assertRaises(ValueError):
            job.run(source, source.partitions()[0], 10)

    def test_the_key_and_the_value_are_exact(self) -> None:
        job = gof.PitJob("normal", ("mu", "sigma"), (0.5, 2.0), 3, 1e-12)
        self.assertEqual(
            job.checkpoint_key(),
            (
                "pit",
                gof.PIT_TAG,
                gof.PIT_SCHEMA_VERSION,
                "normal",
                ("mu", "sigma"),
                ((0.5).hex(), (2.0).hex()),
                3,
                (1e-12).hex(),
            ),
        )
        self.assertEqual((gof.PIT_TAG, gof.PIT_SCHEMA_VERSION), ("scale.pit_counts", 1))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            job.exponent = 4  # type: ignore[misc]
        self.assertFalse(hasattr(job, "__dict__"))
        self.assertEqual(job.identity().counts.shape, (15,))


class EnclosureTests(unittest.TestCase):
    def test_a_hand_counted_case(self) -> None:
        # B = 2, no strips: 3 of 10 values are under the edge 1/2 and 7 are over it.
        enclosure = gof.enclose(np.array([3, 0, 7]), 10, 1)
        self.assertEqual(enclosure.ks_lower, math.nextafter(0.2, -1.0))
        self.assertEqual(enclosure.ks_upper, math.nextafter(0.7, 2.0))
        self.assertEqual(enclosure.strip_rows, 0)
        self.assertAlmostEqual(enclosure.tv_lower, 0.2, places=15)
        self.assertAlmostEqual(enclosure.tv_upper, 0.2, places=15)

    def test_strip_rows_widen_the_bounds_and_the_total_variation(self) -> None:
        # B = 2: 3 clearly under, 2 in the strip of the edge, 5 clearly over
        enclosure = gof.enclose(np.array([3, 2, 5]), 10, 1)
        self.assertEqual(enclosure.ks_lower, 0.0)
        self.assertEqual(enclosure.ks_upper, math.nextafter(0.7, 2.0))
        self.assertEqual(enclosure.strip_rows, 2)
        self.assertAlmostEqual(enclosure.tv_lower, 0.0, places=15)
        self.assertAlmostEqual(enclosure.tv_upper, 0.2, places=15)

    def test_fewer_than_two_bins_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least"):
            gof.enclose(np.array([4]), 4, 0)

    def test_a_perfectly_uniform_sample_has_a_zero_lower_bound(self) -> None:
        enclosure = gof.enclose(np.array([5, 0, 5]), 10, 1)
        self.assertEqual(enclosure.ks_lower, 0.0)
        self.assertAlmostEqual(enclosure.ks_upper, 0.5, places=12)
        self.assertEqual((enclosure.tv_lower, enclosure.tv_upper), (0.0, 0.0))

    def test_the_bounds_stay_inside_zero_and_one(self) -> None:
        enclosure = gof.enclose(np.array([10, 0, 0]), 10, 1)
        self.assertEqual(enclosure.ks_upper, 1.0)
        self.assertGreaterEqual(enclosure.ks_lower, 0.0)

    def test_the_enclosure_contains_the_exact_distance_and_is_at_most_one_bin_wide(self) -> None:
        rng = np.random.default_rng(11)
        for exponent in (1, 2, 4, 7):
            for shape in (1.0, 1.3, 0.8):
                probabilities = np.sort(rng.beta(shape, 1.0, 400))
                exact = ks_of(probabilities)
                enclosure = brute_enclosure(probabilities.tolist(), exponent, 0.0)
                with self.subTest(exponent=exponent, shape=shape):
                    self.assertLessEqual(enclosure.ks_lower, exact)
                    self.assertLessEqual(exact, enclosure.ks_upper)
                    self.assertLessEqual(
                        enclosure.ks_upper - enclosure.ks_lower, 1.0 / (1 << exponent) + 1e-15
                    )

    def test_the_enclosure_contains_the_exact_distance_with_strips(self) -> None:
        probabilities = np.sort(np.random.default_rng(12).beta(1.1, 1.0, 600))
        exact = ks_of(probabilities)
        enclosure = brute_enclosure(probabilities.tolist(), 3, 1e-3)
        self.assertGreater(enclosure.strip_rows, 0)
        self.assertLessEqual(enclosure.ks_lower, exact)
        self.assertLessEqual(exact, enclosure.ks_upper)
        bare = brute_enclosure(probabilities.tolist(), 3, 0.0)
        self.assertLessEqual(enclosure.ks_lower, bare.ks_lower)
        self.assertGreaterEqual(enclosure.ks_upper, bare.ks_upper)

    def test_values_off_by_the_declared_error_cannot_escape_the_enclosure(self) -> None:
        rng = np.random.default_rng(13)
        true = np.sort(rng.beta(1.2, 1.0, 500))
        error = 2e-3
        noisy = np.clip(true + rng.uniform(-error, error, true.shape), 0.0, 1.0)
        exact = ks_of(true)
        enclosure = brute_enclosure(noisy.tolist(), 4, error)
        self.assertLessEqual(enclosure.ks_lower, exact)
        self.assertLessEqual(exact, enclosure.ks_upper)

    def test_the_width_shrinks_as_the_bins_grow(self) -> None:
        probabilities = np.random.default_rng(14).beta(1.2, 1.0, 5000).tolist()
        widths = []
        for exponent in (2, 4, 6, 8, 10):
            enclosure = brute_enclosure(probabilities, exponent, 1e-12)
            widths.append(enclosure.ks_upper - enclosure.ks_lower)
        self.assertEqual(widths, sorted(widths, reverse=True))
        self.assertLessEqual(widths[-1], 1.0 / 1024 + 1e-12)

    def test_the_exact_integer_arithmetic_handles_a_huge_count(self) -> None:
        rows = 10**12
        counts = np.array([rows // 2 + 1, 0, rows - rows // 2 - 1])
        enclosure = gof.enclose(counts, rows, 1)
        self.assertAlmostEqual(enclosure.ks_lower, 1.0 / rows, delta=1e-24)


class ImmutabilityTests(unittest.TestCase):
    def test_the_values_are_immutable_and_slotted(self) -> None:
        values, result = fitted("gamma", 2000, 3)
        report = gof.large_n_gof(NumpySource(values), result, bin_exponent=4)
        for value in (report, gof.Enclosure(0.0, 1.0, 0.0, 1.0, 0)):
            with self.subTest(type=type(value).__name__):
                first = next(iter(type(value).__dataclass_fields__))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(value, first, None)
                self.assertFalse(hasattr(value, "__dict__"))


class DecisionTests(unittest.TestCase):
    def enclosure(self, lower: float, upper: float) -> gof.Enclosure:
        return gof.Enclosure(lower, upper, 0.0, 0.0, 0)

    def test_no_threshold_no_decision(self) -> None:
        self.assertIsNone(gof.decide(self.enclosure(0.1, 0.2), None))

    def test_the_edges_of_the_threshold(self) -> None:
        box = self.enclosure(0.10, 0.20)
        self.assertIs(gof.decide(box, 0.20), gof.GofDecision.PASS)
        self.assertIs(gof.decide(box, 0.25), gof.GofDecision.PASS)
        self.assertIs(gof.decide(box, math.nextafter(0.20, 0.0)), gof.GofDecision.INCONCLUSIVE)
        self.assertIs(gof.decide(box, 0.15), gof.GofDecision.INCONCLUSIVE)
        self.assertIs(gof.decide(box, 0.10), gof.GofDecision.INCONCLUSIVE)
        self.assertIs(gof.decide(box, math.nextafter(0.10, 0.0)), gof.GofDecision.FAIL)
        self.assertIs(gof.decide(box, 0.05), gof.GofDecision.FAIL)
        self.assertIs(gof.decide(box, 0.0), gof.GofDecision.FAIL)

    def test_the_decisions_are_named_pass_fail_inconclusive(self) -> None:
        self.assertEqual([item.value for item in gof.GofDecision], ["PASS", "FAIL", "INCONCLUSIVE"])


class ReportTests(unittest.TestCase):
    def test_the_enclosure_contains_the_exact_distance_for_every_family(self) -> None:
        for family in FAMILIES:
            values, result = fitted(family, 20_000, 3)
            for exponent in (6, 10):
                with self.subTest(family=family, exponent=exponent):
                    out = gof.large_n_gof(
                        NumpySource(values, partition_rows=7000), result, bin_exponent=exponent
                    )
                    exact = exact_ks(values, result)
                    self.assertLessEqual(out.ks_lower, exact)
                    self.assertLessEqual(exact, out.ks_upper)
                    self.assertLessEqual(out.ks_upper - out.ks_lower, 1.0 / (1 << exponent) + 1e-9)
                    self.assertEqual((out.rows, out.bins), (20_000, 1 << exponent))
                    self.assertIs(out.family, result.fit.family)  # type: ignore[union-attr]

    def test_a_badly_specified_model_is_far_and_a_good_one_is_near(self) -> None:
        values = sample("lognormal", 30_000, 5)
        good = fit(NumpySource(values), "lognormal")
        bad = fit(NumpySource(values), "normal")
        near = gof.large_n_gof(NumpySource(values), good, bin_exponent=12)
        far = gof.large_n_gof(NumpySource(values), bad, bin_exponent=12)
        self.assertLess(near.ks_upper, 0.02)
        self.assertGreater(far.ks_lower, 0.05)
        self.assertGreater(far.tv_lower, near.tv_upper)
        for report, result in ((near, good), (far, bad)):
            exact = exact_ks(values, result)
            self.assertLessEqual(report.ks_lower, exact)
            self.assertLessEqual(exact, report.ks_upper)

    def test_the_width_shrinks_as_the_bins_grow_on_a_fit(self) -> None:
        values, result = fitted("weibull_min", 20_000, 3)
        widths = []
        for exponent in (4, 8, 12, 16):
            out = gof.large_n_gof(NumpySource(values), result, bin_exponent=exponent)
            widths.append(out.ks_upper - out.ks_lower)
        self.assertEqual(widths, sorted(widths, reverse=True))
        self.assertLess(widths[-1], widths[0] / 1000)

    def test_a_threshold_is_decided(self) -> None:
        values, result = fitted("normal", 20_000, 3)
        base = gof.large_n_gof(NumpySource(values), result, bin_exponent=12)
        self.assertIsNone(base.threshold)
        self.assertIsNone(base.decision)

        def decision(threshold: float) -> gof.GofDecision | None:
            return gof.large_n_gof(
                NumpySource(values), result, bin_exponent=12, threshold=threshold
            ).decision

        self.assertIs(decision(0.5), gof.GofDecision.PASS)
        self.assertIs(decision(0.0), gof.GofDecision.FAIL)
        self.assertIs(decision((base.ks_lower + base.ks_upper) / 2), gof.GofDecision.INCONCLUSIVE)
        self.assertIs(decision(base.ks_upper), gof.GofDecision.PASS)
        self.assertIs(decision(base.ks_lower), gof.GofDecision.INCONCLUSIVE)
        self.assertIs(decision(math.nextafter(base.ks_lower, 0.0)), gof.GofDecision.FAIL)
        self.assertEqual(
            gof.large_n_gof(NumpySource(values), result, bin_exponent=12, threshold=0.5).threshold,
            0.5,
        )

    def test_the_report_names_the_declared_error_and_the_counts(self) -> None:
        values, result = fitted("weibull_min", 5000, 3)
        out = gof.large_n_gof(NumpySource(values), result, bin_exponent=8)
        self.assertEqual(
            out.cdf_error,
            gof.cdf_error_bound(FamilyId.WEIBULL_MIN, result.fit.parameters),  # type: ignore[union-attr]
        )
        self.assertEqual(len(out.counts_digest), 64)
        again = gof.large_n_gof(NumpySource(values), result, bin_exponent=8)
        self.assertEqual(again.counts_digest, out.counts_digest)
        other = gof.large_n_gof(NumpySource(values), result, bin_exponent=9)
        self.assertNotEqual(other.counts_digest, out.counts_digest)
        self.assertEqual(out.strip_rows, 0)
        self.assertEqual(out.scale.row_count, 5000)
        self.assertEqual(out.scale.state, result.scale.state)

    def test_strip_rows_are_counted(self) -> None:
        values, result = fitted("exponential", 5000, 3)
        with mock.patch.object(gof, "cdf_error_bound", return_value=2e-3):
            out = gof.large_n_gof(NumpySource(values), result, bin_exponent=4)
        self.assertGreater(out.strip_rows, 0)
        self.assertEqual(out.cdf_error, 2e-3)
        exact = exact_ks(values, result)
        self.assertLessEqual(out.ks_lower, exact)
        self.assertLessEqual(exact, out.ks_upper)

    def test_the_pass_is_one_gof_pass(self) -> None:
        values, result = fitted("gamma", 5000, 3)
        source = ReadLog(NumpySource(values, partition_rows=2000))
        out = gof.large_n_gof(source, result)
        passes = out.scale.passes
        assert passes is not None
        self.assertEqual(
            (passes.pass_budget, passes.minimum_passes, passes.maximum_passes, passes.pass_count),
            (1, 1, 1, 1),
        )
        self.assertEqual([(r.role, r.points) for r in passes.records], [("gof", 1)])
        self.assertEqual(source.reads, [0, 1, 2])

    def test_a_single_use_source_is_enough(self) -> None:
        values, result = fitted("gamma", 3000, 3)

        class Once(NumpySource):
            single_use = True

        out = gof.large_n_gof(Once(values), result, bin_exponent=6)
        self.assertEqual(out.rows, 3000)


class RefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values, self.result = fitted("weibull_min", 3000, 3)

    def test_a_failed_fit_has_nothing_to_compare_with(self) -> None:
        for bad in (
            fit(NumpySource(np.full(10, 3.0)), "weibull_min"),
            fit(NumpySource(np.array([-1.0, 2.0, 3.0])), "lognormal"),
            object(),
        ):
            with self.assertRaises(ScaleInferenceError) as caught:
                gof.large_n_gof(NumpySource(self.values), bad)  # type: ignore[arg-type]
            self.assertIs(caught.exception.code, ScaleInferenceErrorCode.GOF_FIT_NOT_AVAILABLE)

    def test_the_arguments_are_validated_before_anything_is_read(self) -> None:
        source = ReadLog(NumpySource(self.values))
        cases: list[tuple[type[Exception], dict[str, Any]]] = [
            (TypeError, {"bin_exponent": 4.0}),
            (TypeError, {"bin_exponent": True}),
            (ValueError, {"bin_exponent": 0}),
            (ValueError, {"bin_exponent": 21}),
            (TypeError, {"threshold": 1}),
            (TypeError, {"threshold": "0.1"}),
            (ValueError, {"threshold": -0.1}),
            (ValueError, {"threshold": 1.5}),
            (TypeError, {"inference_pass_budget": 1.0}),
            (ValueError, {"inference_pass_budget": 0}),
            (ValueError, {"max_rows": 0}),
            (ValueError, {"executor": "x"}),
            (ValueError, {"resume": True}),
        ]
        for error, arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(error):
                gof.large_n_gof(source, self.result, **arguments)
        with self.assertRaises(TypeError):
            gof.large_n_gof(object(), self.result)  # type: ignore[arg-type]
        self.assertEqual(source.reads, [])

    def test_the_extreme_bin_exponents_are_accepted(self) -> None:
        small = gof.large_n_gof(NumpySource(self.values), self.result, bin_exponent=1)
        self.assertEqual(small.bins, 2)
        large = gof.large_n_gof(NumpySource(self.values), self.result, bin_exponent=20)
        self.assertEqual(large.bins, 1 << 20)
        self.assertLess(large.ks_upper - large.ks_lower, 2e-6)

    def test_a_product_beyond_the_exact_range_is_refused_before_any_read(self) -> None:
        source = ReadLog(NumpySource(self.values))
        with mock.patch.object(gof, "MAX_PRODUCT", 3000 * 64):
            with self.assertRaises(ScaleStateError) as caught:
                gof.large_n_gof(source, self.result, bin_exponent=6)
            self.assertIs(caught.exception.code, ScaleStateErrorCode.OBSERVATION_LIMIT_EXCEEDED)
            self.assertEqual(source.reads, [])
            gof.large_n_gof(source, self.result, bin_exponent=5)
        self.assertEqual(source.reads, [0])

    def test_an_error_too_large_for_the_bins_is_refused_before_any_read(self) -> None:
        source = ReadLog(NumpySource(self.values))
        with mock.patch.object(gof, "cdf_error_bound", return_value=0.01):
            with self.assertRaises(ScaleInferenceError) as caught:
                gof.large_n_gof(source, self.result, bin_exponent=8)
        self.assertIs(caught.exception.code, ScaleInferenceErrorCode.GOF_RESOLUTION_UNSUPPORTED)
        self.assertEqual(source.reads, [])

    def test_a_distribution_function_that_fails_or_is_not_a_probability(self) -> None:
        target = "veridist.statistics.distributions.cdf"
        for failure in (ArithmeticError, ValueError):
            with mock.patch(target, side_effect=failure):
                with self.assertRaises(ScaleInferenceError) as caught:
                    gof.large_n_gof(NumpySource(self.values), self.result)
            self.assertIs(caught.exception.code, ScaleInferenceErrorCode.GOF_NOT_REPRESENTABLE)
        with mock.patch(
            target, side_effect=lambda family, values, **p: np.full(values.shape, np.nan)
        ):
            with self.assertRaises(ScaleInferenceError) as caught:
                gof.large_n_gof(NumpySource(self.values), self.result)
        self.assertIs(caught.exception.code, ScaleInferenceErrorCode.GOF_NOT_REPRESENTABLE)

    def test_a_gamma_shape_the_kernel_cannot_evaluate_is_refused(self) -> None:
        values = np.random.default_rng(1).gamma(20000.0, 1.0, 2000)
        result = fit(NumpySource(values), "gamma")
        with self.assertRaises(ScaleInferenceError) as caught:
            gof.large_n_gof(NumpySource(values), result)
        self.assertIs(caught.exception.code, ScaleInferenceErrorCode.GOF_NOT_REPRESENTABLE)

    def test_another_source_is_partition_changed(self) -> None:
        with self.assertRaises(ScaleSourceError) as caught:
            gof.large_n_gof(NumpySource(self.values[:-1]), self.result)
        self.assertIs(caught.exception.code, ScaleSourceErrorCode.PARTITION_CHANGED)

    def test_the_budget_is_one_pass_and_a_larger_one_is_accepted(self) -> None:
        out = gof.large_n_gof(NumpySource(self.values), self.result, inference_pass_budget=4)
        assert out.scale.passes is not None
        self.assertEqual(out.scale.passes.pass_budget, 4)
        self.assertEqual(out.scale.passes.pass_count, 1)
        with mock.patch.object(gof, "GOF_PASSES", 2):
            with self.assertRaises(PassBudgetError) as caught:
                gof.large_n_gof(NumpySource(self.values), self.result)
        self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_INSUFFICIENT)

    def test_the_decision_does_not_change_the_numbers(self) -> None:
        plain = gof.large_n_gof(NumpySource(self.values), self.result)
        decided = gof.large_n_gof(NumpySource(self.values), self.result, threshold=0.5)
        self.assertEqual(replace(decided, threshold=None, decision=None), plain)


if __name__ == "__main__":
    unittest.main()
