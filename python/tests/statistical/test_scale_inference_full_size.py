"""The checks of inference from a scale fit that are too slow for the mutation run, at full size.

The tests of ``tests/unit``, ``tests/property`` and ``tests/reference`` that cover the same code
use few rows and few cases so that a mutant is killed in milliseconds. Here the same tests run
with the larger data, the whole grids, every interruption point and real worker processes, and
the checks whose cost is the size itself (the most quantities, the most bins, forty digits at
hundreds of rows) are made as they are specified.
"""

from __future__ import annotations

import multiprocessing
import tempfile
import unittest
from pathlib import Path

from tests.property import test_scale_inference_invariance as invariance
from tests.reference import test_scale_cdf_error_reference as cdf_reference
from tests.reference import test_scale_profile_reference as profile_reference
from tests.unit import test_scale_gof as gof_tests
from tests.unit import test_scale_inference_checkpoint as checkpoint
from tests.unit import test_scale_profile_intervals as interval_tests
from tests.unit.scale_executor_fixtures import NoLeaksTestCase, parquet_dataset
from tests.unit.scale_inference_fixtures import gof_bits, profile_bits
from tests.unit.scale_inference_oracle import weibull_data
from veridist.scale import _gof as gof
from veridist.scale import _profile_intervals as pi
from veridist.scale._executors import ProcessExecutor, ThreadExecutor
from veridist.scale._fit import fit
from veridist.scale._sources import NumpySource

ALL_QUANTITIES = (pi.mean(), pi.quantile(0.1), pi.survival(30.0))


class FullProfileInvarianceTests(invariance.ProfileInvarianceTests):
    SIZE = 900
    QUANTITIES = ALL_QUANTITIES
    SWEEP_QUANTITIES = (pi.quantile(0.1),)
    SINGLETONS = True
    LABELS = None


class FullGofInvarianceTests(invariance.GofInvarianceTests):
    SIZE = 900


class FullProfileReferenceTests(profile_reference.ProfileReferenceTests):
    ROWS = 200
    ITERATIONS = 60


class FullCdfErrorReferenceTests(cdf_reference.CdfErrorReferenceTests):
    CASES = cdf_reference.CASES
    POINTS = 30


class FullProfileResumeTests(checkpoint.ProfileResumeTests):
    SIZE = 800
    MAX_ROWS = 64
    QUANTITIES = (pi.quantile(0.1),)

    def test_a_resumed_run_equals_the_uninterrupted_one_at_every_commit(self) -> None:
        for executor in (None, ThreadExecutor(3)):
            self.check_resumes(executor, lambda total: range(1, total))


class FullGofResumeTests(checkpoint.GofResumeTests):
    SIZE = 800
    MAX_ROWS = 64

    def test_a_resumed_report_equals_the_uninterrupted_one_at_every_commit(self) -> None:
        for executor in (None, ThreadExecutor(2)):
            self.check_resumes(executor, range(1, checkpoint.PARTITIONS))


class FullRefusalTests(checkpoint.RefusalTests):
    SIZE = 800
    MAX_ROWS = 64
    QUANTITIES = (pi.quantile(0.1),)


class InMemoryAgreementTests(unittest.TestCase):
    """The intervals equal the in-memory profile intervals at several sizes and levels."""

    def test_the_parameters_at_several_sizes_and_levels(self) -> None:
        for family in interval_tests.DATA:
            for size, seed in ((400, 1), (3000, 2)):
                values, result = interval_tests.scale_fit(family, size, seed)
                uncertainty = interval_tests.in_memory_uncertainty(family, values)
                for level in (0.9, 0.95, 0.99):
                    with self.subTest(family=family, size=size, level=level):
                        out = pi.profile_intervals(NumpySource(values), result, level=level)
                        reference = uncertainty.interval_details(level, "profile")
                        self.assertEqual(set(out.parameters), set(reference))
                        for name, interval in out.parameters.items():
                            expected = reference[name]
                            self.assertAlmostEqual(
                                interval.lower / expected.lower, 1.0, delta=1e-7, msg=name
                            )
                            self.assertAlmostEqual(
                                interval.upper / expected.upper, 1.0, delta=1e-7, msg=name
                            )

    def test_the_weibull_quantities_at_eight_hundred_rows(self) -> None:
        values, result = interval_tests.scale_fit("weibull_min", 800, 3)
        requested = [pi.mean(), pi.quantile(0.1), pi.quantile(0.5), pi.survival(30.0)]
        out = pi.profile_intervals(NumpySource(values), result, quantities=requested)
        uncertainty = interval_tests.in_memory_uncertainty("weibull_min", values)
        reference = {
            "mean": uncertainty.mean(0.95, "profile"),
            "quantile:0.1": uncertainty.quantile(0.1, 0.95, "profile"),
            "quantile:0.5": uncertainty.quantile(0.5, 0.95, "profile"),
            "survival:30.0": uncertainty.survival(30.0, 0.95, "profile"),
        }
        self.assertEqual(list(out.derived), list(reference))
        for name, expected in reference.items():
            got = out.derived[name]
            with self.subTest(name=name):
                self.assertAlmostEqual(got.estimate / expected.estimate, 1.0, delta=1e-6)  # type: ignore[union-attr]
                self.assertAlmostEqual(got.lower / expected.lower, 1.0, delta=1e-6)  # type: ignore[union-attr]
                self.assertAlmostEqual(got.upper / expected.upper, 1.0, delta=1e-6)  # type: ignore[union-attr]

    def test_the_most_quantities_are_searched_for_real(self) -> None:
        values, result = interval_tests.scale_fit("weibull_min", 200, 5)
        out = pi.profile_intervals(
            NumpySource(values),
            result,
            quantities=[pi.survival(float(i + 1)) for i in range(64)],
        )
        self.assertEqual(len(out.derived), 64)
        assert out.scale.passes is not None
        self.assertLessEqual(out.scale.passes.records[-1].points, 2 + 2 * 65)


class MostBinsTests(unittest.TestCase):
    def test_the_most_bins_with_the_cuts_built_one_edge_at_a_time(self) -> None:
        values, result = gof_tests.fitted("weibull_min", 3000, 3)
        large = gof.large_n_gof(NumpySource(values), result, bin_exponent=20)
        self.assertEqual(large.bins, 1 << 20)
        self.assertLess(large.ks_upper - large.ks_lower, 2e-6)


class ProcessInferenceTests(NoLeaksTestCase):
    real_processes = True

    def test_profile_and_goodness_of_fit_on_processes_equal_the_serial_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = parquet_dataset(Path(directory), weibull_data(1.7, 40.0, 900, 5), 3, 2)
            result = fit(source, "weibull_min")
            quantities = [pi.quantile(0.1)]
            base = pi.profile_intervals(source, result, quantities=quantities)
            parallel = pi.profile_intervals(
                source, result, quantities=quantities, executor=ProcessExecutor(2)
            )
            self.assertEqual(profile_bits(parallel), profile_bits(base))
            assert parallel.scale.execution is not None and parallel.scale.passes is not None
            self.assertEqual(parallel.scale.execution.executor, "processes")
            self.assertEqual(len(parallel.scale.execution.passes), parallel.scale.passes.pass_count)
            self.assertTrue(
                all(item.executor == "processes" for item in parallel.scale.execution.passes)
            )
            report = gof.large_n_gof(source, result, bin_exponent=8)
            report_parallel = gof.large_n_gof(
                source, result, bin_exponent=8, executor=ProcessExecutor(2)
            )
            self.assertEqual(gof_bits(report_parallel), gof_bits(report))
            self.assertEqual(multiprocessing.active_children(), [])


if __name__ == "__main__":
    unittest.main()
