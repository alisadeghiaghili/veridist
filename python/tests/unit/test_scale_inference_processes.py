"""Real worker processes read the passes of inference: same bits as the serial run, nothing left."""

from __future__ import annotations

import multiprocessing
import tempfile
from pathlib import Path

from tests.unit.scale_executor_fixtures import NoLeaksTestCase, parquet_dataset
from tests.unit.scale_inference_fixtures import gof_bits, profile_bits
from tests.unit.scale_inference_oracle import weibull_data
from veridist.scale import _gof as gof
from veridist.scale import _profile_intervals as pi
from veridist.scale._executors import ProcessExecutor
from veridist.scale._fit import fit


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
