"""Process executors read the passes of inference: same bits as the serial run, nothing left.

The pool is an in-process stand-in (``FakePool``) that runs each task where it is submitted, so
the partitions are read by the code of the worker task without spawning an interpreter; the
test that starts real worker processes is in ``tests/statistical`` (``full_size``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from tests.unit.scale_executor_fixtures import NoLeaksTestCase, PoolRecorder, parquet_dataset
from tests.unit.scale_inference_fixtures import gof_bits, profile_bits
from tests.unit.scale_inference_oracle import weibull_data
from veridist.scale import _gof as gof
from veridist.scale import _profile_intervals as pi
from veridist.scale._executors import ProcessExecutor
from veridist.scale._fit import fit
from veridist.scale._sources import NumpySource


class ProcessInferenceTests(NoLeaksTestCase):
    def test_profile_and_goodness_of_fit_on_processes_equal_the_serial_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            values = weibull_data(1.7, 40.0, 300, 5)
            source = parquet_dataset(Path(directory), values, 3, 2)
            result = fit(NumpySource(values), "weibull_min")
            quantities: list[pi.Quantity] = []
            base = pi.profile_intervals(source, result, quantities=quantities)
            recorder = PoolRecorder()
            executor = ProcessExecutor(2, pool_factory=recorder)
            parallel = pi.profile_intervals(
                source, result, quantities=quantities, executor=executor
            )
            self.assertEqual(profile_bits(parallel), profile_bits(base))
            assert parallel.scale.execution is not None and parallel.scale.passes is not None
            self.assertEqual(parallel.scale.execution.executor, "processes")
            self.assertEqual(len(parallel.scale.execution.passes), parallel.scale.passes.pass_count)
            self.assertTrue(
                all(item.executor == "processes" for item in parallel.scale.execution.passes)
            )
            self.assertEqual(len(recorder.pools), 1)
            self.assertEqual(len(recorder.pools[0].tasks), 6 * parallel.scale.passes.pass_count)
            report = gof.large_n_gof(source, result, bin_exponent=8)
            report_parallel = gof.large_n_gof(source, result, bin_exponent=8, executor=executor)
            self.assertEqual(gof_bits(report_parallel), gof_bits(report))
            self.assertEqual(len(recorder.pools), 2)
            self.assertEqual(len(recorder.pools[1].tasks), 6)
