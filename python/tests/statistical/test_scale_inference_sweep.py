"""A few million rows: profile intervals and large-n goodness of fit within the time limit.

A Weibull fit over 3 million rows is followed by its profile-likelihood intervals (shape, scale
and the B10 life), by the large-n goodness-of-fit report of the right model and of a wrong one,
and by the same intervals and report read on threads. The intervals agree with the Wald intervals
of the fit to the order the likelihood allows, the enclosure of the distance is below its
threshold for the right model and above it for the wrong one, the thread run equals the serial
run bit for bit, and the whole run stays within its time limit.
"""

from __future__ import annotations

import time
import unittest

from tests.unit.scale_executor_fixtures import NoLeaksTestCase
from tests.unit.scale_inference_fixtures import gof_bits, profile_bits
from tests.unit.scale_inference_oracle import weibull_data
from veridist.scale import _gof as gof
from veridist.scale import _profile_intervals as pi
from veridist.scale._executors import ThreadExecutor
from veridist.scale._fit import fit
from veridist.scale._sources import NumpySource

LIMIT_SECONDS = 30.0
ROWS = 3_000_000
PARTITION_ROWS = 750_000


class InferenceSweepTests(NoLeaksTestCase):
    def test_millions_of_rows_are_inferred_within_the_time_limit(self) -> None:
        started = time.monotonic()
        values = weibull_data(1.7, 40.0, ROWS, 31)
        source = NumpySource(values, partition_rows=PARTITION_ROWS)
        executor = ThreadExecutor(4)
        result = fit(source, "weibull_min", executor=executor)
        self.assertEqual(result.scale.row_count, ROWS)

        quantities = [pi.quantile(0.1)]
        serial = pi.profile_intervals(source, result, quantities=quantities)
        threaded = pi.profile_intervals(source, result, quantities=quantities, executor=executor)
        self.assertEqual(profile_bits(threaded), profile_bits(serial))
        assert serial.scale.passes is not None
        self.assertLessEqual(serial.scale.passes.pass_count, 4)

        wald = result.fit.uncertainty().confidence_intervals(0.95, "wald")  # type: ignore[union-attr]
        for name, interval in serial.parameters.items():
            low, high = wald[name]
            self.assertLess(interval.lower, result.fit.parameters[name])  # type: ignore[union-attr]
            self.assertGreater(interval.upper, result.fit.parameters[name])  # type: ignore[union-attr]
            self.assertAlmostEqual(interval.lower / low, 1.0, delta=1e-3)
            self.assertAlmostEqual(interval.upper / high, 1.0, delta=1e-3)
        self.assertLess(serial.parameters["shape"].lower, 1.7)
        self.assertGreater(serial.parameters["shape"].upper, 1.7)
        self.assertLess(serial.resolution.confirmation_residual, 1e-4)

        right = gof.large_n_gof(source, result, bin_exponent=14, threshold=0.01, executor=executor)
        self.assertEqual(right.decision, gof.GofDecision.PASS)
        self.assertLess(right.ks_upper, 0.01)
        self.assertLess(right.ks_upper - right.ks_lower, 1.0 / (1 << 14) + 1e-9)
        wrong = fit(source, "exponential", executor=executor)
        far = gof.large_n_gof(source, wrong, bin_exponent=14, threshold=0.01, executor=executor)
        self.assertEqual(far.decision, gof.GofDecision.FAIL)
        self.assertGreater(far.ks_lower, 0.01)
        self.assertGreater(far.tv_lower, right.tv_upper)
        self.assertEqual(
            gof_bits(gof.large_n_gof(source, result, bin_exponent=14, threshold=0.01)),
            gof_bits(right),
        )
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, LIMIT_SECONDS, f"took {elapsed:.1f} s")


if __name__ == "__main__":
    unittest.main()
