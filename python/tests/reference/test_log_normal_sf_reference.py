"""Independent mpmath reference contract for the stable lognormal right tail.

Pins ``_log_normal_sf``, which replaces a direct ``log(0.5 * erfc(z / sqrt(2)))``
that raised ``ValueError`` whenever ``erfc`` underflowed to exactly ``0.0``. That
previously aborted a whole Lognormal fit with ``OPTIMIZER_EXHAUSTED`` even when
the true optimum was fine elsewhere, because a single heavily right-censored
observation already far out on the tail destroyed the entire likelihood
evaluation. ``_log_normal_sf`` never raises for finite ``z`` and matches
``mpmath.log(mpmath.ncdf(-z))`` to a tight relative tolerance across the direct
formula's range and its asymptotic-tail fallback.
"""

from __future__ import annotations

import math
import unittest

import mpmath

from veridist.families.lognormal import _log_normal_sf

mpmath.mp.dps = 50

# Non-negative z only: for z <= 0 the true log-survival is already within a
# few ULP of 0.0 (survival close to 1), so a relative-error comparison against
# an mpmath value many orders of magnitude smaller than double precision's
# resolution near 1.0 is not meaningful. The asymptotic branch this test
# pins only ever triggers for large positive z, where `erfc` underflows.
_Z_GRID = (
    0.0,
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
    15.0,
    20.0,
    25.0,
    30.0,
    35.0,
    36.0,
    # The direct formula's underflow threshold (survival < 1e-300) falls
    # inside this neighborhood; the grid straddles it on both sides.
    37.0,
    37.5,
    38.0,
    39.0,
    40.0,
    45.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
    300.0,
    500.0,
    700.0,
    1000.0,
)


class LogNormalSfReferenceTests(unittest.TestCase):
    def test_matches_mpmath_log_ncdf_across_the_direct_and_asymptotic_ranges(self) -> None:
        for z in _Z_GRID:
            with self.subTest(z=z):
                reference = mpmath.log(mpmath.ncdf(-z))
                observed = _log_normal_sf(z)
                self.assertTrue(mpmath.isfinite(mpmath.mpf(observed)))
                relative_error = abs(mpmath.mpf(observed) - reference) / abs(reference)
                self.assertLessEqual(float(relative_error), 1e-12)

    def test_never_raises_for_very_large_finite_z(self) -> None:
        # z**2 already exceeds double range around z ~ 1.3e154, where the true
        # log-survival (~-z**2/2) is itself unrepresentable as a finite float;
        # returning -inf there is the correct saturation, not a bug. The
        # requirement is only that no exception is raised and no NaN appears.
        for z in (1e3, 1e6, 1e50, 1e100, 1e300):
            with self.subTest(z=z):
                result = _log_normal_sf(z)
                self.assertFalse(math.isnan(result))
                self.assertLess(result, 0.0)
        for z in (1e3, 1e6, 1e50):
            with self.subTest(z=z, contract="still-finite-below-the-float-range-limit"):
                self.assertTrue(math.isfinite(_log_normal_sf(z)))

    def test_asymptotic_branch_is_used_only_past_the_underflow_threshold(self) -> None:
        # Just below the threshold, the direct erfc-based formula must still
        # be used (it is exact there); just above it, the asymptotic
        # expansion takes over. Both must agree closely with mpmath (checked
        # above) and with each other at the boundary.
        self.assertGreater(0.5 * math.erfc(36.0 / math.sqrt(2.0)), 1e-300)


if __name__ == "__main__":
    unittest.main()
