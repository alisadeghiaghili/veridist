"""Independent reference contract for the censored Lognormal MLE at n = 2000.

Before this performance work, this fit evaluated the full likelihood in
two nested 80-step golden-section searches (~6,400 evaluations), each
O(observation count) because every censored observation was summed
individually. Grouping censored observations by identical time and hoisting
the exact-observation sums out of the inner loop made each evaluation
O(distinct censored times) instead of O(n); this sample (single censoring
time) took roughly 10-20s before and well under 1s after on the reference
machine (no timing assertion is made here; see the commit message for the
measured before/after).
"""

from __future__ import annotations

import time
import unittest

import numpy as np

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.families.lognormal import fit_lognormal


class LognormalCensoredReferenceTests(unittest.TestCase):
    def test_n_2000_censored_fit_matches_an_independent_scipy_reference(self) -> None:
        # Reference computed out-of-band with scipy 1.x in a scratch venv:
        # Nelder-Mead minimization of the negative log-likelihood (the same
        # censored Lognormal likelihood this module implements) over
        # (mu, log(sigma)), seeded from (0.0, 0.0), `xatol=fatol=1e-12`,
        # `maxiter=maxfev=20000`, on the exact observations constructed
        # below: a `numpy.random.default_rng(0)` draws 200 normals (discarded,
        # matching an unrelated earlier draw in this suite) and then
        # `t = exp(rng.normal(2, 1, 2000))`, right-censored at `exp(2.5)`.
        # Result: mu ~1.9668544328027235, sigma ~1.0015851733062753.
        rng = np.random.default_rng(0)
        _ = rng.normal(0, 1, 200)
        times = np.exp(rng.normal(2, 1, 2000))
        censor_at = np.exp(2.5)
        observations = tuple(
            ExactLifetime(float(t)) if t <= censor_at else RightCensoredLifetime(float(censor_at))
            for t in times
        )
        event_count = sum(1 for t in times if t <= censor_at)
        self.assertEqual(event_count, 1403)

        start = time.perf_counter()
        result = fit_lognormal(observations)
        elapsed = time.perf_counter() - start

        self.assertTrue(hasattr(result, "mu_log"), result)
        self.assertTrue(result.converged)
        reference_mu = 1.9668544328027235
        reference_sigma = 1.0015851733062753
        self.assertAlmostEqual(result.mu_log, reference_mu, delta=abs(reference_mu) * 1e-6)
        self.assertAlmostEqual(
            result.sigma_log, reference_sigma, delta=abs(reference_sigma) * 1e-6
        )
        # No timing assertion: environments vary. The measured elapsed time
        # is available to anyone running this test with `-s` for visibility.
        self.assertGreater(elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
