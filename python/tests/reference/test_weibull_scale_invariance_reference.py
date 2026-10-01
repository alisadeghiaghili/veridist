"""Independent reference and scale-invariance contracts for the Weibull MLE.

Before this fix, `fit_weibull` searched for the log-shape over a hard-coded
`[-6, 6]` bracket and worked directly with `exp(shape * log(time))`, so the
same sample reported a different (and sometimes boundary-pinned) fit merely
because the recorded times were in a different unit. The profile search now
divides every time by the sample's geometric mean before optimizing, which
makes the recovered shape, scale and log-likelihood invariant to the time
unit (up to floating-point noise from representing `time * factor` itself,
not from the algorithm).
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from veridist.domain.lifetimes import ExactLifetime, RightCensoredLifetime
from veridist.families.weibull import fit_weibull

_SCALE_FACTORS = (1e-8, 1.0, 1e12)


def _censored_weibull_sample(
    *, seed: int, n: int, shape: float, scale: float, censor_at: float
) -> tuple[tuple[float, bool], ...]:
    rng = np.random.default_rng(seed)
    raw = scale * rng.weibull(shape, size=n)
    return tuple((float(min(t, censor_at)), bool(t <= censor_at)) for t in raw)


def _observations_at_scale(
    sample: tuple[tuple[float, bool], ...], factor: float
) -> tuple[ExactLifetime | RightCensoredLifetime, ...]:
    return tuple(
        ExactLifetime(time * factor) if is_event else RightCensoredLifetime(time * factor)
        for time, is_event in sample
    )


class WeibullScaleInvarianceReferenceTests(unittest.TestCase):
    def test_censored_fit_matches_an_independent_scipy_reference_at_every_scale(self) -> None:
        # Reference computed out-of-band with scipy 1.x in a scratch venv:
        # Nelder-Mead minimization of the negative log-likelihood (the same
        # censored Weibull likelihood this module implements) over
        # (log(shape), log(scale)), seeded from (0, 0), `xatol=fatol=1e-12`,
        # on `numpy.random.default_rng(1).weibull(1.7, size=300)` censored at
        # 1.2 (true shape 1.7, true scale 1.0). Result: shape ~1.733211347873287,
        # scale ~0.993470433132818 (225 of 300 observations were events).
        reference_shape = 1.733211347873287
        reference_scale = 0.993470433132818
        sample = _censored_weibull_sample(seed=1, n=300, shape=1.7, scale=1.0, censor_at=1.2)
        event_count = sum(1 for _, is_event in sample if is_event)
        self.assertEqual(event_count, 225)

        for factor in _SCALE_FACTORS:
            with self.subTest(factor=factor):
                result = fit_weibull(_observations_at_scale(sample, factor))
                self.assertTrue(hasattr(result, "shape"), result)
                self.assertAlmostEqual(
                    result.shape, reference_shape, delta=reference_shape * 1e-6
                )
                self.assertAlmostEqual(
                    result.scale / factor, reference_scale, delta=reference_scale * 1e-6
                )
                self.assertTrue(result.converged)

    def test_shape_scale_and_log_likelihood_are_invariant_to_the_time_unit(self) -> None:
        # A second, independently drawn sample (different seed/size) checked
        # at tighter tolerance: the geometric-mean rescaling is an exact
        # algebraic transformation, so the only remaining discrepancy between
        # scales is floating-point noise in representing `time * factor`
        # itself. This sample demonstrates that noise is far below 1e-9
        # relative for shape and scale.
        sample = _censored_weibull_sample(seed=10, n=80, shape=1.7, scale=1.0, censor_at=1.2)
        event_count = sum(1 for _, is_event in sample if is_event)
        self.assertEqual(event_count, 66)

        results = {
            factor: fit_weibull(_observations_at_scale(sample, factor))
            for factor in _SCALE_FACTORS
        }
        baseline = results[1.0]
        self.assertTrue(hasattr(baseline, "shape"), baseline)

        for factor in (1e-8, 1e12):
            with self.subTest(factor=factor):
                result = results[factor]
                self.assertTrue(hasattr(result, "shape"), result)
                relative_shape_error = abs(result.shape - baseline.shape) / baseline.shape
                self.assertLessEqual(relative_shape_error, 1e-9)
                relative_scale_error = abs(
                    result.scale / factor - baseline.scale
                ) / baseline.scale
                self.assertLessEqual(relative_scale_error, 1e-9)
                # The log-likelihood is not itself scale-invariant (a change
                # of time unit rescales the density via its Jacobian), but
                # the reported value must track that exactly: only exact
                # observations (not censored ones) contribute a Jacobian
                # term, so the shift is `-event_count * log(factor)`.
                expected_difference = -event_count * math.log(factor)
                actual_difference = result.log_likelihood - baseline.log_likelihood
                tolerance = abs(expected_difference) * 1e-9 + 1e-9
                self.assertAlmostEqual(actual_difference, expected_difference, delta=tolerance)


if __name__ == "__main__":
    unittest.main()
