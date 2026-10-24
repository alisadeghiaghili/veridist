"""The gamma fits on ordinary data are unchanged by the lower-tail log-density fix.

The expected numbers were produced by the implementation before the fix, with the data below:
the 25-point reference sample of ``test_new_fits_reference`` and a 40-point sample from a
linear congruential generator (so no library generator is involved) whose fitted shape is
about 100, which is in the range the fix touches for the log-density.  The log-density is not
used by ``fit_gamma``: its likelihood is assembled from sums of logarithms and, for censored
times, the incomplete-gamma log-survival, so the fits are bit-for-bit what they were.
"""

from __future__ import annotations

import unittest

from veridist.domain.lifetimes import ExactLifetime, LifetimeObservation, RightCensoredLifetime
from veridist.families.gamma import GammaFitSuccess, fit_gamma

GAMMA_DATA = (
    8.504, 18.042, 10.226, 3.112, 8.333, 2.251, 8.956, 15.848, 7.33, 9.592,
    12.179, 0.416, 15.786, 18.654, 7.692, 1.47, 24.849, 5.343, 2.069, 6.589,
    9.366, 6.347, 7.002, 3.888, 10.515,
)  # fmt: skip


def _lcg_sample(count: int) -> tuple[float, ...]:
    state = 12345
    values = []
    for _ in range(count):
        state = (1103515245 * state + 12345) % 2**31
        values.append(round(6.0 + 3.0 * (state / 2**31), 6))
    return tuple(values)


TIGHT_DATA = _lcg_sample(40)

#: ``(data, censoring point, shape, scale, log-likelihood)`` from the pre-fix implementation.
RECORDED_FITS = (
    (GAMMA_DATA, None, 1.8736867467768328, 4.789680033462338, -77.41990743392316),
    (GAMMA_DATA, 14.0, 1.7324507637371105, 5.387995552927517, -64.60926965520176),
    (TIGHT_DATA, None, 100.3782479461872, 0.07489651920434386, -45.13737549258501),
    (TIGHT_DATA, 8.0, 80.66717257994178, 0.0940210807198613, -44.642073718986154),
)


def _observations(data: tuple[float, ...], censoring: float | None) -> list[LifetimeObservation]:
    return [
        RightCensoredLifetime(censoring)
        if censoring is not None and time > censoring
        else ExactLifetime(time)
        for time in data
    ]


class GammaFitRegressionTests(unittest.TestCase):
    def test_the_sample_generator_is_the_recorded_one(self) -> None:
        self.assertEqual(TIGHT_DATA[:3], (7.965462, 6.914443, 8.024882))

    def test_fitted_parameters_and_log_likelihood_are_unchanged(self) -> None:
        for data, censoring, shape, scale, log_likelihood in RECORDED_FITS:
            with self.subTest(size=len(data), censoring=censoring):
                result = fit_gamma(_observations(data, censoring))
                assert isinstance(result, GammaFitSuccess)
                # The search is deterministic; the slack only absorbs libm differences between
                # platforms, which an optimizer that stops at a tolerance can amplify.
                self.assertAlmostEqual(result.shape / shape, 1.0, delta=1e-7)
                self.assertAlmostEqual(result.scale / scale, 1.0, delta=1e-7)
                self.assertAlmostEqual(result.log_likelihood, log_likelihood, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
