"""Goodness of fit still refuses censored and truncated data, with a typed reason (CENS-GOF-01).

The refit Monte Carlo and ``assess_families`` take finite, exactly observed real numbers and
nothing else: any observation object, of any of the eight kinds and with or without a
truncation window, is a ``TypeError`` with the text of the goodness-of-fit record, raised before
the first fit and before the generator is touched.  ``large_n_gof`` needs a successful scale
fit, so the failure of an in-memory fit of such data, or an in-memory fit of any kind, is
refused with its typed code.  Fitting such data is available (``test_censored_fits_contract``);
checking it by simulation is a separate piece of work.
"""

from __future__ import annotations

import unittest
from itertools import product
from types import SimpleNamespace

import numpy as np

from tests.reference import censored_support as support
from veridist import fit_normal
from veridist.inference import GofStatistic, assess_families, refit_monte_carlo_gof
from veridist.scale import NumpySource, large_n_gof
from veridist.scale._errors import ScaleInferenceError, ScaleInferenceErrorCode

KINDS = ("exact", "right", "left", "interval")
WINDOWS = (None, (1.0, float("inf")), (float("-inf"), 9.0), (1.0, 9.0))
FAMILIES = ("exponential", "gamma", "weibull_min", "lognormal", "normal", "gumbel_right")
PLAIN = [3.1, 4.2, 2.7, 5.5, 3.9, 4.8, 2.2, 3.3]
MESSAGE = "observations must be plain real numbers"


def observations(lifetime: bool, kind: str, window: tuple[float, float] | None) -> list[object]:
    family = "weibull_min" if lifetime else "normal"
    upper = 6.0 if kind == "interval" else None
    row = support.observation(family, kind, 4.0, upper, window)
    return [*PLAIN, row]


class RefusalTests(unittest.TestCase):
    def test_every_observation_kind_and_window_is_a_type_error(self) -> None:
        for family, (kind, window) in product(FAMILIES, product(KINDS, WINDOWS)):
            lifetime = family in ("exponential", "gamma", "weibull_min", "lognormal")
            rows = observations(lifetime, kind, window)
            with self.subTest(family=family, kind=kind, window=window):
                rng = np.random.default_rng(1)
                state = rng.bit_generator.state
                with self.assertRaises(TypeError) as captured:
                    refit_monte_carlo_gof(
                        observations=rows,
                        family=family,
                        statistics=frozenset({GofStatistic.AD}),
                        replicates=3,
                        rng=rng,
                    )
                self.assertIn(MESSAGE, str(captured.exception))
                self.assertIn("not supported by refit goodness-of-fit", str(captured.exception))
                self.assertEqual(rng.bit_generator.state, state)

    def test_assess_families_refuses_them_before_any_fit(self) -> None:
        for kind, window in product(KINDS, WINDOWS):
            rows = observations(True, kind, window)
            with self.subTest(kind=kind, window=window):
                rng = np.random.default_rng(2)
                state = rng.bit_generator.state
                with self.assertRaises(TypeError) as captured:
                    assess_families(observations=rows, replicates=3, rng=rng)
                self.assertIn(MESSAGE, str(captured.exception))
                self.assertEqual(rng.bit_generator.state, state)

    def test_plain_numbers_are_still_accepted(self) -> None:
        rng = np.random.default_rng(3)
        result = refit_monte_carlo_gof(
            observations=PLAIN,
            family="normal",
            statistics=frozenset({GofStatistic.AD}),
            replicates=4,
            rng=rng,
        )
        self.assertEqual(result.requested_replicates, 4)

    def test_the_failure_of_an_in_memory_fit_is_not_a_scale_fit(self) -> None:
        failure = fit_normal([support.observation("normal", "left", 4.0, None, None)])
        self.assertTrue(hasattr(failure, "code"))
        source = NumpySource(np.array(PLAIN))
        for result in (SimpleNamespace(fit=failure), SimpleNamespace(fit=None)):
            with self.assertRaises(ScaleInferenceError) as captured:
                large_n_gof(source, result)  # type: ignore[arg-type]
            self.assertIs(captured.exception.code, ScaleInferenceErrorCode.GOF_FIT_NOT_AVAILABLE)


if __name__ == "__main__":
    unittest.main()
