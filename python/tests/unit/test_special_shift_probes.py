"""Exact, millisecond-fast pins on the digamma/trigamma recurrence shift.

``_shifted`` fixes how many terms of the recurrence run before the asymptotic series
takes over. A change that keeps digamma and trigamma mathematically correct but makes
the step count grow with the argument (for example ``_ASYMPTOTIC_FROM + x``) is not
caught by a value check at small arguments, and at a large argument it turns a
millisecond call into a minutes-long one. These tests pin the exact step count, the
exact shifted point and the exact terms, so every such change fails immediately and
cheaply, and they bound the work done at a large argument without running it.
"""

from __future__ import annotations

import unittest
from itertools import islice
from unittest.mock import patch

from veridist.statistics import _special
from veridist.statistics._special import _ASYMPTOTIC_FROM, _shifted, digamma, trigamma

#: ``(x, shifted point, number of recurrence terms)`` with ``terms == (x + i for i in range(n))``.
_SHIFT_CASES = (
    (0.5, 20.5, 20),
    (1.0, 21.0, 20),
    (2.5, 20.5, 18),
    (10.0, 21.0, 11),
    (19.0, 21.0, 2),
    (19.5, 20.5, 1),
    (20.0, 21.0, 1),
    (20.5, 21.5, 1),
    (20.999, 21.999, 1),
    (21.0, 21.0, 0),
    (25.0, 25.0, 0),
    (1.0e3, 1.0e3, 0),
)

#: mpmath ``digamma`` / ``psi(1, .)`` at 40 digits, rounded to binary64.
_REFERENCE = (
    (0.5, -1.9635100260214235, 4.934802200544679),
    (1.0, -0.5772156649015329, 1.6449340668482264),
    (2.5, 0.7031566406452432, 0.49035775610023485),
    (19.5, 2.944554343425595, 0.05261944121365593),
    (20.0, 2.970523992242149, 0.05127082293520312),
    (25.0, 3.198742512851974, 0.04081066325722558),
    (1.0e6, 13.815510057964191, 1.0000005000001667e-06),
)

#: The recurrence never needs more than this many terms for a positive argument.
_MAX_TERMS = 21


class ShiftedExactnessTests(unittest.TestCase):
    """Pin ``_shifted`` exactly: count, shifted point and every term."""

    def test_asymptotic_threshold_is_twenty(self) -> None:
        self.assertEqual(_ASYMPTOTIC_FROM, 20.0)

    def test_shifted_point_and_terms_are_exact(self) -> None:
        for x, top, count in _SHIFT_CASES:
            with self.subTest(x=x):
                shifted, below = _shifted(x)
                self.assertEqual(len(below), count)
                self.assertEqual(shifted, top)
                self.assertEqual(below, tuple(x + float(i) for i in range(count)))

    def test_shifted_point_reaches_the_threshold_and_is_the_last_term_plus_one(self) -> None:
        for x, _, _ in _SHIFT_CASES:
            with self.subTest(x=x):
                shifted, below = _shifted(x)
                self.assertGreaterEqual(shifted, _ASYMPTOTIC_FROM)
                self.assertEqual(shifted, x + len(below))
                if below:
                    # The recurrence stops at the first point at or above the threshold.
                    self.assertEqual(shifted, below[-1] + 1.0)
                    self.assertLess(below[-1], _ASYMPTOTIC_FROM + 1.0)

    def test_the_step_count_does_not_grow_with_the_argument(self) -> None:
        for x in (21.0, 25.0, 100.0, 1.0e3, 1.0e6):
            with self.subTest(x=x):
                shifted, below = _shifted(x)
                self.assertEqual(below, ())
                self.assertEqual(shifted, x)

    def test_the_step_count_is_bounded_for_every_positive_argument(self) -> None:
        for x in (1.0e-300, 5.0e-324, 1.0e-12, 0.25, 0.5, 1.0, 7.0, 19.999999, 20.0):
            with self.subTest(x=x):
                shifted, below = _shifted(x)
                self.assertLessEqual(len(below), _MAX_TERMS)
                self.assertGreaterEqual(len(below), 1)
                self.assertGreaterEqual(shifted, _ASYMPTOTIC_FROM)

    def test_tiny_arguments_use_the_full_recurrence(self) -> None:
        shifted, below = _shifted(1.0e-300)
        self.assertEqual(len(below), 21)
        self.assertEqual(below[0], 1.0e-300)
        self.assertEqual(below[1:], tuple(float(i) for i in range(1, 21)))
        self.assertEqual(shifted, 21.0)


class SpecialFunctionWorkTests(unittest.TestCase):
    """Bound the recurrence work and check the values it must produce."""

    def _recurrence_term_counts(self, function: object, x: float) -> list[int]:
        counts: list[int] = []
        real = _special.fsum

        def counting(items: object) -> float:
            # Read at most one more than the limit so that a runaway count is
            # reported instead of being summed.
            taken = tuple(islice(items, _MAX_TERMS + 2))  # type: ignore[call-overload]
            counts.append(len(taken) - 1)  # the first item is the series tail
            return real(taken)

        with patch.object(_special, "fsum", counting):
            function(x)  # type: ignore[operator]
        return counts

    def test_digamma_and_trigamma_sum_a_bounded_number_of_terms(self) -> None:
        for function in (digamma, trigamma):
            for x, _, count in _SHIFT_CASES:
                with self.subTest(function=function.__name__, x=x):
                    self.assertEqual(self._recurrence_term_counts(function, x), [count])

    def test_large_arguments_run_no_recurrence_at_all(self) -> None:
        for function in (digamma, trigamma):
            with self.subTest(function=function.__name__):
                self.assertEqual(self._recurrence_term_counts(function, 1.0e6), [0])

    def test_values_match_the_reference(self) -> None:
        for x, psi, psi_prime in _REFERENCE:
            with self.subTest(x=x):
                self.assertAlmostEqual(digamma(x) / psi, 1.0, delta=1.0e-13)
                self.assertAlmostEqual(trigamma(x) / psi_prime, 1.0, delta=1.0e-13)

    def test_the_recurrences_hold_across_the_threshold(self) -> None:
        for x in (0.5, 1.0, 5.5, 19.0, 19.5, 20.0, 20.5, 25.0):
            with self.subTest(x=x):
                self.assertAlmostEqual(digamma(x + 1.0) - digamma(x), 1.0 / x, delta=2.0e-15 * 20)
                self.assertAlmostEqual(
                    trigamma(x) - trigamma(x + 1.0), 1.0 / (x * x), delta=2.0e-15 * 20
                )

    def test_invalid_arguments_are_refused_before_any_shift(self) -> None:
        for function in (digamma, trigamma):
            for bad in (0.0, -1.0, float("nan")):
                with self.subTest(function=function.__name__, x=bad):
                    with self.assertRaises(ValueError):
                        function(bad)


if __name__ == "__main__":
    unittest.main()
