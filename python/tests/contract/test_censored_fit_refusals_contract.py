"""What the fits still refuse, and where censored and truncated data is not taken.

The fits take left- and interval-censored observations and truncation windows (see
``test_censored_fits_contract``); what stays refused is pinned here.  The exponential reducers
(``ExponentialReductionState``, ``reduce_exponential_chunks``) keep refusing them with the typed
``DATA_KIND_FIT_UNAVAILABLE`` and the reason "not supported until fitting for this data kind is
available": their state holds the number of events and the total time alone, which is not the
sufficient statistic of such data.  The ``censoring=`` and ``truncation=`` keywords keep their
exception types and codes and say, in their message, to declare censoring and truncation with
the typed observations or the column constructors (CENS-PROV-01).  Analytic weights and a fixed
Weibull shape with such data are refused with typed codes, and goodness of fit and the scale
engine do not take observation objects at all (CENS-GOF-01).
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from veridist import (
    CapabilityError,
    ExactLifetime,
    ExactValue,
    IntervalCensoredLifetime,
    IntervalCensoredValue,
    LeftCensoredLifetime,
    LeftCensoredValue,
    RightCensoredLifetime,
    RightCensoredValue,
    fit_exponential,
    fit_gamma,
    fit_gumbel_right,
    fit_lognormal,
    fit_normal,
    fit_weibull,
    scale,
)
from veridist.engine.errors import CapabilityCode
from veridist.inference import assess_families, refit_monte_carlo_gof
from veridist.statistics.exponential import ExponentialReductionState, reduce_exponential_chunks

UNAVAILABLE = (
    "DATA_KIND_FIT_UNAVAILABLE: not supported until fitting for this data kind is available"
)
LIFETIME_FITS = (
    ("exponential", fit_exponential),
    ("weibull_min", fit_weibull),
    ("lognormal", fit_lognormal),
    ("gamma", fit_gamma),
)
VALUE_FITS = (("normal", fit_normal), ("gumbel_right", fit_gumbel_right))


def refused_lifetimes() -> list[Any]:
    return [
        LeftCensoredLifetime(2.0),
        IntervalCensoredLifetime(1.0, 2.0),
        ExactLifetime(3.0, truncation_lower=1.0),
        ExactLifetime(3.0, truncation_upper=9.0),
        RightCensoredLifetime(3.0, truncation_lower=1.0, truncation_upper=9.0),
    ]


def refused_values() -> list[Any]:
    return [
        LeftCensoredValue(2.0),
        IntervalCensoredValue(1.0, 2.0),
        ExactValue(3.0, truncation_lower=1.0),
        ExactValue(3.0, truncation_upper=9.0),
        RightCensoredValue(3.0, truncation_lower=1.0, truncation_upper=9.0),
    ]


class FitRefusalTests(unittest.TestCase):
    def assert_unavailable(self, call: Any) -> None:
        with self.assertRaises(CapabilityError) as captured:
            call()
        self.assertIs(captured.exception.code, CapabilityCode.DATA_KIND_FIT_UNAVAILABLE)
        self.assertEqual(str(captured.exception), UNAVAILABLE)

    def test_a_fixed_weibull_shape_is_refused_for_the_new_data_kinds_only(self) -> None:
        for row in refused_lifetimes():
            data = [ExactLifetime(3.0), ExactLifetime(5.0), row]
            with self.subTest(row=row):
                with self.assertRaises(CapabilityError) as captured:
                    fit_weibull(data, fixed_shape=1.5)
                self.assertIs(captured.exception.code, CapabilityCode.DATA_KIND_FIT_UNAVAILABLE)
                self.assertIn("fixed_shape", str(captured.exception))
        plain = fit_weibull([ExactLifetime(3.0), ExactLifetime(5.0)], fixed_shape=1.5)
        self.assertEqual(plain.shape, 1.5)

    def test_analytic_weights_are_refused_for_every_data_kind(self) -> None:
        for row in refused_lifetimes():
            data = [ExactLifetime(3.0), ExactLifetime(5.0), row]
            with self.subTest(row=row):
                with self.assertRaises(CapabilityError) as captured:
                    fit_weibull(data, analytic_weights=[1.0, 1.0, 1.0])
                self.assertIs(captured.exception.code, CapabilityCode.ANALYTIC_WEIGHTS_UNSUPPORTED)

    def test_the_exponential_reducers_refuse_them_too_even_after_an_overflow(self) -> None:
        for row in refused_lifetimes():
            with self.subTest(row=row):
                self.assert_unavailable(lambda r=row: ExponentialReductionState.empty().add(r))
                self.assert_unavailable(lambda r=row: reduce_exponential_chunks([[r]]))
                overflow = [ExactLifetime(1e308), ExactLifetime(1e308), ExactLifetime(1e308)]
                self.assert_unavailable(
                    lambda r=row, o=overflow: reduce_exponential_chunks([o, [r]])
                )

    def test_the_wrong_group_is_still_a_type_error_before_any_capability_check(self) -> None:
        for row in (*refused_values(), ExactValue(3.0)):
            with self.subTest(row=row):
                with self.assertRaises(TypeError):
                    fit_weibull([ExactLifetime(3.0), row])
                with self.assertRaises(TypeError):
                    fit_exponential([ExactLifetime(3.0), row])
        for row in (*refused_lifetimes(), ExactLifetime(3.0)):
            with self.assertRaises(TypeError):
                fit_normal([ExactValue(3.0), row])

    def test_supported_data_is_unchanged(self) -> None:
        data = [ExactLifetime(3.0), ExactLifetime(5.0), RightCensoredLifetime(6.0)]
        self.assertEqual(fit_exponential(data).observation_count, 3)
        self.assertTrue(fit_weibull(data + [ExactLifetime(4.0)]).converged)

    def test_goodness_of_fit_and_the_scale_engine_do_not_take_observation_objects(self) -> None:
        rng = np.random.default_rng(1)
        rows = [LeftCensoredLifetime(2.0), ExactLifetime(3.0), ExactLifetime(4.0)]
        with self.assertRaises(TypeError) as captured:
            assess_families(observations=rows, replicates=5, rng=rng)
        self.assertIn("observations must be plain real numbers", str(captured.exception))
        with self.assertRaises(TypeError):
            refit_monte_carlo_gof(
                observations=rows,
                family="weibull_min",
                statistics=frozenset(),
                replicates=5,
                rng=rng,
            )
        with self.assertRaises(TypeError):
            scale.fit(tuple(rows), "weibull_min")


class KeywordMessageTests(unittest.TestCase):
    def test_the_keywords_keep_their_codes_and_point_to_the_constructors(self) -> None:
        data = [ExactLifetime(3.0), ExactLifetime(5.0)]
        cases = (
            ({"censoring": "left"}, CapabilityCode.LEFT_CENSORING_UNSUPPORTED, "censoring"),
            ({"censoring": "interval"}, CapabilityCode.INTERVAL_CENSORING_UNSUPPORTED, "censoring"),
            ({"truncation": (0.5, None)}, CapabilityCode.TRUNCATION_UNSUPPORTED, "truncation"),
        )
        for function in (fit_weibull, fit_lognormal, fit_gamma):
            for keywords, code, name in cases:
                with self.subTest(fit=function.__name__, keywords=keywords):
                    with self.assertRaises(CapabilityError) as captured:
                        function(data, **keywords)
                    error = captured.exception
                    self.assertIs(error.code, code)
                    self.assertTrue(str(error).startswith(code.value + ": "))
                    assert error.message is not None
                    self.assertIn(f"the {name} keyword", error.message)
                    for fragment in (
                        "LeftCensoredLifetime",
                        "IntervalCensoredLifetime",
                        "truncation_lower",
                        "lifetimes_from_columns",
                        "values_from_columns",
                    ):
                        self.assertIn(fragment, error.message)
        for function in (fit_normal, fit_gumbel_right):
            with self.assertRaises(CapabilityError) as captured:
                function([ExactValue(1.0), ExactValue(2.0)], censoring="left")
            self.assertIs(captured.exception.code, CapabilityCode.LEFT_CENSORING_UNSUPPORTED)

    def test_the_censoring_keyword_message_says_what_it_takes(self) -> None:
        with self.assertRaises(CapabilityError) as captured:
            fit_weibull([ExactLifetime(3.0)], censoring="left")
        self.assertIn("takes only 'right'", str(captured.exception))

    def test_analytic_weights_say_they_arrive_later(self) -> None:
        with self.assertRaises(CapabilityError) as captured:
            fit_weibull([ExactLifetime(3.0)], analytic_weights=[1.0])
        self.assertIs(captured.exception.code, CapabilityCode.ANALYTIC_WEIGHTS_UNSUPPORTED)
        self.assertEqual(
            str(captured.exception),
            "ANALYTIC_WEIGHTS_UNSUPPORTED: analytic weights are not supported yet; "
            "they arrive in a later release",
        )

    def test_an_unknown_censoring_value_is_still_a_value_error(self) -> None:
        with self.assertRaises(ValueError):
            fit_weibull([ExactLifetime(3.0)], censoring="sideways")

    def test_a_capability_error_without_a_message_is_just_its_code(self) -> None:
        error = CapabilityError(CapabilityCode.TRUNCATION_UNSUPPORTED)
        self.assertEqual(error.args, ("TRUNCATION_UNSUPPORTED",))
        self.assertIsNone(error.message)
        with_message = CapabilityError(CapabilityCode.TRUNCATION_UNSUPPORTED, "why")
        self.assertEqual(str(with_message), "TRUNCATION_UNSUPPORTED: why")
        self.assertEqual(with_message.message, "why")
        with self.assertRaises(TypeError):
            CapabilityError(CapabilityCode.TRUNCATION_UNSUPPORTED, 3)  # type: ignore[arg-type]

    def test_the_new_code_is_a_member_of_the_closed_set(self) -> None:
        self.assertEqual(
            {code.value for code in CapabilityCode},
            {
                "ANALYTIC_WEIGHTS_UNSUPPORTED",
                "DATA_KIND_FIT_UNAVAILABLE",
                "INTERVAL_CENSORING_UNSUPPORTED",
                "LEFT_CENSORING_UNSUPPORTED",
                "TRUNCATION_UNSUPPORTED",
            },
        )


if __name__ == "__main__":
    unittest.main()
