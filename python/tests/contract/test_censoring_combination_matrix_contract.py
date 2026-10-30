"""The combination matrix, table driven over its full cross product (CENS-OBS-02).

Every censoring kind (exact, right, left, interval) x truncation type (none, left, right,
two-sided) x weight kind (none, integer frequency, analytic) is either supported by an operation
or refused with a typed code; nothing falls through.  The expected table below is written out
independently of the production table (every cell is supported except those with analytic
weights, for the log-likelihood entry points and, since fitting of censored and truncated data
is available, for the fits too) and then checked three ways: against
:func:`~veridist.domain._combinations.cell_status`, against the real log-likelihood entry points
and against the real fits, for a lifetime family and a real-line family.
"""

from __future__ import annotations

import unittest
from itertools import product
from math import isfinite
from typing import Any

from tests.reference import censored_support as support
from veridist import (
    CapabilityError,
    ExactLifetime,
    ExactValue,
    FamilyId,
    fit,
    reduce_lifetime_log_likelihood_chunks,
    reduce_value_log_likelihood_chunks,
)
from veridist.domain._combinations import (
    ANALYTIC_WEIGHTS_MESSAGE,
    FIT_UNAVAILABLE_MESSAGE,
    CellOperation,
    CensoringKind,
    TruncationKind,
    WeightKind,
    all_cells,
    cell_status,
    observation_cell,
)
from veridist.engine.errors import CapabilityCode

CENSORING = ("exact", "right", "left", "interval")
TRUNCATION = ("none", "left", "right", "two_sided")
WEIGHTS = ("none", "frequency", "analytic")
WINDOWS = {
    "none": None,
    "left": (2.0, float("inf")),
    "right": (float("-inf"), 9.0),
    "two_sided": (2.0, 9.0),
}
GROUPS = (
    (
        "weibull_min",
        {"shape": 1.5, "scale": 6.0},
        ExactLifetime,
        reduce_lifetime_log_likelihood_chunks,
    ),
    ("normal", {"mu": 4.0, "sigma": 2.0}, ExactValue, reduce_value_log_likelihood_chunks),
)


def expected(operation: str, censoring: str, truncation: str, weights: str) -> str | None:
    """The independent expectation: ``None`` when supported, else the refusal code."""

    if weights == "analytic":
        return "ANALYTIC_WEIGHTS_UNSUPPORTED"
    return None


def cell_observation(family: str, censoring: str, truncation: str) -> Any:
    window = WINDOWS[truncation]
    upper = 6.0 if censoring == "interval" else None
    point = 4.0 if censoring == "interval" else 5.0
    return support.observation(family, censoring, point, upper, window)


class TableTests(unittest.TestCase):
    def test_the_cross_product_has_forty_eight_cells_in_a_fixed_order(self) -> None:
        cells = all_cells()
        self.assertEqual(len(cells), 48)
        self.assertEqual(len(set(cells)), 48)
        self.assertEqual(
            [(c.value, t.value, w.value) for c, t, w in cells],
            list(product(CENSORING, TRUNCATION, WEIGHTS)),
        )

    def test_cell_status_equals_the_independent_table_for_both_operations(self) -> None:
        for operation in ("log_likelihood", "fit"):
            for censoring, truncation, weights in product(CENSORING, TRUNCATION, WEIGHTS):
                with self.subTest(operation=operation, cell=(censoring, truncation, weights)):
                    status = cell_status(
                        CellOperation(operation),
                        CensoringKind(censoring),
                        TruncationKind(truncation),
                        WeightKind(weights),
                    )
                    code = expected(operation, censoring, truncation, weights)
                    self.assertEqual(status.supported, code is None)
                    self.assertEqual(None if status.code is None else status.code.value, code)
                    if code is None:
                        self.assertIsNone(status.message)
                    else:
                        self.assertEqual(code, "ANALYTIC_WEIGHTS_UNSUPPORTED")
                        self.assertEqual(status.message, ANALYTIC_WEIGHTS_MESSAGE)

    def test_a_cell_is_made_of_the_matrix_enumerations_only(self) -> None:
        for arguments in (
            ("fit", CensoringKind.EXACT, TruncationKind.NONE, WeightKind.NONE),
            (CellOperation.FIT, "exact", TruncationKind.NONE, WeightKind.NONE),
            (CellOperation.FIT, CensoringKind.EXACT, "none", WeightKind.NONE),
            (CellOperation.FIT, CensoringKind.EXACT, TruncationKind.NONE, "none"),
        ):
            with self.assertRaises(TypeError):
                cell_status(*arguments)  # type: ignore[arg-type]

    def test_the_refusal_texts(self) -> None:
        self.assertEqual(
            FIT_UNAVAILABLE_MESSAGE, "not supported until fitting for this data kind is available"
        )
        self.assertEqual(
            ANALYTIC_WEIGHTS_MESSAGE,
            "analytic weights are not supported yet; they arrive in a later release",
        )

    def test_an_observation_is_classified_by_kind_and_window(self) -> None:
        for family, _, _, _ in GROUPS:
            for censoring, truncation in product(CENSORING, TRUNCATION):
                with self.subTest(family=family, censoring=censoring, truncation=truncation):
                    cell = observation_cell(cell_observation(family, censoring, truncation))
                    self.assertEqual((cell[0].value, cell[1].value), (censoring, truncation))
        with self.assertRaises(TypeError):
            observation_cell(1.0)


class LogLikelihoodMatrixTests(unittest.TestCase):
    def test_every_cell_is_supported_or_refused_as_the_table_says(self) -> None:
        for family, parameters, _, reducer in GROUPS:
            for censoring, truncation, weights in product(CENSORING, TRUNCATION, WEIGHTS):
                cell = (family, censoring, truncation, weights)
                with self.subTest(cell=cell):
                    chunk = [cell_observation(family, censoring, truncation)]
                    options: dict[str, Any] = {}
                    if weights == "frequency":
                        options["frequency_weights"] = [[3]]
                    if weights == "analytic":
                        options["analytic_weights"] = [[0.5]]
                    code = expected("log_likelihood", censoring, truncation, weights)
                    if code is None:
                        result = reducer(FamilyId(family), [chunk], **options, **parameters)
                        self.assertTrue(hasattr(result, "total_log_likelihood"), result)
                        self.assertTrue(isfinite(result.total_log_likelihood))
                        self.assertEqual(
                            result.observation_count, 3 if weights == "frequency" else 1
                        )
                    else:
                        with self.assertRaises(CapabilityError) as captured:
                            reducer(FamilyId(family), [chunk], **options, **parameters)
                        self.assertIs(captured.exception.code, CapabilityCode(code))
                        self.assertEqual(captured.exception.message, ANALYTIC_WEIGHTS_MESSAGE)

    def test_weights_that_are_not_non_negative_integers_are_refused(self) -> None:
        for family, parameters, _, reducer in GROUPS:
            chunk = [cell_observation(family, "exact", "none")]
            for bad, error in ((-1, TypeError), (1.5, TypeError), (True, TypeError)):
                with self.subTest(family=family, bad=bad):
                    with self.assertRaises(error):
                        reducer(FamilyId(family), [chunk], frequency_weights=[[bad]], **parameters)
            with self.assertRaises(ValueError):
                reducer(FamilyId(family), [chunk], frequency_weights=[[1, 1]], **parameters)
            with self.assertRaises(ValueError):
                reducer(FamilyId(family), [chunk], frequency_weights=[], **parameters)
            with self.assertRaises(ValueError):
                reducer(FamilyId(family), [chunk], frequency_weights=[[1], [1]], **parameters)


class FitMatrixTests(unittest.TestCase):
    def test_every_cell_is_supported_or_refused_as_the_table_says(self) -> None:
        for family, _, exact, _ in GROUPS:
            for censoring, truncation, weights in product(CENSORING, TRUNCATION, WEIGHTS):
                cell = (family, censoring, truncation, weights)
                with self.subTest(cell=cell):
                    sample = [exact(3.0), exact(5.0), exact(7.0)]
                    observations = [*sample, cell_observation(family, censoring, truncation)]
                    options: dict[str, Any] = {}
                    if weights == "frequency":
                        options["frequency_weights"] = [1, 2, 1, 1]
                    if weights == "analytic":
                        options["analytic_weights"] = [1.0, 1.0, 1.0, 1.0]
                    code = expected("fit", censoring, truncation, weights)
                    if code is None:
                        result = fit(family, observations, **options)
                        self.assertTrue(hasattr(result, "observation_count"))
                        summary = result.observation_summary
                        self.assertEqual(summary.total, result.observation_count)
                        new = censoring in ("left", "interval") or truncation != "none"
                        self.assertEqual(summary.is_new_data, new)
                    else:
                        with self.assertRaises(CapabilityError) as captured:
                            fit(family, observations, **options)
                        self.assertIs(captured.exception.code, CapabilityCode(code))

    def test_a_value_type_to_a_lifetime_family_or_the_reverse_is_still_a_type_error(self) -> None:
        for kind in CENSORING:
            for truncation in TRUNCATION:
                with self.subTest(kind=kind, truncation=truncation):
                    with self.assertRaises(TypeError):
                        fit("weibull_min", [cell_observation("normal", kind, truncation)])
                    with self.assertRaises(TypeError):
                        fit("normal", [cell_observation("weibull_min", kind, truncation)])

    def test_the_supported_fit_cells_have_their_earlier_results(self) -> None:
        plain = [ExactLifetime(3.0), ExactLifetime(5.0), ExactLifetime(7.0)]
        weighted = fit(
            "weibull_min", [ExactLifetime(3.0), ExactLifetime(7.0)], frequency_weights=[1, 1]
        )
        self.assertEqual(
            fit("weibull_min", plain + [ExactLifetime(4.0)]),
            fit("weibull_min", plain + [ExactLifetime(4.0)], frequency_weights=[1, 1, 1, 1]),
        )
        self.assertTrue(weighted.converged)


if __name__ == "__main__":
    unittest.main()
