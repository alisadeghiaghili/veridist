"""Contract of the execution record of a scale fit: the closed names, the checks, the JSON."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace
from typing import Any

from tests.contract.test_scale_provenance_schema import passes, record
from veridist.engine.provenance import (
    ScaleExecutionPassProvenance,
    ScaleExecutionProvenance,
    scale_provenance_to_canonical_json_bytes,
)

REASONS = (
    "single_partition",
    "single_use_source",
    "engine_managed",
    "in_memory_data",
    "undeclared_source",
    "spool_first_pass",
)


def execution_pass(**changes: Any) -> ScaleExecutionPassProvenance:
    base = ScaleExecutionPassProvenance("threads", 4, 6, 1, None)
    return replace(base, **changes)


def execution(**changes: Any) -> ScaleExecutionProvenance:
    base = ScaleExecutionProvenance(
        "threads", 8, (execution_pass(), execution_pass(workers=3, retries=0))
    )
    return replace(base, **changes)


def serial_pass(reason: str | None = None, **changes: Any) -> ScaleExecutionPassProvenance:
    return execution_pass(executor="serial", workers=1, downgrade=reason, **changes)


class ExecutionPassRecordTests(unittest.TestCase):
    def test_a_valid_pass_keeps_its_fields(self) -> None:
        item = execution_pass()
        self.assertEqual(
            (item.executor, item.workers, item.tasks, item.retries, item.downgrade),
            ("threads", 4, 6, 1, None),
        )
        with self.assertRaises(FrozenInstanceError):
            item.workers = 2  # type: ignore[misc]

    def test_the_executor_is_one_of_three_names(self) -> None:
        for name in ("serial", "threads", "processes"):
            execution_pass(executor=name, workers=1)
        for bad in ("thread", "Serial", "", "fork"):
            with self.subTest(executor=bad), self.assertRaises(ValueError):
                execution_pass(executor=bad)

    def test_counts_are_validated(self) -> None:
        for bad in (0, -1, True, 1.5):
            with self.subTest(workers=bad), self.assertRaises((TypeError, ValueError)):
                execution_pass(workers=bad)
        for field in ("tasks", "retries"):
            self.assertEqual(getattr(execution_pass(**{field: 0}), field), 0)
            for bad in (-1, True, 1.5):
                with self.subTest(field=field, value=bad):
                    with self.assertRaises((TypeError, ValueError)):
                        execution_pass(**{field: bad})

    def test_the_downgrade_is_a_known_reason_on_a_serial_pass(self) -> None:
        for reason in REASONS:
            self.assertEqual(serial_pass(reason).downgrade, reason)
        for bad in ("other", "", "SINGLE_PARTITION"):
            with self.subTest(downgrade=bad), self.assertRaises(ValueError):
                serial_pass(bad)
        with self.assertRaises(ValueError):
            execution_pass(executor="threads", workers=2, downgrade="engine_managed")

    def test_a_serial_pass_has_one_worker(self) -> None:
        serial_pass()
        with self.assertRaises(ValueError):
            execution_pass(executor="serial", workers=2)


class ExecutionRecordTests(unittest.TestCase):
    def test_a_valid_record_keeps_its_fields_and_derives_the_totals(self) -> None:
        item = execution()
        self.assertEqual((item.executor, item.requested_workers), ("threads", 8))
        self.assertEqual(len(item.passes), 2)
        self.assertEqual(item.effective_workers, 4)
        self.assertEqual(item.retries, 1)
        self.assertIsNone(item.downgrade)
        with self.assertRaises(FrozenInstanceError):
            item.executor = "serial"  # type: ignore[misc]

    def test_the_totals_of_several_passes(self) -> None:
        item = execution(
            passes=(
                serial_pass("spool_first_pass", retries=2),
                execution_pass(workers=6, retries=3),
                serial_pass("engine_managed"),
            )
        )
        self.assertEqual(item.effective_workers, 6)
        self.assertEqual(item.retries, 6)
        self.assertEqual(item.downgrade, "spool_first_pass")

    def test_a_record_without_passes_has_one_worker_and_no_retries(self) -> None:
        item = execution(passes=())
        self.assertEqual((item.effective_workers, item.retries, item.downgrade), (1, 0, None))

    def test_the_request_is_validated(self) -> None:
        with self.assertRaises(ValueError):
            execution(executor="fork")
        for bad in (0, -2, True, 2.0):
            with self.subTest(workers=bad), self.assertRaises((TypeError, ValueError)):
                execution(requested_workers=bad)
        self.assertEqual(ScaleExecutionProvenance("serial", 1, (serial_pass(),)).executor, "serial")
        with self.assertRaises(ValueError):
            ScaleExecutionProvenance("serial", 2, ())

    def test_passes_must_be_a_tuple_of_pass_records(self) -> None:
        with self.assertRaises(TypeError):
            execution(passes=[execution_pass()])
        with self.assertRaises(TypeError):
            execution(passes=("threads",))

    def test_a_pass_cannot_have_more_workers_than_were_asked_for(self) -> None:
        execution(requested_workers=4)
        with self.assertRaises(ValueError):
            execution(requested_workers=3)

    def test_a_pass_that_was_not_downgraded_used_the_requested_executor(self) -> None:
        with self.assertRaises(ValueError):
            execution(passes=(execution_pass(executor="processes"),))
        execution(passes=(serial_pass("single_partition"),))

    def test_the_serial_executor_is_never_downgraded(self) -> None:
        with self.assertRaises(ValueError):
            ScaleExecutionProvenance("serial", 1, (serial_pass("single_partition"),))


class ScaleExecutionFieldTests(unittest.TestCase):
    def test_the_field_defaults_to_none_and_is_typed(self) -> None:
        self.assertIsNone(record().execution)
        with self.assertRaises(TypeError):
            record(execution="threads")
        with self.assertRaises(TypeError):
            record(execution=execution_pass())

    def test_a_one_pass_record_has_one_entry(self) -> None:
        item = execution(passes=(execution_pass(),))
        self.assertEqual(record(execution=item).execution, item)
        with self.assertRaises(ValueError):
            record(execution=execution())
        with self.assertRaises(ValueError):
            record(execution=execution(passes=()))

    def test_a_multi_pass_record_has_one_entry_per_pass(self) -> None:
        recorded = passes()
        entries = tuple(execution_pass() for _ in range(recorded.pass_count))
        item = execution(passes=entries)
        self.assertEqual(record(passes=recorded, execution=item).execution, item)
        with self.assertRaises(ValueError):
            record(passes=recorded, execution=execution(passes=entries[:-1]))
        with self.assertRaises(ValueError):
            record(passes=recorded, execution=execution(passes=(*entries, execution_pass())))

    def test_the_canonical_json_has_the_record_and_omits_it_when_absent(self) -> None:
        plain = json.loads(scale_provenance_to_canonical_json_bytes(record()))
        self.assertNotIn("execution", plain)
        item = execution(passes=(serial_pass("engine_managed"),))
        decoded = json.loads(scale_provenance_to_canonical_json_bytes(record(execution=item)))
        self.assertEqual(
            decoded["execution"],
            {
                "downgrade": "engine_managed",
                "effective_workers": 1,
                "executor": "threads",
                "passes": [
                    {
                        "downgrade": "engine_managed",
                        "executor": "serial",
                        "retries": 1,
                        "tasks": 6,
                        "workers": 1,
                    }
                ],
                "requested_workers": 8,
                "retries": 1,
            },
        )
        del decoded["execution"]
        self.assertEqual(decoded, plain)

    def test_the_json_is_compact_and_sorted(self) -> None:
        item = execution(passes=(execution_pass(),))
        encoded = scale_provenance_to_canonical_json_bytes(record(execution=item))
        self.assertEqual(
            encoded, json.dumps(json.loads(encoded), sort_keys=True, separators=(",", ":")).encode()
        )
