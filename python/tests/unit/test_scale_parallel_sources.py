"""What a source says about parallel reads, and the partitioned ``NumpySource``."""

from __future__ import annotations

import hashlib
import pickle
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import write_parquet
from tests.unit.scale_source_oracle import reference_digest
from veridist.engine import provenance as engine_provenance
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._errors import (
    ScaleSourceError,
    ScaleSourceErrorCode,
    ScaleStateError,
    ScaleStateErrorCode,
)
from veridist.scale._one_pass import ExponentialState
from veridist.scale._sources import (
    DowngradeReason,
    ExecutorKind,
    FingerprintLevel,
    NumpySource,
    Partition,
    fold,
    is_retryable,
    parallel_obstacle,
)

KINDS = (ExecutorKind.THREADS, ExecutorKind.PROCESSES)


class VocabularyTests(unittest.TestCase):
    def test_the_kinds_and_the_reasons_are_closed(self) -> None:
        self.assertEqual([kind.value for kind in ExecutorKind], ["serial", "threads", "processes"])
        self.assertEqual(
            [reason.value for reason in DowngradeReason],
            [
                "single_partition",
                "single_use_source",
                "engine_managed",
                "in_memory_data",
                "undeclared_source",
                "spool_first_pass",
            ],
        )

    def test_the_provenance_record_accepts_exactly_these_names(self) -> None:
        self.assertEqual(
            engine_provenance._SCALE_EXECUTORS, frozenset(kind.value for kind in ExecutorKind)
        )
        self.assertEqual(
            engine_provenance._SCALE_DOWNGRADES,
            frozenset(reason.value for reason in DowngradeReason),
        )


class HelperTests(unittest.TestCase):
    def test_a_source_without_the_method_is_undeclared(self) -> None:
        for kind in KINDS:
            self.assertIs(parallel_obstacle(object(), kind, ()), DowngradeReason.UNDECLARED_SOURCE)

    def test_the_answer_is_the_sources_and_it_is_asked_about_the_kind_and_partitions(self) -> None:
        seen: list[tuple[Any, Any]] = []

        class Source:
            def parallel_obstacle(self, kind: Any, partitions: Any) -> Any:
                seen.append((kind, partitions))
                return DowngradeReason.ENGINE_MANAGED if kind is ExecutorKind.THREADS else None

        partitions = (Partition("a", 0, 1, None, FingerprintLevel.NONE),)
        self.assertIs(
            parallel_obstacle(Source(), ExecutorKind.THREADS, partitions),
            DowngradeReason.ENGINE_MANAGED,
        )
        self.assertIsNone(parallel_obstacle(Source(), ExecutorKind.PROCESSES, partitions))
        self.assertEqual(
            seen, [(ExecutorKind.THREADS, partitions), (ExecutorKind.PROCESSES, partitions)]
        )

    def test_retryable_unless_the_source_says_otherwise_or_is_single_use(self) -> None:
        class Plain:
            pass

        class Flagged:
            def __init__(self, retryable: object, single_use: object = False) -> None:
                self.retryable = retryable
                self.single_use = single_use

        self.assertTrue(is_retryable(Plain()))
        self.assertTrue(is_retryable(Flagged(True)))
        self.assertFalse(is_retryable(Flagged(False)))
        self.assertFalse(is_retryable(Flagged(True, True)))
        self.assertFalse(is_retryable(Flagged(False, True)))
        self.assertTrue(is_retryable(Flagged(True, 1)))
        self.assertFalse(is_retryable(Flagged(0)))
        self.assertFalse(is_retryable(Flagged(1)))


class SourceDeclarationTests(unittest.TestCase):
    def test_numpy_data_is_read_in_place_by_threads_and_never_copied_to_processes(self) -> None:
        source = NumpySource(np.arange(1.0, 5.0))
        parts = source.partitions()
        self.assertIsNone(source.parallel_obstacle(ExecutorKind.THREADS, parts))
        self.assertIs(
            source.parallel_obstacle(ExecutorKind.PROCESSES, parts), DowngradeReason.IN_MEMORY_DATA
        )
        self.assertTrue(is_retryable(source))

    def test_arrow_data_in_memory_is_read_by_threads_only(self) -> None:
        for data in (pa.table({"x": [1.0, 2.0]}), pa.table({"x": [1.0, 2.0]}).to_batches()[0]):
            source = ArrowSource(data, "x")
            with self.subTest(data=type(data).__name__):
                self.assertIsNone(source.parallel_obstacle(ExecutorKind.THREADS, ()))
                self.assertIs(
                    source.parallel_obstacle(ExecutorKind.PROCESSES, ()),
                    DowngradeReason.IN_MEMORY_DATA,
                )
                self.assertTrue(is_retryable(source))

    def test_an_arrow_stream_cannot_be_shared_and_is_not_retried(self) -> None:
        table = pa.table({"x": [1.0, 2.0]})
        source = ArrowSource(
            pa.RecordBatchReader.from_batches(table.schema, table.to_batches()), "x"
        )
        for kind in KINDS:
            with self.subTest(kind=kind):
                self.assertIs(source.parallel_obstacle(kind, ()), DowngradeReason.SINGLE_USE_SOURCE)
        self.assertFalse(is_retryable(source))

    def test_parquet_files_are_read_by_workers_of_either_kind(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_parquet(Path(directory) / "a.parquet", {"x": [1.0, 2.0]})
            source = ParquetSource(directory, "x")
            for kind in KINDS:
                with self.subTest(kind=kind):
                    self.assertIsNone(source.parallel_obstacle(kind, source.partitions()))
            self.assertEqual(ParquetSource.WORKER_MODULES, ("pyarrow", "pyarrow.parquet"))
            copy = pickle.loads(pickle.dumps(source))
            self.assertEqual(copy.partitions(), source.partitions())
            partition = source.partitions()[0]
            self.assertEqual(
                [batch.tolist() for batch in copy.batches(partition)],
                [batch.tolist() for batch in source.batches(partition)],
            )


def reference_partitions(values: np.ndarray, step: int) -> list[tuple[str, int, int, str]]:
    """``(id, ordinal, rows, fingerprint)`` of each range, from the documented record."""

    found = []
    count = max(1, -(-len(values) // step))
    for index in range(count):
        part = values[index * step : (index + 1) * step]
        content = hashlib.sha256(part.tobytes()).hexdigest()
        found.append(
            (
                f"array#{index}",
                index,
                len(part),
                reference_digest("numpy", part.dtype.str, (len(part),), index * step, content),
            )
        )
    return found


class PartitionedNumpyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.arange(1.0, 11.0)

    def test_the_array_is_cut_into_consecutive_ranges_with_their_own_fingerprints(self) -> None:
        for step in (1, 3, 4, 5, 9, 10, 11, 100):
            source = NumpySource(self.values, partition_rows=step)
            with self.subTest(step=step):
                found = [
                    (item.id, item.ordinal, item.rows, item.fingerprint)
                    for item in source.partitions()
                ]
                self.assertEqual(found, reference_partitions(self.values, step))
                for item in source.partitions():
                    self.assertIs(item.fingerprint_level, FingerprintLevel.CONTENT)

    def test_the_last_range_is_the_short_one_and_the_counts_are_exact(self) -> None:
        sizes = [item.rows for item in NumpySource(self.values, partition_rows=4).partitions()]
        self.assertEqual(sizes, [4, 4, 2])
        sizes = [item.rows for item in NumpySource(self.values, partition_rows=5).partitions()]
        self.assertEqual(sizes, [5, 5])
        sizes = [item.rows for item in NumpySource(self.values, partition_rows=10).partitions()]
        self.assertEqual(sizes, [10])

    def test_equal_ranges_at_different_places_have_different_fingerprints(self) -> None:
        source = NumpySource(np.ones(6), partition_rows=2)
        fingerprints = {item.fingerprint for item in source.partitions()}
        self.assertEqual(len(fingerprints), 3)

    def test_a_changed_value_changes_only_its_range(self) -> None:
        base = NumpySource(self.values, partition_rows=4).partitions()
        changed = self.values.copy()
        changed[5] = 99.0
        other = NumpySource(changed, partition_rows=4).partitions()
        self.assertEqual(
            [a.fingerprint == b.fingerprint for a, b in zip(base, other, strict=True)],
            [True, False, True],
        )

    def test_an_empty_array_has_one_empty_range(self) -> None:
        source = NumpySource(np.array([], dtype=np.float64), partition_rows=3)
        (item,) = source.partitions()
        self.assertEqual((item.id, item.ordinal, item.rows), ("array#0", 0, 0))
        self.assertEqual(list(source.batches(item)), [])

    def test_the_batches_of_a_range_are_its_rows_in_order_and_bounded(self) -> None:
        source = NumpySource(self.values, partition_rows=4)
        parts = source.partitions()
        for max_rows, expected in (
            (1, [[5.0], [6.0], [7.0], [8.0]]),
            (3, [[5.0, 6.0, 7.0], [8.0]]),
            (9, [[5.0, 6.0, 7.0, 8.0]]),
        ):
            with self.subTest(max_rows=max_rows):
                batches = source.batches(parts[1], max_rows=max_rows)
                self.assertEqual([batch.tolist() for batch in batches], expected)
        self.assertEqual([b.tolist() for b in source.batches(parts[2])], [[9.0, 10.0]])

    def test_every_row_is_read_once_across_the_ranges(self) -> None:
        for step in (1, 3, 4, 10, 25):
            source = NumpySource(self.values, partition_rows=step)
            rows = np.concatenate(
                [batch for item in source.partitions() for batch in source.batches(item)]
            )
            with self.subTest(step=step):
                self.assertEqual(rows.tolist(), self.values.tolist())

    def test_other_dtypes_are_widened_a_batch_at_a_time(self) -> None:
        ints = np.arange(1, 8, dtype=np.int32)
        source = NumpySource(ints, partition_rows=3)
        batches = [b for item in source.partitions() for b in source.batches(item)]
        self.assertEqual([b.dtype for b in batches], [np.float64] * 3)
        self.assertEqual(np.concatenate(batches).tolist(), ints.astype(float).tolist())

    def test_a_partition_of_another_source_or_range_is_refused(self) -> None:
        source = NumpySource(self.values, partition_rows=4)
        for identity in (
            "array",
            "array#3",
            "array#99",
            "array#x",
            "array#-1",
            "array#",
            "other#0",
            "#0",
            "array#٣",
            "array#0#0",
        ):
            bad = Partition(identity, 0, 4, None, FingerprintLevel.NONE)
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                source.batches(bad)
        with self.assertRaises(ValueError):
            source.batches("array#0")  # type: ignore[arg-type]

    def test_the_whole_array_partition_is_refused_by_a_partitioned_source_and_conversely(
        self,
    ) -> None:
        whole = NumpySource(self.values)
        (only,) = whole.partitions()
        self.assertEqual(only.id, "array")
        for identity in ("array#0", "other"):
            with self.subTest(identity=identity), self.assertRaises(ValueError):
                whole.batches(Partition(identity, 0, 10, None, FingerprintLevel.NONE))
        self.assertEqual(np.concatenate(list(whole.batches(only))).tolist(), self.values.tolist())

    def test_the_last_valid_range_is_read_and_the_next_one_is_not(self) -> None:
        source = NumpySource(self.values, partition_rows=5)
        self.assertEqual(
            sum(
                len(b)
                for b in source.batches(Partition("array#1", 1, 5, None, FingerprintLevel.NONE))
            ),
            5,
        )
        with self.assertRaises(ValueError):
            source.batches(Partition("array#2", 2, 0, None, FingerprintLevel.NONE))

    def test_the_partitioning_changes_no_state(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.exponential(2.0, 101)
        whole = fold(NumpySource(values), ExponentialState)
        for step in (1, 2, 7, 50, 101, 500):
            with self.subTest(step=step):
                split = fold(NumpySource(values, partition_rows=step), ExponentialState)
                self.assertEqual(split.to_bytes(), whole.to_bytes())

    def test_partition_rows_is_a_positive_built_in_integer_or_none(self) -> None:
        for bad in (0, -1):
            with self.subTest(rows=bad), self.assertRaises(ValueError):
                NumpySource(self.values, partition_rows=bad)
        for wrong in (True, 2.0, "2"):
            with self.subTest(rows=wrong), self.assertRaises(TypeError):
                NumpySource(self.values, partition_rows=wrong)  # type: ignore[arg-type]
        self.assertEqual(len(NumpySource(self.values, partition_rows=None).partitions()), 1)

    def test_the_value_policy_still_applies_to_every_range(self) -> None:
        values = np.array([1.0, 2.0, np.nan, 4.0])
        source = NumpySource(values, partition_rows=2)
        first, second = source.partitions()
        self.assertEqual(len(list(source.batches(first))), 1)
        with self.assertRaises(ScaleSourceError) as caught:
            list(source.batches(second))
        self.assertEqual(caught.exception.code, ScaleSourceErrorCode.NAN_VALUE)


class ErrorPickleTests(unittest.TestCase):
    def test_a_source_error_survives_pickling_with_and_without_a_hint(self) -> None:
        for code in ScaleSourceErrorCode:
            for hint in (None, "do this instead"):
                original = ScaleSourceError(code, hint)
                copy = pickle.loads(pickle.dumps(original))
                with self.subTest(code=code, hint=hint):
                    self.assertIs(type(copy), ScaleSourceError)
                    self.assertIs(copy.code, code)
                    self.assertEqual(copy.hint, hint)
                    self.assertEqual(str(copy), str(original))
                    self.assertEqual(copy.args, original.args)

    def test_a_state_error_survives_pickling(self) -> None:
        for code in ScaleStateErrorCode:
            copy = pickle.loads(pickle.dumps(ScaleStateError(code)))
            with self.subTest(code=code):
                self.assertIs(type(copy), ScaleStateError)
                self.assertIs(copy.code, code)
                self.assertEqual(str(copy), code.value)
