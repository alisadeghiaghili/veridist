"""RED contracts for the 0.6 resumable CSV execution milestone."""

from __future__ import annotations

import hashlib
import inspect
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId
from veridist.adapters.csv_lifetimes import CsvLifetimeChunk
from veridist.domain.lifetimes import ExactLifetime
from veridist.engine.checkpoint import CheckpointRecord, SQLiteCheckpointStore
from veridist.engine.delivery import ChunkEnvelope
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.statistics.exponential import ExponentialCheckpointReducer, ExponentialReductionState

_SOURCE_ID = "src_0123456789abcdef0123456789abcdef"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V1CheckpointedCsvTests(unittest.TestCase):
    def _store(
        self,
        directory: str,
        revision: str = "revision-a",
        *,
        source_id: str = _SOURCE_ID,
        source_schema: str = "csv-lifetime-v1",
        reducer_id: str = "exponential-reduction-v1",
        accumulator_schema: str = "exponential-reduction-v1",
    ) -> SQLiteCheckpointStore:
        state = (
            b'{"compensation":"0x0.0p+0","event_count":0,'
            b'"observation_count":0,"total_time":"0x0.0p+0"}'
        )
        initial = CheckpointRecord.create(
            format_version=1,
            source_id=source_id,
            source_schema=source_schema,
            source_revision=revision,
            reducer_id=reducer_id,
            accumulator_schema=accumulator_schema,
            plan_digest="plan",
            cursor=0,
            committed_ranges=(),
            generation=0,
            operation_token=None,
            operation_digest=None,
            state=state,
        )
        path = (
            Path(directory)
            / f"{source_id}-{source_schema}-{reducer_id}-{accumulator_schema}-{revision}.sqlite3"
        )
        return SQLiteCheckpointStore.create(path, initial)

    def test_checkpointed_csv_api_is_keyword_explicit(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        signature = inspect.signature(fit_exponential_checkpointed_csv)
        self.assertEqual(
            tuple(signature.parameters),
            ("path", "schema", "source_id", "limits", "store", "source_revision", "cancel"),
        )
        self.assertTrue(
            all(
                item.kind is inspect.Parameter.KEYWORD_ONLY
                for item in signature.parameters.values()
            )
        )

    def test_resume_does_not_apply_committed_rows_twice(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n2,0\n3,1\n", encoding="utf-8")
            revision = _sha256(source)
            store = self._store(directory, revision)
            first = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(24, 48),
                store=store,
                source_revision=revision,
                cancel=lambda cursor: cursor >= 2,
            )
            self.assertEqual(first.code, "CANCELLED")
            committed_cursor = store.read().cursor
            self.assertEqual(committed_cursor, 2)
            resumed = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(24, 48),
                store=SQLiteCheckpointStore(store.path),
                source_revision=revision,
                cancel=None,
            )
        self.assertEqual(resumed.fit.observation_count, 3)
        self.assertEqual(resumed.fit.event_count, 2)
        self.assertEqual(resumed.fit.total_time, 6.0)

    def test_source_revision_change_cannot_advance_checkpoint(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n", encoding="utf-8")
            store = self._store(directory, _sha256(source))
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=store,
                source_revision="revision-b",
                cancel=None,
            )
            self.assertEqual(result.code, "SOURCE_REVISION_MISMATCH")
            self.assertEqual(store.read().cursor, 0)

    def test_rewritten_file_is_rejected_even_when_the_old_revision_is_reused(self) -> None:
        """DS2a repro: a rewritten CSV must not silently advance the checkpoint.

        Before the fix, only the checkpoint's own recorded revision was
        compared to the caller-supplied string; neither was tied to the
        file's actual bytes, so replacing the file and reusing the old
        revision string, or pairing it with a different public source id,
        both produced a COMPLETE result that silently mixed rows from two
        different files.
        """
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n2,1\n3,0\n4,1\n", encoding="utf-8")
            original_revision = _sha256(source)
            store = self._store(directory, original_revision)
            cancelled = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=store,
                source_revision=original_revision,
                cancel=lambda cursor: cursor >= 2,
            )
            self.assertEqual(cancelled.code, "CANCELLED")
            generation_before = store.read().generation

            source.write_text("time,event_observed\n100,1\n200,1\n3,0\n4,1\n", encoding="utf-8")
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=SQLiteCheckpointStore(store.path),
                source_revision=original_revision,
                cancel=None,
            )
            self.assertEqual(result.code, "SOURCE_REVISION_MISMATCH")
            self.assertEqual(store.read().generation, generation_before)

    def test_mismatched_public_source_id_is_rejected(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n", encoding="utf-8")
            revision = _sha256(source)
            store = self._store(directory, revision, source_id=_SOURCE_ID)
            generation_before = store.read().generation
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId("src_fedcba9876543210fedcba9876543210"),
                limits=CsvLifetimeLimits(32, 64),
                store=store,
                source_revision=revision,
                cancel=None,
            )
            self.assertEqual(result.code, "SOURCE_ID_MISMATCH")
            self.assertEqual(store.read().generation, generation_before)

    def test_mismatched_source_schema_is_rejected(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n", encoding="utf-8")
            revision = _sha256(source)
            store = self._store(directory, revision, source_schema="other")
            generation_before = store.read().generation
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=store,
                source_revision=revision,
                cancel=None,
            )
            self.assertEqual(result.code, "SOURCE_SCHEMA_MISMATCH")
            self.assertEqual(store.read().generation, generation_before)

    def test_create_checkpointed_csv_store_round_trips_cancel_and_resume(self) -> None:
        from veridist.execution import (
            create_checkpointed_csv_store,
            fit_exponential_checkpointed_csv,
        )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n2,1\n3,0\n4,1\n", encoding="utf-8")
            revision = _sha256(source)

            uninterrupted_store = create_checkpointed_csv_store(
                Path(directory) / "uninterrupted.sqlite3",
                csv_path=source,
                source_id=PublicSourceId(_SOURCE_ID),
            )
            uninterrupted = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=uninterrupted_store,
                source_revision=revision,
                cancel=None,
            )
            self.assertEqual(uninterrupted.code, "COMPLETE")

            store = create_checkpointed_csv_store(
                Path(directory) / "resumed.sqlite3",
                csv_path=source,
                source_id=PublicSourceId(_SOURCE_ID),
            )
            cancelled = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=store,
                source_revision=revision,
                cancel=lambda cursor: cursor >= 2,
            )
            self.assertEqual(cancelled.code, "CANCELLED")
            resumed = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=SQLiteCheckpointStore(store.path),
                source_revision=revision,
                cancel=None,
            )
        self.assertEqual(resumed.code, "COMPLETE")
        assert uninterrupted.fit is not None
        assert resumed.fit is not None
        self.assertEqual(resumed.fit.observation_count, uninterrupted.fit.observation_count)
        self.assertEqual(resumed.fit.event_count, uninterrupted.fit.event_count)
        self.assertEqual(resumed.fit.total_time, uninterrupted.fit.total_time)
        self.assertEqual(resumed.fit.rate, uninterrupted.fit.rate)

    def test_create_checkpointed_csv_store_rejects_invalid_types(self) -> None:
        from veridist.execution import create_checkpointed_csv_store

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n", encoding="utf-8")
            with self.assertRaises(TypeError):
                create_checkpointed_csv_store(
                    Path(directory) / "store.sqlite3",
                    csv_path="not-a-path",  # type: ignore[arg-type]
                    source_id=PublicSourceId(_SOURCE_ID),
                )
            with self.assertRaises(TypeError):
                create_checkpointed_csv_store(
                    Path(directory) / "store.sqlite3",
                    csv_path=source,
                    source_id="not-a-public-source-id",  # type: ignore[arg-type]
                )

    def test_checkpoint_internal_revision_disagrees_with_a_correctly_identified_file(
        self,
    ) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n", encoding="utf-8")
            current_revision = _sha256(source)
            stale_revision = hashlib.sha256(b"a-different-file-entirely").hexdigest()
            store = self._store(directory, stale_revision)
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=store,
                source_revision=current_revision,
                cancel=None,
            )
        self.assertEqual(result.code, "SOURCE_REVISION_MISMATCH")

    def test_mid_run_source_revision_change_is_rejected(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class RevisionChangesAfterFirstRead:
            def __init__(self, initial: CheckpointRecord) -> None:
                self._initial = initial
                self._reads = 0

            def read(self) -> CheckpointRecord:
                self._reads += 1
                if self._reads == 1:
                    return self._initial
                return replace(self._initial, source_revision="changed-mid-run")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n", encoding="utf-8")
            revision = _sha256(source)
            initial = self._store(directory, revision).read()
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=RevisionChangesAfterFirstRead(initial),
                source_revision=revision,
                cancel=None,
            )
        self.assertEqual(result.code, "SOURCE_REVISION_MISMATCH")

    def test_cancel_on_the_first_row_of_a_run_commits_nothing(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n2,1\n", encoding="utf-8")
            revision = _sha256(source)
            store = self._store(directory, revision)
            generation_before = store.read().generation
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=store,
                source_revision=revision,
                cancel=lambda cursor: True,
            )
            self.assertEqual(result.code, "CANCELLED")
            self.assertEqual(store.read().cursor, 0)
            self.assertEqual(store.read().generation, generation_before)

    def test_already_committed_leading_chunk_is_skipped_without_reapplying(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class TwoChunkAdapter:
            def __init__(self, *unused: object) -> None:
                return None

            def iter_chunks(self) -> object:
                yield CsvLifetimeChunk(
                    ChunkEnvelope(
                        source_id=_SOURCE_ID,
                        chunk_id="chunk-0",
                        sequence_number=0,
                        row_start=0,
                        row_stop=2,
                        byte_size=1,
                    ),
                    (ExactLifetime(Decimal("1")), ExactLifetime(Decimal("2"))),
                    1,
                )
                yield CsvLifetimeChunk(
                    ChunkEnvelope(
                        source_id=_SOURCE_ID,
                        chunk_id="chunk-1",
                        sequence_number=1,
                        row_start=2,
                        row_stop=3,
                        byte_size=1,
                    ),
                    (ExactLifetime(Decimal("3")),),
                    1,
                )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "unused.csv"
            source.write_text("time,event_observed\n1,1\n2,1\n3,1\n", encoding="utf-8")
            revision = _sha256(source)
            reducer = ExponentialCheckpointReducer()
            already_committed_state = reducer.encode_state(
                ExponentialReductionState(2, 2, 3.0, 0.0)
            )
            record = CheckpointRecord.create(
                format_version=1,
                source_id=_SOURCE_ID,
                source_schema="csv-lifetime-v1",
                source_revision=revision,
                reducer_id=reducer.reducer_id,
                accumulator_schema=reducer.accumulator_schema,
                plan_digest="plan",
                cursor=2,
                committed_ranges=((0, 2),),
                generation=1,
                operation_token="rows-0-2",
                operation_digest="a" * 64,
                state=already_committed_state,
            )
            store = SQLiteCheckpointStore.create(Path(directory) / "two-chunk.sqlite3", record)
            with patch("veridist.execution.CsvLifetimeAdapter", TwoChunkAdapter):
                result = fit_exponential_checkpointed_csv(
                    path=source,
                    schema=CsvLifetimeSchema("time", "event_observed"),
                    source_id=PublicSourceId(_SOURCE_ID),
                    limits=CsvLifetimeLimits(32, 64),
                    store=store,
                    source_revision=revision,
                    cancel=None,
                )
        self.assertEqual(result.code, "COMPLETE")
        self.assertEqual(result.fit.observation_count, 3)
        self.assertEqual(result.fit.total_time, 6.0)

    def test_checkpointed_csv_fit_result_rejects_invalid_shapes(self) -> None:
        from veridist.execution import CheckpointedCsvFitResult, fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n", encoding="utf-8")
            revision = _sha256(source)
            store = self._store(directory, revision)
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=store,
                source_revision=revision,
                cancel=None,
            )

        with self.assertRaises(ValueError):
            CheckpointedCsvFitResult("", None)
        with self.assertRaises(ValueError):
            CheckpointedCsvFitResult("COMPLETE", None)
        with self.assertRaises(ValueError):
            CheckpointedCsvFitResult("CANCELLED", result.fit)

    def test_contract_input_types_fail_before_storage_or_source_access(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        common = {
            "path": Path("source.csv"),
            "schema": CsvLifetimeSchema("time", "event_observed"),
            "source_id": PublicSourceId(_SOURCE_ID),
            "limits": CsvLifetimeLimits(32, 64),
            "store": object(),
            "source_revision": "revision-a",
            "cancel": None,
        }
        for name, value in (
            ("path", "source.csv"),
            ("schema", object()),
            ("source_id", object()),
            ("limits", object()),
            ("cancel", object()),
        ):
            with self.subTest(name=name):
                arguments = dict(common)
                arguments[name] = value
                with self.assertRaises(TypeError):
                    fit_exponential_checkpointed_csv(**arguments)

    def test_incompatible_reducer_metadata_is_rejected_before_reading_csv(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text("time,event_observed\n1,1\n", encoding="utf-8")
            revision = _sha256(source)
            for reducer_id, schema, expected in (
                ("other-reducer", "exponential-reduction-v1", "REDUCER_MISMATCH"),
                ("exponential-reduction-v1", "other-schema", "ACCUMULATOR_SCHEMA_MISMATCH"),
            ):
                with self.subTest(expected=expected):
                    store = self._store(
                        directory, revision, reducer_id=reducer_id, accumulator_schema=schema
                    )
                    result = fit_exponential_checkpointed_csv(
                        path=source,
                        schema=CsvLifetimeSchema("time", "event_observed"),
                        source_id=PublicSourceId(_SOURCE_ID),
                        limits=CsvLifetimeLimits(32, 64),
                        store=store,
                        source_revision=revision,
                        cancel=None,
                    )
                    self.assertEqual(result.code, expected)

    def test_checksum_mismatch_is_rejected_before_constructing_the_adapter(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class CorruptStore:
            def read(self) -> CheckpointRecord:
                return replace(self_record, checksum="not-a-valid-checksum")

        with tempfile.TemporaryDirectory() as directory:
            self_record = self._store(directory).read()
            result = fit_exponential_checkpointed_csv(
                path=Path(directory) / "unread.csv",
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=CorruptStore(),
                source_revision="revision-a",
                cancel=None,
            )
        self.assertEqual(result.code, "CHECKPOINT_CHECKSUM_MISMATCH")

    def test_range_gap_from_adapter_boundary_is_not_silently_repaired(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class StaticStore:
            def read(self) -> CheckpointRecord:
                return record

        class GappedAdapter:
            def __init__(self, *unused: object) -> None:
                return None

            def iter_chunks(self) -> object:
                yield CsvLifetimeChunk(
                    ChunkEnvelope(
                        source_id=_SOURCE_ID,
                        chunk_id="chunk-gap",
                        sequence_number=0,
                        row_start=1,
                        row_stop=2,
                        byte_size=1,
                    ),
                    (ExactLifetime(Decimal("1")),),
                    1,
                )

        with tempfile.TemporaryDirectory() as directory:
            # The GappedAdapter patch below bypasses real CSV parsing, but the
            # revision contract still hashes the real file at `path`, so one
            # must exist for this to reach the adapter-boundary check at all.
            source = Path(directory) / "unused.csv"
            source.write_text("time,event_observed\n1,1\n2,1\n", encoding="utf-8")
            revision = _sha256(source)
            record = self._store(directory, revision).read()
            with patch("veridist.execution.CsvLifetimeAdapter", GappedAdapter):
                result = fit_exponential_checkpointed_csv(
                    path=source,
                    schema=CsvLifetimeSchema("time", "event_observed"),
                    source_id=PublicSourceId(_SOURCE_ID),
                    limits=CsvLifetimeLimits(32, 64),
                    store=StaticStore(),
                    source_revision=revision,
                    cancel=None,
                )
        self.assertEqual(result.code, "RANGE_MISMATCH")

    def test_engine_contract_error_from_checkpoint_boundary_is_returned_as_a_code(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class FailingStore:
            def read(self) -> CheckpointRecord:
                raise EngineContractError(FailureCode.CHECKPOINT_STORAGE_FAILED)

        result = fit_exponential_checkpointed_csv(
            path=Path("unread.csv"),
            schema=CsvLifetimeSchema("time", "event_observed"),
            source_id=PublicSourceId(_SOURCE_ID),
            limits=CsvLifetimeLimits(32, 64),
            store=FailingStore(),
            source_revision="revision-a",
            cancel=None,
        )
        self.assertEqual(result.code, "CHECKPOINT_STORAGE_FAILED")

    def test_final_checkpoint_revision_is_rechecked_after_an_empty_source(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class RevisionChangingStore:
            def __init__(self, initial: CheckpointRecord) -> None:
                self._initial = initial
                self._reads = 0

            def read(self) -> CheckpointRecord:
                self._reads += 1
                if self._reads == 1:
                    return self._initial
                return replace(self._initial, source_revision="revision-b")

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "empty.csv"
            source.write_text("time,event_observed\n", encoding="utf-8")
            revision = _sha256(source)
            initial = self._store(directory, revision).read()
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(32, 64),
                store=RevisionChangingStore(initial),
                source_revision=revision,
                cancel=None,
            )
        self.assertEqual(result.code, "SOURCE_REVISION_MISMATCH")

    def test_large_csv_commits_bounded_batches_instead_of_each_row(self) -> None:
        from veridist.execution import fit_exponential_checkpointed_csv

        class CountingStore:
            def __init__(self, delegate: SQLiteCheckpointStore) -> None:
                self.delegate = delegate
                self.compare_and_swap_calls = 0

            def read(self) -> CheckpointRecord:
                return self.delegate.read()

            def compare_and_swap(
                self, expected_generation: int, candidate: CheckpointRecord
            ) -> CheckpointRecord:
                self.compare_and_swap_calls += 1
                return self.delegate.compare_and_swap(expected_generation, candidate)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "lifetimes.csv"
            source.write_text(
                "time,event_observed\n" + "".join(f"{index + 1},1\n" for index in range(200)),
                encoding="utf-8",
            )
            revision = _sha256(source)
            store = CountingStore(self._store(directory, revision))
            result = fit_exponential_checkpointed_csv(
                path=source,
                schema=CsvLifetimeSchema("time", "event_observed"),
                source_id=PublicSourceId(_SOURCE_ID),
                limits=CsvLifetimeLimits(4096, 4096),
                store=store,
                source_revision=revision,
                cancel=None,
            )

        self.assertEqual(result.code, "COMPLETE")
        self.assertEqual(result.fit.observation_count, 200)
        self.assertLess(store.compare_and_swap_calls, 20)


if __name__ == "__main__":
    unittest.main()
