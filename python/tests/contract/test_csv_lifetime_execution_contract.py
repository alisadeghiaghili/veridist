"""RED execution contracts staged after the CSV adapter behavior is green."""

from __future__ import annotations

import re
import unittest
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from veridist.adapters.csv_lifetimes import (
    CsvLifetimeAdapter,
    CsvLifetimeChunk,
    CsvLifetimeLimits,
    CsvLifetimeSchema,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.outcome import FailedOutcome, FailureStage, UnknownMissingRanges
from veridist.engine.provenance import PublicSourceId, SourceMutationStatus
from veridist.execution import (
    ExponentialSourceFitResult,
    fit_exponential_csv,
    fit_exponential_source,
)
from veridist.families.exponential import ExponentialFitFailure, ExponentialFitFailureCode

SOURCE_ID = PublicSourceId("src_0123456789abcdef0123456789abcdef")
SCHEMA = CsvLifetimeSchema(time_column="time", event_observed_column="event_observed")


class _TrackingSource:
    """Local execution fixture; importing another test class duplicates collection."""

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.open_count = 0
        self.close_count = 0

    def identity(self, path: Path) -> object:
        del path
        return "fixture-revision"

    def open_binary(self, path: Path) -> BytesIO:
        del path
        self.open_count += 1
        return _CountingBytesIO(self.payload, self)


class _CountingBytesIO(BytesIO):
    def __init__(self, payload: bytes, owner: _TrackingSource) -> None:
        super().__init__(payload)
        self.owner = owner

    def close(self) -> None:
        if not self.closed:
            self.owner.close_count += 1
        super().close()


def _adapter(
    payload: bytes, *, chunk_bytes: int = 2048
) -> tuple[CsvLifetimeAdapter, _TrackingSource]:
    source = _TrackingSource(payload)
    return (
        CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            SCHEMA,
            SOURCE_ID,
            CsvLifetimeLimits(chunk_bytes, chunk_bytes),
            source,
        ),
        source,
    )


class CsvLifetimeExecutionContracts(unittest.TestCase):
    """The adapter is family-neutral; only this layer creates scientific fits."""

    def test_csv_exec01_header_only_is_complete_empty_sample(self) -> None:
        adapter, source = _adapter(b"time,event_observed\n")

        result = fit_exponential_source(adapter)

        self.assertIsInstance(result, ExponentialSourceFitResult)
        assert isinstance(result, ExponentialSourceFitResult)
        self.assertIsInstance(result.fit, ExponentialFitFailure)
        assert isinstance(result.fit, ExponentialFitFailure)
        self.assertEqual(result.fit.code, ExponentialFitFailureCode.EMPTY_SAMPLE)
        self.assertTrue(result.execution.outcome.complete)
        self.assertEqual(source.close_count, 1)

    def test_csv_exec02_open_failure_is_closed_failed_outcome_without_fit(self) -> None:
        result = fit_exponential_csv(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
        )

        self.assertIsInstance(result, ExponentialSourceFitResult)
        assert isinstance(result, ExponentialSourceFitResult)
        self.assertIsNone(result.fit)
        self.assertIsInstance(result.execution.outcome, FailedOutcome)
        assert isinstance(result.execution.outcome, FailedOutcome)
        self.assertEqual(result.execution.outcome.failure.code, FailureCode.SOURCE_OPEN_FAILED)
        self.assertEqual(result.execution.outcome.failure.stage.value, "preflight")
        self.assertIsInstance(result.execution.outcome.coverage, UnknownMissingRanges)

    def test_csv_exec03_overflow_consumes_the_full_source_and_is_a_complete_nonestimate(
        self,
    ) -> None:
        adapter, source = _adapter(b"time,event_observed\n1e308,1\n1e308,0\n1,1\n")

        result = fit_exponential_source(adapter)

        self.assertTrue(result.execution.outcome.complete)
        self.assertIsInstance(result.fit, ExponentialFitFailure)
        assert isinstance(result.fit, ExponentialFitFailure)
        self.assertEqual(result.fit.code, ExponentialFitFailureCode.NUMERICAL_OVERFLOW)
        self.assertEqual((result.fit.observation_count, result.fit.event_count), (3, 2))
        self.assertEqual(source.open_count, 1)
        self.assertEqual(source.close_count, 1)

    def test_csv_exec04_mixed_censoring_matches_decimal_oracle_across_chunk_budgets(self) -> None:
        payload = b"time,event_observed\n1,1\n2,0\n3,1\n"
        for budget in (530, 2048):
            with self.subTest(chunk_bytes=budget):
                adapter, source = _adapter(payload, chunk_bytes=budget)
                result = fit_exponential_source(adapter)

                self.assertTrue(result.execution.outcome.complete)
                self.assertIsNotNone(result.fit)
                assert result.fit is not None
                self.assertEqual((result.fit.observation_count, result.fit.event_count), (3, 2))
                self.assertAlmostEqual(result.fit.rate, 2.0 / 6.0)
                self.assertEqual(source.close_count, 1)
                observation = result.execution.provenance.execution
                self.assertEqual(observation.passes.actual_pass_count, 1)
                self.assertEqual(observation.passes.max_passes, 1)
                self.assertGreater(observation.buffer.peak_inflight_bytes, 0)
                self.assertEqual(
                    observation.buffer.peak_inflight_bytes,
                    observation.buffer.largest_retained_chunk_bytes,
                )
                self.assertLessEqual(
                    observation.buffer.peak_inflight_bytes,
                    budget,
                )

    def test_csv_exec05_complete_statistical_nonestimates_and_typed_failure_stages(self) -> None:
        cases = (
            (b"time,event_observed\n1,0\n2,0\n", ExponentialFitFailureCode.NO_OBSERVED_EVENTS),
            (b"time,event_observed\n0,1\n", ExponentialFitFailureCode.UNBOUNDED_LIKELIHOOD),
            (b"time,event_observed\n1e-320,1\n", ExponentialFitFailureCode.NUMERICAL_OVERFLOW),
        )
        for payload, code in cases:
            with self.subTest(code=code):
                adapter, _ = _adapter(payload)
                result = fit_exponential_source(adapter)
                self.assertTrue(result.execution.outcome.complete)
                self.assertIsInstance(result.fit, ExponentialFitFailure)
                assert isinstance(result.fit, ExponentialFitFailure)
                self.assertEqual(result.fit.code, code)

        adapter, _ = _adapter(b"time,event_observed\ninvalid,1\n")
        failed = fit_exponential_source(adapter)
        self.assertIsInstance(failed.execution.outcome, FailedOutcome)
        assert isinstance(failed.execution.outcome, FailedOutcome)
        self.assertIs(failed.execution.outcome.failure.stage, FailureStage.DELIVERY)
        self.assertIs(
            failed.execution.provenance.source.mutation_status,
            SourceMutationStatus.VERIFIED_UNCHANGED,
        )

    def test_csv_exec06_provenance_has_fresh_run_id_and_canonical_settings_hash(self) -> None:
        first, _ = _adapter(b"time,event_observed\n1,1\n")
        second, _ = _adapter(b"time,event_observed\n1,1\n")
        left = fit_exponential_source(first).execution.provenance
        right = fit_exponential_source(second).execution.provenance

        self.assertRegex(left.run_id, re.compile(r"run_[0-9a-f]{32}"))
        self.assertNotEqual(left.run_id, right.run_id)
        self.assertEqual(left.estimator.settings_sha256, right.estimator.settings_sha256)
        self.assertNotEqual(left.estimator.settings_sha256, "0" * 64)

    def test_csv_exec07_result_and_adapter_type_invariants_are_closed(self) -> None:
        with self.assertRaises(TypeError):
            ExponentialSourceFitResult(None, object())  # type: ignore[arg-type]
        adapter, _ = _adapter(b"time,event_observed\n")
        complete = fit_exponential_source(adapter)
        with self.assertRaises(ValueError):
            ExponentialSourceFitResult(None, complete.execution)
        with self.assertRaises(TypeError):
            fit_exponential_source(object())

    def test_csv_exec10_finish_detects_a_chunk_parsed_but_never_delivered(self) -> None:
        payload = b"time,event_observed\n1,1\n2,1\n"
        wide, _ = _adapter(payload, chunk_bytes=4096)
        combined = next(wide.iter_chunks())
        adapter, _ = _adapter(payload, chunk_bytes=combined.retained_payload_bytes - 1)

        original_iter_chunks = CsvLifetimeAdapter.iter_chunks

        def dropped_last(self: CsvLifetimeAdapter) -> Iterator[CsvLifetimeChunk]:
            # Drain the real generator fully -- so the adapter still records
            # its true terminal record count -- then withhold the final
            # chunk, simulating a chunk lost between parsing and delivery.
            chunks = list(original_iter_chunks(self))
            assert len(chunks) == 2
            yield from chunks[:1]

        with patch.object(CsvLifetimeAdapter, "iter_chunks", dropped_last):
            result = fit_exponential_source(adapter)

        self.assertEqual(adapter.terminal_record_count, 2)
        self.assertIsNone(result.fit)
        self.assertIsInstance(result.execution.outcome, FailedOutcome)
        assert isinstance(result.execution.outcome, FailedOutcome)
        self.assertEqual(result.execution.outcome.failure.code, FailureCode.MISSING_CHUNK)
        self.assertEqual(result.execution.outcome.failure.stage, FailureStage.FINALIZATION)

    def test_csv_exec11_provenance_pass_count_reflects_the_adapters_own_enforcer(self) -> None:
        adapter, _ = _adapter(b"time,event_observed\n1,1\n")
        self.assertEqual(adapter.passes.actual_pass_count, 0)

        result = fit_exponential_source(adapter)

        self.assertEqual(adapter.passes.actual_pass_count, 1)
        self.assertEqual(
            result.execution.provenance.execution.passes,
            adapter.passes.observation,
        )

    def test_csv_exec08_iterator_without_close_and_non_tuple_payload_cleanup(self) -> None:
        adapter, _ = _adapter(b"time,event_observed\n")
        with patch.object(CsvLifetimeAdapter, "iter_chunks", return_value=iter(())):
            self.assertTrue(fit_exponential_source(adapter).execution.outcome.complete)

        adapter, source = _adapter(b"time,event_observed\n1,1\n")
        from veridist.engine.delivery import BufferedChunk as RealBufferedChunk

        def list_payload(*, envelope: object, payload: object) -> RealBufferedChunk:
            del payload
            return RealBufferedChunk(envelope=envelope, payload=[])  # type: ignore[arg-type]

        with patch("veridist.execution.BufferedChunk", side_effect=list_payload):
            with self.assertRaises(TypeError, msg="non-tuple payload must propagate"):
                fit_exponential_source(adapter)
        self.assertEqual(source.close_count, 1)

    def test_csv_exec09_non_csv_engine_failures_keep_honest_stage_and_cleanup(self) -> None:
        adapter, source = _adapter(b"time,event_observed\n1,1\n")
        error = EngineContractError(FailureCode.MISSING_CHUNK, {})
        with patch("veridist.execution.DeliveryValidator.accept", side_effect=error):
            result = fit_exponential_source(adapter)
        assert isinstance(result.execution.outcome, FailedOutcome)
        self.assertIs(result.execution.outcome.failure.stage, FailureStage.DELIVERY)
        self.assertIs(
            result.execution.provenance.source.mutation_status,
            SourceMutationStatus.NOT_CHECKED,
        )
        self.assertEqual(source.close_count, 1)

        adapter, _ = _adapter(b"time,event_observed\n")
        with patch("veridist.execution.PassEnforcer.begin_pass", side_effect=error):
            preflight = fit_exponential_source(adapter)
        assert isinstance(preflight.execution.outcome, FailedOutcome)
        self.assertIs(preflight.execution.outcome.failure.stage, FailureStage.PREFLIGHT)


if __name__ == "__main__":
    unittest.main()
