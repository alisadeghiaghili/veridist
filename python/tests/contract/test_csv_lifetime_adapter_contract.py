"""CSV-01--CSV-06 executable RED contracts for strict lifetime CSV ingestion."""

from __future__ import annotations

import os
import re
import sys
import unittest
from io import BytesIO, TextIOWrapper
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from veridist.adapters.csv_lifetimes import (
    CsvLifetimeAdapter,
    CsvLifetimeAdapterError,
    CsvLifetimeChunk,
    CsvLifetimeLimits,
    CsvLifetimeSchema,
    retained_object_graph_bytes,
)
from veridist.domain.lifetimes import ExactLifetime
from veridist.engine.delivery import ChunkEnvelope
from veridist.engine.errors import FailureCode
from veridist.engine.pass_budget import PassBudgetError
from veridist.engine.provenance import PublicSourceId, SourceMutationStatus

SOURCE_ID = PublicSourceId("src_0123456789abcdef0123456789abcdef")
SCHEMA = CsvLifetimeSchema(time_column="time", event_observed_column="event_observed")
_CHUNK_ID = re.compile(r"chk_[0-9a-f]{32}")


class TrackingBinaryStream(BytesIO):
    """A binary stream that fails any whole-file read or seek attempt."""

    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.close_count = 0
        self.read_requests: list[int] = []

    def read(self, size: int = -1) -> bytes:
        self._record_read(size)
        return super().read(size)

    def read1(self, size: int = -1) -> bytes:
        self._record_read(size)
        return super().read1(size)

    def readinto(self, buffer: bytearray) -> int:
        self._record_read(len(buffer))
        return super().readinto(buffer)

    def readinto1(self, buffer: bytearray) -> int:
        self._record_read(len(buffer))
        return super().readinto1(buffer)

    def _record_read(self, size: int) -> None:
        if size < 0 or size > 8192:
            raise AssertionError("unbounded binary read")
        self.read_requests.append(size)

    def readall(self) -> bytes:
        raise AssertionError("readall is forbidden")

    def seek(self, offset: int, whence: int = 0) -> int:
        del offset, whence
        raise AssertionError("seek is forbidden")

    def close(self) -> None:
        self.close_count += 1
        super().close()


class CloseFailingBinaryStream(TrackingBinaryStream):
    def close(self) -> None:
        if self.closed:
            return
        self.close_count += 1
        BytesIO.close(self)
        raise OSError("private close failure")


class MidReadFaultStream(TrackingBinaryStream):
    def read(self, size: int = -1) -> bytes:
        del size
        raise RuntimeError("unexpected mid-read fault")

    def read1(self, size: int = -1) -> bytes:
        del size
        raise RuntimeError("unexpected mid-read fault")


class TrackingSource:
    """Replayable test seam with observable opens, identities, and close lifecycle."""

    def __init__(self, payload: bytes, *, changed_after_open: bool = False) -> None:
        self._changed_after_open = changed_after_open
        self._payload = payload
        self._revision = "rev0"
        self._identity_calls = 0
        self.open_count = 0
        self.streams: list[TrackingBinaryStream] = []

    def identity(self, path: Path) -> object:
        del path
        self._identity_calls += 1
        if self._changed_after_open and self._identity_calls == 1:
            self._revision = "rev1"
            return "rev0"
        return self._revision

    def open_binary(self, path: Path) -> TrackingBinaryStream:
        del path
        self.open_count += 1
        stream = TrackingBinaryStream(self._payload)
        self.streams.append(stream)
        return stream

    @property
    def close_count(self) -> int:
        return sum(stream.close_count for stream in self.streams)


class UnexpectedSource(TrackingSource):
    """A seam for checking that programming faults are never relabelled."""

    def open_binary(self, path: Path) -> TrackingBinaryStream:
        del path
        raise RuntimeError("unexpected source fault")


class IdentityUnavailableSource(TrackingSource):
    def identity(self, path: Path) -> object:
        del path
        if self.open_count:
            raise OSError("private revision unavailable")
        return "rev0"


class InitialIdentityUnavailableSource(TrackingSource):
    def identity(self, path: Path) -> object:
        del path
        raise OSError("private initial identity unavailable")


class InitialIdentityUnexpectedSource(TrackingSource):
    def identity(self, path: Path) -> object:
        del path
        raise RuntimeError("unexpected initial identity fault")


class NoneIdentitySource(TrackingSource):
    def __init__(self, payload: bytes, *, after_open: bool) -> None:
        super().__init__(payload)
        self.after_open = after_open

    def identity(self, path: Path) -> object:
        del path
        if self.after_open or self.open_count:
            return None
        return "rev0"


class PostIdentityFailureSource(TrackingSource):
    def __init__(self, payload: bytes, value: object) -> None:
        super().__init__(payload)
        self.value = value
        self.calls = 0

    def identity(self, path: Path) -> object:
        del path
        self.calls += 1
        if self.calls == 1:
            return "rev0"
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class NonBinarySource:
    def identity(self, path: Path) -> object:
        del path
        return "rev0"

    def open_binary(self, path: Path) -> object:
        del path
        return _ClosableNonBinary()


class _ClosableNonBinary:
    def close(self) -> None:
        pass


class MissingIdentitySource:
    def open_binary(self, path: Path) -> TrackingBinaryStream:
        del path
        return TrackingBinaryStream(b"time,event_observed\n1,1\n")


class FixedStreamSource:
    def __init__(self, stream: TrackingBinaryStream) -> None:
        self.stream = stream

    def identity(self, path: Path) -> object:
        del path
        return "rev0"

    def open_binary(self, path: Path) -> TrackingBinaryStream:
        del path
        return self.stream


class CsvLifetimeAdapterContracts(unittest.TestCase):
    """Strict CSV semantics must be proven through adapter and execution seams."""

    def adapter(
        self,
        payload: bytes,
        *,
        chunk_bytes: int = 2048,
        max_inflight_bytes: int = 4096,
        changed_after_open: bool = False,
    ) -> tuple[CsvLifetimeAdapter, TrackingSource]:
        source = TrackingSource(payload, changed_after_open=changed_after_open)
        return (
            CsvLifetimeAdapter(
                Path("private-lifetime-data.csv"),
                schema=SCHEMA,
                source_id=SOURCE_ID,
                limits=CsvLifetimeLimits(chunk_bytes, max_inflight_bytes),
                opener=source,
            ),
            source,
        )

    def assert_redacted(self, error: CsvLifetimeAdapterError) -> None:
        public = f"{error!r} {error} {error.context!r}"
        self.assertNotIn("private-lifetime-data.csv", public)
        self.assertNotIn("super-secret-cell", public)
        self.assertNotIn("rev1", public)

    def assert_adapter_error(
        self,
        adapter: CsvLifetimeAdapter,
        *,
        code: FailureCode,
        reason: str,
        context: dict[str, object] | None = None,
    ) -> None:
        with self.assertRaises(CsvLifetimeAdapterError) as captured:
            tuple(adapter.iter_chunks())
        self.assertEqual(captured.exception.code, code)
        if context is None:
            self.assertEqual(captured.exception.context["reason"], reason)
            self.assertTrue(set(captured.exception.context) <= {"reason", "record_offset"})
        else:
            self.assertEqual(captured.exception.context, context)
        self.assert_redacted(captured.exception)

    def test_csv01_exact_schema_tokens_and_ascii_decimal_grammar(self) -> None:
        adapter, _ = self.adapter(b"\xef\xbb\xbftime,event_observed\r\n0,1\r\n1.25e+1,0\r\n")
        chunks = tuple(adapter.iter_chunks())
        observations = tuple(item for chunk in chunks for item in chunk.observations)
        self.assertEqual(len(observations), 2)

        for value in (b" 1", b"+1", b"01", b".5", b"1.", b"NaN", b"Inf", b"1e-999999"):
            with self.subTest(value=value):
                invalid, _ = self.adapter(b"time,event_observed\n" + value + b",1\n")
                self.assert_adapter_error(
                    invalid,
                    code=FailureCode.SOURCE_ROW_INVALID,
                    reason="invalid_time",
                )
        for token in (b" 1", b"1 ", b"01", b"true"):
            with self.subTest(token=token):
                invalid, _ = self.adapter(b"time,event_observed\n1," + token + b"\n")
                self.assert_adapter_error(
                    invalid,
                    code=FailureCode.SOURCE_ROW_INVALID,
                    reason="invalid_event_token",
                )

    def test_csv02_logical_offsets_are_stable_but_chunk_ids_vary_with_budget(self) -> None:
        payload = b"time,event_observed\n1,1\n2,0\n3,1\n"
        narrow, _ = self.adapter(payload, chunk_bytes=2048, max_inflight_bytes=4096)
        wide, _ = self.adapter(payload, chunk_bytes=8192, max_inflight_bytes=8192)
        narrow_chunks = tuple(narrow.iter_chunks())
        wide_chunks = tuple(wide.iter_chunks())

        narrow_rows = [
            chunk.envelope.row_identity(index)
            for chunk in narrow_chunks
            for index in range(chunk.envelope.row_count)
        ]
        wide_rows = [
            chunk.envelope.row_identity(index)
            for chunk in wide_chunks
            for index in range(chunk.envelope.row_count)
        ]
        self.assertEqual(narrow_rows, wide_rows)
        self.assertEqual(
            narrow_rows, [(SOURCE_ID.value, 0), (SOURCE_ID.value, 1), (SOURCE_ID.value, 2)]
        )
        self.assertTrue(
            all(_CHUNK_ID.fullmatch(chunk.envelope.chunk_id) for chunk in narrow_chunks)
        )
        self.assertTrue(all(chunk.envelope.chunk_id != SOURCE_ID.value for chunk in narrow_chunks))
        self.assertTrue(all(_CHUNK_ID.fullmatch(chunk.envelope.chunk_id) for chunk in wide_chunks))
        self.assertTrue(
            all(
                SOURCE_ID.value.removeprefix("src_") not in chunk.envelope.chunk_id
                for chunk in narrow_chunks
            )
        )

        multiline, _ = self.adapter(b'time,event_observed\n1,1\n"super-secret-cell\n",0\n')
        self.assert_adapter_error(
            multiline,
            code=FailureCode.SOURCE_ROW_INVALID,
            reason="invalid_time",
            context={"reason": "invalid_time", "record_offset": 1},
        )

    def test_csv03_object_graph_accounting_and_guarded_stream_forbid_materialization(self) -> None:
        adapter, source = self.adapter(b"time,event_observed\n1,1\n2,0\n")
        chunks = tuple(adapter.iter_chunks())
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.retained_payload_bytes, retained_object_graph_bytes(chunk))
            self.assertLessEqual(chunk.retained_payload_bytes, adapter.limits.chunk_bytes)
            self.assertEqual(chunk.envelope.byte_size, chunk.retained_payload_bytes)
        self.assertTrue(source.streams)
        self.assertTrue(all(stream.read_requests for stream in source.streams))
        self.assertTrue(
            all(
                request >= 0 and request <= 8192
                for stream in source.streams
                for request in stream.read_requests
            )
        )

        oversized, _ = self.adapter(
            b"time,event_observed\n1,1\n",
            chunk_bytes=1,
            max_inflight_bytes=1,
        )
        self.assert_adapter_error(
            oversized,
            code=FailureCode.CHUNK_TOO_LARGE,
            reason="record_too_large",
        )

    def test_csv04_single_pass_rejects_second_acquisition_before_a_second_open(self) -> None:
        adapter, source = self.adapter(b"time,event_observed\n1,1\n3,0\n")
        self.assertTrue(tuple(adapter.iter_chunks()))
        with self.assertRaises(PassBudgetError) as captured:
            tuple(adapter.iter_chunks())
        self.assertEqual(captured.exception.code, FailureCode.PASS_BUDGET_EXCEEDED)
        self.assertEqual(source.open_count, 1)
        self.assertEqual(adapter.metadata.replayability.value, "single_pass")

    def test_csv05_failure_codes_reasons_lifecycle_and_closed_outcome_mapping(self) -> None:
        cases = (
            (b"", FailureCode.SOURCE_SCHEMA_INVALID, "header_missing"),
            (b"\xef\xbb\xbf", FailureCode.SOURCE_SCHEMA_INVALID, "header_missing"),
            (b"\xff", FailureCode.SOURCE_DECODE_FAILED, "invalid_utf8"),
            (b"time,time\n1,1\n", FailureCode.SOURCE_SCHEMA_INVALID, "header_duplicate"),
            (b"time\n1\n", FailureCode.SOURCE_SCHEMA_INVALID, "header_columns_mismatch"),
            (
                b"time,\xef\xbb\xbfevent_observed\n1,1\n",
                FailureCode.SOURCE_SCHEMA_INVALID,
                "header_columns_mismatch",
            ),
            (
                b"time,event_observed,extra\n1,1,x\n",
                FailureCode.SOURCE_SCHEMA_INVALID,
                "header_columns_mismatch",
            ),
            (
                b"time,event_observed\n\n1,1\n",
                FailureCode.SOURCE_ROW_INVALID,
                "blank_record",
            ),
            (
                b"time,event_observed\nsuper-secret-cell,1\n",
                FailureCode.SOURCE_ROW_INVALID,
                "invalid_time",
            ),
            (
                b"time,event_observed\n1,1,extra\n",
                FailureCode.SOURCE_ROW_INVALID,
                "malformed_record",
            ),
            (
                b'time,event_observed\n"unterminated,1\n',
                FailureCode.SOURCE_ROW_INVALID,
                "malformed_record",
            ),
        )
        for payload, code, reason in cases:
            with self.subTest(code=code, reason=reason):
                adapter, source = self.adapter(payload)
                self.assert_adapter_error(adapter, code=code, reason=reason)
                self.assertEqual(source.close_count, 1)

        # A blank record is tolerated only when it is the last thing in the
        # file, for example a trailing "\r\n\r\n" an editor or spreadsheet
        # added; it does not change the delivered rows.
        for trailing_blank_payload in (
            b"time,event_observed\n\n",
            b"time,event_observed\n1,1\n\n",
            b"time,event_observed\n1,1\n\n\n",
        ):
            with self.subTest(payload=trailing_blank_payload):
                tolerant, _ = self.adapter(trailing_blank_payload)
                observations = tuple(
                    item for chunk in tolerant.iter_chunks() for item in chunk.observations
                )
                self.assertEqual(
                    observations,
                    (ExactLifetime(1.0),) if b"1,1" in trailing_blank_payload else (),
                )

        changed, source = self.adapter(
            b"time,event_observed\n1,1\n",
            changed_after_open=True,
        )
        self.assert_adapter_error(
            changed,
            code=FailureCode.SOURCE_REVISION_MISMATCH,
            reason="source_mutated",
        )
        self.assertEqual(source.close_count, 1)

        early, source = self.adapter(b"time,event_observed\n" + b"1,1\n" * 30)
        iterator = early.iter_chunks()
        next(iterator)
        iterator.close()
        self.assertEqual(source.close_count, 1)

        unexpected_source = UnexpectedSource(b"time,event_observed\n1,1\n")
        unexpected = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
            opener=unexpected_source,
        )
        with self.assertRaises(RuntimeError):
            tuple(unexpected.iter_chunks())

    def test_csv05b_explicit_text_wrapper_close_is_not_left_to_finalization(self) -> None:
        adapter, _ = self.adapter(b"time,event_observed\n1,1\n")
        wrappers: list[TextIOWrapper] = []

        def retained_wrapper(*args: object, **kwargs: object) -> TextIOWrapper:
            wrapper = TextIOWrapper(*args, **kwargs)  # type: ignore[arg-type]
            wrappers.append(wrapper)
            return wrapper

        with patch("veridist.adapters.csv_lifetimes.TextIOWrapper", retained_wrapper):
            self.assertEqual(len(tuple(adapter.iter_chunks())), 1)
        self.assertEqual(len(wrappers), 1)
        self.assertTrue(wrappers[0].closed)

    def test_csv06_empty_header_and_full_scan_precedence_constructor_and_result_invariants(
        self,
    ) -> None:
        header_only, source = self.adapter(b"time,event_observed\n")
        self.assertEqual(tuple(header_only.iter_chunks()), ())
        self.assertEqual(source.open_count, 1)

        overflow_then_invalid, _ = self.adapter(b"time,event_observed\n1e10000,1\n1,x\n")
        self.assert_adapter_error(
            overflow_then_invalid,
            code=FailureCode.SOURCE_ROW_INVALID,
            reason="invalid_time",
        )

        for columns in (("", "event_observed"), ("time", "time")):
            with self.subTest(columns=columns):
                with self.assertRaises(ValueError):
                    CsvLifetimeSchema(*columns)
        for limits in (
            (True, 1),
            (0, 1),
            (1, 0),
            (2, 1),
            (1, True),
        ):
            with self.subTest(limits=limits):
                with self.assertRaises((TypeError, ValueError)):
                    CsvLifetimeLimits(*limits)
        with self.assertRaises(TypeError):
            CsvLifetimeAdapter(  # type: ignore[arg-type]
                123.456,
                schema=SCHEMA,
                source_id=SOURCE_ID,
                limits=CsvLifetimeLimits(2048, 2048),
            )
        with self.assertRaises(TypeError):
            CsvLifetimeAdapter(
                Path("private-lifetime-data.csv"),
                schema=SCHEMA,
                source_id=SOURCE_ID,
                limits=CsvLifetimeLimits(2048, 2048),
                opener=object(),  # type: ignore[arg-type]
            )
        for field in ("schema", "source_id", "limits"):
            with self.subTest(field=field):
                arguments: dict[str, object] = {
                    "schema": SCHEMA,
                    "source_id": SOURCE_ID,
                    "limits": CsvLifetimeLimits(2048, 2048),
                }
                arguments[field] = object()
                with self.assertRaises(TypeError):
                    CsvLifetimeAdapter(
                        Path("private-lifetime-data.csv"),
                        **arguments,  # type: ignore[arg-type]
                    )

    def test_csv07_public_metadata_and_default_source_failures_are_redacted(self) -> None:
        adapter = CsvLifetimeAdapter(
            Path("private-missing-source.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
        )
        metadata = adapter.metadata
        self.assertEqual(metadata.source_id, SOURCE_ID.value)
        self.assertEqual(metadata.schema_version, "1")
        self.assertEqual(metadata.redaction_reason, "hash_unavailable")
        self.assert_adapter_error(
            adapter,
            code=FailureCode.SOURCE_OPEN_FAILED,
            reason="open_failed",
        )

        with TemporaryDirectory() as directory:
            path = Path(directory) / "lifetimes.csv"
            path.write_bytes(b"time,event_observed\n1,1\n")
            filesystem = CsvLifetimeAdapter(
                path,
                schema=SCHEMA,
                source_id=SOURCE_ID,
                limits=CsvLifetimeLimits(2048, 2048),
            )
            self.assertEqual(len(tuple(filesystem.iter_chunks())), 1)

    def test_csv08_value_guards_and_owned_graph_accounting(self) -> None:
        for code, reason in (
            (FailureCode.CANCELLED, "open_failed"),
            (FailureCode.SOURCE_OPEN_FAILED, "unapproved"),
        ):
            with self.subTest(code=code, reason=reason):
                with self.assertRaises(ValueError):
                    CsvLifetimeAdapterError(code, reason=reason)
        for offset in (True, -1):
            with self.subTest(offset=offset):
                with self.assertRaises((TypeError, ValueError)):
                    CsvLifetimeAdapterError(
                        FailureCode.SOURCE_ROW_INVALID,
                        reason="invalid_time",
                        record_offset=offset,
                    )

        envelope = ChunkEnvelope(
            SOURCE_ID.value, "chk_0123456789abcdef0123456789abcdef", 0, 0, 1, 1
        )
        with self.assertRaises(TypeError):
            CsvLifetimeChunk(  # type: ignore[arg-type]
                object(), (ExactLifetime(1),), 1
            )
        with self.assertRaises(TypeError):
            CsvLifetimeChunk(envelope, [ExactLifetime(1)], 1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            CsvLifetimeChunk(envelope, (object(),), 1)  # type: ignore[arg-type]
        with self.assertRaises((TypeError, ValueError)):
            CsvLifetimeChunk(envelope, (ExactLifetime(1),), True)
        with self.assertRaises(ValueError):
            CsvLifetimeChunk(envelope, (ExactLifetime(1),), 2)
        with self.assertRaises(ValueError):
            CsvLifetimeChunk(envelope, (ExactLifetime(1),), 0)
        mismatched = ChunkEnvelope(
            SOURCE_ID.value, "chk_0123456789abcdef0123456789abcdef", 0, 0, 2, 1
        )
        with self.assertRaises(ValueError):
            CsvLifetimeChunk(mismatched, (ExactLifetime(1),), 1)

        shared = (ExactLifetime(1),)
        self.assertLess(
            retained_object_graph_bytes((shared, shared)),
            2 * retained_object_graph_bytes((shared,)),
        )
        dynamic = "chunk-id-" + "x" * 128
        manual = sys.getsizeof((dynamic,)) + sys.getsizeof(dynamic)
        self.assertEqual(retained_object_graph_bytes((dynamic,)), manual)

    def test_csv09_adapter_boundary_branches(self) -> None:
        nonbinary = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
            opener=NonBinarySource(),
        )
        with self.assertRaises(TypeError):
            tuple(nonbinary.iter_chunks())

        unavailable = IdentityUnavailableSource(b"time,event_observed\n1,1\n")
        identity_unknown = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
            opener=unavailable,
        )
        self.assert_adapter_error(
            identity_unknown,
            code=FailureCode.SOURCE_REVISION_UNAVAILABLE,
            reason="identity_unavailable",
        )
        self.assertEqual(unavailable.close_count, 1)

        initial_identity_unknown = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
            opener=InitialIdentityUnavailableSource(b"time,event_observed\n1,1\n"),
        )
        self.assert_adapter_error(
            initial_identity_unknown,
            code=FailureCode.SOURCE_REVISION_UNAVAILABLE,
            reason="identity_unavailable",
        )

        baseline, _ = self.adapter(b"time,event_observed\n1,1\n2,0\n")
        combined = next(baseline.iter_chunks())
        split, _ = self.adapter(
            b"time,event_observed\n1,1\n2,0\n",
            chunk_bytes=combined.retained_payload_bytes - 1,
            max_inflight_bytes=combined.retained_payload_bytes - 1,
        )
        self.assertEqual([chunk.envelope.sequence_number for chunk in split.iter_chunks()], [0, 1])

    def test_csv10_closed_source_protocol_and_full_scan_after_failure(self) -> None:
        with self.assertRaises(TypeError):
            CsvLifetimeAdapter(
                Path("private-lifetime-data.csv"),
                schema=SCHEMA,
                source_id=SOURCE_ID,
                limits=CsvLifetimeLimits(2048, 2048),
                opener=MissingIdentitySource(),  # type: ignore[arg-type]
            )

        invalid_then_valid, _ = self.adapter(b"time,event_observed\n1,x\n2,1\n")
        self.assert_adapter_error(
            invalid_then_valid,
            code=FailureCode.SOURCE_ROW_INVALID,
            reason="invalid_event_token",
        )

    def test_csv11_translated_failures_drop_private_causes(self) -> None:
        class OpenFailure:
            def identity(self, path: Path) -> object:
                del path
                return "private-revision"

            def open_binary(self, path: Path) -> TrackingBinaryStream:
                del path
                raise OSError("private-path-and-message")

        cases: tuple[CsvLifetimeAdapter, ...] = (
            CsvLifetimeAdapter(
                Path("private-lifetime-data.csv"),
                schema=SCHEMA,
                source_id=SOURCE_ID,
                limits=CsvLifetimeLimits(2048, 2048),
                opener=OpenFailure(),
            ),
            self.adapter(b"time,event_observed\n\xff,1\n")[0],
            self.adapter(b'time,event_observed\n"unterminated,1\n')[0],
        )
        for adapter in cases:
            with self.subTest(adapter=type(adapter.opener).__name__):
                with self.assertRaises(CsvLifetimeAdapterError) as captured:
                    tuple(adapter.iter_chunks())
                self.assertIsNone(captured.exception.__cause__)
                self.assertIsNone(captured.exception.__context__)
                self.assert_redacted(captured.exception)

    def test_csv12_exact_chunk_boundaries_and_ids_are_range_source_sensitive(self) -> None:
        payload = b"time,event_observed\n1,1\n2,0\n"
        wide, _ = self.adapter(payload, chunk_bytes=4096, max_inflight_bytes=4096)
        combined = next(wide.iter_chunks())
        exact_combined, _ = self.adapter(
            payload,
            chunk_bytes=combined.retained_payload_bytes,
            max_inflight_bytes=combined.retained_payload_bytes,
        )
        self.assertEqual([item.envelope.row_count for item in exact_combined.iter_chunks()], [2])
        split, _ = self.adapter(
            payload,
            chunk_bytes=combined.retained_payload_bytes - 1,
            max_inflight_bytes=combined.retained_payload_bytes - 1,
        )
        split_chunks = tuple(split.iter_chunks())
        self.assertEqual(
            [(item.envelope.row_start, item.envelope.row_stop) for item in split_chunks],
            [(0, 1), (1, 2)],
        )
        self.assertEqual([item.envelope.sequence_number for item in split_chunks], [0, 1])
        self.assertTrue(
            all(item.retained_payload_bytes <= split.limits.chunk_bytes for item in split_chunks)
        )
        self.assertNotEqual(combined.envelope.chunk_id, split_chunks[0].envelope.chunk_id)
        repeat, _ = self.adapter(payload, chunk_bytes=4096, max_inflight_bytes=4096)
        self.assertEqual(combined.envelope.chunk_id, next(repeat.iter_chunks()).envelope.chunk_id)

        other = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=PublicSourceId("src_fedcba9876543210fedcba9876543210"),
            limits=CsvLifetimeLimits(4096, 4096),
            opener=TrackingSource(payload),
        )
        self.assertNotEqual(combined.envelope.chunk_id, next(other.iter_chunks()).envelope.chunk_id)

        one_row, _ = self.adapter(b"time,event_observed\n1,1\n", chunk_bytes=4096)
        single = next(one_row.iter_chunks())
        exact_single, _ = self.adapter(
            b"time,event_observed\n1,1\n",
            chunk_bytes=single.retained_payload_bytes,
            max_inflight_bytes=single.retained_payload_bytes,
        )
        self.assertEqual(len(tuple(exact_single.iter_chunks())), 1)
        below_single, _ = self.adapter(
            b"time,event_observed\n1,1\n",
            chunk_bytes=single.retained_payload_bytes - 1,
            max_inflight_bytes=single.retained_payload_bytes - 1,
        )
        self.assert_adapter_error(
            below_single,
            code=FailureCode.CHUNK_TOO_LARGE,
            reason="record_too_large",
            context={"reason": "record_too_large", "record_offset": 0},
        )

    def test_csv13_error_precedence_is_first_semantic_then_mutation(self) -> None:
        first, _ = self.adapter(b"time,event_observed\n1,x\n2,bad\n")
        self.assert_adapter_error(
            first,
            code=FailureCode.SOURCE_ROW_INVALID,
            reason="invalid_event_token",
            context={"reason": "invalid_event_token", "record_offset": 0},
        )
        mutated, _ = self.adapter(
            b"time,event_observed\n1,x\n",
            changed_after_open=True,
        )
        self.assert_adapter_error(
            mutated,
            code=FailureCode.SOURCE_REVISION_MISMATCH,
            reason="source_mutated",
        )

    def test_csv14_cleanup_never_masks_primary_and_midread_fault_closes_once(self) -> None:
        close_failing = CloseFailingBinaryStream(b"time,event_observed\n\xff,1\n")
        protected = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
            opener=FixedStreamSource(close_failing),
        )
        wrappers: list[TextIOWrapper] = []

        def retained_wrapper(*args: object, **kwargs: object) -> TextIOWrapper:
            wrapper = TextIOWrapper(*args, **kwargs)  # type: ignore[arg-type]
            wrappers.append(wrapper)
            return wrapper

        with patch("veridist.adapters.csv_lifetimes.TextIOWrapper", retained_wrapper):
            self.assert_adapter_error(
                protected,
                code=FailureCode.SOURCE_DECODE_FAILED,
                reason="invalid_utf8",
            )
        self.assertEqual(close_failing.close_count, 1)
        self.assertTrue(wrappers[0].closed)

        midread = MidReadFaultStream(b"time,event_observed\n1,1\n")
        unexpected = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
            opener=FixedStreamSource(midread),
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected mid-read fault"):
            tuple(unexpected.iter_chunks())
        self.assertEqual(midread.close_count, 1)

        initial_source = InitialIdentityUnexpectedSource(b"time,event_observed\n1,1\n")
        initial_unexpected = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
            opener=initial_source,
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected initial identity fault"):
            tuple(initial_unexpected.iter_chunks())
        self.assertEqual(initial_source.close_count, 1)

    def test_csv15_identity_none_cleanup_and_strict_malformed_corpus(self) -> None:
        for source in (
            NoneIdentitySource(b"time,event_observed\n1,1\n", after_open=True),
            NoneIdentitySource(b"time,event_observed\n1,1\n", after_open=False),
        ):
            adapter = CsvLifetimeAdapter(
                Path("private-lifetime-data.csv"),
                SCHEMA,
                SOURCE_ID,
                CsvLifetimeLimits(2048, 2048),
                source,
            )
            self.assert_adapter_error(
                adapter,
                code=FailureCode.SOURCE_REVISION_UNAVAILABLE,
                reason="identity_unavailable",
            )
            self.assertEqual(source.close_count, 1)

        cleanup_only = CloseFailingBinaryStream(b"time,event_observed\n1,1\n")
        cleanup_adapter = CsvLifetimeAdapter(
            Path("private-lifetime-data.csv"),
            SCHEMA,
            SOURCE_ID,
            CsvLifetimeLimits(2048, 2048),
            FixedStreamSource(cleanup_only),
        )
        with self.assertRaisesRegex(OSError, "private close failure"):
            tuple(cleanup_adapter.iter_chunks())
        self.assertEqual(cleanup_only.close_count, 1)

        corpus = (
            (b'time,event_observed\n"1"junk,1\n', "malformed_record"),
            (b"time,event_observed\n1\x00,1\n", "invalid_time"),
            (b"time,event_observed\r\n1,1\r\n", None),
            # A trailing blank record (nothing follows it) is tolerated, not
            # a failure; see the dedicated trailing-blank assertions above.
            (b"time,event_observed\n1,1\n\n", None),
            (b'time,event_observed\n"1\n",1\n', "invalid_time"),
        )
        for payload, reason in corpus:
            adapter, _ = self.adapter(payload)
            if reason is None:
                self.assertEqual(len(tuple(adapter.iter_chunks())), 1)
            else:
                self.assert_adapter_error(
                    adapter, code=FailureCode.SOURCE_ROW_INVALID, reason=reason
                )

    def test_csv16_post_identity_failures_and_static_type_accounting(self) -> None:
        for value in (OSError("private post identity"), None):
            source = PostIdentityFailureSource(b"time,event_observed\n1,1\n", value)
            adapter = CsvLifetimeAdapter(
                Path("private-lifetime-data.csv"),
                SCHEMA,
                SOURCE_ID,
                CsvLifetimeLimits(2048, 2048),
                source,
            )
            self.assert_adapter_error(
                adapter,
                code=FailureCode.SOURCE_REVISION_UNAVAILABLE,
                reason="identity_unavailable",
            )
            self.assertEqual(source.close_count, 1)
        self.assertEqual(retained_object_graph_bytes(CsvLifetimeAdapter), 0)

    def test_csv17_failure_phase_and_identity_status_are_typed(self) -> None:
        with self.assertRaises(TypeError):
            CsvLifetimeAdapterError(
                FailureCode.SOURCE_OPEN_FAILED,
                reason="open_failed",
                phase="preflight",  # type: ignore[arg-type]
            )
        with self.assertRaises(TypeError):
            CsvLifetimeAdapterError(
                FailureCode.SOURCE_OPEN_FAILED,
                reason="open_failed",
                mutation_status="not_checked",  # type: ignore[arg-type]
            )
        error = CsvLifetimeAdapterError(
            FailureCode.SOURCE_ROW_INVALID,
            reason="invalid_time",
            record_offset=2,
        )
        verified = error.with_mutation_status(SourceMutationStatus.VERIFIED_UNCHANGED)
        self.assertIsNot(error, verified)
        self.assertEqual(verified.context, error.context)

    def test_csv18_chunk_builder_accounting_operations_scale_linearly(self) -> None:
        def operation_count(rows: int) -> int:
            payload = b"time,event_observed\n" + b"1,1\n" * rows
            adapter, _ = self.adapter(
                payload,
                chunk_bytes=10_000_000,
                max_inflight_bytes=10_000_000,
            )
            with patch(
                "veridist.adapters.csv_lifetimes.retained_object_graph_bytes",
                wraps=retained_object_graph_bytes,
            ) as accounting:
                chunks = tuple(adapter.iter_chunks())
            self.assertEqual(len(chunks), 1)
            return accounting.call_count

        small = operation_count(128)
        large = operation_count(256)
        # Per-record accounting may inspect the newly parsed observation, but
        # the adapter's fixed empty-chunk graph must be computed once at
        # construction, not walked again for every row.
        self.assertLessEqual(small, 4 * 128 + 4)
        self.assertLessEqual(large, 4 * 256 + 4)
        self.assertLessEqual(large, 2 * small + 4)

    def test_csv19_filesystem_initial_identity_is_bound_to_open_handle(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "lifetimes.csv"
            path.write_bytes(b"time,event_observed\n1,1\n")
            adapter = CsvLifetimeAdapter(
                path,
                SCHEMA,
                SOURCE_ID,
                CsvLifetimeLimits(2048, 2048),
            )
            with patch("veridist.adapters.csv_lifetimes.os.fstat", wraps=os.fstat) as fstat:
                self.assertEqual(len(tuple(adapter.iter_chunks())), 1)
            fstat.assert_called_once()

    def test_csv20_path_accepts_str_and_pathlike_and_still_rejects_other_types(self) -> None:
        str_source = TrackingSource(b"time,event_observed\n1,1\n")
        str_adapter = CsvLifetimeAdapter(
            "private-lifetime-data.csv",
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
            opener=str_source,
        )
        self.assertEqual(str_adapter.path, Path("private-lifetime-data.csv"))
        self.assertEqual(len(tuple(str_adapter.iter_chunks())), 1)

        class _FakePathLike:
            def __fspath__(self) -> str:
                return "private-lifetime-data.csv"

        pathlike_source = TrackingSource(b"time,event_observed\n1,1\n")
        pathlike_adapter = CsvLifetimeAdapter(
            _FakePathLike(),  # type: ignore[arg-type]
            schema=SCHEMA,
            source_id=SOURCE_ID,
            limits=CsvLifetimeLimits(2048, 2048),
            opener=pathlike_source,
        )
        self.assertEqual(pathlike_adapter.path, Path("private-lifetime-data.csv"))
        self.assertEqual(len(tuple(pathlike_adapter.iter_chunks())), 1)

        with self.assertRaises(TypeError):
            CsvLifetimeAdapter(  # type: ignore[arg-type]
                12345,
                schema=SCHEMA,
                source_id=SOURCE_ID,
                limits=CsvLifetimeLimits(2048, 2048),
            )

    def test_csv21_default_limits_are_the_documented_constant(self) -> None:
        self.assertEqual(CsvLifetimeLimits.default(), CsvLifetimeLimits(65_536, 65_536))


if __name__ == "__main__":
    unittest.main()
