"""DS-10 contracts for the public engine failure surface."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from enum import StrEnum
from math import inf, nan
from pathlib import Path
from types import MappingProxyType

from tests.contract.buffer_watchdog import bounded_buffer_call
from veridist.engine.data_source import (
    CheckpointMetadata,
    DataSourceCapabilityError,
    DataSourceMetadata,
    Replayability,
    plan_passes,
)
from veridist.engine.delivery import (
    BoundedChunkBuffer,
    BufferedChunk,
    ChunkEnvelope,
    DeliveryContractError,
    DeliveryValidator,
)
from veridist.engine.errors import EngineContractError, FailureCode
from veridist.engine.pass_budget import PassBudgetError

EXPECTED_FAILURE_CODES = {
    "ACCUMULATOR_SCHEMA_MISMATCH",
    "BUFFER_TIMEOUT",
    "CANCELLED",
    "CHECKPOINT_CHECKSUM_MISMATCH",
    "CHECKPOINT_CONFLICT",
    "CHECKPOINT_DECODE_FAILED",
    "CHECKPOINT_FORMAT_UNSUPPORTED",
    "CHECKPOINT_ALREADY_EXISTS",
    "CHECKPOINT_NOT_FOUND",
    "CHECKPOINT_REQUIRED",
    "CHECKPOINT_SCHEMA_MISMATCH",
    "CHECKPOINT_SOURCE_ID_MISMATCH",
    "CHECKPOINT_STORAGE_FAILED",
    "CHUNK_TOO_LARGE",
    "DUPLICATE_CHUNK",
    "INVALID_RETAINED_BYTES",
    "MISSING_CHUNK",
    "MISSING_RANGE_UNKNOWN",
    "OPERATION_DIGEST_CONFLICT",
    "OUT_OF_ORDER_CHUNK",
    "PASS_BUDGET_EXCEEDED",
    "PAYLOAD_CHECKSUM_MISMATCH",
    "PLAN_MISMATCH",
    "RANGE_MISMATCH",
    "REDUCER_FAILURE",
    "REDUCER_MISMATCH",
    "RETRY_EXHAUSTED",
    "RETRY_NOT_ADMISSIBLE",
    "SINK_FAILURE",
    "SOURCE_DECODE_FAILED",
    "SOURCE_ID_MISMATCH",
    "SOURCE_MISMATCH",
    "SOURCE_OPEN_FAILED",
    "SOURCE_ROW_INVALID",
    "SOURCE_REVISION_MISMATCH",
    "SOURCE_REVISION_UNAVAILABLE",
    "SOURCE_SCHEMA_INVALID",
    "SOURCE_SCHEMA_MISMATCH",
    "SPOOL_REQUIRED",
}


class ContextLabel(StrEnum):
    VALUE = "safe-label"


def context_strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        return tuple(
            item
            for nested in value.values()
            for item in context_strings(nested)
        )
    if isinstance(value, tuple):
        return tuple(item for nested in value for item in context_strings(nested))
    return ()


class TypedFailureSurfaceTests(unittest.TestCase):
    def test_ds10_failure_code_registry_is_complete_and_stable(self) -> None:
        self.assertEqual({code.name for code in FailureCode}, EXPECTED_FAILURE_CODES)
        for code in FailureCode:
            with self.subTest(code=code):
                self.assertEqual(code.value, code.name)

    def test_ds10_public_failure_classes_share_one_typed_constructor(self) -> None:
        cases = (
            (EngineContractError, FailureCode.RANGE_MISMATCH),
            (DataSourceCapabilityError, FailureCode.SPOOL_REQUIRED),
            (DeliveryContractError, FailureCode.MISSING_CHUNK),
            (PassBudgetError, FailureCode.PASS_BUDGET_EXCEEDED),
        )
        for error_type, code in cases:
            with self.subTest(error_type=error_type.__name__):
                error = error_type(code, {"attempt": 1})
                self.assertIsInstance(error, EngineContractError)
                self.assertIs(error.code, code)
                with self.assertRaises(TypeError):
                    error_type(code.value, {})  # type: ignore[arg-type]

    def test_ds10_context_is_recursively_immutable_and_enum_normalized(self) -> None:
        source = {
            "nested": {"labels": (ContextLabel.VALUE,)},
            "complete": False,
            "estimate": 1.5,
        }
        error = EngineContractError(FailureCode.RANGE_MISMATCH, source)

        self.assertIsInstance(error.context, MappingProxyType)
        nested = error.context["nested"]
        self.assertIsInstance(nested, MappingProxyType)
        self.assertEqual(nested["labels"], ("safe-label",))
        self.assertEqual(error.context["estimate"], 1.5)

        source["nested"]["labels"] = ("changed",)  # type: ignore[index]
        self.assertEqual(nested["labels"], ("safe-label",))
        with self.assertRaises(TypeError):
            error.context["new"] = "value"  # type: ignore[index]
        with self.assertRaises(TypeError):
            nested["new"] = "value"  # type: ignore[index]

    def test_ds10_context_rejects_unsafe_keys_and_values(self) -> None:
        unsafe_contexts: tuple[object, ...] = (
            ["not a mapping"],
            {1: "non-string key"},
            {"payload": b"raw"},
            {"file": Path("private/source.csv")},
            {"items": [1, 2]},
            {"items": {1, 2}},
            {"exception": RuntimeError("private payload")},
            {"object": object()},
            {"estimate": nan},
            {"estimate": inf},
            {"estimate": -inf},
        )
        for context in unsafe_contexts:
            with self.subTest(context_type=type(context).__name__):
                with self.assertRaises(TypeError):
                    EngineContractError(FailureCode.RANGE_MISMATCH, context)  # type: ignore[arg-type]

    def test_ds10_sensitive_context_keys_are_rejected_even_for_strings(self) -> None:
        sensitive_keys = (
            "payload",
            "path",
            "uri",
            "query",
            "dsn",
            "credential",
            "password",
            "token",
            "source_revision",
            "checksum",
            "digest",
            "exception",
            "message",
            "traceback",
        )
        for key in sensitive_keys:
            with self.subTest(key=key), self.assertRaises(TypeError):
                EngineContractError(FailureCode.RANGE_MISMATCH, {key: "private-value"})

    def test_ds10_text_surfaces_bare_code_only_when_context_is_empty(self) -> None:
        for error in (
            EngineContractError(FailureCode.RANGE_MISMATCH),
            EngineContractError(FailureCode.RANGE_MISMATCH, {}),
        ):
            with self.subTest(context=error.context):
                self.assertEqual(str(error), "RANGE_MISMATCH")
                self.assertEqual(repr(error), "EngineContractError(code=RANGE_MISMATCH)")

    def test_ds10_text_surfaces_a_sorted_context_when_present(self) -> None:
        # Exception text shows numeric and token-like context, sorted by key,
        # so failures are diagnosable from a log line alone.
        error = EngineContractError(
            FailureCode.RANGE_MISMATCH,
            {"row_start": 5, "checkpoint_cursor": 3},
        )

        self.assertEqual(str(error), "RANGE_MISMATCH (checkpoint_cursor=3, row_start=5)")
        self.assertEqual(
            repr(error),
            "EngineContractError(code=RANGE_MISMATCH, "
            "context={checkpoint_cursor=3, row_start=5})",
        )

    def test_ds10_text_never_surfaces_untrusted_context_strings(self) -> None:
        # Context keys are screened, values are not, so free-form strings such
        # as paths or URIs must never reach exception text even under an
        # allowed key; code-defined tokens remain visible.
        sentinel = "C:/private/source.csv?credential=secret"
        error = EngineContractError(
            FailureCode.CHECKPOINT_STORAGE_FAILED,
            {
                "expected": sentinel,
                "actual": (0, 3),
                "nested": {"note": "free text with spaces"},
                "sqlite_errorname": "SQLITE_BUSY",
            },
        )

        for text in (str(error), repr(error)):
            with self.subTest(text=text):
                self.assertNotIn(sentinel, text)
                self.assertNotIn("free text with spaces", text)
                self.assertIn("sqlite_errorname=SQLITE_BUSY", text)
                self.assertIn("expected=<redacted>", text)
                self.assertIn("actual=(0, 3)", text)
                self.assertIn("nested={note=<redacted>}", text)

    def test_ds10_buffer_timeouts_use_one_typed_distinct_code(self) -> None:
        buffer = BoundedChunkBuffer(chunk_bytes=4, max_inflight_bytes=4)

        def buffered(chunk_id: str, sequence: int) -> BufferedChunk:
            return BufferedChunk(
                envelope=ChunkEnvelope(
                    source_id="source",
                    chunk_id=chunk_id,
                    sequence_number=sequence,
                    row_start=sequence,
                    row_stop=sequence + 1,
                    byte_size=4,
                ),
                payload=None,
            )

        bounded_buffer_call(buffer, lambda: buffer.put(buffered("first", 0)))
        with self.assertRaises(DeliveryContractError) as put_error:
            bounded_buffer_call(buffer, lambda: buffer.put(buffered("second", 1), timeout=0.001))
        self.assertIs(put_error.exception.code, FailureCode.BUFFER_TIMEOUT)
        self.assertEqual(put_error.exception.context, {"operation": "put"})

        bounded_buffer_call(buffer, lambda: buffer.get(timeout=0.1))

        with self.assertRaises(DeliveryContractError) as caught:
            bounded_buffer_call(buffer, lambda: buffer.get(timeout=0.001))

        self.assertIs(caught.exception.code, FailureCode.BUFFER_TIMEOUT)
        self.assertNotEqual(caught.exception.code, FailureCode.CANCELLED)
        self.assertEqual(caught.exception.context, {"operation": "get"})

    def test_ds10_public_contexts_redact_raw_source_chunk_and_schema_identifiers(self) -> None:
        sentinel = "file:///private/source.csv?credential=secret"

        validator = DeliveryValidator(f"expected:{sentinel}")
        with self.assertRaises(DeliveryContractError) as delivery_error:
            validator.accept(
                ChunkEnvelope(
                    source_id=f"actual:{sentinel}",
                    chunk_id=f"chunk:{sentinel}",
                    sequence_number=0,
                    row_start=0,
                    row_stop=1,
                    byte_size=1,
                )
            )

        class Source:
            metadata = DataSourceMetadata(
                source_id=f"expected:{sentinel}",
                schema_version="source-v1",
                provenance_schema_version="1",
                replayability=Replayability.CHECKPOINT_REPLAYABLE,
                source_hash="redacted-test-hash",
                checkpoint_schema_version=f"expected-schema:{sentinel}",
            )

        checkpoint_cases = (
            CheckpointMetadata(f"actual:{sentinel}", Source.metadata.checkpoint_schema_version),
            CheckpointMetadata(Source.metadata.source_id, f"actual-schema:{sentinel}"),
        )
        errors = [delivery_error.exception]
        for checkpoint in checkpoint_cases:
            with self.subTest(checkpoint=checkpoint), self.assertRaises(
                DataSourceCapabilityError
            ) as checkpoint_error:
                plan_passes(Source(), required_passes=2, checkpoint=checkpoint)
            errors.append(checkpoint_error.exception)

        for error in errors:
            with self.subTest(code=error.code):
                self.assertFalse(
                    any(sentinel in value for value in context_strings(error.context))
                )
                self.assertNotIn(sentinel, str(error))
                self.assertNotIn(sentinel, repr(error))


if __name__ == "__main__":
    unittest.main()
