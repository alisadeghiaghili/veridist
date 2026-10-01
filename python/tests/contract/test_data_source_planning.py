"""DS-01 through DS-03 contract tests for the preflight data-source planner."""

from __future__ import annotations

import unittest

from veridist.engine.data_source import (
    SUPPORTED_PROVENANCE_SCHEMA_VERSIONS,
    CheckpointMetadata,
    DataSourceCapabilityError,
    DataSourceMetadata,
    Replayability,
    SpoolPolicy,
    plan_passes,
)


class CountingSource:
    """A deliberately iterable fixture whose reads are externally observable."""

    def __init__(self, metadata: DataSourceMetadata) -> None:
        self.metadata = metadata
        self.read_count = 0

    def __iter__(self) -> CountingSource:
        self.read_count += 1
        return self

    def __next__(self) -> object:
        raise StopIteration


def make_metadata(
    replayability: Replayability,
    *,
    source_hash: str | None = "sha256:abc",
    redaction_reason: str | None = None,
    checkpoint_schema_version: str | None = None,
) -> DataSourceMetadata:
    return DataSourceMetadata(
        source_id="dataset:example-001",
        schema_version="1",
        provenance_schema_version="1",
        replayability=replayability,
        source_hash=source_hash,
        redaction_reason=redaction_reason,
        checkpoint_schema_version=checkpoint_schema_version,
    )


class DataSourceMetadataContractTests(unittest.TestCase):
    """DS-01: source identity and redaction metadata are immutable and complete."""

    def test_ds01_metadata_is_frozen_slotted_and_has_supported_provenance(self) -> None:
        metadata = make_metadata(Replayability.REPLAYABLE)

        self.assertEqual(metadata.source_id, "dataset:example-001")
        self.assertIn(metadata.provenance_schema_version, SUPPORTED_PROVENANCE_SCHEMA_VERSIONS)
        self.assertEqual(metadata.source_hash, "sha256:abc")
        self.assertFalse(hasattr(metadata, "__dict__"))
        with self.assertRaises((AttributeError, TypeError)):
            metadata.source_id = "dataset:changed"  # type: ignore[misc]

    def test_ds01_redaction_requires_reason_when_hash_is_absent(self) -> None:
        redacted = make_metadata(
            Replayability.REPLAYABLE,
            source_hash=None,
            redaction_reason="privacy-policy:restricted-input",
        )
        self.assertIsNone(redacted.source_hash)
        self.assertEqual(redacted.redaction_reason, "privacy-policy:restricted-input")

        with self.assertRaises(ValueError):
            make_metadata(Replayability.REPLAYABLE, source_hash=None)

    def test_ds01_metadata_rejects_incomplete_or_unsupported_declarations(self) -> None:
        invalid_declarations = (
            {"source_id": ""},
            {"provenance_schema_version": "2"},
            {"source_hash": ""},
            {"redaction_reason": ""},
            {"source_hash": "sha256:abc", "redaction_reason": "also-present"},
            {"checkpoint_schema_version": ""},
        )
        for overrides in invalid_declarations:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                DataSourceMetadata(
                    source_id=overrides.get("source_id", "dataset:example-001"),
                    schema_version="1",
                    provenance_schema_version=overrides.get("provenance_schema_version", "1"),
                    replayability=Replayability.REPLAYABLE,
                    source_hash=overrides.get("source_hash", None),
                    redaction_reason=overrides.get("redaction_reason", "redacted"),
                    checkpoint_schema_version=overrides.get("checkpoint_schema_version", None),
                )

    def test_ds01_metadata_rejects_non_string_identifiers_with_typeerror(self) -> None:
        for field_name in ("source_id", "schema_version", "provenance_schema_version"):
            overrides: dict[str, object] = {
                "source_id": "dataset:example-001",
                "schema_version": "1",
                "provenance_schema_version": "1",
            }
            overrides[field_name] = 1
            with self.subTest(field_name=field_name), self.assertRaises(TypeError):
                DataSourceMetadata(
                    source_id=overrides["source_id"],  # type: ignore[arg-type]
                    schema_version=overrides["schema_version"],  # type: ignore[arg-type]
                    provenance_schema_version=overrides[  # type: ignore[arg-type]
                        "provenance_schema_version"
                    ],
                    replayability=Replayability.REPLAYABLE,
                    redaction_reason="redacted",
                )


class ReplayabilityPlanningContractTests(unittest.TestCase):
    """DS-02: plans are admitted or rejected without consuming the source."""

    def test_ds02_single_pass_second_pass_is_rejected_before_iteration(self) -> None:
        source = CountingSource(make_metadata(Replayability.SINGLE_PASS))

        with self.assertRaises(DataSourceCapabilityError) as caught:
            plan_passes(source, required_passes=2)

        self.assertEqual(caught.exception.code, "SPOOL_REQUIRED")
        self.assertEqual(caught.exception.context["required_passes"], 2)
        self.assertEqual(caught.exception.context["replayability"], "single_pass")
        self.assertEqual(source.read_count, 0)

    def test_ds02_replayable_source_is_admitted_within_budget(self) -> None:
        source = CountingSource(make_metadata(Replayability.REPLAYABLE))

        plan = plan_passes(source, required_passes=2)

        self.assertEqual(plan.required_passes, 2)
        self.assertFalse(plan.spool_enabled)
        self.assertEqual(source.read_count, 0)

    def test_ds02_checkpoint_replay_requires_compatible_checkpoint(self) -> None:
        source = CountingSource(
            make_metadata(
                Replayability.CHECKPOINT_REPLAYABLE,
                checkpoint_schema_version="checkpoint-v1",
            )
        )
        checkpoint = CheckpointMetadata(
            source_id="dataset:example-001", schema_version="checkpoint-v1"
        )

        plan = plan_passes(source, required_passes=2, checkpoint=checkpoint)

        self.assertEqual(plan.required_passes, 2)
        self.assertEqual(source.read_count, 0)

    def test_ds02_checkpoint_failures_are_typed_and_preflight_only(self) -> None:
        source = CountingSource(
            make_metadata(
                Replayability.CHECKPOINT_REPLAYABLE,
                checkpoint_schema_version="checkpoint-v1",
            )
        )
        with self.assertRaises(DataSourceCapabilityError) as required:
            plan_passes(source, required_passes=2)
        self.assertEqual(required.exception.code, "CHECKPOINT_REQUIRED")

        mismatches = (
            (
                CheckpointMetadata("dataset:other", "checkpoint-v1"),
                "CHECKPOINT_SOURCE_ID_MISMATCH",
            ),
            (
                CheckpointMetadata("dataset:example-001", "checkpoint-v2"),
                "CHECKPOINT_SCHEMA_MISMATCH",
            ),
        )
        for checkpoint, expected_code in mismatches:
            with self.subTest(expected_code=expected_code), self.assertRaises(
                DataSourceCapabilityError
            ) as incompatible:
                plan_passes(source, required_passes=2, checkpoint=checkpoint)
            self.assertEqual(incompatible.exception.code, expected_code)
        self.assertEqual(source.read_count, 0)

    def test_ds02_invalid_pass_budget_is_rejected_before_iteration(self) -> None:
        source = CountingSource(make_metadata(Replayability.REPLAYABLE))
        with self.assertRaises(ValueError):
            plan_passes(source, required_passes=0)
        with self.assertRaises(ValueError):
            CheckpointMetadata("", "checkpoint-v1")
        self.assertEqual(source.read_count, 0)


class ExplicitSpoolingContractTests(unittest.TestCase):
    """DS-03: a spool plan is declared, never silently materialized."""

    def test_ds03_spool_off_has_stable_error_and_zero_reads(self) -> None:
        source = CountingSource(make_metadata(Replayability.SINGLE_PASS))

        with self.assertRaises(DataSourceCapabilityError) as caught:
            plan_passes(source, required_passes=3, spool=SpoolPolicy.disabled())

        self.assertEqual(caught.exception.code, "SPOOL_REQUIRED")
        self.assertEqual(caught.exception.context["spool_enabled"], False)
        self.assertEqual(source.read_count, 0)

    def test_ds03_explicit_spool_records_requirements_without_reading(self) -> None:
        source = CountingSource(make_metadata(Replayability.SINGLE_PASS))
        spool = SpoolPolicy(
            enabled=True,
            disk_budget_bytes=10_000,
            retention="until-run-complete",
            cleanup_required=True,
        )

        plan = plan_passes(source, required_passes=3, spool=spool)

        self.assertTrue(plan.spool_enabled)
        self.assertEqual(plan.spool_requirements.disk_budget_bytes, 10_000)
        self.assertEqual(plan.spool_requirements.retention, "until-run-complete")
        self.assertTrue(plan.spool_requirements.cleanup_required)
        self.assertEqual(plan.provenance["spool"]["cleanup_required"], True)
        self.assertEqual(source.read_count, 0)

    def test_ds03_redacted_provenance_and_invalid_spool_policies(self) -> None:
        redacted_source = CountingSource(
            make_metadata(
                Replayability.REPLAYABLE,
                source_hash=None,
                redaction_reason="privacy-policy:restricted-input",
            )
        )
        plan = plan_passes(redacted_source, required_passes=1)
        self.assertNotIn("source_hash", plan.provenance)
        self.assertEqual(plan.provenance["redaction_reason"], "privacy-policy:restricted-input")

        invalid_policies = (
            {"enabled": False, "disk_budget_bytes": 1},
            {
                "enabled": True,
                "disk_budget_bytes": 0,
                "retention": "keep",
                "cleanup_required": True,
            },
            {
                "enabled": True,
                "disk_budget_bytes": 1,
                "retention": "",
                "cleanup_required": True,
            },
            {"enabled": True, "disk_budget_bytes": 1, "retention": "keep"},
        )
        for arguments in invalid_policies:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                SpoolPolicy(**arguments)
        self.assertEqual(redacted_source.read_count, 0)


if __name__ == "__main__":
    unittest.main()
