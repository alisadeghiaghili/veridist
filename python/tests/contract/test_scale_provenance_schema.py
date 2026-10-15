"""Contracts of the schema 2 scale provenance record, and that schema 1 is untouched."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace
from typing import Any

from tests.contract.test_execution_provenance import provenance
from veridist.engine.data_source import SUPPORTED_PROVENANCE_SCHEMA_VERSIONS
from veridist.engine.outcome import CompleteOutcome, KnownCoverage, KnownExtent, RowRange
from veridist.engine.provenance import (
    PROVENANCE_SCHEMA_VERSION,
    SCALE_PROVENANCE_SCHEMA_VERSION,
    ExecutionProvenance,
    ExecutionReport,
    ScaleEnvironmentProvenance,
    ScalePartitionProvenance,
    ScaleProvenance,
    ScaleStateProvenance,
    scale_provenance_to_canonical_json_bytes,
    to_canonical_json_bytes,
)

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64
FINGERPRINT = "sha256:" + "c" * 64


def partition(**changes: Any) -> ScalePartitionProvenance:
    base = ScalePartitionProvenance("a.parquet#0", 0, 10, FINGERPRINT, "metadata")
    return replace(base, **changes)


def record(**changes: Any) -> ScaleProvenance:
    base = ScaleProvenance(
        "2",
        "parquet",
        (partition(), partition(partition_id="a.parquet#1", ordinal=1, rows=None)),
        2,
        17,
        4096,
        ScaleStateProvenance("scale.normal", 1, DIGEST_A),
        ScaleEnvironmentProvenance("2.5.4", DIGEST_B),
    )
    return replace(base, **changes)


class SchemaVersionTests(unittest.TestCase):
    def test_scale_records_are_schema_2_and_execution_provenance_stays_schema_1(self) -> None:
        self.assertEqual(SCALE_PROVENANCE_SCHEMA_VERSION, "2")
        self.assertEqual(PROVENANCE_SCHEMA_VERSION, "1")
        self.assertEqual(SUPPORTED_PROVENANCE_SCHEMA_VERSIONS, frozenset({"1"}))

    def test_a_schema_1_report_still_encodes_and_still_refuses_schema_2(self) -> None:
        coverage = KnownCoverage(KnownExtent(0, 3), (RowRange(0, 3),), 3, 0)
        report = ExecutionReport(CompleteOutcome(coverage), provenance())
        value = json.loads(to_canonical_json_bytes(report))
        self.assertEqual(value["schema_version"], "1")
        with self.assertRaises(ValueError):
            replace(provenance(), schema_version="2")
        self.assertIsInstance(provenance(), ExecutionProvenance)

    def test_the_record_accepts_only_its_own_version(self) -> None:
        for version in ("1", "3", "", 2):
            with self.subTest(version=version), self.assertRaises(ValueError):
                record(schema_version=version)


class PartitionRecordTests(unittest.TestCase):
    def test_a_valid_record_keeps_its_fields(self) -> None:
        item = partition()
        self.assertEqual(
            (item.partition_id, item.ordinal, item.rows, item.fingerprint, item.fingerprint_level),
            ("a.parquet#0", 0, 10, FINGERPRINT, "metadata"),
        )

    def test_a_partition_without_a_fingerprint_is_allowed_only_at_level_none(self) -> None:
        item = partition(fingerprint=None, fingerprint_level="none")
        self.assertIsNone(item.fingerprint)
        with self.assertRaises(ValueError):
            partition(fingerprint=None, fingerprint_level="content")
        with self.assertRaises(ValueError):
            partition(fingerprint=FINGERPRINT, fingerprint_level="none")

    def test_levels_and_fingerprints_are_validated(self) -> None:
        for level in ("content", "metadata"):
            partition(fingerprint_level=level)
        for level in ("full", "", "NONE"):
            with self.subTest(level=level), self.assertRaises(ValueError):
                partition(fingerprint_level=level)
        for bad in ("sha256:" + "C" * 64, "sha256:" + "c" * 63, "md5:" + "c" * 64, "c" * 64):
            with self.subTest(fingerprint=bad), self.assertRaises(ValueError):
                partition(fingerprint=bad)
        with self.assertRaises(ValueError):
            partition(fingerprint=FINGERPRINT + "0")
        with self.assertRaises(ValueError):
            partition(fingerprint=7)

    def test_identifier_ordinal_and_rows_are_validated(self) -> None:
        for bad_id in ("", 3, None):
            with self.subTest(partition_id=bad_id), self.assertRaises(TypeError):
                partition(partition_id=bad_id)
        for bad_ordinal in (-1, True, 1.0):
            with self.subTest(ordinal=bad_ordinal), self.assertRaises((TypeError, ValueError)):
                partition(ordinal=bad_ordinal)
        for bad_rows in (-1, True, 1.5):
            with self.subTest(rows=bad_rows), self.assertRaises((TypeError, ValueError)):
                partition(rows=bad_rows)
        self.assertEqual(partition(rows=0).rows, 0)
        self.assertIsNone(partition(rows=None).rows)


class StateAndEnvironmentRecordTests(unittest.TestCase):
    def test_state_fields_are_validated(self) -> None:
        ScaleStateProvenance("scale.gamma", 1, DIGEST_A)
        with self.assertRaises(ValueError):
            ScaleStateProvenance("scale gamma", 1, DIGEST_A)
        with self.assertRaises(TypeError):
            ScaleStateProvenance(3, 1, DIGEST_A)  # type: ignore[arg-type]
        for version in (0, -1, True):
            with self.subTest(version=version), self.assertRaises((TypeError, ValueError)):
                ScaleStateProvenance("scale.gamma", version, DIGEST_A)
        for digest in ("A" * 64, "a" * 63, "sha256:" + "a" * 64):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                ScaleStateProvenance("scale.gamma", 1, digest)

    def test_environment_fields_are_validated(self) -> None:
        ScaleEnvironmentProvenance("2.5.4", DIGEST_B)
        ScaleEnvironmentProvenance("2.0.0.dev0+git20240101", DIGEST_B)
        with self.assertRaises(ValueError):
            ScaleEnvironmentProvenance("2.5 4", DIGEST_B)
        with self.assertRaises(ValueError):
            ScaleEnvironmentProvenance("", DIGEST_B)
        with self.assertRaises(ValueError):
            ScaleEnvironmentProvenance("2.5.4", "z" * 64)


class ScaleProvenanceTests(unittest.TestCase):
    def test_a_valid_record_keeps_its_fields(self) -> None:
        item = record()
        self.assertEqual(item.schema_version, "2")
        self.assertEqual(item.source_kind, "parquet")
        self.assertEqual(item.partition_count, 2)
        self.assertEqual(item.row_count, 17)
        self.assertEqual(item.max_rows, 4096)
        self.assertEqual(item.state, ScaleStateProvenance("scale.normal", 1, DIGEST_A))
        self.assertEqual(item.environment, ScaleEnvironmentProvenance("2.5.4", DIGEST_B))
        self.assertEqual(len(item.partitions), 2)

    def test_the_record_is_frozen(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            record().row_count = 1  # type: ignore[misc]

    def test_a_record_may_have_no_state_when_the_data_were_refused(self) -> None:
        self.assertIsNone(record(state=None).state)

    def test_an_empty_source_has_no_partitions(self) -> None:
        item = record(partitions=(), partition_count=0, row_count=0)
        self.assertEqual(item.partitions, ())

    def test_the_partition_count_must_match(self) -> None:
        with self.assertRaises(ValueError):
            record(partition_count=1)
        with self.assertRaises(ValueError):
            record(partition_count=3)

    def test_partitions_must_be_a_tuple_of_partition_records(self) -> None:
        with self.assertRaises(TypeError):
            record(partitions=[partition()], partition_count=1)
        with self.assertRaises(TypeError):
            record(partitions=("a.parquet#0",), partition_count=1)

    def test_partition_identifiers_must_be_unique(self) -> None:
        with self.assertRaises(ValueError):
            record(partitions=(partition(), partition(ordinal=1)))

    def test_counts_and_cap_are_validated(self) -> None:
        for bad in (-1, True, 1.5):
            with self.subTest(row_count=bad), self.assertRaises((TypeError, ValueError)):
                record(row_count=bad)
        for bad in (0, -1, True, 2.0):
            with self.subTest(max_rows=bad), self.assertRaises((TypeError, ValueError)):
                record(max_rows=bad)

    def test_source_kind_is_a_token(self) -> None:
        for bad in ("", "my source", 3):
            with self.subTest(kind=bad), self.assertRaises((TypeError, ValueError)):
                record(source_kind=bad)

    def test_state_and_environment_must_be_typed(self) -> None:
        with self.assertRaises(TypeError):
            record(state="digest")
        with self.assertRaises(TypeError):
            record(environment=None)


class CanonicalJsonTests(unittest.TestCase):
    def test_the_bytes_are_exact(self) -> None:
        expected = (
            '{"environment":{"digest_sha256":"' + DIGEST_B + '","numpy_version":"2.5.4"},'
            '"max_rows":4096,"partition_count":2,"partitions":['
            '{"fingerprint":"' + FINGERPRINT + '","fingerprint_level":"metadata","ordinal":0,'
            '"partition_id":"a.parquet#0","rows":10},'
            '{"fingerprint":"' + FINGERPRINT + '","fingerprint_level":"metadata","ordinal":1,'
            '"partition_id":"a.parquet#1","rows":null}],'
            '"row_count":17,"schema_version":"2","source_kind":"parquet",'
            '"state":{"digest_sha256":"' + DIGEST_A + '","state_schema_version":1,'
            '"state_tag":"scale.normal"}}'
        )
        self.assertEqual(scale_provenance_to_canonical_json_bytes(record()), expected.encode())

    def test_a_missing_state_and_fingerprint_are_null(self) -> None:
        item = record(
            partitions=(partition(fingerprint=None, fingerprint_level="none"),),
            partition_count=1,
            state=None,
        )
        value = json.loads(scale_provenance_to_canonical_json_bytes(item))
        self.assertIsNone(value["state"])
        self.assertIsNone(value["partitions"][0]["fingerprint"])
        self.assertEqual(value["partitions"][0]["fingerprint_level"], "none")

    def test_equal_records_have_equal_bytes_and_a_change_changes_them(self) -> None:
        first = scale_provenance_to_canonical_json_bytes(record())
        self.assertEqual(scale_provenance_to_canonical_json_bytes(record()), first)
        self.assertNotEqual(scale_provenance_to_canonical_json_bytes(record(row_count=18)), first)

    def test_non_ascii_identifiers_are_kept_as_utf8(self) -> None:
        item = record(
            partitions=(partition(partition_id="déjà/فا.parquet#0"),),
            partition_count=1,
        )
        data = scale_provenance_to_canonical_json_bytes(item)
        self.assertIn("déjà/فا.parquet#0".encode(), data)
        self.assertEqual(json.loads(data)["partitions"][0]["partition_id"][:3], "déj")

    def test_only_a_scale_record_is_encoded(self) -> None:
        with self.assertRaises(TypeError):
            scale_provenance_to_canonical_json_bytes(provenance())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
