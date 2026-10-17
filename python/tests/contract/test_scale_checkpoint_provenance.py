"""Contract of the checkpoint record of a scale fit: its checks, its totals and its JSON."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError, replace
from typing import Any

from tests.contract.test_scale_provenance_schema import (
    DIGEST_A,
    DIGEST_B,
    passes,
    record,
    spool,
    spool_partition,
)
from veridist.engine.provenance import (
    ScaleCheckpointPassProvenance,
    ScaleCheckpointProvenance,
    ScalePassRecord,
    scale_provenance_to_canonical_json_bytes,
)


def step(**changes: Any) -> ScaleCheckpointPassProvenance:
    base = ScaleCheckpointPassProvenance(0, 2, False, 1, 1)
    return replace(base, **changes)


def checkpoint(**changes: Any) -> ScaleCheckpointProvenance:
    base = ScaleCheckpointProvenance(
        "sqlite_partition", 2, DIGEST_A, True, False, 0, 0, 0, False, (step(),)
    )
    return replace(base, **changes)


def passes_of(count: int) -> Any:
    records = (
        ScalePassRecord("statistics", 0, None, None),
        *(ScalePassRecord("iteration", 1, None, None) for _ in range(count - 1)),
    )
    return passes(pass_count=count, records=records)


class PassRecordTests(unittest.TestCase):
    def test_a_valid_pass_keeps_its_fields_and_is_immutable(self) -> None:
        item = step()
        self.assertEqual(
            (item.pass_index, item.partitions, item.replayed, item.reused, item.read),
            (0, 2, False, 1, 1),
        )
        with self.assertRaises(FrozenInstanceError):
            item.read = 2  # type: ignore[misc]

    def test_counts_are_non_negative_integers(self) -> None:
        for name in ("pass_index", "partitions", "reused", "read"):
            for bad in (-1, True, 1.0, None):
                with self.subTest(name=name, value=bad), self.assertRaises((TypeError, ValueError)):
                    step(**{name: bad})

    def test_replayed_is_a_bool(self) -> None:
        for bad in (1, 0, None, "yes"):
            with self.subTest(value=bad), self.assertRaises(TypeError):
                step(replayed=bad)

    def test_reused_and_read_cannot_exceed_the_partitions(self) -> None:
        step(reused=2, read=0)
        step(reused=0, read=2)
        step(reused=0, read=0)
        for reused, read in ((2, 1), (3, 0), (0, 3), (1, 2)):
            with self.subTest(reused=reused, read=read), self.assertRaises(ValueError):
                step(reused=reused, read=read)

    def test_a_replayed_pass_reuses_every_partition_and_reads_none(self) -> None:
        item = step(replayed=True, reused=2, read=0)
        self.assertTrue(item.replayed)
        for reused, read in ((1, 0), (2, 1), (0, 0), (1, 1)):
            with self.subTest(reused=reused, read=read), self.assertRaises(ValueError):
                step(replayed=True, reused=reused, read=read)
        step(replayed=True, partitions=0, reused=0, read=0)


class RecordTests(unittest.TestCase):
    def test_a_valid_record_keeps_its_fields_and_is_immutable(self) -> None:
        item = checkpoint()
        self.assertEqual(
            (item.store_kind, item.store_format_version, item.job_digest_sha256),
            ("sqlite_partition", 2, DIGEST_A),
        )
        self.assertEqual((item.resumed, item.incremental, item.restarted), (True, False, False))
        self.assertEqual(item.passes, (step(),))
        with self.assertRaises(FrozenInstanceError):
            item.resumed = False  # type: ignore[misc]

    def test_the_store_and_the_digest_are_validated(self) -> None:
        for bad in ("", "has space", 3):
            with self.subTest(kind=bad), self.assertRaises((TypeError, ValueError)):
                checkpoint(store_kind=bad)
        for bad in (0, -1, True, 2.0):
            with self.subTest(version=bad), self.assertRaises((TypeError, ValueError)):
                checkpoint(store_format_version=bad)
        for bad in ("A" * 64, "a" * 63, "sha256:" + "a" * 64, 5):
            with self.subTest(digest=bad), self.assertRaises((TypeError, ValueError)):
                checkpoint(job_digest_sha256=bad)

    def test_the_flags_are_bools(self) -> None:
        for name in ("resumed", "incremental", "restarted"):
            for bad in (1, 0, None):
                with self.subTest(name=name, value=bad), self.assertRaises(TypeError):
                    checkpoint(**{name: bad})

    def test_the_counts_are_non_negative_integers(self) -> None:
        for name in ("added_partitions", "removed_partitions", "recomputed_partitions"):
            for bad in (-1, True, 1.0):
                with self.subTest(name=name, value=bad), self.assertRaises((TypeError, ValueError)):
                    checkpoint(**{name: bad})

    def test_the_passes_are_a_tuple_of_pass_records_numbered_from_zero(self) -> None:
        for bad in ([step()], ("x",), (step(), "x")):
            with self.subTest(passes=bad), self.assertRaises(TypeError):
                checkpoint(passes=bad)
        checkpoint(passes=(step(), step(pass_index=1)))
        checkpoint(passes=())
        for numbers in ((1,), (0, 2), (1, 0), (0, 0)):
            with self.subTest(numbers=numbers), self.assertRaises(ValueError):
                checkpoint(passes=tuple(step(pass_index=number) for number in numbers))

    def test_a_job_that_was_started_has_nothing_to_reuse_or_amend(self) -> None:
        fresh = checkpoint(resumed=False, passes=(step(reused=0, read=2),))
        self.assertFalse(fresh.resumed)
        for change in (
            {"added_partitions": 1, "incremental": True},
            {"removed_partitions": 1, "incremental": True},
            {"recomputed_partitions": 1},
            {"restarted": True, "recomputed_partitions": 1},
            {"passes": (step(reused=1, read=1),)},
            {"passes": (step(replayed=True, reused=2, read=0),)},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                checkpoint(resumed=False, **change)

    def test_partitions_are_added_or_removed_by_an_incremental_fit_only(self) -> None:
        checkpoint(added_partitions=1, incremental=True)
        checkpoint(removed_partitions=2, incremental=True)
        for change in ({"added_partitions": 1}, {"removed_partitions": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                checkpoint(**change)

    def test_a_restart_follows_from_a_recomputed_partition(self) -> None:
        checkpoint(restarted=True, recomputed_partitions=1)
        with self.assertRaises(ValueError):
            checkpoint(restarted=True, recomputed_partitions=0)
        checkpoint(recomputed_partitions=3)

    def test_the_totals_are_the_sums_over_the_passes(self) -> None:
        item = checkpoint(
            passes=(
                step(replayed=True, reused=2, read=0),
                step(pass_index=1, reused=1, read=1),
                step(pass_index=2, reused=0, read=2),
            )
        )
        self.assertEqual(item.reused_partitions, 3)
        self.assertEqual(item.read_partitions, 3)
        self.assertEqual(item.replayed_passes, 1)
        empty = checkpoint(passes=())
        self.assertEqual(
            (empty.reused_partitions, empty.read_partitions, empty.replayed_passes), (0, 0, 0)
        )


class ScaleRecordTests(unittest.TestCase):
    def test_a_record_has_no_checkpoint_unless_given_one(self) -> None:
        self.assertIsNone(record().checkpoint)
        self.assertEqual(record(checkpoint=checkpoint()).checkpoint, checkpoint())

    def test_the_checkpoint_must_be_typed(self) -> None:
        for bad in ("x", step(), 3):
            with self.subTest(value=bad), self.assertRaises(TypeError):
                record(checkpoint=bad)

    def test_it_has_one_entry_per_data_pass(self) -> None:
        record(checkpoint=checkpoint())
        with self.assertRaises(ValueError):
            record(checkpoint=checkpoint(passes=(step(), step(pass_index=1))))
        with self.assertRaises(ValueError):
            record(checkpoint=checkpoint(passes=()))
        three = tuple(step(pass_index=index) for index in range(3))
        record(passes=passes_of(3), checkpoint=checkpoint(passes=three))
        with self.assertRaises(ValueError):
            record(passes=passes_of(3), checkpoint=checkpoint(passes=three[:2]))

    def test_a_pass_cannot_cover_more_partitions_than_the_source_has(self) -> None:
        record(checkpoint=checkpoint(passes=(step(partitions=2, reused=0, read=0),)))
        with self.assertRaises(ValueError):
            record(checkpoint=checkpoint(passes=(step(partitions=3, reused=0, read=0),)))

    def test_a_spool_without_a_checkpoint_holds_exactly_the_partitions_and_rows(self) -> None:
        subset = spool(partitions=(spool_partition(),), row_count=10)
        with self.assertRaises(ValueError):
            record(spool=subset)

    def test_a_spool_beside_a_checkpoint_holds_some_of_the_partitions_each_once(self) -> None:
        subset = spool(partitions=(spool_partition(ordinal=1),), row_count=10)
        item = record(spool=subset, checkpoint=checkpoint())
        self.assertEqual(item.spool, subset)
        record(spool=spool(partitions=(), row_count=0), checkpoint=checkpoint())
        record(spool=spool(), checkpoint=checkpoint())
        outside = spool(partitions=(spool_partition(ordinal=5),), row_count=10)
        with self.assertRaises(ValueError):
            record(spool=outside, checkpoint=checkpoint())
        twice = spool(
            partitions=(spool_partition(), spool_partition(rows=7, digest_sha256=DIGEST_B)),
            row_count=17,
        )
        with self.assertRaises(ValueError):
            record(spool=twice, checkpoint=checkpoint())


class JsonTests(unittest.TestCase):
    def test_a_record_without_a_checkpoint_has_no_checkpoint_key(self) -> None:
        self.assertNotIn(b"checkpoint", scale_provenance_to_canonical_json_bytes(record()))

    def test_the_checkpoint_is_encoded_with_every_field_and_sorted_keys(self) -> None:
        item = checkpoint(
            incremental=True,
            added_partitions=1,
            removed_partitions=2,
            recomputed_partitions=3,
            restarted=True,
            passes=(step(replayed=True, reused=2, read=0),),
        )
        data = scale_provenance_to_canonical_json_bytes(record(checkpoint=item))
        value = json.loads(data)["checkpoint"]
        self.assertEqual(
            value,
            {
                "added_partitions": 1,
                "incremental": True,
                "job_digest_sha256": DIGEST_A,
                "passes": [
                    {
                        "partitions": 2,
                        "pass_index": 0,
                        "read": 0,
                        "replayed": True,
                        "reused": 2,
                    }
                ],
                "recomputed_partitions": 3,
                "removed_partitions": 2,
                "restarted": True,
                "resumed": True,
                "store_format_version": 2,
                "store_kind": "sqlite_partition",
            },
        )
        expected = (
            '"checkpoint":{"added_partitions":1,"incremental":true,'
            f'"job_digest_sha256":"{DIGEST_A}","passes":[{{"partitions":2,"pass_index":0,'
            '"read":0,"replayed":true,"reused":2}],"recomputed_partitions":3,'
            '"removed_partitions":2,"restarted":true,"resumed":true,'
            '"store_format_version":2,"store_kind":"sqlite_partition"}'
        )
        self.assertIn(expected.encode(), data)

    def test_the_other_fields_are_those_of_the_record_without_one(self) -> None:
        without = json.loads(scale_provenance_to_canonical_json_bytes(record()))
        with_one = json.loads(
            scale_provenance_to_canonical_json_bytes(record(checkpoint=checkpoint()))
        )
        del with_one["checkpoint"]
        self.assertEqual(with_one, without)

    def test_the_encoding_holds_no_path_identifier_or_fingerprint_of_a_partition(self) -> None:
        data = scale_provenance_to_canonical_json_bytes(record(checkpoint=checkpoint()))
        checkpoint_text = json.dumps(json.loads(data)["checkpoint"])
        self.assertNotIn("parquet", checkpoint_text)
        self.assertNotIn("sha256:", checkpoint_text)
