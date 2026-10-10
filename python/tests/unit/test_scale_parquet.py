"""The Parquet source: partitions, ids, fingerprints, content hashing and reading."""

from __future__ import annotations

import gc
import hashlib
import os
import re
import shutil
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tests.unit.scale_arrow_fixtures import chunk_bytes, set_mtime, write_parquet
from tests.unit.scale_source_oracle import reference_digest
from veridist.scale import _arrow
from veridist.scale._arrow import ParquetSource, row_group_record
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._sources import DEFAULT_MAX_ROWS, FingerprintLevel, Partition

CODES = ScaleSourceErrorCode
NS = 1_700_000_000_000_000_000


def code_of(call: Callable[[], object]) -> ScaleSourceErrorCode:
    try:
        call()
    except ScaleSourceError as error:
        return error.code
    raise AssertionError("the call was not refused")


def read(
    source: ParquetSource, partition: Partition, max_rows: int = DEFAULT_MAX_ROWS
) -> list[Any]:
    return list(source.batches(partition, max_rows=max_rows))


def read_values(
    source: ParquetSource, partition: Partition, max_rows: int = DEFAULT_MAX_ROWS
) -> Any:
    batches = read(source, partition, max_rows)
    return np.concatenate(batches).tolist() if batches else []


def write_standard(root: Path) -> None:
    """a.parquet: seven rows in row groups of 3, 3 and 1; sub/b.parquet: five rows."""

    write_parquet(
        root / "a.parquet",
        {
            "x": np.arange(7.0),
            "xy": np.arange(7.0) * 10,
            "s": ["p", "q", "r", "s", "t", "u", "v"],
        },
        row_group_size=3,
    )
    write_parquet(root / "sub" / "b.parquet", {"x": np.arange(10.0, 15.0)}, row_group_size=5)
    (root / "notes.txt").write_text("not parquet", encoding="utf-8")


class ReadOnlyFixture(unittest.TestCase):
    root: Path
    _directory: tempfile.TemporaryDirectory[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls._directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls._directory.name)
        write_standard(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._directory.cleanup()


class MutableFixture(unittest.TestCase):
    root: Path

    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        write_standard(self.root)


class EnumerationTests(ReadOnlyFixture):
    def test_one_partition_per_row_group_in_canonical_order(self) -> None:
        partitions = ParquetSource(self.root, "x").partitions()
        self.assertEqual(
            [p.id for p in partitions],
            ["a.parquet#0", "a.parquet#1", "a.parquet#2", "sub/b.parquet#0"],
        )
        self.assertEqual([p.ordinal for p in partitions], [0, 1, 2, 3])
        self.assertEqual([p.rows for p in partitions], [3, 3, 1, 5])
        self.assertTrue(all(p.fingerprint_level is FingerprintLevel.METADATA for p in partitions))
        self.assertTrue(all(p.resumable for p in partitions))
        self.assertEqual(len({p.fingerprint for p in partitions}), 4)

    def test_the_order_in_which_files_are_given_does_not_matter(self) -> None:
        expected = ParquetSource(self.root, "x").partitions()
        a, b = self.root / "a.parquet", self.root / "sub" / "b.parquet"
        for paths in ([a, b], [b, a], [str(b), str(a)], [self.root / "sub", a]):
            with self.subTest(paths=paths):
                self.assertEqual(ParquetSource(paths, "x", root=self.root).partitions(), expected)

    def test_a_directory_is_searched_recursively_for_parquet_files_only(self) -> None:
        ids = [p.id for p in ParquetSource(self.root, "x").partitions()]
        self.assertFalse(any("notes" in identity for identity in ids))
        self.assertEqual(
            [p.id for p in ParquetSource(self.root / "sub", "x").partitions()], ["b.parquet#0"]
        )

    def test_a_single_directory_is_its_own_root_and_a_declared_root_is_respected(self) -> None:
        self.assertEqual(
            [p.id for p in ParquetSource(self.root / "sub", "x").partitions()], ["b.parquet#0"]
        )
        under_parent = ParquetSource(self.root / "sub", "x", root=self.root).partitions()
        self.assertEqual([p.id for p in under_parent], ["sub/b.parquet#0"])

    def test_without_a_root_a_file_is_identified_by_its_absolute_posix_path(self) -> None:
        path = self.root / "a.parquet"
        partitions = ParquetSource(path, "x").partitions()
        prefix = Path(os.path.abspath(path)).as_posix()
        self.assertEqual([p.id for p in partitions], [f"{prefix}#0", f"{prefix}#1", f"{prefix}#2"])
        both = ParquetSource([path, self.root / "sub" / "b.parquet"], "x").partitions()
        self.assertEqual(len(both), 4)
        prefix = Path(os.path.abspath(self.root)).as_posix()
        self.assertTrue(all(p.id.startswith(prefix) for p in both))

    def test_str_and_pathlike_arguments_are_equivalent(self) -> None:
        as_path = ParquetSource(self.root, "x").partitions()
        self.assertEqual(ParquetSource(str(self.root), "x").partitions(), as_path)
        self.assertEqual(ParquetSource([str(self.root)], "x").partitions(), as_path)

    def test_a_declared_root_must_contain_every_file(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            outside = write_parquet(Path(other) / "o.parquet", {"x": [1.0]})
            with self.assertRaisesRegex(
                ValueError, "^every file must be under the declared root$"
            ):
                ParquetSource([self.root / "a.parquet", outside], "x", root=self.root)

    def test_constructor_refusals(self) -> None:
        column = "column must be a non-empty string"
        flag = "content_hash must be a bool"
        twice = "the same file is listed more than once"
        for call, error, message in (
            (lambda: ParquetSource([], "x"), ValueError, "paths must not be empty"),
            (lambda: ParquetSource(self.root, ""), TypeError, column),
            (lambda: ParquetSource(self.root, None), TypeError, column),  # type: ignore[arg-type]
            (lambda: ParquetSource(self.root, "x", content_hash=1), TypeError, flag),  # type: ignore[arg-type]
            (lambda: ParquetSource(self.root, "x", content_hash=None), TypeError, flag),  # type: ignore[arg-type]
            (lambda: ParquetSource([self.root / "a.parquet", self.root], "x"), ValueError, twice),
            (lambda: ParquetSource([self.root / "a.parquet"] * 2, "x"), ValueError, twice),
        ):
            with self.assertRaisesRegex(error, f"^{re.escape(message)}$"):
                call()
        for missing in (self.root / "nope.parquet", self.root / "nope"):
            self.assertIs(code_of(lambda m=missing: ParquetSource(m, "x")), CODES.SOURCE_NOT_FOUND)
        with tempfile.TemporaryDirectory() as empty:
            self.assertIs(code_of(lambda: ParquetSource(empty, "x")), CODES.SOURCE_NOT_FOUND)
            (Path(empty) / "only.txt").write_text("x", encoding="utf-8")
            self.assertIs(code_of(lambda: ParquetSource(empty, "x")), CODES.SOURCE_NOT_FOUND)

    def test_many_row_groups_keep_numeric_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_parquet(Path(directory) / "m.parquet", {"x": np.arange(24.0)}, row_group_size=2)
            partitions = ParquetSource(directory, "x").partitions()
        self.assertEqual([p.id for p in partitions], [f"m.parquet#{i}" for i in range(12)])
        self.assertEqual([p.ordinal for p in partitions], list(range(12)))

    def test_an_empty_row_group_is_a_partition_of_zero_rows_without_batches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "e.parquet"
            pq.write_table(pa.table({"x": pa.array([], pa.float64())}), path)
            source = ParquetSource(path, "x")
            (partition,) = source.partitions()
            self.assertEqual(partition.rows, 0)
            self.assertEqual(read(source, partition), [])

    def test_a_missing_column_is_refused_when_partitions_are_listed(self) -> None:
        source = ParquetSource(self.root, "xy")  # the column is absent from sub/b.parquet
        self.assertIs(code_of(source.partitions), CODES.COLUMN_NOT_FOUND)
        self.assertIs(code_of(ParquetSource(self.root, "zzz").partitions), CODES.COLUMN_NOT_FOUND)

    def test_a_corrupt_file_is_unreadable_and_the_message_has_no_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "secret-name.parquet"
            bad.write_bytes(b"this is not a parquet file")
            try:
                ParquetSource(bad, "x").partitions()
            except ScaleSourceError as error:
                outcome = (error.code, str(error), type(error.__cause__).__name__)
                del error
            else:
                self.fail("the corrupt file was accepted")
        self.assertEqual(outcome, (CODES.SOURCE_UNREADABLE, "SOURCE_UNREADABLE", "ArrowInvalid"))


class FingerprintTests(MutableFixture):
    def fingerprints(self, **options: Any) -> dict[str, str | None]:
        partitions = ParquetSource(self.root, "x", **options).partitions()
        return {p.id: p.fingerprint for p in partitions}

    def test_an_untouched_file_keeps_its_fingerprints(self) -> None:
        first = self.fingerprints()
        self.assertEqual(self.fingerprints(), first)
        again = ParquetSource(self.root, "x").partitions()
        self.assertEqual(ParquetSource(self.root, "x").partitions(), again)
        self.assertEqual(len(set(first.values())), 4)

    def test_a_rewritten_file_with_other_data_changes_only_its_own_partitions(self) -> None:
        before = self.fingerprints()
        write_parquet(self.root / "a.parquet", {"x": np.arange(7.0) + 100}, row_group_size=3)
        after = self.fingerprints()
        for identity in ("a.parquet#0", "a.parquet#1", "a.parquet#2"):
            self.assertNotEqual(after[identity], before[identity])
        self.assertEqual(after["sub/b.parquet#0"], before["sub/b.parquet#0"])

    def test_the_same_data_rewritten_later_changes_the_fingerprint_through_the_mtime(self) -> None:
        path = self.root / "a.parquet"
        set_mtime(path, NS)
        before = self.fingerprints()
        set_mtime(path, NS + 1_000_000_000)
        changed = self.fingerprints()
        self.assertNotEqual(changed["a.parquet#0"], before["a.parquet#0"])
        self.assertEqual(changed["sub/b.parquet#0"], before["sub/b.parquet#0"])
        set_mtime(path, NS)
        self.assertEqual(self.fingerprints(), before)

    def test_one_microsecond_of_mtime_is_enough(self) -> None:
        path = self.root / "sub" / "b.parquet"
        set_mtime(path, NS)
        before = self.fingerprints()["sub/b.parquet#0"]
        set_mtime(path, NS + 1000)
        self.assertNotEqual(self.fingerprints()["sub/b.parquet#0"], before)

    def test_identical_data_rewritten_with_the_same_mtime_is_indistinguishable(self) -> None:
        path = self.root / "a.parquet"
        set_mtime(path, NS)
        before = self.fingerprints()
        write_parquet(
            path,
            {
                "x": np.arange(7.0),
                "xy": np.arange(7.0) * 10,
                "s": ["p", "q", "r", "s", "t", "u", "v"],
            },
            row_group_size=3,
        )
        set_mtime(path, NS)
        self.assertEqual(self.fingerprints(), before)

    def test_changed_row_group_layout_or_codec_changes_the_fingerprint(self) -> None:
        path = self.root / "a.parquet"
        set_mtime(path, NS)
        before = self.fingerprints()
        data = {"x": np.arange(7.0), "xy": np.arange(7.0) * 10, "s": list("pqrstuv")}
        write_parquet(path, data, row_group_size=3, compression="NONE")
        set_mtime(path, NS)
        self.assertNotEqual(self.fingerprints()["a.parquet#0"], before["a.parquet#0"])

    def test_an_in_place_edit_that_keeps_size_mtime_and_metadata_is_not_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_parquet(
                Path(directory) / "p.parquet",
                {"x": np.arange(1.0, 9.0)},
                row_group_size=4,
                compression="NONE",
                use_dictionary=False,
            )
            set_mtime(path, NS)
            metadata_before = ParquetSource(path, "x").partitions()
            content_before = ParquetSource(path, "x", content_hash=True).partitions()
            start, size = chunk_bytes(path, 0)
            raw = bytearray(path.read_bytes())
            raw[start + size - 1] ^= 0x01
            path.write_bytes(bytes(raw))
            set_mtime(path, NS)
            self.assertEqual(ParquetSource(path, "x").partitions(), metadata_before)
            content_after = ParquetSource(path, "x", content_hash=True).partitions()
            self.assertNotEqual(content_after[0].fingerprint, content_before[0].fingerprint)
            self.assertEqual(content_after[1].fingerprint, content_before[1].fingerprint)

    def test_the_fingerprint_is_the_digest_of_the_documented_record(self) -> None:
        path = self.root / "a.parquet"
        set_mtime(path, NS)
        stat = path.stat()
        with pq.ParquetFile(path) as parquet:
            metadata = parquet.metadata
            row_group = metadata.row_group(1)
            chunk = row_group.column(0)
            stats = chunk.statistics
            self.assertEqual((stats.min, stats.max), (3.0, 5.0))
            chunk_record = (
                "x",
                str(chunk.physical_type),
                str(chunk.compression),
                tuple(sorted(str(e) for e in chunk.encodings)),
                int(chunk.has_dictionary_page),
                chunk.dictionary_page_offset if chunk.has_dictionary_page else None,
                chunk.data_page_offset,
                chunk.total_compressed_size,
                chunk.total_uncompressed_size,
                chunk.num_values,
                (float(stats.min).hex(), float(stats.max).hex(), 0, 3),
            )
            expected = (
                "parquet",
                "a.parquet",
                stat.st_size,
                stat.st_mtime_ns,
                1,
                metadata.num_row_groups,
                3,
                row_group.total_byte_size,
                "x",
                "double",
                (chunk_record,),
                None,
            )
            self.assertEqual(
                row_group_record(
                    "a.parquet",
                    stat.st_size,
                    stat.st_mtime_ns,
                    1,
                    3,
                    row_group,
                    "x",
                    "double",
                    None,
                ),
                expected,
            )
        partition = ParquetSource(self.root, "x").partitions()[1]
        self.assertEqual(partition.fingerprint, reference_digest(*expected))


class RecordTests(ReadOnlyFixture):
    def setUp(self) -> None:
        self._file = pq.ParquetFile(self.root / "a.parquet")
        self.enterContext(self._file)
        self.rg = [self._file.metadata.row_group(i) for i in range(3)]

    def record(self, **overrides: Any) -> tuple[object, ...]:
        fields: dict[str, Any] = {
            "identity": "a.parquet",
            "size": 100,
            "mtime_ns": 5,
            "row_group_index": 0,
            "row_group_count": 3,
            "row_group": self.rg[0],
            "column": "x",
            "column_type": "double",
            "content": None,
        }
        fields.update(overrides)
        return row_group_record(**fields)

    def test_every_component_changes_the_record(self) -> None:
        baseline = self.record()
        for overrides in (
            {"identity": "b.parquet"},
            {"size": 101},
            {"mtime_ns": 6},
            {"row_group_index": 1},
            {"row_group_count": 4},
            {"row_group": self.rg[2]},
            {"column": "xy"},
            {"column_type": "float"},
            {"content": "00"},
        ):
            with self.subTest(overrides=overrides):
                self.assertNotEqual(self.record(**overrides), baseline)

    def test_the_prefix_match_does_not_pick_up_a_column_with_a_longer_name(self) -> None:
        (chunk_record,) = self.record()[10]  # type: ignore[misc]
        self.assertEqual(chunk_record[0], "x")  # type: ignore[index]
        (other,) = self.record(column="xy")[10]  # type: ignore[misc]
        self.assertEqual(other[0], "xy")  # type: ignore[index]

    def test_statistics_text_is_exact(self) -> None:
        self.assertEqual(_arrow._stat_text(1.5), (1.5).hex())
        self.assertEqual(_arrow._stat_text(-0.0), "-0x0.0p+0")
        self.assertEqual(_arrow._stat_text(3), "3")
        self.assertEqual(_arrow._stat_text("a"), "'a'")
        self.assertEqual(_arrow._stat_text(None), "None")

    def test_a_chunk_without_a_dictionary_page_records_none_for_its_offset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_parquet(
                Path(directory) / "p.parquet", {"x": [1.0, 2.0]}, use_dictionary=False
            )
            with pq.ParquetFile(path) as parquet:
                chunk = parquet.metadata.row_group(0).column(0)
                record = _arrow._chunk_record(chunk)
                self.assertFalse(chunk.has_dictionary_page)
                self.assertEqual(record[4:6], (0, None))
                self.assertEqual(record[6], chunk.data_page_offset)
        self.assertEqual(
            self.record()[10][0][4:6],  # type: ignore[index]
            (1, self.rg[0].column(0).dictionary_page_offset),
        )

    def test_column_statistics_are_part_of_the_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plain = write_parquet(Path(directory) / "s.parquet", {"x": [1.0, 2.0]})
            none = write_parquet(
                Path(directory) / "n.parquet", {"x": [1.0, 2.0]}, write_statistics=False
            )
            nulls = write_parquet(
                Path(directory) / "z.parquet", {"x": pa.array([None, None], pa.float64())}
            )
            ints = write_parquet(Path(directory) / "i.parquet", {"x": [4, 9]})

            def statistics(path: Path) -> object:
                with pq.ParquetFile(path) as parquet:
                    record = row_group_record(
                        "f", 1, 1, 0, 1, parquet.metadata.row_group(0), "x", "t", None
                    )
                return record[10][0][-1]  # type: ignore[index]

            self.assertEqual(statistics(plain), ((1.0).hex(), (2.0).hex(), 0, 2))
            self.assertIsNone(statistics(none))
            self.assertEqual(statistics(nulls), (None, None, 2, 0))
            self.assertEqual(statistics(ints), ("4", "9", 0, 2))


class ContentHashTests(MutableFixture):
    def expected_content(self, path: Path, group: int, columns: tuple[int, ...] = (0,)) -> str:
        data = path.read_bytes()
        digest = hashlib.sha256()
        for column in columns:
            start, size = chunk_bytes(path, group, column)
            digest.update(data[start : start + size])
        return digest.hexdigest()

    def test_level_and_digest_of_the_chunk_bytes(self) -> None:
        path = self.root / "a.parquet"
        set_mtime(path, NS)
        stat = path.stat()
        partitions = ParquetSource(path, "x", content_hash=True).partitions()
        for group, partition in enumerate(partitions):
            self.assertIs(partition.fingerprint_level, FingerprintLevel.CONTENT)
            with pq.ParquetFile(path) as parquet:
                record = row_group_record(
                    Path(os.path.abspath(path)).as_posix(),
                    stat.st_size,
                    stat.st_mtime_ns,
                    group,
                    3,
                    parquet.metadata.row_group(group),
                    "x",
                    "double",
                    self.expected_content(path, group),
                )
            self.assertEqual(partition.fingerprint, reference_digest(*record))
        metadata_only = ParquetSource(path, "x").partitions()
        for weaker, stronger in zip(metadata_only, partitions, strict=True):
            self.assertNotEqual(weaker.fingerprint, stronger.fingerprint)

    def test_with_and_without_a_dictionary_page(self) -> None:
        for options in ({"use_dictionary": True}, {"use_dictionary": False}):
            path = write_parquet(
                self.root / "d.parquet", {"x": np.arange(6.0)}, row_group_size=3, **options
            )
            with self.subTest(options=options):
                with pq.ParquetFile(path) as parquet:
                    has_dictionary = parquet.metadata.row_group(0).column(0).has_dictionary_page
                self.assertEqual(has_dictionary, options["use_dictionary"])
                source = ParquetSource(path, "x", content_hash=True)
                first = source.partitions()
                self.assertEqual(first, source.partitions())
                set_mtime(path, NS)
                stat = path.stat()
                with pq.ParquetFile(path) as parquet:
                    record = row_group_record(
                        Path(os.path.abspath(path)).as_posix(),
                        stat.st_size,
                        stat.st_mtime_ns,
                        1,
                        2,
                        parquet.metadata.row_group(1),
                        "x",
                        "double",
                        self.expected_content(path, 1),
                    )
                self.assertEqual(
                    ParquetSource(path, "x", content_hash=True).partitions()[1].fingerprint,
                    reference_digest(*record),
                )

    def test_the_block_size_does_not_change_the_digest(self) -> None:
        path = self.root / "a.parquet"
        set_mtime(path, NS)
        expected = ParquetSource(path, "x", content_hash=True).partitions()
        for block in (1, 7, 64):
            with self.subTest(block=block), mock.patch.object(_arrow, "CONTENT_BLOCK_BYTES", block):
                self.assertEqual(ParquetSource(path, "x", content_hash=True).partitions(), expected)

    def test_a_nested_column_hashes_every_leaf_chunk(self) -> None:
        path = write_parquet(
            self.root / "n.parquet",
            {
                "id": np.arange(4.0),
                "st": pa.array([{"a": 1.0, "b": 2.0}] * 4),
                "st2": np.arange(4.0),
            },
            row_group_size=4,
            use_dictionary=False,
            compression="NONE",
        )
        with pq.ParquetFile(path) as parquet:
            names = [parquet.metadata.row_group(0).column(i).path_in_schema for i in range(4)]
        self.assertEqual(names, ["id", "st.a", "st.b", "st2"])
        set_mtime(path, NS)
        (partition,) = ParquetSource(path, "st", content_hash=True).partitions()
        stat = path.stat()
        with pq.ParquetFile(path) as parquet:
            record = row_group_record(
                Path(os.path.abspath(path)).as_posix(),
                stat.st_size,
                stat.st_mtime_ns,
                0,
                1,
                parquet.metadata.row_group(0),
                "st",
                str(parquet.schema_arrow.field("st").type),
                self.expected_content(path, 0, (1, 2)),
            )
        self.assertEqual([chunk[0] for chunk in record[10]], ["st.a", "st.b"])  # type: ignore[union-attr]
        self.assertEqual(partition.fingerprint, reference_digest(*record))


class ReadingTests(MutableFixture):
    def partitions(self, column: str = "x") -> tuple[ParquetSource, tuple[Partition, ...]]:
        source = ParquetSource(self.root, column)
        return source, source.partitions()

    def test_each_partition_reads_only_its_row_group(self) -> None:
        source, partitions = self.partitions()
        self.assertEqual(
            [read_values(source, p) for p in partitions],
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0], [10.0, 11.0, 12.0, 13.0, 14.0]],
        )

    def test_batches_are_bounded_by_max_rows(self) -> None:
        source, partitions = self.partitions()
        last = partitions[3]
        for max_rows, sizes in ((2, [2, 2, 1]), (1, [1] * 5), (5, [5]), (6, [5]), (4, [4, 1])):
            with self.subTest(max_rows=max_rows):
                batches = read(source, last, max_rows)
                self.assertEqual([len(b) for b in batches], sizes)
                self.assertEqual(np.concatenate(batches).tolist(), [10.0, 11.0, 12.0, 13.0, 14.0])
        self.assertEqual([len(b) for b in read(source, last)], [5])

    def test_the_default_bound_applies(self) -> None:
        path = write_parquet(self.root / "big.parquet", {"x": np.zeros(DEFAULT_MAX_ROWS + 3)})
        source = ParquetSource(path, "x")
        (partition,) = source.partitions()
        self.assertEqual([len(b) for b in read(source, partition)], [DEFAULT_MAX_ROWS, 3])

    def test_batches_are_validated_read_only_views_of_arrow_buffers(self) -> None:
        source, partitions = self.partitions()
        (batch,) = read(source, partitions[0])
        self.assertEqual(batch.dtype, np.float64)
        self.assertFalse(batch.flags["OWNDATA"])
        self.assertFalse(batch.flags["WRITEABLE"])

    def test_the_projection_and_row_group_are_pushed_down_to_the_reader(self) -> None:
        source, partitions = self.partitions()
        calls: list[dict[str, Any]] = []
        original = pq.ParquetFile.iter_batches

        def spy(self_: Any, *args: Any, **kwargs: Any) -> Any:
            calls.append({"args": args, **kwargs})
            return original(self_, *args, **kwargs)

        with mock.patch.object(pq.ParquetFile, "iter_batches", spy):
            read(source, partitions[1], 4)
        self.assertEqual(
            calls, [{"args": (), "batch_size": 4, "row_groups": [1], "columns": ["x"]}]
        )

    def test_other_columns_do_not_matter(self) -> None:
        wide = ParquetSource(self.root / "a.parquet", "xy")
        self.assertEqual(read_values(wide, wide.partitions()[0]), [0.0, 10.0, 20.0])
        source, partitions = self.partitions("x")
        self.assertEqual(read_values(source, partitions[0]), [0.0, 1.0, 2.0])

    def test_the_file_is_released_after_reading_and_after_an_abandoned_read(self) -> None:
        source, partitions = self.partitions()
        read(source, partitions[0])
        iterator = source.batches(partitions[1], max_rows=1)
        next(iterator)
        iterator.close()  # type: ignore[attr-defined]
        os.remove(self.root / "a.parquet")
        self.assertFalse((self.root / "a.parquet").exists())

    def test_a_partition_that_never_came_from_this_source_is_refused(self) -> None:
        source, partitions = self.partitions()
        foreign_text = "^partition does not belong to this source$"
        for identity in (
            "nohash",
            "a.parquet#x",
            "a.parquet#-1",
            "a.parquet#",
            "a.parquet#٣",
            "a.parquet#1.0",
            "missing.parquet#0",
            "#0",
            "sub#0",
            "A.parquet#0",
        ):
            foreign = Partition(identity, 0, None, None, FingerprintLevel.NONE)
            with self.subTest(identity=identity), self.assertRaisesRegex(ValueError, foreign_text):
                source.batches(foreign)
        for bad in ("a.parquet#0", None, 3):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, foreign_text):
                source.batches(bad)  # type: ignore[arg-type]
        for bad_rows, error in ((0, ValueError), (True, TypeError)):
            with self.assertRaises(error):
                source.batches(partitions[0], max_rows=bad_rows)  # type: ignore[arg-type]

    def test_a_partition_without_a_known_row_count_is_read_without_the_count_check(self) -> None:
        source, _ = self.partitions()
        loose = Partition("a.parquet#1", 1, None, None, FingerprintLevel.NONE)
        self.assertEqual(read_values(source, loose), [3.0, 4.0, 5.0])

    def test_a_changed_row_group_is_refused(self) -> None:
        source, partitions = self.partitions()
        wrong_rows = Partition("a.parquet#0", 0, 4, None, FingerprintLevel.NONE)
        self.assertIs(code_of(lambda: next(source.batches(wrong_rows))), CODES.PARTITION_CHANGED)
        beyond = Partition("a.parquet#3", 3, 3, None, FingerprintLevel.NONE)
        self.assertIs(code_of(lambda: next(source.batches(beyond))), CODES.PARTITION_CHANGED)
        beyond_unknown = Partition("a.parquet#9", 9, None, None, FingerprintLevel.NONE)
        refused = code_of(lambda: next(source.batches(beyond_unknown)))
        self.assertIs(refused, CODES.PARTITION_CHANGED)
        last = Partition("a.parquet#2", 2, 1, None, FingerprintLevel.NONE)
        self.assertEqual(read_values(source, last), [6.0])
        write_parquet(self.root / "a.parquet", {"x": np.arange(2.0)}, row_group_size=3)
        self.assertIs(code_of(lambda: next(source.batches(partitions[0]))), CODES.PARTITION_CHANGED)

    def test_a_file_that_disappears_or_breaks_is_unreadable(self) -> None:
        source, partitions = self.partitions()
        (self.root / "a.parquet").write_bytes(b"garbage")
        self.assertIs(code_of(lambda: read(source, partitions[0])), CODES.SOURCE_UNREADABLE)
        gc.collect()  # drop the frames that still hold the broken file open (Windows)
        os.remove(self.root / "a.parquet")
        self.assertIs(code_of(lambda: read(source, partitions[2])), CODES.SOURCE_UNREADABLE)

    def test_a_column_that_vanishes_is_not_found(self) -> None:
        source, partitions = self.partitions()
        write_parquet(self.root / "a.parquet", {"y": np.arange(7.0)}, row_group_size=3)
        self.assertIs(code_of(lambda: read(source, partitions[0])), CODES.COLUMN_NOT_FOUND)

    def test_the_value_policy_applies_to_parquet_columns(self) -> None:
        cases: list[tuple[str, dict[str, Any], ScaleSourceErrorCode | None]] = [
            ("f32", {"x": np.array([0.5, 1.5], dtype=np.float32)}, None),
            ("i32", {"x": np.array([1, 2], dtype=np.int32)}, None),
            ("i64", {"x": np.array([1, 2**53], dtype=np.int64)}, None),
            ("big", {"x": np.array([1, 2**53 + 1], dtype=np.int64)}, CODES.LOSSY_CAST),
            ("null", {"x": pa.array([1.0, None])}, CODES.NULL_VALUE),
            ("nan", {"x": np.array([1.0, np.nan])}, CODES.NAN_VALUE),
            ("inf", {"x": np.array([1.0, np.inf])}, CODES.NON_FINITE_VALUE),
            ("str", {"x": ["1.0", "2.0"]}, CODES.UNSUPPORTED_DTYPE),
            ("bool", {"x": [True, False]}, CODES.UNSUPPORTED_DTYPE),
            ("list", {"x": pa.array([[1.0], [2.0]])}, CODES.UNSUPPORTED_DTYPE),
            ("ts", {"x": pa.array([1, 2], pa.timestamp("s"))}, CODES.UNSUPPORTED_DTYPE),
        ]
        for name, columns, code in cases:
            path = write_parquet(self.root / f"{name}.parquet", columns)
            source = ParquetSource(path, "x")
            (partition,) = source.partitions()
            with self.subTest(case=name):
                if code is None:
                    self.assertEqual(len(np.concatenate(read(source, partition))), 2)
                else:
                    self.assertIs(code_of(lambda s=source, p=partition: read(s, p)), code)

    def test_a_refusal_after_good_batches_keeps_the_earlier_ones(self) -> None:
        path = write_parquet(
            self.root / "late.parquet", {"x": np.array([1.0, 2.0, np.nan, 4.0])}, row_group_size=4
        )
        source = ParquetSource(path, "x")
        (partition,) = source.partitions()
        iterator = source.batches(partition, max_rows=2)
        self.assertEqual(next(iterator).tolist(), [1.0, 2.0])
        self.assertIs(code_of(lambda: next(iterator)), CODES.NAN_VALUE)

    def test_a_nested_column_is_listed_but_refused_when_read(self) -> None:
        path = write_parquet(self.root / "nested.parquet", {"x": pa.array([[1.0], [2.0]])})
        source = ParquetSource(path, "x")
        (partition,) = source.partitions()
        self.assertEqual(partition.rows, 2)
        self.assertIs(code_of(lambda: read(source, partition)), CODES.UNSUPPORTED_DTYPE)

    def test_a_directory_of_files_is_read_through_its_partitions(self) -> None:
        source, partitions = self.partitions()
        everything = [v for p in partitions for v in read_values(source, p)]
        expected = [*np.arange(7.0).tolist(), *np.arange(10.0, 15.0).tolist()]
        self.assertEqual(sorted(everything), expected)


class CopyTests(MutableFixture):
    def test_a_tree_copied_with_its_timestamps_keeps_ids_and_fingerprints(self) -> None:
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / "copy"
            shutil.copytree(self.root, target, copy_function=shutil.copy2)
            original = ParquetSource(self.root, "x").partitions()
            copied = ParquetSource(target, "x").partitions()
        self.assertEqual(copied, original)

    def test_a_tree_copied_without_timestamps_keeps_ids_but_not_fingerprints(self) -> None:
        for path in self.root.rglob("*.parquet"):
            set_mtime(path, NS)
        with tempfile.TemporaryDirectory() as other:
            target = Path(other) / "copy"
            shutil.copytree(self.root, target, copy_function=shutil.copyfile)
            original = ParquetSource(self.root, "x").partitions()
            copied = ParquetSource(target, "x").partitions()
        self.assertEqual([p.id for p in copied], [p.id for p in original])
        for before, after in zip(original, copied, strict=True):
            self.assertNotEqual(before.fingerprint, after.fingerprint)


if __name__ == "__main__":
    unittest.main()
