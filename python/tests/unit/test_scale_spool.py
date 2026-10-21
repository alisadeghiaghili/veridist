"""The opt-in local spool of the multi-pass scale fit: policy, mechanics, refusals, cleanup.

Every test uses temporary directories. Nothing here needs more than a few thousand values.
"""

from __future__ import annotations

import hashlib
import os
import stat
import struct
import sys
import tempfile
import unittest
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pyarrow as pa

from tests.unit.scale_arrow_fixtures import StreamExporter, write_parquet
from tests.unit.scale_fit_oracle import ListSource
from tests.unit.scale_multi_pass_oracle import gumbel_sample, weibull_sample
from veridist.engine.errors import FailureCode
from veridist.engine.pass_budget import PassBudgetError
from veridist.engine.provenance import scale_provenance_to_canonical_json_bytes
from veridist.families.gumbel import GumbelFitFailure, GumbelFitFailureCode
from veridist.families.weibull import WeibullFitFailure, WeibullFitFailureCode
from veridist.scale import _fit, _multi_pass, _spool
from veridist.scale._arrow import ArrowSource, ParquetSource
from veridist.scale._errors import ScaleSourceError, ScaleSourceErrorCode
from veridist.scale._fit import ScaleFitResult, fit
from veridist.scale._sources import (
    DEFAULT_MAX_ROWS,
    FingerprintLevel,
    NumpySource,
    Partition,
    Source,
    is_single_use,
)
from veridist.scale._spool import (
    BATCH_FRAMING_BYTES,
    FILE_FRAMING_BYTES,
    ScaleSpoolPolicy,
    Spool,
    SpooledSource,
    batch_cost,
)

CODES = ScaleSourceErrorCode
SCHEMA = pa.schema([pa.field("x", pa.float64())])
BLOCK_PYARROW = {"pyarrow": None, "pyarrow.parquet": None}
ARROW_HINT = 'install the optional "arrow" extra with: pip install "veridist[arrow]"'


def weibull(size: int = 3000, seed: int = 5) -> np.ndarray:
    return weibull_sample(1.7, 1000.0, size, seed)


def gumbel(size: int = 3000, seed: int = 6) -> np.ndarray:
    return gumbel_sample(50.0, 4.0, size, seed)


def digest_of(values: Any) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def stream_source(values: np.ndarray, rows: int = 700, pulled: list[int] | None = None) -> Any:
    """An Arrow source over a reader: it can be read once. ``pulled`` collects batch sizes."""

    def generate() -> Iterator[Any]:
        for start in range(0, len(values), rows):
            part = values[start : start + rows]
            if pulled is not None:
                pulled.append(len(part))
            yield pa.record_batch([pa.array(part)], names=["x"])

    return ArrowSource(pa.RecordBatchReader.from_batches(SCHEMA, generate()), "x")


class Counting:
    """Delegates to a source and counts the calls of ``batches`` and ``partitions``."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.batch_calls = 0
        self.partition_calls = 0

    @property
    def single_use(self) -> bool:
        return is_single_use(self.inner)

    def partitions(self) -> tuple[Partition, ...]:
        self.partition_calls += 1
        partitions: tuple[Partition, ...] = self.inner.partitions()
        return partitions

    def batches(self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS) -> Iterator[Any]:
        self.batch_calls += 1
        batches: Iterator[Any] = self.inner.batches(partition, max_rows=max_rows)
        return batches


def code_of(call: Any) -> ScaleSourceErrorCode:
    try:
        call()
    except ScaleSourceError as error:
        return error.code
    raise AssertionError("the call was not refused")


def entries(directory: Path | str) -> list[str]:
    return sorted(os.listdir(directory))


def spool_files(directory: Path | str) -> list[Path]:
    """The files of the one spool directory made inside ``directory``."""

    (private,) = entries(directory)
    return sorted((Path(directory) / private).iterdir())


def result_bits(result: ScaleFitResult) -> tuple[bytes, ...]:
    success: Any = result.fit
    numbers = [*success.parameters.values(), success.log_likelihood]
    uncertainty = success.uncertainty()
    for row in uncertainty.covariance:
        numbers.extend(row)
    for lower, upper in uncertainty.confidence_intervals(0.95, "wald").values():
        numbers.extend((lower, upper))
    return tuple(struct.pack("<d", float(value)) for value in numbers)


class TempCase(unittest.TestCase):
    """A test with a temporary directory for the spool to be made in."""

    def setUp(self) -> None:
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.root = Path(holder.name)
        self.directory = self.root / "spools"
        self.directory.mkdir()

    def policy(self, max_bytes: int = 1 << 24) -> ScaleSpoolPolicy:
        return ScaleSpoolPolicy(directory=self.directory, max_bytes=max_bytes)

    def assert_clean(self) -> None:
        self.assertEqual(entries(self.directory), [])

    def assert_no_leak(self, error: BaseException) -> None:
        self.assertNotIn(str(self.directory), str(error))
        self.assertNotIn(str(self.root), repr(error))
        self.assertIsNone(error.__cause__)


class ConstantTests(unittest.TestCase):
    def test_the_layout_constants_are_pinned(self) -> None:
        self.assertEqual(_spool.DIRECTORY_PREFIX, "veridist-spool-")
        self.assertEqual(_spool.FILE_SUFFIX, ".arrow")
        self.assertEqual(_spool.FILE_MODE, 0o600)
        self.assertEqual(_spool.COLUMN_NAME, "x")
        self.assertEqual(_spool.VALUE_BYTES, 8)
        self.assertEqual(BATCH_FRAMING_BYTES, 256)
        self.assertEqual(FILE_FRAMING_BYTES, 1024)

    def test_a_batch_is_charged_its_values_and_its_framing(self) -> None:
        self.assertEqual(batch_cost(0), 256)
        self.assertEqual(batch_cost(1), 264)
        self.assertEqual(batch_cost(1000), 8256)

    def test_the_hints_are_constant_sentences_without_a_path(self) -> None:
        self.assertEqual(
            _spool.HINT_REQUIRED,
            "the source can be read only once but this family reads it several times; "
            "pass spool=ScaleSpoolPolicy(...) to keep a local copy",
        )
        self.assertEqual(
            _spool.HINT_LIMIT,
            "the spool would exceed max_bytes; raise max_bytes of the ScaleSpoolPolicy",
        )
        self.assertEqual(
            _spool.HINT_CHANGED,
            "the spool no longer holds what was written to it; it was changed or damaged",
        )
        self.assertEqual(
            _spool.HINT_UNAVAILABLE,
            "the spool could not be used; check that its directory exists, is writable "
            "and has room",
        )
        self.assertEqual(
            _spool.HINT_CLEANUP,
            "the spool could not be fully deleted; remove the veridist-spool-* directories "
            "left in the spool directory",
        )

    def test_the_charges_bound_what_the_ipc_format_writes(self) -> None:
        """The charge of a file never falls below its real size, whatever the batching."""

        with tempfile.TemporaryDirectory() as holder:
            for sizes in ((), (1,), (1, 1, 1), (7, 3), (1000,), (DEFAULT_MAX_ROWS,), (5,) * 50):
                spool_directory = Path(holder) / "d"
                spool_directory.mkdir()
                data = np.arange(1.0, sum(sizes) + 1.0)
                source = ListSource([data])
                with Spool(ScaleSpoolPolicy(directory=spool_directory, max_bytes=1 << 26)) as spool:
                    spooled = spool.source(_Chunked(source, sizes))
                    (partition,) = spooled.partitions()
                    list(spooled.batches(partition))
                    (path,) = spool_files(spool_directory)
                    size = path.stat().st_size
                    charged = spool._charged
                self.assertLessEqual(size, charged, sizes)
                self.assertEqual(
                    charged, FILE_FRAMING_BYTES + sum(batch_cost(count) for count in sizes), sizes
                )
                spool_directory.rmdir()


class _Chunked:
    """One partition whose batches have exactly the given sizes."""

    def __init__(self, source: ListSource, sizes: tuple[int, ...]) -> None:
        self._source = source
        self._sizes = sizes

    def partitions(self) -> tuple[Partition, ...]:
        return self._source.partitions()

    def batches(self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS) -> Iterator[Any]:
        data = self._source.parts[partition.ordinal]
        start = 0
        for size in self._sizes:
            yield data[start : start + size]
            start += size


class PolicyTests(unittest.TestCase):
    def test_the_fields_and_defaults(self) -> None:
        policy = ScaleSpoolPolicy(directory="spools", max_bytes=10)
        self.assertEqual((policy.directory, policy.max_bytes), ("spools", 10))
        self.assertIs(policy.allow_system_temp, False)
        system = ScaleSpoolPolicy(max_bytes=1, allow_system_temp=True)
        self.assertIsNone(system.directory)
        self.assertIs(system.allow_system_temp, True)

    def test_a_policy_is_frozen_and_takes_keywords_only(self) -> None:
        policy = ScaleSpoolPolicy(directory="spools", max_bytes=10)
        with self.assertRaises(FrozenInstanceError):
            policy.max_bytes = 3  # type: ignore[misc]
        with self.assertRaises(TypeError):
            ScaleSpoolPolicy("spools", 10)  # type: ignore[misc]
        self.assertEqual(replace(policy, max_bytes=11).max_bytes, 11)

    def test_the_budget_is_a_positive_built_in_integer(self) -> None:
        for bad in (True, 1.0, "10", None, 2**20 + 0.5):
            with self.subTest(max_bytes=bad), self.assertRaises(TypeError):
                ScaleSpoolPolicy(directory="spools", max_bytes=bad)  # type: ignore[arg-type]
        for bad in (0, -1):
            with self.subTest(max_bytes=bad), self.assertRaises(ValueError):
                ScaleSpoolPolicy(directory="spools", max_bytes=bad)
        self.assertEqual(ScaleSpoolPolicy(directory="spools", max_bytes=1).max_bytes, 1)

    def test_a_spool_never_chooses_where_to_write(self) -> None:
        with self.assertRaises(ValueError):
            ScaleSpoolPolicy(max_bytes=10)
        with self.assertRaises(ValueError):
            ScaleSpoolPolicy(directory=None, max_bytes=10, allow_system_temp=False)

    def test_a_directory_and_the_system_directory_are_exclusive(self) -> None:
        with self.assertRaises(ValueError):
            ScaleSpoolPolicy(directory="spools", max_bytes=10, allow_system_temp=True)

    def test_the_system_flag_is_a_bool(self) -> None:
        for bad in (1, 0, "yes", None):
            with self.subTest(flag=bad), self.assertRaises(TypeError):
                ScaleSpoolPolicy(max_bytes=10, allow_system_temp=bad)  # type: ignore[arg-type]

    def test_the_directory_is_a_text_path(self) -> None:
        class BytesPath:
            def __fspath__(self) -> bytes:
                return b"spools"

        for bad in (3, b"spools", BytesPath(), ["spools"]):
            with self.subTest(directory=bad), self.assertRaises(TypeError):
                ScaleSpoolPolicy(directory=bad, max_bytes=10)  # type: ignore[arg-type]
        ScaleSpoolPolicy(directory=Path("spools"), max_bytes=10)

    def test_the_base_directory_is_absolute_or_the_system_one(self) -> None:
        self.assertEqual(
            ScaleSpoolPolicy(max_bytes=1, allow_system_temp=True).base_directory(),
            tempfile.gettempdir(),
        )
        self.assertEqual(
            ScaleSpoolPolicy(directory=".", max_bytes=1).base_directory(), os.path.abspath(".")
        )
        with tempfile.TemporaryDirectory() as holder:
            self.assertEqual(
                ScaleSpoolPolicy(directory=Path(holder), max_bytes=1).base_directory(),
                os.path.abspath(holder),
            )
            self.assertEqual(
                ScaleSpoolPolicy(directory=holder, max_bytes=1).base_directory(),
                os.path.abspath(holder),
            )


class SourceDeclarationTests(unittest.TestCase):
    def test_only_a_source_that_says_so_is_single_use(self) -> None:
        values = weibull(10)
        self.assertTrue(is_single_use(stream_source(values)))
        self.assertTrue(is_single_use(ArrowSource(StreamExporter(pa.table({"x": values})), "x")))
        self.assertFalse(is_single_use(ArrowSource(pa.table({"x": values}), "x")))
        self.assertFalse(is_single_use(ArrowSource(pa.record_batch({"x": values}), "x")))
        self.assertFalse(is_single_use(NumpySource(values)))
        self.assertFalse(is_single_use(ListSource([values])))
        self.assertFalse(is_single_use(object()))

        class Loose:
            single_use: Any = "yes"

        self.assertFalse(is_single_use(Loose()))
        Loose.single_use = True
        self.assertTrue(is_single_use(Loose()))

    def test_a_consumed_source_is_still_single_use(self) -> None:
        source = stream_source(weibull(10))
        (partition,) = source.partitions()
        list(source.batches(partition))
        self.assertTrue(source.single_use)


class SpoolMechanicsTests(TempCase):
    def parts(self) -> list[np.ndarray]:
        data = weibull(2500)
        return [data[:1000], data[1000:1000], data[1000:2250], data[2250:]]

    def test_the_first_read_writes_and_every_later_read_replays_the_spool(self) -> None:
        parts = self.parts()
        source = ListSource(parts)
        with Spool(self.policy()) as spool:
            spooled = spool.source(source)
            self.assertIsInstance(spooled, SpooledSource)
            self.assertIsInstance(spooled, Source)
            partitions = spooled.partitions()
            first = [list(spooled.batches(p, max_rows=400)) for p in partitions]
            self.assertEqual(source.batch_calls, [0, 1, 2, 3])
            self.assertEqual(source.batch_sizes, [400, 400, 200, 400, 400, 400, 50, 250])
            for _ in range(3):
                again = [list(spooled.batches(p, max_rows=400)) for p in partitions]
                self.assertEqual(source.batch_calls, [0, 1, 2, 3])
                for before, after in zip(first, again, strict=True):
                    self.assertEqual(
                        [b.tolist() for b in before], [a.tolist() for a in after]
                    )
            self.assertEqual(
                [len(b) for batches in first for b in batches],
                [400, 400, 200, 400, 400, 400, 50, 250],
            )
            self.assertEqual(
                [path.name for path in spool_files(self.directory)],
                ["00000000.arrow", "00000001.arrow", "00000002.arrow", "00000003.arrow"],
            )
        self.assert_clean()

    def test_the_replay_keeps_the_batches_and_the_bits(self) -> None:
        values = np.array([0.0, -0.0, 5e-324, 1.7976931348623157e308, -2.5, 3.0])
        with Spool(self.policy()) as spool:
            spooled = spool.source(ListSource([values]))
            (partition,) = spooled.partitions()
            list(spooled.batches(partition, max_rows=4))
            replayed = list(spooled.batches(partition, max_rows=4))
        self.assertEqual([len(batch) for batch in replayed], [4, 2])
        joined = np.concatenate(replayed)
        self.assertEqual(joined.tobytes(), values.tobytes())
        self.assertEqual(joined.dtype, np.float64)

    def test_the_spool_holds_a_directory_of_its_own_inside_the_given_one(self) -> None:
        with Spool(self.policy()) :
            (private,) = entries(self.directory)
            self.assertTrue(private.startswith("veridist-spool-"))
            self.assertTrue((self.directory / private).is_dir())
            self.assertEqual(spool_files(self.directory), [])
        self.assert_clean()

    def test_a_partition_identifier_never_names_a_path(self) -> None:
        class Odd:
            def partitions(self) -> tuple[Partition, ...]:
                return (
                    Partition("../escape", 0, 3, None, FingerprintLevel.NONE),
                    Partition("a/b\\c:d", 1, 2, None, FingerprintLevel.NONE),
                )

            def batches(
                self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
            ) -> Iterator[Any]:
                yield np.arange(1.0, 3.0 + partition.ordinal * -1.0)

        with Spool(self.policy()) as spool:
            spooled = spool.source(Odd())
            for partition in spooled.partitions():
                list(spooled.batches(partition))
            self.assertEqual(
                [path.name for path in spool_files(self.directory)],
                ["00000000.arrow", "00000001.arrow"],
            )
            self.assertEqual(entries(self.root), ["spools"])

    def test_the_digest_and_rows_of_every_partition_are_recorded_in_read_order(self) -> None:
        parts = self.parts()
        with Spool(self.policy()) as spool:
            spooled = spool.source(ListSource(parts))
            for partition in spooled.partitions():
                list(spooled.batches(partition, max_rows=300))
            record = spool.provenance()
        self.assertEqual(record.row_count, 2500 - 0)
        self.assertEqual([item.ordinal for item in record.partitions], [0, 1, 2, 3])
        self.assertEqual([item.rows for item in record.partitions], [1000, 0, 1250, 250])
        self.assertEqual(
            [item.digest_sha256 for item in record.partitions], [digest_of(p) for p in parts]
        )
        self.assertEqual(record.partitions[1].digest_sha256, hashlib.sha256(b"").hexdigest())

    def test_the_digest_does_not_depend_on_the_batching(self) -> None:
        values = weibull(1000)
        digests = set()
        for rows in (1, 7, 333, 1000, 5000):
            with Spool(self.policy()) as spool:
                spooled = spool.source(ListSource([values]))
                (partition,) = spooled.partitions()
                list(spooled.batches(partition, max_rows=rows))
                digests.add(spool.provenance().partitions[0].digest_sha256)
        self.assertEqual(digests, {digest_of(values)})

    def test_a_widened_source_is_spooled_as_float64(self) -> None:
        with Spool(self.policy()) as spool:
            spooled = spool.source(NumpySource(np.array([1, 2, 3], dtype=np.int32)))
            (partition,) = spooled.partitions()
            list(spooled.batches(partition))
            record = spool.provenance()
            replayed = list(spooled.batches(partition))
        self.assertEqual(record.partitions[0].digest_sha256, digest_of([1.0, 2.0, 3.0]))
        self.assertEqual(replayed[0].dtype, np.float64)

    def test_an_empty_batch_is_passed_on_and_not_written(self) -> None:
        values = weibull(10)
        gap = ListSource([values])

        class Gappy:
            def partitions(self) -> tuple[Partition, ...]:
                return gap.partitions()

            def batches(
                self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
            ) -> Iterator[Any]:
                yield np.empty(0)
                yield values[:4]
                yield np.empty(0)
                yield values[4:]
                yield np.empty(0)

        with Spool(self.policy()) as spool:
            spooled = spool.source(Gappy())
            (partition,) = spooled.partitions()
            first = list(spooled.batches(partition))
            self.assertEqual([len(batch) for batch in first], [0, 4, 0, 6, 0])
            replayed = list(spooled.batches(partition))
            self.assertEqual([len(batch) for batch in replayed], [4, 6])
            self.assertEqual(spool._charged, FILE_FRAMING_BYTES + batch_cost(4) + batch_cost(6))
            self.assertEqual(spool.provenance().partitions[0].digest_sha256, digest_of(values))

    def test_a_partition_that_is_not_one_is_refused_and_a_foreign_one_by_the_source(self) -> None:
        source = ListSource([weibull(10)])
        with Spool(self.policy()) as spool:
            spooled = spool.source(source)
            with self.assertRaises(ValueError):
                spooled.batches("p0")  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                spool.batches(source, "p0", 10)  # type: ignore[arg-type]
            foreign = Partition("array", 0, 3, None, FingerprintLevel.NONE)
            with self.assertRaises(ValueError):
                spool.source(NumpySource(weibull(3))).batches(
                    Partition("other", 0, 3, None, FingerprintLevel.NONE)
                )
            self.assertEqual(spool._begun, set())
            self.assertEqual(
                [p.id for p in spool.source(NumpySource(weibull(3))).partitions()], [foreign.id]
            )

    def test_the_batch_bound_is_validated(self) -> None:
        with Spool(self.policy()) as spool:
            spooled = spool.source(ListSource([weibull(10)]))
            (partition,) = spooled.partitions()
            with self.assertRaises(ValueError):
                spooled.batches(partition, max_rows=0)
            with self.assertRaises(TypeError):
                spooled.batches(partition, max_rows=2.0)  # type: ignore[arg-type]
            self.assertEqual(spool._begun, set())

    def test_a_partition_that_was_begun_and_not_finished_cannot_be_read_again(self) -> None:
        source = ListSource([weibull(10)])
        with Spool(self.policy()) as spool:
            spooled = spool.source(source)
            (partition,) = spooled.partitions()
            unfinished = spooled.batches(partition)
            self.assertEqual(code_of(lambda: spooled.batches(partition)), CODES.SPOOL_CHANGED)
            unfinished.close()
            self.assertEqual(code_of(lambda: spooled.batches(partition)), CODES.SPOOL_CHANGED)
        self.assertEqual(source.batch_calls, [])

    def test_a_spool_needs_a_policy(self) -> None:
        for bad in (None, {"max_bytes": 10}, "spools"):
            with self.subTest(policy=bad), self.assertRaises(TypeError):
                Spool(bad)  # type: ignore[arg-type]

    def test_a_spool_without_pyarrow_is_refused_and_creates_nothing(self) -> None:
        with mock.patch.dict(sys.modules, BLOCK_PYARROW):
            with self.assertRaises(ScaleSourceError) as caught:
                Spool(self.policy())
        self.assertIs(caught.exception.code, CODES.OPTIONAL_DEPENDENCY_MISSING)
        self.assertEqual(caught.exception.hint, ARROW_HINT)
        self.assert_clean()

    def test_a_missing_or_unusable_directory_is_unavailable_and_names_no_path(self) -> None:
        missing = self.root / "absent"
        with self.assertRaises(ScaleSourceError) as caught:
            Spool(ScaleSpoolPolicy(directory=missing, max_bytes=10))
        self.assertIs(caught.exception.code, CODES.SPOOL_UNAVAILABLE)
        self.assertEqual(caught.exception.hint, _spool.HINT_UNAVAILABLE)
        self.assertEqual(str(caught.exception), f"SPOOL_UNAVAILABLE: {_spool.HINT_UNAVAILABLE}")
        self.assertNotIn(str(missing), str(caught.exception))
        self.assertIsNone(caught.exception.__cause__)
        self.assertTrue(caught.exception.__suppress_context__)
        plain = self.root / "plain"
        plain.write_text("x", encoding="utf-8")
        self.assertEqual(
            code_of(lambda: Spool(ScaleSpoolPolicy(directory=plain, max_bytes=10))),
            CODES.SPOOL_UNAVAILABLE,
        )

    def test_the_system_directory_is_used_when_the_policy_allows_it(self) -> None:
        with mock.patch.object(tempfile, "gettempdir", return_value=str(self.directory)):
            with Spool(ScaleSpoolPolicy(max_bytes=10, allow_system_temp=True)):
                (private,) = entries(self.directory)
                self.assertTrue(private.startswith("veridist-spool-"))
        self.assert_clean()

    def test_a_file_is_created_exclusively(self) -> None:
        with Spool(self.policy()) as spool:
            (private,) = entries(self.directory)
            path = self.directory / private / "00000000.arrow"
            path.write_bytes(b"")
            spooled = spool.source(ListSource([weibull(10)]))
            (partition,) = spooled.partitions()
            self.assertEqual(
                code_of(lambda: list(spooled.batches(partition))), CODES.SPOOL_UNAVAILABLE
            )

    def test_the_batches_are_read_only_views_of_float64(self) -> None:
        with Spool(self.policy()) as spool:
            spooled = spool.source(ListSource([weibull(100)]))
            (partition,) = spooled.partitions()
            list(spooled.batches(partition))
            (replayed,) = spooled.batches(partition)
            self.assertEqual(replayed.dtype, np.float64)
            self.assertFalse(replayed.flags.writeable)
            self.assertTrue(replayed.flags.c_contiguous)

    def test_the_replay_can_be_memory_mapped_or_read(self) -> None:
        values = weibull(20000)
        allocated: dict[bool, int] = {}
        digests = set()
        for mapped in (True, False):
            if mapped and sys.platform == "win32":
                continue
            with (
                mock.patch.object(_spool, "MEMORY_MAP", mapped),
                Spool(self.policy()) as spool,
            ):
                spooled = spool.source(ListSource([values]))
                (partition,) = spooled.partitions()
                list(spooled.batches(partition))
                before = pa.total_allocated_bytes()
                replay = spooled.batches(partition)
                held = next(replay)
                allocated[mapped] = pa.total_allocated_bytes() - before
                self.assertEqual(held.shape, (20000,))
                digests.add(hashlib.sha256(held.tobytes()).hexdigest())
                del held
                replay.close()
        self.assertEqual(digests, {digest_of(values)})
        self.assertGreaterEqual(allocated[False], 20000 * 8)
        if True in allocated:
            self.assertEqual(allocated[True], 0)

    def test_the_default_follows_the_platform(self) -> None:
        self.assertEqual(_spool.MEMORY_MAP, sys.platform != "win32")


class LimitTests(TempCase):
    def single(self, count: int = 1000) -> ListSource:
        return ListSource([weibull(count)])

    def test_the_budget_is_exact_for_one_batch(self) -> None:
        total = FILE_FRAMING_BYTES + batch_cost(1000)
        with Spool(self.policy(total)) as spool:
            spooled = spool.source(self.single())
            (partition,) = spooled.partitions()
            list(spooled.batches(partition))
            self.assertEqual(spool._charged, total)
        self.assert_clean()
        with Spool(self.policy(total - 1)) as spool:
            spooled = spool.source(self.single())
            (partition,) = spooled.partitions()
            with self.assertRaises(ScaleSourceError) as caught:
                list(spooled.batches(partition))
            self.assertIs(caught.exception.code, CODES.SPOOL_LIMIT_EXCEEDED)
            self.assertEqual(caught.exception.hint, _spool.HINT_LIMIT)
            self.assertEqual(spool._charged, FILE_FRAMING_BYTES)
        self.assert_clean()

    def test_the_budget_is_exact_across_batches_and_files(self) -> None:
        parts = [weibull(900), weibull(500, seed=9)]
        total = 2 * FILE_FRAMING_BYTES + batch_cost(400) * 2 + batch_cost(100) + batch_cost(400)
        total += batch_cost(100)
        with Spool(self.policy(total)) as spool:
            spooled = spool.source(ListSource(parts))
            for partition in spooled.partitions():
                list(spooled.batches(partition, max_rows=400))
            self.assertEqual(spool._charged, total)
        for short in (total - 1, total - batch_cost(100), total - FILE_FRAMING_BYTES):
            with Spool(self.policy(short)) as spool:
                spooled = spool.source(ListSource(parts))
                with self.assertRaises(ScaleSourceError) as caught:
                    for partition in spooled.partitions():
                        list(spooled.batches(partition, max_rows=400))
                self.assertIs(caught.exception.code, CODES.SPOOL_LIMIT_EXCEEDED)
            self.assert_clean()

    def test_a_file_that_does_not_fit_is_refused_before_the_source_is_read(self) -> None:
        source = self.single()
        with Spool(self.policy(FILE_FRAMING_BYTES - 1)) as spool:
            spooled = spool.source(source)
            (partition,) = spooled.partitions()
            self.assertEqual(
                code_of(lambda: list(spooled.batches(partition))), CODES.SPOOL_LIMIT_EXCEEDED
            )
            self.assertEqual(spool._charged, 0)
            self.assertEqual(spool_files(self.directory), [])
        self.assertEqual(source.batch_sizes, [])
        self.assert_clean()

    def test_a_refused_batch_is_not_written_and_the_budget_is_not_charged(self) -> None:
        source = ListSource([weibull(1000)])
        with Spool(self.policy(FILE_FRAMING_BYTES + batch_cost(600) + batch_cost(399))) as spool:
            spooled = spool.source(source)
            (partition,) = spooled.partitions()
            read: list[Any] = []
            with self.assertRaises(ScaleSourceError) as caught:
                for batch in spooled.batches(partition, max_rows=600):
                    read.append(batch)
            self.assertIs(caught.exception.code, CODES.SPOOL_LIMIT_EXCEEDED)
            self.assertEqual(len(read), 1)
            self.assertEqual(spool._charged, FILE_FRAMING_BYTES + batch_cost(600))
            (path,) = spool_files(self.directory)
            self.assertLess(path.stat().st_size, FILE_FRAMING_BYTES + batch_cost(600))
        self.assert_clean()

    def test_real_files_never_exceed_the_budget(self) -> None:
        data = weibull(3000)
        for rows in (1, 13, 700, 3000):
            charge = FILE_FRAMING_BYTES + sum(
                batch_cost(min(rows, 3000 - start)) for start in range(0, 3000, rows)
            )
            with Spool(self.policy(charge)) as spool:
                spooled = spool.source(ListSource([data]))
                (partition,) = spooled.partitions()
                list(spooled.batches(partition, max_rows=rows))
                size = sum(path.stat().st_size for path in spool_files(self.directory))
            self.assertLessEqual(size, charge, rows)
            self.assert_clean()

    def test_the_fit_refuses_cleans_up_and_names_no_path(self) -> None:
        values = weibull(1000)
        total = FILE_FRAMING_BYTES + batch_cost(1000)
        with self.assertRaises(ScaleSourceError) as caught:
            fit(NumpySource(values), "weibull_min", spool=self.policy(total - 1))
        self.assertIs(caught.exception.code, CODES.SPOOL_LIMIT_EXCEEDED)
        self.assert_no_leak(caught.exception)
        self.assert_clean()
        self.assertEqual(
            fit(NumpySource(values), "weibull_min", spool=self.policy(total)).scale.row_count,
            1000,
        )
        self.assert_clean()


class WriteFailureTests(TempCase):
    class Broken:
        """A writer whose calls fail."""

        def __init__(self, error: BaseException, *, at: str = "write_batch") -> None:
            self.error = error
            self.at = at

        def write_batch(self, batch: Any) -> None:
            if self.at == "write_batch":
                raise self.error

        def close(self) -> None:
            if self.at == "close":
                raise self.error

    def run_first_pass(self, failure: Any) -> ScaleSourceError:
        with Spool(self.policy()) as spool:
            spooled = spool.source(ListSource([weibull(100)]))
            (partition,) = spooled.partitions()
            with self.assertRaises(ScaleSourceError) as caught, failure:
                list(spooled.batches(partition))
        self.assert_clean()
        return caught.exception

    def test_a_failure_to_create_the_file_is_unavailable(self) -> None:
        error = self.run_first_pass(
            mock.patch.object(_spool, "_create_file", side_effect=OSError("denied"))
        )
        self.assertIs(error.code, CODES.SPOOL_UNAVAILABLE)
        self.assertEqual(error.hint, _spool.HINT_UNAVAILABLE)
        self.assertIsNone(error.__cause__)

    def test_a_failure_to_start_the_writer_is_unavailable(self) -> None:
        error = self.run_first_pass(
            mock.patch.object(pa.ipc, "new_file", side_effect=pa.ArrowInvalid("bad"))
        )
        self.assertIs(error.code, CODES.SPOOL_UNAVAILABLE)

    def test_a_failure_to_write_a_batch_is_unavailable(self) -> None:
        for raised in (OSError("disk full"), pa.ArrowIOError("disk full")):
            with self.subTest(error=type(raised).__name__):
                error = self.run_first_pass(
                    mock.patch.object(pa.ipc, "new_file", return_value=self.Broken(raised))
                )
                self.assertIs(error.code, CODES.SPOOL_UNAVAILABLE)
                self.assertIsNone(error.__cause__)

    def test_a_failure_to_finish_the_file_is_unavailable(self) -> None:
        for raised in (OSError("disk full"), pa.ArrowInvalid("bad")):
            with self.subTest(error=type(raised).__name__):
                error = self.run_first_pass(
                    mock.patch.object(
                        pa.ipc, "new_file", return_value=self.Broken(raised, at="close")
                    )
                )
                self.assertIs(error.code, CODES.SPOOL_UNAVAILABLE)

    def test_an_error_of_the_source_is_not_turned_into_a_spool_failure(self) -> None:
        with self.assertRaises(ScaleSourceError) as caught:
            fit(
                NumpySource(np.array([1.0, 2.0, float("nan")])),
                "weibull_min",
                spool=self.policy(),
            )
        self.assertIs(caught.exception.code, CODES.NAN_VALUE)
        self.assert_clean()


class TamperTests(TempCase):
    """A spool file changed after the first pass is never read back as if it were intact."""

    def written(self, values: np.ndarray, rows: int = 4) -> tuple[Spool, SpooledSource, Partition]:
        spool = Spool(self.policy())
        self.addCleanup(spool.__exit__, None, None, None)
        spooled = spool.source(ListSource([values]))
        (partition,) = spooled.partitions()
        list(spooled.batches(partition, max_rows=rows))
        return spool, spooled, partition

    def only_file(self) -> Path:
        (path,) = spool_files(self.directory)
        return path

    def assert_changed(
        self, spooled: SpooledSource, partition: Partition, rows: int = 4
    ) -> list[Any]:
        got: list[Any] = []
        with self.assertRaises(ScaleSourceError) as caught:
            for batch in spooled.batches(partition, max_rows=rows):
                got.append(batch)
        self.assertIs(caught.exception.code, CODES.SPOOL_CHANGED)
        self.assertEqual(caught.exception.hint, _spool.HINT_CHANGED)
        self.assert_no_leak(caught.exception)
        return got

    def rewrite(self, path: Path, batches: list[Any], schema: Any = None) -> None:
        schema = schema or pa.schema([pa.field("x", pa.float64(), nullable=False)])
        with pa.OSFile(str(path), "wb") as sink, pa.ipc.new_file(sink, schema) as writer:
            for batch in batches:
                writer.write_batch(pa.record_batch([batch], schema=schema))

    def test_an_intact_spool_is_read_back_any_number_of_times(self) -> None:
        values = weibull(10)
        _, spooled, partition = self.written(values)
        for _ in range(3):
            joined = np.concatenate(list(spooled.batches(partition, max_rows=4)))
            self.assertEqual(joined.tobytes(), values.tobytes())

    def test_one_changed_value_is_detected_and_the_last_batch_is_not_handed_over(self) -> None:
        values = weibull(10)
        _, spooled, partition = self.written(values)
        changed = values.copy()
        changed[5] = np.nextafter(changed[5], 0.0)
        self.rewrite(
            self.only_file(),
            [pa.array(changed[0:4]), pa.array(changed[4:8]), pa.array(changed[8:10])],
        )
        got = self.assert_changed(spooled, partition)
        self.assertEqual([len(batch) for batch in got], [4, 4])

    def test_a_single_batch_partition_hands_over_nothing_when_changed(self) -> None:
        values = weibull(3)
        _, spooled, partition = self.written(values, rows=10)
        changed = values.copy()
        changed[0] = changed[0] + 1.0
        self.rewrite(self.only_file(), [pa.array(changed)])
        self.assertEqual(self.assert_changed(spooled, partition, rows=10), [])

    def test_a_flipped_byte_in_the_values_is_detected(self) -> None:
        values = weibull(10)
        _, spooled, partition = self.written(values)
        path = self.only_file()
        data = bytearray(path.read_bytes())
        position = bytes(data).index(values[5].tobytes())
        data[position] ^= 0x01
        path.write_bytes(bytes(data))
        self.assert_changed(spooled, partition)

    def test_a_flipped_byte_anywhere_is_never_read_as_intact(self) -> None:
        values = weibull(10)
        _, spooled, partition = self.written(values)
        path = self.only_file()
        original = path.read_bytes()
        for position in range(0, len(original), 17):
            data = bytearray(original)
            data[position] ^= 0xFF
            path.write_bytes(bytes(data))
            try:
                joined = np.concatenate(list(spooled.batches(partition, max_rows=4)))
            except ScaleSourceError as error:
                self.assertIs(error.code, CODES.SPOOL_CHANGED, position)
            else:
                self.assertEqual(joined.tobytes(), values.tobytes(), position)

    def test_a_truncated_empty_or_missing_file_is_detected(self) -> None:
        values = weibull(10)
        spool, spooled, partition = self.written(values)
        path = self.only_file()
        original = path.read_bytes()
        for kept in (len(original) - 10, len(original) // 2, 8, 0):
            path.write_bytes(original[:kept])
            with self.subTest(kept=kept):
                self.assert_changed(spooled, partition)
        path.write_bytes(b"this is not an arrow file at all, not even close to one")
        self.assert_changed(spooled, partition)
        self.assertEqual(spool._handles, [])
        path.unlink()
        self.assert_changed(spooled, partition)
        self.assertEqual(spool._handles, [])

    def test_another_number_of_batches_is_detected(self) -> None:
        values = weibull(10)
        _, spooled, partition = self.written(values)
        self.rewrite(self.only_file(), [pa.array(values[0:4]), pa.array(values[4:10])])
        self.assert_changed(spooled, partition)
        self.rewrite(
            self.only_file(),
            [
                pa.array(values[0:4]),
                pa.array(values[4:8]),
                pa.array(values[8:10]),
                pa.array([], pa.float64()),
            ],
        )
        self.assert_changed(spooled, partition)

    def test_a_batch_larger_than_any_that_was_written_is_detected(self) -> None:
        values = weibull(10)
        _, spooled, partition = self.written(values)
        self.rewrite(
            self.only_file(),
            [pa.array(values), pa.array([], pa.float64()), pa.array([], pa.float64())],
        )
        self.assert_changed(spooled, partition)

    def test_a_batch_as_large_as_the_largest_written_is_accepted(self) -> None:
        values = weibull(10)
        _, spooled, partition = self.written(values)
        self.rewrite(
            self.only_file(),
            [pa.array(values[0:2]), pa.array(values[2:6]), pa.array(values[6:10])],
        )
        joined = np.concatenate(list(spooled.batches(partition, max_rows=4)))
        self.assertEqual(joined.tobytes(), values.tobytes())

    def test_the_pieces_handed_over_never_exceed_the_bound_asked_for(self) -> None:
        values = weibull(10)
        _, spooled, partition = self.written(values, rows=4)
        expected = {
            1: [1] * 10,
            3: [3, 1, 3, 1, 2],
            4: [4, 4, 2],
            100: [4, 4, 2],
        }
        for rows, sizes in expected.items():
            pieces = list(spooled.batches(partition, max_rows=rows))
            with self.subTest(rows=rows):
                self.assertEqual([len(piece) for piece in pieces], sizes)
                self.assertEqual(np.concatenate(pieces).tobytes(), values.tobytes())

    def test_a_change_is_detected_before_the_last_piece_whatever_the_bound(self) -> None:
        values = weibull(10)
        _, spooled, partition = self.written(values, rows=4)
        changed = values.copy()
        changed[9] = changed[9] + 1.0
        self.rewrite(
            self.only_file(),
            [pa.array(changed[0:4]), pa.array(changed[4:8]), pa.array(changed[8:10])],
        )
        self.assertEqual(
            [len(piece) for piece in self.assert_changed(spooled, partition, rows=3)], [3, 1, 3, 1]
        )

    def test_another_type_or_nullability_is_detected(self) -> None:
        values = weibull(10)
        _, spooled, partition = self.written(values)
        path = self.only_file()
        wide = pa.schema([pa.field("x", pa.float32(), nullable=False)])
        self.rewrite(
            path,
            [
                pa.array(values[0:4], pa.float32()),
                pa.array(values[4:8], pa.float32()),
                pa.array(values[8:10], pa.float32()),
            ],
            wide,
        )
        self.assert_changed(spooled, partition)
        nullable = pa.schema([pa.field("x", pa.float64(), nullable=True)])
        self.rewrite(
            path,
            [pa.array(values[0:4]), pa.array(values[4:8]), pa.array(values[8:10])],
            nullable,
        )
        self.assert_changed(spooled, partition)
        renamed = pa.schema([pa.field("y", pa.float64(), nullable=False)])
        self.rewrite(
            path,
            [pa.array(values[0:4]), pa.array(values[4:8]), pa.array(values[8:10])],
            renamed,
        )
        self.assert_changed(spooled, partition)

    def test_a_row_count_that_differs_from_the_record_is_detected(self) -> None:
        values = weibull(10)
        spool, spooled, partition = self.written(values)
        entry = spool._sealed[partition.id]
        spool._sealed[partition.id] = replace(entry, rows=entry.rows + 1)
        self.assert_changed(spooled, partition)
        spool._sealed[partition.id] = replace(entry, rows=entry.rows - 1)
        self.assert_changed(spooled, partition)

    def test_a_digest_that_differs_from_the_record_is_detected(self) -> None:
        values = weibull(10)
        spool, spooled, partition = self.written(values)
        entry = spool._sealed[partition.id]
        spool._sealed[partition.id] = replace(entry, digest="0" * 64)
        self.assert_changed(spooled, partition)

    def test_an_empty_partition_is_verified_too(self) -> None:
        spool, spooled, partition = self.written(np.empty(0))
        self.assertEqual(list(spooled.batches(partition)), [])
        entry = spool._sealed[partition.id]
        spool._sealed[partition.id] = replace(entry, digest=digest_of([1.0]))
        self.assert_changed(spooled, partition)
        spool._sealed[partition.id] = replace(entry, rows=1)
        self.assert_changed(spooled, partition)

    def test_a_tampered_spool_fails_the_fit_before_any_estimate_and_is_cleaned(self) -> None:
        values = weibull(3000)
        original = _fit.PassLedger.begin
        calls = {"begin": 0}

        def tamper(ledger: Any, *args: Any, **kwargs: Any) -> Any:
            calls["begin"] += 1
            if calls["begin"] == 3:
                path = self.only_file()
                data = bytearray(path.read_bytes())
                data[len(data) // 2] ^= 0x10
                path.write_bytes(bytes(data))
            return original(ledger, *args, **kwargs)

        with (
            mock.patch.object(_fit.PassLedger, "begin", tamper),
            self.assertRaises(ScaleSourceError) as caught,
        ):
            fit(NumpySource(values), "weibull_min", spool=self.policy())
        self.assertIs(caught.exception.code, CODES.SPOOL_CHANGED)
        self.assertEqual(calls["begin"], 3)
        self.assert_clean()

    def test_a_tampered_spool_fails_both_families(self) -> None:
        for family, values in (("weibull_min", weibull()), ("gumbel_right", gumbel())):
            original = _fit.PassLedger.begin
            calls = {"begin": 0}

            def tamper(ledger: Any, *args: Any, **kwargs: Any) -> Any:
                calls["begin"] += 1
                if calls["begin"] == 2:
                    self.only_file().write_bytes(b"")
                return original(ledger, *args, **kwargs)

            with self.subTest(family=family):
                with (
                    mock.patch.object(_fit.PassLedger, "begin", tamper),
                    self.assertRaises(ScaleSourceError) as caught,
                ):
                    fit(NumpySource(values), family, spool=self.policy())
                self.assertIs(caught.exception.code, CODES.SPOOL_CHANGED)
                self.assert_clean()


@unittest.skipUnless(os.name == "posix", "permission bits are POSIX")
class PermissionTests(TempCase):
    def test_the_directory_and_the_files_are_private_to_the_owner(self) -> None:
        with Spool(self.policy()) as spool:
            spooled = spool.source(ListSource([weibull(10), weibull(5)]))
            for partition in spooled.partitions():
                list(spooled.batches(partition))
            (private,) = entries(self.directory)
            mode = stat.S_IMODE((self.directory / private).stat().st_mode)
            self.assertEqual(mode, 0o700)
            files = spool_files(self.directory)
            self.assertEqual(len(files), 2)
            for path in files:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_a_loose_umask_does_not_loosen_the_files(self) -> None:
        previous = os.umask(0)
        try:
            with Spool(self.policy()) as spool:
                spooled = spool.source(ListSource([weibull(10)]))
                (partition,) = spooled.partitions()
                list(spooled.batches(partition))
                (path,) = spool_files(self.directory)
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        finally:
            os.umask(previous)

class CleanupTests(TempCase):
    def test_the_directory_is_removed_when_the_context_exits(self) -> None:
        with Spool(self.policy()) as spool:
            spooled = spool.source(ListSource([weibull(100)]))
            (partition,) = spooled.partitions()
            list(spooled.batches(partition))
            self.assertEqual(len(entries(self.directory)), 1)
        self.assert_clean()

    def test_the_directory_is_removed_when_the_block_raises_or_is_interrupted(self) -> None:
        for error in (RuntimeError("boom"), KeyboardInterrupt(), SystemExit(3)):
            with self.subTest(error=type(error).__name__):
                with self.assertRaises(type(error)):
                    with Spool(self.policy()) as spool:
                        spooled = spool.source(ListSource([weibull(100)]))
                        (partition,) = spooled.partitions()
                        list(spooled.batches(partition))
                        raise error
                self.assert_clean()

    def test_the_directory_is_removed_when_the_first_pass_is_interrupted_midway(self) -> None:
        class Interrupted:
            def partitions(self) -> tuple[Partition, ...]:
                return (Partition("p", 0, None, None, FingerprintLevel.NONE),)

            def batches(
                self, partition: Partition, *, max_rows: int = DEFAULT_MAX_ROWS
            ) -> Iterator[Any]:
                yield weibull(10)
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            fit(Interrupted(), "weibull_min", spool=self.policy())
        self.assert_clean()

    def test_the_directory_is_removed_after_a_failure_in_a_later_pass(self) -> None:
        for error in (RuntimeError("boom"), KeyboardInterrupt()):
            for family in ("weibull_min", "gumbel_right"):
                with self.subTest(error=type(error).__name__, family=family):
                    with (
                        mock.patch.object(_multi_pass, "_profile_pass", side_effect=error),
                        self.assertRaises(type(error)),
                    ):
                        fit(NumpySource(weibull()), family, spool=self.policy())
                    self.assert_clean()

    def test_the_directory_is_removed_when_a_pass_is_refused_by_the_budget(self) -> None:
        with self.assertRaises(PassBudgetError) as caught:
            fit(NumpySource(weibull()), "weibull_min", pass_budget=3, spool=self.policy())
        self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_EXCEEDED)
        self.assert_clean()

    def test_the_directory_is_removed_after_a_result_that_is_a_failure(self) -> None:
        result = fit(NumpySource(np.array([-1.0, 2.0])), "weibull_min", spool=self.policy())
        self.assertIsInstance(result.fit, WeibullFitFailure)
        self.assert_clean()

    def test_a_pass_budget_below_the_plan_is_refused_before_a_directory_is_made(self) -> None:
        with self.assertRaises(PassBudgetError) as caught:
            fit(NumpySource(weibull()), "weibull_min", pass_budget=2, spool=self.policy())
        self.assertIs(caught.exception.code, FailureCode.PASS_BUDGET_INSUFFICIENT)
        self.assert_clean()
        with self.assertRaises(PassBudgetError):
            fit(stream_source(weibull()), "weibull_min", pass_budget=2)

    def test_deleting_twice_does_nothing(self) -> None:
        spool = Spool(self.policy())
        spool.__exit__(None, None, None)
        spool.__exit__(None, None, None)
        self.assert_clean()

    def test_a_directory_that_is_already_gone_is_not_a_failure(self) -> None:
        with Spool(self.policy()):
            (private,) = entries(self.directory)
            (self.directory / private).rmdir()
        self.assert_clean()

    def test_a_directory_that_cannot_be_deleted_is_reported_when_nothing_else_is(self) -> None:
        with mock.patch.object(_spool.shutil, "rmtree", side_effect=PermissionError("busy")):
            with self.assertRaises(ScaleSourceError) as caught:
                fit(NumpySource(weibull()), "weibull_min", spool=self.policy())
        self.assertIs(caught.exception.code, CODES.SPOOL_UNAVAILABLE)
        self.assertEqual(caught.exception.hint, _spool.HINT_CLEANUP)
        self.assertNotIn(str(self.directory), str(caught.exception))
        self.assertEqual(len(entries(self.directory)), 1)

    def test_a_directory_that_cannot_be_deleted_never_masks_the_error_in_flight(self) -> None:
        for error in (RuntimeError("boom"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                with (
                    mock.patch.object(_spool.shutil, "rmtree", side_effect=PermissionError("busy")),
                    mock.patch.object(_multi_pass, "_profile_pass", side_effect=error),
                    self.assertRaises(type(error)),
                ):
                    fit(NumpySource(weibull()), "weibull_min", spool=self.policy())

    def test_a_handle_that_cannot_be_closed_is_reported_and_the_rest_is_still_deleted(self) -> None:
        class Stuck:
            def close(self) -> None:
                raise OSError("busy")

        spool = Spool(self.policy())
        spool._handles.append(Stuck())
        with self.assertRaises(ScaleSourceError) as caught:
            spool.__exit__(None, None, None)
        self.assertIs(caught.exception.code, CODES.SPOOL_UNAVAILABLE)
        self.assertEqual(spool._handles, [])
        self.assert_clean()

    def test_every_handle_is_closed_before_the_directory_is_removed(self) -> None:
        closed: list[str] = []

        class Recording:
            def __init__(self, name: str) -> None:
                self.name = name

            def close(self) -> None:
                closed.append(self.name)

        with mock.patch.object(
            _spool.shutil, "rmtree", side_effect=lambda path: closed.append("rmtree")
        ):
            spool = Spool(self.policy())
            spool._handles.extend([Recording("a"), Recording("b")])
            spool.__exit__(None, None, None)
        self.assertEqual(closed, ["a", "b", "rmtree"])
        for private in entries(self.directory):
            (self.directory / private).rmdir()

    def test_an_unfinished_reader_closed_after_the_spool_is_gone_is_harmless(self) -> None:
        with Spool(self.policy()) as spool:
            spooled = spool.source(ListSource([weibull(100)]))
            (partition,) = spooled.partitions()
            list(spooled.batches(partition, max_rows=30))
            replay = spooled.batches(partition, max_rows=30)
            next(replay)
        self.assert_clean()
        replay.close()

    def test_a_reader_leaves_no_handle_open(self) -> None:
        with Spool(self.policy()) as spool:
            spooled = spool.source(ListSource([weibull(100)]))
            (partition,) = spooled.partitions()
            list(spooled.batches(partition))
            self.assertEqual(spool._handles, [])
            list(spooled.batches(partition))
            self.assertEqual(spool._handles, [])
            replay = spooled.batches(partition, max_rows=10)
            next(replay)
            self.assertEqual(len(spool._handles), 1)
            replay.close()
            self.assertEqual(spool._handles, [])


class RefusalTests(TempCase):
    def test_a_single_use_source_is_refused_before_anything_is_read(self) -> None:
        for family in ("weibull_min", "gumbel_right"):
            pulled: list[int] = []
            source = Counting(stream_source(weibull(), pulled=pulled))
            with self.subTest(family=family), self.assertRaises(ScaleSourceError) as caught:
                fit(source, family)
            error = caught.exception
            self.assertIs(error.code, CODES.SPOOL_REQUIRED)
            self.assertEqual(error.hint, _spool.HINT_REQUIRED)
            self.assertEqual(str(error), f"SPOOL_REQUIRED: {_spool.HINT_REQUIRED}")
            self.assertEqual(pulled, [])
            self.assertEqual(source.batch_calls, 0)

    def test_the_refusal_leaves_the_source_unread_so_a_spool_can_still_be_requested(self) -> None:
        values = weibull()
        source = stream_source(values)
        with self.assertRaises(ScaleSourceError):
            fit(source, "weibull_min")
        result = fit(source, "weibull_min", spool=self.policy())
        self.assertEqual(result.scale.row_count, len(values))
        self.assert_clean()

    def test_a_stream_object_is_refused_the_same_way(self) -> None:
        exporter = StreamExporter(pa.table({"x": weibull()}))
        source = ArrowSource(exporter, "x")
        self.assertEqual(code_of(lambda: fit(source, "weibull_min")), CODES.SPOOL_REQUIRED)
        self.assertIsNotNone(fit(source, "weibull_min", spool=self.policy()).fit)

    def test_a_source_that_can_be_read_again_needs_no_spool(self) -> None:
        values = weibull()
        for source in (
            ArrowSource(pa.table({"x": values}), "x"),
            ArrowSource(pa.record_batch({"x": values}), "x"),
            NumpySource(values),
            ListSource([values]),
        ):
            with self.subTest(source=type(source).__name__):
                self.assertIsNotNone(fit(source, "weibull_min").fit)

    def test_a_pass_budget_below_the_plan_is_refused_before_the_spool_requirement(self) -> None:
        source = Counting(stream_source(weibull()))
        with self.assertRaises(PassBudgetError):
            fit(source, "gumbel_right", pass_budget=1)
        self.assertEqual(source.batch_calls, 0)

    def test_a_consumed_source_with_a_spool_still_reports_that_it_was_consumed(self) -> None:
        source = stream_source(weibull())
        (partition,) = source.partitions()
        list(source.batches(partition))
        self.assertEqual(
            code_of(lambda: fit(source, "weibull_min", spool=self.policy())), CODES.SOURCE_CONSUMED
        )
        self.assert_clean()

    def test_the_spool_argument_is_validated_before_any_read(self) -> None:
        for family in ("weibull_min", "exponential"):
            source = Counting(NumpySource(weibull()))
            for bad in ({"max_bytes": 5}, True, "spools", 5, ScaleSpoolPolicy):
                with self.subTest(family=family, spool=bad), self.assertRaises(TypeError):
                    fit(source, family, spool=bad)  # type: ignore[arg-type]
            self.assertEqual(source.batch_calls, 0)
            self.assertEqual(source.partition_calls, 0)

    def test_a_spool_without_pyarrow_is_refused_before_any_read(self) -> None:
        source = Counting(NumpySource(weibull()))
        with mock.patch.dict(sys.modules, BLOCK_PYARROW):
            with self.assertRaises(ScaleSourceError) as caught:
                fit(source, "weibull_min", spool=self.policy())
        self.assertIs(caught.exception.code, CODES.OPTIONAL_DEPENDENCY_MISSING)
        self.assertEqual(caught.exception.hint, ARROW_HINT)
        self.assertEqual(source.batch_calls, 0)
        self.assert_clean()

    def test_an_unusable_directory_is_refused_before_any_read(self) -> None:
        source = Counting(NumpySource(weibull()))
        policy = ScaleSpoolPolicy(directory=self.root / "absent", max_bytes=1 << 20)
        self.assertEqual(
            code_of(lambda: fit(source, "weibull_min", spool=policy)), CODES.SPOOL_UNAVAILABLE
        )
        self.assertEqual(source.batch_calls, 0)


class OnePassTests(TempCase):
    def test_a_one_pass_family_ignores_the_spool_and_creates_nothing(self) -> None:
        values = weibull() + 1.0
        for family in ("exponential", "normal", "gamma", "lognormal"):
            with self.subTest(family=family):
                with mock.patch.object(
                    _fit, "Spool", side_effect=AssertionError("a spool was made")
                ):
                    source = Counting(NumpySource(values))
                    result = fit(source, family, spool=self.policy())
                self.assertEqual(source.batch_calls, 1)
                self.assertIsNone(result.scale.spool)
                self.assert_clean()
                self.assertEqual(entries(self.root), ["spools"])

    def test_a_one_pass_fit_is_the_same_with_and_without_the_policy(self) -> None:
        values = weibull() + 1.0
        for family in ("exponential", "normal", "gamma", "lognormal"):
            with self.subTest(family=family):
                plain = fit(NumpySource(values), family)
                spooled = fit(NumpySource(values), family, spool=self.policy())
                self.assertEqual(result_bits(spooled), result_bits(plain))
                self.assertEqual(spooled.scale, plain.scale)

    def test_a_one_pass_family_reads_a_single_use_source_once_and_needs_no_pyarrow_for_a_spool(
        self,
    ) -> None:
        pulled: list[int] = []
        values = weibull() + 1.0
        result = fit(stream_source(values, pulled=pulled), "gamma", spool=self.policy())
        self.assertEqual(sum(pulled), len(values))
        self.assertIsNone(result.scale.spool)
        with mock.patch.dict(sys.modules, BLOCK_PYARROW):
            plain = fit(NumpySource(values), "gamma", spool=self.policy())
        self.assertIsNone(plain.scale.spool)
        self.assert_clean()

    def test_a_one_pass_family_refuses_a_second_read_of_a_single_use_source_as_before(self) -> None:
        source = stream_source(weibull() + 1.0)
        fit(source, "exponential")
        self.assertEqual(code_of(lambda: fit(source, "exponential")), CODES.SOURCE_CONSUMED)


class FitTests(TempCase):
    def test_a_spool_gives_the_same_result_bits_as_no_spool(self) -> None:
        for family, values in (("weibull_min", weibull()), ("gumbel_right", gumbel())):
            for rows in (DEFAULT_MAX_ROWS, 1000, 97):
                with self.subTest(family=family, rows=rows):
                    plain = fit(NumpySource(values), family, max_rows=rows)
                    spooled = fit(NumpySource(values), family, max_rows=rows, spool=self.policy())
                    self.assertEqual(result_bits(spooled), result_bits(plain))
                    self.assertEqual(spooled.scale.state, plain.scale.state)
                    self.assertEqual(spooled.scale.passes, plain.scale.passes)
                    self.assertEqual(replace(spooled.scale, spool=None), plain.scale)
                    self.assertEqual(
                        scale_provenance_to_canonical_json_bytes(
                            replace(spooled.scale, spool=None)
                        ),
                        scale_provenance_to_canonical_json_bytes(plain.scale),
                    )
                    self.assert_clean()

    def test_a_custom_source_with_several_partitions_gives_the_same_bits(self) -> None:
        values = weibull(2500)
        parts = [values[:700], values[700:700], values[700:1900], values[1900:]]
        plain = fit(ListSource(parts), "weibull_min", max_rows=300)
        source = ListSource(parts)
        spooled = fit(source, "weibull_min", max_rows=300, spool=self.policy())
        self.assertEqual(result_bits(spooled), result_bits(plain))
        self.assertEqual(spooled.scale.state, plain.scale.state)
        self.assertEqual(source.batch_calls, [0, 1, 2, 3])
        assert spooled.scale.spool is not None
        self.assertEqual([p.rows for p in spooled.scale.spool.partitions], [700, 0, 1200, 600])
        self.assertEqual(spooled.scale.partition_count, 4)

    def test_a_single_use_stream_gives_the_result_of_a_re_readable_copy(self) -> None:
        for family, values in (("weibull_min", weibull()), ("gumbel_right", gumbel())):
            with self.subTest(family=family):
                copy = fit(ArrowSource(pa.table({"x": values}), "x"), family)
                spooled = fit(stream_source(values, rows=400), family, spool=self.policy())
                self.assertEqual(result_bits(spooled), result_bits(copy))
                self.assertEqual(spooled.scale.state, copy.scale.state)
                self.assertEqual(spooled.scale.passes, copy.scale.passes)
                self.assertEqual(spooled.scale.row_count, len(values))
                self.assertEqual(spooled.scale.source_kind, "arrow")
                self.assertEqual(spooled.scale.partitions[0].fingerprint_level, "none")
                self.assert_clean()

    def test_the_source_is_read_exactly_once(self) -> None:
        values = weibull()
        plain = Counting(NumpySource(values))
        fit(plain, "weibull_min")
        self.assertGreaterEqual(plain.batch_calls, 3)
        spooled = Counting(NumpySource(values))
        result = fit(spooled, "weibull_min", spool=self.policy())
        self.assertEqual(spooled.batch_calls, 1)
        self.assertEqual(spooled.partition_calls, 1)
        assert result.scale.passes is not None
        self.assertGreaterEqual(result.scale.passes.pass_count, 3)
        several = Counting(ListSource([values[:1000], values[1000:2000], values[2000:]]))
        fit(several, "gumbel_right", spool=self.policy())
        self.assertEqual(several.batch_calls, 3)
        self.assertEqual(several.partition_calls, 1)

    def test_a_stream_is_pulled_once_and_in_order(self) -> None:
        pulled: list[int] = []
        values = weibull(2800)
        fit(stream_source(values, rows=700, pulled=pulled), "weibull_min", spool=self.policy())
        self.assertEqual(pulled, [700, 700, 700, 700])

    def test_parquet_gives_the_same_bits_with_and_without_a_spool(self) -> None:
        values = weibull(2400)
        for family in ("weibull_min", "gumbel_right"):
            data = values if family == "weibull_min" else gumbel(2400)
            root = self.root / f"parquet-{family}"
            write_parquet(root / "a.parquet", {"x": data[:1000]}, row_group_size=300)
            write_parquet(root / "b.parquet", {"x": data[1000:]}, row_group_size=500)
            with self.subTest(family=family):
                plain = fit(ParquetSource(root, "x"), family, max_rows=256)
                spooled = fit(ParquetSource(root, "x"), family, max_rows=256, spool=self.policy())
                self.assertEqual(result_bits(spooled), result_bits(plain))
                self.assertEqual(spooled.scale.state, plain.scale.state)
                self.assertEqual(replace(spooled.scale, spool=None), plain.scale)
                assert spooled.scale.spool is not None
                self.assertEqual(
                    [p.ordinal for p in spooled.scale.spool.partitions],
                    [p.ordinal for p in spooled.scale.partitions],
                )
                self.assertEqual(
                    [p.rows for p in spooled.scale.spool.partitions],
                    [p.rows for p in spooled.scale.partitions],
                )
                self.assert_clean()

    def test_the_provenance_records_the_spool_without_a_path(self) -> None:
        values = weibull(2500)
        parts = [values[:1500], values[1500:]]
        result = fit(ListSource(parts), "weibull_min", spool=self.policy())
        record = result.scale.spool
        assert record is not None
        self.assertEqual(record.row_count, 2500)
        self.assertEqual([item.ordinal for item in record.partitions], [0, 1])
        self.assertEqual([item.rows for item in record.partitions], [1500, 1000])
        self.assertEqual(
            [item.digest_sha256 for item in record.partitions], [digest_of(p) for p in parts]
        )
        encoded = scale_provenance_to_canonical_json_bytes(result.scale)
        self.assertIn(b'"spool":{"partitions":[{"digest_sha256":"', encoded)
        for text in (str(self.directory), str(self.root), "veridist-spool", ".arrow"):
            self.assertNotIn(text.encode(), encoded)
            self.assertNotIn(text, repr(result.scale))

    def test_no_spool_leaves_the_provenance_without_one(self) -> None:
        result = fit(NumpySource(weibull()), "weibull_min")
        self.assertIsNone(result.scale.spool)
        self.assertNotIn(b"spool", scale_provenance_to_canonical_json_bytes(result.scale))

    def test_a_failed_fit_still_records_the_spool_of_what_was_read(self) -> None:
        empty = fit(NumpySource(np.empty(0)), "weibull_min", spool=self.policy())
        self.assertIsInstance(empty.fit, WeibullFitFailure)
        self.assertEqual(empty.fit.code, WeibullFitFailureCode.EMPTY_SAMPLE)  # type: ignore[union-attr]
        assert empty.scale.spool is not None
        self.assertEqual(empty.scale.spool.row_count, 0)
        self.assertEqual(empty.scale.spool.partitions[0].digest_sha256, digest_of([]))
        invalid = fit(NumpySource(np.array([-1.0, 2.0, 3.0])), "weibull_min", spool=self.policy())
        self.assertEqual(invalid.fit.code, WeibullFitFailureCode.INVALID_SUPPORT)  # type: ignore[union-attr]
        assert invalid.scale.spool is not None
        self.assertEqual(invalid.scale.spool.row_count, 3)
        self.assertIsNone(invalid.scale.state)
        constant = fit(NumpySource(np.full(5, 2.0)), "gumbel_right", spool=self.policy())
        self.assertIsInstance(constant.fit, GumbelFitFailure)
        self.assertEqual(constant.fit.code, GumbelFitFailureCode.DEGENERATE_SAMPLE)  # type: ignore[union-attr]
        assert constant.scale.spool is not None
        self.assertEqual(
            constant.scale.spool.partitions[0].digest_sha256, digest_of(np.full(5, 2.0))
        )
        self.assert_clean()

    def test_an_empty_source_with_no_partition_has_an_empty_spool(self) -> None:
        result = fit(ListSource([]), "weibull_min", spool=self.policy())
        assert result.scale.spool is not None
        self.assertEqual((result.scale.spool.row_count, result.scale.spool.partitions), (0, ()))
        self.assertEqual(result.scale.partitions, ())
        self.assert_clean()

    def test_a_spool_in_the_system_directory_when_allowed(self) -> None:
        with mock.patch.object(tempfile, "gettempdir", return_value=str(self.directory)):
            result = fit(
                stream_source(weibull()),
                "weibull_min",
                spool=ScaleSpoolPolicy(max_bytes=1 << 24, allow_system_temp=True),
            )
        self.assertIsNotNone(result.scale.spool)
        self.assert_clean()

    def test_two_fits_may_share_a_directory(self) -> None:
        first = fit(NumpySource(weibull()), "weibull_min", spool=self.policy())
        second = fit(NumpySource(weibull()), "weibull_min", spool=self.policy())
        self.assertEqual(result_bits(first), result_bits(second))
        self.assert_clean()

    def test_a_spooled_fit_of_a_source_with_an_integer_column_matches(self) -> None:
        values = np.arange(1, 2001, dtype=np.int64) % 97 + 1
        plain = fit(NumpySource(values), "weibull_min")
        spooled = fit(NumpySource(values), "weibull_min", spool=self.policy())
        self.assertEqual(result_bits(spooled), result_bits(plain))


if __name__ == "__main__":
    unittest.main()
