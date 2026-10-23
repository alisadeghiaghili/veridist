"""The large-data guide documents the scale engine that actually exists, in every language."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import re
import unittest
from collections import Counter
from pathlib import Path

from veridist import scale
from veridist.scale import _gof, _planner, _profile_intervals, _sources, _tasks
from veridist.scale._executors import MAX_WORKERS

PYTHON_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = PYTHON_ROOT / "docs"
PAGE = DOCS_ROOT / "source" / "large-data.md"
TOOLCHAIN = DOCS_ROOT / "toolchain.py"

FENCE = re.compile(r"```python\n(.*?)```(?:\n\n```text\n(.*?)```)?", re.DOTALL)
HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


def load_toolchain():
    spec = importlib.util.spec_from_file_location("veridist_docs_toolchain_scale", TOOLCHAIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fences(path: Path) -> list[tuple[str, str]]:
    return FENCE.findall(path.read_text(encoding="utf-8").replace("\r\n", "\n"))


def run(source: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(source, "<large-data>", "exec"), {"__name__": "__docs__"})  # noqa: S102
    return buffer.getvalue()


def page_text() -> str:
    return " ".join(PAGE.read_text(encoding="utf-8").split())


def literals(message: str) -> list[str]:
    """The code literals of a message, whether written with backticks or ``<code>``."""

    found = re.findall(r"`([^`]+)`", message)
    found += re.findall(r'<code dir="ltr">(.*?)</code>', message)
    return sorted(found)


def numbers(message: str) -> list[str]:
    plain = re.sub(r"<[^>]+>", " ", message)
    return sorted(re.findall(r"\d+", plain))


class LargeDataExamplesTests(unittest.TestCase):
    def test_every_python_fence_runs_and_prints_its_output(self) -> None:
        examples = fences(PAGE)
        self.assertGreaterEqual(len(examples), 6)
        for index, (source, expected) in enumerate(examples):
            if "pyarrow" in source and not HAS_PYARROW:
                continue
            with self.subTest(example=index):
                printed = run(source)
                if expected:
                    self.assertEqual(printed.split(), expected.split())

    def test_the_page_uses_only_public_names(self) -> None:
        text = PAGE.read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"veridist\.scale\._[a-z]", text))
        for source, _ in fences(PAGE):
            self.assertNotIn("scale._", source)
            for name in re.findall(r"(?<![\w.])scale\.([A-Za-z_][A-Za-z_0-9]*)", source):
                with self.subTest(name=name):
                    self.assertIn(name, scale.__all__)
        for name in re.findall(r"`scale\.([A-Za-z_][A-Za-z_0-9]*)", text):
            with self.subTest(name=name):
                self.assertIn(name, scale.__all__)


class LargeDataContractTests(unittest.TestCase):
    def test_the_entry_points_sources_and_results_are_all_documented(self) -> None:
        text = page_text()
        for name in (
            "fit",
            "profile_intervals",
            "large_n_gof",
            "NumpySource",
            "ParquetSource",
            "ArrowSource",
            "PolarsSource",
            "DuckDBSource",
            "ThreadExecutor",
            "ProcessExecutor",
            "ScaleSpoolPolicy",
            "SQLitePartitionStore",
            "ScaleFitResult",
            "ScaleProvenance",
            "LargeNGof",
            "GofDecision",
            "ProfileUnavailable",
            "PassBudgetError",
        ):
            with self.subTest(name=name):
                self.assertIn(name, scale.__all__)
                self.assertIsNotNone(re.search(rf"`[^`]*{name}[^`]*`", text))

    def test_every_error_code_is_listed_with_its_error_class(self) -> None:
        section = page_text().partition("## Errors and their codes")[2]
        classes = {
            "ScaleSourceError": scale.ScaleSourceErrorCode,
            "ScaleStateError": scale.ScaleStateErrorCode,
            "ScaleCheckpointError": scale.ScaleCheckpointErrorCode,
            "ScaleInferenceError": scale.ScaleInferenceErrorCode,
        }
        for name, enum in classes.items():
            entry = section.partition(f"* `{name}`")[2].partition(" * `")[0]
            self.assertTrue(entry, name)
            listed = set(re.findall(r"`([A-Z_]+)`", entry))
            with self.subTest(error=name):
                self.assertEqual(listed, {member.value for member in enum})
        entry = section.partition("* `PassBudgetError`")[2]
        self.assertEqual(
            set(re.findall(r"`([A-Z_]+)`", entry)),
            {"PASS_BUDGET_INSUFFICIENT", "PASS_BUDGET_EXCEEDED"},
        )

    def test_the_numbers_of_the_page_are_the_numbers_of_the_code(self) -> None:
        text = page_text()
        self.assertEqual(_sources.DEFAULT_MAX_ROWS, 262144)
        self.assertIn("(default 262144)", text)
        self.assertEqual(_planner.PASS_BUDGET_DEFAULT, 8)
        self.assertIn("(default 8)", text)
        self.assertEqual((_planner.MINIMUM_PASSES, _planner.MAXIMUM_PASSES), (3, 17))
        self.assertIn("at least 3 and at most 17 times", text)
        self.assertIn("A budget below 3 is refused", text)
        self.assertEqual(
            (_profile_intervals.PROFILE_MIN_PASSES, _profile_intervals.PROFILE_MAX_PASSES),
            (2, 9),
        )
        self.assertIn("at least 2 and at most 9 passes", text)
        self.assertEqual(_profile_intervals.PROFILE_PASS_BUDGET_DEFAULT, 6)
        self.assertIn("`inference_pass_budget` (default 6)", text)
        self.assertEqual(
            (_gof.DEFAULT_BIN_EXPONENT, _gof.MIN_BIN_EXPONENT, _gof.MAX_BIN_EXPONENT),
            (12, 1, 20),
        )
        self.assertIn("`bin_exponent` (default 12, from 1 to 20)", text)
        self.assertEqual(MAX_WORKERS, 64)
        self.assertIn("capped by 64", text)
        self.assertEqual(_tasks.MAX_PARTITION_RETRIES, 2)
        self.assertIn("at most twice", text)
        from veridist.inference import REFIT_MAX_OBSERVATIONS

        self.assertEqual(REFIT_MAX_OBSERVATIONS, 100000)
        self.assertIn("more than 100000 observations", text)

    def test_every_downgrade_reason_a_bundled_source_can_give_is_documented(self) -> None:
        text = page_text()
        for reason in (
            _sources.DowngradeReason.SINGLE_PARTITION,
            _sources.DowngradeReason.SINGLE_USE_SOURCE,
            _sources.DowngradeReason.ENGINE_MANAGED,
            _sources.DowngradeReason.IN_MEMORY_DATA,
            _sources.DowngradeReason.SPOOL_FIRST_PASS,
        ):
            with self.subTest(reason=reason.value):
                self.assertIn(f"`{reason.value}`", text)

    def test_the_page_states_the_scope_and_the_reproducibility_tiers(self) -> None:
        text = page_text()
        for expected in (
            "Right censoring at scale is not supported yet",
            "bit-identical",
            "numeric environment",
            "bit identity is not promised",
            "There is no p-value",
            "pool has a start-up cost",
            "`spawn`",
            'if __name__ == "__main__":',
            "SPOOL_REQUIRED",
            "never a path",
            "`pip install \"veridist[arrow]\"`",
            "`pip install \"veridist[polars]\"`",
            "`pip install \"veridist[duckdb]\"`",
            "never silently",
            "public API",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, text)

    def test_the_page_is_linked_from_the_index_and_the_api_guide(self) -> None:
        index = (DOCS_ROOT / "source" / "index.md").read_text(encoding="utf-8")
        self.assertIn("large-data", index.partition("```{toctree}")[2])
        for name in ("api.md", "api.fa.md", "api.de.md"):
            with self.subTest(page=name):
                api = (DOCS_ROOT / "source" / name).read_text(encoding="utf-8")
                self.assertIn("large-data.md", api)


class LargeDataLocaleParityTests(unittest.TestCase):
    """Code literals and numbers are the same in every language of the page."""

    def test_literals_and_numbers_are_identical_in_persian_and_german(self) -> None:
        toolchain = load_toolchain()
        import json

        manifest = json.loads((DOCS_ROOT / "i18n" / "parity-manifest.json").read_text("utf-8"))
        sources = [m["source"] for m in manifest["messages"] if m["page"] == "large-data.md"]
        self.assertGreaterEqual(len(sources), 40)
        for locale in ("fa", "de"):
            catalog = toolchain._parse_po(
                DOCS_ROOT / "locales" / locale / "LC_MESSAGES" / "large-data.po"
            )
            for source in sources:
                translation = catalog[source]
                with self.subTest(locale=locale, message=source[:50]):
                    self.assertEqual(literals(translation), literals(source))
                    self.assertEqual(numbers(translation), numbers(source))
                    self.assertEqual(
                        Counter(re.findall(r"[A-Z][A-Z_]{4,}", translation)),
                        Counter(re.findall(r"[A-Z][A-Z_]{4,}", source)),
                    )

    def test_the_page_is_registered_and_translated_in_every_catalog(self) -> None:
        import json

        manifest = json.loads((DOCS_ROOT / "i18n" / "parity-manifest.json").read_text("utf-8"))
        pages = {page["source"]: page for page in manifest["pages"]}
        self.assertEqual(pages["large-data.md"]["id"], "DOC-PAGE-LARGE-DATA")
        self.assertEqual(pages["large-data.md"]["anchor"], "veridist-large-data")
        for locale in ("fa", "de"):
            self.assertTrue(
                (DOCS_ROOT / "locales" / locale / "LC_MESSAGES" / "large-data.po").is_file()
            )


class LargeDataSurroundingDocsTests(unittest.TestCase):
    def test_the_readmes_describe_the_large_data_path_and_the_extras(self) -> None:
        repository = PYTHON_ROOT.parent
        for path in (
            repository / "README.md",
            repository / "README.fa.md",
            repository / "README.de.md",
            PYTHON_ROOT / "README.md",
            PYTHON_ROOT / "README.fa.md",
            PYTHON_ROOT / "README.de.md",
        ):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name, root=path.parent.name):
                self.assertIn("veridist.scale", text)
                for extra in ("veridist[arrow]", "veridist[polars]", "veridist[duckdb]"):
                    self.assertIn(extra, text)
                self.assertIn("large-data.md", text)

    def test_known_limits_has_a_scale_engine_entry_in_every_language(self) -> None:
        for name in ("KNOWN_LIMITS.md", "KNOWN_LIMITS.fa.md", "KNOWN_LIMITS.de.md"):
            text = (PYTHON_ROOT / name).read_text(encoding="utf-8")
            with self.subTest(page=name):
                self.assertIn("SCALE-ENGINE", text)
                self.assertIn("veridist.scale", text)

    def test_no_document_still_says_only_the_csv_path_is_scale_capable(self) -> None:
        stale = (
            "the only bundled file adapter",
            "only bundled file adapter",
            "The package bundles no Parquet, Arrow, dataframe, database, or network",
            "generic dataframe/database adapters",
            "not through a file-fitting API",
        )
        repository = PYTHON_ROOT.parent
        paths = [
            *(DOCS_ROOT / "source").glob("*.md"),
            *PYTHON_ROOT.glob("KNOWN_LIMITS*.md"),
            *repository.glob("docs/capability-guide*.md"),
            *repository.glob("README*.md"),
            *PYTHON_ROOT.glob("README*.md"),
        ]
        for path in paths:
            text = path.read_text(encoding="utf-8")
            for claim in stale:
                with self.subTest(page=path.name, claim=claim):
                    self.assertNotIn(claim, text)

    def test_the_changelog_announces_the_public_scale_engine_once(self) -> None:
        changelog = (PYTHON_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        section = changelog.partition("## [2.2.0] - 2026-10-23")[2].partition("\n## [")[0]
        self.assertEqual(section.count("\n### Added"), 1)
        self.assertEqual(section.count("\n### Changed"), 1)
        self.assertIn("veridist.scale", section.partition("### Added")[2])


if __name__ == "__main__":
    unittest.main()
