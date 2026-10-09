"""The 2.0 API guide and migration guide describe the API that actually exists."""

from __future__ import annotations

import contextlib
import io
import re
import unittest
from pathlib import Path

import veridist

PYTHON_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PYTHON_ROOT / "docs" / "source"
MIGRATION = PYTHON_ROOT / "docs" / "migration-2.0.md"

FENCE = re.compile(r"```python\n(.*?)```(?:\n\n```text\n(.*?)```)?", re.DOTALL)


def fences(path: Path) -> list[tuple[str, str | None]]:
    return FENCE.findall(path.read_text(encoding="utf-8").replace("\r\n", "\n"))


def run(source: str) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(source, "<docs>", "exec"), {"__name__": "__docs__"})  # noqa: S102
    return buffer.getvalue()


class ApiGuideExampleTests(unittest.TestCase):
    def test_every_python_fence_of_the_english_guide_runs_and_prints_its_output(self) -> None:
        examples = fences(SOURCE_ROOT / "api.md")
        self.assertGreaterEqual(len(examples), 6)
        for index, (source, expected) in enumerate(examples):
            with self.subTest(example=index):
                printed = run(source)
                if expected:
                    self.assertEqual(printed.split(), expected.split())

    def test_the_translated_guides_carry_the_same_examples(self) -> None:
        english = fences(SOURCE_ROOT / "api.md")
        for name in ("api.fa.md", "api.de.md"):
            with self.subTest(page=name):
                self.assertEqual(fences(SOURCE_ROOT / name), english)

    def test_the_guide_covers_the_two_point_zero_surface_in_every_language(self) -> None:
        for name in ("api.md", "api.fa.md", "api.de.md"):
            page = (SOURCE_ROOT / name).read_text(encoding="utf-8")
            for expected in (
                "uncertainty()",
                "confidence_intervals",
                "quantile(0.1)",
                "lifetimes_from_arrays",
                "values_from_arrays",
                "reduce_lifetime_log_likelihood_chunks",
                "reduce_value_log_likelihood_chunks",
                "logpdf",
                "FitSuccess",
                "CapabilityError",
                "gumbel_right",
                "UncertaintyUnavailable",
                "migration-2.0.md",
            ):
                with self.subTest(page=name, expected=expected):
                    self.assertIn(expected, page)

    def test_the_guide_no_longer_makes_the_stale_claims(self) -> None:
        stale = (
            "does not yet provide confidence intervals",
            "build likelihoods for censored data",
            "not an array API",
            "five evaluated",
            "fünf geprüften",
        )
        for name in ("api.md", "api.fa.md", "api.de.md", "families-log-density-likelihood.md"):
            page = (SOURCE_ROOT / name).read_text(encoding="utf-8")
            for claim in stale:
                with self.subTest(page=name, claim=claim):
                    self.assertNotIn(claim, page)

    def test_every_name_the_guide_imports_from_the_top_level_is_exported(self) -> None:
        for name in ("api.md", "api.fa.md", "api.de.md"):
            page = (SOURCE_ROOT / name).read_text(encoding="utf-8")
            for block in re.findall(r"from veridist import (\([^)]*\)|[^\n]+)", page):
                for item in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", block):
                    with self.subTest(page=name, item=item):
                        self.assertIn(item, veridist.__all__)


class MigrationGuideTests(unittest.TestCase):
    def test_the_migration_guide_covers_every_required_topic(self) -> None:
        page = MIGRATION.read_text(encoding="utf-8")
        for expected in (
            "cdf(\"gamma\", 2.0, shape=2.0, scale=1.0)",
            "removed in 3.0",
            "SHA-256",
            "(row_start, payload)",
            "create_checkpointed_csv_store",
            "Removed APIs",
            "Business Source License 1.1",
            "non-commercial",
            "uncertainty()",
            "`family` attribute",
            "FamilyId",
            "UncertaintyUnavailable",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, page)

    def test_the_runnable_examples_of_the_migration_guide_run(self) -> None:
        for index, (source, _expected) in enumerate(fences(MIGRATION)):
            if "import" not in source:
                continue
            with self.subTest(example=index):
                run(source)

    def test_the_guide_and_the_readmes_link_to_the_migration_guide(self) -> None:
        for path in (
            SOURCE_ROOT / "api.md",
            PYTHON_ROOT / "README.md",
            PYTHON_ROOT / "README.fa.md",
            PYTHON_ROOT / "README.de.md",
        ):
            with self.subTest(page=path.name):
                self.assertIn("migration-2.0.md", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
