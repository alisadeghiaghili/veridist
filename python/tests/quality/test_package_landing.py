"""Release-metadata contracts for the nested Veridist package."""

from __future__ import annotations

import re
import subprocess
import sys
import tomllib
import unicodedata
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PYTHON_ROOT.parent
PYPROJECT = PYTHON_ROOT / "pyproject.toml"
LANGUAGE_NAVIGATION = (
    "[English](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.md)"
    " | [فارسی](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.fa.md)"
    " | [Deutsch](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md)"
)
README_PATHS = {
    "en": PYTHON_ROOT / "README.md",
    "fa": PYTHON_ROOT / "README.fa.md",
    "de": PYTHON_ROOT / "README.de.md",
}


class PackageLandingContractTests(unittest.TestCase):
    def test_repository_landings_link_to_the_release_package_and_support(self) -> None:
        required = {
            REPOSITORY_ROOT / "README.md": (
                "# veridist",
                "python/README.md",
                "README.fa.md",
                "README.de.md",
                "python/CHANGELOG.md",
                "python/KNOWN_LIMITS.md",
            ),
            REPOSITORY_ROOT / "README.fa.md": (
                "# veridist",
                "python/README.fa.md",
                "python/KNOWN_LIMITS.fa.md",
            ),
            REPOSITORY_ROOT / "README.de.md": (
                "# veridist",
                "python/README.de.md",
                "python/KNOWN_LIMITS.de.md",
            ),
            REPOSITORY_ROOT / "SECURITY.md": (
                "# Security policy",
                "GitHub Security Advisory",
            ),
        }
        for path, phrases in required.items():
            content = " ".join(path.read_text(encoding="utf-8").split()).casefold()
            with self.subTest(path=path):
                for phrase in phrases:
                    self.assertIn(phrase.casefold(), content)

    def test_project_metadata_points_to_packaged_human_facing_material(self) -> None:
        configuration = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
        project = configuration["project"]
        self.assertIn("setuptools>=77", configuration["build-system"]["requires"])
        self.assertEqual(
            project["description"],
            "Evidence-first distribution fitting with explicit execution contracts",
        )
        self.assertNotIn("greenfield", project["description"].casefold())
        self.assertEqual(project["readme"], {"file": "README.md", "content-type": "text/markdown"})
        self.assertEqual(project["license"], "BUSL-1.1")
        self.assertEqual(project["license-files"], ["LICENSE"])
        self.assertEqual(
            project["urls"],
            {
                "Homepage": "https://github.com/alisadeghiaghili/veridist",
                "Documentation": (
                    "https://github.com/alisadeghiaghili/veridist/tree/main/python/docs"
                ),
                "Repository": "https://github.com/alisadeghiaghili/veridist",
                "Issues": "https://github.com/alisadeghiaghili/veridist/issues",
            },
        )

    def test_nested_license_is_an_exact_copy_of_the_repository_license(self) -> None:
        self.assertEqual(
            (PYTHON_ROOT / "LICENSE").read_text(encoding="utf-8").splitlines(),
            (REPOSITORY_ROOT / "LICENSE").read_text(encoding="utf-8").splitlines(),
        )

    def test_all_three_package_readmes_link_each_other_and_install_from_pypi(self) -> None:
        for locale, path in README_PATHS.items():
            with self.subTest(locale=locale):
                content = path.read_text(encoding="utf-8")
                self.assertIn(LANGUAGE_NAVIGATION, content)
                self.assertIn("veridist", content.casefold())
                self.assertRegex(content, r"(?m)^python -m pip install veridist\s*$")

    def test_package_landings_cover_the_adoption_journey_without_copy_assertions(self) -> None:
        for locale, path in README_PATHS.items():
            content = path.read_text(encoding="utf-8")
            with self.subTest(locale=locale):
                self.assertIn("python -m pip install veridist", content)
                self.assertIn("```python", content)
                self.assertIn("```text", content)
                self.assertIn("SQLiteCheckpointStore", content)
                self.assertIn("KNOWN_LIMITS", content)
                self.assertIn("Coverage ≥95%", content)

    def test_all_locales_publish_the_same_executable_quickstart(self) -> None:
        snippets: dict[str, str] = {}
        for locale, path in README_PATHS.items():
            content = path.read_text(encoding="utf-8")
            matches = re.findall(r"```python\n(.*?)```", content, flags=re.DOTALL)
            with self.subTest(locale=locale):
                self.assertEqual(len(matches), 1)
            snippets[locale] = matches[0].strip()

        self.assertEqual(len(set(snippets.values())), 1)
        quickstart = snippets["en"]
        for required in (
            "fit_exponential_csv",
            "CsvLifetimeSchema",
            "CsvLifetimeLimits",
            "PublicSourceId",
            "assert fit.rate == 0.5",
            'assert fit.inference == "not_provided"',
            'assert fit.censoring_assumption == "independent_right_censoring"',
            'path.write_text("time,event_observed\\n1,1\\n1,0\\n", encoding="utf-8")',
        ):
            with self.subTest(required=required):
                self.assertIn(required, quickstart)

        namespace: dict[str, object] = {}
        exec(compile(quickstart, "package-landing-quickstart", "exec"), namespace)

    def test_root_landings_offer_the_same_task_paths(self) -> None:
        for filename in ("README.md", "README.fa.md", "README.de.md"):
            content = (REPOSITORY_ROOT / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                self.assertIn("Coverage ≥95%", content)
                self.assertIn("python -m pip install veridist", content)
                self.assertIn("GitHub Issues", content)
                self.assertIn("python/README", content)
                self.assertIn("KNOWN_LIMITS", content)
                self.assertEqual(len(re.findall(r"```python\n(.*?)```", content, re.DOTALL)), 1)
                self.assertEqual(len(re.findall(r"```text\n(.*?)```", content, re.DOTALL)), 1)

    def test_root_quickstarts_are_equivalent_and_executable(self) -> None:
        snippets: dict[str, str] = {}
        outputs: dict[str, str] = {}
        root_readmes = {"en": "README.md", "fa": "README.fa.md", "de": "README.de.md"}
        for locale, filename in root_readmes.items():
            content = (REPOSITORY_ROOT / filename).read_text(encoding="utf-8")
            snippets[locale] = re.search(
                r"```python\n(.*?)```", content, re.DOTALL
            ).group(1).strip()
            outputs[locale] = re.search(r"```text\n(.*?)```", content, re.DOTALL).group(1).strip()

        self.assertEqual(len(set(snippets.values())), 1)
        self.assertEqual(set(outputs.values()), {"rate=0.5; events=1; censored=1"})
        namespace: dict[str, object] = {}
        exec(compile(snippets["en"], "root-landing-quickstart", "exec"), namespace)

    def test_checkpoint_resume_example_is_executable(self) -> None:
        example = PYTHON_ROOT / "examples" / "checkpoint_resume.py"
        result = subprocess.run(
            [sys.executable, str(example)],
            cwd=PYTHON_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "rows=2; events=1; total_time=3.75")

    def test_readme_local_links_and_anchors_resolve(self) -> None:
        readmes = (
            REPOSITORY_ROOT / "README.md",
            REPOSITORY_ROOT / "README.fa.md",
            REPOSITORY_ROOT / "README.de.md",
            *README_PATHS.values(),
        )
        for readme in readmes:
            content = readme.read_text(encoding="utf-8")
            for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", content):
                href = target.split(maxsplit=1)[0].strip("<>")
                if href.startswith(("http://", "https://", "mailto:")):
                    continue
                relative_path, separator, anchor = href.partition("#")
                destination = (readme.parent / relative_path).resolve() if relative_path else readme
                with self.subTest(readme=readme, href=href, contract="destination"):
                    self.assertTrue(destination.exists())
                if separator:
                    headings = re.findall(
                        r"(?m)^#{1,6}\s+(.+?)\s*$",
                        destination.read_text(encoding="utf-8"),
                    )
                    anchors = {self._github_like_anchor(heading) for heading in headings}
                    with self.subTest(readme=readme, href=href, contract="anchor"):
                        self.assertIn(anchor, anchors)

    def test_localized_readme_structures_are_parallel(self) -> None:
        readme_groups = (
            (
                REPOSITORY_ROOT / "README.md",
                REPOSITORY_ROOT / "README.fa.md",
                REPOSITORY_ROOT / "README.de.md",
            ),
            tuple(README_PATHS.values()),
        )
        for paths in readme_groups:
            heading_counts = {
                path.name: len(
                    re.findall(r"(?m)^## ", path.read_text(encoding="utf-8"))
                )
                + len(re.findall(r"(?mi)^<h2\b", path.read_text(encoding="utf-8")))
                for path in paths
            }
            with self.subTest(paths=tuple(path.name for path in paths)):
                self.assertEqual(len(set(heading_counts.values())), 1, heading_counts)

    @staticmethod
    def _github_like_anchor(heading: str) -> str:
        normalized = "".join(
            character
            for character in heading.casefold()
            if character.isalnum() or character in " -_"
        )
        return re.sub(r"[\s-]+", "-", normalized.strip())

    def test_all_landing_pages_are_clean_nfc_without_retired_claims_or_bidi_controls(self) -> None:
        retired_claims = (
            "does not provide a distribution-fitting API",
            "API برازش توزیع ارائه نمی‌کند",
            "keine API zur Verteilungsanpassung",
        )
        mojibake_markers = ("Ã", "Â", "Ø", "Ù", "�")
        bidi_controls = tuple(chr(codepoint) for codepoint in range(0x202A, 0x202F)) + tuple(
            chr(codepoint) for codepoint in range(0x2066, 0x206A)
        )
        for locale, path in README_PATHS.items():
            content = path.read_text(encoding="utf-8")
            with self.subTest(locale=locale, contract="nfc"):
                self.assertEqual(content, unicodedata.normalize("NFC", content))
            with self.subTest(locale=locale, contract="mojibake"):
                self.assertFalse(any(marker in content for marker in mojibake_markers))
            with self.subTest(locale=locale, contract="retired"):
                self.assertFalse(any(claim in content for claim in retired_claims))
            with self.subTest(locale=locale, contract="bidi-controls"):
                self.assertFalse(any(control in content for control in bidi_controls))

    def test_persian_rtl_wrappers_never_contain_ltr_code_fences(self) -> None:
        for path in (REPOSITORY_ROOT / "README.fa.md", README_PATHS["fa"]):
            content = path.read_text(encoding="utf-8")
            with self.subTest(path=path, contract="wrapper"):
                self.assertRegex(
                    content,
                    r'<div lang="fa" dir="rtl"(?: align="right")?>',
                )
            for fence in re.finditer(r"```(?:console|python|text|mermaid)?", content):
                prefix = content[: fence.start()]
                with self.subTest(path=path, fence=fence.group(0), offset=fence.start()):
                    self.assertEqual(
                        len(
                            re.findall(
                                r'<div lang="fa" dir="rtl"(?: align="right")?>',
                                prefix,
                            )
                        ),
                        prefix.count("</div>"),
                    )

    def test_source_manifest_includes_all_package_landing_files(self) -> None:
        manifest = (PYTHON_ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            manifest,
            [
                "include LICENSE",
                "include README.md",
                "include README.fa.md",
                "include README.de.md",
                "include CHANGELOG.md",
                "include KNOWN_LIMITS.md",
                "include KNOWN_LIMITS.fa.md",
                "include KNOWN_LIMITS.de.md",
                "include src/veridist/py.typed",
            ],
        )

    def test_candidate_changelog_and_known_limits_are_complete_and_parallel(self) -> None:
        changelog = (PYTHON_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        for required in (
            "## [1.0.0] - 2026-09-11",
            "AIC/BIC, adequacy-gated model selection",
            "CvM goodness-of-fit",
            "KNOWN_LIMITS.md",
        ):
            self.assertIn(required, changelog)

        limits = {
            locale: (PYTHON_ROOT / filename).read_text(encoding="utf-8")
            for locale, filename in {
                "en": "KNOWN_LIMITS.md",
                "fa": "KNOWN_LIMITS.fa.md",
                "de": "KNOWN_LIMITS.de.md",
            }.items()
        }
        contract_ids = (
            "FIT-CSV-EXP",
            "CSV-STRICT",
            "SCALAR-FAMILIES",
            "STREAM-SOURCE",
            "MEMORY-BOUND",
            "SCALE-EVIDENCE",
            "LICENSE",
        )
        for locale, content in limits.items():
            with self.subTest(locale=locale):
                self.assertIn("1.0.1", content)
                self.assertIn("BUSL-1.1", content)
                self.assertIn("Apache-2.0", content)
                self.assertIn("2030-09-05", content)
                self.assertEqual(
                    [contract_id for contract_id in contract_ids if f"`{contract_id}`" in content],
                    list(contract_ids),
                )


    def test_citation_guide_keeps_common_formats_and_readmes_link_to_it(self) -> None:
        guide = (REPOSITORY_ROOT / "docs" / "citing-veridist.md").read_text(encoding="utf-8")
        for heading in (
            "## IEEE",
            "## APA 7",
            "## BibTeX",
            "## RIS",
            "## EndNote XML",
            "## CSL-JSON",
            "## Chicago author-date",
            "## MLA 9",
            "## Harvard",
            "## Vancouver",
        ):
            self.assertIn(heading, guide)
        self.assertIn("ver. 1.0.1", guide)
        self.assertIn("CITATION.cff", guide)
        root_readmes = tuple(
            REPOSITORY_ROOT / name for name in ("README.md", "README.fa.md", "README.de.md")
        )
        readmes = (*README_PATHS.values(), *root_readmes)
        for path in readmes:
            with self.subTest(path=path):
                self.assertIn("citing-veridist.md", path.read_text(encoding="utf-8"))

if __name__ == "__main__":
    unittest.main()
