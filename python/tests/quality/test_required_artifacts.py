"""Regression checks for required, version-controlled quality artifacts."""

from __future__ import annotations

import json
import subprocess
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PYTHON_ROOT.parent
MANIFEST = PYTHON_ROOT / "quality" / "coverage-manifest.json"
CAPABILITY_GUIDE = REPOSITORY_ROOT / "docs" / "capability-guide.md"
READINESS = REPOSITORY_ROOT / "docs" / "v1-readiness.md"
EVALUATED_FAMILY_ADR = REPOSITORY_ROOT / "docs" / "adr" / "ADR-0019-evaluated-family-kernel.md"
CSV_ADAPTER_ADR = (
    REPOSITORY_ROOT
    / "docs"
    / "adr"
    / "ADR-0018-csv-lifetime-adapter-and-one-pass-exponential-orchestrator.md"
)
MUTATION_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "mutation.yml"


class RequiredQualityArtifactTests(unittest.TestCase):
    def test_manifest_records_the_authoritative_production_file_count(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        production_count = len(manifest["production_files"])
        self.assertEqual(production_count, 31)
        self.assertIn("src/veridist/engine/streaming.py", manifest["production_files"])
        self.assertEqual(
            manifest["expected_denominators"]["src/veridist/engine/streaming.py"],
            {"statements": 53, "branches": 22},
        )

    def test_readiness_separates_historical_snapshot_from_current_evidence(
        self,
    ) -> None:
        readiness = READINESS.read_text(encoding="utf-8")
        self.assertIn("Historical snapshot: `bfb496d` (preserved verbatim)", readiness)
        self.assertIn("accepted all 23 enumerated\n  production files", readiness)
        self.assertIn("Current unmerged family-kernel candidate", readiness)
        self.assertNotIn("coverage.json` SHA-256", readiness)

    def test_evaluated_family_adr_uses_a_primary_immutable_math_source(self) -> None:
        adr = EVALUATED_FAMILY_ADR.read_text(encoding="utf-8")
        self.assertIn("NIST DLMF §5.11", adr)
        self.assertNotIn("github.com/wch/r-source/blob/trunk", adr)

    def test_release_cell_adrs_are_accepted(self) -> None:
        for adr in (CSV_ADAPTER_ADR, EVALUATED_FAMILY_ADR):
            with self.subTest(adr=adr):
                self.assertIn("Status: Accepted", adr.read_text(encoding="utf-8"))

        index = (REPOSITORY_ROOT / "docs" / "adr" / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "| 0018 | CSV lifetime adapter and one-pass exponential orchestrator | Accepted |",
            index,
        )
        self.assertIn(
            "| 0019 | Evaluated-family kernel and parameter contracts | Accepted |",
            index,
        )

    def test_capability_guide_declares_the_release_scope_and_limits(self) -> None:
        content = " ".join(CAPABILITY_GUIDE.read_text(encoding="utf-8").split())
        for required in (
            "1.0.1",
            "Exponential MLE",
            "Weibull-minimum MLE",
            "Lognormal MLE",
            "exact and independent right-censoring",
            "fixed O(1) reducer state",
            "Refit Monte Carlo KS/AD/CvM",
            "local filesystem",
            "no generic RSS or throughput claim",
            "at least 95% global line and branch coverage",
        ):
            with self.subTest(required=required):
                self.assertIn(required, content)

    def test_capability_guides_keep_the_same_release_topics_in_every_locale(self) -> None:
        guides = {
            "en": REPOSITORY_ROOT / "docs" / "capability-guide.md",
            "fa": REPOSITORY_ROOT / "docs" / "capability-guide.fa.md",
            "de": REPOSITORY_ROOT / "docs" / "capability-guide.de.md",
        }
        required_topics = {
            "en": (
                "lifetime of equipment",
                "right-censoring",
                "What result do I get?",
                "large or a run is interrupted",
                "not supported yet",
                "code quality",
                "Technical details",
            ),
            "fa": (
                "عمر دستگاه‌ها",
                "سانسورشده از راست",
                "چه نتیجه‌ای می‌گیرم؟",
                "داده زیاد باشد یا برنامه قطع شود",
                "هنوز پشتیبانی نمی‌شوند",
                "کیفیت کد ما",
                "جزئیات فنی",
            ),
            "de": (
                "Lebensdauer von Geräten",
                "Rechtszensierung",
                "Welches Ergebnis erhalte ich?",
                "großen Daten oder einer Unterbrechung",
                "noch nicht unterstützt",
                "Codequalität",
                "Technische Details",
            ),
        }
        for locale, path in guides.items():
            content = path.read_text(encoding="utf-8")
            for topic in required_topics[locale]:
                with self.subTest(locale=locale, topic=topic):
                    self.assertIn(topic, content)

    def test_coverage_manifest_exists_is_valid_and_is_not_gitignored(self) -> None:
        self.assertTrue(MANIFEST.is_file(), "required coverage manifest is missing")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(manifest["production_root"], "src/veridist")
        self.assertIn("production_files", manifest)
        self.assertIn("expected_denominators", manifest)

        result = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={PYTHON_ROOT.parent.as_posix()}",
                "check-ignore",
                "-q",
                "quality/coverage-manifest.json",
            ],
            cwd=PYTHON_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            1,
            "quality/coverage-manifest.json must remain visible to Git",
        )

    def test_mutation_checker_and_runner_are_versioned_and_cache_is_ignored(self) -> None:
        for artifact in (
            PYTHON_ROOT / "quality" / "mutation-manifest.json",
            PYTHON_ROOT / "tools" / "check_mutation_evidence.py",
            PYTHON_ROOT / "tools" / "run_mutation.py",
        ):
            with self.subTest(artifact=artifact):
                self.assertTrue(artifact.is_file())
        ignored = (PYTHON_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/mutants/", ignored)
        self.assertIn("/mutation-evidence.json", ignored)
        self.assertIn("/.mutation-tmp/", ignored)

    def test_mutation_cache_ignores_are_limited_to_the_python_root(self) -> None:
        ignored = (PYTHON_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        self.assertIn("/mutants/", ignored)
        self.assertIn("/mutation-evidence.json", ignored)
        self.assertIn("/.mutation-tmp/", ignored)

        ignored_paths = (
            "mutants/cache.py",
            "mutation-evidence.json",
            ".mutation-tmp/cache.json",
        )
        retained_paths = (
            "src/veridist/mutants/example.py",
            "tests/mutants/example.py",
            "subdir/mutation-evidence.json",
            "subdir/.mutation-tmp/cache.json",
        )
        for path in ignored_paths:
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "-q", "--no-index", "--", path],
                    cwd=PYTHON_ROOT,
                    check=False,
                )
                self.assertEqual(result.returncode, 0)
        for path in retained_paths:
            with self.subTest(path=path):
                result = subprocess.run(
                    ["git", "check-ignore", "-q", "--no-index", "--", path],
                    cwd=PYTHON_ROOT,
                    check=False,
                )
                self.assertEqual(result.returncode, 1)

    def test_mutation_workflow_is_pinned_linux_evidence_gate(self) -> None:
        workflow = MUTATION_WORKFLOW.read_text(encoding="utf-8")
        for required in (
            "runs-on: ubuntu-latest",
            'python-version: "3.13"',
            "workflow_dispatch:",
            "pull_request:",
            "types: [published]",
            "MUTMUT_WHEEL_SHA256:",
            "MUTMUT_WHEEL: mutmut-3.7.0-py3-none-any.whl",
            "1d2f9a1bfa4a474b2213df6b17223150b492bf4a85af0eda4fb322297337fb32",
            "importlib.metadata.version('mutmut')",
            "--index-url https://pypi.org/simple",
            "python -m pip install '.[test,docs]'",
            "--mutmut-wheel",
            "--logs-dir",
            "--logs-root",
            "--mutants-root",
            "if: always()",
            "python/mutants/**/*.meta",
            "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683",
            "ref: ${{ github.event.pull_request.head.sha || github.sha }}",
            "fetch-depth: 0",
            "persist-credentials: false",
            "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
        ):
            with self.subTest(required=required):
                self.assertIn(required, workflow)


if __name__ == "__main__":
    unittest.main()
