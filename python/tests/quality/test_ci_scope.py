"""Behavioral contracts for CI lane routing and aggregate gating."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PYTHON_ROOT.parent
CLASSIFIER_PATH = PYTHON_ROOT / "tools" / "ci_scope.py"
LEGACY_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"


def load_classifier():
    spec = importlib.util.spec_from_file_location("veridist_ci_scope", CLASSIFIER_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("CI scope classifier cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CiScopeClassifierTests(unittest.TestCase):
    def test_veridist_only_paths_do_not_run_legacy_product_tests(self) -> None:
        classifier = load_classifier()
        paths = (
            "python/src/veridist/engine/errors.py",
            "python/tests/quality/test_ci_scope.py",
            "docs/adr/ADR-0015-retry-checkpoint-transactional-guarantees.md",
            "docs/evidence/scale-csv-exponential-v1.md",
            "docs/capability-matrix.md",
            "docs/capability-matrix.fa.md",
            "docs/capability-matrix.de.md",
            "docs/readme-design-policy.md",
            "docs/citing-veridist.md",
            "docs/v1-readiness.md",
            "docs/decisions-2026-08-20.md",
            "docs/migration/legacy-salvage-ledger.json",
            "README.md",
            "CHANGELOG.md",
            "README.fa.md",
            "README.de.md",
            "SECURITY.md",
            ".zenodo.json",
            "CITATION.cff",
            "conda-forge-recipe/meta.yaml",
            ".github/workflows/mutation.yml",
            ".github/workflows/scale-evidence.yml",
            ".github/workflows/v1-ci.yml",
            ".github/workflows/v1-release-evidence.yml",
            ".github/workflows/pypi-publish.yml",
            ".github/workflows/ci.yml",
        )
        self.assertFalse(classifier.legacy_relevant(paths))

    def test_legacy_shared_mixed_and_unknown_paths_fail_closed(self) -> None:
        classifier = load_classifier()
        relevant_paths = (
            "distfit_pro/core.py",
            "tests/test_core.py",
            "examples/basic.py",
            "docs/source/index.rst",
            "docs/user_guide/01_getting_started.md",
            "docs/api/index.md",
            "docs/faq.md",
            "docs/unclassified.md",
            "docs/capability-matrix-draft.md",
            "docs/capability-matrix.md.bak",
            "docs/capability-matrices.md",
            "pyproject.toml",
            "LICENSE",
            ".gitignore",
            "new-root-area/file.txt",
        )
        for path in relevant_paths:
            with self.subTest(path=path):
                self.assertTrue(classifier.legacy_relevant((path,)))
        self.assertTrue(classifier.legacy_relevant(("python/src/veridist/__init__.py", "setup.py")))
        self.assertTrue(classifier.legacy_relevant(()))

    def test_classifier_accepts_unicode_spaces_and_inert_shell_metacharacters(self) -> None:
        classifier = load_classifier()
        self.assertFalse(
            classifier.legacy_relevant(
                (
                    "python/tests/data/file with spaces.py",
                    "docs/adr/تصمیم-$GITHUB_OUTPUT.md",
                )
            )
        )

    def test_classifier_rejects_non_repository_paths(self) -> None:
        classifier = load_classifier()
        malformed = ("", "/absolute/path", "../escape", "python\\file.py", "docs/./file.md")
        for path in malformed:
            with self.subTest(path=path), self.assertRaises(ValueError):
                classifier.legacy_relevant((path,))

    def test_nul_decoder_requires_a_complete_unambiguous_stream(self) -> None:
        classifier = load_classifier()
        self.assertEqual(
            classifier.decode_nul_paths(b"python/a.py\0docs/adr/x.md\0"),
            ("python/a.py", "docs/adr/x.md"),
        )
        for payload in (b"python/a.py", b"python/a.py\0\0", b"python/\xff\0"):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                classifier.decode_nul_paths(payload)

    def test_cli_emits_boolean_tokens_and_uses_nonzero_only_for_errors(self) -> None:
        cases = (
            (b"python/a.py\0", "false"),
            (b"distfit_pro/a.py\0", "true"),
            (b"", "true"),
        )
        for payload, expected in cases:
            with self.subTest(expected=expected):
                result = subprocess.run(
                    [sys.executable, str(CLASSIFIER_PATH), "classify", "--nul"],
                    input=payload,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr.decode())
                self.assertEqual(result.stdout.decode().strip(), expected)

        malformed = subprocess.run(
            [sys.executable, str(CLASSIFIER_PATH), "classify", "--nul"],
            input=b"unterminated",
            capture_output=True,
            check=False,
        )
        self.assertNotEqual(malformed.returncode, 0)

    def test_legacy_gate_is_fail_closed_for_every_unexpected_state(self) -> None:
        classifier = load_classifier()
        self.assertTrue(classifier.legacy_gate_allows(False, "success", "skipped"))
        self.assertTrue(classifier.legacy_gate_allows(True, "success", "success"))
        for relevant, scope_result, test_result in (
            (False, "failure", "skipped"),
            (False, "success", "success"),
            (True, "success", "skipped"),
            (True, "success", "failure"),
            (True, "cancelled", "success"),
            (True, "success", "neutral"),
        ):
            with self.subTest(
                relevant=relevant, scope_result=scope_result, test_result=test_result
            ):
                self.assertFalse(classifier.legacy_gate_allows(relevant, scope_result, test_result))

    def test_legacy_gate_cli_propagates_valid_and_invalid_terminal_states(self) -> None:
        cases = (
            ("false", "success", "skipped", 0),
            ("true", "success", "success", 0),
            ("true", "success", "failure", 1),
            ("false", "failure", "skipped", 1),
            ("unknown", "success", "skipped", 2),
        )
        for relevant, scope_result, test_result, expected in cases:
            with self.subTest(
                relevant=relevant, scope_result=scope_result, test_result=test_result
            ):
                result = subprocess.run(
                    [
                        sys.executable,
                        str(CLASSIFIER_PATH),
                        "legacy-gate",
                        "--relevant",
                        relevant,
                        "--scope-result",
                        scope_result,
                        "--test-result",
                        test_result,
                    ],
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, expected)

    def test_legacy_workflow_has_an_explicit_always_run_aggregate_gate(self) -> None:
        workflow = LEGACY_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("name: legacy-ci", workflow)
        self.assertIn("legacy-scope:", workflow)
        self.assertIn("legacy-test:", workflow)
        self.assertIn("legacy-gate:", workflow)
        self.assertIn("name: legacy / gate", workflow)
        scope = workflow.split("  legacy-test:", maxsplit=1)[0]
        self.assertIn("uses: actions/setup-python@v5", scope)
        self.assertIn('python-version: "3.11"', scope)
        self.assertIn("if: always()", workflow)
        self.assertIn("python python/tools/ci_scope.py classify --nul", workflow)
        self.assertIn("--no-renames", workflow)
        self.assertIn("--diff-filter=ACMRTD", workflow)
        self.assertIn("^[0-9a-fA-F]{40}$", workflow)
        self.assertIn("branches: [main, develop]", workflow)
        self.assertNotIn('"release"', workflow)
        self.assertIn('"workflow_dispatch"', workflow)
        self.assertIn('"0000000000000000000000000000000000000000"', workflow)
        self.assertNotIn("paths:", workflow)
        self.assertNotIn("continue-on-error", workflow)
        self.assertNotIn("|| true", workflow)


if __name__ == "__main__":
    unittest.main()
