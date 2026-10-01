"""Red contracts for formal mutation evidence schema v2 fail-closed behavior."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from json import loads
from os import environ
from pathlib import Path
from shutil import copy2, copytree

PYTHON_ROOT = Path(__file__).parents[2]
CHECKER = PYTHON_ROOT / "tools" / "check_mutation_evidence.py"


class MutationEvidenceV2Contracts(unittest.TestCase):
    def test_mutation_config_partitions_every_package_file(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import mutation_evidence

        config = mutation_evidence.mutation_config(PYTHON_ROOT)
        package = PYTHON_ROOT / "src" / "veridist"

        def expand(entries: list[str]) -> set[Path]:
            files: set[Path] = set()
            for entry in entries:
                target = PYTHON_ROOT / entry
                if target.is_dir():
                    files.update(
                        path
                        for path in target.rglob("*")
                        if path.is_file()
                        and "__pycache__" not in path.parts
                        and path.suffix != ".pyc"
                    )
                else:
                    files.add(target)
            return files

        all_files = {
            path
            for path in package.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        mutated = expand(config["source_paths"])
        copied = expand([path for path in config["also_copy"] if path != "tools"])
        self.assertFalse(mutated & copied)
        self.assertEqual(mutated | copied, all_files)

    def test_mutant_tree_shadows_installed_veridist_package(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import mutation_evidence

        config = mutation_evidence.mutation_config(PYTHON_ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mutant_root = root / "mutants"

            def copy_entry(entry: str) -> None:
                source = PYTHON_ROOT / entry
                destination = mutant_root / entry
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_dir():
                    copytree(source, destination)
                else:
                    copy2(source, destination)

            for entry in config["source_paths"]:
                copy_entry(entry)
            for entry in config["also_copy"]:
                if entry != "tools":
                    copy_entry(entry)
            script = (
                "import importlib, json; names=['veridist','veridist.domain',"
                "'veridist.statistics','veridist.families','veridist.engine',"
                "'veridist.adapters','veridist.reporting','veridist.execution']; "
                "files={name: importlib.import_module(name).__file__ for name in names}; "
                "print(json.dumps(files))"
            )
            environment = dict(environ)
            environment["PYTHONPATH"] = str(mutant_root / "src")
            result = subprocess.run(
                [sys.executable, "-c", script],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            files = loads(result.stdout)
            for path in files.values():
                self.assertTrue(Path(path).is_relative_to(mutant_root / "src"))

    def test_checker_exposes_duplicate_and_nonfinite_safe_json_loader(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import check_mutation_evidence

        for source in ('{"x": 1, "x": 2}', '{"x": NaN}', '{"x": Infinity}'):
            with self.subTest(source=source):
                with self.assertRaises(ValueError):
                    check_mutation_evidence.load_json(source)

    def test_schema_v2_is_explicit_and_rejects_legacy_self_consistent_evidence(self) -> None:
        self.assertIn("SCHEMA_VERSION = 2", CHECKER.read_text(encoding="utf-8"))

    def test_official_mutmut_status_mapping_is_complete_and_not_caller_controlled(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import mutation_evidence

        expected = {
            0: "survived",
            1: "killed",
            3: "killed",
            5: "no_tests",
            36: "timeout",
            37: "type_check",
            -11: "segfault",
            None: "not_checked",
        }
        for code, status in expected.items():
            with self.subTest(code=code):
                self.assertEqual(mutation_evidence.official_status(code), status)
        self.assertEqual(mutation_evidence.scoring_status(0), "survived")
        self.assertEqual(mutation_evidence.scoring_status(37), "killed")

    def test_checker_requires_raw_cache_for_non_fixture_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text("[tool.mutmut]\n", encoding="utf-8")
            evidence = root / "evidence.json"
            evidence.write_text("{}", encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--project-root",
                    str(root),
                    "--evidence",
                    str(evidence),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("--mutants-root", result.stderr)

    def test_runner_provenance_contracts_are_present(self) -> None:
        runner = (PYTHON_ROOT / "tools" / "run_mutation.py").read_text(encoding="utf-8")
        required_terms = (
            "--mutmut-wheel",
            "git status --porcelain",
            "input_digest",
            "baseline",
            "mutation",
            '"also_copy": [',
            '"src/veridist/reporting",',
            '"pytest_selection": list(MUTATION_TEST_SELECTION)',
        )
        for required in required_terms:
            with self.subTest(required=required):
                self.assertIn(required, runner)

    def test_raw_cache_validation_requires_exact_meta_schema(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import check_mutation_evidence

        source = "src/veridist/domain/example.py"
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            target = cache / f"{source}.meta"
            target.parent.mkdir(parents=True)
            target.write_text('{"exit_code_by_key": {}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                check_mutation_evidence.file_meta(cache, source)

    def test_raw_cache_rejects_identity_relations_outside_exit_codes(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import check_mutation_evidence

        source = "src/veridist/domain/example.py"
        payload = {
            "exit_code_by_key": {"mutant": 1},
            "hash_by_function_name": {},
            "type_check_error_by_key": {"fabricated": "error"},
            "durations_by_key": {},
            "estimated_durations_by_key": {},
        }
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            target = cache / f"{source}.meta"
            target.parent.mkdir(parents=True)
            target.write_text(__import__("json").dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                check_mutation_evidence.file_meta(cache, source)

    def test_unknown_mutmut_exit_code_is_suspicious(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import mutation_evidence

        self.assertEqual(mutation_evidence.official_status(999), "suspicious")

    def test_mutmut_configuration_requires_exact_tools_copy_and_rejects_extra_options(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import mutation_evidence

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            configuration = (
                "[tool.mutmut]\n"
                'source_paths = ["src/veridist/domain", "src/veridist/statistics", '
                '"src/veridist/families", "src/veridist/engine"]\n'
                'pytest_add_cli_args_test_selection = ["tests/contract", '
                '"tests/reference", "tests/unit", "tests/conformance", "tests/property"]\n'
                "mutate_only_covered_lines = false\n"
            )
            copy = (
                '["tools", "src/veridist/__init__.py", "src/veridist/execution.py", '
                '"src/veridist/inference.py", "src/veridist/py.typed", '
                '"src/veridist/adapters", "src/veridist/reporting"]'
            )
            root.joinpath("pyproject.toml").write_text(
                configuration + f"also_copy = {copy}\n",
                encoding="utf-8",
            )
            self.assertEqual(
                mutation_evidence.mutation_config(root)["also_copy"],
                [
                    "tools",
                    "src/veridist/__init__.py",
                    "src/veridist/execution.py",
                    "src/veridist/inference.py",
                    "src/veridist/py.typed",
                    "src/veridist/adapters",
                    "src/veridist/reporting",
                ],
            )
            self.assertEqual(
                mutation_evidence.mutation_config(root)["pytest_add_cli_args_test_selection"],
                [
                    "tests/contract",
                    "tests/reference",
                    "tests/unit",
                    "tests/conformance",
                    "tests/property",
                ],
            )
            for selection in (
                '["tests"]',
                '["tests/reference", "tests/contract", "tests/unit"]',
                '["tests/contract", "tests/reference", "tests/unit", "tests/scale"]',
            ):
                with self.subTest(selection=selection):
                    root.joinpath("pyproject.toml").write_text(
                        configuration.replace(
                            '["tests/contract", "tests/reference", "tests/unit", '
                            '"tests/conformance", "tests/property"]',
                            selection,
                        )
                        + f"also_copy = {copy}\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValueError):
                        mutation_evidence.mutation_config(root)
            for value in ('[]', '["tools"]', copy + ','):
                with self.subTest(value=value):
                    root.joinpath("pyproject.toml").write_text(
                        configuration + f"also_copy = {value}\n", encoding="utf-8"
                    )
                    with self.assertRaises(ValueError):
                        mutation_evidence.mutation_config(root)
            root.joinpath("pyproject.toml").write_text(
                configuration + f"also_copy = {copy}\nalso_mutate = [\"src/ignored\"]\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                mutation_evidence.mutation_config(root)

    def test_checker_rejects_non_utc_backward_or_invalid_phase_records(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        from tests.quality import test_mutation_evidence

        for field, value in (
            ("started_at", "2026-01-01T00:00:00+00:00"),
            ("ended_at", "2025-12-31T23:59:59Z"),
            ("missing_state", None),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payload = test_mutation_evidence.fixture(root)
                if field == "missing_state":
                    payload["baseline"].pop("state", None)
                else:
                    payload["baseline"][field] = value
                self.assertNotEqual(test_mutation_evidence.check(root, payload).returncode, 0)

    def test_checker_requires_logs_root_for_non_fixture_evidence(self) -> None:
        checker = CHECKER.read_text(encoding="utf-8")
        self.assertIn("--logs-root", checker)

    def test_input_manifest_is_tracked_and_excludes_cache_artifacts(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import mutation_evidence

        files = mutation_evidence.input_files(PYTHON_ROOT)
        self.assertNotIn(".pytest_cache/example.pyc", files)
        self.assertNotIn("mutants/example.meta", files)
        self.assertTrue(all("__pycache__" not in item for item in files))

    def test_input_files_raises_when_a_required_non_test_file_is_missing(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import mutation_evidence

        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            root = repository / "python"
            for module in mutation_evidence.CRITICAL_MODULES:
                target = root / "src" / "veridist" / module / "sample.py"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("value = 1\n", encoding="utf-8")
            required = [
                "tests/test_sample.py",
                "pyproject.toml",
                "quality/mutation-manifest.json",
                "tools/mutation_evidence.py",
                "tools/run_mutation.py",
                "tools/check_mutation_evidence.py",
            ]
            for relative in required:
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x\n", encoding="utf-8")
            workflow = repository / ".github" / "workflows" / "mutation.yml"
            workflow.parent.mkdir(parents=True)
            workflow.write_text("name: a\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(["git", "add", "."], cwd=repository, check=True)

            # Confirm the fully-populated tree is accepted before removing a
            # required non-test file.
            mutation_evidence.input_files(root)

            workflow.unlink()
            with self.assertRaisesRegex(ValueError, "mutation input file is missing"):
                mutation_evidence.input_files(root)

    def test_portable_runner_commands_and_relative_logs_are_contractual(self) -> None:
        runner = (PYTHON_ROOT / "tools" / "run_mutation.py").read_text(encoding="utf-8")
        self.assertIn('["python", "-m", "pytest", "tests"]', runner)
        self.assertIn('["mutmut", "run"]', runner)
        self.assertIn("relative_to(logs_root)", runner)

    def test_runner_handles_not_run_mutation_without_exit_code_access(self) -> None:
        runner = (PYTHON_ROOT / "tools" / "run_mutation.py").read_text(encoding="utf-8")
        self.assertIn('baseline["state"] == "passed"', runner)
        self.assertNotIn('baseline["exit_code"] == mutation["exit_code"]', runner)

    def test_tracked_workflow_byte_changes_input_digest(self) -> None:
        sys.path.insert(0, str(PYTHON_ROOT / "tools"))
        import mutation_evidence

        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            root = repository / "python"
            for module in mutation_evidence.CRITICAL_MODULES:
                target = root / "src" / "veridist" / module / "sample.py"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("value = 1\n", encoding="utf-8")
            for relative in (
                "tests/test_sample.py",
                "pyproject.toml",
                "quality/mutation-manifest.json",
                "tools/mutation_evidence.py",
                "tools/run_mutation.py",
                "tools/check_mutation_evidence.py",
            ):
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x\n", encoding="utf-8")
            workflow = repository / ".github" / "workflows" / "mutation.yml"
            workflow.parent.mkdir(parents=True)
            workflow.write_text("name: a\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(["git", "add", "."], cwd=repository, check=True)
            _, first = mutation_evidence.input_digest(root)
            workflow.write_text("name: b\n", encoding="utf-8")
            files, second = mutation_evidence.input_digest(root)
            self.assertIn("../.github/workflows/mutation.yml", files)
            self.assertNotEqual(first, second)
