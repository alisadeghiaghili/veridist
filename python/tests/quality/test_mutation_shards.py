"""Contracts for the mutation shard map, the mutmut selection overlay and the shard workflow."""

from __future__ import annotations

import fnmatch
import json
import platform
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest import mock

import yaml

PYTHON_ROOT = Path(__file__).parents[2]
REPOSITORY_ROOT = PYTHON_ROOT.parent
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "mutation.yml"
sys.path.insert(0, str(PYTHON_ROOT / "tools"))
import mutation_evidence  # noqa: E402
import run_mutation  # noqa: E402

from tests.quality.test_mutation_evidence import (  # noqa: E402
    FIXTURE_SHARDS,
    PYPROJECT,
    write_manifest,
    write_tree,
)

CRITICAL = mutation_evidence.CRITICAL_MODULES


def production_files() -> list[str]:
    """Independent enumeration of the critical production files (not the checker's own)."""

    return sorted(
        path.relative_to(PYTHON_ROOT).as_posix()
        for module in CRITICAL
        for path in (PYTHON_ROOT / "src" / "veridist" / module).rglob("*.py")
    )


def declared_shards() -> dict[str, list[str]]:
    manifest = json.loads(
        (PYTHON_ROOT / "quality" / "mutation-manifest.json").read_text(encoding="utf-8")
    )
    return dict(manifest["shards"])


class ShardMapCompletenessTests(unittest.TestCase):
    def test_every_production_file_of_a_critical_module_belongs_to_exactly_one_shard(self) -> None:
        owners: dict[str, list[str]] = {}
        for name, files in declared_shards().items():
            for path in files:
                owners.setdefault(path, []).append(name)
        expected = production_files()
        self.assertGreater(len(expected), 50)
        self.assertEqual(sorted(owners), expected)
        for path, names in owners.items():
            with self.subTest(path=path):
                self.assertEqual(len(names), 1, names)

    def test_the_validator_accepts_the_checked_in_manifest(self) -> None:
        manifest = mutation_evidence.mutation_manifest(PYTHON_ROOT)
        self.assertEqual(manifest["shards"], declared_shards())

    def test_every_critical_module_has_a_shard_and_scale_is_split(self) -> None:
        shards = declared_shards()
        for module in CRITICAL:
            with self.subTest(module=module):
                owning = [
                    name
                    for name, files in shards.items()
                    if any(mutation_evidence.module_of(path) == module for path in files)
                ]
                self.assertGreaterEqual(len(owning), 1)
        scale = [
            name
            for name, files in shards.items()
            if any(mutation_evidence.module_of(path) == "scale" for path in files)
        ]
        self.assertGreaterEqual(len(scale), 2)

    def test_the_other_modules_are_not_mixed_into_scale_shards(self) -> None:
        for name, files in declared_shards().items():
            with self.subTest(shard=name):
                modules = {mutation_evidence.module_of(path) for path in files}
                if name.startswith("scale-"):
                    self.assertEqual(modules, {"scale"})
                else:
                    self.assertEqual(len(modules), 1)

    def test_shard_names_and_files_are_canonical(self) -> None:
        for name, files in declared_shards().items():
            with self.subTest(shard=name):
                self.assertRegex(name, r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
                self.assertEqual(files, sorted(set(files)))
                self.assertFalse(any(set("*?[]\\") & set(path) for path in files))

    def test_an_unassigned_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_tree(root)
            (root / "src/veridist/engine/added.py").write_text("x = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not assigned to a mutation shard"):
                mutation_evidence.mutation_manifest(root)

    def test_a_file_in_two_shards_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_tree(root)
            shards = {**FIXTURE_SHARDS, "scale-b": [*FIXTURE_SHARDS["scale-b"]]}
            shards["scale-b"] = sorted([*shards["scale-b"], *FIXTURE_SHARDS["scale-a"]])
            write_manifest(root, shards=shards)
            with self.assertRaisesRegex(ValueError, "more than one mutation shard"):
                mutation_evidence.mutation_manifest(root)

    def test_malformed_shard_declarations_are_rejected(self) -> None:
        engine = "src/veridist/engine/sample.py"
        others = {name: files for name, files in FIXTURE_SHARDS.items() if name != "engine"}
        bad: dict[str, object] = {
            "empty mapping": {},
            "not a mapping": [engine],
            "empty shard": {**FIXTURE_SHARDS, "engine": []},
            "repeated file": {**FIXTURE_SHARDS, "engine": [engine, engine]},
            "unsorted files": {
                **others,
                "engine": [engine, "src/veridist/engine/another.py"],
            },
            "file outside the critical scope": {
                **FIXTURE_SHARDS,
                "engine": [engine, "src/veridist/adapters/sample.py"],
            },
            "unknown file": {**FIXTURE_SHARDS, "engine": [engine, "src/veridist/engine/x.py"]},
            "glob character": {**others, "engine": ["src/veridist/engine/*.py"]},
            "string instead of list": {**others, "engine": engine},
            "non-string file": {**others, "engine": [engine, 3]},
            "upper-case name": {**others, "Engine": [engine]},
            "name with underscore": {**others, "en_gine": [engine]},
            "name with a trailing dash": {**others, "engine-": [engine]},
            "file removed from every shard": {**others, "engine": []},
        }
        for name, shards in bad.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                write_tree(root)
                write_manifest(root, shards=shards)
                with self.assertRaises(ValueError):
                    mutation_evidence.mutation_manifest(root)

    def test_a_manifest_without_the_shard_map_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_tree(root)
            write_manifest(root, shards=None)
            with self.assertRaisesRegex(ValueError, "invalid mutation manifest schema"):
                mutation_evidence.mutation_manifest(root)

    def test_duplicate_shard_names_in_the_manifest_json_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_tree(root)
            path = root / "quality" / "mutation-manifest.json"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace('"shards": {', '"shards": {"domain": ["x"], "domain": ["y"], ', 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                mutation_evidence.mutation_manifest(root)

    def test_module_of_names_the_owning_module(self) -> None:
        self.assertEqual(mutation_evidence.module_of("src/veridist/scale/_fit.py"), "scale")
        self.assertEqual(mutation_evidence.module_of("src/veridist/domain/__init__.py"), "domain")
        with self.assertRaises(ValueError):
            mutation_evidence.module_of("tools/run_mutation.py")


class OnlyMutateOverlayTests(unittest.TestCase):
    def test_the_overlay_adds_only_the_selection_to_the_mutmut_table(self) -> None:
        files = ["src/veridist/scale/_fit.py", "src/veridist/scale/_planner.py"]
        text = mutation_evidence.overlay_only_mutate(PYPROJECT, files)
        base = tomllib.loads(PYPROJECT)
        merged = tomllib.loads(text)
        self.assertEqual(merged["tool"]["mutmut"]["only_mutate"], files)
        del merged["tool"]["mutmut"]["only_mutate"]
        self.assertEqual(merged, base)

    def test_the_overlay_of_the_real_pyproject_equals_the_shard_configuration(self) -> None:
        pyproject = (PYTHON_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        files = declared_shards()["scale-fit"]
        merged = tomllib.loads(mutation_evidence.overlay_only_mutate(pyproject, files))
        config = mutation_evidence.mutation_config(PYTHON_ROOT)
        self.assertEqual(merged["tool"]["mutmut"], mutation_evidence.shard_config(config, files))
        # The rest of the project metadata is untouched.
        original = tomllib.loads(pyproject)
        del merged["tool"]["mutmut"]
        del original["tool"]["mutmut"]
        self.assertEqual(merged, original)

    def test_the_overlay_keeps_crlf_line_endings(self) -> None:
        text = PYPROJECT.replace("\n", "\r\n")
        result = mutation_evidence.overlay_only_mutate(text, ["src/veridist/domain/sample.py"])
        self.assertNotIn("\n", result.replace("\r\n", ""))
        self.assertIn("only_mutate", tomllib.loads(result)["tool"]["mutmut"])

    def test_the_overlay_refuses_ambiguous_or_preconfigured_tables(self) -> None:
        files = ["src/veridist/domain/sample.py"]
        for name, text in (
            ("no table", "[tool.ruff]\nline-length = 100\n"),
            ("two tables", PYPROJECT + "\n[tool.mutmut]\n"),
            ("already selected", PYPROJECT + 'only_mutate = ["src/veridist/domain/x.py"]\n'),
            ("header at the end", "[tool.mutmut]"),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                mutation_evidence.overlay_only_mutate(text, files)

    def test_the_tracked_configuration_may_not_carry_the_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_text(
                mutation_evidence.overlay_only_mutate(PYPROJECT, ["src/veridist/domain/x.py"]),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid mutmut configuration"):
                mutation_evidence.mutation_config(root)

    def test_the_selection_changes_the_effective_digest_per_shard_only(self) -> None:
        base = mutation_evidence.config_digest(PYTHON_ROOT)
        digests = {
            name: mutation_evidence.shard_config_digest(PYTHON_ROOT, files)
            for name, files in declared_shards().items()
        }
        self.assertEqual(len(set(digests.values())), len(digests))
        self.assertNotIn(base, digests.values())

    def test_the_selection_patterns_match_exactly_the_shard_files_under_fnmatch(self) -> None:
        # mutmut 3.7 selects with fnmatch.fnmatch(str(path), pattern) over every source file.
        shards = declared_shards()
        for name, files in shards.items():
            with self.subTest(shard=name):
                matched = {
                    path
                    for path in production_files()
                    if any(fnmatch.fnmatch(path, pattern) for pattern in files)
                }
                self.assertEqual(matched, set(files))


class RunnerShardTests(unittest.TestCase):
    def test_the_overlay_is_present_only_while_mutmut_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_bytes(PYPROJECT.encode("utf-8"))
            files = ["src/veridist/domain/sample.py"]
            seen: dict[str, object] = {}

            def fake(command: list[str], *args: object) -> dict[str, object]:
                text = (root / "pyproject.toml").read_text(encoding="utf-8")
                seen["only_mutate"] = tomllib.loads(text)["tool"]["mutmut"]["only_mutate"]
                seen["command"] = command
                return {"state": "passed"}

            with mock.patch.object(run_mutation, "run_logged", fake):
                result = run_mutation.run_shard_mutation(root, root / "logs", files)
            self.assertEqual(result, {"state": "passed"})
            self.assertEqual(seen["only_mutate"], files)
            self.assertEqual(seen["command"], ["mutmut", "run"])
            self.assertEqual((root / "pyproject.toml").read_bytes(), PYPROJECT.encode("utf-8"))

    def test_the_tracked_configuration_is_restored_when_mutmut_cannot_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "pyproject.toml").write_bytes(PYPROJECT.encode("utf-8"))

            def boom(*args: object) -> dict[str, object]:
                raise RuntimeError("mutmut exploded")

            with mock.patch.object(run_mutation, "run_logged", boom):
                with self.assertRaises(RuntimeError):
                    run_mutation.run_shard_mutation(root, root / "logs", ["src/veridist/a.py"])
            self.assertEqual((root / "pyproject.toml").read_bytes(), PYPROJECT.encode("utf-8"))

    def test_an_unknown_shard_is_refused_before_anything_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            wheel = Path(temporary) / "mutmut.whl"
            wheel.write_bytes(b"not a wheel")
            argv = [
                "run_mutation.py",
                "--project-root",
                str(PYTHON_ROOT),
                "--shard",
                "no-such-shard",
                "--output",
                str(Path(temporary) / "evidence.json"),
                "--mutmut-wheel",
                str(wheel),
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(sys, "platform", "linux"),
                mock.patch.object(platform, "system", lambda: "Linux"),
                mock.patch.object(run_mutation, "run_logged") as logged,
                mock.patch.object(sys, "stderr") as stderr,
            ):
                code = run_mutation.main()
            self.assertEqual(code, 1)
            logged.assert_not_called()
            written = "".join(call.args[0] for call in stderr.write.call_args_list)
            self.assertIn("unknown mutation shard: no-such-shard", written)
            self.assertFalse((Path(temporary) / "evidence.json").exists())

    def test_the_shard_argument_is_required(self) -> None:
        argv = ["run_mutation.py", "--output", "e.json"]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(sys, "stderr"):
            with self.assertRaises(SystemExit) as raised:
                run_mutation.main()
        self.assertEqual(raised.exception.code, 2)


class ShardWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = WORKFLOW.read_text(encoding="utf-8")
        cls.document = yaml.safe_load(cls.text)
        cls.jobs = cls.document["jobs"]

    def test_the_workflow_keeps_exactly_the_shard_job_and_the_required_gate(self) -> None:
        self.assertEqual(set(self.jobs), {"mutation", "mutation-gate"})
        self.assertEqual(self.jobs["mutation-gate"]["name"], "mutation / gate")
        self.assertEqual(self.jobs["mutation-gate"]["needs"], ["mutation"])
        self.assertEqual(self.jobs["mutation-gate"]["if"], "always()")
        self.assertNotIn("continue-on-error", self.text)
        self.assertNotIn("|| true", self.text)

    def test_the_matrix_lists_exactly_the_manifest_shards(self) -> None:
        matrix = self.jobs["mutation"]["strategy"]["matrix"]["shard"]
        self.assertEqual(len(matrix), len(set(matrix)))
        self.assertEqual(set(matrix), set(declared_shards()))
        self.assertIs(self.jobs["mutation"]["strategy"]["fail-fast"], False)
        self.assertEqual(list(self.jobs["mutation"]["strategy"]["matrix"]), ["shard"])

    def test_every_shard_job_is_bounded_and_runs_its_own_shard(self) -> None:
        job = self.jobs["mutation"]
        self.assertEqual(job["timeout-minutes"], 90)
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        self.assertEqual(job["env"]["SHARD"], "${{ matrix.shard }}")
        self.assertIn("${{ matrix.shard }}", job["name"])
        runs = "\n".join(step.get("run", "") for step in job["steps"])
        self.assertIn('--shard "$SHARD"', runs)
        self.assertEqual(runs.count('--shard "$SHARD"'), 2)
        self.assertIn("tools/run_mutation.py", runs)
        self.assertIn("tools/check_mutation_evidence.py", runs)
        self.assertNotIn("${{ matrix", runs)

    def test_each_shard_uploads_evidence_logs_and_raw_cache_even_when_it_fails(self) -> None:
        steps = {step["name"]: step for step in self.jobs["mutation"]["steps"]}
        stage = steps["Stage the raw mutmut cache beside the shard evidence"]
        upload = steps["Retain shard evidence, logs and raw cache"]
        self.assertEqual(stage["if"], "always()")
        self.assertIn("*.meta", stage["run"])
        self.assertEqual(upload["if"], "always()")
        self.assertEqual(
            upload["with"]["name"],
            "mutation-shard-${{ matrix.shard }}-${{ github.run_id }}-${{ github.run_attempt }}",
        )
        self.assertEqual(upload["with"]["path"], "${{ runner.temp }}/mutation-shard")
        run = steps["Run the fail-closed mutation shard"]["run"]
        self.assertIn('bundle="$RUNNER_TEMP/mutation-shard"', run)
        for fragment in (
            '--output "$bundle/mutation-evidence.json"',
            '--logs-dir "$bundle/logs"',
            '--evidence "$bundle/mutation-evidence.json"',
            '--logs-root "$bundle/logs"',
            '--mutants-root "$GITHUB_WORKSPACE/python/mutants"',
        ):
            self.assertIn(fragment, run)

    def test_the_gate_merges_every_downloaded_shard_of_this_run_attempt(self) -> None:
        steps = {step["name"]: step for step in self.jobs["mutation-gate"]["steps"]}
        download = steps["Download every shard artifact of this run attempt"]
        pattern = download["with"]["pattern"]
        self.assertEqual(pattern, "mutation-shard-*-${{ github.run_id }}-${{ github.run_attempt }}")
        self.assertNotIn("merge-multiple", download["with"])
        # The upload names of every shard must be matched by the download pattern.
        upload = steps_by_name(self.jobs["mutation"])["Retain shard evidence, logs and raw cache"]
        template = upload["with"]["name"]
        for shard in declared_shards():
            concrete = template.replace("${{ matrix.shard }}", shard).replace(
                "${{ github.run_id }}-${{ github.run_attempt }}", "7-1"
            )
            self.assertTrue(
                fnmatch.fnmatch(
                    concrete,
                    pattern.replace("${{ github.run_id }}-${{ github.run_attempt }}", "7-1"),
                ),
                concrete,
            )
        merge = steps["Merge the shard evidence and enforce the floors"]["run"]
        self.assertIn("tools/merge_mutation_evidence.py", merge)
        self.assertIn('--shards-root "$RUNNER_TEMP/mutation-shards"', merge)
        self.assertEqual(download["with"]["path"], "${{ runner.temp }}/mutation-shards")

    def test_the_gate_fails_closed_before_it_looks_at_any_artifact(self) -> None:
        first = self.jobs["mutation-gate"]["steps"][0]
        self.assertEqual(first["env"]["MUTATION_RESULT"], "${{ needs.mutation.result }}")
        self.assertIn('"$MUTATION_RESULT" != "success"', first["run"])
        self.assertIn("exit 1", first["run"])
        for step in self.jobs["mutation-gate"]["steps"][1:-1]:
            self.assertNotIn("if", step)

    def test_the_gate_checks_out_the_same_revision_as_the_shards(self) -> None:
        revisions = []
        for job in ("mutation", "mutation-gate"):
            checkout = next(
                step
                for step in self.jobs[job]["steps"]
                if step.get("uses", "").startswith("actions/checkout@")
            )
            revisions.append(checkout["with"])
        self.assertEqual(revisions[0], revisions[1])
        self.assertEqual(
            revisions[0]["ref"], "${{ github.event.pull_request.head.sha || github.sha }}"
        )
        self.assertFalse(revisions[0]["persist-credentials"])

    def test_the_pinned_toolchain_is_unchanged(self) -> None:
        env = self.jobs["mutation"]["env"]
        self.assertEqual(env["MUTMUT_VERSION"], "3.7.0")
        self.assertEqual(env["MUTMUT_WHEEL"], "mutmut-3.7.0-py3-none-any.whl")
        self.assertEqual(env["MUTMUT_WHEEL_SHA256"], mutation_evidence.MUTMUT_WHEEL_SHA256)
        for job in ("mutation", "mutation-gate"):
            setup = next(
                step
                for step in self.jobs[job]["steps"]
                if step.get("uses", "").startswith("actions/setup-python@")
            )
            self.assertEqual(setup["with"]["python-version"], "3.13")
        self.assertRegex(self.text, r"actions/download-artifact@[0-9a-f]{40} # v4\.3\.0")

    def test_pytest_is_pinned_together_with_mutmut_in_the_workflow_and_the_extra(self) -> None:
        env = self.jobs["mutation"]["env"]
        self.assertEqual(env["PYTEST_VERSION"], "9.1.1")
        install = next(
            step
            for step in self.jobs["mutation"]["steps"]
            if step["name"].startswith("Install the project test dependencies")
        )
        self.assertIn('"$RUNNER_TEMP/$MUTMUT_WHEEL" "pytest==$PYTEST_VERSION"', install["run"])
        self.assertIn(
            "importlib.metadata.version('pytest') == '9.1.1'",
            install["run"],
        )
        extra = tomllib.loads((PYTHON_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]["optional-dependencies"]["mutation"]
        self.assertEqual(sorted(extra), ["mutmut==3.7.0", "pytest==9.1.1"])

    def test_the_workflow_file_is_a_mutation_input_and_the_merge_tool_too(self) -> None:
        self.assertIn(
            "../.github/workflows/mutation.yml", mutation_evidence.REQUIRED_NON_TEST_FILES
        )
        self.assertIn("tools/merge_mutation_evidence.py", mutation_evidence.REQUIRED_NON_TEST_FILES)

    def test_other_workflows_still_find_the_job_and_workflow_names_they_depend_on(self) -> None:
        scale = (WORKFLOW.parent / "scale-evidence.yml").read_text(encoding="utf-8")
        self.assertIn('"mutation / gate"', scale)
        publish = (WORKFLOW.parent / "pypi-publish.yml").read_text(encoding="utf-8")
        self.assertRegex(publish, r"for workflow in v1-ci\.yml mutation\.yml; do")
        self.assertIn("workflow_dispatch", self.document[True])


def steps_by_name(job: dict[str, object]) -> dict[str, dict[str, object]]:
    return {step["name"]: step for step in job["steps"]}  # type: ignore[index, attr-defined]


if __name__ == "__main__":
    unittest.main()
