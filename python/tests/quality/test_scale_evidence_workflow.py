"""Contract tests for the manually dispatched, candidate-bound scale gate."""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "scale-evidence.yml"
MUTATION_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "mutation.yml"
TOOLS = REPOSITORY_ROOT / "python" / "tools"


class ScaleEvidenceWorkflowTests(unittest.TestCase):
    def test_candidate_mutation_workflow_publishes_the_required_gate(self) -> None:
        workflow = MUTATION_WORKFLOW.read_text(encoding="utf-8")
        for required in (
            "mutation-gate:",
            "name: mutation / gate",
            "needs: [mutation]",
            "if: always()",
            "MUTATION_RESULT: ${{ needs.mutation.result }}",
            '"$MUTATION_RESULT" != "success"',
        ):
            with self.subTest(required=required):
                self.assertIn(required, workflow)

    def test_gate_is_manual_candidate_bound_and_cross_platform(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        for required in (
            "name: veridist-scale-evidence",
            "workflow_dispatch:",
            "candidate_sha:",
            "required: true",
            "ubuntu-latest",
            "windows-latest",
            "ref: ${{ inputs.candidate_sha }}",
            "fetch-depth: 0",
            "persist-credentials: false",
            "candidate SHA must be a full lowercase Git commit",
            "checked-out candidate SHA differs from requested candidate SHA",
            "refusing scale evidence from a dirty checkout",
            "Verify required candidate checks",
            "GH_TOKEN: ${{ github.token }}",
            "required candidate checks did not pass",
            "veridist / gate",
            "legacy / gate",
            "mutation / gate",
        ):
            with self.subTest(required=required):
                self.assertIn(required, workflow)

    def test_evidence_jobs_are_preflight_gated_and_fail_closed(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("needs: [preflight]", workflow)
        self.assertIn("needs: [scale-evidence, scale-engine-evidence]", workflow)
        self.assertNotIn("continue-on-error", workflow)
        self.assertNotIn("|| true", workflow)
        for required in (
            "tests.scale.test_log_likelihood_scale_evidence",
            "tests.scale.test_csv_exponential_evidence",
            "run_log_likelihood_scale_evidence.py",
            "check_log_likelihood_scale_evidence.py",
            "run_scale_csv_exponential_evidence.py",
            "check_scale_csv_exponential_evidence.py",
            "--expected-git-sha \"${{ needs.preflight.outputs.candidate_sha }}\"",
            "--repo-root ..",
            "1m public-stream likelihood evidence",
            "if: always()",
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2",
        ):
            with self.subTest(required=required):
                self.assertIn(required, workflow)

    def test_artifact_names_are_platform_and_candidate_specific(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("scale-evidence-${{ matrix.os }}-${{ inputs.candidate_sha }}", workflow)
        self.assertIn("likelihood-scale-evidence.json", workflow)
        self.assertIn("csv-exponential-scale-evidence.json", workflow)
        self.assertIn(
            "scale-engine-evidence-${{ matrix.os }}-${{ inputs.candidate_sha }}", workflow
        )
        self.assertIn("scale-engine-evidence.json", workflow)


def _jobs() -> dict[str, Any]:
    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    jobs: dict[str, Any] = document["jobs"]
    return jobs


def _runs(job: dict[str, Any]) -> list[str]:
    return [str(step["run"]) for step in job["steps"] if "run" in step]


def _step(job: dict[str, Any], fragment: str) -> dict[str, Any]:
    matches = [step for step in job["steps"] if fragment in str(step.get("run", ""))]
    assert len(matches) == 1, fragment
    step: dict[str, Any] = matches[0]
    return step


class ScaleEngineEvidenceWorkflowTests(unittest.TestCase):
    def test_the_job_is_preflight_gated_and_runs_on_both_platforms(self) -> None:
        job = _jobs()["scale-engine-evidence"]
        self.assertEqual(job["needs"], ["preflight"])
        self.assertEqual(job["runs-on"], "${{ matrix.os }}")
        self.assertFalse(job["strategy"]["fail-fast"])
        entries = {entry["os"]: entry for entry in job["strategy"]["matrix"]["include"]}
        self.assertEqual(set(entries), {"ubuntu-latest", "windows-latest"})
        self.assertEqual(job["defaults"]["run"], {"shell": "bash", "working-directory": "python"})

    def test_only_the_linux_ladder_includes_the_top_rung(self) -> None:
        job = _jobs()["scale-engine-evidence"]
        entries = {entry["os"]: entry for entry in job["strategy"]["matrix"]["include"]}
        self.assertEqual(entries["ubuntu-latest"]["top_rung"], "--top-rung")
        self.assertEqual(entries["windows-latest"]["top_rung"], "")
        self.assertEqual(
            [step.get("env", {}).get("TOP_RUNG_FLAG") for step in job["steps"] if "env" in step],
            ["${{ matrix.top_rung }}", "${{ matrix.top_rung }}"],
        )

    def test_timeouts_fit_the_hosted_runner_limit_with_room_to_spare(self) -> None:
        job = _jobs()["scale-engine-evidence"]
        self.assertEqual(job["timeout-minutes"], "${{ matrix.timeout }}")
        entries = {entry["os"]: entry["timeout"] for entry in job["strategy"]["matrix"]["include"]}
        self.assertLessEqual(entries["ubuntu-latest"], 350)
        self.assertGreaterEqual(entries["ubuntu-latest"], 300)
        self.assertLess(entries["windows-latest"], entries["ubuntu-latest"])
        self.assertGreaterEqual(entries["windows-latest"], 180)

    def test_the_runner_and_the_checker_are_wired_with_the_same_ladder_flag(self) -> None:
        job = _jobs()["scale-engine-evidence"]
        runner = _step(job, "run_scale_engine_evidence.py")
        checker = _step(job, "check_scale_engine_evidence.py")
        steps = job["steps"]
        self.assertLess(steps.index(runner), steps.index(checker))
        generated = str(runner["run"]).split()
        self.assertIn("tools/run_scale_engine_evidence.py", generated)
        self.assertIn("--output", generated)
        self.assertIn('--temporary-root "${{ runner.temp }}"', str(runner["run"]))
        self.assertIn("$TOP_RUNG_FLAG", generated)
        validated = str(checker["run"])
        for required in (
            "--artifact",
            "--expected-git-sha",
            '"${{ needs.preflight.outputs.candidate_sha }}"',
            "--repo-root ..",
            "$TOP_RUNG_FLAG",
        ):
            with self.subTest(required=required):
                self.assertIn(required, validated)
        self.assertNotIn("--smoke", validated)
        self.assertNotIn("--smoke", str(runner["run"]))
        output = str(runner["run"]).split("--output")[1].split()[0]
        self.assertEqual(output, str(checker["run"]).split("--artifact")[1].split()[0])

    def test_the_flag_the_workflow_passes_is_one_the_tools_accept(self) -> None:
        for name in ("run_scale_engine_evidence.py", "check_scale_engine_evidence.py"):
            with self.subTest(tool=name):
                source = (TOOLS / name).read_text(encoding="utf-8")
                self.assertIn('"--top-rung"', source)

    def test_the_candidate_is_bound_and_the_checkout_is_clean_before_measuring(self) -> None:
        job = _jobs()["scale-engine-evidence"]
        checkout = job["steps"][0]
        self.assertEqual(checkout["with"]["ref"], "${{ needs.preflight.outputs.candidate_sha }}")
        self.assertEqual(checkout["with"]["persist-credentials"], False)
        binding = _step(job, "git status --porcelain")
        self.assertIn('test "$(git rev-parse HEAD)" = "$candidate"', str(binding["run"]))
        measuring = _step(job, "run_scale_engine_evidence.py")
        self.assertLess(job["steps"].index(binding), job["steps"].index(measuring))

    def test_the_job_installs_the_arrow_extra_it_measures_with(self) -> None:
        job = _jobs()["scale-engine-evidence"]
        install = _step(job, "pip install -e")
        self.assertIn('".[test,arrow]"', str(install["run"]))

    def test_the_artifact_is_retained_even_when_validation_fails(self) -> None:
        job = _jobs()["scale-engine-evidence"]
        upload = job["steps"][-1]
        self.assertEqual(upload["if"], "always()")
        self.assertEqual(
            upload["with"]["name"],
            "scale-engine-evidence-${{ matrix.os }}-${{ inputs.candidate_sha }}",
        )
        self.assertEqual(upload["with"]["path"], "${{ runner.temp }}/scale-engine-evidence.json")
        self.assertEqual(upload["with"]["if-no-files-found"], "warn")
        self.assertTrue(str(upload["uses"]).startswith("actions/upload-artifact@"))

    def test_the_preflight_verifies_the_new_tests_with_the_arrow_extra(self) -> None:
        preflight = _jobs()["preflight"]
        install = _step(preflight, "pip install -e")
        self.assertIn('".[test,arrow]"', str(install["run"]))
        verify = _step(preflight, "python -m unittest")
        self.assertIn("tests.scale.test_scale_engine_evidence", str(verify["run"]))

    def test_the_aggregate_gate_waits_for_both_evidence_jobs(self) -> None:
        gate = _jobs()["scale-evidence-gate"]
        self.assertEqual(gate["needs"], ["scale-evidence", "scale-engine-evidence"])
        self.assertEqual(gate["if"], "always()")
        env = gate["steps"][0]["env"]
        self.assertEqual(env["SCALE_EVIDENCE_RESULT"], "${{ needs.scale-evidence.result }}")
        self.assertEqual(
            env["SCALE_ENGINE_EVIDENCE_RESULT"], "${{ needs.scale-engine-evidence.result }}"
        )
        script = str(gate["steps"][0]["run"])
        self.assertIn('"$SCALE_EVIDENCE_RESULT" "$SCALE_ENGINE_EVIDENCE_RESULT"', script)
        self.assertIn('!= "success"', script)

    def test_the_existing_evidence_job_still_measures_what_it_measured(self) -> None:
        job = _jobs()["scale-evidence"]
        self.assertEqual(job["needs"], ["preflight"])
        runs = "\n".join(_runs(job))
        for tool in (
            "run_log_likelihood_scale_evidence.py",
            "check_log_likelihood_scale_evidence.py",
            "run_scale_csv_exponential_evidence.py",
            "check_scale_csv_exponential_evidence.py",
        ):
            with self.subTest(tool=tool):
                self.assertIn(tool, runs)


if __name__ == "__main__":
    unittest.main()
