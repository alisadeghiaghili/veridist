"""Contract tests for the manually dispatched, candidate-bound scale gate."""

from __future__ import annotations

import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "scale-evidence.yml"
MUTATION_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "mutation.yml"


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
        self.assertIn("needs: [scale-evidence]", workflow)
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


if __name__ == "__main__":
    unittest.main()
