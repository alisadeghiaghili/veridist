"""Contracts for the fail-closed Veridist release validation workflow."""

import unittest
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "veridist-release.yml"


class ReleaseWorkflowTests(unittest.TestCase):
    def test_workflow_validates_only_and_binds_tag_version_and_artifacts(self) -> None:
        content = WORKFLOW.read_text(encoding="utf-8")
        for required in (
            "name: veridist-release-validation",
            "workflow_dispatch:",
            "candidate_sha:",
            "release_tag:",
            "ref: ${{ inputs.candidate_sha }}",
            "fetch-depth: 0",
            "persist-credentials: false",
            "working-directory: python",
            "setuptools==84.0.0",
            "wheel==0.48.0",
            "python tools/build_reproducible.py --project . --output dist-a",
            "python tools/build_reproducible.py --project . --output dist-b",
            "cmp dist-a/*.whl dist-b/*.whl",
            "cmp dist-a/*.tar.gz dist-b/*.tar.gz",
            "python tools/check_release_metadata.py --repository-root .. --sdist",
            "python -m twine check dist/*",
            "python tools/check_release_artifacts.py",
            '--release-tag "$RELEASE_TAG"',
            "--artifact dist/*.whl",
            "--artifact dist/*.tar.gz",
            "importlib.metadata",
            "veridist",
            "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02 # v4.6.2",
            "path: python/dist",
            "release tag does not match package version",
            "candidate SHA must be a full lowercase Git commit",
            "checked-out candidate SHA differs from requested candidate SHA",
            "release candidate checkout is dirty",
        ):
            with self.subTest(required=required):
                self.assertIn(required, content)
        self.assertNotIn("twine upload", content)
        self.assertNotIn("pypa/gh-action-pypi-publish", content)


if __name__ == "__main__":
    unittest.main()
