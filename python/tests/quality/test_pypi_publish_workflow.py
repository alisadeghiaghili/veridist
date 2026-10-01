"""Contracts for the isolated, verify-before-publish trusted-publishing workflow."""

from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[3] / ".github" / "workflows" / "pypi-publish.yml"


def test_pypi_publish_workflow_is_release_bound_and_oidc_only() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "types: [published]",
        "environment: pypi",
        "contents: read",
        "actions: read",
        "id-token: write",
        "attestations: write",
        "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
        "packages-dir: python/dist",
        "gh release download",
    ):
        assert required in content
    assert "PYPI_TOKEN" not in content


def test_pypi_publish_workflow_pins_third_party_actions_to_a_commit_sha() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2",
        "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065 # v5.6.0",
        "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33",
        "# release/v1 (v1.14.2)",
    ):
        assert required in content
    assert "@release/v1\n" not in content
    assert "@v4\n" not in content
    assert "@v5\n" not in content


def test_pypi_publish_workflow_checks_out_the_release_tag_before_rebuilding() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "ref: ${{ github.event.release.tag_name }}",
        "fetch-depth: 0",
        "persist-credentials: false",
        'python-version: "3.11"',
        "working-directory: python",
    ):
        assert required in content


def test_pypi_publish_workflow_rebuilds_and_verifies_byte_identical_artifacts() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    for required in (
        'pip install "build>=1.2,<2" ".[test]"',
        "setuptools==84.0.0",
        "wheel==0.48.0",
        "release_epoch=",
        'json.load(open("../.zenodo.json", encoding="utf-8"))["publication_date"]',
        "python tools/build_reproducible.py --project . --output dist-rebuilt --epoch",
        "sha256sum",
    ):
        assert required in content


def test_pypi_publish_workflow_runs_all_release_safety_checks(
) -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "python tools/check_release_metadata.py --repository-root .. --sdist",
        "python tools/check_release_artifacts.py",
        "--project-root . --release-tag",
        "python tools/check_legacy_isolation.py --artifact",
    ):
        assert required in content


def test_pypi_publish_workflow_requires_green_ci_and_mutation_runs_for_the_tag() -> None:
    content = WORKFLOW.read_text(encoding="utf-8")
    for required in (
        "gh run list --repo",
        "--workflow \"$workflow\"",
        "v1-ci.yml mutation.yml",
        "--commit \"$tag_sha\"",
        "--status success",
        "--json databaseId",
        "no successful $workflow run found for commit $tag_sha",
        "dispatch it manually",
    ):
        assert required in content


def test_pypi_publish_workflow_never_inlines_event_values_into_run_scripts() -> None:
    """Every ``${{ github.event.* }}`` value used by a shell step must be passed through env.

    Inlining an attacker-influenced event value directly into a ``run:`` block
    (for example the release tag name) is a classic script-injection vector;
    this workflow always binds such values to an environment variable first
    and references only the shell variable in ``run:``.
    """

    import yaml

    document = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    for job in document["jobs"].values():
        for step in job.get("steps", []):
            run = step.get("run")
            if not isinstance(run, str):
                continue
            assert "github.event." not in run, step.get("name")
