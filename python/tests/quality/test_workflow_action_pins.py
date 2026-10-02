"""Every third-party action in a Veridist workflow is pinned to a commit SHA."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[3] / ".github" / "workflows"

# The legacy workflow is validated byte-for-byte by tools/check_legacy_release_safety.py and
# quality/legacy-ci-manifest.json, so its action references must stay as they are.
UNPINNED_ALLOWLIST = frozenset({"ci.yml"})

_PINNED_USE = re.compile(r"^(?P<action>[^@\s]+)@(?P<sha>[0-9a-f]{40})\s+#\s*\S")
_UNSAFE_SCRIPT_EXPRESSION = re.compile(r"\$\{\{[^}]*(github\.event\.|inputs\.|env\.)[^}]*\}\}")
_USES_LINE = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S.*?)\s*$")


def _uses_lines(path: Path) -> list[str]:
    references = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = _USES_LINE.match(line)
        if match is not None:
            references.append(match.group("ref"))
    return references


def _parsed_uses(path: Path) -> list[str]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    references = []
    for job in document["jobs"].values():
        if "uses" in job:
            references.append(job["uses"])
        for step in job.get("steps", []):
            if "uses" in step:
                references.append(step["uses"])
    return references


class WorkflowActionPinTests(unittest.TestCase):
    def test_every_workflow_is_discovered(self) -> None:
        names = {path.name for path in WORKFLOWS.glob("*.yml")}
        self.assertTrue(UNPINNED_ALLOWLIST <= names)
        self.assertGreater(len(names - UNPINNED_ALLOWLIST), 0)

    def test_every_action_outside_the_allowlist_is_pinned_with_a_tag_comment(self) -> None:
        for path in sorted(WORKFLOWS.glob("*.yml")):
            if path.name in UNPINNED_ALLOWLIST:
                continue
            with self.subTest(workflow=path.name):
                references = _uses_lines(path)
                self.assertTrue(references)
                for reference in references:
                    self.assertRegex(reference, _PINNED_USE)

    def test_line_scan_agrees_with_the_parsed_workflow(self) -> None:
        for path in sorted(WORKFLOWS.glob("*.yml")):
            with self.subTest(workflow=path.name):
                scanned = [reference.split()[0] for reference in _uses_lines(path)]
                self.assertEqual(scanned, _parsed_uses(path))

    def test_the_same_action_resolves_to_one_commit_across_workflows(self) -> None:
        commits: dict[str, set[str]] = {}
        for path in sorted(WORKFLOWS.glob("*.yml")):
            if path.name in UNPINNED_ALLOWLIST:
                continue
            for reference in _uses_lines(path):
                match = _PINNED_USE.match(reference)
                self.assertIsNotNone(match, reference)
                action = match.group("action")
                commits.setdefault(action, set()).add(match.group("sha"))
        self.assertIn("actions/checkout", commits)
        for action, shas in commits.items():
            with self.subTest(action=action):
                self.assertEqual(len(shas), 1, shas)

    def test_the_legacy_workflow_is_the_only_exemption(self) -> None:
        self.assertEqual(UNPINNED_ALLOWLIST, frozenset({"ci.yml"}))
        legacy = _uses_lines(WORKFLOWS / "ci.yml")
        self.assertTrue(any(_PINNED_USE.match(reference) is None for reference in legacy))

    def test_pin_pattern_rejects_mutable_references(self) -> None:
        sha = "a" * 40
        for reference in (
            "actions/checkout@v4",
            "actions/checkout@main",
            f"actions/checkout@{sha}",
            f"actions/checkout@{sha[:-1]}",
            f"actions/checkout@{sha.upper()} # v4",
            f"actions/checkout@{sha}#v4",
        ):
            with self.subTest(reference=reference):
                self.assertIsNone(_PINNED_USE.match(reference))
        self.assertIsNotNone(_PINNED_USE.match(f"actions/checkout@{sha} # v4.2.2"))

    def test_scripts_never_interpolate_event_payload_values(self) -> None:
        # Expressions are expanded into the script text before the shell
        # parses it, so caller-controlled values (event payloads, dispatch
        # inputs, and env values derived from them) must reach run: scripts
        # only through environment variables.
        for path in sorted(WORKFLOWS.glob("*.yml")):
            if path.name in UNPINNED_ALLOWLIST:
                continue
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
            for job_name, job in document["jobs"].items():
                for step in job.get("steps", []):
                    with self.subTest(workflow=path.name, job=job_name):
                        script = step.get("run", "")
                        self.assertIsNone(_UNSAFE_SCRIPT_EXPRESSION.search(script), script)


if __name__ == "__main__":
    unittest.main()
