"""Contracts for merging per-shard mutation evidence into one fail-closed verdict."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PYTHON_ROOT / "tools"))
import merge_mutation_evidence  # noqa: E402

from tests.quality.test_mutation_evidence import (  # noqa: E402
    FIXTURE_SHARDS,
    shard_evidence,
    with_counts,
    write_manifest,
    write_tree,
)

MERGER = PYTHON_ROOT / "tools" / "merge_mutation_evidence.py"


def build(
    root: Path, counts: dict[str, tuple[int, int]] | None = None
) -> dict[str, dict[str, object]]:
    """Return valid evidence for every fixture shard (``counts``: shard -> killed, survived)."""

    write_tree(root)
    payloads: dict[str, dict[str, object]] = {}
    for shard in FIXTURE_SHARDS:
        killed, survived = (counts or {}).get(shard, (100, 0))
        payloads[shard] = with_counts(shard_evidence(root, shard), killed, survived)
    return payloads


def lay_out(shards_root: Path, payloads: dict[str, dict[str, object]]) -> None:
    for name, payload in payloads.items():
        target = shards_root / f"mutation-shard-{name}-1-1"
        target.mkdir(parents=True)
        (target / "mutation-evidence.json").write_text(json.dumps(payload), encoding="utf-8")


def run_merge(
    root: Path, payloads: dict[str, dict[str, object]], output: Path | None = None
) -> subprocess.CompletedProcess[str]:
    shards_root = root / "downloaded"
    lay_out(shards_root, payloads)
    command = [
        sys.executable,
        str(MERGER),
        "--project-root",
        str(root),
        "--shards-root",
        str(shards_root),
        "--fixture",
    ]
    if output is not None:
        command += ["--output", str(output)]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def merge_in_process(root: Path, payloads: dict[str, dict[str, object]]) -> dict[str, object]:
    return merge_mutation_evidence.merge(root, list(copy.deepcopy(payloads).values()))


class MergeAcceptanceTests(unittest.TestCase):
    def test_a_complete_consistent_set_passes_and_reports_merged_totals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root, {"scale-a": (3, 7), "scale-b": (97, 3)})
            output = root / "merged.json"
            result = run_merge(root, payloads, output)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("MUTATION MERGE PASS", result.stdout)
            self.assertIn("module scale: mutation score 0.9091", result.stdout)
            merged = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(merged["schema_version"], 4)
            self.assertEqual(merged["commit"], "fixture")
            self.assertEqual([row["name"] for row in merged["shards"]], sorted(FIXTURE_SHARDS))
            scale = next(row for row in merged["modules"] if row["module"] == "scale")
            self.assertEqual(
                (scale["killed"], scale["survived"], scale["generated"]), (100, 10, 110)
            )
            self.assertEqual(merged["totals"]["generated"], 4 * 100 + 110)
            self.assertEqual(
                merged["score"], merged["totals"]["killed"] / merged["totals"]["generated"]
            )

    def test_a_module_floor_is_met_over_merged_totals_although_one_shard_alone_is_below_it(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root, {"scale-a": (1, 9), "scale-b": (99, 1)})
            self.assertEqual(run_merge(root, payloads).returncode, 0)

    def test_a_module_exactly_at_its_floor_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root, {"domain": (1, 1)})
            result = run_merge(root, payloads)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("module domain: mutation score 0.5000", result.stdout)

    def test_the_merged_evidence_is_not_published_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            del payloads["engine"]
            output = root / "merged.json"
            self.assertEqual(run_merge(root, payloads, output).returncode, 1)
            self.assertFalse(output.exists())


class MergeRejectionTests(unittest.TestCase):
    def test_a_missing_shard_is_rejected_for_every_shard(self) -> None:
        for shard in FIXTURE_SHARDS:
            with self.subTest(shard=shard), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payloads = build(root)
                del payloads[shard]
                result = run_merge(root, payloads)
                self.assertEqual(result.returncode, 1)
                self.assertIn(f"missing shard evidence: {shard}", result.stderr)

    def test_a_duplicate_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            lay_out(root / "downloaded", payloads)
            twin = root / "downloaded" / "mutation-shard-engine-1-2"
            twin.mkdir()
            (twin / "mutation-evidence.json").write_text(
                json.dumps(payloads["engine"]), encoding="utf-8"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(MERGER),
                    "--project-root",
                    str(root),
                    "--shards-root",
                    str(root / "downloaded"),
                    "--fixture",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("duplicate shard evidence: engine", result.stderr)

    def test_unexpected_artifact_entries_are_rejected(self) -> None:
        for kind in ("stray file", "directory without evidence", "no artifacts"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                payloads = build(root)
                shards_root = root / "downloaded"
                if kind != "no artifacts":
                    lay_out(shards_root, payloads)
                else:
                    shards_root.mkdir()
                if kind == "stray file":
                    (shards_root / "notes.txt").write_text("x", encoding="utf-8")
                elif kind == "directory without evidence":
                    (shards_root / "mutation-shard-extra-1-1").mkdir()
                result = subprocess.run(
                    [
                        sys.executable,
                        str(MERGER),
                        "--project-root",
                        str(root),
                        "--shards-root",
                        str(shards_root),
                        "--fixture",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("MUTATION MERGE FAIL", result.stderr)

    def test_a_missing_artifact_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(MERGER),
                    "--project-root",
                    str(root),
                    "--shards-root",
                    str(root / "absent"),
                    "--fixture",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 1)

    def test_a_shard_claiming_another_shards_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            # Rejected by the merge itself ...
            stolen = payloads["scale-a"]["shard"]["files"][0]  # type: ignore[index]
            payloads["scale-b"]["shard"]["files"].append(stolen)  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "more than one shard"):
                merge_in_process(root, payloads)
        # ... and, one layer earlier, by the per-shard check of the same evidence.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            payloads["scale-b"]["shard"]["files"].append(  # type: ignore[index]
                payloads["scale-a"]["shard"]["files"][0]  # type: ignore[index]
            )
            self.assertEqual(run_merge(root, payloads).returncode, 1)

    def test_a_file_that_no_shard_mutated_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            payloads["scale-a"]["shard"]["files"] = []  # type: ignore[index]
            payloads["scale-a"]["files"] = []  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "not mutated: src/veridist/scale/other.py"):
                merge_in_process(root, payloads)

    def test_a_report_for_a_file_the_shard_does_not_own_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            payloads["scale-a"]["files"][0]["path"] = "src/veridist/scale/sample.py"  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "does not own"):
                merge_in_process(root, payloads)

    def test_input_manifest_digest_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            provenance = payloads["engine"]["provenance"]  # type: ignore[index]
            for key in ("input_digest", "pre_input_digest", "post_input_digest"):
                provenance[key] = "e" * 64
            result = run_merge(root, payloads)
            self.assertEqual(result.returncode, 1)
            self.assertIn("input manifest digest differs between shards", result.stderr)

    def test_an_input_that_changed_during_a_shard_run_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            payloads["engine"]["provenance"]["post_input_digest"] = "e" * 64  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "changed while a shard was running"):
                merge_in_process(root, payloads)

    def test_commit_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            payloads["engine"]["source"]["commit"] = "0" * 40  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "source commit differs between shards"):
                merge_in_process(root, payloads)

    def test_mutation_configuration_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            payloads["engine"]["config"]["pytest_selection"] = ["tests/unit"]  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "mutation configuration differs"):
                merge_in_process(root, payloads)

    def test_evidence_for_an_undeclared_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            payloads["engine"]["shard"]["name"] = "scale"  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "missing shard evidence: engine"):
                merge_in_process(root, payloads)

    def test_unresolved_mutants_are_rejected_by_the_shard_check_and_the_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            report = payloads["engine"]["files"][0]  # type: ignore[index]
            report["mutants"][0].update(
                exit_code=None, official_status="not_checked", scoring_status="unresolved"
            )
            report.update(killed=99, unresolved=1)
            payloads["engine"]["modules"][0].update(killed=99, unresolved=1)  # type: ignore[index]
            payloads["engine"]["totals"].update(killed=99, unresolved=1)  # type: ignore[index]
            result = run_merge(root, payloads)
            self.assertEqual(result.returncode, 1)
            self.assertIn("unresolved or zero-scored evidence", result.stderr)
            with self.assertRaisesRegex(ValueError, "unresolved mutants remain: 1"):
                merge_in_process(root, payloads)

    def test_a_module_below_its_floor_fails_although_the_global_score_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root, {"domain": (4, 6)})
            result = run_merge(root, payloads)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn(
                "module domain mutation score 0.4000 is below its minimum 0.50", result.stderr
            )

    def test_a_module_split_over_shards_is_judged_on_the_merged_totals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # scale-a is perfect and scale-b is poor; together they are 50 / 100 = 0.5 < 0.8.
            write_tree(root)
            write_manifest(
                root,
                module_minimum_scores={
                    "domain": 0.5,
                    "statistics": 0.5,
                    "families": 0.5,
                    "engine": 0.5,
                    "scale": 0.8,
                },
            )
            payloads = {
                shard: with_counts(shard_evidence(root, shard), *counts)
                for shard, counts in {
                    "domain": (100, 0),
                    "statistics": (100, 0),
                    "families": (100, 0),
                    "engine": (100, 0),
                    "scale-a": (50, 0),
                    "scale-b": (1, 49),
                }.items()
            }
            result = run_merge(root, payloads)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn(
                "module scale mutation score 0.5100 is below its minimum 0.80", result.stderr
            )

    def test_the_global_floor_fails_when_every_module_clears_its_own(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root, {shard: (7, 3) for shard in FIXTURE_SHARDS})
            result = run_merge(root, payloads)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("mutation score 0.7000 is below the global minimum 0.80", result.stderr)

    def test_a_merge_without_scored_mutants_for_a_module_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            payloads["engine"]["files"][0].update(generated=0, killed=0, survived=0)  # type: ignore[index]
            payloads["engine"]["files"][0]["mutants"] = []  # type: ignore[index]
            with self.assertRaisesRegex(ValueError, "module engine has no scored mutants"):
                merge_in_process(root, payloads)

    def test_the_directory_name_does_not_decide_the_shard_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = build(root)
            shards_root = root / "downloaded"
            lay_out(shards_root, payloads)
            (shards_root / "mutation-shard-engine-1-1").rename(shards_root / "renamed")
            result = subprocess.run(
                [
                    sys.executable,
                    str(MERGER),
                    "--project-root",
                    str(root),
                    "--shards-root",
                    str(shards_root),
                    "--fixture",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
