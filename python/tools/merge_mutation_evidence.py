"""Merge the evidence of every mutation shard into one fail-closed verdict (schema 4).

Each shard job mutates a declared, disjoint subset of the critical production files and
publishes its own evidence together with the raw mutmut cache and logs. This tool is the only
place where the per-module and global minimum scores are enforced, because a module can span
several shards. It accepts nothing it cannot prove: every shard must be present exactly once,
every shard's evidence is re-validated against its raw cache, all shards must be bound to the
same commit and the same input manifest, and the mutated files of all shards must tile the
critical production scope exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from check_mutation_evidence import REPORT_COUNT_KEYS, check, fail
from mutation_evidence import (
    CRITICAL_MODULES,
    module_minimum_scores,
    module_of,
    mutation_manifest,
    score_excluding_type_check,
    source_files,
    strict_json,
)

SCHEMA_VERSION = 4
EVIDENCE_NAME = "mutation-evidence.json"
# Shard-specific configuration; everything else in ``config`` must be identical across shards.
SHARD_CONFIG_KEYS = frozenset({"only_mutate", "effective_config_sha256"})


def load_shards(
    shards_root: Path, project_root: Path, fixture: bool = False
) -> list[dict[str, Any]]:
    """Load and fully re-validate the evidence found in each downloaded shard artifact.

    The layout is one directory per artifact, each holding ``mutation-evidence.json``, the
    ``logs`` directory and the ``mutants`` tree of ``.meta`` files. Any other entry is rejected.
    """

    if not shards_root.is_dir():
        fail("shard artifact directory is missing")
    payloads: list[dict[str, Any]] = []
    for entry in sorted(shards_root.iterdir()):
        evidence = entry / EVIDENCE_NAME
        if not entry.is_dir() or not evidence.is_file():
            fail(f"unexpected entry in the shard artifacts: {entry.name}")
        value = strict_json(evidence.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            fail(f"evidence root must be an object: {entry.name}")
        payload = cast(dict[str, Any], value)
        check(
            payload,
            project_root,
            fixture,
            (entry / "mutants").resolve(),
            (entry / "logs").resolve(),
        )
        payloads.append(payload)
    if not payloads:
        fail("no shard evidence was downloaded")
    return payloads


def _single(values: list[Any], label: str) -> Any:
    unique = {json.dumps(value, sort_keys=True) for value in values}
    if len(unique) != 1:
        fail(f"{label} differs between shards")
    return values[0]


def merge(project_root: Path, shards: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the merged verdict of already validated shard evidence, or raise ``ValueError``."""

    manifest = mutation_manifest(project_root)
    declared: dict[str, list[str]] = manifest["shards"]
    names = [str(payload["shard"]["name"]) for payload in shards]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        fail("duplicate shard evidence: " + ", ".join(duplicates))
    missing = sorted(set(declared) - set(names))
    if missing:
        fail("missing shard evidence: " + ", ".join(missing))
    unknown = sorted(set(names) - set(declared))
    if unknown:
        fail("evidence for an undeclared shard: " + ", ".join(unknown))

    commit = _single([payload["source"]["commit"] for payload in shards], "source commit")
    provenance = [payload["provenance"] for payload in shards]
    input_digest = _single([item["input_digest"] for item in provenance], "input manifest digest")
    if any(
        item["pre_input_digest"] != input_digest or item["post_input_digest"] != input_digest
        for item in provenance
    ):
        fail("an input manifest changed while a shard was running")
    _single(
        [
            {key: value for key, value in payload["config"].items() if key not in SHARD_CONFIG_KEYS}
            for payload in shards
        ],
        "mutation configuration",
    )

    owner: dict[str, str] = {}
    for payload in shards:
        for item in payload["shard"]["files"]:
            path = str(item["path"])
            if path in owner:
                fail(
                    f"file was mutated by more than one shard: {path} "
                    f"({owner[path]}, {payload['shard']['name']})"
                )
            owner[path] = str(payload["shard"]["name"])
    expected = source_files(project_root)
    absent = sorted(set(expected) - set(owner))
    extra = sorted(set(owner) - set(expected))
    if absent or extra:
        fail(
            "mutated files differ from the critical production set"
            + (f"; not mutated: {', '.join(absent)}" if absent else "")
            + (f"; not critical: {', '.join(extra)}" if extra else "")
        )

    totals = {key: 0 for key in REPORT_COUNT_KEYS}
    by_module = {module: {key: 0 for key in REPORT_COUNT_KEYS} for module in CRITICAL_MODULES}
    shard_rows: list[dict[str, Any]] = []
    for payload in sorted(shards, key=lambda item: str(item["shard"]["name"])):
        row: dict[str, Any] = {
            "name": payload["shard"]["name"],
            "files": len(payload["files"]),
            **{key: 0 for key in REPORT_COUNT_KEYS},
        }
        for report in payload["files"]:
            if owner.get(report["path"]) != payload["shard"]["name"]:
                fail(f"shard reports a file it does not own: {report['path']}")
            for key in REPORT_COUNT_KEYS:
                totals[key] += report[key]
                row[key] += report[key]
                by_module[module_of(report["path"])][key] += report[key]
        shard_rows.append(row)

    if totals["unresolved"]:
        fail(f"unresolved mutants remain: {totals['unresolved']}")
    if totals["generated"] != totals["killed"] + totals["survived"]:
        fail("merged mutant totals are inconsistent")
    denominator = totals["killed"] + totals["survived"]
    if denominator == 0:
        fail("no mutants were scored")
    score = totals["killed"] / denominator
    if score < manifest["minimum_score"]:
        fail(
            f"mutation score {score:.4f} is below the global minimum "
            f"{manifest['minimum_score']:.2f}"
        )
    floors = module_minimum_scores(manifest["module_minimum_scores"])
    module_rows: list[dict[str, Any]] = []
    for module in CRITICAL_MODULES:
        counts = by_module[module]
        scored = counts["killed"] + counts["survived"]
        if scored == 0:
            fail(f"module {module} has no scored mutants")
        achieved = counts["killed"] / scored
        if achieved < floors[module]:
            fail(
                f"module {module} mutation score {achieved:.4f} is below its minimum "
                f"{floors[module]:.2f}"
            )
        module_rows.append({"module": module, **counts, "score": achieved})
    return {
        "schema_version": SCHEMA_VERSION,
        "commit": commit,
        "input_digest": input_digest,
        "shards": shard_rows,
        "modules": module_rows,
        "totals": totals,
        "score": score,
        "score_excluding_type_check": score_excluding_type_check(
            totals["killed"], totals["survived"], totals["type_check"]
        ),
    }


def publish(output: Path, merged: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".tmp")
    staging.write_text(json.dumps(merged, allow_nan=False, sort_keys=True) + "\n", encoding="utf-8")
    staging.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args()
    try:
        root = args.project_root.resolve()
        merged = merge(root, load_shards(args.shards_root.resolve(), root, args.fixture))
        if args.output is not None:
            publish(args.output.resolve(), merged)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"MUTATION MERGE FAIL: {error}", file=sys.stderr)
        return 1
    for row in merged["modules"]:
        print(f"module {row['module']}: mutation score {row['score']:.4f}")
    print(f"mutation score {merged['score']:.4f} over {merged['totals']['generated']} mutants")
    print("MUTATION MERGE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
