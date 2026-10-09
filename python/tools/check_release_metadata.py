"""Validate release metadata shared by Python, CFF, Zenodo, and conda-forge."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
import tomllib
from datetime import date
from pathlib import Path

import yaml

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_VERSION_LINE = re.compile(r'^\{% set version = "([^"]+)" %\}$', re.MULTILINE)
_RECIPE_LICENSE = re.compile(r"^\s*license:\s*(\S+)\s*$", re.MULTILINE)
_ZENODO_DOI = re.compile(r"10\.5281/zenodo\.\d+")


def _date(value: object) -> str | None:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        try:
            return date.fromisoformat(value).isoformat()
        except ValueError:
            return None
    return None


def _module_version(source: str) -> str | None:
    """Return the literal ``__version__`` assigned at module level, without importing."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
            value: ast.expr | None = node.value
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
            value = node.value
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Name)
                and target.id == "__version__"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                return value.value
    return None


def _doi_errors(citation: dict[object, object]) -> list[str]:
    """Return violations of the Zenodo DOI rule: every declared DOI is a Zenodo DOI."""

    doi_values: list[object] = []
    if "doi" in citation:
        doi_values.append(citation["doi"])
    identifiers = citation.get("identifiers", [])
    if not isinstance(identifiers, list):
        return ["CITATION.cff identifiers must be a list"]
    for identifier in identifiers:
        if isinstance(identifier, dict) and identifier.get("type") == "doi":
            doi_values.append(identifier.get("value"))
    return [
        f"CITATION.cff DOI is not a Zenodo DOI: {value!r}"
        for value in doi_values
        if not isinstance(value, str) or _ZENODO_DOI.fullmatch(value) is None
    ]


def validate(repository: Path, sdist: Path | None = None) -> list[str]:
    """Return every cross-file release-metadata violation."""

    errors: list[str] = []
    try:
        project = tomllib.loads((repository / "python/pyproject.toml").read_text("utf-8"))[
            "project"
        ]
        version = project["version"]
        license_id = project["license"]
        citation = yaml.safe_load((repository / "CITATION.cff").read_text("utf-8"))
        zenodo = json.loads((repository / ".zenodo.json").read_text("utf-8"))
        recipe = (repository / "conda-forge-recipe/meta.yaml").read_text("utf-8")
        module_source = (repository / "python/src/veridist/__init__.py").read_text("utf-8")
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        yaml.YAMLError,
    ) as error:
        return [f"release metadata is unreadable: {error}"]
    if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
        return ["project version is not a stable semantic version"]
    if _module_version(module_source) != version:
        errors.append("veridist.__version__ differs from the package version")
    if not isinstance(citation, dict):
        return ["CITATION.cff root must be a mapping"]
    if citation.get("cff-version") != "1.2.0":
        errors.append("CITATION.cff must use CFF 1.2.0")
    if citation.get("version") != version:
        errors.append("CITATION.cff version differs from the package")
    released = _date(citation.get("date-released"))
    if released is None:
        errors.append("CITATION.cff lacks a valid release date")
    if citation.get("license") != license_id:
        errors.append("CITATION.cff license differs from the package")
    if not isinstance(citation.get("authors"), list) or not citation["authors"]:
        errors.append("CITATION.cff lacks authors")
    if citation.get("repository-code") != "https://github.com/alisadeghiaghili/veridist":
        errors.append("CITATION.cff repository is not canonical")
    errors.extend(_doi_errors(citation))
    if not isinstance(zenodo, dict):
        return [*errors, ".zenodo.json root must be an object"]
    if zenodo.get("version") != version:
        errors.append("Zenodo version differs from the package")
    if zenodo.get("publication_date") != released:
        errors.append("Zenodo and CFF release dates differ")
    if zenodo.get("upload_type") != "software":
        errors.append("Zenodo upload type must be software")
    if zenodo.get("license") != license_id:
        errors.append("Zenodo license differs from the package")
    if not isinstance(zenodo.get("creators"), list) or not zenodo["creators"]:
        errors.append("Zenodo metadata lacks creators")
    recipe_version = _VERSION_LINE.search(recipe)
    if recipe_version is None or recipe_version.group(1) != version:
        errors.append("conda-forge recipe version differs from the package")
    recipe_license = _RECIPE_LICENSE.search(recipe)
    if recipe_license is None or recipe_license.group(1) != license_id:
        errors.append("conda-forge recipe license differs from the package")
    digest = re.search(r"^\s*sha256:\s*([0-9a-f]+)\s*$", recipe, re.MULTILINE)
    if digest is None or _SHA256.fullmatch(digest.group(1)) is None:
        errors.append("conda-forge recipe lacks an immutable SHA-256")
    elif sdist is not None:
        try:
            actual_digest = hashlib.sha256(sdist.read_bytes()).hexdigest()
        except OSError as error:
            errors.append(f"source distribution is unreadable: {error}")
        else:
            if actual_digest != digest.group(1):
                errors.append(
                    "conda-forge SHA-256 differs from the built source distribution "
                    f"(recipe={digest.group(1)}, built={actual_digest})"
                )
    required_recipe_text = (
        "releases/download/v{{ version }}/veridist-{{ version }}.tar.gz",
        "--no-deps",
        "--no-build-isolation",
        "license_file: LICENSE",
    )
    for required in required_recipe_text:
        if required not in recipe:
            errors.append(f"conda-forge recipe lacks required text: {required}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--sdist", type=Path)
    args = parser.parse_args()
    errors = validate(
        args.repository_root.resolve(), args.sdist.resolve() if args.sdist is not None else None
    )
    if errors:
        print("FAIL: " + "; ".join(errors), file=sys.stderr)
        return 1
    print("PASS: release metadata is complete and version-aligned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
