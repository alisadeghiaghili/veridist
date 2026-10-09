"""Validate Veridist release metadata and distribution payloads."""

from __future__ import annotations

import argparse
import sys
import tarfile
import tomllib
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath


class ReleaseArtifactError(ValueError):
    """Report a release artifact that does not match the source contract."""


def _read_members(artifact: Path) -> dict[str, bytes]:
    if artifact.suffix == ".whl" and zipfile.is_zipfile(artifact):
        with zipfile.ZipFile(artifact) as archive:
            return {
                name: archive.read(name) for name in archive.namelist() if not name.endswith("/")
            }
    if artifact.name.endswith(".tar.gz") and tarfile.is_tarfile(artifact):
        with tarfile.open(artifact) as archive:
            members: dict[str, bytes] = {}
            for member in archive.getmembers():
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is not None:
                        members[member.name] = stream.read()
            return members
    raise ReleaseArtifactError(f"unsupported distribution artifact: {artifact}")


def _required_member(members: dict[str, bytes], name: str, artifact: Path) -> bytes:
    try:
        return members[name]
    except KeyError as error:
        raise ReleaseArtifactError(f"{artifact.name} is missing required member {name}") from error


def _source_payloads(project_root: Path) -> dict[str, bytes]:
    """Return every ``*.py``/``py.typed`` payload under ``src/veridist``, keyed by relative path."""

    root = project_root / "src" / "veridist"
    payloads: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix != ".py" and path.name != "py.typed":
            continue
        payloads[path.relative_to(root).as_posix()] = path.read_bytes()
    return payloads


def _validate_package_payload(
    members: dict[str, bytes],
    *,
    prefix: str,
    expected: dict[str, bytes],
    artifact: Path,
) -> None:
    prefix_path = PurePosixPath(prefix)
    packaged: dict[str, bytes] = {}
    for member, payload in members.items():
        try:
            relative = PurePosixPath(member).relative_to(prefix_path)
        except ValueError:
            continue
        packaged[relative.as_posix()] = payload
    missing = sorted(set(expected) - set(packaged))
    if missing:
        raise ReleaseArtifactError(
            f"{artifact.name} is missing veridist source file(s): {', '.join(missing)}"
        )
    extra = sorted(set(packaged) - set(expected))
    if extra:
        raise ReleaseArtifactError(
            f"{artifact.name} contains unexpected file(s) under {prefix}: {', '.join(extra)}"
        )
    modified = sorted(name for name in expected if packaged[name] != expected[name])
    if modified:
        raise ReleaseArtifactError(
            f"{artifact.name} contains a modified veridist source file: {', '.join(modified)}"
        )


def _expected(project_root: Path, release_tag: str) -> tuple[str, str, str]:
    configuration = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = configuration["project"]
    name = str(project["name"])
    version = str(project["version"])
    license_expression = str(project["license"])
    if release_tag != f"v{version}":
        raise ReleaseArtifactError(
            f"release tag {release_tag!r} does not match package version {version!r}"
        )
    return name, version, license_expression


def _validate_metadata(
    payload: bytes, *, name: str, version: str, license_expression: str, artifact: Path
) -> None:
    metadata = BytesParser().parsebytes(payload)
    expected = {
        "Name": name,
        "Version": version,
        "License-Expression": license_expression,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ReleaseArtifactError(
                f"{artifact.name} metadata {field} is {metadata.get(field)!r}; expected {value!r}"
            )


def validate_artifact(artifact: Path, *, project_root: Path, release_tag: str) -> None:
    """Validate one wheel or source distribution against source metadata.

    Parameters
    ----------
    artifact : pathlib.Path
        Wheel or ``.tar.gz`` source distribution to inspect.
    project_root : pathlib.Path
        Nested Veridist project containing ``pyproject.toml`` and ``LICENSE``.
    release_tag : str
        Exact ``v``-prefixed version requested for validation.

    Raises
    ------
    ReleaseArtifactError
        If tag, metadata, required files, or package payload disagree.

    Examples
    --------
    Validate built distributions through the command-line entry point::

        python tools/check_release_artifacts.py --project-root . \
            --release-tag v2.1.0 --artifact dist/veridist-2.1.0-py3-none-any.whl \
            --artifact dist/veridist-2.1.0.tar.gz
    """
    if not artifact.is_file():
        raise ReleaseArtifactError(f"artifact does not exist: {artifact}")
    name, version, license_expression = _expected(project_root, release_tag)
    members = _read_members(artifact)
    if artifact.suffix == ".whl":
        metadata_name = f"{name}-{version}.dist-info/METADATA"
    else:
        metadata_name = f"{name}-{version}/PKG-INFO"
    metadata_payload = _required_member(members, metadata_name, artifact)
    _validate_metadata(
        metadata_payload,
        name=name,
        version=version,
        license_expression=license_expression,
        artifact=artifact,
    )

    license_payload = (project_root / "LICENSE").read_bytes()
    packaged_licenses = [
        payload for member, payload in members.items() if PurePosixPath(member).name == "LICENSE"
    ]
    if license_payload not in packaged_licenses:
        raise ReleaseArtifactError(f"{artifact.name} does not contain the exact project LICENSE")

    package_members = [PurePosixPath(member).parts for member in members]
    expected_payloads = _source_payloads(project_root)
    if artifact.suffix == ".whl":
        if not any(parts and parts[0] == "veridist" for parts in package_members):
            raise ReleaseArtifactError(f"{artifact.name} does not contain the veridist package")
        if any(parts and parts[0] == "distfit_pro" for parts in package_members):
            raise ReleaseArtifactError(f"{artifact.name} contains the legacy distfit_pro package")
        _validate_package_payload(
            members, prefix="veridist", expected=expected_payloads, artifact=artifact
        )
    else:
        source_prefix = f"{name}-{version}"
        if not any(
            len(parts) >= 3 and parts[0] == source_prefix and parts[1:3] == ("src", "veridist")
            for parts in package_members
        ):
            raise ReleaseArtifactError(f"{artifact.name} does not contain the veridist source tree")
        if any("distfit_pro" in parts for parts in package_members):
            raise ReleaseArtifactError(f"{artifact.name} contains the legacy distfit_pro package")
        for required_name in (
            "CHANGELOG.md",
            "KNOWN_LIMITS.md",
            "KNOWN_LIMITS.fa.md",
            "KNOWN_LIMITS.de.md",
        ):
            packaged = _required_member(members, f"{source_prefix}/{required_name}", artifact)
            if packaged != (project_root / required_name).read_bytes():
                raise ReleaseArtifactError(f"{artifact.name} contains a modified {required_name}")
        _validate_package_payload(
            members,
            prefix=f"{source_prefix}/src/veridist",
            expected=expected_payloads,
            artifact=artifact,
        )


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--release-tag", required=True)
    parser.add_argument("--artifact", type=Path, action="append", required=True)
    parsed = parser.parse_args(arguments)
    try:
        for artifact in parsed.artifact:
            validate_artifact(
                artifact,
                project_root=parsed.project_root,
                release_tag=parsed.release_tag,
            )
    except (OSError, KeyError, ReleaseArtifactError, ValueError) as error:
        print(f"release artifact check failed: {error}", file=sys.stderr)
        return 1
    print("release artifact check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
