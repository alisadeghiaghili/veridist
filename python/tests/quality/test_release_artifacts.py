"""Adversarial contracts for Veridist release distributions."""

from __future__ import annotations

import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.check_release_artifacts import ReleaseArtifactError, validate_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION = "1.0.1"
METADATA = (
    f"Metadata-Version: 2.4\nName: veridist\nVersion: {VERSION}\nLicense-Expression: BUSL-1.1\n\n"
).encode()


def _source_payloads() -> dict[str, bytes]:
    root = PROJECT_ROOT / "src" / "veridist"
    payloads: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix != ".py" and path.name != "py.typed":
            continue
        payloads[path.relative_to(root).as_posix()] = path.read_bytes()
    return payloads


def _wheel(
    path: Path,
    *,
    metadata: bytes = METADATA,
    legacy: bool = False,
    drop: str | None = None,
    tamper: str | None = None,
    extra: str | None = None,
) -> None:
    payloads = dict(_source_payloads())
    if drop is not None:
        del payloads[drop]
    if tamper is not None:
        payloads[tamper] = payloads[tamper] + b"\n# tampered\n"
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in payloads.items():
            archive.writestr(f"veridist/{name}", payload)
        if extra is not None:
            archive.writestr(f"veridist/{extra}", b"# unexpected\n")
        archive.writestr(f"veridist-{VERSION}.dist-info/METADATA", metadata)
        archive.writestr(
            f"veridist-{VERSION}.dist-info/licenses/LICENSE",
            (PROJECT_ROOT / "LICENSE").read_bytes(),
        )
        if legacy:
            archive.writestr("distfit_pro/__init__.py", b"")


def _sdist(
    path: Path,
    *,
    modified_known_limits: bool = False,
    drop: str | None = None,
    tamper: str | None = None,
    extra: str | None = None,
) -> None:
    source_prefix = f"veridist-{VERSION}/src/veridist"
    payloads = dict(_source_payloads())
    if drop is not None:
        del payloads[drop]
    if tamper is not None:
        payloads[tamper] = payloads[tamper] + b"\n# tampered\n"
    members = {
        f"veridist-{VERSION}/PKG-INFO": METADATA,
        f"veridist-{VERSION}/src/veridist.egg-info/PKG-INFO": METADATA,
        f"veridist-{VERSION}/LICENSE": (PROJECT_ROOT / "LICENSE").read_bytes(),
        **{f"{source_prefix}/{name}": payload for name, payload in payloads.items()},
        **{
            f"veridist-{VERSION}/{name}": (PROJECT_ROOT / name).read_bytes()
            for name in (
                "CHANGELOG.md",
                "KNOWN_LIMITS.md",
                "KNOWN_LIMITS.fa.md",
                "KNOWN_LIMITS.de.md",
            )
        },
    }
    if extra is not None:
        members[f"{source_prefix}/{extra}"] = b"# unexpected\n"
    if modified_known_limits:
        members[f"veridist-{VERSION}/KNOWN_LIMITS.md"] = b"modified\n"
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


class ReleaseArtifactContractTests(unittest.TestCase):
    def test_valid_wheel_and_sdist_match_source_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / f"veridist-{VERSION}-py3-none-any.whl"
            sdist = root / f"veridist-{VERSION}.tar.gz"
            _wheel(wheel)
            _sdist(sdist)
            for artifact in (wheel, sdist):
                with self.subTest(artifact=artifact.name):
                    validate_artifact(
                        artifact,
                        project_root=PROJECT_ROOT,
                        release_tag=f"v{VERSION}",
                    )

    def test_rejects_mismatched_tag_metadata_and_legacy_payload(self) -> None:
        cases = {
            "tag": (METADATA, False, "v9.9.9", "does not match package version"),
            "version": (
                METADATA.replace(VERSION.encode(), b"9.9.9"),
                False,
                f"v{VERSION}",
                "metadata Version",
            ),
            "license": (
                METADATA.replace(b"BUSL-1.1", b"MIT"),
                False,
                f"v{VERSION}",
                "metadata License-Expression",
            ),
            "legacy": (METADATA, True, f"v{VERSION}", "legacy distfit_pro"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, (metadata, legacy, tag, message) in cases.items():
                artifact = Path(directory) / f"{name}.whl"
                _wheel(artifact, metadata=metadata, legacy=legacy)
                with self.subTest(case=name), self.assertRaisesRegex(ReleaseArtifactError, message):
                    validate_artifact(artifact, project_root=PROJECT_ROOT, release_tag=tag)

    def test_rejects_modified_release_document_in_sdist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / f"veridist-{VERSION}.tar.gz"
            _sdist(artifact, modified_known_limits=True)
            with self.assertRaisesRegex(ReleaseArtifactError, "modified KNOWN_LIMITS.md"):
                validate_artifact(
                    artifact,
                    project_root=PROJECT_ROOT,
                    release_tag=f"v{VERSION}",
                )

    def test_rejects_a_wheel_holding_only_a_tampered_init_module(self) -> None:
        # This is the historical escape: a wheel containing nothing but a
        # rewritten veridist/__init__.py (plus METADATA and LICENSE) used to
        # pass validation because no byte-for-byte payload check existed.
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / f"veridist-{VERSION}-py3-none-any.whl"
            with zipfile.ZipFile(artifact, "w") as archive:
                archive.writestr(
                    "veridist/__init__.py", "raise SystemExit('tampered')\n"
                )
                archive.writestr(f"veridist-{VERSION}.dist-info/METADATA", METADATA)
                archive.writestr(
                    f"veridist-{VERSION}.dist-info/licenses/LICENSE",
                    (PROJECT_ROOT / "LICENSE").read_bytes(),
                )
            with self.assertRaisesRegex(ReleaseArtifactError, "missing veridist source file"):
                validate_artifact(artifact, project_root=PROJECT_ROOT, release_tag=f"v{VERSION}")

    def test_rejects_tampered_missing_and_extra_payloads(self) -> None:
        cases: dict[str, dict[str, str]] = {
            "tampered": {"tamper": "families/weibull.py"},
            "missing": {"drop": "families/weibull.py"},
            "extra": {"extra": "families/not_real.py"},
        }
        expected_message = {
            "tampered": "modified veridist source file",
            "missing": "missing veridist source file",
            "extra": "unexpected file",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, kwargs in cases.items():
                for builder, suffix in ((_wheel, ".whl"), (_sdist, ".tar.gz")):
                    with self.subTest(case=name, kind=suffix):
                        artifact = Path(directory) / f"{name}-{suffix.lstrip('.')}{suffix}"
                        builder(artifact, **kwargs)  # type: ignore[arg-type]
                        with self.assertRaisesRegex(
                            ReleaseArtifactError, expected_message[name]
                        ):
                            validate_artifact(
                                artifact, project_root=PROJECT_ROOT, release_tag=f"v{VERSION}"
                            )


if __name__ == "__main__":
    unittest.main()
