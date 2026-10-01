"""Build wheel and sdist archives with canonical container metadata.

Byte-for-byte reproducibility here depends on two things this module does
not control. First, both archives store payloads with DEFLATE
(``zipfile.ZIP_DEFLATED`` for the wheel, gzip for the sdist); the compressed
bytes can differ between zlib versions even when the decompressed content is
identical, so builds across hosts with different zlib builds are only
guaranteed to match after decompression, not as raw archive bytes. Second,
``build()`` invokes ``python -m build --no-isolation``, which builds with
whatever ``setuptools``/``wheel`` versions are already installed; pin those
versions in the calling environment (as ``veridist-release.yml`` does) if the
raw archive bytes must match across runs.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import io
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def _normalise_sdist(source: Path, target: Path, epoch: int) -> None:
    with tarfile.open(source, "r:gz") as archive:
        members = sorted(archive.getmembers(), key=lambda member: member.name)
        with target.open("wb") as raw_target:
            with gzip.GzipFile(fileobj=raw_target, mode="wb", filename="", mtime=epoch) as zipped:
                with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as output:
                    for member in members:
                        member.mtime = epoch
                        member.uid = 0
                        member.gid = 0
                        member.uname = ""
                        member.gname = ""
                        member.pax_headers = {}
                        is_executable = bool(member.mode & 0o111)
                        member.mode = 0o755 if member.isdir() or is_executable else 0o644
                        payload = archive.extractfile(member) if member.isfile() else None
                        output.addfile(member, payload)


def _normalise_wheel(source: Path, target: Path, epoch: int) -> None:
    stamp = datetime.fromtimestamp(max(epoch, 315532800), tz=UTC)
    date_time = (stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute, stamp.second)
    with zipfile.ZipFile(source, "r") as archive:
        originals = {item.filename: item for item in archive.infolist()}
        contents = {name: archive.read(name) for name in originals}
    wheel_files = [name for name in contents if name.endswith(".dist-info/WHEEL")]
    record_files = [name for name in contents if name.endswith(".dist-info/RECORD")]
    if len(wheel_files) != 1 or len(record_files) != 1:
        raise RuntimeError("wheel lacks unique WHEEL and RECORD metadata")
    wheel_name, record_name = wheel_files[0], record_files[0]
    wheel_lines = contents[wheel_name].decode("utf-8").splitlines()
    contents[wheel_name] = (
        "\n".join(
            "Generator: veridist-reproducible-build" if line.startswith("Generator:") else line
            for line in wheel_lines
        )
        + "\n"
    ).encode("utf-8")
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name in sorted(contents):
        if name == record_name:
            continue
        payload = contents[name]
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
        writer.writerow((name, f"sha256={digest}", len(payload)))
    writer.writerow((record_name, "", ""))
    contents[record_name] = record.getvalue().encode("utf-8")
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for name in sorted(contents):
            original = originals[name]
            info = zipfile.ZipInfo(name, date_time=date_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3  # Unix, so external_attr mode bits below are honored.
            info.external_attr = 0o644 << 16
            info.flag_bits = original.flag_bits
            output.writestr(info, contents[name], compress_type=zipfile.ZIP_DEFLATED)


def build(project: Path, output: Path, epoch: int) -> tuple[Path, Path]:
    """Build and canonicalise exactly one wheel and one source distribution."""

    if epoch < 315532800:
        raise ValueError("epoch must be on or after 1980-01-01")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="veridist-build-") as directory:
        raw = Path(directory)
        environment = os.environ.copy()
        environment["SOURCE_DATE_EPOCH"] = str(epoch)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--no-isolation",
                "--sdist",
                "--wheel",
                "--outdir",
                str(raw),
            ],
            cwd=project,
            env=environment,
            check=True,
        )
        sdists = list(raw.glob("*.tar.gz"))
        wheels = list(raw.glob("*.whl"))
        if len(sdists) != 1 or len(wheels) != 1:
            raise RuntimeError("build did not produce exactly one wheel and one sdist")
        sdist_target = output / sdists[0].name
        wheel_target = output / wheels[0].name
        _normalise_sdist(sdists[0], sdist_target, epoch)
        _normalise_wheel(wheels[0], wheel_target, epoch)
        return sdist_target, wheel_target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()
    try:
        sdist, wheel = build(args.project.resolve(), args.output.resolve(), args.epoch)
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"FAIL: reproducible build failed: {error}", file=sys.stderr)
        return 1
    print(f"built {sdist.name} and {wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
