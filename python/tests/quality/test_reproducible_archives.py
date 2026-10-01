"""Contracts for deterministic archive canonicalisation."""

from __future__ import annotations

import hashlib
import io
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from tools.build_reproducible import _normalise_sdist, _normalise_wheel

EPOCH = 1_789_084_800


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ReproducibleArchiveTests(unittest.TestCase):
    def test_sdist_metadata_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs: list[Path] = []
            for index, mtime in enumerate((1_700_000_000, 1_800_000_000)):
                raw = root / f"raw-{index}.tar.gz"
                with tarfile.open(raw, "w:gz") as archive:
                    info = tarfile.TarInfo("package/value.txt")
                    info.size = 5
                    info.mtime = mtime
                    info.uid = index + 1
                    archive.addfile(info, io.BytesIO(b"value"))
                target = root / f"normal-{index}.tar.gz"
                _normalise_sdist(raw, target, EPOCH)
                outputs.append(target)
            self.assertEqual(_digest(outputs[0]), _digest(outputs[1]))

    def test_sdist_member_modes_are_normalised(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.tar.gz"
            with tarfile.open(raw, "w:gz") as archive:
                plain = tarfile.TarInfo("package/plain.txt")
                plain.size = 5
                plain.mode = 0o644
                archive.addfile(plain, io.BytesIO(b"value"))
                executable = tarfile.TarInfo("package/run.sh")
                executable.size = 5
                executable.mode = 0o755
                archive.addfile(executable, io.BytesIO(b"value"))
                odd = tarfile.TarInfo("package/odd.txt")
                odd.size = 5
                odd.mode = 0o666
                archive.addfile(odd, io.BytesIO(b"value"))
                directory_member = tarfile.TarInfo("package/sub")
                directory_member.type = tarfile.DIRTYPE
                directory_member.mode = 0o644
                archive.addfile(directory_member)
            target = root / "normal.tar.gz"
            _normalise_sdist(raw, target, EPOCH)
            with tarfile.open(target, "r:gz") as archive:
                modes = {member.name: member.mode for member in archive.getmembers()}
            self.assertEqual(modes["package/plain.txt"], 0o644)
            self.assertEqual(modes["package/run.sh"], 0o755)
            self.assertEqual(modes["package/odd.txt"], 0o644)
            self.assertEqual(modes["package/sub"], 0o755)

    def test_wheel_metadata_is_canonical(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs: list[Path] = []
            for index, stamp in enumerate(((2020, 1, 1, 0, 0, 0), (2025, 1, 1, 0, 0, 0))):
                raw = root / f"raw-{index}.whl"
                with zipfile.ZipFile(raw, "w") as archive:
                    info = zipfile.ZipInfo("package/value.txt", date_time=stamp)
                    archive.writestr(info, b"value")
                    archive.writestr(
                        zipfile.ZipInfo("package-1.0.dist-info/WHEEL", date_time=stamp),
                        f"Wheel-Version: 1.0\nGenerator: backend-{index}\n",
                    )
                    archive.writestr(
                        zipfile.ZipInfo("package-1.0.dist-info/RECORD", date_time=stamp), b""
                    )
                target = root / f"normal-{index}.whl"
                _normalise_wheel(raw, target, EPOCH)
                outputs.append(target)
            self.assertEqual(_digest(outputs[0]), _digest(outputs[1]))

    def test_wheel_external_attr_is_normalised_to_unix_644(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw.whl"
            stamp = (2020, 1, 1, 0, 0, 0)
            with zipfile.ZipFile(raw, "w") as archive:
                payload_info = zipfile.ZipInfo("package/value.txt", date_time=stamp)
                payload_info.external_attr = 0o777 << 16
                payload_info.create_system = 0
                archive.writestr(payload_info, b"value")
                wheel_info = zipfile.ZipInfo("package-1.0.dist-info/WHEEL", date_time=stamp)
                wheel_info.external_attr = 0 << 16
                archive.writestr(wheel_info, "Wheel-Version: 1.0\nGenerator: backend\n")
                record_info = zipfile.ZipInfo("package-1.0.dist-info/RECORD", date_time=stamp)
                archive.writestr(record_info, b"")
            target = root / "normal.whl"
            _normalise_wheel(raw, target, EPOCH)
            with zipfile.ZipFile(target) as archive:
                for info in archive.infolist():
                    self.assertEqual(info.external_attr, 0o644 << 16, info.filename)
                    self.assertEqual(info.create_system, 3, info.filename)


if __name__ == "__main__":
    unittest.main()
