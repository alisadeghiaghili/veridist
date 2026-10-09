"""Adversarial checks for cross-file release metadata."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import tomllib
import unittest
from pathlib import Path

from tools.check_release_metadata import validate

ROOT = Path(__file__).resolve().parents[3]


class ReleaseMetadataTests(unittest.TestCase):
    def _copy_metadata(self, target: Path) -> None:
        (target / "python").mkdir()
        (target / "conda-forge-recipe").mkdir()
        (target / "python/src/veridist").mkdir(parents=True)
        for relative in (
            "python/src/veridist/__init__.py",
            "CITATION.cff",
            ".zenodo.json",
            "python/pyproject.toml",
            "conda-forge-recipe/meta.yaml",
        ):
            destination = target / relative
            shutil.copyfile(ROOT / relative, destination)

    def test_repository_release_metadata_is_aligned(self) -> None:
        self.assertEqual(validate(ROOT), [])

    def test_rejects_version_drift_and_placeholder_recipe_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_metadata(root)
            zenodo_path = root / ".zenodo.json"
            zenodo = json.loads(zenodo_path.read_text("utf-8"))
            zenodo["version"] = "9.9.9"
            zenodo_path.write_text(json.dumps(zenodo), encoding="utf-8")
            recipe_path = root / "conda-forge-recipe/meta.yaml"
            recipe_path.write_text(
                re.sub(
                    r"sha256: [0-9a-f]{64}",
                    "sha256: replace-me",
                    recipe_path.read_text("utf-8"),
                ),
                encoding="utf-8",
            )
            errors = " ".join(validate(root))
            self.assertIn("Zenodo version", errors)
            self.assertIn("SHA-256", errors)

    def test_rejects_package_version_that_differs_from_pyproject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_metadata(root)
            init_path = root / "python/src/veridist/__init__.py"
            init_path.write_text(
                re.sub(
                    r'^__version__ = "[^"]+"$',
                    '__version__ = "9.9.9"',
                    init_path.read_text("utf-8"),
                    flags=re.MULTILINE,
                ),
                encoding="utf-8",
            )
            self.assertIn("__version__ differs", " ".join(validate(root)))

    def test_rejects_a_module_without_a_literal_version(self) -> None:
        for source in ("def broken(:\n", "__version__ = compute()\n", "other = '2.1.0'\n"):
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._copy_metadata(root)
                (root / "python/src/veridist/__init__.py").write_text(source, encoding="utf-8")
                self.assertIn("__version__ differs", " ".join(validate(root)))

    def test_accepts_an_annotated_literal_version(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_metadata(root)
            version = tomllib.loads((ROOT / "python/pyproject.toml").read_text("utf-8"))[
                "project"
            ]["version"]
            (root / "python/src/veridist/__init__.py").write_text(
                f'__version__: str = "{version}"\n', encoding="utf-8"
            )
            self.assertEqual(validate(root), [])

    def test_rejects_conda_license_that_differs_from_pyproject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_metadata(root)
            recipe_path = root / "conda-forge-recipe/meta.yaml"
            recipe_path.write_text(
                re.sub(
                    r"^(\s*)license: \S+$",
                    lambda match: f"{match.group(1)}license: MIT",
                    recipe_path.read_text("utf-8"),
                    flags=re.MULTILINE,
                ),
                encoding="utf-8",
            )
            self.assertIn("recipe license differs", " ".join(validate(root)))

    def test_rejects_conda_recipe_without_a_license_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_metadata(root)
            recipe_path = root / "conda-forge-recipe/meta.yaml"
            recipe_path.write_text(
                re.sub(
                    r"^[ \t]*license: \S+\n",
                    "",
                    recipe_path.read_text("utf-8"),
                    flags=re.MULTILINE,
                ),
                encoding="utf-8",
            )
            self.assertIn("recipe license differs", " ".join(validate(root)))

    def test_rejects_a_built_sdist_that_differs_from_the_recipe_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "veridist.tar.gz"
            artifact.write_bytes(b"different")
            self.assertIn("built source distribution", " ".join(validate(ROOT, artifact)))


if __name__ == "__main__":
    unittest.main()
