"""Executable hygiene constraints for the import-time package boundary."""

from __future__ import annotations

import ast
import importlib
import sys
import tomllib
import unittest
from pathlib import Path

PYTHON_ROOT = Path(__file__).parents[2]
SOURCE_ROOT = PYTHON_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "veridist"
WORKFLOW_PATH = PYTHON_ROOT.parent / ".github" / "workflows" / "v1-ci.yml"


class PackageHygieneTests(unittest.TestCase):
    """Keep the foundation importable without optional dependencies or side effects."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(SOURCE_ROOT))

    @classmethod
    def tearDownClass(cls) -> None:
        sys.path.remove(str(SOURCE_ROOT))

    def test_package_and_declared_namespaces_import_with_stdlib_only(self) -> None:
        package = importlib.import_module("veridist")
        self.assertEqual(package.__version__, "2.0.0")
        for namespace in ("domain", "statistics", "families", "engine"):
            importlib.import_module(f"veridist.{namespace}")

    def test_py_typed_marker_is_shipped_in_source_tree(self) -> None:
        self.assertTrue((PACKAGE_ROOT / "py.typed").is_file())

    def test_source_has_no_eager_optional_imports_or_import_time_side_effects(self) -> None:
        forbidden_import_roots = {"numpy", "pandas", "scipy", "polars", "dask", "ray"}
        for source_file in PACKAGE_ROOT.rglob("*.py"):
            tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(
                            alias.name.split(".")[0], forbidden_import_roots, source_file
                        )
                if isinstance(node, ast.ImportFrom) and node.module is not None:
                    self.assertNotIn(node.module.split(".")[0], forbidden_import_roots, source_file)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotEqual(node.func.id, "print", source_file)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, {"basicConfig", "seed"}, source_file)

    def test_test_extra_provides_the_workflow_coverage_plugin(self) -> None:
        project = tomllib.loads((PYTHON_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        test_extra = project["project"]["optional-dependencies"]["test"]
        self.assertIn("pytest-cov>=6,<8", test_extra)

    def test_project_metadata_declares_human_authorship(self) -> None:
        project = tomllib.loads((PYTHON_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(
            project["project"]["authors"],
            [
                {
                    "name": "Ali Sadeghi Aghili",
                    "email": "alisadeghiaghili@gmail.com",
                }
            ],
        )

    def test_project_metadata_declares_discovery_fields(self) -> None:
        project = tomllib.loads((PYTHON_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]
        self.assertEqual(
            project["keywords"],
            [
                "reliability",
                "survival-analysis",
                "lifetime-data",
                "right-censoring",
                "distribution-fitting",
                "maximum-likelihood",
            ],
        )
        self.assertEqual(
            project["classifiers"],
            [
                "Intended Audience :: Science/Research",
                "Topic :: Scientific/Engineering :: Mathematics",
                "Programming Language :: Python :: 3 :: Only",
                "Programming Language :: Python :: 3.11",
                "Programming Language :: Python :: 3.12",
                "Programming Language :: Python :: 3.13",
                "Programming Language :: Python :: 3.14",
                "Operating System :: OS Independent",
                "Typing :: Typed",
            ],
        )

    def test_license_and_status_classifiers_are_left_to_the_owner(self) -> None:
        project = tomllib.loads((PYTHON_ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]
        for classifier in project["classifiers"]:
            self.assertFalse(classifier.startswith(("License ::", "Development Status ::")))

    def test_workflow_lints_documentation_python_sources(self) -> None:
        workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
        self.assertIn("python -m ruff check src tests docs tools", workflow)


if __name__ == "__main__":
    unittest.main()
