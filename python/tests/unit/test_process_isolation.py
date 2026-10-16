"""The helpers that give a forked test process a fresh interpreter before Polars or DuckDB."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests import process_isolation as isolation


class LibraryDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.addCleanup(isolation._uses_libraries.clear)

    def module(self, name: str, text: str) -> Path:
        path = self.directory / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_a_module_that_names_polars_or_duckdb_is_detected(self) -> None:
        for text in ("import polars as pl\n", "import duckdb\n", "from x import polars_frame\n"):
            with self.subTest(text=text):
                path = self.module("test_a.py", text)
                isolation._uses_libraries.clear()
                self.assertTrue(isolation.uses_thread_pool_libraries(path))

    def test_the_shared_fixtures_of_those_libraries_are_detected(self) -> None:
        path = self.module("test_b.py", "from tests.unit.scale_frame_fixtures import code_of\n")
        self.assertTrue(isolation.uses_thread_pool_libraries(path))

    def test_a_module_that_starts_executors_is_detected(self) -> None:
        for text in (
            "from veridist.scale._executors import ThreadExecutor\n",
            "executor = ProcessExecutor(2)\n",
            "from concurrent.futures import ThreadPoolExecutor\n",
            "from concurrent.futures import ProcessPoolExecutor\n",
            "from tests.unit.scale_executor_fixtures import NoLeaksTestCase\n",
        ):
            with self.subTest(text=text):
                path = self.module("test_g.py", text)
                isolation._uses_libraries.clear()
                self.assertTrue(isolation.uses_thread_pool_libraries(path))

    def test_the_executor_tests_of_this_suite_are_detected(self) -> None:
        unit = Path(__file__).resolve().parent
        for name in (
            "test_scale_executors.py",
            "test_scale_executor_fit.py",
            "test_scale_executor_processes.py",
            "test_scale_spool_parallel.py",
        ):
            with self.subTest(module=name):
                self.assertTrue(isolation.uses_thread_pool_libraries(unit / name))

    def test_modules_that_only_use_the_task_pieces_are_not_detected(self) -> None:
        unit = Path(__file__).resolve().parent
        for name in ("test_scale_tasks.py", "test_scale_parallel_sources.py"):
            with self.subTest(module=name):
                self.assertFalse(isolation.uses_thread_pool_libraries(unit / name))

    def test_other_modules_and_unreadable_files_are_not(self) -> None:
        plain = self.module("test_c.py", "import numpy\nimport pyarrow\n")
        self.assertFalse(isolation.uses_thread_pool_libraries(plain))
        self.assertFalse(isolation.uses_thread_pool_libraries(self.directory / "missing.py"))
        binary = self.directory / "test_d.py"
        binary.write_bytes(b"\xff\xfe\x00polars")
        self.assertFalse(isolation.uses_thread_pool_libraries(binary))

    def test_the_answer_is_remembered(self) -> None:
        path = self.module("test_e.py", "import polars\n")
        self.assertTrue(isolation.uses_thread_pool_libraries(path))
        path.write_text("import numpy\n", encoding="utf-8")
        self.assertTrue(isolation.uses_thread_pool_libraries(path))


class RestartDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.path = Path("test_f.py")
        patcher = mock.patch.dict(isolation._uses_libraries, {self.path: True})
        patcher.start()
        self.addCleanup(patcher.stop)

    def decide(self, *, forked: bool, name: str, environ: dict[str, str]) -> bool:
        with mock.patch.object(isolation, "forked", forked), mock.patch.object(os, "name", name):
            return isolation.must_restart(self.path, environ)

    def test_only_a_forked_posix_process_that_was_not_replaced_restarts(self) -> None:
        self.assertTrue(self.decide(forked=True, name="posix", environ={}))
        self.assertFalse(self.decide(forked=False, name="posix", environ={}))
        self.assertFalse(self.decide(forked=True, name="nt", environ={}))
        guarded = {isolation.GUARD_VARIABLE: "1"}
        self.assertFalse(self.decide(forked=True, name="posix", environ=guarded))

    def test_a_module_without_those_libraries_never_restarts(self) -> None:
        other = Path("test_g.py")
        with (
            mock.patch.dict(isolation._uses_libraries, {other: False}),
            mock.patch.object(isolation, "forked", True),
            mock.patch.object(os, "name", "posix"),
        ):
            self.assertFalse(isolation.must_restart(other, {}))


class OrderingTests(unittest.TestCase):
    class Item:
        def __init__(self, name: str, path: Path) -> None:
            self.name, self.path = name, path

    def items(self) -> list[Item]:
        uses, plain = Path("test_uses.py"), Path("test_plain.py")
        return [
            self.Item("a", uses),
            self.Item("b", plain),
            self.Item("c", uses),
            self.Item("d", plain),
        ]

    def names(self, forked: bool) -> list[str]:
        items = self.items()
        patched = {Path("test_uses.py"): True, Path("test_plain.py"): False}
        with (
            mock.patch.dict(isolation._uses_libraries, patched),
            mock.patch.object(isolation, "forked", forked),
        ):
            isolation.defer_library_tests(items)
        return [item.name for item in items]

    def test_a_forked_process_runs_the_other_tests_first_and_keeps_each_group_in_order(
        self,
    ) -> None:
        self.assertEqual(self.names(forked=True), ["b", "d", "a", "c"])

    def test_an_ordinary_process_keeps_its_order(self) -> None:
        self.assertEqual(self.names(forked=False), ["a", "b", "c", "d"])


class ReplacementCommandTests(unittest.TestCase):
    def test_the_selection_is_replaced_and_every_option_is_kept(self) -> None:
        arguments = isolation.fresh_process_arguments(
            ["--rootdir=.", "--tb=native", "-x", "-q", "-p", "no:randomly", "tests/a.py::T::one"],
            ["tests/a.py::T::one"],
            ["tests/b.py::U::two", "tests/c.py::V::three"],
        )
        self.assertEqual(
            arguments,
            [
                "--rootdir=.",
                "--tb=native",
                "-x",
                "-q",
                "-p",
                "no:randomly",
                "tests/b.py::U::two",
                "tests/c.py::V::three",
            ],
        )

    def test_a_run_without_a_selection_keeps_every_argument(self) -> None:
        self.assertEqual(
            isolation.fresh_process_arguments(["-q"], [], ["tests/a.py::T::one"]),
            ["-q", "tests/a.py::T::one"],
        )

    def test_the_environment_carries_the_import_path_and_the_markers(self) -> None:
        with mock.patch.object(sys, "path", ["", "/work/mutants/src", "/work/mutants"]):
            environment = isolation.fresh_process_environment(
                {"PYTHONPATH": "/extra", "KEEP": "1"}, "/work/mutants/src/veridist/__init__.py"
            )
        self.assertEqual(
            environment,
            {
                "PYTHONPATH": os.pathsep.join(["/work/mutants/src", "/work/mutants", "/extra"]),
                "KEEP": "1",
                isolation.GUARD_VARIABLE: "1",
                isolation.EXPECTED_PACKAGE_VARIABLE: "/work/mutants/src/veridist/__init__.py",
            },
        )

    def test_the_environment_of_a_run_without_a_python_path(self) -> None:
        with mock.patch.object(sys, "path", ["/work/src"]):
            environment = isolation.fresh_process_environment({}, "/work/src/veridist/__init__.py")
        self.assertEqual(environment["PYTHONPATH"], "/work/src")


class ExpectedPackageTests(unittest.TestCase):
    def test_a_matching_or_unset_expectation_lets_the_process_go_on(self) -> None:
        here = os.path.abspath(__file__)
        with mock.patch.object(os, "_exit") as leave:
            isolation.check_expected_package(here, {})
            isolation.check_expected_package(here, {isolation.EXPECTED_PACKAGE_VARIABLE: here})
        leave.assert_not_called()

    def test_another_location_stops_the_process(self) -> None:
        here = os.path.abspath(__file__)
        other = os.path.join(os.path.dirname(here), "elsewhere", "__init__.py")
        with mock.patch.object(os, "_exit") as leave, mock.patch.object(sys, "stderr"):
            isolation.check_expected_package(here, {isolation.EXPECTED_PACKAGE_VARIABLE: other})
        leave.assert_called_once_with(isolation.REFUSED_EXIT_CODE)


class RestartTests(unittest.TestCase):
    def test_the_process_is_replaced_by_pytest_with_the_given_arguments(self) -> None:
        with (
            mock.patch.object(os, "execve") as replace,
            mock.patch.object(sys, "stdout"),
            mock.patch.object(sys, "stderr"),
        ):
            isolation.restart(["-q", "tests/a.py::T::one"], {"A": "1"})
        replace.assert_called_once_with(
            sys.executable, [sys.executable, "-m", "pytest", "-q", "tests/a.py::T::one"], {"A": "1"}
        )

    def test_a_replacement_that_cannot_start_is_reported_to_the_caller(self) -> None:
        with (
            mock.patch.object(os, "execve", side_effect=OSError("no")),
            mock.patch.object(sys, "stdout"),
            mock.patch.object(sys, "stderr"),
            self.assertRaises(OSError),
        ):
            isolation.restart([], {})

    def test_install_records_forks_in_children_only(self) -> None:
        with mock.patch.object(os, "register_at_fork", create=True) as register:
            isolation.install()
        register.assert_called_once_with(after_in_child=isolation._mark_forked)
        with mock.patch.object(isolation, "forked", False):
            isolation._mark_forked()
            self.assertTrue(isolation.forked)


if __name__ == "__main__":
    unittest.main()
