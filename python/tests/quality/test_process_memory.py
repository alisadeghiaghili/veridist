"""Contracts for the process memory helper shared by the evidence collectors."""

from __future__ import annotations

import ctypes
import importlib
import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.process_memory import peak_rss_bytes

TOOLS = Path(__file__).parents[2] / "tools"
EVIDENCE_TOOLS = (
    "collect_v1_execution_evidence",
    "run_scale_csv_exponential_evidence",
    "run_log_likelihood_scale_evidence",
)


class PeakRssTests(unittest.TestCase):
    def test_reports_a_plausible_resident_set_for_this_process(self) -> None:
        value = peak_rss_bytes()
        self.assertIsInstance(value, int)
        self.assertGreater(value, 1_000_000)
        self.assertGreaterEqual(peak_rss_bytes(), value)

    def test_posix_units_differ_between_macos_and_linux(self) -> None:
        usage = SimpleNamespace(ru_maxrss=2048)
        fake = SimpleNamespace(RUSAGE_SELF=0, getrusage=lambda _who: usage)
        with patch.dict(sys.modules, {"resource": fake}):
            with patch.object(sys, "platform", "darwin"):
                self.assertEqual(peak_rss_bytes(), 2048)
            with patch.object(sys, "platform", "linux"):
                self.assertEqual(peak_rss_bytes(), 2048 * 1024)

    def test_windows_failure_is_reported_instead_of_returning_a_placeholder(self) -> None:
        def failing_getter(*_arguments: object) -> int:
            return 0

        windll = SimpleNamespace(
            psapi=SimpleNamespace(GetProcessMemoryInfo=failing_getter),
            kernel32=SimpleNamespace(GetCurrentProcess=lambda: 1),
        )
        with (
            patch.object(sys, "platform", "win32"),
            patch.object(ctypes, "windll", windll, create=True),
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot obtain Windows process RSS"):
                peak_rss_bytes()


class SharedHelperUsageTests(unittest.TestCase):
    def test_each_evidence_tool_uses_the_shared_helper_and_no_private_copy(self) -> None:
        for name in EVIDENCE_TOOLS:
            with self.subTest(tool=name):
                source = (TOOLS / f"{name}.py").read_text(encoding="utf-8")
                self.assertIn("peak_rss_bytes", source)
                self.assertNotIn("def _rss_bytes", source)
                self.assertNotIn("GetProcessMemoryInfo", source)
                self.assertNotIn("import ctypes", source)
                module = importlib.import_module(f"tools.{name}")
                defined_in = inspect.getsourcefile(module.peak_rss_bytes)
                assert defined_in is not None
                self.assertEqual(Path(defined_in).resolve(), TOOLS.resolve() / "process_memory.py")


if __name__ == "__main__":
    unittest.main()
