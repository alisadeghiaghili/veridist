"""Keep tests that use thread-pool libraries out of forked copies of a process that used them.

Mutation testing forks one child per mutant from a runner that has already run the
whole suite in the same process.  Polars and DuckDB start native worker threads on
first use, and a thread does not survive ``fork``: the child inherits a pool whose
workers no longer exist, so the first query that waits for one never returns.  The
runner sees a child that neither passes nor fails until its time limit expires, and
reports the mutant as timed out instead of killed or survived.

The remedy is to give such a child a fresh interpreter at the moment it first needs
one.  Before the first test that uses Polars or DuckDB, the forked child replaces
itself (``execve`` keeps the process id, so the runner still waits for, and limits,
the same process) with a new ``pytest`` that runs the tests that remain.  Every test
that ran before is already known to have passed, because the runner stops at the
first failure.  Those tests are also moved behind all others, so that most children
are decided before then.  A process that was not forked is neither replaced nor
reordered.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

GUARD_VARIABLE = "VERIDIST_TEST_FRESH_PROCESS"
"""Set in the replacement process, so that it is never replaced again."""

EXPECTED_PACKAGE_VARIABLE = "VERIDIST_TEST_EXPECTED_PACKAGE"
"""The file the package must be imported from; a replacement that finds another one stops."""

REFUSED_EXIT_CODE = 70
"""Exit code of a replacement process that imported a different copy of the package."""

_LIBRARIES = re.compile(r"polars|duckdb|scale_frame_fixtures")

forked = False
"""Whether this process was created by ``fork`` after this module was imported."""

_uses_libraries: dict[Path, bool] = {}


def _mark_forked() -> None:
    global forked
    forked = True


def install() -> None:
    """Record, in every forked child, that it is one."""

    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=_mark_forked)


def uses_thread_pool_libraries(path: Path) -> bool:
    """Whether the test module at ``path`` names Polars, DuckDB or their shared fixtures."""

    if path not in _uses_libraries:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            text = ""
        _uses_libraries[path] = _LIBRARIES.search(text) is not None
    return _uses_libraries[path]


def defer_library_tests(items: list[Any]) -> None:
    """Move the tests that use those libraries behind all others, keeping each group's order.

    In a forked child, every test that can end the run (by failing) before the first
    such test saves the cost of a fresh interpreter.  Does nothing in a process that
    was not forked, so that an ordinary run keeps its order.
    """

    if forked:
        items.sort(key=lambda item: uses_thread_pool_libraries(item.path))


def must_restart(path: Path, environ: Mapping[str, str]) -> bool:
    """Whether the test module at ``path`` must run in a fresh process, not this one."""

    return (
        forked
        and os.name == "posix"
        and environ.get(GUARD_VARIABLE) != "1"
        and uses_thread_pool_libraries(path)
    )


def fresh_process_arguments(
    invocation_args: Sequence[str], selected: Sequence[str], remaining: Sequence[str]
) -> list[str]:
    """The arguments of a ``pytest`` that runs ``remaining`` with the options of this run.

    ``selected`` are the test selections of this run, which ``remaining`` replaces.
    """

    chosen = set(selected)
    options = [argument for argument in invocation_args if argument not in chosen]
    return [*options, *remaining]


def fresh_process_environment(environ: Mapping[str, str], package_file: str) -> dict[str, str]:
    """The environment of the replacement: this one, plus what makes it find the same code."""

    paths = [entry for entry in sys.path if entry]
    inherited = environ.get("PYTHONPATH")
    if inherited:
        paths.append(inherited)
    return {
        **environ,
        "PYTHONPATH": os.pathsep.join(paths),
        GUARD_VARIABLE: "1",
        EXPECTED_PACKAGE_VARIABLE: package_file,
    }


def check_expected_package(package_file: str, environ: Mapping[str, str]) -> None:
    """Stop a replacement process that imported the package from somewhere else.

    A replacement that ran the tests against another copy of the code would report
    every mutant as surviving, silently.  It exits with ``REFUSED_EXIT_CODE`` instead.
    """

    expected = environ.get(EXPECTED_PACKAGE_VARIABLE)
    if expected is not None and os.path.realpath(expected) != os.path.realpath(package_file):
        sys.stderr.write(
            "a fresh test process imported the package from a different location than "
            "the process it replaced\n"
        )
        sys.stderr.flush()
        os._exit(REFUSED_EXIT_CODE)


def restart(arguments: Sequence[str], environ: Mapping[str, str]) -> None:
    """Replace this process by ``python -m pytest`` with ``arguments``; it does not return.

    It returns only by raising ``OSError`` when the replacement cannot start, and the
    caller then goes on in this process.
    """

    sys.stdout.flush()
    sys.stderr.flush()
    command = [sys.executable, "-m", "pytest", *arguments]
    os.execve(sys.executable, command, dict(environ))
