"""DS-10 contract: every typed-failure construction in `src/` uses safe keys.

`_freeze_context_value` redacts a failure context by key *name* only -- it
never inspects values (see its docstring and `KNOWN_LIMITS.md`'s
`CONTEXT-REDACTION` entry). That is only safe because every construction
site in this package is reviewed to avoid a forbidden key part. This test
statically walks every source file under `src/` and asserts that nobody
regresses that discipline by introducing a construction whose context (or,
for `CsvLifetimeAdapterError`, whose keyword arguments, which become its
fixed context keys) uses a forbidden part.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from veridist.engine.errors import _SENSITIVE_CONTEXT_KEY_PARTS

PYTHON_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PYTHON_ROOT / "src" / "veridist"

#: Every failure-class constructor whose context must be screened. Each takes
#: `(code, context=None)` except `CsvLifetimeAdapterError`, whose keyword
#: arguments (`reason=`, `record_offset=`) are exactly its fixed context keys.
SCREENED_CONSTRUCTORS = frozenset(
    {
        "EngineContractError",
        "DeliveryContractError",
        "DataSourceCapabilityError",
        "PassBudgetError",
        "StreamSourceError",
        "CsvLifetimeAdapterError",
    }
)


def _string_dict_keys(node: ast.Dict) -> list[str]:
    keys: list[str] = []
    for key in node.keys:
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            keys.append(key.value)
    return keys


def _dict_literals_assigned_to(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> dict[str, list[ast.Dict]]:
    """Map each locally assigned name to every dict literal assigned to it.

    This is a deliberately shallow, same-function, no-control-flow lookup: it
    does not attempt real data-flow analysis. Its only job is to let a call
    like ``EngineContractError(code, context)`` resolve back to the literal
    ``context = {...}`` a few lines above it, which is this codebase's only
    indirection pattern (confirmed by inspection of every call site).
    """

    assigned: dict[str, list[ast.Dict]] = {}
    for node in ast.walk(function):
        target_name: str | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                target_name = target.id
                value = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        if target_name is None or value is None:
            continue
        if isinstance(value, ast.IfExp):
            literal_dicts = [
                item for item in (value.body, value.orelse) if isinstance(item, ast.Dict)
            ]
        elif isinstance(value, ast.Dict):
            literal_dicts = [value]
        else:
            literal_dicts = []
        if literal_dicts:
            assigned.setdefault(target_name, []).extend(literal_dicts)
    return assigned


def _enclosing_function(
    tree: ast.Module, call: ast.Call
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            end = node.end_lineno or call.lineno
            if node.lineno <= call.lineno <= end and (best is None or node.lineno > best.lineno):
                best = node
    return best


def _candidate_keys_for_call(tree: ast.Module, call: ast.Call) -> set[str]:
    keys: set[str] = set()
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        if keyword.arg == "context" and isinstance(keyword.value, ast.Dict):
            keys.update(_string_dict_keys(keyword.value))
        else:
            # Every keyword name is a candidate: for `CsvLifetimeAdapterError`
            # specifically, `reason=`/`record_offset=` are literally the
            # context keys its constructor builds.
            keys.add(keyword.arg)
    if len(call.args) >= 2:
        context_arg = call.args[1]
        if isinstance(context_arg, ast.Dict):
            keys.update(_string_dict_keys(context_arg))
        elif isinstance(context_arg, ast.Name):
            function = _enclosing_function(tree, call)
            if function is not None:
                assigned = _dict_literals_assigned_to(function)
                for literal in assigned.get(context_arg.id, ()):
                    keys.update(_string_dict_keys(literal))
    return keys


def _forbidden_parts(key: str) -> set[str]:
    parts = {part for part in key.casefold().split("_") if part}
    return parts & _SENSITIVE_CONTEXT_KEY_PARTS


class ContextKeyRedactionDisciplineTests(unittest.TestCase):
    def test_every_screened_constructor_in_src_avoids_forbidden_key_parts(self) -> None:
        self.assertTrue(SRC_ROOT.is_dir())
        violations: list[str] = []
        scanned_constructors: set[str] = set()
        for path in sorted(SRC_ROOT.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not isinstance(node.func, ast.Name):
                    continue
                if node.func.id not in SCREENED_CONSTRUCTORS:
                    continue
                scanned_constructors.add(node.func.id)
                for key in _candidate_keys_for_call(tree, node):
                    forbidden = _forbidden_parts(key)
                    if forbidden:
                        violations.append(
                            f"{path.relative_to(PYTHON_ROOT)}:{node.lineno}: "
                            f"{node.func.id} context key {key!r} contains "
                            f"forbidden part(s) {sorted(forbidden)}"
                        )

        self.assertEqual(violations, [])
        # The scan itself must actually have looked at every listed class, so
        # a future rename or removal cannot silently empty this test out.
        self.assertEqual(scanned_constructors, SCREENED_CONSTRUCTORS)


if __name__ == "__main__":
    unittest.main()
