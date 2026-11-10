"""Every `Raises:` block names the classes the function raises itself.

An entry with no match in the body is left alone, since a block may name what
a callee raises.
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "semlog"

_RAISES_HEADING = "Raises:"


def _modules() -> list[Path]:
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _own_raises(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return the exception classes a function raises in its own body.

    A nested function has its own docstring, so its raises are not counted
    here. A bare `raise` re-raises whatever was caught and names no class.
    """
    found: set[str] = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node is not (
            function
        ):
            continue
        if not isinstance(node, ast.Raise) or node.exc is None:
            continue
        called = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
        if isinstance(called, ast.Name):
            found.add(called.id)
        elif isinstance(called, ast.Attribute):
            found.add(called.attr)
    return found


def _nested_raises(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """Return what the functions defined inside this one raise."""
    return {
        name
        for node in ast.walk(function)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node is not function
        for name in _own_raises(node)
    }


def _documented(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str] | None:
    """Return the classes a docstring's `Raises:` block names, or None if it has none."""
    doc = ast.get_docstring(function)
    if doc is None or _RAISES_HEADING not in doc:
        return None
    block = doc.split(_RAISES_HEADING, 1)[1]
    named: set[str] = set()
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        head, sep, _ = stripped.partition(":")
        if sep and head.isidentifier() and head[0].isupper():
            named.add(head)
    return named


def test_every_raises_block_names_what_its_function_raises():
    problems = []
    for module in _modules():
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for function in _functions(tree):
            documented = _documented(function)
            if documented is None:
                continue
            missing = sorted(
                (_own_raises(function) | _nested_raises(function)) - documented
            )
            if missing:
                problems.append(
                    f"{module.relative_to(PACKAGE_ROOT)}:{function.lineno} {function.name} "
                    f"raises {missing} but documents {sorted(documented)}"
                )

    assert problems == [], "\n".join(problems)
