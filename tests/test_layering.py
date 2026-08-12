"""The import rules between the sub-packages.

Every module is parsed and its semlog imports are checked against the
layer it sits in. This is what keeps the dependency direction one way.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "semanticlog"

ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "data": {"core"},
    "enrich": {"core", "data"},
    "onto": {"core", "data", "enrich"},
    "cli": {"core", "data", "enrich", "onto"},
}

EXEMPT = {PACKAGE_ROOT / "__init__.py"}


def _own_package(path: Path) -> str | None:
    parts = path.relative_to(PACKAGE_ROOT).parts
    return parts[0] if len(parts) > 1 else None


def _imported_packages(tree: ast.AST) -> set[tuple[str, int]]:
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "semanticlog" and len(parts) > 1:
                    found.add((parts[1], node.lineno))
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            parts = node.module.split(".")
            if parts[0] == "semanticlog" and len(parts) > 1:
                found.add((parts[1], node.lineno))
    return found


def _modules() -> list[Path]:
    return sorted(
        p
        for p in PACKAGE_ROOT.rglob("*.py")
        if p not in EXEMPT and "__pycache__" not in p.parts
    )


def test_the_package_has_the_sub_packages_the_layering_names():
    present = {p.name for p in PACKAGE_ROOT.iterdir() if (p / "__init__.py").is_file()}
    assert set(ALLOWED) <= present, f"missing: {set(ALLOWED) - present}"


@pytest.mark.parametrize(
    "module", _modules(), ids=lambda p: str(p.relative_to(PACKAGE_ROOT))
)
def test_module_imports_stay_within_its_layer(module: Path):
    package = _own_package(module)
    if package is None or package not in ALLOWED:
        pytest.skip(f"{module.name} is not inside a layered sub-package")

    permitted = ALLOWED[package] | {package}
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))

    violations = [
        f"line {line}: imports semanticlog.{imported}"
        for imported, line in sorted(_imported_packages(tree), key=lambda t: t[1])
        if imported not in permitted
    ]
    assert not violations, (
        f"{module.relative_to(PACKAGE_ROOT)} is in '{package}', which may import "
        f"{sorted(permitted)}:\n" + "\n".join(f"  {v}" for v in violations)
    )


def test_core_imports_nothing_from_the_package():
    offenders = {}
    for module in _modules():
        if _own_package(module) != "core":
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        imported = {name for name, _ in _imported_packages(tree)}
        if imported:
            offenders[module.name] = sorted(imported)
    assert not offenders, f"core must be a leaf, but: {offenders}"
