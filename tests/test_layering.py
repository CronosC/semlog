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
    "onto": {"core", "data"},
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


def test_core_imports_nothing_from_the_other_layers():
    """`core` is a leaf of the layer graph. Its own modules may use each other."""
    offenders = {}
    for module in _modules():
        if _own_package(module) != "core":
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        imported = {name for name, _ in _imported_packages(tree)} - {"core"}
        if imported:
            offenders[module.name] = sorted(imported)
    assert not offenders, f"core must be a leaf, but: {offenders}"


ONTO_ROOT = PACKAGE_ROOT / "onto"

# Which parts of `onto` each part may import. A module inside a sub-package
# counts as that sub-package, and a part may always import itself. `onto` is
# the layer where a build is assembled, so the order runs settings, then the
# things that read them, then the builder that drives all of it.
ONTO_ALLOWED: dict[str, set[str]] = {
    "settings": set(),
    "reason": set(),
    "emitted": set(),
    "inputs": {"settings"},
    "result": {"settings"},
    "annotations": {"settings"},
    "audit": {"settings", "inputs"},
    "sinks": set(),
    "emitters": {"settings", "inputs", "sinks", "annotations", "emitted"},
    "builder": {
        "settings",
        "inputs",
        "sinks",
        "emitters",
        "emitted",
        "annotations",
        "audit",
        "result",
    },
    "pipeline": {"settings", "inputs", "sinks", "builder", "result"},
}


def _onto_part(path: Path) -> str:
    """Return which part of `onto` a module belongs to."""
    return path.relative_to(ONTO_ROOT).parts[0].removesuffix(".py")


def _imported_onto_parts(tree: ast.AST) -> set[tuple[str, int]]:
    """Return the parts of `onto` a module imports, with the line each is on."""
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            parts = name.split(".")
            if parts[:2] == ["semanticlog", "onto"] and len(parts) > 2:
                found.add((parts[2], node.lineno))
    return found


def _onto_modules() -> list[Path]:
    return [
        p for p in _modules() if p.is_relative_to(ONTO_ROOT) and p.name != "__init__.py"
    ]


def test_every_part_of_onto_has_an_import_rule():
    parts = {_onto_part(p) for p in _onto_modules()}
    assert parts == set(ONTO_ALLOWED), (
        "ONTO_ALLOWED and the modules in onto/ disagree. Missing a rule: "
        f"{sorted(parts - set(ONTO_ALLOWED))}. Rule for nothing: "
        f"{sorted(set(ONTO_ALLOWED) - parts)}."
    )


@pytest.mark.parametrize(
    "module", _onto_modules(), ids=lambda p: str(p.relative_to(ONTO_ROOT))
)
def test_onto_module_imports_stay_within_its_part(module: Path):
    """The direction inside `onto`, which the sub-package rule does not cover.

    Without this, a sink could reach back into an emitter and the boundary the
    sink exists to be would stop being one.
    """
    part = _onto_part(module)
    permitted = ONTO_ALLOWED[part] | {part}
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))

    violations = [
        f"line {line}: imports semanticlog.onto.{imported}"
        for imported, line in sorted(_imported_onto_parts(tree), key=lambda t: t[1])
        if imported not in permitted
    ]
    assert not violations, (
        f"{module.relative_to(ONTO_ROOT)} is part of '{part}', which may import "
        f"{sorted(permitted)}:\n" + "\n".join(f"  {v}" for v in violations)
    )
