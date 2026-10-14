"""The import rules between the sub-packages, and between the parts of `onto`."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "semanticlog"
ONTO_ROOT = PACKAGE_ROOT / "onto"

ALLOWED: dict[str, set[str]] = {
    "core": set(),
    "data": {"core"},
    "enrich": {"core", "data"},
    "onto": {"core", "data"},
    "cli": {"core", "data", "enrich", "onto"},
}

# Which parts of `onto` each part may import. A module inside a sub-package
# counts as that sub-package, and a part may always import itself.
ONTO_ALLOWED: dict[str, set[str]] = {
    "settings": set(),
    "reason": {"sinks"},
    "record": set(),
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


def _modules(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(module: Path, prefix: tuple[str, ...]) -> list[tuple[str, int]]:
    """Return the part named right after `prefix` in each absolute import, with its line."""
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        else:
            continue
        for name in names:
            parts = tuple(name.split("."))
            if parts[: len(prefix)] == prefix and len(parts) > len(prefix):
                found.append((parts[len(prefix)], node.lineno))
    return sorted(found, key=lambda pair: pair[1])


def test_every_module_imports_only_the_layers_its_layer_may():
    present = {p.name for p in PACKAGE_ROOT.iterdir() if (p / "__init__.py").is_file()}
    violations = [
        f"missing sub-package {name}" for name in sorted(set(ALLOWED) - present)
    ]
    for module in _modules(PACKAGE_ROOT):
        parts = module.relative_to(PACKAGE_ROOT).parts
        if len(parts) == 1:
            continue
        layer = parts[0]
        permitted = ALLOWED.get(layer, set()) | {layer}
        violations += [
            f"{module.relative_to(PACKAGE_ROOT)}:{line} imports semanticlog.{imported}"
            for imported, line in _imports(module, ("semanticlog",))
            if imported not in permitted
        ]

    assert violations == [], "\n".join(violations)


def test_every_onto_module_imports_only_the_parts_its_part_may():
    violations = []
    for module in _modules(ONTO_ROOT):
        if module.parent == ONTO_ROOT and module.name == "__init__.py":
            continue
        part = module.relative_to(ONTO_ROOT).parts[0].removesuffix(".py")
        if part not in ONTO_ALLOWED:
            violations.append(f"onto/{part} has no entry in ONTO_ALLOWED")
            continue
        permitted = ONTO_ALLOWED[part] | {part}
        violations += [
            f"onto/{module.relative_to(ONTO_ROOT)}:{line} imports semanticlog.onto.{imported}"
            for imported, line in _imports(module, ("semanticlog", "onto"))
            if imported not in permitted
        ]
    present = {
        p.name.removesuffix(".py")
        for p in ONTO_ROOT.iterdir()
        if p.name != "__init__.py"
        and (p.suffix == ".py" or (p / "__init__.py").is_file())
    }
    violations += [
        f"ONTO_ALLOWED names {part}, which onto does not have"
        for part in sorted(set(ONTO_ALLOWED) - present)
    ]

    assert violations == [], "\n".join(violations)
