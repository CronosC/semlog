"""Properties that have to hold of every build, whatever its settings."""

from __future__ import annotations

import re
from collections import defaultdict

import pytest

from semanticlog.core import vocabulary as vocab
from tests.onto.scenarios.test_golden import build_lines
from tests.onto.support.configurations import CONFIGURATIONS

_DECLARATION = re.compile(
    r"^Declaration\((Class|ObjectProperty|DataProperty)\(:([^)]+)\)\)$"
)
_CLASS_ASSERTION = re.compile(r"^ClassAssertion\(:([^\s]+) :([^\s)]+)\)$")
_OBJECT_ASSERTION = re.compile(
    r"^ObjectPropertyAssertion\(:([^\s]+) :([^\s]+) :([^\s)]+)\)$"
)
_SUB_CLASS = re.compile(r"^SubClassOf\(:([^\s]+) :([^\s)]+)\)$")


class Emitted:
    def __init__(self, lines: list[str]) -> None:
        self.classes: set[str] = set()
        self.properties: set[str] = set()
        self.individuals: set[str] = set()
        self.types: dict[str, set[str]] = defaultdict(set)
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.relation_endpoints: set[str] = set()

        for line in lines:
            if m := _DECLARATION.match(line):
                kind, name = m.groups()
                (self.classes if kind == "Class" else self.properties).add(name)
            elif m := _CLASS_ASSERTION.match(line):
                cls, individual = m.groups()
                self.individuals.add(individual)
                self.types[individual].add(cls)
            elif m := _OBJECT_ASSERTION.match(line):
                _, subject, obj = m.groups()
                self.relation_endpoints.update({subject, obj})
            elif m := _SUB_CLASS.match(line):
                child, parent = m.groups()
                self.parents[child].add(parent)

    def ancestors(self, name: str) -> set[str]:
        seen: set[str] = set()
        stack = list(self.parents.get(name, ()))
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(self.parents.get(current, ()))
        return seen


@pytest.fixture(scope="module")
def emitted() -> dict[str, Emitted]:
    import tempfile
    from pathlib import Path

    built = {}
    for name, factory in CONFIGURATIONS.items():
        with tempfile.TemporaryDirectory() as tmp:
            built[name] = Emitted(build_lines(factory(), Path(tmp)))
    return built


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_no_iri_is_both_a_class_and_an_individual(name, emitted):
    e = emitted[name]
    both = e.classes & e.individuals
    assert not both, (
        f"{name}: {sorted(both)[:10]} used as both a class and an individual"
    )


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_no_iri_is_both_a_class_and_a_property(name, emitted):
    e = emitted[name]
    both = e.classes & e.properties
    assert not both, (
        f"{name}: {sorted(both)[:10]} declared as both a class and a property"
    )


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_every_relation_endpoint_exists_as_an_asserted_individual(name, emitted):
    e = emitted[name]
    dangling = e.relation_endpoints - e.individuals
    assert not dangling, (
        f"{name}: {len(dangling)} relation endpoint(s) are not asserted anywhere: "
        f"{sorted(dangling)[:10]}"
    )


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_no_individual_is_both_an_entity_and_an_attribute(name, emitted):
    e = emitted[name]
    roots = {vocab.OBJECT, vocab.EVENT}
    attribute_roots = {vocab.ATTRIBUTE, vocab.OBJECT_ATTRIBUTE, vocab.EVENT_ATTRIBUTE}

    offenders = []
    for individual, classes in e.types.items():
        lineage = set(classes)
        for cls in classes:
            lineage |= e.ancestors(cls)
        if lineage & roots and lineage & attribute_roots:
            offenders.append(individual)

    assert not offenders, (
        f"{name}: {len(offenders)} individual(s) are both an entity and an "
        f"attribute: {sorted(offenders)[:10]}"
    )


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_every_asserted_class_is_declared(name, emitted):
    e = emitted[name]
    used = {cls for classes in e.types.values() for cls in classes}
    undeclared = used - e.classes
    assert not undeclared, (
        f"{name}: asserted into undeclared class(es) {sorted(undeclared)}"
    )
