"""Properties that hold of every build, run over every configuration of the matrix."""

from __future__ import annotations

import re

import pandas as pd
import pytest

from semlog.core import vocabulary as vocab
from semlog.core.enums import (
    AttributeRepresentation,
    EventOrderRepresentation,
    IndividualKind,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
)
from semlog.core.vocabulary import DEFAULT_IRI
from semlog.data.tables import RelationKind
from semlog.onto.builder import OCEDOntoBuilder
from semlog.onto.reason import Reasoner
from semlog.onto.settings import OntoBuildSettings
from semlog.onto.sinks import StreamingSink
from tests.support import stores
from tests.support.build import BACKENDS, emit
from tests.support.owlapi import (
    ontology_annotations,
    ontology_iris,
    undeclared_signature,
)
from tests.support.settings import EL_NAMES, MATRIX, NAMES, configuration

# Constructors OWL 2 EL does not have.
FORBIDDEN_IN_EL = (
    "ObjectUnionOf",
    "ObjectComplementOf",
    "ObjectAllValuesFrom",
    "ObjectMaxCardinality",
    "ObjectExactCardinality",
    "DisjointUnion",
    "InverseObjectProperties",
    "ObjectInverseOf",
    "TransitiveObjectProperty",
    "FunctionalObjectProperty",
    "InverseFunctionalObjectProperty",
    "SymmetricObjectProperty",
    "AsymmetricObjectProperty",
    "DataUnionOf",
    "DataComplementOf",
)

ENTITY_ROOTS = {vocab.OBJECT, vocab.EVENT}
ATTRIBUTE_ROOTS = {vocab.ATTRIBUTE, vocab.OBJECT_ATTRIBUTE, vocab.EVENT_ATTRIBUTE}


def test_the_matrix_builds_every_value_of_every_enum_and_boolean_setting():
    seen: dict[str, set] = {
        "profile": set(),
        "order_representations": set(),
        "inverse_representations": set(),
        "attribute_representations": set(),
        "ocel_extensions": set(),
        "derive_annotations": set(),
    }
    for factory in MATRIX.values():
        settings = factory().settings
        seen["profile"].add(settings.profile)
        seen["derive_annotations"].add(settings.derive_annotations)
        for key in (
            "order_representations",
            "inverse_representations",
            "attribute_representations",
            "ocel_extensions",
        ):
            values = getattr(settings, key)
            seen[key] |= set(values) or {None}

    assert seen == {
        "profile": set(OWLProfile),
        "order_representations": {*EventOrderRepresentation, None},
        "inverse_representations": {*InverseRepresentation, None},
        "attribute_representations": set(AttributeRepresentation),
        "ocel_extensions": {*OCELExtension, None},
        "derive_annotations": {True, False},
    }


def test_a_build_under_another_tbox_iri_writes_no_default_schema_iri(
    tiny_store, tmp_path
):
    path = tmp_path / "a.ofn"
    emit(OntoBuildSettings(tbox_iri="https://a.example/schema#"), path, data=tiny_store)

    assert vocab.TBOX_IRI not in path.read_text(encoding="utf-8")


_KIND_PREFIXES = tuple(f"{kind.value}:" for kind in IndividualKind)
_ASSERTIONS = ("ClassAssertion(", "ObjectPropertyAssertion(", "DataPropertyAssertion(")


def _individual_tokens(line: str) -> list[str]:
    """Return the individual references of one assertion line, before any literal."""
    found = []
    for token in line.rstrip(")").split(" ")[1:]:
        if token.startswith('"'):
            break
        found.append(token)
    return found


@pytest.mark.parametrize("name", NAMES)
def test_every_individual_iri_holds_its_kind_path(built, name):
    onto = built(name)
    individuals = [
        token
        for line in onto.axiom_lines
        if line.startswith(_ASSERTIONS)
        for token in _individual_tokens(line)
    ]

    assert individuals, name
    assert [t for t in individuals if not t.startswith(_KIND_PREFIXES)] == []


def _violations(onto) -> list[str]:
    """Return every way a build breaks a rule that holds for all builds."""
    found = []
    punned = (onto.classes & (onto.individuals | onto.properties)) | (
        onto.object_properties & onto.data_properties
    )
    found += [f"{name} is used in two roles" for name in sorted(punned)]

    asserted_classes = {c for classes in onto.types.values() for c in classes}
    asserted_properties = {p for p, _, _ in onto.relations} | {
        p for p, _ in onto.data_assertions
    }
    undeclared = (asserted_classes - onto.classes) | (
        asserted_properties - onto.properties
    )
    found += [f"{name} is asserted but not declared" for name in sorted(undeclared)]

    dangling = onto.relation_endpoints - onto.individuals
    found += [f"{name} is related but not asserted" for name in sorted(dangling)]

    found += [
        f"{individual} is both an entity and an attribute"
        for individual in sorted(onto.individuals)
        if (lineage := onto.lineage(individual)) & ENTITY_ROOTS
        and lineage & ATTRIBUTE_ROOTS
    ]

    found += [
        f"written {n} times: {line}"
        for line, n in sorted(onto.duplicate_axioms.items())
    ]
    return found


@pytest.mark.parametrize("name", NAMES)
def test_nothing_is_asserted_before_it_is_declared(name, recording_sink):
    config = MATRIX[name]()
    sink = recording_sink(config.settings.profile)

    OCEDOntoBuilder(DEFAULT_IRI, config.settings, sink=sink).build_ontology(
        stores.build(stores.MATRIX_STORE), config.inputs
    )

    assert sink.declaration_order_violations() == []


def test_the_declaration_order_check_finds_a_late_declaration(recording_sink):
    sink = recording_sink()

    sink.assert_individuals("some_class", ["late"], IndividualKind.OBJECT)
    sink.add_class("late", source="a class declared too late")

    assert sink.declaration_order_violations() == [("late", "some_class")]


@pytest.mark.parametrize("name", NAMES)
def test_the_build_is_a_well_formed_ontology(name, built):
    """No punning, nothing used undeclared or unasserted, no individual both an entity and an attribute, no axiom twice."""
    violations = _violations(built(name))

    assert violations == [], "\n".join(violations[:20])


def test_a_reflexive_relation_under_implicit_inverses_writes_no_axiom_twice(
    tiny_store, tmp_path
):
    """An object related to itself gives the same pair forward and reversed."""
    tiny_store.add_relation(
        RelationKind.O2O,
        "contains",
        pd.DataFrame({"a": ["o1"], "b": ["o1"]}),
        from_id="a",
        from_type=["order"],
        to_id="b",
        to_type=["order"],
    )
    config = configuration(
        case_concepts=["order"],
        order_representations=[EventOrderRepresentation.DF],
        inverse_representations=[InverseRepresentation.IMPLICIT],
    )

    onto = emit(config, tmp_path / "reflexive.ofn", data=tiny_store)

    assert ("contains", "o1", "o1") in onto.relations
    assert ("inv_contains", "o1", "o1") in onto.relations
    assert onto.duplicate_axioms == {}


@pytest.mark.parametrize("name", EL_NAMES)
def test_an_el_build_uses_no_constructor_el_lacks(name, built):
    offenders = [
        f"{constructor}: {line}"
        for line in built(name).axiom_lines
        for constructor in FORBIDDEN_IN_EL
        if re.search(rf"\b{constructor}\(", line)
    ]

    assert not offenders, "\n".join(offenders[:10])


@pytest.mark.slow
@pytest.mark.parametrize("name", NAMES)
def test_two_builds_of_one_configuration_are_identical(name, tmp_path):
    first = emit(MATRIX[name](), tmp_path / "first.ofn")
    second = emit(MATRIX[name](), tmp_path / "second.ofn")

    assert first.axiom_lines == second.axiom_lines


@pytest.mark.reasoner
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", NAMES)
def test_every_entity_in_the_signature_is_declared(name, backend, built_path):
    """The OWL API signature also holds the annotation properties and datatypes, which `Emitted` does not."""
    undeclared = undeclared_signature(built_path(name, backend))

    assert undeclared == {}


@pytest.mark.reasoner
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_build_states_its_ontology_iri_and_version(backend, built_path):
    path = built_path(NAMES[0], backend)
    ontology_iri, _ = ontology_iris(path)

    assert ontology_iri == vocab.DEFAULT_IRI.removesuffix("/")
    assert ontology_annotations(path)["buildTool"].startswith("semlog ")


@pytest.mark.reasoner
def test_the_ontology_iri_is_the_one_the_build_was_given(tmp_path):
    path = tmp_path / "named.ofn"
    config = MATRIX["dl_df"]()
    sink = StreamingSink(str(path), "https://example.org/log", config.settings.profile)
    OCEDOntoBuilder(
        "https://example.org/log", config.settings, sink=sink
    ).build_ontology(stores.build(stores.MATRIX_STORE), config.inputs)

    assert ontology_iris(path)[0] == "https://example.org/log"


@pytest.mark.reasoner
@pytest.mark.parametrize("name", NAMES)
def test_every_build_is_consistent_and_every_class_satisfiable(name, built_path):
    # Pellet reads both profiles and accepts a timestamp with microseconds.
    reasoner = Reasoner(str(built_path(name)), backend="Pellet")
    empty = sorted(
        remainder
        for owl_class in reasoner.reasoner.unsatisfiable_classes()
        if (remainder := str(owl_class.iri.get_remainder())) != "Nothing"
    )

    assert reasoner.check() is True
    assert empty == []
