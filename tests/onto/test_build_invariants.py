"""Properties that hold of every build, whatever its settings.

These are the checks that say the emitted file is a usable ontology rather
than a plausible-looking one. Each runs over every configuration in the
matrix, against the build `built` already ran.
"""

from __future__ import annotations

import re

import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import OWLProfile
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.sinks import StreamingSink
from tests.conftest import BACKENDS, emit
from tests.support import stores
from tests.support.owlapi import (
    ontology_annotations,
    ontology_iris,
    undeclared_signature,
)
from tests.support.settings import CONFIGURATIONS, EL_NAMES, NAMES

# Constructors OWL 2 EL does not have. An EL build that emits one of these has
# left the profile it promised to stay inside.
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


@pytest.mark.parametrize("name", NAMES)
def test_no_name_is_both_a_class_and_an_individual(name, built):
    onto = built(name)
    both = onto.classes & onto.individuals
    assert not both, (
        f"{name}: {sorted(both)[:10]} used as both a class and an individual"
    )


@pytest.mark.parametrize("name", NAMES)
def test_no_name_is_both_a_class_and_a_property(name, built):
    onto = built(name)
    both = onto.classes & onto.properties
    assert not both, (
        f"{name}: {sorted(both)[:10]} declared as both a class and a property"
    )


@pytest.mark.parametrize("name", NAMES)
def test_every_asserted_class_is_declared(name, built):
    onto = built(name)
    used = {cls for classes in onto.types.values() for cls in classes}
    undeclared = used - onto.classes
    assert not undeclared, (
        f"{name}: asserted into undeclared class(es) {sorted(undeclared)}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_every_asserted_property_is_declared(name, built):
    onto = built(name)
    used = {prop for prop, _, _ in onto.relations} | {
        prop for prop, _ in onto.data_assertions
    }
    undeclared = used - onto.properties
    assert not undeclared, (
        f"{name}: asserted undeclared propert(ies) {sorted(undeclared)}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_every_relation_endpoint_is_an_asserted_individual(name, built):
    onto = built(name)
    dangling = onto.relation_endpoints - onto.individuals
    assert not dangling, (
        f"{name}: {len(dangling)} relation endpoint(s) are not asserted anywhere: "
        f"{sorted(dangling)[:10]}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_no_individual_is_both_an_entity_and_an_attribute(name, built):
    onto = built(name)
    offenders = [
        individual
        for individual in onto.individuals
        if (lineage := onto.lineage(individual)) & ENTITY_ROOTS
        and lineage & ATTRIBUTE_ROOTS
    ]
    assert not offenders, (
        f"{name}: {len(offenders)} individual(s) are both an entity and an "
        f"attribute: {sorted(offenders)[:10]}"
    )


@pytest.mark.parametrize("name", NAMES)
def test_no_axiom_is_written_twice(name, built):
    repeated = built(name).duplicate_axioms
    assert not repeated, "{} repeated axiom(s):\n{}".format(
        len(repeated),
        "\n".join(f"  {n}x  {line}" for line, n in sorted(repeated.items())),
    )


@pytest.mark.parametrize("name", EL_NAMES)
def test_an_el_build_uses_no_constructor_el_lacks(name, built):
    offenders = [
        (constructor, line)
        for line in built(name).axiom_lines
        for constructor in FORBIDDEN_IN_EL
        if re.search(rf"\b{constructor}\(", line)
    ]
    assert not offenders, (
        f"{name} is an OWL2EL build but emits {len(offenders)} axiom(s) using a "
        "constructor EL does not have:\n"
        + "\n".join(f"  {c}: {line}" for c, line in offenders[:10])
    )


def test_the_matrix_covers_both_profiles():
    assert EL_NAMES
    assert set(NAMES) - set(EL_NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_an_attribute_with_several_owners_keeps_one_domain_axiom(name, built):
    settings = CONFIGURATIONS[name]().settings
    domains = [
        line
        for line in built(name).axiom_lines
        if line.startswith("ObjectPropertyDomain(:has_")
    ]

    if settings.profile is OWLProfile.OWL2DL:
        assert any("ObjectUnionOf" in line for line in domains), (
            f"{name}: channel is carried by both object types, so a DL build "
            "should express its domain as a union"
        )
    else:
        assert not any("ObjectUnionOf" in line for line in domains)


@pytest.mark.parametrize("name", NAMES)
def test_two_builds_of_one_configuration_are_identical(name, tmp_path):
    first = emit(CONFIGURATIONS[name](), tmp_path / "first.ofn")
    second = emit(CONFIGURATIONS[name](), tmp_path / "second.ofn")

    assert first.axiom_lines == second.axiom_lines


@pytest.mark.reasoner
@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("name", NAMES)
def test_every_entity_in_the_signature_is_declared(name, backend, built_path):
    """Nothing is used under a name the file never declares.

    `Emitted` holds the classes and properties a build declared. This asks the
    OWL API for the whole signature, which also holds the annotation properties
    and the datatypes, and those are where a missing declaration goes unseen.
    """
    undeclared = undeclared_signature(built_path(name, backend))
    assert not undeclared, (
        f"{name} through the {backend} sink uses {sum(map(len, undeclared.values()))} "
        "name(s) it never declares:\n"
        + "\n".join(
            f"  {kind}: {', '.join(names)}"
            for kind, names in sorted(undeclared.items())
        )
    )


@pytest.mark.reasoner
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_build_names_the_ontology_it_produced(backend, built_path):
    """A build states an ontology IRI, and says what produced it.

    The IRI is how a consumer says which ontology it loaded. `owl:versionInfo`
    is what a build carries instead of a version IRI, which owlapy has no way
    to set on the ontology it holds in memory.
    """
    path = built_path(NAMES[0], backend)
    ontology_iri, _ = ontology_iris(path)

    assert ontology_iri, f"the {backend} sink wrote no ontology IRI"
    assert "versionInfo" in ontology_annotations(path), (
        f"the {backend} sink wrote no owl:versionInfo, so {ontology_iri} does not "
        "say what built it"
    )


@pytest.mark.reasoner
def test_the_ontology_iri_is_the_one_the_build_was_given(tmp_path):
    """A build puts its names in the IRI it was given, not a fixed one."""
    path = tmp_path / "named.ofn"
    config = CONFIGURATIONS["dl_df"]()
    sink = StreamingSink(str(path), "https://example.org/log", config.settings.profile)
    OCEDOntoBuilder(
        "https://example.org/log", config.settings, sink=sink
    ).build_ontology(stores.build(stores.MATRIX_STORE), config.inputs)

    ontology_iri, _ = ontology_iris(path)
    assert ontology_iri == "https://example.org/log"
