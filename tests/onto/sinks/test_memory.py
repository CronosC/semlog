"""`InMemorySink`, which builds an owlapy ontology in memory."""

from __future__ import annotations

from semanticlog.core.enums import InverseRepresentation, OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import InMemorySink
from tests.support.build import IN_MEMORY


def test_both_directions_of_an_implicit_inverse_survive_an_inverse_axiom(built):
    """owlapy keeps one direction of every assertion between two properties it
    holds an inverse axiom for, so the sink adds that axiom last.
    """
    onto = built("dl_inverses", IN_MEMORY)

    forward = {(s, o) for p, s, o in onto.relations if p == "concerns"}
    backward = {(s, o) for p, s, o in onto.relations if p == "inv_concerns"}

    assert len(forward) > 0
    assert backward == {(o, s) for s, o in forward}
    assert ("concerns", "inv_concerns") in onto.inverses


def test_the_inverse_axioms_are_in_the_ontology_a_build_returns(tiny_store):
    settings = OntoBuildSettings(
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ]
    )

    result = OCEDOntoBuilder(DEFAULT_IRI, settings).build_ontology(tiny_store)

    names = {
        p.iri.remainder for p in result.require_onto().object_properties_in_signature()
    }
    assert "inv_concerns" in names


def test_an_annotation_property_outside_the_base_namespace_keeps_its_iri():
    """owlapy's RDF/XML writer drops a namespace without a `#`, so this reads the store."""
    creator = "http://purl.org/dc/terms/creator"
    sink = InMemorySink(DEFAULT_IRI, OWLProfile.OWL2DL)
    sink.open_ontology({creator: "semlog"})

    ontology_iri = DEFAULT_IRI.removesuffix("/")
    stored = {
        prop.iri
        for prop, _ in sink.onto._onto.world.sparql(
            f"SELECT ?p ?v WHERE {{ <{ontology_iri}> ?p ?v }}"
        )
        if hasattr(prop, "iri")
    }
    assert creator in stored
