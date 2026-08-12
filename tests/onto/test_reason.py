"""Materialising an ontology with a reasoner."""

import pytest
from rdflib import RDF, Graph, URIRef

from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import DEFAULT_IRI, OntoBuildSettings, OWLProfile
from semanticlog.onto.sinks import InMemorySink


@pytest.mark.reasoner
def test_materialize(tmp_path):
    settings = OntoBuildSettings(
        profile=OWLProfile.OWL2EL,
        order_representations=[],
        case_concepts=[],
    )
    base_iri = DEFAULT_IRI
    out_path = str(tmp_path / "temp.owl")

    sink = InMemorySink(base_iri, settings)

    sink.add_class("Human")
    sink.add_class("Parent", "Human")
    sink.add_class("Male")
    sink.add_class("Female")
    sink.add_class("Man", ["Human"])
    sink.add_class("Man", ["Male"])
    sink.add_class("Woman", ["Human"])
    sink.add_class("Woman", ["Female"])

    individuals = ["a", "b", "c", "d", "e"]
    for ind in individuals:
        sink.assert_individual(ind, "Human")
    sink.assert_individual("a", "Man")
    sink.assert_individual("b", "Man")
    sink.assert_individual("c", "Man")
    sink.assert_individual("d", "Woman")
    sink.assert_individual("e", "Woman")

    sink.save(out_path)

    reasoner = Reasoner(out_path)
    reasoner.materialize()
    reasoner.save()

    rg = Graph()
    rg.parse(out_path)

    def u(local: str) -> URIRef:
        return URIRef(base_iri + local)

    def has_triple(s, p, o) -> bool:
        return any(rg.triples((s, p, o)))

    for ind in individuals:
        assert has_triple(u(ind), RDF.type, u("Human")), f"Missing {ind} a Human"

    for ind in ["a", "b", "c"]:
        assert has_triple(u(ind), RDF.type, u("Man")), f"Missing {ind} a Man"
        assert has_triple(u(ind), RDF.type, u("Male")), f"Missing {ind} a Male"

    for ind in ["d", "e"]:
        assert has_triple(u(ind), RDF.type, u("Woman")), f"Missing {ind} a Woman"
        assert has_triple(u(ind), RDF.type, u("Female")), f"Missing {ind} a Female"

    for ind in ["a", "b", "c"]:
        assert not has_triple(u(ind), RDF.type, u("Female")), (
            f"Unexpected {ind} a Female"
        )
