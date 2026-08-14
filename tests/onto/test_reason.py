"""Materialising an ontology with a reasoner."""

from pathlib import Path

import pytest
from rdflib import RDF, Graph, URIRef

from semanticlog.core.enums import OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.reason import REASONER_BACKENDS, Reasoner
from semanticlog.onto.settings import OntoBuildSettings
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
    sink.assert_individuals(individuals, "Human")
    sink.assert_individuals(["a", "b", "c"], "Man")
    sink.assert_individuals(["d", "e"], "Woman")

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


def _small_ontology(path: str) -> None:
    """Write an ontology with something for a reasoner to find."""
    settings = OntoBuildSettings(profile=OWLProfile.OWL2EL, case_concepts=[])
    sink = InMemorySink(DEFAULT_IRI, settings)
    sink.add_class("Human")
    sink.add_class("Man", ["Human"])
    sink.add_property("knows", domain_class=["Human"], range_class="Human")
    sink.assert_individuals(["a", "b"], "Man")
    sink.write_object_assertions("knows", [("a", "b")])
    sink.save(path)


@pytest.mark.reasoner
@pytest.mark.parametrize("backend", REASONER_BACKENDS)
def test_every_backend_materializes(backend, tmp_path):
    """Each backend is asked only for the entailments it can answer."""
    path = str(tmp_path / f"{backend}.owl")
    _small_ontology(path)
    before = Path(path).stat().st_size

    reasoner = Reasoner(path, backend=backend, profile=OWLProfile.OWL2EL)
    reasoner.materialize()

    assert Path(path).stat().st_size > before


@pytest.mark.reasoner
def test_a_backend_that_will_not_read_the_ontology_says_which_to_try(tmp_path):
    path = tmp_path / "not-an-ontology.owl"
    path.write_text("this is not an ontology", encoding="utf-8")

    with pytest.raises(ValueError, match="will not read") as failure:
        Reasoner(str(path))

    assert "Pellet" in str(failure.value)
