"""Materializing an ontology with a reasoner, and what a reasoner makes of a build."""

from pathlib import Path

import pytest
from rdflib import RDF, Graph, URIRef

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.reason import REASONER_BACKENDS, Reasoner
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import InMemorySink
from tests.conftest import STREAMING, emit
from tests.support.ontology import Emitted, read_functional
from tests.support.settings import CONFIGURATIONS, NAMES

# Pellet reads both profiles and takes a timestamp carrying microseconds, which
# is what the matrix writes.
BACKEND = "Pellet"


@pytest.mark.reasoner
def test_materialize(tmp_path):
    settings = OntoBuildSettings(
        profile=OWLProfile.OWL2EL,
        order_representations=[],
    )
    base_iri = DEFAULT_IRI
    out_path = str(tmp_path / "temp.owl")

    sink = InMemorySink(base_iri, settings.profile)

    sink.add_class("Human")
    sink.add_class("Parent", "Human")
    sink.add_class("Male")
    sink.add_class("Female")
    sink.add_class("Man", ["Human"])
    sink.add_class("Man", ["Male"])
    sink.add_class("Woman", ["Human"])
    sink.add_class("Woman", ["Female"])

    individuals = ["a", "b", "c", "d", "e"]
    sink.assert_individuals("Human", individuals)
    sink.assert_individuals("Man", ["a", "b", "c"])
    sink.assert_individuals("Woman", ["d", "e"])

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
    settings = OntoBuildSettings(profile=OWLProfile.OWL2EL)
    sink = InMemorySink(DEFAULT_IRI, settings.profile)
    sink.add_class("Human")
    sink.add_class("Man", ["Human"])
    sink.add_property("knows", domain_class=["Human"], range_class="Human")
    sink.assert_individuals("Man", ["a", "b"])
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


# --------------------------------------------------------------------------
# What a reasoner makes of a build from the matrix
# --------------------------------------------------------------------------

EF_ORDER = vocab.ef_property("order")


def _ef_pairs(onto: Emitted) -> set[tuple[str, str]]:
    """Return the eventually-follows pairs of the order case concept."""
    return {(s, o) for prop, s, o in onto.relations if prop == EF_ORDER}


@pytest.mark.reasoner
def test_the_two_eventually_follows_representations_agree(tmp_path):
    """EF derived by transitivity reaches the pairs EF_EXPLICIT asserts.

    The settings offer the two as alternatives to each other, so a reasoner
    over the axiomatic build has to arrive at exactly the relation the explicit
    build wrote out.
    """
    axiomatic = tmp_path / "axiomatic.ofn"
    explicit = tmp_path / "explicit.ofn"
    emit(CONFIGURATIONS["dl_df_ef"](), axiomatic, STREAMING)
    asserted = _ef_pairs(emit(CONFIGURATIONS["dl_ef_explicit"](), explicit, STREAMING))

    entailed = {
        (
            str(axiom.get_subject().iri.get_remainder()),
            str(axiom.get_object().iri.get_remainder()),
        )
        for axiom in Reasoner(
            str(axiomatic), backend=BACKEND
        ).inferred_property_assertions()
        if str(axiom.get_property().iri.as_str()).endswith(f"#{EF_ORDER}")
    }
    # A backend answers with everything entailed, which already holds what was
    # asserted. Taking both means a backend answering with only the new pairs
    # is still compared against the whole relation.
    reached = _ef_pairs(read_functional(axiomatic)) | entailed

    assert reached == asserted, (
        f"the two representations of eventually-follows differ:\n"
        f"  only reached by the reasoner: {sorted(reached - asserted)[:5]}\n"
        f"  only asserted by EF_EXPLICIT: {sorted(asserted - reached)[:5]}"
    )


@pytest.mark.reasoner
@pytest.mark.parametrize("name", NAMES)
def test_a_built_ontology_is_consistent(name, built_path):
    assert Reasoner(str(built_path(name)), backend=BACKEND).check(), (
        f"{name} is inconsistent, so nothing can be read out of it"
    )


@pytest.mark.reasoner
@pytest.mark.parametrize("name", NAMES)
def test_no_class_of_a_build_is_unsatisfiable(name, built_path):
    """No class of a build is one that nothing could ever belong to.

    A TBox with no disjointness and no cardinality cannot produce one, so this
    holds of every build as it stands. It is what would notice the day an
    axiom makes a class describe nothing.
    """
    reasoner = Reasoner(str(built_path(name)), backend=BACKEND).reasoner
    empty = sorted(
        remainder
        for owl_class in reasoner.unsatisfiable_classes()
        if (remainder := str(owl_class.iri.get_remainder())) != "Nothing"
    )
    assert not empty, f"{name} declares {len(empty)} class(es) nothing can be: {empty}"
