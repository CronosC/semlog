"""Eventually-follows means the same thing whichever strategy produces it.

``EF`` leaves the closure to a reasoner; ``EF_explicit`` materialises it. Both
must yield the same extension, otherwise the choice between them is not the
performance trade-off it is documented to be.
"""

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.data.oced import OCEDStore
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    EventOrderRepresentation,
    OntoBuildSettings,
    OwlProfile,
)

NS = DEFAULT_IRI
EVENTS = ["e1", "e2", "e3"]


@pytest.fixture
def store() -> OCEDStore:
    """A single trace ``e1 -> e2 -> e3`` over one case concept."""
    data = OCEDStore()
    data.add_event(
        "step",
        id_col="id",
        time_col="time",
        data=pd.DataFrame(
            {"id": EVENTS, "time": [datetime(2023, 1, d) for d in (1, 2, 3)]}
        ),
    )
    data.add_object("order", id_col="id", data=pd.DataFrame({"id": ["o1"]}))
    data.add_event_to_object_relation(
        "concerns",
        from_id_col="event_id",
        to_id_col="object_id",
        to_type=["order"],
        data=pd.DataFrame({"event_id": EVENTS, "object_id": ["o1"] * 3}),
    )
    return data


def _build(store: OCEDStore, order: list, path) -> str:
    settings = OntoBuildSettings(
        profile=OwlProfile.OWL2DL, case_concepts=["order"], order_representation=order
    )
    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(store)
    builder.save_onto(str(path))
    return str(path)


def _ef_pairs_from_file(path: str) -> set[tuple[str, str]]:
    """Read the asserted ``ef_order`` pairs out of a saved ontology.

    Parsed as XML rather than matched as text: the two writers disagree on both
    the element name for an individual (``owl:NamedIndividual`` from owlready2,
    ``owl:Thing`` from the OWL API after materialising) and on whether IRIs are
    written in local or absolute form.
    """
    import xml.etree.ElementTree as ET

    rdf = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
    pairs: set[tuple[str, str]] = set()

    for node in ET.parse(path).getroot():
        about = node.get(f"{rdf}about")
        if about is None:
            continue
        subject = about.rsplit("#", 1)[-1]
        if subject not in EVENTS:
            continue
        for child in node:
            if child.tag.rsplit("}", 1)[-1] == "ef_order":
                target = child.get(f"{rdf}resource", "")
                pairs.add((subject, target.rsplit("#", 1)[-1]))
    return pairs


EXPECTED = {
    ("e1", "e1"),
    ("e1", "e2"),
    ("e1", "e3"),
    ("e2", "e2"),
    ("e2", "e3"),
    ("e3", "e3"),
}


def test_ef_explicit_materialises_the_full_closure(store, tmp_path):
    path = _build(
        store,
        [EventOrderRepresentation.DF, EventOrderRepresentation.EF_explicit],
        tmp_path / "explicit.owl",
    )
    assert _ef_pairs_from_file(path) == EXPECTED


def test_ef_axiomatic_asserts_only_the_self_loops(store, tmp_path):
    """Before reasoning, the axiomatic strategy states reflexivity and nothing more.

    The rest of the closure is what the transitivity and ``df ⊑ ef`` axioms are
    for. Reflexivity cannot be an axiom in OWL 2 DL alongside transitivity, so it
    is the one part that has to be asserted per event.
    """
    path = _build(
        store,
        [EventOrderRepresentation.DF, EventOrderRepresentation.EF],
        tmp_path / "axiomatic.owl",
    )
    assert _ef_pairs_from_file(path) == {("e1", "e1"), ("e2", "e2"), ("e3", "e3")}


def test_ef_explicit_refuses_a_materialisation_over_budget(store, tmp_path):
    """The cost is quadratic in trace length, so it cannot be read off the input size."""
    settings = OntoBuildSettings(
        profile=OwlProfile.OWL2DL,
        case_concepts=["order"],
        order_representation=[EventOrderRepresentation.EF_explicit],
        max_ef_pairs=3,
    )
    builder = OCEDOntoBuilder(NS, settings)

    # 3 events in one trace is 6 pairs, over the limit of 3.
    with pytest.raises(ValueError, match="over the max_ef_pairs limit"):
        builder.build_ontology(store)


def test_ef_explicit_within_budget_builds(store, tmp_path):
    settings = OntoBuildSettings(
        profile=OwlProfile.OWL2DL,
        case_concepts=["order"],
        order_representation=[EventOrderRepresentation.EF_explicit],
        max_ef_pairs=6,
    )
    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(store)
    builder.save_onto(str(tmp_path / "within.owl"))

    assert _ef_pairs_from_file(str(tmp_path / "within.owl")) == EXPECTED


def test_reasoning_over_ef_axiomatic_yields_the_same_closure(store, tmp_path):
    """The two strategies agree once a reasoner interprets the axioms."""
    from owlapy.owl_individual import OWLNamedIndividual
    from owlapy.owl_property import OWLObjectProperty

    path = _build(
        store,
        [EventOrderRepresentation.DF, EventOrderRepresentation.EF],
        tmp_path / "axiomatic.owl",
    )
    reasoner = Reasoner(path, profile=OwlProfile.OWL2DL).reasoner
    ef = OWLObjectProperty(NS + "ef_order")

    derived = {
        (event, value.iri.get_remainder())
        for event in EVENTS
        for value in reasoner.object_property_values(OWLNamedIndividual(NS + event), ef)
    }
    assert derived == EXPECTED


def test_materialize_writes_the_ef_closure_to_the_file(store, tmp_path):
    """Materialising records what the reasoner derived, not only class-level facts.

    ``InferredPropertyAssertionGenerator`` yields nothing through owlapy, so until
    the reasoner was queried directly this wrote just the three asserted
    self-loops and silently dropped the closure EF exists to produce.
    """
    path = _build(
        store,
        [EventOrderRepresentation.DF, EventOrderRepresentation.EF],
        tmp_path / "materialised.owl",
    )
    Reasoner(path, profile=OwlProfile.OWL2DL).materialize()

    assert _ef_pairs_from_file(path) == EXPECTED
