"""Directly-follows and the two forms of eventually-follows."""

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    EventOrderRepresentation,
    OntoBuildSettings,
    OWLProfile,
)

NS = DEFAULT_IRI
EVENTS = ["e1", "e2", "e3"]


@pytest.fixture
def store() -> OCEDStore:
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
    return clean_oced_data(data)


def _build(store: OCEDStore, order: list, path) -> str:
    settings = OntoBuildSettings(
        profile=OWLProfile.OWL2DL, case_concepts=["order"], order_representations=order
    )
    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(store)
    builder.sink.save(str(path))
    return str(path)


def _ef_pairs_from_file(path: str) -> set[tuple[str, str]]:
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
        [EventOrderRepresentation.DF, EventOrderRepresentation.EF_EXPLICIT],
        tmp_path / "explicit.owl",
    )
    assert _ef_pairs_from_file(path) == EXPECTED


def test_ef_axiomatic_asserts_only_the_self_loops(store, tmp_path):
    path = _build(
        store,
        [EventOrderRepresentation.DF, EventOrderRepresentation.EF],
        tmp_path / "axiomatic.owl",
    )
    assert _ef_pairs_from_file(path) == {("e1", "e1"), ("e2", "e2"), ("e3", "e3")}


def test_ef_explicit_refuses_a_materialisation_over_budget(store, tmp_path):
    settings = OntoBuildSettings(
        profile=OWLProfile.OWL2DL,
        case_concepts=["order"],
        order_representations=[EventOrderRepresentation.EF_EXPLICIT],
        max_ef_pairs=3,
    )
    builder = OCEDOntoBuilder(NS, settings)

    with pytest.raises(ValueError, match="over the max_ef_pairs limit"):
        builder.build_ontology(store)


def test_ef_explicit_within_budget_builds(store, tmp_path):
    settings = OntoBuildSettings(
        profile=OWLProfile.OWL2DL,
        case_concepts=["order"],
        order_representations=[EventOrderRepresentation.EF_EXPLICIT],
        max_ef_pairs=6,
    )
    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(store)
    builder.sink.save(str(tmp_path / "within.owl"))

    assert _ef_pairs_from_file(str(tmp_path / "within.owl")) == EXPECTED


@pytest.mark.reasoner
def test_reasoning_over_ef_axiomatic_yields_the_same_closure(store, tmp_path):
    from owlapy.owl_individual import OWLNamedIndividual
    from owlapy.owl_property import OWLObjectProperty

    path = _build(
        store,
        [EventOrderRepresentation.DF, EventOrderRepresentation.EF],
        tmp_path / "axiomatic.owl",
    )
    reasoner = Reasoner(path, profile=OWLProfile.OWL2DL).reasoner
    ef = OWLObjectProperty(NS + "ef_order")

    derived = {
        (event, value.iri.remainder)
        for event in EVENTS
        for value in reasoner.object_property_values(OWLNamedIndividual(NS + event), ef)
    }
    assert derived == EXPECTED


@pytest.mark.reasoner
def test_materialize_writes_the_ef_closure_to_the_file(store, tmp_path):
    path = _build(
        store,
        [EventOrderRepresentation.DF, EventOrderRepresentation.EF],
        tmp_path / "materialized.owl",
    )
    Reasoner(path, profile=OWLProfile.OWL2DL).materialize()

    assert _ef_pairs_from_file(path) == EXPECTED
