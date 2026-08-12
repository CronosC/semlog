"""Writing functional syntax straight to a file."""

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core.vocabulary import UnsafeLocalNameError
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    EventOrderRepresentation,
    InverseRepresentation,
    OntoBuildSettings,
    OWLProfile,
)
from semanticlog.onto.sinks import StreamingSink

NS = DEFAULT_IRI


@pytest.fixture
def store() -> OCEDStore:
    data = OCEDStore()
    data.add_event(
        "place order",
        id_col="eid",
        time_col="t",
        data=pd.DataFrame(
            {
                "eid": ["e1", "e2", "e3"],
                "t": [datetime(2024, 1, d) for d in (1, 2, 3)],
                "channel": ["web", "web", "shop"],
                "amount": [10.5, 20.0, 31.25],
            }
        ),
    )
    data.add_object(
        "order",
        id_col="oid",
        data=pd.DataFrame({"oid": ["o1", "o2"], "priority": ["high", "low"]}),
    )
    data.add_event_to_object_relation(
        "concerns",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["e1", "e2", "e3"], "oid": ["o1", "o1", "o2"]}),
    )
    return clean_oced_data(data)


def _settings(**kwargs) -> OntoBuildSettings:
    base = dict(
        profile=OWLProfile.OWL2DL,
        case_concepts=["order"],
        order_representations=[EventOrderRepresentation.DF],
    )
    base.update(kwargs)
    return OntoBuildSettings(**base)


_BUILTINS = {"Thing", "Nothing", "topObjectProperty", "bottomObjectProperty"}


def _signature(path: str) -> dict[str, set[str]]:
    from owlapy.owl_ontology import SyncOntology

    onto = SyncOntology(str(path))

    def names(entities) -> set[str]:
        return {e.iri.remainder for e in entities} - _BUILTINS

    return {
        "classes": names(onto.classes_in_signature()),
        "individuals": names(onto.individuals_in_signature()),
        "object_properties": names(onto.object_properties_in_signature()),
        "data_properties": names(onto.data_properties_in_signature()),
    }


def _build_both(store: OCEDStore, tmp_path, settings: OntoBuildSettings):
    in_memory = tmp_path / "memory.owl"
    OCEDOntoBuilder(NS, settings).build_ontology(store).save(str(in_memory))

    streamed = tmp_path / "streamed.ofn"
    sink = StreamingSink(str(streamed), NS, settings)
    OCEDOntoBuilder(NS, settings, sink=sink).build_ontology(store)

    return _signature(str(in_memory)), _signature(str(streamed))


def test_streamed_signature_matches_in_memory(store, tmp_path):
    memory, streamed = _build_both(store, tmp_path, _settings())
    assert streamed == memory


def test_streamed_signature_matches_with_inverses_and_annotations(store, tmp_path):
    memory, streamed = _build_both(
        store,
        tmp_path,
        _settings(
            inverse_representations=[
                InverseRepresentation.AXIOMATIC,
                InverseRepresentation.IMPLICIT,
            ],
            derive_annotations=True,
        ),
    )
    assert streamed == memory


@pytest.mark.reasoner
def test_streamed_ontology_is_consistent(store, tmp_path):
    from owlapy.owl_ontology import SyncOntology
    from owlapy.owl_reasoner import SyncReasoner

    path = tmp_path / "streamed.ofn"
    settings = _settings(
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.EF,
        ]
    )
    sink = StreamingSink(str(path), NS, settings)
    OCEDOntoBuilder(NS, settings, sink=sink).build_ontology(store)

    reasoner = SyncReasoner(SyncOntology(str(path)), reasoner="HermiT")
    assert reasoner.has_consistent_ontology()


def test_datetime_literals_survive_streaming(store, tmp_path):
    path = tmp_path / "streamed.ofn"
    sink = StreamingSink(str(path), NS, _settings())
    OCEDOntoBuilder(NS, _settings(), sink=sink).build_ontology(store)

    text = path.read_text(encoding="utf-8")
    assert "^^xsd:dateTime" in text
    assert "^^xsd:double" in text


def test_streamed_build_reports_its_path_and_has_no_ontology(store, tmp_path):
    path = tmp_path / "streamed.ofn"
    sink = StreamingSink(str(path), NS, _settings())

    result = OCEDOntoBuilder(NS, _settings(), sink=sink).build_ontology(store)

    assert result.path == str(path)
    assert result.onto is None
    with pytest.raises(ValueError, match="streamed"):
        result.require_onto()


def test_saving_a_streaming_sink_elsewhere_is_refused(store, tmp_path):
    path = tmp_path / "streamed.ofn"
    sink = StreamingSink(str(path), NS, _settings())

    with pytest.raises(ValueError, match="cannot save"):
        sink.save(str(tmp_path / "elsewhere.ofn"))


def test_a_class_expression_is_a_list_of_names_not_a_grammar(tmp_path):
    sink = StreamingSink(str(tmp_path / "s.ofn"), NS, _settings())

    with pytest.raises(UnsafeLocalNameError, match="B and C"):
        sink.add_class("A", "B and C")
