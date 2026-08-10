"""The sink boundary holds: a build step can run without an OWL library.

If an emitter can be driven to completion by a sink that only records calls, then
the emitters really are free of owlapy and the streaming and OWL-API backends
planned behind :class:`AxiomSink` have somewhere to plug in.
"""

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.data.oced import OCEDStore
from semanticlog.onto.abox import EventEmitter, OrderEmitter, TimeEmitter
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    EventOrderRepresentation,
    OntoBuildSettings,
)
from semanticlog.onto.sink import AxiomSink
from semanticlog.onto.upper import TBoxEmitter


class RecordingSink(AxiomSink):
    """An :class:`AxiomSink` that stores calls instead of building an ontology.

    Implements only the ``write_*`` primitives, which is the whole surface a new
    backend has to supply. Everything else is inherited policy.
    """

    def __init__(self, settings: OntoBuildSettings, iri: str = DEFAULT_IRI) -> None:
        super().__init__(iri, settings)
        self.calls: list[tuple] = []

    def _record(self, *call) -> None:
        self.calls.append(call)

    def write_declaration(self, name, kind):
        self._record("declare", name, kind)

    def write_sub_class_of(self, sub, superclass):
        self._record("sub_class", sub, superclass)

    def write_sub_property_of(self, sub, parent, data_prop):
        self._record("sub_property", sub, parent)

    def write_domain(self, prop, expression, data_prop):
        self._record("domain", prop, expression)

    def write_range(self, prop, expression, data_prop):
        self._record("range", prop, expression)

    def write_inverse_properties(self, prop, inverse):
        self._record("inverse", prop, inverse)

    def write_transitive(self, prop):
        self._record("transitive", prop)

    def write_class_assertions(self, class_expression, individuals):
        for individual in individuals:
            self._record("individual", str(individual), class_expression)

    def write_object_assertions(self, prop, pairs):
        for subject, obj in pairs:
            self._record("relation", prop, str(subject), str(obj))

    def write_data_assertions(self, prop, pairs):
        for subject, value in pairs:
            self._record("data", prop, str(subject), value)

    def write_annotations(self, annotation, pairs):
        for name, text in pairs:
            self._record("annotate", str(name), annotation, text)

    def write_ontology_annotation(self, property_iri, value):
        self._record("ontology_annotation", property_iri, value)

    def save(self, path):
        raise NotImplementedError("RecordingSink holds no ontology to save")

    # helpers for assertions

    def of_kind(self, kind: str) -> list[tuple]:
        return [c for c in self.calls if c[0] == kind]


@pytest.fixture
def settings() -> OntoBuildSettings:
    return OntoBuildSettings(
        case_concepts=["order"],
        order_representation=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.EF_explicit,
        ],
    )


@pytest.fixture
def sink(settings) -> RecordingSink:
    return RecordingSink(settings)


@pytest.fixture
def store() -> OCEDStore:
    data = OCEDStore()
    data.add_event(
        "step",
        id_col="id",
        time_col="time",
        data=pd.DataFrame(
            {
                "id": ["e1", "e2"],
                "time": [datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "channel": ["web", "shop"],
            }
        ),
    )
    data.add_object("order", id_col="id", data=pd.DataFrame({"id": ["o1"]}))
    data.add_event_to_object_relation(
        "concerns",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["e1", "e2"], "oid": ["o1", "o1"]}),
    )
    return data


def _emitter(cls, sink, settings):
    return cls(sink, settings, AnnotationWriter(sink, settings))


def test_tbox_emitter_runs_without_an_ontology(sink, settings):
    _emitter(TBoxEmitter, sink, settings).emit(attribute_history=False)

    declared = {(c[1], c[2]) for c in sink.of_kind("declare")}
    assert {("Event", "class"), ("Object", "class"), ("Time", "class")} <= declared
    assert {("observed_at", "object"), ("has_data_value", "data")} <= declared


def test_event_emitter_runs_without_an_ontology(sink, settings, store):
    _emitter(EventEmitter, sink, settings).emit(store)

    individuals = {(c[1], c[2]) for c in sink.of_kind("individual")}
    assert ("e1", "step") in individuals
    assert ("e2", "step") in individuals
    # The string attribute becomes a class and one individual per distinct value.
    assert ("channel_web", "channel") in individuals


def test_time_emitter_runs_without_an_ontology(sink, settings, store):
    _emitter(TimeEmitter, sink, settings).emit(store)

    observed = {(c[2], c[3]) for c in sink.of_kind("relation") if c[1] == "observed_at"}
    assert observed == {
        ("e1", "20240101T000000_000000"),
        ("e2", "20240102T000000_000000"),
    }


def test_order_emitter_runs_without_an_ontology(sink, settings, store):
    _emitter(OrderEmitter, sink, settings).emit(store)

    df_pairs = {(c[2], c[3]) for c in sink.of_kind("relation") if c[1] == "df_order"}
    ef_pairs = {(c[2], c[3]) for c in sink.of_kind("relation") if c[1] == "ef_order"}

    assert df_pairs == {("e1", "e2")}
    # EF_explicit materialises the upper triangle, reflexive pairs included.
    assert ef_pairs == {("e1", "e1"), ("e1", "e2"), ("e2", "e2")}
