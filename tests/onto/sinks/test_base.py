"""The checks and suppression every sink inherits."""

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core.vocabulary import UnsafeLocalNameError
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.emitters import EventEmitter, OrderEmitter, TimeEmitter
from semanticlog.onto.emitters.tbox import TBoxEmitter
from semanticlog.onto.settings import (
    EventOrderRepresentation,
    OntoBuildSettings,
)
from tests.onto.support.recording import RecordingSink, make_emitter


@pytest.fixture
def settings() -> OntoBuildSettings:
    return OntoBuildSettings(
        case_concepts=["order"],
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.EF_EXPLICIT,
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
    return clean_oced_data(data)


def _emitter(cls, sink, settings):
    return make_emitter(cls, sink, settings)


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
    assert ("channel_web", "channel") in individuals


def test_time_emitter_runs_without_an_ontology(sink, settings, store):
    _emitter(TimeEmitter, sink, settings).emit(store)

    observed = {(c[2], c[3]) for c in sink.of_kind("relation") if c[1] == "observed_at"}
    assert observed == {
        ("e1", "20240101T000000_000000"),
        ("e2", "20240102T000000_000000"),
    }


def test_a_repeated_schema_axiom_is_written_once(sink):
    for _ in range(3):
        sink.add_class("channel", "ObjectAttribute")
        sink.add_property(
            "has_channel",
            domain_class=["order", "item"],
            range_class="channel",
            superprop="has_attribute",
        )
        sink.add_transitive("ef_order")
        sink.annotate_entity("channel", label="channel", comment="an attribute")

    assert len(sink.of_kind("declare")) == 2
    assert len(sink.of_kind("sub_class")) == 1
    assert len(sink.of_kind("sub_property")) == 1
    assert len(sink.of_kind("domain")) == 1
    assert len(sink.of_kind("range")) == 1
    assert len(sink.of_kind("transitive")) == 1
    assert len(sink.of_kind("annotate")) == 2


def test_an_unsafe_declared_name_is_refused(sink):
    with pytest.raises(UnsafeLocalNameError, match="bad name"):
        sink.add_class("bad name")
    with pytest.raises(UnsafeLocalNameError, match="has:colon"):
        sink.add_property("has:colon")
    with pytest.raises(UnsafeLocalNameError):
        sink.add_transitive("ef order")


def test_a_class_expression_reaches_the_backend_as_a_list(sink):
    sink.add_class("channel", ["ObjectAttribute", "EventAttribute"])
    sink.add_property("has_channel", domain_class=["order", "item"])

    assert sink.of_kind("sub_class")[0][2] == ["ObjectAttribute", "EventAttribute"]
    assert sink.of_kind("domain")[0][2] == ["order", "item"]


def test_a_single_parent_is_still_accepted_as_a_bare_string(sink):
    sink.add_class("channel", "ObjectAttribute")

    assert sink.of_kind("sub_class")[0][2] == ["ObjectAttribute"]


def test_a_class_under_two_parents_keeps_both(sink):
    sink.add_class("channel", "ObjectAttribute")
    sink.add_class("channel", "EventAttribute")

    assert len(sink.of_kind("declare")) == 1
    assert {tuple(c[2]) for c in sink.of_kind("sub_class")} == {
        ("ObjectAttribute",),
        ("EventAttribute",),
    }


def test_a_different_annotation_on_the_same_entity_is_kept(sink):
    sink.add_class("order", source="an object type")
    sink.annotate_entity("order", label="derived label")
    sink.annotate_entity("order", label="label from the annotation document")

    assert len(sink.of_kind("annotate")) == 2


def test_annotating_an_undeclared_entity_is_refused(sink):
    with pytest.raises(ValueError, match="no build step declared it"):
        sink.annotate_entity("AttributeChange", label="Object attribute change")


def test_row_level_assertions_are_never_deduplicated(sink):
    for _ in range(3):
        sink.write_object_assertions("df_order", [("e1", "e2")])
        sink.write_class_assertions("place", ["e1"])

    assert len(sink.of_kind("relation")) == 3
    assert len(sink.of_kind("individual")) == 3


def test_order_emitter_runs_without_an_ontology(sink, settings, store):
    _emitter(OrderEmitter, sink, settings).emit(store)

    df_pairs = {(c[2], c[3]) for c in sink.of_kind("relation") if c[1] == "df_order"}
    ef_pairs = {(c[2], c[3]) for c in sink.of_kind("relation") if c[1] == "ef_order"}

    assert df_pairs == {("e1", "e2")}
    assert ef_pairs == {("e1", "e1"), ("e1", "e2"), ("e2", "e2")}
