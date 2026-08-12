"""The checks that run before a build emits anything."""

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core.vocabulary import NameCollisionError, UnsafeLocalNameError
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.audit import audit_build
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    EventOrderRepresentation,
    OntoBuildSettings,
)

NS = DEFAULT_IRI


def _store(object_data: pd.DataFrame, activity: str = "act") -> OCEDStore:
    data = OCEDStore()
    data.add_event(
        activity,
        id_col="eid",
        time_col="t",
        data=pd.DataFrame({"eid": ["e1"], "t": [datetime(2024, 1, 1)]}),
    )
    data.add_object("order", id_col="oid", data=object_data)
    data.add_event_to_object_relation(
        "rel",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["e1"], "oid": [object_data["oid"].iloc[0]]}),
    )
    return data


def _build(store: OCEDStore, settings: OntoBuildSettings | None = None):
    return OCEDOntoBuilder(NS, settings or OntoBuildSettings()).build_ontology(store)


def test_ambiguous_underscore_split_is_rejected():
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1"], "a_b": ["c"], "a": ["b_c"]}))
    )

    with pytest.raises(NameCollisionError, match="a_b_c") as excinfo:
        _build(store)

    message = str(excinfo.value)
    assert "'a'" in message
    assert "'a_b'" in message


def test_an_ambiguous_split_is_rejected_before_any_axiom_is_written():
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1"], "a_b": ["c"], "a": ["b_c"]}))
    )
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())

    with pytest.raises(NameCollisionError):
        builder.build_ontology(store)

    assert list(builder.onto.classes_in_signature()) == []


def test_prefix_related_attributes_with_disjoint_values_are_accepted():
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1"], "a_b": ["x"], "a": ["y"]}))
    )
    _build(store)


def test_a_bin_class_cannot_collide_with_a_value_individual():
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1", "o2"], "price": ["low", "high"]}))
    )
    settings = OntoBuildSettings(
        data_representations=[DataRepresentation.VALUES, DataRepresentation.EQUAL_BINS],
        bins=2,
        bin_labels=["low", "high"],
    )
    store.add_object(
        "priced",
        id_col="oid",
        data=pd.DataFrame({"oid": ["p1", "p2"], "price": [1.0, 2.0]}),
        overwrite=True,
    )

    semlog = _build(store, settings)

    names = {c.iri.remainder for c in semlog.onto.classes_in_signature()}
    individuals = {i.iri.remainder for i in semlog.onto.individuals_in_signature()}
    assert "price_bin_low" in names
    assert "price_low" in individuals
    assert not names & individuals


def test_case_concept_property_colliding_with_a_relation_is_rejected():
    data = OCEDStore()
    data.add_event(
        "act",
        id_col="eid",
        time_col="t",
        data=pd.DataFrame({"eid": ["e1"], "t": [datetime(2024, 1, 1)]}),
    )
    data.add_object("order", id_col="oid", data=pd.DataFrame({"oid": ["o1"]}))
    data.add_event_to_object_relation(
        "df_order",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )
    settings = OntoBuildSettings(
        case_concepts=["order"],
        order_representations=[EventOrderRepresentation.DF],
    )

    with pytest.raises(NameCollisionError, match="df_order"):
        _build(data, settings)


def test_an_activity_named_after_an_object_type_is_rejected():
    data = OCEDStore()
    data.add_event(
        "order",
        id_col="eid",
        time_col="t",
        data=pd.DataFrame({"eid": ["e1"], "t": [datetime(2024, 1, 1)]}),
    )
    data.add_object("order", id_col="oid", data=pd.DataFrame({"oid": ["o1"]}))
    data.add_event_to_object_relation(
        "rel",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )

    with pytest.raises(NameCollisionError, match="order"):
        _build(data)


def test_uncleaned_activity_name_is_rejected():
    store = _store(pd.DataFrame({"oid": ["o1"]}), activity="place order")

    with pytest.raises(UnsafeLocalNameError, match="clean_oced_data"):
        _build(store)


def test_uncleaned_attribute_value_is_rejected():
    store = _store(pd.DataFrame({"oid": ["o1"], "status": ["in progress"]}))

    with pytest.raises(UnsafeLocalNameError, match="status_in progress"):
        _build(store)


def test_uncleaned_entity_id_is_rejected():
    store = _store(pd.DataFrame({"oid": ["order/1"]}))

    with pytest.raises(UnsafeLocalNameError, match="order/1"):
        _build(store)


def test_distinct_names_pass():
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1"], "colour": ["red"], "size": ["large"]}))
    )
    _build(store)


def test_same_attribute_on_two_types_is_not_a_collision():
    data = OCEDStore()
    data.add_event(
        "act",
        id_col="eid",
        time_col="t",
        data=pd.DataFrame({"eid": ["e1"], "t": [datetime(2024, 1, 1)], "name": ["x"]}),
    )
    data.add_object(
        "order", id_col="oid", data=pd.DataFrame({"oid": ["o1"], "name": ["y"]})
    )
    data.add_event_to_object_relation(
        "rel",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )

    _build(clean_oced_data(data))


def test_numeric_values_with_decimal_points_are_accepted():
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1", "o2"], "price": [10.5, -3.2]}))
    )

    _build(store)


def test_audit_build_accepts_a_clean_store():
    store = clean_oced_data(_store(pd.DataFrame({"oid": ["o1"], "colour": ["red"]})))

    audit_build(store, OntoBuildSettings())
