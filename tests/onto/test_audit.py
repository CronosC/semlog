"""Names that two things would share are rejected before anything is written."""

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.audit import (
    NameCollisionError,
    UnsafeLocalNameError,
    audit_names,
)
from semanticlog.onto.instantiate import OCEDOntoBuilder
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


def test_ambiguous_underscore_split_is_rejected():
    """``a_b`` + ``c`` and ``a`` + ``b_c`` both name the individual ``a_b_c``."""
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1"], "a_b": ["c"], "a": ["b_c"]}))
    )

    with pytest.raises(NameCollisionError, match="a_b_c") as excinfo:
        OCEDOntoBuilder(NS, OntoBuildSettings()).build_ontology(store)

    message = str(excinfo.value)
    assert "value 'c' of attribute 'a_b'" in message
    assert "value 'b_c' of attribute 'a'" in message


def test_collision_is_raised_before_any_axiom_is_written():
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1"], "a_b": ["c"], "a": ["b_c"]}))
    )
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())

    with pytest.raises(NameCollisionError):
        builder.build_ontology(store)

    assert list(builder.onto.classes_in_signature()) == []


def test_bin_class_colliding_with_a_value_individual_is_rejected():
    """``price`` valued ``low`` and a ``price_low`` bin subclass are one IRI."""
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1", "o2"], "price": ["low", "high"]}))
    )
    settings = OntoBuildSettings(
        data_representation=[DataRepresentation.VALUES, DataRepresentation.EQUAL_BINS],
        bins=2,
        bin_labels=["low", "high"],
    )
    # The bin subclass names come from bin_labels, the individuals from the data.
    store.add_object(
        "priced",
        id_col="oid",
        data=pd.DataFrame({"oid": ["p1", "p2"], "price": [1.0, 2.0]}),
        overwrite=True,
    )

    with pytest.raises(NameCollisionError, match="price_low"):
        audit_names(store, settings)


def test_distinct_names_pass():
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1"], "colour": ["red"], "size": ["large"]}))
    )
    audit_names(store, OntoBuildSettings())


def test_same_attribute_on_two_types_is_not_a_collision():
    """Sharing a column name across types is conflation, not an IRI clash.

    Both tables contribute the same class and property from the same source, so
    the audit must stay quiet. Whether they *should* be one attribute is a
    separate modelling question.
    """
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

    audit_names(clean_oced_data(data), OntoBuildSettings())


def test_uncleaned_activity_name_is_rejected():
    """Skipping ``clean_oced_data`` used to produce malformed IRIs, not an error."""
    store = _store(pd.DataFrame({"oid": ["o1"]}), activity="place order")

    with pytest.raises(UnsafeLocalNameError, match="clean_oced_data"):
        audit_names(store, OntoBuildSettings())


def test_uncleaned_attribute_value_is_rejected():
    store = _store(pd.DataFrame({"oid": ["o1"], "status": ["in progress"]}))

    with pytest.raises(UnsafeLocalNameError, match="status_in progress"):
        audit_names(store, OntoBuildSettings())


def test_uncleaned_entity_id_is_rejected():
    store = _store(pd.DataFrame({"oid": ["order/1"]}))

    with pytest.raises(UnsafeLocalNameError, match="order/1"):
        audit_names(store, OntoBuildSettings())


def test_numeric_values_with_decimal_points_are_accepted():
    """``price_10.5`` is what a float attribute has always produced.

    A dot is legal in an IRI reference even though it is not a valid QName, so
    rejecting it would fail nearly every real log for no gain.
    """
    store = clean_oced_data(
        _store(pd.DataFrame({"oid": ["o1", "o2"], "price": [10.5, -3.2]}))
    )

    audit_names(store, OntoBuildSettings())


def test_case_concept_property_colliding_with_a_relation_is_rejected():
    """A relation called ``df_order`` claims the directly-follows property IRI."""
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
        order_representation=[EventOrderRepresentation.DF],
    )

    with pytest.raises(NameCollisionError, match="df_order"):
        audit_names(data, settings)
