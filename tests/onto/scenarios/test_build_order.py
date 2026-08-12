"""What a build writes, and when it refuses to write anything."""

from datetime import datetime

import pandas as pd
import pytest
from owlapy.iri import IRI
from owlapy.owl_ontology import OWLClass

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OntoBuildSettings,
    OWLProfile,
)

NS = DEFAULT_IRI


def _populate(data: OCEDStore) -> OCEDStore:
    data.add_event(
        "place_order",
        id_col="id",
        time_col="time",
        data=pd.DataFrame(
            {
                "id": ["e1", "e2"],
                "time": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
                "price": [10.0, 20.0],
            }
        ),
    )
    data.add_object("order", id_col="id", data=pd.DataFrame({"id": ["o1", "o2"]}))
    data.add_event_to_object_relation(
        "concerns",
        from_id_col="event_id",
        to_id_col="object_id",
        to_type=["order"],
        data=pd.DataFrame({"event_id": ["e1", "e2"], "object_id": ["o1", "o2"]}),
    )
    return clean_oced_data(data)


@pytest.fixture
def store() -> OCEDStore:
    return _populate(OCEDStore())


def _signature(builder: OCEDOntoBuilder) -> list:
    return list(builder.onto.classes_in_signature()) + list(
        builder.onto.object_properties_in_signature()
    )


def test_construction_emits_nothing(store):
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    assert _signature(builder) == []


def test_invalid_profile_combination_writes_no_axioms(store):
    settings = OntoBuildSettings(
        profile=OWLProfile.OWL2EL,
        inverse_representations=[InverseRepresentation.AXIOMATIC],
    )
    builder = OCEDOntoBuilder(NS, settings)

    with pytest.raises(ValueError, match="AXIOMATIC"):
        builder.build_ontology(store)

    assert _signature(builder) == []


def test_missing_declare_input_writes_no_axioms(store):
    settings = OntoBuildSettings(
        order_representations=[EventOrderRepresentation.DECLARE],
        case_concepts=["order"],
    )
    builder = OCEDOntoBuilder(NS, settings)

    with pytest.raises(ValueError, match="DECLARE"):
        builder.build_ontology(store)

    assert _signature(builder) == []


def test_attribute_history_schema_omitted_without_ocel_mode(store, caplog):
    settings = OntoBuildSettings(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])
    builder = OCEDOntoBuilder(NS, settings)

    semlog = builder.build_ontology(store)

    assert OWLClass(IRI(NS, "AttributeChange")) not in list(
        semlog.onto.classes_in_signature()
    )
    assert "ocel_mode=True" in caplog.text


def test_attribute_history_schema_present_with_ocel_mode():
    ocel_store = _populate(OCEDStore(ocel_mode=True))
    ocel_store.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1"],
                "ocel:type": ["order"],
                "ocel:timestamp": [datetime(2023, 1, 3)],
                "ocel:field": ["status"],
                "status": ["shipped"],
            }
        ),
    )

    settings = OntoBuildSettings(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])
    semlog = OCEDOntoBuilder(NS, settings).build_ontology(ocel_store)

    assert OWLClass(IRI(NS, "AttributeChange")) in list(
        semlog.onto.classes_in_signature()
    )
