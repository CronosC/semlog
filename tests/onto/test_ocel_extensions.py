"""Tests for OcelExtension instantiation: ATTRIBUTE_HISTORY and RELATION_TIMESTAMPS."""

import pandas as pd
import pytest
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty
from owlapy.owl_reasoner import StructuralReasoner

from semanticlog.data.oced import OCEDStore
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    OcelExtension,
    OntoBuildSettings,
    OwlProfile,
)

NS = DEFAULT_IRI


def _individual_iris(onto) -> set[str]:
    return {i.iri.remainder for i in onto.individuals_in_signature()}


def _object_prop_assertions(onto, prop_name: str) -> set[tuple[str, str]]:
    """Return all (subject, object) pairs for the given object property."""
    prop = OWLObjectProperty(IRI(NS, prop_name))
    reasoner = StructuralReasoner(onto)
    pairs = set()
    for ind in onto.individuals_in_signature():
        for val in reasoner.object_property_values(ind, prop):
            pairs.add((ind.iri.remainder, val.iri.remainder))
    return pairs


def _data_prop_subjects(onto, prop_name: str) -> set[str]:
    """Return IRI remainders of all individuals that have the given data property asserted."""
    prop = OWLDataProperty(IRI(NS, prop_name))
    reasoner = StructuralReasoner(onto)
    subjects = set()
    for ind in onto.individuals_in_signature():
        if list(reasoner.data_property_values(ind, prop)):
            subjects.add(ind.iri.remainder)
    return subjects


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def settings_ah():
    s = OntoBuildSettings(ocel_extensions=[OcelExtension.ATTRIBUTE_HISTORY])
    s.profile = OwlProfile.OWL2EL
    s.order_representation = []
    s.case_concepts = []
    s.inverse_representation = []
    return s


@pytest.fixture
def oced_with_changes():
    """OCEDStore in ocel_mode with one object type that has attribute history."""
    oced = OCEDStore(ocel_mode=True)
    oced.add_object(
        "order",
        "ocel:oid",
        pd.DataFrame({"ocel:oid": ["o1", "o2"]}),
    )
    ev = pd.DataFrame(
        {
            "ocel:eid": ["e1"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
        }
    )
    oced.add_event("place_order", "ocel:eid", "ocel:timestamp", ev)
    oced.add_event_to_object_relation(
        "modifies",
        "ocel:eid",
        "ocel:oid",
        ["order"],
        pd.DataFrame({"ocel:eid": ["e1"], "ocel:oid": ["o1"]}),
    )
    changes = pd.DataFrame(
        {
            "ocel:oid": ["o1", "o1", "o2"],
            "ocel:type": ["order", "order", "order"],
            "ocel:timestamp": pd.to_datetime(
                ["2022-01-01", "2022-01-03", "2022-01-02"], utc=True
            ),
            "ocel:field": ["status", "status", "status"],
            "status": ["pending", "shipped", "pending"],
        }
    )
    oced.add_object_changes("order", changes)
    return oced


# ---------------------------------------------------------------------------
# Tests — ATTRIBUTE_HISTORY ABox instantiation
# ---------------------------------------------------------------------------


def test_attribute_history_creates_attribute_change_individuals(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    iris = _individual_iris(builder.onto)
    attr_changes = {i for i in iris if i.startswith("AttrChange_")}
    # 3 change records → 3 AttrChange individuals
    assert len(attr_changes) == 3


def test_attribute_history_links_object_to_attr_change(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    pairs = _object_prop_assertions(builder.onto, "has_attribute_change")
    subjects = {s for s, _ in pairs}
    assert "o1" in subjects
    assert "o2" in subjects


def test_attribute_history_links_attr_change_to_value(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    # The per-attribute subproperty is asserted, not the generic superproperty.
    pairs = _object_prop_assertions(builder.onto, "change_value_status")
    objects = {o for _, o in pairs}
    assert "status_pending" in objects
    assert "status_shipped" in objects


def test_attribute_history_per_attr_prop_in_signature(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    prop_names = {
        p.iri.remainder for p in builder.onto.object_properties_in_signature()
    }
    assert "change_value_status" in prop_names


def test_attribute_history_per_attr_change_subclass(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    reasoner = StructuralReasoner(builder.onto)
    status_change_cls = OWLClass(IRI(NS, "statusChange"))
    attr_change_cls = OWLClass(IRI(NS, "AttributeChange"))
    supers = set(reasoner.super_classes(status_change_cls, direct=True))
    assert attr_change_cls in supers


def test_attribute_history_individuals_in_change_subclass(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    reasoner = StructuralReasoner(builder.onto)
    status_change_cls = OWLClass(IRI(NS, "statusChange"))
    instances = {i.iri.remainder for i in reasoner.instances(status_change_cls)}
    attr_changes = {i for i in instances if i.startswith("AttrChange_")}
    assert len(attr_changes) == 3


def test_attribute_history_links_attr_change_to_time(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    pairs = _object_prop_assertions(builder.onto, "changed_at")
    assert len(pairs) == 3  # one per change record


def test_attribute_history_skipped_when_store_not_in_ocel_mode(tmp_path, settings_ah):
    """If the store was not created with ocel_mode=True, the builder skips gracefully."""
    oced = OCEDStore()  # ocel_mode=False
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1"]}))
    ev = pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2022-01-01"], utc=True)})
    oced.add_event("act", "eid", "ts", ev)
    oced.add_event_to_object_relation(
        "modifies",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )

    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced)  # must not raise

    iris = _individual_iris(builder.onto)
    assert not any(i.startswith("AttrChange_") for i in iris)


@pytest.fixture
def oced_with_numeric_changes():
    oced = OCEDStore(ocel_mode=True)
    oced.add_object("item", "ocel:oid", pd.DataFrame({"ocel:oid": ["i1", "i2"]}))
    ev = pd.DataFrame(
        {"ocel:eid": ["e1"], "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True)}
    )
    oced.add_event("ship", "ocel:eid", "ocel:timestamp", ev)
    oced.add_event_to_object_relation(
        "modifies",
        "ocel:eid",
        "ocel:oid",
        ["item"],
        pd.DataFrame({"ocel:eid": ["e1"], "ocel:oid": ["i1"]}),
    )
    changes = pd.DataFrame(
        {
            "ocel:oid": ["i1", "i2"],
            "ocel:type": ["item", "item"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
            "ocel:field": ["weight", "weight"],
            "weight": [100.0, 200.0],
        }
    )
    oced.add_object_changes("item", changes)
    return oced


def test_attribute_history_numeric_values_has_data_value(
    tmp_path, settings_ah, oced_with_numeric_changes
):
    settings_ah.data_representation = [DataRepresentation.VALUES]
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_numeric_changes)

    subjects = _data_prop_subjects(builder.onto, "has_data_value")
    assert "weight_100.0" in subjects
    assert "weight_200.0" in subjects


def test_attribute_history_numeric_none_skips(tmp_path, oced_with_numeric_changes):
    settings = OntoBuildSettings(ocel_extensions=[OcelExtension.ATTRIBUTE_HISTORY])
    settings.profile = OwlProfile.OWL2EL
    settings.order_representation = []
    settings.case_concepts = []
    settings.inverse_representation = []
    settings.data_representation = []

    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(oced_with_numeric_changes)

    iris = _individual_iris(builder.onto)
    assert not any(i.startswith("AttrChange_") for i in iris)


def test_attribute_history_numeric_bins_only_skips(tmp_path, oced_with_numeric_changes):
    settings = OntoBuildSettings(ocel_extensions=[OcelExtension.ATTRIBUTE_HISTORY])
    settings.profile = OwlProfile.OWL2EL
    settings.order_representation = []
    settings.case_concepts = []
    settings.inverse_representation = []
    settings.data_representation = [DataRepresentation.EQUAL_BINS]

    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(oced_with_numeric_changes)

    iris = _individual_iris(builder.onto)
    assert not any(i.startswith("AttrChange_") for i in iris)


def test_attribute_history_numeric_values_with_bins_ignores_bin_labels(
    tmp_path, oced_with_numeric_changes
):
    settings = OntoBuildSettings(ocel_extensions=[OcelExtension.ATTRIBUTE_HISTORY])
    settings.profile = OwlProfile.OWL2EL
    settings.order_representation = []
    settings.case_concepts = []
    settings.inverse_representation = []
    settings.data_representation = [
        DataRepresentation.VALUES,
        DataRepresentation.EQUAL_BINS,
    ]

    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(oced_with_numeric_changes)

    iris = _individual_iris(builder.onto)
    # Raw value individuals must be present
    assert "weight_100.0" in iris
    assert "weight_200.0" in iris
    # No bin-label individuals
    bin_labels = settings.bin_labels
    assert not any(f"weight_{lbl}" in iris for lbl in bin_labels)


def test_attribute_history_changed_at_time_has_data_value(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    subjects = _data_prop_subjects(builder.onto, "has_data_value")
    # All 3 change records have distinct timestamps → 3 Time individuals with has_data_value
    # (timestamps: 2022-01-01, 2022-01-03, 2022-01-02)
    time_ids = {s for s in subjects if s[0].isdigit()}
    assert len(time_ids) == 3


def test_attribute_history_attribute_class_is_object_attribute(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    reasoner = StructuralReasoner(builder.onto)
    status_cls = OWLClass(IRI(NS, "status"))
    obj_attr_cls = OWLClass(IRI(NS, "ObjectAttribute"))
    supers = set(reasoner.super_classes(status_cls, direct=False))
    assert obj_attr_cls in supers, (
        "status (from history) not subclass of ObjectAttribute"
    )


def test_attribute_history_changed_at_time_has_label(
    tmp_path, settings_ah, oced_with_changes
):
    settings_ah.derive_annotations = True
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    out_path = str(tmp_path / "onto.owl")
    builder.save_onto(out_path)
    content = open(out_path).read()
    # Time individuals for the 3 change timestamps should have human-readable labels
    assert "2022-01-01 00:00:00" in content
    assert "2022-01-02 00:00:00" in content
    assert "2022-01-03 00:00:00" in content
