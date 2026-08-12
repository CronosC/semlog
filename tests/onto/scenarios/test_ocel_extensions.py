"""Object attribute history, end to end."""

import pandas as pd
import pytest
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty
from owlapy.owl_reasoner import StructuralReasoner

from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    OCELExtension,
    OntoBuildSettings,
    OWLProfile,
)

NS = DEFAULT_IRI


def _individual_iris(onto) -> set[str]:
    return {i.iri.remainder for i in onto.individuals_in_signature()}


def _object_prop_assertions(onto, prop_name: str) -> set[tuple[str, str]]:
    prop = OWLObjectProperty(IRI(NS, prop_name))
    reasoner = StructuralReasoner(onto)
    pairs = set()
    for ind in onto.individuals_in_signature():
        for val in reasoner.object_property_values(ind, prop):
            pairs.add((ind.iri.remainder, val.iri.remainder))
    return pairs


def _data_prop_subjects(onto, prop_name: str) -> set[str]:
    prop = OWLDataProperty(IRI(NS, prop_name))
    reasoner = StructuralReasoner(onto)
    subjects = set()
    for ind in onto.individuals_in_signature():
        if list(reasoner.data_property_values(ind, prop)):
            subjects.add(ind.iri.remainder)
    return subjects


@pytest.fixture
def settings_ah():
    s = OntoBuildSettings(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])
    s = s.replace(profile=OWLProfile.OWL2EL)
    s = s.replace(order_representations=[])
    s = s.replace(case_concepts=[])
    s = s.replace(inverse_representations=[])
    return s


@pytest.fixture
def oced_with_changes():
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


def test_attribute_history_creates_attribute_change_individuals(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    iris = _individual_iris(builder.onto)
    attr_changes = {i for i in iris if i.startswith("AttrChange_")}
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
    assert len(pairs) == 3


def test_attribute_history_skipped_when_store_not_in_ocel_mode(tmp_path, settings_ah):
    oced = OCEDStore()
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
    builder.build_ontology(oced)

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
    settings_ah = settings_ah.replace(data_representations=[DataRepresentation.VALUES])
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_numeric_changes)

    subjects = _data_prop_subjects(builder.onto, "has_data_value")
    assert "weight_100.0" in subjects
    assert "weight_200.0" in subjects


def test_attribute_history_numeric_none_skips(tmp_path, oced_with_numeric_changes):
    settings = OntoBuildSettings(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])
    settings = settings.replace(profile=OWLProfile.OWL2EL)
    settings = settings.replace(order_representations=[])
    settings = settings.replace(case_concepts=[])
    settings = settings.replace(inverse_representations=[])
    settings = settings.replace(data_representations=[])

    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(oced_with_numeric_changes)

    iris = _individual_iris(builder.onto)
    assert not any(i.startswith("AttrChange_") for i in iris)


def test_attribute_history_numeric_bins_only_skips(tmp_path, oced_with_numeric_changes):
    settings = OntoBuildSettings(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])
    settings = settings.replace(profile=OWLProfile.OWL2EL)
    settings = settings.replace(order_representations=[])
    settings = settings.replace(case_concepts=[])
    settings = settings.replace(inverse_representations=[])
    settings = settings.replace(data_representations=[DataRepresentation.EQUAL_BINS])

    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(oced_with_numeric_changes)

    iris = _individual_iris(builder.onto)
    assert not any(i.startswith("AttrChange_") for i in iris)


def test_attribute_history_numeric_values_with_bins_ignores_bin_labels(
    tmp_path, oced_with_numeric_changes
):
    settings = OntoBuildSettings(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])
    settings = settings.replace(profile=OWLProfile.OWL2EL)
    settings = settings.replace(order_representations=[])
    settings = settings.replace(case_concepts=[])
    settings = settings.replace(inverse_representations=[])
    settings = settings.replace(
        data_representations=[
            DataRepresentation.VALUES,
            DataRepresentation.EQUAL_BINS,
        ]
    )

    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(oced_with_numeric_changes)

    iris = _individual_iris(builder.onto)
    assert "weight_100.0" in iris
    assert "weight_200.0" in iris
    bin_labels = settings.bin_labels
    assert not any(f"weight_{lbl}" in iris for lbl in bin_labels)


def test_attribute_history_changed_at_time_has_data_value(
    tmp_path, settings_ah, oced_with_changes
):
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    subjects = _data_prop_subjects(builder.onto, "has_data_value")
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
    settings_ah = settings_ah.replace(derive_annotations=True)
    builder = OCEDOntoBuilder(NS, settings_ah)
    builder.build_ontology(oced_with_changes)

    out_path = str(tmp_path / "onto.owl")
    builder.sink.save(out_path)
    content = open(out_path).read()
    assert "2022-01-01 00:00:00" in content
    assert "2022-01-02 00:00:00" in content
    assert "2022-01-03 00:00:00" in content
