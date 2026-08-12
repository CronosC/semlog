"""One full build, read back and checked through a reasoner."""

import copy
import re
from pathlib import Path

import pandas as pd
import pytest
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_ontology import Ontology
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty
from owlapy.owl_reasoner import StructuralReasoner

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.loading.pm4py_loader import PM4PyLoader
from semanticlog.data.oced import EntityKind, OCEDStore, RelationKind
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    EventOrderRepresentation,
    OntoBuildSettings,
    OWLProfile,
)

PIPELINE_OCEL_PATH = str(
    Path(__file__).resolve().parents[2] / "fixtures" / "pipeline.jsonocel"
)

NS = DEFAULT_IRI


def _make_settings() -> OntoBuildSettings:
    s = OntoBuildSettings()
    s = s.replace(profile=OWLProfile.OWL2EL)
    s = s.replace(order_representations=[EventOrderRepresentation.DF])
    s = s.replace(case_concepts=["order"])
    s = s.replace(data_representations=[DataRepresentation.VALUES])
    s = s.replace(inverse_representations=[])
    return s


def _load_reasoner(owl_path: str):
    onto = Ontology(IRI.create(f"file://{owl_path}"))
    return onto, StructuralReasoner(onto)


def _cls(name: str) -> OWLClass:
    return OWLClass(IRI(NS, name))


def _ind(name: str) -> OWLNamedIndividual:
    return OWLNamedIndividual(IRI(NS, name))


def _oprop(name: str) -> OWLObjectProperty:
    return OWLObjectProperty(IRI(NS, name))


def _dprop(name: str) -> OWLDataProperty:
    return OWLDataProperty(IRI(NS, name))


@pytest.fixture(scope="module")
def raw_oced():
    return PM4PyLoader(PIPELINE_OCEL_PATH, {}).get_oced()


@pytest.fixture(scope="module")
def cleaned_oced(raw_oced):
    return clean_oced_data(copy.deepcopy(raw_oced))


@pytest.fixture
def upper_owl(tmp_path) -> str:
    return str(tmp_path / "tmp.owl")


def test_loader_produces_expected_store(raw_oced):
    assert set(raw_oced.get_event_types()) == {"place_order", "pay", "ship_order"}
    assert set(raw_oced.get_object_types()) == {"order", "item"}
    assert set(raw_oced.get_relation_types(RelationKind.E2O)) == {"modifies", "ships"}
    assert set(raw_oced.get_relation_types(RelationKind.O2O)) == {"contains_o2o"}
    assert len(raw_oced.get_ids(EntityKind.EVENT, "place_order")) == 4
    assert len(raw_oced.get_ids(EntityKind.EVENT, "pay")) == 4
    assert len(raw_oced.get_ids(EntityKind.EVENT, "ship_order")) == 4
    assert len(raw_oced.get_ids(EntityKind.OBJECT, "order")) == 3
    assert len(raw_oced.get_ids(EntityKind.OBJECT, "item")) == 4


def test_cleaning_sanitizes_store(raw_oced):
    cleaned = clean_oced_data(copy.deepcopy(raw_oced))

    valid = re.compile(r"^[A-Za-z0-9_]+$")
    for name in (
        cleaned.get_event_types()
        + cleaned.get_object_types()
        + cleaned.get_relation_types()
    ):
        assert valid.match(name), f"Name not sanitized: {name!r}"

    assert set(cleaned.get_event_types()) == {"place_order", "pay", "ship_order"}
    assert set(cleaned.get_object_types()) == {"order", "item"}

    po_df = cleaned.get_event_table("place_order").data
    assert "amount" in po_df.columns
    assert "resource" in po_df.columns

    so_df = cleaned.get_event_table("ship_order").data
    assert "carrier" in so_df.columns

    ord_df = cleaned.get_object_table("order").data
    assert "status" in ord_df.columns

    item_df = cleaned.get_object_table("item").data
    assert "weight" in item_df.columns


def test_event_instantiation(cleaned_oced, upper_owl):
    builder = OCEDOntoBuilder(DEFAULT_IRI, _make_settings())
    builder.events.emit(cleaned_oced)
    builder.sink.save(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    classes = set(onto.classes_in_signature())
    inds = set(onto.individuals_in_signature())

    for act in ("place_order", "pay", "ship_order"):
        assert _cls(act) in classes, f"Missing class: {act}"

    expected_po_ids = set(cleaned_oced.get_ids(EntityKind.EVENT, "place_order"))
    actual_po_inds = {i.iri.remainder for i in reasoner.instances(_cls("place_order"))}
    assert expected_po_ids == actual_po_inds

    assert _cls("resource") in classes
    assert _ind("resource_Alice") in inds
    assert _ind("resource_Bob") in inds
    assert _ind("resource_Alice") in set(reasoner.instances(_cls("resource")))
    assert _ind("resource_Bob") in set(reasoner.instances(_cls("resource")))

    event_attr_cls = _cls("EventAttribute")
    for attr in ("resource", "amount"):
        supers = set(reasoner.super_classes(_cls(attr), direct=False))
        assert event_attr_cls in supers, f"{attr} not subclass of EventAttribute"

    assert _cls("amount") in classes
    has_data_value = _dprop("has_data_value")
    assert has_data_value in set(onto.data_properties_in_signature())
    amount_inds = [i for i in inds if i.iri.remainder.startswith("amount_")]
    assert len(amount_inds) > 0
    sample_amount = amount_inds[0]
    literals = list(
        reasoner.data_property_values(sample_amount, has_data_value, direct=False)
    )
    assert len(literals) > 0, f"No has_data_value literal on {sample_amount}"

    for str_ind in (_ind("resource_Alice"), _ind("resource_Bob")):
        literals = list(
            reasoner.data_property_values(str_ind, has_data_value, direct=False)
        )
        assert len(literals) == 0, f"Unexpected has_data_value on {str_ind}"


def test_object_instantiation(cleaned_oced, upper_owl):
    builder = OCEDOntoBuilder(DEFAULT_IRI, _make_settings())
    builder.objects.emit(cleaned_oced)
    builder.sink.save(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    classes = set(onto.classes_in_signature())
    inds = set(onto.individuals_in_signature())

    assert _cls("order") in classes
    assert _cls("item") in classes

    expected_order_ids = set(cleaned_oced.get_ids(EntityKind.OBJECT, "order"))
    actual_order_inds = {i.iri.remainder for i in reasoner.instances(_cls("order"))}
    assert expected_order_ids == actual_order_inds

    assert _cls("status") in classes
    assert _ind("status_pending") in inds
    assert _ind("status_shipped") in inds
    assert _ind("status_pending") in set(reasoner.instances(_cls("status")))
    assert _ind("status_shipped") in set(reasoner.instances(_cls("status")))

    obj_attr_cls = _cls("ObjectAttribute")
    supers_status = set(reasoner.super_classes(_cls("status"), direct=False))
    assert obj_attr_cls in supers_status, "status not subclass of ObjectAttribute"

    assert _cls("weight") in classes
    has_data_value = _dprop("has_data_value")
    weight_inds = [i for i in inds if i.iri.remainder.startswith("weight_")]
    assert len(weight_inds) > 0
    sample_weight = weight_inds[0]
    literals = list(
        reasoner.data_property_values(sample_weight, has_data_value, direct=False)
    )
    assert len(literals) > 0, f"No has_data_value literal on {sample_weight}"


def test_e2o_relation_instantiation(cleaned_oced, upper_owl):
    builder = OCEDOntoBuilder(DEFAULT_IRI, _make_settings())
    builder.events.emit(cleaned_oced)
    builder.objects.emit(cleaned_oced)
    builder.relations.emit_event_to_object(cleaned_oced)
    builder.sink.save(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    obj_props = set(onto.object_properties_in_signature())

    modifies_prop = _oprop("modifies")
    ships_prop = _oprop("ships")
    assert modifies_prop in obj_props, "Property 'modifies' missing"
    assert ships_prop in obj_props, "Property 'ships' missing"

    mod_rel = cleaned_oced.get_relation("modifies")
    from_col = mod_rel.from_id
    to_col = mod_rel.to_id
    for _, row in mod_rel.data.iterrows():
        from_ind = _ind(str(row[from_col]))
        to_ind = _ind(str(row[to_col]))
        targets = set(
            reasoner.object_property_values(from_ind, modifies_prop, direct=False)
        )
        assert to_ind in targets, (
            f"Missing modifies assertion: {row[from_col]} → {row[to_col]}"
        )

    ships_rel = cleaned_oced.get_relation("ships")
    from_col_s = ships_rel.from_id
    to_col_s = ships_rel.to_id
    for _, row in ships_rel.data.iterrows():
        from_ind = _ind(str(row[from_col_s]))
        to_ind = _ind(str(row[to_col_s]))
        targets = set(
            reasoner.object_property_values(from_ind, ships_prop, direct=False)
        )
        assert to_ind in targets, (
            f"Missing ships assertion: {row[from_col_s]} → {row[to_col_s]}"
        )

    item_ids = set(str(i) for i in cleaned_oced.get_ids(EntityKind.OBJECT, "item"))
    for eid in cleaned_oced.get_ids(EntityKind.EVENT, "ship_order"):
        targets = set(
            reasoner.object_property_values(_ind(str(eid)), modifies_prop, direct=False)
        )
        for t in targets:
            assert t.iri.remainder not in item_ids, (
                f"ship_order event {eid} has unexpected modifies assertion to item {t}"
            )


def test_o2o_relation_instantiation(cleaned_oced, upper_owl):
    builder = OCEDOntoBuilder(DEFAULT_IRI, _make_settings())
    builder.events.emit(cleaned_oced)
    builder.objects.emit(cleaned_oced)
    builder.relations.emit_object_to_object(cleaned_oced)
    builder.sink.save(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    obj_props = set(onto.object_properties_in_signature())

    contains_prop = _oprop("contains_o2o")
    assert contains_prop in obj_props, "Property 'contains_o2o' missing"

    o2o_rel = cleaned_oced.get_relation("contains_o2o")
    from_col = o2o_rel.from_id
    to_col = o2o_rel.to_id
    for _, row in o2o_rel.data.iterrows():
        from_ind = _ind(str(row[from_col]))
        to_ind = _ind(str(row[to_col]))
        targets = set(
            reasoner.object_property_values(from_ind, contains_prop, direct=False)
        )
        assert to_ind in targets, (
            f"Missing contains_o2o assertion: {row[from_col]} → {row[to_col]}"
        )

    item_ids = set(str(i) for i in cleaned_oced.get_ids(EntityKind.OBJECT, "item"))
    for iid in item_ids:
        targets = set(
            reasoner.object_property_values(_ind(iid), contains_prop, direct=False)
        )
        order_ids = set(
            str(i) for i in cleaned_oced.get_ids(EntityKind.OBJECT, "order")
        )
        for t in targets:
            assert t.iri.remainder not in order_ids, (
                f"Unexpected reverse contains_o2o from item {iid} to order {t}"
            )


def test_event_order_instantiation(cleaned_oced, upper_owl):
    builder = OCEDOntoBuilder(DEFAULT_IRI, _make_settings())
    builder.events.emit(cleaned_oced)
    builder.order.emit(cleaned_oced)
    builder.sink.save(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    classes = set(onto.classes_in_signature())
    obj_props = set(onto.object_properties_in_signature())

    assert _oprop("df_order") in obj_props
    df_prop = _oprop("df_order")

    expected_df_pairs = {
        ("po_e1", "pay_e1"),
        ("pay_e1", "po_e4"),
        ("po_e2", "pay_e2"),
        ("pay_e2", "pay_e4"),
        ("po_e3", "pay_e3"),
    }
    for from_id, to_id in expected_df_pairs:
        targets = set(
            reasoner.object_property_values(_ind(from_id), df_prop, direct=False)
        )
        assert _ind(to_id) in targets, f"Missing df_order: {from_id} → {to_id}"

    for end_id in ("po_e4", "pay_e4", "pay_e3"):
        targets = list(
            reasoner.object_property_values(_ind(end_id), df_prop, direct=False)
        )
        assert len(targets) == 0, (
            f"EndEvent {end_id} has unexpected df_order assertion(s)"
        )

    cross_pairs = [("po_e1", "po_e2"), ("po_e1", "po_e3"), ("po_e2", "po_e3")]
    for from_id, to_id in cross_pairs:
        targets = set(
            reasoner.object_property_values(_ind(from_id), df_prop, direct=False)
        )
        assert _ind(to_id) not in targets, (
            f"Unexpected cross-trace df_order: {from_id} → {to_id}"
        )

    assert _cls("StartEvent_order") in classes
    assert _cls("EndEvent_order") in classes

    start_inds = {i.iri.remainder for i in reasoner.instances(_cls("StartEvent_order"))}
    end_inds = {i.iri.remainder for i in reasoner.instances(_cls("EndEvent_order"))}

    for start_id in ("po_e1", "po_e2", "po_e3"):
        assert start_id in start_inds, f"{start_id} not in StartEvent_order"
    for end_id in ("po_e4", "pay_e4", "pay_e3"):
        assert end_id in end_inds, f"{end_id} not in EndEvent_order"

    for mid_id in ("pay_e1", "pay_e2"):
        assert mid_id not in start_inds, f"{mid_id} unexpectedly in StartEvent_order"
        assert mid_id not in end_inds, f"{mid_id} unexpectedly in EndEvent_order"


def test_full_pipeline(cleaned_oced, upper_owl):
    builder = OCEDOntoBuilder(DEFAULT_IRI, _make_settings())
    builder.build_ontology(cleaned_oced)
    builder.sink.save(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    classes = set(onto.classes_in_signature())
    obj_props = set(onto.object_properties_in_signature())
    data_props = set(onto.data_properties_in_signature())

    for cls_name in (
        "place_order",
        "pay",
        "ship_order",
        "order",
        "item",
        "StartEvent_order",
        "EndEvent_order",
        "resource",
        "amount",
        "weight",
        "status",
        "carrier",
    ):
        assert _cls(cls_name) in classes, f"Missing class: {cls_name}"

    event_cls = _cls("Event")
    object_cls = _cls("Object")
    attr_cls = _cls("Attribute")

    for act in ("place_order", "pay", "ship_order"):
        supers = set(reasoner.super_classes(_cls(act), direct=False))
        assert event_cls in supers, f"{act} not subclass of Event"

    for obj_type in ("order", "item"):
        supers = set(reasoner.super_classes(_cls(obj_type), direct=False))
        assert object_cls in supers, f"{obj_type} not subclass of Object"

    event_attr_cls = _cls("EventAttribute")
    obj_attr_cls = _cls("ObjectAttribute")

    supers_amount = set(reasoner.super_classes(_cls("amount"), direct=False))
    assert attr_cls in supers_amount, "amount not subclass of Attribute"
    assert event_attr_cls in supers_amount, "amount not subclass of EventAttribute"

    supers_status = set(reasoner.super_classes(_cls("status"), direct=False))
    assert obj_attr_cls in supers_status, "status not subclass of ObjectAttribute"

    for act in ("place_order", "pay", "ship_order"):
        assert list(reasoner.instances(_cls(act))), f"No individuals for activity {act}"
    for obj_type in ("order", "item"):
        assert list(reasoner.instances(_cls(obj_type))), (
            f"No individuals for object type {obj_type}"
        )
    time_cls = _cls("Time")
    assert list(reasoner.instances(time_cls)), "No Time individuals"

    for prop_name in (
        "observed_at",
        "modifies",
        "ships",
        "contains_o2o",
        "df_order",
    ):
        assert _oprop(prop_name) in obj_props, f"Missing object property: {prop_name}"
    assert _dprop("has_data_value") in data_props

    start_inds = list(reasoner.instances(_cls("StartEvent_order")))
    assert len(start_inds) == 3, (
        f"Expected 3 StartEvent_order instances, got {len(start_inds)}"
    )


def _get_labels(owl_path: str, iri_fragment: str) -> list[str]:
    import owlready2

    onto = owlready2.get_ontology(f"file://{owl_path}").load()
    entity = onto[iri_fragment]
    return list(entity.label) if entity is not None else []


def test_derive_annotations_emits_class_labels_from_name_backmap(tmp_path):
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["e1", "e2"],
            "ts": pd.date_range("2022-01-01", periods=2, tz="UTC"),
        }
    )
    oced.add_event("Place Order", "eid", "ts", ev)
    oced.add_object("Purchase Item", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced = clean_oced_data(oced)

    settings = _make_settings()
    settings = settings.replace(derive_annotations=True)

    owl_path = str(tmp_path / "ann.owl")
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.events.emit(oced)
    builder.objects.emit(oced)
    builder.sink.save(owl_path)

    assert _get_labels(owl_path, "Place_Order") == ["Place Order"]
    assert _get_labels(owl_path, "Purchase_Item") == ["Purchase Item"]


def test_derive_annotations_individual_labels_use_original_names(tmp_path):
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["order-1", "order-2"],
            "ts": pd.date_range("2022-01-01", periods=2, tz="UTC"),
        }
    )
    oced.add_event("Place Order", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["item-1"]}))
    oced = clean_oced_data(oced)

    settings = _make_settings()
    settings = settings.replace(derive_annotations=True)

    owl_path = str(tmp_path / "ann2.owl")
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.events.emit(oced)
    builder.objects.emit(oced)
    builder.sink.save(owl_path)

    assert _get_labels(owl_path, "order_1") == ["Place Order (order-1)"]
    assert _get_labels(owl_path, "order_2") == ["Place Order (order-2)"]
    assert _get_labels(owl_path, "item_1") == ["obj (item-1)"]


def test_derive_annotations_attribute_class_label(tmp_path):
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["e1"],
            "ts": pd.date_range("2022-01-01", periods=1, tz="UTC"),
            "total cost": ["high"],
        }
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced = clean_oced_data(oced)

    settings = _make_settings()
    settings = settings.replace(derive_annotations=True)

    owl_path = str(tmp_path / "ann3.owl")
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.events.emit(oced)
    builder.sink.save(owl_path)

    assert _get_labels(owl_path, "total_cost") == ["total cost"]


def test_derive_annotations_attribute_value_label_uses_original_text(tmp_path):
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["e1"],
            "ts": pd.date_range("2022-01-01", periods=1, tz="UTC"),
            "subject": ["Hello World!"],
        }
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced = clean_oced_data(oced)

    settings = _make_settings()
    settings = settings.replace(derive_annotations=True)

    owl_path = str(tmp_path / "ann4.owl")
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings)
    builder.events.emit(oced)
    builder.sink.save(owl_path)

    assert _get_labels(owl_path, "subject_Hello_World_") == ["Hello World!"]
