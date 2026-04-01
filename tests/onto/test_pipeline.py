"""
Integration tests for the full semlog pipeline.

Covers: PM4PYLoader → clean_oced_data → OCEDUpperOntoBuilder →
OCEDOntoBuilder (events, objects, relations, DF order) → OWL assertions.

The static OCEL fixture is generated on first import and reused across all test
runs.  All OWL assertions are verified with owlapy StructuralReasoner.
"""

import copy
import os
import re

import pandas as pd
import pm4py
import pytest
from owlapy.class_expression import OWLClass
from owlapy.iri import IRI
from owlapy.owl_individual import OWLNamedIndividual
from owlapy.owl_ontology import Ontology
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty
from owlapy.owl_reasoner import StructuralReasoner
from pm4py.objects.ocel.obj import OCEL

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.pm4pyloader import PM4PYLoader
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    DataRepresentation,
    EventOrderRepresentation,
    OntoBuildSettings,
    OwlProfile,
)
from semanticlog.onto.upper import OCEDUpperOntoBuilder

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

PIPELINE_OCEL_PATH = "tests/fixtures/pipeline.jsonocel"

NS = DEFAULT_IRI


# ---------------------------------------------------------------------------
# Fixture construction helpers
# ---------------------------------------------------------------------------


def _empty_o2o() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ocel:oid": pd.Series([], dtype=str),
            "ocel:oid_2": pd.Series([], dtype=str),
            "ocel:qualifier": pd.Series([], dtype=str),
        }
    )


def _write_pipeline_ocel(path: str) -> None:
    """Build and write the pipeline integration-test OCEL fixture to *path*.

    Event and object attributes are embedded inline in the events/objects
    DataFrames so they survive the pm4py JSON round-trip without needing
    the event_changes table (not supported in the installed pm4py version).
    """

    # ------------------------------------------------------------------
    # Timestamps: day N = 2024-01-01 + (N-1) days
    # ------------------------------------------------------------------
    def _day(n: int) -> pd.Timestamp:
        return pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=n - 1)

    # ------------------------------------------------------------------
    # Events — attributes inline per activity
    # ------------------------------------------------------------------
    events = pd.DataFrame(
        {
            "ocel:eid": [
                "po_e1",
                "po_e2",
                "po_e3",
                "po_e4",
                "pay_e1",
                "pay_e2",
                "pay_e3",
                "pay_e4",
                "so_e1",
                "so_e2",
                "so_e3",
                "so_e4",
            ],
            "ocel:activity": [
                "place_order",
                "place_order",
                "place_order",
                "place_order",
                "pay",
                "pay",
                "pay",
                "pay",
                "ship_order",
                "ship_order",
                "ship_order",
                "ship_order",
            ],
            "ocel:timestamp": [
                _day(1),
                _day(2),
                _day(4),
                _day(8),
                _day(3),
                _day(5),
                _day(7),
                _day(9),
                _day(10),
                _day(11),
                _day(12),
                _day(13),
            ],
            # place_order/pay share 'amount' and 'resource'; ship_order gets
            # 'carrier' and 'weight'.  Nulls per activity row are dropped by
            # PM4PYLoader's per-type dropna(axis=1, how='all').
            "amount": [
                100.0,
                200.0,
                150.0,
                80.0,
                100.0,
                200.0,
                150.0,
                80.0,
                None,
                None,
                None,
                None,
            ],
            "resource": [
                "Alice",
                "Bob",
                "Alice",
                "Bob",
                "Alice",
                "Bob",
                "Bob",
                "Alice",
                None,
                None,
                None,
                None,
            ],
            "carrier": [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                "DHL",
                "UPS",
                "DHL",
                "UPS",
            ],
            "weight": [
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                2.5,
                3.0,
                1.5,
                4.0,
            ],
        }
    )

    # ------------------------------------------------------------------
    # Objects — attributes inline per type
    # ------------------------------------------------------------------
    objects = pd.DataFrame(
        {
            "ocel:oid": [
                "ord_1",
                "ord_2",
                "ord_3",
                "item_1",
                "item_2",
                "item_3",
                "item_4",
            ],
            "ocel:type": [
                "order",
                "order",
                "order",
                "item",
                "item",
                "item",
                "item",
            ],
            # 'status' only applies to orders; 'weight' only to items.
            # Nulls are dropped by PM4PYLoader's per-type dropna.
            "status": ["pending", "shipped", "pending", None, None, None, None],
            "weight": [None, None, None, 2.5, 3.0, 1.5, 4.0],
        }
    )

    # ------------------------------------------------------------------
    # E2O relations
    # ------------------------------------------------------------------
    modifies_eids = [
        "po_e1",
        "po_e2",
        "po_e3",
        "po_e4",
        "pay_e1",
        "pay_e2",
        "pay_e3",
        "pay_e4",
    ]
    modifies_oids = [
        "ord_1",
        "ord_2",
        "ord_3",
        "ord_1",
        "ord_1",
        "ord_2",
        "ord_3",
        "ord_2",
    ]
    modifies_ts = [
        _day(1),
        _day(2),
        _day(4),
        _day(8),
        _day(3),
        _day(5),
        _day(7),
        _day(9),
    ]

    ships_eids = ["so_e1", "so_e2", "so_e3", "so_e4"]
    ships_oids = ["item_1", "item_2", "item_3", "item_4"]
    ships_ts = [_day(10), _day(11), _day(12), _day(13)]

    relations = pd.DataFrame(
        {
            "ocel:eid": modifies_eids + ships_eids,
            "ocel:activity": (["place_order"] * 4 + ["pay"] * 4) + (["ship_order"] * 4),
            "ocel:timestamp": modifies_ts + ships_ts,
            "ocel:oid": modifies_oids + ships_oids,
            "ocel:type": (["order"] * 8) + (["item"] * 4),
            "ocel:qualifier": (["modifies"] * 8) + (["ships"] * 4),
        }
    )

    # ------------------------------------------------------------------
    # O2O: contains — order → item
    # ------------------------------------------------------------------
    o2o = pd.DataFrame(
        {
            "ocel:oid": ["ord_1", "ord_1", "ord_2", "ord_3"],
            "ocel:oid_2": ["item_1", "item_2", "item_3", "item_4"],
            "ocel:qualifier": ["contains", "contains", "contains", "contains"],
        }
    )

    ocel = OCEL(events=events, objects=objects, relations=relations, o2o=o2o)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pm4py.write_ocel2_json(ocel, path)


# Generate fixture on import if missing
if not os.path.exists(PIPELINE_OCEL_PATH):
    _write_pipeline_ocel(PIPELINE_OCEL_PATH)


# ---------------------------------------------------------------------------
# Settings / reasoner helpers
# ---------------------------------------------------------------------------


def _make_settings() -> OntoBuildSettings:
    """Return OntoBuildSettings configured for pipeline integration tests."""
    s = OntoBuildSettings()
    s.profile = OwlProfile.OWL2EL
    s.order_representation = [EventOrderRepresentation.DF]
    s.case_concepts = ["order"]
    s.data_representation = DataRepresentation.VALUES
    s.inverse_properties = False
    s.make_inverses_explicit = False
    return s


def _load_reasoner(owl_path: str):
    """Load an owlapy Ontology + StructuralReasoner from *owl_path*."""
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


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_oced():
    """Load the pipeline OCEL fixture via PM4PYLoader (no cleaning)."""
    return PM4PYLoader(PIPELINE_OCEL_PATH, {}).get_oced()


@pytest.fixture(scope="module")
def cleaned_oced(raw_oced):
    """Return a cleaned deep-copy of *raw_oced* (cleaning mutates in place)."""
    return clean_oced_data(copy.deepcopy(raw_oced))


@pytest.fixture
def upper_owl(tmp_path) -> str:
    """Build upper ontology into a temp file and return its absolute path."""
    owl_path = str(tmp_path / "tmp.owl")
    OCEDUpperOntoBuilder(DEFAULT_IRI, _make_settings()).construct_upper_ontology(
        owl_path
    )
    return owl_path


# ---------------------------------------------------------------------------
# Tests — loader
# ---------------------------------------------------------------------------


def test_loader_produces_expected_store(raw_oced):
    """PM4PYLoader populates OCEDStore with the correct activities, objects, and relations."""
    assert set(raw_oced.get_activities()) == {"place_order", "pay", "ship_order"}
    assert set(raw_oced.get_object_types()) == {"order", "item"}
    assert set(raw_oced.get_relation_types("e2o")) == {"modifies", "ships"}
    assert set(raw_oced.get_relation_types("o2o")) == {"contains_o2o"}
    assert len(raw_oced.get_ids("event", "place_order")) == 4
    assert len(raw_oced.get_ids("event", "pay")) == 4
    assert len(raw_oced.get_ids("event", "ship_order")) == 4
    assert len(raw_oced.get_ids("object", "order")) == 3
    assert len(raw_oced.get_ids("object", "item")) == 4


# ---------------------------------------------------------------------------
# Tests — cleaning
# ---------------------------------------------------------------------------


def test_cleaning_sanitizes_store(raw_oced):
    """clean_oced_data produces names matching [A-Za-z0-9_]+ and preserves identity."""
    cleaned = clean_oced_data(copy.deepcopy(raw_oced))

    valid = re.compile(r"^[A-Za-z0-9_]+$")
    for name in (
        cleaned.get_activities()
        + cleaned.get_object_types()
        + cleaned.get_relation_types()
    ):
        assert valid.match(name), f"Name not sanitized: {name!r}"

    assert set(cleaned.get_activities()) == {"place_order", "pay", "ship_order"}
    assert set(cleaned.get_object_types()) == {"order", "item"}

    # Spot-check that key attribute columns survived
    po_df = cleaned.get_events("place_order")["data"]
    assert "amount" in po_df.columns
    assert "resource" in po_df.columns

    so_df = cleaned.get_events("ship_order")["data"]
    assert "carrier" in so_df.columns

    ord_df = cleaned.get_object("order")["data"]
    assert "status" in ord_df.columns

    item_df = cleaned.get_object("item")["data"]
    assert "weight" in item_df.columns


# ---------------------------------------------------------------------------
# Tests — event instantiation
# ---------------------------------------------------------------------------


def test_event_instantiation(cleaned_oced, upper_owl):
    """Event classes, individuals, and attribute individuals are created correctly."""
    builder = OCEDOntoBuilder(upper_owl, _make_settings())
    builder.instantiate_event_data(cleaned_oced)
    builder.save_onto(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    classes = set(onto.classes_in_signature())
    inds = set(onto.individuals_in_signature())

    # Activity classes
    for act in ("place_order", "pay", "ship_order"):
        assert _cls(act) in classes, f"Missing class: {act}"

    # All place_order individuals present
    expected_po_ids = set(cleaned_oced.get_ids("event", "place_order"))
    actual_po_inds = {i.iri.remainder for i in reasoner.instances(_cls("place_order"))}
    assert expected_po_ids == actual_po_inds

    # String attribute class resource + its individuals (cleaning casefolding lowercases values)
    assert _cls("resource") in classes
    assert _ind("resource_alice") in inds
    assert _ind("resource_bob") in inds
    assert _ind("resource_alice") in set(reasoner.instances(_cls("resource")))
    assert _ind("resource_bob") in set(reasoner.instances(_cls("resource")))

    # Numeric attribute class amount + a value individual
    assert _cls("amount") in classes
    # has_data_value on numeric individuals
    has_data_value = _dprop("has_data_value")
    assert has_data_value in set(onto.data_properties_in_signature())
    amount_inds = [i for i in inds if i.iri.remainder.startswith("amount_")]
    assert len(amount_inds) > 0
    sample_amount = amount_inds[0]
    literals = list(
        reasoner.data_property_values(sample_amount, has_data_value, direct=False)
    )
    assert len(literals) > 0, f"No has_data_value literal on {sample_amount}"

    # String individuals must NOT have has_data_value literals
    for str_ind in (_ind("resource_alice"), _ind("resource_bob")):
        literals = list(
            reasoner.data_property_values(str_ind, has_data_value, direct=False)
        )
        assert len(literals) == 0, f"Unexpected has_data_value on {str_ind}"


# ---------------------------------------------------------------------------
# Tests — object instantiation
# ---------------------------------------------------------------------------


def test_object_instantiation(cleaned_oced, upper_owl):
    """Object classes, individuals, and attribute individuals are created correctly."""
    builder = OCEDOntoBuilder(upper_owl, _make_settings())
    builder.instantiate_object_data(cleaned_oced)
    builder.save_onto(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    classes = set(onto.classes_in_signature())
    inds = set(onto.individuals_in_signature())

    assert _cls("order") in classes
    assert _cls("item") in classes

    expected_order_ids = set(cleaned_oced.get_ids("object", "order"))
    actual_order_inds = {i.iri.remainder for i in reasoner.instances(_cls("order"))}
    assert expected_order_ids == actual_order_inds

    # String attr class status + value individuals (casefolded after cleaning)
    assert _cls("status") in classes
    assert _ind("status_pending") in inds
    assert _ind("status_shipped") in inds
    assert _ind("status_pending") in set(reasoner.instances(_cls("status")))
    assert _ind("status_shipped") in set(reasoner.instances(_cls("status")))

    # Numeric attr class weight + has_data_value literals
    assert _cls("weight") in classes
    has_data_value = _dprop("has_data_value")
    weight_inds = [i for i in inds if i.iri.remainder.startswith("weight_")]
    assert len(weight_inds) > 0
    sample_weight = weight_inds[0]
    literals = list(
        reasoner.data_property_values(sample_weight, has_data_value, direct=False)
    )
    assert len(literals) > 0, f"No has_data_value literal on {sample_weight}"


# ---------------------------------------------------------------------------
# Tests — e2o relation instantiation
# ---------------------------------------------------------------------------


def test_e2o_relation_instantiation(cleaned_oced, upper_owl):
    """E2O OWL property assertions match every row in the cleaned OCEDStore."""
    builder = OCEDOntoBuilder(upper_owl, _make_settings())
    builder.instantiate_event_data(cleaned_oced)
    builder.instantiate_object_data(cleaned_oced)
    builder.instantiate_event_object_relation_data(cleaned_oced)
    builder.save_onto(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    obj_props = set(onto.object_properties_in_signature())

    modifies_prop = _oprop("modifies")
    ships_prop = _oprop("ships")
    assert modifies_prop in obj_props, "Property 'modifies' missing"
    assert ships_prop in obj_props, "Property 'ships' missing"

    # Every modifies row must have a corresponding OWL assertion
    mod_rel = cleaned_oced.get_relation("modifies")
    from_col = mod_rel["from_id"]
    to_col = mod_rel["to_id"]
    for _, row in mod_rel["data"].iterrows():
        from_ind = _ind(str(row[from_col]))
        to_ind = _ind(str(row[to_col]))
        targets = set(
            reasoner.object_property_values(from_ind, modifies_prop, direct=False)
        )
        assert (
            to_ind in targets
        ), f"Missing modifies assertion: {row[from_col]} → {row[to_col]}"

    # Every ships row must have a corresponding OWL assertion
    ships_rel = cleaned_oced.get_relation("ships")
    from_col_s = ships_rel["from_id"]
    to_col_s = ships_rel["to_id"]
    for _, row in ships_rel["data"].iterrows():
        from_ind = _ind(str(row[from_col_s]))
        to_ind = _ind(str(row[to_col_s]))
        targets = set(
            reasoner.object_property_values(from_ind, ships_prop, direct=False)
        )
        assert (
            to_ind in targets
        ), f"Missing ships assertion: {row[from_col_s]} → {row[to_col_s]}"

    # Cross-type sanity: no ship event should have a modifies assertion to an item
    item_ids = set(str(i) for i in cleaned_oced.get_ids("object", "item"))
    for eid in cleaned_oced.get_ids("event", "ship_order"):
        targets = set(
            reasoner.object_property_values(_ind(str(eid)), modifies_prop, direct=False)
        )
        for t in targets:
            assert (
                t.iri.remainder not in item_ids
            ), f"ship_order event {eid} has unexpected modifies assertion to item {t}"


# ---------------------------------------------------------------------------
# Tests — o2o relation instantiation
# ---------------------------------------------------------------------------


def test_o2o_relation_instantiation(cleaned_oced, upper_owl):
    """O2O OWL property assertions match every row in the cleaned OCEDStore."""
    builder = OCEDOntoBuilder(upper_owl, _make_settings())
    builder.instantiate_event_data(cleaned_oced)
    builder.instantiate_object_data(cleaned_oced)
    builder.instantiate_object_to_object_relation_data(cleaned_oced)
    builder.save_onto(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    obj_props = set(onto.object_properties_in_signature())

    contains_prop = _oprop("contains_o2o")
    assert contains_prop in obj_props, "Property 'contains_o2o' missing"

    # Every o2o row must have a corresponding OWL assertion
    o2o_rel = cleaned_oced.get_relation("contains_o2o")
    from_col = o2o_rel["from_id"]
    to_col = o2o_rel["to_id"]
    for _, row in o2o_rel["data"].iterrows():
        from_ind = _ind(str(row[from_col]))
        to_ind = _ind(str(row[to_col]))
        targets = set(
            reasoner.object_property_values(from_ind, contains_prop, direct=False)
        )
        assert (
            to_ind in targets
        ), f"Missing contains_o2o assertion: {row[from_col]} → {row[to_col]}"

    # No reverse assertions (item → order)
    item_ids = set(str(i) for i in cleaned_oced.get_ids("object", "item"))
    for iid in item_ids:
        targets = set(
            reasoner.object_property_values(_ind(iid), contains_prop, direct=False)
        )
        order_ids = set(str(i) for i in cleaned_oced.get_ids("object", "order"))
        for t in targets:
            assert (
                t.iri.remainder not in order_ids
            ), f"Unexpected reverse contains_o2o from item {iid} to order {t}"


# ---------------------------------------------------------------------------
# Tests — event order (DF) instantiation
# ---------------------------------------------------------------------------


def test_event_order_instantiation(cleaned_oced, upper_owl):
    """Trace structure, DF pairs, and Start/EndEvent membership are correct."""
    builder = OCEDOntoBuilder(upper_owl, _make_settings())
    builder.instantiate_event_data(cleaned_oced)
    builder.instantiate_event_order(cleaned_oced)
    builder.save_onto(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    classes = set(onto.classes_in_signature())
    obj_props = set(onto.object_properties_in_signature())

    # ---- Trace structure ----
    assert _cls("Trace_order") in classes
    trace_inds = {i.iri.remainder for i in reasoner.instances(_cls("Trace_order"))}
    assert trace_inds == {"ord_1", "ord_2", "ord_3"}

    assert _oprop("part_of") in obj_props

    # All events in modifies relation have a part_of assertion to their trace
    mod_rel = cleaned_oced.get_relation("modifies")
    from_col = mod_rel["from_id"]
    to_col = mod_rel["to_id"]
    part_of_prop = _oprop("part_of")
    for _, row in mod_rel["data"].iterrows():
        ev_ind = _ind(str(row[from_col]))
        trace_ind = _ind(str(row[to_col]))
        targets = set(
            reasoner.object_property_values(ev_ind, part_of_prop, direct=False)
        )
        assert trace_ind in targets, f"Missing part_of: {row[from_col]} → {row[to_col]}"

    # ---- DF correctness ----
    assert _oprop("df_order") in obj_props
    df_prop = _oprop("df_order")

    # Expected DF pairs (derived from trace membership + timestamp ordering):
    # ord_1 events by time: po_e1(d1), pay_e1(d3), po_e4(d8) → pairs: po_e1→pay_e1, pay_e1→po_e4
    # ord_2 events by time: po_e2(d2), pay_e2(d5), pay_e4(d9) → pairs: po_e2→pay_e2, pay_e2→pay_e4
    # ord_3 events by time: po_e3(d4), pay_e3(d7) → pair: po_e3→pay_e3
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

    # EndEvents must not have outgoing df_order assertions
    for end_id in ("po_e4", "pay_e4", "pay_e3"):
        targets = list(
            reasoner.object_property_values(_ind(end_id), df_prop, direct=False)
        )
        assert (
            len(targets) == 0
        ), f"EndEvent {end_id} has unexpected df_order assertion(s)"

    # No cross-trace df_order assertion
    cross_pairs = [("po_e1", "po_e2"), ("po_e1", "po_e3"), ("po_e2", "po_e3")]
    for from_id, to_id in cross_pairs:
        targets = set(
            reasoner.object_property_values(_ind(from_id), df_prop, direct=False)
        )
        assert (
            _ind(to_id) not in targets
        ), f"Unexpected cross-trace df_order: {from_id} → {to_id}"

    # ---- Start/End events ----
    assert _cls("StartEvent") in classes
    assert _cls("EndEvent") in classes

    start_inds = {i.iri.remainder for i in reasoner.instances(_cls("StartEvent"))}
    end_inds = {i.iri.remainder for i in reasoner.instances(_cls("EndEvent"))}

    for start_id in ("po_e1", "po_e2", "po_e3"):
        assert start_id in start_inds, f"{start_id} not in StartEvent"
    for end_id in ("po_e4", "pay_e4", "pay_e3"):
        assert end_id in end_inds, f"{end_id} not in EndEvent"

    # Middle events are neither Start nor End
    for mid_id in ("pay_e1", "pay_e2"):
        assert mid_id not in start_inds, f"{mid_id} unexpectedly in StartEvent"
        assert mid_id not in end_inds, f"{mid_id} unexpectedly in EndEvent"


# ---------------------------------------------------------------------------
# Tests — full pipeline
# ---------------------------------------------------------------------------


def test_full_pipeline(cleaned_oced, upper_owl):
    """End-to-end build_ontology produces a well-formed, correctly populated OWL file."""
    builder = OCEDOntoBuilder(upper_owl, _make_settings())
    builder.build_ontology(cleaned_oced)
    builder.save_onto(upper_owl)

    onto, reasoner = _load_reasoner(upper_owl)
    classes = set(onto.classes_in_signature())
    obj_props = set(onto.object_properties_in_signature())
    data_props = set(onto.data_properties_in_signature())

    # ---- Class existence ----
    for cls_name in (
        "place_order",
        "pay",
        "ship_order",
        "order",
        "item",
        "Trace_order",
        "StartEvent",
        "EndEvent",
        "resource",
        "amount",
        "weight",
        "status",
        "carrier",
    ):
        assert _cls(cls_name) in classes, f"Missing class: {cls_name}"

    # ---- Class hierarchy ----
    event_cls = _cls("Event")
    object_cls = _cls("Object")
    attr_cls = _cls("Attribute")

    for act in ("place_order", "pay", "ship_order"):
        supers = set(reasoner.super_classes(_cls(act), direct=False))
        assert event_cls in supers, f"{act} not subclass of Event"

    for obj_type in ("order", "item"):
        supers = set(reasoner.super_classes(_cls(obj_type), direct=False))
        assert object_cls in supers, f"{obj_type} not subclass of Object"

    supers_amount = set(reasoner.super_classes(_cls("amount"), direct=False))
    assert attr_cls in supers_amount, "amount not subclass of Attribute"

    # ---- Individual existence ----
    for act in ("place_order", "pay", "ship_order"):
        assert list(reasoner.instances(_cls(act))), f"No individuals for activity {act}"
    for obj_type in ("order", "item"):
        assert list(
            reasoner.instances(_cls(obj_type))
        ), f"No individuals for object type {obj_type}"
    time_cls = _cls("Time")
    assert list(reasoner.instances(time_cls)), "No Time individuals"

    # ---- Property existence ----
    for prop_name in (
        "observed_at",
        "modifies",
        "ships",
        "contains_o2o",
        "df_order",
        "part_of",
    ):
        assert _oprop(prop_name) in obj_props, f"Missing object property: {prop_name}"
    assert _dprop("has_data_value") in data_props

    # ---- Reasoner inference: Trace_order has exactly 3 instances ----
    trace_inds = list(reasoner.instances(_cls("Trace_order")))
    assert (
        len(trace_inds) == 3
    ), f"Expected 3 Trace_order instances, got {len(trace_inds)}"

    # ---- StartEvent has exactly 3 instances ----
    start_inds = list(reasoner.instances(_cls("StartEvent")))
    assert (
        len(start_inds) == 3
    ), f"Expected 3 StartEvent instances, got {len(start_inds)}"
