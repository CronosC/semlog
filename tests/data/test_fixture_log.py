"""The committed OCEL 2.0 log, read and cleaned.

The rest of the suite builds against this file, so what it holds is worth
stating once. The README beside it describes the same shape in prose.
"""

from __future__ import annotations

import re

from semanticlog.data.tables import EntityKind, RelationKind

SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def test_the_loader_reads_the_activities_and_object_types(pipeline_raw_store):
    raw = pipeline_raw_store

    assert set(raw.get_event_types()) == {"place_order", "pay", "ship_order"}
    assert set(raw.get_object_types()) == {"order", "item"}


def test_the_loader_reads_both_families_of_relation(pipeline_raw_store):
    raw = pipeline_raw_store

    assert set(raw.get_relation_types(RelationKind.E2O)) == {"modifies", "ships"}
    assert set(raw.get_relation_types(RelationKind.O2O)) == {"contains"}


def test_the_log_holds_four_events_of_each_activity(pipeline_raw_store):
    raw = pipeline_raw_store

    for activity in ("place_order", "pay", "ship_order"):
        assert len(raw.get_ids(EntityKind.EVENT, activity)) == 4


def test_the_log_holds_three_orders_and_four_items(pipeline_raw_store):
    raw = pipeline_raw_store

    assert len(raw.get_ids(EntityKind.OBJECT, "order")) == 3
    assert len(raw.get_ids(EntityKind.OBJECT, "item")) == 4


def test_every_name_is_safe_to_emit_after_cleaning(pipeline_store):
    names = (
        pipeline_store.get_event_types()
        + pipeline_store.get_object_types()
        + pipeline_store.get_relation_types()
    )

    unsafe = [name for name in names if not SAFE_NAME.match(name)]
    assert not unsafe, f"cleaning left {unsafe} unusable as IRI local names"


def test_cleaning_keeps_the_attributes_each_type_carries(pipeline_store):
    assert {"amount", "resource"} <= set(
        pipeline_store.get_event_table("place_order").data.columns
    )
    assert "carrier" in pipeline_store.get_event_table("ship_order").data.columns
    assert "status" in pipeline_store.get_object_table("order").data.columns
    assert "weight" in pipeline_store.get_object_table("item").data.columns
