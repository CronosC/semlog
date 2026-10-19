"""The shape of `tests/fixtures/pipeline.jsonocel` that its README describes and other tests rely on."""

from __future__ import annotations

from semanticlog.data.tables import EntityKind, RelationKind


def test_the_log_holds_three_activities_two_object_types_and_their_relations(
    pipeline_raw_store,
):
    raw = pipeline_raw_store

    assert set(raw.types(EntityKind.EVENT)) == {"place_order", "pay", "ship_order"}
    assert set(raw.types(EntityKind.OBJECT)) == {"order", "item"}
    assert set(raw.get_relation_types(RelationKind.E2O)) == {"modifies", "ships"}
    assert set(raw.get_relation_types(RelationKind.O2O)) == {"contains"}


def test_the_log_holds_four_events_per_activity_and_the_documented_object_ids(
    pipeline_raw_store,
):
    raw = pipeline_raw_store

    assert {
        a: len(raw.ids(EntityKind.EVENT, a)) for a in raw.types(EntityKind.EVENT)
    } == {
        "place_order": 4,
        "pay": 4,
        "ship_order": 4,
    }
    assert sorted(raw.ids(EntityKind.OBJECT, "order")) == ["ord_1", "ord_2", "ord_3"]
    assert sorted(raw.ids(EntityKind.OBJECT, "item")) == [
        "item_1",
        "item_2",
        "item_3",
        "item_4",
    ]


def test_each_type_keeps_the_attributes_the_readme_lists(pipeline_raw_store):
    def columns(kind, name):
        return set(pipeline_raw_store.table(kind, name).data.columns)

    assert {"amount", "resource"} <= columns(EntityKind.EVENT, "place_order")
    assert {"amount", "resource"} <= columns(EntityKind.EVENT, "pay")
    assert {"carrier", "weight"} <= columns(EntityKind.EVENT, "ship_order")
    assert "status" in columns(EntityKind.OBJECT, "order")
    assert "weight" in columns(EntityKind.OBJECT, "item")
