"""Renaming the ids and values whose individual name another claimant holds."""

import pandas as pd
import pytest

from semanticlog.core.errors import InputError
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.cleaning.individuals import verify_unique_names
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind


def _ids(oced, kind=EntityKind.OBJECT, name="obj"):
    return list(oced.ids(kind, name))


def _column(oced, column, obj_type="obj"):
    return list(oced.table(EntityKind.OBJECT, obj_type).data[column])


def test_a_value_whose_individual_is_a_derived_name_moves(make_store, events):
    """`x` with value `yChange` is `act.x.yChange`, the change class of `act.x.y`."""
    oced = make_store(events=events("e1", "e2", x=["yChange", "ok"], **{"x.y": [1, 2]}))

    cleaned = clean_oced_data(oced)

    assert list(cleaned.table(EntityKind.EVENT, "act").data["act.x"]) == [
        "value_yChange",
        "ok",
    ]
    assert cleaned.backmap["act.x.value_yChange"] == "yChange"


def test_a_moved_value_is_numbered_past_a_value_holding_its_new_name(
    make_store, events
):
    oced = make_store(
        events=events("e1", "e2", x=["yChange", "value_yChange"], **{"x.y": [1, 2]})
    )

    cleaned = clean_oced_data(oced)

    assert sorted(cleaned.table(EntityKind.EVENT, "act").data["act.x"]) == [
        "value_yChange",
        "value_yChange_2",
    ]


def test_an_id_named_after_an_object_type_moves(make_store):
    oced = make_store(objects=pd.DataFrame({"oid": ["obj", "o2"]}))

    cleaned = clean_oced_data(oced)

    assert _ids(cleaned) == ["id_obj", "o2"]
    assert cleaned.backmap["id_obj"] == "obj"


def test_an_id_named_after_a_value_individual_moves_and_the_value_stays(make_store):
    oced = make_store(
        objects=pd.DataFrame(
            {"oid": ["obj.status.open", "o2"], "status": ["open", "x"]}
        )
    )

    cleaned = clean_oced_data(oced)

    assert _ids(cleaned) == ["id_obj.status.open", "o2"]
    assert _column(cleaned, "obj.status") == ["open", "x"]


def test_an_id_named_after_a_timestamp_individual_moves(make_store):
    stamp = pd.date_range("2022-01-01", periods=1, tz="UTC")
    oced = make_store(
        events=pd.DataFrame({"eid": ["20220101T000000_000000"], "ts": stamp})
    )

    cleaned = clean_oced_data(oced)

    assert _ids(cleaned, EntityKind.EVENT, "act") == ["id_20220101T000000_000000"]


def test_a_moved_id_skips_an_id_holding_its_new_name(make_store):
    oced = make_store(objects=pd.DataFrame({"oid": ["obj", "id_obj"]}))
    oced.add_relation(
        RelationKind.E2O,
        "rel",
        pd.DataFrame({"eid": ["e1"], "oid": ["obj"]}),
        from_id="eid",
        to_id="oid",
        to_type=["obj"],
    )

    cleaned = clean_oced_data(oced)

    assert _ids(cleaned) == ["id_obj_2", "id_obj"]
    assert list(cleaned.get_relation("rel").data["oid"]) == ["id_obj_2"]


@pytest.mark.parametrize(("bin_labels", "moved"), [(["low"], True), ((), False)])
def test_an_id_named_after_a_bin_class_moves_only_when_the_build_bins(
    make_store, bin_labels, moved
):
    oced = make_store(
        objects=pd.DataFrame({"oid": ["obj.price_bin_low", "o2"], "price": [1.0, 2.0]})
    )

    cleaned = clean_oced_data(oced, bin_labels=bin_labels)

    assert ("obj.price_bin_low" in _ids(cleaned)) is not moved


def test_of_two_attributes_differing_by_a_dot_the_later_value_moves(make_store):
    """`a` with value `b.c` and `a.b` with value `c` are both `obj.a.b.c`."""
    oced = make_store(objects=pd.DataFrame({"oid": ["o1"], "a": ["b.c"], "a.b": ["c"]}))

    cleaned = clean_oced_data(oced)

    assert (_column(cleaned, "obj.a"), _column(cleaned, "obj.a.b")) == (
        ["b.c"],
        ["value_c"],
    )
    assert cleaned.backmap == {"obj.a": "a", "obj.a.b": "a.b", "obj.a.b.value_c": "c"}


def test_a_text_value_moves_for_a_number_it_collides_with(make_store):
    oced = make_store(objects=pd.DataFrame({"oid": ["o1"], "a": ["b.5"], "a.b": [5]}))

    cleaned = clean_oced_data(oced)

    assert (_column(cleaned, "obj.a"), _column(cleaned, "obj.a.b")) == (
        ["value_b.5"],
        [5],
    )


def test_a_number_whose_individual_another_name_holds_is_rejected(make_store):
    """Object type `order.price.5` has the name the value 5 of `order.price` would take."""
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o1"], "price": [5.0]}), obj_type="order"
    )
    oced.add(
        EntityKind.OBJECT, "order.price.5", pd.DataFrame({"x": ["p1"]}), id_col="x"
    )

    with pytest.raises(
        InputError,
        match=r"'order\.price\.5' is the name of both a value of attribute 'order\.price' and object type 'order\.price\.5'\. The value is not text",
    ):
        clean_oced_data(oced)


def test_two_changes_that_would_name_one_record_move_one_object_apart(make_store):
    """`o` changing `x_order.status` and `o_order.x` changing `status` at one time are one record."""
    oced = make_store(
        objects=pd.DataFrame({"oid": ["o", "o_order.x"]}),
        obj_type="order",
        ocel_mode=True,
    )
    oced.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o", "o_order.x"],
                "ocel:type": ["order", "order"],
                "ocel:timestamp": pd.to_datetime(["2024-01-02"] * 2, utc=True),
                "ocel:field": ["x_order.status", "status"],
                "x_order.status": ["v0", None],
                "status": [None, "v1"],
            }
        ),
    )

    cleaned = clean_oced_data(oced)

    assert _ids(cleaned, name="order") == ["o", "id_o_order.x"]
    assert list(cleaned.get_object_changes("order")["ocel:oid"]) == [
        "o",
        "id_o_order.x",
    ]


def test_every_claimant_of_a_shared_name_is_reported(events):
    oced = OCEDStore()
    oced.add(
        EntityKind.OBJECT, "x", pd.DataFrame({"oid": ["x"], "x": [1]}), id_col="oid"
    )
    oced.add(EntityKind.EVENT, "x", events("e1"), id_col="eid", time_col="ts")
    oced.add_relation(
        RelationKind.E2O,
        "x",
        pd.DataFrame({"eid": ["e1"], "oid": ["x"]}),
        from_id="eid",
        to_id="oid",
        to_type=["x"],
    )

    with pytest.raises(
        InputError,
        match=r"'x' is claimed 5 times: activity 'x', attribute 'x', id from object type 'x', object type 'x', relation 'x'",
    ):
        verify_unique_names(oced)


def test_a_timestamp_whose_individual_a_schema_name_holds_is_rejected(make_store):
    oced = make_store(obj_type="20220101T000000_000000")

    with pytest.raises(
        InputError,
        match=r"'20220101T000000_000000' is the name of both a timestamp and object type '20220101T000000_000000'\. Neither can be renamed\.",
    ):
        clean_oced_data(oced)


def test_a_name_still_shared_after_the_last_round_of_renames_is_rejected(make_store):
    """Each round moves object `o`, and its change record lands on the next id."""
    stamp = "20240102T000000_000000"
    ids = ["o"] + [f"AttrChange_{'id_' * k}o_order.status_{stamp}" for k in range(4)]
    oced = make_store(
        objects=pd.DataFrame({"oid": ids}), obj_type="order", ocel_mode=True
    )
    oced.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o"],
                "ocel:type": ["order"],
                "ocel:timestamp": pd.to_datetime(["2024-01-02"], utc=True),
                "ocel:field": ["status"],
                "status": ["x"],
            }
        ),
    )

    with pytest.raises(
        InputError,
        match=r"a change of attribute 'order\.status' of object 'id_id_id_o' after renaming\.\n\nRename one of them",
    ):
        clean_oced_data(oced)
