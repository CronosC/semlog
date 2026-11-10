"""The repairs both OCEL 2.0 loaders make to the tables of a log: qualifiers, numeric ids and latest values."""

import logging

import pandas as pd
import pytest

from semlog.core.enums import OCELExtension
from semlog.data.loading.ocel_frames import normalize_qualifiers, relation_names
from semlog.data.loading.ocel_loader import OCELLoader
from semlog.data.tables import EntityKind, RelationKind
from tests.support.ocel_files import EPOCH, small_log, write_log

HISTORY = [OCELExtension.ATTRIBUTE_HISTORY]

# --------------------------------------------------------------------------
# Qualifiers
# --------------------------------------------------------------------------


def test_a_missing_qualifier_takes_the_fallback_name_of_its_family(tmp_path):
    log = small_log()
    log["events"][0]["relationships"] = [
        {"objectId": "o1", "qualifier": "None"},
        {"objectId": "i1", "qualifier": "N/A"},
    ]
    log["events"][1]["relationships"] = []
    log["objects"][0]["relationships"] = [{"objectId": "i1", "qualifier": "NA"}]

    oced = OCELLoader(write_log(tmp_path, log)).get_oced()

    assert oced.relation_types(RelationKind.E2O) == ["involves"]
    assert oced.relation_types(RelationKind.O2O) == ["relates"]


def test_a_qualifier_both_families_use_is_suffixed_on_both_sides(tmp_path):
    log = small_log()
    log["events"][0]["relationships"] = [
        {"objectId": "o1", "qualifier": "contains"},
        {"objectId": "i1", "qualifier": "contains"},
    ]
    log["events"][1]["relationships"] = []

    oced = OCELLoader(write_log(tmp_path, log)).get_oced()

    assert oced.relation_types() == ["contains_e2o", "contains_o2o"]


def test_a_suffixed_name_the_log_uses_is_numbered_past():
    e2o, o2o = relation_names(
        pd.Series(["contains", "contains_e2o"]), pd.Series(["contains"])
    )

    assert e2o == {"contains": "contains_e2o_2", "contains_e2o": "contains_e2o"}
    assert o2o == {"contains": "contains_o2o"}


def test_a_word_read_as_no_qualifier_is_warned_about_apart_from_an_empty_one(caplog):
    rows = pd.DataFrame({"ocel:qualifier": ["modifies", None, "NA"]})

    with caplog.at_level(logging.WARNING):
        normalized = normalize_qualifiers(rows, "e2o")

    assert list(normalized["ocel:qualifier"]) == ["modifies", "involves", "involves"]
    assert "2 e2o relation(s) had no qualifier" in caplog.text
    assert (
        "1 of those carry the text (NA), which is read as no qualifier" in caplog.text
    )


def test_a_word_on_every_relation_of_a_family_is_not_warned_about_as_a_merge(caplog):
    rows = pd.DataFrame({"ocel:qualifier": ["None", "None"]})

    with caplog.at_level(logging.WARNING):
        normalize_qualifiers(rows, "e2o")

    warnings = [r.getMessage() for r in caplog.records]
    assert len(warnings) == 1
    assert warnings[0].startswith("2 e2o relation(s) had no qualifier")


# --------------------------------------------------------------------------
# Numeric ids
# --------------------------------------------------------------------------


def test_numeric_ids_are_prefixed_with_their_type_everywhere_they_appear(tmp_path):
    log = small_log()
    log["objects"] = [{"id": "10", "type": "order"}, {"id": "20", "type": "order"}]
    log["events"] = [
        {
            "id": eid,
            "type": "ship",
            "time": f"2024-01-0{eid}T00:00:00Z",
            "relationships": [{"objectId": oid, "qualifier": "ships"}],
        }
        for eid, oid in (("1", "10"), ("2", "20"))
    ]

    oced = OCELLoader(write_log(tmp_path, log)).get_oced()

    assert list(oced.ids(EntityKind.OBJECT, "order")) == ["order_10", "order_20"]
    assert list(oced.ids(EntityKind.EVENT, "ship")) == ["ship_1", "ship_2"]
    assert oced.relation("ships").data.values.tolist() == [
        ["ship_1", "order_10"],
        ["ship_2", "order_20"],
    ]


# --------------------------------------------------------------------------
# Latest values
# --------------------------------------------------------------------------


def _status_log(tmp_path, statuses: list[tuple[str, str]]) -> str:
    """Write a log whose order `o1` lists `statuses`, each as (time, value), in that order."""
    log = small_log()
    log["objects"][0]["attributes"] = [
        {"name": "status", "time": time, "value": value} for time, value in statuses
    ]
    return write_log(tmp_path, log)


def _status_of_o1(path: str, extensions=()) -> str:
    orders = OCELLoader(path, extensions).get_oced().table(EntityKind.OBJECT, "order")
    return orders.data.set_index("ocel:oid").loc["o1", "status"]


OUT_OF_ORDER = [
    ("2024-01-05T00:00:00Z", "shipped"),
    ("2024-01-01T00:00:00Z", "new"),
    ("2024-01-09T00:00:00Z", "delivered"),
    ("2024-01-09T00:00:00Z", "closed"),
]


def test_an_object_carries_the_latest_value_the_last_listed_among_equal_times(
    tmp_path,
):
    assert _status_of_o1(_status_log(tmp_path, OUT_OF_ORDER)) == "closed"


def test_the_latest_value_is_the_same_with_and_without_the_history_extension(
    tmp_path,
):
    path = _status_log(tmp_path, OUT_OF_ORDER)

    assert _status_of_o1(path) == _status_of_o1(path, HISTORY) == "closed"


def test_the_changes_hold_every_value_after_the_earliest_in_time_order(tmp_path):
    oced = OCELLoader(_status_log(tmp_path, OUT_OF_ORDER), HISTORY).get_oced()

    assert oced.object_changes("order")["status"].tolist() == [
        "shipped",
        "delivered",
        "closed",
    ]


@pytest.mark.parametrize("suffix", [".jsonocel", ".xmlocel", ".sqlite"])
def test_the_values_after_a_null_earliest_value_are_all_changes(tmp_path, suffix):
    log = small_log()
    log["objects"][0]["attributes"] = [
        {"name": "status", "time": EPOCH, "value": None},
        {"name": "status", "time": "2024-01-03T00:00:00Z", "value": "a"},
        {"name": "status", "time": "2024-01-05T00:00:00Z", "value": "b"},
    ]

    oced = OCELLoader(write_log(tmp_path, log, suffix), HISTORY).get_oced()
    changes = oced.object_changes("order")
    orders = oced.table(EntityKind.OBJECT, "order").data

    assert list(zip(changes["ocel:timestamp"], changes["status"], strict=True)) == [
        (pd.Timestamp("2024-01-03", tz="UTC"), "a"),
        (pd.Timestamp("2024-01-05", tz="UTC"), "b"),
    ]
    assert orders.set_index("ocel:oid").loc["o1", "status"] == "b"
