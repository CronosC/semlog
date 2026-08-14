"""Finding relations that point at entities the store does not hold."""

import logging
import re

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.flatten import get_log
from semanticlog.data.integrity import check_integrity as check_oced_integrity
from semanticlog.data.oced import OCEDStore

CLEAN_RE = re.compile(r"^[A-Za-z0-9_]+$")


def _consistent_oced() -> OCEDStore:
    oced = OCEDStore()
    events = pd.DataFrame(
        {"eid": ["e1", "e2", "e3"], "ts": pd.date_range("2024-01-01", periods=3)}
    )
    objects = pd.DataFrame({"oid": ["o1", "o2"], "attr": ["a", "b"]})
    relations = pd.DataFrame({"eid": ["e1", "e2", "e3"], "oid": ["o1", "o1", "o2"]})
    oced.add_event("activity", "eid", "ts", events)
    oced.add_object("thing", "oid", objects)
    oced.add_event_to_object_relation("modifies", "eid", "oid", ["thing"], relations)
    return oced


def test_clean_store_has_no_violations():
    assert check_oced_integrity(_consistent_oced()) == []


def test_detects_missing_object_id_in_e2o():
    oced = _consistent_oced()
    bad_rel = pd.DataFrame({"eid": ["e1"], "oid": ["DOES_NOT_EXIST"]})
    oced.add_event_to_object_relation(
        "bad_rel", "eid", "oid", ["thing"], bad_rel, overwrite=False
    )
    violations = check_oced_integrity(oced)
    assert any("DOES_NOT_EXIST" in v or "1 object IDs" in v for v in violations)


def test_detects_missing_event_id_in_e2o():
    oced = _consistent_oced()
    bad_rel = pd.DataFrame({"eid": ["GHOST_EVENT"], "oid": ["o1"]})
    oced.add_event_to_object_relation(
        "ghost_rel", "eid", "oid", ["thing"], bad_rel, overwrite=False
    )
    violations = check_oced_integrity(oced)
    assert any("GHOST_EVENT" in v or "event IDs" in v for v in violations)


def test_detects_an_event_with_no_id():
    oced = _consistent_oced()
    events = pd.DataFrame(
        {"eid": ["e4", None], "ts": pd.date_range("2024-02-01", periods=2)}
    )
    oced.add_event("nameless", "eid", "ts", events)

    violations = check_oced_integrity(oced)

    assert violations == ["activity 'nameless': 1 event(s) have no 'eid'"]


def test_detects_an_object_with_no_id():
    """The row has an attribute, so dtype inference keeps it as a nameless object."""
    oced = _consistent_oced()
    oced.add_object(
        "nameless", "oid", pd.DataFrame({"oid": [None, "x1"], "kind": ["a", "b"]})
    )

    violations = check_oced_integrity(oced)

    assert violations == ["object type 'nameless': 1 object(s) have no 'oid'"]


def test_detects_missing_id_in_o2o():
    oced = OCEDStore()
    events = pd.DataFrame({"eid": ["e1"], "ts": pd.date_range("2024-01-01", periods=1)})
    oced.add_event("ev", "eid", "ts", events)
    oced.add_object("A", "oid", pd.DataFrame({"oid": ["a1", "a2"]}))
    oced.add_object("B", "oid", pd.DataFrame({"oid": ["b1"]}))
    bad_o2o = pd.DataFrame({"from_id": ["a99"], "to_id": ["b99"]})
    oced.add_object_to_object_relation("rel", "from_id", ["A"], "to_id", ["B"], bad_o2o)
    violations = check_oced_integrity(oced)
    assert len(violations) == 2


def test_clean_oced_normalizes_ids_symmetrically():
    oced = OCEDStore()
    events = pd.DataFrame(
        {"eid": ["ev-001", "ev-002"], "ts": pd.date_range("2024-01-01", periods=2)}
    )
    objects = pd.DataFrame({"oid": ["obj/A", "obj/B"]})
    relations = pd.DataFrame({"eid": ["ev-001", "ev-002"], "oid": ["obj/A", "obj/B"]})
    oced.add_event("act-type", "eid", "ts", events)
    oced.add_object("obj-type", "oid", objects)
    oced.add_event_to_object_relation("rel-name", "eid", "oid", ["obj-type"], relations)

    oced = clean_oced_data(oced)
    violations = check_oced_integrity(oced)
    assert violations == [], f"Integrity broken after cleaning: {violations}"


def test_all_names_clean_after_normalization_synthetic():
    oced = OCEDStore()
    events = pd.DataFrame(
        {"eid": ["ev-1"], "ts": pd.date_range("2024-01-01", periods=1)}
    )
    objects = pd.DataFrame({"oid": ["o-1"]})
    oced.add_event("act/type", "eid", "ts", events)
    oced.add_object("obj type", "oid", objects)
    oced = clean_oced_data(oced)
    for name in (
        oced.get_event_types() + oced.get_object_types() + oced.get_relation_types()
    ):
        assert CLEAN_RE.match(name), f"Unclean name after normalization: {name!r}"


def test_get_log_drops_no_rows_on_clean_single_type_store(caplog):
    oced = _consistent_oced()
    with caplog.at_level(logging.WARNING):
        get_log(oced, "thing")
    drop_msgs = [r for r in caplog.records if "dropped" in r.message]
    assert drop_msgs == [], "\n".join(r.message for r in drop_msgs)


def test_get_log_reports_the_size_of_the_projection(caplog):
    oced = _consistent_oced()

    with caplog.at_level(logging.INFO, logger="semanticlog.data.flatten"):
        get_log(oced, "thing")

    assert "Extracted flattened log (thing) with 3 events, 2 objects" in caplog.text


def test_get_log_holds_an_event_once_however_many_relations_reach_the_case():
    oced = _consistent_oced()
    oced.add_event_to_object_relation(
        "creates",
        "eid",
        "oid",
        ["thing"],
        pd.DataFrame({"eid": ["e1", "e2", "e3"], "oid": ["o1", "o1", "o2"]}),
    )

    log = get_log(oced, "thing")

    assert len(log) == 3
    assert log.index.is_unique


def test_get_log_leaves_out_an_event_with_no_timestamp(caplog):
    oced = OCEDStore()
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": ["e1", "e2", "e3"],
                "ts": ["2024-01-01T09:00:00Z", "2024-01-02T09:00:00Z", None],
            }
        ),
    )
    oced.add_object("thing", "oid", pd.DataFrame({"oid": ["o1"]}))
    oced.add_event_to_object_relation(
        "modifies",
        "eid",
        "oid",
        ["thing"],
        pd.DataFrame({"eid": ["e1", "e2", "e3"], "oid": ["o1", "o1", "o1"]}),
    )

    with caplog.at_level(logging.WARNING, logger="semanticlog.data.flatten"):
        log = get_log(oced, "thing")

    assert list(log.index) == ["e1", "e2"]
    assert "1 event(s) have no timestamp" in caplog.text


def test_get_log_says_nothing_about_ids_of_another_object_type(caplog):
    """Projecting onto one case notion excludes the other types by definition."""
    oced = _consistent_oced()
    oced.add_object("other", "oid", pd.DataFrame({"oid": ["x1"]}))
    oced.add_event_to_object_relation(
        "touches",
        "eid",
        "oid",
        ["thing", "other"],
        pd.DataFrame({"eid": ["e1"], "oid": ["x1"]}),
    )

    with caplog.at_level(logging.INFO, logger="semanticlog.data.flatten"):
        get_log(oced, "thing")

    assert "another object type" not in caplog.text


def _mixed_dtypes() -> OCEDStore:
    oced = OCEDStore()
    oced.add_event(
        "place order",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": ["e1", "e2", "e3"],
                "ts": ["2024-01-01T09:00:00Z", "2024-01-02T09:00:00Z", None],
                "clerk": ["ann", "bo", "ann"],
                "amount": ["10.5", "20", "30.25"],
                "urgent": ["true", "false", "true"],
                "count": ["1", "2", "3"],
            }
        ),
    )
    oced.add_object(
        "order",
        "oid",
        pd.DataFrame(
            {"oid": ["o1", "o2"], "channel": ["web", None], "weight": ["1.5", "2.5"]}
        ),
    )
    oced.add_event_to_object_relation(
        "concerns",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1", "e2", "e3"], "oid": ["o1", "o1", "o2"]}),
    )
    return oced


def _awkward_names() -> OCEDStore:
    oced = OCEDStore()
    oced.add_event(
        "Place Order (retail)",
        "event id",
        "time:timestamp",
        pd.DataFrame(
            {
                "event id": ["ev-1", "ev-2"],
                "time:timestamp": pd.date_range("2024-01-01", periods=2, tz="UTC"),
                "unit price": ["1.0", "2.0"],
            }
        ),
    )
    oced.add_object(
        "order/type",
        "object-id",
        pd.DataFrame({"object-id": ["obj A", "obj/B"], "Käufer": ["ann", "bo"]}),
    )
    oced.add_event_to_object_relation(
        "relates to",
        "event id",
        "object-id",
        ["order/type"],
        pd.DataFrame({"event id": ["ev-1", "ev-2"], "object-id": ["obj A", "obj/B"]}),
    )
    return oced


def _multi_type_relations() -> OCEDStore:
    oced = OCEDStore()
    oced.add_event(
        "ship",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": ["e1", "e2", "e3"],
                "ts": pd.date_range("2024-01-01", periods=3, tz="UTC"),
            }
        ),
    )
    oced.add_object("order", "oid", pd.DataFrame({"oid": ["o1", "o2"]}))
    oced.add_object("item", "oid", pd.DataFrame({"oid": ["i1"]}))
    oced.add_event_to_object_relation(
        "touches",
        "eid",
        "oid",
        ["order", "item"],
        pd.DataFrame({"eid": ["e1", "e2", "e3"], "oid": ["o1", "o2", "i1"]}),
    )
    oced.add_object_to_object_relation(
        "contains",
        "from_id",
        ["order"],
        "to_id",
        ["item"],
        pd.DataFrame({"from_id": ["o1"], "to_id": ["i1"]}),
    )
    return oced


def _sparse() -> OCEDStore:
    oced = OCEDStore()
    oced.add_event(
        "act",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": ["e1", "e2"],
                "ts": pd.date_range("2024-01-01", periods=2, tz="UTC"),
                "always_null": [None, None],
                "sometimes": ["x", None],
            }
        ),
    )
    oced.add_object(
        "thing",
        "oid",
        pd.DataFrame({"oid": ["t1", "t2"], "blank": [None, None]}),
    )
    oced.add_event_to_object_relation(
        "on",
        "eid",
        "oid",
        ["thing"],
        pd.DataFrame({"eid": ["e1", "e2"], "oid": ["t1", "t2"]}),
    )
    return oced


AWKWARD_STORES = {
    "mixed_dtypes": _mixed_dtypes,
    "awkward_names": _awkward_names,
    "multi_type_relations": _multi_type_relations,
    "sparse": _sparse,
}


@pytest.mark.parametrize("shape", sorted(AWKWARD_STORES))
def test_no_integrity_violations_after_clean(shape):
    oced = clean_oced_data(AWKWARD_STORES[shape]())
    violations = check_oced_integrity(oced)
    assert violations == [], "\n".join(violations)


@pytest.mark.parametrize("shape", sorted(AWKWARD_STORES))
def test_all_names_clean_after_normalization(shape):
    oced = clean_oced_data(AWKWARD_STORES[shape]())
    for name in (
        oced.get_event_types() + oced.get_object_types() + oced.get_relation_types()
    ):
        assert CLEAN_RE.match(name), f"Unclean name in '{shape}': {name!r}"


@pytest.mark.parametrize("shape", sorted(AWKWARD_STORES))
def test_get_log_emits_no_drop_warning(shape, caplog):
    oced = clean_oced_data(AWKWARD_STORES[shape]())
    case_concept = oced.get_most_common_objects(n=1)[0]
    with caplog.at_level(logging.WARNING):
        get_log(oced, case_concept)
    drop_msgs = [
        r for r in caplog.records if "not found in any object store" in r.message
    ]
    assert drop_msgs == [], "\n".join(r.message for r in drop_msgs)
