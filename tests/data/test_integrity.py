"""
Tests for OCEDStore referential integrity and name normalization.

Invariants verified:
  - Every object ID referenced in an e2o relation exists in the object store
  - Every event ID referenced in an e2o relation exists in the event store
  - Every from/to ID in an o2o relation exists in the corresponding object store
  - After clean_oced_data(), all type/relation names match [A-Za-z0-9_]+
  - get_log() drops zero rows on a clean store
"""

import logging
import os
import re

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.data.pm4pyloader import PM4PYLoader

CLEAN_RE = re.compile(r"^[A-Za-z0-9_]+$")

OCEL_DIR = "data/ocel"
_EXCLUDE = {"AOE2.sqlite", "AOE2_small.json", "blockchain.xmlocel"}
_SIZE_LIMIT = 100 * 1024 * 1024  # 100 MB

OCEL_FILES = [
    f
    for f in os.listdir(OCEL_DIR)
    if f not in _EXCLUDE and os.path.getsize(os.path.join(OCEL_DIR, f)) < _SIZE_LIMIT
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def check_oced_integrity(oced: OCEDStore) -> list[str]:
    """
    Check referential integrity of an OCEDStore.
    Returns a list of human-readable violation descriptions; empty list = clean.
    """
    violations = []

    all_event_ids: set = set()
    for entry in oced._events.values():
        all_event_ids.update(entry["data"][entry["id"]].unique())

    object_ids_by_type: dict[str, set] = {}
    for obj_type, entry in oced._objects.items():
        object_ids_by_type[obj_type] = set(entry["data"][entry["id"]].unique())

    for rel_name, rel in oced._relations["event_to_object"].items():
        df = rel["data"]
        from_col = rel["from_id"]
        to_col = rel["to_id"]

        missing_events = set(df[from_col].unique()) - all_event_ids
        if missing_events:
            sample = sorted(str(x) for x in missing_events)[:5]
            violations.append(
                f"E2O '{rel_name}': {len(missing_events)} event IDs not in event store "
                f"(sample: {sample})"
            )

        # Each to_id must exist in at least one of the referenced types (union check)
        all_known: set = set()
        for to_type in rel["to_type"]:
            all_known |= object_ids_by_type.get(to_type, set())
        missing = set(df[to_col].unique()) - all_known
        if missing:
            sample = sorted(str(x) for x in missing)[:5]
            violations.append(
                f"E2O '{rel_name}' → {rel['to_type']}: {len(missing)} object IDs not in "
                f"any referenced object type (sample: {sample})"
            )

    for rel_name, rel in oced._relations["object_to_object"].items():
        df = rel["data"]
        for id_col, types, label in [
            (rel["from_id"], rel["from_type"], "from"),
            (rel["to_id"], rel["to_type"], "to"),
        ]:
            all_known = set()
            for obj_type in types:
                all_known |= object_ids_by_type.get(obj_type, set())
            missing = set(df[id_col].unique()) - all_known
            if missing:
                sample = sorted(str(x) for x in missing)[:5]
                violations.append(
                    f"O2O '{rel_name}' {label} → {types}: {len(missing)} IDs "
                    f"not in any referenced object type (sample: {sample})"
                )

    return violations


# ---------------------------------------------------------------------------
# Synthetic unit tests
# ---------------------------------------------------------------------------


def _consistent_oced() -> OCEDStore:
    """Build a minimal OCEDStore with no integrity violations."""
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
    # Add a relation row pointing to an object ID that doesn't exist
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


def test_detects_missing_id_in_o2o():
    oced = OCEDStore()
    events = pd.DataFrame({"eid": ["e1"], "ts": pd.date_range("2024-01-01", periods=1)})
    oced.add_event("ev", "eid", "ts", events)
    oced.add_object("A", "oid", pd.DataFrame({"oid": ["a1", "a2"]}))
    oced.add_object("B", "oid", pd.DataFrame({"oid": ["b1"]}))
    # from_id "a99" not in A, to_id "b99" not in B
    bad_o2o = pd.DataFrame({"from_id": ["a99"], "to_id": ["b99"]})
    oced.add_object_to_object_relation("rel", "from_id", ["A"], "to_id", ["B"], bad_o2o)
    violations = check_oced_integrity(oced)
    assert len(violations) == 2  # one for from side, one for to side


def test_clean_oced_normalizes_ids_symmetrically():
    """After clean_oced_data(), integrity must hold even if IDs had special chars."""
    oced = OCEDStore()
    events = pd.DataFrame(
        {"eid": ["ev-001", "ev-002"], "ts": pd.date_range("2024-01-01", periods=2)}
    )
    objects = pd.DataFrame({"oid": ["obj/A", "obj/B"]})
    # Relations use the same raw IDs — cleaning must be symmetric
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
        list(oced._events)
        + list(oced._objects)
        + list(oced._relations["event_to_object"])
        + list(oced._relations["object_to_object"])
    ):
        assert CLEAN_RE.match(name), f"Unclean name after normalization: {name!r}"


# ---------------------------------------------------------------------------
# Real OCEL integration tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("file_name", OCEL_FILES)
def test_no_integrity_violations_after_load_and_clean(file_name):
    oced = PM4PYLoader(os.path.join(OCEL_DIR, file_name), {}).get_oced()
    oced = clean_oced_data(oced)
    violations = check_oced_integrity(oced)
    assert violations == [], "\n".join(violations)


def test_get_log_drops_no_rows_on_clean_single_type_store(caplog):
    """get_log must not drop rows when all e2o relations are single-type and consistent."""
    oced = _consistent_oced()
    with caplog.at_level(logging.WARNING):
        oced.get_log("thing")
    drop_msgs = [r for r in caplog.records if "dropped" in r.message]
    assert drop_msgs == [], "\n".join(r.message for r in drop_msgs)


@pytest.mark.parametrize("file_name", OCEL_FILES)
def test_get_log_emits_no_warning(file_name, caplog):
    oced = PM4PYLoader(os.path.join(OCEL_DIR, file_name), {}).get_oced()
    oced = clean_oced_data(oced)
    case_concept = oced.get_most_common_objects(n=1)[0]
    with caplog.at_level(logging.WARNING):
        oced.get_log(case_concept)
    drop_msgs = [
        r for r in caplog.records if "not found in any object store" in r.message
    ]
    assert drop_msgs == [], "\n".join(r.message for r in drop_msgs)


@pytest.mark.parametrize("file_name", OCEL_FILES)
def test_all_names_clean_after_normalization(file_name):
    oced = PM4PYLoader(os.path.join(OCEL_DIR, file_name), {}).get_oced()
    oced = clean_oced_data(oced)
    for name in (
        list(oced._events)
        + list(oced._objects)
        + list(oced._relations["event_to_object"])
        + list(oced._relations["object_to_object"])
    ):
        assert CLEAN_RE.match(name), f"Unclean name in '{file_name}': {name!r}"
