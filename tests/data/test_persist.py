"""An OCEDStore survives a write/read round trip, dtypes and all.

The format exists so a stored log outlives the code that wrote it, which means
the things worth asserting are that nothing is lost, that dtypes come back
exactly, and that a store from a newer semlog is refused rather than
misinterpreted.
"""

import json

import pandas as pd
import pytest

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import EntityKind, OCEDStore, RelationKind
from semanticlog.data.persist import MANIFEST_NAME


@pytest.fixture
def store() -> OCEDStore:
    data = OCEDStore(ocel_mode=True)
    data.add_event(
        "place order",
        "eid",
        "t",
        pd.DataFrame(
            {
                "eid": ["e1", "e2"],
                "t": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
                "channel": ["web", "shop"],
                "amount": [10.5, 20.0],
                "urgent": [True, False],
                "count": [1, 7],
            }
        ),
    )
    data.add_object(
        "order", "oid", pd.DataFrame({"oid": ["o1", "o2"], "priority": ["hi", "lo"]})
    )
    data.add_object("item", "iid", pd.DataFrame({"iid": ["i1"]}))
    data.add_event_to_object_relation(
        "concerns",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1", "e2"], "oid": ["o1", "o2"]}),
    )
    data.add_object_to_object_relation(
        "contains",
        "oid",
        ["order"],
        "iid",
        ["item"],
        pd.DataFrame({"oid": ["o1"], "iid": ["i1"]}),
    )
    data.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1"],
                "ocel:type": ["order"],
                "ocel:timestamp": pd.to_datetime(["2024-01-03"], utc=True),
                "ocel:field": ["priority"],
                "priority": ["lo"],
            }
        ),
    )
    return clean_oced_data(data)


def test_round_trip_preserves_the_whole_store(store, tmp_path):
    store.write(tmp_path / "s.oced")
    loaded = OCEDStore.read(tmp_path / "s.oced")

    assert loaded.get_activities() == store.get_activities()
    assert loaded.get_object_types() == store.get_object_types()
    assert loaded.get_relation_types(RelationKind.ALL) == store.get_relation_types(
        RelationKind.ALL
    )
    assert loaded.backmap == store.backmap
    assert loaded.has_ocel_mode() is True
    assert loaded.get_object_change_types() == store.get_object_change_types()


def test_round_trip_preserves_dtypes(store, tmp_path):
    """The reason for Parquet over CSV: no schema has to be reconstructed."""
    store.write(tmp_path / "s.oced")
    loaded = OCEDStore.read(tmp_path / "s.oced")

    before = store.get_events("place_order").data.dtypes
    after = loaded.get_events("place_order").data.dtypes
    pd.testing.assert_series_equal(before, after)


def test_round_trip_preserves_table_contents(store, tmp_path):
    store.write(tmp_path / "s.oced")
    loaded = OCEDStore.read(tmp_path / "s.oced")

    pd.testing.assert_frame_equal(
        store.get_events("place_order").data, loaded.get_events("place_order").data
    )
    pd.testing.assert_frame_equal(
        store.get_relation("concerns").data, loaded.get_relation("concerns").data
    )


def test_relation_typing_survives(store, tmp_path):
    store.write(tmp_path / "s.oced")
    loaded = OCEDStore.read(tmp_path / "s.oced")

    e2o = loaded.get_relation("concerns")
    o2o = loaded.get_relation("contains")
    assert e2o.to_type == ["order"]
    assert e2o.from_type is None
    assert o2o.from_type == ["order"]


def test_identifiers_are_usable_after_reading(store, tmp_path):
    """update_used_ids runs on read, so the store is ready to be added to."""
    store.write(tmp_path / "s.oced")
    loaded = OCEDStore.read(tmp_path / "s.oced")

    assert set(loaded.get_ids(EntityKind.OBJECT, "order")) == {"o1", "o2"}
    assert loaded.ids_not_used(["brand_new"]) is True
    assert loaded.ids_not_used(["o1"]) is False


def test_writing_twice_replaces_rather_than_accumulates(store, tmp_path):
    path = tmp_path / "s.oced"
    store.write(path)
    store.write(path)

    assert len(list((path / "events").glob("*.parquet"))) == 1


def test_a_directory_without_a_manifest_is_refused(tmp_path):
    (tmp_path / "empty").mkdir()

    with pytest.raises(ValueError, match="not a semlog store"):
        OCEDStore.read(tmp_path / "empty")


def test_a_newer_format_version_is_refused(store, tmp_path):
    path = tmp_path / "s.oced"
    store.write(path)
    manifest = json.loads((path / MANIFEST_NAME).read_text())
    manifest["version"] = 99
    (path / MANIFEST_NAME).write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Upgrade semlog"):
        OCEDStore.read(path)


def test_table_names_that_are_not_valid_filenames_still_round_trip(tmp_path):
    """Table names come from source data, so they cannot be used as filenames."""
    data = OCEDStore()
    data.add_event(
        "pay/invoice: step 1",
        "eid",
        "t",
        pd.DataFrame({"eid": ["e1"], "t": pd.to_datetime(["2024-01-01"], utc=True)}),
    )
    data.add_object("a/b", "oid", pd.DataFrame({"oid": ["o1"]}))

    data.write(tmp_path / "s.oced")
    loaded = OCEDStore.read(tmp_path / "s.oced")

    assert loaded.get_activities() == ["pay/invoice: step 1"]
    assert loaded.get_object_types() == ["a/b"]


def test_pickle_save_still_loads_but_warns(store, tmp_path):
    """Stores written by earlier versions must keep opening for one release."""
    path = tmp_path / "legacy.pkl"

    with pytest.warns(DeprecationWarning, match="OCEDStore.write"):
        store.save(path)

    loaded = OCEDStore.load(path)
    assert loaded.get_activities() == store.get_activities()
