"""
Tests for PM4PYLoader.

Synthetic OCEL test data is written via pm4py.write_ocel2_json (JSON round-trip
is reliable across pm4py versions). SQLite round-trip via write_ocel2_sqlite has
a known pm4py 2.7 bug and is avoided for fixture construction.
"""

import pandas as pd
import pm4py
import pytest
from pm4py.objects.ocel.obj import OCEL

from semanticlog.data.loading.pm4pyloader import PM4PYLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _empty_o2o() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ocel:oid": pd.Series([], dtype=str),
            "ocel:oid_2": pd.Series([], dtype=str),
            "ocel:qualifier": pd.Series([], dtype=str),
        }
    )


def _write_ocel(ocel: OCEL, path: str) -> None:
    pm4py.write_ocel2_json(ocel, path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_ocel(tmp_path):
    """Minimal OCEL2 JSON with two activities, two object types, e2o relations."""
    events = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e2", "e3"],
            "ocel:activity": ["place_order", "place_order", "ship_order"],
            "ocel:timestamp": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03"], utc=True
            ),
        }
    )
    objects = pd.DataFrame(
        {
            "ocel:oid": ["o1", "o2", "i1"],
            "ocel:type": ["order", "order", "item"],
        }
    )
    relations = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e2", "e3"],
            "ocel:activity": ["place_order", "place_order", "ship_order"],
            "ocel:timestamp": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03"], utc=True
            ),
            "ocel:oid": ["o1", "o2", "i1"],
            "ocel:type": ["order", "order", "item"],
            "ocel:qualifier": ["modifies", "modifies", "modifies"],
        }
    )
    ocel = OCEL(events=events, objects=objects, relations=relations, o2o=_empty_o2o())
    path = str(tmp_path / "simple.jsonocel")
    _write_ocel(ocel, path)
    return path


@pytest.fixture
def o2o_ocel(tmp_path):
    """OCEL2 JSON that includes an o2o relation between two object types."""
    events = pd.DataFrame(
        {
            "ocel:eid": ["e1"],
            "ocel:activity": ["act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
        }
    )
    objects = pd.DataFrame(
        {
            "ocel:oid": ["o1", "i1"],
            "ocel:type": ["order", "item"],
        }
    )
    relations = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e1"],
            "ocel:activity": ["act", "act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-01"], utc=True),
            "ocel:oid": ["o1", "i1"],
            "ocel:type": ["order", "item"],
            "ocel:qualifier": ["modifies", "modifies"],
        }
    )
    o2o = pd.DataFrame(
        {
            "ocel:oid": ["o1"],
            "ocel:oid_2": ["i1"],
            "ocel:qualifier": ["contains"],
        }
    )
    ocel = OCEL(events=events, objects=objects, relations=relations, o2o=o2o)
    path = str(tmp_path / "o2o.jsonocel")
    _write_ocel(ocel, path)
    return path


@pytest.fixture
def numeric_ids_ocel(tmp_path):
    """OCEL2 JSON where all object and event IDs are purely numeric strings."""
    events = pd.DataFrame(
        {
            "ocel:eid": ["1", "2"],
            "ocel:activity": ["act", "act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
        }
    )
    objects = pd.DataFrame(
        {
            "ocel:oid": ["10", "20"],
            "ocel:type": ["order", "order"],
        }
    )
    relations = pd.DataFrame(
        {
            "ocel:eid": ["1", "2"],
            "ocel:activity": ["act", "act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
            "ocel:oid": ["10", "20"],
            "ocel:type": ["order", "order"],
            "ocel:qualifier": ["modifies", "modifies"],
        }
    )
    ocel = OCEL(events=events, objects=objects, relations=relations, o2o=_empty_o2o())
    path = str(tmp_path / "numeric.jsonocel")
    _write_ocel(ocel, path)
    return path


@pytest.fixture
def null_qualifier_ocel(tmp_path):
    """OCEL2 JSON where e2o qualifiers are null/empty/placeholder strings."""
    events = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e2", "e3", "e4"],
            "ocel:activity": ["act", "act", "act", "act"],
            "ocel:timestamp": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"], utc=True
            ),
        }
    )
    objects = pd.DataFrame(
        {
            "ocel:oid": ["o1", "o2", "o3", "o4"],
            "ocel:type": ["order", "order", "order", "order"],
        }
    )
    # Use placeholder strings that PM4PYLoader normalises to "modifies"
    relations = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e2", "e3", "e4"],
            "ocel:activity": ["act", "act", "act", "act"],
            "ocel:timestamp": pd.to_datetime(
                ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"], utc=True
            ),
            "ocel:oid": ["o1", "o2", "o3", "o4"],
            "ocel:type": ["order", "order", "order", "order"],
            "ocel:qualifier": ["None", "none", "NA", "N/A"],
        }
    )
    ocel = OCEL(events=events, objects=objects, relations=relations, o2o=_empty_o2o())
    path = str(tmp_path / "null_qualifier.jsonocel")
    _write_ocel(ocel, path)
    return path


@pytest.fixture
def explicit_qualifier_ocel(tmp_path):
    """OCEL2 JSON with two distinct e2o qualifiers."""
    events = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e2"],
            "ocel:activity": ["act", "act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
        }
    )
    objects = pd.DataFrame(
        {
            "ocel:oid": ["o1", "i1"],
            "ocel:type": ["order", "item"],
        }
    )
    relations = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e2"],
            "ocel:activity": ["act", "act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
            "ocel:oid": ["o1", "i1"],
            "ocel:type": ["order", "item"],
            "ocel:qualifier": ["creates", "updates"],
        }
    )
    ocel = OCEL(events=events, objects=objects, relations=relations, o2o=_empty_o2o())
    path = str(tmp_path / "qualifier.jsonocel")
    _write_ocel(ocel, path)
    return path


# ---------------------------------------------------------------------------
# Tests — basic loading
# ---------------------------------------------------------------------------


def test_returns_oced_store(simple_ocel):
    loader = PM4PYLoader(simple_ocel, {})
    oced = loader.get_oced()
    assert oced is not None


def test_loads_correct_object_types(simple_ocel):
    oced = PM4PYLoader(simple_ocel, {}).get_oced()
    assert set(oced.get_object_types()) == {"order", "item"}


def test_loads_correct_activities(simple_ocel):
    oced = PM4PYLoader(simple_ocel, {}).get_oced()
    assert set(oced.get_activities()) == {"place_order", "ship_order"}


def test_loads_correct_object_ids(simple_ocel):
    oced = PM4PYLoader(simple_ocel, {}).get_oced()
    assert set(oced.get_ids("object", "order").tolist()) == {"o1", "o2"}
    assert set(oced.get_ids("object", "item").tolist()) == {"i1"}


def test_loads_correct_event_ids(simple_ocel):
    oced = PM4PYLoader(simple_ocel, {}).get_oced()
    order_events = set(oced.get_ids("event", "place_order").tolist())
    ship_events = set(oced.get_ids("event", "ship_order").tolist())
    assert order_events == {"e1", "e2"}
    assert ship_events == {"e3"}


# ---------------------------------------------------------------------------
# Tests — file format detection
# ---------------------------------------------------------------------------


def test_format_auto_detected_from_sqlite_extension(tmp_path):
    """No explicit file_format in config — PM4PYLoader infers from .sqlite extension."""
    dummy = str(tmp_path / "log.sqlite")
    open(dummy, "w").close()
    loader = PM4PYLoader(dummy, {})
    assert loader.config["file_format"] == "sqlite"


def test_format_auto_detected_from_jsonocel_extension(simple_ocel):
    """No explicit file_format in config — PM4PYLoader infers from .jsonocel extension."""
    loader = PM4PYLoader(simple_ocel, {})
    assert loader.config["file_format"] == "json"


def test_explicit_format_config_accepted(simple_ocel):
    loader = PM4PYLoader(simple_ocel, {"file_format": "json"})
    oced = loader.get_oced()
    assert oced is not None


def test_unknown_extension_raises(tmp_path):
    dummy = str(tmp_path / "log.csv")
    with open(dummy, "w") as f:
        f.write("dummy")
    with pytest.raises(Exception):
        PM4PYLoader(dummy, {})


# ---------------------------------------------------------------------------
# Tests — e2o relations
# ---------------------------------------------------------------------------


def test_e2o_default_modifies_relation_present(simple_ocel):
    oced = PM4PYLoader(simple_ocel, {}).get_oced()
    assert "modifies" in oced.get_relation_types()


def test_explicit_e2o_qualifier_stored_as_relation_name(explicit_qualifier_ocel):
    oced = PM4PYLoader(explicit_qualifier_ocel, {}).get_oced()
    relation_types = set(oced.get_relation_types())
    assert "creates" in relation_types
    assert "updates" in relation_types


def test_null_placeholder_qualifiers_normalised_to_modifies(null_qualifier_ocel):
    """None/none/NA/N/A qualifiers should all become the 'modifies' relation."""
    oced = PM4PYLoader(null_qualifier_ocel, {}).get_oced()
    assert "modifies" in oced.get_relation_types()
    # Verify no placeholder strings leaked through as relation names
    for placeholder in ("None", "none", "NA", "N/A"):
        assert placeholder not in oced.get_relation_types()


# ---------------------------------------------------------------------------
# Tests — o2o relations
# ---------------------------------------------------------------------------


def test_o2o_relation_stored_with_o2o_suffix(o2o_ocel):
    oced = PM4PYLoader(o2o_ocel, {}).get_oced()
    assert "contains_o2o" in oced.get_relation_types()


def test_o2o_original_qualifier_not_stored_without_suffix(o2o_ocel):
    oced = PM4PYLoader(o2o_ocel, {}).get_oced()
    assert "contains" not in oced.get_relation_types()


def test_o2o_null_values_raises(simple_ocel, monkeypatch):
    """get_oced() must raise ValueError when the o2o table returned by pm4py contains nulls."""
    import semanticlog.data.loading.pm4pyloader as loader_module

    # Construct an OCEL whose o2o table has a null, then mock the pm4py reader to return it
    events = pd.DataFrame(
        {
            "ocel:eid": ["e1"],
            "ocel:activity": ["act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
        }
    )
    objects = pd.DataFrame({"ocel:oid": ["o1", "o2"], "ocel:type": ["order", "order"]})
    relations = pd.DataFrame(
        {
            "ocel:eid": ["e1"],
            "ocel:activity": ["act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
            "ocel:oid": ["o1"],
            "ocel:type": ["order"],
            "ocel:qualifier": ["modifies"],
        }
    )
    o2o_with_null = pd.DataFrame(
        {"ocel:oid": ["o1"], "ocel:oid_2": [None], "ocel:qualifier": ["rel"]}
    )
    corrupted = OCEL(
        events=events, objects=objects, relations=relations, o2o=o2o_with_null
    )
    monkeypatch.setattr(loader_module.pm, "read_ocel2_json", lambda _path: corrupted)

    with pytest.raises(ValueError, match="Could not resolve object types"):
        PM4PYLoader(simple_ocel, {}).get_oced()


# ---------------------------------------------------------------------------
# Tests — numeric ID prefixing
# ---------------------------------------------------------------------------


def test_numeric_object_ids_prefixed_with_type(numeric_ids_ocel):
    """Object IDs that are purely numeric strings get '{type}_{id}' prefix."""
    oced = PM4PYLoader(numeric_ids_ocel, {}).get_oced()
    ids = set(oced.get_ids("object", "order").tolist())
    assert ids == {"order_10", "order_20"}


def test_numeric_event_ids_prefixed_with_activity(numeric_ids_ocel):
    """Event IDs that are purely numeric strings get '{activity}_{id}' prefix."""
    oced = PM4PYLoader(numeric_ids_ocel, {}).get_oced()
    ids = set(oced.get_ids("event", "act").tolist())
    assert ids == {"act_1", "act_2"}


def test_numeric_id_prefix_propagates_to_e2o_relation(numeric_ids_ocel):
    """After prefixing, relation tables must reference the new IDs, not the originals."""
    oced = PM4PYLoader(numeric_ids_ocel, {}).get_oced()
    rel = oced.get_relations("e2o")["modifies"]
    to_ids = set(rel.data[rel.to_id].tolist())
    assert to_ids == {"order_10", "order_20"}


def test_string_ids_not_prefixed(simple_ocel):
    """Non-numeric string IDs must not be modified."""
    oced = PM4PYLoader(simple_ocel, {}).get_oced()
    ids = set(oced.get_ids("object", "order").tolist())
    assert ids == {"o1", "o2"}


# ---------------------------------------------------------------------------
# Tests — JSON format (pm4py>=2.7 has read_ocel2_json)
# ---------------------------------------------------------------------------


def test_json_format_loads_successfully(tmp_path):
    """JSON OCEL2 files (.jsonocel) load correctly with pm4py>=2.7."""
    events = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e2"],
            "ocel:activity": ["act", "act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
        }
    )
    objects = pd.DataFrame({"ocel:oid": ["o1", "o2"], "ocel:type": ["order", "order"]})
    relations = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e2"],
            "ocel:activity": ["act", "act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01", "2022-01-02"], utc=True),
            "ocel:oid": ["o1", "o2"],
            "ocel:type": ["order", "order"],
            "ocel:qualifier": ["modifies", "modifies"],
        }
    )
    ocel = OCEL(events=events, objects=objects, relations=relations, o2o=_empty_o2o())
    path = str(tmp_path / "log.jsonocel")
    pm4py.write_ocel2_json(ocel, path)

    result = PM4PYLoader(path, {}).get_oced()
    assert set(result.get_object_types()) == {"order"}
    assert set(result.get_activities()) == {"act"}


# ---------------------------------------------------------------------------
# Tests — ATTRIBUTE_HISTORY and RELATION_TIMESTAMPS extensions
# ---------------------------------------------------------------------------


@pytest.fixture
def object_changes_ocel(tmp_path, monkeypatch):
    """OCEL whose objects have attribute change history (object_changes).

    Uses monkeypatching to bypass the pm4py writer (which has a bug when objects
    carry object_changes but no static attributes) and inject the OCEL directly.
    """
    import semanticlog.data.loading.pm4pyloader as loader_module

    events = pd.DataFrame(
        {
            "ocel:eid": ["e1"],
            "ocel:activity": ["place_order"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
        }
    )
    objects = pd.DataFrame({"ocel:oid": ["o1", "o2"], "ocel:type": ["order", "order"]})
    relations = pd.DataFrame(
        {
            "ocel:eid": ["e1"],
            "ocel:activity": ["place_order"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
            "ocel:oid": ["o1"],
            "ocel:type": ["order"],
            "ocel:qualifier": ["modifies"],
        }
    )
    object_changes = pd.DataFrame(
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
    ocel = OCEL(
        events=events,
        objects=objects,
        relations=relations,
        o2o=_empty_o2o(),
        object_changes=object_changes,
    )
    dummy_path = str(tmp_path / "changes.jsonocel")
    open(dummy_path, "w").close()
    monkeypatch.setattr(loader_module.pm, "read_ocel2_json", lambda _: ocel)
    return dummy_path


def test_no_extension_ocel_mode_is_false(simple_ocel):
    oced = PM4PYLoader(simple_ocel, {}).get_oced()
    assert oced.has_ocel_mode() is False


def test_attribute_history_sets_ocel_mode(object_changes_ocel):
    from semanticlog.onto.settings import OcelExtension

    oced = PM4PYLoader(
        object_changes_ocel, {}, ocel_extensions=[OcelExtension.ATTRIBUTE_HISTORY]
    ).get_oced()
    assert oced.has_ocel_mode() is True


def test_attribute_history_populates_object_changes(object_changes_ocel):
    from semanticlog.onto.settings import OcelExtension

    oced = PM4PYLoader(
        object_changes_ocel, {}, ocel_extensions=[OcelExtension.ATTRIBUTE_HISTORY]
    ).get_oced()
    assert "order" in oced.get_object_change_types()
    changes = oced.get_object_changes("order")
    assert len(changes) == 3


def test_attribute_history_not_loaded_without_extension(object_changes_ocel):
    oced = PM4PYLoader(object_changes_ocel, {}).get_oced()
    assert oced.has_ocel_mode() is False


def test_attribute_history_numeric_ids_prefixed_in_changes(tmp_path, monkeypatch):
    """object_changes ocel:oid values are updated when objects have numeric IDs."""
    import semanticlog.data.loading.pm4pyloader as loader_module
    from semanticlog.onto.settings import OcelExtension

    events = pd.DataFrame(
        {
            "ocel:eid": ["e1"],
            "ocel:activity": ["act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
        }
    )
    objects = pd.DataFrame({"ocel:oid": ["10"], "ocel:type": ["order"]})
    relations = pd.DataFrame(
        {
            "ocel:eid": ["e1"],
            "ocel:activity": ["act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
            "ocel:oid": ["10"],
            "ocel:type": ["order"],
            "ocel:qualifier": ["modifies"],
        }
    )
    object_changes = pd.DataFrame(
        {
            "ocel:oid": ["10"],
            "ocel:type": ["order"],
            "ocel:timestamp": pd.to_datetime(["2022-01-02"], utc=True),
            "ocel:field": ["status"],
            "status": ["shipped"],
        }
    )
    ocel = OCEL(
        events=events,
        objects=objects,
        relations=relations,
        o2o=_empty_o2o(),
        object_changes=object_changes,
    )
    dummy_path = str(tmp_path / "numeric_changes.jsonocel")
    open(dummy_path, "w").close()
    monkeypatch.setattr(loader_module.pm, "read_ocel2_json", lambda _: ocel)

    oced = PM4PYLoader(
        dummy_path, {}, ocel_extensions=[OcelExtension.ATTRIBUTE_HISTORY]
    ).get_oced()
    changes = oced.get_object_changes("order")
    assert set(changes["ocel:oid"].tolist()) == {"order_10"}
