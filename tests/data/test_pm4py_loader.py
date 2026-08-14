"""Reading OCEL 2.0 logs, and the repairs made on the way in."""

import pandas as pd
import pm4py
import pytest
from pm4py.objects.ocel.obj import OCEL

from semanticlog.data.loading.pm4py_loader import PM4PyLoader, _relation_names
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind


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


@pytest.fixture
def simple_ocel(tmp_path):
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
def null_o2o_qualifier_ocel(tmp_path):
    events = pd.DataFrame(
        {
            "ocel:eid": ["e1"],
            "ocel:activity": ["act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"], utc=True),
        }
    )
    objects = pd.DataFrame(
        {
            "ocel:oid": ["o1", "o2", "i1", "i2"],
            "ocel:type": ["order", "order", "item", "item"],
        }
    )
    relations = pd.DataFrame(
        {
            "ocel:eid": ["e1", "e1", "e1", "e1"],
            "ocel:activity": ["act", "act", "act", "act"],
            "ocel:timestamp": pd.to_datetime(["2022-01-01"] * 4, utc=True),
            "ocel:oid": ["o1", "o2", "i1", "i2"],
            "ocel:type": ["order", "order", "item", "item"],
            "ocel:qualifier": ["places", "places", "places", "places"],
        }
    )
    o2o = pd.DataFrame(
        {
            "ocel:oid": ["o1", "o1", "o2", "o2"],
            "ocel:oid_2": ["i1", "i2", "i1", "i2"],
            "ocel:qualifier": ["None", "none", "NA", "N/A"],
        }
    )
    ocel = OCEL(events=events, objects=objects, relations=relations, o2o=o2o)
    path = str(tmp_path / "null_o2o_qualifier.jsonocel")
    _write_ocel(ocel, path)
    return path


@pytest.fixture
def shared_qualifier_ocel(tmp_path):
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
            "ocel:qualifier": ["contains", "contains"],
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
    path = str(tmp_path / "shared_qualifier.jsonocel")
    _write_ocel(ocel, path)
    return path


@pytest.fixture
def numeric_ids_ocel(tmp_path):
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


def test_returns_oced_store(simple_ocel):
    oced = PM4PyLoader(simple_ocel, {}).get_oced()

    assert isinstance(oced, OCEDStore)
    assert oced.get_event_types()
    assert oced.get_object_types()
    assert oced.get_relation_types(RelationKind.E2O)


def test_loads_correct_object_types(simple_ocel):
    oced = PM4PyLoader(simple_ocel, {}).get_oced()
    assert set(oced.get_object_types()) == {"order", "item"}


def test_loads_correct_activities(simple_ocel):
    oced = PM4PyLoader(simple_ocel, {}).get_oced()
    assert set(oced.get_event_types()) == {"place_order", "ship_order"}


def test_loads_correct_object_ids(simple_ocel):
    oced = PM4PyLoader(simple_ocel, {}).get_oced()
    assert set(oced.get_ids(EntityKind.OBJECT, "order").tolist()) == {"o1", "o2"}
    assert set(oced.get_ids(EntityKind.OBJECT, "item").tolist()) == {"i1"}


def test_loads_correct_event_ids(simple_ocel):
    oced = PM4PyLoader(simple_ocel, {}).get_oced()
    order_events = set(oced.get_ids(EntityKind.EVENT, "place_order").tolist())
    ship_events = set(oced.get_ids(EntityKind.EVENT, "ship_order").tolist())
    assert order_events == {"e1", "e2"}
    assert ship_events == {"e3"}


def test_format_auto_detected_from_sqlite_extension(tmp_path):
    dummy = str(tmp_path / "log.sqlite")
    open(dummy, "w").close()
    loader = PM4PyLoader(dummy, {})
    assert loader.config["file_format"] == "sqlite"


def test_format_auto_detected_from_jsonocel_extension(simple_ocel):
    loader = PM4PyLoader(simple_ocel, {})
    assert loader.config["file_format"] == "json"


def test_explicit_format_config_accepted(simple_ocel):
    explicit = PM4PyLoader(simple_ocel, {"file_format": "json"}).get_oced()
    inferred = PM4PyLoader(simple_ocel, {}).get_oced()

    assert explicit.get_event_types() == inferred.get_event_types()
    assert explicit.get_object_types() == inferred.get_object_types()


def test_unknown_extension_raises(tmp_path):
    dummy = str(tmp_path / "log.csv")
    with open(dummy, "w") as f:
        f.write("dummy")
    with pytest.raises(ValueError, match="[Uu]nsupported|[Uu]nknown|extension"):
        PM4PyLoader(dummy, {})


def test_e2o_default_modifies_relation_present(simple_ocel):
    oced = PM4PyLoader(simple_ocel, {}).get_oced()
    assert "modifies" in oced.get_relation_types()


def test_explicit_e2o_qualifier_stored_as_relation_name(explicit_qualifier_ocel):
    oced = PM4PyLoader(explicit_qualifier_ocel, {}).get_oced()
    relation_types = set(oced.get_relation_types())
    assert "creates" in relation_types
    assert "updates" in relation_types


def test_null_e2o_placeholder_qualifiers_normalised_to_modifies(null_qualifier_ocel):
    oced = PM4PyLoader(null_qualifier_ocel, {}).get_oced()
    assert "modifies" in oced.get_relation_types()
    for placeholder in ("None", "none", "NA", "N/A"):
        assert placeholder not in oced.get_relation_types()


def test_null_o2o_placeholder_qualifiers_normalised_to_relates(null_o2o_qualifier_ocel):
    oced = PM4PyLoader(null_o2o_qualifier_ocel, {}).get_oced()
    assert oced.get_relation_types(RelationKind.O2O) == ["relates"]


def test_o2o_relation_keeps_its_own_qualifier(o2o_ocel):
    oced = PM4PyLoader(o2o_ocel, {}).get_oced()
    assert oced.get_relation_types(RelationKind.O2O) == ["contains"]


def test_qualifier_used_by_both_families_suffixed_on_both_sides(shared_qualifier_ocel):
    oced = PM4PyLoader(shared_qualifier_ocel, {}).get_oced()
    assert oced.get_relation_types(RelationKind.E2O) == ["contains_e2o"]
    assert oced.get_relation_types(RelationKind.O2O) == ["contains_o2o"]
    assert "contains" not in oced.get_relation_types()


def test_a_suffixed_name_the_log_already_uses_is_moved_out_of_the_way():
    e2o, o2o = _relation_names(
        pd.Series(["contains", "contains_e2o"]), pd.Series(["contains"])
    )
    assert e2o == {"contains": "contains_e2o_", "contains_e2o": "contains_e2o"}
    assert o2o == {"contains": "contains_o2o"}


def test_o2o_null_values_raises(simple_ocel, monkeypatch):
    import semanticlog.data.loading.pm4py_loader as loader_module

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
        PM4PyLoader(simple_ocel, {}).get_oced()


def test_numeric_object_ids_prefixed_with_type(numeric_ids_ocel):
    oced = PM4PyLoader(numeric_ids_ocel, {}).get_oced()
    ids = set(oced.get_ids(EntityKind.OBJECT, "order").tolist())
    assert ids == {"order_10", "order_20"}


def test_numeric_event_ids_prefixed_with_activity(numeric_ids_ocel):
    oced = PM4PyLoader(numeric_ids_ocel, {}).get_oced()
    ids = set(oced.get_ids(EntityKind.EVENT, "act").tolist())
    assert ids == {"act_1", "act_2"}


def test_numeric_id_prefix_propagates_to_e2o_relation(numeric_ids_ocel):
    oced = PM4PyLoader(numeric_ids_ocel, {}).get_oced()
    rel = oced.get_e2o()["modifies"]
    to_ids = set(rel.data[rel.to_id].tolist())
    assert to_ids == {"order_10", "order_20"}


def test_string_ids_not_prefixed(simple_ocel):
    oced = PM4PyLoader(simple_ocel, {}).get_oced()
    ids = set(oced.get_ids(EntityKind.OBJECT, "order").tolist())
    assert ids == {"o1", "o2"}


def test_json_format_loads_successfully(tmp_path):
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

    result = PM4PyLoader(path, {}).get_oced()
    assert set(result.get_object_types()) == {"order"}
    assert set(result.get_event_types()) == {"act"}


@pytest.fixture
def object_changes_ocel(tmp_path, monkeypatch):
    import semanticlog.data.loading.pm4py_loader as loader_module

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
    oced = PM4PyLoader(simple_ocel, {}).get_oced()
    assert oced.ocel_mode is False


def test_attribute_history_sets_ocel_mode(object_changes_ocel):
    from semanticlog.onto.settings import OCELExtension

    oced = PM4PyLoader(
        object_changes_ocel, {}, ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY]
    ).get_oced()
    assert oced.ocel_mode is True


def test_attribute_history_populates_object_changes(object_changes_ocel):
    from semanticlog.onto.settings import OCELExtension

    oced = PM4PyLoader(
        object_changes_ocel, {}, ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY]
    ).get_oced()
    assert "order" in oced.get_object_change_types()
    changes = oced.get_object_changes("order")
    assert len(changes) == 3


def test_attribute_history_not_loaded_without_extension(object_changes_ocel):
    oced = PM4PyLoader(object_changes_ocel, {}).get_oced()
    assert oced.ocel_mode is False


def test_attribute_history_numeric_ids_prefixed_in_changes(tmp_path, monkeypatch):
    import semanticlog.data.loading.pm4py_loader as loader_module
    from semanticlog.onto.settings import OCELExtension

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

    oced = PM4PyLoader(
        dummy_path, {}, ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY]
    ).get_oced()
    changes = oced.get_object_changes("order")
    assert set(changes["ocel:oid"].tolist()) == {"order_10"}
