"""Reading OCEL 2.0 logs, and the repairs made on the way in."""

import json
import logging
import sqlite3

import pandas as pd
import pm4py
import pytest
from pm4py.objects.ocel.obj import OCEL

from semanticlog.core.enums import OCELExtension
from semanticlog.core.errors import InputError
from semanticlog.data.loading import pm4py_loader
from semanticlog.data.loading.pm4py_loader import (
    PM4PyLoader,
    _normalize_qualifiers,
    _relation_names,
)
from semanticlog.data.tables import EntityKind, RelationKind

HISTORY = [OCELExtension.ATTRIBUTE_HISTORY]


def _stamps(count: int) -> pd.Series:
    return pd.to_datetime(
        [f"2022-01-{day:02d}" for day in range(1, count + 1)], utc=True
    )


def _ocel(
    tmp_path,
    name: str,
    *,
    objects: dict[str, str],
    events: list[tuple[str, str, list[tuple[str, str]]]],
    o2o: list[tuple[str, str, str]] = (),
    changes: pd.DataFrame | None = None,
) -> str:
    """Write one OCEL 2.0 JSON file and return its path.

    Args:
        objects: Object id to type.
        events: Each as event id, activity and its (object id, qualifier)
            relations, in order. The events are dated on consecutive days.
        o2o: Each as object id, object id and qualifier.
        changes: Attribute change rows, if the log carries a history.
    """
    stamps = _stamps(len(events))
    event_rows = pd.DataFrame(
        {
            "ocel:eid": [eid for eid, _, _ in events],
            "ocel:activity": [act for _, act, _ in events],
            "ocel:timestamp": stamps,
        }
    )
    relation_rows = [
        (eid, act, stamp, oid, objects[oid], qualifier)
        for (eid, act, relations), stamp in zip(events, stamps, strict=True)
        for oid, qualifier in relations
    ]
    relations = pd.DataFrame(
        relation_rows,
        columns=[
            "ocel:eid",
            "ocel:activity",
            "ocel:timestamp",
            "ocel:oid",
            "ocel:type",
            "ocel:qualifier",
        ],
    )
    o2o_rows = pd.DataFrame(
        list(o2o) or None, columns=["ocel:oid", "ocel:oid_2", "ocel:qualifier"]
    ).astype(str)
    ocel = OCEL(
        events=event_rows,
        objects=pd.DataFrame(
            {"ocel:oid": list(objects), "ocel:type": list(objects.values())}
        ),
        relations=relations,
        o2o=o2o_rows,
        object_changes=changes,
    )
    path = str(tmp_path / f"{name}.jsonocel")
    pm4py.write_ocel2_json(ocel, path)
    return path


@pytest.fixture
def simple_ocel(tmp_path):
    return _ocel(
        tmp_path,
        "simple",
        objects={"o1": "order", "o2": "order", "i1": "item"},
        events=[
            ("e1", "place_order", [("o1", "places")]),
            ("e2", "place_order", [("o2", "places")]),
            ("e3", "ship_order", [("i1", "ships")]),
        ],
        o2o=[("o1", "i1", "contains")],
    )


@pytest.fixture
def history_ocel(tmp_path):
    """Two orders with numeric ids, each changing its status twice."""
    changes = pd.DataFrame(
        {
            "ocel:oid": ["10", "10", "20", "20"],
            "ocel:type": ["order"] * 4,
            "ocel:timestamp": pd.to_datetime(
                ["2022-01-01", "2022-01-03", "2022-01-02", "2022-01-04"], utc=True
            ),
            "ocel:field": ["status"] * 4,
            "status": ["pending", "shipped", "pending", "shipped"],
        }
    )
    return _ocel(
        tmp_path,
        "changes",
        objects={"10": "order", "20": "order"},
        events=[
            ("e1", "place_order", [("10", "places")]),
            ("e2", "place_order", [("20", "places")]),
        ],
        changes=changes,
    )


def test_each_activity_and_object_type_of_the_log_becomes_a_table(simple_ocel):
    oced = PM4PyLoader(simple_ocel).get_oced()

    assert oced.types(EntityKind.EVENT) == ["place_order", "ship_order"]
    assert list(oced.ids(EntityKind.EVENT, "place_order")) == ["e1", "e2"]
    assert oced.types(EntityKind.OBJECT) == ["item", "order"]
    assert list(oced.ids(EntityKind.OBJECT, "order")) == ["o1", "o2"]


def test_each_qualifier_becomes_a_relation_holding_its_endpoints_and_types(simple_ocel):
    oced = PM4PyLoader(simple_ocel).get_oced()
    contains = oced.get_relation("contains")

    assert oced.get_relation_types(RelationKind.E2O) == ["places", "ships"]
    assert oced.get_relation("places").to_type == ["order"]
    assert list(contains.data.columns) == ["ocel:oid", "ocel:oid_2"]
    assert (contains.from_type, contains.to_type) == (["order"], ["item"])


def test_the_format_is_read_off_the_suffix(tmp_path):
    formats = [
        PM4PyLoader(str(tmp_path / f"log{suffix}")).file_format
        for suffix in (".sqlite", ".jsonocel", ".XMLOCEL")
    ]

    assert formats == ["sqlite", "json", "xml"]


def test_a_suffix_with_no_ocel_format_is_rejected(tmp_path):
    with pytest.raises(InputError, match=r"Cannot infer OCEL format from '.*log\.csv'"):
        PM4PyLoader(str(tmp_path / "log.csv"))


def test_a_missing_qualifier_takes_the_fallback_name_of_its_family(tmp_path):
    path = _ocel(
        tmp_path,
        "placeholders",
        objects={"o1": "order", "i1": "item"},
        events=[("e1", "act", [("o1", "None"), ("i1", "N/A")])],
        o2o=[("o1", "i1", "NA")],
    )

    oced = PM4PyLoader(path).get_oced()

    assert oced.get_relation_types(RelationKind.E2O) == ["modifies"]
    assert oced.get_relation_types(RelationKind.O2O) == ["relates"]


def test_a_qualifier_both_families_use_is_suffixed_on_both_sides(tmp_path):
    path = _ocel(
        tmp_path,
        "shared",
        objects={"o1": "order", "i1": "item"},
        events=[("e1", "act", [("o1", "contains"), ("i1", "contains")])],
        o2o=[("o1", "i1", "contains")],
    )

    oced = PM4PyLoader(path).get_oced()

    assert oced.get_relation_types() == ["contains_e2o", "contains_o2o"]


def test_a_suffixed_name_the_log_uses_is_numbered_past():
    e2o, o2o = _relation_names(
        pd.Series(["contains", "contains_e2o"]), pd.Series(["contains"])
    )

    assert e2o == {"contains": "contains_e2o_2", "contains_e2o": "contains_e2o"}
    assert o2o == {"contains": "contains_o2o"}


def test_a_word_read_as_no_qualifier_is_warned_about_apart_from_an_empty_one(caplog):
    rows = pd.DataFrame({"ocel:qualifier": ["modifies", None, "NA"]})

    with caplog.at_level(logging.WARNING):
        normalized = _normalize_qualifiers(rows, "e2o")

    assert list(normalized["ocel:qualifier"]) == ["modifies"] * 3
    assert "2 e2o relation(s) had no qualifier" in caplog.text
    assert (
        "1 of those carry the text (NA), which is read as no qualifier" in caplog.text
    )


def test_a_word_on_every_relation_of_a_family_is_not_warned_about_as_a_merge(caplog):
    rows = pd.DataFrame({"ocel:qualifier": ["None", "None"]})

    with caplog.at_level(logging.WARNING):
        _normalize_qualifiers(rows, "e2o")

    assert "2 e2o relation(s) had no qualifier" in caplog.text
    assert "carry the text" not in caplog.text


def test_numeric_ids_are_prefixed_with_their_type_everywhere_they_appear(tmp_path):
    path = _ocel(
        tmp_path,
        "numeric",
        objects={"10": "order", "20": "order"},
        events=[("1", "act", [("10", "places")]), ("2", "act", [("20", "places")])],
    )

    oced = PM4PyLoader(path).get_oced()

    assert list(oced.ids(EntityKind.OBJECT, "order")) == ["order_10", "order_20"]
    assert list(oced.ids(EntityKind.EVENT, "act")) == ["act_1", "act_2"]
    assert oced.get_relation("places").data.values.tolist() == [
        ["act_1", "order_10"],
        ["act_2", "order_20"],
    ]


def test_an_o2o_row_naming_no_object_of_the_log_is_rejected(simple_ocel, monkeypatch):
    log = pm4py.read_ocel2_json(simple_ocel)
    log.o2o = pd.DataFrame(
        {"ocel:oid": ["o1"], "ocel:oid_2": [None], "ocel:qualifier": ["rel"]}
    )
    monkeypatch.setattr(pm4py_loader.pm, "read_ocel2_json", lambda _path: log)

    with pytest.raises(
        InputError, match="Could not resolve object types for o2o relation 'rel'"
    ):
        PM4PyLoader(simple_ocel).get_oced()


def test_one_id_for_two_objects_in_a_json_log_is_rejected(tmp_path):
    doc = {
        "objectTypes": [{"name": "order", "attributes": []}],
        "eventTypes": [{"name": "act", "attributes": []}],
        "objects": [
            {"id": "1", "type": "order", "attributes": []},
            {"id": "1", "type": "order", "attributes": []},
        ],
        "events": [],
    }
    path = tmp_path / "twice.jsonocel"
    path.write_text(json.dumps(doc))

    with pytest.raises(
        InputError, match="uses one object id for more than one object: 1"
    ):
        PM4PyLoader(str(path)).get_oced()


def test_one_id_for_two_objects_in_an_xml_log_is_rejected(tmp_path):
    path = tmp_path / "twice.xmlocel"
    path.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<log>
  <object-types><object-type name="order"><attributes/></object-type></object-types>
  <event-types><event-type name="act"><attributes/></event-type></event-types>
  <objects>
    <object id="1" type="order"><attributes/></object>
    <object id="1" type="order"><attributes/></object>
  </objects>
  <events>
    <event id="e1" type="act" time="2022-01-01T00:00:00+00:00">
      <attributes/>
      <objects><relationship object-id="1" qualifier="modifies"/></objects>
    </event>
  </events>
</log>
"""
    )

    with pytest.raises(
        InputError, match="uses one object id for more than one object: 1"
    ):
        PM4PyLoader(str(path)).get_oced()


def test_one_id_for_two_events_in_a_sqlite_log_is_rejected(tmp_path, simple_ocel):
    path = tmp_path / "twice.sqlite"
    pm4py.write_ocel2_sqlite(pm4py.read_ocel2_json(simple_ocel), str(path))
    connection = sqlite3.connect(path)
    connection.execute("INSERT INTO event SELECT * FROM event WHERE ocel_id = 'e1'")
    connection.commit()
    connection.close()

    with pytest.raises(
        InputError, match="uses one event id for more than one event: e1"
    ):
        PM4PyLoader(str(path)).get_oced()


def test_attribute_history_puts_the_first_value_on_the_object_and_later_ones_in_changes(
    history_ocel,
):
    oced = PM4PyLoader(history_ocel, ocel_extensions=HISTORY).get_oced()
    orders = oced.table(EntityKind.OBJECT, "order").data
    changes = oced.get_object_changes("order")

    assert oced.ocel_mode is True
    assert dict(zip(orders["ocel:oid"], orders["status"], strict=True)) == {
        "order_10": "pending",
        "order_20": "pending",
    }
    assert dict(zip(changes["ocel:oid"], changes["status"], strict=True)) == {
        "order_10": "shipped",
        "order_20": "shipped",
    }


def test_without_the_history_extension_no_change_is_read(history_ocel):
    assert PM4PyLoader(history_ocel).get_oced().ocel_mode is False


def test_an_event_id_the_reader_returns_twice_is_rejected(simple_ocel, monkeypatch):
    log = pm4py.read_ocel2_json(simple_ocel)
    log.events = pd.concat([log.events, log.events.iloc[:1]], ignore_index=True)
    monkeypatch.setattr(pm4py_loader.pm, "read_ocel2_json", lambda _path: log)

    with pytest.raises(
        InputError,
        match=r"Event ids are not unique across the log: e1\. Give every event and object an id of its own\.",
    ):
        PM4PyLoader(simple_ocel).get_oced()


def _typed_ocel(tmp_path, suffix: str) -> str:
    """Write a log declaring a text `code` of digits and an integer `rank`, and return its path."""
    doc = {
        "objectTypes": [
            {
                "name": "order",
                "attributes": [
                    {"name": "code", "type": "string"},
                    {"name": "rank", "type": "integer"},
                ],
            }
        ],
        "eventTypes": [{"name": "place_order", "attributes": []}],
        "objects": [
            {
                "id": f"o{i}",
                "type": "order",
                "attributes": [
                    {"name": "code", "time": "1970-01-01T00:00:00Z", "value": str(i)},
                    {"name": "rank", "time": "1970-01-01T00:00:00Z", "value": str(i)},
                ],
            }
            for i in (1, 2)
        ],
        "events": [
            {
                "id": "e1",
                "type": "place_order",
                "time": "2022-01-01T00:00:00Z",
                "attributes": [],
                "relationships": [
                    {"objectId": "o1", "qualifier": "places"},
                    {"objectId": "o2", "qualifier": "places"},
                ],
            }
        ],
    }
    path = tmp_path / f"typed{suffix}"
    if suffix == ".jsonocel":
        path.write_text(json.dumps(doc))
        return str(path)
    path.write_text(
        """<?xml version='1.0' encoding='UTF-8'?>
<log>
  <object-types>
    <object-type name="order">
      <attributes>
        <attribute name="code" type="string"/>
        <attribute name="rank" type="integer"/>
      </attributes>
    </object-type>
  </object-types>
  <event-types>
    <event-type name="place_order"><attributes/></event-type>
  </event-types>
  <objects>
"""
        + "".join(
            f"""    <object id="o{i}" type="order">
      <attributes>
        <attribute name="code" time="1970-01-01T00:00:00Z">{i}</attribute>
        <attribute name="rank" time="1970-01-01T00:00:00Z">{i}</attribute>
      </attributes>
    </object>
"""
            for i in (1, 2)
        )
        + """  </objects>
  <events>
    <event id="e1" type="place_order" time="2022-01-01T00:00:00Z">
      <attributes/>
      <objects>
        <relationship object-id="o1" qualifier="places"/>
        <relationship object-id="o2" qualifier="places"/>
      </objects>
    </event>
  </events>
</log>
"""
    )
    return str(path)


@pytest.mark.parametrize("suffix", [".jsonocel", ".xmlocel"])
def test_an_attribute_keeps_the_type_the_log_declares(tmp_path, suffix):
    path = _typed_ocel(tmp_path, suffix)

    frame = PM4PyLoader(path).get_oced().table(EntityKind.OBJECT, "order").data

    assert list(frame["code"]) == ["1", "2"]
    assert list(frame["rank"]) == [1, 2]
    assert str(frame["rank"].dtype) == "Int64"
