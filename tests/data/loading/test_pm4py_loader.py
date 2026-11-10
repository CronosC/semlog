"""Reading OCEL 2.0 logs through PM4Py."""

import json
import sqlite3

import pandas as pd
import pm4py
import pytest
from pm4py.objects.ocel.obj import OCEL

from semlog.core.enums import OCELExtension
from semlog.core.errors import InputError
from semlog.data.loading import pm4py_loader
from semlog.data.loading.pm4py_loader import PM4PyLoader
from semlog.data.tables import EntityKind, RelationKind
from tests.support.ocel_files import events_in_order

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
    contains = oced.relation("contains")

    assert oced.relation_types(RelationKind.E2O) == ["places", "ships"]
    assert oced.relation("places").to_type == ["order"]
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


def test_attribute_history_puts_the_latest_value_on_the_object(history_ocel):
    oced = PM4PyLoader(history_ocel, ocel_extensions=HISTORY).get_oced()
    orders = oced.table(EntityKind.OBJECT, "order").data

    assert orders[["ocel:oid", "status"]].values.tolist() == [
        ["order_10", "shipped"],
        ["order_20", "shipped"],
    ]


def test_attribute_history_reads_the_values_after_the_earliest_as_changes(
    history_ocel,
):
    oced = PM4PyLoader(history_ocel, ocel_extensions=HISTORY).get_oced()
    changes = oced.object_changes("order")

    assert oced.ocel_mode is True
    assert changes[["ocel:oid", "ocel:timestamp", "status"]].values.tolist() == [
        ["order_10", pd.Timestamp("2022-01-03", tz="UTC"), "shipped"],
        ["order_20", pd.Timestamp("2022-01-04", tz="UTC"), "shipped"],
    ]


def test_without_the_history_extension_no_change_is_read(history_ocel):
    assert PM4PyLoader(history_ocel).get_oced().ocel_mode is False


def _timed_values_log(tmp_path, values: list[tuple[str, str]]) -> str:
    """Write one order whose `status` takes `values`, each as (time, value), in that order."""
    path = str(tmp_path / "timed.jsonocel")
    document = {
        "objectTypes": [
            {"name": "order", "attributes": [{"name": "status", "type": "string"}]}
        ],
        "eventTypes": [{"name": "create", "attributes": []}],
        "objects": [
            {
                "id": "o1",
                "type": "order",
                "attributes": [
                    {"name": "status", "time": time, "value": value}
                    for time, value in values
                ],
            }
        ],
        "events": [
            {
                "id": "e1",
                "type": "create",
                "time": "2024-01-01T09:00:00Z",
                "attributes": [],
                "relationships": [{"objectId": "o1", "qualifier": "for"}],
            }
        ],
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle)
    return path


def _status_of_o1(path: str, extensions=()) -> str:
    orders = (
        PM4PyLoader(path, ocel_extensions=extensions)
        .get_oced()
        .table(EntityKind.OBJECT, "order")
    )
    return orders.data.set_index("ocel:oid").loc["o1", "status"]


def test_a_change_with_an_unparseable_timestamp_loses_to_a_dated_one(tmp_path):
    path = _timed_values_log(
        tmp_path,
        [
            ("2024-01-01T09:00:00Z", "pending"),
            ("2024-01-05T09:00:00Z", "shipped"),
            ("later", "lost"),
        ],
    )
    assert _status_of_o1(path) == "shipped"


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


@pytest.mark.parametrize("listed", [["e10", "e9"], ["e9", "e10"]])
def test_events_at_one_timestamp_are_positioned_in_file_order(tmp_path, listed):
    activities = {"e10": "pay", "e9": "ship"}
    path = tmp_path / "tied.jsonocel"
    path.write_text(
        json.dumps(
            {
                "objectTypes": [{"name": "order", "attributes": []}],
                "eventTypes": [
                    {"name": act, "attributes": []} for act in ("pay", "ship")
                ],
                "objects": [{"id": "o1", "type": "order"}],
                "events": [
                    {
                        "id": eid,
                        "type": activities[eid],
                        "time": "2024-01-01T00:00:00Z",
                        "relationships": [{"objectId": "o1", "qualifier": "on"}],
                    }
                    for eid in listed
                ],
            }
        )
    )

    assert events_in_order(PM4PyLoader(str(path)).get_oced()) == listed


def test_the_event_position_is_no_attribute(simple_ocel):
    oced = PM4PyLoader(simple_ocel).get_oced()

    assert [
        view.attributes for view in oced.tables() if view.name == "place_order"
    ] == [[]]
