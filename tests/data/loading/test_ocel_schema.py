"""Reading the attribute types an OCEL 2.0 file declares."""

import logging
import sqlite3

import pytest

from semlog.data.loading.ocel_schema import declared_types, normalize_type
from semlog.data.tables import EntityKind


@pytest.mark.parametrize(
    ("raw", "declared"),
    [
        ("string", "string"),
        ("TEXT", "string"),
        ("integer", "integer"),
        ("INTEGER", "integer"),
        ("double", "float"),
        ("REAL", "float"),
        ("date", "time"),
        ("TIMESTAMP", "time"),
        ("boolean", "boolean"),
        ("blob", None),
    ],
)
def test_a_declared_type_name_maps_to_its_ocel_type(raw, declared):
    assert normalize_type(raw) == declared


def test_a_sqlite_log_declares_the_column_types_of_its_type_tables(tmp_path):
    path = tmp_path / "log.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE object_map_type (ocel_type TEXT, ocel_type_map TEXT);
        INSERT INTO object_map_type VALUES ('purchase order', 'purchaseorder');
        CREATE TABLE event_map_type (ocel_type TEXT, ocel_type_map TEXT);
        INSERT INTO event_map_type VALUES ('Create Order', 'CreateOrder');
        CREATE TABLE object_purchaseorder (
            ocel_id TEXT, ocel_time TIMESTAMP, code TEXT, amount REAL,
            ocel_changed_field TEXT
        );
        CREATE TABLE event_CreateOrder (ocel_id TEXT, ocel_time TIMESTAMP, n INTEGER);
        """
    )
    connection.close()

    assert declared_types(str(path), "sqlite") == {
        (EntityKind.OBJECT, "purchase order"): {"code": "string", "amount": "float"},
        (EntityKind.EVENT, "Create Order"): {"n": "integer"},
    }


def test_an_unknown_declared_type_is_left_to_inference_with_a_warning(tmp_path, caplog):
    path = tmp_path / "log.jsonocel"
    path.write_text(
        '{"objectTypes": [{"name": "order", "attributes": '
        '[{"name": "photo", "type": "blob"}, {"name": "code", "type": "string"}]}]}'
    )

    with caplog.at_level(logging.WARNING):
        declared = declared_types(str(path), "json")

    assert declared == {(EntityKind.OBJECT, "order"): {"code": "string"}}
    assert "'blob'" in caplog.text
