"""Reading flat single-case logs."""

import pandas as pd
import pytest

from semanticlog.core.errors import ConfigError, InputError
from semanticlog.data.loading.base import LoaderColumns
from semanticlog.data.loading.classical_loader import ClassicalLoader
from semanticlog.data.tables import EntityKind

_CSV_CONTENT = """case:concept:name,concept:name,time:timestamp,resource
c1,A,2022-01-01T08:00:00Z,Alice
c1,B,2022-01-01T09:00:00Z,Bob
c2,A,2022-01-02T08:00:00Z,Alice
c2,C,2022-01-02T09:00:00Z,Charlie
"""

# The same log as _CSV_CONTENT, with the resource under its XES name.
_XES_CONTENT = """<?xml version="1.0" encoding="UTF-8" ?>
<log xes.version="1.0" xes.features="nested-attributes">
  <trace>
    <string key="concept:name" value="c1"/>
    <event>
      <string key="concept:name" value="A"/>
      <date key="time:timestamp" value="2022-01-01T08:00:00+00:00"/>
      <string key="org:resource" value="Alice"/>
    </event>
    <event>
      <string key="concept:name" value="B"/>
      <date key="time:timestamp" value="2022-01-01T09:00:00+00:00"/>
      <string key="org:resource" value="Bob"/>
    </event>
  </trace>
  <trace>
    <string key="concept:name" value="c2"/>
    <event>
      <string key="concept:name" value="A"/>
      <date key="time:timestamp" value="2022-01-02T08:00:00+00:00"/>
      <string key="org:resource" value="Alice"/>
    </event>
    <event>
      <string key="concept:name" value="C"/>
      <date key="time:timestamp" value="2022-01-02T09:00:00+00:00"/>
      <string key="org:resource" value="Charlie"/>
    </event>
  </trace>
</log>
"""

CASE = "case:concept:name"


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "log.csv"
    path.write_text(_CSV_CONTENT)
    return str(path)


def _write(tmp_path, **columns):
    """Write a two-row log with the given columns over the defaults, and return its path."""
    log = {
        CASE: ["c1", "c1"],
        "concept:name": ["a", "b"],
        "time:timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
        **columns,
    }
    path = tmp_path / "log.csv"
    pd.DataFrame(log).to_csv(path, index=False)
    return str(path)


def _all_events(oced, column):
    return {
        value
        for activity in oced.types(EntityKind.EVENT)
        for value in oced.table(EntityKind.EVENT, activity).data[column]
    }


def test_a_file_that_is_neither_csv_nor_xes_is_rejected():
    with pytest.raises(InputError, match=r"Can only load \.xes or \.csv"):
        ClassicalLoader("mylog.json")


def test_each_row_is_an_event_of_its_activity_related_to_its_case(csv_path):
    oced = ClassicalLoader(csv_path).get_oced()
    relation = oced.get_relation("modifies")

    assert oced.types(EntityKind.EVENT) == ["A", "B", "C"]
    assert list(oced.ids(EntityKind.EVENT, "A")) == ["e0", "e2"]
    assert oced.types(EntityKind.OBJECT) == [CASE]
    assert list(oced.ids(EntityKind.OBJECT, CASE)) == ["tc1", "tc2"]
    assert relation.to_type == [CASE]
    assert relation.data.values.tolist() == [
        ["e0", "tc1"],
        ["e1", "tc1"],
        ["e2", "tc2"],
        ["e3", "tc2"],
    ]


def test_an_xes_log_becomes_the_store_a_csv_of_it_would(tmp_path, csv_path):
    xes_path = tmp_path / "log.XES"
    xes_path.write_text(_XES_CONTENT, encoding="utf-8")

    from_xes = ClassicalLoader(str(xes_path)).get_oced()
    from_csv = ClassicalLoader(csv_path).get_oced()

    assert from_xes.types(EntityKind.EVENT) == from_csv.types(EntityKind.EVENT)
    assert list(from_xes.ids(EntityKind.OBJECT, CASE)) == list(
        from_csv.ids(EntityKind.OBJECT, CASE)
    )
    assert _all_events(from_xes, "org:resource") == _all_events(from_csv, "resource")
    assert _all_events(from_xes, "time:timestamp") == _all_events(
        from_csv, "time:timestamp"
    )


def test_a_column_constant_within_each_case_moves_onto_the_case(tmp_path):
    path = _write(tmp_path, region=["north", "north"], cost=[10, 20])

    oced = ClassicalLoader(path).get_oced()

    assert list(oced.table(EntityKind.OBJECT, CASE).data.columns) == [CASE, "region"]
    assert list(oced.table(EntityKind.EVENT, "a").data.columns) == [
        "event_id",
        "time:timestamp",
        "cost",
    ]


def test_an_object_column_becomes_a_type_with_prefixed_ids_and_its_own_relation(
    tmp_path,
):
    path = _write(tmp_path, resource=["ann", None])

    oced = ClassicalLoader(path, LoaderColumns(object_cols=["resource"])).get_oced()

    assert list(oced.ids(EntityKind.OBJECT, "resource")) == ["resource_ann"]
    assert oced.get_relation("modifies_resource").data.values.tolist() == [
        ["e0", "resource_ann"]
    ]


def test_an_event_id_column_of_the_log_is_kept_as_an_attribute(tmp_path):
    path = _write(tmp_path, event_id=["REAL-1", "REAL-2"])

    oced = ClassicalLoader(path).get_oced()

    assert _all_events(oced, "event_id") == {"REAL-1", "REAL-2"}
    assert _all_events(oced, "event_id_2") == {"e0", "e1"}


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ({"time_col": "when"}, r"time_col 'when' is not a column of .*log\.csv"),
        (
            {"object_cols": ["nope"]},
            r"object_cols names \['nope'\], which are not columns",
        ),
    ],
)
def test_a_configured_column_the_log_does_not_have_is_rejected(
    tmp_path, columns, message
):
    path = _write(tmp_path)

    with pytest.raises(ConfigError, match=message):
        ClassicalLoader(path, LoaderColumns(**columns)).get_oced()


def test_rows_with_no_case_id_are_rejected_naming_them(tmp_path):
    path = _write(tmp_path, **{CASE: [None, "  "]})

    with pytest.raises(
        InputError,
        match=r"2 row\(s\) of .* have no value in case_col .* \(rows \[0, 1\]\)",
    ):
        ClassicalLoader(path).get_oced()


def test_a_time_value_that_is_not_iso_8601_is_rejected(tmp_path):
    path = _write(tmp_path, **{"time:timestamp": ["01/02/2024 08:00", None]})

    with pytest.raises(
        InputError,
        match=r"time_col 'time:timestamp' holds 1 value\(s\) that are not ISO 8601 timestamps, such as \['01/02/2024 08:00'\]",
    ):
        ClassicalLoader(path).get_oced()


def test_the_time_column_is_parsed_over_the_whole_log_keeping_gaps(tmp_path):
    path = _write(
        tmp_path,
        **{
            "concept:name": ["a", "b", "b"],
            CASE: ["c1"] * 3,
            "time:timestamp": ["2024-01-01T00:00:00Z", "2024-01-03", None],
        },
    )

    oced = ClassicalLoader(path).get_oced()

    assert list(oced.table(EntityKind.EVENT, "b").data["time:timestamp"]) == [
        pd.Timestamp("2024-01-03", tz="UTC"),
        pd.NaT,
    ]
