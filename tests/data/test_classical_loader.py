"""Reading flat single-case logs."""

import pandas as pd
import pytest

from semanticlog.data.loading.classical_loader import ClassicalLoader
from semanticlog.data.tables import EntityKind, RelationKind

_CSV_CONTENT = """case:concept:name,concept:name,time:timestamp,resource
c1,A,2022-01-01 08:00:00,Alice
c1,B,2022-01-01 09:00:00,Bob
c2,A,2022-01-02 08:00:00,Alice
c2,C,2022-01-02 09:00:00,Charlie
"""


@pytest.fixture
def csv_path(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text(_CSV_CONTENT)
    return str(p)


@pytest.fixture
def csv_path_no_resource(tmp_path):
    content = """case:concept:name,concept:name,time:timestamp
c1,A,2022-01-01 08:00:00
c1,B,2022-01-01 09:00:00
c2,A,2022-01-02 08:00:00
"""
    p = tmp_path / "log_simple.csv"
    p.write_text(content)
    return str(p)


def test_classicalloader_raises_for_unsupported_extension():
    with pytest.raises(ValueError, match="Can only load"):
        ClassicalLoader("mylog.json", {})


def test_get_oced_event_ids_have_e_prefix(csv_path_no_resource):
    oced = ClassicalLoader(csv_path_no_resource, {}).get_oced()
    for activity in oced.get_event_types():
        ev_ids = oced.get_ids(EntityKind.EVENT, activity)
        for eid in ev_ids:
            assert str(eid).startswith("e"), f"Event ID '{eid}' missing 'e' prefix"


def test_get_oced_case_ids_have_t_prefix(csv_path_no_resource):
    oced = ClassicalLoader(csv_path_no_resource, {}).get_oced()
    case_type = "case:concept:name"
    assert case_type in oced.get_object_types()
    case_ids = oced.get_ids(EntityKind.OBJECT, case_type)
    for cid in case_ids:
        assert str(cid).startswith("t"), f"Case ID '{cid}' missing 't' prefix"


def test_get_oced_creates_one_activity_per_unique_activity_value(csv_path_no_resource):
    oced = ClassicalLoader(csv_path_no_resource, {}).get_oced()
    assert set(oced.get_event_types()) == {"A", "B"}


def test_get_oced_activities_match_csv_activity_column(csv_path):
    oced = ClassicalLoader(csv_path, {}).get_oced()
    assert set(oced.get_event_types()) == {"A", "B", "C"}


def test_get_oced_creates_modifies_e2o_relation(csv_path_no_resource):
    oced = ClassicalLoader(csv_path_no_resource, {}).get_oced()
    assert "modifies" in oced.get_relation_types(RelationKind.E2O)


def test_get_oced_modifies_relation_links_events_to_case_type(csv_path_no_resource):
    oced = ClassicalLoader(csv_path_no_resource, {}).get_oced()
    rel = oced.get_relation("modifies")
    assert "case:concept:name" in rel.to_type


def test_get_oced_extra_object_col_creates_object_type_and_relation(csv_path):
    oced = ClassicalLoader(csv_path, {"object_cols": ["resource"]}).get_oced()
    assert "resource" in oced.get_object_types()
    assert "modifies_resource" in oced.get_relation_types(RelationKind.E2O)


def test_get_oced_extra_object_col_resource_ids_present(csv_path):
    """Ids are prefixed with their column, so no two object types can share one."""
    oced = ClassicalLoader(csv_path, {"object_cols": ["resource"]}).get_oced()
    resource_ids = set(oced.get_ids(EntityKind.OBJECT, "resource"))
    assert resource_ids == {"resource_Alice", "resource_Bob", "resource_Charlie"}


# --------------------------------------------------------------------------
# Attributes
# --------------------------------------------------------------------------


def test_a_column_that_varies_within_a_case_stays_on_the_events(csv_path):
    oced = ClassicalLoader(csv_path, {}).get_oced()

    # Alice and Bob both work on c1, so resource cannot describe the case.
    assert "resource" in oced.get_event_table("A").data.columns
    assert "resource" not in oced.get_object_table("case:concept:name").data.columns


def test_a_column_constant_within_a_case_is_lifted_onto_it(tmp_path):
    path = tmp_path / "log.csv"
    pd.DataFrame(
        {
            "case:concept:name": ["c1", "c1", "c2"],
            "concept:name": ["A", "B", "A"],
            "time:timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
            "region": ["north", "north", "south"],
            "cost": [10, 20, 30],
        }
    ).to_csv(path, index=False)

    oced = ClassicalLoader(str(path), {}).get_oced()
    cases = oced.get_object_table("case:concept:name").data

    assert set(cases["region"]) == {"north", "south"}
    assert "cost" not in cases.columns
    assert "cost" in oced.get_event_table("A").data.columns


def test_the_activity_and_the_timestamp_do_not_become_attributes(csv_path):
    oced = ClassicalLoader(csv_path, {}).get_oced()
    events = oced.get_event_table("A").data

    assert "concept:name" not in events.columns
    assert set(events.columns) == {"event_id", "time:timestamp", "resource"}


@pytest.mark.parametrize(
    ("setting", "column"),
    [("act_col", "concept:name"), ("time_col", "time:timestamp")],
)
def test_an_unknown_column_is_named_like_the_case_column_is(tmp_path, setting, column):
    path = tmp_path / "log.csv"
    pd.DataFrame(
        {
            "case:concept:name": ["c1"],
            "concept:name": ["a"],
            "time:timestamp": ["2024-01-01T00:00:00Z"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing_column"):
        ClassicalLoader(str(path), {setting: "missing_column"}).get_oced()


def test_an_unknown_object_col_is_named(tmp_path):
    path = tmp_path / "log.csv"
    pd.DataFrame(
        {
            "case:concept:name": ["c1"],
            "concept:name": ["a"],
            "time:timestamp": ["2024-01-01T00:00:00Z"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="object_cols"):
        ClassicalLoader(str(path), {"object_cols": ["nope"]}).get_oced()


def test_a_custom_case_col_is_actually_used(tmp_path):
    path = tmp_path / "log.csv"
    pd.DataFrame(
        {
            "my_case": ["c1", "c1", "c2"],
            "concept:name": ["a", "b", "a"],
            "time:timestamp": [
                "2024-01-01T00:00:00Z",
                "2024-01-02T00:00:00Z",
                "2024-01-03T00:00:00Z",
            ],
        }
    ).to_csv(path, index=False)

    oced = ClassicalLoader(str(path), {"case_col": "my_case"}).get_oced()

    assert "my_case" in oced.get_object_types()
    assert set(oced.get_ids(EntityKind.OBJECT, "my_case")) == {"tc1", "tc2"}


def test_an_unknown_case_col_is_named(tmp_path):
    path = tmp_path / "log.csv"
    pd.DataFrame(
        {
            "case:concept:name": ["c1"],
            "concept:name": ["a"],
            "time:timestamp": ["2024-01-01T00:00:00Z"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing_column"):
        ClassicalLoader(str(path), {"case_col": "missing_column"}).get_oced()
