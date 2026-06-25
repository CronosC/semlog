import pytest

from semanticlog.data.loading.classicalloader import ClassicalLoader

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# File extension validation
# ---------------------------------------------------------------------------


def test_classicalloader_raises_for_unsupported_extension():
    with pytest.raises(ValueError, match="Can only load"):
        ClassicalLoader("mylog.json", {})


# ---------------------------------------------------------------------------
# Event IDs and case IDs
# ---------------------------------------------------------------------------


def test_get_oced_event_ids_have_e_prefix(csv_path_no_resource):
    oced = ClassicalLoader(csv_path_no_resource, {}).get_oced()
    for activity in oced.get_activities():
        ev_ids = oced.get_ids("event", activity)
        for eid in ev_ids:
            assert str(eid).startswith("e"), f"Event ID '{eid}' missing 'e' prefix"


def test_get_oced_case_ids_have_t_prefix(csv_path_no_resource):
    oced = ClassicalLoader(csv_path_no_resource, {}).get_oced()
    # The case object type is "case:concept:name" by default
    case_type = "case:concept:name"
    assert case_type in oced.get_object_types()
    case_ids = oced.get_ids("object", case_type)
    for cid in case_ids:
        assert str(cid).startswith("t"), f"Case ID '{cid}' missing 't' prefix"


# ---------------------------------------------------------------------------
# Activity structure
# ---------------------------------------------------------------------------


def test_get_oced_creates_one_activity_per_unique_activity_value(csv_path_no_resource):
    oced = ClassicalLoader(csv_path_no_resource, {}).get_oced()
    # CSV has activities: A, B, A, C  → unique: {A, B, C}
    # Wait — csv_path_no_resource has A, B, A → unique: {A, B}
    assert set(oced.get_activities()) == {"A", "B"}


def test_get_oced_activities_match_csv_activity_column(csv_path):
    oced = ClassicalLoader(csv_path, {}).get_oced()
    # csv_path has activities A, B, A, C → unique: {A, B, C}
    assert set(oced.get_activities()) == {"A", "B", "C"}


# ---------------------------------------------------------------------------
# E2O relation
# ---------------------------------------------------------------------------


def test_get_oced_creates_modifies_e2o_relation(csv_path_no_resource):
    oced = ClassicalLoader(csv_path_no_resource, {}).get_oced()
    assert "modifies" in oced.get_relation_types("e2o")


def test_get_oced_modifies_relation_links_events_to_case_type(csv_path_no_resource):
    oced = ClassicalLoader(csv_path_no_resource, {}).get_oced()
    rel = oced.get_relation("modifies")
    assert "case:concept:name" in rel["to_type"]


# ---------------------------------------------------------------------------
# Extra object_cols
# ---------------------------------------------------------------------------


def test_get_oced_extra_object_col_creates_object_type_and_relation(csv_path):
    oced = ClassicalLoader(csv_path, {"object_cols": ["resource"]}).get_oced()
    assert "resource" in oced.get_object_types()
    assert "modifies_resource" in oced.get_relation_types("e2o")


def test_get_oced_extra_object_col_resource_ids_present(csv_path):
    oced = ClassicalLoader(csv_path, {"object_cols": ["resource"]}).get_oced()
    resource_ids = set(oced.get_ids("object", "resource"))
    assert resource_ids == {"Alice", "Bob", "Charlie"}
