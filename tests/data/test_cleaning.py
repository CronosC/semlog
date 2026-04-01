import pandas as pd
import pytest

from semanticlog.data.cleaning import (
    apply_function_to_value,
    clean_column_values,
    clean_dataframe,
    clean_oced_data,
    clean_string,
    clean_value,
    fix_duplicate_names,
    is_timestamp,
    null_check,
)
from semanticlog.data.oced import OCEDStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _oced_with_event(activity: str, attr_col: str = "status", attr_val: str = "ok"):
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["e1", "e2"],
            "ts": pd.date_range("2022-01-01", periods=2, tz="UTC"),
            attr_col: [attr_val, "normal_value"],
        }
    )
    oced.add_event(activity, "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))
    return oced


# ---------------------------------------------------------------------------
# is_timestamp
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s",
    ["2023-03-15T12:45:00Z", "2022-01-01", "2023-03-15 12:45:00"],
)
def test_is_timestamp_true_for_date_strings(s):
    assert is_timestamp(s) is True


@pytest.mark.parametrize("s", ["hello", "not_a_date", "activity_name", "some_value"])
def test_is_timestamp_false_for_plain_words(s):
    assert is_timestamp(s) is False


# ---------------------------------------------------------------------------
# clean_string
# ---------------------------------------------------------------------------


def test_clean_string_replaces_special_chars_with_underscore():
    assert clean_string("Hello World!", casefold=False) == "Hello_World_"


def test_clean_string_casefold_lowercases_result():
    assert clean_string("Hello World!", casefold=True) == "hello_world_"


def test_clean_string_preserves_alphanumeric_and_underscore():
    assert clean_string("abc_123", casefold=False) == "abc_123"


# ---------------------------------------------------------------------------
# clean_value
# ---------------------------------------------------------------------------


def test_clean_value_cleans_plain_string():
    assert clean_value("Hello World!") == "hello_world_"


def test_clean_value_leaves_timestamp_string_unchanged():
    ts = "2023-03-15T12:45:00Z"
    assert clean_value(ts) == ts


def test_clean_value_recursively_cleans_dict():
    result = clean_value({"Hello World!": "Test Value!"})
    assert result == {"hello_world_": "test_value_"}


def test_clean_value_cleans_dataframe_columns_and_values():
    df = pd.DataFrame({"Name!": ["Alice?", "Bob$"], "Age*": [25, 30]})
    cleaned = clean_value(df)
    assert "name_" in cleaned.columns
    assert "age_" in cleaned.columns
    assert cleaned["name_"].iloc[0] == "alice_"
    assert cleaned["age_"].iloc[0] == 25


def test_clean_value_passes_through_int():
    assert clean_value(42) == 42


def test_clean_value_passes_through_float():
    assert clean_value(3.14) == 3.14


def test_clean_value_passes_through_bool():
    assert clean_value(True) is True


def test_clean_value_cleans_list_elements():
    result = clean_value(["Hello!", "World?"])
    assert result == ["hello_", "world_"]


def test_clean_value_passes_through_timestamp_object():
    ts = pd.Timestamp("2022-01-01", tz="UTC")
    assert clean_value(ts) == ts


# ---------------------------------------------------------------------------
# clean_dataframe
# ---------------------------------------------------------------------------


def test_clean_dataframe_cleans_columns_and_string_values():
    df = pd.DataFrame(
        {"Column One!": ["Hello World!", "Goodbye?"], "Column Two$": [1, 2]}
    )
    cleaned = clean_dataframe(df)
    assert "column_one_" in cleaned.columns
    assert "column_two_" in cleaned.columns
    assert cleaned["column_one_"].iloc[0] == "hello_world_"
    assert cleaned["column_two_"].iloc[0] == 1


# ---------------------------------------------------------------------------
# apply_function_to_value
# ---------------------------------------------------------------------------


def test_apply_function_to_value_transforms_string():
    assert apply_function_to_value("hello", str.upper) == "HELLO"


def test_apply_function_to_value_skips_timestamp_string():
    ts = "2023-03-15T12:45:00Z"
    # Timestamp strings are left unchanged regardless of function
    assert apply_function_to_value(ts, str.upper) == ts


def test_apply_function_to_value_recurses_into_dict():
    result = apply_function_to_value(
        {"A": 1, "B": {"C": "hello", "D": 2}},
        lambda x: x.upper() if isinstance(x, str) else x,
    )
    assert result == {"A": 1, "B": {"C": "HELLO", "D": 2}}


def test_apply_function_to_value_transforms_dataframe_columns_and_values():
    df = pd.DataFrame({"col1": ["a", "b"], "col2": [1, 2]})
    result = apply_function_to_value(df, lambda x: str(x).upper())
    expected = pd.DataFrame({"COL1": ["A", "B"], "COL2": ["1", "2"]})
    pd.testing.assert_frame_equal(result, expected)


def test_apply_function_to_value_rename_df_columns_false_leaves_columns_untouched():
    df = pd.DataFrame({"col": ["a", "b"]})
    result = apply_function_to_value(df, str.upper, rename_df_columns=False)
    assert list(result.columns) == ["col"]
    assert result["col"].tolist() == ["A", "B"]


def test_apply_function_to_value_replace_df_values_false_leaves_values_untouched():
    df = pd.DataFrame({"col": ["a", "b"]})
    result = apply_function_to_value(df, str.upper, replace_df_values=False)
    assert list(result.columns) == ["COL"]
    assert result["COL"].tolist() == ["a", "b"]


def test_apply_function_to_value_transforms_list_elements():
    result = apply_function_to_value(["hello", "world"], str.upper)
    assert result == ["HELLO", "WORLD"]


def test_apply_function_to_value_passes_through_int():
    assert apply_function_to_value(42, str.upper) == 42


def test_apply_function_to_value_passes_through_float():
    assert apply_function_to_value(3.14, str.upper) == 3.14


def test_apply_function_to_value_passes_through_timestamp():
    ts = pd.Timestamp("2022-01-01", tz="UTC")
    assert apply_function_to_value(ts, str.upper) == ts


# ---------------------------------------------------------------------------
# clean_column_values
# ---------------------------------------------------------------------------


def test_clean_column_values_applies_function_to_column():
    df = pd.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
    result = clean_column_values(df.copy(), {"A": lambda x: x * 2})
    assert list(result["A"]) == [2, 4, 6]
    assert list(result["B"]) == ["x", "y", "z"]


def test_clean_column_values_raises_for_missing_column():
    df = pd.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(ValueError):
        clean_column_values(df.copy(), {"B": lambda x: x})


# ---------------------------------------------------------------------------
# null_check
# ---------------------------------------------------------------------------


def test_null_check_runs_without_error_on_clean_store():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["e1", "e2"],
            "ts": pd.date_range("2022-01-01", periods=2, tz="UTC"),
        }
    )
    oced.add_event("act", "eid", "ts", ev)
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))
    # Should not raise
    null_check(oced)


# ---------------------------------------------------------------------------
# fix_duplicate_names
# ---------------------------------------------------------------------------


def test_fix_duplicate_names_renames_reserved_relation():
    oced = OCEDStore()
    oced.add_object("A", "oid", pd.DataFrame({"oid": [1, 2]}))
    ev = pd.DataFrame(
        {"eid": ["e1", "e2"], "ts": pd.date_range("2022-01-01", periods=2, tz="UTC")}
    )
    oced.add_event("act", "eid", "ts", ev)
    # "df" is a reserved OWL property name
    oced.add_event_to_object_relation(
        "df", "eid", "oid", ["A"], pd.DataFrame({"eid": ["e1", "e2"], "oid": [1, 2]})
    )

    result = fix_duplicate_names(oced)

    assert "df_" in result.get_relation_types()
    assert "df" not in result.get_relation_types()


def test_fix_duplicate_names_renames_reserved_activity():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("observes", "eid", "ts", ev)  # reserved name
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))

    result = fix_duplicate_names(oced)

    assert "observes_" in result.get_activities()
    assert "observes" not in result.get_activities()


def test_fix_duplicate_names_renames_object_colliding_with_activity():
    # Object type shares a name with the activity — object gets the "_" suffix
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("my_activity", "eid", "ts", ev)
    oced.add_object(
        "my_activity", "oid", pd.DataFrame({"oid": ["o1"]})
    )  # same name as activity

    result = fix_duplicate_names(oced)

    assert "my_activity_" in result.get_object_types()
    assert "my_activity" not in result.get_object_types()


def test_fix_duplicate_names_renames_attribute_col_colliding_with_object_type():
    # Object's attribute column has the same name as the object type itself
    oced = OCEDStore()
    oced.add_object(
        "TypeA",
        "oid",
        pd.DataFrame({"oid": ["o1"], "TypeA": ["some_val"]}),
    )

    result = fix_duplicate_names(oced)

    obj_df = result.get_object("TypeA")["data"]
    assert "TypeA_attr" in obj_df.columns
    assert "TypeA" not in obj_df.columns


def test_fix_duplicate_names_renames_attribute_value_colliding_with_reserved():
    # An event attribute cell value equals a reserved OWL name → gets "_attr_val" suffix
    oced = _oced_with_event("activity", attr_col="status", attr_val="is_a")

    result = fix_duplicate_names(oced)

    status_vals = list(result.get_event("activity")["data"]["status"])
    assert "is_a_attr_val" in status_vals
    assert "normal_value" in status_vals


# ---------------------------------------------------------------------------
# clean_oced_data
# ---------------------------------------------------------------------------


def test_clean_oced_data_sanitizes_special_chars_in_all_strings():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {
            "eid": ["e1"],
            "ts": pd.date_range("2022-01-01", periods=1, tz="UTC"),
            "some attr!": ["val?"],
        }
    )
    oced.add_event("activity type!", "eid", "ts", ev)
    oced.add_object(
        "obj type!", "oid", pd.DataFrame({"oid": ["o1"], "data!": ["val!"]})
    )
    oced.add_event_to_object_relation(
        "rel name!",
        "eid",
        "oid",
        ["obj type!"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )

    result = clean_oced_data(oced)

    assert "activity_type_" in result.get_activities()
    assert "obj_type_" in result.get_object_types()
    assert "rel_name_" in result.get_relation_types()
    ev_data = result.get_event("activity_type_")["data"]
    assert "some_attr_" in ev_data.columns
    assert "val_" in ev_data["some_attr_"].values


def test_clean_oced_data_applies_fix_duplicate_names():
    oced = OCEDStore()
    ev = pd.DataFrame(
        {"eid": ["e1"], "ts": pd.date_range("2022-01-01", periods=1, tz="UTC")}
    )
    oced.add_event("df", "eid", "ts", ev)  # "df" is reserved
    oced.add_object("obj", "oid", pd.DataFrame({"oid": ["o1"]}))

    result = clean_oced_data(oced)

    assert "df_" in result.get_activities()
    assert "df" not in result.get_activities()
