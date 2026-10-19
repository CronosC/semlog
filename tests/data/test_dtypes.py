"""Giving loaded columns the dtype their content implies."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from semanticlog.data.dtypes import cast_declared, infer_dtypes


def test_numeric_and_boolean_columns_pass_through_unchanged():
    df = pd.DataFrame({"ints": [1, 2], "floats": [1.1, 2.2], "bools": [True, False]})

    assert_frame_equal(infer_dtypes(df), df)


def test_iso_timestamps_become_utc_timestamps_to_the_microsecond():
    df = pd.DataFrame(
        {"ts": ["2024-01-01T10:00:00.123456Z", "2024-01-01T11:00:00.789012+01:00"]}
    )

    out = infer_dtypes(df)

    assert list(out["ts"]) == list(
        pd.to_datetime(
            ["2024-01-01T10:00:00.123456Z", "2024-01-01T10:00:00.789012Z"], utc=True
        )
    )


def test_a_column_is_converted_when_enough_of_its_values_parse():
    df = pd.DataFrame({"amount": ["10.1", "11.7", None, "x"]})

    out = infer_dtypes(df, min_parse_ratio=0.66)

    assert out["amount"].tolist()[:2] == pytest.approx([10.1, 11.7])
    assert str(out["amount"].dtype) == "float64"


def test_a_column_is_left_as_text_when_too_few_of_its_values_parse():
    df = pd.DataFrame({"when": ["2022-01-01 00:00:00", "bad", "worse", None]})

    out = infer_dtypes(df, min_parse_ratio=0.5)

    assert out["when"].tolist()[:3] == ["2022-01-01 00:00:00", "bad", "worse"]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["2019", "2020"], [2019, 2020]),
        (["3", None], [3, pd.NA]),
    ],
    ids=["years", "with_a_gap"],
)
def test_a_column_of_integer_literals_becomes_integers(values, expected):
    out = infer_dtypes(pd.DataFrame({"oid": ["o1", "o2"], "n": values}), ["oid"])

    assert str(out["n"].dtype) == "Int64"
    assert out["n"].tolist() == expected


def test_whole_numbers_beyond_int64_stay_float():
    out = infer_dtypes(pd.DataFrame({"big": ["1e20", "2e20", "3"]}))

    assert str(out["big"].dtype) == "float64"


def test_boolean_words_in_any_case_become_booleans_and_a_stray_word_missing():
    df = pd.DataFrame({"flag": ["TRUE", "no", "Yes", "f"] * 5 + ["maybe"]})

    out = infer_dtypes(df)

    assert str(out["flag"].dtype) == "boolean"
    assert out["flag"].tolist() == [True, False, True, False] * 5 + [pd.NA]


def test_a_column_of_mostly_other_words_stays_text():
    out = infer_dtypes(pd.DataFrame({"flag": ["yes", "no", "maybe"]}))

    assert out["flag"].tolist() == ["yes", "no", "maybe"]


def test_empty_rows_and_columns_are_dropped_and_blank_text_is_missing():
    df = pd.DataFrame(
        {"empty": [None, None, None], "x": ["a", None, "  "], "y": [1, None, 3]}
    )

    out = infer_dtypes(df)

    assert list(out.columns) == ["x", "y"]
    assert out["x"].isna().tolist() == [False, True]


def test_an_ignored_column_is_kept_as_it_is_even_when_empty():
    df = pd.DataFrame({"id": [None, None], "ts": ["2022-01-01", "2022-01-02"]})

    out = infer_dtypes(df, ignore_cols=["id", "ts"])

    assert_frame_equal(out, df)


def test_a_declared_string_of_digits_stays_text():
    out = infer_dtypes(pd.DataFrame({"code": ["1", "2"]}), declared={"code": "string"})

    assert out["code"].tolist() == ["1", "2"]


@pytest.mark.parametrize(
    ("declared", "values", "expected"),
    [
        ("integer", ["1", "2.5"], [1, None]),
        ("float", ["1.5", "abc"], [1.5, None]),
        ("boolean", ["yes", "maybe"], [True, None]),
    ],
)
def test_a_value_that_is_not_its_declared_type_is_missing_and_warned_about(
    declared, values, expected, caplog
):
    out = cast_declared(pd.DataFrame({"x": values}), {"x": declared})

    assert [None if pd.isna(v) else v for v in out["x"]] == expected
    assert f"declared {declared}, but 1 of its value(s)" in caplog.text
