"""Giving loaded columns the dtype their content implies."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from semanticlog.data.dtypes import infer_dtypes


def test_infer_dtypes_passes_through_numeric_and_bool():
    df = pd.DataFrame(
        {"ints": [1, 2, 3], "floats": [1.1, 2.2, 3.3], "bools": [True, False, True]}
    )
    before = df.copy(deep=True)
    out = infer_dtypes(df)
    assert_series_equal(out["ints"], before["ints"], check_dtype=True)
    assert_series_equal(out["floats"], before["floats"], check_dtype=True)
    assert_series_equal(out["bools"], before["bools"], check_dtype=True)


def test_infer_dtypes_normalizes_datetime_to_utc_without_truncating():
    ts = pd.to_datetime(
        [
            "2022-01-01 12:00:00.987654+00:00",
            "2022-01-01 13:00:01.123456+01:00",
        ],
        utc=True,
    )
    out = infer_dtypes(pd.DataFrame({"dt": ts}), min_parse_ratio=1.0)
    assert pd.api.types.is_datetime64_any_dtype(out["dt"])
    expected = pd.to_datetime(
        ["2022-01-01 12:00:00.987654+00:00", "2022-01-01 12:00:01.123456+00:00"],
        utc=True,
    )
    assert_series_equal(out["dt"], pd.Series(expected, name="dt"), check_dtype=False)


def test_infer_dtypes_parses_datetime_strings_when_ratio_met():
    df = pd.DataFrame(
        {"maybe_dt": ["2022-01-01 08:00:00", "2022-01-01 09:00:05", None, "garbage"]}
    )
    out = infer_dtypes(df, min_parse_ratio=0.66)
    assert pd.api.types.is_datetime64_any_dtype(out["maybe_dt"])


def test_infer_dtypes_parses_numeric_strings_when_ratio_met():
    df = pd.DataFrame({"maybe_num": ["10.1", "11.7", None, "x"]})
    out = infer_dtypes(df, min_parse_ratio=0.66)
    assert pd.api.types.is_numeric_dtype(out["maybe_num"])
    assert out["maybe_num"].dropna().tolist() == pytest.approx([10.1, 11.7])


def test_infer_dtypes_does_not_convert_when_ratio_too_low():
    df = pd.DataFrame({"weak_dt": ["2022-01-01 00:00:00", "bad", "worse", None]})
    out = infer_dtypes(df, min_parse_ratio=0.5)
    assert not pd.api.types.is_datetime64_any_dtype(out["weak_dt"])


def test_infer_dtypes_respects_ignore_cols():
    df = pd.DataFrame(
        {
            "ignored": ["2022-01-01 12:00:00", "2022-01-01 12:00:01"],
            "convert_me": ["2022-01-01 12:00:00.123", "2022-01-01 12:00:01.999"],
        }
    )
    out = infer_dtypes(df, ignore_cols=["ignored"])
    assert not pd.api.types.is_datetime64_any_dtype(out["ignored"])
    assert pd.api.types.is_datetime64_any_dtype(out["convert_me"])


def test_infer_dtypes_drops_all_na_rows_and_columns():
    df = pd.DataFrame({"all_na_col": [np.nan, np.nan, np.nan], "mix": [1, np.nan, 3]})
    df.loc[1, "mix"] = np.nan
    out = infer_dtypes(df)
    assert "all_na_col" not in out.columns
    assert len(out) == 2
    assert out["mix"].tolist() == [1, 3]


def test_infer_dtypes_leaves_zero_one_strings_numeric():
    df = pd.DataFrame({"flag": ["0", "1", "0", "1"]})
    out = infer_dtypes(df)
    assert str(out["flag"].dtype) == "Int64"


def test_infer_dtypes_coerces_whole_number_strings_to_int64():
    df = pd.DataFrame({"count": ["1", "2", "3", "10"]})
    out = infer_dtypes(df)
    assert str(out["count"].dtype) == "Int64"


def test_sub_second_timestamps_keep_their_precision():
    df = pd.DataFrame(
        {
            "eid": ["e1", "e2"],
            "ts": ["2024-01-01T10:00:00.123456Z", "2024-01-01T10:00:00.789012Z"],
        }
    )

    out = infer_dtypes(df, ignore_cols=["eid"])

    assert out["ts"].nunique() == 2
    assert out["ts"].dt.microsecond.tolist() == [123456, 789012]


def test_a_numeric_column_of_zeros_and_ones_stays_numeric():
    out = infer_dtypes(
        pd.DataFrame({"oid": ["o1", "o2", "o3"], "count": ["0", "1", "1"]}),
        ignore_cols=["oid"],
    )

    assert str(out["count"].dtype) == "Int64"
    assert out["count"].tolist() == [0, 1, 1]


@pytest.mark.parametrize(
    "values",
    [["true", "false"], ["True", "False"], ["yes", "no"], ["TRUE", "false"]],
)
def test_boolean_like_strings_still_become_boolean(values):
    out = infer_dtypes(
        pd.DataFrame({"oid": ["o1", "o2"], "flag": values}), ignore_cols=["oid"]
    )

    assert str(out["flag"].dtype) == "boolean"
    assert out["flag"].tolist() == [True, False]


def test_an_integer_column_with_a_null_stays_an_integer():
    out = infer_dtypes(
        pd.DataFrame({"oid": ["o1", "o2"], "qty": ["3", None]}), ignore_cols=["oid"]
    )

    assert str(out["qty"].dtype) == "Int64"
