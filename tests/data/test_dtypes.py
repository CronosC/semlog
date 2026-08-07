"""Tests for :mod:`semanticlog.data.dtypes`."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from semanticlog.data.dtypes import infer_dtypes

# ---------------------------------------------------------------------------
# infer_dtypes
# ---------------------------------------------------------------------------


def test_infer_dtypes_passes_through_numeric_and_bool():
    df = pd.DataFrame(
        {"ints": [1, 2, 3], "floats": [1.1, 2.2, 3.3], "bools": [True, False, True]}
    )
    before = df.copy(deep=True)
    out = infer_dtypes(df)
    assert_series_equal(out["ints"], before["ints"], check_dtype=True)
    assert_series_equal(out["floats"], before["floats"], check_dtype=True)
    assert_series_equal(out["bools"], before["bools"], check_dtype=True)


def test_infer_dtypes_normalizes_datetime_to_seconds_floored():
    ts = pd.to_datetime(
        [
            "2022-01-01 12:00:00.987654321+00:00",
            "2022-01-01 12:00:01.123456789+00:00",
        ],
        utc=True,
    )
    out = infer_dtypes(pd.DataFrame({"dt": ts}), min_parse_ratio=1.0)
    assert pd.api.types.is_datetime64_any_dtype(out["dt"])
    expected = pd.to_datetime(["2022-01-01 12:00:00", "2022-01-01 12:00:01"], utc=True)
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


def test_infer_dtypes_coerces_zero_one_strings_to_boolean():
    df = pd.DataFrame({"flag": ["0", "1", "0", "1"]})
    out = infer_dtypes(df)
    assert str(out["flag"].dtype) == "boolean"


def test_infer_dtypes_coerces_whole_number_strings_to_int64():
    df = pd.DataFrame({"count": ["1", "2", "3", "10"]})
    out = infer_dtypes(df)
    assert str(out["count"].dtype) == "Int64"
