"""Infer the dtype of loaded columns from their content.

Source logs carry most attributes as text. The dtype decides the literal type
and whether a column can be binned. A column is converted when at least
`DEFAULT_MIN_PARSE_RATIO` of its values parse.
"""

import re

import numpy as np
import pandas as pd

DEFAULT_MIN_PARSE_RATIO = 0.95


_BOOLEAN_STRINGS = frozenset({"true", "false", "yes", "no", "t", "f", "y", "n"})

_DIGITS = re.compile(r"\s*[+-]?\d+\s*")


def _try_datetime(
    series: pd.Series, non_null: int, min_ratio: float
) -> pd.Series | None:
    """Return the column as timestamps, or None if too few values parse."""
    dt = pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
    return dt if (dt.notna().sum() / non_null) >= min_ratio else None


def _try_boolean(
    series: pd.Series, non_null: int, min_ratio: float
) -> pd.Series | None:
    """Return the column as booleans, or None if too few values parse."""
    if not pd.api.types.is_object_dtype(series) and not isinstance(
        series.dtype, pd.StringDtype
    ):
        return None
    lowered = series.astype(str).str.strip().str.casefold().where(series.notna())
    parsed = lowered.isin(_BOOLEAN_STRINGS)
    if (parsed.sum() / non_null) < min_ratio:
        return None
    truthy = {"true", "yes", "t", "y"}
    return lowered.isin(truthy).astype("boolean").where(parsed)


def _try_numeric(
    series: pd.Series, non_null: int, min_ratio: float
) -> pd.Series | None:
    """Return the column as Int64 or float64, or None if too few values parse."""
    num = pd.to_numeric(series, errors="coerce")
    if (num.notna().sum() / non_null) < min_ratio:
        return None
    if (num.dropna() % 1 == 0).all():
        try:
            return num.astype("Int64")
        except (TypeError, OverflowError):
            pass
    return num.astype("float64")


def _all_digits(series: pd.Series) -> bool:
    """Return True when every non-null value of a text column is an integer literal."""
    if not pd.api.types.is_object_dtype(series) and not isinstance(
        series.dtype, pd.StringDtype
    ):
        return False
    values = series.dropna()
    return bool(
        values.map(
            lambda v: isinstance(v, str) and _DIGITS.fullmatch(v) is not None
        ).all()
    )


def infer_dtypes(
    df: pd.DataFrame,
    ignore_cols: list[str] | None = None,
    min_parse_ratio: float = DEFAULT_MIN_PARSE_RATIO,
) -> pd.DataFrame:
    """Return a copy of the table with inferred dtypes.

    All-empty rows are dropped, and all-empty columns not in `ignore_cols`.
    Blank text becomes missing. A column becomes a timestamp, boolean or
    number when enough of its values parse, and a column of integer literals
    becomes a number.

    Args:
        ignore_cols: Columns kept as they are, such as id columns.
        min_parse_ratio: Share of non-empty values that must parse before a
            conversion is accepted.
    """
    if ignore_cols is None:
        ignore_cols = []
    df = df.copy()

    df = df.dropna(how="all")
    empty = [c for c in df.columns if c not in ignore_cols and df[c].isna().all()]
    df = df.drop(columns=empty)

    text_cols = df.select_dtypes(include=["object", "str"]).columns
    df[text_cols] = df[text_cols].replace(r"^\s*$", np.nan, regex=True)

    for col in df.columns:
        if col in ignore_cols:
            continue

        non_null = df[col].notna().sum()
        if non_null == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(
            df[col]
        ):
            continue

        if _all_digits(df[col]):
            num = _try_numeric(df[col], non_null, min_parse_ratio)
            if num is not None:
                df[col] = num
                continue

        dt = _try_datetime(df[col], non_null, min_parse_ratio)
        if dt is not None:
            df[col] = dt
            continue

        boolean = _try_boolean(df[col], non_null, min_parse_ratio)
        if boolean is not None:
            df[col] = boolean
            continue

        num = _try_numeric(df[col], non_null, min_parse_ratio)
        if num is not None:
            df[col] = num
    return df
