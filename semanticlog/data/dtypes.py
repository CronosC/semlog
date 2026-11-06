"""Infer the dtype of loaded columns from their content, or cast them to a declared type.

Source logs carry most attributes as text. The dtype decides the literal type
and whether a column can be binned. A column with a declared type is cast to
it. Any other column is converted when at least `DEFAULT_MIN_PARSE_RATIO` of
its values parse.
"""

import logging
import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MIN_PARSE_RATIO = 0.95

DECLARED_TYPES: tuple[str, ...] = ("string", "integer", "float", "boolean", "time")

_SAMPLE = 3


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
    parsed = _parse_boolean(series)
    if (parsed.notna().sum() / non_null) < min_ratio:
        return None
    return parsed


def _parse_boolean(series: pd.Series) -> pd.Series:
    """Return the text column as booleans, missing where a value is not a boolean word."""
    lowered = series.astype(str).str.strip().str.casefold().where(series.notna())
    truthy = {"true", "yes", "t", "y"}
    return lowered.isin(truthy).astype("boolean").where(lowered.isin(_BOOLEAN_STRINGS))


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


def _as_declared(series: pd.Series, declared: str) -> pd.Series:
    """Return the column cast to a declared type, with missing values where a value does not parse."""
    if declared == "string":
        if pd.api.types.is_object_dtype(series) or isinstance(
            series.dtype, pd.StringDtype
        ):
            return series
        return series.astype(object).where(series.isna(), series.astype(str))
    if declared == "time":
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        return pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
    if declared == "boolean":
        if pd.api.types.is_bool_dtype(series):
            return series.astype("boolean")
        return _parse_boolean(series)
    num = pd.to_numeric(series, errors="coerce")
    if declared == "integer":
        return num.where(num % 1 == 0).astype("Int64")
    return num.astype("float64")


def _warn_lost(col: str, before: pd.Series, after: pd.Series, typed: str) -> None:
    """Warn when a value of `before` is missing in its conversion `after`.

    Args:
        typed: How the column got its type, such as `declared integer`.
    """
    failed = before.notna() & after.isna()
    if not failed.any():
        return
    sample = list(before[failed].astype(str).unique()[:_SAMPLE])
    logger.warning(
        "Attribute '%s' is %s, but %d of its value(s), such as %s, are not. "
        "They are read as missing.",
        col,
        typed,
        int(failed.sum()),
        sample,
    )


def _inferred(col: str, before: pd.Series, after: pd.Series) -> pd.Series:
    """Return `after`, the inferred conversion of `before`, and warn about lost values."""
    if pd.api.types.is_datetime64_any_dtype(after):
        kind = "time"
    elif pd.api.types.is_bool_dtype(after):
        kind = "boolean"
    elif pd.api.types.is_integer_dtype(after):
        kind = "integer"
    else:
        kind = "float"
    _warn_lost(col, before, after, f"read as {kind}")
    return after


def cast_declared(df: pd.DataFrame, declared: Mapping[str, str]) -> pd.DataFrame:
    """Return a copy of the table with each declared column cast to its type.

    A value that does not parse as its declared type becomes missing, and a
    warning counts them per column. Columns the table does not have are
    ignored.

    Args:
        declared: Column name to one of `DECLARED_TYPES`.
    """
    df = df.copy()
    for col, declared_type in declared.items():
        if col not in df.columns:
            continue
        before = df[col]
        after = _as_declared(before, declared_type)
        _warn_lost(col, before, after, f"declared {declared_type}")
        df[col] = after
    return df


def infer_dtypes(
    df: pd.DataFrame,
    ignore_cols: list[str] | None = None,
    min_parse_ratio: float = DEFAULT_MIN_PARSE_RATIO,
    declared: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return a copy of the table with declared or inferred dtypes.

    All-empty rows are dropped, and all-empty columns not in `ignore_cols`.
    Blank text becomes missing. A declared column is cast to its type, see
    `cast_declared`. Any other column becomes a timestamp, boolean or number
    when enough of its values parse, and a column of integer literals becomes
    a number. A value that does not parse in a converted column becomes
    missing, and a warning counts them per column.

    Args:
        ignore_cols: Columns kept as they are, such as id columns.
        min_parse_ratio: Share of non-empty values that must parse before a
            conversion is accepted.
        declared: Column name to one of `DECLARED_TYPES`.
    """
    declared = {
        col: kind
        for col, kind in (declared or {}).items()
        if col not in (ignore_cols or [])
    }
    if ignore_cols is None:
        ignore_cols = []
    df = df.copy()

    df = df.dropna(how="all")
    empty = [c for c in df.columns if c not in ignore_cols and df[c].isna().all()]
    df = df.drop(columns=empty)

    text_cols = df.select_dtypes(include=["object", "str"]).columns
    df[text_cols] = df[text_cols].replace(r"^\s*$", np.nan, regex=True)
    df = cast_declared(df, declared)

    for col in df.columns:
        if col in ignore_cols or col in declared:
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
            df[col] = _inferred(col, df[col], dt)
            continue

        boolean = _try_boolean(df[col], non_null, min_parse_ratio)
        if boolean is not None:
            df[col] = _inferred(col, df[col], boolean)
            continue

        num = _try_numeric(df[col], non_null, min_parse_ratio)
        if num is not None:
            df[col] = _inferred(col, df[col], num)
    return df
