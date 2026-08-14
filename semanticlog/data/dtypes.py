"""Give loaded columns the dtype their content implies.

Source logs carry most attributes as text, and the ontology needs to know
which of them are really timestamps, booleans or numbers, because that decides
the literal type and whether a column can be binned. Each column is converted
only when nearly all of its values parse, so one stray entry does not decide
the type of the column.
"""

import numpy as np
import pandas as pd

DEFAULT_MIN_PARSE_RATIO = 0.95


_BOOLEAN_STRINGS = frozenset({"true", "false", "yes", "no", "t", "f", "y", "n"})


def _try_datetime(
    series: pd.Series, non_null: int, min_ratio: float
) -> pd.Series | None:
    """Return the column as timestamps, or None if too few values parse."""
    dt = pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
    return dt if (dt.notna().sum() / non_null) >= min_ratio else None


def _try_boolean(
    series: pd.Series, non_null: int, min_ratio: float
) -> pd.Series | None:
    """Return the column as booleans, or None if it holds anything else."""
    if not pd.api.types.is_object_dtype(series) and not isinstance(
        series.dtype, pd.StringDtype
    ):
        return None
    lowered = series.dropna().astype(str).str.strip().str.casefold()
    if lowered.empty or not set(lowered) <= _BOOLEAN_STRINGS:
        return None
    if (len(lowered) / non_null) < min_ratio:
        return None
    truthy = {"true", "yes", "t", "y"}
    return (
        series.astype(str).str.strip().str.casefold().isin(truthy).astype("boolean")
    ).where(series.notna())


def _try_numeric(
    series: pd.Series, non_null: int, min_ratio: float
) -> pd.Series | None:
    """Return the column as integers or floats, or None if too few values parse."""
    num = pd.to_numeric(series, errors="coerce")
    if (num.notna().sum() / non_null) < min_ratio:
        return None
    if (num.dropna() % 1 == 0).all():
        return num.astype("Int64")
    return num.astype("float64")


def infer_dtypes(
    df: pd.DataFrame,
    ignore_cols: list[str] | None = None,
    min_parse_ratio: float = DEFAULT_MIN_PARSE_RATIO,
) -> pd.DataFrame:
    """Return a copy of the table with inferred dtypes.

    Rows and columns that are entirely empty are dropped, and blank text is
    treated as missing. Each remaining column is tried as a timestamp, then as
    a boolean, then as a number, and keeps its original text form if none of
    those fit.

    Args:
        df: Table to convert.
        ignore_cols: Columns to leave alone, such as id columns, where a
            numeric-looking id should stay text.
        min_parse_ratio: Share of non-empty values that must parse before a
            conversion is accepted.

    Returns:
        A new table. The input is not modified.
    """
    if ignore_cols is None:
        ignore_cols = []
    df = df.copy()

    df = df.dropna(how="all").dropna(axis=1, how="all")

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
