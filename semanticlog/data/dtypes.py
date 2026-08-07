"""Best-effort dtype inference for freshly loaded event and object tables.

Kept separate from :mod:`semanticlog.data.cleaning` because
:class:`~semanticlog.data.oced.OCEDStore` applies it on ingest, and cleaning
imports the store.
"""

import logging

import numpy as np
import pandas as pd

DEFAULT_MIN_PARSE_RATIO = 0.95


def _try_datetime(
    series: pd.Series, non_null: int, min_ratio: float
) -> pd.Series | None:
    dt = pd.to_datetime(series, errors="coerce", utc=True, format="ISO8601")
    dt = dt.dt.floor("s")
    return dt if (dt.notna().sum() / non_null) >= min_ratio else None


def _try_numeric(
    series: pd.Series, non_null: int, min_ratio: float
) -> pd.Series | None:
    num = pd.to_numeric(series, errors="coerce")
    if (num.notna().sum() / non_null) < min_ratio:
        return None
    vals = set(num.dropna().unique())
    if vals <= {0, 1}:
        return num.astype("boolean")
    if (num.dropna() % 1 == 0).all():
        return num.astype("Int64")
    return num.astype("float64")


def infer_dtypes(
    df: pd.DataFrame,
    ignore_cols: list[str] | None = None,
    min_parse_ratio: float = DEFAULT_MIN_PARSE_RATIO,
    label: str = "table",
) -> pd.DataFrame:
    """Return a copy of ``df`` with all-empty rows/columns dropped and dtypes inferred.

    A string column is converted when at least ``min_parse_ratio`` of its non-null
    values parse as ISO 8601 timestamps or as numbers; values that fail to parse
    become null.  Columns listed in ``ignore_cols`` (typically identifier columns)
    keep their original dtype.

    Args:
        df: Table to convert.
        ignore_cols: Columns to leave untouched.
        min_parse_ratio: Fraction of non-null values that must parse for a conversion.
        label: Name used when logging what was dropped.

    Returns:
        A new DataFrame; the input is never modified.
    """
    if ignore_cols is None:
        ignore_cols = []
    df = df.copy()  # ensure we write to the real frame, not a view

    n_rows_before, cols_before = len(df), set(df.columns)
    df = df.dropna(how="all").dropna(axis=1, how="all")
    _log_drops(label, n_rows_before - len(df), cols_before - set(df.columns))

    # normalize empties for object-like cols
    obj_cols = df.select_dtypes(include="object").columns
    df[obj_cols] = df[obj_cols].replace(r"^\s*$", np.nan, regex=True)

    for col in df.columns:
        if col in ignore_cols:
            continue

        non_null = df[col].notna().sum()
        if non_null == 0:
            continue

        # already numeric/bool? skip
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(
            df[col]
        ):
            continue

        dt = _try_datetime(df[col], non_null, min_parse_ratio)
        if dt is not None:
            df[col] = dt
            continue

        num = _try_numeric(df[col], non_null, min_parse_ratio)
        if num is not None:
            df[col] = num
    return df


def _log_drops(label: str, n_rows: int, dropped_cols: set) -> None:
    if n_rows:
        logging.info(f"infer_dtypes[{label}]: dropped {n_rows} all-empty row(s).")
    if dropped_cols:
        logging.info(
            f"infer_dtypes[{label}]: dropped {len(dropped_cols)} all-empty column(s): "
            f"{sorted(str(c) for c in dropped_cols)}."
        )
