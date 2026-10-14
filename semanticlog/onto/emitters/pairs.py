"""Pairs of names read from a table, for a sink to assert.

A table can hold missing endpoints and repeated rows. Both are dropped here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import pandas as pd

Pair = tuple[Any, Any]


def clean_pairs(df: pd.DataFrame, from_key: str, to_key: str) -> Iterator[Pair]:
    """Yield the distinct pairs of two columns, skipping rows with a missing endpoint.

    Args:
        df: Table holding the pairs.
        from_key: Column holding the subject.
        to_key: Column holding the target, or the literal value.
    """
    rows = df[[from_key, to_key]]
    rows = rows[rows[from_key].notna() & rows[to_key].notna()]
    rows = rows.drop_duplicates([from_key, to_key])
    return rows.itertuples(index=False, name=None)


def reversed_pairs(pairs: Iterable[Pair]) -> Iterator[Pair]:
    """Swap each pair, for asserting the inverse of a property."""
    return ((target, subject) for subject, target in pairs)
