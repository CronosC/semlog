"""The step between a table and the pairs a sink asserts.

A sink takes pairs of names. The tables they are read from carry gaps and
repeats, and what to do about those is a decision about the build rather than
about the axiom boundary, so it is made here.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import pandas as pd

Pair = tuple[Any, Any]


def clean_pairs(df: pd.DataFrame, from_key: str, to_key: str) -> Iterator[Pair]:
    """Yield the pairs of two columns, without gaps and without repeats.

    A row with a missing endpoint carries no name to assert, and a repeated
    pair would be the same axiom twice. The result is a generator, because a
    relation of a large log holds one pair per row and a backend writing them
    out needs to see only one at a time.

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
