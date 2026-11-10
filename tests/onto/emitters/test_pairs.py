"""Pairs of names read from a table, for a sink to assert."""

import pandas as pd

from semlog.onto.emitters.pairs import clean_pairs, reversed_pairs


def test_a_row_with_a_missing_endpoint_yields_no_pair():
    df = pd.DataFrame({"s": ["e1", None, "e3"], "t": ["o1", "o2", None]})

    assert list(clean_pairs(df, "s", "t")) == [("e1", "o1")]


def test_a_repeated_pair_is_yielded_once():
    df = pd.DataFrame(
        {"s": ["e1", "e1", "e1"], "t": ["o1", "o1", "o2"], "x": [1, 2, 3]}
    )

    assert list(clean_pairs(df, "s", "t")) == [("e1", "o1"), ("e1", "o2")]


def test_reversing_swaps_subject_and_target():
    assert list(reversed_pairs([("e1", "o1"), ("e2", "o1")])) == [
        ("o1", "e1"),
        ("o1", "e2"),
    ]
