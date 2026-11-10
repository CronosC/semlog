"""Rewriting one name, Series, value or frame into what an IRI can hold."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from semlog.core.vocabulary import is_safe_local_name
from semlog.data.cleaning.names import (
    clean_dataframe,
    clean_series,
    clean_string,
    clean_value,
    free_name,
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Hello World!", "Hello_World_"),
        (".only.", "_only_"),
        ("a-b.c", "a-b.c"),
        ("", "_"),
    ],
)
def test_unsafe_characters_and_edge_punctuation_become_underscores(source, expected):
    assert clean_string(source) == expected
    assert is_safe_local_name(expected) is True


def test_a_series_is_cleaned_as_each_of_its_strings_would_be():
    values = [".hidden", "v2.1.", "-flag", "a b", ""]

    assert list(clean_series(pd.Series(values))) == [clean_string(v) for v in values]


def test_every_string_inside_a_nested_value_is_cleaned_and_the_rest_kept():
    stamp = pd.Timestamp("2022-01-01", tz="UTC")

    cleaned = clean_value({"k k": ["a b", 3.5, True, stamp]})

    assert cleaned == {"k_k": ["a_b", 3.5, True, stamp]}


def test_a_frame_has_its_column_names_and_text_cells_cleaned():
    df = pd.DataFrame({1: ["a b"], "Age*": [25], "when": ["2023-03-15 12:45:00"]})

    assert_frame_equal(
        clean_value(df),
        pd.DataFrame({"1": ["a_b"], "Age_": [25], "when": ["2023-03-15_12_45_00"]}),
    )


def test_a_kept_column_keeps_its_name_and_has_its_cells_cleaned():
    df = pd.DataFrame({"ocel:type": ["order item"], "sta tus": ["pen ding"]})

    assert_frame_equal(
        clean_dataframe(df, keep=["ocel:type"]),
        pd.DataFrame({"ocel:type": ["order_item"], "sta_tus": ["pen_ding"]}),
    )


def test_a_free_name_is_the_name_then_the_prefixed_one_then_a_counter():
    assert [
        free_name("x", set(), "id_"),
        free_name("x", {"x"}, "id_"),
        free_name("x", {"x", "id_x"}, "id_"),
        free_name("x", {"x"}),
    ] == ["x", "id_x", "id_x_2", "x_2"]
