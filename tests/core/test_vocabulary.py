"""The names a build emits, and the reserved set derived from them."""

import numpy as np
import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.errors import ConfigError, InputError


@pytest.mark.parametrize(
    "iri",
    [
        "https://x.org/a b/",
        "https://x.org/a\tb/",
        'https://x.org/a"b/',
        "https://x.org/{a}/",
        "https://x.org/a|b/",
        "https://x.org/a\\b/",
        "https://x.org/a^b/",
        "https://x.org/a`b/",
        "https://x.org/<a>/",
        "https://x.org/a\x01b/",
    ],
)
def test_an_iri_with_a_character_an_iri_cannot_hold_is_rejected(iri):
    with pytest.raises(
        ConfigError, match="--iri: .* cannot be written into an ontology"
    ):
        vocab.check_iri(iri, "--iri")


def test_an_empty_iri_is_rejected():
    with pytest.raises(ConfigError, match="^iri: an empty IRI cannot be written"):
        vocab.check_iri("", "iri")


def test_an_iri_of_legal_characters_is_accepted():
    vocab.check_iri("https://example.org/ök/log-1_~%20/", "iri")


def test_a_log_iri_ending_in_a_hash_is_rejected():
    with pytest.raises(ConfigError, match="^iri: must end in '/'"):
        vocab.check_log_iri("https://example.org/log#", "iri")


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1e20, "100000000000000000000"), (np.float64(1e-7), "0.0000001")],
)
def test_a_number_is_written_as_plain_digits(value, expected):
    assert vocab.format_number(value) == expected


@pytest.mark.parametrize(
    ("template", "activities", "expected"),
    [
        ("Response", ["a", "b"], "Response__a__b__"),
        ("Exclusive Choice", ["a", "b"], "ExclusiveChoice__a__b__"),
        ("Init", ["a"], "Init__a__"),
    ],
)
def test_a_declare_constraint_name_joins_its_template_and_activities(
    template, activities, expected
):
    assert vocab.declare_constraint(template, activities) == expected


def test_a_number_that_is_not_finite_is_rejected():
    with pytest.raises(InputError, match="inf cannot be written as a number"):
        vocab.format_number(float("inf"))


@pytest.mark.parametrize(
    ("attribute", "value", "expected"),
    [
        ("status", "shipped", "status.shipped"),
        ("amount", 5.0, "amount.5"),
        ("amount", 2.5, "amount.2.5"),
    ],
)
def test_a_value_individual_is_named_by_its_attribute_and_value(
    attribute, value, expected
):
    assert vocab.attribute_value(attribute, value) == expected


def test_an_infinite_value_is_rejected_naming_its_attribute():
    with pytest.raises(InputError, match="Attribute 'amount': inf cannot be written"):
        vocab.attribute_value("amount", float("inf"))


def test_a_missing_value_is_rejected_naming_its_attribute():
    with pytest.raises(InputError, match="Attribute 'amount'.*missing value"):
        vocab.attribute_value("amount", pd.NaT)


@pytest.mark.parametrize("utc", [True, False])
def test_a_timestamp_value_is_named_to_the_microsecond(utc):
    stamp = pd.Series(pd.to_datetime(["2022-01-01 10:30:00"], utc=utc)).iloc[0]

    assert vocab.attribute_value("ts", stamp) == "ts.20220101T103000_000000"


def test_one_instant_under_two_offsets_names_one_timestamp():
    utc = pd.Timestamp("2024-01-01T10:00:00+00:00")
    same_instant = pd.Timestamp("2024-01-01T15:00:00+05:00")

    assert vocab.timestamp_id(utc) == vocab.timestamp_id(same_instant)


def test_a_column_of_timestamps_is_named_as_each_one_would_be():
    zoned = pd.Series(pd.to_datetime(["2024-01-01T10:00:00+05:00"]))
    text = pd.Series(["2022-01-01 10:30"])

    assert list(vocab.timestamp_ids(zoned)) == [vocab.timestamp_id(zoned.iloc[0])]
    assert list(vocab.timestamp_ids(text)) == ["2022_01_01_10_30"]


@pytest.mark.parametrize(
    ("name", "safe"),
    [
        (".hidden", False),
        ("v2.1.", False),
        ("v2.1", True),
    ],
)
def test_punctuation_is_safe_inside_a_name_but_not_at_its_edges(name, safe):
    assert vocab.is_safe_local_name(name) is safe


def test_an_inverse_takes_its_paired_name_or_the_inv_prefix():
    assert vocab.inverse(vocab.HAS_ATTRIBUTE) == vocab.ATTRIBUTE_OF
    assert vocab.inverse(vocab.DF) == "inv_df"
    assert vocab.inverse("df_order") == "inv_df_order"
    assert vocab.inverse("a", {"a": "b"}) == "b"


def test_an_inverse_stays_in_the_namespace_of_its_property():
    assert vocab.is_tbox(vocab.inverse(vocab.DF))
    assert not vocab.is_tbox(vocab.inverse("has_attribute"))
    assert vocab.inverse("has_attribute") == "inv_has_attribute"


def test_change_records_of_one_object_attribute_and_instant_are_numbered_from_two():
    ids = vocab.attribute_change_ids(
        pd.Series(["o1", "o1", "o2"]), "price", pd.Series(["T1", "T1", "T1"])
    )

    assert list(ids) == ["price.o1.T1", "price.o1.T1_2", "price.o2.T1"]


def test_change_records_whose_parts_join_alike_get_two_names():
    """Object `o` with `x_y` and object `o_x` with `y` both joined to `o_x_y`."""
    stamp = pd.Series(["20240102T000000_000000"])

    first = vocab.attribute_change_ids(pd.Series(["o"]), "x_y", stamp)
    second = vocab.attribute_change_ids(pd.Series(["o_x"]), "y", stamp)

    assert first[0] != second[0]


def test_every_schema_name_builder_is_registered_for_the_reserved_set():
    """Every `*_class` and `*_property` builder is in one of the tuples `derived_from_*` reads."""
    registered = {
        build
        for group in (
            vocab.CASE_CONCEPT_CLASSES,
            vocab.CASE_CONCEPT_PROPERTIES,
            vocab.ATTRIBUTE_CLASSES,
            vocab.ATTRIBUTE_PROPERTIES,
            vocab.BIN_CLASSES,
        )
        for build in group
    }
    builders = {
        name: member
        for name, member in vars(vocab).items()
        if callable(member)
        and not name.startswith("_")
        and getattr(member, "__module__", None) == vocab.__name__
        and (name.endswith("_class") or name.endswith("_property"))
    }

    assert "start_class" in builders
    assert (
        sorted(name for name, build in builders.items() if build not in registered)
        == []
    )
