"""The names a build emits, and the reserved set derived from them."""

import numpy as np
import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import (
    InverseRepresentation,
    OCELExtension,
)
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.tbox import construct_tbox

TEST_IRI = "https://test.example.org/vocab#"


def test_reserved_names_cover_every_fixed_name_the_upper_ontology_emits():
    settings = OntoBuildSettings(
        ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY],
        inverse_representations=[InverseRepresentation.AXIOMATIC],
    )
    onto = construct_tbox(TEST_IRI, settings).onto

    emitted = {
        entity.iri.remainder
        for entity in (
            list(onto.classes_in_signature())
            + list(onto.object_properties_in_signature())
            + list(onto.data_properties_in_signature())
        )
    }

    assert emitted <= vocab.RESERVED_NAMES, (
        f"emitted but not reserved: {sorted(emitted - vocab.RESERVED_NAMES)}"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3.5, "3.5"),
        (200.0, "200.0"),
        (-0.0001, "-0.0001"),
        (1e20, "100000000000000000000"),
        (1.5e16, "15000000000000000"),
        (1e-7, "0.0000001"),
    ],
)
def test_a_number_is_written_in_full_however_wide_it_is(value, expected):
    """`str` would switch to exponent notation, which no local name can hold."""
    assert vocab.format_number(value) == expected
    assert vocab.is_safe_local_name(vocab.attribute_value("amount", value))


def test_a_numpy_number_is_written_like_a_python_one():
    """A numpy scalar passes `isinstance(x, float)`, and its own text form does not."""
    assert vocab.format_number(np.float64(2.5)) == "2.5"


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_a_number_that_is_not_one_is_refused(value):
    with pytest.raises(ValueError, match="cannot be written as a number"):
        vocab.format_number(value)


def test_refusing_a_value_names_the_attribute_it_came_from():
    with pytest.raises(ValueError, match="attribute 'amount'"):
        vocab.attribute_value("amount", float("inf"))


def test_reserved_names_cover_the_provenance_annotations():
    assert vocab.PROVENANCE_ANNOTATIONS <= vocab.RESERVED_NAMES


def test_reserved_names_include_generated_inverses():
    assert vocab.inverse(vocab.EVENT_OBJECT_RELATION) in vocab.RESERVED_NAMES
    assert vocab.inverse(vocab.DF) in vocab.RESERVED_NAMES


def test_inverse_prefers_the_explicit_schema_name():
    assert vocab.inverse(vocab.HAS_ATTRIBUTE) == vocab.ATTRIBUTE_OF
    assert vocab.inverse(vocab.OBSERVED_AT) == vocab.OBSERVES


def test_schema_named_inverses_are_symmetric():
    for prop, inv in vocab.SCHEMA_NAMED_INVERSES.items():
        assert vocab.SCHEMA_NAMED_INVERSES[inv] == prop


def test_inverse_falls_back_to_the_generated_form():
    assert vocab.inverse("df_order") == "inv_df_order"


def test_inverse_consults_the_supplied_map_first():
    assert vocab.inverse("a", {"a": "b"}) == "b"
    assert vocab.inverse("c", {"a": "b"}) == "inv_c"


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        (lambda: vocab.start_class("order"), "StartEvent_order"),
        (lambda: vocab.end_class("order"), "EndEvent_order"),
        (lambda: vocab.df_property("order"), "df_order"),
        (lambda: vocab.ef_property("order"), "ef_order"),
        (lambda: vocab.has_attribute_property("price"), "has_price"),
        (lambda: vocab.change_value_property("price"), "change_value_price"),
        (lambda: vocab.attribute_change_class("price"), "priceChange"),
        (lambda: vocab.bin_class("price", "low"), "price_bin_low"),
    ],
)
def test_derived_name_templates(call, expected):
    assert call() == expected


def test_attribute_change_ids_are_vectorised():
    ids = vocab.attribute_change_ids(
        pd.Series(["o1", "o2"]), "price", pd.Series(["T1", "T2"])
    )
    assert list(ids) == ["AttrChange_o1_price_T1", "AttrChange_o2_price_T2"]


def test_attribute_value_interpolates_plain_values():
    assert vocab.attribute_value("status", "shipped") == "status_shipped"
    assert vocab.attribute_value("price", 10) == "price_10"


@pytest.mark.parametrize("utc", [True, False])
def test_attribute_value_renders_timestamps_iri_safely(utc):
    series = pd.Series(pd.to_datetime(["2022-01-01 10:30:00"], utc=utc))
    name = vocab.attribute_value("ts", series.unique()[0])
    assert name == "ts_20220101T103000_000000"


def test_timestamp_id_sanitises_non_timestamps():
    assert vocab.timestamp_id("2022-01-01 10:30") == "2022_01_01_10_30"


def test_timestamp_ids_handles_datetime_and_object_columns():
    stamps = pd.Series(pd.to_datetime(["2022-01-01 10:30:00"], utc=True))
    assert list(vocab.timestamp_ids(stamps)) == ["20220101T103000_000000"]
    assert list(vocab.timestamp_ids(pd.Series(["a-b"]))) == ["a_b"]


def test_every_schema_name_builder_is_registered_for_the_reserved_set():
    """A builder left out of the registry is a collision a build cannot see.

    `derived_schema_names` is what cleaning renames the log against, so a name
    this module can emit but that map never produces is one the input can claim
    first.
    """
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
    assert builders, "the naming convention this test relies on found nothing"
    missing = sorted(
        name for name, member in builders.items() if member not in registered
    )
    assert missing == [], (
        f"{missing} build schema names that derived_schema_names never produces. "
        "Add each to the matching tuple in core/vocabulary.py."
    )
