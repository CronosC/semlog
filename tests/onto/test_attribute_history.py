"""The OCEL 2.0 attribute history extension.

Each recorded edit becomes an individual under `AttributeChange`, hanging off
the object it happened to, pointing at the value it changed to and stamped
with when. A numeric attribute follows whatever the data representation says,
so an edit to one is left out when the numbers are and binned when they are.

The matrix store records both a string attribute and a numeric one changing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import (
    DataRepresentation,
    OCELExtension,
    OWLProfile,
)
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.settings import OntoBuildSettings
from tests.conftest import emit

CHANGES = {
    "AttrChange_o1_channel_20240102T090000_000000",
    "AttrChange_o1_channel_20240106T090000_000000",
    "AttrChange_o2_channel_20240103T090000_000000",
    "AttrChange_o1_price_20240104T090000_000000",
    "AttrChange_o2_price_20240105T090000_000000",
}


def history_settings(**overrides) -> OntoBuildSettings:
    """Return a history build with everything else turned off."""
    base: dict = {
        "profile": OWLProfile.OWL2EL,
        "ocel_extensions": [OCELExtension.ATTRIBUTE_HISTORY],
        "order_representations": [],
    }
    base.update(overrides)
    return OntoBuildSettings(**base)


@pytest.fixture(scope="module")
def history(request):
    return request.getfixturevalue("built")("dl_history")


def test_every_recorded_edit_becomes_an_individual(history):
    changes = {
        n for n in history.individuals if "AttributeChange" in history.lineage(n)
    }

    assert changes == CHANGES


def test_a_change_is_typed_by_the_attribute_it_changed(history):
    assert history.parents["channelChange"] == {"AttributeChange"}
    assert history.parents["priceChange"] == {"AttributeChange"}


def test_a_change_hangs_off_the_object_it_happened_to(history):
    owned = {(s, o) for p, s, o in history.relations if p == vocab.HAS_ATTRIBUTE_CHANGE}

    assert {s for s, _ in owned} == {"o1", "o2"}
    assert {o for _, o in owned} == CHANGES


def test_a_change_points_at_the_value_it_changed_to(history):
    channel = {o for p, _, o in history.relations if p == "change_value_channel"}
    price = {o for p, _, o in history.relations if p == "change_value_price"}

    assert channel == {"channel_shop", "channel_web", "channel_by_hand"}
    assert price == {"price_12.75", "price_41.0"}


def test_a_change_is_stamped_with_when_it_happened(history):
    stamped = {s for p, s, _ in history.relations if p == "changed_at"}
    times = {o for p, _, o in history.relations if p == "changed_at"}

    assert stamped == CHANGES
    assert all("Time" in history.types[t] for t in times)
    assert all((vocab.HAS_DATA_VALUE, t) in history.data_assertions for t in times)


def test_the_changed_attribute_is_an_object_attribute(history):
    assert history.parents["channel"] == {vocab.OBJECT_ATTRIBUTE}
    assert history.parents["price"] == {vocab.OBJECT_ATTRIBUTE}


# --------------------------------------------------------------------------
# A numeric attribute follows the data representation
# --------------------------------------------------------------------------


def test_a_numeric_change_carries_its_number_under_values(awkward_store, tmp_path):
    onto = emit(
        history_settings(data_representations=[DataRepresentation.VALUES]),
        tmp_path / "values.ofn",
        data=awkward_store,
    )
    values = {o for p, _, o in onto.relations if p == "change_value_price"}

    assert values == {"price_12.75", "price_41.0"}
    assert {(vocab.HAS_DATA_VALUE, v) for v in values} <= onto.data_assertions


def test_a_numeric_change_is_left_out_when_the_numbers_are(awkward_store, tmp_path):
    onto = emit(
        history_settings(data_representations=[]),
        tmp_path / "none.ofn",
        data=awkward_store,
    )

    assert not any(p == "change_value_price" for p, _, _ in onto.relations)
    assert not any(
        n.endswith("_price_20240104T090000_000000") for n in onto.individuals
    )


def test_a_numeric_change_is_binned_the_way_the_current_value_is(
    awkward_store, tmp_path
):
    """The history and the object have to agree on what a number looks like."""
    onto = emit(
        history_settings(
            data_representations=[DataRepresentation.EQUAL_BINS],
            bins=2,
            bin_labels=["low", "high"],
        ),
        tmp_path / "bins.ofn",
        data=awkward_store,
    )
    changed_to = {o for p, _, o in onto.relations if p == "change_value_price"}

    assert changed_to == {"price_12.75", "price_41.0"}
    assert onto.types["price_12.75"] == {"price", "price_bin_low"}
    assert onto.types["price_41.0"] == {"price", "price_bin_high"}


def test_a_change_and_a_current_value_land_in_the_same_bin(awkward_store, tmp_path):
    """One cut per attribute, so a number cannot be low on the object and
    high in its history."""
    onto = emit(
        history_settings(
            data_representations=[DataRepresentation.EQUAL_BINS],
            bins=2,
            bin_labels=["low", "high"],
        ),
        tmp_path / "bins.ofn",
        data=awkward_store,
    )
    bins = {"price_bin_low", "price_bin_high"}

    for individual, classes in onto.types.items():
        if individual.startswith("price_"):
            assert len(classes & bins) <= 1, individual


def test_a_numeric_change_keeps_its_number_rather_than_a_bin_label(
    awkward_store, tmp_path
):
    onto = emit(
        history_settings(
            data_representations=[
                DataRepresentation.VALUES,
                DataRepresentation.EQUAL_BINS,
            ],
            bins=2,
            bin_labels=["low", "high"],
        ),
        tmp_path / "both.ofn",
        data=awkward_store,
    )
    values = {o for p, _, o in onto.relations if p == "change_value_price"}

    assert values == {"price_12.75", "price_41.0"}
    assert not any(v in ("price_low", "price_high") for v in values)


def test_a_string_change_is_unaffected_by_the_data_representation(
    awkward_store, tmp_path
):
    onto = emit(
        history_settings(data_representations=[]),
        tmp_path / "none.ofn",
        data=awkward_store,
    )
    values = {o for p, _, o in onto.relations if p == "change_value_channel"}

    assert values == {"channel_shop", "channel_web", "channel_by_hand"}


def test_a_value_shared_with_the_current_state_keeps_one_literal(history):
    """o1 changed its price to 12.75, which is also o4's current price. The
    value individual is one individual, so it carries one literal.
    """
    literals = [
        (prop, subject)
        for prop, subject in history.data_assertions
        if subject == "price_12.75"
    ]

    assert len(literals) == 1
    assert not history.duplicate_axioms


# --------------------------------------------------------------------------
# Two changes at one instant
# --------------------------------------------------------------------------


def _store_with_repeated_change() -> OCEDStore:
    """A log that edits one attribute of one object twice at one moment."""
    data = OCEDStore(ocel_mode=True)
    data.add_object("order", id_col="oid", data=pd.DataFrame({"oid": ["o1"]}))
    data.add_event(
        "step",
        id_col="eid",
        time_col="ts",
        data=pd.DataFrame(
            {"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"])},
        ),
    )
    data.add_event_to_object_relation(
        "concerns",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )
    data.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1", "o1"],
                "ocel:type": ["order", "order"],
                "ocel:timestamp": pd.to_datetime(["2024-01-02", "2024-01-02"]),
                "ocel:field": ["status", "status"],
                "status": ["open", "closed"],
            }
        ),
    )
    return clean_oced_data(data)


def test_two_changes_at_one_instant_stay_two_records(tmp_path):
    """One change record carrying two new values says the object took both."""
    onto = emit(
        history_settings(),
        tmp_path / "repeat.ofn",
        data=_store_with_repeated_change(),
    )
    records = {s for p, s, _ in onto.relations if p == "change_value_status"}

    assert len(records) == 2
    for record in records:
        targets = {
            o
            for p, s, o in onto.relations
            if p == "change_value_status" and s == record
        }
        assert len(targets) == 1


def test_a_repeated_change_is_reported(tmp_path, caplog):
    emit(
        history_settings(),
        tmp_path / "repeat.ofn",
        data=_store_with_repeated_change(),
    )

    assert "more than one change of 'status' at the same instant" in caplog.text


def test_a_change_id_is_unchanged_when_nothing_repeats(history):
    assert CHANGES <= history.individuals
    assert not any(
        name.startswith("AttrChange_") and name.endswith("_2")
        for name in history.individuals
    )


# --------------------------------------------------------------------------
# An attribute cleaning renamed
# --------------------------------------------------------------------------


def _store_with_a_reserved_attribute() -> OCEDStore:
    """An object attribute named after a schema property, with a history."""
    data = OCEDStore(ocel_mode=True)
    data.add_object(
        "order",
        id_col="oid",
        data=pd.DataFrame({"oid": ["o1"], "observes": ["x"]}),
    )
    data.add_event(
        "step",
        id_col="eid",
        time_col="ts",
        data=pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"])}),
    )
    data.add_event_to_object_relation(
        "concerns",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )
    data.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1"],
                "ocel:type": ["order"],
                "ocel:timestamp": pd.to_datetime(["2024-01-02"]),
                "ocel:field": ["observes"],
                "observes": ["y"],
            }
        ),
    )
    return clean_oced_data(data)


def test_a_renamed_attribute_and_its_history_stay_one_attribute(tmp_path):
    """`observes` is the inverse of `observed_at`, so cleaning moves it aside."""
    onto = emit(
        history_settings(),
        tmp_path / "renamed.ofn",
        data=_store_with_a_reserved_attribute(),
    )

    assert "observes_attr" in onto.classes
    assert "observes" not in onto.classes
    assert "change_value_observes_attr" in onto.object_properties
    assert {"observes_attr_x", "observes_attr_y"} <= onto.individuals
