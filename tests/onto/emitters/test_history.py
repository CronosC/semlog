"""`HistoryEmitter`: the OCEL 2.0 attribute history as change records.

The matrix store records a string attribute and a numeric one changing.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import DataRepresentation, OCELExtension, OWLProfile
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.build import emit

CHANGES = {
    "AttrChange_o1_order_channel_20240102T090000_000000",
    "AttrChange_o1_order_channel_20240106T090000_000000",
    "AttrChange_o2_order_channel_20240103T090000_000000",
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


def _store_with_changes(oid: str, field: str, values: list, row: dict) -> OCEDStore:
    """One order and one event, with the given changes of one field, cleaned."""
    data = OCEDStore(ocel_mode=True)
    data.add(
        EntityKind.OBJECT, "order", pd.DataFrame({"oid": [oid], **row}), id_col="oid"
    )
    data.add(
        EntityKind.EVENT,
        "step",
        pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"])}),
        id_col="eid",
        time_col="ts",
    )
    data.add_relation(
        RelationKind.E2O,
        "concerns",
        pd.DataFrame({"eid": ["e1"], "oid": [oid]}),
        from_id="eid",
        to_id="oid",
        to_type=["order"],
    )
    data.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": [oid] * len(values),
                "ocel:type": ["order"] * len(values),
                "ocel:timestamp": pd.to_datetime(["2024-01-02"] * len(values)),
                "ocel:field": [field] * len(values),
                field: values,
            }
        ),
    )
    return clean_oced_data(data)


@pytest.fixture(scope="module")
def history(request):
    return request.getfixturevalue("built")("dl_history")


def test_every_recorded_change_becomes_a_record_under_its_attribute(history):
    changes = {
        n for n in history.individuals if "AttributeChange" in history.lineage(n)
    }

    assert changes == CHANGES
    assert history.parents["order_channelChange"] == {"AttributeChange"}
    assert history.parents["order_channel"] == {vocab.OBJECT_ATTRIBUTE}


def test_a_change_record_hangs_off_the_object_it_happened_to(history):
    owned = {(s, o) for p, s, o in history.relations if p == vocab.HAS_ATTRIBUTE_CHANGE}

    assert {s for s, _ in owned} == {"o1", "o2"}
    assert {o for _, o in owned} == CHANGES


def test_a_change_record_points_at_the_value_it_changed_to(history):
    channel = {o for p, _, o in history.relations if p == "change_value_order_channel"}
    price = {o for p, _, o in history.relations if p == "change_value_price"}

    assert channel == {
        "order_channel_shop",
        "order_channel_web",
        "order_channel_by_hand",
    }
    assert price == {"price_12.75", "price_41"}


def test_a_change_record_is_stamped_with_a_time_individual(history):
    stamped = {s for p, s, _ in history.relations if p == "changed_at"}
    times = {o for p, _, o in history.relations if p == "changed_at"}

    assert stamped == CHANGES
    assert {t for t in times if "Time" in history.types[t]} == times
    assert {(vocab.HAS_DATA_VALUE, t) for t in times} <= history.data_assertions


def test_a_value_shared_with_a_current_value_carries_one_literal(history):
    """o1 changed its price to 12.75, which is also o4's current price."""
    literals = [s for _, s in history.data_assertions if s == "price_12.75"]

    assert literals == ["price_12.75"]


def test_a_numeric_change_is_left_out_when_no_numbers_are_emitted(
    awkward_store, tmp_path
):
    onto = emit(
        history_settings(data_representations=[]),
        tmp_path / "none.ofn",
        data=awkward_store,
    )
    targets = {p for p, _, _ in onto.relations if p.startswith("change_value_")}

    assert targets == {"change_value_order_channel"}


def test_a_numeric_change_is_binned_like_a_current_value(awkward_store, tmp_path):
    onto = emit(
        history_settings(
            data_representations=[DataRepresentation.EQUAL_BINS],
            bins=2,
            bin_labels=["low", "high"],
        ),
        tmp_path / "bins.ofn",
        data=awkward_store,
    )

    assert onto.types["price_12.75"] == {"price", "price_bin_low"}
    assert onto.types["price_41"] == {"price", "price_bin_high"}


def test_two_changes_of_one_attribute_at_one_instant_stay_two_records(tmp_path, caplog):
    store = _store_with_changes("o1", "status", ["open", "closed"], {})

    with caplog.at_level(logging.WARNING):
        onto = emit(history_settings(), tmp_path / "repeat.ofn", data=store)

    records = [(s, o) for p, s, o in onto.relations if p == "change_value_status"]
    assert len({s for s, _ in records}) == 2
    assert len(records) == 2
    assert "more than one change of 'status' at the same instant" in caplog.text


def test_an_attribute_cleaning_renamed_keeps_its_history_under_the_new_name(
    tmp_path,
):
    """`observes` is the inverse of `observed_at`, so cleaning renames it."""
    store = _store_with_changes("o1", "observes", ["y"], {"observes": ["x"]})

    onto = emit(history_settings(), tmp_path / "renamed.ofn", data=store)

    assert "observes" not in onto.classes
    assert "change_value_attribute_observes" in onto.object_properties
    assert {"attribute_observes_x", "attribute_observes_y"} <= onto.individuals


def test_a_change_record_is_labeled_with_the_object_id_the_log_wrote(tmp_path):
    store = _store_with_changes("ORD#123", "status", ["shipped"], {})

    onto = emit(
        history_settings(derive_annotations=True),
        tmp_path / "history.ofn",
        data=store,
    )

    labels = [
        text
        for name, kind, text in onto.annotations
        if kind == "label" and name.startswith("AttrChange_")
    ]
    assert labels == ["status change for order (ORD#123) at 2024-01-02 00:00:00"]
