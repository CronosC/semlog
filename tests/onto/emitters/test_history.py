"""`HistoryEmitter`: the OCEL 2.0 attribute history as change records.

The matrix store records a string attribute and a numeric one changing.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import AttributeRepresentation, OCELExtension, OWLProfile
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.build import emit

CHANGES = {
    "channel.o1.20240102T090000_000000",
    "channel.o1.20240106T090000_000000",
    "channel.o2.20240102T090000_000000",
    "price.o1.20240104T090000_000000",
    "price.o2.20240105T090000_000000",
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


def _store_with_changes(
    oid: str, field: str, values: list, row: dict, other_types: tuple[str, ...] = ()
) -> OCEDStore:
    """One order and one event, with the given changes of one field, cleaned.

    Each of `other_types` is an object type with one object and no attributes.
    """
    data = OCEDStore(ocel_mode=True)
    data.add(
        EntityKind.OBJECT, "order", pd.DataFrame({"oid": [oid], **row}), id_col="oid"
    )
    for obj_type in other_types:
        data.add(
            EntityKind.OBJECT,
            obj_type,
            pd.DataFrame({"oid": [f"{obj_type}_1"]}),
            id_col="oid",
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
    assert history.parents["channelChange"] == {"AttributeChange"}
    assert history.parents["channel"] == {vocab.OBJECT_ATTRIBUTE}


def test_a_change_record_hangs_off_the_object_it_happened_to(history):
    owned = {(s, o) for p, s, o in history.relations if p == vocab.HAS_ATTRIBUTE_CHANGE}

    assert {s for s, _ in owned} == {"o1", "o2"}
    assert {o for _, o in owned} == CHANGES


def test_a_change_record_points_at_the_value_it_changed_to(history):
    channel = {o for p, _, o in history.relations if p == "change_value_channel"}
    price = {o for p, _, o in history.relations if p == "change_value_price"}

    assert channel == {
        "channel.shop",
        "channel.web",
        "channel.by_hand",
    }
    assert price == {"price.12.75", "price.41"}


def test_a_change_record_is_stamped_with_a_time_individual(history):
    stamped = {s for p, s, _ in history.relations if p == "changed_at"}
    times = {o for p, _, o in history.relations if p == "changed_at"}

    assert stamped == CHANGES
    assert {t for t in times if "Time" in history.types[t]} == times
    assert {(vocab.TIMESTAMP, t) for t in times} <= history.data_assertions


def test_a_value_shared_with_a_current_value_carries_one_literal(history):
    """o1 changed its price to 12.75, which is also o4's current price."""
    literals = [s for _, s in history.data_assertions if s == "price.12.75"]

    assert literals == ["price.12.75"]


def test_literals_puts_the_new_value_on_the_change_record(awkward_store, tmp_path):
    onto = emit(
        history_settings(attribute_representations=[AttributeRepresentation.LITERALS]),
        tmp_path / "literals.ofn",
        data=awkward_store,
    )
    carriers = {s for p, s, _v, _t in onto.data_values if p == "has_value_price"}

    assert {
        "price.o1.20240104T090000_000000",
        "price.o2.20240105T090000_000000",
    } <= carriers
    assert "change_value_price" not in onto.declared


def test_the_value_property_domain_includes_the_change_class_under_literals(
    awkward_store, tmp_path
):
    onto = emit(
        history_settings(
            profile=OWLProfile.OWL2DL,
            attribute_representations=[AttributeRepresentation.LITERALS],
        ),
        tmp_path / "domain.ofn",
        data=awkward_store,
    )

    assert onto.domains["has_value_price"] == {"order", "priceChange"}


def test_an_object_attribute_without_changes_gets_no_change_class(
    awkward_store, tmp_path
):
    """The matrix store records no change of `weight`."""
    onto = emit(
        history_settings(
            profile=OWLProfile.OWL2DL,
            attribute_representations=[AttributeRepresentation.LITERALS],
        ),
        tmp_path / "domain.ofn",
        data=awkward_store,
    )

    assert onto.domains["has_value_weight"] == {"item"}
    assert "weightChange" not in onto.declared


def test_a_numeric_change_is_binned_like_a_current_value(awkward_store, tmp_path):
    onto = emit(
        history_settings(
            attribute_representations=[
                AttributeRepresentation.INDIVIDUALS,
                AttributeRepresentation.EQUAL_BINS,
            ],
            bins=2,
            bin_labels=["low", "high"],
        ),
        tmp_path / "bins.ofn",
        data=awkward_store,
    )

    assert onto.types["price.12.75"] == {"price", "price_bin_low"}
    assert onto.types["price.41"] == {"price", "price_bin_high"}


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
    """`status` is an object type, so cleaning renames the attribute."""
    store = _store_with_changes(
        "o1", "status", ["y"], {"status": ["x"]}, other_types=("status",)
    )

    onto = emit(history_settings(), tmp_path / "renamed.ofn", data=store)

    assert "change_value_status" not in onto.object_properties
    assert "change_value_attribute_status" in onto.object_properties
    assert {"attribute_status.x", "attribute_status.y"} <= onto.individuals


def test_a_changed_text_value_carries_the_literal_the_log_wrote(tmp_path):
    store = _store_with_changes("o1", "status", ["in transit"], {})

    onto = emit(
        history_settings(attribute_representations=[AttributeRepresentation.LITERALS]),
        tmp_path / "history.ofn",
        data=store,
    )

    assert {
        value for prop, _s, value, _t in onto.data_values if prop == "has_value_status"
    } == {"in transit"}


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
        if kind == "label" and " change for " in text
    ]
    assert labels == ["status change for order (ORD#123) at 2024-01-02 00:00:00"]


def _store_with_causes() -> OCEDStore:
    """One order with a change at the time of one event, of two events and of none.

    `e1` concerns the order under two qualifiers, `e2` and `e3` share a
    timestamp, and no event of the order occurs at the third change.
    """
    data = OCEDStore(ocel_mode=True)
    data.add(EntityKind.OBJECT, "order", pd.DataFrame({"oid": ["o1"]}), id_col="oid")
    data.add(
        EntityKind.EVENT,
        "step",
        pd.DataFrame(
            {
                "eid": ["e1", "e2", "e3"],
                "ts": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"]),
            }
        ),
        id_col="eid",
        time_col="ts",
    )
    for qualifier in ("creates", "updates"):
        data.add_relation(
            RelationKind.E2O,
            qualifier,
            pd.DataFrame({"eid": ["e1", "e2", "e3"], "oid": ["o1"] * 3}),
            from_id="eid",
            to_id="oid",
            to_type=["order"],
        )
    data.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1"] * 3,
                "ocel:type": ["order"] * 3,
                "ocel:timestamp": pd.to_datetime(
                    ["2024-01-01", "2024-01-02", "2024-01-03"]
                ),
                "ocel:field": ["status"] * 3,
                "status": ["open", "paid", "closed"],
            }
        ),
    )
    return clean_oced_data(data)


def test_a_change_at_the_time_of_one_event_of_its_object_is_caused_by_it(history):
    causes = {(s, o) for p, s, o in history.relations if p == vocab.CAUSED_BY}

    assert causes == {("channel.o2.20240102T090000_000000", "e2")}


def test_a_change_with_no_event_or_several_events_at_its_time_has_no_cause(tmp_path):
    onto = emit(history_settings(), tmp_path / "causes.ofn", data=_store_with_causes())

    causes = {(s, o) for p, s, o in onto.relations if p == vocab.CAUSED_BY}
    assert causes == {("status.o1.20240101T000000_000000", "e1")}


def test_the_build_reports_the_changes_left_without_a_cause(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="semanticlog.onto.emitters.history"):
        emit(history_settings(), tmp_path / "causes.ofn", data=_store_with_causes())

    assert (
        "Linked 1 of 3 attribute change record(s) to the event that caused them. "
        "1 had no event of their object at the time of the change, 1 had several."
    ) in caplog.text


def test_a_build_without_the_history_has_no_caused_by(tmp_path):
    onto = emit(
        history_settings(ocel_extensions=[]),
        tmp_path / "plain.ofn",
        data=_store_with_causes(),
    )

    assert vocab.CAUSED_BY not in onto.object_properties
    assert {p for p, _, _ in onto.relations if p == vocab.CAUSED_BY} == set()
