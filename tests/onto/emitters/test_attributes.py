"""`AttributeEmitter`: attribute classes, their value individuals, and their owners."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import DataRepresentation
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.binning import Binning
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore, attribute_owners
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.attributes import (
    AttributeEmitter,
    report_conflated_attributes,
)
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.build import emit

BIN_SETTINGS = dict(
    data_representations=[DataRepresentation.EQUAL_BINS],
    bins=2,
    bin_labels=["low", "high"],
)
LOW_HIGH = Binning(labels=("low", "high"), edges=(0.0, 5.0, 10.0))


def _order_store(**columns) -> OCEDStore:
    """Orders with the given attribute columns, each touched by one event, cleaned."""
    oids = [f"o{i}" for i in range(1, len(next(iter(columns.values()))) + 1)]
    data = OCEDStore()
    data.add(
        EntityKind.OBJECT, "order", pd.DataFrame({"oid": oids, **columns}), id_col="oid"
    )
    eids = [f"e{i}" for i in range(1, len(oids) + 1)]
    data.add(
        EntityKind.EVENT,
        "step",
        pd.DataFrame({"eid": eids, "ts": [datetime(2024, 1, 1)] * len(eids)}),
        id_col="eid",
        time_col="ts",
    )
    data.add_relation(
        RelationKind.E2O,
        "concerns",
        pd.DataFrame({"eid": eids, "oid": oids}),
        from_id="eid",
        to_id="oid",
        to_type=["order"],
    )
    return clean_oced_data(data)


def _price_emitter(sink, settings: OntoBuildSettings, binning) -> AttributeEmitter:
    """An emitter for one object attribute, `price`, on `order`."""
    return AttributeEmitter(
        sink,
        settings,
        AnnotationWriter(sink, settings, {}),
        EmittedIndividuals(),
        {"price": ["order"]},
        {"price": binning} if binning is not None else {},
    )


def _values(sink) -> set[str]:
    return {name for _, name, _cls in sink.of_kind("individual")}


def _literal_subjects(sink) -> set[str]:
    return {subject for _, _prop, subject, _v in sink.of_kind("data")}


# --------------------------------------------------------------------------
# Owners and domains
# --------------------------------------------------------------------------


def test_an_attribute_of_one_type_has_that_type_as_its_domain(tiny_store, tmp_path):
    onto = emit(OntoBuildSettings(), tmp_path / "tiny.ofn", data=tiny_store)

    assert onto.domains["has_channel"] == {"step"}
    assert onto.ranges["has_channel"] == {"channel"}


def test_an_attribute_of_two_types_has_one_domain_naming_both(awkward_store, tmp_path):
    onto = emit(OntoBuildSettings(), tmp_path / "awkward.ofn", data=awkward_store)
    domain_axioms = [
        line
        for line in onto.axiom_lines
        if line.startswith("ObjectPropertyDomain(:has_channel ")
    ]

    assert onto.domains["has_channel"] == {"order", "item"}
    assert len(domain_axioms) == 1


def test_an_attribute_shared_by_several_types_is_reported(awkward_store, caplog):
    with caplog.at_level(logging.WARNING):
        conflated = report_conflated_attributes(attribute_owners(awkward_store))

    assert conflated == {"channel": ["item", "order"], "clerk": ["place_order", "ship"]}
    assert "4 different entities share 2 attribute names" in caplog.text
    assert "channel (item, order)" in caplog.text


def test_the_owners_of_a_widely_shared_attribute_are_counted(caplog):
    owners = {"commit_message": [f"activity_{i}" for i in range(68)]}

    with caplog.at_level(logging.WARNING):
        report_conflated_attributes(owners)

    assert "commit_message (68 types)" in caplog.text
    assert "activity_0" not in caplog.text


@pytest.mark.reasoner
def test_a_build_with_typed_attributes_is_consistent(awkward_store, tmp_path):
    path = tmp_path / "awkward.owl"
    OCEDOntoBuilder(DEFAULT_IRI, OntoBuildSettings()).build_ontology(
        awkward_store
    ).save(str(path))

    assert Reasoner(str(path)).check() is True


# --------------------------------------------------------------------------
# Value individuals
# --------------------------------------------------------------------------


def test_a_number_too_wide_for_plain_notation_is_named_without_an_exponent(
    tiny_store, tmp_path
):
    table = tiny_store.table(EntityKind.EVENT, "step")
    table.data["price"] = [1.0e20, 2.5, 1.0e-7]

    onto = emit(OntoBuildSettings(), tmp_path / "wide.ofn", data=tiny_store)

    assert {"price_100000000000000000000", "price_0.0000001"} <= onto.individuals
    assert ("has_data_value", "price_100000000000000000000") in onto.data_assertions


def test_attributes_whose_names_share_a_prefix_keep_their_own_values(tmp_path):
    onto = emit(
        OntoBuildSettings(),
        tmp_path / "prefix.ofn",
        data=_order_store(a_b=["x"], a=["y"]),
    )

    assert {"a_b_x", "a_y"} <= onto.individuals


def test_a_bin_class_and_a_value_of_the_same_label_are_named_apart(tmp_path):
    store = _order_store(price=["low", "high"])
    numbers = [float(n) for n in range(1, 13)]
    store.add(
        EntityKind.OBJECT,
        "priced",
        pd.DataFrame({"oid": [f"p{n:.0f}" for n in numbers], "price": numbers}),
        id_col="oid",
    )

    onto = emit(
        OntoBuildSettings(
            **{
                **BIN_SETTINGS,
                "data_representations": [
                    DataRepresentation.VALUES,
                    DataRepresentation.EQUAL_BINS,
                ],
            }
        ),
        tmp_path / "bins.ofn",
        data=store,
    )

    assert "price_bin_low" in onto.classes
    assert "price_low" in onto.individuals
    assert onto.classes & onto.individuals == set()


def test_a_boolean_attribute_keeps_its_values_when_numbers_are_left_out(tmp_path):
    onto = emit(
        OntoBuildSettings(data_representations=[]),
        tmp_path / "bool.ofn",
        data=_order_store(express=[True, False], price=[1.0, 2.0]),
    )

    assert "price" not in onto.declared
    assert {"express_True", "express_False"} <= onto.individuals
    assert ("has_data_value", "express_True") in onto.data_assertions


def test_a_boolean_attribute_is_not_binned(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        onto = emit(
            OntoBuildSettings(**BIN_SETTINGS),
            tmp_path / "bool_bins.ofn",
            data=_order_store(express=[True, False], price=[1.0, 2.0]),
        )

    assert {"express_True", "express_False"} <= onto.individuals
    assert {c for c in onto.classes if c.startswith("express_bin_")} == set()
    assert "'express' has too few distinct values" not in caplog.text


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


def test_a_value_outside_every_bin_is_emitted_without_a_bin(recording_sink, caplog):
    settings = OntoBuildSettings(**BIN_SETTINGS)
    sink = recording_sink(settings.profile)
    rows = pd.DataFrame({"oid": ["o1", "o2", "o3"], "price": [1.0, 7.0, 99.0]})

    with caplog.at_level(logging.WARNING):
        _price_emitter(sink, settings, LOW_HIGH).emit_columns(
            rows, "oid", vocab.OBJECT_ATTRIBUTE
        )

    binned = {name for _, name, cls in sink.of_kind("individual") if "_bin_" in cls}
    assert "fall outside all bin boundaries" in caplog.text
    assert _values(sink) >= {"price_1", "price_7", "price_99"}
    assert binned == {"price_1", "price_7"}
    assert _literal_subjects(sink) == set()


def test_a_column_too_narrow_to_bin_is_emitted_unbinned_and_reported_once(
    recording_sink, caplog
):
    """The same attribute reaches the emitter once per table carrying it."""
    settings = OntoBuildSettings(**BIN_SETTINGS)
    sink = recording_sink(settings.profile)
    emitter = _price_emitter(sink, settings, None)

    with caplog.at_level(logging.WARNING):
        for oid in ("o1", "o2"):
            emitter.emit_columns(
                pd.DataFrame({"oid": [oid], "price": [4.0]}),
                "oid",
                vocab.OBJECT_ATTRIBUTE,
            )

    warnings = [r for r in caplog.records if "too few distinct values" in r.message]
    assert _values(sink) == {"price_4"}
    assert len(warnings) == 1


def test_values_alongside_a_binning_mode_keeps_the_literals(recording_sink):
    settings = OntoBuildSettings(
        **{
            **BIN_SETTINGS,
            "data_representations": [
                DataRepresentation.VALUES,
                DataRepresentation.EQUAL_BINS,
            ],
        }
    )
    sink = recording_sink(settings.profile)
    rows = pd.DataFrame({"oid": ["o1", "o2", "o3"], "price": [1.0, 7.0, 99.0]})

    _price_emitter(sink, settings, LOW_HIGH).emit_columns(
        rows, "oid", vocab.OBJECT_ATTRIBUTE
    )

    assert _literal_subjects(sink) == {"price_1", "price_7", "price_99"}
