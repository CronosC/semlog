"""`AttributeEmitter`: attribute classes, their value individuals, and their owners."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import DataRepresentation
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.binning import Binning
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.attributes import AttributeEmitter
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
        {"price": {EntityKind.OBJECT: ["order"]}},
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

    assert onto.domains["has_step.channel"] == {"step"}
    assert onto.ranges["has_step.channel"] == {"step.channel"}


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
    table.data["step.price"] = [1.0e20, 2.5, 1.0e-7]

    onto = emit(OntoBuildSettings(), tmp_path / "wide.ofn", data=tiny_store)

    assert {
        "step.price.100000000000000000000",
        "step.price.0.0000001",
    } <= onto.individuals
    assert (
        "has_data_value",
        "step.price.100000000000000000000",
    ) in onto.data_assertions


def test_attributes_whose_names_share_a_prefix_keep_their_own_values(tmp_path):
    onto = emit(
        OntoBuildSettings(),
        tmp_path / "prefix.ofn",
        data=_order_store(a_b=["x"], a=["y"]),
    )

    assert {"order.a_b.x", "order.a.y"} <= onto.individuals


def test_a_bin_class_and_a_value_of_the_same_label_are_named_apart(tmp_path):
    store = _order_store(price=[float(n) for n in range(1, 13)])

    onto = emit(
        OntoBuildSettings(
            **{
                **BIN_SETTINGS,
                "data_representations": [
                    DataRepresentation.VALUES,
                    DataRepresentation.EQUAL_BINS,
                ],
                "bin_labels": ["1", "12"],
            }
        ),
        tmp_path / "bins.ofn",
        data=store,
    )

    assert "order.price_bin_1" in onto.classes
    assert "order.price.1" in onto.individuals
    assert onto.classes & onto.individuals == set()


def test_a_boolean_attribute_keeps_its_values_when_numbers_are_left_out(tmp_path):
    onto = emit(
        OntoBuildSettings(data_representations=[]),
        tmp_path / "bool.ofn",
        data=_order_store(express=[True, False], price=[1.0, 2.0]),
    )

    assert "order.price" not in onto.declared
    assert {"order.express.True", "order.express.False"} <= onto.individuals
    assert ("has_data_value", "order.express.True") in onto.data_assertions


def test_a_boolean_attribute_is_not_binned(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        onto = emit(
            OntoBuildSettings(**BIN_SETTINGS),
            tmp_path / "bool_bins.ofn",
            data=_order_store(express=[True, False], price=[1.0, 2.0]),
        )

    assert {"order.express.True", "order.express.False"} <= onto.individuals
    assert {c for c in onto.classes if c.startswith("order.express_bin_")} == set()
    assert "'order.express' has too few distinct values" not in caplog.text


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
    assert _values(sink) >= {"price.1", "price.7", "price.99"}
    assert binned == {"price.1", "price.7"}
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
    assert _values(sink) == {"price.4"}
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

    assert _literal_subjects(sink) == {"price.1", "price.7", "price.99"}
