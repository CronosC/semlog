"""`AttributeEmitter`: attribute classes, their value individuals, and their owners."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import DataRepresentation, OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.binning import Binning
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.attributes import AttributeEmitter, attribute_datatypes
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.build import emit
from tests.support.functional import read_functional

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


def test_the_attribute_classes_and_the_bins_of_one_attribute_are_disjoint(tmp_path):
    onto = emit(
        OntoBuildSettings(**BIN_SETTINGS),
        tmp_path / "bins.ofn",
        data=_order_store(
            price=[float(n) for n in range(1, 13)], channel=["a", "b"] * 6
        ),
    )

    assert frozenset({"order.price", "order.channel"}) in onto.disjoint
    assert frozenset({"order.price_bin_low", "order.price_bin_high"}) in onto.disjoint


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
        "has_value_step.price",
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
    assert ("has_value_order.express", "order.express.True") in onto.data_assertions


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


# --------------------------------------------------------------------------
# Literals and bin definitions
# --------------------------------------------------------------------------

VALUES_AND_BINS = dict(
    data_representations=[DataRepresentation.VALUES, DataRepresentation.EQUAL_BINS],
    bins=2,
    bin_labels=["low", "high"],
)
TWELVE_PRICES = [float(n) for n in range(1, 13)]


def test_a_text_value_carries_a_string_literal(tmp_path):
    onto = emit(
        OntoBuildSettings(),
        tmp_path / "text.ofn",
        data=_order_store(channel=["web", "shop"]),
    )

    assert (
        "has_value_order.channel",
        "order.channel.web",
        "web",
        "string",
    ) in onto.data_values


def test_an_attribute_value_property_is_below_has_data_value(tmp_path):
    onto = emit(
        OntoBuildSettings(), tmp_path / "price.ofn", data=_order_store(price=[1.5])
    )

    assert onto.property_parents["has_value_order.price"] == {vocab.HAS_DATA_VALUE}
    assert "has_value_order.price" in onto.functional


def test_an_attribute_of_a_type_named_with_a_leading_digit_gets_a_value_property(
    tmp_path,
):
    data = OCEDStore()
    data.add(
        EntityKind.EVENT,
        "1. Beratung",
        pd.DataFrame({"eid": ["e1"], "ts": [datetime(2024, 1, 1)], "titel": ["t"]}),
        id_col="eid",
        time_col="ts",
    )

    onto = emit(OntoBuildSettings(), tmp_path / "digit.ofn", data=clean_oced_data(data))

    assert ("has_value_1._Beratung.titel", "1._Beratung.titel.t", "t", "string") in (
        onto.data_values
    )


@pytest.mark.parametrize(
    ("values", "datatype"),
    [
        ([1.5, 2.0], "decimal"),
        ([1, 2], "integer"),
        (["web", "shop"], "string"),
        ([True, False], "boolean"),
    ],
)
def test_an_attribute_value_property_has_the_column_datatype_as_range(
    tmp_path, values, datatype
):
    onto = emit(
        OntoBuildSettings(), tmp_path / "range.ofn", data=_order_store(x=values)
    )

    assert onto.ranges["has_value_order.x"] == {datatype}


def test_mixed_datatypes_give_no_range(caplog):
    store = _order_store(x=["web", 2])

    with caplog.at_level(logging.WARNING):
        datatypes = attribute_datatypes(store, OntoBuildSettings())

    assert "order.x" not in datatypes
    assert "'order.x' holds values of more than one datatype" in caplog.text


def test_a_bin_is_defined_by_its_edges_under_dl(recording_sink):
    sink = recording_sink(OWLProfile.OWL2DL)
    OCEDOntoBuilder(
        DEFAULT_IRI, OntoBuildSettings(**VALUES_AND_BINS), sink=sink
    ).build_ontology(_order_store(price=TWELVE_PRICES))

    definitions = {call[1]: call[2:] for call in sink.of_kind("bin_definition")}
    assert definitions == {
        "order.price_bin_low": ("order.price", "has_value_order.price", None, 6.5),
        "order.price_bin_high": ("order.price", "has_value_order.price", 6.5, None),
    }


def test_a_bin_has_no_definition_under_el(recording_sink):
    sink = recording_sink(OWLProfile.OWL2EL)
    OCEDOntoBuilder(
        DEFAULT_IRI,
        OntoBuildSettings(profile=OWLProfile.OWL2EL, **VALUES_AND_BINS),
        sink=sink,
    ).build_ontology(_order_store(price=TWELVE_PRICES))

    assert sink.of_kind("bin_definition") == []
    assert {c[1] for c in sink.of_kind("declare") if "_bin_" in c[1]} == {
        "order.price_bin_low",
        "order.price_bin_high",
    }


def test_a_bin_without_values_has_no_definition(recording_sink):
    sink = recording_sink(OWLProfile.OWL2DL)
    settings = OntoBuildSettings(
        **{**VALUES_AND_BINS, "data_representations": [DataRepresentation.EQUAL_BINS]}
    )
    OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(
        _order_store(price=TWELVE_PRICES)
    )

    assert sink.of_kind("bin_definition") == []


@pytest.mark.reasoner
def test_a_reasoner_places_an_unasserted_value_in_its_bin(tmp_path):
    path = tmp_path / "bins.ofn"
    emit(
        OntoBuildSettings(**VALUES_AND_BINS),
        path,
        data=_order_store(price=TWELVE_PRICES),
    )
    extra = (
        "ClassAssertion(:order.price :probe)\n"
        'DataPropertyAssertion(:has_value_order.price :probe "9.5"^^xsd:decimal)\n'
    )
    text = path.read_text(encoding="utf-8").rstrip()
    path.write_text(f"{text[:-1]}{extra})\n", encoding="utf-8")

    reasoner = Reasoner(str(path))
    reasoner.materialize(overwrite_old=False, property_assertions=False)
    reasoner.save(str(tmp_path / "materialized.ofn"))

    types = read_functional(tmp_path / "materialized.ofn").types["probe"]
    assert "order.price_bin_high" in types
    assert "order.price_bin_low" not in types
