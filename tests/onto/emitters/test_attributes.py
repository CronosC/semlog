"""`AttributeEmitter`: attribute classes, their value individuals, and their owners."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import (
    AttributeRepresentation,
    IndividualKind,
    InverseRepresentation,
    OWLProfile,
)
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
from tests.support.build import BACKENDS, emit
from tests.support.functional import read_functional

INDIVIDUALS = [AttributeRepresentation.INDIVIDUALS]
LITERALS = [AttributeRepresentation.LITERALS]
BOTH = [AttributeRepresentation.INDIVIDUALS, AttributeRepresentation.LITERALS]
BIN_SETTINGS = dict(
    attribute_representations=[
        AttributeRepresentation.INDIVIDUALS,
        AttributeRepresentation.EQUAL_BINS,
    ],
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


def _two_kind_store(order_weight=2.5, step_weight=3.0) -> OCEDStore:
    """`weight` on the object type `order` and on the activity `step`, cleaned."""
    data = OCEDStore()
    data.add(
        EntityKind.OBJECT,
        "order",
        pd.DataFrame({"oid": ["o1"], "weight": [order_weight]}),
        id_col="oid",
    )
    data.add(
        EntityKind.EVENT,
        "step",
        pd.DataFrame(
            {"eid": ["e1"], "ts": [datetime(2024, 1, 1)], "weight": [step_weight]}
        ),
        id_col="eid",
        time_col="ts",
    )
    data.add_relation(
        RelationKind.E2O,
        "concerns",
        pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
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
# Owners, parents and domains
# --------------------------------------------------------------------------


def test_an_attribute_of_one_type_has_that_type_as_its_domain(tiny_store, tmp_path):
    onto = emit(OntoBuildSettings(), tmp_path / "tiny.ofn", data=tiny_store)

    assert onto.domains["has_channel"] == {"step"}
    assert onto.ranges["has_channel"] == {"channel"}


def test_an_attribute_of_two_types_has_both_as_its_domain(tmp_path):
    onto = emit(OntoBuildSettings(), tmp_path / "two.ofn", data=_two_kind_store())

    assert onto.domains["has_weight"] == {"order", "step"}
    assert onto.ranges["has_weight"] == {"weight"}
    assert onto.property_parents["has_weight"] == {vocab.HAS_ATTRIBUTE}


def test_one_value_under_two_types_gives_one_individual(tmp_path):
    onto = emit(
        OntoBuildSettings(), tmp_path / "one.ofn", data=_two_kind_store(2.5, 2.5)
    )

    assert {n for n in onto.individuals if n.startswith("weight.")} == {"weight.2.5"}
    assert {("has_weight", "o1", "weight.2.5"), ("has_weight", "e1", "weight.2.5")} <= (
        onto.relations
    )


def test_an_attribute_of_event_types_only_sits_below_event_attribute(
    tiny_store, tmp_path
):
    onto = emit(OntoBuildSettings(), tmp_path / "tiny.ofn", data=tiny_store)

    assert onto.parents["channel"] == {vocab.EVENT_ATTRIBUTE}


def test_an_attribute_of_object_types_only_sits_below_object_attribute(tmp_path):
    onto = emit(
        OntoBuildSettings(), tmp_path / "order.ofn", data=_order_store(channel=["web"])
    )

    assert onto.parents["channel"] == {vocab.OBJECT_ATTRIBUTE}


def test_an_attribute_of_both_kinds_sits_below_attribute(tmp_path):
    onto = emit(OntoBuildSettings(), tmp_path / "two.ofn", data=_two_kind_store())

    assert onto.parents["weight"] == {vocab.ATTRIBUTE}


def test_the_attribute_classes_and_the_bins_of_one_attribute_are_disjoint(tmp_path):
    onto = emit(
        OntoBuildSettings(**BIN_SETTINGS),
        tmp_path / "bins.ofn",
        data=_order_store(
            price=[float(n) for n in range(1, 13)], channel=["a", "b"] * 6
        ),
    )

    assert frozenset({"price", "channel"}) in onto.disjoint
    assert frozenset({"price_bin_low", "price_bin_high"}) in onto.disjoint


@pytest.mark.reasoner
def test_an_attribute_of_both_kinds_builds_consistently_under_hermit(tmp_path):
    path = tmp_path / "two.ofn"
    emit(
        OntoBuildSettings(
            inverse_representations=[InverseRepresentation.AXIOMATIC],
        ),
        path,
        data=_two_kind_store(),
    )

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

    assert {
        "price.100000000000000000000",
        "price.0.0000001",
    } <= onto.individuals
    assert (
        "has_value_price",
        "price.100000000000000000000",
    ) in onto.data_assertions


def test_attributes_whose_names_share_a_prefix_keep_their_own_values(tmp_path):
    onto = emit(
        OntoBuildSettings(),
        tmp_path / "prefix.ofn",
        data=_order_store(a_b=["x"], a=["y"]),
    )

    assert {"a_b.x", "a.y"} <= onto.individuals


def test_a_bin_class_and_a_value_of_the_same_label_are_named_apart(tmp_path):
    store = _order_store(price=[float(n) for n in range(1, 13)])

    onto = emit(
        OntoBuildSettings(**{**BIN_SETTINGS, "bin_labels": ["1", "12"]}),
        tmp_path / "bins.ofn",
        data=store,
    )

    assert "price_bin_1" in onto.classes
    assert "price.1" in onto.individuals
    assert onto.classes & onto.individuals == set()


def test_a_boolean_attribute_is_not_binned(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        onto = emit(
            OntoBuildSettings(**BIN_SETTINGS),
            tmp_path / "bool_bins.ofn",
            data=_order_store(express=[True, False], price=[1.0, 2.0]),
        )

    assert {"express.True", "express.False"} <= onto.individuals
    assert {c for c in onto.classes if c.startswith("express_bin_")} == set()
    assert [r.getMessage() for r in caplog.records] == [
        "Attribute 'price' has too few distinct values to bin. "
        "Its values carry no bin class."
    ]


# --------------------------------------------------------------------------
# Binning
# --------------------------------------------------------------------------


def test_a_value_outside_every_bin_is_emitted_without_a_bin(recording_sink, caplog):
    settings = OntoBuildSettings(**BIN_SETTINGS)
    sink = recording_sink(settings.profile)
    rows = pd.DataFrame({"oid": ["o1", "o2", "o3"], "price": [1.0, 7.0, 99.0]})

    with caplog.at_level(logging.WARNING):
        _price_emitter(sink, settings, LOW_HIGH).emit_columns(
            rows, "oid", IndividualKind.OBJECT
        )

    binned = {name for _, name, cls in sink.of_kind("individual") if "_bin_" in cls}
    assert "fall outside all bin boundaries" in caplog.text
    assert _values(sink) >= {"price.1", "price.7", "price.99"}
    assert binned == {"price.1", "price.7"}
    assert _literal_subjects(sink) == {"price.1", "price.7", "price.99"}


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
                IndividualKind.OBJECT,
            )

    warnings = [r for r in caplog.records if "too few distinct values" in r.message]
    assert _values(sink) == {"price.4"}
    assert len(warnings) == 1


def test_a_number_under_individuals_always_carries_its_literal(recording_sink):
    settings = OntoBuildSettings(**BIN_SETTINGS)
    sink = recording_sink(settings.profile)
    rows = pd.DataFrame({"oid": ["o1", "o2", "o3"], "price": [1.0, 7.0, 99.0]})

    _price_emitter(sink, settings, LOW_HIGH).emit_columns(
        rows, "oid", IndividualKind.OBJECT
    )

    assert _literal_subjects(sink) == {"price.1", "price.7", "price.99"}


# --------------------------------------------------------------------------
# Literals and bin definitions
# --------------------------------------------------------------------------

VALUES_AND_BINS = BIN_SETTINGS
TWELVE_PRICES = [float(n) for n in range(1, 13)]


def test_literals_puts_the_literal_on_the_entity(tmp_path):
    onto = emit(
        OntoBuildSettings(attribute_representations=LITERALS),
        tmp_path / "literals.ofn",
        data=_order_store(price=[1.5, 2.0]),
    )

    assert ("has_value_price", "o1", "1.5", "decimal") in onto.data_values


def test_literals_alone_emits_no_attribute_class_and_no_has_property(tmp_path):
    onto = emit(
        OntoBuildSettings(attribute_representations=LITERALS),
        tmp_path / "literals.ofn",
        data=_order_store(channel=["web", "shop"]),
    )

    assert "channel" not in onto.declared
    assert "has_channel" not in onto.declared
    assert "has_value_channel" in onto.declared
    assert onto.individuals & {"channel.web", "channel.shop"} == set()


def test_individuals_and_literals_write_the_literal_in_both_places(tmp_path):
    onto = emit(
        OntoBuildSettings(attribute_representations=BOTH),
        tmp_path / "both.ofn",
        data=_order_store(channel=["web"]),
    )

    assert ("has_channel", "o1", "channel.web") in onto.relations
    assert ("has_value_channel", "channel.web", "web", "string") in onto.data_values
    assert ("has_value_channel", "o1", "web", "string") in onto.data_values


@pytest.mark.parametrize(
    ("representations", "domain"),
    [
        (INDIVIDUALS, {"channel"}),
        (LITERALS, {"order"}),
        (BOTH, {"channel", "order"}),
    ],
    ids=["individuals", "literals", "both"],
)
def test_the_value_property_domain_follows_the_selection(
    tmp_path, representations, domain
):
    onto = emit(
        OntoBuildSettings(attribute_representations=representations),
        tmp_path / "domain.ofn",
        data=_order_store(channel=["web"]),
    )

    assert onto.domains["has_value_channel"] == domain


def test_a_text_value_carries_a_string_literal(tmp_path):
    onto = emit(
        OntoBuildSettings(),
        tmp_path / "text.ofn",
        data=_order_store(channel=["web", "shop"]),
    )

    assert ("has_value_channel", "channel.web", "web", "string") in onto.data_values


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_text_literal_holds_the_value_the_log_wrote(tmp_path, backend):
    onto = emit(
        OntoBuildSettings(attribute_representations=BOTH),
        tmp_path / "source.owl",
        backend=backend,
        data=_order_store(unit=["unit price"]),
    )

    assert ("has_unit", "o1", "unit.unit_price") in onto.relations
    assert {
        (subject, value)
        for prop, subject, value, _t in onto.data_values
        if prop == "has_value_unit"
    } == {("unit.unit_price", "unit price"), ("o1", "unit price")}


def test_two_values_that_clean_to_one_name_keep_their_own_literals(tmp_path):
    onto = emit(
        OntoBuildSettings(),
        tmp_path / "merged.ofn",
        data=_order_store(channel=["a b", "a_b"]),
    )

    literals = {
        subject: value
        for prop, subject, value, _t in onto.data_values
        if prop == "has_value_channel"
    }
    assert len(literals) == 2
    assert sorted(literals.values()) == ["a b", "a_b"]


def test_a_control_character_in_a_value_becomes_the_replacement_character(
    tmp_path,
):
    onto = emit(
        OntoBuildSettings(attribute_representations=LITERALS),
        tmp_path / "control.ofn",
        data=_order_store(note=["bad\x01note"]),
    )

    assert ("has_value_note", "o1", "bad�note", "string") in onto.data_values


def test_an_attribute_value_property_is_below_has_data_value(tmp_path):
    onto = emit(
        OntoBuildSettings(), tmp_path / "price.ofn", data=_order_store(price=[1.5])
    )

    assert onto.property_parents["has_value_price"] == {vocab.HAS_DATA_VALUE}
    assert "has_value_price" in onto.functional


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

    assert onto.ranges["has_value_x"] == {datatype}


def test_an_integer_and_a_decimal_column_give_a_decimal_range(tmp_path):
    onto = emit(
        OntoBuildSettings(), tmp_path / "widened.ofn", data=_two_kind_store(2, 2.5)
    )

    assert onto.ranges["has_value_weight"] == {"decimal"}


def test_mixed_datatypes_give_no_range(caplog):
    store = _order_store(x=["web", 2])

    with caplog.at_level(logging.WARNING):
        datatypes = attribute_datatypes(store, OntoBuildSettings())

    assert "x" not in datatypes
    assert "'x' holds values of more than one datatype" in caplog.text


def test_a_bin_is_defined_by_its_edges_under_dl(recording_sink):
    sink = recording_sink(OWLProfile.OWL2DL)
    OCEDOntoBuilder(
        DEFAULT_IRI, OntoBuildSettings(**VALUES_AND_BINS), sink=sink
    ).build_ontology(_order_store(price=TWELVE_PRICES))

    definitions = {call[1]: call[2:] for call in sink.of_kind("bin_definition")}
    assert definitions == {
        "price_bin_low": ("price", "has_value_price", None, 6.5),
        "price_bin_high": ("price", "has_value_price", 6.5, None),
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
        "price_bin_low",
        "price_bin_high",
    }


@pytest.mark.reasoner
def test_a_reasoner_places_an_unasserted_value_in_its_bin(tmp_path):
    path = tmp_path / "bins.ofn"
    emit(
        OntoBuildSettings(**VALUES_AND_BINS),
        path,
        data=_order_store(price=TWELVE_PRICES),
    )
    extra = (
        "ClassAssertion(:price value:probe)\n"
        'DataPropertyAssertion(:has_value_price value:probe "9.5"^^xsd:decimal)\n'
    )
    text = path.read_text(encoding="utf-8").rstrip()
    path.write_text(f"{text[:-1]}{extra})\n", encoding="utf-8")

    reasoner = Reasoner(str(path))
    reasoner.materialize(property_assertions=False)
    reasoner.save(str(tmp_path / "materialized.ofn"))

    types = read_functional(tmp_path / "materialized.ofn").types["probe"]
    assert "price_bin_high" in types
    assert "price_bin_low" not in types
