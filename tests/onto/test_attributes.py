"""Attribute classes, the values under them, and who owns them.

An attribute name carried by more than one entity type becomes one OWL class
with shared value individuals. That is worth knowing about, so the build says
so, and the property domain has to name every owner.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import DataRepresentation
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.binning import Binning
from semanticlog.data.oced import attribute_owners
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.attributes import (
    AttributeEmitter,
    report_conflated_attributes,
)
from semanticlog.onto.settings import OntoBuildSettings
from tests.conftest import emit

NS = DEFAULT_IRI


def test_an_attribute_owned_by_one_type_gets_that_type_as_its_domain(
    tiny_store, tmp_path
):
    onto = emit(OntoBuildSettings(), tmp_path / "tiny.ofn", data=tiny_store)

    assert onto.domains["has_channel"] == {"step"}
    assert onto.ranges["has_channel"] == {"channel"}


def test_a_number_too_wide_for_plain_notation_still_builds(tiny_store, tmp_path):
    """A value `str` would write as `1e+20` reaches the ontology in full.

    The name and the literal both go through the same formatting, so an
    ordinary log with a large amount or a small measurement builds rather than
    failing on a character an IRI cannot hold.
    """
    table = tiny_store.get_event_table("step")
    table.data["price"] = [1.0e20, 2.5, 1.0e-7]

    onto = emit(OntoBuildSettings(), tmp_path / "wide.ofn", data=tiny_store)

    assert "price_100000000000000000000" in onto.individuals
    assert "price_0.0000001" in onto.individuals
    assert ("has_data_value", "price_100000000000000000000") in onto.data_assertions


def test_an_attribute_owned_by_two_types_gets_one_domain_naming_both(
    awkward_store, tmp_path
):
    onto = emit(OntoBuildSettings(), tmp_path / "awkward.ofn", data=awkward_store)
    domain_axioms = [
        line
        for line in onto.axiom_lines
        if line.startswith("ObjectPropertyDomain(:has_channel ")
    ]

    assert onto.domains["has_channel"] == {"order", "item"}
    assert len(domain_axioms) == 1, "the owners belong in one union, not one axiom each"


def test_a_shared_attribute_is_reported(awkward_store, caplog):
    owners = attribute_owners(awkward_store)

    with caplog.at_level(logging.WARNING):
        conflated = report_conflated_attributes(owners)

    assert conflated == {"channel": ["item", "order"], "clerk": ["place_order", "ship"]}
    assert "4 different entities share 2 attribute names" in caplog.text
    assert "channel (item, order)" in caplog.text


def test_the_owners_of_a_widely_shared_attribute_are_counted_not_listed(caplog):
    owners = {"commit_message": [f"activity_{i}" for i in range(68)]}

    with caplog.at_level(logging.WARNING):
        report_conflated_attributes(owners)

    assert "commit_message (68 types)" in caplog.text
    assert "activity_0" not in caplog.text


def test_a_long_list_of_conflated_attributes_is_cut(caplog):
    owners = {f"attr_{i}": ["item", "order"] for i in range(40)}

    with caplog.at_level(logging.WARNING):
        report_conflated_attributes(owners)

    assert "2 different entities share 40 attribute names" in caplog.text
    assert "more)" in caplog.text


def test_an_unshared_attribute_is_not_reported(tiny_store, caplog):
    owners = attribute_owners(tiny_store)

    with caplog.at_level(logging.WARNING):
        assert report_conflated_attributes(owners) == {}

    assert "Attribute conflation" not in caplog.text


@pytest.mark.parametrize("name", ["tiny", "awkward"])
@pytest.mark.reasoner
def test_a_build_with_typed_attributes_is_consistent(name, store, tmp_path):
    from semanticlog.onto.builder import OCEDOntoBuilder
    from semanticlog.onto.reason import Reasoner

    settings = OntoBuildSettings()
    path = tmp_path / f"{name}.owl"
    OCEDOntoBuilder(NS, settings).build_ontology(store(name)).save(str(path))

    assert Reasoner(str(path), profile=settings.profile).check() is True


# --------------------------------------------------------------------------
# A number the bins cannot place
# --------------------------------------------------------------------------

BIN_SETTINGS = dict(
    data_representations=[DataRepresentation.EQUAL_BINS],
    bins=2,
    bin_labels=["low", "high"],
)


def _binned_emitter(
    sink, settings: OntoBuildSettings, binning: Binning | None
) -> AttributeEmitter:
    """An emitter for one binned object attribute, `price`, on `order`."""
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


def _literals(sink) -> set[str]:
    return {subject for _, _prop, subject, _v in sink.of_kind("data")}


def test_a_value_outside_every_bin_is_emitted_without_its_number(
    recording_sink, caplog
):
    """It has no bin class, so it is reachable by name and by its attribute."""
    settings = OntoBuildSettings(**BIN_SETTINGS)
    sink = recording_sink(settings.profile)
    emitter = _binned_emitter(
        sink, settings, Binning(labels=("low", "high"), edges=(0.0, 5.0, 10.0))
    )
    rows = pd.DataFrame({"oid": ["o1", "o2", "o3"], "price": [1.0, 7.0, 99.0]})

    with caplog.at_level(logging.WARNING):
        emitter.emit_columns(rows, "oid", vocab.OBJECT_ATTRIBUTE)

    assert "fall outside all bin boundaries" in caplog.text
    assert "price_99.0" in _values(sink)
    assert {"price_bin_low", "price_bin_high"} <= {
        cls for _, _n, cls in sink.of_kind("individual")
    }
    assert _literals(sink) == set(), "a build that bins keeps no raw numbers"


def test_a_column_too_narrow_to_bin_is_emitted_without_its_number(
    recording_sink, caplog
):
    settings = OntoBuildSettings(**BIN_SETTINGS)
    sink = recording_sink(settings.profile)
    emitter = _binned_emitter(sink, settings, None)
    rows = pd.DataFrame({"oid": ["o1", "o2"], "price": [4.0, 4.0]})

    with caplog.at_level(logging.WARNING):
        emitter.emit_columns(rows, "oid", vocab.OBJECT_ATTRIBUTE)

    assert "too few distinct values to bin" in caplog.text
    assert _values(sink) == {"price_4.0"}
    assert _literals(sink) == set()


def test_the_same_values_keep_their_numbers_when_the_build_asks_for_them(
    recording_sink,
):
    """VALUES alongside a binning mode is what puts the literal back."""
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
    emitter = _binned_emitter(
        sink, settings, Binning(labels=("low", "high"), edges=(0.0, 5.0, 10.0))
    )
    rows = pd.DataFrame({"oid": ["o1", "o2", "o3"], "price": [1.0, 7.0, 99.0]})

    emitter.emit_columns(rows, "oid", vocab.OBJECT_ATTRIBUTE)

    assert _literals(sink) == {"price_1.0", "price_7.0", "price_99.0"}
