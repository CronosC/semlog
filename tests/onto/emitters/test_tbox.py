"""`TBoxEmitter`: the upper schema, which the settings alone decide."""

from __future__ import annotations

import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import InverseRepresentation, OCELExtension
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.build import emit
from tests.support.tbox import construct_tbox

BASE_CLASSES = {
    vocab.OBJECT,
    vocab.EVENT,
    vocab.ATTRIBUTE,
    vocab.TIME,
    vocab.OBJECT_ATTRIBUTE,
    vocab.EVENT_ATTRIBUTE,
}
BASE_OBJECT_PROPERTIES = {
    vocab.OBSERVED_AT,
    vocab.EVENT_OBJECT_RELATION,
    vocab.OBJECT_OBJECT_RELATION,
    vocab.HAS_ATTRIBUTE,
}


def _schema(**overrides) -> dict[str, set[str]]:
    """Return the names the upper schema declares, by kind."""
    onto = construct_tbox(DEFAULT_IRI, OntoBuildSettings(**overrides)).require_onto()
    return {
        "classes": {c.iri.remainder for c in onto.classes_in_signature()},
        "object": {p.iri.remainder for p in onto.object_properties_in_signature()},
        "data": {p.iri.remainder for p in onto.data_properties_in_signature()},
    }


def test_the_schema_declares_the_base_classes_and_properties():
    assert _schema() == {
        "classes": BASE_CLASSES,
        "object": BASE_OBJECT_PROPERTIES,
        "data": {vocab.HAS_DATA_VALUE},
    }


def test_inverses_add_the_named_and_the_derived_inverse_properties():
    schema = _schema(
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ]
    )

    assert schema["object"] == BASE_OBJECT_PROPERTIES | {
        "observes",
        "attribute_of",
        "inv_event_object_relation",
        "inv_object_object_relation",
    }


def test_the_attribute_history_extension_adds_its_class_and_properties():
    schema = _schema(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])

    assert schema["classes"] == BASE_CLASSES | {"AttributeChange"}
    assert schema["object"] == BASE_OBJECT_PROPERTIES | {
        "has_attribute_change",
        "attribute_change_value",
        "changed_at",
    }


def test_every_name_the_upper_schema_emits_is_in_the_tbox_namespace():
    settings = OntoBuildSettings(
        ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY],
        inverse_representations=[InverseRepresentation.AXIOMATIC],
    )
    onto = construct_tbox(DEFAULT_IRI, settings).require_onto()
    entities = [
        *onto.classes_in_signature(),
        *onto.object_properties_in_signature(),
        *onto.data_properties_in_signature(),
    ]

    assert [e for e in entities if not e.iri.str.startswith(vocab.TBOX_IRI)] == []


def test_the_top_classes_are_pairwise_disjoint(tiny_store, tmp_path):
    settings = OntoBuildSettings(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])
    onto = emit(settings, tmp_path / "tiny.ofn", data=tiny_store)

    assert {
        frozenset({"Object", "Event", "Attribute", "Time"}),
        frozenset({"EventAttribute", "ObjectAttribute"}),
    } <= onto.disjoint


def test_the_time_and_attribute_properties_are_functional_under_dl(
    tiny_store, tmp_path
):
    onto = emit(OntoBuildSettings(), tmp_path / "tiny.ofn", data=tiny_store)

    assert {"observed_at", "has_data_value", "has_step.channel"} <= onto.functional


def test_an_implicit_inverse_of_a_functional_property_is_inverse_functional(
    tiny_store, tmp_path
):
    implicit = OntoBuildSettings(
        inverse_representations=[InverseRepresentation.IMPLICIT]
    )
    axiomatic = OntoBuildSettings(
        inverse_representations=[InverseRepresentation.AXIOMATIC]
    )

    with_twins = emit(implicit, tmp_path / "implicit.ofn", data=tiny_store)
    without = emit(axiomatic, tmp_path / "axiomatic.ofn", data=tiny_store)

    assert "observes" in with_twins.inverse_functional
    assert without.inverse_functional == set()


@pytest.mark.reasoner
def test_an_event_observed_at_two_times_is_inconsistent(tiny_store, tmp_path):
    path = tmp_path / "tiny.ofn"
    onto = emit(OntoBuildSettings(), path, data=tiny_store)
    times = {s: o for p, s, o in onto.relations if p == "observed_at"}

    text = path.read_text(encoding="utf-8").rstrip()
    extra = f"ObjectPropertyAssertion(oced:observed_at :e1 :{times['e2']})"
    path.write_text(f"{text[:-1]}{extra}\n)\n", encoding="utf-8")

    assert Reasoner(str(path)).check() is False
