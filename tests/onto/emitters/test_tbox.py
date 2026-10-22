"""`TBoxEmitter`: the upper schema, which the settings alone decide."""

from __future__ import annotations

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import InverseRepresentation, OCELExtension
from semanticlog.core.vocabulary import DEFAULT_IRI
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
