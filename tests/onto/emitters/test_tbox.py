"""The fixed upper schema."""

import pytest
from owlapy.iri import IRI
from owlapy.owl_ontology import OWLClass
from owlapy.owl_property import OWLDataProperty, OWLObjectProperty

from semanticlog.core import vocabulary as vocab
from semanticlog.onto.builder import construct_tbox
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    InverseRepresentation,
    OntoBuildSettings,
)

NS = DEFAULT_IRI


TEST_IRI = "https://test.example.org/upper#"

TOP_LEVEL_CLASSES = ["Object", "Event", "Attribute", "Time"]
SUB_ATTRIBUTE_CLASSES = ["ObjectAttribute", "EventAttribute"]
CORE_OBJECT_PROPERTIES = [
    "observed_at",
    "event_object_relation",
    "object_object_relation",
    "has_attribute",
]
NAMED_INVERSES = ["observes", "attribute_of"]
AUTO_NAMED_INVERSES = ["inv_event_object_relation", "inv_object_object_relation"]
DATA_PROPERTIES = ["has_data_value"]


def _owl_class(name: str) -> OWLClass:
    return OWLClass(IRI(TEST_IRI, name))


def _obj_prop(name: str) -> OWLObjectProperty:
    return OWLObjectProperty(IRI(TEST_IRI, name))


def _data_prop(name: str) -> OWLDataProperty:
    return OWLDataProperty(IRI(TEST_IRI, name))


@pytest.fixture
def oced_with_inverses():
    return construct_tbox(
        TEST_IRI,
        OntoBuildSettings(
            inverse_representations=[
                InverseRepresentation.AXIOMATIC,
                InverseRepresentation.IMPLICIT,
            ]
        ),
    )


@pytest.fixture
def oced_without_inverses():
    return construct_tbox(TEST_IRI, OntoBuildSettings(inverse_representations=[]))


def obj_prop_names(semlog) -> set[str]:
    return {p.iri.remainder for p in semlog.onto.object_properties_in_signature()}


def test_top_level_classes_all_present(oced_with_inverses):
    classes = list(oced_with_inverses.onto.classes_in_signature())
    for name in TOP_LEVEL_CLASSES:
        assert _owl_class(name) in classes, f"Expected class '{name}' in signature"


def test_sub_attribute_classes_present(oced_with_inverses):
    classes = list(oced_with_inverses.onto.classes_in_signature())
    for name in SUB_ATTRIBUTE_CLASSES:
        assert _owl_class(name) in classes, f"Expected class '{name}' in signature"


def test_core_object_properties_present(oced_with_inverses):
    obj_props = list(oced_with_inverses.onto.object_properties_in_signature())
    for name in CORE_OBJECT_PROPERTIES:
        assert _obj_prop(name) in obj_props, f"Expected object property '{name}'"


def test_data_property_has_data_value_present(oced_with_inverses):
    data_props = list(oced_with_inverses.onto.data_properties_in_signature())
    assert _data_prop("has_data_value") in data_props


def test_named_inverse_observes_present(oced_with_inverses):
    assert "observes" in obj_prop_names(oced_with_inverses)


def test_named_inverse_attribute_of_present(oced_with_inverses):
    assert "attribute_of" in obj_prop_names(oced_with_inverses)


def test_auto_named_inverses_present(oced_with_inverses):
    names = obj_prop_names(oced_with_inverses)
    for name in AUTO_NAMED_INVERSES:
        assert name in names, f"Expected auto-named inverse '{name}'"


def test_no_inverse_properties_when_disabled(oced_without_inverses):
    names = obj_prop_names(oced_without_inverses)
    for name in NAMED_INVERSES + AUTO_NAMED_INVERSES:
        assert name not in names, f"Unexpected inverse property '{name}' when disabled"


def test_a_path_writes_the_schema_to_disk(tmp_path):
    out = tmp_path / "upper.owl"
    construct_tbox(TEST_IRI, OntoBuildSettings(), str(out))
    assert out.stat().st_size > 0


def test_iri_ends_with_hash(oced_with_inverses):
    assert oced_with_inverses.iri.endswith("#")


@pytest.fixture
def oced_with_attribute_history():
    from semanticlog.onto.settings import OCELExtension

    return construct_tbox(
        TEST_IRI,
        OntoBuildSettings(
            ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY],
            inverse_representations=[],
        ),
    )


def test_attribute_history_adds_attribute_change_class(oced_with_attribute_history):
    classes = list(oced_with_attribute_history.onto.classes_in_signature())
    assert _owl_class("AttributeChange") in classes


def test_attribute_history_adds_has_attribute_change_property(
    oced_with_attribute_history,
):
    assert "has_attribute_change" in obj_prop_names(oced_with_attribute_history)


def test_attribute_history_adds_attribute_change_value_property(
    oced_with_attribute_history,
):
    assert "attribute_change_value" in obj_prop_names(oced_with_attribute_history)


def test_attribute_history_adds_changed_at_property(oced_with_attribute_history):
    assert "changed_at" in obj_prop_names(oced_with_attribute_history)


def test_no_extension_does_not_add_attribute_change_class(oced_with_inverses):
    classes = list(oced_with_inverses.onto.classes_in_signature())
    assert _owl_class("AttributeChange") not in classes


def test_the_upper_ontology_declares_exactly_the_names_the_vocabulary_reserves():
    onto = construct_tbox(NS, OntoBuildSettings(case_concepts=["order"])).onto

    classes = {c.iri.remainder for c in onto.classes_in_signature()}
    assert classes == {
        vocab.OBJECT,
        vocab.EVENT,
        vocab.ATTRIBUTE,
        vocab.TIME,
        vocab.OBJECT_ATTRIBUTE,
        vocab.EVENT_ATTRIBUTE,
    }

    properties = {p.iri.remainder for p in onto.object_properties_in_signature()}
    assert properties == {
        vocab.OBSERVED_AT,
        vocab.EVENT_OBJECT_RELATION,
        vocab.OBJECT_OBJECT_RELATION,
        vocab.HAS_ATTRIBUTE,
    }

    data_properties = {p.iri.remainder for p in onto.data_properties_in_signature()}
    assert data_properties == {vocab.HAS_DATA_VALUE}
