"""The fixed upper schema every build starts from.

What the schema holds is decided by the settings alone, so these build it
without any data. The exhaustive checks compare it against the vocabulary,
which is the list of names a build may emit.
"""

from __future__ import annotations

import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import (
    InverseRepresentation,
    OCELExtension,
)
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.tbox import construct_tbox

TEST_IRI = "https://test.example.org/upper#"

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
NAMED_INVERSES = {"observes", "attribute_of"}
AUTO_NAMED_INVERSES = {"inv_event_object_relation", "inv_object_object_relation"}
HISTORY_PROPERTIES = {"has_attribute_change", "attribute_change_value", "changed_at"}


def schema(iri: str = TEST_IRI, **overrides):
    """Return the upper schema the given settings ask for."""
    return construct_tbox(iri, OntoBuildSettings(**overrides))


def classes(semlog) -> set[str]:
    return {c.iri.remainder for c in semlog.onto.classes_in_signature()}


def object_properties(semlog) -> set[str]:
    return {p.iri.remainder for p in semlog.onto.object_properties_in_signature()}


def data_properties(semlog) -> set[str]:
    return {p.iri.remainder for p in semlog.onto.data_properties_in_signature()}


def test_the_schema_declares_exactly_the_names_the_vocabulary_reserves():
    built = schema(DEFAULT_IRI)

    assert classes(built) == BASE_CLASSES
    assert object_properties(built) == BASE_OBJECT_PROPERTIES
    assert data_properties(built) == {vocab.HAS_DATA_VALUE}


def test_inverses_add_the_named_and_the_auto_named_ones():
    built = schema(
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ]
    )

    assert (
        object_properties(built)
        == BASE_OBJECT_PROPERTIES | NAMED_INVERSES | AUTO_NAMED_INVERSES
    )


def test_no_inverse_property_is_declared_when_inverses_are_off():
    built = schema(inverse_representations=[])

    assert not object_properties(built) & (NAMED_INVERSES | AUTO_NAMED_INVERSES)


def test_the_attribute_history_extension_adds_its_own_schema():
    built = schema(
        ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY], inverse_representations=[]
    )

    assert classes(built) == BASE_CLASSES | {"AttributeChange"}
    assert object_properties(built) == BASE_OBJECT_PROPERTIES | HISTORY_PROPERTIES


def test_no_history_schema_is_declared_without_the_extension():
    assert "AttributeChange" not in classes(schema())


def test_the_base_iri_is_made_to_end_in_a_hash():
    assert schema("https://test.example.org/upper").iri.endswith("#")


def test_a_path_writes_the_schema_to_disk(tmp_path):
    out = tmp_path / "upper.owl"

    construct_tbox(TEST_IRI, OntoBuildSettings(), str(out))

    assert out.stat().st_size > 0


@pytest.mark.parametrize("name", sorted(BASE_CLASSES | BASE_OBJECT_PROPERTIES))
def test_every_schema_name_is_reserved_against_the_log(name):
    assert name in vocab.RESERVED_NAMES
