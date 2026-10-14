"""What a whole build does when its input fails the audit, and the input checks only a build makes."""

from __future__ import annotations

import pytest

from semanticlog.core.enums import EventOrderRepresentation, OCELExtension
from semanticlog.core.vocabulary import DEFAULT_IRI, NameCollisionError
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings
from tests.support import stores
from tests.support.settings import declare_constraints, settings

NS = DEFAULT_IRI
DF = EventOrderRepresentation.DF


def _declared(builder: OCEDOntoBuilder) -> set[str]:
    return {c.iri.remainder for c in builder.sink.onto.classes_in_signature()}


def test_a_build_that_fails_the_audit_writes_no_axioms(tiny_store):
    builder = OCEDOntoBuilder(NS, OntoBuildSettings(order_representations=[DF]))

    with pytest.raises(ValueError, match="no_such_type"):
        builder.build_ontology(tiny_store, BuildInputs(case_concepts=["no_such_type"]))

    assert _declared(builder) == set()


def test_case_concepts_are_not_checked_when_no_order_is_emitted(tiny_store):
    builder = OCEDOntoBuilder(NS, OntoBuildSettings(order_representations=[]))

    builder.build_ontology(tiny_store, BuildInputs(case_concepts=["no_such_type"]))

    assert "step" in _declared(builder)


def test_a_constraint_named_after_an_object_type_is_rejected():
    config = settings(
        order_representations=[DF, EventOrderRepresentation.DECLARE],
        declare=declare_constraints("order"),
    )

    with pytest.raises(NameCollisionError, match="'order'"):
        OCEDOntoBuilder(NS, config.settings).build_ontology(
            stores.build(stores.MATRIX_STORE), config.inputs
        )


def test_the_history_schema_is_left_out_for_a_store_without_history(tiny_store, caplog):
    builder = OCEDOntoBuilder(
        NS, OntoBuildSettings(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])
    )

    builder.build_ontology(tiny_store)

    assert "AttributeChange" not in _declared(builder)
    assert "not created with ocel_mode=True" in caplog.text
