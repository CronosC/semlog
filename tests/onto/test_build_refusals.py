"""What a build refuses, and that it writes nothing when it refuses.

A build audits its input and its settings before emitting anything, so a
problem stops the run rather than leaving half an ontology behind. The DECLARE
constraints are the input most likely to belong to a different log, so most of
the checks are about them.
"""

from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from semanticlog.core.enums import (
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OWLProfile,
)
from semanticlog.core.vocabulary import (
    DEFAULT_IRI,
    NameCollisionError,
    UnsafeLocalNameError,
)
from semanticlog.enrich.declare import DeclareResult
from semanticlog.onto.audit import audit_build
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import StreamingSink
from tests.support import stores
from tests.support.settings import CONFIGURATIONS, Configuration, configured

NS = DEFAULT_IRI


def signature(builder: OCEDOntoBuilder) -> list:
    """Return everything the builder's ontology has been told about."""
    return list(builder.sink.onto.classes_in_signature()) + list(
        builder.sink.onto.object_properties_in_signature()
    )


def declare_settings(**constraint_columns) -> Configuration:
    """Return the DECLARE configuration with the given constraint columns."""
    return with_constraints(
        pd.DataFrame(
            {
                "trace_id": ["o1", "o2"],
                "case_concept": ["order", "order"],
                **constraint_columns,
            }
        )
    )


def with_constraints(constraints: pd.DataFrame) -> Configuration:
    """Return the DECLARE configuration carrying one constraints table."""
    config = CONFIGURATIONS["dl_declare"]()
    return dataclasses.replace(
        config,
        inputs=dataclasses.replace(config.inputs, declare_constraints=constraints),
    )


def build(spec: Configuration | OntoBuildSettings):
    config = configured(spec)
    return OCEDOntoBuilder(NS, config.settings).build_ontology(
        stores.build(stores.MATRIX_STORE), config.inputs
    )


# --------------------------------------------------------------------------
# Nothing is written until the input has been checked
# --------------------------------------------------------------------------


def test_constructing_a_builder_emits_nothing():
    assert signature(OCEDOntoBuilder(NS, OntoBuildSettings())) == []


def test_a_profile_that_cannot_carry_the_settings_writes_no_axioms(tiny_store):
    builder = OCEDOntoBuilder(
        NS,
        OntoBuildSettings(
            profile=OWLProfile.OWL2EL,
            inverse_representations=[InverseRepresentation.AXIOMATIC],
        ),
    )

    with pytest.raises(ValueError, match="AXIOMATIC"):
        builder.build_ontology(tiny_store)

    assert signature(builder) == []


def test_ef_under_el_is_refused_for_the_reasoner_not_for_the_profile(tiny_store):
    """TransitiveObjectProperty is inside OWL 2 EL. ELK not answering for
    property assertions is what leaves the derived pairs unreachable.
    """
    settings = OntoBuildSettings(
        profile=OWLProfile.OWL2EL,
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.EF,
        ],
    )

    with pytest.raises(ValueError, match="does not answer for property assertions"):
        settings.validate()


def test_ef_without_directly_follows_is_refused(tiny_store):
    """EF is a transitivity axiom over directly-follows. Without the pairs
    underneath it there is nothing for a reasoner to close over.
    """
    settings = OntoBuildSettings(
        order_representations=[EventOrderRepresentation.EF],
    )

    with pytest.raises(ValueError, match="derives its pairs by transitivity"):
        settings.validate()


def test_missing_declare_constraints_write_no_axioms(tiny_store):
    builder = OCEDOntoBuilder(
        NS,
        OntoBuildSettings(order_representations=[EventOrderRepresentation.DECLARE]),
    )

    with pytest.raises(ValueError, match="DECLARE"):
        builder.build_ontology(tiny_store, BuildInputs(case_concepts=["order"]))

    assert signature(builder) == []


def test_a_case_concept_the_log_does_not_have_writes_no_axioms(tiny_store):
    builder = OCEDOntoBuilder(
        NS,
        OntoBuildSettings(order_representations=[EventOrderRepresentation.DF]),
    )

    with pytest.raises(ValueError, match="no_such_type"):
        builder.build_ontology(tiny_store, BuildInputs(case_concepts=["no_such_type"]))

    assert signature(builder) == []


def test_a_case_concept_cleaning_renamed_is_reported_under_its_new_name(tiny_store):
    tiny_store.backmap = {**tiny_store.backmap, "order": "ORDER"}
    settings = OntoBuildSettings(order_representations=[EventOrderRepresentation.DF])

    with pytest.raises(ValueError, match="'ORDER' was renamed to 'order'"):
        audit_build(tiny_store, settings, BuildInputs(case_concepts=["ORDER"]))


def test_a_case_concept_no_event_reaches_writes_no_axioms(tiny_store):
    """Flattening needs events, which reach objects through the e2o relations."""
    tiny_store.add_object("courier", "cid", pd.DataFrame({"cid": ["c1"]}))
    builder = OCEDOntoBuilder(
        NS,
        OntoBuildSettings(order_representations=[EventOrderRepresentation.DF]),
    )

    with pytest.raises(ValueError, match="no event-to-object relation reaches it"):
        builder.build_ontology(tiny_store, BuildInputs(case_concepts=["courier"]))

    assert signature(builder) == []


def test_case_concepts_are_left_alone_when_no_order_is_emitted(tiny_store):
    settings = OntoBuildSettings(order_representations=[])

    audit_build(tiny_store, settings, BuildInputs(case_concepts=["no_such_type"]))


def test_an_event_with_no_id_writes_no_axioms(tiny_store):
    """It carries no name to be emitted under, and neither does the next one."""
    table = tiny_store.get_event_table("step")
    rows = table.data.copy()
    rows.loc[1, table.id] = None
    tiny_store.replace_event_data("step", rows)
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())

    with pytest.raises(ValueError, match="1 event\\(s\\) have no 'eid'"):
        builder.build_ontology(tiny_store)

    assert signature(builder) == []


def test_a_rejected_declare_input_leaves_no_streamed_file(tmp_path):
    """Not even the partial one it had already written a schema into."""
    target = tmp_path / "out.ofn"
    config = declare_settings(**{"Response@1": [True, False]})
    sink = StreamingSink(str(target), NS, config.settings.profile)

    with pytest.raises(UnsafeLocalNameError):
        OCEDOntoBuilder(NS, config.settings, sink=sink).build_ontology(
            stores.build(stores.MATRIX_STORE), config.inputs
        )

    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# DECLARE constraints that do not belong to the log
# --------------------------------------------------------------------------


def test_an_unsafe_constraint_name_is_rejected():
    with pytest.raises(UnsafeLocalNameError, match="Response@1"):
        build(declare_settings(**{"Response@1": [True, False]}))


def test_a_constraint_named_after_an_object_type_is_rejected():
    with pytest.raises(NameCollisionError, match="'order'"):
        build(declare_settings(order=[True, False]))


def test_a_trace_id_that_is_not_an_object_is_rejected():
    config = with_constraints(
        pd.DataFrame(
            {
                "trace_id": ["from_another_log", "o1"],
                "case_concept": ["order", "order"],
                "Response__place_order__ship__": [True, False],
            }
        )
    )

    with pytest.raises(ValueError, match="from_another_log"):
        audit_build(stores.build(stores.MATRIX_STORE), config.settings, config.inputs)


def test_a_case_concept_the_log_does_not_have_is_rejected():
    config = with_constraints(
        pd.DataFrame(
            {
                "trace_id": ["x1"],
                "case_concept": ["no_such_type"],
                "Response__place_order__ship__": [True],
            }
        )
    )

    with pytest.raises(ValueError, match="not an object type of this log"):
        audit_build(stores.build(stores.MATRIX_STORE), config.settings, config.inputs)


def test_the_matrix_declare_input_passes_the_audit():
    config = CONFIGURATIONS["dl_declare"]()
    audit_build(stores.build(stores.MATRIX_STORE), config.settings, config.inputs)


def test_constraints_from_another_log_warn_without_failing(caplog):
    from semanticlog.cli.run import warn_on_foreign_declare

    declare = DeclareResult(
        constraints=CONFIGURATIONS["dl_declare"]().inputs.declare_constraints,
        case_concepts=["order"],
        source_log="somewhere/else.xml",
        source_sha256="a" * 64,
    )

    warn_on_foreign_declare(declare, "b" * 64)

    assert "not the file being built" in caplog.text


# --------------------------------------------------------------------------
# The attribute history extension needs a store that can carry history
# --------------------------------------------------------------------------


def test_the_history_schema_is_omitted_for_a_store_without_history(tiny_store, caplog):
    settings = OntoBuildSettings(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])

    semlog = OCEDOntoBuilder(NS, settings).build_ontology(tiny_store)

    declared = {c.iri.remainder for c in semlog.onto.classes_in_signature()}
    assert "AttributeChange" not in declared
    assert "ocel_mode=True" in caplog.text
