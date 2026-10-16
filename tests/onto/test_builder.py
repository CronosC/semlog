"""`OCEDOntoBuilder`: one build per builder, its steps in order, and the annotation document."""

from __future__ import annotations

import dataclasses
import logging

import pandas as pd
import pytest

from semanticlog.core.enums import OCELExtension, OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI, UnsafeLocalNameError
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.builder import (
    BUILD_STEPS,
    BuildContext,
    Emitters,
    OCEDOntoBuilder,
)
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings

NO_ORDER = OntoBuildSettings(order_representations=[])


def _event_store(activity: str = "step", oid: str = "o1", **attributes) -> OCEDStore:
    """One event on one order, not cleaned."""
    data = OCEDStore()
    data.add(
        EntityKind.EVENT,
        activity,
        pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"])}),
        id_col="eid",
        time_col="ts",
    )
    data.add(
        EntityKind.OBJECT,
        "order",
        pd.DataFrame({"oid": [oid], **attributes}),
        id_col="oid",
    )
    data.add_relation(
        RelationKind.E2O,
        "concerns",
        pd.DataFrame({"eid": ["e1"], "oid": [oid]}),
        from_id="eid",
        to_id="oid",
        to_type=["order"],
    )
    return data


# --------------------------------------------------------------------------
# One builder, one ontology
# --------------------------------------------------------------------------


def test_a_second_build_raises_and_adds_nothing(tiny_store, pipeline_store):
    builder = OCEDOntoBuilder(DEFAULT_IRI, OntoBuildSettings())
    builder.build_ontology(tiny_store)

    with pytest.raises(ValueError, match="has produced an ontology"):
        builder.build_ontology(pipeline_store)

    names = {c.iri.remainder for c in builder.sink.onto.classes_in_signature()}
    assert "step" in names
    assert "place_order" not in names


def test_a_build_that_fails_while_emitting_spends_the_builder(tiny_store):
    builder = OCEDOntoBuilder(DEFAULT_IRI, OntoBuildSettings())

    with pytest.raises(ValueError, match="cannot be an IRI local name"):
        builder.build_ontology(_event_store(activity="bad name"))

    with pytest.raises(ValueError, match="has produced an ontology"):
        builder.build_ontology(tiny_store)


@pytest.mark.parametrize(
    ("store", "match"),
    [
        (_event_store(activity="place order"), "'place order' from .* clean_oced_data"),
        (_event_store(status=["in progress"]), "status_in progress"),
        (_event_store(oid="order/1"), "order/1"),
    ],
    ids=["activity", "attribute value", "entity id"],
)
def test_a_store_that_was_not_cleaned_is_rejected(store, match):
    with pytest.raises(UnsafeLocalNameError, match=match):
        OCEDOntoBuilder(DEFAULT_IRI, OntoBuildSettings()).build_ontology(store)


# --------------------------------------------------------------------------
# The steps of a build
# --------------------------------------------------------------------------


def test_every_emitter_is_run_by_a_step(tiny_store):
    touched: set[str] = set()

    class Probe:
        """Fake emitter that records being called."""

        def __init__(self, name: str) -> None:
            self._name = name

        def __getattr__(self, item: str):
            touched.add(self._name)
            return lambda *args, **kwargs: None

    fields = [f.name for f in dataclasses.fields(Emitters)]
    context = BuildContext(
        data=tiny_store,
        inputs=BuildInputs(),
        settings=OntoBuildSettings(),
        source=None,
        record=None,
        attribute_history=True,
    )

    for step in BUILD_STEPS:
        step.run(Emitters(**{name: Probe(name) for name in fields}), context)

    assert touched == set(fields)


def test_the_steps_of_a_build_are_announced_in_order(pipeline_store, caplog):
    with caplog.at_level(logging.INFO, logger="semanticlog.onto.builder"):
        OCEDOntoBuilder(DEFAULT_IRI, NO_ORDER).build_ontology(pipeline_store)

    announced = [
        record.message
        for record in caplog.records
        if record.message.startswith(("Instantiating", "No "))
    ]
    assert announced == [
        "Instantiating the upper schema..",
        "Instantiating event data (3 activities)..",
        "No event order to instantiate..",
        "Instantiating time data..",
        "Instantiating object data (2 object types)..",
        "Instantiating event to object relation data (2 relation types)..",
        "Instantiating object to object relation data (1 relation types)..",
    ]


def test_the_history_step_runs_only_when_the_settings_ask_for_it(caplog):
    data = OCEDStore(ocel_mode=True)
    for kind, name, frame, kwargs in [
        (EntityKind.OBJECT, "order", pd.DataFrame({"oid": ["o1"]}), {"id_col": "oid"}),
        (
            EntityKind.EVENT,
            "step",
            pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"])}),
            {"id_col": "eid", "time_col": "ts"},
        ),
    ]:
        data.add(kind, name, frame, **kwargs)
    data.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1"],
                "ocel:type": ["order"],
                "ocel:timestamp": pd.to_datetime(["2024-01-02"]),
                "ocel:field": ["status"],
                "status": ["shipped"],
            }
        ),
    )
    history = "Instantiating object attribute change history.."
    with_history = NO_ORDER.replace(ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY])

    with caplog.at_level(logging.INFO, logger="semanticlog.onto.builder"):
        OCEDOntoBuilder(DEFAULT_IRI, NO_ORDER).build_ontology(clean_oced_data(data))
        without = caplog.text
        OCEDOntoBuilder(DEFAULT_IRI, with_history).build_ontology(clean_oced_data(data))

    assert history not in without
    assert history in caplog.text


def test_the_unions_el_drops_are_reported_once_for_the_build(awkward_store, caplog):
    builder = OCEDOntoBuilder(DEFAULT_IRI, OntoBuildSettings(profile=OWLProfile.OWL2EL))

    with caplog.at_level(logging.INFO, logger="semanticlog.onto.builder"):
        builder.build_ontology(awkward_store)

    omitted = [r.message for r in caplog.records if "omitted" in r.message]
    dropped = builder.sink.dropped_unions
    assert omitted == [
        f"  omitted {dropped} axioms with a union of classes, which OWL 2 EL does "
        "not allow"
    ]


def test_the_declaration_order_check_finds_a_late_declaration(recording_sink):
    sink = recording_sink()

    sink.assert_individuals("some_class", ["late"])
    sink.add_class("late", source="a class declared too late")

    assert sink.declaration_order_violations() == [("late", "some_class")]


# --------------------------------------------------------------------------
# Stores at the edge of empty
# --------------------------------------------------------------------------


def test_an_empty_store_builds_the_upper_schema_alone():
    result = OCEDOntoBuilder(DEFAULT_IRI, NO_ORDER).build_ontology(
        clean_oced_data(OCEDStore())
    )

    onto = result.require_onto()
    assert {c.iri.remainder for c in onto.classes_in_signature()} == {
        "Object",
        "Event",
        "Attribute",
        "Time",
        "ObjectAttribute",
        "EventAttribute",
    }
    assert list(onto.individuals_in_signature()) == []


def test_a_single_event_with_no_object_builds():
    data = OCEDStore()
    data.add(
        EntityKind.EVENT,
        "step",
        pd.DataFrame({"eid": ["e1"], "ts": pd.to_datetime(["2024-01-01"])}),
        id_col="eid",
        time_col="ts",
    )

    result = OCEDOntoBuilder(DEFAULT_IRI, NO_ORDER).build_ontology(
        clean_oced_data(data)
    )

    individuals = {
        i.iri.remainder for i in result.require_onto().individuals_in_signature()
    }
    assert individuals == {"e1", "20240101T000000_000000"}


# --------------------------------------------------------------------------
# The annotation document
# --------------------------------------------------------------------------


def _annotate(tiny_store, recording_sink, annotations, **settings):
    """Build the tiny store with an annotation document and return what was written."""
    sink = recording_sink()
    OCEDOntoBuilder(
        DEFAULT_IRI, OntoBuildSettings(**settings), sink=sink
    ).build_ontology(tiny_store, BuildInputs(annotations=annotations))
    return {(name, kind, text) for _, name, kind, text in sink.of_kind("annotate")}


def test_a_document_entry_annotates_the_entity_it_names(tiny_store, recording_sink):
    written = _annotate(
        tiny_store,
        recording_sink,
        {"step": {"label": "A step", "comment": "Someone did something."}},
    )

    assert written == {
        ("step", "label", "A step"),
        ("step", "comment", "Someone did something."),
    }


def _texts_of(tiny_store, recording_sink, annotations, name):
    """Build with derived annotations and return every text written for one name."""
    sink = recording_sink()
    OCEDOntoBuilder(
        DEFAULT_IRI, OntoBuildSettings(derive_annotations=True), sink=sink
    ).build_ontology(tiny_store, BuildInputs(annotations=annotations))
    return sorted(
        (kind, text)
        for _, entity, kind, text in sink.of_kind("annotate")
        if entity == name
    )


def test_a_document_entry_replaces_the_derived_label_and_comment(
    tiny_store, recording_sink
):
    texts = _texts_of(
        tiny_store,
        recording_sink,
        {"step": {"label": "A step", "comment": "Someone did something."}},
        "step",
    )

    assert texts == [("comment", "Someone did something."), ("label", "A step")]


def test_a_document_label_keeps_the_derived_comment(tiny_store, recording_sink):
    texts = _texts_of(tiny_store, recording_sink, {"step": {"label": "A step"}}, "step")

    assert texts == [("comment", "Events of type 'step'."), ("label", "A step")]


def test_a_document_entry_naming_nothing_the_build_declared_is_reported(
    tiny_store, recording_sink, caplog
):
    with caplog.at_level(logging.WARNING):
        written = _annotate(
            tiny_store,
            recording_sink,
            {
                "step": {"label": "A step"},
                "a step": {"label": "the name before cleaning"},
                "invoice": {"label": "an object type of another log"},
            },
        )

    assert written == {("step", "label", "A step")}
    assert "Applied 1 of 3 annotation entries" in caplog.text
    assert "2 name nothing this build declared" in caplog.text
    assert "invoice" in caplog.text
