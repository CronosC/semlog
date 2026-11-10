"""The relations of the log, as properties between individuals."""

from __future__ import annotations

import pytest

from semanticlog.core.enums import OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.tables import RelationKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters import RelationEmitter
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.build import emit
from tests.support.recording import RecordingSink
from tests.support.settings import settings


def test_emitting_both_relation_families_at_once_is_rejected(tiny_store):
    sink = RecordingSink(OWLProfile.OWL2DL, DEFAULT_IRI)
    settings = OntoBuildSettings()
    emitter = RelationEmitter(
        sink, settings, AnnotationWriter(sink, settings, {}), EmittedIndividuals()
    )

    with pytest.raises(ValueError, match="both relation families"):
        emitter.emit(tiny_store, RelationKind.ALL)


def test_an_e2o_domain_is_the_union_of_its_source_event_types(tmp_path):
    onto = emit(settings(), tmp_path / "matrix.ofn")

    assert onto.domains["handles"] == {"place_order", "ship"}


def test_an_e2o_used_by_one_event_type_has_that_type_as_domain(tiny_store, tmp_path):
    onto = emit(settings(), tmp_path / "tiny.ofn", data=tiny_store)

    assert onto.domains["concerns"] == {"step"}


def test_a_union_domain_falls_back_to_event_under_el(tmp_path):
    onto = emit(settings(profile=OWLProfile.OWL2EL), tmp_path / "matrix.ofn")

    assert onto.domains["handles"] == {"Event"}


def test_an_e2o_comment_names_its_domain_classes(tiny_store):
    sink = RecordingSink(OWLProfile.OWL2DL, DEFAULT_IRI)
    settings = OntoBuildSettings(derive_annotations=True)
    emitter = RelationEmitter(
        sink, settings, AnnotationWriter(sink, settings, {}), EmittedIndividuals()
    )

    emitter.emit(tiny_store, RelationKind.E2O)

    comments = {
        name: text
        for _, name, kind, text in sink.of_kind("annotate")
        if kind == "comment"
    }
    assert comments["concerns"] == "Relation 'concerns' from step to order."
