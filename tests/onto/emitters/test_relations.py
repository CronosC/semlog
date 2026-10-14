from __future__ import annotations

import pytest

from semanticlog.core.enums import OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.tables import RelationKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters import RelationEmitter
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.recording import RecordingSink


def test_emitting_both_relation_families_at_once_is_rejected(tiny_store):
    sink = RecordingSink(OWLProfile.OWL2DL, DEFAULT_IRI)
    settings = OntoBuildSettings()
    emitter = RelationEmitter(
        sink, settings, AnnotationWriter(sink, settings, {}), EmittedIndividuals()
    )

    with pytest.raises(ValueError, match="both relation families"):
        emitter.emit(tiny_store, RelationKind.ALL)
