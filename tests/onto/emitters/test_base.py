"""The base class of every emitter."""

import pandas as pd

from semlog.core.enums import IndividualKind, InverseRepresentation
from semlog.onto.annotations import AnnotationWriter
from semlog.onto.emitted import EmittedIndividuals
from semlog.onto.emitters.base import Emitter
from semlog.onto.settings import OntoBuildSettings

PAIRS = pd.DataFrame({"eid": ["e1", "e2"], "oid": ["o1", "o1"]})
KINDS = (IndividualKind.EVENT, IndividualKind.OBJECT)


def _emitter(sink, *inverses: InverseRepresentation) -> Emitter:
    settings = OntoBuildSettings(inverse_representations=inverses)
    return Emitter(
        sink, settings, AnnotationWriter(sink, settings, {}), EmittedIndividuals()
    )


def test_without_implicit_inverses_only_the_property_itself_is_asserted(
    recording_sink,
):
    sink = recording_sink()
    _emitter(sink).assert_relations("concerns", PAIRS, "eid", "oid", KINDS)

    assert sink.of_kind("relation") == [
        ("relation", "concerns", "e1", "o1"),
        ("relation", "concerns", "e2", "o1"),
    ]


def test_implicit_inverses_assert_the_reversed_pairs_under_the_inverse(
    recording_sink,
):
    sink = recording_sink()
    emitter = _emitter(sink, InverseRepresentation.IMPLICIT)
    emitter.assert_relations("concerns", PAIRS, "eid", "oid", KINDS)

    assert [c for c in sink.of_kind("relation") if c[1] == "inv_concerns"] == [
        ("relation", "inv_concerns", "o1", "e1"),
        ("relation", "inv_concerns", "o1", "e2"),
    ]
