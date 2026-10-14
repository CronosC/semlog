import pandas as pd
import pytest

from semanticlog.core.enums import InverseRepresentation
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.settings import OntoBuildSettings

PAIRS = pd.DataFrame({"eid": ["e1", "e2"], "oid": ["o1", "o1"]})


def _emitter(sink, *inverses: InverseRepresentation) -> Emitter:
    settings = OntoBuildSettings(inverse_representations=inverses)
    return Emitter(
        sink, settings, AnnotationWriter(sink, settings, {}), EmittedIndividuals()
    )


def test_without_implicit_inverses_only_the_property_itself_is_asserted(
    recording_sink,
):
    sink = recording_sink()
    _emitter(sink).assert_relations("concerns", PAIRS, "eid", "oid")

    assert sink.of_kind("relation") == [
        ("relation", "concerns", "e1", "o1"),
        ("relation", "concerns", "e2", "o1"),
    ]


@pytest.mark.parametrize(
    ("inverse_name", "inverse"),
    [(None, "inv_concerns"), ("concerned_by", "concerned_by")],
)
def test_implicit_inverses_assert_the_reversed_pairs_under_the_inverse(
    recording_sink, inverse_name, inverse
):
    sink = recording_sink()
    emitter = _emitter(sink, InverseRepresentation.IMPLICIT)
    emitter.assert_relations("concerns", PAIRS, "eid", "oid", inverse_name)

    assert [c for c in sink.of_kind("relation") if c[1] == inverse] == [
        ("relation", inverse, "o1", "e1"),
        ("relation", inverse, "o1", "e2"),
    ]
