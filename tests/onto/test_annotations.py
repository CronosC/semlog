"""The labels and comments of the upper schema."""

import pytest

from semanticlog.core.enums import InverseRepresentation
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.annotations import TBOX_ANNOTATION_DOCUMENT, tbox_annotations
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings
from tests.conftest import emit
from tests.support.settings import Configuration
from tests.support.tbox import construct_tbox


@pytest.fixture
def full_tbox_names() -> set[str]:
    settings = OntoBuildSettings(
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ]
    )
    semlog = construct_tbox(DEFAULT_IRI, settings, attribute_history=True)
    onto = semlog.require_onto()
    return {
        entity.iri.remainder
        for entity in [
            *onto.classes_in_signature(),
            *onto.object_properties_in_signature(),
            *onto.data_properties_in_signature(),
        ]
    }


def test_the_document_is_installed_alongside_the_package():
    document = tbox_annotations()

    assert document, f"{TBOX_ANNOTATION_DOCUMENT} loaded empty"
    assert all(
        isinstance(entry, dict) and (entry.keys() <= {"label", "comment"})
        for entry in document.values()
    ), "every entry must be a label/comment mapping"


def test_every_annotated_name_exists_in_the_schema(full_tbox_names):
    annotated = set(tbox_annotations())

    orphaned = annotated - full_tbox_names
    assert not orphaned, f"annotated but never declared: {sorted(orphaned)}"


def test_a_supplied_document_reaches_the_build_as_annotations(tmp_path):
    inputs = BuildInputs(annotations=tbox_annotations())

    onto = emit(
        Configuration(OntoBuildSettings(order_representations=[]), inputs),
        tmp_path / "annotated.ofn",
    )

    assert ("Event", "label", tbox_annotations()["Event"]["label"]) in onto.annotations
