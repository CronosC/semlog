"""`BuildRecordEmitter`: the build record, written as annotations on the ontology."""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

from semlog import __version__
from semlog.core.enums import EventOrderRepresentation, OWLProfile
from semlog.core.vocabulary import DEFAULT_IRI
from semlog.onto.builder import OCEDOntoBuilder
from semlog.onto.inputs import BuildInputs
from semlog.onto.settings import OntoBuildSettings
from semlog.onto.sinks import StreamingSink
from tests.support.build import BACKENDS, STREAMING, emit
from tests.support.rdf import read_rdf

HEADER = {"buildTool", "buildConfiguration", "caseConcepts"}


@pytest.fixture
def record(tiny_store, tmp_path):
    """Return a function that builds the tiny store and returns the ontology annotations."""

    def build(settings: OntoBuildSettings | None = None, **kwargs) -> dict[str, str]:
        builder = OCEDOntoBuilder(DEFAULT_IRI, settings or OntoBuildSettings())
        builder.build_ontology(tiny_store, **kwargs)
        path = tmp_path / "prov.owl"
        builder.sink.save(str(path))
        return read_rdf(path, DEFAULT_IRI).ontology_annotations

    return build


def test_the_record_names_the_tool_and_its_version(record):
    assert record()["buildTool"] == f"semlog {__version__}"


def test_the_record_has_no_version_info(record):
    assert "versionInfo" not in record()


def test_the_record_keys_use_the_configured_tbox_iri(tiny_store, tmp_path):
    path = tmp_path / "prov.ofn"
    settings = OntoBuildSettings(tbox_iri="https://example.com/schema#")
    sink = StreamingSink(str(path), DEFAULT_IRI, settings.profile, settings.tbox_iri)
    OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(tiny_store)

    text = path.read_text(encoding="utf-8")
    assert 'Annotation(<https://example.com/schema#caseConcepts> "[]")' in text
    assert "Declaration(AnnotationProperty(oced:caseConcepts))" in text


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_header_holds_only_the_tool_the_settings_and_the_case_concepts(
    tmp_path, backend
):
    path = tmp_path / ("prov.ofn" if backend == STREAMING else "prov.owl")
    written = emit(OntoBuildSettings(), path, backend)

    declared = re.findall(
        r'AnnotationProperty\(oced:(\w+)\)|AnnotationProperty rdf:about="[^"]*#(\w+)"',
        path.read_text(encoding="utf-8"),
    )
    assert set(written.ontology_annotations) == HEADER
    assert {functional or rdf for functional, rdf in declared} == HEADER


def test_the_resolved_settings_are_recorded_as_json(record):
    settings = OntoBuildSettings(
        profile=OWLProfile.OWL2EL,
        order_representations=[EventOrderRepresentation.DF],
    )

    written = json.loads(
        record(settings, inputs=BuildInputs(case_concepts=["order"]))[
            "buildConfiguration"
        ]
    )

    assert written["profile"] == "OWL2EL"
    assert written["order_representations"] == ["df"]


def test_the_recorded_configuration_holds_every_setting():
    config = OntoBuildSettings().to_dict()

    assert set(config) == {f.name for f in dataclasses.fields(OntoBuildSettings)}


def test_the_case_concepts_are_recorded_as_json(record):
    written = record(inputs=BuildInputs(case_concepts=["order"]))

    assert json.loads(written["caseConcepts"]) == ["order"]
