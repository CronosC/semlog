"""`ProvenanceEmitter`: the build record, written as annotations on the ontology."""

from __future__ import annotations

import dataclasses
import json

import pytest

from semanticlog import __version__
from semanticlog.core.enums import EventOrderRepresentation, OWLProfile
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.emitters.provenance import build_configuration
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import StreamingSink
from tests.support.rdf import read_rdf


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
    OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(
        tiny_store, source="log.jsonocel"
    )

    text = path.read_text(encoding="utf-8")
    assert 'Annotation(<https://example.com/schema#sourceLog> "log.jsonocel")' in text
    assert "Declaration(AnnotationProperty(oced:sourceLog))" in text


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
    config = build_configuration(OntoBuildSettings())

    assert set(config) == {f.name for f in dataclasses.fields(OntoBuildSettings)}


@pytest.mark.parametrize(
    ("source", "expected"), [("logs/demo.jsonocel", "logs/demo.jsonocel"), (None, None)]
)
def test_the_source_log_is_recorded_when_it_is_known(record, source, expected):
    assert record(source=source).get("sourceLog") == expected


def test_each_run_record_entry_becomes_one_annotation(record):
    written = record(
        run={
            "sourceLogSha256": "abc123",
            "buildInputStatistics": {"b": 2, "a": 1},
            "buildConfigFile": None,
        }
    )

    assert written["sourceLogSha256"] == "abc123"
    assert written["buildInputStatistics"] == '{"a": 1, "b": 2}'
    assert "buildConfigFile" not in written


def test_a_run_record_key_that_cannot_be_a_local_name_is_rejected(tiny_store):
    builder = OCEDOntoBuilder(DEFAULT_IRI, OntoBuildSettings())

    with pytest.raises(ValueError, match="Run record key 'bad key' cannot be an IRI"):
        builder.build_ontology(tiny_store, run={"bad key": "x"})
