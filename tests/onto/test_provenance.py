"""The record of a build, annotated onto the ontology it produced.

The `.owl` file is the product, so it has to say what made it: the version,
the resolved settings, the log it came from and whatever the caller passed as
the run record.
"""

from __future__ import annotations

import dataclasses
import json

import pytest

from semanticlog.core.enums import (
    EventOrderRepresentation,
    OWLProfile,
)
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.emitters.provenance import build_configuration
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings
from tests.conftest import emit

NS = DEFAULT_IRI


@pytest.fixture
def record(tiny_store, tmp_path):
    """Build the tiny store and return the ontology annotations of the result."""

    def build(settings: OntoBuildSettings | None = None, **kwargs) -> dict[str, str]:
        builder = OCEDOntoBuilder(NS, settings or OntoBuildSettings())
        builder.build_ontology(tiny_store, **kwargs)
        path = tmp_path / "prov.owl"
        builder.sink.save(str(path))
        from tests.support.ontology import read_rdf

        return read_rdf(path, NS).ontology_annotations

    return build


def test_the_semlog_version_is_recorded(record):
    from semanticlog import __version__

    assert record()["versionInfo"] == f"semlog {__version__}"


def test_the_resolved_configuration_is_recorded_as_readable_json(record):
    settings = OntoBuildSettings(
        profile=OWLProfile.OWL2EL,
        order_representations=[EventOrderRepresentation.DF],
    )
    inputs = BuildInputs(case_concepts=["order"])

    written = json.loads(record(settings, inputs=inputs)["buildConfiguration"])

    assert written["profile"] == "OWL2EL"
    assert written["order_representations"] == ["df"]


def test_the_source_log_is_recorded_when_it_is_known(record):
    assert record(source="logs/demo.jsonocel")["sourceLog"] == "logs/demo.jsonocel"


def test_the_source_log_is_left_out_when_it_is_not(record):
    assert "sourceLog" not in record()


def test_the_configuration_records_every_setting(record):
    """A setting left out of the record is one a reader cannot reproduce."""
    config = build_configuration(OntoBuildSettings())

    assert set(config) == {f.name for f in dataclasses.fields(OntoBuildSettings)}


def test_each_entry_of_the_run_record_becomes_one_annotation(record):
    written = record(
        run={"sourceLogSha256": "abc123", "caseConceptSource": "auto(n=3)"}
    )

    assert written["sourceLogSha256"] == "abc123"
    assert written["caseConceptSource"] == "auto(n=3)"


def test_a_structured_run_value_is_embedded_as_sorted_json(record):
    written = record(run={"buildInputStatistics": {"b": 2, "a": 1}})

    assert written["buildInputStatistics"] == json.dumps(
        {"a": 1, "b": 2}, sort_keys=True
    )


def test_a_missing_run_value_is_left_out_rather_than_written_as_null(record):
    assert "buildConfigFile" not in record(run={"buildConfigFile": None})


def test_no_run_record_and_an_empty_one_produce_the_same_ontology(record):
    assert record(run=None) == record(run={})


def test_two_builds_of_one_configuration_record_the_same_thing(tiny_store, tmp_path):
    first = emit(
        OntoBuildSettings(), tmp_path / "first.ofn", data=tiny_store, source="demo"
    )
    second = emit(
        OntoBuildSettings(), tmp_path / "second.ofn", data=tiny_store, source="demo"
    )

    assert first.ontology_annotations == second.ontology_annotations
    assert first.axiom_lines == second.axiom_lines
