"""The record of a build, as annotated onto the ontology."""

import json
from datetime import datetime

import pandas as pd

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.emitters.provenance import build_configuration
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    EventOrderRepresentation,
    OntoBuildSettings,
    OWLProfile,
)

NS = DEFAULT_IRI


def _store() -> OCEDStore:
    data = OCEDStore()
    data.add_event(
        "act",
        id_col="eid",
        time_col="t",
        data=pd.DataFrame({"eid": ["e1"], "t": [datetime(2024, 1, 1)]}),
    )
    data.add_object("order", id_col="oid", data=pd.DataFrame({"oid": ["o1"]}))
    data.add_event_to_object_relation(
        "rel",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["e1"], "oid": ["o1"]}),
    )
    return clean_oced_data(data)


def _annotations(path) -> str:
    return path.read_text(encoding="utf-8")


def test_version_is_recorded(tmp_path):
    from semanticlog import __version__

    path = tmp_path / "prov.owl"
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store())
    builder.sink.save(str(path))

    assert f"semlog {__version__}" in _annotations(path)


def test_configuration_is_recorded_and_readable(tmp_path):
    path = tmp_path / "prov.owl"
    settings = OntoBuildSettings(
        profile=OWLProfile.OWL2EL,
        case_concepts=["order"],
        order_representations=[EventOrderRepresentation.DF],
    )
    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(_store())
    builder.sink.save(str(path))

    text = _annotations(path)
    assert "buildConfiguration" in text
    config = build_configuration(settings)
    assert config["profile"] == "OWL2EL"
    assert config["order_representations"] == ["df"]
    assert json.dumps(config, sort_keys=True)[:40] in text


def test_source_log_is_recorded_when_given(tmp_path):
    path = tmp_path / "prov.owl"
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store(), source="logs/demo.jsonocel")
    builder.sink.save(str(path))

    assert "logs/demo.jsonocel" in _annotations(path)


def test_source_log_is_omitted_when_unknown(tmp_path):
    path = tmp_path / "prov.owl"
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store())
    builder.sink.save(str(path))

    assert "sourceLog" not in _annotations(path)


def test_configuration_excludes_bulky_inputs():
    config = build_configuration(OntoBuildSettings())

    assert "annotation_data" not in config
    assert "declare_result" not in config
    assert "annotation_document" in config


def test_two_runs_of_one_configuration_are_identical(tmp_path):
    outputs = []
    for name in ("first.owl", "second.owl"):
        path = tmp_path / name
        builder = OCEDOntoBuilder(NS, OntoBuildSettings())
        builder.build_ontology(_store(), source="logs/demo.jsonocel")
        builder.sink.save(str(path))
        outputs.append(sorted(path.read_text(encoding="utf-8").splitlines()))

    assert outputs[0] == outputs[1]


def test_the_run_record_becomes_one_annotation_per_entry(tmp_path):
    path = tmp_path / "prov.owl"
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(
        _store(),
        run={"sourceLogSha256": "abc123", "caseConceptSource": "auto(n=3)"},
    )
    builder.sink.save(str(path))

    text = _annotations(path)
    assert "abc123" in text
    assert "auto(n=3)" in text


def test_a_structured_run_value_is_embedded_as_sorted_json(tmp_path):
    path = tmp_path / "prov.owl"
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store(), run={"buildInputStatistics": {"b": 2, "a": 1}})
    builder.sink.save(str(path))

    assert json.dumps({"a": 1, "b": 2}, sort_keys=True) in _annotations(path)


def test_a_missing_run_value_is_omitted_rather_than_written_as_null(tmp_path):
    path = tmp_path / "prov.owl"
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store(), run={"buildConfigFile": None})
    builder.sink.save(str(path))

    assert "buildConfigFile" not in _annotations(path)


def test_no_run_record_leaves_the_ontology_as_it_was(tmp_path):
    outputs = []
    for name, run in (("without.owl", None), ("empty.owl", {})):
        path = tmp_path / name
        builder = OCEDOntoBuilder(NS, OntoBuildSettings())
        builder.build_ontology(_store(), run=run)
        builder.sink.save(str(path))
        outputs.append(sorted(path.read_text(encoding="utf-8").splitlines()))

    assert outputs[0] == outputs[1]
