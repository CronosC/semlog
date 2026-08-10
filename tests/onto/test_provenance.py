"""A saved ontology says which version and configuration produced it.

Downstream projects load the ``.owl`` alone, without the run manifest beside it,
so anything needed to trace a result back to a run has to be inside the file.
"""

import json
from datetime import datetime

import pandas as pd

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.instantiate import OCEDOntoBuilder
from semanticlog.onto.provenance import build_configuration
from semanticlog.onto.settings import (
    DEFAULT_IRI,
    EventOrderRepresentation,
    OntoBuildSettings,
    OwlProfile,
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
    builder.save_onto(str(path))

    assert f"semlog {__version__}" in _annotations(path)


def test_configuration_is_recorded_and_readable(tmp_path):
    path = tmp_path / "prov.owl"
    settings = OntoBuildSettings(
        profile=OwlProfile.OWL2EL,
        case_concepts=["order"],
        order_representation=[EventOrderRepresentation.DF],
    )
    builder = OCEDOntoBuilder(NS, settings)
    builder.build_ontology(_store())
    builder.save_onto(str(path))

    text = _annotations(path)
    assert "buildConfiguration" in text
    # The embedded value is the same dict the manifest records.
    config = build_configuration(settings)
    assert config["profile"] == "OWL2EL"
    assert config["order_representation"] == ["df"]
    assert json.dumps(config, sort_keys=True)[:40] in text


def test_source_log_is_recorded_when_given(tmp_path):
    path = tmp_path / "prov.owl"
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store(), source="logs/demo.jsonocel")
    builder.save_onto(str(path))

    assert "logs/demo.jsonocel" in _annotations(path)


def test_source_log_is_omitted_when_unknown(tmp_path):
    path = tmp_path / "prov.owl"
    builder = OCEDOntoBuilder(NS, OntoBuildSettings())
    builder.build_ontology(_store())
    builder.save_onto(str(path))

    assert "sourceLog" not in _annotations(path)


def test_configuration_excludes_bulky_inputs():
    """Annotation and DECLARE payloads are inputs, not a description of the build."""
    config = build_configuration(OntoBuildSettings())

    assert "annotation_data" not in config
    assert "declare_result" not in config
    assert "annotation_document" in config


def test_two_runs_of_one_configuration_are_identical(tmp_path):
    """No timestamp: provenance must not make a build unreproducible."""
    outputs = []
    for name in ("first.owl", "second.owl"):
        path = tmp_path / name
        builder = OCEDOntoBuilder(NS, OntoBuildSettings())
        builder.build_ontology(_store(), source="logs/demo.jsonocel")
        builder.save_onto(str(path))
        outputs.append(sorted(path.read_text(encoding="utf-8").splitlines()))

    assert outputs[0] == outputs[1]
