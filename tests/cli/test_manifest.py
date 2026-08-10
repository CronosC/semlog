"""The run manifest: what it records, and what it deliberately does not."""

import json
from pathlib import Path

import pandas as pd

from semanticlog.cli.manifest import MANIFEST_VERSION, RunManifest, file_digest
from semanticlog.enrich.declare import DeclareResult
from semanticlog.onto.settings import OntoBuildSettings


def manifest_for(path: Path, **kwargs) -> RunManifest:
    settings = kwargs.pop("settings", OntoBuildSettings(case_concepts=["order"]))
    return RunManifest(
        input_path=path,
        settings=settings,
        case_concept_source=kwargs.pop("case_concept_source", "config"),
        **kwargs,
    )


def test_file_digest_matches_the_content(tmp_path):
    import hashlib

    path = tmp_path / "log.json"
    path.write_bytes(b"some ocel bytes")
    assert file_digest(path) == hashlib.sha256(b"some ocel bytes").hexdigest()


def test_the_manifest_identifies_the_input_by_checksum(tmp_path):
    path = tmp_path / "log.json"
    path.write_bytes(b"payload")

    record = manifest_for(path).to_dict()
    assert record["input"]["path"] == str(path)
    assert record["input"]["bytes"] == len(b"payload")
    assert record["input"]["sha256"] == file_digest(path)


def test_the_manifest_carries_its_own_version_and_semlog_version(tmp_path):
    path = tmp_path / "log.json"
    path.write_bytes(b"x")

    record = manifest_for(path).to_dict()
    assert record["manifest_version"] == MANIFEST_VERSION
    assert record["semlog_version"]


def test_declare_constraints_are_summarised_not_embedded(tmp_path):
    """A manifest stays readable; the constraints live in their own CSV."""
    path = tmp_path / "log.json"
    path.write_bytes(b"x")
    settings = OntoBuildSettings(case_concepts=["order"])
    settings.declare_result = DeclareResult(
        constraints=pd.DataFrame(
            {"constraint": ["Response(a,b)", "Init(a)"], "case_concept": ["o", "o"]}
        ),
        case_concepts=["o"],
    )

    record = manifest_for(path, settings=settings).to_dict()
    assert record["settings"]["declare_constraints"] == {
        "count": 2,
        "case_concepts": ["o"],
    }
    assert "declare_result" not in record["settings"]


def test_annotations_are_counted_not_embedded(tmp_path):
    path = tmp_path / "log.json"
    path.write_bytes(b"x")
    settings = OntoBuildSettings(case_concepts=["order"])
    settings.annotation_data = {"order": {"label": "Order"}, "item": {"label": "Item"}}

    record = manifest_for(path, settings=settings).to_dict()
    assert record["settings"]["annotation_entries"] == 2
    assert "annotation_data" not in record["settings"]


def test_writing_creates_the_directory_and_returns_what_it_wrote(tmp_path):
    path = tmp_path / "log.json"
    path.write_bytes(b"x")
    target = tmp_path / "nested" / "run.manifest.json"

    record = manifest_for(path).write(target)
    assert json.loads(target.read_text()) == record


def test_timings_are_rounded_to_milliseconds(tmp_path):
    path = tmp_path / "log.json"
    path.write_bytes(b"x")

    record = manifest_for(path, timings={"build": 1.23456}).to_dict()
    assert record["timings_s"]["build"] == 1.235
