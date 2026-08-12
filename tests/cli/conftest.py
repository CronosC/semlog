"""Fixtures for the CLI tests: a log, a config writer and result readers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

FIXTURE_OCEL = Path(__file__).resolve().parents[1] / "fixtures" / "pipeline.jsonocel"

_OWL_ONTOLOGY = "{http://www.w3.org/2002/07/owl#}Ontology"

_OFN_ANNOTATION = re.compile(
    r'^AnnotationAssertion\(<[^>]*[#/](?P<name>\w+)> <[^>]*> "(?P<value>.*)"\)$'
)


@pytest.fixture
def ocel() -> Path:
    return FIXTURE_OCEL


@pytest.fixture
def write_config(tmp_path: Path):

    def _write(name: str, body: str) -> Path:
        path = tmp_path / name
        path.write_text(body, encoding="utf-8")
        return path

    return _write


def ontology_annotations(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    if not text.lstrip().startswith("<"):
        return {
            match["name"]: match["value"].replace('\\"', '"').replace("\\\\", "\\")
            for line in text.splitlines()
            if (match := _OFN_ANNOTATION.match(line))
        }

    header = ElementTree.parse(path).getroot().find(_OWL_ONTOLOGY)
    assert header is not None, f"{path} has no owl:Ontology element"
    return {child.tag.rpartition("}")[2]: child.text or "" for child in header}


def run_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = dict(ontology_annotations(path))
    for key in ("buildConfiguration", "buildInputStatistics", "buildOutputs"):
        if key in record:
            record[key] = json.loads(record[key])
    return record
