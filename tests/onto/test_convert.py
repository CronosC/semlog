"""`convert`, which writes a built ontology in further formats. Every test starts a JVM."""

from __future__ import annotations

from pathlib import Path

import pytest

from semanticlog.core.enums import EventOrderRepresentation, OutputFormat
from semanticlog.core.errors import InputError
from semanticlog.onto import reason as reason_module
from semanticlog.onto.convert import convert
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.pipeline import build_log
from semanticlog.onto.settings import OntoBuildSettings
from tests.support import stores
from tests.support.owlapi import load

pytestmark = pytest.mark.reasoner


def _axioms(path: Path) -> set[str]:
    return {str(axiom) for axiom in load(path).getAxioms()}


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("built") / "pipeline.ofn"
    build_log(
        stores.pipeline(),
        OntoBuildSettings(order_representations=[EventOrderRepresentation.DF]),
        BuildInputs(case_concepts=["order"]),
        path,
    )
    return path


@pytest.mark.parametrize("output_format", list(OutputFormat))
def test_each_format_holds_the_axioms_of_the_source(built, tmp_path, output_format):
    target = tmp_path / f"pipeline.{output_format.value}"

    convert(built, {output_format: target})

    source, written = _axioms(built), _axioms(target)
    assert source - written == set()
    # The OWL API declares every individual it writes.
    assert [a for a in written - source if not a.startswith("Declaration(")] == []


def test_one_call_writes_every_requested_format(built, tmp_path):
    targets = {f: tmp_path / f"pipeline.{f.value}" for f in OutputFormat}

    convert(built, targets)

    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        p.name for p in targets.values()
    )


def test_a_failed_save_leaves_no_file_and_names_the_format(
    built, tmp_path, monkeypatch
):
    # The bundled OWL API has no N-Triples writer, so this save fails.
    monkeypatch.setitem(
        reason_module._DOCUMENT_FORMATS, OutputFormat.TTL, "NTriplesDocumentFormat"
    )
    target = tmp_path / "pipeline.ttl"

    with pytest.raises(InputError, match=r"Cannot write .*pipeline\.ttl as ttl"):
        convert(built, {OutputFormat.TTL: target})

    assert list(tmp_path.iterdir()) == []
