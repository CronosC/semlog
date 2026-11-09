from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

from semanticlog.core.enums import EventOrderRepresentation
from semanticlog.core.errors import ConfigError
from semanticlog.core.vocabulary import DEFAULT_IRI, UnsafeLocalNameError
from semanticlog.data.declare import ConceptConstraints, DeclareResult, MiningSettings
from semanticlog.data.loading import load_and_clean
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.pipeline import build_log
from semanticlog.onto.settings import OntoBuildSettings
from tests.support.functional import read_functional
from tests.support.rdf import read_rdf

DF = EventOrderRepresentation.DF
NO_ORDER = OntoBuildSettings(order_representations=[])


def test_a_streamed_build_writes_functional_syntax_into_a_new_directory(
    tiny_store, tmp_path
):
    output = tmp_path / "out" / "tiny.ofn"

    result = build_log(
        tiny_store,
        OntoBuildSettings(order_representations=[DF]),
        BuildInputs(case_concepts=["order"]),
        output,
    )

    assert result.path == str(output)
    assert ("df_order", "e1", "e2") in read_functional(output).relations


def test_a_streamed_build_holds_no_ontology_to_hand_back(tiny_store, tmp_path):
    result = build_log(tiny_store, NO_ORDER, BuildInputs(), tmp_path / "tiny.ofn")

    assert result.onto is None
    with pytest.raises(ValueError, match="is not in memory"):
        result.require_onto()


def test_an_in_memory_build_writes_rdf_xml_and_keeps_the_ontology(tiny_store, tmp_path):
    output = tmp_path / "tiny.owl"

    result = build_log(tiny_store, NO_ORDER, BuildInputs(), output)

    assert result.path == str(output)
    assert "step" in {
        c.iri.remainder for c in result.require_onto().classes_in_signature()
    }
    assert "step" in read_rdf(output, DEFAULT_IRI).classes


def test_the_iri_and_the_header_reach_the_file(tiny_store, tmp_path):
    output = tmp_path / "named.ofn"
    iri = "https://example.org/tiny/"

    result = build_log(tiny_store, NO_ORDER, BuildInputs(), output, iri=iri)

    annotations = read_functional(output).ontology_annotations
    assert result.iri == iri
    assert f"Ontology(<{iri.removesuffix('/')}>" in output.read_text(encoding="utf-8")
    assert set(annotations) == {"buildTool", "buildConfiguration", "caseConcepts"}
    assert json.loads(annotations["caseConcepts"]) == []


def test_a_rejected_build_leaves_no_file_behind(tiny_store, tmp_path):
    output = tmp_path / "rejected" / "out.ofn"
    declare = DeclareResult(
        concepts=(
            ConceptConstraints("order", ("o1",), {"Response@1": frozenset({"o1"})}),
        ),
        mining=MiningSettings("discovery", None, 0.5, 0.5),
    )

    with pytest.raises(UnsafeLocalNameError, match="Response@1"):
        build_log(
            tiny_store,
            OntoBuildSettings(
                order_representations=[DF, EventOrderRepresentation.DECLARE]
            ),
            BuildInputs(case_concepts=["order"], declare=declare),
            output,
        )

    assert list(output.parent.iterdir()) == []


def test_a_build_into_an_ofn_path_writes_functional_syntax(tiny_store, tmp_path):
    output = tmp_path / "tiny.ofn"

    build_log(tiny_store, NO_ORDER, BuildInputs(), output)

    assert output.read_text(encoding="utf-8").startswith("Prefix(")


def test_a_build_into_an_owl_path_writes_rdf_xml(tiny_store, tmp_path):
    output = tmp_path / "tiny.owl"

    build_log(tiny_store, NO_ORDER, BuildInputs(), output)

    assert output.read_text(encoding="utf-8").startswith("<?xml")


def test_a_build_into_a_path_of_another_suffix_is_rejected(tiny_store, tmp_path):
    output = tmp_path / "tiny.ttl"

    with pytest.raises(ConfigError, match=r"tiny\.ttl.*\.ofn.*\.owl"):
        build_log(tiny_store, NO_ORDER, BuildInputs(), output)

    assert not output.exists()


@pytest.mark.parametrize("suffix", ["ofn", "owl"])
def test_a_csv_value_with_a_control_character_builds_into_a_readable_file(
    tmp_path, caplog, suffix
):
    log = tmp_path / "log.csv"
    pd.DataFrame(
        {
            "case:concept:name": ["c1", "c1"],
            "concept:name": ["a", "b"],
            "time:timestamp": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"],
            "note": ["bad\x01note", "fine"],
        }
    ).to_csv(log, index=False)
    output = tmp_path / f"log.{suffix}"

    with caplog.at_level(logging.WARNING):
        build_log(
            load_and_clean(log),
            NO_ORDER.replace(derive_annotations=True),
            BuildInputs(),
            output,
        )

    # rdflib reads RDF/XML through an XML parser, which rejects a control
    # character.
    onto = read_functional(output) if suffix == "ofn" else read_rdf(output, DEFAULT_IRI)
    assert ("note.bad_note", "label", "bad\ufffdnote") in onto.annotations
    assert "U+FFFD" in caplog.text
    assert "labels of note (1)" in caplog.text
