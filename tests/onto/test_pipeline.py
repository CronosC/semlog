from __future__ import annotations

import json

import pytest

from semanticlog.core.enums import EventOrderRepresentation
from semanticlog.core.vocabulary import DEFAULT_IRI, UnsafeLocalNameError
from semanticlog.data.declare import ConceptConstraints, DeclareResult, MiningSettings
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

    result = build_log(tiny_store, NO_ORDER, BuildInputs(), output, streaming=False)

    assert result.path == str(output)
    assert "step" in {
        c.iri.remainder for c in result.require_onto().classes_in_signature()
    }
    assert "step" in read_rdf(output, DEFAULT_IRI).classes


def test_the_iri_and_the_run_record_reach_the_file(tiny_store, tmp_path):
    output = tmp_path / "named.ofn"
    iri = "https://example.org/tiny#"

    result = build_log(
        tiny_store,
        NO_ORDER,
        BuildInputs(),
        output,
        iri=iri,
        source="tiny.jsonocel",
        record={"sourceLogSha256": "abc", "stats": {"events": 3}},
    )

    annotations = read_functional(output).ontology_annotations
    assert result.iri == iri
    assert f"Ontology(<{iri.rstrip('#')}>" in output.read_text(encoding="utf-8")
    assert annotations["sourceLog"] == "tiny.jsonocel"
    assert annotations["sourceLogSha256"] == "abc"
    assert json.loads(annotations["stats"]) == {"events": 3}


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
