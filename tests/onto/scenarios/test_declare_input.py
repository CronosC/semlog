"""Rejecting DECLARE constraints that do not belong to the log."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from semanticlog.core.vocabulary import NameCollisionError, UnsafeLocalNameError
from semanticlog.enrich.declare import META_SUFFIX, DeclareResult
from semanticlog.onto.audit import audit_build
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import DEFAULT_IRI
from tests.onto.support.configurations import CONFIGURATIONS, build_store


def _build(settings):
    return OCEDOntoBuilder(DEFAULT_IRI, settings).build_ontology(build_store())


def _settings(**constraint_columns):
    settings = CONFIGURATIONS["dl_declare"]()
    settings = settings.replace(
        declare_result=DeclareResult(
            constraints=pd.DataFrame(
                {
                    "trace_id": ["o1", "o2"],
                    "case_concept": ["order", "order"],
                    **constraint_columns,
                }
            ),
            case_concepts=["order"],
        )
    )
    return settings


def test_an_unsafe_constraint_name_is_rejected():
    settings = _settings(**{"Response@1": [True, False]})
    with pytest.raises(UnsafeLocalNameError, match="Response@1"):
        _build(settings)


def test_a_constraint_name_colliding_with_an_object_type_is_rejected():
    settings = _settings(order=[True, False])
    with pytest.raises(NameCollisionError, match="'order'"):
        _build(settings)


def test_trace_ids_that_are_not_objects_are_rejected():
    settings = CONFIGURATIONS["dl_declare"]()
    settings = settings.replace(
        declare_result=DeclareResult(
            constraints=pd.DataFrame(
                {
                    "trace_id": ["from_another_log", "o1"],
                    "case_concept": ["order", "order"],
                    "Response__place__ship__": [True, False],
                }
            ),
            case_concepts=["order"],
        )
    )
    with pytest.raises(ValueError, match="from_another_log"):
        audit_build(build_store(), settings)


def test_an_unknown_case_concept_is_rejected():
    settings = CONFIGURATIONS["dl_declare"]()
    settings = settings.replace(
        declare_result=DeclareResult(
            constraints=pd.DataFrame(
                {
                    "trace_id": ["x1"],
                    "case_concept": ["no_such_type"],
                    "Response__place__ship__": [True],
                }
            ),
            case_concepts=["no_such_type"],
        )
    )
    with pytest.raises(ValueError, match="not an object type of this log"):
        audit_build(build_store(), settings)


def test_a_rejected_declare_input_writes_nothing(tmp_path):
    from semanticlog.onto.sinks import StreamingSink

    target = tmp_path / "out.ofn"
    settings = _settings(**{"Response@1": [True, False]})
    sink = StreamingSink(str(target), DEFAULT_IRI, settings)
    with pytest.raises(UnsafeLocalNameError):
        OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(build_store())

    sink.close()
    assert "Response" not in target.read_text(encoding="utf-8")


def test_a_valid_declare_input_still_builds(tmp_path):
    settings = CONFIGURATIONS["dl_declare"]()
    audit_build(build_store(), settings)


def test_write_records_where_the_constraints_came_from(tmp_path):
    path = tmp_path / "constraints.csv"
    DeclareResult(
        constraints=pd.DataFrame({"trace_id": ["o1"], "case_concept": ["order"]}),
        case_concepts=["order"],
        source_log="data/ocel/hiring.xml",
        source_sha256="abc123",
    ).write(str(path))

    meta = json.loads((tmp_path / ("constraints.csv" + META_SUFFIX)).read_text())
    assert meta["source_log"] == "data/ocel/hiring.xml"
    assert meta["source_sha256"] == "abc123"

    restored = DeclareResult.read(str(path))
    assert restored.source_log == "data/ocel/hiring.xml"
    assert restored.source_sha256 == "abc123"


def test_a_csv_without_a_sidecar_still_loads_with_a_warning(tmp_path, caplog):
    path = tmp_path / "bare.csv"
    pd.DataFrame({"trace_id": ["o1"], "case_concept": ["order"]}).to_csv(
        path, index=False
    )

    result = DeclareResult.read(str(path))

    assert result.source_sha256 is None
    assert "do not record which log" in caplog.text


def test_source_survives_the_dict_round_trip():
    original = DeclareResult(
        constraints=pd.DataFrame({"trace_id": ["o1"], "case_concept": ["order"]}),
        case_concepts=["order"],
        source_log="a.xml",
        source_sha256="deadbeef",
    )
    restored = DeclareResult.from_dict(original.to_dict())
    assert (restored.source_log, restored.source_sha256) == ("a.xml", "deadbeef")


def test_a_foreign_digest_warns_without_failing(caplog):
    import dataclasses

    from semanticlog.cli.run import warn_on_foreign_declare

    settings = CONFIGURATIONS["dl_declare"]()
    settings = settings.replace(
        declare_result=dataclasses.replace(
            settings.declare_result,
            source_log="somewhere/else.xml",
            source_sha256="a" * 64,
        )
    )

    warn_on_foreign_declare(settings, "b" * 64)

    assert "not the file being built" in caplog.text


def test_a_declare_result_cannot_be_rewritten_through_frozen_settings():
    import dataclasses

    settings = CONFIGURATIONS["dl_declare"]()
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.declare_result.source_log = "somewhere/else.xml"
