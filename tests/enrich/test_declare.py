"""Tests for :class:`semanticlog.enrich.declare.DeclareResult`.

Serialisation only: none of this needs declare4py or the Lydia backend.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from semanticlog.enrich.declare import DeclareEnricher, DeclareResult
from semanticlog.onto.settings import OntoBuildSettings


@pytest.fixture
def constraints():
    return pd.DataFrame(
        {
            "case_concept": ["order", "order", "item"],
            "trace_id": ["o1", "o2", "i1"],
            "Response__A__B__": [True, False, True],
            "Existence1__A__": [True, True, False],
        }
    )


@pytest.fixture
def result(constraints):
    return DeclareResult(constraints=constraints, case_concepts=["order", "item"])


# ---------------------------------------------------------------------------
# CSV round-trip
# ---------------------------------------------------------------------------


def test_write_then_read_preserves_constraints(result, tmp_path):
    path = str(tmp_path / "declare.csv")
    result.write(path)
    restored = DeclareResult.read(path)
    assert_frame_equal(restored.constraints, result.constraints)


def test_read_infers_case_concepts_from_column(result, tmp_path):
    path = str(tmp_path / "declare.csv")
    result.write(path)
    assert DeclareResult.read(path).case_concepts == ["order", "item"]


def test_write_does_not_emit_an_index_column(result, tmp_path):
    path = tmp_path / "declare.csv"
    result.write(str(path))
    header = path.read_text().splitlines()[0]
    assert header.startswith("case_concept,trace_id")


def test_read_preserves_boolean_dtype(result, tmp_path):
    path = str(tmp_path / "declare.csv")
    result.write(path)
    restored = DeclareResult.read(path)
    assert restored.constraints["Response__A__B__"].dtype == bool


# ---------------------------------------------------------------------------
# dict round-trip
# ---------------------------------------------------------------------------


def test_to_dict_is_json_shaped(result):
    d = result.to_dict()
    assert set(d) == {"constraints", "case_concepts"}
    assert isinstance(d["constraints"], list)
    assert d["constraints"][0]["trace_id"] == "o1"
    assert d["case_concepts"] == ["order", "item"]


def test_from_dict_restores_to_dict_output(result):
    restored = DeclareResult.from_dict(result.to_dict())
    assert_frame_equal(restored.constraints, result.constraints)
    assert restored.case_concepts == result.case_concepts


def test_from_dict_handles_empty_constraints():
    restored = DeclareResult.from_dict({"constraints": [], "case_concepts": []})
    assert restored.constraints.empty
    assert restored.case_concepts == []


# ---------------------------------------------------------------------------
# Integration with OntoBuildSettings
# ---------------------------------------------------------------------------


def test_settings_read_declare_file_lazily(result, tmp_path):
    path = str(tmp_path / "declare.csv")
    result.write(path)

    settings = OntoBuildSettings(declare_file=path)
    loaded = settings.declare_result

    assert loaded is not None
    assert_frame_equal(loaded.constraints, result.constraints)


def test_settings_declare_result_assignment_wins_over_file(result, tmp_path):
    path = str(tmp_path / "declare.csv")
    result.write(path)

    settings = OntoBuildSettings(declare_file=path)
    direct = DeclareResult(
        constraints=pd.DataFrame({"trace_id": ["x"]}), case_concepts=[]
    )
    settings.declare_result = direct

    assert settings.declare_result is direct


def test_settings_reread_when_declare_file_changes(result, tmp_path):
    first = str(tmp_path / "a.csv")
    second = str(tmp_path / "b.csv")
    result.write(first)
    DeclareResult(
        constraints=result.constraints.head(1), case_concepts=["order"]
    ).write(second)

    settings = OntoBuildSettings(declare_file=first)
    assert len(settings.declare_result.constraints) == 3
    settings.declare_file = second
    assert len(settings.declare_result.constraints) == 1


def test_settings_round_trip_through_to_dict(result, tmp_path):
    settings = OntoBuildSettings()
    settings.declare_result = result

    restored = OntoBuildSettings.from_dict(settings.to_dict())

    assert restored.declare_result is not None
    assert_frame_equal(restored.declare_result.constraints, result.constraints)


# ---------------------------------------------------------------------------
# DeclareEnricher availability
# ---------------------------------------------------------------------------


def test_enricher_raises_import_error_without_declare4py(monkeypatch):
    """Without the optional backend the constructor must fail with a clear message."""
    import semanticlog.enrich.declare as declare_module

    monkeypatch.setattr(declare_module, "_DECLARE4PY_AVAILABLE", False)
    with pytest.raises(ImportError, match="Declare4Py"):
        DeclareEnricher(data=None)
