"""Carrying DECLARE constraints. The mining itself needs a solver that is
not installed with the package, so it is not exercised here.
"""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from semanticlog.core.errors import DependencyError
from semanticlog.enrich.declare import DeclareEnricher, DeclareResult


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


def test_to_dict_is_json_shaped(result):
    d = result.to_dict()
    assert set(d) == {
        "constraints",
        "case_concepts",
        "source_log",
        "source_sha256",
    }
    assert isinstance(d["constraints"], list)
    assert d["constraints"][0]["trace_id"] == "o1"
    assert d["case_concepts"] == ["order", "item"]


def test_enricher_reports_a_missing_dependency_without_declare4py(monkeypatch):
    import semanticlog.enrich.declare as declare_module

    monkeypatch.setattr(declare_module, "_DECLARE4PY_AVAILABLE", False)
    with pytest.raises(DependencyError, match="Declare4Py"):
        DeclareEnricher(data=None)


def test_reading_a_csv_without_its_origin_file_warns(result, tmp_path, caplog):
    import logging

    path = tmp_path / "constraints.csv"
    result.write(str(path))
    (tmp_path / "constraints.csv.meta.json").unlink()

    with caplog.at_level(logging.WARNING):
        reread = DeclareResult.read(str(path))

    assert reread.source_log is None
    assert "do not record which log" in caplog.text
