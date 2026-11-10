"""`DeclareResult`: mined DECLARE constraints per case concept, and their JSON file."""

from __future__ import annotations

import json

import pytest

from semanticlog.core.errors import InputError
from semanticlog.data.declare import ConceptConstraints, DeclareResult, MiningSettings

DISCOVERY = MiningSettings(
    method="discovery",
    templates=("Response", "Precedence"),
    constraint_support=0.1,
    itemset_support=0.9,
)


def _result() -> DeclareResult:
    return DeclareResult(
        concepts=(
            ConceptConstraints(
                "order",
                ("o1", "o2", "o3"),
                {"Response__a__b__": frozenset({"o1", "o3"}), "Init__a__": frozenset()},
            ),
            ConceptConstraints(
                "item", ("i1",), {"Response__a__b__": frozenset({"i1"})}
            ),
        ),
        mining=DISCOVERY,
        source_log="log.jsonocel",
    )


def test_a_result_written_and_read_back_is_equal(tmp_path):
    path = tmp_path / "log.declare.json"

    _result().write(path)

    assert DeclareResult.read(path) == _result()


def test_the_written_file_holds_the_mining_settings_and_one_entry_per_concept(tmp_path):
    path = tmp_path / "log.declare.json"

    _result().write(path)
    written = json.loads(path.read_text())

    assert written["mining"] == {
        "method": "discovery",
        "templates": ["Response", "Precedence"],
        "constraint_support": 0.1,
        "itemset_support": 0.9,
    }
    assert list(written["concepts"]) == ["order", "item"]
    assert written["concepts"]["order"]["satisfied"]["Response__a__b__"] == ["o1", "o3"]


def test_the_members_of_a_constraint_span_every_concept_in_trace_order():
    assert _result().members("Response__a__b__") == ["o1", "o3", "i1"]


def test_the_concepts_satisfying_a_constraint_follow_the_concept_order():
    assert _result().case_concepts_of("Response__a__b__") == ["order", "item"]
    assert _result().case_concepts_of("Init__a__") == []


def test_the_constraint_names_are_the_union_over_concepts_in_first_seen_order():
    assert _result().constraint_names() == ["Response__a__b__", "Init__a__"]


def test_a_satisfying_trace_outside_the_concepts_traces_is_rejected():
    with pytest.raises(InputError, match="'order'.*o9"):
        ConceptConstraints("order", ("o1",), {"Init__a__": frozenset({"o9"})})


def test_a_trace_listed_twice_in_one_concept_is_rejected():
    with pytest.raises(InputError, match="'order'.*o1"):
        ConceptConstraints("order", ("o1", "o1"), {})


def test_a_case_concept_listed_twice_is_rejected():
    order = ConceptConstraints("order", ("o1",), {})

    with pytest.raises(InputError, match="'order'"):
        DeclareResult(concepts=(order, order), mining=DISCOVERY)


def test_a_file_that_is_not_json_is_rejected(tmp_path):
    path = tmp_path / "log.declare.json"
    path.write_text("trace_id,case_concept\n")

    with pytest.raises(InputError, match="not readable as JSON.*mine-declare"):
        DeclareResult.read(path)


@pytest.mark.parametrize(
    "document",
    [
        {"concepts": {}},
        {"mining": DISCOVERY.__dict__ | {"templates": None}, "concepts": []},
        {
            "mining": DISCOVERY.__dict__ | {"templates": None},
            "concepts": {"order": {"trace_ids": ["o1"]}},
        },
    ],
)
def test_a_file_missing_part_of_the_structure_is_rejected(tmp_path, document):
    path = tmp_path / "log.declare.json"
    path.write_text(json.dumps(document))

    with pytest.raises(InputError, match="not a DECLARE result"):
        DeclareResult.read(path)
