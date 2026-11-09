"""`DeclareEnricher` with the Declare4Py classes replaced by fakes, and how its output becomes a `DeclareResult`."""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from semanticlog.core.errors import ConfigError, DependencyError, InputError
from semanticlog.data.declare import ConceptConstraints, MiningSettings
from semanticlog.data.tables import EntityKind
from semanticlog.enrich import declare as declare_module
from semanticlog.enrich.declare import (
    DeclareEnricher,
    concept_constraints,
    drop_uninformative,
    template_of,
)
from tests.support import stores

# --------------------------------------------------------------------------
# Mining, with fakes for Declare4Py
# --------------------------------------------------------------------------


class FakeTrace(list):
    """Fake trace, which Declare4Py reads as a sequence of event dicts."""

    def __init__(self, activities, name):
        super().__init__({"concept:name": a} for a in activities)
        self.attributes = {"concept:name": name}


class FakeEventLog:
    """Fake `D4PyEventLog`."""

    def __init__(self, log=None, case_name=None):
        self.wrapped = log
        self.case_name = case_name

    def get_log(self):
        return self.wrapped


class FakeModel:
    """Fake `DeclareModel`."""

    def __init__(self, text: str | None = None):
        self.text = text

    def parse_from_string(self, text: str) -> FakeModel:
        return FakeModel(text)


class FakeAnalyzer:
    """Fake `MPDeclareAnalyzer`. It records what it was handed."""

    calls: list[dict] = []
    # Both vary across the two traces, so neither is dropped as constant.
    metric = pd.DataFrame({"Response[a, b] | | |": [1, 0], "Init[a] | |": [0, 1]})

    def __init__(self, log=None, declare_model=None, consider_vacuity=None):
        FakeAnalyzer.calls.append(
            {"log": log, "model": declare_model, "consider_vacuity": consider_vacuity}
        )

    def run(self) -> FakeAnalyzer:
        return self

    def get_metric(self, metric: str) -> pd.DataFrame:
        assert metric == "state"
        return FakeAnalyzer.metric.copy()


class FakeMiner:
    """Fake `DeclareMiner`. It records the thresholds it was given."""

    calls: list[dict] = []

    def __init__(self, **kwargs):
        FakeMiner.calls.append(kwargs)

    def run(self) -> FakeModel:
        return FakeModel("discovered")


@pytest.fixture
def fakes(monkeypatch):
    """Put the fakes in place of Declare4Py."""
    FakeAnalyzer.calls = []
    FakeMiner.calls = []
    monkeypatch.setattr(declare_module, "_DECLARE4PY_AVAILABLE", True)
    for name, fake in (
        ("D4PyEventLog", FakeEventLog),
        ("DeclareModel", FakeModel),
        ("MPDeclareAnalyzer", FakeAnalyzer),
        ("DeclareMiner", FakeMiner),
    ):
        monkeypatch.setattr(declare_module, name, fake, raising=False)


@pytest.fixture
def log() -> FakeEventLog:
    """A two-trace log with three activities between them."""
    return FakeEventLog([FakeTrace(["a", "b"], "t1"), FakeTrace(["b", "c"], "t2")])


@pytest.fixture
def enricher(fakes, tiny_store, log, monkeypatch):
    """An enricher whose every case concept flattens to `log`."""
    monkeypatch.setattr(
        DeclareEnricher, "get_log", lambda self, cc: (log, [f"{cc}1", f"{cc}2"])
    )
    return DeclareEnricher(tiny_store)


def test_a_missing_declare4py_names_the_command_and_the_install_steps(
    monkeypatch, tiny_store
):
    monkeypatch.setattr(declare_module, "_DECLARE4PY_AVAILABLE", False)

    with pytest.raises(DependencyError, match="^semlog mine-declare needs") as failure:
        DeclareEnricher(tiny_store)

    message = str(failure.value)
    assert "uv sync --extra declare" in message
    assert "lydia" in message
    assert "PATH" in message
    assert "DeclareEnricher" not in message


def test_the_model_declares_every_activity_once(enricher, log):
    assert (
        enricher.construct_model_string(log) == "activity a\nactivity b\nactivity c\n"
    )


def test_a_unary_template_is_instantiated_once_per_activity(enricher, log):
    model = enricher.construct_model_string(log, constraint_set=["Init"])

    assert [line for line in model.splitlines() if line.startswith("Init")] == [
        "Init[a] | |",
        "Init[b] | |",
        "Init[c] | |",
    ]


def test_a_binary_template_takes_every_ordered_pair_of_distinct_activities(
    enricher, log
):
    model = enricher.construct_model_string(log, constraint_set=["Response"])
    pairs = [line for line in model.splitlines() if line.startswith("Response")]

    assert len(pairs) == 6
    assert {"Response[a, b] | | |", "Response[b, a] | | |"} <= set(pairs)
    assert [line for line in pairs if "[a, a]" in line] == []


def test_a_misspelled_template_is_rejected(enricher):
    with pytest.raises(ConfigError, match="Responce"):
        enricher.get_constraints_via_discovery(["order"], constraint_set=["Responce"])


def test_mining_over_no_case_concept_is_rejected(enricher):
    with pytest.raises(ConfigError, match="No case concepts to mine"):
        enricher.get_constraints_via_enumeration([], ["Init"])


def test_enumeration_checks_each_case_concept_against_the_model_it_built(enricher, log):
    result = enricher.get_constraints_via_enumeration(["order", "item"], ["Init"])

    assert [call["model"].text for call in FakeAnalyzer.calls] == [
        enricher.construct_model_string(log, constraint_set=["Init"])
    ] * 2
    assert [(c.case_concept, c.trace_ids) for c in result.concepts] == [
        ("order", ("order1", "order2")),
        ("item", ("item1", "item2")),
    ]
    assert result.mining == MiningSettings("enumeration", ("Init",))


def test_enumeration_passes_the_vacuity_setting_through(enricher):
    enricher.get_constraints_via_enumeration(
        ["order"], ["Init"], vacuous_is_considered=True
    )

    assert FakeAnalyzer.calls[0]["consider_vacuity"] is True


def test_discovery_checks_the_model_the_miner_found_with_its_thresholds(enricher):
    result = enricher.get_constraints_via_discovery(
        ["order"], constraint_support_threshold=0.7, itemset_support_threshold=0.3
    )

    assert (
        FakeMiner.calls[0]["min_support"],
        FakeMiner.calls[0]["itemsets_support"],
    ) == (
        0.7,
        0.3,
    )
    assert FakeAnalyzer.calls[0]["model"].text == "discovered"
    assert result.mining == MiningSettings("discovery", None, 0.7, 0.3)


def test_discovery_keeps_only_the_templates_it_was_restricted_to(enricher):
    """The miner proposes whatever the log supports."""
    result = enricher.get_constraints_via_discovery(["order"], constraint_set=["Init"])

    assert result.constraint_names() == ["Init__a__"]
    assert result.mining.templates == ("Init",)


@pytest.mark.parametrize(
    ("column", "expected"),
    [("Exclusive Choice[a, b] | | |", "Exclusive Choice"), ("trace_id", "trace_id")],
)
def test_the_template_of_a_column_is_the_text_before_the_activities(column, expected):
    assert template_of(column) == expected


# --------------------------------------------------------------------------
# The checker's output, made into class names and membership
# --------------------------------------------------------------------------


def _concept(name: str, columns: dict, trace_ids: list) -> ConceptConstraints:
    return concept_constraints(name, pd.DataFrame(columns | {"trace_id": trace_ids}))


def test_a_constraint_column_becomes_a_class_name_with_its_satisfying_traces():
    concept = _concept("order", {"Response[a, b] | | |": [1, 0]}, ["t1", "t2"])

    assert dict(concept.satisfied) == {"Response__a__b__": frozenset({"t1"})}


def test_a_trace_with_no_answer_does_not_satisfy_the_constraint():
    concept = _concept(
        "order", {"Response[a, b] | | |": [1, 0, float("nan")]}, ["t1", "t2", "t3"]
    )

    assert concept.satisfied["Response__a__b__"] == frozenset({"t1"})


def test_numeric_trace_ids_are_kept_as_text():
    concept = _concept("order", {"Response[a, b] | | |": [1, 0]}, [0, 1])

    assert concept.trace_ids == ("0", "1")


def test_two_constraints_that_get_one_class_name_are_rejected():
    columns = {"Response[a__b, c] | | |": [1, 0], "Response[a, b__c] | | |": [0, 1]}

    with pytest.raises(InputError, match="'order' both get the class name"):
        _concept("order", columns, ["t1", "t2"])


def test_two_constraints_that_get_one_class_name_are_reported_on_one_line():
    columns = {"Response[a__b, c] | | |": [1, 0], "Response[a, b__c] | | |": [0, 1]}

    with pytest.raises(InputError, match="both get the class name") as failure:
        _concept("order", columns, ["t1", "t2"])

    assert "\n" not in str(failure.value)


def test_a_column_that_names_no_constraint_is_a_dependency_error():
    with pytest.raises(DependencyError, match="names no constraint"):
        _concept("order", {"state": [1, 0]}, ["t1", "t2"])


def test_a_constraint_that_holds_of_every_trace_is_dropped():
    order = _concept(
        "order", {"Init[a] | |": [1, 1], "Response[a, b] | | |": [1, 0]}, ["o1", "o2"]
    )

    (kept,) = drop_uninformative([order])

    assert list(kept.satisfied) == ["Response__a__b__"]


def test_a_constraint_constant_within_one_case_concept_only_is_kept():
    order = _concept("order", {"Response[a, b] | | |": [1, 1]}, ["o1", "o2"])
    item = _concept("item", {"Response[a, b] | | |": [1, 0]}, ["i1", "i2"])

    kept = drop_uninformative([order, item])

    assert [dict(c.satisfied) for c in kept] == [
        {"Response__a__b__": frozenset({"o1", "o2"})},
        {"Response__a__b__": frozenset({"i1"})},
    ]


def test_a_constraint_checked_for_one_case_concept_and_held_by_all_is_dropped():
    order = _concept(
        "order", {"Init[a] | |": [1, 1], "Response[a, b] | | |": [1, 0]}, ["o1", "o2"]
    )
    item = _concept("item", {"Response[a, b] | | |": [1, 0]}, ["i1", "i2"])

    kept = drop_uninformative([order, item])

    assert [list(c.satisfied) for c in kept] == [["Response__a__b__"]] * 2


# --------------------------------------------------------------------------
# Flattening, where the trace ids come from
# --------------------------------------------------------------------------


@pytest.fixture
def awkward_enricher(fakes):
    return DeclareEnricher(stores.build("awkward"))


def test_the_trace_ids_follow_the_order_of_the_traces_in_the_log(awkward_enricher):
    """`trace_id` is assigned to the checker's rows by position."""
    log, case_ids = awkward_enricher.get_log("order")

    assert case_ids == [trace.attributes["concept:name"] for trace in log.wrapped]
    assert set(case_ids) <= set(
        stores.build("awkward").table(EntityKind.OBJECT, "order").data["oid"]
    )


def test_the_flattened_log_names_its_activity_and_timestamp_keys(awkward_enricher):
    """`D4PyEventLog` reads both keys from the log's properties."""
    log, _ = awkward_enricher.get_log("order")

    assert log.wrapped.properties["pm4py:param:timestamp_key"] == "time:timestamp"
    assert log.wrapped.properties["pm4py:param:activity_key"] == "concept:name"


def test_a_change_to_the_root_logger_inside_the_import_guard_is_undone():
    root = logging.getLogger()
    before = (root.handlers[:], root.level)

    with declare_module._keep_logging_config():
        root.addHandler(logging.NullHandler())
        root.setLevel(logging.DEBUG)

    assert (root.handlers, root.level) == before
