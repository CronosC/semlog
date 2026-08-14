"""Mining DECLARE constraints, with the solver stood in for.

`DeclareEnricher` needs declare4py and the Lydia solver, neither of which is
installed here. The four backend names are looked up as module globals when a
method runs, so they can be replaced with fakes. That covers the wiring, which
is what is ours: how many passes there are, what reaches the backend and what
comes back. It does not check that declare4py agrees, and cannot.
"""

from __future__ import annotations

import pandas as pd
import pytest

from semanticlog.enrich import declare as declare_module
from semanticlog.enrich.declare import DeclareEnricher, template_of
from tests.support import stores


class FakeTrace(list):
    """A trace, which declare4py reads as a sequence of event dicts."""

    def __init__(self, activities, name):
        super().__init__({"concept:name": a} for a in activities)
        self.attributes = {"concept:name": name}


class FakeEventLog:
    """Stands in for `D4PyEventLog`, which wraps a pm4py log."""

    def __init__(self, log=None, case_name=None):
        self.wrapped = log
        self.case_name = case_name

    def get_log(self):
        return self.wrapped


class FakeModel:
    """Stands in for a parsed or discovered `DeclareModel`."""

    def __init__(self, text: str | None = None):
        self.text = text

    def parse_from_string(self, text: str) -> FakeModel:
        return FakeModel(text)


class FakeAnalyzer:
    """Stands in for `MPDeclareAnalyzer`, recording what it was handed."""

    calls: list[dict] = []
    # Both vary across the two traces, so neither is dropped as saying
    # nothing.
    metric = pd.DataFrame(
        {
            "Response[a, b] | | |": [1, 0],
            "Init[a] | |": [0, 1],
        }
    )

    def __init__(self, log=None, declare_model=None, consider_vacuity=None):
        FakeAnalyzer.calls.append(
            {
                "log": log,
                "model": declare_model,
                "consider_vacuity": consider_vacuity,
            }
        )

    def run(self) -> FakeAnalyzer:
        return self

    def get_metric(self, metric: str) -> pd.DataFrame:
        assert metric == "state"
        return FakeAnalyzer.metric.copy()


class FakeMiner:
    """Stands in for `DeclareMiner`, recording the thresholds it was given."""

    calls: list[dict] = []

    def __init__(self, **kwargs):
        FakeMiner.calls.append(kwargs)

    def run(self) -> FakeModel:
        return FakeModel("discovered")


@pytest.fixture
def enricher(monkeypatch, tiny_store):
    """An enricher whose backend is the fakes above."""
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
    return DeclareEnricher(tiny_store)


@pytest.fixture
def log() -> FakeEventLog:
    """A two-trace log with three activities between them."""
    return FakeEventLog(
        [FakeTrace(["a", "b"], "t1"), FakeTrace(["b", "c"], "t2")],
    )


# --------------------------------------------------------------------------
# Refusing to construct without the solver
# --------------------------------------------------------------------------


def test_the_enricher_refuses_to_run_without_declare4py(monkeypatch, tiny_store):
    monkeypatch.setattr(declare_module, "_DECLARE4PY_AVAILABLE", False)

    with pytest.raises(ImportError, match="Lydia"):
        DeclareEnricher(tiny_store)


# --------------------------------------------------------------------------
# The model string enumeration checks against
# --------------------------------------------------------------------------


def test_every_activity_is_declared_once(enricher, log):
    model = enricher.construct_model_string(log)

    assert model == "activity a\nactivity b\nactivity c\n"


def test_a_unary_template_is_instantiated_once_per_activity(enricher, log):
    model = enricher.construct_model_string(log, constraint_set=["Init"])
    lines = [line for line in model.splitlines() if line.startswith("Init")]

    assert lines == ["Init[a] | |", "Init[b] | |", "Init[c] | |"]


def test_a_binary_template_takes_every_ordered_pair_of_distinct_activities(
    enricher, log
):
    model = enricher.construct_model_string(log, constraint_set=["Response"])
    pairs = [line for line in model.splitlines() if line.startswith("Response")]

    assert len(pairs) == 6
    assert "Response[a, b] | | |" in pairs
    assert "Response[b, a] | | |" in pairs
    assert not any("[a, a]" in line for line in pairs)


def test_the_unary_and_binary_templates_are_told_apart(enricher, log):
    model = enricher.construct_model_string(log, constraint_set=["Init", "Response"])

    assert "Init[a] | |" in model
    assert "Response[a, b] | | |" in model


def test_no_templates_leaves_the_activities_alone(enricher, log):
    model = enricher.construct_model_string(log, constraint_set=[])

    assert model.splitlines() == ["activity a", "activity b", "activity c"]


def test_a_misspelled_template_is_refused(enricher, log):
    """It would otherwise be written out as a binary template and fail
    somewhere inside declare4py."""
    with pytest.raises(ValueError, match="Responce"):
        enricher.construct_model_string(log, constraint_set=["Responce"])


# --------------------------------------------------------------------------
# Enumeration
# --------------------------------------------------------------------------


def test_enumeration_runs_one_pass_per_case_concept(enricher, monkeypatch):
    monkeypatch.setattr(
        DeclareEnricher,
        "get_log",
        lambda self, cc: (FakeEventLog([]), [f"{cc}1", f"{cc}2"]),
    )

    result = enricher.get_constraints_via_enumeration(["order", "item"], ["Init"])

    assert len(FakeAnalyzer.calls) == 2
    assert set(result.constraints["case_concept"]) == {"order", "item"}


def test_enumeration_checks_the_model_it_built(enricher, log, monkeypatch):
    monkeypatch.setattr(
        DeclareEnricher, "get_log", lambda self, cc: (log, ["t1", "t2"])
    )

    enricher.get_constraints_via_enumeration(["order"], ["Init"])

    assert FakeAnalyzer.calls[0]["model"].text == enricher.construct_model_string(
        log, constraint_set=["Init"]
    )


def test_enumeration_labels_each_row_with_its_trace(enricher, log, monkeypatch):
    monkeypatch.setattr(
        DeclareEnricher, "get_log", lambda self, cc: (log, ["t1", "t2"])
    )

    result = enricher.get_constraints_via_enumeration(["order"], ["Init"])

    assert list(result.constraints["trace_id"]) == ["t1", "t2"]


def test_enumeration_passes_the_vacuity_setting_through(enricher, log, monkeypatch):
    monkeypatch.setattr(
        DeclareEnricher, "get_log", lambda self, cc: (log, ["t1", "t2"])
    )

    enricher.get_constraints_via_enumeration(
        ["order"], ["Init"], vacuous_is_considered=True
    )

    assert FakeAnalyzer.calls[0]["consider_vacuity"] is True


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------


def test_discovery_hands_the_thresholds_to_the_miner(enricher, log, monkeypatch):
    monkeypatch.setattr(
        DeclareEnricher, "get_log", lambda self, cc: (log, ["t1", "t2"])
    )

    enricher.get_constraints_via_discovery(
        ["order"], constraint_support_threshold=0.7, itemset_support_threshold=0.3
    )

    assert FakeMiner.calls[0]["min_support"] == 0.7
    assert FakeMiner.calls[0]["itemsets_support"] == 0.3


def test_discovery_checks_the_model_the_miner_found(enricher, log, monkeypatch):
    monkeypatch.setattr(
        DeclareEnricher, "get_log", lambda self, cc: (log, ["t1", "t2"])
    )

    enricher.get_constraints_via_discovery(["order"])

    assert FakeAnalyzer.calls[0]["model"].text == "discovered"


def test_discovery_keeps_only_the_templates_it_was_restricted_to(
    enricher, log, monkeypatch
):
    """The miner proposes whatever the log supports, so the restriction is
    applied to what it found."""
    monkeypatch.setattr(
        DeclareEnricher, "get_log", lambda self, cc: (log, ["t1", "t2"])
    )

    result = enricher.get_constraints_via_discovery(["order"], constraint_set=["Init"])
    mined = set(result.constraints.columns) - {"trace_id", "case_concept"}

    assert mined == {"Init__a__"}


def test_discovery_without_a_restriction_keeps_everything(enricher, log, monkeypatch):
    monkeypatch.setattr(
        DeclareEnricher, "get_log", lambda self, cc: (log, ["t1", "t2"])
    )

    result = enricher.get_constraints_via_discovery(["order"])
    mined = set(result.constraints.columns) - {"trace_id", "case_concept"}

    assert mined == {"Response__a__b__", "Init__a__"}


def test_discovery_refuses_a_misspelled_template(enricher):
    with pytest.raises(ValueError, match="Responce"):
        enricher.get_constraints_via_discovery(["order"], constraint_set=["Responce"])


@pytest.mark.parametrize(
    ("column", "expected"),
    [
        ("Response[a, b] | | |", "Response"),
        ("Init[a] | |", "Init"),
        ("Exclusive Choice[a, b] | | |", "Exclusive Choice"),
        ("trace_id", "trace_id"),
    ],
)
def test_the_template_of_a_column_is_what_comes_before_the_activities(column, expected):
    assert template_of(column) == expected


# --------------------------------------------------------------------------
# Making the miner output fit to become class names
# --------------------------------------------------------------------------


def test_brackets_and_commas_become_underscores(enricher):
    frame = pd.DataFrame({"Response[a, b] | | |": [1, 0], "trace_id": ["t1", "t2"]})

    result = enricher.postprocess_results({"order": frame})

    assert "Response__a__b__" in result.columns


def test_a_constraint_that_holds_of_every_trace_is_dropped(enricher):
    """A class every trace belongs to says nothing about any of them."""
    frame = pd.DataFrame(
        {
            "Init[a] | |": [1, 1],
            "Response[a, b] | | |": [1, 0],
            "trace_id": ["t1", "t2"],
        }
    )

    result = enricher.postprocess_results({"order": frame})

    assert "Init__a__" not in result.columns
    assert "Response__a__b__" in result.columns


def test_a_zero_one_column_becomes_boolean(enricher):
    frame = pd.DataFrame({"Response[a, b] | | |": [1, 0], "trace_id": ["t1", "t2"]})

    result = enricher.postprocess_results({"order": frame})

    assert result["Response__a__b__"].dtype == bool


def test_each_frame_is_labelled_with_its_case_concept(enricher):
    frames = {
        cc: pd.DataFrame({"Response[a, b] | | |": [1, 0], "trace_id": ["x", "y"]})
        for cc in ("order", "item")
    }

    result = enricher.postprocess_results(frames)

    assert list(result["case_concept"]) == ["order", "order", "item", "item"]


# --------------------------------------------------------------------------
# Flattening, which is where the trace ids come from
# --------------------------------------------------------------------------


def test_get_log_returns_the_trace_ids_in_the_order_the_log_holds_them(monkeypatch):
    """`trace_id` is assigned to the checker's rows positionally, so the two
    orders have to agree."""
    monkeypatch.setattr(declare_module, "_DECLARE4PY_AVAILABLE", True)
    monkeypatch.setattr(declare_module, "D4PyEventLog", FakeEventLog, raising=False)
    enricher = DeclareEnricher(stores.build("awkward"))

    log, case_ids = enricher.get_log("order")

    assert case_ids == [trace.attributes["concept:name"] for trace in log.wrapped]
    assert set(case_ids) <= set(
        stores.build("awkward").get_object_table("order").data["oid"]
    )
