"""Directly-follows, and the two ways eventually-follows can be expressed.

Both forms describe the same closure. EF asserts a transitivity axiom and
leaves the pairs to a reasoner, EF_EXPLICIT asserts every pair. The tests that
run a reasoner check that the two agree.

The tiny store is one trace, so it says what a representation does within a
trace. The matrix store is twelve traces over two case concepts, so it says
where each one stops.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import (
    EventOrderRepresentation,
    OWLProfile,
)
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from semanticlog.onto.settings import OntoBuildSettings
from tests.conftest import IN_MEMORY, emit
from tests.support.ontology import read_rdf

NS = DEFAULT_IRI

DF = EventOrderRepresentation.DF
EF = EventOrderRepresentation.EF
EF_EXPLICIT = EventOrderRepresentation.EF_EXPLICIT

EVENTS = ("e1", "e2", "e3")

# The three events of the tiny store are one trace, so eventually-follows is
# the reflexive transitive closure over them.
CLOSURE = {
    ("e1", "e1"),
    ("e1", "e2"),
    ("e1", "e3"),
    ("e2", "e2"),
    ("e2", "e3"),
    ("e3", "e3"),
}

# The matrix store flattened on `order`: twelve traces, the first six of two
# events and the rest of one.
MATRIX_ORDER_TRACES = [(f"e{i}", f"s{i}") for i in range(1, 7)] + [
    (f"e{i}",) for i in range(7, 13)
]

# Flattened on `item` instead: e1 and s1 both touch i1, and the other three
# items are touched once each.
MATRIX_ITEM_TRACES = [("e1", "s1"), ("e2",), ("s2",), ("s3",)]


def _directly_follows(traces) -> set[tuple[str, str]]:
    """Return the consecutive pairs of each trace."""
    return {pair for trace in traces for pair in zip(trace, trace[1:], strict=False)}


def _closure(traces) -> set[tuple[str, str]]:
    """Return the reflexive transitive closure within each trace."""
    return {
        (trace[i], trace[j])
        for trace in traces
        for i in range(len(trace))
        for j in range(i, len(trace))
    }


def _settings(order, **overrides) -> OntoBuildSettings:
    base: dict = {
        "profile": OWLProfile.OWL2DL,
        "case_concepts": ["order"],
        "order_representations": order,
    }
    base.update(overrides)
    return OntoBuildSettings(**base)


def _pairs(onto, prop: str) -> set[tuple[str, str]]:
    return {(s, o) for p, s, o in onto.relations if p == prop}


def test_directly_follows_links_each_event_to_the_next(tiny_store, tmp_path):
    onto = emit(_settings([DF]), tmp_path / "df.ofn", data=tiny_store)

    assert _pairs(onto, "df_order") == {("e1", "e2"), ("e2", "e3")}


def test_ef_explicit_asserts_the_whole_closure(tiny_store, tmp_path):
    onto = emit(
        _settings([DF, EF_EXPLICIT]), tmp_path / "explicit.ofn", data=tiny_store
    )

    assert _pairs(onto, "ef_order") == CLOSURE


def test_ef_axiomatic_asserts_only_the_self_loops(tiny_store, tmp_path):
    onto = emit(_settings([DF, EF]), tmp_path / "axiomatic.ofn", data=tiny_store)

    assert _pairs(onto, "ef_order") == {(e, e) for e in EVENTS}
    assert "ef_order" in onto.transitive


def test_a_second_relation_to_one_order_does_not_make_an_event_follow_itself(
    tiny_store, tmp_path
):
    """OCEL logs qualify e2o links, so one pair can arrive under two names."""
    concerns = tiny_store.get_relation("concerns")
    tiny_store.add_event_to_object_relation(
        "creates",
        from_id_col=concerns.from_id,
        to_id_col=concerns.to_id,
        to_type=list(concerns.to_type),
        data=concerns.data.copy(),
    )

    onto = emit(_settings([DF, EF_EXPLICIT]), tmp_path / "twice.ofn", data=tiny_store)

    assert _pairs(onto, "df_order") == {("e1", "e2"), ("e2", "e3")}
    assert _pairs(onto, "ef_order") == CLOSURE


# --------------------------------------------------------------------------
# Where each representation stops
# --------------------------------------------------------------------------


def test_directly_follows_over_many_traces_pairs_within_each(tmp_path):
    onto = emit(_settings([DF]), tmp_path / "many.ofn")

    assert _pairs(onto, "df_order") == _directly_follows(MATRIX_ORDER_TRACES)


def test_ef_explicit_over_many_traces_closes_within_each(tmp_path):
    onto = emit(_settings([DF, EF_EXPLICIT]), tmp_path / "many.ofn")
    pairs = _pairs(onto, "ef_order")

    # Six traces of two events give three pairs each, six of one give one.
    assert len(pairs) == 24
    assert pairs == _closure(MATRIX_ORDER_TRACES)


def test_ef_explicit_relates_no_two_events_of_different_traces(tmp_path):
    onto = emit(_settings([DF, EF_EXPLICIT]), tmp_path / "many.ofn")
    trace_of = {event: trace[0] for trace in MATRIX_ORDER_TRACES for event in trace}

    crossing = {
        (s, o) for s, o in _pairs(onto, "ef_order") if trace_of[s] != trace_of[o]
    }
    assert not crossing, f"eventually-follows left its trace: {sorted(crossing)}"


def test_ef_axiomatic_over_many_traces_asserts_one_self_pair_per_event(tmp_path):
    onto = emit(_settings([DF, EF]), tmp_path / "many.ofn")
    events = {event for trace in MATRIX_ORDER_TRACES for event in trace}

    assert _pairs(onto, "ef_order") == {(e, e) for e in events}


def test_each_case_concept_orders_its_own_events(tmp_path):
    onto = emit(
        _settings([DF, EF_EXPLICIT], case_concepts=["order", "item"]),
        tmp_path / "two.ofn",
    )

    assert _pairs(onto, "df_item") == _directly_follows(MATRIX_ITEM_TRACES)
    assert _pairs(onto, "ef_item") == _closure(MATRIX_ITEM_TRACES)


def test_an_event_reached_by_two_case_concepts_sits_in_both_chains(tmp_path):
    onto = emit(_settings([DF], case_concepts=["order", "item"]), tmp_path / "two.ofn")

    # e1 places o1 and handles i1, so it starts both of their chains.
    assert ("e1", "s1") in _pairs(onto, "df_order")
    assert ("e1", "s1") in _pairs(onto, "df_item")
    assert {"StartEvent_order", "StartEvent_item"} <= onto.types["e1"]


def test_the_order_is_the_order_of_the_timestamps(tmp_path):
    data = OCEDStore()
    data.add_event(
        "step",
        id_col="eid",
        time_col="ts",
        # Stored as a, b, c but stamped c, a, b.
        data=pd.DataFrame(
            {
                "eid": ["a", "b", "c"],
                "ts": [
                    datetime(2024, 1, 2),
                    datetime(2024, 1, 3),
                    datetime(2024, 1, 1),
                ],
            }
        ),
    )
    data.add_object("order", id_col="oid", data=pd.DataFrame({"oid": ["o1"]}))
    data.add_event_to_object_relation(
        "concerns",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["a", "b", "c"], "oid": ["o1"] * 3}),
    )

    onto = emit(_settings([DF]), tmp_path / "shuffled.ofn", data=clean_oced_data(data))

    assert _pairs(onto, "df_order") == {("c", "a"), ("a", "b")}
    assert onto.types["c"] == {"step", "StartEvent_order"}
    assert onto.types["b"] == {"step", "EndEvent_order"}


# --------------------------------------------------------------------------
# The property hierarchy each representation hangs off
# --------------------------------------------------------------------------


def test_a_case_order_property_sits_under_the_order_super_property(tmp_path):
    onto = emit(_settings([DF, EF_EXPLICIT]), tmp_path / "hierarchy.ofn")

    assert onto.property_parents["df_order"] == {vocab.DF}
    assert onto.property_parents["ef_order"] == {vocab.EF}


def test_ef_axiomatic_puts_directly_follows_under_eventually_follows(tmp_path):
    """Transitivity alone derives nothing. The closure a reasoner reads out
    comes from the directly-follows pairs being eventually-follows pairs too.
    """
    onto = emit(_settings([DF, EF]), tmp_path / "axiomatic.ofn")

    assert onto.property_parents["df_order"] == {vocab.DF, "ef_order"}
    assert "ef_order" in onto.transitive


def test_ef_explicit_leaves_the_two_order_properties_unrelated(tmp_path):
    onto = emit(_settings([DF, EF_EXPLICIT]), tmp_path / "explicit.ofn")

    assert "ef_order" not in onto.property_parents["df_order"]
    assert not onto.transitive


def test_ef_explicit_alone_emits_no_directly_follows(tmp_path):
    """EF_EXPLICIT asserts its own pairs, so it needs nothing to derive them
    over. Only EF does, and it says so in validate().
    """
    onto = emit(_settings([EF_EXPLICIT]), tmp_path / "ef_only.ofn")

    assert "ef_order" in onto.object_properties
    assert "df_order" not in onto.object_properties
    assert vocab.DF not in onto.object_properties
    assert not _pairs(onto, "df_order")


def test_ef_explicit_alone_still_bounds_its_traces(tmp_path):
    """The start and end classes describe the projection, not directly-follows."""
    onto = emit(_settings([EF_EXPLICIT]), tmp_path / "ef_only_bounds.ofn")

    assert {"StartEvent_order", "EndEvent_order"} <= onto.classes


def test_ef_explicit_refuses_a_closure_over_budget(tiny_store):
    settings = _settings([EF_EXPLICIT], max_ef_pairs=3)

    with pytest.raises(ValueError, match="over the max_ef_pairs limit"):
        OCEDOntoBuilder(NS, settings).build_ontology(tiny_store)


def test_ef_explicit_within_budget_builds(tiny_store, tmp_path):
    onto = emit(
        _settings([EF_EXPLICIT], max_ef_pairs=6),
        tmp_path / "within.ofn",
        data=tiny_store,
    )

    assert _pairs(onto, "ef_order") == CLOSURE


@pytest.mark.reasoner
def test_reasoning_over_ef_axiomatic_yields_the_closure_ef_explicit_asserts(
    tiny_store, tmp_path
):
    from owlapy.owl_individual import OWLNamedIndividual
    from owlapy.owl_property import OWLObjectProperty

    path = tmp_path / "axiomatic.owl"
    emit(_settings([DF, EF]), path, backend=IN_MEMORY, data=tiny_store)

    reasoner = Reasoner(str(path), profile=OWLProfile.OWL2DL).reasoner
    ef = OWLObjectProperty(NS + "ef_order")
    derived = {
        (event, value.iri.remainder)
        for event in EVENTS
        for value in reasoner.object_property_values(OWLNamedIndividual(NS + event), ef)
    }

    assert derived == CLOSURE


@pytest.mark.reasoner
def test_materializing_writes_the_closure_back_to_the_file(tiny_store, tmp_path):
    path = tmp_path / "materialized.owl"
    emit(_settings([DF, EF]), path, backend=IN_MEMORY, data=tiny_store)

    Reasoner(str(path), profile=OWLProfile.OWL2DL).materialize()

    assert _pairs(read_rdf(path, NS), "ef_order") == CLOSURE
