"""`OrderEmitter`: directly-follows and the two forms of eventually-follows.

`ef` asserts a transitivity axiom, `ef_explicit` asserts every pair. The tiny
store is one trace, the matrix store twelve traces over two case concepts.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import (
    EventOrderRepresentation,
    InverseRepresentation,
    OWLProfile,
)
from semanticlog.core.errors import ConfigError
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.declare import ConceptConstraints, DeclareResult
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.reason import Reasoner
from tests.support.build import STREAMING, emit
from tests.support.functional import read_functional
from tests.support.settings import (
    CONFIGURATIONS,
    DISCOVERY,
    Configuration,
    configuration,
    ragged_declare_constraints,
)

DF = EventOrderRepresentation.DF
EF = EventOrderRepresentation.EF
EF_EXPLICIT = EventOrderRepresentation.EF_EXPLICIT

EVENTS = ("e1", "e2", "e3")

# The three events of the tiny store are one trace.
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

# Flattened on `item`: e1 and s1 both touch i1, the other three items are
# touched once each.
MATRIX_ITEM_TRACES = [("e1", "s1"), ("e2",), ("s2",), ("s3",)]


def _directly_follows(traces) -> set[tuple[str, str]]:
    return {pair for trace in traces for pair in zip(trace, trace[1:], strict=False)}


def _closure(traces) -> set[tuple[str, str]]:
    """Return the reflexive transitive closure within each trace."""
    return {
        (trace[i], trace[j])
        for trace in traces
        for i in range(len(trace))
        for j in range(i, len(trace))
    }


def _settings(order, **overrides) -> Configuration:
    base: dict = {
        "profile": OWLProfile.OWL2DL,
        "case_concepts": ["order"],
        "order_representations": order,
    }
    base.update(overrides)
    return configuration(**base)


def _pairs(onto, prop: str) -> set[tuple[str, str]]:
    return {(s, o) for p, s, o in onto.relations if p == prop}


def _one_order(events: pd.DataFrame) -> OCEDStore:
    """One order that every event in the frame concerns, cleaned."""
    data = OCEDStore()
    data.add(EntityKind.EVENT, "step", events, id_col="eid", time_col="ts")
    data.add(EntityKind.OBJECT, "order", pd.DataFrame({"oid": ["o1"]}), id_col="oid")
    data.add_relation(
        RelationKind.E2O,
        "concerns",
        pd.DataFrame({"eid": events["eid"], "oid": ["o1"] * len(events)}),
        from_id="eid",
        to_id="oid",
        to_type=["order"],
    )
    return clean_oced_data(data)


# --------------------------------------------------------------------------
# The pairs each representation asserts
# --------------------------------------------------------------------------


def test_directly_follows_pairs_consecutive_events_within_each_trace(tmp_path):
    onto = emit(_settings([DF]), tmp_path / "many.ofn")

    assert _pairs(onto, "df_order") == _directly_follows(MATRIX_ORDER_TRACES)


def test_ef_explicit_asserts_the_closure_within_each_trace(tmp_path):
    onto = emit(_settings([DF, EF_EXPLICIT]), tmp_path / "many.ofn")

    assert _pairs(onto, "ef_order") == _closure(MATRIX_ORDER_TRACES)


def test_ef_asserts_one_self_pair_per_event_and_a_transitivity_axiom(tmp_path):
    onto = emit(_settings([DF, EF]), tmp_path / "many.ofn")
    events = {event for trace in MATRIX_ORDER_TRACES for event in trace}

    assert _pairs(onto, "ef_order") == {(e, e) for e in events}
    assert onto.transitive == {"ef_order"}


def test_each_case_concept_orders_its_own_events(tmp_path):
    onto = emit(
        _settings([DF, EF_EXPLICIT], case_concepts=["order", "item"]),
        tmp_path / "two.ofn",
    )

    assert _pairs(onto, "df_item") == _directly_follows(MATRIX_ITEM_TRACES)
    assert _pairs(onto, "ef_item") == _closure(MATRIX_ITEM_TRACES)


def test_an_event_reached_by_two_case_concepts_sits_in_both_chains(tmp_path):
    onto = emit(_settings([DF], case_concepts=["order", "item"]), tmp_path / "two.ofn")

    assert ("e1", "s1") in _pairs(onto, "df_order")
    assert ("e1", "s1") in _pairs(onto, "df_item")
    assert {"StartEvent_order", "StartEvent_item"} <= onto.types["e1"]


def test_a_second_relation_to_one_order_does_not_make_an_event_follow_itself(
    tiny_store, tmp_path
):
    """OCEL logs qualify e2o links, so one pair can arrive under two names."""
    concerns = tiny_store.get_relation("concerns")
    tiny_store.add_relation(
        RelationKind.E2O,
        "creates",
        concerns.data.copy(),
        from_id=concerns.from_id,
        to_id=concerns.to_id,
        to_type=list(concerns.to_type),
    )

    onto = emit(_settings([DF, EF_EXPLICIT]), tmp_path / "twice.ofn", data=tiny_store)

    assert _pairs(onto, "df_order") == {("e1", "e2"), ("e2", "e3")}
    assert _pairs(onto, "ef_order") == CLOSURE


def test_events_follow_each_other_in_timestamp_order(tmp_path):
    events = pd.DataFrame(
        {
            "eid": ["a", "b", "c"],
            "ts": [datetime(2024, 1, 2), datetime(2024, 1, 3), datetime(2024, 1, 1)],
        }
    )

    onto = emit(_settings([DF]), tmp_path / "shuffled.ofn", data=_one_order(events))

    assert _pairs(onto, "df_order") == {("c", "a"), ("a", "b")}
    assert onto.types["c"] == {"step", "StartEvent_order"}
    assert onto.types["b"] == {"step", "EndEvent_order"}


def test_events_at_one_instant_are_ordered_by_event_id(tmp_path):
    events = pd.DataFrame({"eid": ["c", "a", "b"], "ts": [datetime(2024, 1, 1)] * 3})

    onto = emit(_settings([DF]), tmp_path / "tie.ofn", data=_one_order(events))

    assert _pairs(onto, "df_order") == {("a", "b"), ("b", "c")}


def test_an_event_in_no_relation_is_in_no_trace(tiny_store, tmp_path):
    table = tiny_store.table(EntityKind.EVENT, "step")
    table.data = pd.concat(
        [
            table.data,
            pd.DataFrame(
                {
                    "eid": ["e4"],
                    "ts": pd.to_datetime(["2024-01-04 09:00"]),
                    "step.channel": ["web"],
                    "step.price": [1.0],
                }
            ),
        ],
        ignore_index=True,
    )
    tiny_store.update_used_ids()

    onto = emit(_settings([DF]), tmp_path / "loose.ofn", data=tiny_store)

    assert onto.types["e4"] == {"step"}
    assert {e for pair in _pairs(onto, "df_order") for e in pair} == set(EVENTS)


# --------------------------------------------------------------------------
# The property hierarchy each representation hangs off
# --------------------------------------------------------------------------


def test_ef_explicit_puts_each_order_property_under_its_super_property_only(tmp_path):
    onto = emit(_settings([DF, EF_EXPLICIT]), tmp_path / "hierarchy.ofn")

    assert onto.property_parents["df_order"] == {vocab.DF}
    assert onto.property_parents["ef_order"] == {vocab.EF}
    assert onto.transitive == set()


def test_ef_puts_directly_follows_under_eventually_follows(tmp_path):
    onto = emit(_settings([DF, EF]), tmp_path / "axiomatic.ofn")

    assert onto.property_parents["df_order"] == {vocab.DF, "ef_order"}


def test_ef_explicit_alone_bounds_its_traces_without_directly_follows(tmp_path):
    onto = emit(_settings([EF_EXPLICIT]), tmp_path / "ef_only.ofn")

    assert {"df_order", vocab.DF} & onto.object_properties == set()
    assert "ef_order" in onto.object_properties
    assert {"StartEvent_order", "EndEvent_order"} <= onto.classes


def test_ef_with_implicit_inverses_makes_the_inverse_transitive_too(
    tiny_store, tmp_path
):
    """With no inverse axiom, `inv_ef_order` needs its own transitivity."""
    onto = emit(
        _settings([DF, EF], inverse_representations=[InverseRepresentation.IMPLICIT]),
        tmp_path / "implicit.ofn",
        data=tiny_store,
    )

    assert onto.transitive == {"ef_order", "inv_ef_order"}
    assert "inv_ef_order" in onto.property_parents["inv_df_order"]
    assert _pairs(onto, "inv_ef_order") == {(e, e) for e in EVENTS}


def test_ef_with_an_inverse_axiom_leaves_the_inverse_to_the_reasoner(
    tiny_store, tmp_path
):
    onto = emit(
        _settings(
            [DF, EF],
            inverse_representations=[
                InverseRepresentation.AXIOMATIC,
                InverseRepresentation.IMPLICIT,
            ],
        ),
        tmp_path / "axiomatic_inverse.ofn",
        data=tiny_store,
    )

    assert onto.transitive == {"ef_order"}
    assert ("ef_order", "inv_ef_order") in onto.inverses


def test_ef_explicit_over_max_ef_pairs_is_rejected(tiny_store):
    config = _settings([EF_EXPLICIT], max_ef_pairs=3)

    with pytest.raises(ConfigError, match="6 pairs .* over max_ef_pairs=3"):
        OCEDOntoBuilder(DEFAULT_IRI, config.settings).build_ontology(
            tiny_store, config.inputs
        )


@pytest.mark.reasoner
def test_a_reasoner_derives_from_ef_the_pairs_ef_explicit_asserts(tmp_path):
    axiomatic = tmp_path / "axiomatic.ofn"
    explicit = tmp_path / "explicit.ofn"
    emit(CONFIGURATIONS["dl_df_ef"](), axiomatic, STREAMING)
    asserted = _pairs(
        emit(CONFIGURATIONS["dl_ef_explicit"](), explicit, STREAMING), "ef_order"
    )

    entailed = {
        (
            str(axiom.get_subject().iri.get_remainder()),
            str(axiom.get_object().iri.get_remainder()),
        )
        for axiom in Reasoner(
            str(axiomatic), backend="Pellet"
        ).inferred_property_assertions()
        if str(axiom.get_property().iri.as_str()) == f"{DEFAULT_IRI}ef_order"
    }
    # A backend may answer with every entailed pair or only the new ones.
    reached = _pairs(read_functional(axiomatic), "ef_order") | entailed

    assert reached == asserted


# --------------------------------------------------------------------------
# DECLARE
# --------------------------------------------------------------------------


def _shared_constraint() -> DeclareResult:
    """One constraint that an order trace and an item trace satisfy."""
    return DeclareResult(
        concepts=(
            ConceptConstraints(
                "order", ("o1", "o2"), {"Existence__ship__": frozenset({"o1"})}
            ),
            ConceptConstraints(
                "item", ("i1", "i2"), {"Existence__ship__": frozenset({"i1"})}
            ),
        ),
        mining=DISCOVERY,
    )


def _declare_settings(declare: DeclareResult, profile=OWLProfile.OWL2DL):
    return _settings(
        [EventOrderRepresentation.DECLARE],
        profile=profile,
        case_concepts=["order", "item"],
        declare=declare,
    )


def test_the_constraint_hierarchy_is_below_object(tmp_path):
    onto = emit(_declare_settings(ragged_declare_constraints()), tmp_path / "d.ofn")

    assert onto.parents[vocab.DECLARE_CONSTRAINT] == {vocab.OBJECT}


def test_a_constraint_satisfied_in_one_concept_is_below_its_object_type(tmp_path):
    onto = emit(_declare_settings(ragged_declare_constraints()), tmp_path / "d.ofn")

    assert onto.parents["Existence__ship__"] == {vocab.DECLARE_CONSTRAINT, "item"}


def test_a_constraint_satisfied_in_two_concepts_is_below_their_union_under_dl(
    tmp_path,
):
    onto = emit(_declare_settings(_shared_constraint()), tmp_path / "d.ofn")

    assert (
        "SubClassOf(:Existence__ship__ ObjectUnionOf(:order :item))" in onto.axiom_lines
    )


def test_a_constraint_satisfied_in_two_concepts_has_one_parent_under_el(tmp_path):
    onto = emit(
        _declare_settings(_shared_constraint(), OWLProfile.OWL2EL), tmp_path / "d.ofn"
    )

    assert onto.parents["Existence__ship__"] == {vocab.DECLARE_CONSTRAINT}
