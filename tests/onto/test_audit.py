"""The checks a build runs before it emits anything."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core.enums import EventOrderRepresentation
from semanticlog.core.errors import ConfigError, InputError
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.declare import ConceptConstraints, DeclareResult, MiningSettings
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.audit import audit_build
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings

DF = OntoBuildSettings(order_representations=[EventOrderRepresentation.DF])
DECLARE = OntoBuildSettings(order_representations=[EventOrderRepresentation.DECLARE])


def _store(object_type: str = "order", relation_eid: str | None = "e1") -> OCEDStore:
    """One event and one object, linked by one relation row unless `relation_eid` is None."""
    data = OCEDStore()
    data.add(
        EntityKind.EVENT,
        "act",
        pd.DataFrame({"eid": ["e1"], "t": [datetime(2024, 1, 1)]}),
        id_col="eid",
        time_col="t",
    )
    data.add(
        EntityKind.OBJECT, object_type, pd.DataFrame({"oid": ["o1"]}), id_col="oid"
    )
    data.add(EntityKind.OBJECT, "pallet", pd.DataFrame({"oid": ["p1"]}), id_col="oid")
    if relation_eid is None:
        return data
    data.add_relation(
        RelationKind.E2O,
        "rel",
        pd.DataFrame({"eid": [relation_eid], "oid": ["o1"]}),
        from_id="eid",
        to_id="oid",
        to_type=[object_type],
    )
    return data


def _declare_audit(case_concept: str, trace_ids: tuple[str, ...]) -> None:
    declare = DeclareResult(
        concepts=(
            ConceptConstraints(
                case_concept, trace_ids, {"Response__a__b__": frozenset(trace_ids[:1])}
            ),
        ),
        mining=MiningSettings("discovery", None, 0.5, 0.5),
    )
    audit_build(
        clean_oced_data(_store()),
        DECLARE,
        BuildInputs(case_concepts=["order"], declare=declare),
    )


# --------------------------------------------------------------------------
# Case concepts
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("store", "case_concept", "match"),
    [
        (_store(), "no_such_type", "'no_such_type' is not an object type of this log"),
        (
            _store("purchase order"),
            "purchase order",
            "'purchase order' was renamed to 'purchase_order' by cleaning",
        ),
        (_store(), "pallet", "'pallet' is an object type of this log, but no event"),
    ],
    ids=["unknown", "renamed", "unreachable"],
)
def test_a_case_concept_the_log_cannot_be_projected_onto_is_rejected(
    store, case_concept, match
):
    with pytest.raises(ConfigError, match=match):
        audit_build(
            clean_oced_data(store), DF, BuildInputs(case_concepts=[case_concept])
        )


def test_a_rejected_case_concept_is_answered_with_the_types_the_setting_takes():
    with pytest.raises(ConfigError, match=r"Set case_concepts to one of: order\. "):
        audit_build(clean_oced_data(_store()), DF, BuildInputs(case_concepts=["x"]))


def test_a_log_without_event_to_object_relations_is_answered_with_no_types():
    store = _store(relation_eid=None)

    with pytest.raises(
        ConfigError,
        match=r"No event-to-object relation of this log names an object type, so "
        r"case_concepts cannot name one\. Add event-to-object relations to the "
        r"log, or remove df, ef and ef_explicit from order_representations\.",
    ):
        audit_build(clean_oced_data(store), DF, BuildInputs(case_concepts=["order"]))


@pytest.mark.parametrize(
    ("store", "case_concept", "match"),
    [
        (_store(), "no_such_type", "'no_such_type' is not an object type of this log"),
        (_store(), "pallet", "'pallet' is an object type of this log, but no event"),
    ],
    ids=["unknown", "unreachable"],
)
def test_a_case_concept_the_log_cannot_be_projected_onto_is_rejected_without_event_order(
    store, case_concept, match
):
    with pytest.raises(ConfigError, match=match):
        audit_build(
            clean_oced_data(store),
            OntoBuildSettings(),
            BuildInputs(case_concepts=[case_concept]),
        )


def test_a_log_without_event_to_object_relations_passes_without_event_order():
    audit_build(
        clean_oced_data(_store(relation_eid=None)), OntoBuildSettings(), BuildInputs()
    )


# --------------------------------------------------------------------------
# DECLARE constraints
# --------------------------------------------------------------------------


def test_a_constraint_case_concept_the_log_lacks_is_rejected():
    with pytest.raises(InputError, match="case concept 'invoice' is not an object"):
        _declare_audit("invoice", ("i1",))


def test_a_trace_id_that_is_no_object_of_its_case_concept_is_rejected():
    with pytest.raises(InputError, match="1 of 2 'order' trace id.* starting with o9"):
        _declare_audit("order", ("o1", "o9"))


# --------------------------------------------------------------------------
# Every check at once
# --------------------------------------------------------------------------


def test_every_failed_check_is_reported_in_one_error():
    with pytest.raises(InputError, match=r"1 event id\(s\) are in no event table") as e:
        audit_build(
            _store(relation_eid="ghost"),
            DF,
            BuildInputs(case_concepts=["no_such_type"]),
        )

    assert "'no_such_type' is not an object type of this log" in str(e.value)
