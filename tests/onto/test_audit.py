from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core.enums import EventOrderRepresentation
from semanticlog.core.errors import ConfigError, InputError
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.audit import audit_build
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings

DF = OntoBuildSettings(order_representations=[EventOrderRepresentation.DF])
DECLARE = OntoBuildSettings(order_representations=[EventOrderRepresentation.DECLARE])


def _store(object_type: str = "order", relation_eid: str = "e1") -> OCEDStore:
    """One event and one object, linked by one relation row."""
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
    data.add_relation(
        RelationKind.E2O,
        "rel",
        pd.DataFrame({"eid": [relation_eid], "oid": ["o1"]}),
        from_id="eid",
        to_id="oid",
        to_type=[object_type],
    )
    return data


def _declare_audit(constraints: dict) -> None:
    audit_build(
        clean_oced_data(_store()),
        DECLARE,
        BuildInputs(
            case_concepts=["order"], declare_constraints=pd.DataFrame(constraints)
        ),
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


# --------------------------------------------------------------------------
# DECLARE constraints
# --------------------------------------------------------------------------


def test_constraints_missing_an_index_column_name_it():
    with pytest.raises(InputError, match="no 'case_concept' column"):
        _declare_audit({"trace_id": ["o1"], "Response__a__b__": [True]})


def test_a_constraint_row_with_no_case_concept_is_rejected():
    with pytest.raises(InputError, match="1 row\\(s\\) have no case concept"):
        _declare_audit(
            {
                "case_concept": ["order", None],
                "trace_id": ["o1", "ghost"],
                "Response__a__b__": [True, True],
            }
        )


def test_a_constraint_case_concept_the_log_lacks_is_rejected():
    with pytest.raises(InputError, match="case concept 'invoice' is not an object"):
        _declare_audit(
            {
                "case_concept": ["invoice"],
                "trace_id": ["i1"],
                "Response__a__b__": [True],
            }
        )


def test_a_trace_id_that_is_no_object_of_its_case_concept_is_rejected():
    with pytest.raises(InputError, match="1 of 2 'order' trace id.* starting with o9"):
        _declare_audit(
            {
                "case_concept": ["order", "order"],
                "trace_id": ["o1", "o9"],
                "Response__a__b__": [True, False],
            }
        )


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
