"""`TimeEmitter`, for an event with no timestamp."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from semlog.core import vocabulary as vocab
from semlog.data.cleaning import clean_oced_data
from semlog.data.oced import OCEDStore
from semlog.data.tables import EntityKind, RelationKind
from tests.support.build import emit
from tests.support.settings import CONFIGURATIONS


@pytest.fixture
def undated(tmp_path):
    """Three events on one order, the last of which is undated, built."""
    data = OCEDStore()
    data.add(
        EntityKind.EVENT,
        "step",
        pd.DataFrame(
            {
                "eid": ["e1", "e2", "e3"],
                "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2), None],
            }
        ),
        id_col="eid",
        time_col="ts",
    )
    data.add(EntityKind.OBJECT, "order", pd.DataFrame({"oid": ["o1"]}), id_col="oid")
    data.add_relation(
        RelationKind.E2O,
        "concerns",
        pd.DataFrame({"eid": ["e1", "e2", "e3"], "oid": ["o1", "o1", "o1"]}),
        from_id="eid",
        to_id="oid",
        to_type=["order"],
    )
    return emit(
        CONFIGURATIONS["dl_df"](), tmp_path / "undated.ofn", data=clean_oced_data(data)
    )


def test_an_undated_event_gets_no_time_individual(undated):
    observed = {s for p, s, _ in undated.relations if p == vocab.OBSERVED_AT}
    times = {name for name, classes in undated.types.items() if vocab.TIME in classes}

    assert observed == {"e1", "e2"}
    assert len(times) == 2


def test_an_undated_event_is_still_an_event_of_its_activity(undated):
    assert "step" in undated.types["e3"]
