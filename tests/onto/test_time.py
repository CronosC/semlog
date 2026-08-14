"""Time individuals, and the events that have no moment to point at.

A timestamp becomes an individual so that events sharing a moment share it,
which the matrix store covers. What it does not cover is an event whose
timestamp the log left empty, so these build their own store.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from semanticlog.core import vocabulary as vocab
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from tests.conftest import BACKENDS, emit
from tests.support.settings import CONFIGURATIONS


def _store_with_a_missing_timestamp() -> OCEDStore:
    """Three events on one order, the last of which is undated."""
    data = OCEDStore()
    data.add_event(
        "step",
        id_col="eid",
        time_col="ts",
        data=pd.DataFrame(
            {
                "eid": ["e1", "e2", "e3"],
                "ts": [datetime(2024, 1, 1), datetime(2024, 1, 2), None],
            }
        ),
    )
    data.add_object("order", id_col="oid", data=pd.DataFrame({"oid": ["o1"]}))
    data.add_event_to_object_relation(
        "concerns",
        from_id_col="eid",
        to_id_col="oid",
        to_type=["order"],
        data=pd.DataFrame({"eid": ["e1", "e2", "e3"], "oid": ["o1", "o1", "o1"]}),
    )
    return clean_oced_data(data)


@pytest.fixture(params=BACKENDS)
def undated(request, tmp_path):
    """The store with an undated event, built through each sink."""
    suffix = "ofn" if request.param == "streaming" else "owl"
    return emit(
        CONFIGURATIONS["dl_df"](),
        tmp_path / f"undated.{suffix}",
        request.param,
        data=_store_with_a_missing_timestamp(),
    )


def test_an_undated_event_produces_no_time_individual(undated):
    times = {name for name, classes in undated.types.items() if vocab.TIME in classes}

    assert len(times) == 2, "two events are dated, the third is not"
    assert "nan" not in undated.individuals


def test_an_undated_event_is_still_an_event(undated):
    assert "e3" in undated.types
    assert "step" in undated.types["e3"]


def test_an_undated_event_points_at_no_moment(undated):
    observed = {s for p, s, _ in undated.relations if p == vocab.OBSERVED_AT}

    assert observed == {"e1", "e2"}
