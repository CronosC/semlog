"""Factories for the small stores the cleaning tests build."""

import pandas as pd
import pytest

from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind


def _events(*ids, **attributes):
    """Return an event table with one timestamp per id and the given attribute columns."""
    return pd.DataFrame(
        {
            "eid": list(ids),
            "ts": pd.date_range("2022-01-01", periods=len(ids), tz="UTC"),
            **attributes,
        }
    )


def _store(objects=None, obj_type="obj", activity="act", events=None, ocel_mode=False):
    """Return a store of one activity and one object type, with the tables given or one row each."""
    oced = OCEDStore(ocel_mode=ocel_mode)
    oced.add(
        EntityKind.EVENT,
        activity,
        events if events is not None else _events("e1"),
        id_col="eid",
        time_col="ts",
    )
    oced.add(
        EntityKind.OBJECT,
        obj_type,
        objects if objects is not None else pd.DataFrame({"oid": ["o1"]}),
        id_col="oid",
    )
    return oced


@pytest.fixture
def events():
    return _events


@pytest.fixture
def make_store():
    return _store
