import pandas as pd
import pytest

from semanticlog.data.oced import OCEDStore


@pytest.fixture
def event_df_1():
    return pd.DataFrame(
        {
            "event": ["e1", "e2", "e5"],
            "timestamp": pd.date_range("2022-01-01", periods=3, freq="D", tz="UTC"),
        }
    )


@pytest.fixture
def event_df_2():
    return pd.DataFrame(
        {
            "event": ["e3", "e4"],
            "timestamp": pd.date_range("2022-01-01", periods=2, freq="D", tz="UTC"),
        }
    )


@pytest.fixture
def object_df_a():
    return pd.DataFrame(
        {"object": [1, 2, 3], "object_attribute": ["attr1", "attr2", "attr3"]}
    )


@pytest.fixture
def object_df_b():
    return pd.DataFrame({"object": [4, 5], "object_attribute": ["attr4", "attr5"]})


@pytest.fixture
def e2o_data():
    return pd.DataFrame(
        {"from_id": ["e1", "e2", "e3", "e4", "e5"], "to_id": [1, 1, 2, 2, 2]}
    )


@pytest.fixture
def o2o_data():
    return pd.DataFrame({"from_id": [1, 1, 4, 5], "to_id": [1, 3, 2, 1]})


@pytest.fixture
def minimal_oced(event_df_1, event_df_2, object_df_a, object_df_b, e2o_data, o2o_data):
    oced = OCEDStore()
    oced.add_event("type1", "event", "timestamp", event_df_1)
    oced.add_event("type2", "event", "timestamp", event_df_2)
    oced.add_object("A", "object", object_df_a)
    oced.add_object("B", "object", object_df_b)
    oced.add_event_to_object_relation("rel1", "from_id", "to_id", ["A"], e2o_data)
    oced.add_object_to_object_relation(
        "rel2", "from_id", ["A"], "to_id", ["B"], o2o_data
    )
    return oced
