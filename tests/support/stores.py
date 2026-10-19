"""The three logs the tests build from.

Every test that needs data uses one of these. Which one to reach for:

- `tiny` when the data is not what is under test. Three events in one trace,
  one string attribute and one numeric one.
- `pipeline` when a test should run against a real OCEL 2.0 file. Read from
  `tests/fixtures/pipeline.jsonocel`, described in the README next to it.
- `awkward_shared` when a test is about attributes several types carry, which
  `awkward` qualifies per type. It is `awkward` cleaned without
  `qualify_shared`.
- `awkward` when a hard case has to be reachable. It is also what the build
  configuration matrix builds from, so every configuration meets a name that
  cleaning rewrote, a missing value, two events at one instant, a numeric
  column wide enough to bin and a value at the far end of it.

A store is mutable, so the fixtures hand out a copy and these builders are
called once per session.
"""

from __future__ import annotations

import copy
from datetime import datetime
from pathlib import Path

import pandas as pd

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.loading.pm4py_loader import PM4PyLoader
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind, RelationKind

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PIPELINE_OCEL = FIXTURES / "pipeline.jsonocel"

_DAY = datetime(2024, 1, 1, 9, 0)


def _day(n: int) -> datetime:
    """Return day n of January 2024 at 09:00."""
    return datetime(2024, 1, n, 9, 0)


def tiny() -> OCEDStore:
    """Three events of one activity, on one order.

    The smallest store with a trace to order, an attribute of each kind and a
    relation to follow.
    """
    data = OCEDStore()
    data.add(
        EntityKind.EVENT,
        "step",
        pd.DataFrame(
            {
                "eid": ["e1", "e2", "e3"],
                "ts": [_day(1), _day(2), _day(3)],
                "channel": ["web", "shop", "web"],
                "price": [10.0, 25.5, 3.25],
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
    return clean_oced_data(data)


def pipeline() -> OCEDStore:
    """The committed OCEL 2.0 log, loaded and cleaned."""
    return clean_oced_data(PM4PyLoader(str(PIPELINE_OCEL)).get_oced())


def pipeline_raw() -> OCEDStore:
    """The committed OCEL 2.0 log as the loader produced it, before cleaning."""
    return PM4PyLoader(str(PIPELINE_OCEL)).get_oced()


# Eleven prices, wide enough for distribution binning to fit, with one value at
# the far end.
_PRICES = [
    3.25,
    7.5,
    10.0,
    12.75,
    18.0,
    21.5,
    25.5,
    30.0,
    33.25,
    41.0,
    120.0,
]

_ORDERS = [f"o{i}" for i in range(1, 13)]
_ITEMS = ["i1", "i2", "i3", "i4"]


def awkward(qualify_shared: bool = True) -> OCEDStore:
    """Twelve orders and four items, with every hard case a build has to handle.

    - `place order` and the value `by hand` are renamed by cleaning.
    - One order has no channel and one has no price.
    - `e11` and `e12` carry the same timestamp.
    - `price` spans eleven values, the last well outside the rest.
    - `channel` is carried by both object types and `clerk` by both
      activities. Each is qualified per type unless `qualify_shared` is off,
      and then its domain is a union.
    - `order` records attribute changes, for the attribute history extension.
    """
    data = OCEDStore(ocel_mode=True)
    data.add(
        EntityKind.OBJECT,
        "order",
        pd.DataFrame(
            {
                "oid": _ORDERS,
                "channel": [
                    "web",
                    "shop",
                    "by hand",
                    "web",
                    "shop",
                    "web",
                    "by hand",
                    "shop",
                    "web",
                    "web",
                    "shop",
                    None,
                ],
                "price": [*_PRICES, None],
            }
        ),
        id_col="oid",
    )
    data.add(
        EntityKind.OBJECT,
        "item",
        pd.DataFrame(
            {
                "iid": _ITEMS,
                "channel": ["web", "shop", "web", "by hand"],
                "weight": [1.5, 2.0, 0.75, 3.25],
            }
        ),
        id_col="iid",
    )
    data.add(
        EntityKind.EVENT,
        "place order",
        pd.DataFrame(
            {
                "eid": [f"e{i}" for i in range(1, 13)],
                "ts": [*(_day(n) for n in range(1, 12)), _day(11)],
                "clerk": ["ann", "bo", "ann", "cy", "bo", "ann"] * 2,
                "amount": list(range(1, 13)),
            }
        ),
        id_col="eid",
        time_col="ts",
    )
    data.add(
        EntityKind.EVENT,
        "ship",
        pd.DataFrame(
            {
                "eid": [f"s{i}" for i in range(1, 7)],
                "ts": [datetime(2024, 2, n, 9, 0) for n in range(1, 7)],
                "clerk": ["cy", "ann", "bo", "cy", "ann", "bo"],
                "due": [datetime(2024, 3, n) for n in range(1, 7)],
            }
        ),
        id_col="eid",
        time_col="ts",
    )
    data.add_relation(
        RelationKind.E2O,
        "concerns",
        pd.DataFrame(
            {
                "eid": [f"e{i}" for i in range(1, 13)] + [f"s{i}" for i in range(1, 7)],
                "oid": _ORDERS + _ORDERS[:6],
            }
        ),
        from_id="eid",
        to_id="oid",
        to_type=["order"],
    )
    data.add_relation(
        RelationKind.E2O,
        "handles",
        pd.DataFrame(
            {
                "eid": ["e1", "e2", "s1", "s2", "s3"],
                "iid": ["i1", "i2", "i1", "i3", "i4"],
            }
        ),
        from_id="eid",
        to_id="iid",
        to_type=["item"],
    )
    data.add_relation(
        RelationKind.O2O,
        "contains",
        pd.DataFrame(
            {"oid": ["o1", "o2", "o3", "o4"], "iid": _ITEMS},
        ),
        from_id="oid",
        from_type=["order"],
        to_id="iid",
        to_type=["item"],
    )
    # One string attribute and one numeric one change, so the history
    # extension meets both kinds of value.
    data.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1", "o1", "o2", "o1", "o2"],
                "ocel:type": ["order"] * 5,
                "ocel:timestamp": [_day(2), _day(6), _day(3), _day(4), _day(5)],
                "ocel:field": ["channel", "channel", "channel", "price", "price"],
                "channel": ["shop", "web", "by hand", None, None],
                "price": [None, None, None, 12.75, 41.0],
            }
        ),
    )
    return clean_oced_data(data, qualify_shared=qualify_shared)


BUILDERS = {
    "tiny": tiny,
    "pipeline": pipeline,
    "awkward": awkward,
    "awkward_shared": lambda: awkward(qualify_shared=False),
}

# The store the build configuration matrix and the shape table build from.
MATRIX_STORE = "awkward"


def build(name: str) -> OCEDStore:
    """Return a fresh copy of one of the three stores."""
    return copy.deepcopy(_cached(name))


_CACHE: dict[str, OCEDStore] = {}


def _cached(name: str) -> OCEDStore:
    """Build a store once and keep it, since building it is the slow part."""
    if name not in _CACHE:
        _CACHE[name] = BUILDERS[name]()
    return _CACHE[name]
