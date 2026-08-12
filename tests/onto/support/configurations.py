"""The stores and settings the scenario tests build from.

One place to change what a scenario means, so the golden fixtures and
the build matrix stay in step.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.enrich.declare import DeclareResult
from semanticlog.onto.settings import (
    DataRepresentation,
    EventOrderRepresentation,
    InverseRepresentation,
    OCELExtension,
    OntoBuildSettings,
    OWLProfile,
)


def build_store() -> OCEDStore:
    data = OCEDStore(ocel_mode=True)
    data.add_object(
        "order",
        "oid",
        pd.DataFrame(
            {
                "oid": ["o1", "o2", "o3"],
                "channel": ["web", "shop", "web"],
                "price": [10.0, 25.5, 3.25],
            }
        ),
    )
    data.add_object(
        "item",
        "iid",
        pd.DataFrame({"iid": ["i1", "i2"], "channel": ["web", "shop"]}),
    )
    data.add_event(
        "place",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": ["e1", "e2", "e3"],
                "ts": [
                    datetime(2024, 1, 1, 9, 0),
                    datetime(2024, 1, 2, 9, 0),
                    datetime(2024, 1, 3, 9, 0),
                ],
                "clerk": ["ann", "bo", "ann"],
                "amount": [1, 2, 3],
            }
        ),
    )
    data.add_event(
        "ship",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": ["s1", "s2"],
                "ts": [datetime(2024, 1, 4, 9, 0), datetime(2024, 1, 5, 9, 0)],
                "clerk": ["cy", "ann"],
                "due": [datetime(2024, 2, 1), datetime(2024, 2, 2)],
            }
        ),
    )
    data.add_event_to_object_relation(
        "concerns",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame(
            {
                "eid": ["e1", "e2", "e3", "s1", "s2"],
                "oid": ["o1", "o1", "o2", "o1", "o2"],
            }
        ),
    )
    data.add_event_to_object_relation(
        "handles",
        "eid",
        "iid",
        ["item"],
        pd.DataFrame({"eid": ["e1", "s1", "s2"], "iid": ["i1", "i1", "i2"]}),
    )
    data.add_object_to_object_relation(
        "contains",
        "oid",
        ["order"],
        "iid",
        ["item"],
        pd.DataFrame({"oid": ["o1", "o2"], "iid": ["i1", "i2"]}),
    )
    data.add_object_changes(
        "order",
        pd.DataFrame(
            {
                "ocel:oid": ["o1", "o1"],
                "ocel:type": ["order", "order"],
                "ocel:timestamp": [datetime(2024, 1, 2, 9, 0), datetime(2024, 1, 6)],
                "ocel:field": ["channel", "channel"],
                "channel": ["shop", "web"],
            }
        ),
    )
    return clean_oced_data(data)


def out_of_bin_store() -> OCEDStore:
    rng = np.random.default_rng(0)
    values = [*rng.normal(0, 1, 60).tolist(), 1e12]
    n = len(values)
    data = OCEDStore()
    data.add_object(
        "order",
        "oid",
        pd.DataFrame({"oid": [f"o{i}" for i in range(n)], "price": values}),
    )
    data.add_event(
        "place",
        "eid",
        "ts",
        pd.DataFrame(
            {
                "eid": [f"e{i}" for i in range(n)],
                "ts": [datetime(2024, 1, 1) + timedelta(minutes=i) for i in range(n)],
            }
        ),
    )
    data.add_event_to_object_relation(
        "concerns",
        "eid",
        "oid",
        ["order"],
        pd.DataFrame(
            {"eid": [f"e{i}" for i in range(n)], "oid": [f"o{i}" for i in range(n)]}
        ),
    )
    return clean_oced_data(data)


def declare_result() -> DeclareResult:
    return DeclareResult(
        constraints=pd.DataFrame(
            {
                "trace_id": ["o1", "o2"],
                "case_concept": ["order", "order"],
                "Response__place__ship__": [True, False],
            }
        ),
        case_concepts=["order"],
    )


def ragged_declare_result() -> DeclareResult:
    return DeclareResult(
        constraints=pd.DataFrame(
            {
                "trace_id": ["o1", "o2", "i1", "i2"],
                "case_concept": ["order", "order", "item", "item"],
                "Response__place__ship__": [True, False, np.nan, np.nan],
                "Existence__ship__": [np.nan, np.nan, True, False],
            }
        ),
        case_concepts=["order", "item"],
    )


def settings(**overrides) -> OntoBuildSettings:
    base: dict = {
        "profile": OWLProfile.OWL2DL,
        "order_representations": [EventOrderRepresentation.DF],
        "case_concepts": ["order"],
    }
    base.update(overrides)
    return OntoBuildSettings(**base)


CONFIGURATIONS: dict[str, Callable[[], OntoBuildSettings]] = {
    "dl_df": lambda: settings(),
    "dl_df_ef": lambda: settings(
        order_representations=[EventOrderRepresentation.DF, EventOrderRepresentation.EF]
    ),
    "dl_ef_explicit": lambda: settings(
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.EF_EXPLICIT,
        ]
    ),
    "el_df": lambda: settings(profile=OWLProfile.OWL2EL),
    "el_ef_explicit": lambda: settings(
        profile=OWLProfile.OWL2EL,
        order_representations=[EventOrderRepresentation.EF_EXPLICIT],
    ),
    "dl_inverses": lambda: settings(
        inverse_representations=[
            InverseRepresentation.AXIOMATIC,
            InverseRepresentation.IMPLICIT,
        ],
        derive_annotations=True,
    ),
    "dl_bins": lambda: settings(
        data_representations=[DataRepresentation.VALUES, DataRepresentation.EQUAL_BINS],
        bins=2,
        bin_labels=["low", "high"],
    ),
    "dl_bins_only": lambda: settings(
        data_representations=[DataRepresentation.EQUAL_BINS],
        bins=2,
        bin_labels=["low", "high"],
    ),
    "dl_dist_bins": lambda: settings(
        data_representations=[DataRepresentation.VALUES, DataRepresentation.DIST_BINS],
    ),
    "el_inverses": lambda: settings(
        profile=OWLProfile.OWL2EL,
        inverse_representations=[InverseRepresentation.IMPLICIT],
    ),
    "dl_no_case_concept": lambda: settings(case_concepts=[], order_representations=[]),
    "dl_no_numeric": lambda: settings(data_representations=[]),
    "dl_history": lambda: settings(
        ocel_extensions=[OCELExtension.ATTRIBUTE_HISTORY],
        derive_annotations=True,
    ),
    "dl_declare_multi_cc": lambda: settings(
        case_concepts=["order", "item"],
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.DECLARE,
        ],
        declare_result=ragged_declare_result(),
    ),
    "dl_declare": lambda: settings(
        order_representations=[
            EventOrderRepresentation.DF,
            EventOrderRepresentation.DECLARE,
        ],
        declare_result=declare_result(),
    ),
}
