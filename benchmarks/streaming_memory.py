"""How much memory a streamed build holds. Run by hand, not collected."""

from __future__ import annotations

import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from semanticlog.core.enums import EventOrderRepresentation
from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.data.cleaning import clean_oced_data
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import StreamingSink

ACTS = 5


def rss() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def make_store(n: int) -> OCEDStore:
    per = n // ACTS
    names = ["ann", "bo", "cy"]
    store = OCEDStore()
    for a in range(ACTS):
        store.add_event(
            f"act{a}",
            "eid",
            "ts",
            pd.DataFrame(
                {
                    "eid": [f"e{a}_{i}" for i in range(per)],
                    "ts": pd.to_datetime(np.arange(per), unit="s", utc=True),
                    "clerk": [names[i % 3] for i in range(per)],
                }
            ),
        )
    nobj = max(1, n // 10)
    store.add_object(
        "order",
        "oid",
        pd.DataFrame(
            {
                "oid": [f"o{i}" for i in range(nobj)],
                "channel": [names[i % 3] for i in range(nobj)],
            }
        ),
    )
    for a in range(ACTS):
        store.add_event_to_object_relation(
            f"rel{a}",
            "eid",
            "oid",
            ["order"],
            pd.DataFrame(
                {
                    "eid": [f"e{a}_{i}" for i in range(per)],
                    "oid": [f"o{i % nobj}" for i in range(per)],
                }
            ),
        )
    return clean_oced_data(store)


def main(n: int, path: Path) -> None:
    store = make_store(n)
    after_store = rss()

    settings = OntoBuildSettings(
        case_concepts=["order"], order_representations=[EventOrderRepresentation.DF]
    )
    sink = StreamingSink(str(path), DEFAULT_IRI, settings)
    start = time.perf_counter()
    OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink).build_ontology(store)
    elapsed = time.perf_counter() - start
    peak = rss()

    with path.open(encoding="utf-8") as handle:
        axioms = sum(1 for _ in handle)

    print(f"events           : {n:,}")
    print(f"axioms           : {axioms:,}")
    print(f"build time       : {elapsed:.1f}s")
    print(f"peak over store  : +{peak - after_store:.0f} MB")
    print(f"ledger entries   : {sink.declared_count:,}")


if __name__ == "__main__":
    events = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    if len(sys.argv) > 2:
        main(events, Path(sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            main(events, Path(tmp) / "bench.ofn")
