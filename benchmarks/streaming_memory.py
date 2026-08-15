"""How much memory a streamed build holds. Run by hand, not collected.

Two logs of the same size, differing only in how many distinct values they
carry. A streamed build writes its axioms straight out, so what it holds is
the store plus one entry per distinct value, moment and name. The two cases
are what shows that.
"""

from __future__ import annotations

import gc
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
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import StreamingSink

ACTS = 5
NAMES = ["ann", "bo", "cy"]


def rss() -> float:
    """Resident memory right now, in MB.

    What is held rather than the peak, since the peak of this process never
    goes down again and the second case would inherit the first one's.
    """
    with open("/proc/self/statm", encoding="utf-8") as handle:
        pages = int(handle.read().split()[1])
    return pages * resource.getpagesize() / 1024 / 1024


def make_store(n: int, distinct: bool) -> OCEDStore:
    """Build a log of n events across five activities.

    Args:
        n: How many events.
        distinct: Give every event its own clerk and its own moment. That is
            the case a build grows with. Without it the log holds three
            clerks and one shared range of timestamps, which is what a log
            with a handful of resources looks like.
    """
    per = n // ACTS
    store = OCEDStore()
    for a in range(ACTS):
        stamps = np.arange(per) + (a * per if distinct else 0)
        clerks = (
            [f"clerk{a}_{i}" for i in range(per)]
            if distinct
            else [NAMES[i % 3] for i in range(per)]
        )
        store.add_event(
            f"act{a}",
            "eid",
            "ts",
            pd.DataFrame(
                {
                    "eid": [f"e{a}_{i}" for i in range(per)],
                    "ts": pd.to_datetime(stamps, unit="s", utc=True),
                    "clerk": clerks,
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
                "channel": [NAMES[i % 3] for i in range(nobj)],
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


def measure(n: int, distinct: bool, path: Path) -> None:
    """Build one log, and report what the build cost beyond the store."""
    gc.collect()
    store = make_store(n, distinct)
    gc.collect()
    after_store = rss()

    settings = OntoBuildSettings(order_representations=[EventOrderRepresentation.DF])
    inputs = BuildInputs(case_concepts=["order"])
    sink = StreamingSink(str(path), DEFAULT_IRI, settings.profile)
    builder = OCEDOntoBuilder(DEFAULT_IRI, settings, sink=sink)
    start = time.perf_counter()
    builder.build_ontology(store, inputs)
    elapsed = time.perf_counter() - start
    peak = rss()

    with path.open(encoding="utf-8") as handle:
        axioms = sum(1 for _ in handle)

    kept = {"names": sink.declared_count, **builder.emitted.counts()}
    print(f"  axioms           : {axioms:,}")
    print(f"  build time       : {elapsed:.1f}s")
    print(f"  store            : {after_store:.0f} MB")
    print(f"  held over store  : +{peak - after_store:.0f} MB")
    print("  kept             : " + ", ".join(f"{k}={v:,}" for k, v in kept.items()))


def main(n: int, directory: Path) -> None:
    print(f"events           : {n:,}")
    for distinct in (False, True):
        print("high cardinality" if distinct else "low cardinality")
        measure(n, distinct, directory / f"bench-{distinct}.ofn")


if __name__ == "__main__":
    events = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
    if len(sys.argv) > 2:
        main(events, Path(sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory() as tmp:
            main(events, Path(tmp))
