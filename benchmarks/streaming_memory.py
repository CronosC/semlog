"""Memory held by a streamed build. Linux only, it reads /proc/self/statm.

Builds two logs of one size that differ in the number of distinct values. A
streamed build holds the store plus one entry per distinct value, timestamp
and name.
"""

from __future__ import annotations

import argparse
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
from semanticlog.data.tables import EntityKind, RelationKind
from semanticlog.onto.builder import OCEDOntoBuilder
from semanticlog.onto.inputs import BuildInputs
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import StreamingSink

ACTS = 5
NAMES = ["ann", "bo", "cy"]
DEFAULT_EVENTS = 200_000


def rss() -> float:
    """Return the current resident memory in MB."""
    # ru_maxrss would carry the first case's peak into the second.
    with open("/proc/self/statm", encoding="utf-8") as handle:
        pages = int(handle.read().split()[1])
    return pages * resource.getpagesize() / 1024 / 1024


def make_store(n: int, distinct: bool) -> OCEDStore:
    """Build a log of n events across five activities.

    Args:
        n: How many events.
        distinct: Give every event its own clerk and timestamp. Otherwise
            the log has three clerks and one shared range of timestamps.
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
        store.add(
            EntityKind.EVENT,
            f"act{a}",
            pd.DataFrame(
                {
                    "eid": [f"e{a}_{i}" for i in range(per)],
                    "ts": pd.to_datetime(stamps, unit="s", utc=True),
                    "clerk": clerks,
                }
            ),
            id_col="eid",
            time_col="ts",
        )
    nobj = max(1, n // 10)
    store.add(
        EntityKind.OBJECT,
        "order",
        pd.DataFrame(
            {
                "oid": [f"o{i}" for i in range(nobj)],
                "channel": [NAMES[i % 3] for i in range(nobj)],
            }
        ),
        id_col="oid",
    )
    for a in range(ACTS):
        store.add_relation(
            RelationKind.E2O,
            f"rel{a}",
            pd.DataFrame(
                {
                    "eid": [f"e{a}_{i}" for i in range(per)],
                    "oid": [f"o{i % nobj}" for i in range(per)],
                }
            ),
            from_id="eid",
            to_id="oid",
            to_type=["order"],
        )
    return clean_oced_data(store)


def measure(n: int, distinct: bool, path: Path) -> None:
    """Build one log and print the memory held beyond the store."""
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
    after_build = rss()

    kept = {"names": sink.declared_count, **builder.emitted.counts()}
    print(f"  axioms           : {sink.axiom_count:,}")
    print(f"  build time       : {elapsed:.1f}s")
    print(f"  store            : {after_store:.0f} MB")
    print(f"  held over store  : +{after_build - after_store:.0f} MB")
    print("  kept             : " + ", ".join(f"{k}={v:,}" for k, v in kept.items()))


def run(n: int, directory: Path) -> None:
    """Measure a low and a high cardinality log of n events, writing into `directory`."""
    print(f"events           : {n:,}")
    for distinct in (False, True):
        mode = "high" if distinct else "low"
        print(f"{mode} cardinality")
        measure(n, distinct, directory / f"{mode}.ofn")


def main(argv: list[str] | None = None) -> int:
    """Run the measurement and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "events",
        nargs="?",
        type=int,
        default=DEFAULT_EVENTS,
        help=f"events per log (default: {DEFAULT_EVENTS:,})",
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="where to keep the written files (default: a temporary directory)",
    )
    args = parser.parse_args(argv)
    if args.directory is not None:
        run(args.events, args.directory)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            run(args.events, Path(tmp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
