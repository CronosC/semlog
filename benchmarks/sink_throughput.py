"""How fast `InMemorySink` accepts individuals and relations."""

from __future__ import annotations

import sys
import time

import pandas as pd

from semanticlog.core.vocabulary import DEFAULT_IRI
from semanticlog.onto.emitters.pairs import clean_pairs
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import InMemorySink

DEFAULT_INDIVIDUALS = 100_000
DEFAULT_RELATIONS = 50_000


def _sink() -> InMemorySink:
    return InMemorySink(DEFAULT_IRI, OntoBuildSettings().profile)


def individuals(n: int = DEFAULT_INDIVIDUALS) -> None:
    sink = _sink()
    sink.add_class("Thing", source="benchmark")

    start = time.perf_counter()
    sink.assert_individuals("Thing", [f"i{k}" for k in range(n)])
    elapsed = time.perf_counter() - start

    count = sum(1 for _ in sink.onto.individuals_in_signature())
    print(f"assert_individuals: {n:,} in {elapsed:.2f}s ({count:,} in signature)")


def relations(n: int = DEFAULT_RELATIONS) -> None:
    sink = _sink()
    sink.add_class("Thing", source="benchmark")
    sink.add_property("rel", "Thing", "Thing", source="benchmark")
    sink.assert_individuals("Thing", [f"i{k}" for k in range(n + 1)])
    pairs = pd.DataFrame(
        {"a": [f"i{k}" for k in range(n)], "b": [f"i{k + 1}" for k in range(n)]}
    )

    start = time.perf_counter()
    sink.assert_relations("rel", clean_pairs(pairs, "a", "b"))
    elapsed = time.perf_counter() - start

    print(f"assert_relations:   {n:,} in {elapsed:.2f}s")


if __name__ == "__main__":
    scale = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    individuals(scale or DEFAULT_INDIVIDUALS)
    relations(scale or DEFAULT_RELATIONS)
