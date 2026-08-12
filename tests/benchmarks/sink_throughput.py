"""How fast each sink accepts axioms. Run by hand, not collected."""

from __future__ import annotations

import time

import pandas as pd

from semanticlog.onto.settings import DEFAULT_IRI, OntoBuildSettings
from semanticlog.onto.sinks import InMemorySink


def _sink() -> InMemorySink:
    return InMemorySink(DEFAULT_IRI, OntoBuildSettings(case_concepts=["order"]))


def individuals(n: int = 100_000) -> None:
    sink = _sink()
    sink.add_class("Thing", source="benchmark")

    start = time.perf_counter()
    sink.assert_individuals([f"i{k}" for k in range(n)], "Thing")
    elapsed = time.perf_counter() - start

    count = sum(1 for _ in sink.onto.individuals_in_signature())
    print(f"assert_individuals: {n:,} in {elapsed:.2f}s ({count:,} in signature)")


def relations(n: int = 50_000) -> None:
    sink = _sink()
    sink.add_class("Thing", source="benchmark")
    sink.add_property("rel", "Thing", "Thing", source="benchmark")
    sink.assert_individuals([f"i{k}" for k in range(n + 1)], "Thing")
    pairs = pd.DataFrame(
        {"a": [f"i{k}" for k in range(n)], "b": [f"i{k + 1}" for k in range(n)]}
    )

    start = time.perf_counter()
    sink.assert_relations(pairs, "rel", "a", "b")
    elapsed = time.perf_counter() - start

    print(f"assert_relations:   {n:,} in {elapsed:.2f}s")


if __name__ == "__main__":
    individuals()
    relations()
