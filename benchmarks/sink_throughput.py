"""How fast `InMemorySink` accepts individuals and relations."""

from __future__ import annotations

import argparse
import sys
import time

import pandas as pd

from semlog.core.enums import IndividualKind
from semlog.core.vocabulary import DEFAULT_IRI
from semlog.onto.emitters.pairs import clean_pairs
from semlog.onto.settings import OntoBuildSettings
from semlog.onto.sinks import InMemorySink

DEFAULT_INDIVIDUALS = 100_000
DEFAULT_RELATIONS = 50_000


def _sink() -> InMemorySink:
    """Return an empty sink under the default profile."""
    return InMemorySink(DEFAULT_IRI, OntoBuildSettings().profile)


def individuals(n: int = DEFAULT_INDIVIDUALS) -> None:
    """Assert n individuals of one class and print the time it took."""
    sink = _sink()
    sink.add_class("Thing", source="benchmark")

    start = time.perf_counter()
    sink.assert_individuals("Thing", [f"i{k}" for k in range(n)], IndividualKind.OBJECT)
    elapsed = time.perf_counter() - start

    count = sum(1 for _ in sink.onto.individuals_in_signature())
    print(f"assert_individuals: {n:,} in {elapsed:.2f}s ({count:,} in signature)")


def relations(n: int = DEFAULT_RELATIONS) -> None:
    """Assert n relations between n + 1 individuals and print the time it took."""
    sink = _sink()
    sink.add_class("Thing", source="benchmark")
    sink.add_property("rel", "Thing", "Thing", source="benchmark")
    sink.assert_individuals(
        "Thing", [f"i{k}" for k in range(n + 1)], IndividualKind.OBJECT
    )
    pairs = pd.DataFrame(
        {"a": [f"i{k}" for k in range(n)], "b": [f"i{k + 1}" for k in range(n)]}
    )

    start = time.perf_counter()
    sink.assert_relations(
        "rel",
        clean_pairs(pairs, "a", "b"),
        (IndividualKind.OBJECT, IndividualKind.OBJECT),
    )
    elapsed = time.perf_counter() - start

    print(f"assert_relations:   {n:,} in {elapsed:.2f}s")


def main(argv: list[str] | None = None) -> int:
    """Run both measurements and return the exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "n",
        nargs="?",
        type=int,
        help=f"individuals and relations to assert (default: {DEFAULT_INDIVIDUALS:,} and {DEFAULT_RELATIONS:,})",
    )
    args = parser.parse_args(argv)
    individuals(args.n or DEFAULT_INDIVIDUALS)
    relations(args.n or DEFAULT_RELATIONS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
