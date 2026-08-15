"""What a build has already asserted as an individual.

An attribute reaches the build once per table carrying it, and the attribute
history reaches it again, so the same value, bin or moment arrives several
times and belongs in the ontology once. More than one emitter needs that
answer, and it is the memory a build spends per distinct value rather than per
event, so it is held here instead of in a set per emitter.
"""

from __future__ import annotations

from collections import defaultdict


class EmittedIndividuals:
    """The individuals a build has asserted so far, by what they stand for.

    Each method hands back the set for one kind, to be read for what is new
    and then updated with whatever was emitted.
    """

    def __init__(self) -> None:
        self._values: dict[str, set[str]] = defaultdict(set)
        self._bins: dict[str, set[str]] = defaultdict(set)
        self._moments: set[str] = set()

    def values_of(self, attribute: str) -> set[str]:
        """The value individuals asserted for one attribute.

        A value carries its literal when it is first asserted, so this is
        also what says whether the literal has been written.
        """
        return self._values[attribute]

    def bins_of(self, attribute: str) -> set[str]:
        """The value individuals of one attribute already placed in a bin."""
        return self._bins[attribute]

    @property
    def moments(self) -> set[str]:
        """The time individuals asserted, which every activity shares."""
        return self._moments

    def counts(self) -> dict[str, int]:
        """How many of each kind are recorded.

        What a build spends beyond its axioms, which is what makes the claim
        about streamed memory measurable.
        """
        return {
            "values": sum(len(names) for names in self._values.values()),
            "bins": sum(len(names) for names in self._bins.values()),
            "moments": len(self._moments),
        }
