"""The individuals a build has asserted.

An attribute appears once per table that carries it and again in the
attribute history, so the same value, bin or timestamp arrives several times.
`EmittedIndividuals` records each one so it is asserted once. Its size grows
with the number of distinct values.
"""

from __future__ import annotations

from collections import defaultdict


class EmittedIndividuals:
    """The individuals a build has asserted, grouped by kind.

    Each method returns the set for one kind. A caller reads it to find new
    names and adds the ones it emits.
    """

    def __init__(self) -> None:
        self._values: dict[str, set[str]] = defaultdict(set)
        self._bins: dict[str, set[str]] = defaultdict(set)
        self._timestamps: set[str] = set()

    def values_of(self, attribute: str) -> set[str]:
        """Return the value individuals asserted for one attribute."""
        return self._values[attribute]

    def bins_of(self, attribute: str) -> set[str]:
        """Return the value individuals of one attribute placed in a bin."""
        return self._bins[attribute]

    @property
    def timestamps(self) -> set[str]:
        """The time individuals asserted, for every activity and the attribute history."""
        return self._timestamps

    def counts(self) -> dict[str, int]:
        """Return how many individuals of each kind are recorded."""
        return {
            "values": sum(len(names) for names in self._values.values()),
            "bins": sum(len(names) for names in self._bins.values()),
            "timestamps": len(self._timestamps),
        }
