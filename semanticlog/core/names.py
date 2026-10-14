"""The registry of local names a build emits.

Classes, properties and individuals share one namespace. `NameRegistry`
records the owner of each name and returns a free name to an owner whose name
is taken. Claiming a name also claims every name the build derives from it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator

from semanticlog.core.vocabulary import is_safe_local_name

UPPER_SCHEMA = "the upper schema"


def candidates(name: str, prefix: str) -> Iterator[str]:
    """Yield the candidate names for one claimant, in order of preference.

    Yields:
        The bare name, then the prefixed name, then the prefixed name with a
        counter from 2 upward.
    """
    yield name
    yield prefix + name
    counter = 2
    while True:
        yield f"{prefix}{name}_{counter}"
        counter += 1


class NameRegistry:
    """The owner of each schema name a build will emit.

    Individuals are not registered. A pass over ids or values compares them
    against `names` in bulk.

    Args:
        reserved: Names no claimant may take.
        owner: The owner recorded for the reserved names.
    """

    def __init__(self, reserved: Iterable[str] = (), owner: str = UPPER_SCHEMA) -> None:
        self._owners: dict[str, str] = dict.fromkeys(reserved, owner)

    def __contains__(self, name: object) -> bool:
        return name in self._owners

    @property
    def names(self) -> frozenset[str]:
        """Every name claimed so far."""
        return frozenset(self._owners)

    def owner(self, name: str) -> str | None:
        """Return the owner of a name, or None if it is free."""
        return self._owners.get(name)

    def starting_with(self, prefix: str) -> frozenset[str]:
        """Return the claimed names that start with a prefix."""
        return frozenset(name for name in self._owners if name.startswith(prefix))

    def allocate(
        self,
        name: str,
        owner: str,
        *,
        prefix: str,
        pending: Collection[str] = (),
        derives: Callable[[str], Iterable[str]] | None = None,
        accepts: Callable[[str], bool] = is_safe_local_name,
    ) -> str:
        """Claim and return the first free candidate for `name`.

        Args:
            pending: Names of unallocated claimants of the same kind. A
                candidate in this set is skipped.
            derives: Returns the names a build derives from a candidate. They
                are claimed with it, and a candidate with a taken derived name
                is skipped.
            accepts: Predicate every candidate must pass.

        Raises:
            ValueError: If `prefix + name` does not pass `accepts`.
        """
        if not accepts(prefix + name):
            raise ValueError(
                f"Prefix {prefix!r} gives {prefix + name!r}, which the claimant "
                "cannot emit. Pass a prefix whose names pass `accepts`."
            )

        def wanted(candidate: str) -> set[str]:
            return {candidate, *(derives(candidate) if derives is not None else ())}

        def fits(candidate: str) -> bool:
            if not accepts(candidate):
                return False
            return all(
                claim not in self._owners and (claim == name or claim not in pending)
                for claim in wanted(candidate)
            )

        free = next(c for c in candidates(name, prefix) if fits(c))
        for claim in wanted(free):
            self._owners[claim] = owner
        return free
