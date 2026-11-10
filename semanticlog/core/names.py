"""The registry of the schema names a build emits.

Classes and properties share one namespace. `NameRegistry` records each
claimed name and returns a free name to a claimant whose name is taken.
Claiming a name also claims every name the build derives from it.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Iterator

from semanticlog.core.vocabulary import is_safe_local_name


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
    """The schema names a build will emit.

    Individuals are not registered. Each kind has a path of its own.
    """

    def __init__(self) -> None:
        self._claimed: set[str] = set()

    def allocate(
        self,
        name: str,
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
                claim not in self._claimed and (claim == name or claim not in pending)
                for claim in wanted(candidate)
            )

        free = next(c for c in candidates(name, prefix) if fits(c))
        self._claimed.update(wanted(free))
        return free
