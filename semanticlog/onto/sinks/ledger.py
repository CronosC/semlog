"""Who claimed which name, and which schema statements have been made.

Two things have to be tracked across a whole build regardless of where the
axioms end up. One name may only ever mean one thing, and a schema statement
is worth making once. Both are bookkeeping over names rather than over axioms,
so they sit beside `AxiomSink` rather than inside it, and a backend never sees
them.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import islice

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.vocabulary import NameCollisionError

_BY_KIND = {
    "class": "classes",
    "object": "object_properties",
    "data": "data_properties",
}


class NameLedger:
    """Records what each emitted name was claimed for, and refuses a second claim.

    A name that is unsafe in an IRI, or that would be used for two different
    things, is refused here with a message naming both sources.
    """

    def __init__(self) -> None:
        self._declared: dict[str, tuple[str, str]] = {}
        self._written_schema: set[tuple] = set()
        self._named_inverses: dict[str, str] = dict(vocab.SCHEMA_NAMED_INVERSES)

    def declare(self, name: str, kind: str, source: str) -> None:
        """Claim a name for one thing.

        Args:
            name: Local name being claimed.
            kind: One of `class`, `object` or `data`.
            source: Where the claim comes from, quoted in any error.

        Raises:
            UnsafeLocalNameError: If the name cannot appear in an IRI.
            NameCollisionError: If something else already claimed it.
        """
        self.check_name(name, source)
        previous = self._declared.get(name)
        if previous is None:
            self._declared[name] = (kind, source)
            return
        prev_kind, prev_source = previous
        if prev_kind == kind and prev_source == source:
            return
        raise NameCollisionError(
            f"{name!r} would be emitted for more than one thing, which silently "
            f"merges them in the ontology:\n"
            f"  as {prev_kind} from {prev_source}\n"
            f"  as {kind} from {source}\n\n"
            "Rename the colliding source in the input data before building."
        )

    def check_name(self, name: str, source: str) -> None:
        """Refuse a name that cannot be an IRI local name.

        Raises:
            UnsafeLocalNameError: Naming where the name came from.
        """
        if not vocab.is_safe_local_name(name):
            raise vocab.UnsafeLocalNameError(
                f"{name!r} from {source} cannot be an IRI local name. Names come "
                "from the input data, so run clean_oced_data(store) before "
                "building, or check any input that bypasses it such as a "
                "DECLARE constraints CSV."
            )

    def accepts_individual(self, name: str) -> bool:
        """Whether a name can be emitted as an individual as it stands.

        This is the hot path of a build, since a relation names two of them
        per row, so it stays as cheap as a regex match and a dict lookup and
        leaves the message about a refusal to `check_individual`.
        """
        return (
            bool(vocab.SAFE_LOCAL_NAME.fullmatch(name)) and name not in self._declared
        )

    def check_individual(self, name: str, source: str) -> None:
        """Refuse one individual name that cannot be emitted as it stands.

        Args:
            name: Candidate individual name, already text.
            source: Where it comes from, quoted in any error.

        Raises:
            UnsafeLocalNameError: If the name cannot appear in an IRI.
            NameCollisionError: If the name is also declared as a class or a
                property, which would make one IRI a member of itself.
        """
        self.check_name(name, source)
        declared = self._declared.get(name)
        if declared is None:
            return
        kind, declared_by = declared
        raise NameCollisionError(
            f"{name!r} is emitted both as an individual and as a {kind}, "
            "which makes one IRI a member of itself:\n"
            f"  as {kind} from {declared_by}\n"
            f"  as individual from {source}\n\n"
            "Rename the colliding source in the input data before building."
        )

    def check_individuals(self, names: Iterable[str], source: str) -> list[str]:
        """Check a batch of individual names, and return the ones to emit.

        Individuals are checked in bulk because there are many of them, and
        only the first offender of each sort is reported. Missing values are
        dropped rather than checked, so a caller writes what this returns and
        not the names it passed in.

        Args:
            names: Candidate individual names.
            source: Where they come from, quoted in any error.

        Returns:
            The distinct names, as text, in the order they arrived.

        Raises:
            UnsafeLocalNameError: If a name cannot appear in an IRI.
            NameCollisionError: If a name is also declared as a class or a
                property, which would make one IRI a member of itself.
        """
        series = (
            names
            if isinstance(names, pd.Series)
            else pd.Series(list(names), dtype="object")
        )
        series = series[series.notna()]
        # A column of ids is already text, and converting it again would copy
        # every string in the log.
        if pd.api.types.infer_dtype(series, skipna=True) != "string":
            series = series.astype(str)
        series = series.drop_duplicates()
        if series.empty:
            return []

        unsafe = series[~series.str.fullmatch(vocab.SAFE_LOCAL_NAME)]
        if not unsafe.empty:
            self.check_name(unsafe.iloc[0], source)

        if self._declared:
            clashing = series[series.isin(self._declared)]
            if not clashing.empty:
                self.check_individual(str(clashing.iloc[0]), source)
        return series.tolist()

    def is_declared(self, name: str) -> bool:
        """Whether a class or property of this name has been declared."""
        return name in self._declared

    @property
    def count(self) -> int:
        """How many distinct classes and properties have been declared."""
        return len(self._declared)

    def by_kind(self) -> dict[str, int]:
        """How many classes, object properties and data properties there are."""
        counts = {"classes": 0, "object_properties": 0, "data_properties": 0}
        for kind, _ in self._declared.values():
            counts[_BY_KIND[kind]] += 1
        return counts

    def since(self, mark: int) -> dict[str, list[str]]:
        """Return the names declared after a mark, grouped by kind.

        Args:
            mark: A `count` read before the step being measured.

        Returns:
            Names by kind, in the order they were declared.
        """
        names: dict[str, list[str]] = {}
        for name, (kind, _) in islice(self._declared.items(), mark, None):
            names.setdefault(kind, []).append(name)
        return names

    def first_declaration(self, name: str, kind: str) -> bool:
        """Report whether this entity has yet to be declared, and record it."""
        return self._first(("declare", name, kind))

    def first_sub_class(self, sub: str, parents: Sequence[str]) -> bool:
        """Report whether this class has yet to be placed under these parents."""
        return self._first(("sub_class", sub, tuple(parents)))

    def first_sub_property(self, sub: str, parent: str, data_prop: bool) -> bool:
        """Report whether this property has yet to be placed under that one."""
        return self._first(("sub_property", sub, parent, data_prop))

    def first_domain(
        self, prop: str, expression: Sequence[str], data_prop: bool
    ) -> bool:
        """Report whether this domain has yet to be stated for the property."""
        return self._first(("domain", prop, tuple(expression), data_prop))

    def first_range(
        self, prop: str, expression: Sequence[str], data_prop: bool
    ) -> bool:
        """Report whether this range has yet to be stated for the property."""
        return self._first(("range", prop, tuple(expression), data_prop))

    def first_inverse(self, prop: str, inverse: str) -> bool:
        """Report whether these two have yet to be stated as inverses."""
        return self._first(("inverse", prop, inverse))

    def first_transitive(self, prop: str) -> bool:
        """Report whether this property has yet to be declared transitive."""
        return self._first(("transitive", prop))

    def first_annotation(self, kind: str, name: str, text: str) -> bool:
        """Report whether this entity has yet to carry this label or comment."""
        return self._first(("annotate", kind, name, text))

    def _first(self, statement: tuple) -> bool:
        """Report whether a schema statement is new, and record it.

        Args:
            statement: What identifies it. The first element says which sort
                it is, so two sorts cannot collide on one key. Every caller is
                a method above, so nothing outside builds one of these.
        """
        if statement in self._written_schema:
            return False
        self._written_schema.add(statement)
        return True

    def inverse_name(self, prop: str) -> str:
        """Return the inverse of a property, including inverses named in this build."""
        return vocab.inverse(prop, self._named_inverses)

    def claim_inverse(self, prop: str, inverse: str, source: str) -> None:
        """Record that two properties are each other's inverse.

        Raises:
            NameCollisionError: If the name is already the inverse of another
                property, since recording the second would drop the first.
        """
        claimed_by = self._named_inverses.get(inverse)
        if claimed_by is not None and claimed_by != prop:
            raise NameCollisionError(
                f"{inverse!r} is already the inverse of {claimed_by!r} and "
                f"cannot also be the inverse of {prop!r}: a property has one "
                "inverse, and recording the second would silently drop the first.\n"
                f"  from {source}\n\n"
                "Pass a distinct inverse_name, or leave it unset to get "
                f"{self.inverse_name(prop)!r}."
            )
        self._named_inverses[prop] = inverse
        self._named_inverses[inverse] = prop
