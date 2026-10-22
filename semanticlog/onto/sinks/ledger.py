"""What each emitted name is declared as, and which schema statements have been written.

`NameLedger` raises on a name declared as two entities. Its `first_*` methods
return False for a schema statement offered a second time, and the sink then
skips the write. A TBox `Term` and a log name with the same text are different
names.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import islice

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import SinkEntity
from semanticlog.core.vocabulary import NameCollisionError

_BY_KIND: dict[SinkEntity, str] = {
    SinkEntity.CLASS: "classes",
    SinkEntity.OBJECT_PROPERTY: "object_properties",
    SinkEntity.DATA_PROPERTY: "data_properties",
    SinkEntity.ANNOTATION_PROPERTY: "annotation_properties",
}

Key = tuple[bool, str]


def _key(name: str) -> Key:
    """Return the ledger key of a name: its namespace and its text."""
    return (vocab.is_tbox(name), str(name))


def _keys(names: Sequence[str]) -> tuple[Key, ...]:
    return tuple(_key(name) for name in names)


_PROPERTY_KINDS = frozenset(
    {
        SinkEntity.OBJECT_PROPERTY,
        SinkEntity.DATA_PROPERTY,
        SinkEntity.ANNOTATION_PROPERTY,
    }
)


class NameLedger:
    """Records what each emitted name is declared as.

    A name that cannot appear in an IRI, or that is declared as two different
    entities, raises with a message naming both sources.
    """

    def __init__(self) -> None:
        self._declared: dict[Key, tuple[SinkEntity, str, str]] = {}
        self._log_names: set[str] = set()
        self._written_schema: set[tuple[object, ...]] = set()
        self._named_inverses: dict[Key, str] = {
            _key(prop): inverse for prop, inverse in vocab.SCHEMA_NAMED_INVERSES.items()
        }

    def declare(self, name: str, kind: SinkEntity, source: str) -> None:
        """Declare a name as one entity.

        A name declared again as the same kind from the same source is
        accepted. A different source is a second entity.

        Args:
            name: Local name.
            kind: Entity kind.
            source: Origin of the name, quoted in an error.

        Raises:
            UnsafeLocalNameError: If the name cannot appear in an IRI, or a
                property name does not start with a letter or an underscore.
            NameCollisionError: If the name is declared as another kind or
                from another source.
        """
        self.check_name(name, source)
        if kind in _PROPERTY_KINDS and not (name[0].isalpha() or name[0] == "_"):
            raise vocab.UnsafeLocalNameError(
                f"{name!r} from {source} cannot be a property name. A property "
                "name starts with a letter or an underscore. Run "
                "clean_oced_data(store) before building, or rename it in the "
                "input."
            )
        key = _key(name)
        previous = self._declared.get(key)
        if previous is None:
            self._declared[key] = (kind, source, name)
            if not key[0]:
                self._log_names.add(key[1])
            return
        prev_kind, prev_source, _ = previous
        if prev_kind == kind and prev_source == source:
            return
        raise NameCollisionError(
            f"{name!r} names two entities:\n"
            f"  as {prev_kind.value} from {prev_source}\n"
            f"  as {kind.value} from {source}\n\n"
            "Rename one of them in the input data before building."
        )

    def check_name(self, name: str, source: str) -> None:
        """Raise if a name cannot be an IRI local name.

        Raises:
            UnsafeLocalNameError: If it cannot. The message names the source.
        """
        if not vocab.is_safe_local_name(name):
            raise vocab.UnsafeLocalNameError(
                f"{name!r} from {source} cannot be an IRI local name. Run "
                "clean_oced_data(store) before building, or clean the input "
                "that bypasses it, such as a DECLARE constraints CSV."
            )

    def accepts_individual(self, name: str) -> bool:
        """Whether a name is safe and not declared as a class or property.

        `check_individual` gives the error message.
        """
        return (
            bool(vocab.SAFE_LOCAL_NAME.fullmatch(name)) and name not in self._log_names
        )

    def check_individual(self, name: str, source: str) -> None:
        """Raise if a name cannot be emitted as an individual.

        Args:
            name: Individual name, as text.
            source: Origin of the name, quoted in the error.

        Raises:
            UnsafeLocalNameError: If the name cannot appear in an IRI.
            NameCollisionError: If the name is declared as a class or a
                property.
        """
        self.check_name(name, source)
        declared = self._declared.get((False, str(name)))
        if declared is None:
            return
        kind, declared_by, _ = declared
        raise NameCollisionError(
            f"{name!r} is emitted both as an individual and as a {kind.value}:\n"
            f"  as {kind.value} from {declared_by}\n"
            f"  as individual from {source}\n\n"
            "Rename one of them in the input data before building."
        )

    def check_individuals(self, names: Iterable[str], source: str) -> list[str]:
        """Check a batch of individual names and return the ones to emit.

        Missing values and duplicates are dropped. Only the first unsafe name
        and the first collision are reported. Write the returned list.

        Args:
            names: Candidate individual names.
            source: Origin of the names, quoted in an error.

        Returns:
            The distinct names, as text, in input order.

        Raises:
            UnsafeLocalNameError: If a name cannot appear in an IRI.
            NameCollisionError: If a name is declared as a class or a
                property.
        """
        series = (
            names
            if isinstance(names, pd.Series)
            else pd.Series(list(names), dtype="object")
        )
        series = series[series.notna()]
        # Converting a text column would copy every string in the log.
        if pd.api.types.infer_dtype(series, skipna=True) != "string":
            series = series.astype(str)
        series = series.drop_duplicates()
        if series.empty:
            return []

        unsafe = series[~series.str.fullmatch(vocab.SAFE_LOCAL_NAME)]
        if not unsafe.empty:
            self.check_name(unsafe.iloc[0], source)

        if self._log_names:
            clashing = series[series.isin(self._log_names)]
            if not clashing.empty:
                self.check_individual(str(clashing.iloc[0]), source)
        return series.tolist()

    def is_declared(self, name: str) -> bool:
        """Whether a class or property of this name has been declared."""
        return _key(name) in self._declared

    @property
    def count(self) -> int:
        """How many distinct classes and properties have been declared."""
        return len(self._declared)

    def by_kind(self) -> dict[str, int]:
        """How many names of each kind have been declared."""
        counts = dict.fromkeys(_BY_KIND.values(), 0)
        for kind, _, _ in self._declared.values():
            counts[_BY_KIND[kind]] += 1
        return counts

    def since(self, mark: int) -> dict[str, list[str]]:
        """Return the names declared after a mark, grouped by kind.

        Args:
            mark: A `count` read before the step being measured.

        Returns:
            Names keyed by the value of their `SinkEntity`, in declaration
            order.
        """
        names: dict[str, list[str]] = {}
        for kind, _, name in islice(self._declared.values(), mark, None):
            names.setdefault(kind.value, []).append(name)
        return names

    def first_declaration(self, name: str, kind: SinkEntity) -> bool:
        """Return True the first time this declaration is offered, False after."""
        return self._first(("declare", _key(name), kind))

    def first_sub_class(self, sub: str, parents: Sequence[str]) -> bool:
        """Return True the first time this subclass axiom is offered, False after."""
        return self._first(("sub_class", _key(sub), _keys(parents)))

    def first_sub_property(self, sub: str, parent: str, data_prop: bool) -> bool:
        """Return True the first time this subproperty axiom is offered, False after."""
        return self._first(("sub_property", _key(sub), _key(parent), data_prop))

    def first_domain(
        self, prop: str, expression: Sequence[str], data_prop: bool
    ) -> bool:
        """Return True the first time this domain is offered, False after."""
        return self._first(("domain", _key(prop), _keys(expression), data_prop))

    def first_range(
        self, prop: str, expression: Sequence[str], data_prop: bool
    ) -> bool:
        """Return True the first time this range is offered, False after."""
        return self._first(("range", _key(prop), _keys(expression), data_prop))

    def first_inverse(self, prop: str, inverse: str) -> bool:
        """Return True the first time this inverse axiom is offered, False after."""
        return self._first(("inverse", _key(prop), _key(inverse)))

    def first_transitive(self, prop: str) -> bool:
        """Return True the first time this transitivity axiom is offered, False after."""
        return self._first(("transitive", _key(prop)))

    def first_annotation(self, kind: str, name: str, text: str) -> bool:
        """Return True the first time this annotation is offered, False after."""
        return self._first(("annotate", kind, _key(name), text))

    def _first(self, statement: tuple[object, ...]) -> bool:
        """Record a schema statement and return True if it is new.

        Args:
            statement: Key of the statement. The first element is its sort.
        """
        if statement in self._written_schema:
            return False
        self._written_schema.add(statement)
        return True

    def inverse_name(self, prop: str) -> str:
        """Return the inverse name of a property, from the vocabulary or from this build."""
        named = self._named_inverses.get(_key(prop))
        return named if named is not None else vocab.inverse(prop)

    def claim_inverse(self, prop: str, inverse: str, source: str) -> None:
        """Record that two properties are each other's inverse.

        Raises:
            NameCollisionError: If the name is the inverse of another property.
        """
        claimed_by = self._named_inverses.get(_key(inverse))
        if claimed_by is not None and _key(claimed_by) != _key(prop):
            raise NameCollisionError(
                f"{inverse!r} is the inverse of {claimed_by!r} and cannot also "
                f"be the inverse of {prop!r}.\n"
                f"  from {source}\n\n"
                "Pass a distinct inverse_name, or leave it unset to get "
                f"{self.inverse_name(prop)!r}."
            )
        self._named_inverses[_key(prop)] = inverse
        self._named_inverses[_key(inverse)] = prop
