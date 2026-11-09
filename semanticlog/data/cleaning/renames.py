"""The record of the source name behind each cleaned name.

`RenameLog` holds it, and `clean_oced_data` sets it on the store as the
backmap, one map per kind of name. The ontology is annotated with the source
names.
"""

from __future__ import annotations

from enum import Enum

from semanticlog.core import vocabulary as vocab

VALUE_KIND = "value"


class RenameKind(str, Enum):
    """What sort of name a rename applies to.

    Attribute values are no kind. `RenameLog` keys them by column, and the
    backmap holds them under `VALUE_KIND`.
    """

    ACTIVITY = "activity"
    OBJECT_TYPE = "object_type"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    ID = "id"

    @property
    def word(self) -> str:
        """The word a message uses for one name of this kind."""
        return WORDS[self.value][0]


WORDS: dict[str, tuple[str, str]] = {
    RenameKind.ACTIVITY.value: ("activity", "activities"),
    RenameKind.OBJECT_TYPE.value: ("object type", "object types"),
    RenameKind.RELATION.value: ("relation", "relations"),
    RenameKind.ATTRIBUTE.value: ("attribute", "attributes"),
    RenameKind.ID.value: ("id", "ids"),
    VALUE_KIND: ("value", "values"),
}


class RenameLog:
    """Records the source name of each cleaned name, through every later rename."""

    def __init__(self) -> None:
        self._original: dict[RenameKind, dict[str, str]] = {k: {} for k in RenameKind}
        self._current: dict[RenameKind, dict[str, str]] = {k: {} for k in RenameKind}
        self._values: dict[tuple[str, str], str] = {}

    def record_value(self, column: str, value: str, original: str) -> None:
        """Record the source form of one cleaned attribute value."""
        self._values[(column, value)] = original

    def rename_value(self, column: str, old: str, new: str) -> None:
        """Record that a value was rewritten again, keeping its source form."""
        self._values[(column, new)] = self._values.pop((column, old), old)

    def rename_value_column(self, old: str, new: str) -> None:
        """Move the values of a renamed column under its new name."""
        moved = {
            (new, value): original
            for (column, value), original in self._values.items()
            if column == old
        }
        if not moved:
            return
        self._values = {
            key: original for key, original in self._values.items() if key[0] != old
        }
        self._values.update(moved)

    def value_renames(self) -> dict[str, str]:
        """Return the value renames, as individual name to source value."""
        renamed = {}
        for (column, value), original in self._values.items():
            if value != original:
                renamed[vocab.attribute_value(column, value)] = original
        return renamed

    def record(self, kind: RenameKind, old: str, new: str) -> None:
        """Record that a name changed, keeping the source name of `old`."""
        root = self._current[kind].get(old, old)
        self._current[kind].pop(old, None)
        self._original[kind].pop(old, None)
        if new != root:
            self._original[kind][new] = root
        self._current[kind][new] = root

    def backmap(self) -> dict[str, dict[str, str]]:
        """Return the renames as current name to source name, keyed by kind value and `value`.

        A value is keyed by its individual name. A kind that renamed nothing
        is left out.
        """
        maps = {kind.value: dict(self._original[kind]) for kind in RenameKind}
        maps[VALUE_KIND] = self.value_renames()
        return {kind: names for kind, names in maps.items() if names}

    def by_kind(self) -> dict[str, list[str]]:
        """Return the renames as `source -> current`, keyed by kind value and `value`.

        A kind that renamed nothing is left out.
        """
        grouped: dict[str, list[str]] = {}
        for kind in RenameKind:
            pairs = [
                f"{source} -> {name}" for name, source in self._original[kind].items()
            ]
            if pairs:
                grouped[kind.value] = pairs
        values = [
            f"{source} -> {name}" for name, source in self.value_renames().items()
        ]
        if values:
            grouped[VALUE_KIND] = values
        return grouped
