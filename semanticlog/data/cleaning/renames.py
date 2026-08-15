"""The record of what cleaning called each thing in the source.

Cleaning makes names safe at the cost of making them uglier, and the ontology
is annotated with the original wording. `RenameLog` holds it, and
`clean_oced_data` leaves it on the store as the backmap.
"""

from __future__ import annotations

from enum import Enum

from semanticlog.core import vocabulary as vocab


class RenameKind(str, Enum):
    """What sort of name a rename applies to.

    Attribute values are not here. They are keyed by the column they sit in
    rather than by a single name, so `RenameLog` holds them separately.
    """

    ACTIVITY = "activity"
    OBJECT_TYPE = "object_type"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    ID = "id"


class RenameLog:
    """Records what each cleaned name was called in the source.

    A name can be rewritten more than once, first by cleaning and then again
    to resolve a collision. The log always keeps the source name, not the
    intermediate one.

    Attribute values are held by the column they belong to rather than by the
    individual name they will have, because that name is `attribute_value` of
    the two and the column half can still change after the value is recorded.
    The individual names are composed in `flat`, once every rename has run.
    """

    def __init__(self) -> None:
        self._original: dict[RenameKind, dict[str, str]] = {k: {} for k in RenameKind}
        self._current: dict[RenameKind, dict[str, str]] = {k: {} for k in RenameKind}
        self._values: dict[tuple[str, str], str] = {}

    def record_value(self, column: str, value: str, original: str) -> None:
        """Record what one cleaned attribute value was called in the source.

        Args:
            column: Cleaned name of the column the value sits in.
            value: The value as it now reads.
            original: The value as the source wrote it.
        """
        self._values[(column, value)] = original

    def rename_value(self, column: str, old: str, new: str) -> None:
        """Record that a value was rewritten again, keeping its source name."""
        self._values[(column, new)] = self._values.pop((column, old), old)

    def rename_value_column(self, old: str, new: str) -> None:
        """Follow a renamed attribute, so its values still name the right thing."""
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

    def record(
        self, kind: RenameKind, old: str, new: str, original: str | None = None
    ) -> None:
        """Record that a name changed.

        Args:
            kind: What sort of name changed.
            old: The name before this change.
            new: The name after it.
            original: The source name to record, when it is neither `old` nor
                already known from an earlier rename of `old`.
        """
        root = original if original is not None else self._current[kind].get(old, old)
        self._current[kind].pop(old, None)
        self._original[kind].pop(old, None)
        if new != root:
            self._original[kind][new] = root
        self._current[kind][new] = root

    def flat(self) -> dict[str, str]:
        """Return every rename in one map, as current name to source name."""
        merged: dict[str, str] = {}
        for kind in RenameKind:
            merged.update(self._original[kind])
        merged.update(self.value_renames())
        return merged

    def by_kind(self) -> dict[str, list[str]]:
        """Return the renames grouped by kind, each as `source -> current`.

        Returns:
            Lists keyed by the kind's value, plus `value` for attribute
            values. A kind that renamed nothing is left out.
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
            grouped["value"] = values
        return grouped
