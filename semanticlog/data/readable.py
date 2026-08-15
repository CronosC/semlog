"""The read side of a store, as everything above `data` sees it.

`OCEDStore` both holds the tables and rewrites them, and rewriting them is
`data`'s business: loading fills the store and cleaning renames what it holds.
Everything above only reads. Saying so in the type is what keeps it that way,
since a build annotated with `ReadableStore` cannot reach a method that
changes the log it is building from.

`OCEDStore` satisfies this by having the methods, so nothing has to declare
it. Where the two drift apart, every call that hands a store to a build stops
type checking.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Protocol

import pandas as pd

from semanticlog.data.tables import (
    EntityKind,
    EventTable,
    ObjectTable,
    RelationKind,
    RelationTable,
    TableView,
)


class ReadableStore(Protocol):
    """Everything a build, an enrichment or a check asks of a store."""

    @property
    def backmap(self) -> dict[str, str]:
        """Cleaned name to original name, for everything cleaning renamed."""

    @property
    def ocel_mode(self) -> bool:
        """Whether this store may hold attribute change history."""

    def get_original(self, iri_or_local_name: str, base_iri: str = ...) -> str | None:
        """Look up what a name in the store was called in the source log."""

    def tables(self, *, changes: bool = True) -> Iterator[TableView]:
        """Yield every table in the store, with its structural columns named."""

    def get_event_types(self) -> list[str]:
        """Return every activity name in the store."""

    def get_event_table(self, event_type: str) -> EventTable:
        """Return the event table of one activity."""

    def get_object_types(self) -> list[str]:
        """Return every object type name in the store."""

    def get_object_table(self, obj_type: str) -> ObjectTable:
        """Return the object table of one type."""

    def get_object_tables(self) -> Mapping[str, ObjectTable]:
        """Return every object table, keyed by type and read-only."""

    def get_most_common_objects(self, n: int = 1) -> list[str]:
        """Return the object types with the most instances."""

    def get_e2o(self) -> Mapping[str, RelationTable]:
        """Return the event-to-object relations, keyed by name and read-only."""

    def get_o2o(self) -> Mapping[str, RelationTable]:
        """Return the object-to-object relations, keyed by name and read-only."""

    def get_relation(self, relation: str) -> RelationTable:
        """Return one relation by name, from either family."""

    def get_relation_types(self, kind: RelationKind = RelationKind.ALL) -> list[str]:
        """Return relation names, optionally of one family only."""

    def get_ids(self, kind: EntityKind, entity_type: str) -> Iterable:
        """Return the distinct ids of one activity or object type."""

    def get_object_changes(self, obj_type: str) -> pd.DataFrame:
        """Return the recorded attribute changes of one object type."""

    def get_object_change_types(self) -> list[str]:
        """Return the object types that have recorded changes."""
