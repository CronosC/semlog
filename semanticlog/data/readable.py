"""The read-only view of a store used by the layers above `data`.

A build, an enrichment or a check takes a `ReadableStore`. `OCEDStore`
satisfies the protocol structurally.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Protocol

import pandas as pd

from semanticlog.data.tables import (
    EntityKind,
    EntityTable,
    RelationKind,
    RelationTable,
    TableView,
)


class ReadableStore(Protocol):
    """The read methods a build, an enrichment or a check calls on a store."""

    @property
    def backmap(self) -> Mapping[str, str]:
        """Cleaned name to original name, for everything cleaning renamed."""

    @property
    def ocel_mode(self) -> bool:
        """Whether this store may hold attribute change history."""

    def get_original(self, iri_or_local_name: str, base_iri: str = ...) -> str | None:
        """Look up what a name in the store was called in the source log."""

    def tables(self, *, changes: bool = True) -> Iterator[TableView]:
        """Yield every table in the store, with its structural columns named."""

    def types(self, kind: EntityKind) -> list[str]:
        """Return every activity or object type name in the store."""

    def table(self, kind: EntityKind, name: str) -> EntityTable:
        """Return the table of one activity or object type."""

    def tables_of(self, kind: EntityKind) -> Mapping[str, EntityTable]:
        """Return the tables of one kind, keyed by type and read-only."""

    def ids(self, kind: EntityKind, name: str) -> Iterable[object]:
        """Return the distinct ids of one activity or object type."""

    def get_most_common_objects(self, n: int = 1) -> list[str]:
        """Return the object types with the most instances."""

    def relations_of(self, kind: RelationKind) -> Mapping[str, RelationTable]:
        """Return one relation family, keyed by name and read-only."""

    def get_relation(self, relation: str) -> RelationTable:
        """Return one relation by name, from either family."""

    def get_relation_types(self, kind: RelationKind = RelationKind.ALL) -> list[str]:
        """Return relation names, optionally of one family only."""

    def get_object_changes(self, obj_type: str) -> pd.DataFrame:
        """Return the recorded attribute changes of one object type."""

    def get_object_change_types(self) -> list[str]:
        """Return the object types that have recorded changes."""
