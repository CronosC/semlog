"""The tables an OCED store is made of.

Each table pairs a DataFrame with the column names that give it meaning, so
that nothing downstream has to guess which column holds an id or a timestamp.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

# What marks a column of a change table as structure rather than an attribute.
OCEL_PREFIX = "ocel:"


class EntityKind(str, Enum):
    """Which half of the store an entity belongs to."""

    EVENT = "event"
    OBJECT = "object"


class RelationKind(str, Enum):
    """Which family of relations a lookup is about.

    Attributes:
        ALL: Both families.
        E2O: Event to object.
        O2O: Object to object.
    """

    ALL = "all"
    E2O = "e2o"
    O2O = "o2o"


@dataclass
class EventTable:
    """The events of one activity.

    Attributes:
        id: Column holding the event id.
        time: Column holding the event timestamp.
        data: One row per event. Every other column is an attribute.
    """

    id: str
    time: str
    data: pd.DataFrame


@dataclass
class ObjectTable:
    """The objects of one type.

    Attributes:
        id: Column holding the object id.
        data: One row per object. Every other column is an attribute.
    """

    id: str
    data: pd.DataFrame


@dataclass
class RelationTable:
    """The rows of one named relation.

    Attributes:
        from_id: Column holding the id of the source entity.
        to_id: Column holding the id of the target entity.
        to_type: Types the target ids may belong to.
        data: One row per related pair.
        from_type: Types the source ids may belong to. None for an
            event-to-object relation, where the source is always an event.
    """

    from_id: str
    to_id: str
    to_type: list[str]
    data: pd.DataFrame
    from_type: list[str] | None = None


@dataclass(frozen=True)
class TableView:
    """One table of a store, as a pass over the whole store sees it.

    Every stage that rewrites names walks events, objects and change history
    alike, and the only thing that differs between them is which columns carry
    structure rather than an attribute. A view states that, so a pass says
    what it does to a table once instead of once per kind of table.

    Attributes:
        frame: The rows themselves.
        name: Activity or object type this table holds.
        kind: Which half of the store the rows are entities of, or None for a
            table whose rows are not entities.
        key_columns: Columns that carry structure. An id, a timestamp, or the
            `ocel:` columns of a change table.
        origin: How this table is named in a message, such as
            `activity 'ship'`.
        id_column: Column holding the id of the entity each row is, or None
            for a table whose rows are not entities.
        replace: Put a rewritten frame back into the store.
    """

    frame: pd.DataFrame
    name: str
    kind: EntityKind | None
    key_columns: frozenset[str]
    origin: str
    id_column: str | None
    replace: Callable[[pd.DataFrame], None]

    @property
    def attributes(self) -> list[str]:
        """The attribute columns, which are all the ones that carry no structure."""
        return [str(c) for c in self.frame.columns if c not in self.key_columns]

    @property
    def structural(self) -> list[str]:
        """The structural columns other than the id, such as an event timestamp."""
        return [c for c in self.key_columns if c != self.id_column]


@dataclass
class Relations:
    """Both relation families of a store, keyed by relation name.

    Attributes:
        event_to_object: Relations from events to objects.
        object_to_object: Relations between objects.
    """

    event_to_object: dict[str, RelationTable] = field(default_factory=dict)
    object_to_object: dict[str, RelationTable] = field(default_factory=dict)
