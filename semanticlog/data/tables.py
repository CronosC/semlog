"""The tables an OCED store is made of.

Each table pairs a DataFrame with the names of its id and timestamp columns.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd

from semanticlog.core.enums import IndividualKind

OCEL_PREFIX = "ocel:"


class EntityKind(str, Enum):
    """The kind of an entity, an event or an object.

    Attributes:
        EVENT: An event, which has a timestamp.
        OBJECT: An object, which events relate to.
    """

    EVENT = "event"
    OBJECT = "object"

    @property
    def word(self) -> str:
        """The word for this kind in a message, `activity` or `object type`."""
        return "activity" if self is EntityKind.EVENT else "object type"

    @property
    def individual(self) -> IndividualKind:
        """The kind of individual an entity of this kind becomes."""
        return (
            IndividualKind.EVENT if self is EntityKind.EVENT else IndividualKind.OBJECT
        )

    @property
    def plural(self) -> str:
        """The plural word for this kind in a message, `activities` or `object types`."""
        return "activities" if self is EntityKind.EVENT else "object types"


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

    @classmethod
    def families(cls) -> tuple[RelationKind, RelationKind]:
        """The two families, in the order a store lists them."""
        return (cls.E2O, cls.O2O)

    @property
    def word(self) -> str:
        """How a message about the store names this family."""
        return {
            RelationKind.ALL: "relation",
            RelationKind.E2O: "event-to-object",
            RelationKind.O2O: "object-to-object",
        }[self]


@dataclass
class EntityTable:
    """The events of one activity, or the objects of one type.

    An event table has a timestamp column. An object table does not.

    Attributes:
        id: Column holding the entity id.
        data: One row per entity. Every column other than `id` and `time` is
            an attribute.
        time: Column holding the event timestamp, or None for an object table.
    """

    id: str
    data: pd.DataFrame
    time: str | None = None

    @property
    def key_columns(self) -> frozenset[str]:
        """The id column and, for events, the time column."""
        if self.time is None:
            return frozenset({self.id})
        return frozenset({self.id, self.time})


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
    """One table of a store, with its structural columns named.

    Attributes:
        frame: The rows themselves.
        name: Activity or object type this table holds.
        kind: The kind of entity the rows are, or None for a table whose rows
            are not entities.
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
        """The columns not in `key_columns`."""
        return [str(c) for c in self.frame.columns if c not in self.key_columns]

    @property
    def filled_attributes(self) -> list[str]:
        """The attribute columns that hold at least one value.

        A change table holds a column for every attribute that changes in the
        log, empty in the rows of the types that do not carry it.
        """
        return [c for c in self.attributes if self.frame[c].notna().any()]

    @property
    def structural(self) -> list[str]:
        """The structural columns other than the id, such as an event timestamp."""
        return [c for c in self.key_columns if c != self.id_column]


@dataclass
class Relations:
    """Both relation families of a store, keyed by kind and then by name.

    Attributes:
        families: The relation tables of each family. `RelationKind.ALL` names
            both families at once and is not a key.
    """

    families: dict[RelationKind, dict[str, RelationTable]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for kind in RelationKind.families():
            self.families.setdefault(kind, {})

    def of(self, kind: RelationKind) -> dict[str, RelationTable]:
        """Return the mutable mapping of one family."""
        return self.families[kind]

    def items(self) -> Iterator[tuple[str, RelationTable]]:
        """Yield every relation with its name, event-to-object first."""
        for kind in RelationKind.families():
            yield from self.families[kind].items()
