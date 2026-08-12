"""The tables an OCED store is made of.

Each table pairs a DataFrame with the column names that give it meaning, so
that nothing downstream has to guess which column holds an id or a timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


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


@dataclass
class Relations:
    """Both relation families of a store, keyed by relation name.

    Attributes:
        event_to_object: Relations from events to objects.
        object_to_object: Relations between objects.
    """

    event_to_object: dict[str, RelationTable] = field(default_factory=dict)
    object_to_object: dict[str, RelationTable] = field(default_factory=dict)
