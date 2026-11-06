"""The tables of one OCEL 2.0 log, as the file holds them.

`OCELStore` keeps every event, object, relation and attribute value of the
log, including the ones an OCED store has no place for, such as earlier
values of an object attribute.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class OCELStore:
    """The OCEL 2.0 tables of one log, one row per entry of the file.

    Ids are text. Rows keep the order of the file, apart from the attribute
    values of an object, which are ordered by time. A value whose attribute
    has a declared type holds that type. A missing event attribute value is
    not recorded. A missing object attribute value is, since it can be the
    first value of its attribute.

    Attributes:
        events: `id`, `type`, `time` as a UTC timestamp, and `position`, the
            place of the event in the file counted from 0.
        objects: `id` and `type`.
        e2o: `event_id`, `object_id` and `qualifier`, which is None where
            the file gives none.
        o2o: `source_id`, `target_id` and `qualifier`.
        event_values: `event_id`, `name` and `value`.
        object_values: `object_id`, `name`, `time` as a UTC timestamp, missing
            where it does not parse, and `value`.
        declared: `kind`, an `EntityKind` value, `type`, `name` and
            `declared`, one of `DECLARED_TYPES`.
    """

    events: pd.DataFrame
    objects: pd.DataFrame
    e2o: pd.DataFrame
    o2o: pd.DataFrame
    event_values: pd.DataFrame
    object_values: pd.DataFrame
    declared: pd.DataFrame
