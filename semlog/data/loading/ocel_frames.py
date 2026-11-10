"""The OCEL 2.0 tables both OCEL loaders produce, and their conversion to a store.

`Frames` holds the tables of one log in PM4Py's column names. `fill_store`
turns them into an OCED store, one table per type and one relation per
qualifier.
"""

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from semlog.core.errors import InputError
from semlog.data.cleaning.names import free_name
from semlog.data.dtypes import cast_declared
from semlog.data.loading.base import Inference
from semlog.data.loading.ocel_schema import DeclaredTypes
from semlog.data.oced import OCEDStore
from semlog.data.tables import EntityKind, RelationKind

logger = logging.getLogger(__name__)

OCEL_FORMATS: dict[str, str] = {
    ".xml": "xml",
    ".xmlocel": "xml",
    ".json": "json",
    ".jsonocel": "json",
    ".sqlite": "sqlite",
}

OCEL_SUFFIXES: frozenset[str] = frozenset(OCEL_FORMATS)

# Text some exporters write for a missing qualifier.
_MISSING_QUALIFIER_WORDS: list[str] = ["None", "none", "NA", "N/A"]
_MISSING_QUALIFIER_STRINGS: list[str] = [*_MISSING_QUALIFIER_WORDS, ""]
_MISSING_QUALIFIERS: list = [None, np.nan, *_MISSING_QUALIFIER_STRINGS]
_FALLBACK_QUALIFIERS: dict[str, str] = {"e2o": "involves", "o2o": "relates"}


def _numerically_indexed(
    frame: pd.DataFrame, type_column: str, id_column: str
) -> set[str]:
    """Return the types whose ids are all numeric."""
    numeric = set()
    for name, group in frame.groupby(type_column):
        ids = group[id_column].dropna()
        if len(ids) and pd.to_numeric(ids, errors="coerce").notna().all():
            numeric.add(str(name))
    return numeric


def _prefixed(ids: pd.Series, owners: pd.Series, prefix_types: set[str]) -> pd.Series:
    """Prefix the ids owned by a type in `prefix_types` with that type."""
    if not prefix_types:
        return ids
    needs = owners.astype(str).isin(prefix_types)
    if not needs.any():
        return ids
    return ids.astype(str).where(~needs, owners.astype(str) + "_" + ids.astype(str))


def _object_types(ids: pd.Series, owner: dict[str, str]) -> list[str]:
    """Return the types the given object ids belong to, by the id-to-type map."""
    return sorted({str(owner[i]) for i in ids.unique() if i in owner})


def normalize_qualifiers(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Give relations with no qualifier their family's fallback name.

    Args:
        kind: `e2o` or `o2o`.

    Returns:
        A copy with every missing qualifier filled in.
    """
    df = df.copy()
    fallback = _FALLBACK_QUALIFIERS[kind]
    qualifier = df["ocel:qualifier"]
    blank = int((qualifier.isna() | qualifier.eq("")).sum())
    worded = qualifier[qualifier.isin(_MISSING_QUALIFIER_WORDS)]
    if blank + len(worded):
        logger.warning(
            "%s %s relation(s) had no "
            "qualifier and were assigned the fallback relation name "
            "'%s'.",
            blank + len(worded),
            kind,
            fallback,
        )
    words = sorted(worded.astype(str).unique())
    sole_qualifier = len(words) == 1 and len(worded) == len(qualifier)
    if len(worded) and not sole_qualifier:
        logger.warning(
            "%s of those carry the text "
            "(%s), which is read as no qualifier and merged "
            "into '%s'. Rename it in the log to keep it as a separate "
            "relation.",
            len(worded),
            ", ".join(words),
            fallback,
        )
    df["ocel:qualifier"] = qualifier.replace(_MISSING_QUALIFIERS, fallback)
    return df


def relation_names(
    e2o_qualifiers: pd.Series, o2o_qualifiers: pd.Series
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the relation name each qualifier becomes, per family.

    A qualifier used by one family keeps its name. A qualifier used by both
    families gets the suffix of each family, and a number appended if the
    suffixed name is also a qualifier.

    Returns:
        The event-to-object and object-to-object maps, each from qualifier to
        relation name.
    """
    e2o = {str(q) for q in e2o_qualifiers.unique()}
    o2o = {str(q) for q in o2o_qualifiers.unique()}
    e2o_names = {q: q for q in e2o}
    o2o_names = {q: q for q in o2o}

    taken = e2o | o2o
    for qualifier in sorted(e2o & o2o):
        for names, suffix in ((e2o_names, "_e2o"), (o2o_names, "_o2o")):
            name = free_name(qualifier + suffix, taken)
            taken.add(name)
            names[qualifier] = name
    return e2o_names, o2o_names


def repeated(ids: Iterable[str]) -> list[str]:
    """Return the ids that occur more than once, sorted."""
    return sorted(i for i, n in Counter(ids).items() if n > 1)


def reject_repeated_ids(path: str, repeated: dict[str, list[str]]) -> None:
    """Raise if a file uses one id for two objects or two events.

    Args:
        repeated: The repeated ids under `object` and under `event`.

    Raises:
        InputError: If either list is not empty.
    """
    for what, ids in repeated.items():
        if ids:
            shown = ", ".join(ids[:5])
            more = f" and {len(ids) - 5} more" if len(ids) > 5 else ""
            raise InputError(
                f"{path} uses one {what} id for more than one {what}: "
                f"{shown}{more}. Give every event "
                "and object an id of its own, for example by prefixing "
                "ids with their type."
            )


@dataclass
class Frames:
    """The tables of one OCEL file.

    Attributes:
        objects: One row per object, with its type.
        events: One row per event, with its activity, sorted by timestamp and
            then by place in the file.
        e2o: Event-to-object relation rows.
        o2o: Object-to-object relation rows.
        changes: Attribute change rows, or None when the log carries none.
    """

    objects: pd.DataFrame
    events: pd.DataFrame
    e2o: pd.DataFrame
    o2o: pd.DataFrame
    changes: pd.DataFrame | None


def prefix_numeric_ids(frames: Frames) -> None:
    """Prefix the ids of every type or activity whose ids are all numeric, in place."""
    numeric_types = _numerically_indexed(frames.objects, "ocel:type", "ocel:oid")
    numeric_activities = _numerically_indexed(
        frames.events, "ocel:activity", "ocel:eid"
    )
    if numeric_types:
        logger.info(
            "Prefixing numeric object ids of: %s", ", ".join(sorted(numeric_types))
        )
    if numeric_activities:
        logger.info(
            "Prefixing numeric event ids of: %s", ", ".join(sorted(numeric_activities))
        )

    # Read before any id is rewritten. The o2o table holds the original ids.
    owner = frames.objects.set_index("ocel:oid")["ocel:type"]

    frames.objects["ocel:oid"] = _prefixed(
        frames.objects["ocel:oid"], frames.objects["ocel:type"], numeric_types
    )
    frames.events["ocel:eid"] = _prefixed(
        frames.events["ocel:eid"], frames.events["ocel:activity"], numeric_activities
    )
    frames.e2o["ocel:oid"] = _prefixed(
        frames.e2o["ocel:oid"], frames.e2o["ocel:type"], numeric_types
    )
    frames.e2o["ocel:eid"] = _prefixed(
        frames.e2o["ocel:eid"], frames.e2o["ocel:activity"], numeric_activities
    )
    if frames.changes is not None:
        frames.changes["ocel:oid"] = _prefixed(
            frames.changes["ocel:oid"], frames.changes["ocel:type"], numeric_types
        )
    for column in ("ocel:oid", "ocel:oid_2"):
        frames.o2o[column] = _prefixed(
            frames.o2o[column], frames.o2o[column].map(owner), numeric_types
        )


def _latest_values(objects: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Return the objects frame with the latest changed value of each attribute.

    For every object and field in the change rows, the value of the row with
    the greatest timestamp replaces the object's value. A timestamp that does
    not parse sorts before every one that does. Rows at one timestamp keep
    their order, and the last wins. A field with no column of its own in the
    change rows is skipped.

    Returns:
        A new frame. The input is not modified.
    """
    objects = objects.copy()
    order = pd.to_datetime(changes["ocel:timestamp"], errors="coerce", utc=True)
    ranked = changes.assign(_order=order).sort_values(
        "_order", kind="mergesort", na_position="first"
    )
    for field, rows in ranked.groupby("ocel:field", sort=False):
        name = str(field)
        if name not in rows.columns:
            continue
        latest = rows.drop_duplicates("ocel:oid", keep="last").set_index("ocel:oid")[
            name
        ]
        held = objects["ocel:oid"].isin(latest.index)
        if name not in objects.columns:
            objects[name] = None
        objects.loc[held, name] = objects.loc[held, "ocel:oid"].map(latest)
    return objects


def fill_store(
    frames: Frames, load_attr_history: bool, declared: DeclaredTypes
) -> OCEDStore:
    """Return a store with one table per type and one relation per qualifier.

    Args:
        declared: The attribute types the log declares, per event or object
            type.

    Raises:
        InputError: If an object-to-object relation names an object the log
            does not contain.
    """
    oced = OCEDStore(ocel_mode=load_attr_history)
    inference = Inference()

    objects = frames.objects
    if frames.changes is not None:
        objects = _latest_values(objects, frames.changes)

    for object_type, frame in objects.groupby("ocel:type"):
        frame = frame.dropna(axis=1, how="all").drop(columns=["ocel:type"])
        oced.add(
            EntityKind.OBJECT,
            str(object_type),
            inference.infer(
                frame,
                "ocel:oid",
                declared.get((EntityKind.OBJECT, str(object_type))),
            ),
            id_col="ocel:oid",
        )

    events = frames.events.reset_index(drop=True)
    position = free_name("ocel:position", {str(c) for c in events.columns})
    for activity, frame in events.groupby("ocel:activity"):
        frame = frame.drop(columns=["ocel:activity"])
        inferred = inference.infer(
            frame,
            "ocel:eid",
            declared.get((EntityKind.EVENT, str(activity))),
        )
        oced.add(
            EntityKind.EVENT,
            str(activity),
            inferred.assign(**{position: inferred.index}),
            id_col="ocel:eid",
            time_col="ocel:timestamp",
            position_col=position,
        )

    e2o_names, o2o_names = relation_names(
        frames.e2o["ocel:qualifier"], frames.o2o["ocel:qualifier"]
    )
    owner = dict(
        zip(frames.objects["ocel:oid"], frames.objects["ocel:type"], strict=True)
    )

    for qualifier, frame in frames.o2o.groupby("ocel:qualifier"):
        from_types = _object_types(frame["ocel:oid"], owner)
        to_types = _object_types(frame["ocel:oid_2"], owner)
        if not from_types or not to_types:
            raise InputError(
                f"Could not resolve object types for o2o relation {qualifier!r}: "
                "one of its endpoints names no object in this log."
            )
        oced.add_relation(
            RelationKind.O2O,
            o2o_names[str(qualifier)],
            frame[["ocel:oid", "ocel:oid_2"]],
            from_id="ocel:oid",
            from_type=from_types,
            to_id="ocel:oid_2",
            to_type=to_types,
        )

    for qualifier, frame in frames.e2o.groupby("ocel:qualifier"):
        oced.add_relation(
            RelationKind.E2O,
            e2o_names[str(qualifier)],
            frame[["ocel:eid", "ocel:oid"]],
            from_id="ocel:eid",
            to_id="ocel:oid",
            to_type=sorted(map(str, frame["ocel:type"].unique())),
        )

    if load_attr_history and frames.changes is not None:
        for object_type, frame in frames.changes.groupby("ocel:type"):
            frame = cast_declared(
                frame, declared.get((EntityKind.OBJECT, str(object_type)), {})
            )
            oced.add_object_changes(str(object_type), frame.reset_index(drop=True))

    inference.report()
    return oced
