"""Reader for OCEL 2.0 logs, via pm4py.

The XML, JSON and SQLite forms of OCEL 2.0 all arrive through pm4py as one
`OCEL` object, which this module reshapes into an OCED store. Two things are
repaired on the way in: relations with no qualifier get the fallback name of
their family so they can still become a property, and ids that are only unique
within their type are prefixed with that type so they become unique across the
log.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py as pm
from pm4py.objects.ocel.obj import OCEL

from semanticlog.core.enums import OCELExtension
from semanticlog.core.errors import InputError
from semanticlog.data.loading.base import Inference, Loader
from semanticlog.data.oced import OCEDStore

logger = logging.getLogger(__name__)

_FORMAT_BY_SUFFIX: dict[str, str] = {
    ".xml": "xml",
    ".xmlocel": "xml",
    ".json": "json",
    ".jsonocel": "json",
    ".sqlite": "sqlite",
}

OCEL_SUFFIXES: frozenset[str] = frozenset(_FORMAT_BY_SUFFIX)

_READER_NAMES: dict[str, str] = {
    "xml": "read_ocel2_xml",
    "json": "read_ocel2_json",
    "sqlite": "read_ocel2_sqlite",
}

_MISSING_QUALIFIER_STRINGS: list[str] = ["None", "none", "NA", "N/A", ""]
_MISSING_QUALIFIERS: list = [None, np.nan, *_MISSING_QUALIFIER_STRINGS]
_FALLBACK_QUALIFIERS: dict[str, str] = {"e2o": "modifies", "o2o": "relates"}


def _require_unique(ids: pd.Series, what: str) -> None:
    """Raise if any id occurs more than once, naming the first few."""
    duplicated = ids[ids.duplicated()].unique()
    if len(duplicated):
        raise InputError(
            f"{what} are not unique across the log: "
            f"{', '.join(map(str, duplicated[:5]))}"
            + (f" and {len(duplicated) - 5} more" if len(duplicated) > 5 else "")
        )


def _numerically_indexed(
    frame: pd.DataFrame, type_column: str, id_column: str
) -> set[str]:
    """Return the types whose ids are all numbers, and so unique per type only."""
    numeric = set()
    for name, group in frame.groupby(type_column):
        ids = group[id_column].dropna()
        if len(ids) and pd.to_numeric(ids, errors="coerce").notna().all():
            numeric.add(str(name))
    return numeric


def _prefixed(ids: pd.Series, owners: pd.Series, prefix_types: set[str]) -> pd.Series:
    """Prefix ids with their own type, for the types that need it."""
    if not prefix_types:
        return ids
    needs = owners.astype(str).isin(prefix_types)
    if not needs.any():
        return ids
    return ids.astype(str).where(~needs, owners.astype(str) + "_" + ids.astype(str))


def _object_types(ids: pd.Series, objects: pd.DataFrame) -> list[str]:
    """Return the types the given object ids belong to."""
    owner = dict(zip(objects["ocel:oid"], objects["ocel:type"], strict=True))
    return sorted({str(owner[i]) for i in ids.unique() if i in owner})


def _normalize_qualifiers(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Give relations with no qualifier their family's fallback name.

    Args:
        df: Relation rows of one family.
        kind: `e2o` or `o2o`, which picks the fallback name and names the
            family in the warning.

    Returns:
        A copy with every missing qualifier filled in.
    """
    df = df.copy()
    fallback = _FALLBACK_QUALIFIERS[kind]
    qualifier = df["ocel:qualifier"]
    n_missing = int(
        qualifier.isna().sum() + qualifier.isin(_MISSING_QUALIFIER_STRINGS).sum()
    )
    if n_missing:
        logger.warning(
            f"pm4pyloader: {n_missing} {kind} relation(s) had no qualifier and were "
            f"assigned the fallback relation name '{fallback}'."
        )
    df["ocel:qualifier"] = qualifier.replace(_MISSING_QUALIFIERS, fallback)
    return df


def _free_relation_name(name: str, taken: set[str]) -> str:
    """Return the name, or it with underscores appended until it is unused."""
    while name in taken:
        name += "_"
    return name


def _relation_names(
    e2o_qualifiers: pd.Series, o2o_qualifiers: pd.Series
) -> tuple[dict[str, str], dict[str, str]]:
    """Return the relation name each qualifier becomes, per family.

    A qualifier used by one family only keeps its own name. One used by both
    would become a single property carrying the domain of an event and of an
    object, so each side takes the suffix of its family, and an underscore
    after that if the suffixed name is a qualifier of the log as well.

    Args:
        e2o_qualifiers: Qualifiers of the event-to-object relations.
        o2o_qualifiers: Qualifiers of the object-to-object relations.

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
            name = _free_relation_name(qualifier + suffix, taken)
            taken.add(name)
            names[qualifier] = name
    return e2o_names, o2o_names


class PM4PyLoader(Loader):
    """Loads an OCEL 2.0 log in XML, JSON or SQLite form."""

    def __init__(
        self,
        path: str,
        config: dict[str, str],
        ocel_extensions: list[OCELExtension] | None = None,
    ):
        """Prepare to read an OCEL 2.0 file.

        Args:
            path: File to read.
            config: Loader options. `file_format` forces a format instead of
                inferring one from the suffix. The resolved format is written
                back into the dict.
            ocel_extensions: Optional parts of the standard to load. Without
                `ATTRIBUTE_HISTORY` only the latest value of each object
                attribute is read.

        Raises:
            ValueError: If the format is unsupported or cannot be inferred.
        """
        super().__init__(path, config)
        self._ocel_extensions: list[OCELExtension] = ocel_extensions or []
        self.file_format = self._resolve_file_format(path, config)
        config["file_format"] = self.file_format

    @staticmethod
    def _resolve_file_format(path: str, config: dict[str, str]) -> str:
        """Take the format from the config, or infer it from the suffix."""
        given = config.get("file_format")
        if given is not None:
            if given not in _READER_NAMES:
                raise InputError(
                    f"Unsupported file_format '{given}'. "
                    f"Supported formats: {', '.join(sorted(_READER_NAMES))}."
                )
            return given
        inferred = _FORMAT_BY_SUFFIX.get(Path(path).suffix.lower())
        if inferred is None:
            raise InputError(
                f"Cannot infer OCEL format from '{path}'. Supported extensions: "
                f"{', '.join(sorted(_FORMAT_BY_SUFFIX))}."
            )
        return inferred

    def get_oced(self) -> OCEDStore:
        """Read the log and return it as a store.

        Events become one table per activity, objects one table per type, and
        each relation qualifier becomes one named relation. A qualifier used
        by both families is suffixed with `_e2o` and `_o2o`, so it stays two
        relations.

        Returns:
            A store, in OCEL mode when attribute history was requested. A log
            that carries none leaves the store in OCEL mode with no changes
            recorded.

        Raises:
            ValueError: If event or object ids are not unique, or an
                object-to-object relation names an object the log does not
                contain.
        """
        load_attr_history = OCELExtension.ATTRIBUTE_HISTORY in self._ocel_extensions

        logger.info(f"Loading ocel data from file {self.path}..")
        read: Callable[[str], OCEL] = getattr(pm, _READER_NAMES[self.file_format])

        frames = _read_frames(read(self.path), load_attr_history)
        _prefix_numeric_ids(frames)
        return _fill_store(frames, load_attr_history)


@dataclass
class _Frames:
    """The tables of one OCEL file, on their way into a store.

    Attributes:
        objects: One row per object, with its type.
        events: One row per event, with its activity.
        e2o: Event-to-object relation rows.
        o2o: Object-to-object relation rows.
        changes: Attribute change rows, or None when the history was not asked
            for or the log carries none.
    """

    objects: pd.DataFrame
    events: pd.DataFrame
    e2o: pd.DataFrame
    o2o: pd.DataFrame
    changes: pd.DataFrame | None


def _read_frames(log: OCEL, load_attr_history: bool) -> _Frames:
    """Take copies of what the log holds, and check its ids are unique.

    Raises:
        ValueError: If event or object ids repeat within their family.
    """
    changes = log.object_changes if load_attr_history else None
    frames = _Frames(
        objects=log.objects.copy(),
        events=log.events.copy(),
        e2o=_normalize_qualifiers(log.relations, "e2o"),
        o2o=_normalize_qualifiers(log.o2o, "o2o"),
        changes=changes.copy() if changes is not None and len(changes) else None,
    )
    _require_unique(frames.events["ocel:eid"], "Event ids")
    _require_unique(frames.objects["ocel:oid"], "Object ids")
    return frames


def _prefix_numeric_ids(frames: _Frames) -> None:
    """Prefix ids that are only unique within their own type.

    A log numbering each type from one would otherwise give two objects the
    same id, and one id is one individual. The owner of each object id is read
    before anything is rewritten, since the relation tables still name the
    originals.

    Args:
        frames: Tables to rewrite, modified in place.
    """
    numeric_types = _numerically_indexed(frames.objects, "ocel:type", "ocel:oid")
    numeric_activities = _numerically_indexed(
        frames.events, "ocel:activity", "ocel:eid"
    )
    if numeric_types:
        logger.info(
            f"Prefixing numeric object ids of: {', '.join(sorted(numeric_types))}"
        )
    if numeric_activities:
        logger.info(
            "Prefixing numeric event ids of: " + ", ".join(sorted(numeric_activities))
        )

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


def _fill_store(frames: _Frames, load_attr_history: bool) -> OCEDStore:
    """Turn the read tables into a store, one table per type and per qualifier.

    Raises:
        ValueError: If an object-to-object relation names an object the log
            does not contain.
    """
    oced = OCEDStore(ocel_mode=load_attr_history)
    inference = Inference()

    for object_type, frame in frames.objects.groupby("ocel:type"):
        frame = frame.dropna(axis=1, how="all").drop(columns=["ocel:type"])
        oced.add_object(
            str(object_type), "ocel:oid", inference.infer(frame, "ocel:oid")
        )

    for activity, frame in frames.events.groupby("ocel:activity"):
        frame = frame.drop(columns=["ocel:activity"])
        oced.add_event(
            str(activity),
            "ocel:eid",
            "ocel:timestamp",
            inference.infer(frame, "ocel:eid"),
        )

    e2o_names, o2o_names = _relation_names(
        frames.e2o["ocel:qualifier"], frames.o2o["ocel:qualifier"]
    )

    for qualifier, frame in frames.o2o.groupby("ocel:qualifier"):
        from_types = _object_types(frame["ocel:oid"], frames.objects)
        to_types = _object_types(frame["ocel:oid_2"], frames.objects)
        if not from_types or not to_types:
            raise InputError(
                f"Could not resolve object types for o2o relation {qualifier!r}: "
                "one of its endpoints names no object in this log."
            )
        oced.add_object_to_object_relation(
            o2o_names[str(qualifier)],
            "ocel:oid",
            from_types,
            "ocel:oid_2",
            to_types,
            frame,
        )

    for qualifier, frame in frames.e2o.groupby("ocel:qualifier"):
        oced.add_event_to_object_relation(
            e2o_names[str(qualifier)],
            "ocel:eid",
            "ocel:oid",
            sorted(map(str, frame["ocel:type"].unique())),
            frame[["ocel:eid", "ocel:oid"]],
        )

    if frames.changes is not None:
        for object_type, frame in frames.changes.groupby("ocel:type"):
            oced.add_object_changes(str(object_type), frame.reset_index(drop=True))

    inference.report()
    return oced
