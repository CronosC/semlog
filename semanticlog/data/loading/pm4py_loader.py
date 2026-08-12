"""Reader for OCEL 2.0 logs, via pm4py.

The XML, JSON and SQLite forms of OCEL 2.0 all arrive through pm4py as one
`OCEL` object, which this module reshapes into an OCED store. Two things are
repaired on the way in: relations with no qualifier get a fallback name so
they can still become a property, and ids that are only unique within their
type are prefixed with that type so they become unique across the log.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pm4py as pm
from pm4py.objects.ocel.obj import OCEL

from semanticlog.core.enums import OCELExtension
from semanticlog.data.loading.base import Loader
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
_FALLBACK_QUALIFIER = "modifies"


def _require_unique(ids: pd.Series, what: str) -> None:
    """Raise if any id occurs more than once, naming the first few."""
    duplicated = ids[ids.duplicated()].unique()
    if len(duplicated):
        raise ValueError(
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
    """Give relations with no qualifier the fallback name, and report how many."""
    df = df.copy()
    qualifier = df["ocel:qualifier"]
    n_missing = int(
        qualifier.isna().sum() + qualifier.isin(_MISSING_QUALIFIER_STRINGS).sum()
    )
    if n_missing:
        logger.warning(
            f"pm4pyloader: {n_missing} {kind} relation(s) had no qualifier and were "
            f"assigned the fallback relation name '{_FALLBACK_QUALIFIER}'."
        )
    df["ocel:qualifier"] = qualifier.replace(_MISSING_QUALIFIERS, _FALLBACK_QUALIFIER)
    return df


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
                raise ValueError(
                    f"Unsupported file_format '{given}'. "
                    f"Supported formats: {', '.join(sorted(_READER_NAMES))}."
                )
            return given
        inferred = _FORMAT_BY_SUFFIX.get(Path(path).suffix.lower())
        if inferred is None:
            raise ValueError(
                f"Cannot infer OCEL format from '{path}'. Supported extensions: "
                f"{', '.join(sorted(_FORMAT_BY_SUFFIX))}."
            )
        return inferred

    def get_oced(self) -> OCEDStore:
        """Read the log and return it as a store.

        Events become one table per activity, objects one table per type, and
        each relation qualifier becomes one named relation. Object-to-object
        relations are suffixed with `_o2o`, so a qualifier used by both
        families stays two relations.

        Returns:
            A store, in OCEL mode when attribute history was requested and the
            log actually carries some.

        Raises:
            ValueError: If event or object ids are not unique, or an
                object-to-object relation names an object the log does not
                contain.
        """
        load_attr_history = OCELExtension.ATTRIBUTE_HISTORY in self._ocel_extensions

        logger.info(f"Loading ocel data from file {self.path}..")
        read: Callable[[str], OCEL] = getattr(pm, _READER_NAMES[self.file_format])
        log = read(self.path)

        objects = log.objects.copy()
        events = log.events.copy()
        relations = _normalize_qualifiers(log.relations, "e2o")
        o2o = _normalize_qualifiers(log.o2o, "o2o")
        changes = log.object_changes if load_attr_history else None
        changes = changes.copy() if changes is not None and len(changes) else None

        _require_unique(events["ocel:eid"], "Event ids")
        _require_unique(objects["ocel:oid"], "Object ids")

        numeric_types = _numerically_indexed(objects, "ocel:type", "ocel:oid")
        numeric_activities = _numerically_indexed(events, "ocel:activity", "ocel:eid")
        if numeric_types:
            logger.info(
                f"Prefixing numeric object ids of: {', '.join(sorted(numeric_types))}"
            )
        if numeric_activities:
            logger.info(
                "Prefixing numeric event ids of: "
                + ", ".join(sorted(numeric_activities))
            )

        objects["ocel:oid"] = _prefixed(
            objects["ocel:oid"], objects["ocel:type"], numeric_types
        )
        events["ocel:eid"] = _prefixed(
            events["ocel:eid"], events["ocel:activity"], numeric_activities
        )
        relations["ocel:oid"] = _prefixed(
            relations["ocel:oid"], relations["ocel:type"], numeric_types
        )
        relations["ocel:eid"] = _prefixed(
            relations["ocel:eid"], relations["ocel:activity"], numeric_activities
        )
        if changes is not None:
            changes["ocel:oid"] = _prefixed(
                changes["ocel:oid"], changes["ocel:type"], numeric_types
            )
        owner = log.objects.set_index("ocel:oid")["ocel:type"]
        for column in ("ocel:oid", "ocel:oid_2"):
            o2o[column] = _prefixed(o2o[column], o2o[column].map(owner), numeric_types)

        oced = OCEDStore(ocel_mode=load_attr_history)

        for object_type, frame in objects.groupby("ocel:type"):
            frame = frame.dropna(axis=1, how="all").drop(columns=["ocel:type"])
            oced.add_object(str(object_type), "ocel:oid", frame)

        for activity, frame in events.groupby("ocel:activity"):
            frame = frame.drop(columns=["ocel:activity"])
            oced.add_event(str(activity), "ocel:eid", "ocel:timestamp", frame)

        for qualifier, frame in o2o.groupby("ocel:qualifier"):
            from_types = _object_types(frame["ocel:oid"], objects)
            to_types = _object_types(frame["ocel:oid_2"], objects)
            if not from_types or not to_types:
                raise ValueError(
                    f"Could not resolve object types for o2o relation {qualifier!r}: "
                    "one of its endpoints names no object in this log."
                )
            oced.add_object_to_object_relation(
                f"{qualifier}_o2o",
                "ocel:oid",
                from_types,
                "ocel:oid_2",
                to_types,
                frame,
            )

        for qualifier, frame in relations.groupby("ocel:qualifier"):
            oced.add_event_to_object_relation(
                str(qualifier),
                "ocel:eid",
                "ocel:oid",
                sorted(map(str, frame["ocel:type"].unique())),
                frame[["ocel:eid", "ocel:oid"]],
            )

        if changes is not None:
            for object_type, frame in changes.groupby("ocel:type"):
                oced.add_object_changes(str(object_type), frame.reset_index(drop=True))

        return oced
