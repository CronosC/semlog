"""Reader for OCEL 2.0 logs, through PM4Py.

`PM4PyLoader` reads the XML, JSON and SQLite forms into an OCED store.
"""

import json
import logging
import sqlite3
import xml.etree.ElementTree as ElementTree
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pm4py as pm
from pm4py.objects.ocel.obj import OCEL

from semlog.core.enums import OCELExtension
from semlog.core.errors import InputError
from semlog.data.loading.base import Loader
from semlog.data.loading.ocel_frames import (
    OCEL_FORMATS,
    Frames,
    fill_store,
    normalize_qualifiers,
    prefix_numeric_ids,
    reject_repeated_ids,
    repeated,
)
from semlog.data.loading.ocel_schema import declared_types
from semlog.data.oced import OCEDStore

logger = logging.getLogger(__name__)

_READER_NAMES: dict[str, str] = {
    "xml": "read_ocel2_xml",
    "json": "read_ocel2_json",
    "sqlite": "read_ocel2_sqlite",
}


def _require_unique(ids: pd.Series, what: str) -> None:
    """Raise if any id occurs more than once, naming the first few."""
    duplicated = ids[ids.duplicated()].unique()
    if len(duplicated):
        raise InputError(
            f"{what} are not unique across the log: "
            f"{', '.join(map(str, duplicated[:5]))}"
            + (f" and {len(duplicated) - 5} more" if len(duplicated) > 5 else "")
            + ". Give every event and object an id of its own."
        )


class PM4PyLoader(Loader):
    """Loads an OCEL 2.0 log in XML, JSON or SQLite form, picked by suffix.

    Args:
        ocel_extensions: Optional parts of the standard to load. Without
            `ATTRIBUTE_HISTORY` only the latest value of each object
            attribute is read.

    Attributes:
        file_format: The format read, `xml`, `json` or `sqlite`.

    Raises:
        InputError: If no format can be inferred from the suffix.
    """

    def __init__(
        self, path: str, ocel_extensions: Sequence[OCELExtension] = ()
    ) -> None:
        self.path = path
        self._ocel_extensions: list[OCELExtension] = list(ocel_extensions)
        inferred = OCEL_FORMATS.get(Path(path).suffix.lower())
        if inferred is None:
            raise InputError(
                f"Cannot infer OCEL format from '{path}'. Supported extensions: "
                f"{', '.join(sorted(OCEL_FORMATS))}."
            )
        self.file_format = inferred

    def get_oced(self) -> OCEDStore:
        """Read the log and return it as a store.

        Events become one table per activity, objects one table per type, and
        each relation qualifier becomes one named relation. A qualifier used
        by both families is suffixed with `_e2o` and `_o2o`. A relation with
        no qualifier takes `involves` or `relates`. The ids of a type whose
        ids are all numeric are prefixed with the type. An attribute keeps the
        type the log declares for it, and only undeclared ones are inferred.
        Each event carries its position in the order PM4Py reads the events,
        which for XML and JSON is their order in the file.

        An object carries the value of each attribute with the greatest
        timestamp. The value PM4Py lists first for an attribute is read as its
        initial value. With `ATTRIBUTE_HISTORY` the later values are also
        recorded as changes.

        Returns:
            A store, in OCEL mode when attribute history was requested. A log
            that carries none leaves the store in OCEL mode with no changes
            recorded.

        Raises:
            InputError: If the file uses one id twice, or an object-to-object
                relation names an object the log does not contain.
        """
        load_attr_history = OCELExtension.ATTRIBUTE_HISTORY in self._ocel_extensions

        logger.info("Loading OCEL data from file %s", self.path)
        read: Callable[[str], OCEL] = getattr(pm, _READER_NAMES[self.file_format])

        frames = _read_frames(self._read(read))
        prefix_numeric_ids(frames)
        declared = declared_types(self.path, self.file_format)
        return fill_store(frames, load_attr_history, declared)

    def _read(self, read: Callable[[str], OCEL]) -> OCEL:
        """Read the file with PM4Py.

        Raises:
            InputError: If the file uses one id for two objects or two
                events.
        """
        reject_repeated_ids(self.path, _repeated_ids(self.path, self.file_format))
        return read(self.path)


def _repeated_ids(path: str, file_format: str) -> dict[str, list[str]]:
    """Return the object ids and event ids a file uses more than once.

    Args:
        file_format: `json`, `xml` or `sqlite`.

    Returns:
        The repeated ids under `object` and under `event`.
    """
    if file_format == "json":
        with open(path, encoding="utf-8") as handle:
            doc = json.load(handle)
        return {
            "object": repeated(str(o["id"]) for o in doc.get("objects", [])),
            "event": repeated(str(e["id"]) for e in doc.get("events", [])),
        }
    if file_format == "xml":
        ids: dict[str, list[str]] = {"object": [], "event": []}
        # The file is the user's own log.
        for _, element in ElementTree.iterparse(path):  # noqa: S314
            if element.tag in ids and "id" in element.attrib:
                ids[element.tag].append(element.attrib["id"])
                element.clear()
        return {what: repeated(found) for what, found in ids.items()}
    queries = {
        "object": "SELECT ocel_id FROM object GROUP BY ocel_id HAVING COUNT(*) > 1",
        "event": "SELECT ocel_id FROM event GROUP BY ocel_id HAVING COUNT(*) > 1",
    }
    connection = sqlite3.connect(path)
    try:
        return {
            what: [str(row[0]) for row in connection.execute(query)]
            for what, query in queries.items()
        }
    finally:
        connection.close()


def _read_frames(log: OCEL) -> Frames:
    """Return copies of the tables of the log with qualifiers filled in.

    The change rows are read whether or not the history is kept, because the
    latest value of an object attribute is among them.

    Raises:
        InputError: If an event or object id repeats.
    """
    changes = log.object_changes
    frames = Frames(
        objects=log.objects.copy(),
        events=log.events.copy(),
        e2o=normalize_qualifiers(log.relations, "e2o"),
        o2o=normalize_qualifiers(log.o2o, "o2o"),
        changes=changes.copy() if changes is not None and len(changes) else None,
    )
    _require_unique(frames.events["ocel:eid"], "Event ids")
    _require_unique(frames.objects["ocel:oid"], "Object ids")
    return frames
