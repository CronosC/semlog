"""Reader for OCEL 2.0 logs.

`OCELLoader` reads the JSON, XML and SQLite forms into an `OCELStore` and
converts that into an OCED store.
"""

import logging
from collections.abc import Sequence

import pandas as pd

from semanticlog.core.enums import OCELExtension
from semanticlog.data.loading.base import Loader
from semanticlog.data.loading.ocel_frames import (
    Frames,
    fill_store,
    normalize_qualifiers,
    prefix_numeric_ids,
)
from semanticlog.data.loading.ocel_reader import read_ocel
from semanticlog.data.loading.ocel_schema import DeclaredTypes
from semanticlog.data.loading.ocel_store import OCELStore
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind

logger = logging.getLogger(__name__)


class OCELLoader(Loader):
    """Loads an OCEL 2.0 log in XML, JSON or SQLite form, picked by suffix.

    Args:
        ocel_extensions: Optional parts of the standard to load. Without
            `ATTRIBUTE_HISTORY` only the latest value of each object
            attribute is read.
    """

    def __init__(
        self, path: str, ocel_extensions: Sequence[OCELExtension] = ()
    ) -> None:
        self.path = path
        self._ocel_extensions: list[OCELExtension] = list(ocel_extensions)

    def get_oced(self) -> OCEDStore:
        """Read the log and return it as a store.

        Events become one table per activity, objects one table per type, and
        each relation qualifier becomes one named relation. A qualifier used
        by both families is suffixed with `_e2o` and `_o2o`. A relation with
        no qualifier takes `involves` or `relates`. The ids of a type whose
        ids are all numeric are prefixed with the type. An attribute keeps the
        type the log declares for it, and only undeclared ones are inferred.
        Each event carries its position in the log, ordered by timestamp and
        then by place in the file. For SQLite that place is the row order of
        the `event` table.

        An object carries the value of each attribute with the greatest
        timestamp, the last listed among equal ones. With `ATTRIBUTE_HISTORY`
        every value after the earliest is also recorded as a change.

        Returns:
            A store, in OCEL mode when attribute history was requested.

        Raises:
            InputError: If the suffix names no OCEL form, the file does not
                parse in it, the log has no events, the file uses one id
                twice, or an event time is not an ISO 8601 timestamp.
        """
        logger.info("Loading OCEL data from file %s", self.path)
        store = read_ocel(self.path)
        frames = to_frames(store, self.path)
        prefix_numeric_ids(frames)
        history = OCELExtension.ATTRIBUTE_HISTORY in self._ocel_extensions
        return fill_store(frames, history, _declared(store.declared))


def _declared(declared: pd.DataFrame) -> DeclaredTypes:
    """Return the declared types of a store keyed by kind and type."""
    types: DeclaredTypes = {}
    for kind, type_name, name, declared_type in declared.itertuples(index=False):
        types.setdefault((EntityKind(kind), str(type_name)), {})[str(name)] = str(
            declared_type
        )
    return types


def _wide(values: pd.DataFrame, owner: str) -> pd.DataFrame:
    """Return one row per owner and one column per attribute, in order of first appearance."""
    names = list(dict.fromkeys(values["name"]))
    wide = values.pivot(index=owner, columns="name", values="value").reindex(
        columns=names
    )
    wide.columns.name = None
    # An all-missing column stays object, so later values of any type fit in.
    filled = [c for c in wide.columns if wide[c].notna().any()]
    wide[filled] = wide[filled].infer_objects()
    return wide


def _known(frame: pd.DataFrame, known: pd.Series, what: str, path: str) -> pd.DataFrame:
    """Return the relation rows marked known, warning about the rest."""
    if not known.all():
        logger.warning(
            "%d %s relation(s) of %s name an event or object the log does not "
            "contain and were left out.",
            int((~known).sum()),
            what,
            path,
        )
    return frame[known]


def to_frames(store: OCELStore, path: str) -> Frames:
    """Return the tables of a store in PM4Py's column names, ready for `fill_store`.

    Objects carry the earliest value of each attribute, even a missing one,
    and every later value that is not missing is a change. Relation rows naming an event or object the log does not
    contain are dropped with a warning, and repeated ones are dropped.

    Args:
        path: The file the store was read from, named in warnings.
    """
    object_type = store.objects.set_index("id")["type"]
    values = store.object_values
    earliest = ~values.duplicated(["object_id", "name"])
    objects = store.objects.rename(columns={"id": "ocel:oid", "type": "ocel:type"})
    objects = objects.join(_wide(values[earliest], "object_id"), on="ocel:oid")

    later = values[~earliest & values["value"].notna()]
    changes = None
    if len(later):
        changes = pd.DataFrame(
            {
                "ocel:oid": later["object_id"],
                "ocel:type": later["object_id"].map(object_type),
                "ocel:timestamp": later["time"],
                "ocel:field": later["name"],
            }
        )
        for name in dict.fromkeys(later["name"]):
            changes[name] = later["value"].where(later["name"] == name)
        changes = changes.reset_index(drop=True).infer_objects()

    ordered = store.events.sort_values(["time", "position"], kind="stable")
    rank = pd.Series(range(len(ordered)), index=ordered["id"])
    event_values = store.event_values
    event_values = event_values[
        ~event_values.duplicated(["event_id", "name"], keep="last")
    ]
    events = (
        ordered[["id", "time", "type"]]
        .rename(
            columns={
                "id": "ocel:eid",
                "time": "ocel:timestamp",
                "type": "ocel:activity",
            }
        )
        .join(_wide(event_values, "event_id"), on="ocel:eid")
        .reset_index(drop=True)
    )

    event_type = store.events.set_index("id")["type"]
    e2o = pd.DataFrame(
        {
            "ocel:eid": store.e2o["event_id"],
            "ocel:activity": store.e2o["event_id"].map(event_type),
            "ocel:oid": store.e2o["object_id"],
            "ocel:type": store.e2o["object_id"].map(object_type),
            "ocel:qualifier": store.e2o["qualifier"],
        }
    )
    e2o = _known(
        e2o, e2o["ocel:activity"].notna() & e2o["ocel:type"].notna(), "e2o", path
    )
    e2o = e2o.iloc[e2o["ocel:eid"].map(rank).argsort(kind="stable")]
    o2o = pd.DataFrame(
        {
            "ocel:oid": store.o2o["source_id"],
            "ocel:oid_2": store.o2o["target_id"],
            "ocel:qualifier": store.o2o["qualifier"],
        }
    )
    known_objects = set(object_type.index)
    o2o = _known(
        o2o,
        o2o["ocel:oid"].isin(known_objects) & o2o["ocel:oid_2"].isin(known_objects),
        "o2o",
        path,
    )
    return Frames(
        objects=objects,
        events=events,
        e2o=_unique(normalize_qualifiers(e2o, "e2o"), ["ocel:eid", "ocel:oid"]),
        o2o=_unique(normalize_qualifiers(o2o, "o2o"), ["ocel:oid", "ocel:oid_2"]),
        changes=changes,
    )


def _unique(relations: pd.DataFrame, ends: list[str]) -> pd.DataFrame:
    """Return the relation rows with repeated ones dropped and a fresh index."""
    return relations.drop_duplicates([*ends, "ocel:qualifier"]).reset_index(drop=True)
