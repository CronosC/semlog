"""Object attribute history, as one individual per recorded change.

Every recorded change becomes a change record holding the new value and its
timestamp. When exactly one event of the object occurred at that timestamp,
the record links to it through `caused_by`. Without the history, an object
carries only the latest value of each attribute.

This runs only for a store loaded with the attribute history extension.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from semlog.core import vocabulary as vocab
from semlog.core.enums import IndividualKind
from semlog.data.readable import ReadableStore
from semlog.data.tables import EntityKind, RelationKind
from semlog.onto.annotations import AnnotationWriter
from semlog.onto.emitted import EmittedIndividuals
from semlog.onto.emitters.attributes import AttributeEmitter, attribute_source
from semlog.onto.emitters.base import Emitter
from semlog.onto.emitters.time import TimeEmitter
from semlog.onto.settings import OntoBuildSettings
from semlog.onto.sinks import AxiomSink

logger = logging.getLogger(__name__)


def changed_attributes(data: ReadableStore) -> frozenset[str]:
    """Return the attributes that have at least one recorded change with a value."""
    changed = set()
    for obj_type in data.object_change_types():
        changes = data.object_changes(obj_type)
        for field, rows in changes.groupby("ocel:field"):
            name = str(field)
            if name in rows.columns and rows[name].notna().any():
                changed.add(name)
    return frozenset(changed)


@dataclass
class _CauseCounts:
    """How many change records a build emitted, and how many got no cause."""

    records: int = 0
    no_event: int = 0
    several: int = 0


def _utc(stamps: pd.Series) -> pd.Series:
    """Return timestamps as UTC nanosecond timestamps, a naive one read as UTC."""
    return pd.to_datetime(stamps, utc=True, errors="coerce").astype(
        "datetime64[ns, UTC]"
    )


def _events_by_object_and_time(data: ReadableStore) -> pd.DataFrame:
    """Return the events of each object at each timestamp.

    An event linked to one object by several relations counts once.

    Returns:
        One row per object and UTC timestamp at which at least one event of
        the object occurred, with the columns `object`, `time`, `event`, one
        of those events, and `events`, how many there are.
    """
    times = [
        pd.DataFrame(
            {
                "event": table.data[table.id].to_numpy(),
                "time": _utc(table.data[table.time]).reset_index(drop=True),
            }
        )
        for table in data.tables_of(EntityKind.EVENT).values()
        if table.time is not None
    ]
    links = [
        pd.DataFrame(
            {
                "event": relation.data[relation.from_id].to_numpy(),
                "object": relation.data[relation.to_id].to_numpy(),
            }
        )
        for relation in data.relations_of(RelationKind.E2O).values()
    ]
    if not times or not links:
        return pd.DataFrame(
            {
                "object": pd.Series(dtype=object),
                "time": pd.Series(dtype="datetime64[ns, UTC]"),
                "event": pd.Series(dtype=object),
                "events": pd.Series(dtype="int64"),
            }
        )
    pairs = pd.concat(links, ignore_index=True).dropna().drop_duplicates()
    stamped = pairs.merge(
        pd.concat(times, ignore_index=True).dropna(), on="event", how="inner"
    )
    return (
        stamped.groupby(["object", "time"], sort=False)
        .agg(event=("event", "first"), events=("event", "size"))
        .reset_index()
    )


class HistoryEmitter(Emitter):
    """Emits the change records of object attributes.

    It calls the attribute and time emitters, which skip values and
    timestamps written by the main pass.
    """

    def __init__(
        self,
        sink: AxiomSink,
        settings: OntoBuildSettings,
        annotations: AnnotationWriter,
        emitted: EmittedIndividuals,
        attributes: AttributeEmitter,
        times: TimeEmitter,
    ) -> None:
        super().__init__(sink, settings, annotations, emitted)
        self.attributes = attributes
        self.times = times
        self._change_classes: dict[str, None] = {}

    def emit_disjointness(self) -> None:
        """State that the change classes of different attributes share no record."""
        self.sink.add_disjoint_classes(
            self._change_classes, source="the attribute change classes"
        )

    def emit(self, data: ReadableStore) -> None:
        """Emit a change record for every recorded attribute change.

        Each record links the object to the new value and to the timestamp of
        the change. Values go through `AttributeEmitter`, so they take the
        attribute representations of the build and a number is binned the
        same way as on the object.
        """
        causes = _events_by_object_and_time(data)
        counts = _CauseCounts()
        for obj_type in data.object_change_types():
            changes_df = data.object_changes(obj_type)
            for attr_name, grp in changes_df.groupby("ocel:field"):
                self._emit_attribute(obj_type, str(attr_name), grp, causes, counts)
        if counts.records:
            logger.info(
                "  Linked %d of %d attribute change record(s) to the event that "
                "caused them. %d had no event of their object at the time of "
                "the change, %d had several.",
                counts.records - counts.no_event - counts.several,
                counts.records,
                counts.no_event,
                counts.several,
            )

    def _emit_attribute(
        self,
        obj_type: str,
        attr_name: str,
        grp: pd.DataFrame,
        causes: pd.DataFrame,
        counts: _CauseCounts,
    ) -> None:
        """Emit the recorded changes of one attribute of one object type.

        Args:
            obj_type: Object type the changes belong to.
            attr_name: Attribute that changed, as `ocel:field` names it.
            grp: The change rows naming that attribute.
            causes: The events of each object at each timestamp, as
                `_events_by_object_and_time` returns them.
            counts: Tally of the records left without a cause, updated in
                place.
        """
        if attr_name not in grp.columns:
            logger.warning(
                "ocel:field value '%s' has no "
                "matching column in object_changes for type '%s'. "
                "The attribute is skipped.",
                attr_name,
                obj_type,
            )
            return

        changes = grp[["ocel:oid", "ocel:timestamp", attr_name]].copy()
        changes = changes[changes[attr_name].notna()].copy()
        if changes.empty:
            return

        attr_change_cls, change_prop = self._emit_schema(attr_name)

        changes["time_str"] = vocab.timestamp_ids(changes["ocel:timestamp"])
        self._warn_on_repeated_changes(changes, attr_name, obj_type)
        changes["attr_change_id"] = vocab.attribute_change_ids(
            changes["ocel:oid"], attr_name, changes["time_str"]
        )
        self.sink.assert_individuals(
            attr_change_cls, changes["attr_change_id"], IndividualKind.CHANGE
        )
        self.annotations.attribute_change_individuals(
            changes["attr_change_id"],
            attr_name,
            obj_type,
            changes["ocel:oid"],
            changes["ocel:timestamp"],
        )
        self.assert_relations(
            vocab.HAS_ATTRIBUTE_CHANGE,
            changes,
            "ocel:oid",
            "attr_change_id",
            (IndividualKind.OBJECT, IndividualKind.CHANGE),
        )
        self.attributes.emit_column_values(
            changes[["attr_change_id", attr_name]],
            "attr_change_id",
            attr_name,
            change_prop,
            IndividualKind.CHANGE,
        )
        self.times.emit_individuals(changes["time_str"], changes["ocel:timestamp"])
        self.assert_relations(
            vocab.CHANGED_AT,
            changes,
            "attr_change_id",
            "time_str",
            (IndividualKind.CHANGE, IndividualKind.TIME),
        )
        self._emit_causes(changes, causes, counts)

    def _emit_causes(
        self, changes: pd.DataFrame, causes: pd.DataFrame, counts: _CauseCounts
    ) -> None:
        """Link each change record to the one event of its object at its timestamp.

        A record whose object has no event or several events at that
        timestamp gets no link and is counted.
        """
        keys = pd.DataFrame(
            {
                "object": changes["ocel:oid"].to_numpy(),
                "time": _utc(changes["ocel:timestamp"]).reset_index(drop=True),
            }
        )
        matched = keys.merge(causes, how="left", on=["object", "time"])
        events = matched["events"].fillna(0).to_numpy()
        linked = pd.DataFrame(
            {
                "attr_change_id": changes["attr_change_id"].to_numpy(),
                "event": matched["event"].to_numpy(),
            }
        )[events == 1]
        counts.records += len(changes)
        counts.no_event += int((events == 0).sum())
        counts.several += int((events > 1).sum())
        self.assert_relations(
            vocab.CAUSED_BY,
            linked,
            "attr_change_id",
            "event",
            (IndividualKind.CHANGE, IndividualKind.EVENT),
        )

    def _emit_schema(self, attr_name: str) -> tuple[str, str]:
        """Declare the change class of one attribute and what its records carry.

        The value property is declared in every mode. The attribute class and
        the change property, which reaches the value individual, are declared
        under `individuals` only.

        Returns:
            The change class and the property reaching the new value.
        """
        source = attribute_source(attr_name)
        attr_change_cls = vocab.attribute_change_class(attr_name)
        self._change_classes[attr_change_cls] = None
        change_source = f"the attribute history of {source}"
        self.sink.add_class(
            attr_change_cls, vocab.ATTRIBUTE_CHANGE, source=change_source
        )
        self.attributes.declare_value_property(attr_name)
        change_prop = vocab.change_value_property(attr_name)
        if not self.settings.emits_value_individuals:
            self.annotations.attribute_change_schema(attr_name, attr_change_cls)
            return attr_change_cls, change_prop
        self.attributes.add_attribute_class(attr_name)
        self.add_object_property(
            change_prop,
            superprop=vocab.ATTRIBUTE_CHANGE_VALUE,
            range_class=attr_name,
            source=change_source,
        )
        self.declare_functional(change_prop)
        self.annotations.attribute_change_schema(
            attr_name, attr_change_cls, change_prop
        )
        return attr_change_cls, change_prop

    @staticmethod
    def _warn_on_repeated_changes(
        changes: pd.DataFrame, attr_name: str, obj_type: str
    ) -> None:
        """Warn about objects with two changes of one attribute at one timestamp.

        `attribute_change_ids` numbers those records.
        """
        repeated = changes.loc[
            changes.duplicated(["ocel:oid", "time_str"], keep=False), "ocel:oid"
        ].unique()
        if len(repeated):
            logger.warning(
                "%s %r object(s) record more than one "
                "change of '%s' at the same instant. Their change "
                "records are numbered (sample: "
                "%s).",
                len(repeated),
                obj_type,
                attr_name,
                ", ".join(sorted(map(str, repeated))[:5]),
            )
