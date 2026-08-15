"""Object attribute history, as one individual per recorded change.

Without this, an object carries only the latest value of each attribute. With
it, every change in the log becomes a change record that holds the new value
and the moment it took effect, so an object's history can be queried too.

This runs only for a store loaded with the attribute history extension.
"""

from __future__ import annotations

import logging

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.data.readable import ReadableStore
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.attributes import AttributeEmitter, attribute_source
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.emitters.time import TimeEmitter
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink

logger = logging.getLogger(__name__)


class HistoryEmitter(Emitter):
    """Emits the change records of object attributes.

    It reuses the attribute and time emitters, so a value or a moment already
    written by the main build is not written a second time.
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

    def emit(self, data: ReadableStore) -> None:
        """Emit a change record for every recorded attribute change.

        Each record links the object to the new value and to the moment of
        the change. Values reach the ontology through the same emitter the
        main pass uses, so a number is binned or kept exactly as it is on the
        object itself.
        """
        for obj_type in data.get_object_change_types():
            changes_df = data.get_object_changes(obj_type)
            for attr_name, grp in changes_df.groupby("ocel:field"):
                self._emit_attribute(obj_type, str(attr_name), grp)

    def _emit_attribute(self, obj_type: str, attr_name: str, grp: pd.DataFrame) -> None:
        """Emit the recorded changes of one attribute of one object type.

        Args:
            obj_type: Object type the changes belong to.
            attr_name: Attribute that changed, as `ocel:field` names it.
            grp: The change rows naming that attribute.
        """
        if attr_name not in grp.columns:
            logger.warning(
                f"ATTRIBUTE_HISTORY: ocel:field value '{attr_name}' has no "
                f"matching column in object_changes for type '{obj_type}', "
                "so it is skipped. The object_changes structure is unexpected."
            )
            return

        tmp = grp[["ocel:oid", "ocel:timestamp", attr_name]].copy()
        tmp = tmp[tmp[attr_name].notna()].copy()
        if tmp.empty:
            return
        if pd.api.types.is_numeric_dtype(tmp[attr_name]) and not (
            self.settings.data_representations
        ):
            return

        attr_change_cls, change_prop = self._emit_schema(attr_name)

        tmp["ts_str"] = vocab.timestamp_ids(tmp["ocel:timestamp"])
        self._warn_on_repeated_changes(tmp, attr_name, obj_type)
        tmp["attr_change_id"] = vocab.attribute_change_ids(
            tmp["ocel:oid"], attr_name, tmp["ts_str"]
        )
        self.sink.assert_individuals(attr_change_cls, tmp["attr_change_id"])
        self.annotations.attribute_change_individuals(
            tmp["attr_change_id"],
            attr_name,
            obj_type,
            tmp["ocel:oid"],
            tmp["ocel:timestamp"],
        )
        self.assert_relations(
            vocab.HAS_ATTRIBUTE_CHANGE, tmp, "ocel:oid", "attr_change_id"
        )
        self.attributes.emit_column_values(
            tmp[["attr_change_id", attr_name]],
            "attr_change_id",
            attr_name,
            change_prop,
        )
        self.times.emit_individuals(tmp["ts_str"], tmp["ocel:timestamp"])
        self.assert_relations(vocab.CHANGED_AT, tmp, "attr_change_id", "ts_str")

    def _emit_schema(self, attr_name: str) -> tuple[str, str]:
        """Declare the change class and change property of one attribute.

        Returns:
            The change class and the property reaching the new value.
        """
        source = attribute_source(attr_name)
        self.sink.add_class(attr_name, vocab.OBJECT_ATTRIBUTE, source=source)
        attr_change_cls = vocab.attribute_change_class(attr_name)
        change_source = f"the attribute history of {source}"
        self.sink.add_class(
            attr_change_cls, vocab.ATTRIBUTE_CHANGE, source=change_source
        )
        change_prop = vocab.change_value_property(attr_name)
        self.sink.add_property(
            change_prop,
            superprop=vocab.ATTRIBUTE_CHANGE_VALUE,
            range_class=attr_name,
            create_inverse=self.settings.has_inverses,
            axiomatic_inverse=self.settings.axiomatic_inverses,
            source=change_source,
        )
        self.annotations.attribute_change_schema(
            attr_name, attr_change_cls, change_prop
        )
        return attr_change_cls, change_prop

    @staticmethod
    def _warn_on_repeated_changes(
        tmp: pd.DataFrame, attr_name: str, obj_type: str
    ) -> None:
        """Report objects whose log records two changes at one instant.

        An object, an attribute and a moment name one change record, so these
        rows are the ones `attribute_change_ids` has to number apart.
        """
        repeated = tmp.loc[
            tmp.duplicated(["ocel:oid", "ts_str"], keep=False), "ocel:oid"
        ].unique()
        if len(repeated):
            logger.warning(
                f"ATTRIBUTE_HISTORY: {len(repeated)} {obj_type!r} object(s) record "
                f"more than one change of '{attr_name}' at the same instant, so "
                "those change records are numbered to keep them apart (sample: "
                f"{', '.join(sorted(map(str, repeated))[:5])})."
            )
