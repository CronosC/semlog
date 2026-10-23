"""Object attribute history, as one individual per recorded change.

Every recorded change becomes a change record holding the new value and its
timestamp. Without it, an object carries only the latest value of each
attribute.

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
                "ocel:field value '%s' has no "
                "matching column in object_changes for type '%s'. "
                "The attribute is skipped.",
                attr_name,
                obj_type,
            )
            return

        tmp = grp[["ocel:oid", "ocel:timestamp", attr_name]].copy()
        tmp = tmp[tmp[attr_name].notna()].copy()
        if tmp.empty:
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
        self.attributes.declare_value_property(attr_name, vocab.OBJECT_ATTRIBUTE)
        change_prop = vocab.change_value_property(attr_name)
        if not self.settings.emits_value_individuals:
            self.annotations.attribute_change_schema(attr_name, attr_change_cls)
            return attr_change_cls, change_prop
        self.attributes.add_attribute_class(attr_name, vocab.OBJECT_ATTRIBUTE)
        self.sink.add_property(
            change_prop,
            superprop=vocab.ATTRIBUTE_CHANGE_VALUE,
            range_class=attr_name,
            create_inverse=self.settings.has_inverses,
            axiomatic_inverse=self.settings.axiomatic_inverses,
            source=change_source,
        )
        self.declare_functional(change_prop)
        self.annotations.attribute_change_schema(
            attr_name, attr_change_cls, change_prop
        )
        return attr_change_cls, change_prop

    @staticmethod
    def _warn_on_repeated_changes(
        tmp: pd.DataFrame, attr_name: str, obj_type: str
    ) -> None:
        """Warn about objects with two changes of one attribute at one timestamp.

        `attribute_change_ids` numbers those records.
        """
        repeated = tmp.loc[
            tmp.duplicated(["ocel:oid", "ts_str"], keep=False), "ocel:oid"
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
