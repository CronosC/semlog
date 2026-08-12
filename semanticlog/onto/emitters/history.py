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
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitters.attributes import AttributeEmitter, attribute_source
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.emitters.time import TimeEmitter
from semanticlog.onto.settings import DataRepresentation, OntoBuildSettings
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
        attributes: AttributeEmitter,
        times: TimeEmitter,
    ) -> None:
        super().__init__(sink, settings, annotations)
        self.attributes = attributes
        self.times = times

    def emit(self, data: OCEDStore) -> None:
        """Emit a change record for every recorded attribute change.

        Each record links the object to the new value and to the moment of
        the change. Numeric attributes are skipped when the build does not
        keep numeric values, matching how the main attribute emitter treats
        them.
        """
        self.annotations.bind(data.backmap)
        has_inverses = self.settings.has_inverses
        for obj_type in data.get_object_change_types():
            changes_df = data.get_object_changes(obj_type)

            for attr_name, grp in changes_df.groupby("ocel:field"):
                attr_name = str(attr_name)
                if attr_name not in grp.columns:
                    logger.warning(
                        f"ATTRIBUTE_HISTORY: ocel:field value '{attr_name}' has no "
                        f"matching column in object_changes for type '{obj_type}', "
                        "so it is skipped. The object_changes structure is unexpected."
                    )
                    continue

                tmp = grp[["ocel:oid", "ocel:timestamp", attr_name]].copy()
                tmp = tmp[tmp[attr_name].notna()].copy()
                if tmp.empty:
                    continue

                s = tmp[attr_name]
                if (
                    pd.api.types.is_numeric_dtype(s)
                    and DataRepresentation.VALUES
                    not in self.settings.data_representations
                ):
                    continue

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
                    create_inverse=has_inverses,
                    source=change_source,
                )
                self.annotations.attribute_change_schema(
                    attr_name, attr_change_cls, change_prop
                )

                tmp["ts_str"] = vocab.timestamp_ids(tmp["ocel:timestamp"])
                tmp["attr_change_id"] = vocab.attribute_change_ids(
                    tmp["ocel:oid"], attr_name, tmp["ts_str"]
                )
                tmp["time_id"] = tmp["ts_str"]

                self.sink.assert_individuals(
                    tmp["attr_change_id"].tolist(), attr_change_cls
                )
                self.annotations.attribute_change_individuals(
                    tmp["attr_change_id"],
                    attr_name,
                    obj_type,
                    tmp["ocel:oid"],
                    tmp["ocel:timestamp"],
                )
                self.sink.assert_relations(
                    tmp,
                    vocab.HAS_ATTRIBUTE_CHANGE,
                    from_key="ocel:oid",
                    to_key="attr_change_id",
                )
                self.attributes.emit_values(
                    tmp,
                    "attr_change_id",
                    attr_name,
                    change_prop,
                    pd.api.types.is_datetime64_any_dtype(s)
                    or pd.api.types.is_numeric_dtype(s),
                )
                self.times.emit_individuals(tmp["time_id"], tmp["ocel:timestamp"])
                self.sink.assert_relations(
                    tmp,
                    vocab.CHANGED_AT,
                    from_key="attr_change_id",
                    to_key="time_id",
                )
