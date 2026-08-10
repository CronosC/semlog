"""ABox construction for OCEL 2.0 object attribute change history."""

from __future__ import annotations

import logging

import pandas as pd

from semanticlog.data.oced import OCEDStore
from semanticlog.onto import vocabulary as vocab
from semanticlog.onto.annotations import AnnotationWriter, timestamp_labels
from semanticlog.onto.attributes import AttributeEmitter
from semanticlog.onto.emit import Emitter
from semanticlog.onto.settings import DataRepresentation, OntoBuildSettings
from semanticlog.onto.sink import AxiomSink


class HistoryEmitter(Emitter):
    """Turns ``log.object_changes`` records into timestamped AttributeChange individuals.

    Active only when ``OcelExtension.ATTRIBUTE_HISTORY`` is enabled.
    """

    def __init__(
        self,
        sink: AxiomSink,
        settings: OntoBuildSettings,
        annotations: AnnotationWriter,
        attributes: AttributeEmitter | None = None,
    ) -> None:
        super().__init__(sink, settings, annotations)
        self.attributes = attributes or AttributeEmitter(sink, settings, annotations)

    def emit(self, data: OCEDStore) -> None:
        """Instantiate object attribute change history as AttributeChange individuals.

        For each change record ``(ocel:oid, ocel:field, value, ocel:timestamp)``,
        creates one ``AttributeChange`` individual and asserts:

        - ``has_attribute_change``: object → AttributeChange
        - ``attribute_change_value``: AttributeChange → attribute-value individual
        - ``changed_at``: AttributeChange → Time individual

        Attribute-value individuals are linked via a per-attribute subproperty of
        ``attribute_change_value`` (e.g., ``change_value_price``).  Datetime attributes
        and numeric attributes with ``DataRepresentation.VALUES`` also get a
        ``has_data_value`` literal.  Binning modes (``EQUAL_BINS``,
        ``DIST_BINS``) are intentionally ignored for changes — only the exact
        numeric value is recorded.

        Args:
            data: OCEDStore created with ``ocel_mode=True`` that contains object
                change history loaded via ``add_object_changes``.
        """
        self.annotations.bind(data.backmap)
        has_inverses = self.settings.has_inverses
        for obj_type in data.get_object_change_types():
            changes_df = data.get_object_changes(obj_type)

            for attr_name, grp in changes_df.groupby("ocel:field"):
                attr_name = str(attr_name)
                if attr_name not in grp.columns:
                    logging.warning(
                        f"ATTRIBUTE_HISTORY: ocel:field value '{attr_name}' has no "
                        f"matching column in object_changes for type '{obj_type}' — "
                        "skipping. This indicates unexpected object_changes structure."
                    )
                    continue

                self.sink.add_class(attr_name, vocab.OBJECT_ATTRIBUTE)
                attr_change_cls = vocab.attribute_change_class(attr_name)
                self.sink.add_class(attr_change_cls, vocab.ATTRIBUTE_CHANGE)
                change_prop = vocab.change_value_property(attr_name)
                self.sink.add_property(
                    change_prop,
                    superprop=vocab.ATTRIBUTE_CHANGE_VALUE,
                    range_class=attr_name,
                    create_inverse=has_inverses,
                )
                self.annotations.attribute_change_schema(
                    attr_name, attr_change_cls, change_prop
                )

                tmp = grp[["ocel:oid", "ocel:timestamp", attr_name]].copy()
                tmp = tmp[tmp[attr_name].notna()].copy()
                if tmp.empty:
                    continue

                s = tmp[attr_name]
                if pd.api.types.is_numeric_dtype(s):
                    if (
                        DataRepresentation.VALUES
                        not in self.settings.data_representation
                    ):
                        continue

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
                # Only datetime and numeric changes carry a literal alongside the
                # value individual; strings are represented by the individual alone.
                self.attributes.emit_values(
                    tmp,
                    "attr_change_id",
                    attr_name,
                    change_prop,
                    pd.api.types.is_datetime64_any_dtype(s)
                    or pd.api.types.is_numeric_dtype(s),
                )
                self.sink.assert_individuals(
                    tmp["time_id"].unique().tolist(), vocab.TIME
                )
                unique_change_times = tmp[
                    ["time_id", "ocel:timestamp"]
                ].drop_duplicates("time_id")
                self.annotations.time_individuals(
                    unique_change_times["time_id"],
                    timestamp_labels(unique_change_times["ocel:timestamp"]),
                )
                self.sink.assert_relations(
                    unique_change_times,
                    vocab.HAS_DATA_VALUE,
                    from_key="time_id",
                    to_key="ocel:timestamp",
                    data_prop=True,
                )
                self.sink.assert_relations(
                    tmp,
                    vocab.CHANGED_AT,
                    from_key="attr_change_id",
                    to_key="time_id",
                )
