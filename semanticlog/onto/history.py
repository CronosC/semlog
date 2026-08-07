"""ABox construction for OCEL 2.0 object attribute change history."""

from __future__ import annotations

import logging

import pandas as pd

from semanticlog.data.oced import OCEDStore
from semanticlog.onto.annotations import timestamp_labels
from semanticlog.onto.attributes import AttributeInstantiationMixin
from semanticlog.onto.settings import DataRepresentation


class AttributeHistoryMixin(AttributeInstantiationMixin):
    """Turns ``log.object_changes`` records into timestamped AttributeChange individuals.\n\n    Mixed into :class:`~semanticlog.onto.instantiate.OCEDOntoBuilder`; active only\n    when ``OcelExtension.ATTRIBUTE_HISTORY`` is enabled."""

    def instantiate_object_attribute_history(self, data: OCEDStore) -> None:
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

                self.add_class(attr_name, "ObjectAttribute")
                attr_change_cls = attr_name + "Change"
                self.add_class(attr_change_cls, "AttributeChange")
                change_prop = self.generate_attr_change_value_prop_name(attr_name)
                self.add_property(
                    change_prop,
                    superprop="attribute_change_value",
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

                tmp["ts_str"] = tmp["ocel:timestamp"].apply(
                    self.generate_timestamp_string
                )
                tmp["attr_change_id"] = (
                    "AttrChange_"
                    + tmp["ocel:oid"].astype(str)
                    + "_"
                    + attr_name
                    + "_"
                    + tmp["ts_str"]
                )
                tmp["time_id"] = tmp["ts_str"]

                self.assert_individuals(tmp["attr_change_id"].tolist(), attr_change_cls)
                self.annotations.attribute_change_individuals(
                    tmp["attr_change_id"],
                    attr_name,
                    obj_type,
                    tmp["ocel:oid"],
                    tmp["ocel:timestamp"],
                )
                self.assert_relations(
                    tmp,
                    "has_attribute_change",
                    from_key="ocel:oid",
                    to_key="attr_change_id",
                )
                s = tmp[attr_name]
                if pd.api.types.is_datetime64_any_dtype(s):
                    self._instantiate_attr(
                        tmp,
                        "attr_change_id",
                        attr_name,
                        change_prop,
                        lambda v, a=attr_name: (
                            f"{a}_{self.generate_timestamp_string(v)}"
                        ),
                        also_assert_data_value=True,
                    )
                elif pd.api.types.is_numeric_dtype(s):
                    self._instantiate_attr(
                        tmp,
                        "attr_change_id",
                        attr_name,
                        change_prop,
                        lambda v, a=attr_name: f"{a}_{v}",
                        also_assert_data_value=True,
                    )
                else:
                    self._instantiate_attr(
                        tmp,
                        "attr_change_id",
                        attr_name,
                        change_prop,
                        lambda v, a=attr_name: f"{a}_{str(v)}",
                        also_assert_data_value=False,
                    )
                self.assert_individuals(tmp["time_id"].unique().tolist(), "Time")
                unique_change_times = tmp[
                    ["time_id", "ocel:timestamp"]
                ].drop_duplicates("time_id")
                self.annotations.time_individuals(
                    unique_change_times["time_id"],
                    timestamp_labels(unique_change_times["ocel:timestamp"]),
                )
                self.assert_relations(
                    unique_change_times,
                    "has_data_value",
                    from_key="time_id",
                    to_key="ocel:timestamp",
                    data_prop=True,
                )
                self.assert_relations(
                    tmp, "changed_at", from_key="attr_change_id", to_key="time_id"
                )
