"""ABox construction for event and object attribute values."""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from semanticlog.data.binning import bin_series, equal_bin_series
from semanticlog.onto.builder import OCEDBaseBuilder
from semanticlog.onto.settings import DataRepresentation


class AttributeInstantiationMixin(OCEDBaseBuilder):
    """Turns attribute columns into OWL classes, properties and value individuals.\n\n    Mixed into :class:`~semanticlog.onto.instantiate.OCEDOntoBuilder`; the\n    ``OCEDBaseBuilder`` base is declared so that the axiom-emitting helpers\n    this mixin calls are visible to the type checker."""

    def instantiate_attributes(
        self,
        df: pd.DataFrame,
        key_col: str,
        attribute_parent: str,
    ) -> None:
        """Instantiate attribute values for a single event or object type.

        For each non-key column in ``df``: creates an attribute class under
        ``attribute_parent``, adds a ``has_attribute`` sub-property, and asserts
        attribute-value individuals using the configured ``data_representation``
        strategy (values, equal-frequency bins, distribution-fitted bins).

        Args:
            df: Event or object DataFrame with the key column and attribute columns.
            key_col: Column name of the individual's IRI local name.
            attribute_parent: Parent class name (``"EventAttribute"`` or ``"ObjectAttribute"``).
        """
        dr = self.settings.data_representation
        for attr in (c for c in df.columns if c != key_col):
            mask = df[attr].notna()
            if not mask.any():
                continue
            df_attr = df.loc[mask, [key_col, attr]]
            s = df_attr[attr]

            if pd.api.types.is_numeric_dtype(s) and not dr:
                continue

            prop = self.generate_specialized_has_attribute_name(attr)
            self.add_class(attr, attribute_parent)
            self.add_property(
                prop,
                superprop="has_attribute",
                create_inverse=self.settings.has_inverses,
            )
            owner = "event" if attribute_parent == "EventAttribute" else "object"
            self.annotations.attribute(attr, prop, owner)

            if pd.api.types.is_datetime64_any_dtype(s):
                self._instantiate_datetime_attr(df_attr, key_col, attr, prop)
            elif pd.api.types.is_numeric_dtype(s):
                self._instantiate_numeric_attr(df_attr, key_col, attr, prop, dr)
            else:
                self._instantiate_string_attr(df_attr, key_col, attr, prop)

    def _instantiate_datetime_attr(
        self,
        df_attr: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
    ) -> None:
        self._instantiate_attr(
            df_attr,
            key_col,
            attr,
            prop,
            value_to_id=lambda v, a=attr: f"{a}_{self.generate_timestamp_string(v)}",
            also_assert_data_value=True,
        )

    def _instantiate_numeric_attr(
        self,
        df_attr: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        dr: list[DataRepresentation],
    ) -> None:
        s = df_attr[attr]
        if DataRepresentation.VALUES in dr:
            self._instantiate_attr(
                df_attr,
                key_col,
                attr,
                prop,
                value_to_id=lambda v, a=attr: f"{a}_{v}",
                also_assert_data_value=True,
            )
        quant = next(
            (
                x
                for x in dr
                if x
                in (
                    DataRepresentation.EQUAL_BINS,
                    DataRepresentation.DIST_BINS,
                )
            ),
            None,
        )
        if quant is None:
            return
        if quant == DataRepresentation.EQUAL_BINS:
            binned = equal_bin_series(
                s,
                bins=self.settings.bins,
                labels=self.settings.bin_labels,
            )
        else:
            binned = bin_series(
                s,
                bins=self.settings.bins,
                labels=self.settings.bin_labels,
            )
        df_merge = df_attr.copy()
        df_merge["_bin"] = binned
        out_of_bin = df_attr.loc[df_merge["_bin"].isna()]
        df_q = df_merge[df_merge["_bin"].notna()]
        for bin_label, bin_group in df_q.groupby("_bin", observed=True):
            bin_class = f"{attr}_{str(bin_label)}"
            self.add_class(bin_class, attr)
            if DataRepresentation.VALUES in dr:
                value_inds = [f"{attr}_{v}" for v in bin_group[attr].unique()]
                self.assert_individuals(value_inds, bin_class)
            else:
                self.assert_individuals(bin_group[key_col].tolist(), bin_class)
        if not out_of_bin.empty:
            logging.warning(
                f"BINNING: {len(out_of_bin)} value(s) for attribute "
                f"'{attr}' fall outside all bin boundaries; using raw "
                "numeric representation as fallback."
            )
            self._instantiate_attr(
                out_of_bin,
                key_col,
                attr,
                prop,
                value_to_id=lambda v, a=attr: f"{a}_{v}",
                also_assert_data_value=True,
            )

    def _instantiate_string_attr(
        self,
        df_attr: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
    ) -> None:
        self._instantiate_attr(
            df_attr,
            key_col,
            attr,
            prop,
            value_to_id=lambda v, a=attr: f"{a}_{str(v)}",
            also_assert_data_value=False,
        )

    def _instantiate_attr(
        self,
        df: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        value_to_id: Callable[..., str],
        also_assert_data_value: bool,
    ) -> None:
        tmp = df.copy()

        # map each distinct value to an attribute-individual IRI/name
        vals = tmp[attr].unique()
        mapping = {v: value_to_id(v) for v in vals}
        tmp["attr_ind"] = tmp[attr].map(mapping)

        # create individuals for the attribute values (unique only — one per distinct value)
        self.assert_individuals(tmp["attr_ind"].unique().tolist(), attr)

        self.annotations.attribute_values(mapping)

        # connect objects/events (key_col) to attribute-individuals via the property
        self.assert_relations(tmp, prop=prop, from_key=key_col, to_key="attr_ind")

        # for numeric/datetime, also attach the literal value
        if also_assert_data_value:
            self.assert_relations(
                tmp,
                prop="has_data_value",
                from_key="attr_ind",
                to_key=attr,
                data_prop=True,
            )
