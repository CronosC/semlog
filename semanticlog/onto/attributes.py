"""ABox construction for event and object attribute values."""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from semanticlog.data.binning import bin_series, equal_bin_series
from semanticlog.data.oced import OCEDStore
from semanticlog.onto import vocabulary as vocab
from semanticlog.onto.emit import Emitter
from semanticlog.onto.settings import DataRepresentation


def attribute_owners(data: OCEDStore) -> dict[str, list[str]]:
    """Map each attribute column name to the entity types that carry it.

    An attribute class is named after the column alone, so a column appearing on
    two types produces one class shared between them. Knowing the full set of
    owners up front is what lets the domain be written as their union: emitting a
    domain per table instead would be read as an intersection, and an attribute
    on both ``order`` and ``item`` would end up with an unsatisfiable domain.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for activity in data.get_activities():
        table = data.get_events(activity)
        for column in set(table.data.columns) - {table.id, table.time}:
            owners[column].add(activity)
    for obj_type in data.get_object_types():
        obj = data.get_object(obj_type)
        for column in set(obj.data.columns) - {obj.id}:
            owners[column].add(obj_type)
    return {attribute: sorted(types) for attribute, types in owners.items()}


def report_conflated_attributes(owners: dict[str, list[str]]) -> dict[str, list[str]]:
    """Log the attributes that more than one entity type shares, and return them.

    Sharing a column name merges two attributes into one class with one set of
    value individuals, which is rarely what the source data meant. Renaming them
    would change IRIs, so this reports rather than acts: it makes the scale of the
    conflation visible on real logs before anyone decides what to do about it.
    """
    conflated = {a: types for a, types in owners.items() if len(types) > 1}
    if conflated:
        detail = "; ".join(
            f"{attribute!r} on {', '.join(types)}"
            for attribute, types in sorted(conflated.items())
        )
        logging.warning(
            f"ATTRIBUTE CONFLATION: {len(conflated)} attribute name(s) are shared by "
            f"more than one entity type and become a single OWL class with shared "
            f"value individuals: {detail}"
        )
    return conflated


class AttributeEmitter(Emitter):
    """Turns attribute columns into OWL classes, properties and value individuals."""

    _owners: dict[str, list[str]] = {}

    def bind_owners(self, owners: dict[str, list[str]]) -> None:
        """Point the emitter at which entity types carry each attribute.

        Without this the ``has_{attr}`` properties get no domain, since a single
        table cannot know whether other tables share the column.
        """
        self._owners = owners

    def emit(
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
            attribute_parent: Parent class name (``"EventAttribute"`` or
                ``"ObjectAttribute"``).
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

            prop = vocab.has_attribute_property(attr)
            self.sink.add_class(attr, attribute_parent)
            owners = self._owners.get(attr, [])
            self.sink.add_property(
                prop,
                domain_class=" or ".join(owners) if owners else None,
                range_class=attr,
                superprop=vocab.HAS_ATTRIBUTE,
                create_inverse=self.settings.has_inverses,
            )
            owner = "event" if attribute_parent == vocab.EVENT_ATTRIBUTE else "object"
            self.annotations.attribute(attr, prop, owner)

            if pd.api.types.is_datetime64_any_dtype(s):
                self.emit_values(df_attr, key_col, attr, prop, True)
            elif pd.api.types.is_numeric_dtype(s):
                self._emit_numeric(df_attr, key_col, attr, prop, dr)
            else:
                self.emit_values(df_attr, key_col, attr, prop, False)

    def _emit_numeric(
        self,
        df_attr: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        dr: list[DataRepresentation],
    ) -> None:
        s = df_attr[attr]
        if DataRepresentation.VALUES in dr:
            self.emit_values(df_attr, key_col, attr, prop, True)
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
            bin_cls = vocab.bin_class(attr, bin_label)
            self.sink.add_class(bin_cls, attr)
            if DataRepresentation.VALUES in dr:
                value_inds = [
                    vocab.attribute_value(attr, v) for v in bin_group[attr].unique()
                ]
                self.sink.assert_individuals(value_inds, bin_cls)
            else:
                self.sink.assert_individuals(bin_group[key_col].tolist(), bin_cls)
        if not out_of_bin.empty:
            logging.warning(
                f"BINNING: {len(out_of_bin)} value(s) for attribute "
                f"'{attr}' fall outside all bin boundaries; using raw "
                "numeric representation as fallback."
            )
            self.emit_values(out_of_bin, key_col, attr, prop, True)

    def emit_values(
        self,
        df: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        also_assert_data_value: bool,
    ) -> None:
        """Create one individual per distinct value and link the owners to it.

        Args:
            df: Rows holding the owner key and the attribute column.
            key_col: Column naming the owning event or object individual.
            attr: Attribute name, used as both the class and the name prefix.
            prop: Property linking owner to value individual.
            also_assert_data_value: Also attach the raw value as a literal, which
                is wanted for numeric and datetime attributes but not for strings.
        """
        tmp = df.copy()

        # map each distinct value to an attribute-individual IRI/name
        vals = tmp[attr].unique()
        mapping = {v: vocab.attribute_value(attr, v) for v in vals}
        tmp["attr_ind"] = tmp[attr].map(mapping)

        # create individuals for the attribute values (unique only — one per distinct value)
        self.sink.assert_individuals(tmp["attr_ind"].unique().tolist(), attr)

        self.annotations.attribute_values(mapping)

        # connect objects/events (key_col) to attribute-individuals via the property
        self.sink.assert_relations(tmp, prop=prop, from_key=key_col, to_key="attr_ind")

        # for numeric/datetime, also attach the literal value
        if also_assert_data_value:
            self.sink.assert_relations(
                tmp,
                prop=vocab.HAS_DATA_VALUE,
                from_key="attr_ind",
                to_key=attr,
                data_prop=True,
            )
