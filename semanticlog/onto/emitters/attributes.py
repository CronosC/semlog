"""Attributes, as classes whose members are the values they took.

An attribute becomes a class, each distinct value of it becomes an individual
of that class, and the entity carrying the value is linked to that individual
by a `has_` property. Values are shared, so two events with the same value
point at one individual and a query can reach both from it.

Numeric attributes can also be reduced to bins, which become subclasses of the
attribute class.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.data.binning import dist_bin_series, equal_bin_series
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.settings import DataRepresentation

logger = logging.getLogger(__name__)


def attribute_owners(data: OCEDStore) -> dict[str, list[str]]:
    """Find which entity types carry each attribute.

    The owners become the domain of the `has_` property, so an attribute used
    by two activities gets both of them as its domain.

    Args:
        data: Store to scan.

    Returns:
        Attribute name to the sorted types that carry it.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for activity in data.get_event_types():
        table = data.get_event_table(activity)
        for column in set(table.data.columns) - {table.id, table.time}:
            owners[column].add(activity)
    for obj_type in data.get_object_types():
        obj = data.get_object_table(obj_type)
        for column in set(obj.data.columns) - {obj.id}:
            owners[column].add(obj_type)
    return {attribute: sorted(types) for attribute, types in owners.items()}


def attribute_source(attribute: str) -> str:
    """Return how an attribute is described in a name collision message."""
    return f"attribute {attribute!r}"


def report_conflated_attributes(owners: dict[str, list[str]]) -> dict[str, list[str]]:
    """Warn about attribute names shared by more than one entity type.

    One name becomes one class with one pool of value individuals, so a
    `weight` on events and a `weight` on objects end up meaning the same
    thing. Since that is sometimes intended, this warns instead of failing.

    Args:
        owners: Attribute name to the types that carry it.

    Returns:
        The entries with more than one owner.
    """
    conflated = {a: types for a, types in owners.items() if len(types) > 1}
    if conflated:
        detail = "; ".join(
            f"{attribute!r} on {', '.join(types)}"
            for attribute, types in sorted(conflated.items())
        )
        logger.warning(
            f"ATTRIBUTE CONFLATION: {len(conflated)} attribute name(s) are shared by "
            f"more than one entity type and become a single OWL class with shared "
            f"value individuals: {detail}"
        )
    return conflated


class AttributeEmitter(Emitter):
    """Emits attribute classes, their values, and the links to their owners.

    It remembers the values it has already emitted per attribute, since the
    same attribute reaches it once per entity type that carries it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._owners: dict[str, list[str]] = {}
        self._emitted_values: dict[str, set[str]] = {}

    def bind_owners(self, owners: dict[str, list[str]]) -> None:
        """Supply the owner map used as the domain of each `has_` property."""
        self._owners = owners

    @staticmethod
    def check_converging_values(data: OCEDStore) -> None:
        """Refuse attribute names that would produce the same value individual.

        A value individual is named `attribute_value`, and an underscore is
        legal in both halves, so `a` with value `b_c` and `a_b` with value `c`
        would both be `a_b_c`. Only attribute pairs that differ by an
        underscore can collide this way, so only those are compared.

        Args:
            data: Store to check.

        Raises:
            NameCollisionError: If two attributes have a value in common under
                this naming, quoting one example.
        """
        owners = attribute_owners(data)
        candidates = [
            (shorter, longer)
            for shorter in owners
            for longer in owners
            if shorter != longer and longer.startswith(f"{shorter}_")
        ]
        if not candidates:
            return

        values = {
            attribute: AttributeEmitter._distinct_values(data, attribute, types)
            for attribute, types in owners.items()
            if any(attribute in pair for pair in candidates)
        }
        for shorter, longer in candidates:
            shared = {
                vocab.attribute_value(shorter, v) for v in values.get(shorter, ())
            } & {vocab.attribute_value(longer, v) for v in values.get(longer, ())}
            if shared:
                example = sorted(shared)[0]
                raise vocab.NameCollisionError(
                    f"{example!r} would be emitted for a value of attribute "
                    f"{shorter!r} and for a value of attribute {longer!r}, which "
                    "makes one individual the value of both:\n"
                    f"  {shorter!r} and {longer!r} differ only by an underscore, "
                    "which is also legal inside a value.\n\n"
                    "Rename one of the two columns in the input data before building."
                )

    @staticmethod
    def _distinct_values(
        data: OCEDStore, attribute: str, types: list[str]
    ) -> set[object]:
        """Collect the distinct values an attribute takes across its owners."""
        found: set[object] = set()
        activities = set(data.get_event_types())
        for entity_type in types:
            if entity_type in activities:
                table = data.get_event_table(entity_type).data
            else:
                table = data.get_object_table(entity_type).data
            if attribute in table.columns:
                found |= set(table[attribute].dropna().unique())
        return found

    def emit_columns(
        self,
        df: pd.DataFrame,
        key_col: str,
        attribute_parent: str,
    ) -> None:
        """Emit every attribute column of one table.

        Timestamps and text always become value individuals. Numbers follow
        the data representation of the build, and are skipped entirely when it
        selects none. Columns that are empty for this table are skipped, which
        is how an attribute that only some activities carry stays off the
        others.

        Args:
            df: Table holding the entities and their attributes.
            key_col: Column holding the entity id.
            attribute_parent: Class the attribute classes go below, either the
                event or the object attribute class.
        """
        dr = self.settings.data_representations
        for attr in (c for c in df.columns if c != key_col):
            mask = df[attr].notna()
            if not mask.any():
                continue
            df_attr = df.loc[mask, [key_col, attr]]
            s = df_attr[attr]

            if pd.api.types.is_numeric_dtype(s) and not dr:
                continue

            prop = vocab.has_attribute_property(attr)
            source = attribute_source(attr)
            self.sink.add_class(attr, attribute_parent, source=source)
            owners = self._owners.get(attr, [])
            self.sink.add_property(
                prop,
                domain_class=owners or None,
                range_class=attr,
                superprop=vocab.HAS_ATTRIBUTE,
                create_inverse=self.settings.has_inverses,
                source=source,
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
        dr: tuple[DataRepresentation, ...],
    ) -> None:
        """Emit a numeric column as raw values, as bins, or as both."""
        raw_values = DataRepresentation.VALUES in dr
        if raw_values:
            self.emit_values(df_attr, key_col, attr, prop, True)

        binned = self._bin(df_attr[attr])
        if binned is None:
            return

        df_binned = df_attr.assign(_bin=binned)
        in_bin = df_binned[df_binned["_bin"].notna()]
        out_of_bin = df_binned[df_binned["_bin"].isna()]

        if not raw_values:
            self.emit_values(in_bin, key_col, attr, prop, False)

        self._emit_bin_classes(in_bin, attr)
        self._emit_out_of_bin(out_of_bin, key_col, attr, prop, raw_values)

    def _bin(self, values: pd.Series) -> pd.Series | None:
        """Bin a column with the selected strategy, or None if binning is off."""
        mode = self.settings.binning_mode
        if mode is None:
            return None
        binner = (
            equal_bin_series
            if mode is DataRepresentation.EQUAL_BINS
            else dist_bin_series
        )
        return binner(values, bins=self.settings.bins, labels=self.settings.bin_labels)

    def _emit_bin_classes(self, in_bin: pd.DataFrame, attr: str) -> None:
        """Emit one subclass per bin, holding the value individuals in it."""
        for bin_label, bin_group in in_bin.groupby("_bin", observed=True):
            bin_cls = vocab.bin_class(attr, bin_label)
            self.sink.add_class(
                bin_cls, attr, source=f"a bin of {attribute_source(attr)}"
            )
            value_inds = [
                vocab.attribute_value(attr, v) for v in bin_group[attr].unique()
            ]
            self.sink.assert_individuals(value_inds, bin_cls)

    def _emit_out_of_bin(
        self,
        out_of_bin: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        raw_values: bool,
    ) -> None:
        """Report values that fell outside every bin, and keep them as numbers.

        A value with no bin would otherwise vanish from the ontology, so when
        raw values are not already being emitted, these are.
        """
        if out_of_bin.empty:
            return
        logger.warning(
            f"BINNING: {len(out_of_bin)} value(s) for attribute '{attr}' fall "
            "outside all bin boundaries and carry no bin class."
            + ("" if raw_values else " Using the raw numeric value instead.")
        )
        if not raw_values:
            self.emit_values(out_of_bin, key_col, attr, prop, True)

    def emit_values(
        self,
        df: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        also_assert_data_value: bool,
    ) -> None:
        """Emit the value individuals of one column and link them to their owners.

        Args:
            df: Table holding the owners and the values.
            key_col: Column holding the owner id.
            attr: Attribute the values belong to.
            prop: Property linking an owner to a value.
            also_assert_data_value: Also keep the literal on the value
                individual, which is wanted for numbers and timestamps and
                redundant for text.
        """
        tmp = df.copy()

        vals = tmp[attr].unique()
        mapping = {v: vocab.attribute_value(attr, v) for v in vals}
        tmp["attr_ind"] = tmp[attr].map(mapping)

        seen = self._emitted_values.setdefault(attr, set())
        fresh = [name for name in mapping.values() if name not in seen]
        seen.update(fresh)

        self.sink.assert_individuals(fresh, attr)

        self.annotations.attribute_values(
            {value: name for value, name in mapping.items() if name in fresh}
        )

        self.sink.assert_relations(tmp, prop=prop, from_key=key_col, to_key="attr_ind")

        if also_assert_data_value:
            self.sink.assert_relations(
                tmp,
                prop=vocab.HAS_DATA_VALUE,
                from_key="attr_ind",
                to_key=attr,
                data_prop=True,
            )
