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
from semanticlog.core.enums import DataRepresentation
from semanticlog.core.summary import count_phrase, name_lines
from semanticlog.data.binning import Binning, fit_dist_bins, fit_equal_bins
from semanticlog.data.oced import OCEDStore
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.settings import OntoBuildSettings

logger = logging.getLogger(__name__)

# Above this many owners, the count reads better than the list of names.
_MAX_NAMED_OWNERS = 3


def _numeric_columns(data: OCEDStore) -> dict[str, list[pd.Series]]:
    """Collect every numeric column of each attribute, across the whole store.

    An attribute reaches the build once per table that carries it, and the
    attribute history reaches it again. All of those share one class and one
    pool of value individuals, so binning has to see all of them at once.
    """
    columns: dict[str, list[pd.Series]] = defaultdict(list)
    for view in data.tables():
        for column in view.attributes:
            if not pd.api.types.is_numeric_dtype(view.frame[column]):
                continue
            values = view.frame[column].dropna()
            if not values.empty:
                columns[column].append(values)
    return columns


def attribute_binnings(
    data: OCEDStore, settings: OntoBuildSettings
) -> dict[str, Binning]:
    """Fit the bins of every numeric attribute the build will bin.

    Each attribute is cut once, over every value it takes anywhere in the
    store, so one number falls in the same bin whichever table it was read
    from. An attribute with too few distinct values to bin is absent, and the
    emitter keeps its raw values instead.

    Args:
        data: Store to fit over.
        settings: Settings, for the binning mode, the bin count and the labels.

    Returns:
        Attribute name to its cut. Empty when the build does not bin.
    """
    mode = settings.binning_mode
    if mode is None:
        return {}
    fit = fit_equal_bins if mode is DataRepresentation.EQUAL_BINS else fit_dist_bins

    binnings: dict[str, Binning] = {}
    for attribute, columns in _numeric_columns(data).items():
        values = pd.concat(columns, ignore_index=True)
        if not pd.api.types.is_numeric_dtype(values):
            logger.warning(
                f"BINNING: attribute '{attribute}' holds more than one numeric "
                "type across the tables carrying it, so it is not binned."
            )
            continue
        binning = fit(values, bins=settings.bins, labels=settings.bin_labels)
        if binning is not None:
            binnings[attribute] = binning
    return binnings


def attribute_owners(data: OCEDStore) -> dict[str, list[str]]:
    """Find which entity types carry each attribute.

    The owners become the domain of the `has_` property, so an attribute used
    by two activities gets both of them as its domain. The attribute history
    counts as its object type carrying the attribute, since an attribute that
    is empty for every object is dropped on load and would otherwise reach the
    build with no owner at all.

    Args:
        data: Store to scan.

    Returns:
        Attribute name to the sorted types that carry it.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for view in data.tables():
        for column in view.attributes:
            owners[column].add(view.name)
    return {attribute: sorted(types) for attribute, types in owners.items()}


def attribute_source(attribute: str) -> str:
    """Return how an attribute is described in a name collision message."""
    return f"attribute {attribute!r}"


def report_conflated_attributes(owners: dict[str, list[str]]) -> dict[str, list[str]]:
    """Warn about attribute names shared by more than one entity type.

    One name becomes one class with one pool of value individuals, so a
    `weight` on events and a `weight` on objects end up meaning the same
    thing. Since that is sometimes intended, this warns instead of failing.
    The owners of an attribute are named while there are few of them, and
    counted once there are many.

    Args:
        owners: Attribute name to the types that carry it.

    Returns:
        The entries with more than one owner.
    """
    conflated = {a: types for a, types in owners.items() if len(types) > 1}
    if not conflated:
        return conflated

    entities = len({owner for types in conflated.values() for owner in types})
    logger.warning(
        "Attribute conflation: "
        f"{count_phrase([(entities, 'different entity', 'different entities')])} share "
        f"{count_phrase([(len(conflated), 'attribute name', 'attribute names')])} "
        "and their values are merged in one class each."
    )
    shared = [
        f"{attribute} ({_owner_summary(types)})"
        for attribute, types in sorted(conflated.items())
    ]
    for line in name_lines(
        "attributes", shared, complete=logger.isEnabledFor(logging.DEBUG)
    ):
        logger.warning(line)
    return conflated


def _owner_summary(types: list[str]) -> str:
    """Name the types carrying an attribute, or count them when there are many."""
    if len(types) > _MAX_NAMED_OWNERS:
        return count_phrase([(len(types), "type", "types")])
    return ", ".join(types)


class AttributeEmitter(Emitter):
    """Emits attribute classes, their values, and the links to their owners.

    It remembers the values it has already emitted per attribute, since the
    same attribute reaches it once per entity type that carries it.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._owners: dict[str, list[str]] = {}
        self._binnings: dict[str, Binning] = {}
        self._emitted_values: dict[str, set[str]] = {}
        self._emitted_literals: dict[str, set[str]] = {}
        self._emitted_bins: dict[str, set[str]] = {}

    def bind_owners(self, owners: dict[str, list[str]]) -> None:
        """Supply the owner map used as the domain of each `has_` property."""
        self._owners = owners

    def bind_binnings(self, binnings: dict[str, Binning]) -> None:
        """Supply the cut of each attribute the build bins."""
        self._binnings = binnings

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

            self.emit_column_values(df_attr, key_col, attr, prop)

    def emit_column_values(
        self, df_attr: pd.DataFrame, key_col: str, attr: str, prop: str
    ) -> None:
        """Emit one attribute column under the data representation of the build.

        The class and the property are expected to be declared already. This
        is the one place that decides what a value looks like in the output,
        so the attribute history reaches values the same way the main pass
        does.

        Args:
            df_attr: Two columns, the owner and the value, with no gaps.
            key_col: Column holding the owner id.
            attr: Attribute the values belong to.
            prop: Property linking an owner to a value.
        """
        s = df_attr[attr]
        if pd.api.types.is_datetime64_any_dtype(s):
            self.emit_values(df_attr, key_col, attr, prop, True)
        elif pd.api.types.is_numeric_dtype(s):
            self._emit_numeric(
                df_attr, key_col, attr, prop, self.settings.data_representations
            )
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

        if self.settings.binning_mode is None:
            return

        binning = self._binnings.get(attr)
        if binning is None:
            self._emit_unbinnable(df_attr, key_col, attr, prop, raw_values)
            return

        df_binned = df_attr.assign(_bin=binning.apply(df_attr[attr]))
        in_bin = df_binned[df_binned["_bin"].notna()]
        out_of_bin = df_binned[df_binned["_bin"].isna()]

        if not raw_values:
            self.emit_values(in_bin, key_col, attr, prop, False)

        self._emit_bin_classes(in_bin, attr)
        self._emit_out_of_bin(out_of_bin, key_col, attr, prop, raw_values)

    def _emit_unbinnable(
        self,
        df_attr: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        raw_values: bool,
    ) -> None:
        """Emit the values of a column that has too few of them to bin.

        A class per distinct number would be a bin in name only, so the column
        gets no bin classes. Its values are still individuals, carrying the
        literal only when the build asked for raw values.
        """
        logger.warning(
            f"BINNING: attribute '{attr}' has too few distinct values to bin"
            + ("." if raw_values else ", so its values carry no bin class.")
        )
        if not raw_values:
            self.emit_values(df_attr, key_col, attr, prop, False)

    def _emit_bin_classes(self, in_bin: pd.DataFrame, attr: str) -> None:
        """Emit one subclass per bin, holding the value individuals in it.

        A value individual belongs to one bin however many tables the number
        was read from, so the ones already placed are skipped.
        """
        placed = self._emitted_bins.setdefault(attr, set())
        for bin_label, bin_group in in_bin.groupby("_bin", observed=True):
            bin_cls = vocab.bin_class(attr, bin_label)
            self.sink.add_class(
                bin_cls, attr, source=f"a bin of {attribute_source(attr)}"
            )
            fresh = [
                name
                for value in bin_group[attr].unique()
                if (name := vocab.attribute_value(attr, value)) not in placed
            ]
            placed.update(fresh)
            self.sink.assert_individuals(fresh, bin_cls)

    def _emit_out_of_bin(
        self,
        out_of_bin: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        raw_values: bool,
    ) -> None:
        """Report values that fell outside every bin, and emit them anyway.

        A value with no bin would otherwise vanish from the ontology. It is
        emitted as an individual like every other value of the attribute,
        carrying the literal only when the build asked for raw values.
        """
        if out_of_bin.empty:
            return
        logger.warning(
            f"BINNING: {len(out_of_bin)} value(s) for attribute '{attr}' fall "
            "outside all bin boundaries and carry no bin class."
        )
        if not raw_values:
            self.emit_values(out_of_bin, key_col, attr, prop, False)

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
                individual. Wanted for numbers and timestamps, redundant for
                text, and off for every value of a build that bins its numbers
                instead of keeping them.
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
            self._emit_literals(tmp, attr)

    def _emit_literals(self, tmp: pd.DataFrame, attr: str) -> None:
        """Keep the literal on each value individual, once per individual.

        A value individual is named after its value, so it carries one
        literal however many owners point at it and however many passes reach
        it. The attribute history reaches values the main pass has already
        emitted, so the ones already written are skipped here.
        """
        written = self._emitted_literals.setdefault(attr, set())
        fresh = tmp[~tmp["attr_ind"].isin(written)].drop_duplicates("attr_ind")
        if fresh.empty:
            return
        written.update(fresh["attr_ind"])
        self.sink.assert_relations(
            fresh,
            prop=vocab.HAS_DATA_VALUE,
            from_key="attr_ind",
            to_key=attr,
            data_prop=True,
        )
