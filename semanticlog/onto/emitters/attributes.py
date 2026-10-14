"""Attributes, as classes whose members are the values they took.

An attribute becomes a class, each distinct value of it becomes an individual
of that class, and the entity carrying the value is linked to that individual
by a `has_` property. Two entities with the same value point at one
individual.

Numeric attributes can also be reduced to bins, which become subclasses of the
attribute class.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import DataRepresentation
from semanticlog.core.summary import counted, report_names
from semanticlog.data.binning import Binning, fit_dist_bins, fit_equal_bins
from semanticlog.data.readable import ReadableStore
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink

logger = logging.getLogger(__name__)

_MAX_NAMED_OWNERS = 3


def is_number_column(series: pd.Series) -> bool:
    """Whether a column holds numbers that `data_representations` governs.

    A boolean column is not a number column. It keeps its values and its
    literals under every data representation and is never binned.
    """
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(
        series
    )


def _numeric_columns(data: ReadableStore) -> dict[str, list[pd.Series]]:
    """Collect every numeric column of each attribute across every table.

    One attribute is binned over all of them together, including the change
    history.
    """
    columns: dict[str, list[pd.Series]] = defaultdict(list)
    for view in data.tables():
        for column in view.attributes:
            if not is_number_column(view.frame[column]):
                continue
            values = view.frame[column].dropna()
            if not values.empty:
                columns[column].append(values)
    return columns


def attribute_binnings(
    data: ReadableStore, settings: OntoBuildSettings
) -> dict[str, Binning]:
    """Fit the bins of every numeric attribute the build will bin.

    Each attribute is cut once over every value in the store. An attribute
    with too few distinct values to bin is not in the result.

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
        if not is_number_column(values):
            logger.warning(
                "Attribute '%s' has different numeric types in "
                "different tables and is not binned.",
                attribute,
            )
            continue
        binning = fit(values, bins=settings.bins, labels=settings.bin_labels)
        if binning is not None:
            binnings[attribute] = binning
    return binnings


def attribute_source(attribute: str) -> str:
    """Return how an attribute is described in a name collision message."""
    return f"attribute {attribute!r}"


def report_conflated_attributes(owners: dict[str, list[str]]) -> dict[str, list[str]]:
    """Warn about attribute names shared by more than one entity type.

    One name becomes one class and one set of value individuals, whichever
    types carry it.

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
        "%s share %s. The values of each name are merged in one class.",
        counted(entities, "different entity", "different entities"),
        counted(len(conflated), "attribute name", "attribute names"),
    )
    shared = [
        f"{attribute} ({_owner_summary(types)})"
        for attribute, types in sorted(conflated.items())
    ]
    report_names(logger, "attributes", shared, level=logging.WARNING)
    return conflated


def _owner_summary(types: list[str]) -> str:
    """Name the types carrying an attribute, or count them when there are many."""
    if len(types) > _MAX_NAMED_OWNERS:
        return counted(len(types), "type", "types")
    return ", ".join(types)


class AttributeEmitter(Emitter):
    """Emits attribute classes, their values, and the links to their owners.

    One attribute is emitted once per entity type and again from the
    attribute history. `emitted.values_of` records which values are written.

    Args:
        sink: Where axioms go.
        settings: Settings of this build.
        annotations: Writer for the labels and comments.
        emitted: The individuals the build has asserted.
        owners: Attribute name to the types carrying it. Domain of each
            `has_` property.
        binnings: The cut of each attribute the build bins.
    """

    def __init__(
        self,
        sink: AxiomSink,
        settings: OntoBuildSettings,
        annotations: AnnotationWriter,
        emitted: EmittedIndividuals,
        owners: Mapping[str, list[str]],
        binnings: Mapping[str, Binning],
    ) -> None:
        super().__init__(sink, settings, annotations, emitted)
        self._owners = owners
        self._binnings = binnings
        self._warned_unbinnable: set[str] = set()

    def emit_columns(
        self,
        df: pd.DataFrame,
        key_col: str,
        attribute_parent: str,
    ) -> None:
        """Emit every attribute column of one table.

        Timestamps, booleans and text become value individuals. Numbers
        follow `data_representations` and are skipped when it is empty. A
        column empty in this table is skipped.

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

            if is_number_column(s) and not dr:
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
                axiomatic_inverse=self.settings.axiomatic_inverses,
                source=source,
            )
            owner = "event" if attribute_parent == vocab.EVENT_ATTRIBUTE else "object"
            self.annotations.attribute(attr, prop, owner)

            self.emit_column_values(df_attr, key_col, attr, prop)

    def emit_column_values(
        self, df_attr: pd.DataFrame, key_col: str, attr: str, prop: str
    ) -> None:
        """Emit one attribute column under the data representation of the build.

        The class and the property must be declared. The attribute history
        calls this too.

        Args:
            df_attr: Two columns, the owner and the value, with no gaps.
            key_col: Column holding the owner id.
            attr: Attribute the values belong to.
            prop: Property linking an owner to a value.
        """
        s = df_attr[attr]
        if pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            self.emit_values(df_attr, key_col, attr, prop, True)
        elif is_number_column(s):
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

        No bin classes are emitted. The values are individuals, with the
        literal only when `VALUES` is selected. The warning is written once
        per attribute.
        """
        if attr not in self._warned_unbinnable:
            self._warned_unbinnable.add(attr)
            logger.warning(
                "Attribute '%s' has too few distinct values to bin. Its "
                "values carry no bin class.",
                attr,
            )
        if not raw_values:
            self.emit_values(df_attr, key_col, attr, prop, False)

    def _emit_bin_classes(self, in_bin: pd.DataFrame, attr: str) -> None:
        """Emit one subclass per bin, holding the value individuals in it.

        Values placed in a bin by an earlier table are skipped.
        """
        placed = self.emitted.bins_of(attr)
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
            self.sink.assert_individuals(bin_cls, fresh)

    def _emit_out_of_bin(
        self,
        out_of_bin: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        raw_values: bool,
    ) -> None:
        """Warn about values outside every bin and emit them as individuals.

        The literal is written only when `VALUES` is selected.
        """
        if out_of_bin.empty:
            return
        logger.warning(
            "%s value(s) of attribute '%s' fall outside "
            "all bin boundaries and carry no bin class.",
            len(out_of_bin),
            attr,
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
            also_assert_data_value: Also write the literal on the value
                individual. True for numbers, booleans and timestamps, False
                for text and for binned numbers without `VALUES`.
        """
        tmp = df.copy()

        vals = tmp[attr].unique()
        mapping = {v: vocab.attribute_value(attr, v) for v in vals}
        tmp["attr_ind"] = tmp[attr].map(mapping)

        seen = self.emitted.values_of(attr)
        fresh = [name for name in mapping.values() if name not in seen]
        fresh_names = set(fresh)
        seen.update(fresh)

        self.sink.assert_individuals(attr, fresh)

        if self.annotations.enabled:
            self.annotations.attribute_values(
                {value: name for value, name in mapping.items() if name in fresh_names}
            )

        self.assert_relations(prop, tmp, key_col, "attr_ind")

        if also_assert_data_value:
            self._emit_literals(tmp, attr, fresh_names)

    def _emit_literals(self, tmp: pd.DataFrame, attr: str, fresh: set[str]) -> None:
        """Write the literal on each value individual in `fresh`.

        The pass that first emits a value decides whether it gets a literal.
        This matters only for an attribute that is numeric in one table and
        text in another.
        """
        if not fresh:
            return
        rows = tmp[tmp["attr_ind"].isin(fresh)].drop_duplicates("attr_ind")
        self.assert_data(vocab.HAS_DATA_VALUE, rows, "attr_ind", attr)
