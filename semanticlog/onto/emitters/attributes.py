"""Attributes, as value individuals, as literals on their owners, or both.

Under `individuals` an attribute becomes a class, each distinct value of it
becomes an individual of that class carrying its literal, and the entity
holding the value is linked to that individual by a `has_` property. Two
entities with the same value point at one individual. Under `literals` the
entity carries the literal through the attribute's value property.

Numeric attributes can also be reduced to bins, which become subclasses of the
attribute class.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import pandas as pd

from semanticlog.core import vocabulary as vocab
from semanticlog.core.enums import AttributeRepresentation, IndividualKind, OWLProfile
from semanticlog.data.binning import Binning, fit_dist_bins, fit_equal_bins
from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind
from semanticlog.onto.annotations import AnnotationWriter
from semanticlog.onto.emitted import EmittedIndividuals
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.settings import OntoBuildSettings
from semanticlog.onto.sinks import AxiomSink
from semanticlog.onto.sinks.literals import (
    BOOLEAN,
    DATE_TIME,
    DECIMAL,
    INTEGER,
    STRING,
    profile_datatype,
)

logger = logging.getLogger(__name__)


def is_number_column(series: pd.Series) -> bool:
    """Whether a column holds numbers a bin mode can cut.

    A boolean column is not a number column. It keeps its values and its
    literals under every attribute representation and is never binned.
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


def _column_datatype(series: pd.Series) -> str | None:
    """Return the XSD datatype every value of a column is written under, or None.

    None stands for a column whose values take more than one datatype, or one
    with no datatype a data property range may name.
    """
    if pd.api.types.is_bool_dtype(series):
        return BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATE_TIME
    if pd.api.types.is_integer_dtype(series):
        return INTEGER
    if pd.api.types.is_float_dtype(series):
        return DECIMAL
    return _INFERRED_DATATYPES.get(pd.api.types.infer_dtype(series, skipna=True))


_INFERRED_DATATYPES: dict[str, str] = {
    "string": STRING,
    "boolean": BOOLEAN,
    "integer": INTEGER,
    "floating": DECIMAL,
    "decimal": DECIMAL,
    "datetime": DATE_TIME,
    "datetime64": DATE_TIME,
}


def attribute_datatypes(
    data: ReadableStore, settings: OntoBuildSettings
) -> dict[str, str]:
    """Return the datatype of the literals of each attribute, for its value property range.

    An attribute whose columns take more than one datatype is left out and
    logged.

    Args:
        data: Store to read.
        settings: Settings, for the profile the literals are written under.

    Returns:
        Attribute name to the local name of its XSD datatype.
    """
    found: dict[str, set[str | None]] = defaultdict(set)
    for view in data.tables():
        for column in view.filled_attributes:
            found[column].add(_column_datatype(view.frame[column].dropna()))
    datatypes: dict[str, str] = {}
    for attribute, kinds in found.items():
        if len(kinds) == 1 and None not in kinds:
            (kind,) = kinds
            datatypes[attribute] = profile_datatype(str(kind), settings.profile)
        else:
            logger.warning(
                "Attribute '%s' holds values of more than one datatype. Its value "
                "property gets no range.",
                attribute,
            )
    return datatypes


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
    fit = (
        fit_equal_bins if mode is AttributeRepresentation.EQUAL_BINS else fit_dist_bins
    )

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
            logger.info(
                "  %s: bins %s cut at %s",
                attribute,
                ", ".join(binning.labels),
                ", ".join(f"{edge:g}" for edge in binning.edges[1:-1]),
            )
    return binnings


def attribute_source(attribute: str) -> str:
    """Return how an attribute is described in a name collision message."""
    return f"attribute {attribute!r}"


class AttributeEmitter(Emitter):
    """Emits attribute classes, their values, and the links to their owners.

    One attribute is emitted once per entity type and again from the
    attribute history. `emitted.values_of` records which values are written.

    Args:
        sink: Where axioms go.
        settings: Settings of this build.
        annotations: Writer for the labels and comments.
        emitted: The individuals the build has asserted.
        owners: Attribute name to the types carrying it, per kind. Domain of
            each `has_` property.
        binnings: The cut of each attribute the build bins.
        datatypes: The datatype of each attribute's literals. An attribute
            without one gets a value property with no range.
        attribute_history: Whether the build emits the attribute change
            history. A change record then carries a literal under `literals`,
            so the change class joins the domain of the value property.
        groups: Each attribute name of the log to the qualified attributes
            that came from it. Under `shared_attributes` each name becomes a
            class and a property above its attributes.
    """

    def __init__(
        self,
        sink: AxiomSink,
        settings: OntoBuildSettings,
        annotations: AnnotationWriter,
        emitted: EmittedIndividuals,
        owners: Mapping[str, Mapping[EntityKind, list[str]]],
        binnings: Mapping[str, Binning],
        datatypes: Mapping[str, str] | None = None,
        attribute_history: bool = False,
        groups: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        super().__init__(sink, settings, annotations, emitted)
        self._owners = owners
        self._binnings = binnings
        self._datatypes = datatypes or {}
        self._attribute_history = attribute_history
        self._groups = groups or {}
        self._group_of = {
            member: group
            for group, members in self._groups.items()
            for member in members
        }
        self._declared_groups: set[str] = set()
        self._warned_unbinnable: set[str] = set()
        self._annotated: set[str] = set()
        self._classes: dict[str, None] = {}
        self._bin_classes: dict[str, dict[str, None]] = defaultdict(dict)

    def add_attribute_class(self, attr: str, parent: str) -> None:
        """Declare an attribute class below its parent and record it."""
        self.sink.add_class(attr, parent, source=attribute_source(attr))
        self._classes[attr] = None

    def _owner_types(self, attr: str) -> list[str]:
        """Return the sorted types carrying a value of `attr`."""
        by_kind = self._owners.get(attr, {})
        return sorted({t for types in by_kind.values() for t in types})

    def _value_domain(self, attr: str, parent: str) -> list[str]:
        """Return the classes whose members carry the literal of `attr`."""
        domain = []
        if self.settings.emits_value_individuals:
            domain.append(attr)
        if self.settings.emits_entity_literals:
            domain.extend(self._owner_types(attr))
            if self._attribute_history and parent == vocab.OBJECT_ATTRIBUTE:
                domain.append(vocab.attribute_change_class(attr))
        return domain

    def declare_value_property(self, attr: str, parent: str) -> str:
        """Declare the data property that carries the literal of a value of `attr`.

        Its domain is the attribute class under `individuals`, the owner
        types under `literals`, and both under both. The change class of an
        object attribute joins the domain when the build emits the history,
        and is declared here so the domain can name it.

        Args:
            attr: The attribute.
            parent: Class the attribute class goes below, either the event or
                the object attribute class.

        Returns:
            The property name.
        """
        prop = vocab.value_property(attr)
        domain = self._value_domain(attr, parent)
        change_cls = vocab.attribute_change_class(attr)
        if change_cls in domain:
            self.sink.add_class(
                change_cls,
                vocab.ATTRIBUTE_CHANGE,
                source=f"the attribute history of {attribute_source(attr)}",
            )
        self.sink.add_property(
            prop,
            domain_class=domain or None,
            range_class=self._datatypes.get(attr),
            superprop=vocab.HAS_DATA_VALUE,
            data_prop=True,
            source=attribute_source(attr),
        )
        self.sink.add_functional(prop, data_prop=True)
        if prop not in self._annotated:
            self._annotated.add(prop)
            self.annotations.attribute_value_property(
                attr, prop, self._owners.get(attr, {})
            )
        return prop

    def declare_attribute(self, attr: str, parent: str) -> str:
        """Declare what one attribute needs before its values are emitted.

        The value property is declared in every mode. The attribute class and
        the `has_` property are declared under `individuals` only.

        Returns:
            The `has_` property name, declared only under `individuals`.
        """
        self.declare_value_property(attr, parent)
        prop = vocab.has_attribute_property(attr)
        if not self.settings.emits_value_individuals:
            return prop
        self.add_attribute_class(attr, parent)
        superprop: str = vocab.HAS_ATTRIBUTE
        group = self._group_of.get(attr) if self.settings.shared_attributes else None
        if group is not None:
            self._declare_group(group)
            self.sink.add_class(attr, group, source=attribute_source(attr))
            superprop = vocab.attribute_group_property(group)
        self.sink.add_property(
            prop,
            domain_class=self._owner_types(attr) or None,
            range_class=attr,
            superprop=superprop,
            create_inverse=self.settings.has_inverses,
            axiomatic_inverse=self.settings.axiomatic_inverses,
            source=attribute_source(attr),
        )
        self.declare_functional(prop)
        if attr not in self._annotated:
            self._annotated.add(attr)
            self.annotations.attribute(attr, prop, self._owners.get(attr, {}))
        return prop

    def _declare_group(self, group: str) -> None:
        """Declare the class and property of an attribute name, once.

        The class sits directly below `Attribute`, because one name can be
        on an event type and an object type. The property sits below
        `has_attribute` with the owner types of every member as its domain.
        """
        if group in self._declared_groups:
            return
        self._declared_groups.add(group)
        source = f"attribute name {group!r}"
        self.sink.add_class(group, vocab.ATTRIBUTE, source=source)
        prop = vocab.attribute_group_property(group)
        owners = sorted(
            {t for member in self._groups[group] for t in self._owner_types(member)}
        )
        self.sink.add_property(
            prop,
            domain_class=owners or None,
            range_class=group,
            superprop=vocab.HAS_ATTRIBUTE,
            create_inverse=self.settings.has_inverses,
            axiomatic_inverse=self.settings.axiomatic_inverses,
            source=source,
        )
        self.annotations.attribute_group(group, prop)

    def emit_disjointness(self) -> None:
        """State that the attribute classes, and the bins of each, share no value."""
        self.sink.add_disjoint_classes(self._classes, source="the attribute classes")
        for attr, bins in self._bin_classes.items():
            self.sink.add_disjoint_classes(bins, source=f"the bins of {attr!r}")

    def emit_columns(
        self,
        df: pd.DataFrame,
        key_col: str,
        attribute_parent: str,
        owner: IndividualKind,
    ) -> None:
        """Emit every attribute column of one table.

        A column empty in this table is skipped.

        Args:
            df: Table holding the entities and their attributes.
            key_col: Column holding the entity id.
            attribute_parent: Class the attribute classes go below, either the
                event or the object attribute class.
            owner: The kind of individual the entities are.
        """
        for attr in (c for c in df.columns if c != key_col):
            mask = df[attr].notna()
            if not mask.any():
                continue
            df_attr = df.loc[mask, [key_col, attr]]
            prop = self.declare_attribute(attr, attribute_parent)
            self.emit_column_values(df_attr, key_col, attr, prop, owner)

    def emit_column_values(
        self,
        df_attr: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        owner: IndividualKind,
    ) -> None:
        """Emit one attribute column, and the bins of a numeric one.

        The attribute must be declared. The attribute history calls this too.

        Args:
            df_attr: Two columns, the owner and the value, with no gaps.
            key_col: Column holding the owner id.
            attr: Attribute the values belong to.
            prop: Property linking an owner to a value individual, used under
                `individuals`.
            owner: The kind of individual the owners are.
        """
        self.emit_values(df_attr, key_col, attr, prop, owner)
        if is_number_column(df_attr[attr]):
            self._emit_bins(df_attr, attr)

    def emit_values(
        self,
        df: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        owner: IndividualKind,
    ) -> None:
        """Emit the values of one column as the attribute representations say.

        Under `individuals`, each fresh value becomes an individual carrying
        its literal, and every owner is linked to its value's individual.
        Under `literals`, every owner carries the literal itself.

        Args:
            df: Table holding the owners and the values.
            key_col: Column holding the owner id.
            attr: Attribute the values belong to.
            prop: Property linking an owner to a value individual.
            owner: The kind of individual the owners are.
        """
        if self.settings.emits_value_individuals:
            self._emit_individuals(df, key_col, attr, prop, owner)
        if self.settings.emits_entity_literals:
            self.assert_data(vocab.value_property(attr), df, key_col, attr, owner)

    def _emit_individuals(
        self,
        df: pd.DataFrame,
        key_col: str,
        attr: str,
        prop: str,
        owner: IndividualKind,
    ) -> None:
        """Emit the value individuals of one column and link them to their owners.

        The pass that first emits a value writes its literal. This matters
        only for an attribute that is numeric in one table and text in
        another.
        """
        tmp = df.copy()

        vals = tmp[attr].unique()
        mapping = {v: vocab.attribute_value(attr, v) for v in vals}
        tmp["attr_ind"] = tmp[attr].map(mapping)

        seen = self.emitted.values_of(attr)
        fresh = [name for name in mapping.values() if name not in seen]
        fresh_names = set(fresh)
        seen.update(fresh)

        self.sink.assert_individuals(attr, fresh, IndividualKind.VALUE)

        if self.annotations.enabled:
            self.annotations.attribute_values(
                {value: name for value, name in mapping.items() if name in fresh_names}
            )

        self.assert_relations(
            prop, tmp, key_col, "attr_ind", (owner, IndividualKind.VALUE)
        )

        if fresh_names:
            rows = tmp[tmp["attr_ind"].isin(fresh_names)].drop_duplicates("attr_ind")
            self.assert_data(
                vocab.value_property(attr), rows, "attr_ind", attr, IndividualKind.VALUE
            )

    def _emit_bins(self, df_attr: pd.DataFrame, attr: str) -> None:
        """Place the values of a numeric column in their bin classes.

        Nothing is emitted without a binning mode. An attribute with too few
        distinct values to bin is reported once, and values outside every bin
        are reported per table.
        """
        if self.settings.binning_mode is None:
            return
        binning = self._binnings.get(attr)
        if binning is None:
            if attr not in self._warned_unbinnable:
                self._warned_unbinnable.add(attr)
                logger.warning(
                    "Attribute '%s' has too few distinct values to bin. Its "
                    "values carry no bin class.",
                    attr,
                )
            return

        df_binned = df_attr.assign(_bin=binning.apply(df_attr[attr]))
        in_bin = df_binned[df_binned["_bin"].notna()]
        out_of_bin = df_binned[df_binned["_bin"].isna()]
        self._emit_bin_classes(in_bin, attr)
        if not out_of_bin.empty:
            logger.warning(
                "%s value(s) of attribute '%s' fall outside "
                "all bin boundaries and carry no bin class.",
                len(out_of_bin),
                attr,
            )

    def _emit_bin_classes(self, in_bin: pd.DataFrame, attr: str) -> None:
        """Emit one subclass per bin, holding the value individuals in it.

        Values placed in a bin by an earlier table are skipped. Under OWL 2
        DL each bin is also defined by its value range.
        """
        placed = self.emitted.bins_of(attr)
        binning = self._binnings[attr]
        for bin_label, bin_group in in_bin.groupby("_bin", observed=True):
            bin_cls = vocab.bin_class(attr, bin_label)
            self.sink.add_class(
                bin_cls, attr, source=f"a bin of {attribute_source(attr)}"
            )
            self._bin_classes[attr][bin_cls] = None
            if self.settings.profile != OWLProfile.OWL2EL:
                lower, upper = binning.bounds(str(bin_label))
                self.sink.add_bin_definition(
                    bin_cls,
                    attr,
                    vocab.value_property(attr),
                    None if math.isinf(lower) else lower,
                    None if math.isinf(upper) else upper,
                )
            if bin_cls not in self._annotated:
                self._annotated.add(bin_cls)
                self.annotations.bin_class(
                    bin_cls,
                    attr,
                    self._owners.get(attr, {}),
                    str(bin_label),
                    binning.bounds(str(bin_label)),
                )
            fresh = [
                name
                for value in bin_group[attr].unique()
                if (name := vocab.attribute_value(attr, value)) not in placed
            ]
            placed.update(fresh)
            self.sink.assert_individuals(bin_cls, fresh, IndividualKind.VALUE)
