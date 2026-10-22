"""Allocating every schema name of a clean store out of one registry.

Object types, attributes, relations and activities each become a class or a
property. The registry claims each name together with the names a build
derives from it, and a claimant whose name is taken is renamed.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Collection, Iterable
from functools import partial

from semanticlog.core import vocabulary as vocab
from semanticlog.core.names import NameRegistry
from semanticlog.core.vocabulary import is_safe_local_name
from semanticlog.data.cleaning.renames import RenameKind, RenameLog
from semanticlog.data.cleaning.rewrite import apply_attribute_renames, attribute_names
from semanticlog.data.oced import OCEDStore
from semanticlog.data.tables import EntityKind

# Each prefix starts with a letter, so a prefixed name is a safe property name.
KIND_PREFIX: dict[RenameKind, str] = {
    RenameKind.OBJECT_TYPE: "object_",
    RenameKind.ATTRIBUTE: "attribute_",
    RenameKind.RELATION: "relation_",
    RenameKind.ACTIVITY: "event_",
    RenameKind.ID: "id_",
}

VALUE_PREFIX = "value_"

# A property name is written as an RDF/XML element, so it cannot open with a
# digit, a dot or a hyphen. Only a relation type becomes a property name
# unchanged.
_UNSAFE_PROPERTY_START = re.compile(r"^[^A-Za-z_]")


def _opens_a_property(name: str) -> bool:
    """Whether a name can be emitted as a property and as an IRI local name."""
    return is_safe_local_name(name) and not _UNSAFE_PROPERTY_START.match(name)


def _allocate_kind(
    registry: NameRegistry,
    log: RenameLog,
    kind: RenameKind,
    names: Collection[str],
    rename: Callable[[str, str], None],
    *,
    derives: Callable[[str], Iterable[str]] | None = None,
    accepts: Callable[[str], bool] = is_safe_local_name,
) -> None:
    """Allocate a free name for each claimant of one kind, renaming the ones that move."""
    pending = set(names)
    for name in sorted(names):
        free = registry.allocate(
            name,
            f"{kind.word} {name!r}",
            prefix=KIND_PREFIX[kind],
            pending=pending,
            derives=derives,
            accepts=accepts,
        )
        pending.discard(name)
        if free != name:
            rename(name, free)
            log.record(kind, name, free)


def allocate_schema_names(
    data: OCEDStore, log: RenameLog, bin_labels: Collection[str]
) -> NameRegistry:
    """Give every object type, attribute, relation and activity a free name.

    They are claimed in that order.

    Args:
        data: Store to fix, modified in place. Its names must be safe.
        log: Rename log to append to.
        bin_labels: Bin labels the build will use. Each bin class name is
            reserved.

    Returns:
        The registry, holding every schema name the build will emit.
    """
    registry = NameRegistry()
    labels = list(bin_labels)

    def rename_attribute(old: str, new: str) -> None:
        apply_attribute_renames(data, {old: new})
        log.rename_value_column(old, new)

    _allocate_kind(
        registry,
        log,
        RenameKind.OBJECT_TYPE,
        data.types(EntityKind.OBJECT),
        partial(data.rename, EntityKind.OBJECT),
        derives=vocab.derived_from_object_type,
    )
    _allocate_kind(
        registry,
        log,
        RenameKind.ATTRIBUTE,
        attribute_names(data),
        rename_attribute,
        derives=lambda name: vocab.derived_from_attribute(name, labels),
    )
    _allocate_kind(
        registry,
        log,
        RenameKind.RELATION,
        data.get_relation_types(),
        data.rename_relation,
        derives=vocab.derived_from_relation,
        accepts=_opens_a_property,
    )
    _allocate_kind(
        registry,
        log,
        RenameKind.ACTIVITY,
        data.types(EntityKind.EVENT),
        partial(data.rename, EntityKind.EVENT),
    )
    return registry
