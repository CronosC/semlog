"""Scoping every attribute to the type that carries it.

OCEL 2.0 declares each attribute for one event type or one object type.
`qualify_attributes` renames every attribute to `<type>.<attribute>` in each
table, change tables included, so every type gets its own attribute class,
values and bins.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from semanticlog.core.errors import InputError
from semanticlog.data.cleaning.renames import RenameKind, RenameLog
from semanticlog.data.cleaning.rewrite import remap
from semanticlog.data.oced import OCEDStore

logger = logging.getLogger(__name__)


def qualified_attribute(type_name: str, attribute: str) -> str:
    """Return the name of an attribute scoped to the type that carries it."""
    return f"{type_name}.{attribute}"


def qualify_attributes(data: OCEDStore, log: RenameLog) -> None:
    """Prefix every attribute with the type that carries it, in place.

    A change table counts as its object type. The names in the store must be
    clean.

    Args:
        data: Store to rewrite.
        log: Rename log the new names are recorded in, each keeping the
            source name of the attribute.

    Raises:
        InputError: If two pairs of type and attribute give one name, such as
            type `a.b` with attribute `c` and type `a` with attribute `b.c`.
    """
    new_names: dict[str, set[str]] = defaultdict(set)
    claimed: dict[str, tuple[str, str]] = {}
    for view in data.tables():
        is_changes = "ocel:field" in view.frame.columns
        columns = view.filled_attributes if is_changes else view.attributes
        renames = {c: qualified_attribute(view.name, c) for c in columns}
        if not renames:
            continue
        for old, new in renames.items():
            owner = claimed.setdefault(new, (view.name, old))
            if owner != (view.name, old):
                raise InputError(
                    f"Attribute {old!r} of {view.name!r} and attribute "
                    f"{owner[1]!r} of {owner[0]!r} both become {new!r}. Rename "
                    "one of them in the input."
                )
        frame = view.frame.rename(columns=renames)
        if is_changes:
            frame = remap(frame, ["ocel:field"], renames)
        view.replace(frame)
        for old, new in renames.items():
            new_names[old].add(new)

    for old, news in new_names.items():
        log.split(RenameKind.ATTRIBUTE, old, sorted(news))
