"""Keeping an attribute name that several types carry apart per type.

`qualify_shared_attributes` renames such an attribute to `<type>_<attribute>`
in each table, change tables included, so every type gets its own attribute
class, values and bins.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from semanticlog.data.cleaning.renames import RenameKind, RenameLog
from semanticlog.data.cleaning.rewrite import remap
from semanticlog.data.oced import OCEDStore, attribute_owners_by_kind

logger = logging.getLogger(__name__)


def qualify_shared_attributes(data: OCEDStore, log: RenameLog) -> None:
    """Prefix each attribute carried by more than one type with the type, in place.

    An attribute only one type carries keeps its name. A change table counts
    as its object type. The names in the store must be clean.

    Args:
        data: Store to rewrite.
        log: Rename log the new names are recorded in, each keeping the
            source name of the attribute.
    """
    shared = {
        name
        for name, by_kind in attribute_owners_by_kind(data).items()
        if sum(len(types) for types in by_kind.values()) > 1
    }
    if not shared:
        return

    new_names: dict[str, set[str]] = defaultdict(set)
    for view in data.tables():
        renames = {c: f"{view.name}_{c}" for c in view.filled_attributes if c in shared}
        if not renames:
            continue
        frame = view.frame.rename(columns=renames)
        if "ocel:field" in frame.columns:
            frame = remap(frame, ["ocel:field"], renames)
        view.replace(frame)
        for old, new in renames.items():
            new_names[old].add(new)

    for old, news in new_names.items():
        log.split(RenameKind.ATTRIBUTE, old, sorted(news))
        log.split_value_column(old, sorted(news))
    logger.info(
        "Qualified %d attribute name(s) carried by more than one type: %s",
        len(shared),
        ", ".join(sorted(shared)),
    )
