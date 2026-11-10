"""Measurements of a loaded store."""

from __future__ import annotations

from collections import defaultdict

from semlog.data.readable import ReadableStore
from semlog.data.tables import EntityKind


def rows_by_kind(
    data: ReadableStore,
) -> dict[EntityKind | None, list[tuple[str, int]]]:
    """Return each entity type with its row count, grouped by `EntityKind`."""
    grouped: dict[EntityKind | None, list[tuple[str, int]]] = defaultdict(list)
    for view in data.tables(changes=False):
        grouped[view.kind].append((view.name, len(view.frame)))
    return grouped
