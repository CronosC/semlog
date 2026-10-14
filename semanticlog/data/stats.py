"""Measurements of a log file and of a loaded store."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

from semanticlog.data.readable import ReadableStore
from semanticlog.data.tables import EntityKind, RelationKind

_HASH_CHUNK = 1 << 20


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def rows_by_kind(
    data: ReadableStore,
) -> dict[EntityKind | None, list[tuple[str, int]]]:
    """Return each entity type with its row count, grouped by `EntityKind`."""
    grouped: dict[EntityKind | None, list[tuple[str, int]]] = defaultdict(list)
    for view in data.tables(changes=False):
        grouped[view.kind].append((view.name, len(view.frame)))
    return grouped


def _relation_rows(data: ReadableStore, kind: RelationKind) -> int:
    """Count the rows across one relation family."""
    return sum(
        len(data.get_relation(name).data) for name in data.get_relation_types(kind)
    )


def store_stats(data: ReadableStore) -> dict[str, int]:
    """Return the counts of a loaded log, as the provenance record holds them."""
    rows = rows_by_kind(data)
    events, objects = rows[EntityKind.EVENT], rows[EntityKind.OBJECT]
    return {
        "activities": len(events),
        "events": sum(n for _, n in events),
        "object_types": len(objects),
        "objects": sum(n for _, n in objects),
        "event_to_object_relations": _relation_rows(data, RelationKind.E2O),
        "object_to_object_relations": _relation_rows(data, RelationKind.O2O),
    }
