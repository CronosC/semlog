"""Reading and writing an :class:`~semanticlog.data.oced.OCEDStore` on disk.

A store is written as a directory: one Parquet file per table, plus a JSON
manifest describing what each file is and how the tables fit together.

This replaces pickling the live object. A pickle records the class layout as much
as the data, so it stops loading the moment a dataclass gains a field, and only
semlog can read it at all. For a research artifact that is the wrong trade: the
stored log outlives the code that wrote it, and other tools should be able to
open it. Parquet keeps the dtypes exactly as written, which CSV would force us to
reconstruct from a recorded schema.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd

if TYPE_CHECKING:
    from semanticlog.data.oced import OCEDStore, RelationTable

#: Bumped when the on-disk layout changes in a way older readers cannot handle.
FORMAT = "semlog-oced"
FORMAT_VERSION = 1

MANIFEST_NAME = "manifest.json"


def _table_path(root: Path, folder: str, index: int) -> tuple[Path, str]:
    """Return the absolute path and its manifest-relative form.

    Numbered rather than named after the table: table names come from source data
    and may contain characters that are not valid in a filename. The manifest
    carries the real name.
    """
    relative = f"{folder}/{index}.parquet"
    return root / relative, relative


def write_store(data: OCEDStore, path: str | Path) -> Path:
    """Write ``data`` to a directory at ``path``, replacing anything already there.

    Args:
        data: The store to write.
        path: Destination directory. Created if missing.

    Returns:
        The directory written to.
    """
    from semanticlog.data.oced import RelationKind

    root = Path(path)
    if root.exists():
        shutil.rmtree(root)
    for folder in ("events", "objects", "e2o", "o2o", "changes"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "ocel_mode": data.has_ocel_mode(),
        "backmap": data.backmap,
        "events": [],
        "objects": [],
        "event_to_object": [],
        "object_to_object": [],
        "object_changes": [],
    }

    for index, activity in enumerate(data.get_activities()):
        table = data.get_events(activity)
        target, relative = _table_path(root, "events", index)
        table.data.to_parquet(target, index=False)
        manifest["events"].append(
            {
                "activity": activity,
                "id": table.id,
                "time": table.time,
                "file": relative,
            }
        )

    for index, obj_type in enumerate(data.get_object_types()):
        obj = data.get_object(obj_type)
        target, relative = _table_path(root, "objects", index)
        obj.data.to_parquet(target, index=False)
        manifest["objects"].append({"type": obj_type, "id": obj.id, "file": relative})

    for key, kind, folder in (
        ("event_to_object", RelationKind.E2O, "e2o"),
        ("object_to_object", RelationKind.O2O, "o2o"),
    ):
        family = cast("dict[str, RelationTable]", data.get_relations(kind))
        for index, (name, relation) in enumerate(family.items()):
            target, relative = _table_path(root, folder, index)
            relation.data.to_parquet(target, index=False)
            manifest[key].append(
                {
                    "name": name,
                    "from_id": relation.from_id,
                    "to_id": relation.to_id,
                    "to_type": list(relation.to_type),
                    "from_type": (
                        list(relation.from_type)
                        if relation.from_type is not None
                        else None
                    ),
                    "file": relative,
                }
            )

    if data.has_ocel_mode():
        for index, obj_type in enumerate(data.get_object_change_types()):
            target, relative = _table_path(root, "changes", index)
            data.get_object_changes(obj_type).to_parquet(target, index=False)
            manifest["object_changes"].append({"type": obj_type, "file": relative})

    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return root


def read_store(path: str | Path) -> OCEDStore:
    """Read a store written by :func:`write_store`.

    Args:
        path: The directory to read.

    Returns:
        A reconstructed :class:`~semanticlog.data.oced.OCEDStore`.

    Raises:
        ValueError: If the directory is not a semlog store, or was written by a
            newer format version than this code understands.
    """
    from semanticlog.data.oced import (
        EventTable,
        ObjectTable,
        OCEDStore,
        Relations,
        RelationTable,
    )

    root = Path(path)
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"{root} is not a semlog store: no {MANIFEST_NAME} in it.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(
            f"{root} is not a semlog store: format {manifest.get('format')!r}."
        )
    if manifest["version"] > FORMAT_VERSION:
        raise ValueError(
            f"{root} was written in format version {manifest['version']}, but this "
            f"semlog reads up to version {FORMAT_VERSION}. Upgrade semlog to read it."
        )

    def frame(entry: dict) -> pd.DataFrame:
        return pd.read_parquet(root / entry["file"])

    store = OCEDStore(ocel_mode=manifest["ocel_mode"])
    events = {
        entry["activity"]: EventTable(
            id=entry["id"], time=entry["time"], data=frame(entry)
        )
        for entry in manifest["events"]
    }
    objects = {
        entry["type"]: ObjectTable(id=entry["id"], data=frame(entry))
        for entry in manifest["objects"]
    }

    def relations_of(key: str) -> dict[str, RelationTable]:
        return {
            entry["name"]: RelationTable(
                from_id=entry["from_id"],
                to_id=entry["to_id"],
                to_type=entry["to_type"],
                from_type=entry["from_type"],
                data=frame(entry),
            )
            for entry in manifest[key]
        }

    store.replace_tables(
        events=events,
        objects=objects,
        relations=Relations(
            event_to_object=relations_of("event_to_object"),
            object_to_object=relations_of("object_to_object"),
        ),
    )
    if manifest["ocel_mode"]:
        store.replace_object_changes(
            {entry["type"]: frame(entry) for entry in manifest["object_changes"]}
        )
    store.backmap = manifest["backmap"]
    store.update_used_ids()
    return store
