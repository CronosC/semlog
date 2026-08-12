"""The on-disk form of an OCED store.

A store is a directory of Parquet tables plus a JSON manifest naming them.
Parquet is used so that the dtypes come back exactly as written, and the
manifest carries what a DataFrame cannot: which column is an id, which is a
timestamp, and which types each relation connects.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from semanticlog.data.tables import (
    EventTable,
    ObjectTable,
    Relations,
    RelationTable,
)

if TYPE_CHECKING:
    from semanticlog.data.oced import OCEDStore

FORMAT = "semlog-oced"
FORMAT_VERSION = 1

MANIFEST_NAME = "manifest.json"


def _table_path(root: Path, folder: str, index: int) -> tuple[Path, str]:
    """Return the absolute path of one table and the path to record for it."""
    relative = f"{folder}/{index}.parquet"
    return root / relative, relative


def _refuse_unless_a_store(root: Path) -> None:
    """Raise unless the directory is empty or holds a store to overwrite."""
    if not any(root.iterdir()):
        return
    if (root / MANIFEST_NAME).is_file():
        return
    raise ValueError(
        f"{root} is not empty and has no {MANIFEST_NAME}, so it is not a semlog "
        "store. Refusing to delete it. Choose an empty or non-existent directory, "
        "or remove it yourself if that is what you meant."
    )


def write_store(data: OCEDStore, path: str | Path) -> Path:
    """Write a store as Parquet tables plus a manifest.

    Args:
        data: Store to write.
        path: Directory to write into. An existing store there is replaced.

    Returns:
        The directory written.

    Raises:
        ValueError: If the directory is not empty and holds no manifest, which
            means it is not a store and will not be deleted.
    """
    from semanticlog.data.oced import RelationKind

    root = Path(path)
    if root.exists():
        _refuse_unless_a_store(root)
        shutil.rmtree(root)
    for folder in ("events", "objects", "e2o", "o2o", "changes"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "ocel_mode": data.ocel_mode,
        "backmap": data.backmap,
        "events": [],
        "objects": [],
        "event_to_object": [],
        "object_to_object": [],
        "object_changes": [],
    }

    for index, activity in enumerate(data.get_event_types()):
        table = data.get_event_table(activity)
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
        obj = data.get_object_table(obj_type)
        target, relative = _table_path(root, "objects", index)
        obj.data.to_parquet(target, index=False)
        manifest["objects"].append({"type": obj_type, "id": obj.id, "file": relative})

    for key, kind, folder in (
        ("event_to_object", RelationKind.E2O, "e2o"),
        ("object_to_object", RelationKind.O2O, "o2o"),
    ):
        family = data.get_e2o() if kind is RelationKind.E2O else data.get_o2o()
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

    if data.ocel_mode:
        for index, obj_type in enumerate(data.get_object_change_types()):
            target, relative = _table_path(root, "changes", index)
            data.get_object_changes(obj_type).to_parquet(target, index=False)
            manifest["object_changes"].append({"type": obj_type, "file": relative})

    (root / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return root


def read_store(path: str | Path) -> OCEDStore:
    """Read a store written by `write_store`.

    Args:
        path: Directory holding the store.

    Returns:
        The store, with its name backmap and used-id set restored.

    Raises:
        ValueError: If the directory holds no manifest, the manifest is for
            another format, or it was written by a newer semlog.
    """
    from semanticlog.data.oced import OCEDStore

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
