"""The `inspect` command: what one log holds, without building anything."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from semanticlog.core.enums import OCELExtension
from semanticlog.data.cleaning.renames import RenameKind
from semanticlog.data.loading import load_and_clean
from semanticlog.data.stats import rows_by_kind
from semanticlog.data.tables import EntityKind


@dataclass
class InspectReport:
    """What one log contains, as `semlog inspect` reports it.

    Attributes:
        input_path: The log this describes.
        activities: Activity name and event count, most events first.
        object_types: Object type and object count, most objects first.
        relations: Relation name and row count, most rows first.
        time_range: Earliest and latest timestamp, or None if there are none.
        renamed: Activity and object type names cleaning changed, as cleaned
            name to source name.
    """

    input_path: Path
    activities: list[tuple[str, int]]
    object_types: list[tuple[str, int]]
    relations: list[tuple[str, int]]
    time_range: tuple[str, str] | None
    renamed: dict[str, str]

    def render(self) -> str:
        """Return the report as the text the command prints."""
        lines = [f"{self.input_path}", ""]
        lines.append(f"activities ({len(self.activities)})")
        lines += [f"  {n:>9,}  {name}" for name, n in self.activities]
        lines.append("")
        lines.append(f"object types ({len(self.object_types)})")
        lines += [f"  {n:>9,}  {name}" for name, n in self.object_types]
        lines.append("")
        lines.append(f"relations ({len(self.relations)})")
        lines += [f"  {n:>9,}  {name}" for name, n in self.relations]
        lines.append("")
        if self.time_range is not None:
            lines.append(
                f"time range           {self.time_range[0]} .. {self.time_range[1]}"
            )
        if self.renamed:
            lines.append("")
            lines.append("renamed")
            lines += [
                f"  {clean:<28} was {original}"
                for clean, original in sorted(self.renamed.items())
            ]
        return "\n".join(lines)


def inspect_input(
    path: Path,
    extensions: Sequence[OCELExtension] = (),
    columns: Mapping[str, Any] | None = None,
) -> InspectReport:
    """Return a summary of one log.

    The log is cleaned first, so the names shown are the ones a build emits.
    An activity or object type name that cleaning changed is listed with its
    source spelling.

    Args:
        columns: Column names for a flat `.csv` or `.xes` log.

    Raises:
        ConfigError: If a column setting names no column the loader reads.
        InputError: If no loader reads the file extension, or the log is
            malformed.
    """
    data = load_and_clean(path, extensions, columns=columns)

    def by_count(counted: list[tuple[str, int]]) -> list[tuple[str, int]]:
        return sorted(counted, key=lambda item: item[1], reverse=True)

    rows = rows_by_kind(data)
    activities = by_count(rows[EntityKind.EVENT])
    object_types = by_count(rows[EntityKind.OBJECT])
    relations = by_count(
        [
            (name, len(data.get_relation(name).data))
            for name in data.get_relation_types()
        ]
    )

    stamps = [
        table.data[table.time]
        for table in data.tables_of(EntityKind.EVENT).values()
        if table.time is not None
    ]
    non_empty = [s for s in stamps if len(s)]
    time_range = (
        (str(min(s.min() for s in non_empty)), str(max(s.max() for s in non_empty)))
        if non_empty
        else None
    )

    schema_names = [(a, RenameKind.ACTIVITY) for a, _ in activities] + [
        (t, RenameKind.OBJECT_TYPE) for t, _ in object_types
    ]
    renamed = {
        name: original
        for name, kind in schema_names
        if (original := data.get_original(kind.value, name)) is not None
        and original != name
    }

    return InspectReport(
        input_path=path,
        activities=activities,
        object_types=object_types,
        relations=relations,
        time_range=time_range,
        renamed=renamed,
    )
