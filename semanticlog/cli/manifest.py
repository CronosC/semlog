"""The run manifest written beside every ontology.

The ``.owl`` file is the product, but on its own it records nothing about how it
came to be.  The manifest closes that gap: it names the input and its checksum,
the fully resolved build settings, where the case concepts came from, and the
semlog version that ran.  Together the two files are enough to reproduce a
result or to tell two results apart.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from semanticlog.onto.settings import OntoBuildSettings

#: Bumped whenever the manifest layout changes incompatibly.
MANIFEST_VERSION = 1

_HASH_CHUNK = 1 << 20


def semlog_version() -> str:
    """The installed semlog version, or ``"unknown"`` outside an installation."""
    try:
        return version("semlog")
    except PackageNotFoundError:
        return "unknown"


def file_digest(path: Path) -> str:
    """SHA-256 of ``path``, read in chunks so large logs do not need to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _settings_record(settings: OntoBuildSettings) -> dict[str, Any]:
    """Serialise settings, replacing the bulk inputs with references to them.

    ``OntoBuildSettings.to_dict`` embeds the full DECLARE constraint table and
    the full annotation document, both of which belong in their own files rather
    than in a manifest meant to stay readable.
    """
    record = settings.to_dict()
    declare = record.pop("declare_result", None)
    record["declare_constraints"] = (
        {
            "count": len(declare["constraints"]),
            "case_concepts": declare["case_concepts"],
        }
        if declare is not None
        else None
    )
    annotations = record.pop("annotation_data", None)
    record["annotation_entries"] = len(annotations) if annotations else 0
    return record


@dataclass
class RunManifest:
    """The record of one build, written as JSON next to its ontology.

    Attributes:
        input_path: The OCEL file that was built.
        settings: The settings the build actually ran with.
        case_concept_source: Either ``"config"`` or ``"auto(n=N)"``.
        outputs: Artifact kind to path, with ``None`` for artifacts not written.
        stats: Counts describing the input and the resulting ontology.
        timings: Wall-clock seconds per pipeline stage.
        config_file: The config this run was resolved from, if any.
        command: The command line as typed.
        materialized: Whether a reasoner ran over the result.
    """

    input_path: Path
    settings: OntoBuildSettings
    case_concept_source: str
    outputs: dict[str, str | None] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    config_file: Path | None = None
    command: str = ""
    materialized: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Render the manifest as the JSON-compatible dict that gets written."""
        return {
            "manifest_version": MANIFEST_VERSION,
            "semlog_version": semlog_version(),
            "created": datetime.now(UTC).astimezone().isoformat(timespec="seconds"),
            "command": self.command,
            "config_file": None if self.config_file is None else str(self.config_file),
            "input": {
                "path": str(self.input_path),
                "bytes": self.input_path.stat().st_size,
                "sha256": file_digest(self.input_path),
            },
            "case_concepts": {
                "value": list(self.settings.case_concepts),
                "source": self.case_concept_source,
            },
            "settings": _settings_record(self.settings),
            "outputs": self.outputs,
            "stats": self.stats,
            "timings_s": {k: round(v, 3) for k, v in self.timings.items()},
            "materialized": self.materialized,
        }

    def write(self, path: Path) -> dict[str, Any]:
        """Write the manifest to ``path`` and return what was written."""
        record = self.to_dict()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, default=str)
            handle.write("\n")
        return record
