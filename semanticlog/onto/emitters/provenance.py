"""The record of how an ontology was produced, written into the ontology.

`ProvenanceEmitter` writes the version, settings and run record as ontology
annotations.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from semanticlog.core import vocabulary as vocab
from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.settings import OntoBuildSettings

OWL_VERSION_INFO = "http://www.w3.org/2002/07/owl#versionInfo"


def build_configuration(settings: OntoBuildSettings) -> dict:
    """Return the settings to record.

    Per-log values are recorded by the caller in the run record.
    """
    return settings.to_dict()


def annotation_text(value: Any) -> str:
    """Return a record entry as text. Non-strings are rendered as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, default=str)


class ProvenanceEmitter(Emitter):
    """Annotates the ontology with the version, settings and run record."""

    def emit(
        self,
        source: str | None = None,
        run: Mapping[str, Any] | None = None,
    ) -> None:
        """Open the ontology with the provenance record as its annotations.

        This is the first emitter a build runs.

        Args:
            source: Path of the log this ontology was built from.
            run: Other values to record about the run, such as the input
                checksum and the command. Each key becomes one annotation
                property, and entries with no value are skipped.

        Raises:
            ValueError: If a key of `run` cannot be an IRI local name.
        """
        from semanticlog import __version__

        base = vocab.TBOX_IRI
        record = {
            OWL_VERSION_INFO: f"semlog {__version__}",
            f"{base}{vocab.BUILD_CONFIGURATION}": json.dumps(
                build_configuration(self.settings), sort_keys=True
            ),
        }
        if source is not None:
            record[f"{base}{vocab.SOURCE_LOG}"] = source
        for name, value in (run or {}).items():
            if value is None:
                continue
            if not vocab.is_safe_local_name(name):
                raise ValueError(
                    f"Run record key {name!r} cannot be an IRI local name. Use "
                    "letters, digits, underscores, dots and hyphens only."
                )
            record[f"{base}{name}"] = annotation_text(value)
        self.sink.open_ontology(record)
