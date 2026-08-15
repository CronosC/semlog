"""The record of how an ontology was produced, written into the ontology.

An ontology file on its own says nothing about the settings behind it, and two
settings of one log produce different ontologies. Annotating the record onto
the ontology keeps it with the file, where a sidecar could be separated from
it.
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

    The settings hold only the choices a build is configured with, so all of
    them go in. What is resolved per log rather than configured in advance is
    recorded by the caller, as part of the run.
    """
    return settings.to_dict()


def annotation_text(value: Any) -> str:
    """Render a record entry as annotation text, as JSON unless it is already text."""
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
        """Open the ontology, annotated with the provenance record.

        The whole record goes to the sink at once, which is what opens the
        ontology, so this is the first emitter a build runs.

        Args:
            source: Path of the log this ontology was built from.
            run: Anything else worth recording about the run, such as the
                input checksum and the command. Each key becomes one
                annotation property, and entries with no value are skipped.
        """
        from semanticlog import __version__

        base = self.sink.iri
        record = {
            OWL_VERSION_INFO: f"semlog {__version__}",
            f"{base}{vocab.BUILD_CONFIGURATION}": json.dumps(
                build_configuration(self.settings), sort_keys=True
            ),
        }
        if source is not None:
            record[f"{base}{vocab.SOURCE_LOG}"] = source
        for name, value in (run or {}).items():
            if value is not None:
                record[f"{base}{name}"] = annotation_text(value)
        self.sink.open_ontology(record)
