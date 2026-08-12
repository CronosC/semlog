"""The record of how an ontology was produced, written into the ontology.

An `.owl` file on its own says nothing about the settings behind it, and two
settings of one log produce different ontologies. Annotating the record onto
the ontology keeps it with the file, where a sidecar could be separated from
it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from semanticlog.onto.emitters.base import Emitter
from semanticlog.onto.settings import OntoBuildSettings

OWL_VERSION_INFO = "http://www.w3.org/2002/07/owl#versionInfo"

_BULKY = ("annotation_data", "declare_result")


def build_configuration(settings: OntoBuildSettings) -> dict:
    """Return the settings to record, without the loaded file contents.

    The annotation and DECLARE content is left out. It can be large, it is
    already identified by the path and the input checksum, and it would
    otherwise be repeated in full inside the ontology.
    """
    config = settings.to_dict()
    for key in _BULKY:
        config.pop(key, None)
    return config


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
        """Write the provenance annotations.

        Args:
            source: Path of the log this ontology was built from.
            run: Anything else worth recording about the run, such as the
                input checksum and the command. Each key becomes one
                annotation property, and entries with no value are skipped.
        """
        from semanticlog import __version__

        self.sink.write_ontology_annotation(OWL_VERSION_INFO, f"semlog {__version__}")
        self.sink.write_ontology_annotation(
            f"{self.sink.iri}buildConfiguration",
            json.dumps(build_configuration(self.settings), sort_keys=True),
        )
        if source is not None:
            self.sink.write_ontology_annotation(f"{self.sink.iri}sourceLog", source)
        for name, value in (run or {}).items():
            if value is None:
                continue
            self.sink.write_ontology_annotation(
                f"{self.sink.iri}{name}", annotation_text(value)
            )
