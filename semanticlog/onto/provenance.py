"""Records in the ontology how the ontology was made.

The ``.owl`` file is the product, and downstream projects load it on its own
without importing semlog or reading the run manifest beside it. An ontology that
does not say which version and which configuration produced it cannot be traced
back to a run, and two configurations of the same log produce different
ontologies.

Deliberately excluded: anything that varies between two runs of the same input.
A build stays byte-for-byte reproducible, so timestamps live in the manifest,
which is where the rest of the per-run detail already is.
"""

from __future__ import annotations

import json

from semanticlog.onto.emit import Emitter
from semanticlog.onto.settings import OntoBuildSettings

OWL_VERSION_INFO = "http://www.w3.org/2002/07/owl#versionInfo"

#: Settings entries that are inputs to the build rather than descriptions of it,
#: and are large enough to bloat the header. Their paths are recorded instead.
_BULKY = ("annotation_data", "declare_result")


def build_configuration(settings: OntoBuildSettings) -> dict:
    """Return the settings as a compact, deterministic dict for embedding."""
    config = settings.to_dict()
    for key in _BULKY:
        config.pop(key, None)
    return config


class ProvenanceEmitter(Emitter):
    """Writes the tool version and build configuration onto the ontology."""

    def emit(self, source: str | None = None) -> None:
        """Annotate the ontology with what produced it.

        Args:
            source: Path or identifier of the input log, when the caller knows
                it. ``build_ontology`` receives a store rather than a path, so the
                CLI supplies this and library callers usually do not.
        """
        from semanticlog import __version__

        self.sink.write_ontology_annotation(OWL_VERSION_INFO, f"semlog {__version__}")
        self.sink.write_ontology_annotation(
            f"{self.sink.iri}buildConfiguration",
            json.dumps(build_configuration(self.settings), sort_keys=True),
        )
        if source is not None:
            self.sink.write_ontology_annotation(f"{self.sink.iri}sourceLog", source)
