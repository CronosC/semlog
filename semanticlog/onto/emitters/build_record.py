"""The record of how an ontology was produced, written into the ontology.

`BuildRecordEmitter` writes the tool, the settings and the case concepts as
ontology annotations.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from semanticlog.core import vocabulary as vocab
from semanticlog.onto.emitters.base import Emitter


class BuildRecordEmitter(Emitter):
    """Annotates the ontology with the tool, the settings and the case concepts."""

    def emit(self, case_concepts: Sequence[str]) -> None:
        """Open the ontology with the build record as its annotations.

        This is the first emitter a build runs. The settings and the case
        concepts are written as JSON text.
        """
        from semanticlog import __version__

        base = self.sink.tbox_iri
        self.sink.open_ontology(
            {
                f"{base}{vocab.BUILD_TOOL}": f"semlog {__version__}",
                f"{base}{vocab.BUILD_CONFIGURATION}": json.dumps(
                    self.settings.to_dict(), sort_keys=True
                ),
                f"{base}{vocab.CASE_CONCEPTS}": json.dumps(list(case_concepts)),
            }
        )
