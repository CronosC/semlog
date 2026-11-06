"""Reading back an ontology a command wrote, in whichever syntax its sink used."""

from __future__ import annotations

import json
from pathlib import Path

from semanticlog.core.vocabulary import DEFAULT_IRI
from tests.support.emitted import Emitted
from tests.support.functional import read_functional
from tests.support.rdf import read_rdf

_JSON_ENTRIES = ("buildConfiguration", "caseConcepts")


def read_built(path: Path) -> Emitted:
    """Read an ontology the CLI wrote."""
    if path.read_text(encoding="utf-8").startswith("Prefix("):
        return read_functional(path)
    return read_rdf(path, DEFAULT_IRI)


def record(path: Path) -> dict:
    """Return the provenance record of a written ontology, with its JSON entries parsed."""
    written = dict(read_built(path).ontology_annotations)
    for key in _JSON_ENTRIES:
        if key in written:
            written[key] = json.loads(written[key])
    return written
