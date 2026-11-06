"""Reading the provenance record back out of an ontology file.

`ProvenanceEmitter` writes the record as ontology annotations. These functions
find one annotation in the head of a written file, in functional syntax or in
RDF/XML, and parse its JSON. The file may be one semlog wrote or one the OWL
API rewrote.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
from xml.sax.saxutils import unescape

from semanticlog.core import vocabulary as vocab

RECORD_HEAD_CHARS = 1 << 16


def _record_patterns(
    annotation: str,
) -> tuple[tuple[re.Pattern[str], Callable[[str], str]], ...]:
    """Return the pattern and unescaper for one annotation, per syntax.

    In functional syntax the property is a full IRI, as semlog writes it, or a
    prefixed name, as the OWL API writes it after `materialize`.
    """
    prop = rf'(?:<[^>]*{annotation}>|[^\s:()<>"]*:{annotation})'
    return (
        (
            re.compile(rf'Annotation\({prop} "((?:[^"\\]|\\.)*)"\)'),
            _unescape_functional,
        ),
        (re.compile(rf"<[^<>]*{annotation}[^<>]*>(.*?)</", re.DOTALL), _unescape_xml),
    )


def _unescape_functional(text: str) -> str:
    """Undo the escaping of a quoted literal in OWL functional syntax."""
    return re.sub(r"\\(.)", r"\1", text)


def _unescape_xml(text: str) -> str:
    """Undo the escaping of character data in RDF/XML."""
    return unescape(text, {"&quot;": '"', "&apos;": "'"})


def recorded_annotation(path: Path, annotation: str) -> Any:
    """Return one JSON annotation of the ontology header, or None.

    Args:
        path: Ontology file to read.
        annotation: Local name of the annotation property.

    Returns:
        The parsed value, or None when the file cannot be read, holds no such
        annotation, or holds one that is not JSON.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = handle.read(RECORD_HEAD_CHARS)
    except OSError:
        return None

    for pattern, decode in _record_patterns(annotation):
        found = pattern.search(head)
        if found is None:
            continue
        try:
            return json.loads(decode(found.group(1)))
        except ValueError:
            return None
    return None


def recorded_configuration(path: Path) -> dict[str, Any] | None:
    """Return the settings an ontology records itself as built from.

    Args:
        path: Ontology file to read.

    Returns:
        The recorded settings, or None if the file holds none that can be read.
    """
    recorded = recorded_annotation(path, vocab.BUILD_CONFIGURATION)
    return recorded if isinstance(recorded, dict) else None
