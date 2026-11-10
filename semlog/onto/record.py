"""Reading the build record back out of an ontology file.

`BuildRecordEmitter` writes the record as ontology annotations. `read_record`
finds them in the head of a written file, in functional syntax or in RDF/XML,
and parses their JSON. The file may be one semlog wrote or one the OWL
API rewrote.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.sax.saxutils import unescape

from semlog.core import vocabulary as vocab

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


@dataclass(frozen=True)
class BuildRecord:
    """The settings and case concepts an ontology records itself as built from."""

    configuration: dict[str, Any]
    case_concepts: list[str]


def _annotation(head: str, annotation: str) -> Any:
    """Return one JSON annotation found in `head`, or None if absent or not JSON."""
    for pattern, decode in _record_patterns(annotation):
        found = pattern.search(head)
        if found is None:
            continue
        try:
            return json.loads(decode(found.group(1)))
        except ValueError:
            return None
    return None


def read_record(path: Path) -> BuildRecord | None:
    """Return the build record in the head of an ontology file.

    Returns:
        The record, or None when the file cannot be read or its head lacks
        a readable configuration or case concepts entry.
    """
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            head = handle.read(RECORD_HEAD_CHARS)
    except OSError:
        return None

    configuration = _annotation(head, vocab.BUILD_CONFIGURATION)
    case_concepts = _annotation(head, vocab.CASE_CONCEPTS)
    if not isinstance(configuration, dict) or not isinstance(case_concepts, list):
        return None
    return BuildRecord(configuration, case_concepts)
