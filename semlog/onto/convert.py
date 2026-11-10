"""Writing a built ontology in further formats through the OWL API."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

from semlog.core.enums import OutputFormat
from semlog.onto.reason import load_ontology, save_ontology

logger = logging.getLogger(__name__)


def convert(source: Path, targets: Mapping[OutputFormat, Path]) -> None:
    """Write the ontology in `source` once per target format.

    The source is loaded once. Each file is written under `PARTIAL_SUFFIX` and
    moved into place when complete. The targets written before a failure stay.

    Raises:
        DependencyError: If there is no usable JVM, or Java ran out of memory.
        InputError: If the OWL API cannot read the source or write a target.
            A target that fails leaves no file.
    """
    if not targets:
        return
    ontology = load_ontology(source)
    for output_format, target in targets.items():
        save_ontology(ontology, output_format, target)
        logger.info("Ontology saved as %s: %s", output_format.value, target)
