"""Writing a built ontology in further formats through the OWL API."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

from owlapy.owl_ontology import SyncOntology

from semanticlog.core.enums import OutputFormat
from semanticlog.core.errors import InputError
from semanticlog.onto.reason import save_ontology, start_jvm

logger = logging.getLogger(__name__)


def convert(source: Path, targets: Mapping[OutputFormat, Path]) -> None:
    """Write the ontology in `source` once per target format.

    The source is loaded once. Each file is written under `PARTIAL_SUFFIX` and
    moved into place when complete. The targets written before a failure stay.

    Raises:
        DependencyError: If there is no JVM to run the OWL API on.
        InputError: If the OWL API cannot read the source or write a target.
            A target that fails leaves no file.
    """
    if not targets:
        return
    start_jvm()
    try:
        ontology = SyncOntology(str(source))
    except Exception as exc:
        raise InputError(
            f"The OWL API cannot read {source}:\n  {exc}\n"
            "Build the ontology again, or check that a build wrote the file."
        ) from exc
    for output_format, target in targets.items():
        save_ontology(ontology, output_format, target)
        logger.info("Ontology saved as %s: %s", output_format.value, target)
